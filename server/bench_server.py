#!/usr/bin/env python
"""HTTP-level benchmark of an OpenAI server (swift-mk's serve.py or stock `vllm serve`), same client for both.

  ~/llm-lab/venv/bin/python server/bench_server.py --url http://127.0.0.1:8011 --label swift-mk [--prompts code,prose,math,long8k] [--reps 3] [--multiturn]
Run from the repo root (the 8K prompt is built from topics/*.md like E0 / engine/tests/bench_spec.py).

Per prompt: POST /v1/completions with the chat-rendered token ids (thinking off), temperature 0, ignore_eos, max_tokens N (default 385), stream + return_token_ids.
  TTFT   = request sent -> first SSE chunk (= first token) [separate requests, each with a unique 1-token prefix, so no prefix cache / snapshot can hit]
  decode = (tokens after the first chunk) / (last chunk time - first chunk time)  (= E0's 384 / (t(385) - t(1)) / bench_spec's rate)
  gaps   = inter-chunk time (streaming granularity)
--multiturn: 3-turn chat over a ~1.5K-token document through /v1/chat/completions, TTFT of each turn warm (resumed) and cold (after an unrelated request).
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

MODEL = Path.home() / "models" / "swift-1.5-int4"
SHORT = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}


def build_prompts(names):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    chat = lambda text: tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)  # noqa: E731
    out = {}
    for n in names:
        if n in SHORT:
            text = chat(SHORT[n])
        elif n == "long8k":
            raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
            ids = tok(raw, add_special_tokens=False)["input_ids"][:8000]
            text = chat(tok.decode(ids) + "\n\nSummarise the main claims above in a numbered list, citing the section names.")
        else:
            raise SystemExit(f"unknown prompt {n}")
        out[n] = tok(text, add_special_tokens=False)["input_ids"]
    return tok, out


def stream_completion(cl, url, ids, n, model):
    body = dict(model=model, prompt=ids, max_tokens=n, temperature=0, ignore_eos=True, stream=True, stream_options={"include_usage": True}, return_token_ids=True)
    t0 = time.perf_counter()
    times, counts = [], []
    with cl.stream("POST", url + "/v1/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("choices"):
                tid = d["choices"][0].get("token_ids") or []
                if tid:
                    times.append(time.perf_counter())
                    counts.append(len(tid))
    total = sum(counts)
    first = counts[0]
    return dict(ttft=times[0] - t0, decode=(total - first) / (times[-1] - times[0]), total=total, first=first, chunks=len(times), gaps=[(b - a) * 1e3 for a, b in zip(times, times[1:])], e2e=times[-1] - t0)


def multiturn(cl, url, model, tok, reps):
    raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    doc = tok.decode(tok(raw, add_special_tokens=False)["input_ids"][:1500])
    qs = ["Question 1: list the three most important topics above, one line each.", "Question 2: which one of them is most likely to be wrong, in one sentence?",
          "Question 3: give a one-sentence summary of your previous answer."]
    res = []
    for rep in range(reps):
        convo = [{"role": "user", "content": f"[doc {rep}]\n" + doc + "\n\n" + qs[0]}]
        row = []
        for turn in range(3):
            if turn:
                convo = convo + [{"role": "assistant", "content": ans}, {"role": "user", "content": qs[turn]}]  # noqa: F821

            def ask(c):
                t0 = time.perf_counter()
                tf, text = None, ""
                with cl.stream("POST", url + "/v1/chat/completions", json=dict(model=model, messages=c, max_tokens=60, temperature=0, stream=True, stream_options={"include_usage": True},
                                                                              chat_template_kwargs={"enable_thinking": False}), timeout=600) as r:
                    r.raise_for_status()
                    usage = None
                    for line in r.iter_lines():
                        if line.startswith("data:") and line.strip() != "data: [DONE]":
                            d = json.loads(line[5:])
                            if d.get("choices") and d["choices"][0]["delta"].get("content"):
                                tf = tf or time.perf_counter() - t0
                                text += d["choices"][0]["delta"]["content"]
                            if d.get("usage"):
                                usage = d["usage"]
                return tf, text, usage

            tf, ans, usage = ask(convo)
            cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
            ask([{"role": "user", "content": "What is 17*23? Answer with the number only."}])  # unrelated: replaces the server's resumable state
            tc, _, _ = ask(convo)
            row.append(dict(turn=turn + 1, prompt=usage["prompt_tokens"], cached=cached, ttft_warm=tf, ttft_cold=tc))
        res.append(row)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--label", default="server")
    ap.add_argument("--model", default="swift")
    ap.add_argument("--prompts", default="code,prose,math,long8k")
    ap.add_argument("--n", type=int, default=385)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--multiturn", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    tok, prompts = build_prompts(a.prompts.split(","))
    cl = httpx.Client()
    out = dict(label=a.label, n=a.n, prompts={})
    stream_completion(cl, a.url, prompts[next(iter(prompts))], 16, a.model)  # warm up (graphs, caches)
    for name, ids in prompts.items():
        reps = a.reps if name != "long8k" else max(1, a.reps - 1)
        rows = [stream_completion(cl, a.url, ids, a.n, a.model) for _ in range(reps)]  # exact E0 prompt (rep 2+ may be served from a prefix cache: only decode is used)
        tt = []
        for rep in range(reps):  # TTFT: a unique 1-token prefix per rep defeats every prefix cache / snapshot; 8 tokens are enough
            uniq = tok(f"#{rep}{name}\n", add_special_tokens=False)["input_ids"]
            tt.append(stream_completion(cl, a.url, uniq + ids, 8, a.model)["ttft"])
        med = lambda k: statistics.median(r[k] for r in rows)  # noqa: E731
        gaps = sorted(g for r in rows for g in r["gaps"])
        out["prompts"][name] = dict(prompt_tokens=len(ids), decode_tok_s=med("decode"), ttft_s=statistics.median(tt), chunks=med("chunks"), tok_first=rows[0]["first"], gap_ms_p50=gaps[len(gaps) // 2],
                                    gap_ms_p95=gaps[int(len(gaps) * .95)], reps=[round(r["decode"], 1) for r in rows], ttft_reps=[round(x, 3) for x in tt])
        o = out["prompts"][name]
        print(f"{a.label:12s} {name:7s} prompt {o['prompt_tokens']:5d}: decode {o['decode_tok_s']:6.1f} tok/s {o['reps']}  TTFT {o['ttft_s'] * 1e3:7.0f} ms {[int(x * 1e3) for x in tt]}  "
              f"chunk gap p50/p95 {o['gap_ms_p50']:.0f}/{o['gap_ms_p95']:.0f} ms, {o['chunks']:.0f} chunks", flush=True)
    if a.multiturn:
        out["multiturn"] = multiturn(cl, a.url, a.model, tok, max(1, a.reps - 1))
        for t in range(3):
            w = statistics.median(r[t]["ttft_warm"] for r in out["multiturn"])
            c = statistics.median(r[t]["ttft_cold"] for r in out["multiturn"])
            r0 = out["multiturn"][0][t]
            print(f"{a.label:12s} multi-turn turn {t + 1}: prompt {r0['prompt']} tokens, cached {r0['cached']}: TTFT warm {w * 1e3:.0f} ms, cold {c * 1e3:.0f} ms")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
