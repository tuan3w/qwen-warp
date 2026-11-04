#!/usr/bin/env python
"""Smoke test of the qwen-warp OpenAI server.  Needs an RTX 3090 (no other GPU has been tested; the server takes ~21 GB) and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4).

  python tests/test_server_smoke.py [--port 8011] [--profile default|long] [--keep]

1. direct engine reference (separate process, exits before the server starts): greedy ids of 4 prompts through SpecEngine + FastPrefill + cycles
2. starts serve.py, then talks to it with the `openai` python client:
   greedy chat ids == direct engine ids | streaming (many chunks, same text, usage) | stop strings | abort on client disconnect (next request is not delayed) |
   FIFO queue (concurrent requests == sequential results) | sampling (seed reproducible, seeds differ, defaults work) | unsupported params -> 400 |
   tool-call round trip (qwen3_coder parser) | /v1/completions | multi-turn prefix reuse (cached_tokens > 0, same answer as a cold run)
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
SERVE = HERE.parent / "server" / "serve.py"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}
NEW = 64
EOS = {248046, 248044}


def render(tok, messages, **kw):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False, **kw)


def long_doc(tok, n):
    raw = (ENGINE / "spec.py").read_text() + "\n\n" + (ENGINE / "prefill.py").read_text()  # real, deterministic text that ships with the repo
    return tok.decode(tok(raw, add_special_tokens=False)["input_ids"][:n])


# ------------------------------------------------------------------------------------------------ 1. direct engine reference
def ref_main(out, profile="default"):
    sys.path.insert(0, str(ENGINE))
    from transformers import AutoTokenizer
    from spec import SpecEngine
    from prefill import FastPrefill

    tok = AutoTokenizer.from_pretrained(MODEL)
    users = dict(PROMPTS, long=long_doc(tok, 600) + "\n\nSummarise the main claims above in two sentences.")
    prompts = {n: (users[n] if n == "long" else render(tok, [{"role": "user", "content": t}])) for n, t in users.items()}  # "long": raw text, no chat markers (no snapshot split in the server)
    long = profile == "long"  # the server's --profile long = int4 KV + host-pinned tables
    e = SpecEngine(ctx_max=2048, verbose=False, kv_fp8=2 if long else 0, host_tables=long)
    fp = FastPrefill(e)
    res = {}
    for name, text in prompts.items():
        ids = tok(text, add_special_tokens=False)["input_ids"]
        e.reset()
        toks = [fp.prefill_spec(ids, 3)]
        while len(toks) < NEW and not (set(toks) & EOS):
            toks += e.cycles(3, 4)[0]
        toks = toks[:NEW]
        cut = next((i for i, t in enumerate(toks) if t in EOS), None)
        res[name] = dict(user=users[name], prompt_ids=ids, gen_ids=toks if cut is None else toks[: cut + 1])
    json.dump(res, open(out, "w"))


# ------------------------------------------------------------------------------------------------ 2. server tests
class T:
    def __init__(self):
        self.res = []

    def check(self, name, ok, detail=""):
        self.res.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--keep", action="store_true", help="leave the server running")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--pld", default="auto", choices=["0", "1", "auto"], help="the server's --prompt-lookup (MTP chain + prompt-lookup tail; default auto = on at this --max-model-len); the greedy ids must still equal the direct engine's")
    ap.add_argument("--profile", default="default", choices=["default", "long"], help="server --profile (long = int4 KV + host tables); the direct-engine reference then uses the same KV dtype and host tables")
    ap.add_argument("--ref", help="internal: write the direct-engine reference json and exit")
    a = ap.parse_args()
    if a.ref:
        return ref_main(a.ref, a.profile)
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    py = sys.executable
    ref_path = tempfile.mktemp(suffix=".json")
    print("direct engine reference ...", flush=True)
    subprocess.run([py, __file__, "--ref", ref_path, "--profile", a.profile], check=True, env=env)
    ref = json.load(open(ref_path))
    os.unlink(ref_path)
    time.sleep(3)

    log = open("/tmp/swift_mk_smoke_server.log", "w")
    t0 = time.time()
    srv = subprocess.Popen([py, str(SERVE), "--port", str(a.port), "--max-model-len", str(a.max_model_len), "--profile", a.profile] + ["--prompt-lookup", a.pld], env=env, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    from openai import OpenAI

    base = f"http://127.0.0.1:{a.port}"
    t = T()
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit("server died, see /tmp/swift_mk_smoke_server.log")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        t.check("health + models", httpx.get(base + "/v1/models").json()["data"][0]["id"] == "swift", f"started in {time.time() - t0:.1f} s")
        cl = OpenAI(base_url=base + "/v1", api_key="x", timeout=120)
        nothink = dict(chat_template_kwargs={"enable_thinking": False}, return_token_ids=True)

        def chat(text=None, messages=None, **kw):
            messages = messages or [{"role": "user", "content": text}]
            return cl.chat.completions.create(model="swift", messages=messages, temperature=kw.pop("temperature", 0), max_tokens=kw.pop("max_tokens", NEW), **kw)

        # greedy parity with the direct engine
        for name, r in ref.items():
            if name == "long":  # fast-prefill path via /v1/completions with raw prompt ids: no <|im_start|>, so no snapshot split and the segmentation is the reference's (a split changes the accumulation order)
                out = cl.completions.create(model="swift", prompt=r["prompt_ids"], temperature=0, max_tokens=NEW, extra_body=dict(return_token_ids=True))
            else:
                out = chat(r["user"], extra_body=nothink)
            ch = out.choices[0]
            ids = ch.model_extra.get("token_ids") if ch.model_extra else None
            pids = (out.model_extra or {}).get("prompt_token_ids") or (ch.model_extra or {}).get("prompt_token_ids")
            fd = next((i for i, (x, y) in enumerate(zip(ids or [], r["gen_ids"])) if x != y), None)
            t.check(f"greedy chat == direct engine ids [{name}]", ids == r["gen_ids"] and pids == r["prompt_ids"], f"{len(ids or [])}/{len(r['gen_ids'])} ids, first difference at index {fd}, prompt ids equal {pids == r['prompt_ids']}")
        base_out = chat(PROMPTS["code"], extra_body=nothink)
        base_text = base_out.choices[0].message.content

        # streaming
        t_chunks, text, usage = [], "", None
        st = cl.chat.completions.create(model="swift", messages=[{"role": "user", "content": PROMPTS["code"]}], temperature=0, max_tokens=NEW, stream=True,
                                        stream_options={"include_usage": True}, extra_body=nothink)
        for c in st:
            if c.choices and c.choices[0].delta.content:
                t_chunks.append(time.time())
                text += c.choices[0].delta.content
            if c.usage:
                usage = c.usage
        gaps = [(b - a_) * 1e3 for a_, b in zip(t_chunks, t_chunks[1:])]
        t.check("streaming == non-streaming text, many chunks, usage", text == base_text and len(t_chunks) > 5 and usage and usage.completion_tokens == base_out.usage.completion_tokens,
                f"{len(t_chunks)} chunks, median gap {sorted(gaps)[len(gaps) // 2]:.0f} ms, usage {usage and usage.completion_tokens}")

        # stop strings
        words = [w for w in base_text.split() if len(w) > 4]
        stop = words[len(words) // 3]
        so = chat(PROMPTS["code"], stop=[stop], extra_body=nothink)
        s_text = so.choices[0].message.content
        t.check("stop string cuts the text before the match", so.choices[0].finish_reason == "stop" and stop not in s_text and base_text.startswith(s_text) and len(s_text) < len(base_text), f"stop={stop!r} len {len(s_text)}/{len(base_text)}")
        eo = chat("Say hi.", max_tokens=5, extra_body=nothink)
        t.check("max_tokens -> finish_reason length", eo.choices[0].finish_reason == "length" and eo.usage.completion_tokens == 5, f"{eo.choices[0].finish_reason} {eo.usage.completion_tokens}")
        t.check("EOS -> finish_reason stop", chat("Say hi.", max_tokens=100, extra_body=nothink).choices[0].finish_reason == "stop")

        # abort: a client that drops a 2000-token stream must free the engine (the next request would otherwise wait ~15 s)
        with httpx.stream("POST", base + "/v1/chat/completions", json=dict(model="swift", messages=[{"role": "user", "content": PROMPTS["prose"]}], temperature=0, max_tokens=2000, stream=True,
                                                                         chat_template_kwargs={"enable_thinking": False}), timeout=60) as r:
            n = 0
            for line in r.iter_lines():
                if line.startswith("data:"):
                    n += 1
                if n >= 5:
                    break
        time.sleep(0.05)
        t1 = time.time()
        chat("Say hi.", max_tokens=8, extra_body=nothink)
        dt = time.time() - t1
        t.check("abort on client disconnect frees the engine", dt < 3.0, f"next request took {dt:.2f} s")

        # FIFO queue: concurrent requests give the sequential results
        seq = {n: chat(p, max_tokens=40, extra_body=nothink).choices[0].message.content for n, p in PROMPTS.items()}
        got, ths = {}, []
        for n, p in PROMPTS.items():
            def go(n=n, p=p):
                got[n] = chat(p, max_tokens=40, extra_body=nothink).choices[0].message.content

            th = threading.Thread(target=go)
            th.start()
            ths.append(th)
            time.sleep(0.02)
        for th in ths:
            th.join()
        t.check("3 concurrent requests == sequential results (FIFO, state isolation)", got == seq)

        # sampling
        # (a repeated prompt resumes from its snapshot = the exact T=1 head step; a cold short prompt runs through the K10 batched path = same maths up to bf16 accumulation order,
        # so a sampled continuation can differ between the two: compare cold with cold (an unrelated request in between replaces the lineage) and warm with warm)
        def samp(seed):
            return chat(PROMPTS["prose"], temperature=0.8, top_p=0.95, seed=seed, max_tokens=48, extra_body=dict(top_k=20, **nothink)).choices[0].message.content

        s1 = samp(7)
        chat(PROMPTS["code"], max_tokens=1, extra_body=nothink)
        s2 = samp(7)
        s3 = samp(8)
        s4 = samp(8)  # warm (resumed) twice
        t.check("sampling: same seed reproducible (cold/cold and warm/warm), other seed differs", s1 == s2 and s3 == s4 and s1 != s3, f"{s1[:40]!r} / {s3[:40]!r}")
        d = cl.chat.completions.create(model="swift", messages=[{"role": "user", "content": "Name a colour."}], max_tokens=20, extra_body=nothink)
        t.check("request without sampling params uses generation_config defaults (T=1, top_p .95, top_k 20)", d.choices[0].message.content.strip() != "")
        for name, bad in (("logprobs", {"logprobs": True, "top_logprobs": 2}), ("bad_words", {"extra_body": {"bad_words": ["hi"]}}), ("thinking_token_budget", {"extra_body": {"thinking_token_budget": 8}})):
            try:
                chat("hi", max_tokens=4, **bad)
                t.check(f"unsupported {name} -> 400", False, "accepted")
            except Exception as ex:  # noqa: BLE001
                t.check(f"unsupported {name} -> 400", getattr(ex, "status_code", None) == 400, str(ex)[:120])

        # tools (qwen3_coder parser)
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        msgs = [{"role": "user", "content": "What is the weather in Hanoi right now? Use the tool."}]
        r1 = chat(messages=msgs, tools=tools, max_tokens=300, extra_body=nothink).choices[0]
        calls = r1.message.tool_calls
        ok = bool(calls) and calls[0].function.name == "get_weather" and "hanoi" in json.loads(calls[0].function.arguments).get("city", "").lower()
        t.check("tool call parsed (name + arguments)", ok, f"{r1.message.content!r} {calls}")
        if calls:
            msgs += [{"role": "assistant", "content": r1.message.content or "", "tool_calls": [{"id": calls[0].id, "type": "function", "function": {"name": "get_weather", "arguments": calls[0].function.arguments}}]},
                     {"role": "tool", "tool_call_id": calls[0].id, "content": json.dumps({"temp_c": 31, "sky": "sunny"})}]
            r2 = chat(messages=msgs, tools=tools, max_tokens=200, extra_body=nothink).choices[0]
            t.check("tool result round trip: final answer uses the tool output", "31" in (r2.message.content or ""), r2.message.content)

        # completions
        c = cl.completions.create(model="swift", prompt="The capital of France is", max_tokens=6, temperature=0)
        t.check("/v1/completions", "Paris" in c.choices[0].text, c.choices[0].text)

        # multi-turn prefix reuse: 3 turns over a ~1.5K-token document
        doc = long_doc(__import__("transformers").AutoTokenizer.from_pretrained(MODEL), 1500)
        convo = [{"role": "user", "content": doc + "\n\nQuestion 1: list the three most important topics above, one line each."}]
        cached, outs, out_ids = [], [], []
        for turn, q in enumerate([None, "Question 2: which one of them is most likely to be wrong, in one sentence?", "Question 3: give a one-sentence summary of your previous answer."]):
            if q:
                convo = convo + [{"role": "assistant", "content": outs[-1]}, {"role": "user", "content": q}]
            t1 = time.time()
            o = chat(messages=convo, max_tokens=60, extra_body=nothink)
            ttft = time.time() - t1
            outs.append(o.choices[0].message.content)
            out_ids.append(o.choices[0].token_ids)
            cached.append((o.usage.prompt_tokens_details.cached_tokens if o.usage.prompt_tokens_details else 0, o.usage.prompt_tokens, ttft))
        t.check("multi-turn: turns 2 and 3 resume from a snapshot", cached[1][0] > 0 and cached[2][0] > 0, f"(cached, prompt, total s) per turn: {cached}")
        chat("Unrelated question: what is 17*23?", max_tokens=8, extra_body=nothink)  # replaces the lineage -> cold
        cold = chat(messages=convo, max_tokens=60, extra_body=nothink)
        chat("Unrelated question: what is 19*23?", max_tokens=8, extra_body=nothink)
        cold2 = chat(messages=convo, max_tokens=60, extra_body=nothink)
        t.check("multi-turn: cold re-prefill is deterministic (two cold runs, same ids)", cold.choices[0].token_ids == cold2.choices[0].token_ids)
        # A resumed prefill and a full prefill do not accumulate in the same order, so a greedy answer may drift from the cold one at a near-tie token
        # (seen in 3 of 12 documents x {speculative, plain} runs, at tokens 6 to 30 of ~35).  The invariant that does hold: the restored state is not broken,
        # so the answers agree from the start.  A corrupt snapshot changes the first tokens.
        w, c = out_ids[2], cold.choices[0].token_ids
        first_diff = next((i for i in range(min(len(w), len(c))) if w[i] != c[i]), None)
        t.check("multi-turn: warm turn 3 starts like the cold re-prefill answer (first >= 3 greedy ids equal)", first_diff is None or first_diff >= 3,
                f"first difference at token {first_diff} of {len(w)}/{len(c)}" if first_diff is not None else f"identical, {len(w)} tokens")
        t.check("cold re-prefill reports cached_tokens == 0", (cold.usage.prompt_tokens_details.cached_tokens if cold.usage.prompt_tokens_details else 0) == 0)
    finally:
        if not a.keep:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
    bad = [n for n, ok in t.res if not ok]
    print(f"\n{len(t.res) - len(bad)}/{len(t.res)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
