#!/usr/bin/env python
"""Short-prompt TTFT through an OpenAI server (swift-mk's serve.py or stock `vllm serve`), same client for both (K10).

  ~/llm-lab/venv/bin/python server/bench_ttft.py --url http://127.0.0.1:8011 --label swift-mk [--lens 8,20,40,64,128,256,512] [--reps 5]
Run from the repo root (filler text = topics/*.md).  For each length L: /v1/completions with a prompt of exactly L token ids (chat-template rendered, thinking off, for
L >= 24; plain corpus ids below), temperature 0, max_tokens 4, stream; TTFT = request sent -> first SSE chunk carrying a token.  Every rep uses a different filler slice,
so no prefix cache / snapshot can hit.  Prints the median and every rep.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

MODEL = Path.home() / "models" / "swift-1.5-int4"


def ttft(cl, url, model, ids):
    body = dict(model=model, prompt=ids, max_tokens=4, temperature=0, ignore_eos=True, stream=True, return_token_ids=True)
    t0 = time.perf_counter()
    first = None
    with cl.stream("POST", url + "/v1/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():  # read to the end: abandoning the stream would leave the engine finishing this request while the next one queues behind it
            if first is None and line.startswith("data:") and line.strip() != "data: [DONE]":
                d = json.loads(line[5:])
                if d.get("choices") and d["choices"][0].get("token_ids"):
                    first = time.perf_counter() - t0
    if first is None:
        raise RuntimeError("no token")
    return first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--label", default="server")
    ap.add_argument("--model", default="swift")
    ap.add_argument("--lens", default="8,20,40,64,128,256,512")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    corpus = tok(raw, add_special_tokens=False)["input_ids"]

    def prompt(L, rep):
        off = 500 + 997 * rep
        if L < 24:
            return corpus[off : off + L]
        k = L - 14
        for _ in range(8):
            text = tok.apply_chat_template([{"role": "user", "content": tok.decode(corpus[off : off + k])}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) == L:
                return ids
            k += L - len(ids)
        return ids[:L] if len(ids) > L else ids + corpus[off : off + L - len(ids)]

    cl = httpx.Client()
    for _ in range(2):
        ttft(cl, a.url, a.model, prompt(40, 99))  # warm-up
    res = {}
    for L in [int(x) for x in a.lens.split(",")]:
        ts = [ttft(cl, a.url, a.model, prompt(L, r)) * 1e3 for r in range(a.reps)]
        res[L] = ts
        print(f"{a.label:12s} prompt {L:4d} tokens: TTFT median {statistics.median(ts):7.1f} ms   {[round(t) for t in ts]}", flush=True)
    if a.out:
        json.dump({str(k): v for k, v in res.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
