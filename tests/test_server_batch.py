#!/usr/bin/env python
"""The batched server (--max-num-seqs > 1) end to end: ONE server start, many HTTP checks.  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL).

Tier: 2

  python tests/test_server_batch.py [--port 8021] [--slots 4] [--max-model-len 8192] [--profile default|long] [--keep]

The server runs in the exact mode (SMK_BATCH_K=1: every request is decoded by batched cycles of one draft, whatever the batch size; server/smk_batch.py), in which a request's
token ids do not depend on the other requests (the row-wise kernels are row-count invariant, tests/test_batch_engine.py part A).  So every check below compares ids EXACTLY with the
same request sent alone to the same server:
  concurrent clients (more than --slots: some queue) | greedy and seeded sampling | one request with logit_bias + penalties next to plain ones (no leak either way) |
  a client that disconnects mid-stream (the others unchanged, the next request is served at once) | requests joining and leaving at different times (max_tokens prefixes) |
  a long prompt arriving while others stream (they keep streaming: the largest gap between their chunks stays under a bound) | multi-turn cached_tokens (> 0, same answer as cold).
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "server" / "serve.py"
PROMPTS = ["Write a Python function that merges two sorted lists.", "Describe a quiet harbour at dawn in one paragraph.", "What is 23 * 47? Show the steps.",
           "List three facts about Jupiter.", "Write a limerick about a cat.", "Explain recursion to a child in three sentences.", "Give a short recipe for pancakes.",
           "Name four sorting algorithms and one property of each."]
NEW = 96
GAP_BOUND_S = 1.5  # a running stream may pause for one prefill unit (~0.3 s at 512 rows) plus one decode group; a frozen stream would wait for the whole ~3 s prefill


class T:
    def __init__(self):
        self.fails = []

    def check(self, name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)
        if not ok:
            self.fails.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--profile", default="default", choices=["default", "long"])
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1", VLLM_SERVER_DEV_MODE="1", SMK_BATCH_K="1")
    log = open("/tmp/swift_mk_batch_server.log", "w")
    t0 = time.time()
    srv = subprocess.Popen([sys.executable, str(SERVE), "--port", str(a.port), "--max-model-len", str(a.max_model_len), "--profile", a.profile, "--max-num-seqs", str(a.slots)],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    from openai import OpenAI

    base = f"http://127.0.0.1:{a.port}"
    t = T()
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit("server died, see /tmp/swift_mk_batch_server.log")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        t.check("server up", True, f"{time.time() - t0:.1f} s, {a.slots} slots, exact mode")
        cl = OpenAI(base_url=base + "/v1", api_key="x", timeout=300)
        nothink = dict(chat_template_kwargs={"enable_thinking": False}, return_token_ids=True)

        def chat(text, max_tokens=NEW, messages=None, **kw):
            out = cl.chat.completions.create(model="swift", messages=messages or [{"role": "user", "content": text}], max_tokens=max_tokens, extra_body={**nothink, **kw.pop("extra", {})},
                                             temperature=kw.pop("temperature", 0), **kw)
            ch = out.choices[0]
            return dict(ids=(ch.model_extra or {}).get("token_ids"), text=ch.message.content, cached=getattr(getattr(out.usage, "prompt_tokens_details", None), "cached_tokens", None),
                        usage=out.usage.completion_tokens)

        def concurrently(fns):
            res = [None] * len(fns)

            def go(i, f):
                res[i] = f()

            th = [threading.Thread(target=go, args=(i, f)) for i, f in enumerate(fns)]
            for x in th:
                x.start()
            for x in th:
                x.join()
            return res

        jobs = [(p, dict()) for p in PROMPTS[:4]] + [(p, dict(temperature=0.8, seed=100 + i, top_p=0.95)) for i, p in enumerate(PROMPTS[4:])]
        # 1. references: every request alone
        t1 = time.time()
        refs = [chat(p, **dict(kw)) for p, kw in jobs]
        alone_s = time.time() - t1
        # 2. all of them at once (2 x the slots: half queue)
        t1 = time.time()
        got = concurrently([lambda p=p, kw=kw: chat(p, **dict(kw)) for p, kw in jobs])
        conc_s = time.time() - t1
        same = [g["ids"] == r["ids"] for g, r in zip(got, refs)]
        ntok = sum(len(r["ids"]) for r in refs)
        t.check(f"{len(jobs)} concurrent requests == each alone (greedy and seeded sampling)", all(same), f"{sum(same)}/{len(same)} identical; {ntok} tokens alone {alone_s:.1f} s, concurrent {conc_s:.1f} s")
        # 3. processors next to plain requests: no leak either way
        biased = dict(extra=dict(logit_bias={str(refs[0]["ids"][1]): -100, str(refs[0]["ids"][3]): 5}, presence_penalty=0.8, frequency_penalty=0.4, repetition_penalty=1.15))
        bref = chat(PROMPTS[0], **dict(biased))
        got = concurrently([lambda: chat(PROMPTS[0], **dict(biased))] + [lambda p=p: chat(p) for p in PROMPTS[1:4]])
        t.check("logit_bias + penalties next to plain requests", got[0]["ids"] == bref["ids"] and all(g["ids"] == r["ids"] for g, r in zip(got[1:], refs[1:4])) and bref["ids"] != refs[0]["ids"],
                f"biased == its alone run {got[0]['ids'] == bref['ids']}, plain == refs {[g['ids'] == r['ids'] for g, r in zip(got[1:], refs[1:4])]}, bias changed the output {bref['ids'] != refs[0]['ids']}")
        # 4. abort mid-batch: one streaming client disconnects after a few chunks
        def stream(text, n_chunks=None, gaps=None, **kw):
            body = dict(model="swift", messages=[{"role": "user", "content": text}], max_tokens=kw.pop("max_tokens", NEW), temperature=0, stream=True, **nothink)
            txt, last, k = "", time.time(), 0
            with httpx.stream("POST", base + "/v1/chat/completions", json=body, timeout=300) as rs:
                for line in rs.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[6:])
                    now = time.time()
                    if gaps is not None and d["choices"] and d["choices"][0]["delta"].get("content"):
                        gaps.append((now, now - last))
                        last = now
                    if d["choices"]:
                        txt += d["choices"][0]["delta"].get("content") or ""
                    k += 1
                    if n_chunks is not None and k >= n_chunks:
                        return txt  # closing the connection aborts the request
            return txt

        got = concurrently([lambda: stream(PROMPTS[0], n_chunks=4)] + [lambda p=p: stream(p) for p in PROMPTS[1:4]])
        t1 = time.time()
        after = chat(PROMPTS[1])
        t.check("a client disconnects mid-stream: the other streams are unchanged, the next request is served at once",
                all(g == r["text"] for g, r in zip(got[1:], refs[1:4])) and after["ids"] == refs[1]["ids"] and time.time() - t1 < 15,
                f"others == refs {[g == r['text'] for g, r in zip(got[1:], refs[1:4])]}, next request {time.time() - t1:.1f} s")
        # 5. requests joining and leaving at different times, different max_tokens
        def later(d, p, n):
            time.sleep(d)
            return chat(p, max_tokens=n)

        lens = [96, 20, 60, 8, 40, 96]
        got = concurrently([lambda i=i: later(0.25 * i, PROMPTS[i % 4], lens[i]) for i in range(6)])
        t.check("requests join and leave while others decode (max_tokens 8..96)", all(g["ids"] == refs[i % 4]["ids"][: lens[i]] for i, g in enumerate(got)),
                f"{[g['ids'] == refs[i % 4]['ids'][: lens[i]] for i, g in enumerate(got)]}")
        # 6. a long prompt arrives while three requests stream: they must not freeze for its whole prefill
        long_text = "Here is a log to summarise later.\n" + "\n".join(f"line {i}: the service handled request {i * 7} in {i % 13} ms with status {200 + i % 3}" for i in range(220))
        gaps = [[], [], []]
        res = {}

        def long_req():
            time.sleep(2.0)
            t2 = time.time()
            res["long"] = chat(long_text + "\nHow many lines are there? Answer with a number only.", max_tokens=8)
            res["long_s"] = (t2, time.time())

        got = concurrently([lambda i=i: stream(PROMPTS[i], gaps=gaps[i], max_tokens=400) for i in range(3)] + [long_req])
        lo, hi = res["long_s"]
        during = [g for gs in gaps for (when, g) in gs if lo <= when <= hi + 0.5]
        worst = max(during, default=0.0)
        t.check("a long prompt prefills while three requests stream: they keep streaming", bool(during) and worst < GAP_BOUND_S,
                f"long request {hi - lo:.1f} s ({res['long']['usage']} tokens out), {len(during)} chunks of the others meanwhile, largest gap {worst:.2f} s (bound {GAP_BOUND_S} s)")
        # 7. multi-turn: cached_tokens and the cached answer == the cold answer
        sysmsg = " ".join(f"Rule {i}: be precise and brief about topic {i}." for i in range(80))
        m1 = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Say hello."}]
        r1 = chat(None, messages=m1)
        m2 = m1 + [{"role": "assistant", "content": r1["text"]}, {"role": "user", "content": "Now count to five."}]
        r2 = chat(None, messages=m2)
        httpx.post(base + "/reset_prefix_cache")
        r2c = chat(None, messages=m2)
        t.check("follow-up turn reports cached_tokens and equals the cold answer", (r2["cached"] or 0) > 0 and r2c["cached"] == 0 and r2["ids"] == r2c["ids"],
                f"cached {r2['cached']} then {r2c['cached']} after reset, same ids {r2['ids'] == r2c['ids']}")
    finally:
        if not a.keep:
            srv.terminate()
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
    print(f"{'FAILED: ' + ', '.join(t.fails) if t.fails else 'all passed'} in {time.time() - t0:.1f} s")
    sys.exit(1 if t.fails else 0)


if __name__ == "__main__":
    main()
