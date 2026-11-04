#!/usr/bin/env python
"""The `seed` contract and the `top_k` rules of the qwen-warp OpenAI server.  Needs an RTX 3090 (the server takes ~21 GB) and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4).

  python tests/test_server_seed.py [--port 8013] [--reps 9] [--pld auto|0|1] [--k 3] [--tree DIR] [--dump FILE]

Starts serve.py once.  A request with a seed must give the same tokens whatever the cache state (cold prefill, resumed from a snapshot of an earlier turn, repeated), so every
scenario repeats ONE seeded request `reps` times, alternating the preceding requests:  cold = an unrelated request in between (the engine's cache is replaced),
turn = the earlier turns of the same chat first (the request resumes from their snapshot), repeat = the request again right after itself.
All token ids (temperature 1, 48 tokens) must be equal, and both cached_tokens == 0 and > 0 must have occurred.  Scenarios: 2- and 3-turn chats over a ~1.5K-token document with
thinking off and on, a raw 1000-token completion prompt (a repeat resumes right before its last token), n = 2 (choices differ, the pair is reproducible), greedy.
top_k: -1, 0, 1, 20, 64 are accepted (<= 0 = no filter = the engine's 64 best candidates, the same as 64), 65 and 1000 give HTTP 400 naming top_k and the limit
(at temperature 0 top_k is ignored, as in vLLM).
--dump FILE writes the token ids of fixed seeded probes (to compare configurations / runs).
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
SERVE = HERE.parent / "server" / "serve.py"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
NEW = 48
QS = ["Question 1: list the three most important topics above, one line each.",
      "Question 2: which one of them is most likely to be wrong, in one sentence?",
      "Question 3: give a one-sentence summary of your previous answer."]
ANS = ["1. the speculative decoding driver\n2. the sampling configuration\n3. the fast prefill path", "The sampling configuration, because it has the most parameters to keep consistent."]


def long_doc(tok, n):
    raw = (ENGINE / "spec.py").read_text() + "\n\n" + (ENGINE / "prefill.py").read_text()  # real, deterministic text that ships with the repo
    return tok.decode(tok(raw, add_special_tokens=False)["input_ids"][:n])


class T:
    def __init__(self):
        self.res = []

    def check(self, name, ok, detail=""):
        self.res.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=9, help="repetitions of the seeded request per scenario (multiple of 3)")
    ap.add_argument("--pld", default="auto", choices=["0", "1", "auto"])
    ap.add_argument("--k", type=int, default=3, choices=[0, 1, 2, 3], help="the server's --num-speculative-tokens")
    ap.add_argument("--tree", help="serve from this checkout instead of this one (run the test against an older revision)")
    ap.add_argument("--dump", help="write the token ids of fixed seeded probes to this json file")
    a = ap.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    log = open("/tmp/swift_mk_seed_server.log", "w")
    srv = subprocess.Popen([sys.executable, str(Path(a.tree) / "server" / "serve.py" if a.tree else SERVE), "--port", str(a.port), "--max-model-len", str(a.max_model_len), "--prompt-lookup", a.pld, "--num-speculative-tokens", str(a.k)],
                           env=env, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    from transformers import AutoTokenizer

    base = f"http://127.0.0.1:{a.port}"
    t = T()
    t0 = time.time()
    dump = {}
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit("server died, see /tmp/swift_mk_seed_server.log")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        tok = AutoTokenizer.from_pretrained(MODEL)
        doc = long_doc(tok, 1500)
        http = httpx.Client(timeout=300)

        def post(path, body):
            r = http.post(base + path, json=body)
            return r.status_code, r.json()

        def chat(messages, think=False, **kw):
            body = dict(model="swift", messages=messages, max_tokens=kw.pop("max_tokens", NEW), temperature=kw.pop("temperature", 1.0), return_token_ids=True,
                        chat_template_kwargs={"enable_thinking": think}, **kw)
            code, j = post("/v1/chat/completions", body)
            assert code == 200, (code, j)
            return [c["token_ids"] for c in j["choices"]], (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0

        def convo(n_turns):
            m = [{"role": "user", "content": doc + "\n\n" + QS[0]}]
            for i in range(1, n_turns):
                m += [{"role": "assistant", "content": ANS[i - 1]}, {"role": "user", "content": QS[i]}]
            return m

        def unrelated(i):
            return chat([{"role": "user", "content": f"Unrelated question number {i}: what is {17 + i} times 23? Answer briefly."}], seed=1000 + i, max_tokens=6)

        def repeat_check(name, once, cold, turn, reps=a.reps, record=None, resumed_only=False):
            """once() -> (ids, cached); cold() / turn() run the requests that precede it."""
            outs = []
            for r in range(reps):
                [cold, turn, lambda: None][r % 3]()
                outs.append(once())
            ids = [json.dumps(o[0]) for o in outs]
            cached = [o[1] for o in outs]
            t.check(f"{name}: {reps} seeded runs give the same token ids", len(set(ids)) == 1, f"{len(set(ids))} distinct; cached_tokens {cached}")
            if not resumed_only:  # (n > 1 reports the cached tokens of its last child)
                t.check(f"{name}: cold (cached_tokens 0) and resumed (> 0) runs both occurred", 0 in cached and any(c > 0 for c in cached), f"cached_tokens {cached}")
            if record:
                dump[record] = outs[0][0]
            return outs

        seed = 123
        for think in (False, True):
            for turns in (2, 3):
                m = convo(turns)
                repeat_check(f"chat {turns} turns, thinking {'on' if think else 'off'}", lambda m=m, think=think: chat(m, think, seed=seed),
                             lambda: unrelated(1), lambda m=m, think=think, turns=turns: [chat(convo(i), think, seed=7, max_tokens=3) for i in range(1, turns)], record=f"chat{turns}_think{int(think)}")
        # raw prompts without chat markers: one segment per 4096 tokens; a repeat resumes right before the last prompt token
        def complete(prompt_ids):
            code, j = post("/v1/completions", dict(model="swift", prompt=prompt_ids, max_tokens=NEW, temperature=1.0, seed=seed, return_token_ids=True))
            assert code == 200, (code, j)
            return [c["token_ids"] for c in j["choices"]], (j["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0

        ids = tok(doc, add_special_tokens=False)["input_ids"][:1000]
        repeat_check("raw completion prompt (1000 tokens)", lambda: complete(ids), lambda: unrelated(2), lambda: None, record="raw")
        # a prompt of 9000 tokens that extends an earlier one of 5000: the earlier request's last snapshot (before its last token) is NOT a segment end of the longer prompt
        # (its plan ends at 4096, 8192 and 8999), only the cut at 4096 is shared
        full = tok(long_doc(tok, 12000), add_special_tokens=False)["input_ids"]
        big = (full * 2)[:9000]
        repeat_check("raw completion prompt (9000 tokens) after a 5000-token prefix", lambda: complete(big), lambda: unrelated(5), lambda: complete(big[:5000]), reps=6, record="raw9000")
        # n = 2: the children get the seeds seed + index (vLLM); the second child resumes from the first one's snapshot, so the pair covers a cold and a resumed run in every repetition
        m2 = convo(2)
        pair = chat(m2, False, seed=seed, n=2)[0]
        single = [chat(m2, False, seed=seed + i)[0][0] for i in (0, 1)]
        t.check("n=2: the choices differ and equal the single requests with seed and seed + 1", pair[0] != pair[1] and pair == single)
        repeat_check("n=2, chat 2 turns", lambda: chat(m2, False, seed=seed, n=2), lambda: unrelated(3), lambda: chat(convo(1), False, seed=7, max_tokens=3), record="n2", resumed_only=True)
        # greedy output does not depend on the cache either
        repeat_check("greedy, chat 3 turns", lambda: chat(convo(3), False, temperature=0), lambda: unrelated(4), lambda: [chat(convo(i), False, seed=7, max_tokens=3) for i in (1, 2)], reps=6, record="greedy")
        s1, s2 = chat(convo(2), False, seed=seed)[0][0], chat(convo(2), False, seed=seed + 1)[0][0]
        t.check("another seed gives another sequence", s1 != s2)
        u1, u2 = chat(convo(1), False)[0][0], chat(convo(1), False)[0][0]
        t.check("without a seed the sequences differ (not reproducible by design)", u1 != u2)

        # top_k rules
        def top_k(v, seed=5):
            return post("/v1/chat/completions", dict(model="swift", messages=[{"role": "user", "content": "Write one sentence about the sea."}], max_tokens=24, temperature=1.0, seed=seed,
                                                     top_k=v, return_token_ids=True, chat_template_kwargs={"enable_thinking": False}))

        res = {v: top_k(v) for v in (-1, 0, 1, 20, 64, 65, 1000)}
        t.check("top_k -1, 0, 1, 20, 64 are accepted", all(res[v][0] == 200 for v in (-1, 0, 1, 20, 64)), str({v: res[v][0] for v in res}))
        err = {v: res[v][1].get("error", {}) for v in (65, 1000)}
        t.check("top_k 65 and 1000 -> 400 naming top_k and the limit 64", all(res[v][0] == 400 and err[v].get("param") == "top_k" and "top_k" in err[v].get("message", "") and "64" in err[v].get("message", "") for v in (65, 1000)), str(err))
        tk = {v: res[v][1]["choices"][0]["token_ids"] for v in (-1, 0, 1, 20, 64)}
        t.check("top_k -1 == 0 == 64 (no filter = the engine's 64 best candidates), same seed same tokens", tk[-1] == tk[0] == tk[64])
        t.check("top_k 20 differs from 64 for some seed (the parameter acts)", any(top_k(20, seed=s)[1]["choices"][0]["token_ids"] != top_k(64, seed=s)[1]["choices"][0]["token_ids"] for s in range(6, 14)))
        g = chat([{"role": "user", "content": "Write one sentence about the sea."}], temperature=0, max_tokens=24)[0][0]
        t.check("top_k 1 at temperature 1 == greedy", tk[1] == g)
        code, _ = post("/v1/chat/completions", dict(model="swift", messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0, top_k=1000))
        t.check("top_k is ignored at temperature 0 (vLLM resets it for greedy requests): top_k 1000 accepted", code == 200, str(code))
        if a.dump:
            json.dump(dump, open(a.dump, "w"))
    finally:
        srv.send_signal(signal.SIGINT)
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            srv.kill()
    bad = [n for n, ok in t.res if not ok]
    print(f"\n{len(t.res) - len(bad)}/{len(t.res)} checks passed in {time.time() - t0:.0f} s" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
