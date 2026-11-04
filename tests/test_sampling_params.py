#!/usr/bin/env python
"""End-to-end test of the supported sampling parameters through the OpenAI server: presence_penalty, frequency_penalty, repetition_penalty, min_p, logit_bias, min_tokens, allowed_token_ids
(+ validation 400s and the parameters that stay unsupported).  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4).

  python tests/test_sampling_params.py [--port 8043] [--profile default|long] [--keep]

Every assertion compares against a measured baseline of the same request without the parameter, so it can fail: a processor that is not applied, applied to the wrong tokens, or leaking into the next request is caught.
Semantics (vLLM v1 sampler order): allowed mask -> min_tokens -> logit_bias -> repetition (prompt + generated) -> frequency / presence (generated) -> temperature -> min_p -> top-k / top-p.
"""
import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "server" / "serve.py"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
EOS = [248046, 248044]
NOTHINK = dict(chat_template_kwargs={"enable_thinking": False}, return_token_ids=True)
LOOP = "the cat sat on the mat. " * 8 + "the cat sat on the"  # a raw completion prompt that a sampled continuation keeps repeating
LIST = "".join(f"{i} " for i in range(1, 11)) * 5 + "1 2 3"
STORY = "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships."


class T:
    def __init__(self):
        self.res = []

    def check(self, name, ok, detail=""):
        self.res.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)


def distinct(ids):
    return len(set(ids)) / max(len(ids), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8043)
    ap.add_argument("--keep", action="store_true", help="leave the server running")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--profile", default="default", choices=["default", "long"])
    a = ap.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    log = open(f"/tmp/swift_mk_params_server_{a.port}.log", "w")
    t0 = time.time()
    srv = subprocess.Popen([sys.executable, str(SERVE), "--port", str(a.port), "--max-model-len", str(a.max_model_len), "--profile", a.profile], env=env, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    from openai import OpenAI
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = f"http://127.0.0.1:{a.port}"
    t = T()
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit(f"server died, see {log.name}")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        t.check("server up", True, f"started in {time.time() - t0:.1f} s")
        cl = OpenAI(base_url=base + "/v1", api_key="x", timeout=300)

        def chat(text, **kw):
            body = dict(NOTHINK, **kw.pop("extra_body", {}))
            return cl.chat.completions.create(model="swift", messages=[{"role": "user", "content": text}], temperature=kw.pop("temperature", 0), max_tokens=kw.pop("max_tokens", 64), extra_body=body, **kw)

        def comp(prompt, **kw):
            return cl.completions.create(model="swift", prompt=prompt, temperature=kw.pop("temperature", 0), max_tokens=kw.pop("max_tokens", 64), extra_body=dict(return_token_ids=True, **kw.pop("extra_body", {})), **kw)

        def ids_of(o, i=0):
            return o.choices[i].token_ids if hasattr(o.choices[i], "token_ids") else o.choices[i].model_extra["token_ids"]

        def post(path, **body):
            r = httpx.post(base + path, json=dict(model="swift", **body), timeout=120)
            return r.status_code, r.json()

        def one_tok(s):
            ids = tok(s, add_special_tokens=False)["input_ids"]
            assert len(ids) == 1, (s, ids)
            return ids[0]

        # ---------------------------------------------------------------------------------------- accepted / rejected
        for name, val in (("presence_penalty", 0.5), ("frequency_penalty", 0.5), ("repetition_penalty", 1.2), ("min_p", 0.1), ("min_tokens", 2), ("logit_bias", {"1000": 1.0}), ("allowed_token_ids", [1000, 1001, 248046])):
            code, body = post("/v1/chat/completions", messages=[{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.7, chat_template_kwargs={"enable_thinking": False}, **{name: val})
            t.check(f"{name} is accepted (HTTP 200)", code == 200, f"HTTP {code} {str(body)[:200]}")
        unsupported = {"logprobs": dict(logprobs=True, top_logprobs=2), "bad_words": dict(bad_words=["hello"]), "thinking_token_budget": dict(thinking_token_budget=10),
                       "structured_outputs": dict(structured_outputs={"regex": "a+"})}
        for name, kw in unsupported.items():
            code, body = post("/v1/chat/completions", messages=[{"role": "user", "content": "hi"}], max_tokens=4, **kw)
            t.check(f"unsupported {name} -> HTTP 400 naming the parameter", code == 400 and name in json.dumps(body), f"HTTP {code} {str(body)[:200]}")

        # ---------------------------------------------------------------------------------------- baselines
        greedy_a = chat(STORY, max_tokens=48)
        base_ids = ids_of(greedy_a)

        # ---------------------------------------------------------------------------------------- presence / frequency: repetition on a looping prompt, n = 8 runs each
        runs = {}
        for label, kw in (("none", {}), ("presence 1.0", dict(presence_penalty=1.0)), ("frequency 1.0", dict(frequency_penalty=1.0)), ("presence -1.0", dict(presence_penalty=-1.0))):
            o = comp(LOOP, temperature=0.7, max_tokens=96, n=8, extra_body=dict(ignore_eos=True), **kw)
            runs[label] = [distinct(ids_of(o, i)) for i in range(8)]
        mean = {k: sum(v) / len(v) for k, v in runs.items()}
        for label in ("presence 1.0", "frequency 1.0"):
            t.check(f"{label.split()[0]}_penalty reduces repetition (distinct-token ratio of 8 runs: mean +0.1, at least 6 runs above the unpenalised mean)", mean[label] >= mean["none"] + 0.1 and sum(x > mean["none"] for x in runs[label]) >= 6,
                    f"none {mean['none']:.3f}, {label} {mean[label]:.3f} (min {min(runs[label]):.3f})")
        t.check("negative presence_penalty increases repetition (mean distinct ratio below 0.9 x the unpenalised runs)", mean["presence -1.0"] < 0.9 * mean["none"], f"{mean['presence -1.0']:.3f} vs {mean['none']:.3f}")

        # ---------------------------------------------------------------------------------------- repetition_penalty on the prompt
        g0 = ids_of(comp(LIST, max_tokens=24))
        g1 = ids_of(comp(LIST, max_tokens=24, extra_body=dict(repetition_penalty=3.0)))
        text0 = tok.decode(g0)
        t.check("repeated-list prompt: the unpenalised greedy continuation repeats the list", text0.startswith("4 5 6 7 8 9 10") or text0.startswith(" 4 5 6 7 8 9 10"), repr(text0[:40]))
        t.check("repetition_penalty 3 changes the repeated-list continuation (prompt tokens are penalised)", g0 != g1 and g0[:8] != g1[:8], f"{tok.decode(g1)[:50]!r}")
        g_one = ids_of(comp(LIST, max_tokens=24, extra_body=dict(repetition_penalty=1.0)))
        t.check("repetition_penalty 1.0 is the identity", g_one == g0)

        # ---------------------------------------------------------------------------------------- min_p
        n_eq = {}
        for mp in (0.0, 0.99):
            o = chat(STORY, temperature=1.0, max_tokens=48, n=8, extra_body=dict(min_p=mp, top_k=20))
            n_eq[mp] = [sum(x == y for x, y in zip(ids_of(o, i), base_ids)) / len(base_ids) for i in range(8)]
        t.check("min_p 0.99 behaves like greedy (mean token agreement with the greedy answer >= 0.95, min_p 0 well below)", sum(n_eq[0.99]) / 8 >= 0.95 and sum(n_eq[0.0]) / 8 < 0.6,
                f"min_p 0.99: {sum(n_eq[0.99]) / 8:.3f}, min_p 0: {sum(n_eq[0.0]) / 8:.3f}")

        # ---------------------------------------------------------------------------------------- logit_bias
        the = one_tok(" the")
        for temp in (0.0, 0.8):
            o = chat(STORY, temperature=temp, max_tokens=24, extra_body=dict(logit_bias={str(the): 100}, top_k=20))
            t.check(f"logit_bias +100 forces the token (temperature {temp}, all 24 tokens)", ids_of(o) == [the] * 24, str(ids_of(o)[:6]))
        first = base_ids[0]
        o = chat(STORY, max_tokens=48, extra_body=dict(logit_bias={str(first): -100}))
        t.check("logit_bias -100 on the greedy first token changes the greedy output and removes the token", ids_of(o)[0] != first and first not in ids_of(o), f"{ids_of(o)[:4]} vs {base_ids[:4]}")
        o = chat(STORY, max_tokens=48, extra_body=dict(logit_bias={str(first): 0.0}))
        t.check("logit_bias 0 is the identity", ids_of(o) == base_ids)

        # ---------------------------------------------------------------------------------------- min_tokens
        short = chat("Say hi.", max_tokens=100)
        n_short = short.usage.completion_tokens
        t.check("baseline: 'Say hi.' ends early", n_short < 20 and ids_of(short)[-1] in EOS, f"{n_short} tokens")
        for temp in (0.0, 0.8):
            o = chat("Say hi.", temperature=temp, max_tokens=60, extra_body=dict(min_tokens=25, top_k=20))
            ids = ids_of(o)
            t.check(f"min_tokens 25 delays the stop token (temperature {temp})", len(ids) >= 25 and not set(ids[:24]) & set(EOS) and (len(ids) == 60 or ids[-1] in EOS), f"{len(ids)} tokens, stop tokens in the first 24: {set(ids[:24]) & set(EOS)}")

        # ---------------------------------------------------------------------------------------- allowed_token_ids
        digits = [one_tok(str(d)) for d in range(10)]
        allowed = digits + [one_tok(" "), EOS[0]]
        outs = [chat(STORY, max_tokens=48, extra_body=dict(allowed_token_ids=allowed))]  # greedy: n must be 1
        outs.append(chat(STORY, temperature=1.0, max_tokens=48, n=4, extra_body=dict(allowed_token_ids=allowed, top_k=20)))
        toks = [x for o in outs for i in range(len(o.choices)) for x in ids_of(o, i)]
        t.check("allowed_token_ids restricts every generated token to the set (greedy and 4 sampled runs)", len(toks) > 20 and all(x in allowed for x in toks), f"{sum(x not in allowed for x in toks)} of {len(toks)} tokens outside the set")
        o = chat(STORY, temperature=0.8, max_tokens=16, extra_body=dict(allowed_token_ids=[the], top_k=20))
        t.check("allowed_token_ids with a single token forces it", ids_of(o) == [the] * 16)
        rnd = random.Random(5).sample(range(1000, 240000), 10000) + EOS
        o = chat(STORY, temperature=1.0, max_tokens=48, n=4, extra_body=dict(allowed_token_ids=rnd, top_k=20))
        sset = set(rnd)
        t.check("allowed_token_ids with 10,000 tokens restricts every token", all(x in sset for i in range(4) for x in ids_of(o, i)), f"{sum(len(ids_of(o, i)) for i in range(4))} tokens")
        o = chat(STORY, max_tokens=48, extra_body=dict(allowed_token_ids=list(range(248320))))
        t.check("allowed_token_ids = the whole vocabulary is the identity", ids_of(o) == base_ids)

        # ---------------------------------------------------------------------------------------- temperature 0 + processors, determinism, isolation
        chat("Unrelated: what is 16*23?", max_tokens=8)  # both penalised runs cold
        pa = ids_of(chat(STORY, max_tokens=48, extra_body=dict(presence_penalty=2.0, frequency_penalty=1.0)))
        chat("Unrelated: what is 17*23?", max_tokens=8)  # makes both runs cold
        pb = ids_of(chat(STORY, max_tokens=48, extra_body=dict(presence_penalty=2.0, frequency_penalty=1.0)))
        t.check("temperature 0 with penalties is deterministic and differs from the plain greedy answer", pa == pb and pa != base_ids, f"differs at index {next((i for i, (x, y) in enumerate(zip(pa, base_ids)) if x != y), None)}")
        after = ids_of(chat(STORY, max_tokens=48))
        t.check("no leakage: the next request without parameters equals the baseline greedy answer", after == base_ids)
        chat("Unrelated: what is 18*23?", max_tokens=8)  # the first seeded run must be cold like the second (a resumed prefill is not numerically identical to a cold one)
        s1 = chat(STORY, temperature=0.8, max_tokens=48, seed=7, extra_body=dict(top_k=20, presence_penalty=0.5, min_p=0.02))
        chat("Unrelated: what is 19*23?", max_tokens=8)
        s2 = chat(STORY, temperature=0.8, max_tokens=48, seed=7, extra_body=dict(top_k=20, presence_penalty=0.5, min_p=0.02))
        s3 = chat(STORY, temperature=0.8, max_tokens=48, seed=8, extra_body=dict(top_k=20, presence_penalty=0.5, min_p=0.02))
        t.check("sampling with processors: same seed reproducible (cold / cold), other seed differs", ids_of(s1) == ids_of(s2) and ids_of(s1) != ids_of(s3))

        # ---------------------------------------------------------------------------------------- prefix reuse: the prompt-token set is rebuilt on a resumed request
        doc = "".join(f"Fact {i}: the quokka number {i} lives on island {i * 7} and eats {i + 3} leaves. " for i in range(60))
        msgs = [{"role": "user", "content": doc + "Question 1: how many quokkas are listed above?"}]
        cl.chat.completions.create(model="swift", messages=msgs, temperature=0, max_tokens=24, extra_body=NOTHINK)
        msgs2 = msgs + [{"role": "assistant", "content": "There are sixty."}, {"role": "user", "content": "Question 2: which island has the quokka number 7? Repeat the fact exactly."}]
        kw = dict(model="swift", messages=msgs2, temperature=0, max_tokens=40, extra_body=dict(NOTHINK, repetition_penalty=1.3, frequency_penalty=0.3))
        warm = cl.chat.completions.create(**kw)
        cached = warm.usage.prompt_tokens_details.cached_tokens if warm.usage.prompt_tokens_details else 0
        cl.chat.completions.create(model="swift", messages=[{"role": "user", "content": "Unrelated: what is 21*23?"}], max_tokens=8, temperature=0, extra_body=NOTHINK)
        cold = cl.chat.completions.create(**kw)
        plain = cl.chat.completions.create(model="swift", messages=msgs2, temperature=0, max_tokens=40, extra_body=NOTHINK)
        w, c, p = ids_of(warm), ids_of(cold), ids_of(plain)
        fd = next((i for i in range(min(len(w), len(c))) if w[i] != c[i]), None)
        # warm vs cold are not bit-identical (resumed T=1 head step vs batched prefill), and a penalty on a prompt that is repeated flattens the top of the distribution: report the agreement, assert the effect
        t.check("resumed request (cached_tokens > 0) applies the penalties to the cached prompt part: the penalised warm answer differs from the plain warm answer", cached > 0 and w != p,
                f"cached {cached}; penalised warm vs cold: first difference {fd}, plain vs penalised first difference {next((i for i in range(min(len(w), len(p))) if w[i] != p[i]), None)}")
        t.check("the repetition penalty changes the answer of that prompt (so the check above is not vacuous)", c != p, f"{tok.decode(c)[:50]!r} vs {tok.decode(p)[:50]!r}")

        # ---------------------------------------------------------------------------------------- validation
        def bad(name, param, **kw):
            code, body = post("/v1/chat/completions", messages=[{"role": "user", "content": "hi"}], max_tokens=8, chat_template_kwargs={"enable_thinking": False}, **kw)
            msg = json.dumps(body)
            t.check(f"validation: {name} -> HTTP 400 naming {param}", code == 400 and param in msg, f"HTTP {code} {msg[:160]}")

        bad("presence_penalty 3", "presence_penalty", presence_penalty=3.0)
        bad("frequency_penalty -2.5", "frequency_penalty", frequency_penalty=-2.5)
        bad("repetition_penalty 0", "repetition_penalty", repetition_penalty=0.0)
        bad("min_p 1.5", "min_p", min_p=1.5)
        bad("min_tokens above max_tokens", "min_tokens", min_tokens=20)
        bad("logit_bias with an out-of-vocabulary id", "logit_bias", logit_bias={"300000": 1.0})
        bad("logit_bias with a non-integer key", "logit_bias", logit_bias={"abc": 1.0})
        bad("logit_bias with 1025 entries", "logit_bias", logit_bias={str(i): 1.0 for i in range(1025)})
        bad("empty allowed_token_ids", "allowed_token_ids", allowed_token_ids=[])
        bad("allowed_token_ids with an out-of-vocabulary id", "allowed_token_ids", allowed_token_ids=[5, 999999])
        bad("allowed_token_ids made of stop tokens only + min_tokens", "min_tokens", allowed_token_ids=EOS, min_tokens=3)
        code, _ = post("/v1/chat/completions", messages=[{"role": "user", "content": "hi"}], max_tokens=8, logit_bias={str(i): 1.0 for i in range(1024)}, chat_template_kwargs={"enable_thinking": False})
        t.check("logit_bias with exactly 1024 entries is accepted", code == 200, f"HTTP {code}")
    finally:
        if not a.keep:
            srv.send_signal(signal.SIGINT)
            try:
                srv.wait(30)
            except subprocess.TimeoutExpired:
                srv.kill()
    bad_ = [n for n, ok in t.res if not ok]
    print(f"\n{len(t.res) - len(bad_)}/{len(t.res)} checks passed" + (f"; FAILED: {bad_}" if bad_ else ""))
    sys.exit(1 if bad_ else 0)


if __name__ == "__main__":
    main()
