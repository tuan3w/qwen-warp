#!/usr/bin/env python
"""Sampler exactness on REAL logits.  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4).

  python tests/test_sampler_dist.py [--draws 20000]

Teacher-forces the greedy continuation of three prompts (a tool call with thinking on, code, a list of numbers; the server's sampling defaults: temperature 1, top_k 20,
top_p 0.95) through the engine's sampling head (quantised head + exact bf16 rescoring of 64 candidates) and, at every continuation position, compares the engine's emitted-token
distribution (N draws through the device sampler, smk_test_draw) with the torch reference: the exact bf16 lm_head on the same final hidden state, temperature -> top_k -> top_p.
Per position: no draw outside the reference support and chi-square over the reference tokens (expected count >= 5, rest pooled) below the 1e-6 critical value.  Gates the
positions of the kinds that matter (prose, code, digits, `<tool_call>\\n`, `</think>`, `<|im_end|>`) and reports the total variation distance.  Negative control: the same
draws against a reference at temperature 0.95 must be rejected at some position, otherwise the test could not fail.  This is the measurement behind "the sampler is not why
tool calls are malformed" (tests/test_tool_call_rate.py): the engine samples the distribution the bf16 head defines.
"""
import argparse
import math
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
T, TOP_K, TOP_P = 1.0, 20, 0.95
Z_CRIT = 4.75  # 1e-6 one-sided


def chi2_crit(df, z=Z_CRIT):  # Wilson-Hilferty approximation of the chi-square quantile
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def processed(logits, temperature, top_k, top_p):
    lg = logits.double() / temperature
    v, ix = torch.topk(lg, top_k)
    pr = torch.softmax(v, 0)
    keep = (torch.cumsum(pr, 0) - pr) < top_p  # keep token i iff the mass before it is < top_p (at least the first)
    keep[0] = True
    return ix[keep].cpu(), (pr[keep] / pr[keep].sum()).cpu()


def compare(draws, ids, p):
    """-> (draws outside the support, chi2, df, tv); draws: int tensor of N tokens, (ids, p): reference support and probabilities."""
    n = len(draws)
    uniq, cnt = torch.unique(draws.long(), return_counts=True)
    count = dict(zip(uniq.tolist(), cnt.tolist()))
    support = set(ids.tolist())
    outside = sum(c for t, c in count.items() if t not in support)
    obs, exp, pool_o, pool_e = [], [], 0.0, 0.0
    tv = 0.5 * outside / n
    for t, pr in zip(ids.tolist(), p.tolist()):
        o = count.get(t, 0)
        tv += 0.5 * abs(o / n - pr)
        if pr * n >= 5:
            obs.append(o), exp.append(pr * n)
        else:
            pool_o, pool_e = pool_o + o, pool_e + pr * n
    if pool_e >= 5:
        obs.append(pool_o), exp.append(pool_e)
    chi2 = sum((o - e) ** 2 / e for o, e in zip(obs, exp))
    return outside, chi2, max(len(obs) - 1, 0), tv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=20000)
    a = ap.parse_args()
    sys.path.insert(0, str(ENGINE))
    import pack
    from spec import SpecEngine
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    V = pack.V
    e = SpecEngine(ctx_max=2048, verbose=False)
    lm = e.G.lm_head
    cases = {
        "tool": dict(messages=[{"role": "user", "content": "What is the weather in Hanoi?"}], tools=TOOLS, n=140),
        "code": dict(messages=[{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number using memoisation."}], n=100, enable_thinking=False),
        "numbers": dict(messages=[{"role": "user", "content": "List the first 12 prime numbers separated by commas, then their sum."}], n=60, enable_thinking=False),
    }
    IM_END, THINK_END, TC = (tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "</think>", "<tool_call>"))
    rows = []  # (case, kind, outside, chi2, df, tv, chi2 against the T = 0.95 reference, df of it, entropy)
    for name, c in cases.items():
        kw = {k: v for k, v in c.items() if k not in ("messages", "n")}
        prompt = tok.apply_chat_template(c["messages"], add_generation_prompt=True, tokenize=False, **kw)
        ids = tok(prompt, add_special_tokens=False)["input_ids"]
        e.set_sampling(0.0)
        gen, _ = e.generate(ids, c["n"], k=3, eos={248046, 248044})
        e.set_sampling(T, TOP_K, TOP_P, seed=20241104)
        e.reset()
        feed = ids + gen[:-1]  # row of the position p predicts feed[p + 1]
        for s in range(0, len(ids) - 1, 4):
            e._step(feed[s:min(s + 4, len(ids) - 1)], 0)
        start = len(ids) - 1
        for s in range(start, len(feed), 4):
            chunk = feed[s:s + 4]
            e._step(chunk, 1)
            hid = e.dbg_hidden(len(chunk))[65]
            for r in range(len(chunk)):
                pos = s + r
                nxt = (ids + gen)[pos + 1]
                h = hid[r].to(torch.bfloat16).float().cuda()
                lg = torch.cat([torch.mv(lm[i:i + 31040].float(), h) for i in range(0, V, 31040)])
                ref_ids, ref_p = processed(lg, T, TOP_K, TOP_P)
                draws = e.test_draw(a.draws, ctr0=pos * a.draws, row=r)
                outside, chi2, df, tv = compare(draws, ref_ids, ref_p)
                ids95, p95 = processed(lg, 0.95, TOP_K, TOP_P)
                _, chi2_95, df95, _ = compare(draws, ids95, p95)
                kind = ("<|im_end|>" if nxt == IM_END else "</think>" if nxt == THINK_END else "tool_call-newline" if (feed[pos] == 198 and feed[pos - 1] == TC) else
                        "digit" if tok.decode([nxt]).strip().isdigit() else name)
                ent = float(-(ref_p * ref_p.clamp_min(1e-30).log()).sum())
                rows.append((name, kind, outside, chi2, df, tv, chi2_95, df95, ent))
    fails, kinds = [], {}
    for name, kind, outside, chi2, df, tv, _, _, ent in rows:
        bad = outside > 0 or (df > 0 and chi2 > chi2_crit(df))
        kinds.setdefault(kind, []).append((tv, ent, bad))
        if bad:
            fails.append(f"{name}/{kind}: {outside} draws outside the support, chi2 {chi2:.1f} (df {df}, critical {chi2_crit(df) if df else 0:.1f})")
    print(f"positions {len(rows)}, {a.draws} draws each")
    for kind, v in sorted(kinds.items()):
        print(f"  {kind:18s} positions {len(v):3d}  mean TV {sum(x[0] for x in v) / len(v):.4f}  max TV {max(x[0] for x in v):.4f}  max entropy {max(x[1] for x in v):.2f} nats  rejected {sum(x[2] for x in v)}")
    want = {"digit", "tool_call-newline", "</think>", "<|im_end|>", "code", "tool"}
    missing = want - set(kinds)
    print(f"{'PASS' if not fails else 'FAIL'}  engine draws == torch bf16-head reference (temperature {T}, top_k {TOP_K}, top_p {TOP_P}) at every position" + (f"  [{fails[:3]}]" if fails else ""))
    print(f"{'PASS' if not missing else 'FAIL'}  every kind of position covered" + (f"  [missing {sorted(missing)}]" if missing else ""))
    rej = [r for r in rows if r[7] > 0 and r[6] > chi2_crit(r[7])]
    print(f"{'PASS' if rej else 'FAIL'}  negative control: draws vs a temperature-0.95 reference are rejected at {len(rej)} position(s) (the test can fail)")
    ok = not fails and not missing and bool(rej)
    print("SAMPLER DIST " + ("OK" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
