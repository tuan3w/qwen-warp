"""E0-style benchmark of on-device MTP speculative decoding: decode tok/s (greedy or, with --temperature, sampling; bf16 KV) at k = 0 (plain T=1 graph), 1, 2, 3 for the E0 prompts.

  SMK_BUILD=build/k4 ~/llm-lab/venv/bin/python tests/bench_spec.py [--prompts code,prose,math,long8k] [--ks 0,1,2,3] [--n 384] [--reps 3] [--group 8]
Run from the repo root (the 8K prompt is built from topics/*.md like E0).  Needs the GPU lease; hold it for the whole run.
Rate = tokens emitted after the prefill's first token / wall time of the decode phase (prefill excluded) = E0's 384 / (t(385) - t(1)); the last group of
cycles can overshoot 384 tokens, the overshoot is counted in both numerator and denominator.  Reports mean accepted length (tokens/cycle), per-position
acceptance P(>= i drafts accepted) (vLLM definition) and the ms/cycle breakdown from replaying the cycle parts on the final state.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spec import SpecEngine  # noqa: E402

MODEL = Path.home() / "models" / "swift-1.5-int4"
SHORT = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}


def prompts(names):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)

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
    return out


def breakdown(e, k, reps=20):
    """ms per cycle part (replays on the current state): full cycle, verify+accept, MTP stage, catch-up only."""
    res = {}
    parts = [(0, "cycle"), (1, "verify"), (2, "mtp_stage"), (3, "catchup")]
    if e.sampling:
        parts += [(4, "samp_target"), (5, "samp_draft")]
    for which, name in parts:
        e.launch_part(k, which, 3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e.launch_part(k, which, reps)
        torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t0) / reps * 1e3
    res["drafts"] = res["mtp_stage"] - res["catchup"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="code,prose,math,long8k")
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--reps-long", type=int, default=1)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--quant", default="int4clip")
    ap.add_argument("--head-quant", default="int4")
    ap.add_argument("--head-mode", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0, help="> 0: sampling graphs (K7); 0 = greedy")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--draft-sample", type=int, default=1, help="1: drafts sampled from q (draft head's filtered distribution), 0: greedy drafts (q = delta)")
    ap.add_argument("--draft-temperature", type=float, default=0.0, help="proposal q temperature (0 = the target temperature)")
    ap.add_argument("--draft-top-k", type=int, default=0)
    ap.add_argument("--draft-top-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fast-prefill", action="store_true", help="use prefill.FastPrefill (K6) for the prompt instead of prefill-by-decode-steps")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-breakdown", action="store_true")
    a = ap.parse_args()
    names = a.prompts.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    P = prompts(names)
    samp = dict(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, draft_sample=a.draft_sample, draft_temperature=a.draft_temperature, draft_top_k=a.draft_top_k, draft_top_p=a.draft_top_p, seed=a.seed)
    mx = max(len(v) for v in P.values())
    e = SpecEngine(ctx_max=mx + a.n + 600, quant=a.quant, head_quant=a.head_quant, head_mode=a.head_mode)
    e.set_sampling(**samp)
    fp = None
    if a.fast_prefill:
        from prefill import FastPrefill

        fp = FastPrefill(e)
    print(f"engine+MTP ready; sampling {samp if a.temperature > 0 else 'greedy'}; prompts " + ", ".join(f"{n}:{len(v)}" for n, v in P.items()), flush=True)
    results = {}
    # warm-up: graph capture for every k
    for k in ks:
        if k:
            e.generate(P[names[0]][:40], 24, k=k, group=2)
        else:
            e.reset(); (fp or e).prefill(P[names[0]][:40]); e.plain_sample(8) if e.sampling else e.decode(8)
    for n in names:
        ids = P[n]
        for k in ks:
            reps = a.reps_long if n == "long8k" else a.reps
            rates, stats = [], None
            for r in range(reps):
                e.reset()
                if k == 0:
                    (fp or e).prefill(ids)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    e.plain_sample(a.n) if e.sampling else e.decode(a.n)
                    dt = time.perf_counter() - t0
                    rates.append(a.n / dt)
                    stats = dict(tokens_per_cycle=1.0, hist=None, pos_acc=None)
                else:
                    e.set_force(None)
                    (fp or e).prefill_spec(ids, k)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    while True:
                        new, ss = e.cycles(k, a.group)
                        if ss[12] >= a.n:
                            break
                    dt = time.perf_counter() - t0
                    cyc = ss[13]
                    hist = ss[16:20]
                    rates.append(ss[12] / dt)
                    stats = dict(tokens_per_cycle=ss[12] / cyc, hist=hist, cycles=cyc,
                                 pos_acc=[sum(hist[i:]) / cyc for i in range(1, k + 1)], ms_per_cycle_wall=dt / cyc * 1e3)
            bd = None
            if k and not a.no_breakdown:
                bd = breakdown(e, k)
            rate = statistics.median(rates)
            results[f"{n}/k{k}"] = dict(prompt_tokens=len(ids), tok_s=rate, reps=rates, **stats, breakdown=bd)
            pa = "" if not stats["pos_acc"] else " pos-acc " + "/".join(f"{x:.2f}" for x in stats["pos_acc"])
            bs = "" if not bd else "  ms: cycle {cycle:.2f} verify {verify:.2f} catchup {catchup:.2f} drafts {drafts:.2f}".format(**bd)
            if bd and "samp_target" in bd:
                bs += "  sampling kernels: target rows {samp_target:.3f} drafts {samp_draft:.3f}".format(**bd)
            print(f"{n:6s} k={k}: {rate:6.1f} tok/s (reps {[round(x, 1) for x in rates]})  tokens/cycle {stats['tokens_per_cycle']:.2f}{pa}{bs}", flush=True)
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
