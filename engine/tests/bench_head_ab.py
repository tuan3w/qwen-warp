"""K9 A/B benchmark: quantised lm_head + exact rescoring vs the BF16 head, interleaved in ONE process (same build, same engine state; Engine.set_head toggles the head path).

  SMK_BUILD=build/k9 ~/llm-lab/venv/bin/python tests/bench_head_ab.py --head int8|int4 [--reps 5] [--what step,verify,spec] [--prompts code,prose,math,long8k] [--json out.json]
Needs the GPU lease (hold it for the whole run).  step: plain decode ms/step (ctx ~ code prompt, `smk_engine_decode(n)` replays), verify: T=2..4 head steps, spec: k=3 MTP tok/s
(greedy and sampling T=1, top_k 20, top_p .95 -- bench_spec.py's loop), `reps` interleaved repeats per mode (order alternates), medians reported.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_spec import prompts  # noqa: E402
from spec import SpecEngine  # noqa: E402


def med(v):
    return statistics.median(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="int8")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--what", default="step,verify,spec")
    ap.add_argument("--prompts", default="code,prose,math,long8k")
    ap.add_argument("--n", type=int, default=384, help="tokens per spec rep")
    ap.add_argument("--nstep", type=int, default=128)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--fast-prefill", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    what = set(a.what.split(","))
    names = a.prompts.split(",")
    P = prompts(names)
    mx = max(len(v) for v in P.values())
    e = SpecEngine(ctx_max=mx + a.n + 600, kv_fp8=a.fp8, head=a.head, verbose=False)
    fp = None
    if a.fast_prefill:
        from prefill import FastPrefill

        fp = FastPrefill(e)
    pre = (fp or e)
    res = {"head": a.head, "reps": a.reps}
    modes = ["bf16", "q"]
    print(f"engine ready (head {a.head}); engine bytes {e.lib.smk_engine_bytes(e.h) / 2**30:.2f} GiB, torch {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    def order(r):
        return modes if r % 2 == 0 else modes[::-1]

    ids = P[names[0]]
    if "step" in what or "verify" in what:
        e.set_sampling(temperature=0.0)
        step = {m: [] for m in modes}
        ver = {m: {T: [] for T in (2, 3, 4)} for m in modes}
        for r in range(a.reps):
            for m in order(r):
                e.set_head(m)
                e.reset()
                e.prefill(ids)
                e.decode(8)
                torch.cuda.synchronize()
                if "step" in what:
                    t0 = time.perf_counter()
                    e.decode(a.nstep)
                    torch.cuda.synchronize()
                    step[m].append((time.perf_counter() - t0) / a.nstep * 1e3)
                if "verify" in what:
                    for T in (2, 3, 4):
                        e._step(list(range(1000, 1000 + T)), 1)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        for _ in range(40):
                            e.lib.smk_engine_step(e.h, T, 1, e.st)
                        torch.cuda.synchronize()
                        ver[m][T].append((time.perf_counter() - t0) / 40 * 1e3)
        if "step" in what:
            res["step_ms"] = {m: step[m] for m in modes}
            print(f"plain T=1 step (ctx {len(ids)}): bf16 head {med(step['bf16']):.3f} ms ({1e3 / med(step['bf16']):.1f} tok/s) | {a.head}+rescore {med(step['q']):.3f} ms ({1e3 / med(step['q']):.1f} tok/s)"
                  f"  => {med(step['bf16']) - med(step['q']):.3f} ms, x{med(step['bf16']) / med(step['q']):.3f}   reps bf16 {[round(x, 3) for x in step['bf16']]} q {[round(x, 3) for x in step['q']]}", flush=True)
        if "verify" in what:
            res["verify_ms"] = {m: {T: ver[m][T] for T in ver[m]} for m in modes}
            for T in (2, 3, 4):
                print(f"verify T={T} head step: bf16 {med(ver['bf16'][T]):.3f} ms | {a.head}+rescore {med(ver['q'][T]):.3f} ms  => {med(ver['bf16'][T]) - med(ver['q'][T]):.3f} ms, x{med(ver['bf16'][T]) / med(ver['q'][T]):.3f}", flush=True)
    if "spec" in what:
        res["spec"] = {}
        for samp_name, samp in (("greedy", dict(temperature=0.0)), ("T1", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, seed=1))):
            e.set_sampling(**samp)
            for n in names:
                ids = P[n]
                rates = {m: [] for m in modes}
                tpc = {m: [] for m in modes}
                for r in range(a.reps if n != "long8k" else max(2, a.reps // 2)):
                    for m in order(r):
                        e.set_head(m)
                        e.generate(P[names[0]][:40], 24, k=a.k, group=2)  # graph recapture after the toggle (outside the timed region)
                        e.reset()
                        e.set_force(None)
                        pre.prefill_spec(ids, a.k)
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        while True:
                            new, ss = e.cycles(a.k, 8)
                            if ss[12] >= a.n:
                                break
                        dt = time.perf_counter() - t0
                        rates[m].append(ss[12] / dt)
                        tpc[m].append(ss[12] / ss[13])
                res["spec"][f"{samp_name}/{n}"] = dict(rates=rates, tokens_per_cycle=tpc)
                print(f"spec k={a.k} {samp_name:6s} {n:6s}: bf16 {med(rates['bf16']):6.1f} tok/s | {a.head}+rescore {med(rates['q']):6.1f} tok/s  x{med(rates['q']) / med(rates['bf16']):.3f}   "
                      f"tokens/cycle {med(tpc['bf16']):.2f} vs {med(tpc['q']):.2f}   reps bf16 {[round(x, 1) for x in rates['bf16']]} q {[round(x, 1) for x in rates['q']]}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
