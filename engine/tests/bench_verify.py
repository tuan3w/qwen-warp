"""K11 step 0: verify-step time V(T), T = 1..8 rows, of the real 64-layer CUDA graph (K9 int4 head + exact R=16 rescoring, DeltaNet ring of 9 slots, attention T>4, W4 mma GEMV).

  SMK_BUILD=build/k11 ~/llm-lab/venv/bin/python tests/bench_verify.py [--ctxs 64,2048,8192] [--reps 7] [--fp8 0] [--head int4]   (needs the GPU lease)
V(T) = median over `reps` blocks of n replays of smk_engine_step(T, head=1) (all rows "accepted": pos advances by T per replay, so the context grows by n*T/2 on average; n is small).
The T order is shuffled per rep, so clock / thermal drift is spread over all T.  Prints a markdown table (ms) and the increments V(T) - V(T-1).
"""
import argparse
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="64,2048,8192")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--head", default=None)
    ap.add_argument("--tmax", type=int, default=8)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctxs.split(",")]
    Ts = list(range(1, args.tmax + 1))
    eng = Engine(ctx_max=max(ctxs) + args.n * 8 * (args.reps + 2) + 64, kv_fp8=args.fp8, head=args.head, tmax=args.tmax)
    g = torch.Generator().manual_seed(0)
    rows = {}
    for ctx in ctxs:
        ids = torch.randint(1000, 100000, (ctx,), generator=g).tolist()
        eng.reset()
        t0 = time.time()
        eng.prefill(ids)
        torch.cuda.synchronize()
        print(f"ctx {ctx}: prefill {time.time() - t0:.1f} s", flush=True)
        for T in Ts:  # capture every graph and warm up
            eng._step(list(range(1000, 1000 + T)), 1)
            for _ in range(3):
                eng.lib.smk_engine_step(eng.h, T, 1, eng.st)
        torch.cuda.synchronize()
        res = {T: [] for T in Ts}
        rng = random.Random(1)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for rep in range(args.reps):
            order = Ts[:]
            rng.shuffle(order)
            for T in order:
                ev0.record()
                for _ in range(args.n):
                    eng.lib.smk_engine_step(eng.h, T, 1, eng.st)
                ev1.record()
                ev1.synchronize()
                res[T].append(ev0.elapsed_time(ev1) / args.n)
        med = {T: sorted(v)[len(v) // 2] for T, v in res.items()}
        rows[ctx] = (med, res)
    print(f"\nV(T) ms, plain verify step incl. head ({eng.head_kind}), kv_fp8={args.fp8}, tmax={args.tmax} (median of {args.reps} x {args.n} replays; [min-max] spread)")
    print("| ctx | " + " | ".join(f"T={T}" for T in Ts) + " |")
    print("|---|" + "---|" * len(Ts))
    for ctx in ctxs:
        med, res = rows[ctx]
        print(f"| {ctx} | " + " | ".join(f"{med[T]:.3f}" for T in Ts) + " |")
    print("increments V(T)-V(T-1):")
    for ctx in ctxs:
        med, _ = rows[ctx]
        print(f"| {ctx} | " + " | ".join(f"{med[T] - med[T - 1]:+.3f}" for T in Ts[1:]) + " |")
    print("spread (max-min over reps):")
    for ctx in ctxs:
        _, res = rows[ctx]
        print(f"| {ctx} | " + " | ".join(f"{max(res[T]) - min(res[T]):.3f}" for T in Ts) + " |")
    if args.json:
        import json

        json.dump({str(c): {str(T): rows[c][0][T] for T in Ts} for c in ctxs}, open(args.json, "w"))


if __name__ == "__main__":
    main()
