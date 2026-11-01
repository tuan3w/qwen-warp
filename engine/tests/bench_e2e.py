"""Plain-decode (T=1) tok/s of the full 64-layer CUDA graph at several context lengths.

  ~/llm-lab/venv/bin/python tests/bench_e2e.py [--ctxs 64,2048] [--n 128] [--fp8 0|1] [--nsys]  (needs the GPU lease)
Context is built by prefilling random-but-fixed token ids (T=4 chunks); tok/s = n / wall time of `smk_engine_decode(n)` (n graph replays, one sync).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctxs", default="64,2048")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--nograph", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    ctxs = [int(c) for c in args.ctxs.split(",")]
    eng = Engine(ctx_max=max(ctxs) + args.n * (args.reps + 2) + 16 + 400, kv_fp8=args.fp8)
    g = torch.Generator().manual_seed(0)
    for ctx in ctxs:
        ids = torch.randint(1000, 100000, (ctx,), generator=g).tolist()
        eng.reset()
        t0 = time.time()
        eng.prefill(ids)
        torch.cuda.synchronize()
        tp = time.time() - t0
        eng.decode(8)  # warm
        res = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.nograph:
                for _ in range(args.n):
                    eng._step([0], 1, graph=False) if False else eng.lib.smk_engine_step_nograph(eng.h, 1, 1, eng.st)
                torch.cuda.synchronize()
            else:
                eng.decode(args.n)
            dt = time.perf_counter() - t0
            res.append(dt / args.n * 1e3)
        res.sort()
        if args.verify:  # T=2..4 verify-shaped steps (head on, all rows accepted: pos advances by T), tokens fixed
            for T in (2, 3, 4):
                eng._step(list(range(1000, 1000 + T)), 1)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(40):
                    eng.lib.smk_engine_step(eng.h, T, 1, eng.st)
                torch.cuda.synchronize()
                print(f"  T={T} step at ctx {ctx}: {(time.perf_counter() - t0) / 40 * 1e3:.3f} ms", flush=True)
        print(f"ctx {ctx} (prefill {tp:.1f}s, kv_fp8={args.fp8}): {res[len(res) // 2]:.3f} ms/step = {1e3 / res[len(res) // 2]:.1f} tok/s  (reps {[round(r, 3) for r in res]})", flush=True)


if __name__ == "__main__":
    main()
