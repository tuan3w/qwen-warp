"""K10 short-prompt TTFT benchmark: wall time of the whole prefill (prompt -> first token + drafts ready, MTP KV included) at several short prompt lengths.

  cd <repo root>; SMK_BUILD=build_k10 ~/llm-lab/venv/bin/python engine/tests/bench_short_ttft.py [--lens 8,20,40,64,128,256,512] [--modes slow,fast,w4] [--reps 5]
modes:  slow = engine prefill-by-decode (T=4 graph chunks);  fast = K6 FastPrefill (dequant + cuBLAS, min_fast=0);  w4 = FastPrefill with the K10 W4 mma GEMM below `w4_max` rows
        (chunks of 64 rows; `--w4_max N`), cuBLAS above;  cur = FastPrefill with the shipped defaults (what the server runs).
time = median of reps of  (reset, prefill_spec(ids, 3), sync)  = request -> first token on the engine side (no tokenizer, no HTTP).
Prompt text = topics/*.md (real tokens).  Needs the GPU lease for the whole run.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

ENG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(ENG / "tests"))
from spec import SpecEngine  # noqa: E402
from prefill import FastPrefill  # noqa: E402

MODEL = Path.home() / "models" / "swift-1.5-int4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default="8,20,40,64,128,256,512")
    ap.add_argument("--modes", default="slow,fast,cur")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--w4_max", type=int, default=4096)
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--prof", type=int, default=0)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    all_ids = tok(raw, add_special_tokens=False)["input_ids"]
    e = SpecEngine(ctx_max=max(lens) + 128, kv_fp8=args.fp8, verbose=False)
    res = {}
    for mode in args.modes.split(","):
        if mode == "slow":
            fp = FastPrefill(e, min_fast=10**9)
        elif mode == "fast":
            fp = FastPrefill(e, min_fast=0, w4_max=0)
        elif mode == "w4":
            fp = FastPrefill(e, min_fast=0, w4_max=args.w4_max)
        elif mode == "cur":
            fp = FastPrefill(e)
        else:
            raise SystemExit(mode)
        for n in lens:
            ids = all_ids[1000 : 1000 + n]
            e.reset()
            fp.prefill_spec(ids, 3)  # warmup (autotune, graphs, allocator)
            torch.cuda.synchronize()
            ts = []
            for r in range(args.reps):
                e.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fp.prefill_spec(ids, 3)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[(mode, n)] = statistics.median(ts) * 1e3
            line = f"{mode:5s} N {n:4d}: {res[(mode, n)]:7.1f} ms (reps {[round(x * 1e3, 1) for x in ts]})"
            if args.prof and mode != "slow":
                fp.prof = {}
                e.reset()
                fp.prefill_spec(ids, 3)
                torch.cuda.synchronize()
                prof, fp.prof = fp.prof, None
                line += "\n      GPU ms: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(prof.items(), key=lambda kv: -kv[1])) + f" (sum {sum(prof.values()):.1f})"
            print(line, flush=True)
    print("\n| prompt tokens | " + " | ".join(args.modes.split(",")) + " |")
    print("|---|" + "---|" * len(args.modes.split(",")))
    for n in lens:
        print(f"| {n} | " + " | ".join(f"{res[(m, n)]:.0f} ms" for m in args.modes.split(",")) + " |")


if __name__ == "__main__":
    main()
