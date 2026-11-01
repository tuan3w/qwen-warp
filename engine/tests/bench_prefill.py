"""K6 prefill benchmark: tok/s at several prompt lengths, time breakdown, peak memory.

  cd <repo root>; SMK_BUILD=build_k6 ~/llm-lab/venv/bin/python engine/tests/bench_prefill.py [--lens 128,1024,4096,8192] [--seg 4096,8192] [--fp8 0] [--reps 3]
tok/s = prompt tokens / wall time of FastPrefill.prefill_spec (all prompt tokens incl. the MTP KV commit and the final head=1 decode step + draft start).
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
    ap.add_argument("--lens", default="128,1024,4096,8192")
    ap.add_argument("--seg", default="4096,8192")
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2048)
    args = ap.parse_args()
    lens = [int(x) for x in args.lens.split(",")]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    all_ids = tok(raw, add_special_tokens=False)["input_ids"]
    all_ids = all_ids * (max(lens) // len(all_ids) + 1)  # timing only: repeat if the corpus is shorter than the longest prompt
    ctx = max(lens) + 64
    e = SpecEngine(ctx_max=ctx, kv_fp8=args.fp8, verbose=False)
    free0, total = torch.cuda.mem_get_info()
    print(f"engine {e.lib.smk_engine_bytes(e.h) / 2**30:.2f} GiB (+spec {e.lib.smk_spec_bytes(e.sp) / 2**30:.2f}), torch weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"device used {(total - free0) / 2**30:.2f} / {total / 2**30:.2f} GiB, kv_fp8={args.fp8}", flush=True)
    for seg in [int(x) for x in args.seg.split(",")]:
        fp = FastPrefill(e, seg=seg, rows=args.rows, min_fast=0)
        for n in lens:
            ids = all_ids[:n]
            e.reset()
            fp.prefill_spec(ids, 3)  # warmup (triton autotune for this shape, allocator)
            torch.cuda.synchronize()
            ts = []
            for _ in range(args.reps):
                e.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fp.prefill_spec(ids, 3)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            t = statistics.median(ts)
            # profiled run: per-category GPU ms
            fp.prof = {}
            e.reset()
            fp.prefill_spec(ids, 3)
            torch.cuda.synchronize()
            prof, fp.prof = fp.prof, None
            free, _ = torch.cuda.mem_get_info()
            print(f"seg {seg} N {n}: {t * 1e3:.0f} ms = {n / t:.0f} tok/s (reps {[round(x * 1e3) for x in ts]}); scratch+torch peak device used {(total - free) / 2**30:.2f} GiB "
                  f"(torch max alloc {torch.cuda.max_memory_allocated() / 2**30:.2f}, reserved {torch.cuda.memory_reserved() / 2**30:.2f} GiB)", flush=True)
            tot = sum(prof.values())
            print("   GPU ms: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(prof.items(), key=lambda kv: -kv[1])) + f"  (sum {tot:.0f})", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
