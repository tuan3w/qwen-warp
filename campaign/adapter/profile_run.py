#!/usr/bin/env python3
"""profile_run.py <tree> <prompt>: nsys target of `try --profile` (shipped greedy cycles = k=3 + prompt-lookup tail on one dev prompt; the capture window covers 32 cycles)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import hlib  # noqa: E402
import metric  # noqa: E402

tree, name = sys.argv[1], sys.argv[2]
P = hlib.jload(hlib.PRIV / "dev_prompts.json")
e, fp = metric.load_engine(Path(tree), max(len(P[name]), 1500) + 1200)  # the warm-up runs 1500 tokens of the copy prompt
metric.warmup(e, fp, P["code"], P["edit"])
e.reset()
first = fp.prefill_spec(P[name], 3)
metric.set_pld(e, True)
e.pld_begin(list(P[name]) + [first])
e.cycles_pld(8)
torch.cuda.synchronize()
e0 = e._ss[12]
c0 = e._ss[13]
torch.cuda.cudart().cudaProfilerStart()
t0 = metric.time.perf_counter()
for _ in range(16):
    e.cycles_pld(2)
torch.cuda.synchronize()
dt = metric.time.perf_counter() - t0
print(f"PROFILED_CYCLES 32 {1e3 * dt / 32:.3f} ms/cycle {(e._ss[12] - e0) / max(1, e._ss[13] - c0):.3f}", flush=True)
torch.cuda.cudart().cudaProfilerStop()
