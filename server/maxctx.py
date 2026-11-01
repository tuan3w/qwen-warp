#!/usr/bin/env python
"""VRAM probe: build the Runner (engine + MTP + prefill scratch, warmup prefill of one segment) at --max-model-len CTX and report device memory.
  SMK_BUILD=... ~/llm-lab/venv/bin/python server/maxctx.py CTX [fp8] [pld]       (needs the GPU lease; run one process per size; pld = K11 9-slot ring)
"""
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from smk_runner import GenParams, Runner  # noqa: E402

ctx, fp8, pld = int(sys.argv[1]), int("fp8" in sys.argv[2:]), "pld" in sys.argv[2:]
free0, total = torch.cuda.mem_get_info()
try:
    r = Runner(ctx_max=ctx + Runner.slack(2, 3, pld), kv_fp8=fp8, k=3, group=2, pld=pld)
    # worst case: a prompt of ctx-512 tokens in segments of 4096 (the first segment sizes the scratch, later ones gather the KV history)
    import numpy as np

    ids = np.random.default_rng(1).integers(1000, 100000, size=ctx - 512).tolist()
    t0 = time.time()
    for _ in r.run(ids, GenParams(max_tokens=8)):
        pass
    dt = time.time() - t0
    free1, _ = torch.cuda.mem_get_info()
    print(f"OK ctx {ctx} kv {'fp8' if fp8 else 'bf16'}{' pld' if pld else ''}: used {(total - free1) / 2**30:.2f} GiB of {total / 2**30:.2f} (before the engine: {(total - free0) / 2**30:.2f}); {ctx - 512}-token prompt prefill {dt:.1f} s = {(ctx - 512) / dt:.0f} tok/s")
except Exception as ex:  # noqa: BLE001
    print(f"FAIL ctx {ctx} kv {'fp8' if fp8 else 'bf16'}{' pld' if pld else ''}: {type(ex).__name__}: {str(ex)[:150]}")
