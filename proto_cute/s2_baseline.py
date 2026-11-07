import torch, sys
from w4_common import *
SHAPES = {"down": (5120, 17408), "gateup": (34816, 5120), "qkv": (16384, 5120), "o": (5120, 6144)}
torch.manual_seed(0)
for name, (N, K) in SHAPES.items():
    nc = 3 if N * K < 100e6 else 2
    nbytes = N * K // 2 + N * K // 128 * 2
    ws = copies(N, K, nc)
    for T in (1, 4, 8):
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
        # correctness vs torch (natural layout copy for the reference)
        wn, sn = make_w4(N, K, seed=0)
        baseline_call(*ws[0], x, out, N, K, T)
        torch.cuda.empty_cache()
        ref = ref_gemv(x, wn, sn); err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        us = bench([lambda w=w, s=s: baseline_call(w, s, x, out, N, K, T) for w, s in ws])
        print(f"{name:7s} N={N:6d} K={K:6d} T={T}  {us:7.1f} us  {nbytes/us/1e3:6.1f} GB/s  relerr {err:.1e}", flush=True)
    del ws
