#!/usr/bin/env python
"""K1 micro-benchmark: 48 DeltaNet layers (distinct state buffers => L2 is cold per layer, like the real step) captured in one CUDA graph.
Needs the GPU lease.  Prints ms/layer, ms/48-layer step and effective GB/s on state traffic (1 read + T snapshot writes of 3.15 MB)."""
import argparse, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_deltanet import DN, CD, SL, NV, DV, DK  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libswiftmk_deltanet.so"))
ap.add_argument("--layers", type=int, default=48)
ap.add_argument("--iters", type=int, default=50)
ap.add_argument("--tag", default="")
args = ap.parse_args()
dn = DN(args.lib)
torch.manual_seed(0)
L = args.layers
bf = torch.bfloat16
mk = lambda *s, sc=1.0: (torch.randn(*s, device="cuda") * sc).to(bf)
W = [{"conv_w": mk(CD, 4, sc=0.5), "A_log": mk(48, sc=0.5), "dt_bias": mk(48), "norm_w": mk(128)} for _ in range(L)]
S = [(torch.randn(SL, NV, DV, DK, device="cuda") * 0.05) for _ in range(L)]
C = [torch.zeros(SL, CD, 3, device="cuda", dtype=bf) for _ in range(L)]
cur = torch.zeros(1, dtype=torch.int32, device="cuda")
qkv, z, a, b = mk(8, CD), mk(8, 6144), mk(8, 48), mk(8, 48)
y = torch.empty(8, 6144, device="cuda", dtype=bf)
state_mb = NV * DV * DK * 4 / 1e6

res = {}
for T in range(1, 9):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for l in range(L):
            dn.step(T, qkv, z, a, b, W[l], S[l], C[l], cur, y, stream=s.cuda_stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for l in range(L):
            dn.step(T, qkv, z, a, b, W[l], S[l], C[l], cur, y, stream=torch.cuda.current_stream().cuda_stream)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(args.iters):
        ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize()
        ts.append(ev0.elapsed_time(ev1))
    ts.sort()
    ms = ts[len(ts) // 2]
    gbs = (1 + T) * state_mb * 1e6 * L / (ms * 1e-3) / 1e9
    res[T] = ms
    print(f"T={T}: {ms:.3f} ms / {L}-layer step  = {ms / L * 1000:.1f} us/layer, state traffic {(1 + T) * state_mb:.2f} MB/layer -> {gbs:.0f} GB/s"
          f"  (min {ts[0]:.3f}, max {ts[-1]:.3f})")
print(f"ratio T=4/T=1: {res[4] / res[1]:.2f}  T=8/T=4: {res[8] / res[4]:.2f}  T=8/T=1: {res[8] / res[1]:.2f}")
