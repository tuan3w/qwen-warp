import sys, torch
from w4_common import *
import cute_w4
SHAPES = {"down": (5120, 17408), "gateup": (34816, 5120), "qkv": (16384, 5120), "o": (5120, 6144)}
CFGS = [dict(NW=8, KS=4, U=2, grid=164), dict(NW=8, KS=4, U=3, grid=164), dict(NW=8, KS=8, U=2, grid=164), dict(NW=8, KS=2, U=3, grid=164), dict(NW=16, KS=8, U=2, grid=82), dict(NW=8, KS=4, U=2, grid=246)]
for name, (N, K) in SHAPES.items():
    nb = N * K // 2 + N * K // 128 * 2
    ws = copies(N, K, 3 if N * K < 100e6 else 2)
    for T in (1, 8):
        x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
        o_ref = torch.empty(T, N, device="cuda", dtype=torch.bfloat16); o = torch.zeros_like(o_ref)
        baseline_call(*ws[0], x, o_ref, N, K, T)
        us0 = bench([lambda w=w, s=s: baseline_call(w, s, x, o_ref, N, K, T) for w, s in ws])
        baseline_call(*ws[0], x, o_ref, N, K, T)
        res = []
        for cfg in CFGS:
            try:
                fn = cute_w4.get(N, K, T, ws[0][0], ws[0][1], x, o, **cfg)
            except AssertionError: continue
            o.zero_(); fn(*cute_w4.wrap(ws[0][0], ws[0][1], x, o), cute_w4.cur()); torch.cuda.synchronize()
            d = (o.float() - o_ref.float()).abs().max().item()
            fns = []
            for w, s in ws:
                a = cute_w4.wrap(w, s, x, o); fns.append(lambda a=a: fn(*a, cute_w4.cur()))
            res.append((bench(fns), d, cfg)); assert d < 5e-3 * o_ref.float().abs().max().item(), (cfg, d)
        best = min(res, key=lambda r: r[0])
        print(f"{name:7s} T={T}  cuda {us0:6.1f}us {nb/us0/1e3:5.0f}GB/s | cute best {best[0]:6.1f}us {nb/best[0]/1e3:5.0f}GB/s diff {best[1]} {best[2]} | all: " + " ".join(f"{r[0]:.1f}" for r in res), flush=True)
    del ws; torch.cuda.empty_cache()
