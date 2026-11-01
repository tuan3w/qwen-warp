"""K2 attention micro-benchmark: a CUDA graph of 16 layers (distinct KV caches, so every layer streams from DRAM like a real step).

    ~/llm-lab/venv/bin/python tests/bench_attn.py [--prof]   (take the GPU lease around it)
Prints ms/layer, KV GB/s (live KV bytes read / time), and the 16-layer step sum for ctx 2K/8K/32K, T=1/4, fp8 and bf16 KV.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_attn as ta  # noqa: E402

NL = 16


def build(ctx, T, fp8, layers=NL):
    w, zs = ta.real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = ta.real_events(z, L)
    ctx_max = ctx + 8
    ws = torch.empty(ta.lib().swiftmk_attn_ws_bytes(), dtype=torch.uint8, device="cuda")
    pos0 = ctx - T
    dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
    Ls = []
    src = torch.randn(4, 4096, 256, device="cuda")
    for i in range(layers):
        ly = ta.Layer(w["qn"], w["kn"], w["ks"], w["vs"], ctx_max, fp8, ws)
        # realistic-scale random fill without per-layer big temporaries
        for kvi in range(2):
            for off in range(0, pos0, 4096):
                n = min(4096, pos0 - off)
                ly.kv[kvi, :, off : off + n] = ((src[:, :n] * (0.8 if kvi == 0 else 0.3) + i * 0.01) / (ly.ks if kvi == 0 else ly.vs)).to(dt) if fp8 else (src[:, :n] * (0.8 if kvi == 0 else 0.3)).to(dt)
        ly.pos0.fill_(pos0)
        Ls.append(ly)
    q, k, v = QG[1 : 1 + T].contiguous(), KK[1 : 1 + T].contiguous(), VV[1 : 1 + T].contiguous()
    out = torch.empty(T, 6144, dtype=torch.bfloat16, device="cuda")
    return Ls, (q, k, v, out)


def capture(Ls, args, T, fp8):
    q, k, v, out = args
    for ly in Ls[:1]:
        ly.step(q, k, v, int(ly.pos0.item()), out=out)  # warm-up outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        st = torch.cuda.current_stream().cuda_stream
        for ly in Ls:
            err = ta.lib().swiftmk_attn_step(T, q.data_ptr(), k.data_ptr(), v.data_ptr(), ly.qn.data_ptr(), ly.kn.data_ptr(), ly.kv.data_ptr(), ly.ctx_max,
                                             int(fp8), ly.ks, ly.vs, ly.pos0.data_ptr(), out.data_ptr(), ly.ws.data_ptr(), st)
            assert err == 0
    return g


def timeit(g, reps=30):
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--ctx", type=int, nargs="*", default=[2048, 8192, 32768])
    ap.add_argument("--layers", type=int, default=NL)
    ap.add_argument("--dtype", choices=["fp8", "bf16", "both"], default="both")
    ap.add_argument("--T", type=int, nargs="*", default=[1, 4])  # K11: any of 1..8
    a = ap.parse_args()
    for fp8 in {"fp8": (True,), "bf16": (False,), "both": (True, False)}[a.dtype]:
        for ctx in a.ctx:
            for T in a.T:
                Ls, args = build(ctx, T, fp8, a.layers)
                g = capture(Ls, args, T, fp8)
                ms = timeit(g)
                per = ms / a.layers
                kvb = ctx * 2 * 4 * 256 * (1 if fp8 else 2)
                print(f"{'fp8 ' if fp8 else 'bf16'} ctx {ctx:6d} T={T}: {per*1e3:7.1f} us/layer  KV {kvb/per/1e6:6.0f} GB/s  16-layer step {per*16:.3f} ms")
                if a.prof:
                    from torch.profiler import ProfilerActivity, profile

                    with profile(activities=[ProfilerActivity.CUDA]) as p:
                        for _ in range(5):
                            g.replay()
                        torch.cuda.synchronize()
                    print(p.key_averages().table(sort_by="cuda_time_total", row_limit=4, max_name_column_width=60))
                del Ls, g
                torch.cuda.empty_cache()
