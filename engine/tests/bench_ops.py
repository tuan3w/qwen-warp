"""Micro-benchmarks of the K3 ops, L2-cold (nb distinct weight buffers per op, >= 400 MB, one CUDA graph), random data of the real shapes.

  ~/llm-lab/venv/bin/python tests/bench_ops.py [--sweep] [--lm-head]    (needs the GPU lease)
Prints a markdown table (GB/s = weight+scale bytes / time per op).
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import smk
from smk import check, epi_segs, epi_simple, ptr, stream

DEV = "cuda"


def graph_time(fn_list, reps=30, warm=5):
    """fn_list: list of closures (one per buffer); returns ms per closure (median over reps of whole-graph time / len)."""
    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) / len(fn_list))
    ts.sort()
    return ts[len(ts) // 2]


def rand_w4(N, K, nb):
    ws = [torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=DEV) for _ in range(nb)]
    ss = [(torch.rand(N, K // 128, device=DEV) * 0.02 / 5).to(torch.bfloat16) for _ in range(nb)]
    return ws, ss


def nbuf(bytes_each, target=400 << 20):
    return max(2, min(16, -(-target // bytes_each)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--lm-head", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    lib = smk.load("libswiftmk_core.so")
    torch.manual_seed(0)
    a_ = torch.randn(8192, 8192, device=DEV, dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < 2.0:  # warm the clocks (first measurements otherwise run ~7% slow)
        a_ @ a_
    torch.cuda.synchronize()
    del a_
    props = torch.cuda.get_device_properties(0)
    print(f"# {props.name}, {props.multi_processor_count} SMs, torch {torch.__version__}")
    rows = []
    shapes = [("gateup+silu", 34816, 5120), ("down", 5120, 17408), ("qkv|z", 16384, 5120), ("attn q|k|v", 14336, 5120), ("o_proj", 5120, 6144)]
    for name, N, K in shapes:
        nbytes = N * K // 2 + N * K // 64
        nb = nbuf(nbytes)
        ws, ss = rand_w4(N, K, nb)
        x = (torch.randn(4, K, device=DEV) * 0.5).to(torch.bfloat16)
        outs = [torch.empty(4, N, dtype=torch.bfloat16, device=DEV)]
        if name.startswith("gateup"):
            e = epi_simple(smk.SMK_EPI_SILU, outs[0])
        elif name == "qkv|z":
            o1, o2 = torch.empty(4, 10240, dtype=torch.bfloat16, device=DEV), torch.empty(4, 6144, dtype=torch.bfloat16, device=DEV)
            e = epi_segs(N, [o1, o2])
        elif name == "attn q|k|v":
            o1, o2, o3 = (torch.empty(4, w, dtype=torch.bfloat16, device=DEV) for w in (12288, 1024, 1024))
            e = epi_segs(N, [o1, o2, o3])
        else:
            e = epi_segs(N, [outs[0]])
        for T in (1, 2, 4) if not args.quick else (1, 4):
            best = None
            cfgs = range(6) if args.sweep else [None]
            for cfg in cfgs:
                def mk(i, cfg=cfg):
                    s = stream()
                    if cfg is None:
                        return lambda: check(lib.smk_w4_gemv(ptr(ws[i]), ptr(ss[i]), ptr(x), N, K, T, e, stream()))
                    return lambda: lib.smk_w4_gemv_cfg(ptr(ws[i]), ptr(ss[i]), ptr(x), N, K, T, e, cfg, stream())
                if cfg is not None and lib.smk_w4_gemv_cfg(ptr(ws[0]), ptr(ss[0]), ptr(x), N, K, T, e, cfg, stream()) != 0:
                    continue
                ms = graph_time([mk(i) for i in range(nb)])
                gbs = nbytes / ms / 1e6
                if args.sweep:
                    print(f"  sweep {name} T={T} cfg {cfg}: {ms * 1e3:.1f} us {gbs:.0f} GB/s", flush=True)
                if best is None or ms < best[0]:
                    best = (ms, cfg)
            ms, cfg = best
            rows.append((f"w4 {name}", f"T={T}" + (f" cfg{cfg}" if args.sweep else ""), N, K, ms * 1e3, nbytes / ms / 1e6))
            print(f"w4 {name} T={T}: {ms * 1e3:.1f} us  {nbytes / ms / 1e6:.0f} GB/s", flush=True)
        del ws, ss
        torch.cuda.empty_cache()

    # small a|b GEMV
    nb = 16
    wab = [torch.randn(96, 5120, device=DEV).to(torch.bfloat16) for _ in range(nb)]
    x = torch.randn(4, 5120, device=DEV).to(torch.bfloat16)
    oa, ob = torch.empty(4, 48, dtype=torch.bfloat16, device=DEV), torch.empty(4, 48, dtype=torch.bfloat16, device=DEV)
    for T in (1, 4):
        ms = graph_time([(lambda i=i: lib.smk_bf16_gemv_small(ptr(wab[i]), ptr(x), 96, 48, 5120, T, ptr(oa), ptr(ob), stream())) for i in range(nb)])
        nbytes = 96 * 5120 * 2
        rows.append(("bf16 small a|b", f"T={T}", 96, 5120, ms * 1e3, nbytes / ms / 1e6))
        print(f"bf16 small a|b T={T}: {ms * 1e3:.1f} us ({nbytes / ms / 1e6:.0f} GB/s; 1 MB, latency-bound)", flush=True)

    # rmsnorm / embed / argmax (latency)
    h = torch.randn(4, 5120, device=DEV).to(torch.bfloat16)
    y = torch.randn(4, 5120, device=DEV).to(torch.bfloat16)
    w = torch.randn(5120, device=DEV).to(torch.bfloat16)
    xn = torch.empty_like(h)
    for T in (1, 4):
        ms = graph_time([lambda: lib.smk_rmsnorm(ptr(h), ptr(y), ptr(w), ptr(xn), None, T, 5120, 1e-6, stream()) for _ in range(32)])
        rows.append(("rmsnorm+residual", f"T={T}", T, 5120, ms * 1e3, 0))
        print(f"rmsnorm+residual T={T}: {ms * 1e3:.2f} us", flush=True)
    logits = torch.randn(4, 248320, device=DEV)
    ws_ = torch.zeros(lib.smk_argmax_ws_bytes(), dtype=torch.uint8, device=DEV)
    tok = torch.empty(4, dtype=torch.int32, device=DEV)
    for T in (1, 4):
        ms = graph_time([lambda: lib.smk_argmax(ptr(logits), T, 248320, ptr(tok), None, ptr(ws_), stream()) for _ in range(16)])
        rows.append(("argmax", f"T={T}", T, 248320, ms * 1e3, T * 248320 * 4 / ms / 1e6))
        print(f"argmax T={T} over 248320 fp32: {ms * 1e3:.1f} us ({T * 248320 * 4 / ms / 1e6:.0f} GB/s from L2)", flush=True)
    emb = torch.randn(1024, 5120, device=DEV).to(torch.bfloat16)
    tk = torch.randint(0, 1024, (4,), dtype=torch.int32, device=DEV)
    out = torch.empty(4, 5120, dtype=torch.bfloat16, device=DEV)
    ms = graph_time([lambda: lib.smk_embed(ptr(emb), ptr(tk), ptr(out), 4, 5120, stream()) for _ in range(32)])
    print(f"embed T=4: {ms * 1e3:.2f} us", flush=True)
    rows.append(("embed", "T=4", 4, 5120, ms * 1e3, 0))

    # lm_head
    if args.lm_head:
        N, K = 248320, 5120
        nbytes = N * K * 2
        wl = [torch.randn(N, K, device=DEV).to(torch.bfloat16) for _ in range(2)]
        x = torch.randn(4, K, device=DEV).to(torch.bfloat16)
        lg = torch.empty(4, N, dtype=torch.float32, device=DEV)
        for T in (2, 1, 4, 1):  # first entry warms clocks
            ms = graph_time([(lambda i=i: lib.smk_bf16_gemv_f32(ptr(wl[i]), ptr(x), N, K, T, ptr(lg), stream())) for i in range(2)], reps=10)
            rows.append(("bf16 lm_head", f"T={T}", N, K, ms * 1e3, nbytes / ms / 1e6))
            print(f"lm_head T={T}: {ms:.3f} ms  {nbytes / ms / 1e6:.0f} GB/s", flush=True)
    print("\n| op | T | N | K | us | GB/s |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]:.1f} | {r[5]:.0f} |")


if __name__ == "__main__":
    main()
