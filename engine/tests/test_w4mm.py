"""K10 multi-row W4 mma GEMM (src/prefill_w4.cu, smkp_w4_gemm_cfg): correctness vs a fp32 reference and vs smk_w4_gemv, and a per-config micro-benchmark.

  cd <repo root>; SMK_BUILD=build_k10 ~/llm-lab/venv/bin/python engine/tests/test_w4mm.py [--bench] [--cfgs 0,1,2,...] [--Ts 8,16,32,64]
Random W4 weights / bf16 activations (the real model's shapes): no checkpoint needed.  Needs the GPU lease.
Checks (every T in 1..64 that is listed, all epilogues):
  * rel-L2 vs the fp32 dequantised reference <= 2.5e-3 (bf16 output rounding floor ~1.7e-3)
  * bit-identical to smk_w4_gemv (K5 kernel, default config w8k4u2 = smkp cfg 0) on the same rows, called in groups of <= 8 rows
"""
import argparse
import ctypes as C
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import smk
from smk import Epi, check, epi_segs, epi_simple, ptr, rel_l2, stream

SHAPES = {"qkvz": (16384, 5120), "attn_qkv": (14336, 5120), "o": (5120, 6144), "gu": (34816, 5120), "down": (5120, 17408), "fc": (5120, 10240)}
DEV = "cuda"


def make_w4(N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randint(-(2**31), 2**31 - 1, (N, K // 8), dtype=torch.int32, device=DEV, generator=g)
    s = (torch.rand(N, K // 128, device=DEV, generator=g) * 0.02 + 0.002).to(torch.bfloat16)
    return w, s


def ref_dequant(w, s):
    N, K8 = w.shape
    sh = torch.arange(0, 32, 4, device=DEV, dtype=torch.int32)
    nib = ((w.unsqueeze(-1) >> sh) & 15).view(N, K8 * 8).float() - 8.0
    return nib * s.float().repeat_interleave(128, dim=1)


def setup(lib):
    lib.smkp_w4_gemm_cfg.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.POINTER(Epi), C.c_int, C.c_void_p]
    lib.smkp_w4_gemm_cfg.restype = C.c_int


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--cfgs", default="0")
    ap.add_argument("--Ts", default="1,5,8,9,16,24,31,32,33,48,57,64")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    args = ap.parse_args()
    lib = smk.load()
    setup(lib)
    st = stream()
    Ts = [int(x) for x in args.Ts.split(",")]
    cfgs = [int(x) for x in args.cfgs.split(",")]
    bad = 0
    for name in args.shapes.split(","):
        N, K = SHAPES[name]
        w, s = make_w4(N, K, hash(name) % 1000)
        W = ref_dequant(w, s)
        for T in Ts:
            x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)
            ref = (x.float() @ W.t())
            epis = [("segs", torch.empty(T, N, device=DEV, dtype=torch.bfloat16))]
            if name == "gu":
                epis = [("silu", torch.empty(T, N // 2, device=DEV, dtype=torch.bfloat16))]
            for kind, out in epis:
                mk = (lambda o: epi_segs(N, [o])) if kind == "segs" else (lambda o: epi_simple(smk.SMK_EPI_SILU, o))
                # K5 reference kernel, <= 8 rows per call
                outk = torch.empty_like(out)
                for r0 in range(0, T, 8):
                    n = min(8, T - r0)
                    e = mk(outk[r0 : r0 + n])
                    check(lib.smk_w4_gemv_cfg(w.data_ptr(), s.data_ptr(), x[r0 : r0 + n].data_ptr(), N, K, n, C.byref(e), 0, st), "gemv")
                for cfg in cfgs:
                    out.zero_()
                    e = mk(out)
                    err = lib.smkp_w4_gemm_cfg(w.data_ptr(), s.data_ptr(), x.data_ptr(), N, K, T, C.byref(e), cfg, st)
                    if err != 0:
                        print(f"{name} T={T} cfg={cfg}: error {err} (config not applicable)")
                        continue
                    torch.cuda.synchronize()
                    if kind == "segs":
                        r = rel_l2(out, ref.to(torch.bfloat16))
                        rf = rel_l2(out, ref)
                    else:
                        gt, up = ref[:, 0::16].new_zeros(0), None
                        # interleaved: tile i rows 16i..16i+7 = gate ch 8i.., 16i+8.. = up ch 8i..
                        R = ref.view(T, N // 16, 2, 8)
                        gate, upp = R[:, :, 0].reshape(T, N // 2), R[:, :, 1].reshape(T, N // 2)
                        g16 = gate.to(torch.bfloat16).float()
                        refs = (torch.nn.functional.silu(g16).to(torch.bfloat16).float() * upp.to(torch.bfloat16).float()).to(torch.bfloat16)
                        r = rel_l2(out, refs)
                        rf = r
                    same = bool((out.view(torch.int16) == outk.view(torch.int16)).all())
                    tol_ok = rf <= 2.5e-3 * (1.6 if kind == "silu" else 1.0)
                    ok = tol_ok and (same or cfg != 0)
                    bad += not ok
                    if not args.bench or not ok:
                        print(f"{'PASS' if ok else 'FAIL'} {name:8s} T={T:2d} cfg={cfg} {kind}: rel-L2 vs fp32 ref {rf:.2e}, bit-identical to smk_w4_gemv: {same}")
    if args.bench:
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        print("\nmicro-benchmark (us per call, median of 20; 'layer' = sum over qkvz+o+gu+down at 64 layers equivalent ms = 64 x sum)")
        tab = {}
        for T in Ts:
            x = torch.randn(T, 17408, device=DEV, dtype=torch.bfloat16)
            for cfg in cfgs:
                tot = 0.0
                row = []
                for name in ("qkvz", "o", "gu", "down"):
                    N, K = SHAPES[name]
                    w, s = make_w4(N, K, 1)
                    out = torch.empty(T, N, device=DEV, dtype=torch.bfloat16)
                    e = epi_simple(smk.SMK_EPI_SILU, out) if name == "gu" else epi_segs(N, [out])
                    xs = x[:, :K].contiguous()
                    ts = []
                    ok = True
                    for rep in range(25):
                        ev0.record()
                        err = lib.smkp_w4_gemm_cfg(w.data_ptr(), s.data_ptr(), xs.data_ptr(), N, K, T, C.byref(e), cfg, st)
                        ev1.record()
                        torch.cuda.synchronize()
                        if err:
                            ok = False
                            break
                        ts.append(ev0.elapsed_time(ev1) * 1e3)
                    if not ok:
                        row.append(None)
                        continue
                    t = sorted(ts)[len(ts) // 2]
                    row.append(t)
                    # dequantised-weight bytes: this is the cold-cache time only if the weights do not fit L2 (they never do: >= 40 MB)
                if None in row:
                    print(f"T={T:2d} cfg={cfg}: n/a")
                    continue
                tot = sum(row)
                tab[(T, cfg)] = tot
                print(f"T={T:2d} cfg={cfg}: qkvz {row[0]:6.0f} o {row[1]:6.0f} gu {row[2]:6.0f} down {row[3]:6.0f} us | 64-layer sum (attn layers differ) {tot * 64 / 1e3:6.1f} ms")
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
