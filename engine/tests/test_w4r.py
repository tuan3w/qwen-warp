"""Repacked W4 layout (pack.repack_w4; smk_w4r_gemv, smkp_w4r_gemm, smkp_dequant_w4r) against the natural-layout kernels on the same random weights.

  ~/llm-lab/venv/bin/python tests/test_w4r.py [--shapes qkvz,o,gu,down,fc] [--Ts 1,2,3,4,5,8,16,33,64]
Random W4 weights / bf16 activations (the real model's shapes): no checkpoint needed.  Needs the GPU lease.
Checks, per shape:
  * smkp_dequant_w4r == smkp_dequant_w4: bit-identical bf16 weights (row range with row0 != 0); repack_w4 keeps the tensors' shape / identity;
  * smkp_w4r_gemm vs smk_w4r_gemv (groups of <= 8 rows), all listed T up to 64: same tolerance as below (the natural pair is bit-identical, tests/test_w4mm.py; the repacked pair differs in the fp32 order of rare elements);
  * smk_w4r_gemv(_cfg) (every configuration the shape allows, same applicability as the natural kernel) vs smk_w4_gemv(_cfg): NOT bit-identical by construction: the repacked word order puts nibble j / j+4
    at columns 2j / 2j+1, i.e. permutes the k order inside an 8-column group, and the tensor-core fp32 accumulation is order-sensitive.  Required: <= DIFF_FRAC of the outputs differ, none by more than
    4 bf16 ulp (at the value, but not finer than at the output rms), and both within the bf16 rounding floor of the fp32 dequantised reference (rel-L2 <= 2.5e-3; SiLU epilogue 4e-3).
"""
DIFF_FRAC, MAX_ULP = 1e-3, 4.0  # measured max 3.3 (gu, SiLU epilogue: silu(gate) * up amplifies the reorder noise of two accumulators); all other shapes <= 2
import argparse
import ctypes as C
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pack
import smk
from smk import Epi, check, epi_segs, epi_simple, stream

SHAPES = {"qkvz": (16384, 5120), "attn_qkv": (14336, 5120), "o": (5120, 6144), "gu": (34816, 5120), "down": (5120, 17408), "fc": (5120, 10240), "head": (65536, 5120)}
DEV = "cuda"


def make_w4(N, K, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    w = torch.randint(-(2**31), 2**31 - 1, (N, K // 8), dtype=torch.int32, device=DEV, generator=g)
    s = (torch.rand(N, K // 128, device=DEV, generator=g) * 0.02 + 0.002).to(torch.bfloat16)
    return w, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--Ts", default="1,2,3,4,5,8,9,16,33,64")
    args = ap.parse_args()
    lib = smk.load()
    vp, ci = C.c_void_p, C.c_int
    for n, a in {
        "smk_w4r_gemv": [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), vp],
        "smk_w4r_gemv_cfg": [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), ci, vp],
        "smkp_w4_gemm": [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), vp],
        "smkp_w4r_gemm": [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), vp],
        "smkp_dequant_w4": [vp, vp, vp, ci, ci, ci, vp],
        "smkp_dequant_w4r": [vp, vp, vp, ci, ci, ci, vp],
    }.items():
        getattr(lib, n).argtypes = a
        getattr(lib, n).restype = ci
    st = stream()
    Ts = [int(x) for x in args.Ts.split(",")]
    bad = n_checks = 0

    def eq(a, b):
        return bool((a.view(torch.int16) == b.view(torch.int16)).all())

    def report(name, ok, extra=""):
        nonlocal bad, n_checks
        n_checks += 1
        bad += not ok
        if not ok:
            print(f"FAIL {name} {extra}")

    for name in args.shapes.split(","):
        N, K = SHAPES[name]
        wn, sn = make_w4(N, K, sum(map(ord, name)))
        wr, sr = wn.clone(), sn.clone()
        ret = pack.repack_w4(wr, sr)
        report(f"{name} repack shape", wr.shape == wn.shape and sr.shape == sn.shape and ret[0] is wr and ret[1] is sr)
        report(f"{name} repack changes the layout", not bool((wr == wn).all()))
        # dequant to bf16 (the prefill's weight buffer): natural vs repacked
        rows = min(N, 2048)
        row0 = 0 if N <= 2048 else (N - rows) // 2 // 16 * 16
        d1 = torch.empty(rows, K, device=DEV, dtype=torch.bfloat16)
        d2 = torch.empty_like(d1)
        check(lib.smkp_dequant_w4(wn.data_ptr(), sn.data_ptr(), d1.data_ptr(), K, row0, rows, st), "dequant")
        check(lib.smkp_dequant_w4r(wr.data_ptr(), sr.data_ptr(), d2.data_ptr(), K, row0, rows, st), "dequant_r")
        torch.cuda.synchronize()
        report(f"{name} dequant rows {row0}..{row0 + rows}", eq(d1, d2))
        silu = name == "gu"
        sh = torch.arange(0, 32, 4, device=DEV, dtype=torch.int32)
        Wref = (((wn.unsqueeze(-1) >> sh) & 15).view(N, K).float() - 8.0) * sn.float().repeat_interleave(128, dim=1)

        def close(o1, o2, x, tag):
            d = (o1.float() - o2.float()).abs()
            frac = (d != 0).float().mean().item()
            ulp = o1.float().abs().clamp_min(o1.float().pow(2).mean().sqrt().item()) * 2**-8  # 1 bf16 ulp at the value, but not finer than at the output rms (cancellation: tiny values carry the fp32 reorder noise of the large partial sums)
            mx = (d / ulp).max().item()
            ref = x.float() @ Wref.t()
            if silu:
                R = ref.view(x.shape[0], N // 16, 2, 8)
                ref = (torch.nn.functional.silu(R[:, :, 0].to(torch.bfloat16).float()) * R[:, :, 1].to(torch.bfloat16).float()).reshape(x.shape[0], N // 2)
            err = max(((o.float() - ref).norm() / ref.norm()).item() for o in (o1, o2))
            report(f"{name} {tag}", frac <= DIFF_FRAC and mx <= MAX_ULP and err <= (4e-3 if silu else 2.5e-3), f"differing {frac:.2e} max {mx:.2f} ulp rel-L2 vs fp32 {err:.2e}")
            return frac

        nd = []
        for T in Ts:
            x = torch.randn(T, K, device=DEV, dtype=torch.bfloat16)

            def mk(o):
                return epi_simple(smk.SMK_EPI_SILU, o) if silu else epi_segs(N, [o])

            shp = (T, N // 2) if silu else (T, N)
            og = torch.empty(shp, device=DEV, dtype=torch.bfloat16)
            for r0 in range(0, T, 8):
                n = min(8, T - r0)
                check(lib.smk_w4r_gemv(wr.data_ptr(), sr.data_ptr(), x[r0 : r0 + n].data_ptr(), N, K, n, C.byref(mk(og[r0 : r0 + n])), st), "gemv_r")
            o6 = torch.empty(shp, device=DEV, dtype=torch.bfloat16)
            check(lib.smkp_w4r_gemm(wr.data_ptr(), sr.data_ptr(), x.data_ptr(), N, K, T, C.byref(mk(o6)), st), "gemm_r")
            torch.cuda.synchronize()
            nd.append(close(og, o6, x, f"T={T} smkp_w4r_gemm vs smk_w4r_gemv"))
            if T <= 8:
                o1 = torch.empty(shp, device=DEV, dtype=torch.bfloat16)
                check(lib.smk_w4_gemv(wn.data_ptr(), sn.data_ptr(), x.data_ptr(), N, K, T, C.byref(mk(o1)), st), "gemv")
                nd.append(close(o1, og, x, f"T={T} smk_w4r_gemv vs smk_w4_gemv"))
                for cfg in range(6):
                    o3, o4 = (torch.zeros(shp, device=DEV, dtype=torch.bfloat16) for _ in range(2))
                    e1 = lib.smk_w4_gemv_cfg(wn.data_ptr(), sn.data_ptr(), x.data_ptr(), N, K, T, C.byref(mk(o3)), cfg, st)
                    e2 = lib.smk_w4r_gemv_cfg(wr.data_ptr(), sr.data_ptr(), x.data_ptr(), N, K, T, C.byref(mk(o4)), cfg, st)
                    torch.cuda.synchronize()
                    report(f"{name} T={T} cfg={cfg} applicability agrees", (e1 == 0) == (e2 == 0), f"{e1} {e2}")
                    if e1 == 0 and e2 == 0:
                        nd.append(close(o3, o4, x, f"T={T} cfg={cfg} smk_w4r_gemv_cfg vs smk_w4_gemv_cfg"))
        print(f"{name}: done, natural-vs-repacked differing-element fraction over {len(nd)} comparisons: mean {sum(nd) / len(nd):.1e} max {max(nd):.1e}", flush=True)
    print(f"{n_checks - bad}/{n_checks} checks passed" + ("" if not bad else f", {bad} FAILED"))
    print("ALL PASS" if not bad else "FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
