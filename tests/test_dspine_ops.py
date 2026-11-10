#!/usr/bin/env python
"""Operator-level test of the W4A16 GEMV (smk_w4r_gemv) at the shapes the DSpine drafter runs it with, against an fp64 reference built from the SAME packed weights.  GPU, no checkpoint, ~20 s.

  python tests/test_dspine_ops.py [--report]
Method (llama.cpp tests/test-backend-ops.cpp): per case a reference result, NMSE = mse(a, b) / mse(a, 0) against a tolerance; any NaN fails; an inf must match in sign; the op is run again
and the bytes must be identical (determinism); the output buffer is poisoned with NaN and fenced by guard words, so an element the kernel forgets to write (partial coverage of a tile) or a
write past the end (a sentinel overwritten) fails instead of passing on stale memory.  The activations are not only N(0, 1): heavy tails, one huge outlier channel (the target's residual
stream reaches 2.6e4) and a tiny scale, because rounding and overflow bugs live there.
--report prints the measured numbers (used to set the tolerances below).
"""

import argparse
import sys
from pathlib import Path

import torch

ENG = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENG))
import ctypes as C  # noqa: E402

import pack  # noqa: E402
import pack_mtp  # noqa: E402
import smk  # noqa: E402

T = 8
# (name, N, K, epilogue) : the drafter's GEMVs (qkv q|k|v, o, gate|up with SILU, down, fc, kv)
SHAPES = [
    ("qkv", 6144, 5120),
    ("o", 5120, 4096),
    ("down", 5120, 17408),
    ("fc", 5120, 25600),
    ("kv", 10240, 5120),
    ("gate_up_rows", 34816, 5120),
]
NMSE_MAX = 1.0e-10  # measured 8.6e-14 worst (fp32 accumulation order only); a bf16 rounding of the weight or the activation would give >= 1e-6
GUARD = 1024


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def nmse(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float(((a - b) ** 2).sum() / (a**2).sum().clamp_min(1e-300))


def activations(kind, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, K, device="cuda", generator=g)
    if kind == "heavy_tail":
        x = x / (
            torch.rand(T, K, device="cuda", generator=g).clamp_min(1e-3)
        )  # Cauchy-like tails
    elif kind == "outlier_channel":
        x[:, K // 3] = 2.6e4 * torch.sign(torch.randn(T, device="cuda", generator=g))
    elif kind == "tiny":
        x = x * 1e-4
    elif kind == "zeros":
        x = torch.zeros_like(x)
    return x.to(torch.bfloat16)


def run_gemv(lib, w, s, x, N, K, kind):
    """-> fp32 [T, N] (F32 epilogue) or bf16 [T, N] (SEGS epilogue) with a NaN-poisoned, guarded output."""
    elem = 4 if kind == "f32" else 2
    buf = torch.full(
        (T * N + 2 * GUARD,),
        float("nan"),
        device="cuda",
        dtype=torch.float32 if kind == "f32" else torch.bfloat16,
    )
    out = buf[GUARD : GUARD + T * N]
    e = smk.Epi()
    if kind == "f32":
        e.kind = smk.SMK_EPI_F32
    else:
        e.kind = smk.SMK_EPI_SEGS
        e.end[0] = e.end[1] = e.end[2] = N
    e.out[0] = out.data_ptr()
    smk.check(
        lib.smk_w4r_gemv(
            C.c_void_p(w.data_ptr()),
            C.c_void_p(s.data_ptr()),
            C.c_void_p(x.data_ptr()),
            N,
            K,
            T,
            C.byref(e),
            C.c_void_p(torch.cuda.current_stream().cuda_stream),
        ),
        "smk_w4r_gemv",
    )
    torch.cuda.synchronize()
    del elem
    return buf, out.view(T, N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    lib = smk.load("libswiftmk.so")
    lib.smk_w4r_gemv.argtypes = [
        C.c_void_p,
        C.c_void_p,
        C.c_void_p,
        C.c_int,
        C.c_int,
        C.c_int,
        C.POINTER(smk.Epi),
        C.c_void_p,
    ]
    lib.smk_w4r_gemv.restype = C.c_int
    worst = 0.0
    for name, N, K in SHAPES:
        torch.manual_seed(0)
        wf = torch.randn(N, K, dtype=torch.bfloat16) * 0.02
        qn, sn = pack_mtp.quant_w4(wf, True, device="cuda")
        ref_w = pack_mtp.dequant_w4(
            qn, sn
        ).double()  # [N, K] fp64 reference of exactly what the kernel multiplies with
        w, s = qn.clone(), sn.clone()
        pack.repack_w4(w, s)
        for kind in ("normal", "heavy_tail", "outlier_channel", "tiny", "zeros"):
            x = activations(kind, K, seed=7)
            ref = x.double() @ ref_w.T  # [T, N]
            buf, out = run_gemv(lib, w, s, x, N, K, "f32")
            tag = f"{name} N={N} K={K} T={T} '{kind}'"
            check(
                bool(
                    (buf[:GUARD].isnan()).all() and (buf[GUARD + T * N :].isnan()).all()
                ),
                f"{tag}: guard words around the output untouched (no out-of-bounds write)",
            )
            check(
                not bool(out.isnan().any()),
                f"{tag}: every output element written, none NaN",
            )
            check(bool(torch.isfinite(out).all()), f"{tag}: finite for finite input")
            if kind == "zeros":
                check(
                    bool((out == 0).all()), f"{tag}: zero activations give exact zeros"
                )
                continue
            e = nmse(ref, out)
            worst = max(worst, e)
            if args.report:
                print(f"REPORT {tag}: nmse {e:.3e}", flush=True)
            else:
                check(
                    e <= NMSE_MAX,
                    f"{tag}: NMSE vs fp64 reference {e:.2e} <= {NMSE_MAX:.0e}",
                )
            _, again = run_gemv(lib, w, s, x, N, K, "f32")
            check(
                bool(torch.equal(out, again)),
                f"{tag}: bit-identical on a second run (deterministic)",
            )
            # rounding contract: the bf16 (SEGS) epilogue is the fp32 result rounded once to bf16
            _, o16 = run_gemv(lib, w, s, x, N, K, "bf16")
            check(
                bool(torch.equal(o16, out.to(torch.bfloat16))),
                f"{tag}: bf16 epilogue == fp32 result rounded once to bf16",
            )
    print(f"worst NMSE {worst:.3e}")
    print("OK")


if __name__ == "__main__":
    main()
