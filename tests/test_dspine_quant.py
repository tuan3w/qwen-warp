#!/usr/bin/env python
"""Numerical contract of the W4A16 quantiser (engine/pack_mtp.py quant_w4 / dequant_w4) that quantises the DSpine drafter at load time.  CPU only, no checkpoint, seconds.

  python tests/test_dspine_quant.py
Method (llama.cpp tests/test-quantize-fns.cpp, tests/test-backend-ops.cpp): the error of the quantiser is measured on distributions that look like real weights (heavy tails, outlier
columns, not only N(0,1)) against bounds taken from measurements with headroom; the layout is decoded a second time by an independent bit-level reader; every extreme input must give
finite output (a NaN or inf scale is the typical silent rounding bug).  Bounds were measured 2026-11-10 on 512 x 2048 matrices (seed 0):
  N(0,1) 0.0103, Student-t(3) 0.0354, Laplace 0.0193, outlier columns 0.0093 NMSE with the MSE-optimal clip.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import pack_mtp  # noqa: E402

GROUP = 128
HEADROOM = 1.25  # allowed growth over the measured NMSE before the test fails


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def nmse(a, b):
    a, b = a.double(), b.double()
    return float(((a - b) ** 2).sum() / (a**2).sum())


def weights(kind, n=512, k=2048, seed=0):
    torch.manual_seed(seed)  # torch.distributions samplers ignore a local Generator: seed the global RNG too, or the data differs between runs
    g = torch.Generator().manual_seed(seed)
    if kind == "normal":
        w = torch.randn(n, k, generator=g)
    elif kind == "student_t3":
        w = torch.distributions.StudentT(3.0).sample((n, k))
    elif kind == "laplace":
        w = torch.distributions.Laplace(0.0, 1.0).sample((n, k))
    elif (
        kind == "outlier_cols"
    ):  # ~1% of the input channels 100x larger: the pattern of the massive activations / weights of real LLMs
        w = torch.randn(n, k, generator=g) * (
            1 + 99 * (torch.rand(1, k, generator=g) < 0.01)
        )
    else:
        raise ValueError(kind)
    return w.to(torch.bfloat16)


def decode_word(word, nib):
    """independent reader of the checkpoint layout: nibble e of the uint32 word j holds column 8j + e, stored offset by 8."""
    return int(((word & 0xFFFFFFFF) >> (4 * nib)) & 0xF) - 8


def main():
    measured = {
        "normal": 0.0103,
        "student_t3": 0.0396,
        "laplace": 0.0191,
        "outlier_cols": 0.0090,
    }
    for kind, bound in measured.items():
        w = weights(kind)
        q, s = pack_mtp.quant_w4(w, True, device="cpu")
        qn, sn = pack_mtp.quant_w4(w, False, device="cpu")
        e_clip, e_plain = (
            nmse(w.float(), pack_mtp.dequant_w4(q, s)),
            nmse(w.float(), pack_mtp.dequant_w4(qn, sn)),
        )
        check(
            e_clip <= bound * HEADROOM,
            f"{kind}: NMSE with the MSE-optimal clip {e_clip:.5f} <= {bound * HEADROOM:.5f}",
        )
        # the clip search tries ratio 1.0 as one candidate per group, so it can never lose to the plain round-to-nearest quantiser
        check(
            e_clip <= e_plain + 1e-12,
            f"{kind}: clip search ({e_clip:.5f}) never worse than plain quantisation ({e_plain:.5f})",
        )
        # plain quantisation: every element within half a step of its group's grid, step = amax / 7.5 (+ bf16 rounding of the scale)
        d = pack_mtp.dequant_w4(qn, sn)
        amax = w.float().abs().view(w.shape[0], -1, GROUP).amax(-1)
        rel = ((w.float() - d).abs().view(w.shape[0], -1, GROUP).amax(-1) / amax).max()
        check(
            float(rel) <= 0.075,
            f"{kind}: plain quantisation error <= 7.5% of the group maximum (measured {float(rel):.4f})",
        )

    # layout: every dequantised value equals (nibble - 8) * scale, decoded here without dequant_w4
    w = weights("normal", n=8, k=256, seed=1)
    q, s = pack_mtp.quant_w4(w, True, device="cpu")
    d = pack_mtp.dequant_w4(q, s)
    bad = 0
    for r in range(8):
        for c in range(256):
            ref = decode_word(int(q[r, c // 8]), c % 8) * float(s[r, c // GROUP])
            bad += abs(ref - float(d[r, c])) > 0
    check(
        bad == 0,
        "dequant_w4 equals the independent bit-level decode of the packed words (2048 values)",
    )
    check(
        q.dtype == torch.int32
        and q.shape == (8, 32)
        and s.shape == (8, 2)
        and s.dtype == torch.bfloat16,
        "packed shapes / dtypes: int32 [N, K/8], bf16 scale [N, K/128]",
    )
    nib = torch.stack(
        [((q.to(torch.int64) & 0xFFFFFFFF) >> (4 * e)) & 0xF for e in range(8)], -1
    )
    check(
        int(nib.min()) >= 0 and int(nib.max()) <= 15,
        "every nibble is a valid 4-bit code",
    )

    # extremes: no NaN / inf in scales or values (zero group, denormal-scale, huge, mixed magnitudes)
    cases = {
        "zeros": torch.zeros(16, 256),
        "tiny 1e-30": torch.full((16, 256), 1e-30),
        "huge 1e30": torch.randn(16, 256) * 1e30,
        "one outlier 3e4": torch.cat(
            [torch.randn(16, 255), torch.full((16, 1), 3.0e4)], 1
        ),
        "mixed magnitudes": torch.randn(16, 256)
        * (10.0 ** torch.randint(-20, 20, (16, 256)).float()),
    }
    for name, x in cases.items():
        x = x.to(torch.bfloat16)
        q, s = pack_mtp.quant_w4(x, True, device="cpu")
        d = pack_mtp.dequant_w4(q, s)
        check(
            bool(torch.isfinite(s.float()).all() and torch.isfinite(d).all()),
            f"extreme input '{name}': scales and dequantised values are finite",
        )
    z = pack_mtp.dequant_w4(
        *pack_mtp.quant_w4(torch.zeros(4, 128).to(torch.bfloat16), True, device="cpu")
    )
    check(bool((z == 0).all()), "an all-zero group dequantises to exact zeros")
    # sign / ordering: a larger weight never dequantises smaller within a group (monotone grid)
    x = torch.linspace(-3, 3, 128).view(1, 128).to(torch.bfloat16)
    d = pack_mtp.dequant_w4(*pack_mtp.quant_w4(x, True, device="cpu"))[0]
    check(bool((d[1:] >= d[:-1]).all()), "dequantisation is monotone within a group")
    print("OK")


if __name__ == "__main__":
    main()
