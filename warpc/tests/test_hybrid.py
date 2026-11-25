"""Stage 2 families 1-3 against the hand engine's kernels (libswiftmk) on identical device inputs: the zero-centred
RMSNorm with and without the residual add, the T = 1 bf16 head GEMV, the small a|b GEMV. All three are bit-identical by
contract; the A/B also times each pair as a decode step runs it (CUDA graph, median of 200 replays).

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tests/test_hybrid.py
  WARPC_TEST_SWIFT27=<27B checkpoint dir>  also runs the plan test (CPU)

Needs a CUDA device and $WARPC_SWIFTMK_LIB; the GPU tests skip without either.
"""

import ctypes as C
import os
from collections import Counter

import torch

from ab_util import ab_us, stream
from harness import Skip, run
from warpc import frontend, sources
from warpc.compiler import lower
from warpc.composite import CompositeBackend
from warpc.cuda import CudaBackend, glue, schedule
from warpc.swiftmk import SwiftmkBackend
from warpc.swiftmk import lib as smk

FAMILIES = ["g_rmsnorm_zc", "g_rmsnorm_zc_add", "g_embed", "g_head", "g_gemv_small"]


def _libs():
    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    return glue.load(10 * major + minor), smk.load()


def _call(fn, *args):
    """`fn(*args, stream)` with tensors passed as device pointers: a callable for the timing helpers."""
    ptrs = [a.data_ptr() if isinstance(a, torch.Tensor) else a for a in args]
    return lambda: fn(*ptrs, stream())


def _bf16(*shape, gen, scale=1.0):
    x = torch.randn(*shape, generator=gen) * scale
    x[..., ::97] *= 20  # outliers: residual-stream like magnitudes
    return x.to(torch.bfloat16).cuda()


def test_norm_is_bitwise():
    L, S = _libs()
    gen = torch.Generator().manual_seed(1)
    for H in (5120, 1024, 8192, 128):
        w = _bf16(H, gen=gen, scale=0.3)
        for T in range(1, 9):
            for add in (False, True):
                h = _bf16(T, H, gen=gen)
                y = _bf16(T, H, gen=gen, scale=0.5)
                h_a, h_b, xn_a, xn_b = h.clone(), h.clone(), torch.empty_like(h), torch.empty_like(h)
                ya = y.data_ptr() if add else None
                S.check(
                    S.smk_rmsnorm(
                        h_a.data_ptr(), ya, w.data_ptr(), xn_a.data_ptr(), None, T, H, C.c_float(1e-6), stream()
                    ),
                    "smk",
                )
                rc = L.warpc_rmsnorm_zc(
                    h_b.data_ptr(), ya, w.data_ptr(), xn_b.data_ptr(), T, H, C.c_float(1e-6), stream()
                )
                assert rc == 0, rc
                torch.cuda.synchronize()
                assert torch.equal(xn_a, xn_b), f"H={H} T={T} add={add}: xn differs"
                assert torch.equal(h_a, h_b), f"H={H} T={T} add={add}: the residual stream differs"


def test_head_is_bitwise_and_speed():
    L, S = _libs()
    gen = torch.Generator().manual_seed(2)
    for n, k in ((4096, 5120), (1000, 1024)):
        w = _bf16(n, k, gen=gen, scale=0.05)
        x = _bf16(1, k, gen=gen)
        a, b = (
            torch.empty(1, n, dtype=torch.float32, device="cuda"),
            torch.empty(1, n, dtype=torch.float32, device="cuda"),
        )
        S.check(S.smk_bf16_gemv_f32(w.data_ptr(), x.data_ptr(), n, k, 1, a.data_ptr(), stream()), "smk")
        assert L.warpc_head_gemv(w.data_ptr(), x.data_ptr(), n, k, b.data_ptr(), stream()) == 0
        torch.cuda.synchronize()
        assert torch.equal(a, b), f"N={n} K={k}: logits differ"
    n, k = 248320, 5120
    w = _bf16(n, k, gen=gen, scale=0.05)
    x = _bf16(1, k, gen=gen)
    a, b = torch.empty(1, n, dtype=torch.float32, device="cuda"), torch.empty(1, n, dtype=torch.float32, device="cuda")
    ours = _call(L.warpc_head_gemv, w, x, n, k, b)
    hand = _call(S.smk_bf16_gemv_f32, w, x, n, k, 1, a)
    ours()
    hand()
    torch.cuda.synchronize()
    assert torch.equal(a, b), "full vocabulary: logits differ"
    u, h = ab_us(ours, hand)
    print(f"  head 248320x5120: ours {u:.1f} us, hand {h:.1f} us ({h / u * 100:.1f}% of hand)")


def test_head_against_the_dense_gemv_on_dense_shapes():
    """The head GEMV serves every model's lm_head: against the dense table's k_gemv mode 2 (a different fp32 order:
    logits equal to 1e-5 relative) on the vocabularies and widths of the Qwen3 models, with the speed of each."""
    L, _ = _libs()
    gen = torch.Generator().manual_seed(5)
    for n, k in ((151936, 1024), (151936, 2560), (151936, 4096)):
        w = _bf16(n, k, gen=gen, scale=0.05)
        x = _bf16(1, k, gen=gen)
        a, b = (torch.empty(1, n, dtype=torch.float32, device="cuda") for _ in range(2))
        ends, outs = (C.c_int * 3)(), (C.c_void_p * 3)(a.data_ptr(), None, None)
        rpw = schedule.gemv_rows_per_warp(n, k, 2)
        old = _call(L.warpc_gemv, w, None, x, n, k, 1, 2, ends, outs, None, C.c_float(0.0), rpw)
        new = _call(L.warpc_head_gemv, w, x, n, k, b)
        assert old() == 0 and new() == 0
        torch.cuda.synchronize()
        err = float((a - b).abs().max() / a.abs().max())
        assert err < 1e-5, f"N={n} K={k}: relative difference {err:.3g}"
        u, h = ab_us(new, old)
        print(f"  dense head {n}x{k}: new {u:.1f} us, k_gemv {h:.1f} us ({h / u * 100:.1f}% of k_gemv)")


def test_small_gemv_is_bitwise():
    L, S = _libs()
    gen = torch.Generator().manual_seed(3)
    n, na, k = 96, 48, 5120
    w = _bf16(n, k, gen=gen, scale=0.05)
    for T in range(1, 9):
        x = _bf16(T, k, gen=gen)
        ha = [torch.empty(T, m, dtype=torch.bfloat16, device="cuda") for m in (na, n - na)]
        oa = [torch.empty_like(t) for t in ha]
        hand = _call(S.smk_bf16_gemv_small, w, x, n, na, k, T, ha[0], ha[1])
        ours = _call(L.warpc_gemv_small, w, x, n, na, k, T, oa[0], oa[1])
        S.check(hand(), "smk")
        assert ours() == 0
        torch.cuda.synchronize()
        for p, q in zip(ha, oa, strict=True):
            assert torch.equal(p, q), f"T={T}: a|b differ"
    x = _bf16(1, k, gen=gen)
    a0, a1 = (torch.empty(1, m, dtype=torch.bfloat16, device="cuda") for m in (na, n - na))
    ours = _call(L.warpc_gemv_small, w, x, n, na, k, 1, a0, a1)
    hand = _call(S.smk_bf16_gemv_small, w, x, n, na, k, 1, a0, a1)
    u, h = ab_us(ours, hand)
    print(f"  a|b 96x5120 T=1: ours {u:.2f} us, hand {h:.2f} us ({h / u * 100:.1f}% of hand)")


def test_norm_speed():
    L, S = _libs()
    gen = torch.Generator().manual_seed(4)
    H = 5120
    w, h0, y = _bf16(H, gen=gen), _bf16(1, H, gen=gen), _bf16(1, H, gen=gen)
    xn = torch.empty_like(h0)
    eps = C.c_float(1e-6)
    ours = _call(L.warpc_rmsnorm_zc, h0, y, w, xn, 1, H, eps)
    hand = _call(S.smk_rmsnorm, h0, y, w, xn, None, 1, H, eps)
    u, h = ab_us(ours, hand)
    print(f"  rmsnorm+add 5120 T=1: ours {u:.2f} us, hand {h:.2f} us ({h / u * 100:.1f}% of hand)")
    tok = torch.tensor([5, 77, 123], dtype=torch.int32, device="cuda")
    table = _bf16(256, H, gen=gen)
    out_a = torch.empty(3, H, dtype=torch.bfloat16, device="cuda")
    out_b = torch.empty_like(out_a)
    S.check(S.smk_embed(table.data_ptr(), tok.data_ptr(), out_a.data_ptr(), 3, H, stream()), "smk embed")
    assert L.warpc_embed(table.data_ptr(), tok.data_ptr(), out_b.data_ptr(), 3, H, stream()) == 0
    torch.cuda.synchronize()
    assert torch.equal(out_a, out_b)


def test_27b_plan_claims_the_families():
    """The composite with the three families admitted lowers the 27B graph onto them (CPU: headers and config)."""
    p = os.environ.get("WARPC_TEST_SWIFT27")
    if not p:
        raise Skip("WARPC_TEST_SWIFT27 (the 27B checkpoint directory) not set")
    g = frontend.build(sources.open(p), ctx_max=4608, kv="bf16")
    plan = lower(g, CompositeBackend(CudaBackend(), SwiftmkBackend("bf16"), kernels=FAMILIES), rows_max=8)
    got = Counter(t.kernel for t in plan.tasks)
    want = {
        "g_embed": 1,
        "g_rmsnorm_zc": 1,
        "g_rmsnorm_zc_add": 128,
        "g_gemv_small": 48,
        "g_head": 1,
        "s_w4_gemv": 192,
        "s_w4_gemv_silu": 64,
        "s_deltanet": 48,
        "s_attn": 16,
    }
    assert got == want, (got, want)


if __name__ == "__main__":
    run(globals())
