"""KV formats: int4_rot's reference quantiser byte-identical to the hand engine's writers, its dequantiser identical to
the engine's gather, its layout equal to the engine's per-layer buffer, and warpc's own prep kernel writing the same
bytes.

  uv run --package warpc python warpc/tests/test_kvcache.py

WARPC_SWIFTMK_LIB (the engine library, built from the v0.12.0 checkout) and a CUDA device enable the
engine comparisons. The engine is built for 4 KV heads of head dim 256; only its V rows reach the quantiser unchanged
(K is normed and roped first), so the engine's bytes are compared on V and its dequantiser on both.
"""

import ctypes
import itertools
import os

import torch

from harness import Skip, run
from warpc.cuda import glue, schedule
from warpc.kvcache import FORMATS, int4_rot_dequantize, int4_rot_quantize, rotate

ENGINE_NKV, ENGINE_HD = 4, 256
INT4_ROT = FORMATS["int4_rot"]


def engine_kv_bytes(ctx: int) -> int:
    """swiftmk_attn_kv_bytes(ctx, 2) (attn.cu): codes + meta, sink rows, ring rows."""
    return 2 * ENGINE_NKV * ctx * (128 + 4) + 2 * ENGINE_NKV * 4 * 256 * 2 + 2 * ENGINE_NKV * 512 * 256 * 2


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


def engine():
    path = os.environ.get("WARPC_SWIFTMK_LIB")
    if not path:
        raise Skip("WARPC_SWIFTMK_LIB is not set")
    require_gpu()
    lib = ctypes.CDLL(path)
    p, i, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_float
    lib.swiftmk_attn_kv_bytes.restype = ctypes.c_size_t
    lib.swiftmk_attn_kv_bytes.argtypes = [i, i]
    lib.swiftmk_attn_ws_bytes.restype = ctypes.c_size_t
    lib.smkp_attn_prep.argtypes = [i, i, p, i, p, i, p, i, p, p, p, i, i, f, f, p, p, p, p, p]
    lib.smkp_gather_kv.argtypes = [p, i, i, i, f, f, p, p, p]
    lib.swiftmk_attn_step.argtypes = [i, p, p, p, p, p, p, i, i, f, f, p, p, p, p, p]
    return lib


def rows(n: int, seed: int, heads: int = ENGINE_NKV, hd: int = ENGINE_HD) -> torch.Tensor:
    """bf16 rows [n, heads, hd] across scales, with heavy-tailed channels, an all-zero row and constant rows."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, heads, hd, generator=g)
    x *= torch.exp(torch.empty(n, heads, 1).uniform_(-9, 6, generator=g))
    hot = torch.rand(n, heads, hd, generator=g) < 0.01
    x = torch.where(hot, x * 40, x)
    x[1] = 0
    x[2] = 0.5
    x[3, :, ::2] = -3
    return x.bfloat16()


def section(buf: torch.Tensor, ctx: int, name: str, nkv: int = ENGINE_NKV, hd: int = ENGINE_HD) -> torch.Tensor:
    """Section `name` of an int4_rot layer buffer (uint8, CPU): [nkv][rows][row bytes], fp16 rows as [nkv][rows][hd]."""
    lay = INT4_ROT.layout(ctx, nkv, hd, 8)
    n = {"c": ctx * hd // 2, "s": ctx * hd // 64, "x": 4 * hd * 2, "r": 512 * hd * 2}[name[1]]
    s = buf[lay[name] : lay[name] + nkv * n]
    return s.view(nkv, ctx, -1) if name[1] in "cs" else s.view(torch.float16).view(nkv, -1, hd)


def check_cache(buf, ctx: int, x: torch.Tensor, pos0: int, ring_from: int, which: str = "v") -> None:
    """The K or V planes of a layer buffer for rows x [T, nkv, hd] at positions pos0.. against the reference: codes,
    scale codes, the sink rows and the ring rows of positions >= ring_from."""
    T, nkv, hd = x.shape
    codes, meta, exact = int4_rot_quantize(x.float())
    for name, want in (("c", codes), ("s", meta)):
        got = section(buf, ctx, which + name, nkv, hd)[:, pos0 : pos0 + T]
        bad = (got != want.transpose(0, 1)).any(-1)
        assert not bad.any(), f"{which}{name} differs in {int(bad.sum())} of {bad.numel()} rows"
    sink, ring = section(buf, ctx, which + "x", nkv, hd), section(buf, ctx, which + "r", nkv, hd)
    for p in range(pos0, min(pos0 + T, 4)):
        assert torch.equal(sink[:, p].view(torch.int16), exact[p - pos0].view(torch.int16)), f"{which} sink row {p}"
    for p in range(ring_from, pos0 + T):
        assert torch.equal(ring[:, p % 512].view(torch.int16), exact[p - pos0].view(torch.int16)), f"{which} ring {p}"


def test_int4_rot_layout_is_the_engines_at_hd_256():
    for ctx in (1, 4096, 262144):
        assert INT4_ROT.nbytes(ctx, ENGINE_NKV, ENGINE_HD, 8) == engine_kv_bytes(ctx)
    lay = INT4_ROT.layout(4096, ENGINE_NKV, ENGINE_HD, 8)
    assert lay["kx"] == 2 * ENGINE_NKV * 4096 * 132  # kvrot::sink_off
    assert lay["kr"] == lay["kx"] + 2 * ENGINE_NKV * 4 * 256 * 2  # kvrot::ring_off
    # 4.125 bits per element at every head dim, plus the fixed sink and ring
    for hd in (64, 128, 256):
        per_pos = INT4_ROT.nbytes(2049, 2, hd, 8) - INT4_ROT.nbytes(2048, 2, hd, 8)
        assert 2 * 2 * hd * 4.125 / 8 <= per_pos <= 2 * 2 * hd * 4.125 / 8 + 64, (hd, per_pos)


def test_int4_rot_reference_round_trip_at_every_head_dim():
    g = torch.Generator().manual_seed(3)
    for hd in (64, 128, 256):
        x = torch.randn(64, 2, hd, generator=g) * 3
        y = rotate(x)
        assert torch.allclose(rotate(y), x, atol=1e-5), hd  # orthonormal and its own inverse
        codes, meta, exact = int4_rot_quantize(x)
        assert codes.shape == (64, 2, hd // 2) and meta.shape == (64, 2, hd // 64) and exact.dtype == torch.float16
        err = (int4_rot_dequantize(codes, meta) - y).norm() / y.norm()
        # measured 0.085 at every head dim (with the reader's 1.0075 gain): 16 levels on Gaussian groups of 64
        assert err < 0.1, (hd, float(err))


def test_int4_rot_bytes_equal_the_engine_writers():
    lib = engine()
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    ctx = 2048
    assert lib.swiftmk_attn_kv_bytes(ctx, 2) == INT4_ROT.nbytes(ctx, ENGINE_NKV, ENGINE_HD, 8)
    norm = torch.zeros(ENGINE_HD, dtype=torch.bfloat16, device="cuda")

    # the prefill writer (prefill.cu k_attn_prep), 1100 rows from position 0: it writes the ring for the last 512 only
    T = 1100
    v = rows(T, 0)
    k = rows(T, 1).cuda()
    vd = v.cuda()
    kv = torch.zeros(INT4_ROT.nbytes(ctx, ENGINE_NKV, ENGINE_HD, 8), dtype=torch.uint8, device="cuda")
    ld = ENGINE_NKV * ENGINE_HD
    args = (T, 0, None, 0, k.data_ptr(), ld, vd.data_ptr(), ld, None, norm.data_ptr(), kv.data_ptr(), ctx, 2, 1.0, 1.0)
    assert lib.smkp_attn_prep(*args, None, None, None, None, stream) == 0
    torch.cuda.synchronize()
    check_cache(kv.cpu(), ctx, v, 0, T - 512)

    # the engine's gather (kvrot::dequant_row, sink rows below position 4) un-rotates and rounds to bf16
    k_all = torch.empty(T, ENGINE_NKV, ENGINE_HD, dtype=torch.bfloat16, device="cuda")
    v_all = torch.empty_like(k_all)
    assert lib.smkp_gather_kv(kv.data_ptr(), ctx, 2, T, 1.0, 1.0, k_all.data_ptr(), v_all.data_ptr(), stream) == 0
    torch.cuda.synchronize()
    buf = kv.cpu()
    for name, got in (("k", k_all), ("v", v_all)):
        y = int4_rot_dequantize(
            section(buf, ctx, name + "c")[:, :T].transpose(0, 1), section(buf, ctx, name + "s")[:, :T].transpose(0, 1)
        )
        y[:4] = section(buf, ctx, name + "x").transpose(0, 1).float()
        assert torch.equal(got.cpu().view(torch.int16), rotate(y).bfloat16().view(torch.int16)), (
            f"{name} gather differs"
        )

    # the decode writer (attn.cu attn_prep_i4_kernel, one scale candidate per warp), 8-row steps
    kv.zero_()
    q_proj = torch.zeros(8, 24 * 512, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(8, 24 * 256, dtype=torch.bfloat16, device="cuda")
    ws = torch.empty(lib.swiftmk_attn_ws_bytes(), dtype=torch.uint8, device="cuda")
    pos0 = torch.zeros(1, dtype=torch.int32, device="cuda")
    for step in range(3):
        v = rows(8, 10 + step)
        k, vd = rows(8, 20 + step).cuda(), v.cuda()
        pos0.fill_(8 * step)
        args = (8, q_proj.data_ptr(), k.data_ptr(), vd.data_ptr(), norm.data_ptr(), norm.data_ptr(), kv.data_ptr(), ctx)
        assert (
            lib.swiftmk_attn_step(*args, 2, 1.0, 1.0, pos0.data_ptr(), None, out.data_ptr(), ws.data_ptr(), stream) == 0
        )
        torch.cuda.synchronize()
        check_cache(kv.cpu(), ctx, v, 8 * step, 8 * step)


def test_warpc_prep_kernel_writes_the_engine_bytes():
    """warpc_attn_prep_q on int4_rot without qk-norm and rope, so K reaches the writer unchanged too: K and V bytes
    equal the reference at every head dim, the query is the rotated row in fp16, and at the engine's shape (4 KV heads
    of 256) the V sections equal what the engine's prefill writer stores for the same rows."""
    require_gpu()
    major, minor = torch.cuda.get_device_capability()
    L = glue.load(10 * major + minor)
    stream = torch.cuda.current_stream().cuda_stream
    ctx, steps = 1024, 4
    for hd, hkv in ((64, 2), (128, 2), (256, ENGINE_NKV)):
        hq = 2 * hkv
        meta = (ctypes.c_longlong * 13)(*schedule.kv_meta(INT4_ROT, ctx, hkv, hd, 8))
        kv = torch.zeros(INT4_ROT.nbytes(ctx, hkv, hd, 8), dtype=torch.uint8, device="cuda")
        qout = torch.empty(8, hq, hd, dtype=torch.float16, device="cuda")
        pos0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        steps_rows = []
        for step in range(steps):
            q, k, v = rows(8, 3 * step, hq, hd), rows(8, 3 * step + 1, hkv, hd), rows(8, 3 * step + 2, hkv, hd)
            qd, kd, vd = q.cuda(), k.cuda(), v.cuda()
            pos0.fill_(8 * step)
            args = (8, qd.data_ptr(), kd.data_ptr(), vd.data_ptr(), None, None, qout.data_ptr(), kv.data_ptr(), meta)
            rc = L.warpc_attn_prep_q(*args, pos0.data_ptr(), hq, hkv, hd, 0, None, 0.0, ctx, 0, stream)
            assert rc == 0, rc
            torch.cuda.synchronize()
            assert torch.equal(qout.cpu().view(torch.int16), rotate(q.float()).half().view(torch.int16)), (hd, "q")
            buf = kv.cpu()
            check_cache(buf, ctx, k, 8 * step, 8 * step, "k")
            check_cache(buf, ctx, v, 8 * step, 8 * step, "v")
            steps_rows.append(v)
        if hd != ENGINE_HD or not os.environ.get("WARPC_SWIFTMK_LIB"):
            continue
        lib = engine()
        v = torch.cat(steps_rows).cuda()
        ekv = torch.zeros_like(kv)
        ld = ENGINE_NKV * ENGINE_HD
        norm = torch.zeros(ENGINE_HD, dtype=torch.bfloat16, device="cuda")
        args = (
            8 * steps,
            0,
            None,
            0,
            v.data_ptr(),
            ld,
            v.data_ptr(),
            ld,
            None,
            norm.data_ptr(),
            ekv.data_ptr(),
            ctx,
            2,
        )
        assert lib.smkp_attn_prep(*args, 1.0, 1.0, None, None, None, None, ctypes.c_void_p(stream)) == 0
        torch.cuda.synchronize()
        for name in ("vc", "vs", "vx", "vr"):
            got, want = section(buf, ctx, name), section(ekv.cpu(), ctx, name)
            assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name


def test_row_sections_and_aux_range_partition_the_buffer() -> None:
    """The per-position planes of every format fit their layout without overlapping, and the exact rows (sink, ring)
    are the contiguous tail `aux_range` names: what a KV export carries per row and what a snapshot carries as state."""
    nkv, hd, rows_max = 4, 256, 8
    for name, fmt in FORMATS.items():
        for ctx in (64, 1000):
            secs = fmt.row_sections(ctx, nkv, hd, rows_max)
            spans = sorted((s.offset, s.offset + s.nbytes) for s in secs.values())
            assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans)), (name, spans)
            total = fmt.nbytes(ctx, nkv, hd, rows_max)
            assert spans[-1][1] <= total and all(s.rows == ctx for s in secs.values()), name
            aux = fmt.aux_range(ctx, nkv, hd, rows_max)
            assert (aux is not None) == fmt.exact_rows, name
            if aux:
                assert aux[0] >= spans[-1][1] and aux[0] + aux[1] == total, (name, aux, spans[-1])
    sections = INT4_ROT.row_sections(1000, nkv, hd, rows_max)
    assert {n: s.row_bytes for n, s in sections.items()} == {"kc": 128, "vc": 128, "ks": 4, "vs": 4}
    assert INT4_ROT.aux_range(1000, nkv, hd, rows_max)[1] == 2 * nkv * 4 * hd * 2 + 2 * nkv * 512 * hd * 2


if __name__ == "__main__":
    run(globals())
