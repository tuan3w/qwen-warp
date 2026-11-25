"""The W4G128 GEMV family (warpc/cuda/w4.py, kernels/w4g128_gemv.cu): the capabilities against the hand engine's, the
repack, and the kernel's gates at kernel level.

  gate 1: bitwise against a CPU model of the accumulation order in the kernel header (`refs`; the tensor core's
          accumulation is modelled by `tc_mma`)
  gate 2: row invariance, rows 1..8 of one launch have the bits of the same row run alone
  gate 3: bitwise against the hand kernel smk_w4r_gemv
  speed:  GB/s of both on the 27B shapes, CUDA-graph timed

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> WARPC_TEST_SWIFT27=<27B checkpoint dir> \\
      uv run --package warpc python warpc/tests/test_w4.py

The kernel tests need a CUDA device (gate 3 and the speed test also $WARPC_SWIFTMK_LIB); they skip without. The caps
test uses $WARPC_TEST_SWIFT27 when set and a 4-layer graph of the same shapes otherwise. The speed test is skipped
unless $WARPC_W4_SPEED is set (it takes the card for about a minute).
"""

import ctypes as C
import os
import statistics
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import torch

from harness import Skip, run
from test_compiler import Qwen35Shapes
from warpc import frontend, sources
from warpc.compiler import CapsTable, Pack
from warpc.compiler.fuse import fuse
from warpc.cuda import glue, w4
from warpc.cuda.pack import Packer
from warpc.cuda.table import RTX3090, TABLE
from warpc.ir import WeightDecl

# (N, K, epilogue) of the 27B's W4 GEMVs by bytes: attention q|k|v, the 14336-row projection of the other mixers (the
# qkv|z of DeltaNet is 16384), o / out, gate|up with the silu epilogue, down
SHAPES = [
    (16384, 5120, w4.EPI_SEGS),
    (14336, 5120, w4.EPI_SEGS),
    (5120, 6144, w4.EPI_SEGS),
    (34816, 5120, w4.EPI_SILU),
    (5120, 17408, w4.EPI_SEGS),
]


def _device():
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    return glue.load(major * 10 + minor)


def _hand():
    from warpc.swiftmk import lib as smk

    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    return smk.load()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _random_w4(n: int, k: int, gen: torch.Generator) -> dict:
    packed = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int64, generator=gen).to(torch.int32)
    scale = (torch.rand(n, k // 128, generator=gen) * 0.02 + 1e-3).to(torch.bfloat16)
    return {"packed": packed, "scale": scale}


class _Source:
    """A Source of natural-layout W4G128 tensors held in memory."""

    def __init__(self, raw: dict):
        self._raw = raw

    def raw(self, name: str) -> dict:
        return self._raw[name]


def _pack(raw: dict, kind: str, device: str = "cuda") -> torch.Tensor:
    names = list(raw)
    decls = {w: WeightDecl(w, (raw[w]["packed"].shape[0], raw[w]["packed"].shape[1] * 8), "w4g128", w) for w in names}
    return Packer(decls, _Source(raw), device).build(Pack("|".join(names), kind, names))


def _launch(lib, buf, x, n, k, T, kind, ends, outs, ks=None):
    """One launch of the kernel on the pack `buf`; outs: the segment outputs, or the one silu output."""
    o, e = (C.c_void_p * 3)(), (C.c_int * 3)()
    for j, t in enumerate(outs):
        o[j] = t.data_ptr()
    for j in range(3):
        e[j] = ends[min(j, len(ends) - 1)] if ends else n
    p = buf.data_ptr()
    args = (p, p + w4.scale_offset(n, k), x.data_ptr(), n, k, T, kind, e, o, ks or w4.w4_ks(n, k), _stream())
    return lib.warpc_w4g128_gemv(*args)


def _hand_launch(hand, buf, x, n, k, T, kind, ends, outs):
    from warpc.swiftmk import lib as smk

    epi = smk.Epi()
    epi.kind = {w4.EPI_SEGS: smk.EPI_SEGS, w4.EPI_SILU: smk.EPI_SILU, w4.EPI_F32: smk.EPI_F32}[kind]
    for j, t in enumerate(outs):
        epi.out[j] = t.data_ptr()
    for j in range(3):
        epi.end[j] = ends[min(j, len(ends) - 1)] if ends else n
    p = buf.data_ptr()
    hand.check(hand.smk_w4r_gemv(p, p + w4.scale_offset(n, k), x.data_ptr(), n, k, T, C.byref(epi), _stream()), "hand")


def _outs(kind, ends, n, T):
    if kind == w4.EPI_F32:
        return [torch.empty(T, n, dtype=torch.float32, device="cuda")]
    if kind == w4.EPI_SILU:
        return [torch.empty(T, n // 2, dtype=torch.bfloat16, device="cuda")]
    widths = [b - a for a, b in zip((0, *ends), ends, strict=False)]
    return [torch.empty(T, m, dtype=torch.bfloat16, device="cuda") for m in widths]


def _ends(n: int) -> tuple[int, ...]:
    """Three unequal 16-multiple segments covering n rows (n % 48 == 16 or 32 allowed: the last takes the rest)."""
    a = (n // 2 // 16) * 16
    b = a + ((n - a) // 2 // 16) * 16
    return (a, b, n)


def _widths(ends: tuple[int, ...]) -> list[int]:
    return [b - a for a, b in zip((0, *ends), ends, strict=False)]


def _make(kind: int, n: int, k: int, gen) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Random weights of an (n, k) GEMV packed for `kind`: (pack, ends)."""
    if kind == w4.EPI_SILU:
        raw = {"g": _random_w4(n // 2, k, gen), "u": _random_w4(n // 2, k, gen)}
        return _pack(raw, "w4g_gateup"), ()
    ends = _ends(n)
    raw = {f"w{j}": _random_w4(m, k, gen) for j, m in enumerate(_widths(ends))}
    return _pack(raw, "w4g_cat"), ends


def _case(kind: int, gen, n=1280, k=1024):
    """(pack, ends, x [8, k]) of random weights."""
    buf, ends = _make(kind, n, k, gen)
    return buf, ends, (torch.randn(8, k, generator=gen) * 2).to(torch.bfloat16).cuda()


def test_pack_layout_is_the_hand_repack():
    """w4g_cat / w4g_gateup are byte for byte the hand engine's w4r_cat / w4r_gateup."""
    from warpc.swiftmk.pack import Packer as HandPacker

    gen = torch.Generator().manual_seed(1)
    k = 512
    for names, ns, mine, theirs in (
        (("a", "b", "c"), (96, 32, 48), "w4g_cat", "w4r_cat"),
        (("a",), (64,), "w4g", "w4r"),
        (("g", "u"), (80, 80), "w4g_gateup", "w4r_gateup"),
    ):
        raw = {nm: _random_w4(n, k, gen) for nm, n in zip(names, ns, strict=True)}
        decls = {nm: WeightDecl(nm, (n, k), "w4g128", nm) for nm, n in zip(names, ns, strict=True)}
        pack = Pack("|".join(names), mine, list(names))
        a = Packer(decls, _Source(raw), "cpu").build(pack)
        b = HandPacker(decls, _Source(raw), "cpu").build(Pack(pack.name, theirs, list(names)))
        assert torch.equal(a, b), mine


def _regions(caps, g):
    res = fuse(g, CapsTable(caps, RTX3090, TABLE.unfused_ctas), 8)
    return res


def test_caps_claim_the_hand_step():
    """On the 27B (or its 4-layer shape twin) the two caps claim exactly what the hand engine's s_w4_gemv /
    s_w4_gemv_silu claim, region for region: covered nodes, operand roles, launch attrs, pack parts."""
    from warpc.swiftmk import table as hand

    path = os.environ.get("WARPC_TEST_SWIFT27")
    g = frontend.build(sources.open(Path(path)) if path else Qwen35Shapes(), ctx_max=4608, kv="bf16")
    mine = _regions(w4.CAPS, g).regions
    theirs = _regions([c for c in hand.CAPS if c.kernel.startswith("s_w4")], g).regions
    layers = 64 if path else 4
    assert Counter(r.kernel for r in mine) == {"g_w4_gemv": 3 * layers, "g_w4_gemv_silu": layers}, Counter(
        r.kernel for r in mine
    )
    rename = {"s_w4_gemv": "g_w4_gemv", "s_w4_gemv_silu": "g_w4_gemv_silu"}
    kinds = {"w4r": "w4g", "w4r_cat": "w4g_cat", "w4r_gateup": "w4g_gateup"}

    def key(r, k, packs):
        return (
            k,
            sorted(r.covers),
            r.ins,
            r.outs,
            r.weights,
            r.attrs,
            [(p.name, packs[p.kind] if packs else p.kind, p.parts) for p in r.packs],
        )

    a = sorted(key(r, r.kernel, None) for r in mine)
    b = sorted(key(r, rename[r.kernel], kinds) for r in theirs)
    assert a == b
    ends = {tuple(r.attrs["ends"]) for r in mine if r.kernel == "g_w4_gemv"}
    assert (12288, 13312, 14336) in ends and (10240, 16384) in ends, ends


def test_unsupported_shapes_are_refused_by_limits():
    """K % 128 and ends % 16 are limits of the caps, not silent miscomputes."""
    from warpc.compiler.caps import limits_ok

    cap = next(c for c in w4.CAPS if c.kernel == "g_w4_gemv")
    c = {"N": 4096, "K": 5120, "T": 8, "ends": 16, "dtype": "bf16"}
    assert limits_ok(cap.kernel, cap.limits, c) == []
    for bad in ({"K": 5184}, {"N": 4104}, {"ends": 8}, {"T": 9}):
        assert limits_ok(cap.kernel, cap.limits, {**c, **bad}), bad


def test_ks_choice():
    assert [w4.w4_ks(n, k) for n, k, _ in SHAPES] == [4, 8, 4, 4, 4]
    assert w4.w4_ks(5136, 5120) == 8  # 321 tiles: odd, KS 4 needs pairs
    assert w4.w4_ks(5120, 128 * 5) == 1  # 5 groups: only KS 1 divides


# ------------------------------------------------------------------------------------------------ CPU model (gate 1)
# The tensor core of sm_86 for bf16 m16n8k16, as measured with a probe kernel (3.2M random mma in each of three
# regimes: int weights x bf16 activations with a partial sum as C, C = 0, C spanning 2^-30..2^14: all bitwise equal to
# this model): the 16 k are two blocks of 8 (hardware k 0..7, 8..15) added in sequence, C first. A block adds its 8
# exact products and the incoming accumulator in one fused step: every addend is truncated (toward zero) to a multiple
# of 2^(emax - 24), where emax = max(exponent(a) + exponent(b) over the products, exponent(c)), the aligned addends are
# summed exactly, and the sum is truncated toward zero to fp32.


def _exp2(v: np.ndarray) -> np.ndarray:
    """floor(log2|v|), -1e6 for 0."""
    return np.where(v == 0, -(10**6), np.frexp(v)[1] - 1).astype(np.int64)


def _rz_f32(s: np.ndarray) -> np.ndarray:
    r = s.astype(np.float32)
    over = np.abs(r.astype(np.float64)) > np.abs(s)
    return np.where(over, np.nextafter(r, np.float32(0)), r).astype(np.float64)


def tc_mma(c: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """c [...] fp32 values (as float64), a / b [..., 16] bf16 values: one mma m16n8k16 per output element."""
    for i in (0, 8):
        pa, pb = a[..., i : i + 8], b[..., i : i + 8]
        prod = pa * pb
        e = np.where(prod == 0, -(10**6), _exp2(pa) + _exp2(pb))
        emax = np.maximum(e.max(-1), _exp2(c))
        emax = np.where(emax < -(10**5), 0, emax)
        q = np.exp2((emax - 24).astype(np.float64))
        s = np.trunc(prod / q[..., None]).sum(-1) * q + np.trunc(c / q) * q
        c = _rz_f32(s)
    return c


# hardware k = 0..15 of one mma (step u, half h) -> column inside the 128-column group: chunk t feeds k = 2t, 2t + 1,
# 2t + 8, 2t + 9 with the word u's columns 4h + 0, 1, 2, 3 (w4g128_gemv.cuh); the list is in ascending k
_COLS = np.array(
    [
        [[32 * t + 8 * u + 4 * h + i + 2 * half for half in (0, 1) for t in range(4) for i in (0, 1)] for h in (0, 1)]
        for u in range(4)
    ]
)


def refs(x: torch.Tensor, packed: torch.Tensor, scale: torch.Tensor, ks: int) -> np.ndarray:
    """The kernel's fp32 sums [T, N] for natural-layout weights (words [N, K/8], scales [N, K/128]) by the order of the
    kernel header: per 128-column group 8 chained mma, acc = fma(scale, part, acc) over the warp's groups, the KS
    accumulators added left to right."""
    n, k8 = packed.shape
    nkg = k8 // 16
    sh = torch.arange(8, dtype=torch.int64) * 4
    nib = (packed.to(torch.int64).unsqueeze(-1) >> sh) & 0xF  # [N, K/8, 8]: column 8j + e
    w = (nib.reshape(n, -1) - 8).to(torch.float64).numpy()
    xs = x.to(torch.float64).numpy()
    sc = scale.to(torch.float64).numpy()
    accs = []
    for r in range(ks):
        acc = np.zeros((n, xs.shape[0]))
        for kg in range(r, nkg, ks):
            wg, xg = w[:, kg * 128 : (kg + 1) * 128], xs[:, kg * 128 : (kg + 1) * 128]
            part = np.zeros((n, xs.shape[0]))
            for u in range(4):
                for h in (0, 1):
                    cols = _COLS[u, h]
                    part = tc_mma(part, wg[:, None, cols], xg[None, :, cols])
            acc = (sc[:, kg, None] * part + acc).astype(np.float32).astype(np.float64)  # fma: exact product in f64
        accs.append(acc)
    out = accs[0]
    for a in accs[1:]:
        out = (out + a).astype(np.float32).astype(np.float64)
    return out.T


def _bf16(s: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(s.astype(np.float32)).to(torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def test_gate1_cpu_model():
    """Bitwise against the CPU model of the accumulation order: the fp32 sums and the three bf16 segments of every KS,
    and the silu pair."""
    lib = _device()
    gen = torch.Generator().manual_seed(11)
    n, k = 384, 5120
    x = (torch.randn(8, k, generator=gen) * torch.exp2(torch.randint(-2, 3, (8, k), generator=gen).float())).to(
        torch.bfloat16
    )
    ends = _ends(n)
    raw = {f"w{j}": _random_w4(m, k, gen) for j, m in enumerate(_widths(ends))}
    buf = _pack(raw, "w4g_cat")
    nat_w = torch.cat([raw[w]["packed"] for w in raw])
    nat_s = torch.cat([raw[w]["scale"] for w in raw])
    for ks in (1, 4, 8):
        sums = refs(x, nat_w, nat_s, ks)  # [T, N]
        f32 = _outs(w4.EPI_F32, (), n, 8)
        segs = _outs(w4.EPI_SEGS, ends, n, 8)
        assert _launch(lib, buf, x.cuda(), n, k, 8, w4.EPI_F32, (), f32, ks) == 0
        assert _launch(lib, buf, x.cuda(), n, k, 8, w4.EPI_SEGS, ends, segs, ks) == 0
        torch.cuda.synchronize()
        bad32 = int((_bits(f32[0]) != torch.from_numpy(sums.astype(np.float32)).view(torch.int32)).sum())
        got = torch.cat([t.cpu() for t in segs], 1)
        bad16 = int((got.view(torch.int16) != _bf16(sums).view(torch.int16)).sum())
        print(f"gate 1: KS={ks}: {bad32} of {got.numel()} fp32 sums, {bad16} bf16 outputs differ from the CPU model")
        assert bad32 == 0 and bad16 == 0
    # silu pair: the model's gate / up sums, the epilogue chain on the device with the kernel's formula
    m = 256
    raw = {"g": _random_w4(m, k, gen), "u": _random_w4(m, k, gen)}
    buf = _pack(raw, "w4g_gateup")
    ks = w4.w4_ks(2 * m, k)
    g = _bf16(refs(x, raw["g"]["packed"], raw["g"]["scale"], ks)).cuda().float()
    u = _bf16(refs(x, raw["u"]["packed"], raw["u"]["scale"], ks)).cuda().float()
    want = ((g / (1 + torch.exp(-g))).bfloat16().float() * u).bfloat16()
    o = _outs(w4.EPI_SILU, (), 2 * m, 8)
    assert _launch(lib, buf, x.cuda(), 2 * m, k, 8, w4.EPI_SILU, (), o) == 0
    torch.cuda.synchronize()
    bad = int((_bits(o[0]) != want.cpu().view(torch.int16)).sum())
    print(f"gate 1: silu pair KS={ks}: {bad} of {want.numel()} outputs differ from the CPU model")
    assert bad == 0


def test_gate2_row_invariance():
    """Row t of a T-row launch equals the bits of that row run alone, for every T, every epilogue and every KS."""
    lib = _device()
    gen = torch.Generator().manual_seed(2)
    for kind in (w4.EPI_SEGS, w4.EPI_SILU, w4.EPI_F32):
        buf, ends, x = _case(kind, gen)
        n, k = 1280, 1024
        for ks in (1, 4, 8):
            solo = []
            for t in range(8):
                o = _outs(kind, ends, n, 1)
                assert _launch(lib, buf, x[t : t + 1], n, k, 1, kind, ends, o, ks) == 0
                solo.append(o)
            for T in range(1, 9):
                o = _outs(kind, ends, n, T)
                assert _launch(lib, buf, x[:T], n, k, T, kind, ends, o, ks) == 0
                torch.cuda.synchronize()
                for j, seg in enumerate(o):
                    for t in range(T):
                        same = torch.equal(_bits(seg[t]), _bits(solo[t][j][0]))
                        assert same, f"kind {kind} ks {ks} T={T} row {t} seg {j}"
    print("gate 2: rows 1..8 equal the row run alone (every epilogue incl. fp32, KS 1 / 4 / 8)")


def test_gate3_equals_the_hand_kernel():
    """Bitwise against smk_w4r_gemv, on random weights at the 27B shapes (K-split chosen as the hand does)."""
    lib, hand = _device(), _hand()
    gen = torch.Generator().manual_seed(3)
    for n, k, kind in SHAPES:
        buf, ends = _make(kind, n, k, gen)
        x = (torch.randn(8, k, generator=gen) * 2).to(torch.bfloat16).cuda()
        for epi in (kind, w4.EPI_F32):  # the epilogue, and the fp32 sums under it (all the bits of the order)
            oends = () if epi == w4.EPI_F32 else ends
            worst = 0
            for T in range(1, 9):
                a, b = _outs(epi, oends, n, T), _outs(epi, oends, n, T)
                assert _launch(lib, buf, x, n, k, T, epi, oends, a) == 0
                _hand_launch(hand, buf, x, n, k, T, epi, oends, b)
                torch.cuda.synchronize()
                worst += sum(int((_bits(u) != _bits(v)).sum()) for u, v in zip(a, b, strict=True))
            name = {w4.EPI_SEGS: "segs", w4.EPI_SILU: "silu", w4.EPI_F32: "fp32"}[epi]
            print(f"gate 3: N={n} K={k} {name}: {worst} differing elements, T=1..8")
            assert worst == 0


def _graph_us(fn, launches=10, replays=200) -> float:
    """CUDA-graph time of one call of fn (us): `launches` back-to-back calls per graph, `replays` replays."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(launches):
                fn()
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(replays):
            g.replay()
        b.record()
        b.synchronize()
    return a.elapsed_time(b) * 1e3 / (replays * launches)


def test_speed_against_the_hand_kernel():
    """GB/s of both kernels on the 27B shapes (weights + scales + x bytes), interleaved A/B rounds, best of rounds."""
    if not os.environ.get("WARPC_W4_SPEED"):
        raise Skip("WARPC_W4_SPEED not set")
    lib, hand = _device(), _hand()
    gen = torch.Generator().manual_seed(4)
    rows = []
    for n, k, kind in SHAPES:
        buf = torch.randint(0, 256, (n * k // 2 + n * k // 64,), dtype=torch.uint8, generator=gen).cuda()
        sc = buf[n * k // 2 :].view(torch.bfloat16)
        sc.copy_((torch.rand(sc.numel(), generator=gen) * 0.02 + 1e-3).to(torch.bfloat16))
        ends = () if kind == w4.EPI_SILU else _ends(n)
        x = torch.randn(8, k, generator=gen).to(torch.bfloat16).cuda()
        for T in (1, 2, 4, 8):
            a, b = _outs(kind, ends, n, T), _outs(kind, ends, n, T)
            mine = partial(_launch, lib, buf, x, n, k, T, kind, ends, a)
            theirs = partial(_hand_launch, hand, buf, x, n, k, T, kind, ends, b)
            ta, tb = [], []
            for _ in range(5):
                tb.append(_graph_us(theirs))
                ta.append(_graph_us(mine))
            nbytes = n * k // 2 + n * k // 64 + T * k * 2
            ua, ub = min(ta), min(tb)
            epi = "silu" if kind == w4.EPI_SILU else "segs"
            rows.append((n, k, epi, T, ua, ub, nbytes / ua / 1e3, nbytes / ub / 1e3))
    print("   N      K  epi  T   warpc us  hand us  warpc GB/s  hand GB/s  ratio")
    for n, k, e, T, ua, ub, ga, gb in rows:
        print(f"{n:6d} {k:6d} {e:5s} {T} {ua:9.1f} {ub:8.1f} {ga:10.1f} {gb:10.1f} {ga / gb:6.3f}")
    t1 = [r[6] / r[7] for r in rows if r[3] == 1]
    print(f"T=1 ratio: min {min(t1):.3f} median {statistics.median(t1):.3f}")


if __name__ == "__main__":
    run(globals())
