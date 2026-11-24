"""The swiftmk backend's binding of the hand engine's kernel library: the ABI guard, the W4 repack and the int4 head
copy checked at kernel level (the repacked GEMV equals the natural-layout GEMV bit for bit), and the executor's DeltaNet
ring cursors against the hand engine's own k_advance.

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tests/test_swiftmk.py

Needs a CUDA device and $WARPC_SWIFTMK_LIB (built by `make engine`); the tests skip without either.
"""

import ctypes as C
import itertools
import os
from collections import Counter
from pathlib import Path

import torch

from harness import Skip, run
from warpc import frontend, sources
from warpc.compiler import Pack, lower
from warpc.errors import LoweringError
from warpc.ir import WeightDecl
from warpc.kvcache import FORMATS
from warpc.sources import ModelConfig
from warpc.swiftmk import SwiftmkBackend
from warpc.swiftmk import lib as smk
from warpc.swiftmk.executor import ring_advance
from warpc.swiftmk.pack import Packer, quant_head_i4, w4_scale_offset

# the hand engine's 64-layer step (engine.cu smk_run_step): per layer the input norm (+ the residual add from layer 1),
# the mixer's GEMVs and kernel, the post-attention norm, the MLP's dual and down GEMVs; then the final norm and the head
STEP_27B = {
    "s_embed": 1,
    "s_rmsnorm": 1,
    "s_rmsnorm_add": 128,
    "s_w4_gemv": 192,
    "s_w4_gemv_silu": 64,
    "s_ab_gemv": 48,
    "s_deltanet": 48,
    "s_attn": 16,
}
FIRST_LAYER = ["s_embed", "s_rmsnorm", "s_w4_gemv", "s_ab_gemv", "s_deltanet", "s_w4_gemv", "s_rmsnorm_add"]


def _lib() -> smk.Library:
    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    return smk.load()


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


class _W4Source:
    """A Source of natural-layout W4G128 tensors and bf16 tensors held in memory."""

    def __init__(self, raw: dict, dense: dict):
        self._raw, self._dense = raw, dense

    def raw(self, name: str) -> dict:
        return self._raw[name]

    def get(self, name: str) -> torch.Tensor:
        return self._dense[name]


def _random_w4(n: int, k: int, gen: torch.Generator) -> dict:
    packed = torch.randint(-(2**31), 2**31, (n, k // 8), dtype=torch.int64, generator=gen).to(torch.int32)
    scale = (torch.rand(n, k // 128, generator=gen) * 0.02 + 1e-3).to(torch.bfloat16)
    return {"packed": packed, "scale": scale}


def _epi(kind: int, outs: list[torch.Tensor], ends: tuple[int, int, int] = (0, 0, 0)) -> smk.Epi:
    e = smk.Epi()
    e.kind = kind
    for j, o in enumerate(outs):
        e.out[j] = o.data_ptr()
    for j in range(3):
        e.end[j] = ends[j]
    return e


def test_load_needs_env():
    saved = os.environ.pop(smk.ENV, None)
    try:
        smk.load()
    except RuntimeError as e:
        assert smk.ENV in str(e), e
    else:
        raise AssertionError("load() without $WARPC_SWIFTMK_LIB must raise")
    finally:
        if saved is not None:
            os.environ[smk.ENV] = saved


def test_abi_guard():
    lib = _lib()
    assert lib.sha256 == smk.sha256(lib.path)
    assert smk.abi_mismatches(lib.cdll) == []
    for ctx in (385, 4608, 66048):
        assert lib.kv_bytes(ctx, "int4_rot") == FORMATS["int4_rot"].nbytes(ctx, 4, 256, 8)
        assert lib.kv_bytes(ctx, "bf16") == 2 * 4 * ctx * 256 * 2


def test_w4r_segments_equal_natural():
    """w4r_cat (q|k|v rows, three segments) through smk_w4r_gemv == the natural layout through smk_w4_gemv, bit for bit,
    at every step row count."""
    lib = _lib()
    gen = torch.Generator().manual_seed(1)
    segs, k = (384, 128, 64), 1024
    raw = {f"w{j}": _random_w4(n, k, gen) for j, n in enumerate(segs)}
    decls = {f"w{j}": WeightDecl(f"w{j}", (n, k), "w4g128", f"w{j}") for j, n in enumerate(segs)}
    packer = Packer(decls, _W4Source(raw, {}))
    rep = packer.build(Pack("w0|w1|w2", "w4r_cat", list(decls)))
    n = sum(segs)
    nat_w = torch.cat([raw[w]["packed"] for w in decls]).cuda()
    nat_s = torch.cat([raw[w]["scale"] for w in decls]).cuda()
    ends = (segs[0], segs[0] + segs[1], n)
    for T in range(1, 9):
        x = torch.randn(T, k, generator=gen).to(torch.bfloat16).cuda()
        want = [torch.empty(T, m, dtype=torch.bfloat16, device="cuda") for m in segs]
        got = [torch.empty_like(o) for o in want]
        e1, e2 = _epi(smk.EPI_SEGS, want, ends), _epi(smk.EPI_SEGS, got, ends)
        lib.check(lib.smk_w4_gemv(nat_w.data_ptr(), nat_s.data_ptr(), x.data_ptr(), n, k, T, e1, _stream()), "w4")
        p = rep.data_ptr()
        lib.check(lib.smk_w4r_gemv(p, p + w4_scale_offset(n, k), x.data_ptr(), n, k, T, e2, _stream()), "w4r")
        torch.cuda.synchronize()
        for a, b in zip(want, got, strict=True):
            assert torch.equal(a, b), f"T={T}: repacked GEMV differs from the natural layout"


def test_w4r_gateup_equal_natural():
    """w4r_gateup through the SiLU epilogue == the natural interleave (engine.h smk_interleave_gateup) through
    smk_w4_gemv, and the natural result equals bf16(bf16(silu(bf16(gate))) * bf16(up)) of separate GEMVs."""
    lib = _lib()
    gen = torch.Generator().manual_seed(2)
    n, k = 256, 512
    raw = {"g": _random_w4(n, k, gen), "u": _random_w4(n, k, gen)}
    decls = {w: WeightDecl(w, (n, k), "w4g128", w) for w in raw}
    rep = Packer(decls, _W4Source(raw, {})).build(Pack("g|u", "w4r_gateup", ["g", "u"]))
    nat = {}
    for key in ("packed", "scale"):
        g, u = raw["g"][key], raw["u"][key]
        nat[key] = torch.stack([g.view(n // 8, 8, -1), u.view(n // 8, 8, -1)], 1).reshape(2 * n, -1).cuda()
    for T in (1, 3, 8):
        x = torch.randn(T, k, generator=gen).to(torch.bfloat16).cuda()
        want = torch.empty(T, n, dtype=torch.bfloat16, device="cuda")
        got = torch.empty_like(want)
        st = _stream()
        rc = lib.smk_w4_gemv(
            nat["packed"].data_ptr(), nat["scale"].data_ptr(), x.data_ptr(), 2 * n, k, T, _epi(smk.EPI_SILU, [want]), st
        )
        lib.check(rc, "w4 silu")
        p = rep.data_ptr()
        e = _epi(smk.EPI_SILU, [got])
        lib.check(lib.smk_w4r_gemv(p, p + w4_scale_offset(2 * n, k), x.data_ptr(), 2 * n, k, T, e, st), "w4r silu")
        gu = [torch.empty(T, n, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
        for w, o in zip(("g", "u"), gu, strict=True):
            pw, ps = raw[w]["packed"].cuda(), raw[w]["scale"].cuda()
            ends = (n, n, n)
            lib.check(lib.smk_w4_gemv(pw.data_ptr(), ps.data_ptr(), x.data_ptr(), n, k, T, _epi(0, [o], ends), st), w)
        torch.cuda.synchronize()
        assert torch.equal(want, got), f"T={T}: repacked gate|up differs from the natural interleave"
        silu = torch.nn.functional.silu(gu[0].float()).to(torch.bfloat16)
        assert torch.equal(want, (silu.float() * gu[1].float()).to(torch.bfloat16)), f"T={T}: SiLU epilogue chain"


def test_head_i4r_matches_natural():
    """head_i4r through smk_headq_gemv(I4R) matches the natural int4 copy through smk_headq_gemv(I4). The two layouts'
    kernels sum the fp32 products in different orders (measured: 1 ulp apart on unrounded logits), so the layout is
    checked to 1e-5 relative (the bf16 GEMV tests above compare rounded outputs, equal on their fixed inputs)."""
    lib = _lib()
    gen = torch.Generator().manual_seed(3)
    n, k = 2048, 5120  # head_q.h: K is the model's hidden size
    w = (torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16)
    decls = {"head": WeightDecl("head", (n, k), "dense", "head")}
    rep = Packer(decls, _W4Source({}, {"head": w})).build(Pack("head", "head_i4r", ["head"]))
    # the natural copy: the same quantisation without the repack
    nw = torch.empty(n, k // 8, dtype=torch.int32, device="cuda")
    ns = torch.empty(n, k // 128, dtype=torch.bfloat16, device="cuda")
    quant_head_i4(w.cuda(), nw, ns)
    for T in (1, 2, 8):
        x = torch.randn(T, k, generator=gen).to(torch.bfloat16).cuda()
        want = torch.empty(T, n, dtype=torch.float32, device="cuda")
        got = torch.empty_like(want)
        st = _stream()
        lib.check(lib.smk_headq_gemv(2, nw.data_ptr(), ns.data_ptr(), x.data_ptr(), n, k, T, want.data_ptr(), st), "i4")
        p = rep.data_ptr()
        rc = lib.smk_headq_gemv(smk.HEADQ_I4R, p, p + w4_scale_offset(n, k), x.data_ptr(), n, k, T, got.data_ptr(), st)
        lib.check(rc, "i4r")
        torch.cuda.synchronize()
        err = float((want - got).abs().max() / want.abs().max())
        assert err < 1e-5, f"T={T}: repacked int4 head differs from the natural layout by {err:.3g} (relative)"


def _swift27() -> Path:
    p = os.environ.get("WARPC_TEST_SWIFT27")
    if not p:
        raise Skip("WARPC_TEST_SWIFT27 (the 27B checkpoint directory) not set")
    return Path(p)


def test_27b_lowers_to_the_hand_step():
    """The 27B graph claims exactly the hand engine's launches, for every KV format and both heads (CPU: headers
    and config only)."""
    src = sources.open(_swift27())
    for kv in ("bf16", "fp8", "int4_rot"):
        if kv not in FORMATS:
            continue
        g = frontend.build(src, ctx_max=4608, kv=kv)
        for head, kernel in (("bf16", "s_head"), ("int4_refine", "s_head_refine")):
            plan = lower(g, SwiftmkBackend(head), rows_max=8)
            assert Counter(t.kernel for t in plan.tasks) == {**STEP_27B, kernel: 1}, plan.summary()
            tasks = plan.tasks
            assert [t.kernel for t in tasks[: len(FIRST_LAYER)]] == FIRST_LAYER
            assert tasks[2].attrs["ends"] == [10240, 16384] and tasks[3].attrs["na"] == 48
            attn = next(t for t in tasks if t.kernel == "s_attn")
            assert attn.attrs["cache"] == kv and ("k_scale" in attn.weights) == (kv == "fp8")
            qkv = tasks[tasks.index(attn) - 1]
            assert qkv.kernel == "s_w4_gemv" and qkv.attrs["ends"] == [12288, 13312, 14336]
        try:
            lower(g, SwiftmkBackend(), rows_max=9)
        except LoweringError as e:
            assert "T=9 > 8" in str(e), str(e)
        else:
            raise AssertionError("rows_max 9 exceeds the DeltaNet and attention kernels' 8 rows")


class _ShapeSource:
    def __init__(self, config: ModelConfig):
        self.config = config

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> None:
        return None


def test_dense_graph_is_refused_by_the_offset_limit():
    """A dense Qwen3 graph (norms x̂ w, no +1 offset) must not lower onto the hand kernels, which compute x̂ (1 + w)."""
    hf = {
        "model_type": "qwen3",
        "vocab_size": 512,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 1,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e7,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
    }
    g = frontend.build(_ShapeSource(ModelConfig.from_hf(hf)), ctx_max=256)
    try:
        lower(g, SwiftmkBackend(), rows_max=8)
    except LoweringError as e:
        assert "offset=0.0 not in (1.0,)" in str(e), str(e)
    else:
        raise AssertionError("a graph without norm offsets lowered onto the swiftmk table")


# engine.h smk_layer_w / smk_model_w, transcribed for the ring test's engine
class _LayerW(C.Structure):
    _fields_ = [
        ("kind", C.c_int),
        *[(n, C.c_void_p) for n in ("ln1", "ln2", "qkv_w", "qkv_s", "ab", "conv_w", "A_log", "dt_bias", "norm_w")],
        *[(n, C.c_void_p) for n in ("q_norm", "k_norm")],
        ("k_scale", C.c_float),
        ("v_scale", C.c_float),
        *[(n, C.c_void_p) for n in ("o_w", "o_s", "gu_w", "gu_s", "dn_w", "dn_s")],
    ]


class _ModelW(C.Structure):
    _fields_ = [
        ("layers", _LayerW * 64),
        ("embed", C.c_void_p),
        ("final_norm", C.c_void_p),
        ("lm_head", C.c_void_p),
        ("head_q_kind", C.c_int),
        ("head_q", C.c_void_p),
        ("head_qs", C.c_void_p),
    ]


DS_CUR, DS_PEND, DS_N = 1, 35, 37  # engine_impl.h: ds[] indices at SMK_TMAX 16


def test_swiftmk_ring():
    """The executor's cur / pend update equals the hand engine's k_advance: a tmax 8, one-state-slot engine (the
    compare27 configuration) steps T = 1..8 three times, crossing the 16-row delta-buffer wrap repeatedly, and its
    device scalars are read after every step. The ring scalars do not depend on the weights: every weight pointer is
    one zero buffer (rows read: token 0 of the embedding, at most the 89 MB of the gate|up matrix)."""
    lib = _lib()
    cd = lib.cdll
    cd.smk_engine_create_slot.argtypes = [C.POINTER(_ModelW), C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p]
    cd.smk_engine_create_slot.restype = C.c_void_p
    cd.smk_engine_destroy.argtypes = [C.c_void_p]
    cd.smk_engine_set_tokens.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_void_p]
    cd.smk_engine_step.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p]
    cd.smk_engine_reset.argtypes = [C.c_void_p, C.c_void_p]
    cd.smk_engine_read.argtypes = [C.c_void_p, C.c_int, C.c_void_p, C.c_size_t]
    zero = torch.zeros(96 << 20, dtype=torch.uint8, device="cuda")
    z = zero.data_ptr()
    m = _ModelW()
    for i in range(64):
        L = m.layers[i]
        L.kind = int(i % 4 == 3)
        for name, _ in _LayerW._fields_:
            if name not in ("kind", "k_scale", "v_scale"):
                setattr(L, name, z)
        L.k_scale = L.v_scale = 1.0
    m.embed = m.final_norm = m.lm_head = z
    tmax = 8
    h = cd.smk_engine_create_slot(C.byref(m), 64, 0, tmax, 1, None)
    assert h, "smk_engine_create_slot failed"
    try:
        st = _stream()
        lib.check(cd.smk_engine_reset(h, st), "reset")
        cur = torch.zeros(1, dtype=torch.int32, device="cuda")
        pend = torch.zeros(1, dtype=torch.int32, device="cuda")
        ds = (C.c_int * DS_N)()
        hand, ours = [], []
        for T in [*range(1, tmax + 1)] * 3:
            toks = (C.c_int * T)(*([0] * T))
            lib.check(cd.smk_engine_set_tokens(h, toks, T, st), "set_tokens")
            lib.check(cd.smk_engine_step(h, T, 0, st), "step")
            torch.cuda.synchronize()
            lib.check(cd.smk_engine_read(h, 0, C.cast(ds, C.c_void_p), C.sizeof(ds)), "read")
            hand.append((ds[DS_CUR], ds[DS_PEND]))
            ring_advance(cur, pend, T, tmax + 1)
            ours.append((int(cur), int(pend)))
        wraps = sum(1 for (_, a), (_, b) in itertools.pairwise(hand) if (a >> 13) != (b >> 13))
        assert wraps >= 4, f"the T sequence crossed the delta-buffer wrap only {wraps} times"
        assert ours == hand, f"ring cursors differ:\n hand {hand}\n ours {ours}"
    finally:
        cd.smk_engine_destroy(h)


if __name__ == "__main__":
    run(globals())
