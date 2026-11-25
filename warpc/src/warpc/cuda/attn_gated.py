"""Stage 2 family 5: the gated hd-256 attention of the Qwen3.5 hybrid.

A family module exports `CAPS` (capabilities appended to the CUDA table), `LAUNCHERS` (kernel name -> `(lib, mem, task,
T, stream)`), `SIGNATURES` (ctypes signatures of its extern "C" entries, merged into glue.SIGNATURES) and `PACKS` (pack
kind -> `(packer, pack) -> device tensor`, merged into pack.Packer).

`g_attn_gated` replaces the hand engine's swiftmk_attn_step (kernels/qwen35_attn.cu): the q projection is read as
[T][Hq][query | gate], q and k get the zero-centred per-head norm and the partial rope, the output is multiplied by
sigmoid(gate). The cache is bf16 or int4_rot (the hand engine's rotated 4-bit format); the fp8 cache has no kernel here.
"""

from __future__ import annotations

import ctypes as C

from ..compiler import Absorb, Cap, Pack, Region, Verdict
from ..kvcache import FORMATS
from ..sources.config import _plain_inv_freq
from . import schedule

ROPE_THETA, ROPE_DIM, HEAD_DIM = 1e7, 64, 256
# the kernel computes the rope angle in fp64 from theta^(-l/32) (a constant table of the kernel): the plan's fp32 table
# must be the one of theta 1e7 over 64 rotary dims
ROPE_TABLE = tuple(_plain_inv_freq(ROPE_THETA, ROPE_DIM).tolist())
CACHES = ("bf16", "int4_rot")
_ROWS = schedule.STEP_ROWS
# query columns (rows x query heads per KV head) of one block: six n-tiles of 8, the split kernel's instantiations
MAX_COLS = 48
SPLIT_BLOCKS = 80  # blocks of the split launch (one wave on 82 SMs): splits per KV head = SPLIT_BLOCKS // Hkv
# keys per split below which the context is not split further: the hand engine's (the splits fix which keys share an
# online softmax, so the output is bit for bit the hand's)
MIN_SPLIT_KEYS = 128


def split_blocks(hkv: int) -> int:
    return max(1, SPLIT_BLOCKS // hkv)


NUMERICS = (
    "prep (q / k zero-centred norm and rope in fp32 with fp64 angles, one rounding into the cache: the hand engine's"
    " writers) + split-K flash decode on tensor cores (fp32 accumulation and softmax, P rounded to the cache's 16-bit"
    " type) + merge applying o / (1 + exp(-gate)) with one bf16 rounding; KV cache {cache}"
)
NUMERICS_EXACT = "; the hand engine's arithmetic and reduction orders operation for operation: bit for bit its output"
NUMERICS_I4 = (
    "; int4_rot: rotated 16-level companded codes, 8-bit log2 scale per 64 coefficients, fp16 query, first 4 positions"
    " and the 496 before the step's last row exact (fp16); scores accumulated in fp16 per scale group"
)


def _attn_nodes(m, A) -> dict:
    att = A.node(m.anchors[0])
    rope = A.prod[att.ins[0]]
    return {
        "att": m.anchors[0],
        "rope": rope,
        "qn": A.prod.get(A.node(rope).ins[0]),
        "kn": A.prod.get(A.node(rope).ins[1]),
        "gate": m.post[0],
    }


def _attn_consts(m, A):
    g = A.graph
    r = _attn_nodes(m, A)
    att, rope = A.node(r["att"]), A.node(r["rope"])
    st = g.states[att.attrs["state"]]
    _, ctx, hkv, hd = st.shape
    hq = g.types[att.ins[0]].shape[1]
    norms = [A.node(i) for i in (r["qn"], r["kn"]) if i in m.covers and A.node(i).op == "rmsnorm"]
    sp = next(A.node(i) for i in m.covers if A.node(i).op == "split")
    return {
        "Hq": hq,
        "Hkv": hkv,
        "G": hq // hkv if hkv and hq % hkv == 0 else 0,
        "hd": hd,
        "rd": rope.attrs["rotary_dim"],
        "rope": int(tuple(rope.attrs["inv_freq"]) == ROPE_TABLE),
        "norms": len(norms),
        "split": tuple(sp.attrs["sizes"]),
        "cache": st.cache,
        "ctx": ctx,
        "cols": A.rows_max * (hq // hkv if hkv and hq % hkv == 0 else 0),
        "T": A.rows_max,
        "dtype": g.types[att.ins[0]].dtype,
    }


def _attn_info(c: dict) -> Verdict:
    why = ["query heads are not a multiple of KV heads"] if c["G"] == 0 else []
    return Verdict(why, c["Hkv"] * split_blocks(c["Hkv"]))


def _attn_check(m, A):
    """The kernel reads the q projection as [T][Hq][query | gate], the k and v projections as [T][Hkv * hd], applies the
    per-head norms then the rope to q and k, and multiplies the output by sigmoid(gate) of the same projection."""
    r = _attn_nodes(m, A)
    att, rope = A.node(r["att"]), A.node(r["rope"])
    if A.prod.get(att.ins[1]) != r["rope"] or rope.outs != [att.ins[0], att.ins[1]]:
        return "q and k must come from one rope node"
    sp = next(i for i in m.covers if A.node(i).op == "split")
    spn = A.node(sp)
    qn, kn = A.node(r["qn"]), A.node(r["kn"])
    if qn.ins[0] != spn.outs[0]:
        return "q must be the norm of the first half of the split q projection"
    if qn.attrs["eps"] != kn.attrs["eps"] or qn.attrs.get("offset", 0.0) != kn.attrs.get("offset", 0.0):
        return "the q and k norms differ in eps or offset"
    qp = A.prod.get(spn.ins[0])
    if qp not in m.covers or A.node(qp).op != "reshape" or A.prod.get(A.node(qp).ins[0]) in m.covers:
        return "the split must read a head reshape of the q projection"

    def outside_reshape(v):
        p = A.prod.get(v)
        ok = p in m.covers and A.node(p).op == "reshape" and A.prod.get(A.node(p).ins[0]) not in m.covers
        return p if ok else None

    if outside_reshape(kn.ins[0]) is None or outside_reshape(att.ins[2]) is None:
        return "k and v must be head reshapes of projection outputs"
    gate = A.node(r["gate"])
    gp = A.prod.get(gate.ins[1])
    if gate.ins[0] != att.outs[0] or gp not in m.covers or A.node(gp).ins[0] != spn.outs[1]:
        return "the output gate must be the second half of the split q projection"
    return None


def _emit_attn(m, A):
    g = A.graph
    r = _attn_nodes(m, A)
    att, qn, kn = A.node(r["att"]), A.node(r["qn"]), A.node(r["kn"])
    sp = next(A.node(i) for i in m.covers if A.node(i).op == "split")
    weights = {"q_norm": qn.attrs["weight"], "k_norm": kn.attrs["weight"]}
    cache = g.states[att.attrs["state"]].cache
    c = m.consts
    return Region(
        "g_attn_gated",
        set(m.covers),
        ins={
            "q": A.node(A.prod[sp.ins[0]]).ins[0],
            "k": A.node(A.prod[kn.ins[0]]).ins[0],
            "v": A.node(A.prod[att.ins[2]]).ins[0],
        },
        outs={"y": A.node(r["gate"]).outs[0]},
        weights=weights,
        states={"kv": att.attrs["state"]},
        attrs={
            "Hq": c["Hq"],
            "Hkv": c["Hkv"],
            "hd": c["hd"],
            "ctx": c["ctx"],
            "scale": att.attrs["scale"],
            "eps": qn.attrs["eps"],
            "offset": qn.attrs.get("offset", 0.0),
            "cache": cache,
        },
        packs=[Pack(w, "dense", [w]) for w in weights.values()],
        numerics=NUMERICS.format(cache=cache) + (NUMERICS_I4 if cache == "int4_rot" else "") + NUMERICS_EXACT,
    )


# What the kernel does not reproduce of the interpreter: the bf16 rounding of the norm output, of cos / sin and of the
# rope products (fp32 through, one rounding into the cache), of P (fp32), of the attention output and of sigmoid(gate)
# (the product is rounded once)
CAPS = [
    Cap(
        "g_attn_gated",
        "attention",
        pre=(
            Absorb("reshape", min=4, max=4),
            Absorb("split", min=1, max=1),
            Absorb("rmsnorm", min=2, max=2),
            Absorb("rope", min=1, max=1),
        ),
        post=(Absorb("sigmoid_mul", min=1, max=1),),
        limits={
            "hd_in": (HEAD_DIM,),
            "rd_in": (ROPE_DIM,),
            "rope_in": (1,),
            "split_in": ((HEAD_DIM, HEAD_DIM),),
            "norms_in": (2,),
            "G_max": 8,
            "cols_max": MAX_COLS,
            "cache_in": CACHES,
            "T_max": _ROWS,
            "dtype_in": ("bf16",),
        },
        rounds={"sigmoid_mul": ("out",)},
        tolerance=(
            "rmsnorm.out",
            "rope.trig",
            "rope.prod",
            "rope.out",
            "attention.p",
            "attention.out",
            "sigmoid_mul.sigmoid",
        ),
        launches=3,
        constants=_attn_consts,
        info=_attn_info,
        check=_attn_check,
        emit=_emit_attn,
    ),
]

_I, _P, _F = C.c_int, C.c_void_p, C.c_float
SIGNATURES: dict[str, tuple[list, type]] = {
    "warpc_attn_g_prep": ([_I, *[_P] * 9, _I, _I, _I, _F, _F, _P], _I),
    "warpc_attn_g_decode": ([_I, *[_P] * 7, _I, _I, _I, _F, _I, _I, _P], _I),
}


def _attention(lib: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    hq, hkv, hd, ctx = a["Hq"], a["Hkv"], a["hd"], a["ctx"]
    q, k, v, y = mem.buffer(t.ins["q"]), mem.buffer(t.ins["k"]), mem.buffer(t.ins["v"]), mem.buffer(t.outs["y"])
    qn, kn = mem.packs[t.weights["q_norm"]], mem.packs[t.weights["k_norm"]]
    kv, pos = mem.states[t.states["kv"]], mem.pos
    fmt = FORMATS[a["cache"]]
    smax = split_blocks(hkv)
    ws = mem.scratch("attn_gated_ws", mem.rows_max * hq * smax * (hd + 4) * 4)
    qbuf = mem.scratch("attn_gated_q", mem.rows_max * hq * hd * 2)
    # read on the host during the call: need not outlive it
    meta = (C.c_longlong * 13)(*schedule.kv_meta(fmt, ctx, hkv, hd, mem.rows_max)) if fmt.quantized else None

    def check(rc: int) -> None:
        if rc != 0:
            raise RuntimeError(f"{t.kernel} failed: {lib.warpc_error_string(rc).decode()} (CUDA error {rc})")

    eps, offset = C.c_float(a["eps"]), C.c_float(a["offset"])
    check(lib.warpc_attn_g_prep(T, q, k, v, qn, kn, qbuf, kv, meta, pos, hq, hkv, ctx, eps, offset, st))
    check(
        lib.warpc_attn_g_decode(
            T, qbuf, kv, meta, pos, q, y, ws, hq, hkv, ctx, C.c_float(a["scale"]), smax, MIN_SPLIT_KEYS, st
        )
    )


LAUNCHERS = {"g_attn_gated": _attention}
PACKS: dict = {}
