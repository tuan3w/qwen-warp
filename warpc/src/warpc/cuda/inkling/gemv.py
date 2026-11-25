"""Inkling-2 kernel family `gemv`: capabilities, launchers and ctypes signatures.

g_gemv_multi    up to 8 `linear` anchors of one visit (wq, wr, wk, k taps, wv, v taps, sink, gate) on up to 3 distinct
                inputs: one launch, each segment reading its own input and writing its own bf16 output
g_gemv_convres  the attention output projection with the short-convolution epilogue, f32 result (a block's first layer)
g_gemv_convres_add  the same plus the residual stream: out = f32(h) + f32(conv result)
g_head_softcap  the unembedding: bf16-rounded logits, soft cap, the vocabulary split and the argmax

The first three run kernels/inkling_gemv.cuh (k_gemv's arithmetic: a segment is bit-identical to `warpc_gemv` mode 0);
the head is `warpc_gemv` mode 2 with its `cap` and `rnd` arguments, the dense `g_head`'s kernel (a separate capability
because the dense head's static `tolerance` (the unrounded accumulator) is not true of this one). The weights are the
dense packs the dense GEMVs read (`dense`, `dense_cat`).
"""

from __future__ import annotations

import ctypes as C

from ...compiler import Absorb, Cap, Pack, Region, Verdict
from .. import schedule

_PTR, _INT, _FLOAT = C.c_void_p, C.c_int, C.c_float
SEGS = 8  # IG_SEGS of inkling_gemv.cuh
SEGMENT_INPUTS = 3
CONV_WIDTH = 4  # IG_TAPS


def _check(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


def _shape(A, i: int):
    return A.graph.weights[A.node(i).attrs["weight"]].shape


def _x_type(A, i: int):
    return A.graph.types[A.node(i).ins[0]]


# ------------------------------------------------------------------------------------------------ g_gemv_multi


def _multi_consts(m, A):
    shapes = [_shape(A, i) for i in m.anchors]
    return {
        "N": sum(s[0] for s in shapes),
        "K": shapes[0][1],
        "T": A.rows_max,
        "nin": len({A.node(i).ins[0] for i in m.anchors}),
        "dtype": _x_type(A, m.anchors[0]).dtype,
    }


def _multi_info(c: dict) -> Verdict:
    return Verdict(ctas=schedule.gemv_ctas(c["N"], c["K"], 0))


def _multi_check(m, A):
    # the dense g_gemv serves up to 3 anchors on one input: this kernel is for what it cannot (a dense model's plans
    # stay as they were)
    if len(m.anchors) <= 3 and m.consts["nin"] == 1:
        return "g_gemv serves up to 3 anchors on one input"
    k = _shape(A, m.anchors[0])[1]
    for i in m.anchors:
        if _shape(A, i)[1] != k:
            return "the segments' K differ"
        t = _x_type(A, i)
        if t.dtype != "bf16" or t.shape[-1] != k or len(t.shape) != 2:
            return "a segment's input is not a bf16 [T, K] activation"
    return None


def _emit_multi(m, A):
    lins = [A.node(i) for i in m.anchors]
    ws = [n.attrs["weight"] for n in lins]
    ns = [A.graph.weights[w].shape[0] for w in ws]
    xs = list(dict.fromkeys(n.ins[0] for n in lins))
    name = "|".join(ws)
    return Region(
        "g_gemv_multi",
        set(m.covers),
        ins={f"x{j}": x for j, x in enumerate(xs)},
        outs={f"o{j}": n.outs[0] for j, n in enumerate(lins)},
        weights={"w": name},
        packs=[Pack(name, "dense_cat" if len(ws) > 1 else "dense", ws)],
        attrs={
            "N": sum(ns),
            "K": A.graph.weights[ws[0]].shape[1],
            "segs": ns,
            "xsel": [xs.index(n.ins[0]) for n in lins],
        },
        numerics="bf16 weights, fp32 accumulation, 32-lane butterfly sum (order depends on K only), one bf16 rounding",
    )


# Rows per warp by step rows T (index T - 1), the best of 1 / 2 / 4 measured on an RTX 3090 at the real shapes
# (N = 8,600 / 8,664 and K = 2,304; N = 2,304 and K = 3,072; weights rotated through DRAM, 24 launches per CUDA graph):
# T = 1: 46.7 us at 4 against 49.6 at the dense rule's 2; T = 8 at 4: 54.9 us against 57.4 at 2; the o_proj at T = 8:
# 24.7 us at 2 against 27.0 at 4.
MULTI_RPW = (4, 4, 4, 2, 2, 4, 4, 4)
CONVRES_RPW = (1, 1, 1, 2, 4, 4, 2, 2)


def _multi(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    segs, n = a["segs"], len(a["segs"])
    ends, xs, outs = (_INT * SEGS)(), (_PTR * SEGS)(), (_PTR * SEGS)()
    acc = 0
    for j in range(n):
        acc += segs[j]
        ends[j] = acc
        xs[j] = mem.buffer(t.ins[f"x{a['xsel'][j]}"])
        outs[j] = mem.buffer(t.outs[f"o{j}"])
    rpw = MULTI_RPW[T - 1]
    while rpw > 1 and any(s % rpw for s in segs[:-1]):
        rpw //= 2
    rc = L.warpc_ink_gemv(mem.packs[t.weights["w"]], a["N"], a["K"], T, n, ends, xs, outs, rpw, st)
    _check(L, rc, t.kernel)


# ------------------------------------------------------------------------------------------------ g_gemv_convres


def _convres_consts(m, A):
    lin = A.node(m.anchors[0])
    conv = A.node(m.post[0])
    cw = A.graph.weights[conv.attrs["weight"]].shape
    return {
        "N": A.graph.weights[lin.attrs["weight"]].shape[0],
        "K": A.graph.weights[lin.attrs["weight"]].shape[1],
        "T": A.rows_max,
        "C": cw[0],
        "W": cw[1],
        "dtype": _x_type(A, m.anchors[0]).dtype,
    }


def _convres_info(c: dict) -> Verdict:
    return Verdict(ctas=schedule.gemv_ctas(c["N"], c["K"], 0))


def _convres_check(m, A):
    if m.consts["C"] != m.consts["N"]:
        return "the convolution's channels are not the projection's rows"
    return None


def _convres_add_check(m, A):
    if _convres_check(m, A):
        return _convres_check(m, A)
    # the residual operand must be the cast of the stream h folded into the prologue (the kernel reads h as bf16)
    cast, add = A.node(m.post[1]), A.node(m.post[2])
    rest = [v for v in add.ins if v != cast.outs[0]]
    if len(rest) != 1 or A.prod.get(rest[0]) not in m.pre:
        return "the add's other operand is not the cast of a stream the kernel reads"
    return None


def _emit_convres(m, A):
    lin, conv = A.node(m.anchors[0]), A.node(m.post[0])
    add = A.node(m.post[2]) if len(m.post) > 2 else None
    ins = {"x": lin.ins[0]}
    if add is not None:
        ins["h"] = A.node(m.pre[0]).ins[0]
    w, cw = lin.attrs["weight"], conv.attrs["weight"]
    return Region(
        "g_gemv_convres" if add is None else "g_gemv_convres_add",
        set(m.covers),
        ins=ins,
        outs={"out": A.node(m.post[-1]).outs[0]},
        weights={"w": w, "cw": cw},
        states={"conv": conv.attrs["state"]},
        packs=[Pack(w, "dense", [w]), Pack(cw, "dense", [cw])],
        attrs={
            "N": A.graph.weights[w].shape[0],
            "K": A.graph.weights[w].shape[1],
            "divisor": conv.attrs["divisor"],
            "residual": add is not None,
        },
        numerics="GEMV as g_gemv_multi, bf16; the short convolution in fp32 over the state row (tap order), "
        "(conv + y) / divisor rounded once to bf16, f32 widening, residual add in f32",
    )


def _convres(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    rc = L.warpc_ink_gemv_convres(
        mem.packs[t.weights["w"]],
        mem.buffer(t.ins["x"]),
        a["N"],
        a["K"],
        T,
        mem.packs[t.weights["cw"]],
        mem.states[t.states["conv"]],
        mem.buffer(t.ins["h"]) if a["residual"] else None,
        mem.buffer(t.outs["out"]),
        C.c_float(a["divisor"]),
        CONVRES_RPW[T - 1],
        st,
    )
    _check(L, rc, t.kernel)


# ------------------------------------------------------------------------------------------------ g_head_softcap


def _head_consts(m, A):
    lin, split = A.node(m.anchors[0]), A.node(m.post[2])
    w = A.graph.weights[lin.attrs["weight"]]
    return {
        "N": split.attrs["sizes"][0],
        "Nw": w.shape[0],
        "K": w.shape[1],
        "T": A.rows_max,
        "dtype": _x_type(A, m.anchors[0]).dtype,
    }


def _head_info(c: dict) -> Verdict:
    return Verdict(ctas=schedule.gemv_ctas(c["N"], c["K"], 2))


def _head_check(m, A):
    split = A.node(m.post[2])
    sizes = split.attrs["sizes"]
    if len(sizes) != 2 or sizes[0] > m.consts["Nw"] or sum(sizes) != m.consts["Nw"]:
        return "the split is not (logits, padding) of the projection's rows"
    pad = split.outs[1]
    if pad in A.outputs or A.consumers(pad):
        return "the padding rows are read"
    return None


def _emit_head(m, A):
    lin, cap, split, amax = A.node(m.anchors[0]), A.node(m.post[1]), A.node(m.post[2]), A.node(m.post[3])
    w = A.graph.weights[lin.attrs["weight"]]
    return Region(
        "g_head_softcap",
        set(m.covers),
        ins={"x": lin.ins[0]},
        outs={"logits": split.outs[0], "ids": amax.outs[0]},
        weights={"w": w.name},
        packs=[Pack(w.name, "dense", [w.name])],
        attrs={"N": split.attrs["sizes"][0], "Nw": w.shape[0], "K": w.shape[1], "cap": cap.attrs["cap"]},
        numerics="fp32 accumulator rounded to bf16 (the interpreter's linear), cap * tanh(. / cap) in fp32; only the "
        "first N rows of the weight are read; argmax lowest index on ties",
    )


def _head(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    outs = (C.c_void_p * 3)(logits, 0, 0)
    rc = L.warpc_gemv(
        mem.packs[t.weights["w"]],
        None,
        mem.buffer(t.ins["x"]),
        a["N"],
        a["K"],
        T,
        2,
        None,
        outs,
        None,
        C.c_float(0.0),
        schedule.gemv_rows_per_warp(a["N"], a["K"], 2),
        None,
        0,
        C.c_float(a["cap"]),
        1,
        1,
        st,
    )
    _check(L, rc, t.kernel)
    ws = mem.scratch("argmax_ws", L.warpc_argmax_ws_bytes())
    _check(L, L.warpc_argmax(logits, T, a["N"], ids, ws, st), t.kernel)


# ------------------------------------------------------------------------------------------------ table entries

_ROWS = schedule.STEP_ROWS
_LIMITS = {"K_multiple": 256, "T_max": _ROWS, "dtype_in": ("bf16",)}

_CONV = Absorb("causal_conv1d", min=1, max=1, attrs={"residual": True})
_CAST = Absorb("cast", min=1, max=1, attrs={"dtype": "f32"})
_CONV_LIMITS = {**_LIMITS, "W_in": (CONV_WIDTH,)}
_CONV_ROUNDS = {"linear": ("out",), "causal_conv1d": ("out",)}

CAPS: list = [
    Cap(
        "g_gemv_convres_add",
        "linear",
        anchor_schemes=("dense",),
        pre=(Absorb("cast", min=1, max=1, attrs={"dtype": "f32"}),),
        post=(_CONV, _CAST, Absorb("add", min=1, max=1, exposes=True)),
        limits=_CONV_LIMITS,
        rounds=_CONV_ROUNDS,
        constants=_convres_consts,
        info=_convres_info,
        check=_convres_add_check,
        emit=_emit_convres,
    ),
    Cap(
        "g_gemv_convres",
        "linear",
        anchor_schemes=("dense",),
        post=(_CONV, _CAST),
        limits=_CONV_LIMITS,
        rounds=_CONV_ROUNDS,
        constants=_convres_consts,
        info=_convres_info,
        check=_convres_check,
        emit=_emit_convres,
    ),
    Cap(
        "g_head_softcap",
        "linear",
        anchor_schemes=("dense",),
        post=(
            Absorb("cast", min=1, max=1, attrs={"dtype": "f32"}),
            Absorb("softcap", min=1, max=1),
            Absorb("split", min=1, max=1, exposes=True),
            Absorb("argmax", min=1, max=1, exposes=True, reduction_ok=True),
        ),
        limits=_LIMITS,
        rounds={"linear": ("out",)},
        launches=3,
        constants=_head_consts,
        info=_head_info,
        check=_head_check,
        emit=_emit_head,
    ),
    Cap(
        "g_gemv_multi",
        "linear",
        anchor_schemes=("dense",),
        segments=SEGS,
        segment_inputs=SEGMENT_INPUTS,
        operand_cap=2 * SEGS + SEGMENT_INPUTS + SEGS,
        limits={**_LIMITS, "nin_max": SEGMENT_INPUTS},
        rounds={"linear": ("out",)},
        constants=_multi_consts,
        info=_multi_info,
        check=_multi_check,
        emit=_emit_multi,
    ),
]

LAUNCHERS: dict = {
    "g_gemv_multi": _multi,
    "g_gemv_convres": _convres,
    "g_gemv_convres_add": _convres,
    "g_head_softcap": _head,
}

SIGNATURES: dict = {
    "warpc_ink_gemv": ([_PTR, _INT, _INT, _INT, _INT, _PTR, _PTR, _PTR, _INT, _PTR], _INT),
    "warpc_ink_gemv_convres": ([_PTR, _PTR, _INT, _INT, _INT, _PTR, _PTR, _PTR, _PTR, _FLOAT, _INT, _PTR], _INT),
}
