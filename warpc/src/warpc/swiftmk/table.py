"""The swiftmk capability table: the exported kernels of the hand engine's library (engine/include/*.h) as fusion
capabilities, each with the limits its C entry hard-codes.

The hand kernels are written for one model (Qwen3.5 27B: hidden 5120, 24 / 4 attention heads of 256, 48 DeltaNet value
heads of 128, partial rope over 64 dims at theta 1e7), so most limits pin a shape or a constant: another model fails at
lowering with a LoweringError naming the limit. Every attr an op carries is a limit here, so a graph the kernels would
compute differently (a norm without the +1 offset, another eps, another rope table) is refused rather than miscomputed.
"""

from __future__ import annotations

import itertools
import math

import torch

from ..compiler import Absorb, Cap, CapsTable, Pack, Region, Target
from ..cuda.table import RTX3090
from .lib import KV_DTYPE

TARGET: Target = RTX3090
ROWS_MAX = 16  # SMK_RMAX: the GEMV / norm / head kernels take up to 16 rows
MIXER_ROWS_MAX = 8  # swiftmk_attn_step and swiftmk_deltanet_step_lazy2 (one sequence): T <= 8
DN_SLOTS_MAX = 17  # SWIFTMK_DN_MAX_SLOTS
HEAD_R = 16  # engine_impl.h HEAD_R_GREEDY: candidates the int4 head rescores exactly
ROPE_THETA, ROPE_DIM = 1e7, 64  # attn.cu: angles in fp64 from theta, not from a table


def _rope_table() -> tuple[float, ...]:
    """The fp32 inverse frequencies a graph built for theta 1e7 over 64 rotary dims carries (HF's formula)."""
    j = torch.arange(0, ROPE_DIM, 2, dtype=torch.int64).float()
    return tuple((1.0 / (ROPE_THETA ** (j / ROPE_DIM))).tolist())


ROPE_TABLE = _rope_table()

NUM_W4 = "int4 g128 weights in the repacked tile layout, mma m16n8k16, fp32 accumulation, one bf16 rounding"
NUM_NORM = "zero-centred rmsnorm: fp32 statistics, x̂ (1 + w), one bf16 rounding; residual h = bf16(h + y) in place"


def _w(A, i: int, key: str = "weight"):
    return A.graph.weights[A.node(i).attrs[key]]


def _ttype(A, v: str):
    return A.graph.types[v]


def _not_head(m, A):
    for i in m.anchors:
        if A.consumers(A.node(i).outs[0], "cast"):
            return "a linear feeding a cast is the lm_head: the head kernel's"
    return None


# ------------------------------------------------------------------------------------------------ embed / norms


def _embed_consts(m, A):
    return {"H": _w(A, m.anchors[0]).shape[1]}


def _emit_embed(m, A):
    n = A.node(m.anchors[0])
    w = n.attrs["weight"]
    return Region(
        "s_embed",
        set(m.covers),
        ins={"tok": "@tokens"},
        outs={"x": n.outs[0]},
        weights={"table": w},
        attrs={"H": A.graph.weights[w].shape[1]},
        packs=[Pack(w, "dense", [w])],
        numerics="row gather (smk_embed)",
    )


def _norm_consts(m, A):
    n = A.node(m.anchors[0])
    x = _ttype(A, n.ins[0])
    return {
        "H": x.shape[-1],
        "T": A.rows_max,
        "rank": len(x.shape),
        "dtype": x.dtype,
        "offset": n.attrs.get("offset", 0.0),
    }


def _emit_rmsnorm(m, A):
    n = A.node(m.anchors[0])
    w = n.attrs["weight"]
    attrs = {"H": _ttype(A, n.ins[0]).shape[-1], "eps": n.attrs["eps"]}
    if not m.pre:
        return Region(
            "s_rmsnorm",
            set(m.covers),
            ins={"h": n.ins[0]},
            outs={"xn": n.outs[0]},
            weights={"w": w},
            packs=[Pack(w, "dense", [w])],
            attrs=attrs,
            numerics=NUM_NORM.split(";")[0],
        )
    an = A.node(m.inplace_node)
    y = an.ins[1] if an.ins[0] == m.h else an.ins[0]
    return Region(
        "s_rmsnorm_add",
        set(m.covers),
        ins={"h": m.h, "y": y},
        outs={"h_out": an.outs[0], "xn": n.outs[0]},
        weights={"w": w},
        packs=[Pack(w, "dense", [w])],
        attrs=attrs,
        inplace=("h_out", "h"),
        numerics=NUM_NORM,
    )


# ------------------------------------------------------------------------------------------------ GEMVs


def _w4_consts(m, A):
    ns = [_w(A, i).shape[0] for i in m.anchors]
    return {
        "N": sum(ns),
        "K": _w(A, m.anchors[0]).shape[1],
        "T": A.rows_max,
        "ends": math.gcd(*itertools.accumulate(ns)),  # every segment end a multiple of 16 (engine.h SMK_EPI_SEGS)
        "dtype": _ttype(A, A.node(m.anchors[0]).ins[0]).dtype,
    }


def _emit_w4(m, A):
    lins = [A.node(i) for i in m.anchors]
    ws = [n.attrs["weight"] for n in lins]
    ns = [A.graph.weights[w].shape[0] for w in ws]
    name = "|".join(ws)
    return Region(
        "s_w4_gemv",
        set(m.covers),
        ins={"x": lins[0].ins[0]},
        outs={f"o{j}": n.outs[0] for j, n in enumerate(lins)},
        weights={"w": name},
        packs=[Pack(name, "w4r_cat" if len(ws) > 1 else "w4r", ws)],
        attrs={"N": sum(ns), "K": A.graph.weights[ws[0]].shape[1], "ends": list(itertools.accumulate(ns))},
        numerics=NUM_W4 + "; output segments by row range",
    )


def _silu_consts(m, A):
    g = _w(A, m.anchors[0])
    return {"N": g.shape[0], "K": g.shape[1], "T": A.rows_max, "dtype": _ttype(A, A.node(m.anchors[0]).ins[0]).dtype}


def _silu_check(m, A):
    if _w(A, m.anchors[0]).shape != _w(A, m.anchors[1]).shape:
        return "gate and up differ in shape"
    return None


def _emit_silu(m, A):
    lg, lu = A.node(m.anchors[0]), A.node(m.anchors[1])
    wg, wu = lg.attrs["weight"], lu.attrs["weight"]
    n, k = A.graph.weights[wg].shape
    name = f"{wg}|{wu}"
    return Region(
        "s_w4_gemv_silu",
        set(m.covers),
        ins={"x": lg.ins[0]},
        outs={"out": A.node(m.pair).outs[0]},
        weights={"w": name},
        packs=[Pack(name, "w4r_gateup", [wg, wu])],
        attrs={"N": 2 * n, "K": k},
        numerics=NUM_W4 + "; gate|up interleaved per 8 rows, bf16(bf16(silu(bf16(gate))) * bf16(up))",
    )


def _ab_consts(m, A):
    return {
        "segs": len(m.anchors),
        "K": _w(A, m.anchors[0]).shape[1],
        "T": A.rows_max,
        "dtype": _ttype(A, A.node(m.anchors[0]).ins[0]).dtype,
    }


def _emit_ab(m, A):
    la, lb = A.node(m.anchors[0]), A.node(m.anchors[1])
    wa, wb = la.attrs["weight"], lb.attrs["weight"]
    na, nb = A.graph.weights[wa].shape[0], A.graph.weights[wb].shape[0]
    name = f"{wa}|{wb}"
    return Region(
        "s_ab_gemv",
        set(m.covers),
        ins={"x": la.ins[0]},
        outs={"o0": la.outs[0], "o1": lb.outs[0]},
        weights={"w": name},
        packs=[Pack(name, "dense_cat", [wa, wb])],
        attrs={"N": na + nb, "na": na, "K": A.graph.weights[wa].shape[1], "side": True},
        numerics="bf16 weights, one warp per row, fp32 accumulation, one bf16 rounding",
    )


def _head_consts(m, A):
    w = _w(A, m.anchors[0])
    return {"N": w.shape[0], "K": w.shape[1], "T": A.rows_max, "dtype": _ttype(A, A.node(m.anchors[0]).ins[0]).dtype}


def _emit_head(kernel: str):
    def emit(m, A):
        ln, cast, amax = A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[1])
        w = A.graph.weights[ln.attrs["weight"]]
        weights, packs = {"w": w.name}, [Pack(w.name, "dense", [w.name])]
        numerics = "fp32 logits = the fp32 accumulator of the bf16 head, argmax lowest index on ties"
        if kernel == "s_head_refine":
            weights["wq"] = f"{w.name}.i4r"
            packs.append(Pack(f"{w.name}.i4r", "head_i4r", [w.name]))
            numerics = (
                f"approximate fp32 logits of an int4 g128 copy of the head, the top {HEAD_R} per row rescored"
                " exactly with the bf16 head; argmax over the rescored candidates"
            )
        return Region(
            kernel,
            set(m.covers),
            ins={"x": ln.ins[0]},
            outs={"logits": cast.outs[0], "ids": amax.outs[0]},
            weights=weights,
            packs=packs,
            attrs={"N": w.shape[0], "K": w.shape[1]},
            numerics=numerics,
        )

    return emit


def _emit_w4_head(m, A):
    ln, cast, amax = A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[1])
    w = A.graph.weights[ln.attrs["weight"]]
    return Region(
        "s_w4_head",
        set(m.covers),
        ins={"x": ln.ins[0]},
        outs={"logits": cast.outs[0], "ids": amax.outs[0]},
        weights={"w": w.name},
        packs=[Pack(w.name, "w4r", [w.name])],
        attrs={"N": w.shape[0], "K": w.shape[1]},
        numerics=NUM_W4 + "; fp32 logits = the fp32 accumulator, argmax lowest index on ties",
    )


# ------------------------------------------------------------------------------------------------ DeltaNet


def _dn_nodes(m, A) -> dict:
    """The covered nodes by role: conv, silu, split, gdr (the anchor), gnorm, out (the final reshape)."""
    by = {A.node(i).op: i for i in m.covers if A.node(i).op != "reshape"}
    return {
        "conv": by["causal_conv1d"],
        "silu": by["silu"],
        "split": by["split"],
        "gdr": m.anchors[0],
        "gnorm": m.post[0],
        "out": m.post[1],
    }


def _dn_consts(m, A):
    r = _dn_nodes(m, A)
    gdr, gn = A.node(r["gdr"]), A.node(r["gnorm"])
    cw = _w(A, r["conv"])
    q, v = _ttype(A, gdr.ins[0]), _ttype(A, gdr.ins[2])
    return {
        "C": cw.shape[0],
        "W": cw.shape[1],
        "Hk": q.shape[1],
        "dk": q.shape[2],
        "Hv": v.shape[1],
        "dv": v.shape[2],
        "split": tuple(A.node(r["split"]).attrs["sizes"]),
        "eps": gdr.attrs["eps"],
        "norm_eps": gn.attrs["eps"],
        "scale": gdr.attrs["scale"],
        "T": A.rows_max,
        "slots": A.rows_max + 1,
        "dtype": _ttype(A, A.node(r["conv"]).ins[0]).dtype,
    }


def _dn_check(m, A):
    """The kernel reads qkv as the conv input and q | k | v as consecutive column ranges of it, z as the gated norm's
    gate before its head reshape, and writes the flat [T, Hv*dv] output."""
    r = _dn_nodes(m, A)
    sp, gdr, gn = A.node(r["split"]), A.node(r["gdr"]), A.node(r["gnorm"])
    if A.node(r["silu"]).ins[0] != A.node(r["conv"]).outs[0] or sp.ins[0] != A.node(r["silu"]).outs[0]:
        return "the split must read silu(causal_conv1d(x))"
    for j, v in enumerate(gdr.ins[:3]):
        p = A.prod.get(v)
        if p not in m.covers or A.node(p).op != "reshape" or A.node(p).ins[0] != sp.outs[j]:
            return "q, k, v must be the head reshapes of the split's outputs, in order"
    if gn.ins[0] != gdr.outs[0]:
        return "the gated norm must normalise the delta rule's output"
    zp = A.prod.get(gn.ins[1])
    if zp not in m.covers or A.node(zp).op != "reshape" or A.prod.get(A.node(zp).ins[0]) in m.covers:
        return "the gated norm's gate must be a head reshape of an outside value"
    _, hv, dv = _ttype(A, gdr.ins[2]).shape
    if _ttype(A, A.node(r["out"]).outs[0]).shape[1:] != (hv * dv,):
        return "the output must be flattened to [T, Hv * dv]"
    return None


def _emit_dn(m, A):
    r = _dn_nodes(m, A)
    conv, gdr, gn = A.node(r["conv"]), A.node(r["gdr"]), A.node(r["gnorm"])
    weights = {
        "conv_w": conv.attrs["weight"],
        "A_log": gdr.attrs["a_log"],
        "dt_bias": gdr.attrs["dt_bias"],
        "norm_w": gn.attrs["weight"],
    }
    return Region(
        "s_deltanet",
        set(m.covers),
        ins={"qkv": conv.ins[0], "z": A.node(A.prod[gn.ins[1]]).ins[0], "a": gdr.ins[3], "b": gdr.ins[4]},
        outs={"y": A.node(r["out"]).outs[0]},
        weights=weights,
        states={"conv": conv.attrs["state"], "rec": gdr.attrs["state"]},
        packs=[Pack(w, "dense", [w]) for w in weights.values()],
        attrs={"Hv": m.consts["Hv"], "dv": m.consts["dv"]},
        numerics=(
            "one launch: conv + silu in fp32 (no rounding between), l2-norm, gates, the delta rule on an fp32 state"
            " stored [Hv][dv][dk], gated rmsnorm; row t reads rows < t; lazy ring snapshots (deltanet.h)"
        ),
    )


# ------------------------------------------------------------------------------------------------ attention


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
    norms = [A.node(i) for i in (r["qn"], r["kn"]) if i in m.covers and A.node(i).op == "rmsnorm"]
    sp = next(A.node(i) for i in m.covers if A.node(i).op == "split")
    return {
        "Hq": _ttype(A, att.ins[0]).shape[1],
        "Hkv": hkv,
        "hd": hd,
        "rd": rope.attrs["rotary_dim"],
        "rope": int(tuple(rope.attrs["inv_freq"]) == ROPE_TABLE),
        "scale": att.attrs["scale"],
        "norms": len(norms),
        "eps": norms[0].attrs["eps"] if norms else 0.0,
        "keps": norms[-1].attrs["eps"] if norms else 0.0,
        "offset": norms[0].attrs.get("offset", 0.0) if norms else 0.0,
        "koffset": norms[-1].attrs.get("offset", 0.0) if norms else 0.0,
        "split": tuple(sp.attrs["sizes"]),
        "cache": st.cache,
        "ctx": ctx,
        "T": A.rows_max,
        "dtype": _ttype(A, att.ins[0]).dtype,
    }


def _attn_check(m, A):
    """The kernel reads the q projection as [T][Hq][q 256 | gate 256], k and v projections as [T][Hkv * 256], applies
    the per-head norms then the rope to q and k, and multiplies the output by sigmoid(gate) of the same projection."""
    r = _attn_nodes(m, A)
    att, rope = A.node(r["att"]), A.node(r["rope"])
    if A.prod.get(att.ins[1]) != r["rope"] or rope.outs != [att.ins[0], att.ins[1]]:
        return "q and k must come from one rope node"
    sp = next(i for i in m.covers if A.node(i).op == "split")
    spn = A.node(sp)
    qn, kn = A.node(r["qn"]), A.node(r["kn"])
    if qn.ins[0] != spn.outs[0]:
        return "q must be the norm of the first half of the split q projection"
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
        return "the output gate must be sigmoid(the second half of the split q projection)"
    return None


def _emit_attn(m, A):
    g = A.graph
    r = _attn_nodes(m, A)
    att, qn, kn = A.node(r["att"]), A.node(r["qn"]), A.node(r["kn"])
    sp = next(A.node(i) for i in m.covers if A.node(i).op == "split")
    weights = {"q_norm": qn.attrs["weight"], "k_norm": kn.attrs["weight"]}
    packs = [Pack(w, "dense", [w]) for w in weights.values()]
    cache = g.states[att.attrs["state"]].cache
    if cache == "fp8":
        for key in ("k_scale", "v_scale"):
            weights[key] = att.attrs[key]
            packs.append(Pack(att.attrs[key], "dense", [att.attrs[key]]))
    numerics = (
        "prep (qk-norm, rope with fp64 angles, KV write) + split flash decode (mma, fp32 softmax) + reduce with"
        f" the sigmoid gate, one bf16 rounding; KV cache {cache}"
    )
    return Region(
        "s_attn",
        set(m.covers),
        ins={
            "q": A.node(A.prod[sp.ins[0]]).ins[0],
            "k": A.node(A.prod[kn.ins[0]]).ins[0],
            "v": A.node(A.prod[att.ins[2]]).ins[0],
        },
        outs={"y": A.node(r["gate"]).outs[0]},
        weights=weights,
        states={"kv": att.attrs["state"]},
        attrs={"ctx": m.consts["ctx"], "cache": cache},
        packs=packs,
        numerics=numerics,
    )


def _unfused_ctas(n, A) -> int:
    """CTAs priced for a node alone: GEMVs by 16-row tiles, everything else as one wave's worth."""
    if n.op == "linear":
        return max(1, A.graph.weights[n.attrs["weight"]].shape[0] // 128)
    return TARGET.sms


_RESADD = Absorb("add", min=1, max=1, exposes=True, inplace=True)
_NORM_LIMITS = {
    "offset_in": (1.0,),
    "H_multiple": 8,
    "H_max": 8192,
    "rank_in": (2,),
    "dtype_in": ("bf16",),
    "T_max": ROWS_MAX,
}
_W4_LIMITS = {"N_multiple": 16, "K_multiple": 128, "T_max": ROWS_MAX, "dtype_in": ("bf16",)}
_HEAD_POST = (
    Absorb("cast", min=1, max=1, exposes=True, attrs={"dtype": "f32"}),
    Absorb("argmax", min=1, max=1, exposes=True, reduction_ok=True),
)

# order = the cost model's tie-break
CAPS = [
    Cap(
        "s_attn",
        "attention",
        pre=(
            Absorb("reshape", min=4, max=4),
            Absorb("split", min=1, max=1),
            Absorb("rmsnorm", min=2, max=2),
            Absorb("rope", min=1, max=1),
        ),
        post=(Absorb("sigmoid_mul", min=1, max=1),),
        limits={
            "Hq_in": (24,),
            "Hkv_in": (4,),
            "hd_in": (256,),
            "rd_in": (ROPE_DIM,),
            "rope_in": (1,),
            "scale_in": (1 / 16,),
            "norms_in": (2,),
            "eps_in": (1e-6,),
            "keps_in": (1e-6,),
            "offset_in": (1.0,),
            "koffset_in": (1.0,),
            "split_in": ((256, 256),),
            "cache_in": tuple(KV_DTYPE),
            "T_max": MIXER_ROWS_MAX,
            "dtype_in": ("bf16",),
        },
        rounds={"rmsnorm": ("out",), "rope": ("out",), "sigmoid_mul": ("out",)},
        tolerance=("rope.trig", "rope.prod", "attention.p", "attention.out", "sigmoid_mul.sigmoid"),
        launches=3,
        constants=_attn_consts,
        check=_attn_check,
        emit=_emit_attn,
    ),
    Cap(
        "s_deltanet",
        "gated_delta_rule",
        pre=(
            Absorb("reshape", min=4, max=4),
            Absorb("split", min=1, max=1),
            Absorb("silu", min=1, max=1),
            Absorb("causal_conv1d", min=1, max=1),
        ),
        post=(Absorb("gated_rmsnorm", min=1, max=1, reduction_ok=True), Absorb("reshape", min=1, max=1)),
        limits={
            "C_in": (10240,),
            "W_in": (4,),
            "Hk_in": (16,),
            "Hv_in": (48,),
            "dk_in": (128,),
            "dv_in": (128,),
            "split_in": ((2048, 2048, 6144),),
            "eps_in": (1e-6,),
            "norm_eps_in": (1e-6,),
            "scale_in": (128**-0.5,),
            "T_max": MIXER_ROWS_MAX,
            "slots_max": DN_SLOTS_MAX,
            "dtype_in": ("bf16",),
        },
        rounds={"gated_delta_rule": ("out",), "gated_rmsnorm": ("out",)},
        tolerance=(
            "causal_conv1d.out",
            "silu.out",
            "gated_delta_rule.beta",
            "gated_rmsnorm.norm",
            "gated_rmsnorm.scale",
        ),
        constants=_dn_consts,
        check=_dn_check,
        emit=_emit_dn,
    ),
    Cap(
        "s_w4_gemv_silu",
        "linear",
        anchor_schemes=("w4g128",),
        pair=("silu_mul",),
        limits={"N_multiple": 8, "K_multiple": 128, "T_max": ROWS_MAX, "dtype_in": ("bf16",)},
        rounds={"linear": ("out",), "silu_mul": ("silu", "out")},
        constants=_silu_consts,
        check=_silu_check,
        emit=_emit_silu,
    ),
    Cap(
        "s_w4_gemv",
        "linear",
        anchor_schemes=("w4g128",),
        segments=3,
        limits={**_W4_LIMITS, "ends_multiple": 16},
        rounds={"linear": ("out",)},
        constants=_w4_consts,
        check=_not_head,
        emit=_emit_w4,
    ),
    Cap(
        "s_w4_head",
        "linear",
        anchor_schemes=("w4g128",),
        post=_HEAD_POST,
        limits=_W4_LIMITS,
        rounds={"cast": ("out",)},
        tolerance=("linear.out",),
        launches=2,
        constants=_head_consts,
        emit=_emit_w4_head,
    ),
    Cap(
        "s_head",
        "linear",
        anchor_schemes=("dense",),
        post=_HEAD_POST,
        limits={"K_multiple": 256, "N_multiple": 4, "T_max": ROWS_MAX, "dtype_in": ("bf16",)},
        rounds={"cast": ("out",)},
        tolerance=("linear.out",),
        launches=2,
        constants=_head_consts,
        emit=_emit_head("s_head"),
    ),
    Cap(
        "s_head_refine",
        "linear",
        anchor_schemes=("dense",),
        post=_HEAD_POST,
        limits={"K_in": (5120,), "N_multiple": 4, "N_min": 128 * HEAD_R, "T_max": ROWS_MAX, "dtype_in": ("bf16",)},
        rounds={"cast": ("out",)},
        tolerance=("linear.out", "argmax.out"),
        launches=2,
        constants=_head_consts,
        gather=("w",),  # smk_headq_select_rescore reads only the candidate rows; the int4 copy wq is scanned
        emit=_emit_head("s_head_refine"),
    ),
    Cap(
        "s_ab_gemv",
        "linear",
        anchor_schemes=("dense",),
        segments=2,
        limits={"segs_in": (2,), "K_multiple": 8, "T_max": ROWS_MAX, "dtype_in": ("bf16",)},
        rounds={"linear": ("out",)},
        constants=_ab_consts,
        check=_not_head,
        emit=_emit_ab,
    ),
    Cap(
        "s_rmsnorm_add",
        "rmsnorm",
        pre=(_RESADD,),
        limits=_NORM_LIMITS,
        rounds={"add": ("out",), "rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm,
    ),
    Cap(
        "s_rmsnorm",
        "rmsnorm",
        limits=_NORM_LIMITS,
        rounds={"rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm,
    ),
    Cap(
        "s_embed",
        "embed",
        limits={"H_multiple": 8},
        rounds={"embed": ("out",)},
        constants=_embed_consts,
        gather=("table",),
        emit=_emit_embed,
    ),
]

HEADS = {"bf16": "s_head", "int4_refine": "s_head_refine"}


def table(head: str) -> CapsTable:
    """The table with the head kernel `head` names (HEADS) and without the other one."""
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r} (known: {', '.join(HEADS)})")
    drop = {k for h, k in HEADS.items() if h != head}
    return CapsTable([c for c in CAPS if c.kernel not in drop], TARGET, _unfused_ctas, {"reshape": 0, "split": 0})
