"""Inkling-2 kernel family `attn`: capabilities, launchers and ctypes signatures.

g_attn_rel  one visit's whole attention block in three launches (kernels/inkling_attn.cu): prep (the four reshapes,
            three l2_clips, the log scaling of a full layer, the k / v short convolutions with their state, the
            unweighted v norm, the KV write at slot pos mod R), a split-K flash decode with the relative bias gathered
            per key distance, the per-head sink and the window, and a merge that applies the output gate:
            out = bf16(f32(att) * sigmoid(f32(gate))).

The projections stay in the GEMV family: the kernel reads their bf16 outputs (q, r, k, v, the sink logits, the gate and,
on a full layer, the two dynamic-conv tap rows) in place. The relative-bias projection is read transposed ([E][d_rel],
one 32-byte row per key distance): the pack kind `dense_t`.
"""

from __future__ import annotations

import ctypes as C

from ...compiler import Absorb, Cap, Pack, Region, Verdict
from .. import schedule

_PTR, _INT, _FLOAT, _SIZE = C.c_void_p, C.c_int, C.c_float, C.c_size_t

HD = 128  # IA_HD of kernels/inkling_attn.cu
D_REL = 16  # IA_DREL
CONV_WIDTH = 4  # IA_W
MAX_G = 8  # query heads per KV head the split kernel is instantiated for

SIGNATURES = {
    "warpc_ink_attn_prep": ([_INT, *[_PTR] * 16, *[_INT] * 5, *[_FLOAT] * 7, _PTR], _INT),
    "warpc_ink_attn_split": ([_INT, *[_PTR] * 6, *[_INT] * 7, _FLOAT, _PTR], _INT),
    "warpc_ink_attn_merge": ([_INT, *[_PTR] * 5, *[_INT] * 4, _PTR], _INT),
    "warpc_ink_attn_ws_bytes": ([_INT, _INT, _INT], _SIZE),
}


def _kv_extent(window: int | None, rows: int) -> int:
    """The most keys one row reads: its window, or the whole state of a positional cache; a ring is never read beyond
    its window."""
    return min(window, rows) if window else rows


def _nodes(m, A, op: str) -> list[int]:
    return [i for i in sorted(m.covers) if A.node(i).op == op]


def _constants(m, A):
    g = A.graph
    att = A.node(m.anchors[0])
    st = g.states[att.attrs["state"]]
    _, rows, hkv, hd = st.shape
    hq = g.types[att.ins[0]].shape[1]
    window = att.attrs.get("window")
    extent = _kv_extent(window, rows)
    nc = schedule.attn_chunks(extent)
    proj = g.weights.get(att.attrs.get("rel_proj", ""))
    return {
        "hd": hd,
        "Hq": hq,
        "Hkv": hkv,
        "G": hq // hkv if hkv and hq % hkv == 0 else 0,
        "R": rows,
        "T": A.rows_max,
        "cache": st.cache,
        "dtype": g.types[att.ins[0]].dtype,
        "d_rel": proj.shape[0] if proj is not None else 0,
        "E": proj.shape[1] if proj is not None else 0,
        "nc": nc,
        "nsplit": schedule.attn_splits(extent, nc),
    }


def _info(c: dict) -> Verdict:
    why = []
    if c["G"] == 0:
        why.append("query heads are not a multiple of KV heads")
    if c["nsplit"] > schedule.ATTN_MAX_SPLITS_CHUNKED:
        why.append(f"{c['nsplit']} split blocks per head exceed the kernel's {schedule.ATTN_MAX_SPLITS_CHUNKED}")
    return Verdict(why, c["Hkv"] * min(c["nsplit"], schedule.ATTN_MAX_SPLITS_CHUNKED))


def _roles(m, A):
    """The graph nodes of the visit's attention block bound to the kernel's roles, or a reason string."""
    att = A.node(m.anchors[0])
    if len(att.ins) != 6 or "rel_proj" not in att.attrs or not att.attrs.get("sinks"):
        return "attention without relative bias and sinks"
    q, k, v, _pos, r, sinks = att.ins
    covers = m.covers

    def prod(value: str, op: str) -> int | None:
        n = A.prod.get(value)
        return n if n is not None and n in covers and A.node(n).op == op else None

    def clipped(value: str, what: str):
        """[log_scale <-] l2_clip <- reshape <- the projection output."""
        ln = prod(value, "log_scale")
        if ln is not None:
            value = A.node(ln).ins[0]
        cn = prod(value, "l2_clip")
        rn = prod(A.node(cn).ins[0], "reshape") if cn is not None else None
        if rn is None:
            return f"{what} is not (a log scaling of) an l2_clip of a reshaped projection"
        return ln, cn, rn

    qc, rc = clipped(q, "q"), clipped(r, "r")
    for c in (qc, rc):
        if isinstance(c, str):
            return c
    if (qc[0] is None) != (rc[0] is None):
        return "q and r must both be log scaled or neither"
    if qc[0] is not None and any(A.node(qc[0]).attrs[a] != A.node(rc[0]).attrs[a] for a in ("alpha", "floor")):
        return "q and r are log scaled differently"
    if "scale" not in A.node(qc[1]).attrs or "scale" not in A.node(rc[1]).attrs:
        return "the q and r clips carry a learned scale"
    kn = prod(k, "l2_clip")
    krn = prod(A.node(kn).ins[0], "reshape") if kn is not None else None
    if kn is None or krn is None or "scale" in A.node(kn).attrs:
        return "k is not an unscaled l2_clip of a reshaped convolution"
    vn = prod(v, "rmsnorm")
    vrn = prod(A.node(vn).ins[0], "reshape") if vn is not None else None
    if vn is None or vrn is None or "weight" in A.node(vn).attrs:
        return "v is not an unweighted rmsnorm of a reshaped convolution"
    kind = {"causal_conv1d", "dyn_conv1d"}
    convs = []
    for rn in (krn, vrn):
        cv = A.prod.get(A.node(rn).ins[0])
        if cv is None or cv not in covers or A.node(cv).op not in kind:
            return "k and v must come from a short convolution of their projection"
        convs.append(cv)
    if A.node(convs[0]).op != A.node(convs[1]).op:
        return "k and v convolutions differ in kind"
    kc, vc = A.node(convs[0]), A.node(convs[1])
    if kc.op == "causal_conv1d":
        if not (kc.attrs.get("residual") and vc.attrs.get("residual")):
            return "the causal convolutions of k and v carry the residual"
        if kc.attrs.get("divisor") != vc.attrs.get("divisor"):
            return "k and v convolutions differ in divisor"
    elif kc.attrs["heads"] != vc.attrs["heads"] or kc.attrs["divisor"] != vc.attrs["divisor"]:
        return "k and v dynamic convolutions differ in heads or divisor"
    # the output gate: x = f32(att) then x * sigmoid(f32(gate)), rounded to bf16
    chain = m.chains[0]
    if len(chain) != 3 or A.node(chain[1]).ins[0] != A.node(chain[0]).outs[0]:
        return "the output gate is not sigmoid_mul(f32(attention), f32(gate))"
    gate_cast = A.prod.get(A.node(chain[1]).ins[1])
    if gate_cast is None or gate_cast not in covers or A.node(gate_cast).op != "cast":
        return "the gate is not a cast of a projection output"
    return {
        "q": qc,
        "r": rc,
        "k": (kn, krn, convs[0]),
        "v": (vn, vrn, convs[1]),
        "gate_cast": gate_cast,
        "dyn": kc.op == "dyn_conv1d",
        "sinks": sinks,
    }


def _check(m, A):
    roles = _roles(m, A)
    if isinstance(roles, str):
        return roles
    g = A.graph
    c = m.consts
    channels = c["Hkv"] * c["hd"]
    for which in ("k", "v"):
        cv = A.node(roles[which][2])
        if g.types[cv.ins[0]].shape[-1] != channels:
            return "k / v convolution width is not Hkv * hd"
        shape = g.weights[cv.attrs["weight"]].shape
        if shape != ((CONV_WIDTH, channels) if roles["dyn"] else (channels, CONV_WIDTH)):
            return f"convolution weight {shape} is not the {CONV_WIDTH}-tap kernel's"
        if roles["dyn"] and cv.attrs["heads"] != c["Hkv"]:
            return "dynamic convolution heads differ from the KV heads"
    return None


def _emit(m, A):
    att = A.node(m.anchors[0])
    roles = _roles(m, A)
    c = m.consts
    dyn = roles["dyn"]
    ql, qc, qr = roles["q"]
    _, rc, rr = roles["r"]
    kn, _, kcv = roles["k"]
    vn, _, vcv = roles["v"]
    kcn, vcn = A.node(kcv), A.node(vcv)
    chain = m.chains[0]
    ins = {
        "q": A.node(qr).ins[0],
        "r": A.node(rr).ins[0],
        "k": kcn.ins[0],
        "v": vcn.ins[0],
        "sink": roles["sinks"],
        "gate": A.node(roles["gate_cast"]).ins[0],
    }
    if dyn:
        ins |= {"ktap": kcn.ins[1], "vtap": vcn.ins[1]}
    rel = att.attrs["rel_proj"]
    weights = {
        "q_scale": A.node(qc).attrs["scale"],
        "r_scale": A.node(rc).attrs["scale"],
        "k_conv": kcn.attrs["weight"],
        "v_conv": vcn.attrs["weight"],
        "rel": f"{rel}.t",
    }
    packs = [Pack(w, "dense", [w]) for role, w in weights.items() if role != "rel"]
    packs.append(Pack(weights["rel"], "dense_t", [rel]))
    log = A.node(ql).attrs if ql is not None else None
    return Region(
        "g_attn_rel",
        set(m.covers),
        ins=ins,
        outs={"y": A.node(chain[-1]).outs[0]},
        weights=weights,
        states={"kv": att.attrs["state"], "kconv": kcn.attrs["state"], "vconv": vcn.attrs["state"]},
        attrs={
            "Hq": c["Hq"],
            "Hkv": c["Hkv"],
            "R": c["R"],
            "E": c["E"],
            "window": att.attrs.get("window") or 0,
            "nc": c["nc"],
            "nsplit": c["nsplit"],
            "scale": att.attrs["scale"],
            "dyn": dyn,
            "logs": log is not None,
            "alpha": log["alpha"] if log else 0.0,
            "floor": log["floor"] if log else 1,
            "thr_q": A.node(qc).attrs["threshold"],
            "thr_r": A.node(rc).attrs["threshold"],
            "thr_k": A.node(kn).attrs["threshold"],
            "eps": A.node(vn).attrs["eps"],
            "div": kcn.attrs["divisor"],
        },
        packs=packs,
        numerics=(
            "prep (clips, log scale, k / v short convs with bf16 states, unweighted v norm, KV write at pos mod R) + "
            "split-K flash decode in fp32 (relative bias rounded to bf16 per key, sink logit, window; unrounded P) + "
            "merge with the sink and the output gate bf16(bf16(att) * sigmoid(gate))"
        ),
    )


_ROUND = ("out",)
CAPS = [
    Cap(
        "g_attn_rel",
        "attention",
        pre=(
            Absorb("reshape", min=4, max=4),
            Absorb("l2_clip", min=3, max=3),
            Absorb("log_scale", min=0, max=2),
            Absorb("causal_conv1d", min=0, max=2),
            Absorb("dyn_conv1d", min=0, max=2),
            Absorb("rmsnorm", min=1, max=1),
            Absorb("cast", min=1, max=1, attrs={"dtype": "f32"}),
        ),
        post=(
            Absorb("cast", min=1, max=1, attrs={"dtype": "f32"}),
            Absorb("sigmoid_mul", min=1, max=1),
            Absorb("cast", min=1, max=1, attrs={"dtype": "bf16"}),
        ),
        limits={
            "hd_in": (HD,),
            "G_max": MAX_G,
            "T_max": schedule.STEP_ROWS,
            "dtype_in": ("bf16",),
            "cache_in": ("bf16",),
            "d_rel_in": (D_REL,),
        },
        rounds={
            "l2_clip": _ROUND,
            "log_scale": _ROUND,
            "causal_conv1d": _ROUND,
            "dyn_conv1d": _ROUND,
            "rmsnorm": _ROUND,
            "attention": ("bias", "out"),
            "cast": _ROUND,
        },
        tolerance=("attention.p",),
        launches=3,
        operand_cap=20,
        constants=_constants,
        info=_info,
        check=_check,
        emit=_emit,
    )
]


def _check_rc(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


def _launch(L: C.CDLL, mem, t, T: int, st: int) -> None:
    a = t.attrs
    hq, hkv, nsplit, nc, dyn = a["Hq"], a["Hkv"], a["nsplit"], a["nc"], a["dyn"]
    rows = mem.rows_max
    ws = mem.scratch("ink_attn_ws", L.warpc_ink_attn_ws_bytes(rows, hq, nsplit))
    qo = mem.scratch("ink_attn_q", rows * hq * HD * 2)
    ro = mem.scratch("ink_attn_r", rows * hq * D_REL * 2)
    kv, pos = mem.states[t.states["kv"]], mem.pos
    packs = mem.packs
    rc = L.warpc_ink_attn_prep(
        T,
        mem.buffer(t.ins["q"]),
        mem.buffer(t.ins["r"]),
        mem.buffer(t.ins["k"]),
        mem.buffer(t.ins["v"]),
        mem.buffer(t.ins["ktap"]) if dyn else None,
        mem.buffer(t.ins["vtap"]) if dyn else None,
        packs[t.weights["q_scale"]],
        packs[t.weights["r_scale"]],
        packs[t.weights["k_conv"]],
        packs[t.weights["v_conv"]],
        mem.states[t.states["kconv"]],
        mem.states[t.states["vconv"]],
        qo,
        ro,
        kv,
        pos,
        hq,
        hkv,
        a["R"],
        int(dyn),
        int(a["logs"]),
        C.c_float(a["thr_q"]),
        C.c_float(a["thr_r"]),
        C.c_float(a["thr_k"]),
        C.c_float(a["eps"]),
        C.c_float(a["div"]),
        C.c_float(a["alpha"]),
        C.c_float(float(a["floor"])),
        st,
    )
    _check_rc(L, rc, f"{t.kernel} prep")
    rc = L.warpc_ink_attn_split(
        T,
        qo,
        ro,
        packs[t.weights["rel"]],
        kv,
        pos,
        ws,
        hq,
        hkv,
        a["R"],
        a["E"],
        a["window"],
        nc,
        nsplit,
        C.c_float(a["scale"]),
        st,
    )
    _check_rc(L, rc, f"{t.kernel} split")
    rc = L.warpc_ink_attn_merge(
        T,
        mem.buffer(t.ins["sink"]),
        mem.buffer(t.ins["gate"]),
        pos,
        ws,
        mem.buffer(t.outs["y"]),
        hq,
        a["window"],
        nc,
        nsplit,
        st,
    )
    _check_rc(L, rc, f"{t.kernel} merge")


LAUNCHERS = {"g_attn_rel": _launch}
