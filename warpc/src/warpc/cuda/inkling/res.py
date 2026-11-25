"""Inkling-2 kernel family `res` (kernels/inkling_res*.cu): the block-residual read `g_attn_res`, the loop-message
controller `g_msg_gate` and the short convolution with residual `g_convres`.

  g_attn_res  attn_res; up to 48 segments over identical rows (one query each: a full layer's three, a pass's 48 message
              reads); per segment an optional rmsnorm (weighted) + cast bf16 epilogue
  g_msg_gate  message_gate; up to 4 independent gates (a loop layer's q / k / v merges); per gate an optional rmsnorm
              (weighted) + cast bf16 epilogue
  g_convres   causal_conv1d with residual / divisor, bf16 [C, 3] state; epilogue chain: cast f32, add f32 (the residual
              stream), cast bf16

The attn_res queries are f32-scheme weights (pack kind `f32`; a stacked [N, H] query is one pack and the task's `row`
picks the row).
"""

from __future__ import annotations

import ctypes as C
from collections.abc import Callable

from ...compiler import Absorb, Cap, Pack, Region
from ...runtime import Memory

_PTR, _INT, _UINT, _FLOAT, _SIZE = C.c_void_p, C.c_int, C.c_uint, C.c_float, C.c_size_t
_ROWS = 8  # kernels/dense_common.cuh STEP_ROWS
# kernels/inkling_res.cuh: AR_MAXROWS, AR_MAXSEG, 256 * AR_KE, MG_MAXSEG, 128 * MG_MAXNB
AR_ROWS, AR_SEGMENTS, AR_H = 6, 48, 2304
MG_SEGMENTS, MG_H = 4, 8192


# ---------------------------------------------------------------- caps
def _norm_chain_why(m, A) -> str | None:
    """Why the epilogue chains of a match are not the kernel's: each is nothing or (weighted rmsnorm, cast bf16), the
    norms of one launch sharing eps."""
    eps = set()
    for chain in m.chains:
        if not chain:
            continue
        if len(chain) != 2:
            return "the rmsnorm epilogue needs its bf16 cast after it"
        nn = A.node(chain[0])
        if "weight" not in nn.attrs or nn.attrs.get("offset", 0.0) != 0.0:
            return "the norm epilogue computes x-hat * w: no weightless or offset norm"
        eps.add(nn.attrs["eps"])
    if len(eps) > 1:
        return "the norms of one launch have different eps"
    return None


def _ar_consts(m, A):
    n = A.node(m.anchors[0])
    t = A.graph.types[n.ins[0]]
    return {"rows": len(n.ins), "H": t.shape[-1], "T": A.rows_max, "rank": len(t.shape)}


def _ar_check(m, A):
    nodes = [A.node(i) for i in m.anchors]
    if len({n.attrs["eps"] for n in nodes}) > 1:
        return "the reads of one launch have different eps"
    for n in nodes:
        q = A.graph.weights[n.attrs["query"]]
        if q.scheme != "f32":
            return f"query {q.name} is {q.scheme}: the kernel reads fp32 queries"
    return _norm_chain_why(m, A)


def _emit_attn_res(m, A):
    g = A.graph
    nodes = [A.node(i) for i in m.anchors]
    ins = nodes[0].ins
    ins_roles = {f"r{i}": v for i, v in enumerate(ins)}
    outs, weights, packs = {}, {}, {}
    rows, eps_n = [], 0.0
    for j, (n, chain) in enumerate(zip(nodes, m.chains, strict=True)):
        q = n.attrs["query"]
        weights[f"q{j}"] = q
        packs[q] = Pack(q, "f32", [q])
        rows.append(n.attrs.get("row", 0))
        if chain:
            nn = A.node(chain[0])
            outs[f"n{j}"] = A.node(chain[-1]).outs[0]
            weights[f"nw{j}"] = nn.attrs["weight"]
            packs[nn.attrs["weight"]] = Pack(nn.attrs["weight"], "dense", [nn.attrs["weight"]])
            eps_n = nn.attrs["eps"]
        else:
            outs[f"f{j}"] = n.outs[0]
    f32mask = sum(1 << i for i, v in enumerate(ins) if g.types[v].dtype == "f32")
    return Region(
        "g_attn_res",
        set(m.covers),
        ins=ins_roles,
        outs=outs,
        weights=weights,
        packs=list(packs.values()),
        attrs={
            "n": len(nodes),
            "rows": len(ins),
            "H": g.types[ins[0]].shape[-1],
            "eps": nodes[0].attrs["eps"],
            "eps_n": eps_n,
            "f32mask": f32mask,
            "row": rows,
        },
        numerics="fp32; row statistics, softmax and the norm of the mix from one reduction of the rows' Gram matrix",
    )


def _mg_consts(m, A):
    n = A.node(m.anchors[0])
    down = A.graph.weights[n.attrs["down"]]
    return {"R": down.shape[0], "H": A.graph.types[n.ins[0]].shape[-1], "T": A.rows_max}


def _emit_msg_gate(m, A):
    nodes = [A.node(i) for i in m.anchors]
    ins, outs, weights, packs = {}, {}, {}, {}
    eps_n = 0.0
    for j, (n, chain) in enumerate(zip(nodes, m.chains, strict=True)):
        ins[f"x{j}"], ins[f"m{j}"] = n.ins
        for role in ("down", "forget", "gain"):
            w = n.attrs[role]
            weights[f"{role}{j}"] = w
            packs[w] = Pack(w, "dense", [w])
        if chain:
            nn = A.node(chain[0])
            outs[f"n{j}"] = A.node(chain[-1]).outs[0]
            weights[f"nw{j}"] = nn.attrs["weight"]
            packs[nn.attrs["weight"]] = Pack(nn.attrs["weight"], "dense", [nn.attrs["weight"]])
            eps_n = nn.attrs["eps"]
        else:
            outs[f"f{j}"] = n.outs[0]
    return Region(
        "g_msg_gate",
        set(m.covers),
        ins=ins,
        outs=outs,
        weights=weights,
        packs=list(packs.values()),
        attrs={
            "n": len(nodes),
            "H": A.graph.types[nodes[0].ins[0]].shape[-1],
            "eps": nodes[0].attrs["eps"],
            "eps_n": eps_n,
        },
        numerics="fp32 tanh gates; the normed pair and the rank-64 hidden rounded to bf16 as the reference does",
    )


def _conv_consts(m, A):
    n = A.node(m.anchors[0])
    w = A.graph.weights[n.attrs["weight"]]
    return {
        "C": w.shape[0],
        "W": w.shape[1],
        "T": A.rows_max,
        "dtype": A.graph.types[n.ins[0]].dtype,
        "residual": bool(n.attrs.get("residual", False)),
    }


def _emit_convres(m, A):
    n = A.node(m.anchors[0])
    chain = [A.node(i) for i in m.post]
    add = next((c for c in chain if c.op == "add"), None)
    value = n.outs[0]
    if chain and chain[0].op == "cast":
        value = chain[0].outs[0]
    res = None
    if add is not None:
        res = next(v for v in add.ins if v != value)
    out_f32 = bool(chain) and chain[-1].attrs.get("dtype", "f32") == "f32"
    w, s = n.attrs["weight"], n.attrs["state"]
    return Region(
        "g_convres",
        set(m.covers),
        ins={"x": n.ins[0], **({"res": res} if res else {})},
        outs={"out": chain[-1].outs[0] if chain else n.outs[0]},
        weights={"w": w},
        states={"state": s},
        packs=[Pack(w, "dense", [w])],
        attrs={
            "C": A.graph.weights[w].shape[0],
            "div": float(n.attrs.get("divisor", 1.0)),
            "res": res is not None,
            "out_f32": out_f32,
        },
        numerics="fp32 taps summed in order, residual, divided by the divisor (multiplied by its fp32 reciprocal, as "
        "torch does on the GPU), one rounding",
    )


CAPS = [
    Cap(
        "g_attn_res",
        "attn_res",
        segments=AR_SEGMENTS,
        post=(Absorb("rmsnorm", reduction_ok=True), Absorb("cast", attrs={"dtype": "bf16"})),
        limits={"rows_max": AR_ROWS, "H_max": AR_H, "T_max": _ROWS, "rank_in": (2,)},
        rounds={"cast": ("out",)},
        operand_cap=AR_ROWS + 2 * AR_SEGMENTS + 8,
        constants=_ar_consts,
        check=_ar_check,
        emit=_emit_attn_res,
    ),
    Cap(
        "g_msg_gate",
        "message_gate",
        segments=MG_SEGMENTS,
        segment_inputs=MG_SEGMENTS,
        post=(Absorb("rmsnorm", reduction_ok=True), Absorb("cast", attrs={"dtype": "bf16"})),
        limits={"R_in": (64,), "H_multiple": 8, "H_max": MG_H, "T_max": _ROWS},
        rounds={"message_gate": ("input", "hidden"), "cast": ("out",)},
        launches=2,
        operand_cap=8 * MG_SEGMENTS,
        constants=_mg_consts,
        check=lambda m, A: _norm_chain_why(m, A),
        emit=_emit_msg_gate,
    ),
    Cap(
        "g_convres",
        "causal_conv1d",
        post=(
            Absorb("cast", attrs={"dtype": "f32"}),
            Absorb("add"),
            Absorb("cast", attrs={"dtype": "bf16"}),
        ),
        limits={"W_in": (4,), "residual_in": (True,), "dtype_in": ("bf16",), "T_max": _ROWS},
        rounds={"causal_conv1d": ("out",), "cast": ("out",)},
        constants=_conv_consts,
        emit=_emit_convres,
    ),
]


# ---------------------------------------------------------------- launchers
def _check(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


def _ptrs(n: int, values) -> C.Array:
    """A host array of device pointers (None = NULL); the kernel copies it into its parameters during the call."""
    return (C.c_void_p * n)(*values)


def _attn_res(L: C.CDLL, mem: Memory, t, T: int, st: int) -> None:
    a = t.attrs
    n, H = a["n"], a["H"]
    rows = _ptrs(a["rows"], [mem.buffer(t.ins[f"r{i}"]) for i in range(a["rows"])])
    q = _ptrs(n, [mem.packs[t.weights[f"q{j}"]] + a["row"][j] * H * 4 for j in range(n)])
    outf = _ptrs(n, [mem.buffer(t.outs[f"f{j}"]) if f"f{j}" in t.outs else None for j in range(n)])
    outn = _ptrs(n, [mem.buffer(t.outs[f"n{j}"]) if f"n{j}" in t.outs else None for j in range(n)])
    nw = _ptrs(n, [mem.packs[t.weights[f"nw{j}"]] if f"nw{j}" in t.weights else None for j in range(n)])
    rc = L.warpc_ink_attn_res(
        a["rows"], rows, a["f32mask"], n, q, outf, outn, nw, T, H, C.c_float(a["eps"]), C.c_float(a["eps_n"]), st
    )
    _check(L, rc, t.kernel)


def _msg_gate(L: C.CDLL, mem: Memory, t, T: int, st: int) -> None:
    a = t.attrs
    n, H = a["n"], a["H"]

    def role(kind: str, f: Callable[[str], int]):
        return _ptrs(n, [f(t.weights[f"{kind}{j}"]) if f"{kind}{j}" in t.weights else None for j in range(n)])

    x = _ptrs(n, [mem.buffer(t.ins[f"x{j}"]) for j in range(n)])
    m = _ptrs(n, [mem.buffer(t.ins[f"m{j}"]) for j in range(n)])
    outf = _ptrs(n, [mem.buffer(t.outs[f"f{j}"]) if f"f{j}" in t.outs else None for j in range(n)])
    outn = _ptrs(n, [mem.buffer(t.outs[f"n{j}"]) if f"n{j}" in t.outs else None for j in range(n)])
    pack = mem.packs.__getitem__
    ws = mem.scratch("ink_msg_gate", L.warpc_ink_mg_ws_bytes(H))
    rc = L.warpc_ink_msg_gate(
        n,
        x,
        m,
        role("down", pack),
        role("forget", pack),
        role("gain", pack),
        outf,
        outn,
        role("nw", pack),
        ws,
        T,
        H,
        C.c_float(a["eps"]),
        C.c_float(a["eps_n"]),
        st,
    )
    _check(L, rc, t.kernel)


def _convres(L: C.CDLL, mem: Memory, t, T: int, st: int) -> None:
    a = t.attrs
    rc = L.warpc_ink_conv_res(
        mem.buffer(t.ins["x"]),
        mem.buffer(t.ins["res"]) if a["res"] else None,
        mem.packs[t.weights["w"]],
        mem.states[t.states["state"]],
        mem.buffer(t.outs["out"]),
        int(a["out_f32"]),
        T,
        a["C"],
        C.c_float(a["div"]),
        st,
    )
    _check(L, rc, t.kernel)


LAUNCHERS = {"g_attn_res": _attn_res, "g_msg_gate": _msg_gate, "g_convres": _convres}

SIGNATURES = {
    "warpc_ink_attn_res": ([_INT, _PTR, _UINT, _INT, _PTR, _PTR, _PTR, _PTR, _INT, _INT, _FLOAT, _FLOAT, _PTR], _INT),
    "warpc_ink_mg_ws_bytes": ([_INT], _SIZE),
    "warpc_ink_msg_gate": ([_INT, *[_PTR] * 8, _PTR, _INT, _INT, _FLOAT, _FLOAT, _PTR], _INT),
    "warpc_ink_conv_res": ([_PTR, _PTR, _PTR, _PTR, _PTR, _INT, _INT, _INT, _FLOAT, _PTR], _INT),
}
