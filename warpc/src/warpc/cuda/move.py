"""Data-movement kernels of the MTP draft graph (kernels/move.cu): `concat`, `select_row` and `take`, each its own
capability on its own op (no fusion closure) and exact: bytes are copied, an id is an exact fp32 integer.

A family module exports CAPS, LAUNCHERS, SIGNATURES and PACKS (see w4.py)."""

from __future__ import annotations

import ctypes as C

from ..compiler import Cap, Region
from . import schedule

_PTR, _INT = C.c_void_p, C.c_int
SIGNATURES = {
    "warpc_concat": ([_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR], _INT),
    "warpc_select_row": ([_PTR, _PTR, _PTR, _INT, _INT, _PTR], _INT),
    "warpc_take": ([_PTR, _PTR, _PTR, _PTR], _INT),
}
PACKS: dict = {}

ITEMSIZE = {"bf16": 2, "f32": 4}


def _check(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


# ------------------------------------------------------------------------------------------------ concat
def _concat_consts(m, A):
    n = A.node(m.anchors[0])
    x, y = (A.graph.types[v] for v in n.ins)
    return {"a": x.shape[1], "b": y.shape[1], "T": A.rows_max, "dtype": x.dtype}


def _emit_concat(m, A):
    n = A.node(m.anchors[0])
    x, y = (A.graph.types[v] for v in n.ins)
    return Region(
        "g_concat",
        set(m.covers),
        ins={"x": n.ins[0], "y": n.ins[1]},
        outs={"out": n.outs[0]},
        attrs={"a": x.shape[1], "b": y.shape[1]},
        numerics="bytes copied: out[t] = [x[t], y[t]]",
    )


def _concat(L, mem, t, T, st):
    a = t.attrs
    rc = L.warpc_concat(
        mem.buffer(t.ins["x"]),
        mem.buffer(t.ins["y"]),
        mem.buffer(t.outs["out"]),
        T,
        a["a"],
        a["b"],
        ITEMSIZE["bf16"],
        st,
    )
    _check(L, rc, t.kernel)


# ------------------------------------------------------------------------------------------------ select_row
def _select_consts(m, A):
    x = A.graph.types[A.node(m.anchors[0]).ins[0]]
    return {"D": x.shape[1], "dtype": x.dtype}


def _emit_select_row(m, A):
    n = A.node(m.anchors[0])
    x = A.graph.types[n.ins[0]]
    return Region(
        "g_select_row",
        set(m.covers),
        ins={"x": n.ins[0], "i": n.ins[1]},
        outs={"out": n.outs[0]},
        attrs={"D": x.shape[1], "esz": ITEMSIZE[x.dtype]},
        numerics="bytes copied: out = x[i[0]], i a device scalar read in the kernel",
    )


def _select_row(L, mem, t, T, st):
    a = t.attrs
    rc = L.warpc_select_row(
        mem.buffer(t.ins["x"]), mem.buffer(t.ins["i"]), mem.buffer(t.outs["out"]), a["D"], a["esz"], st
    )
    _check(L, rc, t.kernel)


# ------------------------------------------------------------------------------------------------ take
def _emit_take(m, A):
    n = A.node(m.anchors[0])
    return Region(
        "g_take",
        set(m.covers),
        ins={"i": n.ins[0]},
        outs={"y": n.outs[0]},
        constants={"table": tuple(float(v) for v in n.attrs["table"])},
        numerics="y = table[i[0]]: the table is an fp32 plan constant (ids < 2^24 are exact)",
    )


def _take(L, mem, t, T, st):
    rc = L.warpc_take(mem.buffer(t.ins["i"]), mem.constants[t.constants["table"]], mem.buffer(t.outs["y"]), st)
    _check(L, rc, t.kernel)


CAPS = [
    Cap(
        "g_concat",
        "concat",
        limits={"a_multiple": 8, "b_multiple": 8, "T_max": schedule.STEP_ROWS, "dtype_in": ("bf16",)},
        rounds={"concat": ("out",)},
        constants=_concat_consts,
        emit=_emit_concat,
    ),
    Cap(
        "g_select_row",
        "select_row",
        limits={"D_multiple": 8, "dtype_in": ("bf16", "f32")},
        rounds={"select_row": ("out",)},
        constants=_select_consts,
        emit=_emit_select_row,
    ),
    Cap("g_take", "take", emit=_emit_take),
]

LAUNCHERS = {"g_concat": _concat, "g_select_row": _select_row, "g_take": _take}
