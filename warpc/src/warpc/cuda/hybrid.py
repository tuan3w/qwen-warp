"""Stage 2 families 1-3 of the Qwen3.5 hybrid: the zero-centred RMSNorm (with the residual add), and the small a|b
GEMV (the T = 1 bf16 head is `launch._head`'s). The kernels are kernels/qwen35_core.cu; their floating-point order is
the hand engine's, so each is bit-identical to its libswiftmk counterpart (smk_rmsnorm, smk_bf16_gemv_small,
smk_bf16_gemv_f32).

A family module exports CAPS, LAUNCHERS, SIGNATURES and PACKS (see w4.py)."""

from __future__ import annotations

import ctypes as C

from ..compiler import Absorb, Cap, Pack, Region
from . import schedule

_PTR, _INT, _FLOAT = C.c_void_p, C.c_int, C.c_float
SIGNATURES = {
    "warpc_rmsnorm_zc": ([_PTR, _PTR, _PTR, _PTR, _INT, _INT, _FLOAT, _PTR], _INT),
    "warpc_head_gemv": ([_PTR, _PTR, _INT, _INT, _PTR, _PTR], _INT),
    "warpc_gemv_small": ([_PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _PTR, _PTR], _INT),
}
PACKS: dict = {}

ROWS = schedule.STEP_ROWS
SMALL_N_MAX = 128  # the one-block-per-row kernel serves projections too narrow to fill the GPU (a|b: 96 rows)


def _check(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: CUDA error {rc}")


# ------------------------------------------------------------------------------------------------ zero-centred norm
def _norm_consts(m, A):
    n = A.node(m.anchors[0])
    x = A.graph.types[n.ins[0]]
    return {
        "H": x.shape[-1],
        "T": A.rows_max,
        "rank": len(x.shape),
        "dtype": x.dtype,
        "offset": n.attrs.get("offset", 0.0),
    }


def _emit_rmsnorm(kernel: str, add: bool):
    def emit(m, A):
        n = A.node(m.anchors[0])
        w = n.attrs["weight"]
        attrs = {"H": A.graph.types[n.ins[0]].shape[-1], "eps": n.attrs["eps"]}
        numerics = "zero-centred rmsnorm: fp32 statistics in the hand engine's block tree, x-hat (1 + w), one rounding"
        if not add:
            return Region(
                kernel,
                set(m.covers),
                ins={"x": n.ins[0]},
                outs={"xn": n.outs[0]},
                weights={"w": w},
                packs=[Pack(w, "dense", [w])],
                attrs=attrs,
                numerics=numerics,
            )
        an = A.node(m.inplace_node)
        y = an.ins[1] if an.ins[0] == m.h else an.ins[0]
        return Region(
            kernel,
            set(m.covers),
            ins={"h": m.h, "y": y},
            outs={"h_out": an.outs[0], "xn": n.outs[0]},
            weights={"w": w},
            packs=[Pack(w, "dense", [w])],
            attrs=attrs,
            inplace=("h_out", "h"),
            numerics="h = bf16(h + y) in place; " + numerics,
        )

    return emit


def _rmsnorm(L, mem, t, T, st):
    add = t.kernel.endswith("_add")
    rc = L.warpc_rmsnorm_zc(
        mem.buffer(t.ins["h"] if add else t.ins["x"]),
        mem.buffer(t.ins["y"]) if add else None,
        mem.packs[t.weights["w"]],
        mem.buffer(t.outs["xn"]),
        T,
        t.attrs["H"],
        C.c_float(t.attrs["eps"]),
        st,
    )
    _check(L, rc, t.kernel)


# ------------------------------------------------------------------------------------------------ small GEMV
def _lin_shape(A, i):
    return A.graph.weights[A.node(i).attrs["weight"]].shape


def _small_consts(m, A):
    shapes = [_lin_shape(A, i) for i in m.anchors]
    return {
        "segs": len(shapes),
        "N": sum(s[0] for s in shapes),
        "K": shapes[0][1],
        "T": A.rows_max,
        "dtype": A.graph.types[A.node(m.anchors[0]).ins[0]].dtype,
    }


def _small_check(m, A):
    for i in m.anchors:
        if A.consumers(A.node(i).outs[0], "cast"):
            return "a linear feeding a cast is the lm_head"
    return None


def _emit_small(m, A):
    lins = [A.node(i) for i in m.anchors]
    ws = [n.attrs["weight"] for n in lins]
    ns = [A.graph.weights[w].shape[0] for w in ws]
    name = "|".join(ws)
    return Region(
        "g_gemv_small",
        set(m.covers),
        ins={"x": lins[0].ins[0]},
        outs={f"o{j}": n.outs[0] for j, n in enumerate(lins)},
        weights={"w": name},
        packs=[Pack(name, "dense_cat", ws)],
        # side: may overlap the task before it when that task reads the same input (the a|b projections beside qkv|z)
        attrs={"N": sum(ns), "na": ns[0], "K": A.graph.weights[ws[0]].shape[1], "side": True},
        numerics="bf16 weights, one 128-thread block per row, per-thread fma chains, warp butterfly,"
        " (w0 + w1) + (w2 + w3), one bf16 rounding",
    )


def _small(L, mem, t, T, st):
    a = t.attrs
    o0, o1 = mem.buffer(t.outs["o0"]), mem.buffer(t.outs["o1"])
    rc = L.warpc_gemv_small(mem.packs[t.weights["w"]], mem.buffer(t.ins["x"]), a["N"], a["na"], a["K"], T, o0, o1, st)
    _check(L, rc, t.kernel)


_RESADD = Absorb("add", min=1, max=1, exposes=True, inplace=True)
# H_max: one 16-byte chunk per thread and at most 1024 threads
_NORM_LIMITS = {
    "offset_in": (1.0,),
    "H_multiple": 8,
    "H_max": 8192,
    "T_max": ROWS,
    "rank_in": (2,),
    "dtype_in": ("bf16",),
}

CAPS = [
    Cap(
        "g_rmsnorm_zc_add",
        "rmsnorm",
        pre=(_RESADD,),
        limits=_NORM_LIMITS,
        rounds={"add": ("out",), "rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm("g_rmsnorm_zc_add", True),
    ),
    Cap(
        "g_rmsnorm_zc",
        "rmsnorm",
        limits=_NORM_LIMITS,
        rounds={"rmsnorm": ("out",)},
        constants=_norm_consts,
        emit=_emit_rmsnorm("g_rmsnorm_zc", False),
    ),
    Cap(
        "g_gemv_small",
        "linear",
        anchor_schemes=("dense",),
        segments=2,
        limits={"segs_in": (2,), "N_max": SMALL_N_MAX, "K_multiple": 8, "T_max": ROWS, "dtype_in": ("bf16",)},
        rounds={"linear": ("out",)},
        constants=_small_consts,
        check=_small_check,
        emit=_emit_small,
    ),
]

LAUNCHERS = {"g_rmsnorm_zc": _rmsnorm, "g_rmsnorm_zc_add": _rmsnorm, "g_gemv_small": _small}
