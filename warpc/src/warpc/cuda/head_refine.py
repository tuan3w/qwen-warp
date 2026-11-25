"""Stage 2 family 7: the int4 head with exact rescoring (the hand engine's `s_head_refine`).

The lm_head runs twice: the approximate logits from an int4 g128 copy of the head (kernels/w4g128_gemv.cu, F32 epilogue,
the K split 8 the hand engine uses for the head), then the top R = 16 candidates of every row are rescored exactly with
the bf16 head and the argmax is taken over the exact candidate logits (kernels/qwen35_head_refine.cu, which states the
specification). The logits buffer holds the exact logits on the candidates and the approximate ones elsewhere, so a
plan using this kernel is reference-tolerance on `linear.out` and `argmax.out`; its ids equal the bf16 head's iff the
bf16 top-1 is among the candidates.

The capability matches the same nodes as the exact `g_head`: `CudaBackend(head="int4_refine")` selects it (table.HEADS).
"""

from __future__ import annotations

import ctypes as C

import torch

from ..compiler import Absorb, Cap, Pack, Region
from . import schedule, w4

_PTR, _INT = C.c_void_p, C.c_int

R = 16  # candidates rescored per row (the hand engine's HEAD_R_GREEDY)
KS = 8  # the K split of the approximate GEMV (w4g128_gemv.cuh: part of its numerics, the hand engine's head config)
CHUNK, NCH_MAX = 2048, 128  # qwen35_head_refine.cu: logits per chunk, chunk slots per row

SIGNATURES = {
    "warpc_headq_ws_bytes": ([], C.c_size_t),
    "warpc_headq_ws_cand": ([_PTR], _PTR),
    "warpc_headq_ws_exact": ([_PTR], _PTR),
    # warpc_headq_select(logits, w, x, N, K, T, R, out_tok, ws, stream) -> cudaError_t
    "warpc_headq_select": ([_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _PTR, _PTR], _INT),
}

# ---------------------------------------------------------------------------------------------------------- pack

_ROWS = 2048  # rows per quantisation chunk


def quant_head_i4(w: torch.Tensor, ow: torch.Tensor, os_: torch.Tensor) -> None:
    """int4 g128 round-to-nearest of the bf16 rows `w` into natural W4 `ow` / `os_`: scale = bf16(amax / 7) per 128
    columns, q = round(w / scale) clamped to [-8, 7], stored as q + 8 (the hand engine's K9 head, bit for bit)."""
    n, k = w.shape
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    g = w.float().view(n, k // w4.GROUP, w4.GROUP)
    sb = (g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7).to(torch.bfloat16).float()
    q = torch.round(g / sb).clamp_(-8, 7)
    word = ((q.to(torch.int64) + 8).view(n, k // 8, 8) << sh).sum(-1)
    ow.copy_(torch.where(word >= 2**31, word - 2**32, word).to(torch.int32))
    os_.copy_(sb.squeeze(-1).to(torch.bfloat16))


def _pack_head(packer, pack: Pack) -> torch.Tensor:
    """head_i4g: the dense head quantised to int4 g128 and repacked as the W4G128 GEMV's layout."""
    d = packer.weights[pack.parts[0]]
    host = packer._host_dense(pack.parts[0])
    n, k = d.shape
    buf, w, s = w4._empty(n, k, packer.device)
    for r0 in range(0, n, _ROWS):
        quant_head_i4(host[r0 : r0 + _ROWS].to(packer.device), w[r0 : r0 + _ROWS], s[r0 : r0 + _ROWS])
    w4.repack(w, s)
    return buf


PACKS = {"head_i4g": _pack_head}


# ------------------------------------------------------------------------------------------------------ launcher


def _head_refine(L: C.CDLL, mem, t, T: int, st: int) -> None:
    n, k = t.attrs["N"], t.attrs["K"]
    wq = mem.packs[t.weights["wq"]]
    x, logits, ids = mem.buffer(t.ins["x"]), mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    outs = (C.c_void_p * 3)()
    outs[0] = logits
    rc = L.warpc_w4g128_gemv(wq, wq + w4.scale_offset(n, k), x, n, k, T, w4.EPI_F32, None, outs, KS, st)
    if rc == 0:
        ws = mem.scratch("headq_ws", L.warpc_headq_ws_bytes())  # zeroed once; the kernels leave it zeroed
        rc = L.warpc_headq_select(logits, mem.packs[t.weights["w"]], x, n, k, T, R, ids, ws, st)
    if rc != 0:
        raise RuntimeError(f"{t.kernel} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


LAUNCHERS = {"g_head_refine": _head_refine}


# ------------------------------------------------------------------------------------------------ capability


def _consts(m, A):
    w = A.graph.weights[A.node(m.anchors[0]).attrs["weight"]]
    return {
        "N": w.shape[0],
        "K": w.shape[1],
        "T": A.rows_max,
        "dtype": A.graph.types[A.node(m.anchors[0]).ins[0]].dtype,
    }


def _emit(m, A):
    ln, cast, amax = A.node(m.anchors[0]), A.node(m.post[0]), A.node(m.post[1])
    w = A.graph.weights[ln.attrs["weight"]]
    q = f"{w.name}.i4g"
    return Region(
        "g_head_refine",
        set(m.covers),
        ins={"x": ln.ins[0]},
        outs={"logits": cast.outs[0], "ids": amax.outs[0]},
        weights={"w": w.name, "wq": q},
        packs=[Pack(w.name, "dense", [w.name]), Pack(q, "head_i4g", [w.name])],
        attrs={"N": w.shape[0], "K": w.shape[1]},
        numerics=(
            f"approximate fp32 logits of an int4 g128 copy of the head (mma, K split {KS}), the top {R} per row"
            " rescored exactly with the bf16 head (fp32, lane-chunk fma chains and a butterfly sum); argmax over the"
            " rescored candidates, lowest index on ties"
        ),
    )


CAPS = [
    Cap(
        "g_head_refine",
        "linear",
        anchor_schemes=("dense",),
        post=(
            Absorb("cast", min=1, max=1, exposes=True, attrs={"dtype": "f32"}),
            Absorb("argmax", min=1, max=1, exposes=True, reduction_ok=True),
        ),
        limits={
            "K_multiple": 128 * KS,
            "N_multiple": 4,
            "N_min": 128 * R,
            "N_max": CHUNK * NCH_MAX,
            "T_max": schedule.STEP_ROWS,
            "dtype_in": ("bf16",),
        },
        rounds={"cast": ("out",)},
        tolerance=("linear.out", "argmax.out"),
        launches=5,
        constants=_consts,
        gather=("w",),  # warpc_headq_select rescores the candidate rows only; the int4 copy wq is scanned
        emit=_emit,
    ),
]
