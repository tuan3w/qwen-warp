"""Stage 2 family 4: the W4G128 GEMV (qkv / z / out / q|k|v / o / gate|up|down of the Qwen3.5 hybrid).

A family module exports `CAPS` (capabilities appended to the CUDA table), `LAUNCHERS` (kernel name -> `(lib, mem, task,
T, stream)`), `SIGNATURES` (ctypes signatures of its extern "C" entries, merged into glue.SIGNATURES) and `PACKS` (pack
kind -> `(packer, pack) -> device tensor`, merged into pack.Packer).

The kernel is `kernels/w4g128_gemv.cu(h)` (numerics and weight layout are specified in the header): int4 weights
with one bf16 scale per 128 columns, mma m16n8k16, T <= STEP_ROWS rows per weight pass. Two capabilities mirror the
hand engine's `s_w4_gemv` / `s_w4_gemv_silu` (swiftmk/table.py) role for role, so a plan using them is task for task
the hand step.
"""

from __future__ import annotations

import ctypes as C
import itertools
import math

import torch

from ..compiler import Cap, Pack, Region
from . import schedule

_PTR, _INT = C.c_void_p, C.c_int

GROUP = 128  # columns per scale
TILE = 16  # rows per mma tile

# warpc_w4g128_gemv(w, scales, x, N, K, T, kind, ends, outs, ks, stream) -> cudaError_t
SIGNATURES = {"warpc_w4g128_gemv": ([_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _PTR, _INT, _PTR], _INT)}

EPI_SEGS, EPI_SILU, EPI_F32 = 0, 1, 2  # F32: unrounded fp32 sums [T][N], the gates' view of the numerics

# The K split (warps per 16-row tile, 4 / 8 / 1) is part of the kernel's numerics: the partial sums of the splits are
# added in a fixed order, so another KS is another rounding. Preferred KS by N: the hand engine's in-graph sweep found
# 8 best for the 14336-row attention q|k|v and 4 for every other shape of the 27B (+-0.2% between configurations); the
# same choice makes this kernel's bits equal the hand kernel's.
KS_PREFERRED = {14336: 8}
KS_DEFAULT = 4
NUM_W4 = (
    "int4 g128 weights in the repacked tile layout, mma m16n8k16 per 128-column group, fp32 group partials scaled by"
    " fma and summed in a fixed order over the K split, one bf16 rounding"
)


def w4_ks(n: int, k: int) -> int:
    """The K split of an (N, K) GEMV: the preferred one if it divides the work (K / 128 groups % KS, 16-row tiles % (8 /
    KS)), else the other of 4 / 8, else 1 (always valid)."""
    first = KS_PREFERRED.get(n, KS_DEFAULT)
    for ks in (first, 12 - first, 1):
        if (k // GROUP) % ks == 0 and (n // TILE) % (8 // ks) == 0:
            return ks
    raise ValueError(f"no K split for N={n}, K={k}")


def scale_offset(n: int, k: int) -> int:
    """Byte offset of the scales in a pack: the weights are N * K / 2 bytes."""
    return n * k // 2


# ---------------------------------------------------------------------------------------------------------- packs

_ROWS = 2048  # rows per repack chunk: bounds the int64 temporaries to ~0.3 GB at K = 17408


def _empty(n: int, k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(storage, words int32 [n, k/8], scales bf16 [n, k/128]): the two views share the one uint8 storage."""
    off = scale_offset(n, k)
    buf = torch.empty(off + n * (k // GROUP) * 2, dtype=torch.uint8, device=device)
    return buf, buf[:off].view(torch.int32).view(n, k // 8), buf[off:].view(torch.bfloat16).view(n, k // GROUP)


def repack(w: torch.Tensor, s: torch.Tensor) -> None:
    """In place, natural W4 (words [N, K/8]: nibble e of word j = column 8j + e; scales [N, K/128]) -> the layout of
    w4g128_gemv.cuh: per (16-row tile, 128-column group) one 1 KB unit [row g, 16 B chunk t][row g + 8, chunk t], the
    units of a tile contiguous; inside a word nibble j / j + 4 = columns 2j / 2j + 1; one u32 scale per (tile, group, g)
    = bf16(row g) | bf16(row g + 8) << 16."""
    rows, k8 = w.shape
    nkg = k8 // 16
    if rows % TILE or k8 % 16:
        raise ValueError(f"W4 repack needs N % 16 == 0 and K % 128 == 0, got [{rows}, {k8 * 8}]")
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=w.device)
    for r0 in range(0, rows, _ROWS):
        n = min(_ROWS, rows - r0)
        nt = n // TILE
        wc = w[r0 : r0 + n].view(nt, 2, 8, nkg, 4, 4)  # [tile, half, g, kg, t, u]
        nib = ((wc.to(torch.int64).unsqueeze(-1) >> sh) & 0xF)[..., order]
        word = (nib << sh).sum(-1)  # < 2**32
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        w[r0 : r0 + n].copy_(word.permute(0, 3, 1, 2, 4, 5).reshape(n, k8))  # [tile, kg, half, g, t, u]
        sc = s[r0 : r0 + n].view(nt, 2, 8, nkg).permute(0, 3, 2, 1).contiguous()  # [tile, kg, g, half]
        s[r0 : r0 + n].copy_(sc.view(n, nkg))


def _natural(packer, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = packer.weights[name]
    if d.scheme != "w4g128":
        raise ValueError(f"{name}: a w4g128 pack of a {d.scheme} weight")
    raw = packer.source.raw(d.source)
    n, k = d.shape
    if tuple(raw["packed"].shape) != (n, k // 8) or tuple(raw["scale"].shape) != (n, k // GROUP):
        raise ValueError(f"{name}: W4G128 tensors {tuple(raw['packed'].shape)} / {tuple(raw['scale'].shape)}")
    return raw["packed"], raw["scale"]


def _pack_rows(packer, pack: Pack) -> torch.Tensor:
    """w4g / w4g_cat: the parts' rows stacked in order (segments of one launch)."""
    k = packer.weights[pack.parts[0]].shape[1]
    n = sum(packer.weights[p].shape[0] for p in pack.parts)
    buf, w, s = _empty(n, k, packer.device)
    r = 0
    for p in pack.parts:
        words, scale = _natural(packer, p)
        if words.shape[1] * 8 != k:
            raise ValueError(f"pack {pack.name}: {p} has K {words.shape[1] * 8}, not {k}")
        w[r : r + words.shape[0]].copy_(words)
        s[r : r + words.shape[0]].copy_(scale)
        r += words.shape[0]
    repack(w, s)
    return buf


def _pack_gateup(packer, pack: Pack) -> torch.Tensor:
    """w4g_gateup: gate and up interleaved per 8 rows (a 16-row tile = 8 gate rows | the 8 matching up rows)."""
    (gw, gs), (uw, us) = (_natural(packer, p) for p in pack.parts)
    n, k8 = gw.shape
    if gw.shape != uw.shape or n % 8:
        raise ValueError(f"pack {pack.name}: gate / up shapes differ or rows are not a multiple of 8")
    buf, w, s = _empty(2 * n, k8 * 8, packer.device)
    for dst, g, u in ((w, gw, uw), (s, gs, us)):
        v = dst.view(n // 8, 2, 8, dst.shape[1])
        v[:, 0].copy_(g.view(n // 8, 8, -1))
        v[:, 1].copy_(u.view(n // 8, 8, -1))
    repack(w, s)
    return buf


PACKS = {"w4g": _pack_rows, "w4g_cat": _pack_rows, "w4g_gateup": _pack_gateup}


# ------------------------------------------------------------------------------------------------------ launchers


def _launch(L: C.CDLL, mem, t, T: int, st: int, kind: int) -> None:
    n, k = t.attrs["N"], t.attrs["K"]
    outs, ends = (C.c_void_p * 3)(), (C.c_int * 3)()
    if kind == EPI_SEGS:
        for j in range(len(t.attrs["ends"])):
            outs[j] = mem.buffer(t.outs[f"o{j}"])
        for j in range(3):
            ends[j] = t.attrs["ends"][min(j, len(t.attrs["ends"]) - 1)]
    else:
        outs[0] = mem.buffer(t.outs["out"])
    w = mem.packs[t.weights["w"]]
    rc = L.warpc_w4g128_gemv(
        w, w + scale_offset(n, k), mem.buffer(t.ins["x"]), n, k, T, kind, ends, outs, w4_ks(n, k), st
    )
    if rc != 0:
        raise RuntimeError(f"{t.kernel} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


LAUNCHERS = {
    "g_w4_gemv": lambda L, mem, t, T, st: _launch(L, mem, t, T, st, EPI_SEGS),
    "g_w4_gemv_silu": lambda L, mem, t, T, st: _launch(L, mem, t, T, st, EPI_SILU),
}


# ---------------------------------------------------------------------------------------------------- capabilities


def _w(A, i: int):
    return A.graph.weights[A.node(i).attrs["weight"]]


def _not_head(m, A):
    for i in m.anchors:
        if A.consumers(A.node(i).outs[0], "cast"):
            return "a linear feeding a cast is the lm_head: g_head's, not a bf16 GEMV"
    return None


def _consts(m, A):
    ns = [_w(A, i).shape[0] for i in m.anchors]
    return {
        "N": sum(ns),
        "K": _w(A, m.anchors[0]).shape[1],
        "T": A.rows_max,
        "ends": math.gcd(*itertools.accumulate(ns)),  # every segment end a multiple of 16
        "dtype": A.graph.types[A.node(m.anchors[0]).ins[0]].dtype,
    }


def _emit(m, A):
    lins = [A.node(i) for i in m.anchors]
    ws = [n.attrs["weight"] for n in lins]
    ns = [A.graph.weights[w].shape[0] for w in ws]
    name = "|".join(ws)
    return Region(
        "g_w4_gemv",
        set(m.covers),
        ins={"x": lins[0].ins[0]},
        outs={f"o{j}": n.outs[0] for j, n in enumerate(lins)},
        weights={"w": name},
        packs=[Pack(name, "w4g_cat" if len(ws) > 1 else "w4g", ws)],
        attrs={"N": sum(ns), "K": A.graph.weights[ws[0]].shape[1], "ends": list(itertools.accumulate(ns))},
        numerics=NUM_W4 + "; output segments by row range",
    )


def _silu_consts(m, A):
    g = _w(A, m.anchors[0])
    return {
        "N": g.shape[0],
        "K": g.shape[1],
        "T": A.rows_max,
        "dtype": A.graph.types[A.node(m.anchors[0]).ins[0]].dtype,
    }


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
        "g_w4_gemv_silu",
        set(m.covers),
        ins={"x": lg.ins[0]},
        outs={"out": A.node(m.pair).outs[0]},
        weights={"w": name},
        packs=[Pack(name, "w4g_gateup", [wg, wu])],
        attrs={"N": 2 * n, "K": k},
        numerics=NUM_W4 + "; gate|up interleaved per 8 rows, bf16(bf16(silu(bf16(gate))) * bf16(up))",
    )


CAPS = [
    Cap(
        "g_w4_gemv_silu",
        "linear",
        anchor_schemes=("w4g128",),
        pair=("silu_mul",),
        limits={"N_multiple": 8, "K_multiple": GROUP, "T_max": schedule.STEP_ROWS, "dtype_in": ("bf16",)},
        rounds={"linear": ("out",), "silu_mul": ("silu", "out")},
        constants=_silu_consts,
        check=_silu_check,
        emit=_emit_silu,
    ),
    Cap(
        "g_w4_gemv",
        "linear",
        anchor_schemes=("w4g128",),
        segments=3,
        limits={
            "N_multiple": TILE,
            "K_multiple": GROUP,
            "T_max": schedule.STEP_ROWS,
            "dtype_in": ("bf16",),
            "ends_multiple": TILE,
        },
        rounds={"linear": ("out",)},
        constants=_consts,
        check=_not_head,
        emit=_emit,
    ),
]
