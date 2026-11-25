"""Window caps of the swiftmk table: the hand engine's fast prefill (engine/prefill.py `FastPrefill`) as launchers.

A window is a step of up to 4,096 rows executed eagerly, not captured. Its Plan is the decode graph lowered a second
time at rows_max = N (17 to 4,096) against a table that adds these caps: the decode caps fall out of it by their
row limit (T_max 8 or 16), these fall out of the decode plan by T_min. Each window cap is a decode cap of this table
with a window launcher: the same match, operands, packs (the repacked W4 layout every kernel of the backend reads;
a window GEMM dequantises it to a bf16 scratch, never a second persistent copy) and states, so the window plan shares
packs, states and cursors with the decode plan.

What a launcher runs, and what makes it equal to `FastPrefill(fold=False)` bit for bit (the gate of the stage):

  w_w4_gemv        windows of at most W4_MAX rows: smkp_w4r_gemm on the repacked weights (the decode kernel's
                   accumulation, row by row, in passes of the cheapest row count); larger: smkp_dequant_w4r +
                   torch.mm (cuBLAS, fp32 accumulate, bf16 out) over the whole window, the MLP's down projection in
                   blocks of MLP_ROWS rows
  w_w4_gemv_silu   the same split; larger: smkp_dequant_w4r_gu + the CUTLASS DualGemm gate|up + SiLU*mul in MLP_ROWS
                   blocks
  w_ab_gemv        torch.mm on the bf16 a|b weights
  w_deltanet       smkp_gates, smkp_conv, fla's chunk_gated_delta_rule carrying the fp32 state of ring slot `cur`,
                   smkp_gnorm
  w_attn           smkp_gather_kv (history), smkp_attn_prep (norm, rope, KV write), FlashAttention-2 over token-major
                   bf16 K / V, smkp_attn_gate
  w_rmsnorm(_add)  smk_rmsnorm (the decode launcher: it takes any row count)
  s_embed          smk_embed (the decode launcher, no row limit)

Preconditions the Runtime guarantees (they are not checked per launch): the DeltaNet ring is materialised (`pend` 0)
and a window leaves `cur` where it is; `pos` holds the window's first position. fla and vllm (FlashAttention-2) are
optional libraries (`warpc[prefill]`), imported when the first window is scheduled.
"""

from __future__ import annotations

import ctypes as C
import dataclasses
import functools
import importlib.util
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from ..compiler import Cap, CapsTable, Task
from ..runtime import Memory
from . import table as tbl
from .launch import Enqueue, _embed, _rmsnorm
from .lib import EPI_SEGS, EPI_SILU, KV_DTYPE, Epi
from .pack import w4_scale_offset

if TYPE_CHECKING:
    from .executor import SwiftmkExecutor

WINDOW_ROWS_MAX = 4096  # most rows of a window (the hand's `seg`)
WINDOW_ROWS_MIN = 17  # a plan of fewer rows is the decode plan's (its GEMV and norm caps take up to 16 rows)
W4_MAX = 240  # windows of at most this many rows use the W4 mma kernel
MLP_ROWS = 2048  # the MLP runs in blocks of this many rows (the hand's `rows`), so M of its GEMMs never exceeds it
KV_HEADS, HD = 4, 256
# ms of one weight pass at 8 * NT rows of the W4 mma kernel (engine/prefill.py W4_PASS_MS)
_PASS_MS = (16.2, 19.5, 23.2, 29.1, 33.9, 40.5, 48.3, 66.0)

LIBRARIES = {"fla": "flash-linear-attention", "vllm": "vllm"}


def require_libraries() -> None:
    """The window plan's optional libraries: fla (the chunked DeltaNet scan) and vllm (its FlashAttention-2)."""
    missing = [pip for mod, pip in LIBRARIES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        raise ImportError(
            f"the window prefill needs {' and '.join(missing)} (pip install warpc[prefill]); it cannot run without"
        )


# ---- the window table


def _emit_as(kernel: str, emit: Callable, **attrs) -> Callable:
    """The decode cap's emit with the window kernel name (and attrs the window launcher reads); the a|b side-stream mark
    is a decode overlap, not a window one."""

    def run(m, A):
        r = emit(m, A)
        kept = {k: v for k, v in r.attrs.items() if k != "side"}
        return dataclasses.replace(r, kernel=kernel, attrs={**kept, **attrs})

    return run


def _emit_w4(m, A):
    """A down projection reads the SwiGLU activation: its GEMM runs in MLP_ROWS blocks, as the DualGemm's did."""
    lin = A.node(m.anchors[0])
    prod = A.prod.get(lin.ins[0])
    mlp = prod is not None and A.node(prod).op == "silu_mul"
    return _emit_as("w_w4_gemv", tbl._emit_w4, mlp_down=bool(mlp))(m, A)


def _window(cap: Cap, kernel: str, emit: Callable | None = None, **limits) -> Cap:
    lim = {k: v for k, v in cap.limits.items() if k not in ("T_max", "slots_max")}
    lim |= {"T_min": WINDOW_ROWS_MIN, "T_max": WINDOW_ROWS_MAX, **limits}
    return dataclasses.replace(cap, kernel=kernel, limits=lim, emit=emit or _emit_as(kernel, cap.emit))


def window_caps(by_kernel: dict[str, Cap]) -> list[Cap]:
    c = by_kernel
    return [
        _window(c["s_attn"], "w_attn"),
        _window(c["s_deltanet"], "w_deltanet"),
        _window(c["s_w4_gemv_silu"], "w_w4_gemv_silu"),
        _window(c["s_w4_gemv"], "w_w4_gemv", _emit_w4),
        _window(c["s_ab_gemv"], "w_ab_gemv"),
        _window(c["s_rmsnorm_add"], "w_rmsnorm_add"),
        _window(c["s_rmsnorm"], "w_rmsnorm"),
    ]


def with_windows(caps: CapsTable) -> CapsTable:
    """`caps` plus the window caps (decode caps first: the cost model's tie-break order)."""
    extra = window_caps(caps.by_kernel)
    return CapsTable([*caps.caps, *extra], caps.target, caps.unfused_ctas, caps.unfused_launches, caps.prefer_rank)


# ---- device views

_ITEMSIZE = {torch.bfloat16: 2, torch.float32: 4, torch.int32: 4}


class _Raw:
    """A device range as a __cuda_array_interface__ (torch.as_tensor shares it)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


@functools.lru_cache(maxsize=8192)
def view(ptr: int, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """The tensor of `shape` over device memory at `ptr` (no copy, no ownership)."""
    nbytes = math.prod(shape) * _ITEMSIZE[dtype]
    return torch.as_tensor(_Raw(ptr, nbytes), device="cuda").view(dtype).view(shape)


def _scalar(ptr: int) -> int:
    return int(view(ptr, (1,), torch.int32).item())


@functools.cache
def _api(cdll: C.CDLL) -> SimpleNamespace:
    """The prefill entries of libswiftmk.so with their prototypes (engine/prefill.py setup_prefill_api)."""
    vp, ci, cf = C.c_void_p, C.c_int, C.c_float
    sig = {
        "smkp_dequant_w4r": [vp, vp, vp, ci, ci, ci, vp],
        "smkp_dequant_w4r_gu": [vp, vp, vp, ci, ci, vp],
        "smkp_w4r_gemm": [vp, vp, vp, ci, ci, ci, C.POINTER(Epi), vp],
        "smkp_swiglu_gemm": [vp, ci, vp, vp, vp, ci, ci, ci, vp],
        "smkp_gates": [vp, vp, vp, vp, vp, ci, vp],
        "smkp_conv": [vp, ci, vp, vp, vp, vp, vp, vp, ci, vp],
        "smkp_gnorm": [vp, vp, ci, vp, vp, ci, vp],
        "smkp_gather_kv": [vp, ci, ci, ci, cf, cf, vp, vp, vp],
        "smkp_attn_prep": [ci, ci, vp, ci, vp, ci, vp, ci, vp, vp, vp, ci, ci, cf, cf, vp, vp, vp, vp, vp],
        "smkp_attn_gate": [vp, vp, ci, vp, ci, vp],
    }
    out = {}
    for name, argtypes in sig.items():
        f = getattr(cdll, name)
        f.argtypes, f.restype = argtypes, ci
        out[name] = f
    return SimpleNamespace(**out)


def _libs():
    require_libraries()
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    return chunk_gated_delta_rule, flash_attn_varlen_func


@functools.lru_cache(maxsize=4)
def _cu(rows: int, keys: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cu_seqlens of one sequence of `rows` queries over `keys` keys."""
    return (
        torch.tensor([0, rows], dtype=torch.int32, device="cuda"),
        torch.tensor([0, keys], dtype=torch.int32, device="cuda"),
    )


@functools.cache
def _w4_plan(rows: int) -> tuple[int, ...]:
    """Row counts of the W4 mma calls that cover `rows` rows at the least total time (a pass streams the weights once
    whatever its row count): 40 -> (40,), 64 -> (40, 24), 100 -> (56, 44)."""
    best, choice = [0.0] * (rows + 1), [0] * (rows + 1)
    for r in range(1, rows + 1):
        best[r], choice[r] = min((best[max(0, r - 8 * nt)] + _PASS_MS[nt - 1], nt) for nt in range(1, 9))
    parts, r = [], rows
    while r > 0:
        parts.append(min(r, 8 * choice[r]))
        r -= 8 * choice[r]
    return tuple(parts)


def _cursor(ex: SwiftmkExecutor) -> int:
    """The DeltaNet ring slot the state lives in (a device scalar: read once per window, when it is scheduled)."""
    return _scalar(ex.cur_ptr)


# ---- GEMMs


def _w4_small(ex: SwiftmkExecutor, mem: Memory, task: Task, T: int, epi_of: Callable) -> Enqueue:
    """The W4 mma kernel over T <= W4_MAX rows in `_w4_plan` passes; epi_of(r, n): the epilogue of rows r .. r + n."""
    lib, api = ex.lib, _api(ex.lib.cdll)
    n, k = task.attrs["N"], task.attrs["K"]
    w = mem.packs[task.weights["w"]]
    x = mem.buffer(task.ins["x"])

    def run(st: int) -> None:
        r = 0
        for rows in _w4_plan(T):
            epi = epi_of(r, rows)
            lib.check(
                api.smkp_w4r_gemm(w, w + w4_scale_offset(n, k), x + r * k * 2, n, k, rows, C.byref(epi), st),
                task.kernel,
            )
            r += rows

    return run


def _w_w4(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    n, k, ends = t.attrs["N"], t.attrs["K"], t.attrs["ends"]
    starts = [0, *ends[:-1]]
    outs = [mem.buffer(t.outs[f"o{j}"]) for j in range(len(ends))]
    widths = [e - s for s, e in zip(starts, ends, strict=True)]
    if T <= W4_MAX:

        def epi_of(r: int, rows: int) -> Epi:
            epi = Epi()
            epi.kind = EPI_SEGS
            for j in range(3):
                epi.end[j] = ends[j] if j < len(ends) else n
                epi.out[j] = outs[j] + r * widths[j] * 2 if j < len(ends) else None
            return epi

        return _w4_small(ex, mem, t, T, epi_of)
    lib, api = ex.lib, _api(ex.lib.cdll)
    w = mem.packs[t.weights["w"]]
    dq = mem.scratch("w_dq", n * k * 2)
    seg = mem.scratch("w_seg", mem.rows_max * n * 2) if len(ends) > 1 else 0
    x = mem.buffer(t.ins["x"])
    block = MLP_ROWS if t.attrs.get("mlp_down") else T

    def run(st: int) -> None:
        lib.check(api.smkp_dequant_w4r(w, w + w4_scale_offset(n, k), dq, k, 0, n, st), t.kernel)
        weight = view(dq, (n, k)).t()
        for r0 in range(0, T, block):
            r1 = min(T, r0 + block)
            xin = view(x, (T, k))[r0:r1]
            if len(ends) == 1:
                torch.mm(xin, weight, out=view(outs[0], (T, n))[r0:r1])
                continue
            tmp = view(seg, (T, n))[r0:r1]
            torch.mm(xin, weight, out=tmp)
            for o, s, wd in zip(outs, starts, widths, strict=True):
                view(o, (T, wd))[r0:r1].copy_(tmp[:, s : s + wd])

    return run


def _w_swiglu(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    n, k = t.attrs["N"], t.attrs["K"]  # n = 2 * I rows, gate and up interleaved per 8
    inter = n // 2
    out = mem.buffer(t.outs["out"])
    if T <= W4_MAX:

        def epi_of(r: int, rows: int) -> Epi:
            epi = Epi()
            epi.kind = EPI_SILU
            epi.out[0] = out + r * inter * 2
            return epi

        return _w4_small(ex, mem, t, T, epi_of)
    lib, api = ex.lib, _api(ex.lib.cdll)
    w = mem.packs[t.weights["w"]]
    dq = mem.scratch("w_dq", n * k * 2)
    x = mem.buffer(t.ins["x"])

    def run(st: int) -> None:
        lib.check(api.smkp_dequant_w4r_gu(w, w + w4_scale_offset(n, k), dq, k, n, st), t.kernel)
        gate, up = dq, dq + inter * k * 2
        for r0 in range(0, T, MLP_ROWS):
            rows = min(MLP_ROWS, T - r0)
            lib.check(
                api.smkp_swiglu_gemm(x + r0 * k * 2, k, gate, up, out + r0 * inter * 2, rows, inter, k, st), t.kernel
            )

    return run


def _w_ab(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    na, nb, k = t.attrs["na"], t.attrs["N"] - t.attrs["na"], t.attrs["K"]
    w, x = mem.packs[t.weights["w"]], mem.buffer(t.ins["x"])
    o0, o1 = mem.buffer(t.outs["o0"]), mem.buffer(t.outs["o1"])
    tmp = mem.scratch("w_ab", mem.rows_max * (na + nb) * 2)

    def run(st: int) -> None:
        ab = view(tmp, (T, na + nb))
        torch.mm(view(x, (T, k)), view(w, (na + nb, k)).t(), out=ab)
        view(o0, (T, na)).copy_(ab[:, :na])
        view(o1, (T, nb)).copy_(ab[:, na:])

    return run


# ---- DeltaNet


def _w_deltanet(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    chunk_gated_delta_rule, _ = _libs()
    lib, api = ex.lib, _api(ex.lib.cdll)
    hv, dv, conv_c, conv_w = t.attrs["Hv"], t.attrs["dv"], 10240, 3
    cs = mem.states[t.states["conv"]] + _cursor(ex) * conv_c * conv_w * 2  # ring slot `cur`
    rec = mem.states[t.states["rec"]]  # one physical state slot, [Hv][dv][dk] fp32
    qkv, z, a, b = (mem.buffer(t.ins[r]) for r in ("qkv", "z", "a", "b"))
    conv_wt, a_log, dt_bias, norm_w = (mem.packs[t.weights[r]] for r in ("conv_w", "A_log", "dt_bias", "norm_w"))
    y = mem.buffer(t.outs["y"])
    rows = mem.rows_max
    ab = mem.scratch("w_ab", rows * hv * 2 * 2)
    g, beta = mem.scratch("w_g", rows * hv * 4), mem.scratch("w_beta", rows * hv * 4)
    prev = mem.scratch("w_conv_prev", conv_c * conv_w * 2)
    tmp = mem.scratch("w_tmp", rows * 12288 * 2)
    qc, kc, vc = (
        tmp,
        tmp + T * 2048 * 2,
        tmp + T * 4096 * 2,
    )  # q | k | v after the conv: [T][2048], [T][2048], [T][6144]

    def run(st: int) -> None:
        both = view(ab, (T, 2 * hv))
        torch.cat((view(a, (T, hv)), view(b, (T, hv))), dim=1, out=both)
        lib.check(api.smkp_gates(ab, a_log, dt_bias, g, beta, T, st), t.kernel)
        view(prev, (conv_c * conv_w,)).copy_(view(cs, (conv_c * conv_w,)))
        lib.check(api.smkp_conv(qkv, conv_c, prev, conv_wt, qc, kc, vc, cs, T, st), t.kernel)
        state = view(rec, (hv, dv, 128), torch.float32)
        o, final = chunk_gated_delta_rule(
            view(qc, (1, T, 16, 128)),
            view(kc, (1, T, 16, 128)),
            view(vc, (1, T, hv, dv)),
            view(g, (1, T, hv), torch.float32),
            view(beta, (1, T, hv), torch.float32),
            scale=128**-0.5,
            initial_state=state.view(1, hv, dv, 128),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            state_v_first=True,
        )
        state.copy_(final.view(hv, dv, 128))
        lib.check(api.smkp_gnorm(o.data_ptr(), z, hv * dv, norm_w, y, T, st), t.kernel)

    return run


# ---- attention


def _w_attn(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    _, flash = _libs()
    lib, api = ex.lib, _api(ex.lib.cdll)
    ctx, cache = t.attrs["ctx"], t.attrs["cache"]
    dtype = KV_DTYPE[cache]
    ks, vs = (ex.scalars[t.weights[r]] if cache == "fp8" else 0.0 for r in ("k_scale", "v_scale"))
    kv = mem.states[t.states["kv"]]
    q, k, v = (mem.buffer(t.ins[r]) for r in ("q", "k", "v"))
    q_norm, k_norm = mem.packs[t.weights["q_norm"]], mem.packs[t.weights["k_norm"]]
    y = mem.buffer(t.outs["y"])
    kall = mem.scratch("w_kall", ctx * KV_HEADS * HD * 2)
    vall = mem.scratch("w_vall", ctx * KV_HEADS * HD * 2)
    tmp = mem.scratch("w_tmp", mem.rows_max * 12288 * 2)
    qatt, out = tmp, tmp + T * 6144 * 2
    pos0 = _scalar(mem.pos)
    total = pos0 + T
    cuq, cuk = _cu(T, total)
    ldq, ldk = 12288, 1024  # the q projection [T][24][q 256 | gate 256] and the k, v projections, one buffer each

    def run(st: int) -> None:
        if pos0:
            lib.check(api.smkp_gather_kv(kv, ctx, dtype, pos0, ks, vs, kall, vall, st), t.kernel)
        lib.check(
            api.smkp_attn_prep(
                T, pos0, q, ldq, k, ldk, v, ldk, q_norm, k_norm, kv, ctx, dtype, ks, vs, qatt, kall, vall, None, st
            ),
            t.kernel,
        )
        flash(
            view(qatt, (T, 24, HD)),
            view(kall, (ctx, KV_HEADS, HD))[:total],
            view(vall, (ctx, KV_HEADS, HD))[:total],
            max_seqlen_q=T,
            cu_seqlens_q=cuq,
            max_seqlen_k=total,
            cu_seqlens_k=cuk,
            causal=True,
            softmax_scale=1.0 / 16,
            fa_version=2,
            out=view(out, (T, 24, HD)),
        )
        lib.check(api.smkp_attn_gate(out, q, ldq, y, T, st), t.kernel)

    return run


WINDOW_LAUNCHERS: dict[str, Callable[[SwiftmkExecutor, Memory, Task, int], Enqueue]] = {
    "w_rmsnorm": _rmsnorm,
    "w_rmsnorm_add": _rmsnorm,
    "w_w4_gemv": _w_w4,
    "w_w4_gemv_silu": _w_swiglu,
    "w_ab_gemv": _w_ab,
    "w_deltanet": _w_deltanet,
    "w_attn": _w_attn,
    "s_embed": _embed,
}
