"""Per-kernel launchers: one task of a swiftmk plan -> a callable that enqueues its libswiftmk calls on a stream.

A launcher resolves every pointer when the step is scheduled (before the warm-up and before each capture), so the
returned callable only makes the C calls.
"""

from __future__ import annotations

import ctypes as C
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..compiler import Task
from ..runtime import Memory
from .lib import EPI_F32, EPI_SEGS, EPI_SILU, HEADQ_I4R, KV_DTYPE, Epi
from .pack import w4_scale_offset
from .table import HEAD_R

if TYPE_CHECKING:
    from .executor import SwiftmkExecutor

Enqueue = Callable[[int], None]

RECURRENT_SLOT_BYTES = (
    48 * 128 * 128 * 4
)  # deltanet.h SWIFTMK_DN_STATE_SLOT_FLOATS: one state slot, the delta buffer after it


def _embed(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib = ex.lib
    args = (mem.packs[t.weights["table"]], mem.buffer(t.ins["tok"]), mem.buffer(t.outs["x"]), T, t.attrs["H"])
    return lambda st: lib.check(lib.smk_embed(*args, st), t.kernel)


def _rmsnorm(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib = ex.lib
    y = mem.buffer(t.ins["y"]) if "y" in t.ins else None
    w, xn = mem.packs[t.weights["w"]], mem.buffer(t.outs["xn"])
    args = (mem.buffer(t.ins["h"]), y, w, xn, None, T, t.attrs["H"], C.c_float(t.attrs["eps"]))
    return lambda st: lib.check(lib.smk_rmsnorm(*args, st), t.kernel)


def _w4(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int, epi: Epi) -> Enqueue:
    lib = ex.lib
    n, k = t.attrs["N"], t.attrs["K"]
    w = mem.packs[t.weights["w"]]
    args = (w, w + w4_scale_offset(n, k), mem.buffer(t.ins["x"]), n, k, T, C.byref(epi))
    return lambda st: lib.check(lib.smk_w4r_gemv(*args, st), t.kernel)


def _w4_segments(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    ends = t.attrs["ends"]
    epi = Epi()
    epi.kind = EPI_SEGS
    for j in range(3):
        epi.end[j] = ends[j] if j < len(ends) else t.attrs["N"]
        epi.out[j] = mem.buffer(t.outs[f"o{j}"]) if j < len(ends) else None
    return _w4(ex, mem, t, T, epi)


def _w4_silu(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    epi = Epi()
    epi.kind = EPI_SILU
    epi.out[0] = mem.buffer(t.outs["out"])
    return _w4(ex, mem, t, T, epi)


def _ab(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib, a = ex.lib, t.attrs
    o0, o1 = mem.buffer(t.outs["o0"]), mem.buffer(t.outs["o1"])
    args = (mem.packs[t.weights["w"]], mem.buffer(t.ins["x"]), a["N"], a["na"], a["K"], T, o0, o1)
    return lambda st: lib.check(lib.smk_bf16_gemv_small(*args, st), t.kernel)


def _deltanet(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib = ex.lib
    rec = mem.states[t.states["rec"]]
    ws = mem.scratch("deltanet_ws", lib.swiftmk_deltanet_ws_bytes())  # zeroed once; the kernel leaves it zeroed
    ins = [mem.buffer(t.ins[r]) for r in ("qkv", "z", "a", "b")]
    weights = [mem.packs[t.weights[r]] for r in ("conv_w", "A_log", "dt_bias", "norm_w")]
    args = (
        T,
        *ins,
        *weights,
        rec,
        mem.states[t.states["conv"]],
        ex.cur_ptr,
        ex.slots,
        1,  # one physical state slot: exact for the lazy entry (deltanet.h swiftmk_deltanet_step_lazy2)
        mem.buffer(t.outs["y"]),
        ws,
        rec + RECURRENT_SLOT_BYTES,
        ex.pend_ptr,
        None,
    )
    return lambda st: lib.check(lib.swiftmk_deltanet_step_lazy2(*args, st), t.kernel)


def _attention(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib, cache = ex.lib, t.attrs["cache"]
    ws = mem.scratch("attn_ws", lib.swiftmk_attn_ws_bytes())
    ks, vs = (ex.scalars[t.weights[r]] if cache == "fp8" else 0.0 for r in ("k_scale", "v_scale"))
    args = (
        T,
        mem.buffer(t.ins["q"]),
        mem.buffer(t.ins["k"]),
        mem.buffer(t.ins["v"]),
        mem.packs[t.weights["q_norm"]],
        mem.packs[t.weights["k_norm"]],
        mem.states[t.states["kv"]],
        t.attrs["ctx"],
        KV_DTYPE[cache],
        C.c_float(ks),
        C.c_float(vs),
        mem.pos,
        None,  # no MRoPE offset: text positions (attn.h rdelta)
        mem.buffer(t.outs["y"]),
        ws,
    )
    return lambda st: lib.check(lib.swiftmk_attn_step(*args, st), t.kernel)


def _w4_head(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib = ex.lib
    n, k = t.attrs["N"], t.attrs["K"]
    w = mem.packs[t.weights["w"]]
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    epi = Epi()
    epi.kind = EPI_F32
    epi.out[0] = logits
    args = (w, w + w4_scale_offset(n, k), mem.buffer(t.ins["x"]), n, k, T, C.byref(epi))
    ws = mem.scratch("argmax_ws", lib.smk_argmax_ws_bytes())  # zeroed once; the kernel re-zeroes it

    def run(st: int) -> None:
        lib.check(lib.smk_w4r_gemv(*args, st), t.kernel)
        lib.check(lib.smk_argmax(logits, T, n, ids, None, ws, st), t.kernel)

    return run


def _head(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib, n, k = ex.lib, t.attrs["N"], t.attrs["K"]
    w, x = mem.packs[t.weights["w"]], mem.buffer(t.ins["x"])
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    ws = mem.scratch("argmax_ws", lib.smk_argmax_ws_bytes())  # zeroed once; the kernel re-zeroes it

    def run(st: int) -> None:
        lib.check(lib.smk_bf16_gemv_f32(w, x, n, k, T, logits, st), t.kernel)
        lib.check(lib.smk_argmax(logits, T, n, ids, None, ws, st), t.kernel)

    return run


def _head_refine(ex: SwiftmkExecutor, mem: Memory, t: Task, T: int) -> Enqueue:
    lib, n, k = ex.lib, t.attrs["N"], t.attrs["K"]
    w, wq, x = mem.packs[t.weights["w"]], mem.packs[t.weights["wq"]], mem.buffer(t.ins["x"])
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    ws = mem.scratch("headq_ws", lib.smk_headq_ws_bytes())  # zeroed once (head_q.h)

    def run(st: int) -> None:
        lib.check(lib.smk_headq_gemv(HEADQ_I4R, wq, wq + w4_scale_offset(n, k), x, n, k, T, logits, st), t.kernel)
        lib.check(lib.smk_headq_select_rescore(logits, w, x, n, k, T, HEAD_R, ids, None, ws, st), t.kernel)

    return run


LAUNCHERS: dict[str, Callable[[SwiftmkExecutor, Memory, Task, int], Enqueue]] = {
    "s_embed": _embed,
    "s_rmsnorm": _rmsnorm,
    "s_rmsnorm_add": _rmsnorm,
    "s_w4_gemv": _w4_segments,
    "s_w4_gemv_silu": _w4_silu,
    "s_ab_gemv": _ab,
    "s_deltanet": _deltanet,
    "s_attn": _attention,
    "s_w4_head": _w4_head,
    "s_head": _head,
    "s_head_refine": _head_refine,
}
