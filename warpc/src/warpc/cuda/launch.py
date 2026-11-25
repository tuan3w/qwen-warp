"""Per-kernel launchers: one task of a plan -> the ctypes calls of its kernel on a stream.

A launcher takes the kernel library, reads the device pointers of one runtime through `Memory` and the shapes from the
task's attrs; it allocates nothing on the device after the eager warm-up, so the same calls can be captured into a CUDA
graph.
"""

from __future__ import annotations

import ctypes as C
from collections.abc import Callable

from ..compiler import Task
from ..kvcache import FORMATS
from ..runtime import Memory
from . import schedule


def _check(L: C.CDLL, rc: int, what: str) -> None:
    if rc != 0:
        raise RuntimeError(f"{what} failed: {L.warpc_error_string(rc).decode()} (CUDA error {rc})")


def _outputs(mem: Memory, t: Task, mode: int):
    outs, ends = (C.c_void_p * 3)(), (C.c_int * 3)()
    if mode == 0:
        segs = t.attrs["segs"]
        for j in range(len(segs)):
            outs[j] = mem.buffer(t.outs[f"o{j}"])
        for j in range(3):
            ends[j] = sum(segs[: j + 1])
    elif mode == 1:
        outs[0] = mem.buffer(t.outs["out"])
    elif mode == 2:
        outs[0] = mem.buffer(t.outs["logits"])
    else:
        outs[0] = mem.buffer(t.ins["h"])  # the residual stream, updated in place
    return outs, ends


def _gemv(L: C.CDLL, mem: Memory, t: Task, T: int, st: int, mode: int) -> None:
    a = t.attrs
    outs, ends = _outputs(mem, t, mode)
    rc = L.warpc_gemv(
        mem.packs[t.weights["w"]],
        mem.packs[t.weights["w2"]] if mode == 1 else None,
        mem.buffer(t.ins["x"]),
        a["N"],
        a["K"],
        T,
        mode,
        ends,
        outs,
        mem.packs[t.weights["nw"]] if "nw" in t.weights else None,
        C.c_float(a.get("eps", 0.0)),
        schedule.gemv_rows_per_warp(a["N"], a["K"], mode),
        st,
    )
    _check(L, rc, t.kernel)


def _q4gemv(L: C.CDLL, mem: Memory, t: Task, T: int, st: int, mode: int) -> None:
    a = t.attrs
    outs, ends = _outputs(mem, t, mode)
    rc = L.warpc_q4_gemv(
        mem.packs[t.weights["w"]],
        mem.packs[t.weights["w2"]] if mode == 1 else None,
        mem.buffer(t.ins["x"]),
        a["N"],
        a["K"],
        T,
        mode,
        ends,
        outs,
        schedule.gemv_rows_per_warp(a["N"], a["K"], mode),
        st,
    )
    _check(L, rc, t.kernel)


def _embed(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    table, tok, x = mem.packs[t.weights["table"]], mem.buffer(t.ins["tok"]), mem.buffer(t.outs["x"])
    _check(L, L.warpc_embed(table, tok, x, T, t.attrs["H"], st), t.kernel)


def _rmsnorm(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    add = t.kernel == "g_rmsnorm_add"
    rc = L.warpc_rmsnorm(
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


def _attention(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    a = t.attrs
    hq, hkv, hd, rd, ctx, nc = a["Hq"], a["Hkv"], a["hd"], a["rd"], a["ctx"], a["nc"]
    scale, qk = C.c_float(a["scale"]), int(a["qk_norm"])
    inv_freq = mem.constants[t.constants["inv_freq"]]
    eps = C.c_float(a.get("eps", 0.0))  # read only with qk-norm
    q, k, v, y = mem.buffer(t.ins["q"]), mem.buffer(t.ins["k"]), mem.buffer(t.ins["v"]), mem.buffer(t.outs["y"])
    qn = mem.packs[t.weights["q_norm"]] if qk else None
    kn = mem.packs[t.weights["k_norm"]] if qk else None
    kv, pos = mem.states[t.states["kv"]], mem.pos
    ws = mem.scratch("attn_ws", L.warpc_attn_ws_bytes(mem.rows_max, hq, hd, ctx, schedule.ATTN_CHUNK))
    fmt = FORMATS[a["cache"]]

    def check(rc: int) -> None:
        _check(L, rc, t.kernel)

    if not fmt.quantized and T == 1 and nc == 1 and hd == 128 and rd == 128:
        # one row: the prep folded into the split kernel, then the merge (two launches instead of three, the same bits)
        rope = (rd, inv_freq, eps, ctx, qk)
        check(L.warpc_attn_decode_fused(q, k, v, qn, kn, kv, pos, y, ws, hq, hkv, hd, *rope, scale, st))
        return
    qbuf = mem.scratch("attn_q", mem.rows_max * hq * hd * 2)
    if not fmt.quantized:
        check(L.warpc_attn_prep(T, q, k, v, qn, kn, qbuf, kv, pos, hq, hkv, hd, rd, inv_freq, eps, ctx, qk, st))
        if nc == 1:
            check(L.warpc_attn_decode(T, qbuf, kv, pos, y, ws, hq, hkv, hd, ctx, scale, st))
        else:
            check(L.warpc_attn_decode_x(T, qbuf, kv, pos, y, ws, hq, hkv, hd, ctx, scale, nc, st))
        return
    # read on the host during the call: need not outlive it
    meta = (C.c_longlong * 13)(*schedule.kv_meta(fmt, ctx, hkv, hd, mem.rows_max))
    check(L.warpc_attn_prep_q(T, q, k, v, qn, kn, qbuf, kv, meta, pos, hq, hkv, hd, rd, inv_freq, eps, ctx, qk, st))
    check(L.warpc_attn_decode_q(T, qbuf, kv, meta, pos, y, ws, hq, hkv, hd, ctx, scale, nc, st))


def _head(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    _gemv(L, mem, t, T, st, 2)
    ws = mem.scratch("argmax_ws", L.warpc_argmax_ws_bytes())
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    _check(L, L.warpc_argmax(logits, T, t.attrs["N"], ids, ws, st), t.kernel)


LAUNCHERS: dict[str, Callable[[C.CDLL, Memory, Task, int, int], None]] = {
    "g_embed": _embed,
    "g_rmsnorm": _rmsnorm,
    "g_rmsnorm_add": _rmsnorm,
    "g_gemv": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 0),
    "g_gemv_dual": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 1),
    "g_gemv_resadd": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 3),
    "g_q4gemv": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 0),
    "g_q4gemv_dual": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 1),
    "g_q4gemv_resadd": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 3),
    "g_attn": _attention,
    "g_head": _head,
}


# L2 prefetch of a residual GEMV's weights while the attention before it leaves DRAM idle. Measured on Qwen3-0.6B
# decode: +2.6% tok/s for bf16 weights, +0.2% (noise) for a Q4_0 o_proj, so bf16 consumers only. Half of the consumer,
# at most 4 MiB (more slows the attention down more than it saves), issued over (3 + MiB) us from 4 blocks; a consumer
# above 16 MiB gains nothing from 4 MiB and is skipped.
def prefetch_bytes(producer: Task, consumer: Task) -> int:
    """Bytes of `consumer`'s weights to prefetch beside `producer` in a single-row step, 0 for none."""
    if producer.kernel != "g_attn" or consumer.kernel != "g_gemv_resadd":
        return 0
    total = consumer.attrs["N"] * consumer.attrs["K"] * 2
    return 0 if total > 16 << 20 else min(4 << 20, total // 2) & ~127


def prefetch(L: C.CDLL, mem: Memory, consumer: Task, nbytes: int, stream: int) -> None:
    spread_ns = 3000 + 1000 * round(nbytes / (1 << 20))
    rc = L.warpc_prefetch(mem.packs[consumer.weights["w"]], nbytes, 4, spread_ns, stream)
    _check(L, rc, f"prefetch for {consumer.kernel}")
