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
from . import attn_gated, deltanet, head_refine, hybrid, inkling, schedule, w4


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
    elif mode in (1, 4):
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
        mem.buffer(t.ins["row"]) if mode == 4 else None,
        a.get("act", 0),
        C.c_float(a.get("cap", 0.0)),
        0,
        schedule.gemv_k_split(a["K"], mode, "nw" in t.weights),
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
    rc = L.warpc_embed(table, tok, x, T, t.attrs["H"], C.c_float(t.attrs.get("scale", 1.0)), st)
    _check(L, rc, t.kernel)


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


def _rmsnorm_post(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    rc = L.warpc_rmsnorm_post(
        mem.buffer(t.ins["h"]),  # the residual stream, updated in place
        mem.buffer(t.ins["x"]),
        mem.packs[t.weights["w"]],
        mem.packs[t.weights["s"]] if "s" in t.weights else None,
        T,
        t.attrs["H"],
        C.c_float(t.attrs["eps"]),
        st,
    )
    _check(L, rc, t.kernel)


def _ple(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    a = t.attrs
    outs = (C.c_void_p * a["L"])(*(mem.buffer(t.outs[f"p{j}"]) for j in range(a["L"])))  # read during the call
    rc = L.warpc_ple(
        mem.buffer(t.ins["ctx"]),
        mem.buffer(t.ins["tok"]),
        mem.packs[t.weights["table"]],
        mem.packs[t.weights["w"]],
        outs,
        T,
        a["L"],
        a["D"],
        C.c_float(a["ctx_scale"]),
        C.c_float(a["out_scale"]),
        C.c_float(a["eps"]),
        C.c_float(a["scale"]),
        st,
    )
    _check(L, rc, t.kernel)


def _attention(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    a = t.attrs
    read = t.kernel == "g_attn_read"  # the cache is another layer's: q-only prep, no KV write
    hq, hkv, hd, rd, ctx, nc = a["Hq"], a["Hkv"], a["hd"], a["rd"], a["ctx"], a["nc"]
    scale, qk, vn, window = C.c_float(a["scale"]), int(a["qk_norm"]), int(a.get("v_norm", False)), a.get("window") or 0
    inv_freq = mem.constants[t.constants["inv_freq"]]
    eps = C.c_float(a.get("eps", 0.0))  # read only with a norm
    q, y = mem.buffer(t.ins["q"]), mem.buffer(t.outs["y"])
    k, v = (None, None) if read else (mem.buffer(t.ins["k"]), mem.buffer(t.ins["v"]))
    qn = mem.packs[t.weights["q_norm"]] if qk else None
    kn = mem.packs[t.weights["k_norm"]] if qk and not read else None
    kv, pos = mem.states[t.states["kv"]], mem.pos
    fmt = FORMATS[a["cache"]]
    rows = ctx if fmt.quantized else schedule.kv_rows(ctx)  # the state's rows as allocated (CudaExecutor.state_nbytes)
    # the workspace holds the partials of nc = 1
    keys = schedule.attn_splits(rows, 1, window or None) * schedule.ATTN_CHUNK
    ws = mem.scratch("attn_ws", L.warpc_attn_ws_bytes(mem.rows_max, hq, hd, keys, schedule.ATTN_CHUNK))

    def check(rc: int) -> None:
        _check(L, rc, t.kernel)

    if not fmt.quantized and T == 1 and nc == 1 and hd == 128 and rd == 128 and not (read or window or vn):
        # one row: the prep folded into the split kernel, then the merge (two launches instead of three, the same bits)
        rope = (rd, inv_freq, eps, rows, qk)
        check(L.warpc_attn_decode_fused(q, k, v, qn, kn, kv, pos, y, ws, hq, hkv, hd, *rope, scale, st))
        return
    one_row = schedule.attn_splits(rows, 1, window or None, schedule.ATTN_ONE_ROW_CHUNK)
    if not fmt.quantized and T == 1 and hd in (256, 512) and rd == hd and one_row <= schedule.ATTN_MAX_SPLITS_CHUNKED:
        # one row, head dim 256 / 512: prep, split and merge in one launch (the merge a second one on a long context)
        keys = one_row * schedule.ATTN_ONE_ROW_CHUNK
        ws = mem.scratch("attn_ws", L.warpc_attn_ws_bytes(1, hq, hd, keys, schedule.ATTN_ONE_ROW_CHUNK))
        cnt = mem.scratch("attn_cnt", 4 * hkv)  # one ticket counter per KV head, left zeroed by the kernel
        args = (q, k, v, qn, kn, kv, pos, y, ws, cnt, hq, hkv, hd, rd, inv_freq, eps, rows, qk, vn, scale, window)
        check(L.warpc_attn_decode1(*args, int(read), st))
        return
    qbuf = mem.scratch("attn_q", mem.rows_max * hq * hd * 2)
    if not fmt.quantized:
        # Hkv = 0: the q heads only
        prep = (T, q, k, v, qn, kn, qbuf, kv, pos, hq, 0 if read else hkv, hd, rd, inv_freq, eps, rows, qk, vn, st)
        check(L.warpc_attn_prep(*prep))
        if nc == 1:
            check(L.warpc_attn_decode(T, qbuf, kv, pos, y, ws, hq, hkv, hd, rows, scale, window, st))
        else:
            check(L.warpc_attn_decode_x(T, qbuf, kv, pos, y, ws, hq, hkv, hd, rows, scale, nc, window, st))
        return
    # read on the host during the call: need not outlive it
    meta = (C.c_longlong * 13)(*schedule.kv_meta(fmt, ctx, hkv, hd, mem.rows_max))
    check(L.warpc_attn_prep_q(T, q, k, v, qn, kn, qbuf, kv, meta, pos, hq, hkv, hd, rd, inv_freq, eps, ctx, qk, st))
    check(L.warpc_attn_decode_q(T, qbuf, kv, meta, pos, y, ws, hq, hkv, hd, ctx, scale, nc, st))


def _head(L: C.CDLL, mem: Memory, t: Task, T: int, st: int) -> None:
    n, k = t.attrs["N"], t.attrs["K"]
    if T == 1 and n % 4 == 0 and k % 256 == 0:
        # one row: the CUDA-core head GEMV whose fp32 order is libswiftmk's (smk_bf16_gemv_f32), bit-identical logits
        rc = L.warpc_head_gemv(
            mem.packs[t.weights["w"]], mem.buffer(t.ins["x"]), n, k, mem.buffer(t.outs["logits"]), st
        )
        _check(L, rc, t.kernel)
    else:
        _gemv(L, mem, t, T, st, 2)
    ws = mem.scratch("argmax_ws", L.warpc_argmax_ws_bytes())
    logits, ids = mem.buffer(t.outs["logits"]), mem.buffer(t.outs["ids"])
    _check(L, L.warpc_argmax(logits, T, t.attrs["N"], ids, ws, st), t.kernel)


LAUNCHERS: dict[str, Callable[[C.CDLL, Memory, Task, int, int], None]] = {
    "g_embed": _embed,
    "g_rmsnorm": _rmsnorm,
    "g_rmsnorm_add": _rmsnorm,
    "g_rmsnorm_post": _rmsnorm_post,
    "g_gemv": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 0),
    "g_gemv_dual": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 1),
    "g_gemv_row": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 4),
    "g_gemv_resadd": lambda L, mem, t, T, st: _gemv(L, mem, t, T, st, 3),
    "g_q4gemv": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 0),
    "g_q4gemv_dual": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 1),
    "g_q4gemv_resadd": lambda L, mem, t, T, st: _q4gemv(L, mem, t, T, st, 3),
    "g_attn": _attention,
    "g_attn_read": _attention,
    "g_head": _head,
    "g_ple": _ple,
}
for _family in (inkling, hybrid, w4, attn_gated, deltanet, head_refine):
    LAUNCHERS.update(_family.LAUNCHERS)


# L2 prefetch of the GEMV that consumes an attention's output (a residual GEMV, or the o projection of a layer that
# norms it before the add: Gemma) while the attention before it leaves DRAM idle. Measured on Qwen3-0.6B
# decode: +2.6% tok/s for bf16 weights, +0.2% (noise) for a Q4_0 o_proj, so bf16 consumers only. Half of the consumer,
# at most 4 MiB (more slows the attention down more than it saves), issued over (3 + MiB) us from 4 blocks; a consumer
# above 16 MiB gains nothing from 4 MiB and is skipped. Gemma 4 E2B, 3 MiB of o beside the attention: -0.89% per token,
# the best of the five boundaries tried there (the others lost).
def prefetch_bytes(producer: Task, consumer: Task) -> int:
    """Bytes of `consumer`'s weights to prefetch beside `producer` in a single-row step, 0 for none."""
    if producer.kernel not in ("g_attn", "g_attn_read"):
        return 0
    if consumer.kernel != "g_gemv_resadd" and not (
        consumer.kernel == "g_gemv"
        and consumer.ins.get("x") == producer.outs["y"]
        and consumer.attrs["K"] == producer.attrs["Hq"] * producer.attrs["hd"]
    ):
        return 0
    total = consumer.attrs["N"] * consumer.attrs["K"] * 2
    return 0 if total > 16 << 20 else min(4 << 20, total // 2) & ~127


def prefetch(L: C.CDLL, mem: Memory, consumer: Task, nbytes: int, stream: int) -> None:
    spread_ns = 3000 + 1000 * round(nbytes / (1 << 20))
    rc = L.warpc_prefetch(mem.packs[consumer.weights["w"]], nbytes, 4, spread_ns, stream)
    _check(L, rc, f"prefetch for {consumer.kernel}")
