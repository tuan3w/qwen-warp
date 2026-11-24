"""SwiftmkExecutor: the runtime half of the swiftmk backend. It packs the weights in the hand kernels' layouts, sizes
the states in the hand engine's layouts, schedules the libswiftmk launches (the a|b GEMV beside the qkv|z GEMV, as the
hand engine overlaps them) and owns the DeltaNet ring cursors `cur` / `pend` every DeltaNet layer shares.

State layouts (bytes per layer):
  kv          swiftmk_attn_kv_bytes(ctx, kv_dtype): [2][4][ctx][256] bf16 / fp8, or int4_rot codes, meta, sink, ring
  conv        [rows_max + 1 ring slots][C][W - 1] bf16
  recurrent   one fp32 state [Hv][dv][dk] (the transpose of the graph's [Hv, dk, dv]) + the layer's delta buffer
              (swiftmk_deltanet_dbuf_floats() fp32)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from ..compiler import Pack, Plan, Task
from ..ir import Graph, StateDecl
from ..runtime import Launch, Memory
from ..sources import Source
from .launch import LAUNCHERS
from .lib import Library
from .pack import Packer

LAZY_ROWS = 16  # deltanet.h SWIFTMK_DN_LAZY_ROWS: rows of a layer's delta buffer


def ring_advance(cur: torch.Tensor, pend: torch.Tensor, T: int, slots: int) -> None:
    """The ring part of the hand engine's k_advance after a step of T rows (engine.cu k_advance, engine_impl.h
    dn_pend_after with a = T), as in-place torch ops on the int32 device scalars `cur` / `pend`: graph-capturable."""
    if T < 2:
        pend.zero_()
    else:
        pc, base, half = pend & 255, (pend >> 8) & 31, (pend >> 13) & 1
        base = torch.where(pc == 0, cur, base)
        wrap = (pc > 0) & (pc + T > LAZY_ROWS)
        pc = torch.where(wrap, 0, pc)
        base = torch.where(wrap, cur, base)
        half = torch.where(wrap, half ^ 1, half)
        pend.copy_((pc + T) | (base << 8) | (half << 13))
    cur.add_(T).remainder_(slots)


class SwiftmkExecutor:
    def __init__(self, lib: Library, graph: Graph, plan: Plan, source: Source):
        self.lib = lib
        self.slots = plan.rows_max + 1  # deltanet.h: a step of T rows needs T + 1 ring slots
        self.scalars: dict[str, float] = {}  # [1] dense packs (the fp8 KV scales) as the host floats the launch takes
        self._packer = Packer(graph.weights, source)
        self._cursors = torch.zeros(2, dtype=torch.int32, device="cuda")  # cur, pend (engine_impl.h DS_CUR, DS_PEND)
        self._cur, self._pend = self._cursors[0:1], self._cursors[1:2]
        self.cur_ptr, self.pend_ptr = self._cur.data_ptr(), self._pend.data_ptr()

    def state_nbytes(self, state: StateDecl, rows_max: int) -> int:
        if state.kind == "kv":
            return self.lib.kv_bytes(state.shape[1], state.cache)
        if state.kind == "conv":
            return (rows_max + 1) * math.prod(state.shape) * 2
        if state.kind == "recurrent":
            return math.prod(state.shape) * 4 + self.lib.swiftmk_deltanet_dbuf_floats() * 4
        raise ValueError(f"state {state.name}: kind {state.kind!r} has no swiftmk layout")

    def pack(self, pack: Pack) -> torch.Tensor:
        t = self._packer.build(pack)
        if pack.kind == "dense" and t.numel() == 1:
            self.scalars[pack.name] = float(t.float())
        return t

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        launches: list[Launch] = []
        prev: Task | None = None
        for t in tasks:
            fn = LAUNCHERS[t.kernel](self, mem, t, T)
            if t.kernel == "s_ab_gemv" and prev is not None and prev.kernel == "s_w4_gemv" and prev.ins == t.ins:
                # the hand engine's overlap (engine.cu): in_proj_a|b on the side stream beside the qkv|z GEMV, which
                # neither reads what a|b writes nor writes what it reads
                p_in, p_out = set(prev.ins.values()), set(prev.outs.values())
                s_in, s_out = set(t.ins.values()), set(t.outs.values())
                if p_in & s_out or p_out & (s_in | s_out):
                    raise AssertionError(f"{t.kernel} cannot run beside {prev.kernel}: they share a buffer")
                launches[-1] = Launch(launches[-1].main, (*launches[-1].side, fn))
            else:
                launches.append(Launch(fn))
            prev = t
        return launches

    def advance(self, mem: Memory, T: int, stream: int) -> None:
        if torch.cuda.current_stream().cuda_stream != stream:
            raise ValueError("advance runs on the current stream")
        ring_advance(self._cur, self._pend, T, self.slots)

    def reset(self, mem: Memory) -> None:
        self._cursors.zero_()
