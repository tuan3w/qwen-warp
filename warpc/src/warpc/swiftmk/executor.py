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
from collections.abc import Mapping, Sequence

import torch

from .. import pinned
from ..compiler import Pack, Plan, Task
from ..ir import Graph, StateDecl
from ..kvcache import FORMATS, RowSection
from ..runtime import Launch, Memory, merge_sides
from ..sources import Source
from .launch import LAUNCHERS, RECURRENT_SLOT_BYTES
from .lib import Library
from .pack import Packer
from .window import WINDOW_LAUNCHERS

LAZY_ROWS = 16  # deltanet.h SWIFTMK_DN_LAZY_ROWS: rows of a layer's delta buffer


def ring_commit(cur: torch.Tensor, pend: torch.Tensor, accepted: torch.Tensor, T: int, slots: int) -> None:
    """The ring part of the hand engine's accept step after a step of T rows of which the first `accepted` are kept
    (engine.cu k_advance / k_commit, engine_impl.h dn_pend_after), as in-place torch ops on the int32 device scalars
    `cur` / `pend`: graph-capturable. `accepted` is a device int32 [1] (a plain step passes T): the state is "slot
    `base` plus the first pc delta rows", and a rejected row sits beyond pc and is overwritten by the next step."""
    if T < 2:
        pend.zero_()
    else:
        pc, base, half = pend & 255, (pend >> 8) & 31, (pend >> 13) & 1
        base = torch.where(pc == 0, cur, base)
        wrap = (pc > 0) & (pc + T > LAZY_ROWS)
        pc = torch.where(wrap, 0, pc)
        base = torch.where(wrap, cur, base)
        half = torch.where(wrap, half ^ 1, half)
        pend.copy_((pc + accepted) | (base << 8) | (half << 13))
    cur.add_(accepted).remainder_(slots)


class SwiftmkExecutor:
    def __init__(self, lib: Library, graph: Graph, plan: Plan, source: Source):
        self.lib, self.graph = lib, graph
        self.slots = plan.rows_max + 1  # deltanet.h: a step of T rows needs T + 1 ring slots
        self.scalars: dict[str, float] = {}  # [1] dense packs (the fp8 KV scales) as the host floats the launch takes
        self._packer = Packer(graph.weights, source)
        self._host_packer = Packer(graph.weights, source, device="cpu")
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

    def pack(self, pack: Pack, *, host: bool = False) -> torch.Tensor:
        if host:
            return pinned.like(self._host_packer.build(pack))
        t = self._packer.build(pack)
        if pack.kind == "dense" and t.numel() == 1:
            self.scalars[pack.name] = float(t.float())
        return t

    def pack_nbytes(self, pack: Pack) -> int:
        return self._packer.nbytes(pack)

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        launchers = {**LAUNCHERS, **WINDOW_LAUNCHERS}
        return merge_sides(tasks, [Launch(launchers[t.kernel](self, mem, t, T)) for t in tasks])

    def advance(self, mem: Memory, T: int, accepted: torch.Tensor, stream: int) -> None:
        if torch.cuda.current_stream().cuda_stream != stream:
            raise ValueError("advance runs on the current stream")
        ring_commit(self._cur, self._pend, accepted, T, self.slots)

    def _aux(self, state: StateDecl) -> tuple[int, int] | None:
        _, ctx, nkv, hd = state.shape
        return FORMATS[state.cache].aux_range(ctx, nkv, hd, self.slots - 1)

    def materialize(self, states: Mapping[str, torch.Tensor]) -> None:
        """swiftmk_deltanet_materialize writes a layer's pending state to slot `cur` and clears `pend` for every layer
        of one call; a call per layer (the layers are separate tensors) restores `pend` first."""
        pend = self._pend.clone()
        stream = torch.cuda.current_stream().cuda_stream
        for name, t in states.items():
            if self.graph.states[name].kind == "recurrent":
                self._pend.copy_(pend)
                p = t.data_ptr()
                rc = self.lib.swiftmk_deltanet_materialize(
                    p, p + RECURRENT_SLOT_BYTES, 1, self.cur_ptr, self.pend_ptr, 1, stream
                )
                self.lib.check(rc, "swiftmk_deltanet_materialize")

    def save(self, mem: Memory, states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.materialize(states)
        cur = int(self._cur)  # a host read: a snapshot synchronises anyway
        out = {}
        for name, t in states.items():
            d = self.graph.states[name]
            if d.kind == "recurrent":
                hv, dk, dv = d.shape
                rec = t[: hv * dk * dv * 4].view(torch.float32).view(hv, dv, dk)
                out[name] = rec.transpose(1, 2).contiguous()
            elif d.kind == "conv":
                out[name] = t.view(torch.bfloat16).view(self.slots, *d.shape)[cur % self.slots].clone()
            elif (aux := self._aux(d)) is not None:
                out[f"{name}.aux"] = t[aux[0] : aux[0] + aux[1]].clone()
        return out

    def load(self, mem: Memory, states: Mapping[str, torch.Tensor], saved: Mapping[str, torch.Tensor]) -> None:
        for name in states:
            d = self.graph.states[name]
            key = f"{name}.aux" if d.kind == "kv" else name
            if d.kind == "kv" and self._aux(d) is None:
                continue
            if key not in saved:
                raise ValueError(f"the snapshot has no state {key!r}")
            want = (self._aux(d)[1],) if d.kind == "kv" else d.shape
            if tuple(saved[key].shape) != tuple(want):
                raise ValueError(f"snapshot state {key!r} has shape {tuple(saved[key].shape)}, not {tuple(want)}")
        for name, t in states.items():
            d = self.graph.states[name]
            if d.kind == "recurrent":
                hv, dk, dv = d.shape
                t[: hv * dk * dv * 4].view(torch.float32).view(hv, dv, dk).copy_(saved[name].transpose(1, 2))
            elif d.kind == "conv":
                t.view(torch.bfloat16).view(self.slots, *d.shape)[0].copy_(saved[name])
            elif (aux := self._aux(d)) is not None:
                t[aux[0] : aux[0] + aux[1]].copy_(saved[f"{name}.aux"])
        self._cursors.zero_()

    def kv_sections(self, state: StateDecl, rows_max: int) -> dict[str, RowSection]:
        _, ctx, nkv, hd = state.shape
        return FORMATS[state.cache].row_sections(ctx, nkv, hd, rows_max)

    def reset(self, mem: Memory) -> None:
        self._cursors.zero_()
