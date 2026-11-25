"""The CUDA backend: its capability table for the compiler, and its executor for the runtime (packs, state layouts,
launches of the kernels in kernels/)."""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence

import torch

from .. import pinned
from ..compiler import Pack, Plan, Task
from ..ir import STATE_DTYPE, Graph, StateDecl
from ..kvcache import FORMATS, RowSection
from ..runtime import Launch, Memory, merge_sides
from ..sources import Source
from . import glue, launch, schedule
from .pack import Packer
from .table import table

_STATE_BYTES = {"bf16": 2, "f32": 4}
_STATE_TORCH = {"bf16": torch.bfloat16, "f32": torch.float32}


def state_nbytes(state: StateDecl, rows_max: int) -> int:
    """Bytes of a state in this backend's layout. kv: the format's layout for steps of up to `rows_max` rows (the
    quantised formats' exact-row ring holds the window plus one step; a bf16 state's rows are rounded up to whole
    attention chunks, so a ring of 519 rows holds 576 and its 64-key chunks never wrap); conv / recurrent: the plain
    [shape] array of STATE_DTYPE."""
    if state.kind == "kv":
        _, ctx, nkv, hd = state.shape
        fmt = FORMATS[state.cache]
        return fmt.nbytes(ctx if fmt.quantized else schedule.kv_rows(ctx), nkv, hd, rows_max)
    return math.prod(state.shape) * _STATE_BYTES[STATE_DTYPE[state.kind]]


class CudaExecutor:
    """Launches the CUDA backend's tasks; it keeps no cursors of its own (the runtime owns position and feedback). Its
    DeltaNet and conv states are the graph's own arrays, updated in place by every step: a step cannot be partly
    accepted (the speculative verify needs the lazy kernel of a later stage), but `save` / `load` are plain copies."""

    def __init__(self, graph: Graph, source: Source):
        major, minor = torch.cuda.get_device_capability()
        self._lib = glue.load(10 * major + minor)
        self.graph = graph
        self._packer = Packer(graph.weights, source)
        self._host_packer = Packer(graph.weights, source, device="cpu")

    def state_nbytes(self, state: StateDecl, rows_max: int) -> int:
        return state_nbytes(state, rows_max)

    def pack(self, pack: Pack, *, host: bool = False) -> torch.Tensor:
        return pinned.like(self._host_packer.build(pack)) if host else self._packer.build(pack)

    def pack_nbytes(self, pack: Pack) -> int:
        """The declared bytes of the parts: exact for dense, W4 and Q4_0 packs and their concatenations, an estimate
        for a kernel family's own layout (the int4 head copy is priced as its W4 bytes)."""
        w = self.graph.weights
        if pack.kind == "head_i4g":
            n, k = w[pack.parts[0]].shape
            return n * k // 2 + n * (k // 128) * 2
        return sum(w[p].nbytes for p in pack.parts)

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        out = []
        for i, t in enumerate(tasks):
            main = functools.partial(launch.LAUNCHERS[t.kernel], self._lib, mem, t, T)
            nbytes = launch.prefetch_bytes(t, tasks[i + 1]) if T == 1 and i + 1 < len(tasks) else 0
            side = (functools.partial(launch.prefetch, self._lib, mem, tasks[i + 1], nbytes),) if nbytes else ()
            out.append(Launch(main, side))
        return merge_sides(tasks, out)

    def advance(self, mem: Memory, T: int, accepted: torch.Tensor, stream: int) -> None:
        pass

    def materialize(self, states: Mapping[str, torch.Tensor]) -> None:
        pass

    def _aux(self, state: StateDecl, rows_max: int) -> tuple[int, int] | None:
        _, ctx, nkv, hd = state.shape
        return FORMATS[state.cache].aux_range(ctx, nkv, hd, rows_max)

    def save(self, mem: Memory, states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for name, t in states.items():
            d = self.graph.states[name]
            if d.kind == "kv":
                if (aux := self._aux(d, mem.rows_max)) is not None:
                    out[f"{name}.aux"] = t[aux[0] : aux[0] + aux[1]].clone()
            else:
                out[name] = t.view(_STATE_TORCH[STATE_DTYPE[d.kind]]).view(d.shape).clone()
        return out

    def load(self, mem: Memory, states: Mapping[str, torch.Tensor], saved: Mapping[str, torch.Tensor]) -> None:
        todo = []
        for name, t in states.items():
            d = self.graph.states[name]
            if d.kind == "kv":
                aux = self._aux(d, mem.rows_max)
                if aux is None:
                    continue
                key, dst, want = f"{name}.aux", t[aux[0] : aux[0] + aux[1]], (aux[1],)
            else:
                key, dst, want = name, t.view(_STATE_TORCH[STATE_DTYPE[d.kind]]).view(d.shape), d.shape
            if key not in saved:
                raise ValueError(f"the snapshot has no state {key!r}")
            if tuple(saved[key].shape) != tuple(want):
                raise ValueError(f"snapshot state {key!r} has shape {tuple(saved[key].shape)}, not {tuple(want)}")
            todo.append((dst, saved[key]))
        for dst, src in todo:
            dst.copy_(src)

    def kv_sections(self, state: StateDecl, rows_max: int) -> dict[str, RowSection]:
        _, ctx, nkv, hd = state.shape
        fmt = FORMATS[state.cache]
        return fmt.row_sections(ctx if fmt.quantized else schedule.kv_rows(ctx), nkv, hd, rows_max)

    def reset(self, mem: Memory) -> None:
        pass


class CudaBackend:
    """`head` picks the lm_head kernel: "bf16" (the exact head) or "int4_refine" (an int4 g128 copy of the head with the
    top 16 per row rescored exactly: approximate logits, the hand engine's default head)."""

    name = "cuda"

    def __init__(self, head: str = "bf16"):
        self.head = head
        self.caps = table(head)

    def executor(self, graph: Graph, plan: Plan, source: Source) -> CudaExecutor:
        return CudaExecutor(graph, source)
