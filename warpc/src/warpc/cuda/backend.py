"""The CUDA backend: its capability table for the compiler, and its executor for the runtime (packs, state layouts,
launches of the kernels in kernels/)."""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import torch

from ..compiler import Pack, Plan, Task
from ..ir import STATE_DTYPE, Graph, StateDecl
from ..kvcache import FORMATS
from ..runtime import Launch, Memory
from ..sources import Source
from . import glue, launch
from .pack import Packer
from .table import TABLE

_STATE_BYTES = {"bf16": 2, "f32": 4}


def state_nbytes(state: StateDecl, rows_max: int) -> int:
    """Bytes of a state in this backend's layout. kv: the format's layout for steps of up to `rows_max` rows (the
    quantised formats' exact-row ring holds the window plus one step); conv / recurrent: the plain [shape] array of
    STATE_DTYPE."""
    if state.kind == "kv":
        _, ctx, nkv, hd = state.shape
        return FORMATS[state.cache].nbytes(ctx, nkv, hd, rows_max)
    return math.prod(state.shape) * _STATE_BYTES[STATE_DTYPE[state.kind]]


class CudaExecutor:
    """Launches the CUDA backend's tasks; it keeps no cursors of its own (the runtime owns position and feedback)."""

    def __init__(self, graph: Graph, source: Source):
        major, minor = torch.cuda.get_device_capability()
        self._lib = glue.load(10 * major + minor)
        self._packer = Packer(graph.weights, source)

    def state_nbytes(self, state: StateDecl, rows_max: int) -> int:
        return state_nbytes(state, rows_max)

    def pack(self, pack: Pack) -> torch.Tensor:
        return self._packer.build(pack)

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        out = []
        for i, t in enumerate(tasks):
            main = functools.partial(launch.LAUNCHERS[t.kernel], self._lib, mem, t, T)
            nbytes = launch.prefetch_bytes(t, tasks[i + 1]) if T == 1 and i + 1 < len(tasks) else 0
            side = (functools.partial(launch.prefetch, self._lib, mem, tasks[i + 1], nbytes),) if nbytes else ()
            out.append(Launch(main, side))
        return out

    def advance(self, mem: Memory, T: int, stream: int) -> None:
        pass

    def reset(self, mem: Memory) -> None:
        pass


class CudaBackend:
    name = "cuda"
    caps = TABLE

    def executor(self, graph: Graph, plan: Plan, source: Source) -> CudaExecutor:
        return CudaExecutor(graph, source)
