"""The pre-flight device memory estimate of `compile`: what the runtimes will allocate, summed from the executors'
declarations before anything is allocated, against the free device memory. A context that cannot fit raises a
ValueError that names the numbers, instead of an out-of-memory error in the middle of loading 17 GB of weights.
Host placement of the gather-only tables (`host_tables`) is explicit and never chosen here."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .compiler import Plan
from .ir import Graph
from .runtime import Arena, Executor

GIB = 2**30
GRAPH_BYTES = 9 << 20  # device memory of one captured step graph (6 - 12 MiB measured on the 27B: 4 ms to capture)
MARGIN_FIXED = 256 << 20  # kernel workspaces, the speculative program's tensors
MARGIN_FRACTION = 0.06  # allocator pool and fragmentation, of the weights and states (1.0 GiB at the 27B)
_ITEMSIZE = {"bf16": 2, "f32": 4, "i32": 4}


@dataclass(frozen=True)
class Part:
    """One runtime to be built: its graph and plan, an executor to size it with, the step graphs it will capture."""

    graph: Graph
    plan: Plan
    executor: Executor
    variants: int


@dataclass(frozen=True)
class Estimate:
    packs: int  # device-placed weight packs, each shared pack once
    host_packs: int  # pinned host memory
    states: int
    buffers: int
    graphs: int
    margin: int

    @property
    def device(self) -> int:
        return self.packs + self.states + self.buffers + self.graphs + self.margin


def estimate(parts: list[Part], arena: Arena, extra: int = 0) -> Estimate:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    packs = host = states = buffers = graphs = 0
    for p in parts:
        for key, pack in p.plan.packs.items():
            if key in seen:
                continue
            seen.add(key)
            nbytes = p.executor.pack_nbytes(pack)
            if arena.on_host(pack):
                host += nbytes
            else:
                packs += nbytes
        states += sum(p.executor.state_nbytes(s, p.plan.rows_max) for s in p.graph.states.values())
        buffers += sum(p.plan.rows_max * b.dim * _ITEMSIZE[b.dtype] for b in p.plan.buffers.values())
        graphs += p.variants * GRAPH_BYTES
    margin = MARGIN_FIXED + int(MARGIN_FRACTION * (packs + states))
    return Estimate(packs, host, states, buffers + extra, graphs, margin)


def preflight(parts: list[Part], arena: Arena, *, ctx_max: int, extra: int = 0) -> Estimate:
    """Raise ValueError when the estimate exceeds the free device memory; else return it. `extra`: bytes of buffers
    outside the parts (the window variant's, `windowrt.estimate_bytes`). Memory torch's allocator holds free (a model
    dropped in this process) counts as free: the runtime allocates through torch, which reuses it."""
    est = estimate(parts, arena, extra)
    free = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    if est.device > free:
        hint = "use a smaller ctx_max" if arena.host_gather else "use host_tables=True or a smaller ctx_max"
        raise ValueError(f"ctx_max {ctx_max} needs {est.device / GIB:.1f} GiB, {free / GIB:.1f} GiB free; {hint}")
    return est
