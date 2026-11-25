"""CompositeBackend: two backends behind one capability table. The primary's kernels are preferred wherever both match
the same nodes (the fallback covers what the primary cannot, or has not been admitted to yet); each task then runs
through the executor of the backend that owns its kernel.

Ownership is per task kernel, and so per pack (the owner of the first task that reads it) and per state (the owner of
the task that touches it): a layer's KV, conv or recurrent state is sized and laid out by exactly one backend.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection, Sequence

import torch

from .compiler import Backend, CapsTable, Pack, Plan, Task
from .ir import Graph, StateDecl
from .runtime import Executor, Launch, Memory, merge_sides
from .sources import Source


class CompositeBackend:
    """`kernels`: the primary's kernel names admitted (None: all of them); every kernel of the fallback is."""

    def __init__(self, primary: Backend, fallback: Backend, *, kernels: Collection[str] | None = None):
        known = {c.kernel for c in primary.caps.caps}
        if kernels is not None and (bad := set(kernels) - known):
            raise ValueError(f"{primary.name} has no kernels {sorted(bad)} (known: {sorted(known)})")
        taken = [c for c in primary.caps.caps if kernels is None or c.kernel in kernels]
        pt, ft = primary.caps, fallback.caps
        self.caps = CapsTable(
            [*taken, *ft.caps],
            pt.target,
            pt.unfused_ctas,
            {**ft.unfused_launches, **pt.unfused_launches},
            prefer_rank=True,
        )
        self.name = f"{primary.name}+{fallback.name}"
        self._owner = {**{c.kernel: primary for c in taken}, **{c.kernel: fallback for c in ft.caps}}

    def executor(self, graph: Graph, plan: Plan, source: Source) -> CompositeExecutor:
        return CompositeExecutor(graph, plan, source, self._owner)


class _OwnerMemory:
    """The Memory one executor sees: scratch names are private to the owner (two backends may call their workspaces
    alike with different sizes and initialisation contracts)."""

    @staticmethod
    def of(mem: Memory, owner: str) -> Memory:
        return dataclasses.replace(mem, scratch=lambda name, nbytes: mem.scratch(f"{owner}.{name}", nbytes))


class CompositeExecutor:
    def __init__(self, graph: Graph, plan: Plan, source: Source, owner_of: dict[str, Backend]):
        self._owner_of = owner_of
        used = {owner_of[t.kernel].name: owner_of[t.kernel] for t in plan.tasks}
        self._executors: dict[str, Executor] = {n: b.executor(graph, plan, source) for n, b in used.items()}
        self._pack_owner: dict[str, str] = {}
        self._state_owner: dict[str, str] = {}
        for t in plan.tasks:
            name = owner_of[t.kernel].name
            for p in t.weights.values():
                self._pack_owner.setdefault(p, name)
            for s in t.states.values():
                self._state_owner.setdefault(s, name)

    def state_nbytes(self, state: StateDecl, rows_max: int) -> int:
        return self._executors[self._state_owner[state.name]].state_nbytes(state, rows_max)

    def pack(self, pack: Pack) -> torch.Tensor:
        return self._executors[self._pack_owner[pack.name]].pack(pack)

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        launches = []
        for t in tasks:
            name = self._owner_of[t.kernel].name
            (launch,) = self._executors[name].schedule(_OwnerMemory.of(mem, name), [t], T)
            launches.append(launch)
        return merge_sides(tasks, launches)

    def advance(self, mem: Memory, T: int, stream: int) -> None:
        for name, ex in self._executors.items():
            ex.advance(_OwnerMemory.of(mem, name), T, stream)

    def reset(self, mem: Memory) -> None:
        for name, ex in self._executors.items():
            ex.reset(_OwnerMemory.of(mem, name))
