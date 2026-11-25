"""Graph -> Plan over a backend's capability table: the fusion pass claims the nodes, then ordering and buffer
assignment (with in-place residual updates) follow from the dataflow alone."""

from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Protocol

from ..errors import LoweringError
from ..ir import Graph
from .caps import CapsTable
from .fuse import Region, fuse
from .plan import Buffer, Plan, Task

if TYPE_CHECKING:
    from ..runtime import Executor
    from ..sources import Source


class Backend(Protocol):
    """A backend: its capability table (what lowering may claim) and its runtime half."""

    name: str
    caps: CapsTable

    def executor(self, graph: Graph, plan: Plan, source: Source) -> Executor:
        """The executor that packs, sizes states and launches the tasks of `plan` (lowered from `graph`)."""
        ...


def lower(graph: Graph, backend: Backend, *, rows_max: int) -> Plan:
    """Lower `graph` for steps of up to `rows_max` rows. Raises LoweringError naming every node no capability claims,
    with the reasons."""
    res = fuse(graph, backend.caps, rows_max)
    if res.unclaimed:
        lines = "\n  ".join(f"{op} x{n}: {why}" for op, why, n in res.backlog(graph))
        count = f"{len(res.unclaimed)} of {len(graph.nodes)} nodes"
        raise LoweringError(f"{backend.name}: no kernel for {count}:\n  {lines}")
    regions = res.regions
    order = _toposort(regions)
    plan = Plan(graph.name, rows_max, backend.name)
    _assign_buffers(graph, regions, order, plan)

    def buffer(v: str) -> str:
        return v if v.startswith("@") else plan.buffer_of[v]

    def constant(role: str, values: tuple[float, ...]) -> str:
        """The plan constant holding `values`: one per distinct table, shared by every task that reads it."""
        name = next((n for n, v in plan.constants.items() if v == values), f"{role}.{len(plan.constants)}")
        plan.constants[name] = values
        return name

    names: dict[str, tuple] = {}
    for ri in order:
        r = regions[ri]
        for p in r.packs:
            key = (p.kind, tuple(p.parts))
            if names.setdefault(p.name, key) != key:
                raise LoweringError(f"pack {p.name} is built as {names[p.name]} and as {key}")
            plan.packs.setdefault(key, p)
        plan.tasks.append(
            Task(
                kernel=r.kernel,
                ins={k: buffer(v) for k, v in r.ins.items()},
                outs={k: buffer(v) for k, v in r.outs.items()},
                weights=dict(r.weights),
                states=dict(r.states),
                constants={role: constant(role, values) for role, values in r.constants.items()},
                attrs=dict(r.attrs),
                covers=sorted(r.covers),
                numerics=r.numerics,
                tolerance=r.tolerance,
            )
        )
    return plan


def _toposort(regions: list[Region]) -> list[int]:
    """Kahn's algorithm over region dependencies; ties go to the region whose first covered node comes first."""
    producer: dict[str, int] = {}
    for ri, r in enumerate(regions):
        for v in r.outs.values():
            producer[v] = ri
    waiting = [{producer[v] for v in r.ins.values() if producer.get(v, ri) != ri} for ri, r in enumerate(regions)]
    users: dict[int, list[int]] = {}
    for ri, d in enumerate(waiting):
        for dep in d:
            users.setdefault(dep, []).append(ri)
    heap = [(min(regions[ri].covers), ri) for ri in range(len(regions)) if not waiting[ri]]
    heapq.heapify(heap)
    done: list[int] = []
    while heap:
        _, ri = heapq.heappop(heap)
        done.append(ri)
        for rj in users.get(ri, []):
            waiting[rj].discard(ri)
            if not waiting[rj]:
                heapq.heappush(heap, (min(regions[rj].covers), rj))
    if len(done) != len(regions):
        raise LoweringError("cyclic region dependencies")
    return done


def _assign_buffers(g: Graph, regions: list[Region], order: list[int], plan: Plan) -> None:
    """Linear scan over the task order. A value gets a buffer of rows_max rows; a buffer is reused by a later value of
    the same (dim, dtype) once its last reader has run; a region's `inplace` output takes over its input's buffer (the
    residual stream); a graph output keeps a buffer of its own."""
    last_use: dict[str, int] = {}
    for pos, ri in enumerate(order):
        for v in regions[ri].ins.values():
            last_use[v] = pos
    outputs = set(g.outputs.values())
    for v in outputs:
        last_use[v] = len(order)
    buf_of = plan.buffer_of
    free: dict[tuple, list[str]] = {}

    def release(b: str) -> None:
        bf = plan.buffers[b]
        free.setdefault((bf.dim, bf.dtype), []).append(b)

    for pos, ri in enumerate(order):
        r = regions[ri]
        moved: set[str] = set()
        for role, v in r.outs.items():
            if r.inplace and r.inplace[0] == role:
                src = r.ins[r.inplace[1]]
                if last_use.get(src, -1) > pos:
                    raise LoweringError(f"{r.kernel}: output {v} updates {src} in place, but a later task reads {src}")
                buf_of[v] = buf_of[src]
                plan.buffers[buf_of[v]].values.append(v)
                moved.add(src)
                continue
            dim = 1
            for d in g.types[v].shape[1:]:
                dim *= d
            dt = g.types[v].dtype
            lst = [] if v in outputs else free.get((dim, dt), [])
            if lst:
                b = lst.pop()
            else:
                b = f"b{len(plan.buffers) + 1}"
                plan.buffers[b] = Buffer(b, dim, dt)
            buf_of[v] = b
            plan.buffers[b].values.append(v)
        for v in dict.fromkeys(r.ins.values()):  # insertion order: buffer names must not depend on the string hash seed
            if v.startswith("@") or v in moved or v not in buf_of:
                continue
            if last_use.get(v, -1) == pos:
                release(buf_of[v])
        for v in r.outs.values():
            if v not in last_use and v in buf_of:  # written, never read, not a graph output
                release(buf_of[v])
