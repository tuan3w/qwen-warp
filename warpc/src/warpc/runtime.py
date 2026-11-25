"""Runs a plan on the GPU through a backend's Executor: packs the weights, allocates the plan's buffers and the
states, and captures one CUDA graph per step variant (T = 1 .. rows_max rows without the head, T = 1 with the head).
Position and the fed-back token are device scalars the runtime updates inside the graphs, so one graph serves every
position and greedy decode needs no host round trip per token.

The seam: everything backend-specific (pack layouts, state layouts, kernel launches, backend cursors) is behind
`Executor`; the runtime owns buffers, states, scratch, streams, the position and the token feedback.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from .compiler import Pack, Plan, Task
from .ir import Graph, StateDecl, states_of
from .sources import Source

if TYPE_CHECKING:
    from .compiler import Backend

_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}


@dataclass(frozen=True)
class Memory:
    """The device pointers of one runtime, as an executor's launches read them."""

    rows_max: int  # rows the buffers hold
    pos: int  # int32 scalar: tokens already cached
    tokens: int  # int32 [rows_max]: the input tokens of the step
    ids: int  # int32 [rows_max]: the argmax ids of the head step
    buffers: dict[str, int]
    packs: dict[str, int]  # pack name -> device pointer
    states: dict[str, int]
    constants: dict[str, int]  # fp32 tables, placed once at construction
    scratch: Callable[[str, int], int]  # (name, bytes) -> a persistent zeroed device scratch of at least that size

    def buffer(self, name: str) -> int:
        return self.tokens if name == "@tokens" else self.buffers[name]


@dataclass(frozen=True)
class Launch:
    """One task's work. `main` is enqueued on the main stream; every `side` callable is forked onto the one side
    stream before `main` and joined before the next Launch's `main` (work that overlaps `main` and reads nothing it
    writes)."""

    main: Callable[[int], None]  # (CUDA stream handle) -> None
    side: tuple[Callable[[int], None], ...] = ()


class Executor(Protocol):
    """A backend's runtime half, built per (graph, plan, source) by `Backend.executor`."""

    def state_nbytes(self, state: StateDecl, rows_max: int) -> int:
        """Bytes of `state` in the layout of the kernel that owns it, for steps of up to rows_max rows."""
        ...

    def pack(self, pack: Pack) -> torch.Tensor:
        """The device tensor of one plan pack (raises on a kind the backend does not know)."""
        ...

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        """The launches of `tasks` for a step of T rows, in order."""
        ...

    def advance(self, mem: Memory, T: int, stream: int) -> None:
        """End of a step of T rows: the backend's own cursors (none for most backends), on `stream`."""
        ...

    def reset(self, mem: Memory) -> None:
        """An empty cache: the backend's own cursors back to their start (states are already zeroed)."""
        ...


def _live_tasks(graph: Graph, plan: Plan, needed: set[str]) -> list[Task]:
    """Dead-task elimination: a node is live if it feeds a needed value or writes a state; a task is live if a node it
    covers is."""
    need, live = set(needed), set()
    for i in range(len(graph.nodes) - 1, -1, -1):
        n = graph.nodes[i]
        if states_of(n) or any(o in need for o in n.outs):
            live.add(i)
            need.update(n.ins)
    return [t for t in plan.tasks if any(c in live for c in t.covers)]


class _Scratch:
    """Persistent zeroed device scratch by name, grown on demand during the eager warm-ups and frozen before the
    first capture: a captured graph holds the pointer, so growth after the freeze raises. A separate object, not a
    runtime method, so Memory holds no reference back to its runtime: a dropped runtime is freed by refcount, never by a
    cyclic collection, which can run inside another runtime's capture and tear down CUDA graphs there (an invalidated
    capture)."""

    def __init__(self):
        self._tensors: dict[str, torch.Tensor] = {}
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def __call__(self, name: str, nbytes: int) -> int:
        t = self._tensors.get(name)
        if t is None or t.numel() < nbytes:
            if self._frozen:
                raise RuntimeError(f"scratch {name!r} grows to {nbytes} bytes after the warm-ups")
            t = self._tensors[name] = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        return t.data_ptr()


class Runtime:
    """Greedy decode of one sequence: prompt steps of up to plan.rows_max rows, a KV cache of the positions the
    graph's KV states hold."""

    def __init__(self, graph: Graph, plan: Plan, source: Source, backend: Backend):
        if not torch.cuda.is_available():
            raise RuntimeError("the runtime needs a CUDA device")
        contexts = {s.shape[1] for s in graph.states.values() if s.kind == "kv"}
        if len(contexts) != 1:
            raise ValueError(f"the graph's KV states must hold one number of positions, not {sorted(contexts)}")
        if "ids" not in graph.outputs or "logits" not in graph.outputs:
            raise ValueError("the graph must output 'logits' and 'ids'")
        self._graph, self._plan = graph, plan
        self._ctx_max, self._rows_max = contexts.pop(), plan.rows_max
        self._executor = backend.executor(graph, plan, source)
        self._packs = {p.name: self._executor.pack(p) for p in plan.packs.values()}
        rows = plan.rows_max

        def zeros(n: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(n, dtype=dtype, device="cuda")

        self._buffers = {b.name: zeros(rows * b.dim, _DTYPES[b.dtype]) for b in plan.buffers.values()}
        self._states = {s.name: zeros(self._executor.state_nbytes(s, rows), torch.uint8) for s in graph.states.values()}
        self._tokens = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self._pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._scratch = _Scratch()
        self._constants = {n: torch.tensor(v, dtype=torch.float32, device="cuda") for n, v in plan.constants.items()}
        self._ids = self._buffers[plan.buffer_of[graph.outputs["ids"]]]
        self._logits = self._buffers[plan.buffer_of[graph.outputs["logits"]]]
        self._vocab_size = graph.types[graph.outputs["logits"]].shape[-1]
        self._memory = Memory(
            rows_max=rows,
            pos=self._pos.data_ptr(),
            tokens=self._tokens.data_ptr(),
            ids=self._ids.data_ptr(),
            buffers={k: v.data_ptr() for k, v in self._buffers.items()},
            packs={k: v.data_ptr() for k, v in self._packs.items()},
            states={k: v.data_ptr() for k, v in self._states.items()},
            constants={k: v.data_ptr() for k, v in self._constants.items()},
            scratch=self._scratch,
        )
        self._position = 0  # host mirror of the device position: tokens in the KV cache
        heads = {False: set(), True: {graph.outputs["ids"]}}
        self._tasks = {head: _live_tasks(graph, plan, needed) for head, needed in heads.items()}
        self._side_stream = torch.cuda.Stream()
        variants = [(T, False) for T in range(1, rows + 1)] + [(1, True)]
        # eager warm-ups load the kernel modules and size the scratch; then every capture sees its final pointers
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for T, head in variants:
                self._pos.zero_()  # every warm-up writes from position 0: the sum of the variants' rows exceeds ctx
                self._run(T, head)
        torch.cuda.current_stream().wait_stream(warm)
        self._scratch.freeze()
        self.reset()
        self._graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        for T, head in variants:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._run(T, head)
            self._graphs[(T, head)] = g
        self.reset()

    def _run(self, T: int, head: bool) -> None:
        main, mem, side = torch.cuda.current_stream(), self._memory, self._side_stream
        for launch in self._executor.schedule(mem, self._tasks[head], T):
            if not launch.side:
                launch.main(main.cuda_stream)
                continue
            fork, join = torch.cuda.Event(), torch.cuda.Event()
            fork.record(main)
            side.wait_event(fork)
            for fn in launch.side:
                fn(side.cuda_stream)
            launch.main(main.cuda_stream)
            join.record(side)
            main.wait_event(join)
        self._executor.advance(mem, T, main.cuda_stream)
        self._pos.add_(T)
        if head:
            self._tokens[:1].copy_(self._ids[T - 1 : T])

    def _require_room(self, n: int) -> None:
        if self._position + n > self._ctx_max:
            have = f"{self._position} cached + {n} new tokens"
            raise ValueError(f"{have} exceed the KV cache of {self._ctx_max} positions")

    def reset(self) -> None:
        """Empty the cache: the next prefill starts at position 0."""
        for s in self._states.values():
            s.zero_()
        self._pos.zero_()
        self._executor.reset(self._memory)
        self._position = 0

    def prefill(self, ids: Sequence[int]) -> int:
        """Append the prompt `ids` (chunks of up to rows_max rows, the last token as a head step); returns the next
        token (argmax)."""
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError("prefill needs at least one token")
        self._require_room(len(ids))
        i, n = 0, len(ids) - 1
        while i < n:
            T = min(self._rows_max, n - i)
            self._tokens[:T].copy_(torch.tensor(ids[i : i + T], dtype=torch.int32))
            self._graphs[(T, False)].replay()
            i += T
        self._position += n
        return self.step(ids[-1])

    def step(self, token: int) -> int:
        """One decode step on `token`; returns the next token (argmax)."""
        self._require_room(1)
        self._tokens[:1].fill_(int(token))
        self._graphs[(1, True)].replay()
        self._position += 1
        return int(self._ids[0])

    def decode(self, n: int) -> list[int]:
        """n greedy steps continuing the last prefill or step, each fed the previous token on the device; returns the n
        tokens."""
        if self._position == 0:
            raise ValueError("decode continues a prefill or step: the cache is empty")
        self._require_room(n)
        out = torch.empty(n, dtype=torch.int32, device="cuda")
        g = self._graphs[(1, True)]
        for i in range(n):
            g.replay()
            out[i : i + 1].copy_(self._ids[:1])
        self._position += n
        return out.tolist()

    @property
    def logits(self) -> torch.Tensor:
        """fp32 [vocab] logits of the last head step (a view, overwritten by the next one)."""
        return self._logits[: self._vocab_size]
