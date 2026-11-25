"""Runs a plan on the GPU through a backend's Executor: packs the weights, allocates the plan's buffers and the
states, and captures one CUDA graph per step variant (T rows, and the graph outputs the step computes).
Position and the fed-back token are device scalars the runtime updates inside the graphs, so one graph serves every
position and greedy decode needs no host round trip per token.

The seam: everything backend-specific (pack layouts, state layouts, kernel launches, backend cursors) is behind
`Executor`; the runtime owns buffers, states, streams, the position and the token feedback. Runtimes of one model (the
target and a draft head) share weight packs, kernel scratch and the side stream through an `Arena`.

`Decoder` is what `Model` forwards to: a `Runtime` is the plain one (a step per token), `warpc.speculate.Speculator`
the one that drafts and verifies. Both keep the same state variables (position, the backend's ring cursors, the
non-positional states), so `snapshot` / `restore` and the KV export are one mechanism for both.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from . import pinned
from .compiler import Pack, Plan, Task
from .ir import Graph, StateDecl, states_of
from .kvcache import KVChunk, RowSection
from .sources import Source

if TYPE_CHECKING:
    from .compiler import Backend

_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}


@dataclass(frozen=True)
class Memory:
    """The device pointers of one runtime, as an executor's launches read them."""

    rows_max: int  # rows the buffers hold
    pos: int  # int32 scalar: tokens already cached
    inputs: dict[str, int]  # every graph input but "pos", by "@<name>": the tensors the runtime binds
    buffers: dict[str, int]
    packs: dict[str, int]  # pack name -> device (or pinned host) pointer
    states: dict[str, int]
    constants: dict[str, int]  # fp32 tables, placed once at construction
    scratch: Callable[[str, int], int]  # (name, bytes) -> a persistent zeroed device scratch of at least that size

    def buffer(self, name: str) -> int:
        return self.inputs[name] if name.startswith("@") else self.buffers[name]


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

    def pack(self, pack: Pack, *, host: bool = False) -> torch.Tensor:
        """The tensor of one plan pack (raises on a kind the backend does not know): on the device, or in pinned host
        memory when `host` (a pack its kernels read only by row gather). No side effects besides the result: the
        Arena builds a pack once and shares it between runtimes."""
        ...

    def pack_nbytes(self, pack: Pack) -> int:
        """Device bytes of `pack`, without building it."""
        ...

    def schedule(self, mem: Memory, tasks: Sequence[Task], T: int) -> list[Launch]:
        """The launches of `tasks` for a step of T rows, in order."""
        ...

    def advance(self, mem: Memory, T: int, accepted: torch.Tensor, stream: int) -> None:
        """End of a step of T rows of which the first `accepted` (a device int32 [1]; T for a plain step) are kept:
        the backend's own cursors (none for most backends), on `stream`. Nothing is copied: a rejected row is
        overwritten by the next step."""
        ...

    def materialize(self, states: Mapping[str, torch.Tensor]) -> None:
        """Write the backend's pending lazy state rows into the live state tensors `states` (before a step that
        reads or writes the state from outside the kernels: a snapshot, a window prefill); nothing for a backend
        whose steps update the state in place."""
        ...

    def save(self, mem: Memory, states: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """The non-positional state out of the live state tensors `states`: pending lazy rows materialised, then, in
        the graph's layout (recurrent f32 [H, dk, dv], conv bf16 [C, W - 1]), a copy per state; "<kv>.aux" (uint8)
        for a KV format with exact rows. Device tensors, owned by the caller."""
        ...

    def load(self, mem: Memory, states: Mapping[str, torch.Tensor], saved: Mapping[str, torch.Tensor]) -> None:
        """Write `save`'s tensors (device or host) into the live state tensors `states` and put the cursors back to
        the start; ValueError naming the key when a state or "<kv>.aux" is missing (nothing is written then)."""
        ...

    def kv_sections(self, state: StateDecl, rows_max: int) -> dict[str, RowSection]:
        """The per-position planes of a KV state in this backend's layout."""
        ...

    def reset(self, mem: Memory) -> None:
        """An empty cache: the backend's own cursors back to their start (states are already zeroed)."""
        ...


def merge_sides(tasks: Sequence[Task], launches: Sequence[Launch]) -> list[Launch]:
    """`launches[i]` is task i's Launch. A task marked attrs["side"] (a small GEMV that may overlap a heavier one: the
    a|b projections beside the qkv|z GEMV, as the hand engine does) rides as `side` of the preceding Launch when the
    preceding task reads the same input. Refused if the two race on a buffer: the side task must neither read what the
    main task writes nor write what it reads or writes."""
    out: list[Launch] = []
    for i, (t, ln) in enumerate(zip(tasks, launches, strict=True)):
        prev = tasks[i - 1] if i else None
        if not t.attrs.get("side") or prev is None or prev.ins != t.ins:
            out.append(ln)
            continue
        p_in, p_out = set(prev.ins.values()), set(prev.outs.values())
        s_in, s_out = set(t.ins.values()), set(t.outs.values())
        if p_in & s_out or p_out & (s_in | s_out):
            raise AssertionError(f"{t.kernel} cannot run beside {prev.kernel}: they share a buffer")
        out[-1] = Launch(out[-1].main, (*out[-1].side, ln.main, *ln.side))
    return out


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


class Arena:
    """What the runtimes of one model share, created once by `compile`: the kernel scratch, the side stream and the
    weight packs (one device tensor per distinct (kind, parts), so a draft head reads the target's embedding table).
    `host_gather`: a pack every reading task lists in `Task.gather` (a table read only by row gather) lives in pinned
    host memory instead of on the device."""

    def __init__(self, *, host_gather: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("the runtime needs a CUDA device")
        self.scratch = _Scratch()
        self.side_stream = torch.cuda.Stream()
        self.host_gather = host_gather
        self._packs: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}
        self._gather_only: dict[tuple[str, tuple[str, ...]], bool] = {}

    def claim(self, plans: Sequence[Plan]) -> None:
        """Record which packs every task that reads them reads by row gather only; before the first `pack`."""
        for plan in plans:
            key_of = {p.name: key for key, p in plan.packs.items()}
            for t in plan.tasks:
                for role, name in t.weights.items():
                    key = key_of[name]
                    self._gather_only[key] = self._gather_only.get(key, True) and role in t.gather

    def on_host(self, pack: Pack) -> bool:
        return self.host_gather and self._gather_only.get((pack.kind, tuple(pack.parts)), False)

    def pack(self, executor: Executor, pack: Pack) -> torch.Tensor:
        key = (pack.kind, tuple(pack.parts))
        t = self._packs.get(key)
        if t is None:
            t = self._packs[key] = executor.pack(pack, host=self.on_host(pack))
        return t

    def freeze(self) -> None:
        """No more warm-ups: the scratch keeps its size from here on (before the first capture)."""
        self.scratch.freeze()


@dataclass
class Snapshot:
    """The state of a cache that is not positional, enough to continue from `position` without the tokens before it:
    the KV rows below `position` are positional and stay in (or are imported into) the live cache."""

    position: int  # committed tokens (a multiple of 64 and of 8 when a resume must equal a cold window prefill)
    anchor: int | None  # the token to feed next (speculative state); None for a plain Runtime
    hidden: (
        torch.Tensor | None
    )  # [H] bf16 final-norm row of position - 1 (the MTP pair of the last cached row); None without MTP
    states: dict[str, torch.Tensor]  # by StateDecl name, graph layout; "<kv>.aux" uint8 for a format with exact rows
    device: bool  # on the device, else pinned host


class Decoder(Protocol):
    """What `Model` forwards to."""

    position: int

    def reset(self) -> None: ...

    def append(self, ids: Sequence[int], *, head: bool = True) -> int:
        """Continue the cache from `position` with `ids`; returns the next token (argmax of the last row), or -1 with
        head=False (states only). After a `decode`, `ids` starts with the anchor (the last token it returned)."""
        ...

    def decode(self, n: int, *, stop: Collection[int] = ()) -> list[int]:
        """Exactly n tokens after the anchor, or up to and including the first stop id. The cache then holds the
        tokens before the last one returned (the new anchor): `position` grew by the number of tokens returned."""
        ...

    def snapshot(self, into: Snapshot | None = None, *, device: bool = False) -> Snapshot: ...

    def restore(self, snap: Snapshot) -> None: ...

    def kv_export(self, start: int, stop: int, into: KVChunk | None = None) -> KVChunk: ...

    def kv_import(self, chunk: KVChunk, start: int) -> None: ...


class Runtime:
    """Greedy decode of one sequence: steps of up to plan.rows_max rows, a KV cache of the positions the graph's KV
    states hold. The plain `Decoder`.

    `variants`: the (T, graph outputs) step graphs captured; default: T = 1 .. rows_max without outputs and one row with
    "ids" (the head). A runtime built on a shared `arena` is not finalized: the caller warms what else it enqueues
    (`warm`), then calls `arena.freeze()` and `finalize()` (a capture must see the scratch at its final size).

    `window`: (graph, plan) of the same model lowered at a larger rows_max with window kernels (`WindowVariant`): prompt
    chunks of more than 8 rows run through it eagerly, in the rows of `append(ends=...)`."""

    def __init__(
        self,
        graph: Graph,
        plan: Plan,
        source: Source,
        backend: Backend,
        arena: Arena | None = None,
        *,
        variants: Sequence[tuple[int, Collection[str]]] | None = None,
        window: tuple[Graph, Plan] | None = None,
    ):
        # a ring (a sliding-window state) holds fewer rows than there are positions: the positions the graph can run
        # are those of its largest KV state
        contexts = {s.shape[1] for s in graph.states.values() if s.kind == "kv"}
        if not contexts:
            raise ValueError("the graph has no KV state")
        if "ids" not in graph.outputs or "logits" not in graph.outputs:
            raise ValueError("the graph must output 'logits' and 'ids'")
        private = arena is None
        self.arena = Arena() if private else arena
        if private:
            self.arena.claim([plan])
        self._graph, self._plan = graph, plan
        self._ctx_max, self._rows_max = max(contexts), plan.rows_max
        self._executor = backend.executor(graph, plan, source)
        self._packs = {p.name: self.arena.pack(self._executor, p) for p in plan.packs.values()}
        rows = plan.rows_max

        def zeros(shape, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device="cuda")

        def shape_of(v: str) -> tuple[int, ...]:
            return tuple(rows if d == "T" else d for d in graph.types[v].shape)

        self._buffers = {b.name: zeros(rows * b.dim, _DTYPES[b.dtype]) for b in plan.buffers.values()}
        self._states = {s.name: zeros(self._executor.state_nbytes(s, rows), torch.uint8) for s in graph.states.values()}
        self.inputs = {n: zeros(shape_of(n), _DTYPES[t.dtype]) for n, t in graph.inputs.items() if n != "pos"}
        self.pos = zeros(1, torch.int32)
        self.outputs = {
            alias: self._buffers[plan.buffer_of[v]][: math.prod(shape_of(v))].view(shape_of(v))
            for alias, v in graph.outputs.items()
        }
        self._constants = {n: torch.tensor(v, dtype=torch.float32, device="cuda") for n, v in plan.constants.items()}
        self._accepted = {T: torch.full((1,), T, dtype=torch.int32, device="cuda") for T in range(1, rows + 1)}
        self._vocab_size = graph.types[graph.outputs["logits"]].shape[-1]
        self._memory = Memory(
            rows_max=rows,
            pos=self.pos.data_ptr(),
            inputs={f"@{k}": v.data_ptr() for k, v in self.inputs.items()},
            buffers={k: v.data_ptr() for k, v in self._buffers.items()},
            packs={k: v.data_ptr() for k, v in self._packs.items()},
            states={k: v.data_ptr() for k, v in self._states.items()},
            constants={k: v.data_ptr() for k, v in self._constants.items()},
            scratch=self.arena.scratch,
        )
        self.position = 0  # host mirror of the device position: tokens in the KV cache
        self._live: dict[frozenset[str], list[Task]] = {}
        if variants is None:
            variants = [(T, ()) for T in range(1, rows + 1)] + [(1, ("ids",))]
        self._variants = [(T, frozenset(o)) for T, o in variants]
        self._graphs: dict[tuple[int, frozenset[str]], torch.cuda.CUDAGraph] = {}
        # eager warm-ups load the kernel modules and size the scratch; then every capture sees its final pointers
        for T, outs in self._variants:
            self.warm(T, outs)
        self._window = None
        if window is not None:  # the eager window variant sizes its scratch here too, before the freeze
            from .windowrt import WindowVariant

            self._window = WindowVariant(self, *window)
            self._window.run(self, list(range(self._window.rows)))
        self.reset()
        if private:
            self.finalize()

    # ---- steps

    def _tasks(self, outputs: frozenset[str]) -> list[Task]:
        tasks = self._live.get(outputs)
        if tasks is None:
            unknown = outputs - set(self._graph.outputs)
            if unknown:
                raise ValueError(f"the graph has no outputs {sorted(unknown)} (has {sorted(self._graph.outputs)})")
            needed = {self._graph.outputs[o] for o in outputs}
            tasks = self._live[outputs] = _live_tasks(self._graph, self._plan, needed)
        return tasks

    def enqueue(self, T: int, outputs: Collection[str], *, advance: bool = True) -> None:
        """One step of T rows on the current stream: the launches of every task the graph outputs `outputs` (and the
        states) need, capture-safe. advance: the step is final (position += T, the backend's cursors, the head's
        argmax fed back as the next input token); advance=False is a speculative step whose rows `commit` keeps."""
        outs = frozenset(outputs)
        main, mem, side = torch.cuda.current_stream(), self._memory, self.arena.side_stream
        for launch in self._executor.schedule(mem, self._tasks(outs), T):
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
        if advance:
            self.commit(T, self._accepted[T])
            if "ids" in outs:
                self.inputs["tokens"][:1].copy_(self.outputs["ids"][T - 1 : T])

    def commit(self, T: int, accepted: torch.Tensor) -> None:
        """Keep the first `accepted` (device int32 [1]) of the T rows of the last `enqueue(advance=False)`."""
        self._executor.advance(self._memory, T, accepted, torch.cuda.current_stream().cuda_stream)
        self.pos.add_(accepted)

    def warm(self, T: int, outputs: Collection[str]) -> None:
        """One eager step from position 0 (before `finalize`): loads the kernel modules and sizes the scratch. Every
        warm-up writes from position 0: the sum of the variants' rows exceeds the context."""
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            self.pos.zero_()
            self.enqueue(T, outputs)
        torch.cuda.current_stream().wait_stream(warm)

    def finalize(self) -> None:
        """Freeze the arena's scratch and capture the variants."""
        self.arena.freeze()
        self.reset()
        for T, outs in self._variants:
            self._graphs[(T, outs)] = self.capture(lambda T=T, outs=outs: self.enqueue(T, outs))
        self.reset()

    @staticmethod
    def capture(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
        """`fn`'s launches on the capture stream as one CUDA graph (the scratch must be frozen)."""
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        return g

    def graph(self, T: int, outputs: Collection[str] = ()) -> torch.cuda.CUDAGraph:
        g = self._graphs.get((T, frozenset(outputs)))
        if g is None:
            raise ValueError(f"no captured step of {T} rows with outputs {sorted(outputs)}")
        return g

    def _require_room(self, n: int) -> None:
        if self.position + n > self._ctx_max:
            have = f"{self.position} cached + {n} new tokens"
            raise ValueError(f"{have} exceed the KV cache of {self._ctx_max} positions")

    # ---- Decoder

    def reset(self) -> None:
        """Empty the cache: the next append starts at position 0."""
        for s in self._states.values():
            s.zero_()
        self.pos.zero_()
        self._executor.reset(self._memory)
        self.position = 0

    def append(
        self,
        ids: Sequence[int],
        *,
        head: bool = True,
        on_hidden: Callable[[torch.Tensor, int, Sequence[int]], None] | None = None,
        ends: Sequence[int] | None = None,
    ) -> int:
        """Append `ids` (with `head` the last token is a head step); returns the next token (argmax), or -1 without
        `head`. The other tokens run in chunks: `ends` are the absolute positions where chunks end (ascending, the last
        one `position + len(ids) - 1` with `head`, `position + len(ids)` without; a chunk is an 8-row step sequence, or
        with a window variant one window when it has more than 8 rows), default chunks of the largest step (a window
        variant: its rows). A prefill is bit-exact only along the same chunks, so a caller that resumes passes the ends
        of a plan that is a function of the prompt (`windows.plan_windows`). `on_hidden(hidden [T, H], pos0, next_ids)`
        runs after every chunk of T rows at positions pos0 .. pos0 + T - 1 (not after the head step) with the final-norm
        rows and the tokens that follow them (fewer than T for the last chunk of a headless append, whose last token has
        no successor yet)."""
        ids = [int(i) for i in ids]
        if not ids:
            raise ValueError("append needs at least one token")
        self._require_room(len(ids))
        n = len(ids) - 1 if head else len(ids)
        p0 = self.position
        step = self._window.rows if self._window else self._rows_max
        if ends is None:
            ends = [*range(p0 + step, p0 + n, step), p0 + n] if n else []
        elif list(ends) != sorted(set(ends)) or (ends and (ends[0] <= p0 or ends[-1] != p0 + n)):
            raise ValueError(f"ends {list(ends)} must ascend from {p0} and end at {p0 + n}")
        outs = ("hidden",) if on_hidden else ()
        tokens = self.inputs["tokens"]
        i = 0
        for end in ends:
            while i < end - p0:
                T = min(end - p0 - i, step)
                if self._window and self._rows_max < T:
                    hidden = self._window.run(self, ids[i : i + T])
                    if on_hidden:
                        on_hidden(hidden, p0 + i, ids[i + 1 : i + 1 + T])
                    i += T
                    continue
                for j in range(0, T, self._rows_max):
                    t = min(self._rows_max, T - j)
                    tokens[:t].copy_(torch.tensor(ids[i + j : i + j + t], dtype=torch.int32))
                    self.graph(t, outs).replay()
                    if on_hidden:
                        on_hidden(self.outputs["hidden"][:t], self.position, ids[i + j + 1 : i + j + 1 + t])
                    self.position += t
                i += T
        return self.step(ids[-1], hidden=bool(on_hidden)) if head else -1

    def step(self, token: int, *, hidden: bool = False) -> int:
        """One decode step on `token`; returns the next token (argmax). hidden: the final-norm row of `token` is left
        in outputs["hidden"][0]."""
        self._require_room(1)
        self.inputs["tokens"][:1].fill_(int(token))
        self.graph(1, ("ids", "hidden") if hidden else ("ids",)).replay()
        self.position += 1
        return int(self.outputs["ids"][0])

    def decode(self, n: int, *, stop: Collection[int] = ()) -> list[int]:
        """n greedy steps continuing the last append or step, each fed the previous token on the device; returns the n
        tokens. With `stop` it ends after the first stop id and syncs once per token: the cache must not pass the
        stop (the speculative decoder does this on the device)."""
        if self.position == 0:
            raise ValueError("decode continues an append or step: the cache is empty")
        self._require_room(n)
        g, ids = self.graph(1, ("ids",)), self.outputs["ids"]
        if stop:
            stop, out = set(stop), []
            for _ in range(n):
                g.replay()
                self.position += 1
                out.append(int(ids[0]))
                if out[-1] in stop:
                    break
            return out
        out = torch.empty(n, dtype=torch.int32, device="cuda")
        for i in range(n):
            g.replay()
            out[i : i + 1].copy_(ids[:1])
        self.position += n
        return out.tolist()

    def snapshot(self, into: Snapshot | None = None, *, device: bool = False) -> Snapshot:
        """The non-positional state at `position` (materialising pending backend state). `into` reuses its tensors
        (and its placement); else a new snapshot, in pinned host memory unless `device`."""
        saved = self._executor.save(self._memory, self._states)
        if into is None:
            states = {k: v if device else _pinned(v) for k, v in saved.items()}
            snap = Snapshot(self.position, None, None, states, device)
        else:
            if set(into.states) != set(saved):
                raise ValueError(f"snapshot into: keys {sorted(into.states)} differ from {sorted(saved)}")
            for k, v in saved.items():
                into.states[k].copy_(v, non_blocking=True)
            into.position, into.anchor, into.hidden = self.position, None, None
            snap = into
        if not snap.device:
            torch.cuda.current_stream().synchronize()
        return snap

    def restore(self, snap: Snapshot) -> None:
        """Back to `snap`: its states written into the live cache, the cursors at the start, position = snap.position.
        KV rows below the position are the caller's (positional: they are valid iff the snapshot is on this lineage,
        else `kv_import` them)."""
        if snap.position > self._ctx_max:
            raise ValueError(f"snapshot at {snap.position} exceeds the KV cache of {self._ctx_max} positions")
        self._executor.load(self._memory, self._states, snap.states)
        self.pos.fill_(snap.position)
        self.position = snap.position

    def _kv_states(self) -> list[tuple[StateDecl, dict[str, RowSection]]]:
        out = []
        for d in self._graph.states.values():
            if d.kind == "kv":
                if d.shape[1] < self._ctx_max:
                    raise ValueError(f"state {d.name} is a ring: its rows are not positional")
                out.append((d, self._executor.kv_sections(d, self._rows_max)))
        return out

    def _kv_plane(self, d: StateDecl, s: RowSection, start: int, stop: int) -> torch.Tensor:
        buf = self._states[d.name][s.offset : s.offset + s.nbytes]
        return buf.view(s.heads, s.rows, s.row_bytes)[:, start:stop]

    def kv_export(self, start: int, stop: int, into: KVChunk | None = None) -> KVChunk:
        """Rows [start, stop) of every KV state, to pinned host memory (`into` reuses a chunk of the same shape)."""
        if not 0 <= start <= stop <= self.position:
            raise ValueError(f"rows [{start}, {stop}) are not within the {self.position} cached positions")
        planes = {}
        for d, sections in self._kv_states():
            for name, s in sections.items():
                key = f"{d.name}/{name}"
                shape = (s.heads, stop - start, s.row_bytes)
                dst = into.planes[key] if into is not None else pinned.empty(shape, torch.uint8)
                dst.copy_(self._kv_plane(d, s, start, stop), non_blocking=True)
                planes[key] = dst
        torch.cuda.current_stream().synchronize()
        return into if into is not None else KVChunk(start, stop, planes)

    def kv_import(self, chunk: KVChunk, start: int) -> None:
        """Write the rows of `chunk` to positions start .. start + (chunk.stop - chunk.start) - 1 of every KV state."""
        n = chunk.stop - chunk.start
        if start < 0 or start + n > self._ctx_max:
            raise ValueError(f"rows [{start}, {start + n}) exceed the KV cache of {self._ctx_max} positions")
        for d, sections in self._kv_states():
            for name, s in sections.items():
                key = f"{d.name}/{name}"
                if key not in chunk.planes:
                    raise ValueError(f"the chunk has no plane {key!r}")
                self._kv_plane(d, s, start, start + n).copy_(chunk.planes[key], non_blocking=True)
        torch.cuda.current_stream().synchronize()

    @property
    def logits(self) -> torch.Tensor:
        """fp32 [vocab] logits of the last head step (a view, overwritten by the next one)."""
        return self.outputs["logits"][0, : self._vocab_size]


def _pinned(t: torch.Tensor) -> torch.Tensor:
    return pinned.like(t, non_blocking=True)
