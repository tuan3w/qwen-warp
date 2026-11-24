"""Runs a plan on the GPU: packs the weights, allocates the plan's buffers and the KV states, and captures one CUDA
graph per step variant (T = 1 .. rows_max rows without the head, T = 1 with the head). Position and the fed-back token
are device scalars updated inside the graphs, so one graph serves every position and greedy decode needs no host round
trip per token."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from ..compiler import Plan, Task
from ..ir import Graph
from ..sources import Source
from . import glue, launch
from .backend import CudaBackend
from .pack import Packer

_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}


def _live_tasks(graph: Graph, plan: Plan, needed: set[str]) -> list[Task]:
    """Dead-task elimination: a node is live if it feeds a needed value or writes a state; a task is live if a node it
    covers is."""
    need, live = set(needed), set()
    for i in range(len(graph.nodes) - 1, -1, -1):
        n = graph.nodes[i]
        if "state" in n.attrs or any(o in need for o in n.outs):
            live.add(i)
            need.update(n.ins)
    return [t for t in plan.tasks if any(c in live for c in t.covers)]


class Runtime:
    """Greedy decode of one sequence: prompt steps of up to plan.rows_max rows, a KV cache of the positions the graph's
    states hold."""

    def __init__(self, graph: Graph, plan: Plan, source: Source):
        if not torch.cuda.is_available():
            raise RuntimeError("the CUDA runtime needs a CUDA device")
        contexts = {s.shape[1] for s in graph.states.values()}
        if len(contexts) != 1:
            raise ValueError(f"the graph's KV states must hold one number of positions, not {sorted(contexts)}")
        if "ids" not in graph.outputs or "logits" not in graph.outputs:
            raise ValueError("the graph must output 'logits' and 'ids'")
        self._graph, self._plan = graph, plan
        self._ctx_max, self._rows_max = contexts.pop(), plan.rows_max
        major, minor = torch.cuda.get_device_capability()
        packer = Packer(graph.weights, source)
        self._packs = {name: packer.build(p) for name, p in plan.packs.items()}
        del packer
        rows = plan.rows_max

        def zeros(n: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.zeros(n, dtype=dtype, device="cuda")

        self._buffers = {b.name: zeros(rows * b.dim, _DTYPES[b.dtype]) for b in plan.buffers.values()}
        self._states = {s.name: zeros(CudaBackend.kv_bytes(s, rows), torch.uint8) for s in graph.states.values()}
        self._tokens = torch.zeros(rows, dtype=torch.int32, device="cuda")
        self._pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self._scratch: dict[str, torch.Tensor] = {}
        self._ids = self._buffers[plan.buffer_of[graph.outputs["ids"]]]
        self._logits = self._buffers[plan.buffer_of[graph.outputs["logits"]]]
        self._vocab_size = graph.types[graph.outputs["logits"]].shape[-1]
        self._memory = launch.Memory(
            glue=glue.load(10 * major + minor),
            rows_max=rows,
            pos=self._pos.data_ptr(),
            tokens=self._tokens.data_ptr(),
            ids=self._ids.data_ptr(),
            buffers={k: v.data_ptr() for k, v in self._buffers.items()},
            packs={k: v.data_ptr() for k, v in self._packs.items()},
            states={k: v.data_ptr() for k, v in self._states.items()},
            scratch=self._scratch_ptr,
        )
        self._position = 0  # host mirror of the device position: tokens in the KV cache
        heads = {False: set(), True: {graph.outputs["ids"]}}
        self._tasks = {head: _live_tasks(graph, plan, needed) for head, needed in heads.items()}
        self._prefetch_stream = torch.cuda.Stream()
        self._graphs: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}
        for T in range(1, rows + 1):
            self._capture(T, False)
        self._capture(1, True)
        self.reset()

    def _scratch_ptr(self, name: str, nbytes: int) -> int:
        """Pointer to a persistent device scratch of at least nbytes; first used in the eager warm-up, before any
        capture."""
        t = self._scratch.get(name)
        if t is None or t.numel() < nbytes:
            t = self._scratch[name] = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        return t.data_ptr()

    def _run(self, T: int, head: bool) -> None:
        main, mem = torch.cuda.current_stream(), self._memory
        tasks = self._tasks[head]
        for i, t in enumerate(tasks):
            nbytes = launch.prefetch_bytes(t, tasks[i + 1]) if T == 1 and i + 1 < len(tasks) else 0
            if not nbytes:
                launch.launch(mem, t, T, main.cuda_stream)
                continue
            # the prefetch runs beside t on its own stream, joined before the next task
            fork, join = torch.cuda.Event(), torch.cuda.Event()
            fork.record(main)
            self._prefetch_stream.wait_event(fork)
            launch.prefetch(mem, tasks[i + 1], nbytes, self._prefetch_stream.cuda_stream)
            launch.launch(mem, t, T, main.cuda_stream)
            join.record(self._prefetch_stream)
            main.wait_event(join)
        launch.advance(mem, T, head, main.cuda_stream)

    def _capture(self, T: int, head: bool) -> None:
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        # eager warm-up: loads the kernel modules and allocates the scratch before capture
        with torch.cuda.stream(warm):
            self._run(T, head)
        torch.cuda.current_stream().wait_stream(warm)
        self.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._run(T, head)
        self._graphs[(T, head)] = g

    def _require_room(self, n: int) -> None:
        if self._position + n > self._ctx_max:
            have = f"{self._position} cached + {n} new tokens"
            raise ValueError(f"{have} exceed the KV cache of {self._ctx_max} positions")

    def reset(self) -> None:
        """Empty the KV cache: the next prefill starts at position 0."""
        for s in self._states.values():
            s.zero_()
        self._pos.zero_()
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
