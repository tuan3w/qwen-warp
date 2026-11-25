"""Helpers of the kernel A/B gates (warpc kernels against the hand engine's libswiftmk on the same device inputs)."""

from __future__ import annotations

import ctypes
import re
import statistics
from collections.abc import Callable

import torch


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def graph_us(fn: Callable[[], None], inner: int = 20, replays: int = 200) -> float:
    """Median microseconds per call of `fn` (which enqueues its launches on the current stream): `inner` calls captured
    into one CUDA graph, `replays` timed replays with CUDA events, as a decode step runs them."""
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(10):
        g.replay()
    times = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        times.append((a, b))
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in times) * 1e3 / inner


def ab_us(ours: Callable[[], None], hand: Callable[[], None], rounds: int = 3) -> tuple[float, float]:
    """(ours, hand) median microseconds per call, measured alternately `rounds` times (the clocks drift between
    the two otherwise)."""
    o, h = [], []
    for _ in range(rounds):
        o.append(graph_us(ours))
        h.append(graph_us(hand))
    return statistics.median(o), statistics.median(h)


class Tap:
    """Records the inputs and outputs of chosen tasks during an EAGER decode of a compiled model (stage-1 real inputs
    for the family gates). `tap = Tap(backend, kernels, layers)` wraps a backend (the hand engine's `SwiftmkBackend`,
    or a composite); compile with `tap.backend`, then `tap.run(model, ids, n)` prefills `ids` in rows_max-row steps and
    decodes `n` greedy tokens eagerly (no CUDA graphs: the graphs the Runtime captured at construction do not record).

    `tap.records`: one dict per recorded task launch in step order: kernel, layer, T, pos (cached tokens before the
    step), ins / outs {role: bf16-or-fp32 tensor [T, dim] on the device}. Only buffers are recorded (roles of
    `Task.ins` / `Task.outs`), not weights or states."""

    def __init__(self, backend, kernels: tuple[str, ...], layers: tuple[int, ...] | None = None):
        self.records: list[dict] = []
        self.active = False
        self.pos = 0  # tokens cached before the step being run
        self._kernels, self._layers = set(kernels), None if layers is None else set(layers)
        self._inner = backend
        self.backend = _TappedBackend(backend, self)

    def wants(self, task) -> bool:
        if task.kernel not in self._kernels:
            return False
        if self._layers is None:
            return True
        names = [*task.states.values(), *task.weights.values()]
        return any((m := re.match(r"L(\d+)\.", n)) and int(m[1]) in self._layers for n in names)

    def run(self, model, ids, n: int) -> list[int]:
        """Eager prefill of `ids` and `n` decode steps with recording; returns the generated ids."""
        rt = model._runtime
        rt.reset()
        ids = [int(i) for i in ids]
        self.active = True
        try:
            i, last = 0, len(ids) - 1
            while i < last:
                T = min(rt._rows_max, last - i)
                rt._tokens[:T].copy_(torch.tensor(ids[i : i + T], dtype=torch.int32))
                self.pos = i
                rt._run(T, False)
                i += T
            rt._tokens[:1].fill_(ids[-1])
            out = []
            for step in range(n + 1):
                self.pos = last + step
                rt._run(1, True)
                out.append(rt._ids[:1].clone())
            torch.cuda.synchronize()
        finally:
            self.active = False
        return [int(t) for t in torch.cat(out)]


class _TappedBackend:
    def __init__(self, inner, tap: Tap):
        self.name, self.caps, self._inner, self._tap = inner.name, inner.caps, inner, tap

    def executor(self, graph, plan, source):
        return _TappedExecutor(self._inner.executor(graph, plan, source), plan, self._tap)


class _TappedExecutor:
    def __init__(self, inner, plan, tap: Tap):
        self._inner, self._plan, self._tap = inner, plan, tap

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def schedule(self, mem, tasks, T):
        from warpc.runtime import Launch, merge_sides

        launches = []
        for t in tasks:
            (ln,) = self._inner.schedule(mem, [t], T)
            launches.append(Launch(self._recorded(mem, t, T, ln.main), ln.side) if self._tap.wants(t) else ln)
        return merge_sides(tasks, launches)

    def _recorded(self, mem, t, T, main):
        tap, plan = self._tap, self._plan
        dtypes = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}

        def grab(role_buffers):
            out = {}
            for role, name in role_buffers.items():
                if not name.startswith("@"):
                    b = plan.buffers[name]
                    out[role] = (name, torch.empty(T, b.dim, dtype=dtypes[b.dtype], device="cuda"))
            return out

        def run(stream: int) -> None:
            if not tap.active:
                main(stream)
                return
            ins, outs = grab(t.ins), grab(t.outs)
            torch.cuda.synchronize()
            for name, x in ins.values():
                _copy_rows(mem.buffer(name), x)
            main(stream)
            torch.cuda.synchronize()
            for name, x in outs.values():
                _copy_rows(mem.buffer(name), x)
            m = next((re.match(r"L(\d+)\.", s) for s in t.states.values() if re.match(r"L(\d+)\.", s)), None)
            tap.records.append(
                {
                    "kernel": t.kernel,
                    "layer": int(m[1]) if m else None,
                    "T": T,
                    "pos": tap.pos,
                    "ins": {r: x for r, (_, x) in ins.items()},
                    "outs": {r: x for r, (_, x) in outs.items()},
                }
            )

        return run


def _copy_rows(src_ptr: int, dst: torch.Tensor) -> None:
    """Device-to-device copy of the rows of a plan buffer (pointer `src_ptr`) into `dst`."""
    from warpc.swiftmk import lib

    cd = lib.load().cdll
    cd.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rc = cd.cudaMemcpy(dst.data_ptr(), src_ptr, dst.numel() * dst.element_size(), 3)
    if rc != 0:
        raise RuntimeError(f"cudaMemcpy failed ({rc})")
