"""warpc behind the `Engine` seam: plain greedy decode on the CUDA or swiftmk backend (no speculation).

`reuse=True` is snapshot / restore at the end of the prompt (the KV rows below it stay in the cache; exact rows of a
quantised format travel in the snapshot), so every round of a cell decodes at the positions of round 0.
"""

from __future__ import annotations

import gc
import re
import subprocess
import time
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

from . import Cannot, EngineSpec, Turn, Unchecked, reason_text

ROWS = 8  # prompt step rows: warpc.cuda.schedule.STEP_ROWS, the largest a launch handles
WINDOW_ROWS = 4096  # the window prefill's rows (compile docs: 17 .. 4096)
_FREE_MEMORY = re.compile(r"needs [\d.]+ GiB, [\d.]+ GiB free")


def total_gib() -> float | None:
    """The card's memory in GiB from nvidia-smi (no CUDA context); None without it."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return float(out.splitlines()[0]) / 1024
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


class WarpcEngine:
    def __init__(self, spec: EngineSpec):
        if spec.mtp:
            raise Cannot("unsupported", "MTP speculation needs the speculative turn of the 27B speed suites")
        if spec.backend not in ("cuda", "swiftmk"):
            raise Cannot("unsupported", f"backend {spec.backend}: this adapter runs the cuda and swiftmk backends")
        self.spec = spec
        self.model = None
        self._snapshot = None
        self._snapshot_ids: list[int] | None = None

    def _backend(self):
        if self.spec.backend == "swiftmk":
            from warpc.swiftmk import SwiftmkBackend

            return SwiftmkBackend(self.spec.head, window=bool(self.spec.window))
        from warpc.cuda import CudaBackend

        return CudaBackend(self.spec.head)

    def _sizing(self, graph, source):
        """What `memory.estimate` asks an executor: the declared sizes. The real executors need a CUDA context (the
        cuda one loads its kernels, the swiftmk one allocates cursors), so their own methods are called without
        constructing one; the swiftmk layouts come from its library, which is only loaded."""
        from warpc.cuda.backend import CudaExecutor

        if self.spec.backend == "cuda":
            return SimpleNamespace(
                pack_nbytes=lambda pack: CudaExecutor.pack_nbytes(SimpleNamespace(graph=graph), pack),
                state_nbytes=lambda state, rows: CudaExecutor.state_nbytes(None, state, rows),
            )
        from warpc.swiftmk import lib
        from warpc.swiftmk.executor import SwiftmkExecutor
        from warpc.swiftmk.pack import Packer

        try:
            library = lib.load()
        except RuntimeError as e:
            raise Unchecked(reason_text(e)) from e
        me = SimpleNamespace(lib=library, graph=graph, _packer=Packer(graph.weights, source))
        return SimpleNamespace(
            pack_nbytes=lambda pack: SwiftmkExecutor.pack_nbytes(me, pack),
            state_nbytes=lambda state, rows: SwiftmkExecutor.state_nbytes(me, state, rows),
        )

    def check(self, ctx_max: int) -> float:
        """Lower the graph on the CPU (a missing kernel names its reason) and estimate the device memory from the
        executors' declared sizes against the card's total; the estimate in GiB."""
        from warpc import frontend, memory, sources, windowrt
        from warpc.compiler import lower
        from warpc.errors import LoweringError, UnsupportedModel
        from warpc.runtime import Arena

        backend = self._backend()
        try:
            source = sources.open(self.spec.model)
            graph = frontend.build(source, ctx_max=ctx_max, rows_max=ROWS, kv=self.spec.kv)
            plan = lower(graph, backend, rows_max=ROWS)
            window = None
            if self.spec.window:
                wgraph = graph.pruned(["hidden"])
                window = lower(wgraph, backend, rows_max=WINDOW_ROWS)
        except LoweringError as e:
            raise Cannot("kernel_missing", reason_text(e)) from e
        except UnsupportedModel as e:
            raise Cannot("unsupported", reason_text(e)) from e
        except ValueError as e:
            raise Cannot("model_limit", reason_text(e)) from e
        arena = object.__new__(Arena)  # its constructor needs CUDA; `claim` and `on_host` read these three
        arena.host_gather, arena._gather_only, arena._packs = bool(self.spec.host_tables), {}, {}
        arena.claim([plan] if window is None else [plan, window])
        extra = windowrt.estimate_bytes(window, ctx_max) if window else 0
        est = memory.estimate([memory.Part(graph, plan, self._sizing(graph, source), 9)], arena, extra).device
        gib = est / memory.GIB
        if (total := total_gib()) is not None and gib > total:
            raise Cannot("memory", f"ctx_max {ctx_max} needs {gib:.1f} GiB of the card's {total:.1f} GiB")
        return gib

    def load(self, ctx_max: int) -> dict[str, Any]:
        import torch

        import warpc
        from warpc.errors import LoweringError, UnsupportedModel

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            self.model = warpc.compile(
                self.spec.model,
                ctx_max=ctx_max,
                kv=self.spec.kv,
                backend=self._backend(),
                host_tables=bool(self.spec.host_tables),
                window_rows=WINDOW_ROWS if self.spec.window else 0,
            )
        except LoweringError as e:
            raise Cannot("kernel_missing", reason_text(e)) from e
        except UnsupportedModel as e:
            raise Cannot("unsupported", reason_text(e)) from e
        except torch.OutOfMemoryError as e:
            raise Cannot("memory", reason_text(e)) from e
        except ValueError as e:
            if _FREE_MEMORY.search(str(e)):
                raise Cannot("memory", reason_text(e)) from e
            raise
        return {"compile_s": time.perf_counter() - t0, "plan": self.model.plan.summary()}

    def _ends(self, ids: list[int]) -> list[int] | None:
        if not self.spec.window:
            return None
        from warpc.windows import plan_windows

        return plan_windows(ids)

    def turn(self, ids: Sequence[int], n_new: int, *, reuse: bool) -> Turn:
        import torch

        model, ids = self.model, [int(i) for i in ids]
        if n_new < 2:
            raise ValueError("a turn decodes at least 2 tokens (the first delivery and one more)")
        torch.cuda.synchronize()
        ends = self._ends(ids)
        if not reuse:
            model.reset()
            self._snapshot = self._snapshot_ids = None
            cached = 0
            t0 = time.perf_counter()
            first = model.append(ids, ends=ends)  # returns with the token on the host
        elif self._snapshot is not None and self._snapshot_ids == ids:
            cached = self._snapshot.position
            t0 = time.perf_counter()
            model.restore(self._snapshot)
            first = model.step(ids[-1])
        else:
            # the first use of a prompt: a cold prefill that also takes the snapshot, which belongs to this turn's time
            model.reset()
            cached = 0
            t0 = time.perf_counter()
            if len(ids) > 1:
                model.append(ids[:-1], head=False, ends=ends)
            self._snapshot, self._snapshot_ids = model.snapshot(device=True), ids
            first = model.step(ids[-1])
        t1 = time.perf_counter()
        rest = model.decode(n_new - 1)
        t2 = time.perf_counter()
        return Turn([first, *rest], t1 - t0, t2 - t1, cached)

    def gpu_seconds(self, ids: Sequence[int], n_new: int) -> float:
        """The summed GPU time of the `n_new - 1` replays of a decode round started from the prompt's snapshot: what
        `Turn.decode_s` has on top of it is the host's share."""
        from .. import protocol

        self.turn(ids, 2, reuse=True)
        self.model.restore(self._snapshot)
        self.model.step(list(ids)[-1])
        return protocol.replay_seconds(self.model.decoder.graph(1, ("ids",)).replay, n_new - 1)

    def close(self) -> None:
        import torch

        self.model = self._snapshot = None
        gc.collect()
        torch.cuda.empty_cache()
