"""compile(): checkpoint -> graph -> plan -> runtime over a backend, behind one Model object."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import torch

from . import frontend, memory, sources
from .compiler import Backend, Plan, lower
from .cuda import CudaBackend
from .cuda.schedule import STEP_ROWS
from .kvcache import KVChunk
from .runtime import Arena, Decoder, Runtime, Snapshot
from .sources import ModelConfig, Source
from .sources.requantize import Requantized
from .speculate import Speculator
from .swiftmk import SwiftmkBackend
from .swiftmk.window import require_libraries
from .windowrt import estimate_bytes

# kernels whose recurrent state tolerates a verify step that keeps only part of its rows (lazy pending rows); another
# kernel updates the state in place, so a rejected row could not be rolled back
ROLLBACK_KERNELS = {"s_deltanet"}


class Model:
    """A compiled model: greedy decode on the GPU with a KV cache of `ctx_max` positions. A forwarder over one
    `Decoder`: a `Runtime` (a step per token) or, when compiled with `speculate`, a `Speculator` (`decoder.k` drafts
    per cycle: 0 is plain decode, see warpc.speculate)."""

    def __init__(self, config: ModelConfig, plan: Plan, runtime: Runtime, decoder: Decoder):
        self.config, self.plan, self._runtime, self.decoder = config, plan, runtime, decoder

    @property
    def position(self) -> int:
        """Tokens in the KV cache."""
        return self.decoder.position

    def reset(self) -> None:
        """Empty the KV cache."""
        self.decoder.reset()

    def append(self, ids: Sequence[int], *, head: bool = True, ends: Sequence[int] | None = None) -> int:
        """Append `ids` to the cache and return the next token (argmax of the last row); head=False appends the
        states only and returns -1. After a `decode`, `ids` starts with the token that returned last (the anchor).
        `ends`: absolute chunk ends (see `Runtime.append`; `windows.plan_windows`)."""
        return self.decoder.append(ids, head=head, ends=ends)

    def decode(self, n: int, *, stop: Collection[int] = ()) -> list[int]:
        """Exactly n greedy tokens continuing the last append or decode (fewer, ending with the first stop id, if one
        comes): with no host round trip per token. The cache then holds every token before the last one returned.
        With speculation the tokens equal plain greedy decode except at near-ties (see warpc.speculate)."""
        return self.decoder.decode(n, stop=stop)

    def step(self, token: int) -> int:
        """Append one token; returns the next token (argmax). The plain step, whatever the decoder."""
        return self._runtime.step(token)

    @property
    def logits(self) -> torch.Tensor:
        """fp32 [vocab] logits of the last plain head step (a view, overwritten by the next one)."""
        return self._runtime.logits

    def generate(self, ids: Sequence[int], max_new_tokens: int, *, stop: Collection[int] = ()) -> list[int]:
        """Greedy continuation of the prompt `ids` from an empty cache: the first `max_new_tokens` generated tokens
        (up to and including the first stop id), at the decoder's current k."""
        self.reset()
        if max_new_tokens <= 0:
            return []
        first = self.append(ids)
        if first in stop:
            return [first]
        return [first, *self.decode(max_new_tokens - 1, stop=stop)]

    def snapshot(self, into: Snapshot | None = None, *, device: bool = False) -> Snapshot:
        """The non-positional state at `position`: `restore` returns to it (the KV rows below it stay in the cache).
        In pinned host memory unless `device` (a copy on the device is 0.4 ms, to the host 15 ms); `into` reuses one."""
        return self.decoder.snapshot(into, device=device)

    def restore(self, snap: Snapshot) -> None:
        self.decoder.restore(snap)

    def kv_export(self, start: int, stop: int, into: KVChunk | None = None) -> KVChunk:
        return self.decoder.kv_export(start, stop, into)

    def kv_import(self, chunk: KVChunk, start: int) -> None:
        self.decoder.kv_import(chunk, start)


def compile(
    source: Source | str | Path,
    *,
    ctx_max: int = 4096,
    rows_max: int = STEP_ROWS,
    kv: str = "bf16",
    backend: Backend | None = None,
    speculate: int = 0,
    host_tables: bool = False,
    window_rows: int = 0,
) -> Model:
    """Compile a checkpoint (a Source, a *.gguf file or an HF directory) for greedy decode with a KV cache of `ctx_max`
    positions; prompt steps take up to `rows_max` rows. `backend`: the kernels to lower onto (default CudaBackend()).

    speculate = k_max > 0 builds the MTP head (the checkpoint's `mtp.*` tensors, re-quantised to W4G128) and the cycle
    programs for k = 1 .. k_max; `Model.decoder.k` picks the drafts per cycle (default k_max). It needs a backend whose
    DeltaNet keeps pending rows (the swiftmk kernels): the model's step is verified k + 1 rows at a time and a partly
    accepted step must roll back by not advancing. host_tables=True keeps the embedding and the rescoring head, which
    the kernels read by row gather only, in pinned host memory: 4.7 GiB less device memory, 0 - 2% of decode speed;
    never chosen automatically. The memory the build will allocate is estimated first: a context that cannot fit
    raises ValueError before anything is loaded.

    window_rows = N (17 .. 4096) adds the window prefill: the same graph lowered a second time at N rows against the
    swiftmk window kernels (the hand engine's fast prefill: W4 dequantisation + cuBLAS, fla's DeltaNet scan,
    FlashAttention-2; needs `warpc[prefill]`), run eagerly by `append(ids, ends=...)` for chunks of more than 8 rows.
    Its scratch is about 1.2 GiB at 4,096 rows plus 4 KiB per context position. Without a backend it selects
    `SwiftmkBackend(window=True)`."""
    if isinstance(source, str | Path):
        source = sources.open(source)
    if window_rows:
        if not 17 <= window_rows <= min(4096, ctx_max):
            raise ValueError(f"window_rows {window_rows}: 17 .. min(4096, ctx_max)")
        require_libraries()
        backend = SwiftmkBackend(window=True) if backend is None else backend
        if "w_attn" not in backend.caps.by_kernel:
            raise ValueError(
                f"window_rows needs a backend with window kernels, SwiftmkBackend(window=True), not {backend.name}"
            )
    backend = CudaBackend() if backend is None else backend
    graph = frontend.build(source, ctx_max=ctx_max, rows_max=rows_max, kv=kv)
    plan = lower(graph, backend, rows_max=rows_max)
    window = None
    if window_rows:
        wgraph = graph.pruned(["hidden"])
        window = (wgraph, lower(wgraph, backend, rows_max=window_rows))
    extra = estimate_bytes(window[1], ctx_max) if window else 0
    arena = Arena(host_gather=host_tables)
    if not speculate:
        arena.claim([plan] if window is None else [plan, window[1]])
        parts = [memory.Part(graph, plan, backend.executor(graph, plan, source), 9)]
        memory.preflight(parts, arena, ctx_max=ctx_max, extra=extra)
        runtime = Runtime(graph, plan, source, backend, arena, window=window)
        arena.freeze()
        runtime.finalize()
        return Model(source.config, plan, runtime, runtime)
    if not 1 <= speculate < rows_max:
        raise ValueError(f"speculate {speculate}: 1 .. rows_max - 1 = {rows_max - 1} drafts per cycle")
    inplace = {t.kernel for t in plan.tasks if any(r.endswith("rec") for r in t.states)} - ROLLBACK_KERNELS
    if inplace:
        raise ValueError(
            f"speculation needs a backend whose DeltaNet keeps pending rows; {sorted(inplace)} update it in place"
        )
    draft_source = Requantized.mtp(source)
    draft_graph = frontend.build_draft(draft_source, ctx_max=ctx_max, rows_max=rows_max)
    draft_plan = lower(draft_graph, backend, rows_max=rows_max)
    arena.claim([plan, draft_plan] + ([window[1]] if window else []))
    target_variants = [(T, ()) for T in range(1, rows_max + 1)] + [(T, ("hidden",)) for T in range(1, rows_max + 1)]
    target_variants += [(1, ("ids",)), (1, ("ids", "hidden"))]
    draft_variants = [(T, ()) for T in range(1, rows_max + 1)]
    cycle_graphs = 2 * speculate
    memory.preflight(
        [
            memory.Part(graph, plan, backend.executor(graph, plan, source), len(target_variants) + cycle_graphs),
            memory.Part(draft_graph, draft_plan, backend.executor(draft_graph, draft_plan, draft_source), 8),
        ],
        arena,
        ctx_max=ctx_max,
        extra=extra,
    )
    target = Runtime(graph, plan, source, backend, arena, variants=target_variants, window=window)
    draft = Runtime(draft_graph, draft_plan, draft_source, backend, arena, variants=draft_variants)
    spec = Speculator(target, draft, k_max=speculate)
    arena.freeze()
    target.finalize()
    draft.finalize()
    spec.finalize()
    return Model(source.config, plan, target, spec)
