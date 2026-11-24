"""compile(): checkpoint -> graph -> plan -> runtime over a backend, behind one Model object."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from . import frontend, sources
from .compiler import Backend, Plan, lower
from .cuda import CudaBackend
from .cuda.schedule import STEP_ROWS
from .runtime import Runtime
from .sources import ModelConfig, Source


class Model:
    """A compiled model: greedy decode on the GPU with a KV cache of `ctx_max` positions."""

    def __init__(self, config: ModelConfig, plan: Plan, runtime: Runtime):
        self.config, self.plan, self._runtime = config, plan, runtime

    def reset(self) -> None:
        """Empty the KV cache."""
        self._runtime.reset()

    def prefill(self, ids: Sequence[int]) -> int:
        """Append the prompt `ids` to the cache; returns the next token (argmax)."""
        return self._runtime.prefill(ids)

    def step(self, token: int) -> int:
        """Append one token; returns the next token (argmax)."""
        return self._runtime.step(token)

    def decode(self, n: int) -> list[int]:
        """n greedy tokens continuing the last prefill or step, with no host round trip per token."""
        return self._runtime.decode(n)

    @property
    def logits(self) -> torch.Tensor:
        """fp32 [vocab] logits of the last step (a view, overwritten by the next one)."""
        return self._runtime.logits

    def generate(self, ids: Sequence[int], max_new_tokens: int) -> list[int]:
        """Greedy continuation of the prompt `ids` from an empty cache: the first `max_new_tokens` generated tokens."""
        self.reset()
        if max_new_tokens <= 0:
            return []
        first = self._runtime.prefill(ids)
        return [first, *self._runtime.decode(max_new_tokens - 1)]


def compile(
    source: Source | str | Path,
    *,
    ctx_max: int = 4096,
    rows_max: int = STEP_ROWS,
    kv: str = "bf16",
    backend: Backend | None = None,
) -> Model:
    """Compile a checkpoint (a Source, a *.gguf file or an HF directory) for greedy decode with a KV cache of `ctx_max`
    positions; prompt steps take up to `rows_max` rows. `backend`: the kernels to lower onto (default CudaBackend())."""
    if isinstance(source, str | Path):
        source = sources.open(source)
    backend = CudaBackend() if backend is None else backend
    graph = frontend.build(source, ctx_max=ctx_max, kv=kv)
    plan = lower(graph, backend, rows_max=rows_max)
    return Model(source.config, plan, Runtime(graph, plan, source, backend))
