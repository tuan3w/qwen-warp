"""Source -> ir.Graph, one module per model family."""

from __future__ import annotations

from ..ir import Graph
from ..kvcache import FORMATS
from ..sources import Source
from . import dense, gemma4, qwen35

ROWS_MAX = 8  # the default step bound of `compile`
_BUILDERS = {
    "llama": dense.build,
    "mistral": dense.build,
    "qwen3": dense.build,
    "qwen3_5": qwen35.build,
    "gemma4": gemma4.build,
}


def build(source: Source, *, ctx_max: int, rows_max: int = ROWS_MAX, kv: str = "bf16") -> Graph:
    """The graph of one forward window: inputs tokens / pos i32[T], outputs logits f32[T, V] and ids i32[T] (argmax);
    a KV cache of `ctx_max` positions in format `kv` per attention layer. `rows_max` bounds the rows of a step: a
    sliding-window layer keeps a ring of window + rows_max - 1 positions."""
    if kv not in FORMATS:
        raise ValueError(f"unknown KV format {kv!r} (known: {', '.join(FORMATS)})")
    return _BUILDERS[source.config.family](source, ctx_max=ctx_max, rows_max=rows_max, kv=kv)
