"""Source -> ir.Graph, one module per model family."""

from __future__ import annotations

from ..ir import Graph
from ..kvcache import FORMATS
from ..sources import Source
from . import dense, gemma4, inkling, qwen35

ROWS_MAX = 8  # the default step bound of `compile`
_BUILDERS = {
    "llama": dense.build,
    "mistral": dense.build,
    "qwen3": dense.build,
    "qwen3_5": qwen35.build,
    "gemma4": gemma4.build,
    "inkling2_loop": inkling.build,
}


def build(source: Source, *, ctx_max: int, rows_max: int = ROWS_MAX, kv: str = "bf16") -> Graph:
    """The graph of one forward window: inputs tokens / pos i32[T], outputs logits f32[T, V] and ids i32[T] (argmax);
    a KV cache of `ctx_max` positions in format `kv` per attention layer. `rows_max` bounds the rows of a step: a
    sliding-window layer keeps a ring of window + rows_max - 1 positions."""
    if kv not in FORMATS:
        raise ValueError(f"unknown KV format {kv!r} (known: {', '.join(FORMATS)})")
    return _BUILDERS[source.config.family](source, ctx_max=ctx_max, rows_max=rows_max, kv=kv)


def build_draft(source: Source, *, ctx_max: int, rows_max: int = ROWS_MAX, draft_kv: str = "bf16") -> Graph:
    """The multi-token-prediction head of a Qwen3.5 checkpoint, one window of T (token, hidden) pairs: inputs tokens /
    pos i32[T], hidden bf16[T, H], sel i32[1]; outputs hidden bf16[1, H] (the post-norm row `sel`), logits f32[1, NV]
    and ids i32[1]. Its attention layer keeps a KV cache `mtp.kv` of `ctx_max` positions in format `draft_kv`,
    positional like the target's, so `rows_max` sizes nothing."""
    if draft_kv not in FORMATS:
        raise ValueError(f"unknown KV format {draft_kv!r} (known: {', '.join(FORMATS)})")
    return qwen35.build_draft(source, ctx_max=ctx_max, draft_kv=draft_kv)
