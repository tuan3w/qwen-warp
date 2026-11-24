"""The swiftmk backend: the hand engine's kernel library (libswiftmk.so) as a warpc backend, for parity with the hand
engine (stage 1 of the Qwen3.5 work). Steps take at most 8 rows (the DeltaNet and attention entries' limit)."""

from __future__ import annotations

from ..compiler import Plan
from ..ir import Graph
from ..sources import Source
from . import lib
from .executor import SwiftmkExecutor
from .table import HEADS, table


class SwiftmkBackend:
    """head "bf16": the bf16 lm_head + argmax (smk_bf16_gemv_f32, smk_argmax); "int4_refine": an int4 g128 copy of the
    head + exact bf16 rescoring of its top 16 per row (the hand engine's default head)."""

    name = "swiftmk"

    def __init__(self, head: str = "bf16"):
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r} (known: {', '.join(HEADS)})")
        self.head = head
        self.caps = table(head)

    def executor(self, graph: Graph, plan: Plan, source: Source) -> SwiftmkExecutor:
        return SwiftmkExecutor(lib.load(), graph, plan, source)
