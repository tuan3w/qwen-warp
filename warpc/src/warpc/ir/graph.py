"""The semantic graph: one forward window of T rows (T is the symbol "T"), so prefill and decode are the same graph.

Cross-call memory is only through declared states, and every op that touches a state names it in its attrs.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

Dim = int | str
DTYPES = ("bf16", "f32", "i32")
SCHEMES = ("dense", "q4_0")
# GGUF Q4_0 block: one fp16 scale and 32 4-bit codes
Q4_0_BLOCK_ELEMS = 32
Q4_0_BLOCK_BYTES = 18


@dataclass(frozen=True)
class TType:
    shape: tuple[Dim, ...]
    dtype: str

    def __post_init__(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unknown dtype {self.dtype!r}")


@dataclass(frozen=True)
class WeightDecl:
    """A read-only tensor of logical shape `shape`, read from `source` (a full HF tensor name).

    scheme "dense": the source tensor in bf16; "q4_0": the source's on-disk GGUF Q4_0 blocks (`Source.raw`).
    """

    name: str
    shape: tuple[int, ...]
    scheme: str
    source: str

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"weight {self.name}: unknown scheme {self.scheme!r}")

    @property
    def nbytes(self) -> int:
        """Bytes of the weight on device in its scheme."""
        if self.scheme == "q4_0":
            return self.shape[0] * self.shape[1] // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES
        return 2 * math.prod(self.shape)


@dataclass(frozen=True)
class StateDecl:
    """A positional KV cache [2 (K, V), ctx, Hkv, hd] of format `cache` (a name in `warpc.kvcache.FORMATS`)."""

    name: str
    shape: tuple[int, ...]
    cache: str = "bf16"


@dataclass
class Node:
    op: str
    ins: list[str]
    outs: list[str]
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OpDef:
    """An op: `infer(graph, attrs, input types) -> output types`, raising ValueError on ill-typed use."""

    name: str
    infer: Callable[[Graph, dict, list[TType]], list[TType]]


OPS: dict[str, OpDef] = {}


class Graph:
    def __init__(self, name: str):
        self.name = name
        self.inputs: dict[str, TType] = {}
        self.weights: dict[str, WeightDecl] = {}
        self.states: dict[str, StateDecl] = {}
        self.nodes: list[Node] = []
        self.types: dict[str, TType] = {}
        self.outputs: dict[str, str] = {}  # alias -> value name
        self._count = 0

    def input(self, name: str, t: TType) -> str:
        if name in self.types:
            raise ValueError(f"duplicate value {name}")
        self.inputs[name] = t
        self.types[name] = t
        return name

    def weight(self, w: WeightDecl) -> str:
        if w.name in self.weights:
            raise ValueError(f"duplicate weight {w.name}")
        self.weights[w.name] = w
        return w.name

    def state(self, s: StateDecl) -> str:
        if s.name in self.states:
            raise ValueError(f"duplicate state {s.name}")
        self.states[s.name] = s
        return s.name

    def op(self, op: str, ins: list[str], **attrs) -> str:
        """Append a single-output node; returns its output value name. attrs["name"] only prefixes that name."""
        outs = self.op_multi(op, ins, **attrs)
        if len(outs) != 1:
            raise ValueError(f"{op} has {len(outs)} outputs: use op_multi")
        return outs[0]

    def op_multi(self, op: str, ins: list[str], **attrs) -> tuple[str, ...]:
        """Append a node typed by the op's `infer`; returns its output value names in order."""
        if op not in OPS:
            raise ValueError(f"unknown op {op}")
        for v in ins:
            if v not in self.types:
                raise ValueError(f"{op}: unknown input value {v}")
        base = attrs.pop("name", None) or op
        otypes = OPS[op].infer(self, attrs, [self.types[v] for v in ins])
        outs = []
        for i, t in enumerate(otypes):
            self._count += 1
            v = f"{base}.{self._count}" if len(otypes) == 1 else f"{base}.{self._count}.{i}"
            if v in self.types:
                raise ValueError(f"{op}: duplicate value name {v}")
            self.types[v] = t
            outs.append(v)
        self.nodes.append(Node(op, list(ins), outs, attrs))
        return tuple(outs)

    def output(self, **values: str) -> None:
        for alias, v in values.items():
            if v not in self.types:
                raise ValueError(f"unknown output {v}")
            self.outputs[alias] = v

    def verify(self) -> None:
        """Every input defined before use, every stored type equal to a fresh inference, every output defined."""
        seen = set(self.inputs)
        for n in self.nodes:
            for v in n.ins:
                if v not in seen:
                    raise ValueError(f"{n.op}: {v} used before definition")
            if [self.types[o] for o in n.outs] != OPS[n.op].infer(self, n.attrs, [self.types[v] for v in n.ins]):
                raise ValueError(f"{n.op}: stored output types differ from inference")
            seen.update(n.outs)
        for v in self.outputs.values():
            if v not in seen:
                raise ValueError(f"output {v} never defined")
