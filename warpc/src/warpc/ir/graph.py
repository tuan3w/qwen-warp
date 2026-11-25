"""The semantic graph: one forward window of T rows (T is the symbol "T"), so prefill and decode are the same graph.

Cross-call memory is only through declared states, and every op that touches a state names it in its attrs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

Dim = int | str
DTYPES = ("bf16", "f32", "i32")
# GGUF block formats the fused kernels read, by scheme: (elements, bytes) per block. q4_0: an fp16 scale and 32 4-bit
# codes; q4_1: scale, minimum and 32 codes; q6_k: 256 6-bit codes, 16 int8 sub-scales and an fp16 scale
BLOCK_FORMATS = {"q4_0": (32, 18), "q4_1": (32, 20), "q6_k": (256, 210)}
# the columns a fused kernel takes at a time: a matrix of another width is widened to bf16 instead
K_MULTIPLE = {"q4_0": 512, "q4_1": 512, "q6_k": 256}
SCHEMES = ("dense", *BLOCK_FORMATS, "w4g128", "f32")
# compressed-tensors pack-quantized int4, symmetric: one bf16 scale per 128 columns, no zero point
W4G128_GROUP = 128
STATE_KINDS = ("kv", "conv", "recurrent")
# kv: the element type is the cache format's (warpc.kvcache.FORMATS)
STATE_DTYPE = {"conv": "bf16", "recurrent": "f32"}


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

    scheme "dense": the source tensor in bf16; a `BLOCK_FORMATS` scheme: the source's on-disk GGUF blocks
    (`Source.raw`); "w4g128": the source's int4 codes packed 8 per int32 word and bf16 scales of 128 columns
    (`Source.raw`); "f32": the source's exact fp32 values (`Source.get(name, torch.float32)`), 4 bytes per element on
    the device. "dense" and "w4g128" may be rank 3, [E, N, K]: E stacked expert matrices.
    """

    name: str
    shape: tuple[int, ...]
    scheme: str
    source: str

    def __post_init__(self):
        if self.scheme not in SCHEMES:
            raise ValueError(f"weight {self.name}: unknown scheme {self.scheme!r}")
        if self.scheme == "w4g128" and (len(self.shape) < 2 or self.shape[-1] % W4G128_GROUP):
            raise ValueError(f"weight {self.name}: w4g128 needs a 2-D or 3-D shape with K a multiple of {W4G128_GROUP}")

    @property
    def nbytes(self) -> int:
        """Bytes of the weight on device in its scheme."""
        if self.scheme in BLOCK_FORMATS:
            elems, nbytes = BLOCK_FORMATS[self.scheme]
            return self.shape[0] * self.shape[1] // elems * nbytes
        if self.scheme == "w4g128":
            k = self.shape[-1]
            return math.prod(self.shape[:-1]) * (k // 2 + k // W4G128_GROUP * 2)
        return (4 if self.scheme == "f32" else 2) * math.prod(self.shape)


@dataclass(frozen=True)
class StateDecl:
    """kv: a cache [2 (K, V), R, Hkv, hd] in format `cache` (a name in `warpc.kvcache.FORMATS`): positional when R
    holds every position, else a ring (position p lives in slot p mod R; a window-w attention needs R >= w + rows - 1
    for a step of `rows` rows, and the frontend sizes R = min(ctx_max, w + rows_max - 1)). conv: the last W-1 inputs
    of a depthwise causal conv, [C, W-1], oldest first. recurrent: one matrix per head, [H, dk, dv]. Element type:
    STATE_DTYPE (conv, recurrent) or the cache format (kv)."""

    name: str
    kind: str
    shape: tuple[int, ...]
    cache: str = "bf16"  # kv only

    def __post_init__(self):
        if self.kind not in STATE_KINDS:
            raise ValueError(f"state {self.name}: unknown kind {self.kind!r}")


@dataclass
class Node:
    op: str
    ins: list[str]
    outs: list[str]
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OpDef:
    """An op: `infer(graph, attrs, input types) -> output types`, raising ValueError on ill-typed use. `weights` list
    the attrs that name a graph weight (an absent optional attr names none); `states` the attrs that name a state the
    op writes (an op that writes a cache also reads it); `reads` the attrs that name a state it only reads, one an
    earlier node of the graph wrote; `state_kind` is the kind every named state must have."""

    name: str
    infer: Callable[[Graph, dict, list[TType]], list[TType]]
    weights: tuple[str, ...] = ()
    states: tuple[str, ...] = ()
    state_kind: str = ""
    reads: tuple[str, ...] = ()


OPS: dict[str, OpDef] = {}


def weights_of(n: Node) -> list[str]:
    """The graph weights node `n` reads, in its op's attr order."""
    return [n.attrs[k] for k in OPS[n.op].weights if k in n.attrs]


def states_of(n: Node) -> list[str]:
    """The graph states node `n` writes (and reads)."""
    return [n.attrs[k] for k in OPS[n.op].states if k in n.attrs]


def reads_of(n: Node) -> list[str]:
    """The graph states node `n` only reads."""
    return [n.attrs[k] for k in OPS[n.op].reads if k in n.attrs]


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
        node = Node(op, list(ins), [], attrs)
        self._check_operands(node)
        otypes = OPS[op].infer(self, attrs, [self.types[v] for v in ins])
        for i, t in enumerate(otypes):
            self._count += 1
            v = f"{base}.{self._count}" if len(otypes) == 1 else f"{base}.{self._count}.{i}"
            if v in self.types:
                raise ValueError(f"{op}: duplicate value name {v}")
            self.types[v] = t
            node.outs.append(v)
        self.nodes.append(node)
        return tuple(node.outs)

    def _check_operands(self, n: Node) -> None:
        """Every weight and state the node names is declared, every state of the kind its op expects."""
        d = OPS[n.op]
        for w in weights_of(n):
            if w not in self.weights:
                raise ValueError(f"{n.op}: unknown weight {w}")
        for s in (*states_of(n), *reads_of(n)):
            if s not in self.states:
                raise ValueError(f"{n.op}: unknown state {s}")
            if self.states[s].kind != d.state_kind:
                raise ValueError(f"{n.op}: state {s} is {self.states[s].kind}, the op needs {d.state_kind}")

    def output(self, **values: str) -> None:
        for alias, v in values.items():
            if v not in self.types:
                raise ValueError(f"unknown output {v}")
            self.outputs[alias] = v

    def pruned(self, outputs: Iterable[str]) -> Graph:
        """The graph without the nodes that neither feed the named outputs (aliases) nor write a state: the same
        weights, states and inputs, the kept nodes in order, only those outputs. A window plan lowers it
        without the head."""
        need = {self.outputs[a] for a in outputs}
        keep = []
        for n in reversed(self.nodes):
            if states_of(n) or any(o in need for o in n.outs):
                keep.append(n)
                need.update(n.ins)
        g = Graph(self.name)
        g.inputs, g.weights, g.states, g.types = self.inputs, self.weights, self.states, self.types
        g.nodes = keep[::-1]
        g.outputs = {a: self.outputs[a] for a in outputs}
        g._count = self._count
        return g

    def verify(self) -> None:
        """Every input defined before use, every named weight and state declared (states of the op's kind), every
        state an op only reads written by an earlier node, every stored type equal to a fresh inference, every output
        defined."""
        seen, written = set(self.inputs), set()
        for n in self.nodes:
            for v in n.ins:
                if v not in seen:
                    raise ValueError(f"{n.op}: {v} used before definition")
            self._check_operands(n)
            for s in reads_of(n):
                if s not in written:
                    raise ValueError(f"{n.op}: state {s} is read before any node of the graph writes it")
            if [self.types[o] for o in n.outs] != OPS[n.op].infer(self, n.attrs, [self.types[v] for v in n.ins]):
                raise ValueError(f"{n.op}: stored output types differ from inference")
            seen.update(n.outs)
            written.update(states_of(n))
        for v in self.outputs.values():
            if v not in seen:
                raise ValueError(f"output {v} never defined")
