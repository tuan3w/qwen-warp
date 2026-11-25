"""Kernel capability records: what a backend kernel absorbs next to its heavy op, as data the fusion pass reads.

A `Cap` is one kernel:

  anchor     the graph op the kernel is built around ("linear", "rmsnorm", "attention", "embed")
  emit       (Match, Analysis) -> Region: binds a legal match to the kernel's launch roles
  segments   how many anchors on one shared input a single launch serves (the q|k|v GEMV: 3); the outputs stay separate;
             with segment_inputs > 1 the anchors may differ in input (one visit's six projections on three inputs)
  pair       ops that merge two same-input anchors into one output (silu_mul on a gate|up pair: the dual GEMV)
  pre        `Absorb` records of producers folded into the kernel's input side (rmsnorm prologue, the residual add in
             front of a norm, the reshape / qk-norm / rope in front of attention)
  post       `Absorb` records of a chain of consumers folded into the output side (residual add in place, cast + argmax
             of the head)
  limits     static kernel limits, evaluated on the constants of a match (K multiple, head dims, rows)
  rounds     per absorbed op, which rounding points of the reference interpreter the kernel reproduces (see REF_POINTS)
  tolerance  the gaps (op.point) a kernel is known to leave open, so a plan containing it is reference-tolerance, not
             bit-exact
  operand_cap  operands (inputs, outputs, weights, states) one launch can bind when that differs from the pass's
               default: a kernel that writes one buffer per layer (the PLE row kernel, 35) takes them as an array

The legality tiers live in the pass (structural, memory) and in `Cap.check` / `Cap.info` (semantic).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# The reference interpreter computes in the activation dtype (bf16) except where an op says fp32 inside, so a bf16
# value is rounded at every op boundary. REF_POINTS[op] names every place the interpreter rounds while computing `op`
# with a bf16 output, "out" being the boundary itself; ops not listed round once, at the boundary; views round nothing.
REF_POINTS: dict[str, tuple[str, ...]] = {
    "silu_mul": ("silu", "out"),  # silu(gate) is a bf16 tensor before the product with up
    "gelu_mul": ("gelu", "out"),  # gelu_tanh(gate) is a bf16 tensor before the product with up
    "rope": ("trig", "prod", "out"),  # cos / sin cast to bf16, x*cos and rot(x)*sin are bf16 tensors, then their sum
    "attention": ("p", "out"),  # the softmax probabilities are rounded to bf16 before P V
    "attention_read": ("p", "out"),
    "sigmoid_mul": ("sigmoid", "out"),  # sigmoid(gate) is a bf16 tensor before the product
    "gated_delta_rule": ("beta", "out"),  # sigmoid(b) is rounded before the fp32 recurrence
    "gated_rmsnorm": ("norm", "scale", "out"),  # bf16(x-hat), bf16(x-hat * w), then the fp32 gate product
    "message_gate": ("input", "hidden"),  # the normed pair and its down projection are bf16 tensors; the output is f32
    "moe_experts": ("gate", "up", "act", "down", "out"),  # g, u, the activation and y are bf16 tensors per expert
}
# Rounding points an op has beyond REF_POINTS when its attrs ask for them: the relative bias is rounded to the
# activation dtype before it joins the scores.
_ATTR_POINTS = {"attention": (("rel_proj", "bias"),)}
# ops whose interior roundings are to the activation dtype whatever dtype they return (an f32 op output has none,
# except these)
FIXED_ROUNDING = frozenset({"message_gate"})
VIEW_OPS = frozenset({"reshape", "split"})
REDUCTION_OPS = frozenset(
    {
        "rmsnorm",
        "argmax",
        "attention",
        "attention_read",
        "gated_rmsnorm",
        "l2_clip",
        "attn_res",
        "message_gate",
        "moe_route",
    }
)
_LIMIT_SUFFIXES = ("_multiple", "_in", "_max", "_min")


def ref_points(op: str, out_dtype: str, attrs: dict | None = None) -> tuple[str, ...]:
    """Rounding points of one interpreter node: none for views; for an output that is not the activation dtype only the
    fixed ones (`FIXED_ROUNDING`, minus the boundary itself); else REF_POINTS or ("out",), plus the points the node's
    attrs add."""
    if op in VIEW_OPS:
        return ()
    points = REF_POINTS.get(op, ("out",))
    if out_dtype == "bf16":
        extra = tuple(p for attr, p in _ATTR_POINTS.get(op, ()) if attrs and attr in attrs)
        return (*extra, *points)
    return tuple(p for p in points if p != "out") if op in FIXED_ROUNDING else ()


@dataclass(frozen=True)
class Target:
    """Cost-model constants of the machine a table's kernels run on."""

    launch_us: float  # fixed cost of one launch inside a CUDA graph
    dram_bandwidth: float  # bytes / s
    sms: int  # streaming multiprocessors (the CTA-utilisation term)


@dataclass(frozen=True)
class Verdict:
    """The semantic tier's answer on one match: why the kernel cannot run it (empty: it can), and the CTAs its launch
    uses when the cost model should not guess them."""

    reasons: list[str] = field(default_factory=list)
    ctas: int | None = None


@dataclass(frozen=True)
class Absorb:
    """One op a kernel folds in; `pre` / `post` of the holding `Cap` says on which side.

    min / max       how many nodes of this op a match absorbs (attention: 3 reshapes, 0 or 2 qk-norms, 1 rope)
    exposes         the node's output is ALSO written to memory (the residual stream after the add, the logits before
                    the argmax)
    inplace         the node updates one of its operands in place (residual add): the memory tier proves nobody reads
                    it afterwards
    reduction_ok    the op is a reduction the kernel really implements (reductions in an epilogue are illegal by
                    default)
    attrs           attribute values the node must carry (cast: {"dtype": "f32"})
    """

    op: str
    min: int = 0
    max: int = 1
    exposes: bool = False
    inplace: bool = False
    reduction_ok: bool = False
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Cap:
    kernel: str
    anchor: str
    emit: Callable  # (Match, Analysis) -> Region
    anchor_schemes: tuple[str, ...] = ()  # weight schemes a linear anchor may read: empty = any
    segments: int = 1
    segment_inputs: int = 1  # > 1: the segments may read different inputs (pairwise independent anchors)
    pair: tuple[str, ...] = ()
    pre: tuple[Absorb, ...] = ()
    post: tuple[Absorb, ...] = ()
    limits: dict = field(default_factory=dict)  # see `limits_ok`
    rounds: dict[str, tuple[str, ...]] = field(default_factory=dict)
    tolerance: tuple[str, ...] = ()  # acknowledged gaps "op.point"
    launches: int = 1  # launches one task of this kernel makes
    constants: Callable | None = None  # (Match, Analysis) -> dict of graph constants the limits and the cost model read
    info: Callable | None = None  # (constants) -> Verdict; None = limits only
    check: Callable | None = None  # semantic predicate (Match, Analysis) -> reason string or None
    operand_cap: int | None = None  # None: fuse.OPERAND_CAP

    def __post_init__(self):
        for key in self.limits:
            if not key.endswith(_LIMIT_SUFFIXES):
                raise ValueError(f"{self.kernel}: unknown limit {key}")

    def ops(self) -> set[str]:
        return {self.anchor, *self.pair, *(a.op for a in self.pre), *(a.op for a in self.post)}

    def evaluate(self, c: dict) -> Verdict:
        reasons = limits_ok(self.kernel, self.limits, c)
        if self.info is None:
            return Verdict(reasons)
        extra = self.info(c)
        return Verdict(reasons + extra.reasons, extra.ctas)

    def gaps(self, op: str, out_dtype: str, attrs: dict | None = None) -> tuple[str, ...]:
        """Reference rounding points of `op` this kernel does not reproduce when it absorbs the op."""
        have = set(self.rounds.get(op, ()))
        return tuple(p for p in ref_points(op, out_dtype, attrs) if p not in have)


def limits_ok(kernel: str, limits: dict, c: dict) -> list[str]:
    """Evaluate static limits on constants: `<c>_multiple`, `<c>_in`, `<c>_max`, `<c>_min`. A limit on a constant the
    match does not define is a table error, not a pass."""
    bad = []
    for key, lim in limits.items():
        suffix = next(s for s in _LIMIT_SUFFIXES if key.endswith(s))
        name = key[: -len(suffix)]
        if name not in c:
            raise ValueError(f"{kernel}: limit {key} reads a constant {name!r} the kernel's constants do not define")
        v = c[name]
        if suffix == "_multiple" and v % lim:
            bad.append(f"{name}={v} is not a multiple of {lim}")
        elif suffix == "_in" and v not in lim:
            bad.append(f"{name}={v} not in {tuple(lim)}")
        elif suffix == "_max" and v > lim:
            bad.append(f"{name}={v} > {lim}")
        elif suffix == "_min" and v < lim:
            bad.append(f"{name}={v} < {lim}")
    return bad


class CapsTable:
    """The ordered capabilities of one backend (order = cost-model tie-break) and the target its cost model prices.

    unfused_ctas      (node, Analysis) -> CTAs the backend's kernel for that node alone would launch
    unfused_launches  op -> launches the backend makes for that node alone (default 1; a view is 0)
    prefer_rank       of two capabilities that match the same nodes, the earlier one wins whatever the cost model says
                      (a composite table lists its preferred kernels first: their gain estimates need not be comparable)
    """

    def __init__(
        self,
        caps: list[Cap],
        target: Target,
        unfused_ctas: Callable,
        unfused_launches: dict[str, int] | None = None,
        prefer_rank: bool = False,
    ):
        self.caps = list(caps)
        self.target = target
        self.by_kernel = {c.kernel: c for c in self.caps}
        if len(self.by_kernel) != len(self.caps):
            raise ValueError("duplicate kernel in capability table")
        self.rank = {c.kernel: i for i, c in enumerate(self.caps)}
        self.unfused_ctas = unfused_ctas
        self.unfused_launches = dict(unfused_launches or {})
        self.prefer_rank = prefer_rank

    def ops(self) -> set[str]:
        return set().union(*(c.ops() for c in self.caps)) if self.caps else set()
