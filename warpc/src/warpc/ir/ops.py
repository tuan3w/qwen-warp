"""The dense op vocabulary: shape/dtype inference per op; `warpc.reference` defines the semantics.

An op that reads a weight names it in attrs["weight"]; an op that reads or writes a state names it in attrs["state"].
"""

from __future__ import annotations

from .graph import OPS, Graph, OpDef, StateDecl, TType, WeightDecl

__all__ = ["OPS"]


def _op(name: str):
    def register(fn):
        OPS[name] = OpDef(name, fn)
        return fn

    return register


def _weight(g: Graph, attrs: dict) -> WeightDecl:
    name = attrs["weight"]
    if name not in g.weights:
        raise ValueError(f"unknown weight {name}")
    return g.weights[name]


def _state(g: Graph, attrs: dict) -> StateDecl:
    name = attrs["state"]
    if name not in g.states:
        raise ValueError(f"unknown state {name}")
    return g.states[name]


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


@_op("embed")
def _embed(g, a, ins):
    """tokens i32[T] -> rows of the [V, H] table, bf16 [T, H]."""
    (t,) = ins
    w = _weight(g, a)
    _need(t.dtype == "i32" and len(t.shape) == 1 and len(w.shape) == 2, "embed: tokens i32[T], table [V, H]")
    return [TType((t.shape[0], w.shape[1]), "bf16")]


@_op("rmsnorm")
def _rmsnorm(g, a, ins):
    """x * rsqrt(mean(x^2, -1) + eps) * w over the last dim; fp32 inside, result in x dtype."""
    (x,) = ins
    w = _weight(g, a)
    _need(w.shape == (x.shape[-1],), f"rmsnorm: weight {w.shape} vs last dim {x.shape[-1]}")
    return [x]


@_op("linear")
def _linear(g, a, ins):
    """x[..., K] @ W[N, K]^T."""
    (x,) = ins
    w = _weight(g, a)
    _need(len(w.shape) == 2 and w.shape[1] == x.shape[-1], f"linear: weight {w.shape} vs x {x.shape}")
    return [TType((*x.shape[:-1], w.shape[0]), x.dtype)]


@_op("add")
def _add(g, a, ins):
    """Elementwise a + b in the activation dtype."""
    x, y = ins
    _need(x == y, f"add: {x} vs {y}")
    return [x]


@_op("silu_mul")
def _silu_mul(g, a, ins):
    """silu(gate) * up."""
    x, y = ins
    _need(x == y, f"silu_mul: {x} vs {y}")
    return [x]


@_op("reshape")
def _reshape(g, a, ins):
    """Reshape to attrs["shape"] (explicit ints, leading "T" allowed)."""
    (x,) = ins
    shape = tuple(a["shape"])

    def size(s):
        p = 1
        for d in s:
            p *= 1 if d == "T" else d
        return p

    _need(size(shape) == size(x.shape) and (shape[0] == "T") == (x.shape[0] == "T"), f"reshape: {x.shape} -> {shape}")
    return [TType(shape, x.dtype)]


@_op("cast")
def _cast(g, a, ins):
    """Dtype conversion to attrs["dtype"]."""
    (x,) = ins
    return [TType(x.shape, a["dtype"])]


@_op("argmax")
def _argmax(g, a, ins):
    """Last-dim argmax -> i32."""
    (x,) = ins
    return [TType(x.shape[:-1], "i32")]


@_op("rope")
def _rope(g, a, ins):
    """Rotate the first rotary_dim dims of q [T, Hq, D] and k [T, Hkv, D] (half-split pairs) at positions pos i32[T]."""
    q, k, pos = ins
    _need(
        pos.dtype == "i32"
        and pos.shape == (q.shape[0],)
        and a["rotary_dim"] <= q.shape[-1]
        and a["rotary_dim"] % 2 == 0,
        f"rope: q {q.shape}, pos {pos.shape}, rotary_dim {a['rotary_dim']}",
    )
    return [q, k]


@_op("attention")
def _attention(g, a, ins):
    """Write k, v at positions pos into the KV state (in its cache format), then causal softmax attention of row t
    over cache[0..pos_t]; GQA; fp32 softmax."""
    q, k, v, pos = ins
    s = _state(g, a)
    hq, d = q.shape[1], q.shape[2]
    hkv = k.shape[1]
    _need(
        len(s.shape) == 4
        and s.shape[0] == 2
        and s.shape[2:] == (hkv, d)
        and hq % hkv == 0
        and v.shape == k.shape
        and pos.dtype == "i32",
        f"attention: state {s.shape}, q {q.shape}, k {k.shape}, v {v.shape}",
    )
    return [TType((q.shape[0], hq * d), q.dtype)]
