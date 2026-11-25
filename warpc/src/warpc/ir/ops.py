"""The op vocabulary: shape/dtype inference per op; `warpc.reference` defines the semantics.

An op that reads weights names them in the attrs its OpDef lists in `weights` (most: attrs["weight"]); an op that reads
or writes a state names it in attrs["state"].
"""

from __future__ import annotations

from .graph import OPS, Graph, OpDef, StateDecl, TType, WeightDecl

__all__ = ["OPS"]


def _op(
    name: str,
    *,
    weights: tuple[str, ...] = ("weight",),
    states: tuple[str, ...] = (),
    state_kind: str = "",
    reads: tuple[str, ...] = (),
):
    def register(fn):
        OPS[name] = OpDef(name, fn, weights, states, state_kind, reads)
        return fn

    return register


def _weight(g: Graph, attrs: dict, key: str = "weight") -> WeightDecl:
    name = attrs[key]
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
    """tokens i32[T] -> rows of the [V, H] table times attrs["scale"] (default 1.0; the scale is rounded to the
    activation dtype first, then the product), bf16 [T, H]."""
    (t,) = ins
    w = _weight(g, a)
    _need(t.dtype == "i32" and len(t.shape) == 1 and len(w.shape) == 2, "embed: tokens i32[T], table [V, H]")
    return [TType((t.shape[0], w.shape[1]), "bf16")]


@_op("rmsnorm")
def _rmsnorm(g, a, ins):
    """x * rsqrt(mean(x^2, -1) + eps) * (offset + w) over the last dim (attrs["offset"], default 0.0); fp32 inside,
    result in x dtype. Without attrs["weight"] the result is x-hat alone (no offset either)."""
    (x,) = ins
    if "weight" not in a:
        _need("offset" not in a, "rmsnorm: an offset needs the weight")
        return [x]
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


@_op("add", weights=())
def _add(g, a, ins):
    """Elementwise a + b in the activation dtype."""
    x, y = ins
    _need(x == y, f"add: {x} vs {y}")
    return [x]


@_op("silu_mul", weights=())
def _silu_mul(g, a, ins):
    """silu(gate) * up."""
    x, y = ins
    _need(x == y, f"silu_mul: {x} vs {y}")
    return [x]


@_op("gelu_mul", weights=())
def _gelu_mul(g, a, ins):
    """gelu_tanh(gate) * up: gelu_tanh(gate) is rounded to the activation dtype before the product."""
    x, y = ins
    _need(x == y, f"gelu_mul: {x} vs {y}")
    return [x]


@_op("scale", weights=("weight",))
def _scale(g, a, ins):
    """x * attrs["factor"] (a float) xor x * w[0] (attrs["weight"], a [1] weight), fp32 inside, one rounding."""
    (x,) = ins
    _need(("factor" in a) != ("weight" in a), "scale: exactly one of factor and weight")
    if "weight" in a:
        _need(_weight(g, a).shape == (1,), "scale: weight must be a [1] weight")
    return [x]


@_op("softcap", weights=())
def _softcap(g, a, ins):
    """cap * tanh(x / cap), fp32 inside, one rounding to x's dtype (an fp32 x stays fp32)."""
    (x,) = ins
    _need(a["cap"] > 0, f"softcap: cap {a['cap']}")
    return [x]


@_op("silu", weights=())
def _silu(g, a, ins):
    """x * sigmoid(x), fp32 inside, one rounding."""
    (x,) = ins
    return [x]


@_op("sigmoid_mul", weights=())
def _sigmoid_mul(g, a, ins):
    """x * sigmoid(gate): sigmoid(gate) is rounded to the activation dtype before the product."""
    x, y = ins
    _need(x == y, f"sigmoid_mul: {x} vs {y}")
    return [x]


@_op("reshape", weights=())
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


@_op("split", weights=())
def _split(g, a, ins):
    """x[..., sum(sizes)] -> one output [..., s] per s in attrs["sizes"], consecutive slices of the last dim."""
    (x,) = ins
    sizes = tuple(a["sizes"])
    _need(len(sizes) >= 2 and all(s > 0 for s in sizes), f"split: sizes {sizes}")
    _need(sum(sizes) == x.shape[-1], f"split: sizes {sizes} vs last dim {x.shape[-1]}")
    return [TType((*x.shape[:-1], s), x.dtype) for s in sizes]


@_op("cast", weights=())
def _cast(g, a, ins):
    """Dtype conversion to attrs["dtype"]."""
    (x,) = ins
    return [TType(x.shape, a["dtype"])]


@_op("argmax", weights=())
def _argmax(g, a, ins):
    """Last-dim argmax -> i32."""
    (x,) = ins
    return [TType(x.shape[:-1], "i32")]


@_op("rope", weights=())
def _rope(g, a, ins):
    """Rotate the first rotary_dim dims of q [T, Hq, D] and k [T, Hkv, D] (half-split pairs) at positions pos i32[T]:
    pair j (dims j and j + rotary_dim / 2) by the angle pos * inv_freq[j], inv_freq a tuple of rotary_dim / 2 fp32
    values. (q, pos) rotates q alone: the same function of one tensor."""
    q, *k, pos = ins
    _need(len(k) <= 1, "rope: (q, k, pos) or (q, pos)")
    _need(
        pos.dtype == "i32"
        and pos.shape == (q.shape[0],)
        and a["rotary_dim"] <= q.shape[-1]
        and a["rotary_dim"] % 2 == 0
        and len(a["inv_freq"]) == a["rotary_dim"] // 2,
        f"rope: q {q.shape}, pos {pos.shape}, rotary_dim {a['rotary_dim']}, {len(a['inv_freq'])} inverse frequencies",
    )
    return [q, *k]


def _window(a: dict, s: StateDecl, op: str) -> None:
    window = a.get("window")
    _need(window is None or (isinstance(window, int) and window > 0), f"{op}: window {window!r}")
    _need(window is None or s.cache == "bf16", f"{op}: a window needs a bf16 cache, not {s.cache}")


@_op("attention", weights=("k_scale", "v_scale"), states=("state",), state_kind="kv")
def _attention(g, a, ins):
    """Write k, v at positions pos into the KV state (in its cache format), then causal softmax attention of row t
    over cache[0..pos_t]; GQA; fp32 softmax. Cache format fp8 reads its per-tensor scales from the [1] weights
    attrs["k_scale"] / attrs["v_scale"], which no other format takes. attrs["window"] (int, default None): row t sees
    only the positions p with 0 <= pos_t - p < window. A state with R rows holds position p in slot p mod R; the step
    needs R >= window + rows - 1 (the write precedes the read), else the reference raises."""
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
    _window(a, s, "attention")
    scales = [key for key in ("k_scale", "v_scale") if key in a]
    if s.cache == "fp8":
        _need(len(scales) == 2, "attention: cache fp8 needs the k_scale and v_scale weights")
        for key in scales:
            _need(_weight(g, a, key).shape == (1,), f"attention: {key} must be a [1] weight")
    else:
        _need(not scales, f"attention: cache {s.cache} takes no scales")
    return [TType((q.shape[0], hq * d), q.dtype)]


@_op("attention_read", weights=(), reads=("state",), state_kind="kv")
def _attention_read(g, a, ins):
    """`attention` without the write: row t attends over a bf16 state an earlier node of the graph wrote (the owner
    of a layer that shares its KV), same attrs["scale"] / attrs["window"]."""
    q, pos = ins
    s = _state(g, a)
    hq, d = q.shape[1], q.shape[2]
    _need(
        len(s.shape) == 4 and s.shape[0] == 2 and s.shape[3] == d and hq % s.shape[2] == 0 and pos.dtype == "i32",
        f"attention_read: state {s.shape}, q {q.shape}",
    )
    _need(s.cache == "bf16", f"attention_read: cache {s.cache} (only bf16)")
    _window(a, s, "attention_read")
    return [TType((q.shape[0], hq * d), q.dtype)]


@_op("causal_conv1d", states=("state",), state_kind="conv")
def _causal_conv1d(g, a, ins):
    """Depthwise causal conv over time, no bias, no activation: y[t, c] = sum_j w[c, j] * u[t + j, c] over the rows
    u = [state (the W-1 inputs before this window, oldest first) | x], fp32 inside, one rounding; the state then holds
    the last W-1 rows of u."""
    (x,) = ins
    w, s = _weight(g, a), _state(g, a)
    _need(
        len(x.shape) == 2 and len(w.shape) == 2 and w.shape[0] == x.shape[1], f"causal_conv1d: x {x.shape}, w {w.shape}"
    )
    _need(w.shape[1] >= 2 and s.shape == (w.shape[0], w.shape[1] - 1), f"causal_conv1d: state {s.shape}, w {w.shape}")
    return [x]


@_op("gated_delta_rule", weights=("a_log", "dt_bias"), states=("state",), state_kind="recurrent")
def _gated_delta_rule(g, a, ins):
    """Gated DeltaNet, the recurrent rule row by row in fp32 over the state S [Hv, dk, dv]: beta = sigmoid(b) rounded
    to the activation dtype, g = -exp(a_log) * softplus(a + dt_bias); q, k l2-normalised (attrs["eps"]) and repeated
    Hv / Hk times, q scaled by attrs["scale"]; S = exp(g) S; d = (v - S^T k) beta; S += k d^T; o = S^T q, rounded."""
    q, k, v, ga, gb = ins
    s = _state(g, a)
    _need(len(q.shape) == 3 and q == k, f"gated_delta_rule: q {q.shape} vs k {k.shape}")
    _need(len(v.shape) == 3 and v.shape[0] == q.shape[0], f"gated_delta_rule: v {v.shape}")
    T, hk, dk = q.shape
    _, hv, dv = v.shape
    _need(hv % hk == 0, f"gated_delta_rule: {hv} value heads are not a multiple of {hk} key heads")
    _need(ga.shape == (T, hv) and gb.shape == (T, hv), f"gated_delta_rule: a {ga.shape}, b {gb.shape} vs [T, {hv}]")
    for key in ("a_log", "dt_bias"):
        _need(_weight(g, a, key).shape == (hv,), f"gated_delta_rule: {key} must be [{hv}]")
    _need(s.shape == (hv, dk, dv), f"gated_delta_rule: state {s.shape} vs [{hv}, {dk}, {dv}]")
    return [v]


@_op("gated_rmsnorm")
def _gated_rmsnorm(g, a, ins):
    """bf16(x * rsqrt(mean(x^2, -1) + eps)) * w (rounded), then * silu(z) in fp32, rounded: no offset."""
    x, z = ins
    w = _weight(g, a)
    _need(x == z, f"gated_rmsnorm: x {x} vs z {z}")
    _need(w.shape == (x.shape[-1],), f"gated_rmsnorm: weight {w.shape} vs last dim {x.shape[-1]}")
    return [x]
