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


@_op("attention", weights=("k_scale", "v_scale", "rel_proj"), states=("state",), state_kind="kv")
def _attention(g, a, ins):
    """Write k, v at positions pos into the KV state (in its cache format), then causal softmax attention of row t
    over cache[0..pos_t]; GQA; fp32 softmax. Cache format fp8 reads its per-tensor scales from the [1] weights
    attrs["k_scale"] / attrs["v_scale"], which no other format takes. attrs["window"] (int, default None): row t sees
    only the positions p with 0 <= pos_t - p < window. A state with R rows holds position p in slot p mod R; the step
    needs R >= window + rows - 1 (the write precedes the read), else the reference raises. Inkling's two extras, bf16
    caches only: the weight attrs["rel_proj"] [d_rel, E] with the input rel [T, Hq, d_rel] after pos adds the bias
    bias[t, h, d] = (rel[t, h] @ rel_proj)[d], rounded to the activation dtype, to the score of the key d = pos_t - p
    positions back (nothing when d >= E); attrs["sinks"] True with the input sinks [T, Hq] after rel adds one logit
    per row and head to the softmax with a zero value (the probability mass it takes is dropped)."""
    q, k, v, pos, *extra = ins
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
    t = q.shape[0]
    _need(len(extra) == ("rel_proj" in a) + bool(a.get("sinks")), f"attention: {len(extra)} extra inputs for its attrs")
    if extra:
        _need(s.cache == "bf16", f"attention: a relative bias or sinks need a bf16 cache, not {s.cache}")
    if "rel_proj" in a:
        proj, rel = _weight(g, a, "rel_proj"), extra[0]
        _need(
            len(proj.shape) == 2 and rel == TType((t, hq, proj.shape[0]), q.dtype),
            f"attention: rel {rel} vs rel_proj {proj.shape}",
        )
    if a.get("sinks"):
        _need(extra[-1] == TType((t, hq), q.dtype), f"attention: sinks {extra[-1]} vs [{t}, {hq}]")
    return [TType((t, hq * d), q.dtype)]


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
    the last W-1 rows of u. attrs["residual"] (default False) adds x to the sum and attrs["divisor"] (default 1.0)
    divides it, before the rounding: y = (conv + x * residual) / divisor."""
    (x,) = ins
    w, s = _weight(g, a), _state(g, a)
    _need(
        len(x.shape) == 2 and len(w.shape) == 2 and w.shape[0] == x.shape[1], f"causal_conv1d: x {x.shape}, w {w.shape}"
    )
    _need(w.shape[1] >= 2 and s.shape == (w.shape[0], w.shape[1] - 1), f"causal_conv1d: state {s.shape}, w {w.shape}")
    _need(a.get("divisor", 1.0) > 0, "causal_conv1d: divisor must be positive")
    return [x]


@_op("dyn_conv1d", states=("state",), state_kind="conv")
def _dyn_conv1d(g, a, ins):
    """Depthwise causal conv whose taps depend on the row: x [T, C] and taps [T, heads * W]; channel c belongs to head
    c // (C / heads), and tap lag l (0 = the current row) of row t has the weight weight[l, c] + taps[t, head * W + l];
    y[t, c] = (sum_l w_l * u[t - l, c] + x[t, c]) / divisor over u = [state | x] as in causal_conv1d, fp32 inside, one
    rounding."""
    x, taps = ins
    w, s = _weight(g, a), _state(g, a)
    heads = a["heads"]
    width, c = w.shape if len(w.shape) == 2 else (0, 0)
    _need(
        len(x.shape) == 2 and c == x.shape[1] and width >= 2 and c % heads == 0,
        f"dyn_conv1d: x {x.shape}, weight {w.shape}, heads {heads}",
    )
    _need(s.shape == (c, width - 1), f"dyn_conv1d: state {s.shape}, weight {w.shape}")
    _need(taps.shape == (x.shape[0], heads * width) and taps.dtype == x.dtype, f"dyn_conv1d: taps {taps}, x {x}")
    _need(a["divisor"] > 0, "dyn_conv1d: divisor must be positive")
    return [x]


@_op("l2_clip", weights=("scale",))
def _l2_clip(g, a, ins):
    """Per-head L2 clip of x [T, H, D]: x_s = x * scale (the optional [H, D] weight), then x_s * min(1, threshold /
    (|x_s| + 1e-6)) over the last dim, fp32 inside, one rounding."""
    (x,) = ins
    _need(len(x.shape) == 3 and a["threshold"] > 0, f"l2_clip: x {x.shape}, threshold {a.get('threshold')}")
    if "scale" in a:
        _need(_weight(g, a, "scale").shape == tuple(x.shape[1:]), f"l2_clip: scale vs x {x.shape}")
    return [x]


@_op("log_scale", weights=())
def _log_scale(g, a, ins):
    """Position-dependent gain of x [T, H, D] at positions pos i32[T]: x * (1 + alpha * log(max((pos + 1) / floor, 1))),
    fp32 inside, one rounding."""
    x, pos = ins
    _need(len(x.shape) == 3 and pos.dtype == "i32" and pos.shape == (x.shape[0],), f"log_scale: x {x.shape}, pos {pos}")
    return [x]


@_op("attn_res", weights=("query",))
def _attn_res(g, a, ins):
    """Block attention residual read over rows r_1..r_n [T, H] (any dtype, widened to fp32; the last is the partial
    block): normalise each unweighted (attrs["eps"]), p = softmax_n(<query, norm_n> * H^-0.5), out = sum_n p_n *
    norm_n, f32 [T, H]. The weight `query` is [H], or [N, H] with attrs["row"] choosing the row."""
    _need(
        len(ins) >= 1 and all(t.shape == ins[0].shape and len(t.shape) == 2 for t in ins), "attn_res: equal [T, H] rows"
    )
    q = _weight(g, a, "query")
    h = ins[0].shape[-1]
    if "row" in a:
        _need(len(q.shape) == 2 and q.shape[1] == h and 0 <= a["row"] < q.shape[0], f"attn_res: query {q.shape}, row")
    else:
        _need(q.shape == (h,), f"attn_res: query {q.shape} vs H {h}")
    return [TType(ins[0].shape, "f32")]


@_op("message_gate", weights=("down", "forget", "gain"))
def _message_gate(g, a, ins):
    """Merge a loop message m into x (both f32 [T, H]): n = [rms(x); rms(m)] unweighted (attrs["eps"]), rounded to the
    activation dtype; h = n @ down[R, 2H]^T rounded to the activation dtype; f = tanh(h @ forget[H, R]^T), g = tanh(h @
    gain[H, R]^T) in fp32; (1 - f) * x + g * m, f32."""
    x, m = ins
    _need(x == m and x.dtype == "f32" and len(x.shape) == 2, f"message_gate: x {x}, m {m}")
    down, forget, gain = (_weight(g, a, k) for k in ("down", "forget", "gain"))
    r, h = down.shape[0], x.shape[-1]
    _need(down.shape == (r, 2 * h) and forget.shape == (h, r) and gain.shape == (h, r), "message_gate: weight shapes")
    return [x]


@_op("moe_route", weights=("weight", "bias"))
def _moe_route(g, a, ins):
    """Sigmoid router: s = sigmoid(x_f32 @ weight[E, H]^T); the top_k experts by s + bias[E], descending (ties to the
    lower id); w = s[idx] / (sum + 1e-20) * route_scale, the sum a left fold in idx order. idx i32 [T, k], w f32
    [T, k]."""
    (x,) = ins
    w, b = _weight(g, a), _weight(g, a, "bias")
    k = a["top_k"]
    _need(
        len(w.shape) == 2 and w.shape[1] == x.shape[-1] and b.shape == (w.shape[0],) and 0 < k <= w.shape[0],
        f"moe_route: weight {w.shape}, bias {b.shape}, top_k {k}",
    )
    _need(a["scoring"] == "sigmoid", f"moe_route: scoring {a['scoring']!r} (only sigmoid)")
    return [TType((x.shape[0], k), "i32"), TType((x.shape[0], k), "f32")]


@_op("moe_experts", weights=("gate", "up", "down"))
def _moe_experts(g, a, ins):
    """Routed SwiGLU experts, stacked weights gate / up [E, I, H] and down [E, H, I]. For each expert some row routes
    to, in ascending id: g = x @ gate^T and u = x @ up^T (rounded), act = softcap(g, gate_cap) * sigmoid(g) *
    softcap(u, up_cap) in fp32 (both caps None: silu(g) * u), rounded; y = act @ down^T rounded; out_t += y * w_t,e in
    fp32 over the experts; the sum is rounded to x's dtype."""
    x, idx, w = ins
    gate, up, down = (_weight(g, a, k) for k in ("gate", "up", "down"))
    _need(len(gate.shape) == 3 and up.shape == gate.shape, f"moe_experts: gate {gate.shape}, up {up.shape}")
    e, i, h = gate.shape
    _need(down.shape == (e, h, i) and h == x.shape[-1], f"moe_experts: down {down.shape}, x {x.shape}")
    _need(
        idx.dtype == "i32" and w.dtype == "f32" and idx.shape == w.shape and idx.shape[0] == x.shape[0],
        f"moe_experts: idx {idx}, w {w}",
    )
    _need((a.get("gate_cap") is None) == (a.get("up_cap") is None), "moe_experts: both caps or neither")
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
