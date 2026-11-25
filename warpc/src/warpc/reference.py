"""Reference interpreter: the definition of what each op means, in plain torch. Every lowering is checked against it.

Norms, softmax, the KV cache numerics and the Gated DeltaNet recurrence are fp32 inside; everything else runs in the
activation dtype (bf16 for real models, fp32 in tests where it makes the comparison with transformers tight).

Weights: dense and GGUF block-format (q4_0, q4_1, q6_k) weights are the source's (dequantised) values in the declared
shape; a w4g128 weight stays packed on the device and is dequantised to fp32 per use, so a large checkpoint's reference
holds about its packed size.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .ir import STATE_DTYPE, Graph
from .kvcache import FORMATS
from .sources import Source
from .sources.w4g128 import dequantize

EXPERT_CHUNK = 16  # stacked w4g128 experts dequantised at once


class Interpreter:
    def __init__(self, graph: Graph, source: Source, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.graph, self.device, self.dtype = graph, device, dtype
        self.weights: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for name, w in graph.weights.items():
            if w.scheme == "w4g128":
                self.weights[name] = {k: v.to(device) for k, v in source.raw(w.source).items()}
            elif w.scheme == "f32":
                self.weights[name] = source.get(w.source, torch.float32).reshape(w.shape).to(device)
            else:
                # a q4_0 weight means exactly its decoded values
                self.weights[name] = source.get(w.source).reshape(w.shape).to(device)
        self.states: dict = {}
        self.reset()

    def reset(self) -> None:
        """Empty every state: KV caches, conv windows and recurrent matrices."""
        self.states = {}
        for name, s in self.graph.states.items():
            if s.kind == "kv":
                self.states[name] = FORMATS[s.cache].reference_cache(s.shape, device=self.device, dtype=self.dtype)
            else:
                dtype = self.dtype if STATE_DTYPE[s.kind] == "bf16" else torch.float32
                self.states[name] = torch.zeros(s.shape, dtype=dtype, device=self.device)

    def _torch_dtype(self, name: str) -> torch.dtype:
        return {"bf16": self.dtype, "f32": torch.float32, "i32": torch.int32}[name]

    @torch.no_grad()
    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """One window: graph inputs by name -> graph outputs by alias."""
        env = {name: inputs[name].to(self.device) for name in self.graph.inputs}
        for n in self.graph.nodes:
            outs = _IMPL[n.op](self, n.attrs, *[env[v] for v in n.ins])
            for name, value in zip(n.outs, outs if isinstance(outs, tuple) else (outs,), strict=True):
                env[name] = value
        return {alias: env[v] for alias, v in self.graph.outputs.items()}

    def generate(self, ids: Sequence[int], max_new_tokens: int) -> list[int]:
        """Greedy decode from an empty cache: the prompt as one window, then one row per step."""
        self.reset()
        n = len(ids)
        out = self.run({"tokens": torch.tensor(ids, dtype=torch.int32), "pos": torch.arange(n, dtype=torch.int32)})
        tokens = [int(out["ids"][-1])]
        while len(tokens) < max_new_tokens:
            step = {"tokens": torch.tensor(tokens[-1:], dtype=torch.int32), "pos": torch.tensor([n], dtype=torch.int32)}
            tokens.append(int(self.run(step)["ids"][-1]))
            n += 1
        return tokens


def _embed(it, a, tokens):
    rows = it.weights[a["weight"]][tokens.long()].to(it.dtype)
    # the scale is rounded to the activation dtype first, as transformers does
    return rows if "scale" not in a else rows * torch.tensor(a["scale"], dtype=it.dtype, device=rows.device)


def _normed(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _rmsnorm(it, a, x):
    if "weight" not in a:
        return _normed(x, a["eps"]).to(x.dtype)
    w = it.weights[a["weight"]].float()
    offset = a.get("offset", 0.0)
    return (_normed(x, a["eps"]) * (offset + w if offset else w)).to(x.dtype)


def _linear(it, a, x):
    w = it.weights[a["weight"]]
    if isinstance(w, dict):
        # (nibble - 8) * scale is exact in fp32, not in bf16: one fp32 product, one rounding
        return F.linear(x.float(), dequantize(w["packed"], w["scale"])).to(x.dtype)
    return F.linear(x, w.to(it.dtype))


def _add(it, a, x, y):
    return x + y


def _silu_mul(it, a, gate, up):
    return F.silu(gate) * up


def _gelu_mul(it, a, gate, up):
    return F.gelu(gate, approximate="tanh") * up


def _scale(it, a, x):
    s = it.weights[a["weight"]].float() if "weight" in a else a["factor"]
    return (x.float() * s).to(x.dtype)


def _softcap(it, a, x):
    cap = a["cap"]
    return (cap * torch.tanh(x.float() / cap)).to(x.dtype)


def _silu(it, a, x):
    return F.silu(x.float()).to(x.dtype)


def _sigmoid_mul(it, a, x, gate):
    return x * torch.sigmoid(gate)


def _reshape(it, a, x):
    return x.reshape([x.shape[0] if d == "T" else d for d in a["shape"]])


def _split(it, a, x):
    return tuple(torch.split(x, list(a["sizes"]), -1))


def _cast(it, a, x):
    return x.to(it._torch_dtype(a["dtype"]))


def _argmax(it, a, x):
    return x.argmax(-1).to(torch.int32)


def _concat(it, a, x, y):
    return torch.cat([x, y], -1)


def _select_row(it, a, x, i):
    return x[i.long()]


def _take(it, a, i):
    return torch.tensor(a["table"], dtype=torch.int32, device=i.device)[i.long()]


def _rope(it, a, q, *rest):
    *ks, pos = rest
    rd = a["rotary_dim"]
    inv = torch.tensor(a["inv_freq"], device=q.device, dtype=torch.float32)
    freqs = pos.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], -1)
    cos, sin = emb.cos().to(q.dtype)[:, None, :], emb.sin().to(q.dtype)[:, None, :]
    half = rd // 2

    def rotate(x):
        xr, xp = x[..., :rd], x[..., rd:]
        rh = torch.cat([-xr[..., half:], xr[..., :half]], -1)
        return torch.cat([xr * cos + rh * sin, xp], -1)

    return (rotate(q), *(rotate(k) for k in ks))


def _window(a: dict) -> dict:
    """The window keyword, only when set: caches without window support take none."""
    return {} if a.get("window") is None else {"window": a["window"]}


def _attention(it, a, q, k, v, pos, *extra):
    cache = it.states[a["state"]]
    p = pos.long()
    scales = {s: it.weights[a[s]] for s in ("k_scale", "v_scale") if s in a}
    options = _window(a)
    if "rel_proj" in a:
        rel, *extra = extra
        options["bias"] = rel @ it.weights[a["rel_proj"]].to(rel.dtype)  # [T, Hq, E], rounded once
    if a.get("sinks"):
        options["sinks"] = extra[0]
    cache.write(p, k, v, **scales)
    return cache.attend(q, p, a["scale"], **scales, **options)


def _attention_read(it, a, q, pos):
    return it.states[a["state"]].attend(q, pos.long(), a["scale"], **_window(a))


def _causal_conv1d(it, a, x):
    state = it.states[a["state"]]
    w = it.weights[a["weight"]].float()  # [C, W]
    width = w.shape[1]
    u = torch.cat([state.t().to(x.dtype), x], 0)  # [W - 1 + T, C], oldest first
    windows = u.float().unfold(0, width, 1)  # [T, C, W]
    state.copy_(u[-(width - 1) :].t())
    y = (windows * w).sum(-1)
    if a.get("residual"):
        y = y + x.float()
    return (y / a.get("divisor", 1.0)).to(x.dtype)


def _dyn_conv1d(it, a, x, taps):
    state = it.states[a["state"]]
    static = it.weights[a["weight"]].float()  # [W, C], row 0 = the current row
    width, c = static.shape
    heads, t = a["heads"], x.shape[0]
    u = torch.cat([state.t().to(x.dtype), x], 0)
    windows = u.float().unfold(0, width, 1).flip(-1)  # [T, C, W], index = lag
    state.copy_(u[-(width - 1) :].t())
    w = static.t().reshape(heads, c // heads, width)[None] + taps.float().reshape(t, heads, 1, width)
    y = (windows.reshape(t, heads, c // heads, width) * w).sum(-1).reshape(t, c)
    return ((y + x.float()) / a["divisor"]).to(x.dtype)


def _l2_clip(it, a, x):
    xf = x.float()
    if "scale" in a:
        xf = xf * it.weights[a["scale"]].float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    return (xf * (a["threshold"] / (norm + 1e-6)).clamp(max=1.0)).to(x.dtype)


def _log_scale(it, a, x, pos):
    gain = 1.0 + a["alpha"] * torch.log(((pos.float() + 1.0) / a["floor"]).clamp(min=1.0))
    return (x.float() * gain[:, None, None]).to(x.dtype)


def _attn_res(it, a, *rows):
    keys = _normed(torch.stack([r.float() for r in rows], 1), a["eps"])  # [T, N, H]
    query = it.weights[a["query"]].float()
    if "row" in a:
        query = query[a["row"]]
    p = torch.softmax(torch.einsum("h,tnh->tn", query, keys) * keys.shape[-1] ** -0.5, -1)
    return torch.einsum("tn,tnh->th", p, keys)


def _message_gate(it, a, x, m):
    n = torch.cat([_normed(x, a["eps"]), _normed(m, a["eps"])], -1).to(it.dtype)
    h = F.linear(n, it.weights[a["down"]].to(it.dtype)).float()
    forget = torch.tanh(F.linear(h, it.weights[a["forget"]].float()))
    gain = torch.tanh(F.linear(h, it.weights[a["gain"]].float()))
    return (1 - forget) * x + gain * m


def _moe_route(it, a, x):
    scores = torch.sigmoid(F.linear(x.float(), it.weights[a["weight"]].float()))
    order = torch.sort(scores + it.weights[a["bias"]].float(), dim=-1, descending=True, stable=True).indices
    idx = order[:, : a["top_k"]]
    w = scores.gather(1, idx)
    total = w[:, 0]
    for j in range(1, w.shape[1]):
        total = total + w[:, j]
    return idx.to(torch.int32), w / (total[:, None] + 1e-20) * a["route_scale"]


def _expert_rows(weight, experts: list[int]) -> torch.Tensor:
    """The matrices of `experts` of a stacked weight: fp32 [n, N, K] dequantised from w4g128, else as stored."""
    idx = torch.as_tensor(
        experts, dtype=torch.long, device=(weight["packed"] if isinstance(weight, dict) else weight).device
    )
    if isinstance(weight, dict):
        return dequantize(weight["packed"][idx], weight["scale"][idx])
    return weight[idx]


def _expert_linear(it, x, rows, packed: bool) -> torch.Tensor:
    """x [m, K] @ rows[N, K]^T: fp32 products and one rounding for dequantised w4g128 rows (as `_linear`)."""
    if packed:
        return F.linear(x.float(), rows).to(x.dtype)
    return F.linear(x, rows.to(it.dtype))


def _moe_experts(it, a, x, idx, w):
    gate, up, down = (it.weights[a[k]] for k in ("gate", "up", "down"))
    packed = isinstance(gate, dict)
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    hit = torch.unique(idx).tolist()  # ascending: the order of the official index_add_
    gate_cap, up_cap = a.get("gate_cap"), a.get("up_cap")
    for c0 in range(0, len(hit), EXPERT_CHUNK):
        chunk = hit[c0 : c0 + EXPERT_CHUNK]
        wg, wu, wd = (_expert_rows(m, chunk) for m in (gate, up, down))
        for j, e in enumerate(chunk):
            tok, slot = torch.where(idx == e)
            cur = x[tok]
            g = _expert_linear(it, cur, wg[j], packed).float()
            u = _expert_linear(it, cur, wu[j], packed).float()
            if gate_cap is None:
                act = g * torch.sigmoid(g) * u
            else:
                act = torch.tanh(g / gate_cap) * gate_cap * torch.sigmoid(g) * (torch.tanh(u / up_cap) * up_cap)
            y = _expert_linear(it, act.to(x.dtype), wd[j], packed)
            out.index_add_(0, tok, y.float() * w[tok, slot, None])
    return out.to(x.dtype)


def _gated_delta_rule(it, a, q, k, v, ga, gb):
    S = it.states[a["state"]]  # [Hv, dk, dv] fp32, updated in place
    hv = v.shape[1]
    rep = hv // q.shape[1]
    beta = torch.sigmoid(gb).float()  # rounded to the activation dtype first
    g = -it.weights[a["a_log"]].float().exp() * F.softplus(ga.float() + it.weights[a["dt_bias"]].float())

    def l2norm(x):
        x = x.float().repeat_interleave(rep, dim=1)
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + a["eps"])

    q, k, vf = l2norm(q) * a["scale"], l2norm(k), v.float()
    out = torch.empty_like(vf)
    s = S.clone()
    for t in range(v.shape[0]):
        s = s * g[t].exp()[:, None, None]
        mem = (s * k[t][:, :, None]).sum(-2)  # S^T k: [Hv, dv]
        d = (vf[t] - mem) * beta[t][:, None]
        s = s + k[t][:, :, None] * d[:, None, :]
        out[t] = (s * q[t][:, :, None]).sum(-2)
    S.copy_(s)
    return out.to(v.dtype)


def _gated_rmsnorm(it, a, x, z):
    n = _normed(x, a["eps"]).to(x.dtype)
    scaled = it.weights[a["weight"]].to(x.dtype) * n
    return (scaled.float() * F.silu(z.float())).to(x.dtype)


_IMPL = {
    "embed": _embed,
    "rmsnorm": _rmsnorm,
    "linear": _linear,
    "add": _add,
    "silu_mul": _silu_mul,
    "gelu_mul": _gelu_mul,
    "scale": _scale,
    "softcap": _softcap,
    "silu": _silu,
    "sigmoid_mul": _sigmoid_mul,
    "reshape": _reshape,
    "split": _split,
    "cast": _cast,
    "argmax": _argmax,
    "concat": _concat,
    "select_row": _select_row,
    "take": _take,
    "rope": _rope,
    "attention": _attention,
    "attention_read": _attention_read,
    "causal_conv1d": _causal_conv1d,
    "dyn_conv1d": _dyn_conv1d,
    "l2_clip": _l2_clip,
    "log_scale": _log_scale,
    "attn_res": _attn_res,
    "message_gate": _message_gate,
    "moe_route": _moe_route,
    "moe_experts": _moe_experts,
    "gated_delta_rule": _gated_delta_rule,
    "gated_rmsnorm": _gated_rmsnorm,
}
