"""Reference interpreter: the definition of what each op means, in plain torch. Every lowering is checked against it.

Norms, softmax, the KV cache numerics and the Gated DeltaNet recurrence are fp32 inside; everything else runs in the
activation dtype (bf16 for real models, fp32 in tests where it makes the comparison with transformers tight).

Weights: dense and q4_0 weights are the source's (dequantised) values in the declared shape; a w4g128 weight stays
packed on the device and is dequantised to fp32 per use, so a large checkpoint's reference holds about its packed size.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .ir import STATE_DTYPE, Graph
from .kvcache import FORMATS
from .sources import Source
from .sources.w4g128 import dequantize


class Interpreter:
    def __init__(self, graph: Graph, source: Source, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.graph, self.device, self.dtype = graph, device, dtype
        self.weights: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        for name, w in graph.weights.items():
            if w.scheme == "w4g128":
                self.weights[name] = {k: v.to(device) for k, v in source.raw(w.source).items()}
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
    return it.weights[a["weight"]][tokens.long()].to(it.dtype)


def _normed(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


def _rmsnorm(it, a, x):
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


def _rope(it, a, q, k, pos):
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

    return rotate(q), rotate(k)


def _attention(it, a, q, k, v, pos):
    cache = it.states[a["state"]]
    p = pos.long()
    scales = {s: it.weights[a[s]] for s in ("k_scale", "v_scale") if s in a}
    cache.write(p, k, v, **scales)
    return cache.attend(q, p, a["scale"], **scales)


def _causal_conv1d(it, a, x):
    state = it.states[a["state"]]
    w = it.weights[a["weight"]].float()  # [C, W]
    width = w.shape[1]
    u = torch.cat([state.t().to(x.dtype), x], 0)  # [W - 1 + T, C], oldest first
    windows = u.float().unfold(0, width, 1)  # [T, C, W]
    state.copy_(u[-(width - 1) :].t())
    return (windows * w).sum(-1).to(x.dtype)


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
    "silu": _silu,
    "sigmoid_mul": _sigmoid_mul,
    "reshape": _reshape,
    "split": _split,
    "cast": _cast,
    "argmax": _argmax,
    "rope": _rope,
    "attention": _attention,
    "causal_conv1d": _causal_conv1d,
    "gated_delta_rule": _gated_delta_rule,
    "gated_rmsnorm": _gated_rmsnorm,
}
