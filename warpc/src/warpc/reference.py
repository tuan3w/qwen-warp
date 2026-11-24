"""Reference interpreter: the definition of what each op means, in plain torch. Every lowering is checked against it.

Norms, softmax and the KV cache numerics are fp32 inside; everything else runs in the activation dtype (bf16 for real
models, fp32 in tests where it makes the comparison with transformers tight).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from .ir import Graph
from .kvcache import FORMATS
from .sources import Source


class Interpreter:
    def __init__(self, graph: Graph, source: Source, *, device: str = "cpu", dtype: torch.dtype = torch.bfloat16):
        self.graph, self.device, self.dtype = graph, device, dtype
        # every scheme reads the source's dequantised tensor: a q4_0 weight means exactly its decoded values
        self.weights = {name: source.get(w.source).to(device) for name, w in graph.weights.items()}
        self.states: dict = {}
        self.reset()

    def reset(self) -> None:
        """Empty every KV cache."""
        self.states = {
            name: FORMATS[s.cache].reference_cache(s.shape, device=self.device, dtype=self.dtype)
            for name, s in self.graph.states.items()
        }

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


def _rmsnorm(it, a, x):
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + a["eps"])
    return (y * it.weights[a["weight"]].float()).to(x.dtype)


def _linear(it, a, x):
    return F.linear(x, it.weights[a["weight"]].to(it.dtype))


def _add(it, a, x, y):
    return x + y


def _silu_mul(it, a, gate, up):
    return F.silu(gate) * up


def _reshape(it, a, x):
    return x.reshape([x.shape[0] if d == "T" else d for d in a["shape"]])


def _cast(it, a, x):
    return x.to(it._torch_dtype(a["dtype"]))


def _argmax(it, a, x):
    return x.argmax(-1).to(torch.int32)


def _rope(it, a, q, k, pos):
    rd, theta = a["rotary_dim"], a["theta"]
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2, device=q.device, dtype=torch.float32) / rd))
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
    cache.write(p, k, v)
    return cache.attend(q, p, a["scale"])


_IMPL = {
    "embed": _embed,
    "rmsnorm": _rmsnorm,
    "linear": _linear,
    "add": _add,
    "silu_mul": _silu_mul,
    "reshape": _reshape,
    "cast": _cast,
    "argmax": _argmax,
    "rope": _rope,
    "attention": _attention,
}
