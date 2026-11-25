"""Dense decoder graph (llama, mistral, qwen3): GQA attention with rope, optional per-head q/k RMSNorm, SwiGLU MLP."""

from __future__ import annotations

from ..ir import Graph, StateDecl
from ..sources import Source
from . import common


def build(source: Source, *, ctx_max: int, kv: str) -> Graph:
    c = source.config
    H, FF, hd = c.hidden_size, c.intermediate_size, c.head_dim
    nq, nkv, eps = c.num_heads, c.num_kv_heads, c.rms_eps
    inv_freq = c.rope_inv_freq()
    missing: list[str] = []
    g = Graph(c.family)
    tokens, pos = common.inputs(g)

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return common.linear(g, source, missing, x, decl, tensor, n, k)

    def rmsnorm(x: str, decl: str, tensor: str, size: int) -> str:
        return common.rmsnorm(g, source, missing, x, decl, tensor, size, eps=eps)

    embed = common.weight(g, source, missing, "embed", "model.embed_tokens.weight", (c.vocab_size, H))
    x = g.op("embed", [tokens], weight=embed, name="x")
    for i in range(c.num_layers):
        p, n = f"model.layers.{i}", f"L{i}"
        state = g.state(StateDecl(f"{n}.kv", "kv", (2, ctx_max, nkv, hd), cache=kv))
        h = rmsnorm(x, f"{n}.ln1", f"{p}.input_layernorm.weight", H)
        q = g.op("reshape", [linear(h, f"{n}.q", f"{p}.self_attn.q_proj.weight", nq * hd, H)], shape=["T", nq, hd])
        k = g.op("reshape", [linear(h, f"{n}.k", f"{p}.self_attn.k_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
        v = g.op("reshape", [linear(h, f"{n}.v", f"{p}.self_attn.v_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
        if c.qk_norm:
            q = rmsnorm(q, f"{n}.qn", f"{p}.self_attn.q_norm.weight", hd)
            k = rmsnorm(k, f"{n}.kn", f"{p}.self_attn.k_norm.weight", hd)
        q, k = g.op_multi("rope", [q, k, pos], rotary_dim=hd, inv_freq=inv_freq)
        att = g.op("attention", [q, k, v, pos], state=state, scale=hd**-0.5)
        x = g.op("add", [x, linear(att, f"{n}.o", f"{p}.self_attn.o_proj.weight", H, nq * hd)], name="x")
        h = rmsnorm(x, f"{n}.ln2", f"{p}.post_attention_layernorm.weight", H)
        gate = linear(h, f"{n}.gate", f"{p}.mlp.gate_proj.weight", FF, H)
        up = linear(h, f"{n}.up", f"{p}.mlp.up_proj.weight", FF, H)
        m = g.op("silu_mul", [gate, up])
        x = g.op("add", [x, linear(m, f"{n}.down", f"{p}.mlp.down_proj.weight", H, FF)], name="x")
    common.head(g, source, missing, x, "model.", eps=eps)
    return g
