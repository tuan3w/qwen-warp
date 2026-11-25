"""Qwen3.5 hybrid graph: Gated DeltaNet layers and gated full-attention layers (config.layer_types), zero-centred
RMSNorms, partial rope, SwiGLU MLP; linears W4G128 where the checkpoint packs them."""

from __future__ import annotations

from ..ir import Graph, StateDecl
from ..sources import Source
from . import common


def build(source: Source, *, ctx_max: int, rows_max: int, kv: str) -> Graph:
    c = source.config
    la = c.linear
    H, FF, hd, rd = c.hidden_size, c.intermediate_size, c.head_dim, c.rotary_dim
    nq, nkv, eps = c.num_heads, c.num_kv_heads, c.rms_eps
    inv_freq = c.rope_inv_freq()
    norm = {"eps": eps, "offset": c.norm_offset}
    missing: list[str] = []
    g = Graph(c.family)
    tokens, pos = common.inputs(g)
    prefix = common.text_prefix(source)

    def weight(decl: str, tensor: str, shape: tuple[int, ...]) -> str:
        return common.weight(g, source, missing, decl, tensor, shape)

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return common.linear(g, source, missing, x, decl, tensor, n, k)

    def rmsnorm(x: str, decl: str, tensor: str, size: int) -> str:
        return common.rmsnorm(g, source, missing, x, decl, tensor, size, **norm)

    def deltanet(h: str, n: str, p: str) -> str:
        kd, vd = la.num_k_heads * la.k_head_dim, la.num_v_heads * la.v_head_dim
        conv_dim = 2 * kd + vd
        qkv = linear(h, f"{n}.qkv", f"{p}.in_proj_qkv.weight", conv_dim, H)
        z = linear(h, f"{n}.z", f"{p}.in_proj_z.weight", vd, H)
        a = linear(h, f"{n}.a", f"{p}.in_proj_a.weight", la.num_v_heads, H)
        b = linear(h, f"{n}.b", f"{p}.in_proj_b.weight", la.num_v_heads, H)
        conv_state = g.state(StateDecl(f"{n}.conv", "conv", (conv_dim, la.conv_kernel - 1)))
        conv_w = weight(f"{n}.conv_w", f"{p}.conv1d.weight", (conv_dim, la.conv_kernel))
        c_ = g.op("silu", [g.op("causal_conv1d", [qkv], weight=conv_w, state=conv_state)])
        q, k, v = g.op_multi("split", [c_], sizes=[kd, kd, vd])
        q = g.op("reshape", [q], shape=["T", la.num_k_heads, la.k_head_dim])
        k = g.op("reshape", [k], shape=["T", la.num_k_heads, la.k_head_dim])
        v = g.op("reshape", [v], shape=["T", la.num_v_heads, la.v_head_dim])
        rec = g.state(StateDecl(f"{n}.rec", "recurrent", (la.num_v_heads, la.k_head_dim, la.v_head_dim)))
        o = g.op(
            "gated_delta_rule",
            [q, k, v, a, b],
            a_log=weight(f"{n}.a_log", f"{p}.A_log", (la.num_v_heads,)),
            dt_bias=weight(f"{n}.dt_bias", f"{p}.dt_bias", (la.num_v_heads,)),
            state=rec,
            eps=1e-6,  # transformers' l2norm default, not rms_norm_eps
            scale=la.k_head_dim**-0.5,
        )
        zr = g.op("reshape", [z], shape=["T", la.num_v_heads, la.v_head_dim])
        nw = weight(f"{n}.gnorm", f"{p}.norm.weight", (la.v_head_dim,))
        o = g.op("gated_rmsnorm", [o, zr], weight=nw, eps=eps)
        return linear(g.op("reshape", [o], shape=["T", vd]), f"{n}.out", f"{p}.out_proj.weight", H, vd)

    def attention(h: str, n: str, p: str) -> str:
        qg = g.op("reshape", [linear(h, f"{n}.q", f"{p}.q_proj.weight", 2 * nq * hd, H)], shape=["T", nq, 2 * hd])
        k = g.op("reshape", [linear(h, f"{n}.k", f"{p}.k_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
        v = g.op("reshape", [linear(h, f"{n}.v", f"{p}.v_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
        q, gate = g.op_multi("split", [qg], sizes=[hd, hd])
        q = rmsnorm(q, f"{n}.qn", f"{p}.q_norm.weight", hd)
        k = rmsnorm(k, f"{n}.kn", f"{p}.k_norm.weight", hd)
        q, k = g.op_multi("rope", [q, k, pos], rotary_dim=rd, inv_freq=inv_freq)
        state = g.state(StateDecl(f"{n}.kv", "kv", (2, ctx_max, nkv, hd), cache=kv))
        scales = {}
        if kv == "fp8":
            scales = {s: weight(f"{n}.{s}", f"{p}.{s}", (1,)) for s in ("k_scale", "v_scale")}
        att = g.op("attention", [q, k, v, pos], state=state, scale=hd**-0.5, **scales)
        gated = g.op("sigmoid_mul", [att, g.op("reshape", [gate], shape=["T", nq * hd])])
        return linear(gated, f"{n}.o", f"{p}.o_proj.weight", H, nq * hd)

    embed = weight("embed", f"{prefix}embed_tokens.weight", (c.vocab_size, H))
    x = g.op("embed", [tokens], weight=embed, name="x")
    for i, kind in enumerate(c.layer_types):
        p, n = f"{prefix}layers.{i}", f"L{i}"
        h = rmsnorm(x, f"{n}.ln1", f"{p}.input_layernorm.weight", H)
        y = attention(h, n, f"{p}.self_attn") if kind == "attention" else deltanet(h, n, f"{p}.linear_attn")
        x = g.op("add", [x, y], name="x")
        h = rmsnorm(x, f"{n}.ln2", f"{p}.post_attention_layernorm.weight", H)
        gate = linear(h, f"{n}.gate", f"{p}.mlp.gate_proj.weight", FF, H)
        up = linear(h, f"{n}.up", f"{p}.mlp.up_proj.weight", FF, H)
        m = g.op("silu_mul", [gate, up])
        x = g.op("add", [x, linear(m, f"{n}.down", f"{p}.mlp.down_proj.weight", H, FF)], name="x")
    common.head(g, source, missing, x, prefix, **norm)
    return g
