"""Gemma 4 text decoder graph (E2B): sliding-window and global attention layers with their own head dims, KV sharing,
per-layer input embeddings (PLE), sandwich norms, a learned scalar per layer, gelu MLP, softcapped logits. The
semantics are transformers' `models/gemma4/modeling_gemma4.py`, read line by line.

Per layer (D = head dim of the layer's type, `owner` = the layer whose KV cache it reads):

  h = ln1(x); q = rope(qn(Wq h)); owner: k = rope(kn(Wk h)), v = rmsnorm_noweight(Wv h) -> `attention` writes the cache
  x += ln2(Wo attention(q, k, v)); x += ln4(Wdown gelu_mul(Wgate ln3(x), Wup ln3(x)))
  x += ln5(Wpproj gelu_mul(Wpgate x, ple_i)); x *= layer_scalar

A layer that shares KV projects q only and runs `attention_read` on its owner's state: the last owning layer of its own
type. Sliding layers keep a ring of min(ctx_max, window + rows_max - 1) positions, global layers every position.
"""

from __future__ import annotations

from ..errors import UnsupportedModel
from ..ir import Graph, StateDecl
from ..sources import Source
from . import common

# Gemma4TextAttention.scaling: the q / k norms carry the scale
ATTN_SCALE = 1.0


def build(source: Source, *, ctx_max: int, rows_max: int, kv: str) -> Graph:
    if kv != "bf16":
        raise UnsupportedModel(
            f"gemma4 with KV format {kv!r} (only bf16: the global layers' head dim 512 has no quantised kernel)"
        )
    c = source.config
    ge = c.gemma
    H, L, eps = c.hidden_size, c.num_layers, c.rms_eps
    nq, nkv, D = c.num_heads, c.num_kv_heads, ge.per_layer_hidden
    missing: list[str] = []
    g = Graph(c.family)
    tokens, pos = common.inputs(g)
    prefix = common.text_prefix(source)

    def weight(decl: str, tensor: str, shape: tuple[int, ...]) -> str:
        return common.weight(g, source, missing, decl, tensor, shape)

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return common.linear(g, source, missing, x, decl, tensor, n, k)

    def rmsnorm(x: str, decl: str, tensor: str, size: int) -> str:
        return common.rmsnorm(g, source, missing, x, decl, tensor, size, eps=eps)

    embed = weight("embed", f"{prefix}embed_tokens.weight", (c.vocab_size, H))
    x = g.op("embed", [tokens], weight=embed, scale=c.embed_scale, name="x")
    ple = None
    if D:
        table = weight("ple_embed", f"{prefix}embed_tokens_per_layer.weight", (ge.per_layer_vocab, L * D))
        tok = g.op("embed", [tokens], weight=table, scale=D**0.5, name="ple_tok")
        ctx = linear(x, "ple_proj", f"{prefix}per_layer_model_projection.weight", L * D, H)
        ctx = g.op("reshape", [g.op("scale", [ctx], factor=H**-0.5)], shape=["T", L, D])
        ctx = g.op(
            "reshape", [rmsnorm(ctx, "ple_norm", f"{prefix}per_layer_projection_norm.weight", D)], shape=["T", L * D]
        )
        ple = g.op_multi("split", [g.op("scale", [g.op("add", [ctx, tok])], factor=2.0**-0.5)], sizes=[D] * L)

    for i, kind in enumerate(c.layer_types):
        p, n = f"{prefix}layers.{i}", f"L{i}"
        full = kind == "full_attention"
        hd = ge.global_head_dim if full else c.head_dim
        window = None if full else c.sliding_window
        owner = ge.owner(c.layer_types, i)
        shared = owner != i
        ff = c.intermediate_size * (2 if ge.double_wide_mlp and shared else 1)
        inv_freq = c.rope_inv_freq(kind)
        h = rmsnorm(x, f"{n}.ln1", f"{p}.input_layernorm.weight", H)
        q = g.op("reshape", [linear(h, f"{n}.q", f"{p}.self_attn.q_proj.weight", nq * hd, H)], shape=["T", nq, hd])
        q = rmsnorm(q, f"{n}.qn", f"{p}.self_attn.q_norm.weight", hd)
        if shared:
            q = g.op("rope", [q, pos], rotary_dim=hd, inv_freq=inv_freq)
            att = g.op("attention_read", [q, pos], state=f"L{owner}.kv", scale=ATTN_SCALE, window=window)
        else:
            rows = ctx_max if window is None else min(ctx_max, window + rows_max - 1)
            state = g.state(StateDecl(f"{n}.kv", "kv", (2, rows, nkv, hd)))
            k = g.op(
                "reshape", [linear(h, f"{n}.k", f"{p}.self_attn.k_proj.weight", nkv * hd, H)], shape=["T", nkv, hd]
            )
            v = g.op(
                "reshape", [linear(h, f"{n}.v", f"{p}.self_attn.v_proj.weight", nkv * hd, H)], shape=["T", nkv, hd]
            )
            k = rmsnorm(k, f"{n}.kn", f"{p}.self_attn.k_norm.weight", hd)
            v = g.op("rmsnorm", [v], eps=eps)
            q, k = g.op_multi("rope", [q, k, pos], rotary_dim=hd, inv_freq=inv_freq)
            att = g.op("attention", [q, k, v, pos], state=state, scale=ATTN_SCALE, window=window)
        o = rmsnorm(
            linear(att, f"{n}.o", f"{p}.self_attn.o_proj.weight", H, nq * hd),
            f"{n}.ln2",
            f"{p}.post_attention_layernorm.weight",
            H,
        )
        x = g.op("add", [x, o], name="x")
        h = rmsnorm(x, f"{n}.ln3", f"{p}.pre_feedforward_layernorm.weight", H)
        gate = linear(h, f"{n}.gate", f"{p}.mlp.gate_proj.weight", ff, H)
        up = linear(h, f"{n}.up", f"{p}.mlp.up_proj.weight", ff, H)
        m = linear(g.op("gelu_mul", [gate, up]), f"{n}.down", f"{p}.mlp.down_proj.weight", H, ff)
        x = g.op("add", [x, rmsnorm(m, f"{n}.ln4", f"{p}.post_feedforward_layernorm.weight", H)], name="x")
        if D:
            pgate = linear(x, f"{n}.pgate", f"{p}.per_layer_input_gate.weight", D, H)
            inj = linear(g.op("gelu_mul", [pgate, ple[i]]), f"{n}.pproj", f"{p}.per_layer_projection.weight", H, D)
            x = g.op("add", [x, rmsnorm(inj, f"{n}.ln5", f"{p}.post_per_layer_input_norm.weight", H)], name="x")
        x = g.op("scale", [x], weight=weight(f"{n}.scalar", f"{p}.layer_scalar", (1,)), name="x")
    common.head(g, source, missing, x, prefix, softcap=c.logit_softcap, eps=eps)
    return g
