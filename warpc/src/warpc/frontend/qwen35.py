"""Qwen3.5 hybrid graph: Gated DeltaNet layers and gated full-attention layers (config.layer_types), zero-centred
RMSNorms, partial rope, SwiGLU MLP; linears W4G128 where the checkpoint packs them. `build_draft` is the checkpoint's
multi-token-prediction head: one full-attention layer (the very blocks of the target's) behind a fused (token, hidden)
projection."""

from __future__ import annotations

from ..errors import UnsupportedModel
from ..ir import Graph, StateDecl, TType
from ..sources import Source
from ..sources.requantize import Requantized
from . import common


def attention(
    g: Graph, source: Source, missing: list[str], h: str, n: str, p: str, *, pos: str, ctx_max: int, kv: str
) -> str:
    """The gated full-attention block over the normed rows `h`: weights `{n}.q` ... from tensors `{p}.q_proj.weight`
    ..., its KV state `{n}.kv` (a cache of `ctx_max` positions in format `kv`); returns the o_proj output."""
    c = source.config
    H, hd, nq, nkv = c.hidden_size, c.head_dim, c.num_heads, c.num_kv_heads

    def linear(x: str, decl: str, tensor: str, n_out: int, k: int) -> str:
        return common.linear(g, source, missing, x, decl, tensor, n_out, k)

    def rmsnorm(x: str, decl: str, tensor: str, size: int) -> str:
        return common.rmsnorm(g, source, missing, x, decl, tensor, size, eps=c.rms_eps, offset=c.norm_offset)

    qg = g.op("reshape", [linear(h, f"{n}.q", f"{p}.q_proj.weight", 2 * nq * hd, H)], shape=["T", nq, 2 * hd])
    k = g.op("reshape", [linear(h, f"{n}.k", f"{p}.k_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
    v = g.op("reshape", [linear(h, f"{n}.v", f"{p}.v_proj.weight", nkv * hd, H)], shape=["T", nkv, hd])
    q, gate = g.op_multi("split", [qg], sizes=[hd, hd])
    q = rmsnorm(q, f"{n}.qn", f"{p}.q_norm.weight", hd)
    k = rmsnorm(k, f"{n}.kn", f"{p}.k_norm.weight", hd)
    q, k = g.op_multi("rope", [q, k, pos], rotary_dim=c.rotary_dim, inv_freq=c.rope_inv_freq())
    state = g.state(StateDecl(f"{n}.kv", "kv", (2, ctx_max, nkv, hd), cache=kv))
    scales = {}
    if kv == "fp8":
        scales = {s: common.weight(g, source, missing, f"{n}.{s}", f"{p}.{s}", (1,)) for s in ("k_scale", "v_scale")}
    att = g.op("attention", [q, k, v, pos], state=state, scale=hd**-0.5, **scales)
    gated = g.op("sigmoid_mul", [att, g.op("reshape", [gate], shape=["T", nq * hd])])
    return linear(gated, f"{n}.o", f"{p}.o_proj.weight", H, nq * hd)


def mlp(g: Graph, source: Source, missing: list[str], h: str, n: str, p: str) -> str:
    """The SwiGLU block over the normed rows `h`: weights `{n}.gate` / `.up` / `.down` from the tensors
    `{p}.gate_proj.weight` ...; returns the down_proj output."""
    c = source.config
    H, FF = c.hidden_size, c.intermediate_size
    gate = common.linear(g, source, missing, h, f"{n}.gate", f"{p}.gate_proj.weight", FF, H)
    up = common.linear(g, source, missing, h, f"{n}.up", f"{p}.up_proj.weight", FF, H)
    return common.linear(g, source, missing, g.op("silu_mul", [gate, up]), f"{n}.down", f"{p}.down_proj.weight", H, FF)


def build(source: Source, *, ctx_max: int, rows_max: int, kv: str) -> Graph:
    c = source.config
    la = c.linear
    H, eps = c.hidden_size, c.rms_eps
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

    embed = weight("embed", f"{prefix}embed_tokens.weight", (c.vocab_size, H))
    x = g.op("embed", [tokens], weight=embed, name="x")
    for i, kind in enumerate(c.layer_types):
        p, n = f"{prefix}layers.{i}", f"L{i}"
        h = rmsnorm(x, f"{n}.ln1", f"{p}.input_layernorm.weight", H)
        if kind == "attention":
            y = attention(g, source, missing, h, n, f"{p}.self_attn", pos=pos, ctx_max=ctx_max, kv=kv)
        else:
            y = deltanet(h, n, f"{p}.linear_attn")
        x = g.op("add", [x, y], name="x")
        h = rmsnorm(x, f"{n}.ln2", f"{p}.post_attention_layernorm.weight", H)
        x = g.op("add", [x, mlp(g, source, missing, h, n, f"{p}.mlp")], name="x")
    common.head(g, source, missing, x, prefix, **norm)
    return g


def build_draft(source: Source, *, ctx_max: int, draft_kv: str) -> Graph:
    """The multi-token-prediction head over a window of T (token, hidden) pairs: inputs tokens / pos i32[T], hidden
    bf16[T, H] (the target's post-final-norm rows), sel i32[1]; outputs hidden bf16[1, H] (the post-norm row `sel`),
    logits f32[1, NV] and ids i32[1] (the draft token, a full-vocabulary id). The layer is the target's
    full-attention block with its own KV state `mtp.kv`; the head is the pseudo-tensor `mtp.draft_head.weight` of a
    `Requantized` source (its ids mapped back through `take`), else the target's full head."""
    c = source.config
    if c.family != "qwen3_5" or c.mtp_layers != 1:
        raise UnsupportedModel(f"no one-layer MTP head: family {c.family}, {c.mtp_layers} MTP layers")
    H = c.hidden_size
    norm = {"eps": c.rms_eps, "offset": c.norm_offset}
    missing: list[str] = []
    g = Graph(f"{c.family}_mtp")
    tokens, pos = common.inputs(g)
    hidden = g.input("hidden", TType(("T", H), "bf16"))
    sel = g.input("sel", TType((1,), "i32"))
    prefix, p = common.text_prefix(source), "mtp.layers.0"

    def rmsnorm(x: str, decl: str, tensor: str) -> str:
        return common.rmsnorm(g, source, missing, x, decl, tensor, H, **norm)

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return common.linear(g, source, missing, x, decl, tensor, n, k)

    embed = common.weight(g, source, missing, "embed", f"{prefix}embed_tokens.weight", (c.vocab_size, H))
    e = rmsnorm(g.op("embed", [tokens], weight=embed), "mtp.pre_e", "mtp.pre_fc_norm_embedding.weight")
    r = rmsnorm(hidden, "mtp.pre_h", "mtp.pre_fc_norm_hidden.weight")
    x = linear(g.op("concat", [e, r]), "mtp.fc", "mtp.fc.weight", H, 2 * H)
    h = rmsnorm(x, "mtp.ln1", f"{p}.input_layernorm.weight")
    x = g.op(
        "add", [x, attention(g, source, missing, h, "mtp", f"{p}.self_attn", pos=pos, ctx_max=ctx_max, kv=draft_kv)]
    )
    h = rmsnorm(x, "mtp.ln2", f"{p}.post_attention_layernorm.weight")
    x = g.op("add", [x, mlp(g, source, missing, h, "mtp", f"{p}.mlp")])
    hr = g.op("select_row", [rmsnorm(x, "mtp.norm", "mtp.norm.weight"), sel], name="hidden")
    if isinstance(source, Requantized):
        head = linear(hr, "mtp.draft_head", "mtp.draft_head.weight", len(source.draft_ids), H)
        table = tuple(int(i) for i in source.draft_ids)
    else:
        tensor = f"{prefix}embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"
        head = linear(hr, "lm_head", tensor, c.vocab_size, H)
        table = None
    logits = g.op("cast", [head], dtype="f32", name="logits")
    ids = g.op("argmax", [logits])
    if table is not None:  # the op refuses ids that an fp32 constant cannot hold (>= 2**24)
        ids = g.op("take", [ids], table=table)
    g.output(hidden=hr, logits=logits, ids=ids)
    common.finish(g, missing)
    return g
