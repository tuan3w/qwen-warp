"""Dense decoder graph (llama, mistral, qwen3): GQA attention with rope, optional per-head q/k RMSNorm, SwiGLU MLP."""

from __future__ import annotations

from .errors import UnsupportedModel
from .ir import Graph, StateDecl, TType, WeightDecl
from .kvcache import FORMATS
from .sources import Source


def build(source: Source, *, ctx_max: int, kv: str = "bf16") -> Graph:
    """The graph of one forward window: inputs tokens / pos i32[T], outputs logits f32[T, V] and ids i32[T] (argmax)."""
    c = source.config
    if kv not in FORMATS:
        raise ValueError(f"unknown KV format {kv!r} (known: {', '.join(FORMATS)})")
    H, V, FF, hd = c.hidden_size, c.vocab_size, c.intermediate_size, c.head_dim
    nq, nkv, eps = c.num_heads, c.num_kv_heads, c.rms_eps
    inv_freq = c.rope_inv_freq()
    missing = []

    g = Graph(c.family)
    tokens = g.input("tokens", TType(("T",), "i32"))
    pos = g.input("pos", TType(("T",), "i32"))

    def weight(decl: str, tensor: str, shape: tuple[int, ...], *, linear: bool = False) -> str:
        if tensor not in source:
            missing.append(tensor)
        scheme = "q4_0" if linear and tensor in source and source.quant(tensor) == "Q4_0" else "dense"
        return g.weight(WeightDecl(decl, shape, scheme, tensor))

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return g.op("linear", [x], weight=weight(decl, tensor, (n, k), linear=True))

    def rmsnorm(x: str, decl: str, tensor: str, size: int, **attrs) -> str:
        return g.op("rmsnorm", [x], weight=weight(decl, tensor, (size,)), eps=eps, **attrs)

    x = g.op("embed", [tokens], weight=weight("embed", "model.embed_tokens.weight", (V, H)), name="x")
    for i in range(c.num_layers):
        p, n = f"model.layers.{i}", f"L{i}"
        state = g.state(StateDecl(f"{n}.kv", (2, ctx_max, nkv, hd), cache=kv))
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
    x = rmsnorm(x, "final_norm", "model.norm.weight", H, name="hidden")
    head = "model.embed_tokens.weight" if c.tie_embeddings else "lm_head.weight"
    logits = g.op("cast", [linear(x, "lm_head", head, V, H)], dtype="f32", name="logits")
    g.output(logits=logits, ids=g.op("argmax", [logits], name="ids"))
    if missing:
        raise UnsupportedModel(f"checkpoint lacks {len(missing)} tensors the graph reads: {', '.join(missing[:4])}")
    g.verify()
    return g
