"""Inkling-2 (inkling2_loop) graph: 24 layers of weights run as 48 visits (layers loop_start..loop_end three times).

Per visit the layer reads the block bank with `attn_res` (a block of `block` layers is a source of the layers after it),
attends without rope (a relative bias from the per-head states `r`, per-head sinks, a sliding window on the 20 local
layers) through short convolutions, then runs a sigmoid-router MoE of softcapped SwiGLU experts. Passes after the first
read four messages per loop layer (`attn_res` over the previous pass's last block and final state) and merge them into
the layer's inputs with `message_gate`. Every visit has its own KV cache and four conv states.

Outputs: logits f32 [T, unpadded vocab], ids, and `h{v:02d}`, the bf16 output of each visit (debug: the oracle compares
them per visit).
"""

from __future__ import annotations

import math

from ..errors import UnsupportedModel
from ..ir import Graph, StateDecl
from ..sources import Source
from . import common

PREFIX = "model.llm."
# the controllers of loop layers: message kind -> checkpoint name; the order of the four message queries
_KINDS = ("attn", "k", "v", "mlp")


def build(source: Source, *, ctx_max: int, rows_max: int, kv: str) -> Graph:
    if kv != "bf16":
        raise UnsupportedModel(f"KV format {kv!r}: the relative bias and sinks have a bf16 cache only")
    c = source.config
    ik = c.inkling
    H, K, eps, nvis = c.hidden_size, ik.conv_kernel, c.rms_eps, 0
    missing: list[str] = []
    g = Graph(c.family)
    tokens, pos = common.inputs(g)
    declared: dict[str, str] = {}

    def weight(decl: str, tensor: str, shape: tuple[int, ...], *, linear: bool = False, exact: bool = False) -> str:
        """Each weight is declared once; the three passes share it."""
        if decl not in declared:
            declared[decl] = common.weight(
                g,
                source,
                missing,
                decl,
                PREFIX + tensor,
                shape,
                schemes=common.LINEAR_SCHEMES if linear else (),
                scheme="f32" if exact else None,
            )
        return declared[decl]

    def linear(x: str, decl: str, tensor: str, n: int, k: int) -> str:
        return g.op("linear", [x], weight=weight(decl, tensor, (n, k), linear=True))

    def cast(x: str, dtype: str) -> str:
        return x if g.types[x].dtype == dtype else g.op("cast", [x], dtype=dtype)

    def norm(x: str, decl: str, tensor: str) -> str:
        return g.op("rmsnorm", [x], weight=weight(decl, tensor, (H,)), eps=eps)

    def unit_norm(x: str) -> str:
        return g.op("rmsnorm", [x], eps=eps)

    def act(x: str) -> str:
        return cast(x, "bf16")

    def f32(x: str) -> str:
        return cast(x, "f32")

    def message_gate(x: str, msg: str, i: int, kind: str) -> str:
        p, n = f"layers.{i}.controllers.{kind}", f"L{i}.c_{kind}"
        r = ik.gate_rank
        return g.op(
            "message_gate",
            [x, msg],
            down=weight(f"{n}.down", f"{p}.down_weight", (r, 2 * H)),
            forget=weight(f"{n}.forget", f"{p}.forget_weight", (H, r)),
            gain=weight(f"{n}.gain", f"{p}.gain_weight", (H, r)),
            eps=eps,
        )

    def visit(i: int, v: int, h: str, blocks: list[str], msgs: list[str] | None) -> tuple[str, list[str]]:
        p, n, s = f"layers.{i}", f"L{i}", f"V{v}"
        full = c.layer_types[i] == "full_attention"
        nq, nkv, hd = (
            (c.num_heads, c.num_kv_heads, c.head_dim) if full else (ik.swa_heads, ik.swa_kv_heads, ik.swa_head_dim)
        )
        ring = ctx_max if full else min(ctx_max, c.sliding_window + rows_max - 1)
        kv_state = g.state(StateDecl(f"{s}.kv", "kv", (2, ring, nkv, hd)))
        conv = {
            name: g.state(StateDecl(f"{s}.{name}", "conv", (width, K - 1)))
            for name, width in (("kconv", nkv * hd), ("vconv", nkv * hd), ("attn_conv", H), ("mlp_conv", H))
        }

        # attention inputs: the block-residual read, the messages, the norms
        if i == 0:
            inputs = [h, h, h]
        elif full:
            query = weight(f"{n}.aq", f"attn_res.attn_queries.{i - 1}", (3, H), exact=True)
            inputs = [g.op("attn_res", [*blocks, h], query=query, row=row, eps=eps) for row in range(3)]
        else:
            query = weight(f"{n}.aq", f"attn_res.attn_queries.{i - 1}", (H,), exact=True)
            inputs = [g.op("attn_res", [*blocks, h], query=query, eps=eps)] * 3
        if msgs is not None:
            inputs = [message_gate(x, msgs[j], i, _KINDS[j]) for j, x in enumerate(inputs)]
        if full:
            hq, hk, hv = (
                act(norm(x, f"{n}.{k}norm", f"{p}.{k}_input_norm.weight")) for x, k in zip(inputs, "qkv", strict=True)
            )
        elif msgs is None:
            hq = hk = hv = act(norm(inputs[0], f"{n}.anorm", f"{p}.attn_norm.weight"))
        else:
            hq, hk, hv = (act(norm(x, f"{n}.anorm", f"{p}.attn_norm.weight")) for x in inputs)

        # q, r: per-head states, clipped (full layers also log-scaled)
        def clipped(x: str, heads: int, dim: int, decl: str, tensor: str, threshold: float) -> str:
            x = g.op("reshape", [x], shape=["T", heads, dim])
            x = g.op("l2_clip", [x], scale=weight(decl, tensor, (heads, dim)), threshold=threshold)
            return g.op("log_scale", [x, pos], alpha=ik.log_alpha, floor=ik.log_floor) if full else x

        q = clipped(
            linear(hq, f"{n}.wq", f"{p}.attn.wq_du.weight", nq * hd, H),
            nq,
            hd,
            f"{n}.qclip",
            f"{p}.attn.q_l2_clip.scale",
            ik.qk_clip,
        )
        r = clipped(
            linear(hq, f"{n}.wr", f"{p}.attn.wr_du.weight", nq * ik.d_rel, H),
            nq,
            ik.d_rel,
            f"{n}.rclip",
            f"{p}.attn.rel_l2_clip.scale",
            ik.rel_clip,
        )

        def short_conv(x: str, hx: str, which: str) -> str:
            """k or v: a static short conv on sliding layers, an input-dependent one on full layers."""
            if not full:
                return g.op(
                    "causal_conv1d",
                    [x],
                    weight=weight(f"{n}.{which}sconv", f"{p}.attn.{which}_sconv.weight", (nkv * hd, K)),
                    state=conv[f"{which}conv"],
                    residual=True,
                    divisor=math.sqrt(K + 1),
                )
            taps = linear(hx, f"{n}.{which}taps", f"{p}.attn.{which}_sconv.weight_proj", nkv * K, H)
            return g.op(
                "dyn_conv1d",
                [x, taps],
                weight=weight(f"{n}.{which}static", f"{p}.attn.{which}_sconv.static_w", (K, nkv * hd)),
                state=conv[f"{which}conv"],
                heads=nkv,
                divisor=math.sqrt(2 * K + 1),
            )

        k = g.op(
            "reshape",
            [short_conv(linear(hk, f"{n}.wk", f"{p}.attn.wk_dv.weight", nkv * hd, H), hk, "k")],
            shape=["T", nkv, hd],
        )
        k = g.op("l2_clip", [k], threshold=ik.qk_clip)
        vx = short_conv(linear(hv, f"{n}.wv", f"{p}.attn.wv_dv.weight", nkv * hd, H), hv, "v")
        vv = unit_norm(g.op("reshape", [vx], shape=["T", nkv, hd]))
        sinks = linear(hq, f"{n}.wsink", f"{p}.attn.attn_sink_proj.weight", nq, H)
        att = g.op(
            "attention",
            [q, k, vv, pos, r, sinks],
            state=kv_state,
            scale=hd**-0.5,
            window=None if full else c.sliding_window,
            rel_proj=weight(
                f"{n}.relproj",
                f"{p}.attn.rel_logits_proj.proj",
                (ik.d_rel, ik.rel_extent if full else c.sliding_window),
            ),
            sinks=True,
        )
        gate = linear(hq, f"{n}.wgate", f"{p}.attn.gate_attn.weight", nq * hd, H)
        gated = act(g.op("sigmoid_mul", [f32(att), f32(gate)]))
        o = linear(gated, f"{n}.wo", f"{p}.attn.wo_ud.weight", H, nq * hd)

        def conv_res(x: str, decl: str, tensor: str, state: str) -> str:
            return g.op(
                "causal_conv1d",
                [x],
                weight=weight(decl, tensor, (H, K)),
                state=conv[state],
                residual=True,
                divisor=math.sqrt(K + 1),
            )

        ao = f32(conv_res(o, f"{n}.asconv", f"{p}.attn_sconv.weight", "attn_conv"))

        # the residual stream is f32 inside the layer; a block starts at every `block`-th layer
        if i % ik.block == 0:
            blocks = [*blocks, h]
            res = ao
        else:
            res = g.op("add", [f32(h), ao])
        m = g.op(
            "attn_res", [*blocks, res], query=weight(f"{n}.mq", f"attn_res.mlp_queries.{i}", (H,), exact=True), eps=eps
        )
        if msgs is not None:
            m = message_gate(m, msgs[3], i, "mlp")
        xm = act(norm(m, f"{n}.mnorm", f"{p}.mlp_norm.weight"))
        E, F = ik.experts, ik.expert_width
        idx, w = g.op_multi(
            "moe_route",
            [xm],
            weight=weight(f"{n}.router", f"{p}.mlp.gate.weight", (E, H)),
            bias=weight(f"{n}.rbias", f"{p}.mlp.gate.bias", (E,), exact=True),
            top_k=ik.top_k,
            route_scale=ik.route_scale,
            scoring="sigmoid",
        )
        experts = {
            part: weight(f"{n}.e_{part}", f"{p}.mlp.experts.{part}_proj.weight", shape, linear=True)
            for part, shape in (("gate", (E, F, H)), ("up", (E, F, H)), ("down", (E, H, F)))
        }
        mo = g.op("moe_experts", [xm, idx, w], **experts, gate_cap=ik.gate_cap, up_cap=ik.up_cap)
        out = act(g.op("add", [res, f32(conv_res(mo, f"{n}.msconv", f"{p}.mlp_sconv.weight", "mlp_conv"))]))
        g.output(**{f"h{v:02d}": out})
        return out, blocks

    embed = weight("embed", "embed.weight", (c.vocab_size, H))
    h = norm(g.op("embed", [tokens], weight=embed), "embed_norm", "embed_norm.weight")
    blocks: list[str] = []
    for i in range(ik.loop_start):
        h, blocks = visit(i, nvis, h, blocks, None)
        nvis += 1
    prelude_h, prelude_blocks = h, blocks
    width = ik.loop_end - ik.loop_start
    for passno in range(ik.passes):
        msgs: list[list[str]] | None = None
        if passno > 0:
            # the previous pass's blocks after the prelude's and the one it started (the input of its first block
            # boundary layer), plus its final state: one read per message of each loop layer
            rows = [*blocks[len(prelude_blocks) + 1 :], h]
            h, blocks = prelude_h, prelude_blocks
            queries = {
                name: weight(name, f"middle_cycle_cross_pass_{name}", (2 * width, H), exact=True)
                for name in ("queries", "kv_queries")
            }
            msgs = [
                [
                    g.op("attn_res", rows, query=queries[q], row=2 * li + j, eps=eps, name="msg")
                    for q, j in (("queries", 0), ("kv_queries", 0), ("kv_queries", 1), ("queries", 1))
                ]
                for li in range(width)
            ]
        for li, i in enumerate(range(ik.loop_start, ik.loop_end)):
            h, blocks = visit(i, nvis, h, blocks, None if msgs is None else msgs[li])
            nvis += 1
    for i in range(ik.loop_end, c.num_layers):
        h, blocks = visit(i, nvis, h, blocks, None)
        nvis += 1

    final = g.op(
        "attn_res", [*blocks, h], query=weight("output_query", "attn_res.output_query", (H,), exact=True), eps=eps
    )
    xn = act(norm(final, "final_norm", "norm.weight"))
    unembed = linear(xn, "unembed", "unembed.weight", c.vocab_size, H)
    capped = g.op("softcap", [f32(unembed)], cap=c.logit_softcap, name="logits")
    logits, _padding = g.op_multi("split", [capped], sizes=[ik.unpadded_vocab, c.vocab_size - ik.unpadded_vocab])
    g.output(logits=logits, ids=g.op("argmax", [logits], name="ids"))
    common.finish(g, missing)
    return g
