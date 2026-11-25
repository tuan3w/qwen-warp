"""Official-implementation oracle for the Inkling-2 graph: a tiny random model run by transformers' `inkling2_loop`.

Runs in the isolated reference environment (transformers at the commit of the official model, which the warpc
environment does not contain), not in warpc's: `$WARPC_INKLING_REF_PYTHON warpc/tests/oracle/inkling_hf.py VARIANT DIR`.
It writes DIR/ckpt (a checkpoint in the layout of Inkling-2-Mini: config.json in the checkpoint's key spelling and
model.safetensors) and DIR/ref.pt (what the official model computes on it).

VARIANT a: experts as dense stacked tensors, hidden 64. VARIANT b: hidden 128, expert width 128, the experts packed as
compressed-tensors W4G128 by this script (the official model gets their exact fp32 dequantisation).

Weights: every parameter is rounded to a bf16 value before the official model runs (what the warpc source hands out
exactly), except the ones the checkpoint stores in fp32: the attention-residual queries and the router bias.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import DynamicCache, Inkling2LoopForCausalLM, Inkling2LoopTextConfig

T = 24
WINDOWS = (10, 1, 1, 4, 3, 1, 4)
PREFIX = "model.llm."
LOOP = {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": False}
LOCAL = [0, 1, 3, 4, 7]  # the layers with a sliding window: the others are full (2, 5, 6)
# the checkpoint's spelling (the official config class accepts these as aliases or extra keys)
BASE = {
    "vocab_size": 128,
    "unpadded_vocab_size": 120,
    "num_hidden_layers": 8,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "swa_num_attention_heads": 4,
    "swa_num_key_value_heads": 2,
    "swa_head_dim": 16,
    "sliding_window_size": 6,  # shorter than T: the window mask is exercised
    "d_rel": 8,
    "rel_extent": 5,  # shorter than T: the bias is zero beyond 5 positions on the full layers
    "log_scaling_n_floor": 4,  # positions >= 4 are log-scaled
    "log_scaling_alpha": 0.5,
    "local_layer_ids": LOCAL,
    "sconv_kernel_size": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 3,
    "route_scale": 2.0,
    "qk_l2_clip_threshold": 2.0,  # small enough that clipping is active
    "rel_l2_clip_threshold": 1.5,
    "n_layers_per_attn_res_block": 2,
    "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0,
    "final_logit_softcapping": 8.0,
    "rms_norm_eps": 1e-6,
    "loop_config": LOOP,
    "use_gate_bias": True,
    "n_shared_experts": 0,
    "q_bias": False,
    "o_bias": False,
    "use_value_norm_no_gamma": True,
    "attn_res_normalize_values": True,
}
VARIANTS = {
    "a": {"hidden_size": 64, "intermediate_size": 24},
    "b": {"hidden_size": 128, "intermediate_size": 128},
}
W4G128 = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "experts": {
            "targets": ["re:.*mlp\\.experts\\.(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
            "input_activations": None,
        }
    },
    "ignore": [],
    "warpc_layout": "stacked-experts",
}
# HF suffix (relative to `model.layers.I.`) -> checkpoint suffix (relative to `layers.I.`)
SIMPLE = {
    "self_attn.q_proj.linear.weight": "attn.wq_du.weight",
    "self_attn.k_proj.linear.weight": "attn.wk_dv.weight",
    "self_attn.v_proj.linear.weight": "attn.wv_dv.weight",
    "self_attn.o_proj.linear.weight": "attn.wo_ud.weight",
    "self_attn.o_proj.gate_proj.weight": "attn.gate_attn.weight",
    "self_attn.sink_proj.weight": "attn.attn_sink_proj.weight",
    "self_attn.q_proj.norm.scale": "attn.q_l2_clip.scale",
    "self_attn.relative_position_bias.linear.weight": "attn.wr_du.weight",
    "self_attn.relative_position_bias.norm.scale": "attn.rel_l2_clip.scale",
    "self_attn.relative_position_bias.proj": "attn.rel_logits_proj.proj",
    "self_attn.k_proj.sconv.static_weight": "attn.k_sconv.static_w",
    "self_attn.v_proj.sconv.static_weight": "attn.v_sconv.static_w",
    "self_attn.k_proj.sconv.weight_proj.weight": "attn.k_sconv.weight_proj",
    "self_attn.v_proj.sconv.weight_proj.weight": "attn.v_sconv.weight_proj",
    "self_attn.k_proj.sconv.conv1d.weight": "attn.k_sconv.weight",
    "self_attn.v_proj.sconv.conv1d.weight": "attn.v_sconv.weight",
    "post_attention_convolution.conv1d.weight": "attn_sconv.weight",
    "post_mlp_convolution.conv1d.weight": "mlp_sconv.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "q_input_layernorm.weight": "q_input_norm.weight",
    "k_input_layernorm.weight": "k_input_norm.weight",
    "v_input_layernorm.weight": "v_input_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    "mlp.gate.weight": "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias": "mlp.gate.bias",
    "mlp.experts.down_proj": "mlp.experts.down_proj.weight",
}
# HF `{q,k,v,mlp}_message_gate` -> checkpoint `controllers.{attn,k,v,mlp}`; HF part -> checkpoint tensor
GATES = {"q": "attn", "k": "k", "v": "v", "mlp": "mlp"}
GATE_PARTS = {"down_proj": "down_weight", "message_gate_proj": "gain_weight", "forget_gate_proj": "forget_weight"}
for _hf, _ck in GATES.items():
    for _part, _name in GATE_PARTS.items():
        SIMPLE[f"{_hf}_message_gate.{_part}.weight"] = f"controllers.{_ck}.{_name}"


def make_model(hidden: int, width: int):
    """The official model with random weights, each rounded to a bf16 value except the fp32 ones."""
    cfg = Inkling2LoopTextConfig(**BASE, hidden_size=hidden, intermediate_size=width, layer_types=layer_types())
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = Inkling2LoopForCausalLM(cfg).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("query") or n.endswith(".proj"):
                p.copy_(torch.randn_like(p) * 0.5)
            elif "static_weight" in n or "convolution" in n or "sconv" in n:
                p.copy_(torch.randn_like(p) * 0.4)
            elif n.endswith("scale") or p.dim() == 1:
                p.copy_(1.0 + 0.2 * torch.randn_like(p))
            elif "gate_up_proj" in n or ("down_proj" in n and p.dim() == 3):
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5 * 1.5)
            elif "message_gate" in n:
                p.copy_(torch.randn_like(p) * 0.2)  # larger message gates make the looped net amplify rounding noise
            elif "gate.weight" in n and "message" not in n and "o_proj" not in n:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5 * 3)  # router: confident, so near-ties are rare
            else:
                p.copy_(torch.randn_like(p) * p.shape[-1] ** -0.5)
        for n, b in m.named_buffers():
            if "e_score_correction_bias" in n:
                b.copy_(torch.randn_like(b) * 0.1)
        for n, t in [*m.named_parameters(), *m.named_buffers()]:
            if not (n.endswith("query") or "e_score_correction_bias" in n):
                t.copy_(t.bfloat16().float())
    return cfg, m


def layer_types() -> list[str]:
    return ["hybrid_sliding" if i in LOCAL else "hybrid" for i in range(BASE["num_hidden_layers"])]


def quantize(w: torch.Tensor):
    """fp32 [..., N, K] -> (packed int32 [..., N, K / 8], scale bf16 [..., N, K / 128], exact dequantisation)."""
    g = w.reshape(*w.shape[:-1], -1, 128)
    scale = (g.abs().amax(-1) / 7).bfloat16()
    s = scale.float()[..., None]
    q = torch.where(s > 0, torch.round(g / torch.where(s > 0, s, torch.ones_like(s))), torch.zeros_like(g)).clamp(-8, 7)
    nibbles = (q + 8).to(torch.int64).reshape(*w.shape[:-1], -1, 8)
    word = sum(nibbles[..., e] << (4 * e) for e in range(8))
    packed = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
    return packed, scale, (q * s).reshape(w.shape)


def to_checkpoint(m, cfg, quantized: bool) -> dict[str, torch.Tensor]:
    """The model's state as the tensors of the Inkling-2-Mini checkpoint (the inverse of transformers' conversion)."""
    sd = {k: v.detach().float().cpu() for k, v in m.state_dict().items()}
    out: dict[str, torch.Tensor] = {}
    taken: set[str] = set()

    def put(src: str, dst: str) -> None:
        out[PREFIX + dst] = sd[src]
        taken.add(src)

    put("model.embed_tokens.weight", "embed.weight")
    put("model.embed_tokens.embed_norm.weight", "embed_norm.weight")
    put("model.norm.weight", "norm.weight")
    put("model.final_input.query", "attn_res.output_query")
    # the checkpoint's unembedding is padded to vocab_size with rows that are not zero
    head = sd["lm_head.weight"]
    pad = torch.randn(cfg.vocab_size - head.shape[0], head.shape[1]) * 3.0
    out[PREFIX + "unembed.weight"] = torch.cat([head, pad]).bfloat16().float()
    taken.add("lm_head.weight")
    q = sd["model.message_input.query"]  # [loop layers, 4 (q, k, v, mlp), H]
    out[PREFIX + "middle_cycle_cross_pass_queries"] = q[:, [0, 3]].flatten(0, 1).contiguous()
    out[PREFIX + "middle_cycle_cross_pass_kv_queries"] = q[:, 1:3].flatten(0, 1).contiguous()
    taken.add("model.message_input.query")
    for i in range(cfg.num_hidden_layers):
        h = f"model.layers.{i}."
        if i > 0:
            put(h + "attention_input.query", f"attn_res.attn_queries.{i - 1}")
        put(h + "mlp_input.query", f"attn_res.mlp_queries.{i}")
        for hf, ck in SIMPLE.items():
            if h + hf in sd:
                put(h + hf, f"layers.{i}.{ck}")
        gate, up = sd[h + "mlp.experts.gate_up_proj"].chunk(2, dim=1)
        taken.add(h + "mlp.experts.gate_up_proj")
        base = PREFIX + f"layers.{i}.mlp.experts."
        if quantized:
            del out[base + "down_proj.weight"]
            for part, w in (("gate", gate), ("up", up), ("down", sd[h + "mlp.experts.down_proj"])):
                packed, scale, exact = quantize(w)
                assert torch.equal(exact, w), f"{base}{part}: the model's experts are not W4G128-representable"
                out[base + f"{part}_proj.weight_packed"], out[base + f"{part}_proj.weight_scale"] = packed, scale
        else:
            out[base + "gate_proj.weight"], out[base + "up_proj.weight"] = gate, up
    assert not set(sd) - taken, sorted(set(sd) - taken)
    exact = ("attn_queries", "mlp_queries", "output_query", "cross_pass", ".gate.bias")
    return {
        k: v if any(s in k for s in exact) else (v.bfloat16() if v.dtype == torch.float32 else v)
        for k, v in out.items()
    }


def dequantize_into(m, cfg) -> None:
    """Replace the official model's expert weights by the exact dequantisation of their W4G128 packing."""
    with torch.no_grad():
        for layer in m.model.layers:
            gate, up = layer.mlp.experts.gate_up_proj.chunk(2, dim=1)
            layer.mlp.experts.gate_up_proj.copy_(torch.cat([quantize(gate)[2], quantize(up)[2]], dim=1))
            layer.mlp.experts.down_proj.copy_(quantize(layer.mlp.experts.down_proj)[2])


def noise_floor(m, ids, visits) -> list[float]:
    """Relative L2 per visit between the official model and itself with the embedding table perturbed by 1e-7
    relative: the sensitivity of this random network to fp32 rounding, which bounds what two correct implementations
    agree to."""
    base = [v.clone() for v in visits]
    visits.clear()
    saved = m.model.embed_tokens.weight.detach().clone()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        m.model.embed_tokens.weight.mul_(1 + 1e-7 * torch.randn(saved.shape, generator=g))
        m(input_ids=ids, use_cache=False)
        m.model.embed_tokens.weight.copy_(saved)
    pert = list(visits)
    visits.clear()
    return [((a - b).norm() / a.norm()).item() for a, b in zip(base, pert, strict=True)]


def main(variant: str, out_dir: Path) -> None:
    hidden, width = VARIANTS[variant]["hidden_size"], VARIANTS[variant]["intermediate_size"]
    cfg, m = make_model(hidden, width)
    quantized = variant == "b"
    if quantized:
        dequantize_into(m, cfg)
    ckpt = out_dir / "ckpt"
    ckpt.mkdir(parents=True)
    save_file(to_checkpoint(m, cfg, quantized), str(ckpt / "model.safetensors"))
    text = {**BASE, "hidden_size": hidden, "intermediate_size": width, "model_type": "inkling2_loop_text"}
    config = {
        "model_type": "inkling2_loop",
        "architectures": ["Inkling2LoopForConditionalGeneration"],
        "text_config": text,
    }
    if quantized:
        config["quantization_config"] = W4G128
    (ckpt / "config.json").write_text(json.dumps(config, indent=2))

    visits: list[torch.Tensor] = []
    for layer in m.model.layers:
        layer.register_forward_hook(
            lambda mod, args, kwargs, out: visits.append(out[0].detach().clone()), with_kwargs=True
        )
    ids = torch.randint(0, BASE["unpadded_vocab_size"], (1, T), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        logits = m(input_ids=ids, use_cache=False).logits[0]
    ref_visits = [v.clone() for v in visits]
    assert len(ref_visits) == cfg.total_num_hidden_layers, len(ref_visits)
    floor = noise_floor(m, ids, visits)

    cache = DynamicCache(config=cfg)
    parts, at = [], 0
    with torch.no_grad():
        for w in WINDOWS:
            parts.append(m(input_ids=ids[:, at : at + w], cache=cache, use_cache=True).logits[0])
            at += w
    torch.save(
        {"ids": ids[0], "visits": ref_visits, "floor": floor, "logits": logits, "cache_logits": torch.cat(parts)},
        out_dir / "ref.pt",
    )


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
