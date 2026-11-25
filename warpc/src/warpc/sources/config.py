"""ModelConfig: the hyperparameters the frontends read, and the guard that refuses what they cannot reproduce.

`from_hf` reads the keys every family shares once (`_common`), then the family's own (`_FAMILIES`); each family's guard
names every setting its graph would silently get wrong.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..errors import UnsupportedModel

# dense family -> per-head q/k RMSNorm before rope
_QK_NORM = {"llama": False, "mistral": False, "qwen3": True}
_REQUIRED = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "rms_norm_eps",
)
_LLAMA3_REQUIRED = ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")
_QWEN35_REQUIRED = (
    "layer_types",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)
# config.json layer_types -> the frontend's layer kinds
_QWEN35_LAYERS = {"full_attention": "attention", "linear_attention": "linear_attention"}
_GEMMA4_REQUIRED = ("layer_types", "sliding_window", "rope_parameters", "head_dim")
_GEMMA4_LAYERS = ("sliding_attention", "full_attention")
# the checkpoint's own key names (the HF class accepts aliases of some; this reads one spelling)
_INKLING_REQUIRED = (
    "num_key_value_heads",
    "head_dim",
    "swa_num_attention_heads",
    "swa_num_key_value_heads",
    "swa_head_dim",
    "sliding_window_size",
    "local_layer_ids",
    "d_rel",
    "rel_extent",
    "log_scaling_n_floor",
    "log_scaling_alpha",
    "qk_l2_clip_threshold",
    "rel_l2_clip_threshold",
    "sconv_kernel_size",
    "n_routed_experts",
    "num_experts_per_tok",
    "route_scale",
    "mlp_gate_softcap",
    "mlp_up_softcap",
    "final_logit_softcapping",
    "n_layers_per_attn_res_block",
    "unpadded_vocab_size",
    "loop_config",
)


def _rope_settings(hf: dict) -> dict:
    """The rope scaling dict: `rope_scaling` (config.json of transformers < 5) or `rope_parameters` (>= 5)."""
    return hf.get("rope_scaling") or hf.get("rope_parameters") or {}


def _rope_kind(hf: dict) -> str | None:
    rope = _rope_settings(hf)
    return rope.get("rope_type") or rope.get("type")


def _common_unsupported(text: dict, *, windowed: bool = False) -> list[str]:
    """Settings of the text config that no family's graph reproduces (`windowed`: the family has sliding-window layers
    of its own)."""
    bad = []
    # Qwen configs switch their window off with use_sliding_window false; Mistral configs have no switch (absent/null).
    if not windowed and text.get("sliding_window") and text.get("use_sliding_window") is not False:
        bad.append(f"sliding_window={text['sliding_window']}")
    if text.get("attention_bias"):
        bad.append("attention_bias=True")
    if text.get("mlp_bias"):
        bad.append("mlp_bias=True")
    act = text.get("hidden_act", "silu")
    if act != "silu":
        bad.append(f"hidden_act='{act}' (only silu)")
    bad += [f"missing {key}" for key in _REQUIRED if text.get(key) is None]
    return bad


def _dense_unsupported(text: dict, hf: dict) -> list[str]:
    bad = []
    kind = _rope_kind(text)
    if kind == "llama3":
        rope = _rope_settings(text)
        bad += [f"missing rope scaling {key}" for key in _LLAMA3_REQUIRED if rope.get(key) is None]
    elif kind not in (None, "default"):
        bad.append(f"rope scaling '{kind}' (rope_scaling / rope_parameters; supported: llama3)")
    if hf.get("quantization_config"):
        method = hf["quantization_config"].get("quant_method", "?")
        bad.append(f"quantization_config '{method}' (only float safetensors)")
    return bad


def _w4g128_unsupported(q: dict) -> list[str]:
    """compressed-tensors pack-quantized int4, group 128, symmetric, weights only: anything else, by name."""
    method, fmt = q.get("quant_method"), q.get("format")
    if method != "compressed-tensors" or fmt != "pack-quantized":
        return [f"quantization_config {method!r} / {fmt!r} (only compressed-tensors pack-quantized)"]
    bad = []
    for name, group in (q.get("config_groups") or {}).items():
        w = group.get("weights") or {}
        want = {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "symmetric": True}
        bad += [f"{name} weights {k}={w.get(k)!r} (only {v!r})" for k, v in want.items() if w.get(k) != v]
        if w.get("dynamic"):
            bad.append(f"{name} dynamic weights")
        if w.get("actorder") not in (None, "static"):
            bad.append(f"{name} actorder {w.get('actorder')!r} (a g_idx permutation)")
        for act in ("input_activations", "output_activations"):
            if group.get(act):
                bad.append(f"{name} {act} quantisation")
        if group.get("format", q["format"]) != "pack-quantized":
            bad.append(f"{name} format {group.get('format')!r}")
    if not q.get("config_groups"):
        bad.append("quantization_config without config_groups")
    return bad


def _qwen35_unsupported(text: dict, hf: dict) -> list[str]:
    bad = [f"missing {key}" for key in _QWEN35_REQUIRED if text.get(key) is None]
    kinds = sorted({t for t in text.get("layer_types") or () if t not in _QWEN35_LAYERS})
    if kinds:
        bad.append(f"layer_types {kinds} (only {', '.join(_QWEN35_LAYERS)})")
    if not text.get("attn_output_gate", True):
        bad.append("attn_output_gate=False")
    if text.get("output_gate_type", "swish") != "swish":
        bad.append(f"output_gate_type={text['output_gate_type']!r} (only swish)")
    if text.get("mamba_ssm_dtype", "float32") != "float32":
        bad.append(f"mamba_ssm_dtype={text['mamba_ssm_dtype']!r} (only float32)")
    kind = _rope_kind(text)
    if kind not in (None, "default"):
        # mrope_section is accepted: text positions are equal in all three sections, which is plain partial rope
        bad.append(f"rope type '{kind}' (only default)")
    if hf.get("quantization_config"):
        bad += _w4g128_unsupported(hf["quantization_config"])
    return bad


def _inkling_unsupported(text: dict, hf: dict) -> list[str]:
    bad = [f"missing {key}" for key in _INKLING_REQUIRED if text.get(key) is None]
    if text.get("n_shared_experts"):
        bad.append(f"n_shared_experts={text['n_shared_experts']} (no shared expert)")
    for key in ("q_bias", "o_bias"):
        if text.get(key):
            bad.append(f"{key}=True")
    # the keys of the checkpoint's config that switch a behaviour the graph fixes: absent means the official default
    for key in ("use_value_norm_no_gamma", "attn_res_normalize_values", "use_gate_bias"):
        if text.get(key) is False:
            bad.append(f"{key}=False")
    if text.get("scoring_func", "sigmoid") != "sigmoid":
        bad.append(f"scoring_func={text['scoring_func']!r} (only sigmoid)")
    loop = text.get("loop_config") or {}
    if loop.get("is_varloop"):
        bad.append("loop_config.is_varloop")
    block = text.get("n_layers_per_attn_res_block")
    if block and loop.get("core_start") is not None:
        edges = (loop["core_start"], loop.get("core_end"))
        if loop["core_start"] <= 0 or any(e is None or e % block for e in edges):
            bad.append(f"loop_config core {edges} (a positive multiple of n_layers_per_attn_res_block={block})")
    if hf.get("quantization_config"):
        q = hf["quantization_config"]
        if q.get("quant_method") == "modelopt":
            bad.append(f"quantization_config 'modelopt' {q.get('quant_algo')} (requantise the experts to W4G128)")
        else:
            bad += _w4g128_unsupported(q)
            if q.get("warpc_layout") != "stacked-experts":
                bad.append(f"quantization_config warpc_layout {q.get('warpc_layout')!r} (only 'stacked-experts')")
    return bad


def _plain_inv_freq(theta: float, rotary_dim: int) -> torch.Tensor:
    """fp32 theta^(-2j / rotary_dim) per rotary pair j, the expression of transformers and of llama.cpp's converter."""
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))


def llama3_freq_factors(theta: float, rotary_dim: int, rope: dict) -> tuple[float, ...]:
    """Llama 3.1 rope scaling as one divisor of the plain inverse frequency per rotary pair: llama.cpp's `rope_freqs`
    tensor, with its converter's fp32 expressions, so an HF checkpoint and its GGUF conversion configure alike."""
    factor, low, high = (rope[key] for key in ("factor", "low_freq_factor", "high_freq_factor"))
    context = rope["original_max_position_embeddings"]
    wavelen = 2 * math.pi / _plain_inv_freq(theta, rotary_dim)
    smooth = (context / wavelen - low) / (high - low)
    medium = 1 / ((1 - smooth) / factor + smooth)
    factors = torch.where(wavelen > context / low, float(factor), medium)
    return tuple(torch.where(wavelen < context / high, 1.0, factors).tolist())


@dataclass(frozen=True)
class Gemma4Text:
    """What Gemma 4 adds to a dense decoder: global (full-attention) layers with their own head dim and proportional
    rope, KV sharing, per-layer input embeddings (PLE), a wider MLP on the sharing layers."""

    kv_shared_layers: int  # the last n layers own no K / V: they read the last owner of their layer type
    global_head_dim: int
    global_kv_heads: int
    global_rope_theta: float
    global_rope_partial: float  # the fraction of a global head's dims that rotate (proportional rope)
    per_layer_hidden: int  # PLE width per layer; 0: no per-layer embeddings
    per_layer_vocab: int
    double_wide_mlp: bool  # sharing layers have twice the intermediate size

    def owner(self, layer_types: tuple[str, ...], layer: int) -> int:
        """The layer whose KV cache `layer` reads (itself unless it shares)."""
        first_shared = len(layer_types) - self.kv_shared_layers
        if layer < first_shared:
            return layer
        kind = layer_types[layer]
        owners = [i for i in range(first_shared) if layer_types[i] == kind]
        if not owners:
            raise UnsupportedModel(f"layer {layer} ({kind}) shares KV but no earlier layer owns that type")
        return owners[-1]


@dataclass(frozen=True)
class LinearAttention:
    """Gated DeltaNet geometry."""

    num_k_heads: int
    num_v_heads: int
    k_head_dim: int
    v_head_dim: int
    conv_kernel: int


@dataclass(frozen=True)
class InklingConfig:
    """Inkling-2 geometry: the sliding layers' own head counts and window, the relative bias, the clips, the experts,
    the block residual and the loop."""

    swa_heads: int
    swa_kv_heads: int
    swa_head_dim: int
    d_rel: int
    rel_extent: int  # relative-bias extent of the full layers (a sliding layer's is its window)
    log_floor: int  # full layers scale q and r by 1 + log_alpha * log(max((pos + 1) / log_floor, 1))
    log_alpha: float
    qk_clip: float
    rel_clip: float
    conv_kernel: int
    experts: int
    top_k: int
    expert_width: int
    route_scale: float
    gate_cap: float
    up_cap: float
    block: int  # layers per attention-residual block
    loop_start: int
    loop_end: int
    passes: int
    gate_rank: int
    unpadded_vocab: int  # rows of the unembedding that are tokens; the rest of vocab_size is padding


@dataclass(frozen=True)
class ModelConfig:
    family: str  # "llama" | "mistral" | "qwen3" | "qwen3_5" | "gemma4" | "inkling2_loop"
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float | None  # None: no rope (Inkling)
    # per rotary pair: the divisor of theta^(-2j / rotary_dim) (Llama 3.1 scaling); None: plain theta
    rope_freq_factors: tuple[float, ...] | None
    qk_norm: bool
    tie_embeddings: bool
    rotary_dim: int  # the leading dims of a head that rope rotates (dense: head_dim)
    norm_offset: float = 0.0  # every RMSNorm computes x-hat * (norm_offset + w)
    attn_output_gate: bool = False  # the attention output is multiplied by sigmoid of half of q_proj
    layer_types: tuple[str, ...] = ()  # per layer; qwen3_5 "attention" | "linear_attention", gemma4 and inkling2_loop
    # the config's "sliding_attention" | "full_attention"; empty: all attention
    linear: LinearAttention | None = None
    quantization: str | None = None  # "w4g128": compressed-tensors int4 linears (Inkling: the experts)
    activation: str = "silu"  # the MLP gate: "silu" | "gelu_tanh"
    logit_softcap: float | None = None  # logits = cap * tanh(logits / cap)
    embed_scale: float = 1.0  # the embedding rows are multiplied by this
    sliding_window: int | None = None  # the window of the sliding_attention layers
    gemma: Gemma4Text | None = None
    inkling: InklingConfig | None = None

    def __post_init__(self):
        factors = self.rope_freq_factors
        if factors is not None and len(factors) != self.rotary_dim // 2:
            raise UnsupportedModel(f"{len(factors)} rope frequency factors for {self.rotary_dim // 2} rotary pairs")
        if self.layer_types and len(self.layer_types) != self.num_layers:
            raise UnsupportedModel(f"{len(self.layer_types)} layer types for {self.num_layers} layers")
        if self.gemma is not None:
            self.gemma.owner(self.layer_types, self.num_layers - 1)  # every sharing layer has an owner of its type

    def rope_inv_freq(self, layer_type: str | None = None) -> tuple[float, ...]:
        """The inverse frequency of each of the rotary_dim / 2 rotary pairs, fp32 values: the angle of pair j at
        position p is p * inv_freq[j]. Gemma 4's full-attention layers (`layer_type` "full_attention") rotate the
        first `partial` of the pairs of their global head with theta^(-2j / global_head_dim) and leave the rest at
        frequency 0, the identity (transformers' proportional rope)."""
        if layer_type == "full_attention" and self.gemma is not None:
            g = self.gemma
            angles = int(g.global_rope_partial * g.global_head_dim // 2)
            rotated = 1.0 / (
                g.global_rope_theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.float32) / g.global_head_dim)
            )
            zeros = torch.zeros(g.global_head_dim // 2 - angles, dtype=torch.float32)
            return tuple(torch.cat([rotated, zeros]).tolist())
        inv = _plain_inv_freq(self.rope_theta, self.rotary_dim)
        if self.rope_freq_factors is not None:
            inv = inv / torch.tensor(self.rope_freq_factors, dtype=torch.float32)
        return tuple(inv.tolist())

    @classmethod
    def from_hf(cls, hf: dict) -> ModelConfig:
        """From an HF config.json dict; raises UnsupportedModel naming each unsupported setting."""
        model_type = hf.get("model_type")
        family = _FAMILY_OF.get(model_type)
        if family is None:
            supported = ", ".join(_FAMILY_OF)
            raise UnsupportedModel(
                f"cannot build this model faithfully: model_type {model_type!r} (supported: {supported})"
            )
        text = (hf.get("text_config") or hf) if family in ("qwen3_5", "gemma4", "inkling2_loop") else hf
        unsupported, extras = _FAMILIES[family]
        bad = _common_unsupported(text, windowed=family == "gemma4") + unsupported(text, hf)
        if bad:
            raise UnsupportedModel(f"cannot build this model faithfully: {'; '.join(bad)}")
        common = _common(text)
        return cls(family=family, **{**common, **extras(text, hf, common)})


def _common(text: dict) -> dict:
    """The keys every family reads, each parsed once."""
    heads = int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or text["hidden_size"] // heads)
    rope = _rope_settings(text)
    return {
        "vocab_size": int(text["vocab_size"]),
        "hidden_size": int(text["hidden_size"]),
        "intermediate_size": int(text["intermediate_size"]),
        "num_layers": int(text["num_hidden_layers"]),
        "num_heads": heads,
        "num_kv_heads": int(text.get("num_key_value_heads") or heads),
        "head_dim": head_dim,
        "rms_eps": float(text["rms_norm_eps"]),
        "rope_theta": float(text.get("rope_theta") or rope.get("rope_theta") or 10000.0),
        # the class default of every supported family
        "tie_embeddings": bool(text.get("tie_word_embeddings", False)),
    }


def _dense_extras(text: dict, hf: dict, common: dict) -> dict:
    hd, theta = common["head_dim"], common["rope_theta"]
    llama3 = _rope_kind(text) == "llama3"
    return {
        "rope_freq_factors": llama3_freq_factors(theta, hd, _rope_settings(text)) if llama3 else None,
        "qk_norm": _QK_NORM[text["model_type"]],
        "rotary_dim": hd,
    }


def _qwen35_extras(text: dict, hf: dict, common: dict) -> dict:
    factor = text.get("partial_rotary_factor") or _rope_settings(text).get("partial_rotary_factor") or 1.0
    return {
        "rope_freq_factors": None,
        "qk_norm": True,
        "rotary_dim": int(common["head_dim"] * factor),
        "norm_offset": 1.0,
        "attn_output_gate": True,
        "layer_types": tuple(_QWEN35_LAYERS[t] for t in text["layer_types"]),
        "linear": LinearAttention(
            num_k_heads=int(text["linear_num_key_heads"]),
            num_v_heads=int(text["linear_num_value_heads"]),
            k_head_dim=int(text["linear_key_head_dim"]),
            v_head_dim=int(text["linear_value_head_dim"]),
            conv_kernel=int(text["linear_conv_kernel_dim"]),
        ),
        "quantization": "w4g128" if hf.get("quantization_config") else None,
    }


def _gemma4_unsupported(text: dict, hf: dict) -> list[str]:
    bad = [f"missing {key}" for key in _GEMMA4_REQUIRED if text.get(key) is None]
    kinds = sorted({t for t in text.get("layer_types") or () if t not in _GEMMA4_LAYERS})
    if kinds:
        bad.append(f"layer_types {kinds} (only {', '.join(_GEMMA4_LAYERS)})")
    for key, why in (
        ("enable_moe_block", "the MoE block"),
        ("attention_k_eq_v", "attention_k_eq_v (keys are values on the global layers)"),
        ("use_bidirectional_attention", "bidirectional attention"),
        ("attn_logit_softcapping", "attention logit softcapping"),
    ):
        if text.get(key):
            bad.append(f"{key}={text[key]!r} ({why})")
    act = text.get("hidden_activation", "gelu_pytorch_tanh")
    if act != "gelu_pytorch_tanh":
        bad.append(f"hidden_activation={act!r} (only gelu_pytorch_tanh)")
    rope = text.get("rope_parameters") or {}
    for kind in _GEMMA4_LAYERS:
        params = rope.get(kind)
        if params is None:
            bad.append(f"missing rope_parameters for {kind}")
            continue
        rope_type = params.get("rope_type", "default")
        if rope_type not in ("default", "proportional"):
            bad.append(f"rope_type {rope_type!r} for {kind} (only default, proportional)")
        if params.get("factor", 1.0) != 1.0:
            bad.append(f"rope factor {params['factor']} for {kind}")
    for layer, over in (text.get("per_layer_config") or {}).items():
        extra = sorted(set(over) - {"head_dim"})
        if extra:
            bad.append(f"per_layer_config[{layer}] overrides {extra} (only head_dim)")
    if hf.get("quantization_config") or text.get("quantization_config"):
        bad.append("quantization_config (only float safetensors)")
    return bad


def _gemma4_extras(text: dict, hf: dict, common: dict) -> dict:
    types = tuple(text["layer_types"])
    rope = text["rope_parameters"]
    over = text.get("per_layer_config") or {}
    # transformers writes the global head dim as a per-layer head_dim override of the full-attention layers
    per_layer = next(
        (over[str(i)]["head_dim"] for i, t in enumerate(types) if t == "full_attention" and str(i) in over), None
    )
    global_head_dim = int(text.get("global_head_dim") or per_layer or common["head_dim"])
    sliding, full = rope["sliding_attention"], rope["full_attention"]
    return {
        "rope_theta": float(sliding["rope_theta"]),
        "rope_freq_factors": None,
        "qk_norm": True,
        "rotary_dim": common["head_dim"],
        "layer_types": types,
        "activation": "gelu_tanh",
        "logit_softcap": text.get("final_logit_softcapping"),
        "embed_scale": float(common["hidden_size"] ** 0.5),
        "sliding_window": int(text["sliding_window"]),
        "gemma": Gemma4Text(
            kv_shared_layers=int(text.get("num_kv_shared_layers") or 0),
            global_head_dim=global_head_dim,
            global_kv_heads=common["num_kv_heads"],
            global_rope_theta=float(full["rope_theta"]),
            global_rope_partial=float(full.get("partial_rotary_factor", 1.0)),
            per_layer_hidden=int(text.get("hidden_size_per_layer_input") or 0),
            per_layer_vocab=int(text.get("vocab_size_per_layer_input") or common["vocab_size"]),
            double_wide_mlp=bool(text.get("use_double_wide_mlp")),
        ),
    }


def _inkling_extras(text: dict, hf: dict, common: dict) -> dict:
    full = set(range(common["num_layers"])) - set(text["local_layer_ids"])
    loop = text["loop_config"]
    return {
        "rope_theta": None,
        "rope_freq_factors": None,
        "qk_norm": False,
        "rotary_dim": 0,
        "layer_types": tuple(
            "full_attention" if i in full else "sliding_attention" for i in range(common["num_layers"])
        ),
        "quantization": "w4g128" if hf.get("quantization_config") else None,
        "sliding_window": int(text["sliding_window_size"]),
        "logit_softcap": float(text["final_logit_softcapping"]),
        "inkling": InklingConfig(
            swa_heads=int(text["swa_num_attention_heads"]),
            swa_kv_heads=int(text["swa_num_key_value_heads"]),
            swa_head_dim=int(text["swa_head_dim"]),
            d_rel=int(text["d_rel"]),
            rel_extent=int(text["rel_extent"]),
            log_floor=int(text["log_scaling_n_floor"]),
            log_alpha=float(text["log_scaling_alpha"]),
            qk_clip=float(text["qk_l2_clip_threshold"]),
            rel_clip=float(text["rel_l2_clip_threshold"]),
            conv_kernel=int(text["sconv_kernel_size"]),
            experts=int(text["n_routed_experts"]),
            top_k=int(text["num_experts_per_tok"]),
            expert_width=common["intermediate_size"],
            route_scale=float(text["route_scale"]),
            gate_cap=float(text["mlp_gate_softcap"]),
            up_cap=float(text["mlp_up_softcap"]),
            block=int(text["n_layers_per_attn_res_block"]),
            loop_start=int(loop["core_start"]),
            loop_end=int(loop["core_end"]),
            passes=int(loop["num_passes"]),
            gate_rank=int(loop["gate_rank"]),
            unpadded_vocab=int(text["unpadded_vocab_size"]),
        ),
    }


_FAMILY_OF = {
    **{f: f for f in _QK_NORM},
    "qwen3_5": "qwen3_5",
    "qwen3_5_text": "qwen3_5",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
    "inkling2_loop": "inkling2_loop",
}
_FAMILIES = {
    **dict.fromkeys(_QK_NORM, (_dense_unsupported, _dense_extras)),
    "qwen3_5": (_qwen35_unsupported, _qwen35_extras),
    "gemma4": (_gemma4_unsupported, _gemma4_extras),
    "inkling2_loop": (_inkling_unsupported, _inkling_extras),
}
