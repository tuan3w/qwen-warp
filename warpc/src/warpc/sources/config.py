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


def _rope_settings(hf: dict) -> dict:
    """The rope scaling dict: `rope_scaling` (config.json of transformers < 5) or `rope_parameters` (>= 5)."""
    return hf.get("rope_scaling") or hf.get("rope_parameters") or {}


def _rope_kind(hf: dict) -> str | None:
    rope = _rope_settings(hf)
    return rope.get("rope_type") or rope.get("type")


def _common_unsupported(text: dict) -> list[str]:
    """Settings of the text config that no family's graph reproduces."""
    bad = []
    # Qwen configs switch their window off with use_sliding_window false; Mistral configs have no switch (absent/null).
    if text.get("sliding_window") and text.get("use_sliding_window") is not False:
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
class LinearAttention:
    """Gated DeltaNet geometry."""

    num_k_heads: int
    num_v_heads: int
    k_head_dim: int
    v_head_dim: int
    conv_kernel: int


@dataclass(frozen=True)
class ModelConfig:
    family: str  # "llama" | "mistral" | "qwen3" | "qwen3_5"
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float
    # per rotary pair: the divisor of theta^(-2j / rotary_dim) (Llama 3.1 scaling); None: plain theta
    rope_freq_factors: tuple[float, ...] | None
    qk_norm: bool
    tie_embeddings: bool
    rotary_dim: int  # the leading dims of a head that rope rotates (dense: head_dim)
    norm_offset: float = 0.0  # every RMSNorm computes x-hat * (norm_offset + w)
    attn_output_gate: bool = False  # the attention output is multiplied by sigmoid of half of q_proj
    layer_types: tuple[str, ...] = ()  # per layer "attention" | "linear_attention"; empty: all attention
    linear: LinearAttention | None = None
    quantization: str | None = None  # "w4g128": compressed-tensors int4 linears

    def __post_init__(self):
        factors = self.rope_freq_factors
        if factors is not None and len(factors) != self.rotary_dim // 2:
            raise UnsupportedModel(f"{len(factors)} rope frequency factors for {self.rotary_dim // 2} rotary pairs")
        if self.layer_types and len(self.layer_types) != self.num_layers:
            raise UnsupportedModel(f"{len(self.layer_types)} layer types for {self.num_layers} layers")

    def rope_inv_freq(self) -> tuple[float, ...]:
        """The inverse frequency of each of the rotary_dim / 2 rotary pairs, fp32 values: the angle of pair j at
        position p is p * inv_freq[j]."""
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
        text = (hf.get("text_config") or hf) if family == "qwen3_5" else hf
        unsupported, extras = _FAMILIES[family]
        bad = _common_unsupported(text) + unsupported(text, hf)
        if bad:
            raise UnsupportedModel(f"cannot build this model faithfully: {'; '.join(bad)}")
        common = _common(text)
        return cls(family=family, **common, **extras(text, hf, common))


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


_FAMILY_OF = {**{f: f for f in _QK_NORM}, "qwen3_5": "qwen3_5", "qwen3_5_text": "qwen3_5"}
_FAMILIES = {
    **dict.fromkeys(_QK_NORM, (_dense_unsupported, _dense_extras)),
    "qwen3_5": (_qwen35_unsupported, _qwen35_extras),
}
