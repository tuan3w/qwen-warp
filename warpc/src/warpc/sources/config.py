"""ModelConfig: the hyperparameters the dense frontend reads, and the guard that refuses what it cannot reproduce."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..errors import UnsupportedModel

# family -> per-head q/k RMSNorm before rope
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


def _rope_settings(hf: dict) -> dict:
    """The rope scaling dict: `rope_scaling` (config.json of transformers < 5) or `rope_parameters` (>= 5)."""
    return hf.get("rope_scaling") or hf.get("rope_parameters") or {}


def _unsupported_settings(hf: dict) -> list[str]:
    """Every setting of an HF config.json dict that the dense graph would silently get wrong, by name."""
    bad = []
    model_type = hf.get("model_type")
    if model_type not in _QK_NORM:
        bad.append(f"model_type {model_type!r} (supported: {', '.join(_QK_NORM)})")
    rope = _rope_settings(hf)
    kind = rope.get("rope_type") or rope.get("type")
    if kind == "llama3":
        bad += [f"missing rope scaling {key}" for key in _LLAMA3_REQUIRED if rope.get(key) is None]
    elif kind not in (None, "default"):
        bad.append(f"rope scaling '{kind}' (rope_scaling / rope_parameters; supported: llama3)")
    # Qwen configs switch their window off with use_sliding_window false; Mistral configs have no switch (absent/null).
    if hf.get("sliding_window") and hf.get("use_sliding_window") is not False:
        bad.append(f"sliding_window={hf['sliding_window']}")
    if hf.get("attention_bias"):
        bad.append("attention_bias=True")
    if hf.get("mlp_bias"):
        bad.append("mlp_bias=True")
    act = hf.get("hidden_act", "silu")
    if act != "silu":
        bad.append(f"hidden_act='{act}' (only silu)")
    if hf.get("quantization_config"):
        method = hf["quantization_config"].get("quant_method", "?")
        bad.append(f"quantization_config '{method}' (only float safetensors)")
    bad += [f"missing {key}" for key in _REQUIRED if hf.get(key) is None]
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
class ModelConfig:
    family: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rms_eps: float
    rope_theta: float
    # per rotary pair: the divisor of theta^(-2j / head_dim) (Llama 3.1 scaling); None: plain theta
    rope_freq_factors: tuple[float, ...] | None
    qk_norm: bool
    tie_embeddings: bool

    def __post_init__(self):
        factors = self.rope_freq_factors
        if factors is not None and len(factors) != self.head_dim // 2:
            raise UnsupportedModel(f"{len(factors)} rope frequency factors for {self.head_dim // 2} rotary pairs")

    def rope_inv_freq(self) -> tuple[float, ...]:
        """The inverse frequency of each of the head_dim / 2 rotary pairs, fp32 values: the angle of pair j at
        position p is p * inv_freq[j]."""
        inv = _plain_inv_freq(self.rope_theta, self.head_dim)
        if self.rope_freq_factors is not None:
            inv = inv / torch.tensor(self.rope_freq_factors, dtype=torch.float32)
        return tuple(inv.tolist())

    @classmethod
    def from_hf(cls, hf: dict) -> ModelConfig:
        """From an HF config.json dict; raises UnsupportedModel naming each unsupported setting."""
        bad = _unsupported_settings(hf)
        if bad:
            raise UnsupportedModel(f"cannot build this model faithfully: {'; '.join(bad)}")
        heads = int(hf["num_attention_heads"])
        head_dim = int(hf.get("head_dim") or hf["hidden_size"] // heads)
        rope = _rope_settings(hf)
        theta = float(hf.get("rope_theta") or rope.get("rope_theta") or 10000.0)
        llama3 = (rope.get("rope_type") or rope.get("type")) == "llama3"
        return cls(
            family=hf["model_type"],
            vocab_size=int(hf["vocab_size"]),
            hidden_size=int(hf["hidden_size"]),
            intermediate_size=int(hf["intermediate_size"]),
            num_layers=int(hf["num_hidden_layers"]),
            num_heads=heads,
            num_kv_heads=int(hf.get("num_key_value_heads") or heads),
            head_dim=head_dim,
            rms_eps=float(hf["rms_norm_eps"]),
            rope_theta=theta,
            rope_freq_factors=llama3_freq_factors(theta, head_dim, rope) if llama3 else None,
            qk_norm=_QK_NORM[hf["model_type"]],
            # the class default of every supported family
            tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
        )
