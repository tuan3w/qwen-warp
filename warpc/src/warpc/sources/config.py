"""ModelConfig: the hyperparameters the dense frontend reads, and the guard that refuses what it cannot reproduce."""

from __future__ import annotations

from dataclasses import dataclass

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


def _unsupported_settings(hf: dict) -> list[str]:
    """Every setting of an HF config.json dict that the dense graph would silently get wrong, by name."""
    bad = []
    model_type = hf.get("model_type")
    if model_type not in _QK_NORM:
        bad.append(f"model_type {model_type!r} (supported: {', '.join(_QK_NORM)})")
    rope = hf.get("rope_scaling") or hf.get("rope_parameters") or {}
    kind = rope.get("rope_type") or rope.get("type")
    if kind not in (None, "default"):
        bad.append(f"rope scaling '{kind}' (rope_scaling / rope_parameters)")
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
    qk_norm: bool
    tie_embeddings: bool

    @classmethod
    def from_hf(cls, hf: dict) -> ModelConfig:
        """From an HF config.json dict; raises UnsupportedModel naming each unsupported setting."""
        bad = _unsupported_settings(hf)
        if bad:
            raise UnsupportedModel(f"cannot build this model faithfully: {'; '.join(bad)}")
        heads = int(hf["num_attention_heads"])
        rope = hf.get("rope_parameters") or {}
        theta = hf.get("rope_theta") or rope.get("rope_theta") or 10000.0
        return cls(
            family=hf["model_type"],
            vocab_size=int(hf["vocab_size"]),
            hidden_size=int(hf["hidden_size"]),
            intermediate_size=int(hf["intermediate_size"]),
            num_layers=int(hf["num_hidden_layers"]),
            num_heads=heads,
            num_kv_heads=int(hf.get("num_key_value_heads") or heads),
            head_dim=int(hf.get("head_dim") or hf["hidden_size"] // heads),
            rms_eps=float(hf["rms_norm_eps"]),
            rope_theta=float(theta),
            qk_norm=_QK_NORM[hf["model_type"]],
            # the class default of every supported family
            tie_embeddings=bool(hf.get("tie_word_embeddings", False)),
        )
