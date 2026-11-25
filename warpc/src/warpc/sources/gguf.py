"""GGUF file (llama.cpp checkpoint) addressed by HF tensor names, with an HF-equivalent ModelConfig.

GGUF stores a matrix with ne = [in, out], i.e. row-major [out, in] like HF, so no transpose is needed. llama.cpp's
converter permutes the q / k rows of the "llama" architecture for its adjacent-pair rope; they are permuted back here so
every architecture reads with the half-split rope of the graph. Q8_0 / Q4_0 / Q4_1 are decoded by the functions below
(bit-identical to ggml's dequantize_row_*); the other non-IQ formats by `gguf.dequantize`.
"""

from __future__ import annotations

import re
from pathlib import Path

import gguf
import numpy as np
import torch

from ..errors import UnsupportedModel
from .config import ModelConfig

# GGUF architecture -> (HF model_type, q / k rows permuted by the converter); llama.cpp converts Mistral as "llama"
ARCHS = {"llama": ("llama", True), "qwen3": ("qwen3", False)}
# bytes per 32-element block: fp16 scale (+ fp16 min for Q4_1) and the codes
_BLOCK_BYTES = {"Q8_0": 34, "Q4_0": 18, "Q4_1": 20}

_GLOBAL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}
_LAYER_NAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}
_LAYER = re.compile(r"blk\.(\d+)\.(.+)")
_FLOATS = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def hf_name(gguf_name: str) -> str | None:
    """The HF tensor name of a GGUF tensor, or None if the dense graph does not read it."""
    if gguf_name in _GLOBAL_NAMES:
        return _GLOBAL_NAMES[gguf_name]
    m = _LAYER.fullmatch(gguf_name)
    if m is None or m.group(2) not in _LAYER_NAMES:
        return None
    return f"model.layers.{m.group(1)}.{_LAYER_NAMES[m.group(2)]}"


def _blocks(rows: np.ndarray, block_bytes: int, scales: int) -> tuple[list[np.ndarray], np.ndarray]:
    """uint8 [..., row_bytes] -> (the fp16 scales heading each block as fp32 [..., blocks, 1], the block payloads)."""
    b = rows.reshape(*rows.shape[:-1], -1, block_bytes)
    head = [b[..., 2 * i : 2 * i + 2].copy().view("<f2").astype(np.float32) for i in range(scales)]
    return head, b[..., 2 * scales :]


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """Byte j of a 32-element block holds element j in its low nibble and element j + 16 in its high nibble."""
    return np.concatenate([qs & 0x0F, qs >> 4], -1).astype(np.float32)


def dequantize_q8_0(rows: np.ndarray) -> np.ndarray:
    """Q8_0 blocks (fp16 d, 32 x int8 q) -> fp32 d * q."""
    (d,), qs = _blocks(rows, _BLOCK_BYTES["Q8_0"], 1)
    return (d * qs.view(np.int8).astype(np.float32)).reshape(*rows.shape[:-1], -1)


def dequantize_q4_0(rows: np.ndarray) -> np.ndarray:
    """Q4_0 blocks (fp16 d, 16 bytes of nibbles q) -> fp32 d * (q - 8)."""
    (d,), qs = _blocks(rows, _BLOCK_BYTES["Q4_0"], 1)
    return (d * (_nibbles(qs) - 8)).reshape(*rows.shape[:-1], -1)


def dequantize_q4_1(rows: np.ndarray) -> np.ndarray:
    """Q4_1 blocks (fp16 d, fp16 m, 16 bytes of nibbles q) -> fp32 d * q + m."""
    (d, m), qs = _blocks(rows, _BLOCK_BYTES["Q4_1"], 2)
    return (d * _nibbles(qs) + m).reshape(*rows.shape[:-1], -1)


DECODERS = {"Q8_0": dequantize_q8_0, "Q4_0": dequantize_q4_0, "Q4_1": dequantize_q4_1}


def _unpermute(rows: np.ndarray, heads: int) -> np.ndarray:
    """Inverse of llama.cpp's q / k permute: per head, rows (pair, 2) back to (2, pair)."""
    return rows.reshape(heads, -1, 2, *rows.shape[1:]).swapaxes(1, 2).reshape(rows.shape)


def _f32(value) -> float:
    """The float64 of a float32's shortest decimal, as an HF config.json holds it (1e-06, not 9.99999997e-07)."""
    return float(str(np.float32(value)))


class GGUFSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        reader = gguf.GGUFReader(self.path)

        def value(name: str):
            field = reader.get_field(name)
            return None if field is None else field.contents()

        arch = value(gguf.Keys.General.ARCHITECTURE)
        if arch not in ARCHS:
            raise UnsupportedModel(f"GGUF architecture {arch!r} (supported: {', '.join(ARCHS)})")
        model_type, permuted = ARCHS[arch]
        tensors = {t.name: t for t in reader.tensors}
        iq = sorted(n for n, t in tensors.items() if t.tensor_type.name.startswith("IQ"))
        if iq:
            raise UnsupportedModel(f"IQ* block formats are not supported: {', '.join(iq[:4])}")
        unknown = [n for n in tensors if hf_name(n) is None and not n.endswith(".bias") and n != "rope_freqs.weight"]
        if unknown:
            raise UnsupportedModel(f"GGUF tensors the dense graph does not read: {', '.join(unknown[:4])}")

        def key(k):
            return value(k.format(arch=arch))

        heads, kv_heads = key(gguf.Keys.Attention.HEAD_COUNT), key(gguf.Keys.Attention.HEAD_COUNT_KV)
        if isinstance(kv_heads, list):
            if len(set(kv_heads)) != 1:
                raise UnsupportedModel(f"per-layer KV head counts {sorted(set(kv_heads))}")
            kv_heads = kv_heads[0]
        scaling = key(gguf.Keys.Rope.SCALING_TYPE)
        if scaling in (None, "none") and "rope_freqs.weight" in tensors:
            scaling = "llama3"  # llama.cpp stores Llama 3.1 rope scaling as a frequency-factor tensor
        window = key(gguf.Keys.Attention.SLIDING_WINDOW)
        eps, theta = key(gguf.Keys.Attention.LAYERNORM_RMS_EPS), key(gguf.Keys.Rope.FREQ_BASE)
        hf = {
            "model_type": model_type,
            "vocab_size": int(tensors["token_embd.weight"].shape[1]) if "token_embd.weight" in tensors else None,
            "hidden_size": key(gguf.Keys.LLM.EMBEDDING_LENGTH),
            "intermediate_size": key(gguf.Keys.LLM.FEED_FORWARD_LENGTH),
            "num_hidden_layers": key(gguf.Keys.LLM.BLOCK_COUNT),
            "num_attention_heads": heads,
            "num_key_value_heads": kv_heads,
            "rms_norm_eps": None if eps is None else _f32(eps),
            "rope_theta": 10000.0 if theta is None else _f32(theta),
            "rope_scaling": None if scaling in (None, "none") else {"rope_type": scaling},
            "sliding_window": window,
            "attention_bias": any(re.fullmatch(r"blk\.\d+\.attn_.*\.bias", n) for n in tensors),
            "mlp_bias": any(re.fullmatch(r"blk\.\d+\.ffn_.*\.bias", n) for n in tensors),
            "tie_word_embeddings": "output.weight" not in tensors,
        }
        if heads and "blk.0.attn_q.weight" in tensors:
            hf["head_dim"] = int(tensors["blk.0.attn_q.weight"].shape[1]) // heads
        self.config = ModelConfig.from_hf(hf)

        self._index: dict[str, tuple[gguf.ReaderTensor, int]] = {}
        for n, t in tensors.items():
            name = hf_name(n)
            if name is None:
                continue
            unpermute = 0
            if permuted and name.endswith("q_proj.weight"):
                unpermute = self.config.num_heads
            elif permuted and name.endswith("k_proj.weight"):
                unpermute = self.config.num_kv_heads
            self._index[name] = (t, unpermute)

    def __contains__(self, name: str) -> bool:
        return name in self._index

    def get(self, name: str) -> torch.Tensor:
        t, heads = self._index[name]
        kind = t.tensor_type.name
        shape = tuple(int(n) for n in reversed(t.shape))
        if kind in _FLOATS:
            x = np.ascontiguousarray(t.data).view(_FLOATS[kind]).reshape(shape)
        elif kind in DECODERS:
            x = DECODERS[kind](np.asarray(t.data)).reshape(shape)
        else:
            x = np.asarray(gguf.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape)
        if heads:
            x = _unpermute(x, heads)
        out = torch.from_numpy(np.require(x, requirements=("C", "W")))
        return out.view(torch.bfloat16) if kind == "BF16" else out.to(torch.bfloat16)

    def quant(self, name: str) -> str | None:
        kind = self._index[name][0].tensor_type.name
        return None if kind in _FLOATS else kind

    def raw(self, name: str) -> np.ndarray:
        t, heads = self._index[name]
        if t.tensor_type.name in _FLOATS or len(t.shape) != 2:
            raise ValueError(f"{name}: {t.tensor_type.name} {len(t.shape)}-D tensor has no block rows")
        rows = np.asarray(t.data).reshape(int(t.shape[1]), -1)
        return np.ascontiguousarray(_unpermute(rows, heads) if heads else rows)
