"""Checkpoint access + compressed-tensors W4A16 (pack-quantized) dequantisation.

Everything here is CPU/torch only. No GPU, no transformers dependency.

Checkpoint facts (see NOTES.md for the full table):
  * `model.safetensors.index.json` maps tensor name -> shard. MTP tensors (`mtp.*`) live in
    `model_mtp.safetensors` and are also listed in the index.
  * A quantised Linear `X` is stored as three tensors
        X.weight_packed : int32  [N, K/8]   8 nibbles / int32, nibble i (LSB first) = column 8*j+i
        X.weight_scale  : bf16   [N, K/128] one scale per (row, 128-column group)
        X.weight_shape  : int64  [2] = [N, K]
    Symmetric uint4b8 convention:  w[n,k] = (nibble[n,k] - 8) * scale[n, k//128].
  * There is NO `weight_g_idx` / zero-point tensor in this checkpoint (checked over the whole index:
    `QuantizedLinear.has_g_idx()` is asserted false), although the config says actorder=static.
    Groups are therefore the contiguous blocks k//128.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

MODEL_DIR = Path(os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4")))
GROUP = 128
PACK = 8  # nibbles per int32


class Checkpoint:
    """Lazy by-name access to the safetensors shards via the index json (mmap, no bulk load)."""

    def __init__(self, model_dir: Path | str = MODEL_DIR):
        self.dir = Path(model_dir)
        self.index: dict[str, str] = json.load(open(self.dir / "model.safetensors.index.json"))["weight_map"]
        self._handles: dict[str, object] = {}

    def _h(self, shard: str):
        h = self._handles.get(shard)
        if h is None:
            h = self._handles[shard] = safe_open(str(self.dir / shard), framework="pt", device="cpu")
        return h

    def names(self, prefix: str = "") -> list[str]:
        return sorted(n for n in self.index if n.startswith(prefix))

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> torch.Tensor:
        return self._h(self.index[name]).get_tensor(name)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._h(self.index[name]).get_slice(name).get_shape())

    def dtype(self, name: str) -> str:
        return self._h(self.index[name]).get_slice(name).get_dtype()

    def iter_tensors(self, prefix: str = ""):
        """Yield (name, tensor) in index order restricted to `prefix` (one tensor materialised at a time)."""
        for n in self.names(prefix):
            yield n, self.get(n)

    def quantized_linear(self, base: str) -> "QuantizedLinear":
        return QuantizedLinear.load(self, base)


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """int32 [N, K/8] -> uint8 [N, K] raw nibbles in 0..15 (column 8*j+i = bits 4i..4i+3 of word j).

    Little-endian: byte b of the row holds columns (2b, 2b+1) = (low nibble, high nibble).
    """
    assert packed.dtype == torch.int32
    b = packed.contiguous().view(torch.uint8)  # [N, K/2]
    lo = b & 0xF
    hi = b >> 4
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def dequant_w4a16(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Dequantise a compressed-tensors W4A16 symmetric g128 weight to `dtype`, shape [N, K]."""
    n, k8 = packed.shape
    k = k8 * PACK
    q = unpack_nibbles(packed).to(torch.float32).sub_(8.0)  # [N, K] in -8..7
    q = q.view(n, k // GROUP, GROUP).mul_(scale.to(torch.float32).unsqueeze(-1))
    return q.view(n, k).to(dtype)


def quantize_w4a16(w: torch.Tensor, group: int = GROUP):
    """Symmetric int4 g128 quantiser matching compressed-tensors `memoryless_minmax` (scale=absmax/7.5, q in -8..7).
    Returns (packed int32 [N,K/8], scale bf16 [N,K/g]). Used only by the self-test."""
    n, k = w.shape
    wg = w.float().view(n, k // group, group)
    scale = (wg.abs().amax(-1) / 7.5).clamp_min(1e-12).to(torch.bfloat16)
    q = torch.clamp(torch.round(wg / scale.float().unsqueeze(-1)), -8, 7).to(torch.int32) + 8  # 0..15
    q = q.view(n, k // PACK, PACK)
    shifts = torch.arange(PACK, dtype=torch.int32) * 4
    packed = (q << shifts).sum(-1, dtype=torch.int32)
    return packed, scale


class QuantizedLinear:
    """A W4A16 linear kept packed in RAM (4.5 bit/weight); dequantised on the fly in row chunks."""

    def __init__(self, packed: torch.Tensor, scale: torch.Tensor, n: int, k: int):
        assert packed.shape == (n, k // PACK) and scale.shape == (n, k // GROUP)
        self.packed, self.scale, self.n, self.k = packed, scale, n, k

    @classmethod
    def load(cls, ck: Checkpoint, base: str) -> "QuantizedLinear":
        assert not ck.has(base + ".weight_g_idx") and not ck.has(base + ".weight_zero_point"), base
        shp = ck.get(base + ".weight_shape").tolist()
        return cls(ck.get(base + ".weight_packed"), ck.get(base + ".weight_scale"), int(shp[0]), int(shp[1]))

    def dequant(self, dtype=torch.float32) -> torch.Tensor:
        return dequant_w4a16(self.packed, self.scale, dtype)

    def matmul(self, x: torch.Tensor, chunk_rows: int = 256) -> torch.Tensor:
        """x [M, K] float32 -> [M, N] float32 == x @ W^T, never materialising the full fp32 W."""
        out = torch.empty(x.shape[0], self.n, dtype=torch.float32)
        for r in range(0, self.n, chunk_rows):
            w = dequant_w4a16(self.packed[r : r + chunk_rows], self.scale[r : r + chunk_rows])
            out[:, r : r + chunk_rows] = x @ w.T
        return out


class DenseLinear:
    """BF16 (unquantised) linear kept as bf16, multiplied in fp32 in row chunks (lm_head, MTP, in_proj_a/b)."""

    def __init__(self, w: torch.Tensor):
        self.w = w  # bf16 [N, K]
        self.n, self.k = w.shape

    def matmul(self, x: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
        if self.n <= chunk_rows:
            return x @ self.w.float().T
        out = torch.empty(x.shape[0], self.n, dtype=torch.float32)
        for r in range(0, self.n, chunk_rows):
            out[:, r : r + chunk_rows] = x @ self.w[r : r + chunk_rows].float().T
        return out


def load_state_dict(ck: Checkpoint, prefix: str, dtype=torch.float32) -> dict[str, torch.Tensor]:
    """All tensors under `prefix` (e.g. 'model.language_model.layers.3.') as an HF-style state dict keyed by the name
    relative to `prefix`: every quantised `X.weight_packed/_scale/_shape` triple becomes a dense `X.weight` (dequantised
    to `dtype`), `*_scale` KV-scales (k_scale/v_scale) and everything else are passed through (floating ones as `dtype`)."""
    sd: dict[str, torch.Tensor] = {}
    for n in ck.names(prefix):
        rel = n[len(prefix):]
        if rel.endswith(".weight_packed"):
            base = n[: -len(".weight_packed")]
            sd[rel[: -len("_packed")]] = ck.quantized_linear(base).dequant(dtype)
        elif rel.endswith((".weight_scale", ".weight_shape")):
            continue
        else:
            t = ck.get(n)
            sd[rel] = t.to(dtype) if t.is_floating_point() else t
    return sd


# --------------------------------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------------------------------
def selftest(ck: Checkpoint | None = None) -> None:
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32

    ck = ck or Checkpoint()
    # 1. no g_idx / zero-point anywhere in the index
    bad = [n for n in ck.index if n.endswith(("weight_g_idx", "weight_zero_point"))]
    assert not bad, bad

    base = "model.language_model.layers.3.self_attn.k_proj"  # small: [1024, 5120]
    ql = ck.quantized_linear(base)
    # 2. our unpack == compressed-tensors unpack_from_int32 (offset: value = nibble - 8)
    ref = unpack_from_int32(ql.packed, 4, torch.Size([ql.n, ql.k]))  # int8, already offset: nibble - 8
    ours = unpack_nibbles(ql.packed)
    assert torch.equal(ref.to(torch.int32) + 8, ours.to(torch.int32)), "nibble order differs from compressed-tensors"
    assert int(ours.max()) == 15 and int(ours.min()) == 0
    # 3. our packing == compressed-tensors pack_to_int32
    q_signed = (ours.to(torch.int16) - 8).to(torch.int8)
    assert torch.equal(pack_to_int32(q_signed, 4), ql.packed), "re-pack mismatch vs compressed-tensors"
    # 4. dequant -> requantise with the checkpoint's own scales reproduces the nibbles exactly
    w = ql.dequant()
    q2 = torch.round(w.view(ql.n, -1, GROUP) / ql.scale.float().unsqueeze(-1)).clamp(-8, 7).reshape(ql.n, ql.k) + 8
    assert torch.equal(q2.to(torch.uint8), ours), "dequant->requant not idempotent"
    # 5. self-made quantiser round trip: quantise random fp32, pack, dequant and check error <= 0.6*scale (extreme clamp 7.5->7 + bf16 scale rounding)
    x = torch.randn(64, 512) * 0.05
    p, s = quantize_w4a16(x)
    xd = dequant_w4a16(p, s)
    err = (x - xd).abs().view(64, -1, GROUP)
    assert bool((err <= s.float().unsqueeze(-1) * 0.6 + 1e-6).all())
    # 6. chunked matmul == dense matmul
    xin = torch.randn(3, ql.k)
    assert torch.allclose(ql.matmul(xin, chunk_rows=300), xin @ w.T, rtol=1e-4, atol=1e-4)
    print(f"weights selftest OK  ({base}: N={ql.n} K={ql.k}, w std={w.std():.4f}, mean nibble={ours.float().mean():.3f})")


if __name__ == "__main__":
    selftest()
