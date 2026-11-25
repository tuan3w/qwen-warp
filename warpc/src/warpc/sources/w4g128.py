"""compressed-tensors pack-quantized int4, group 128, symmetric (W4G128): `weight_packed` int32 [N, K / 8], nibble e of
word j is column 8j + e and holds the code q + 8; `weight_scale` bf16 [N, K / 128]; the value is (nibble - 8) * scale,
no zero point, no group permutation."""

from __future__ import annotations

import torch

from ..ir import W4G128_GROUP


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """int32 [N, K / 8] -> the nibbles as int32 [N, K], column 8j + e from nibble e of word j."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    return ((packed[..., None] >> shifts) & 15).reshape(*packed.shape[:-1], -1)


def dequantize(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 [N, K]: (nibble - 8) * scale, exact (at most 4 + 8 significant bits)."""
    q = (unpack(packed) - 8).float()
    n, k = q.shape
    if scale.shape != (n, k // W4G128_GROUP):
        raise ValueError(f"W4G128: scales {tuple(scale.shape)} for a [{n}, {k}] matrix")
    return (q.reshape(n, -1, W4G128_GROUP) * scale.float()[..., None]).reshape(n, k)
