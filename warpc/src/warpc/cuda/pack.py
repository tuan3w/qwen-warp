"""Packer: the device tensors of the CUDA backend's packs, read from a Source.

dense      one weight as the source stores it
dense_cat  the rows of the dense `parts`, concatenated in order
dense_t    one rank-2 dense weight transposed, [K, N] -> [N, K] contiguous
f32        one f32-scheme weight as the source stores it, fp32 on the device
q4_0       one GGUF Q4_0 matrix as stored: raw blocks [N, K / 32 * 18] bytes, read in place by the fused dequant GEMV
q4_0_cat   the rows of the Q4_0 `parts`, concatenated in order

and the kernel families' own (`FAMILY_PACKS`): w4g128_experts, one MoE layer's stacked gate, up and down.
"""

from __future__ import annotations

import math

import torch

from ..compiler import Pack
from ..ir import Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS, WeightDecl
from ..sources import Source
from . import attn_gated, deltanet, head_refine, hybrid, inkling, w4

PACK_KINDS = ("dense", "dense_cat", "dense_t", "f32", "q4_0", "q4_0_cat")
# layouts only a kernel family reads: kind -> (packer, pack) -> device tensor
FAMILY_PACKS = {
    **inkling.PACKS,
    **hybrid.PACKS,
    **w4.PACKS,
    **attn_gated.PACKS,
    **deltanet.PACKS,
    **head_refine.PACKS,
}


class Packer:
    """Builds packs on `device`. A dense weight read by several packs (a tied embedding and lm_head) is one device
    tensor; concatenated packs are assembled on the host, so the device holds only the result."""

    def __init__(self, weights: dict[str, WeightDecl], source: Source, device: str = "cuda"):
        self.weights, self.source, self.device = weights, source, device
        self._dense: dict[str, torch.Tensor] = {}

    def _host_dense(self, name: str) -> torch.Tensor:
        d = self.weights[name]
        if d.scheme != "dense":
            raise ValueError(f"{name}: a dense pack of a {d.scheme} weight")
        t = self.source.get(d.source)
        if t.numel() != math.prod(d.shape):
            raise ValueError(f"{name}: source {d.source} has shape {tuple(t.shape)}, the graph declares {d.shape}")
        return t.reshape(d.shape).to(torch.bfloat16)

    def _host_f32(self, name: str) -> torch.Tensor:
        d = self.weights[name]
        if d.scheme != "f32":
            raise ValueError(f"{name}: an f32 pack of a {d.scheme} weight")
        t = self.source.get(d.source, torch.float32)
        if t.numel() != math.prod(d.shape):
            raise ValueError(f"{name}: source {d.source} has shape {tuple(t.shape)}, the graph declares {d.shape}")
        return t.reshape(d.shape).float()

    def _host_q4(self, name: str) -> torch.Tensor:
        d = self.weights[name]
        if d.scheme != "q4_0" or len(d.shape) != 2 or d.shape[1] % Q4_0_BLOCK_ELEMS:
            raise ValueError(f"{name}: a q4_0 pack of a {d.scheme} weight of shape {d.shape}")
        raw = self.source.raw(d.source)["blocks"]
        if tuple(raw.shape) != (d.shape[0], d.shape[1] // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES):
            raise ValueError(f"{name}: Q4_0 blocks of shape {tuple(raw.shape)} for a {list(d.shape)} matrix")
        return raw

    def build(self, pack: Pack) -> torch.Tensor:
        if pack.kind in FAMILY_PACKS:
            return FAMILY_PACKS[pack.kind](self, pack)
        if pack.kind not in PACK_KINDS:
            raise ValueError(
                f"pack {pack.name}: kind {pack.kind!r} is not the CUDA backend's"
                f" ({', '.join((*PACK_KINDS, *FAMILY_PACKS))})"
            )
        if pack.kind == "dense":
            source = self.weights[pack.parts[0]].source
            if source not in self._dense:
                self._dense[source] = self._host_dense(pack.parts[0]).contiguous().to(self.device)
            return self._dense[source]
        if pack.kind == "dense_cat":
            return torch.cat([self._host_dense(p) for p in pack.parts], 0).to(self.device)
        if pack.kind == "dense_t":
            if len(self.weights[pack.parts[0]].shape) != 2:
                raise ValueError(f"{pack.name}: a transposed pack of {self.weights[pack.parts[0]].shape}")
            return self._host_dense(pack.parts[0]).t().contiguous().to(self.device)
        if pack.kind == "f32":
            return self._host_f32(pack.parts[0]).contiguous().to(self.device)
        if pack.kind == "q4_0":
            return self._host_q4(pack.parts[0]).contiguous().to(self.device)
        return torch.cat([self._host_q4(p) for p in pack.parts], 0).to(self.device)
