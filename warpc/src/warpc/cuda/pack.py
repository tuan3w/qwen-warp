"""Packer: the device tensors of the CUDA backend's packs, read from a Source.

dense      one weight as the source stores it
dense_cat  the rows of the dense `parts`, concatenated in order
dense_t    one rank-2 dense weight transposed, [K, N] -> [N, K] contiguous
f32        one f32-scheme weight as the source stores it, fp32 on the device
q4_0       one GGUF Q4_0 matrix repacked into the planes of the fused dequant GEMV (`repack_q4`)
q4_0_cat   the rows of the Q4_0 `parts`, concatenated in order
q4_1, q4_1_cat   the same for GGUF Q4_1 matrices
q6_k       one GGUF Q6_K matrix repacked for the fused head GEMV (`repack_q6_k`)

and the kernel families' own (`FAMILY_PACKS`): w4g128_experts, one MoE layer's stacked gate, up and down.
"""

from __future__ import annotations

import math

import torch

from ..compiler import Pack
from ..ir import BLOCK_FORMATS, WeightDecl
from ..sources import Source
from . import attn_gated, deltanet, head_refine, hybrid, inkling, schedule, w4

PACK_KINDS = ("dense", "dense_cat", "dense_t", "f32", "q4_0", "q4_0_cat", "q4_1", "q4_1_cat", "q6_k")
# layouts only a kernel family reads: kind -> (packer, pack) -> device tensor
FAMILY_PACKS = {**inkling.PACKS, **hybrid.PACKS, **w4.PACKS, **attn_gated.PACKS, **deltanet.PACKS, **head_refine.PACKS}

_CHUNK_ROWS = 16384  # rows repacked at a time on the device (bounds the temporaries of the 150k-row heads)


def repack_q4(raw: torch.Tensor, scheme: str) -> torch.Tensor:
    """GGUF Q4_0 / Q4_1 blocks uint8 [N, K / 32 * block] -> the planes of kernels/dense_q4_gemv.cuh, uint8 [N, row].

    Per block the 32 nibbles (columns 0..15 in the low nibbles of the 16 bytes, 16..31 in the high ones) are taken in
    column order and cut into the 8-column words of k_gemv's lane map (chunk k = 32 j + lane in round j); the planes
    hold, per pair of rounds, the words of the 32 lanes (both rounds of a lane adjacent), then the scales (Q4_1: then
    the minimums) of the 8 blocks each round's lanes read, in (group, round) order."""
    elems, block = BLOCK_FORMATS[scheme]
    n = raw.shape[0]
    blocks = raw.reshape(n, -1, block)
    columns = blocks.shape[1] * elems
    pairs = columns // 512
    codes = blocks[..., block - 16 :]
    nibbles = torch.cat([codes & 15, codes >> 4], -1)  # column order
    words = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(n, pairs, 2, 32, 4)  # (pair, round, lane, byte)
    planes = [words.permute(0, 1, 3, 2, 4).reshape(n, -1)]
    for h in range((block - 16) // 2):  # the fp16 values heading a block: d (and m)
        head = blocks[..., 2 * h : 2 * h + 2].reshape(n, pairs, 2, 8, 2)  # (pair, round, group, byte)
        planes.append(head.permute(0, 1, 3, 2, 4).reshape(n, -1))
    return _padded(torch.cat(planes, 1), schedule.packed_row_bytes(scheme, columns))


def repack_q6_k(raw: torch.Tensor) -> torch.Tensor:
    """GGUF Q6_K blocks uint8 [N, K / 256 * 210] -> the planes of kernels/dense_q6_gemv.cuh, uint8 [N, row].

    A block's 256 codes (6 bits: ggml's ql nibble and qh pair) are taken in column order; lane l of the warp owns
    columns 8l .. 8l + 7: its 8 low nibbles are one little-endian u32, its 8 high pairs one u16."""
    n = raw.shape[0]
    blocks = raw.reshape(n, -1, 210)
    nb = blocks.shape[1]
    ql = blocks[..., :128].reshape(n, nb, 2, 1, 64)
    ql = torch.cat([ql & 15, ql >> 4], 3).reshape(n, nb, 256)
    shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=raw.device).reshape(1, 1, 1, 4, 1)
    qh = ((blocks[..., 128:192].reshape(n, nb, 2, 1, 32) >> shifts) & 3).reshape(n, nb, 256)
    lanes = (ql | (qh << 4)).reshape(n, nb, 32, 8)
    low, high = lanes & 15, (lanes >> 4).reshape(n, nb, 32, 2, 4)
    low_bytes = (low[..., 0::2] | (low[..., 1::2] << 4)).reshape(n, -1)
    high_bytes = (high[..., 0] | (high[..., 1] << 2) | (high[..., 2] << 4) | (high[..., 3] << 6)).reshape(n, -1)
    planes = [low_bytes, high_bytes, blocks[..., 192:208].reshape(n, -1), blocks[..., 208:210].reshape(n, -1)]
    return _padded(torch.cat(planes, 1), schedule.packed_row_bytes("q6_k", nb * 256))


def _padded(rows: torch.Tensor, row_bytes: int) -> torch.Tensor:
    return torch.nn.functional.pad(rows, (0, row_bytes - rows.shape[1])) if rows.shape[1] != row_bytes else rows


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

    def _packed(self, name: str, scheme: str) -> torch.Tensor:
        """The repacked rows of the block-format weight `name`, on the device."""
        d = self.weights[name]
        elems, block = BLOCK_FORMATS[scheme]
        if d.scheme != scheme or len(d.shape) != 2 or d.shape[1] % elems:
            raise ValueError(f"{name}: a {scheme} pack of a {d.scheme} weight of shape {d.shape}")
        raw = self.source.raw(d.source)["blocks"]
        if tuple(raw.shape) != (d.shape[0], d.shape[1] // elems * block):
            raise ValueError(f"{name}: {scheme} blocks of shape {tuple(raw.shape)} for a {list(d.shape)} matrix")
        repack = repack_q6_k if scheme == "q6_k" else lambda r: repack_q4(r, scheme)
        row_bytes = schedule.packed_row_bytes(scheme, d.shape[1])
        out = torch.empty(d.shape[0], row_bytes, dtype=torch.uint8, device=self.device)
        for i in range(0, d.shape[0], _CHUNK_ROWS):
            out[i : i + _CHUNK_ROWS] = repack(raw[i : i + _CHUNK_ROWS].to(self.device))
        return out

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
        scheme = pack.kind.removesuffix("_cat")
        parts = [self._packed(p, scheme) for p in pack.parts]
        return parts[0] if len(parts) == 1 else torch.cat(parts, 0)
