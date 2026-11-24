"""Device layouts of the hand kernels' weights, built from a Source on the GPU.

kind        parts                  layout
dense       one weight             as stored, reshaped to the declared shape (conv1d [C,1,W] -> [C,W])
dense_cat   in_proj_a, in_proj_b   rows concatenated, a first ([96, 5120])
w4r         one W4G128 linear      smk_w4r_gemv's repacked layout (repack_w4)
w4r_cat     2 or 3 W4G128 linears  rows concatenated in order, then repacked (segment ends are multiples of 16)
w4r_gateup  gate, up               rows interleaved per 8 (gate 8i..8i+7, then up 8i..8i+7), then repacked
head_i4r    lm_head (bf16)         int4 g128 round-to-nearest copy (quant_head_i4), then repacked

A W4 pack is one uint8 tensor: the int32 words [N, K/8] followed by the scales [N, K/128] (2 bytes each), so the scales
start at byte `w4_scale_offset(N, K)`. Both halves keep the checkpoint's byte counts; `repack_w4` only permutes them.
"""

from __future__ import annotations

import math

import torch

from ..compiler import Pack
from ..ir import W4G128_GROUP, WeightDecl
from ..sources import Source

KINDS = ("dense", "dense_cat", "w4r", "w4r_cat", "w4r_gateup", "head_i4r")
_ROWS = 2048  # rows per repack / quantisation chunk: bounds the int64 temporaries to ~0.3 GB at K = 17408


def w4_scale_offset(n: int, k: int) -> int:
    return n * k // 2


def w4_empty(n: int, k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(storage, words int32 [n, k/8], scales bf16 [n, k/128]): the two views share the one uint8 storage."""
    off = w4_scale_offset(n, k)
    buf = torch.empty(off + n * (k // W4G128_GROUP) * 2, dtype=torch.uint8, device=device)
    return buf, buf[:off].view(torch.int32).view(n, k // 8), buf[off:].view(torch.bfloat16).view(n, k // W4G128_GROUP)


def repack_w4(w: torch.Tensor, s: torch.Tensor) -> None:
    """In place, natural W4 (nibble e of word j = column 8j+e) -> the smk_w4r_gemv layout (engine.h): per (16-row tile,
    128-k group) one contiguous 1 KB unit [row g, 16 B chunk t (32 lanes)][row g+8, chunk t], the units of a tile
    contiguous; inside a word nibble j / j+4 = columns 2j / 2j+1; one u32 scale per (tile, group, g) = bf16 row g |
    bf16 row g+8 << 16."""
    n_rows, k8 = w.shape
    nkg = k8 // 16
    if n_rows % 16 or k8 % 16:
        raise ValueError(f"W4 repack needs N % 16 == 0 and K % 128 == 0, got [{n_rows}, {k8 * 8}]")
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=w.device)
    for r0 in range(0, n_rows, _ROWS):
        n = min(_ROWS, n_rows - r0)
        nt = n // 16
        wc = w[r0 : r0 + n].view(nt, 2, 8, nkg, 4, 4)  # [tile, half, g, kg, t, u]
        nib = ((wc.to(torch.int64).unsqueeze(-1) >> sh) & 0xF)[..., order]
        word = (nib << sh).sum(-1)  # < 2**32
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        w[r0 : r0 + n].copy_(word.permute(0, 3, 1, 2, 4, 5).reshape(n, k8))  # [tile, kg, half, g, t, u]
        sc = s[r0 : r0 + n].view(nt, 2, 8, nkg).permute(0, 3, 2, 1).contiguous()  # [tile, kg, g, half]
        s[r0 : r0 + n].copy_(sc.view(n, nkg))


def quant_head_i4(w: torch.Tensor, ow: torch.Tensor, os_: torch.Tensor) -> None:
    """int4 g128 round-to-nearest of the bf16 rows `w` into natural W4 `ow` / `os_`: scale = bf16(amax / 7) per 128
    columns, q = round(w / scale) clamped to [-8, 7], stored as q + 8 (the hand engine's K9 head, bit for bit)."""
    n, k = w.shape
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    g = w.float().view(n, k // W4G128_GROUP, W4G128_GROUP)
    sb = (g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7).to(torch.bfloat16).float()
    q = torch.round(g / sb).clamp_(-8, 7)
    word = ((q.to(torch.int64) + 8).view(n, k // 8, 8) << sh).sum(-1)
    ow.copy_(torch.where(word >= 2**31, word - 2**32, word).to(torch.int32))
    os_.copy_(sb.squeeze(-1).to(torch.bfloat16))


class Packer:
    """Builds the device tensors of swiftmk packs from `source` on `device`."""

    def __init__(self, weights: dict[str, WeightDecl], source: Source, device: str = "cuda"):
        self.weights, self.source, self.device = weights, source, device

    def _decl(self, name: str, scheme: str) -> WeightDecl:
        d = self.weights[name]
        if d.scheme != scheme:
            raise ValueError(f"{name}: a {scheme} pack of a {d.scheme} weight")
        return d

    def _dense(self, name: str) -> torch.Tensor:
        d = self._decl(name, "dense")
        t = self.source.get(d.source)
        if t.numel() != math.prod(d.shape):
            raise ValueError(f"{name}: source {d.source} has shape {tuple(t.shape)}, the graph declares {d.shape}")
        return t.reshape(d.shape).to(torch.bfloat16)

    def _w4(self, name: str) -> dict[str, torch.Tensor]:
        d = self._decl(name, "w4g128")
        raw = self.source.raw(d.source)
        n, k = d.shape
        if tuple(raw["packed"].shape) != (n, k // 8) or tuple(raw["scale"].shape) != (n, k // W4G128_GROUP):
            raise ValueError(f"{name}: W4G128 tensors {tuple(raw['packed'].shape)} / {tuple(raw['scale'].shape)}")
        return raw

    def build(self, pack: Pack) -> torch.Tensor:
        kind, parts = pack.kind, pack.parts
        if kind == "dense":
            return self._dense(parts[0]).contiguous().to(self.device)
        if kind == "dense_cat":
            return torch.cat([self._dense(p) for p in parts], 0).to(self.device)
        if kind in ("w4r", "w4r_cat"):
            if (kind == "w4r") != (len(parts) == 1):
                raise ValueError(f"pack {pack.name}: {kind} of {len(parts)} weights")
            k = self.weights[parts[0]].shape[1]
            n = sum(self.weights[p].shape[0] for p in parts)
            buf, w, s = w4_empty(n, k, self.device)
            r = 0
            for p in parts:
                raw = self._w4(p)
                m = raw["packed"].shape[0]
                if raw["packed"].shape[1] * 8 != k:
                    raise ValueError(f"pack {pack.name}: {p} has K {raw['packed'].shape[1] * 8}, not {k}")
                w[r : r + m].copy_(raw["packed"])
                s[r : r + m].copy_(raw["scale"])
                r += m
            repack_w4(w, s)
            return buf
        if kind == "w4r_gateup":
            gate, up = (self._w4(p) for p in parts)
            n, k8 = gate["packed"].shape
            if gate["packed"].shape != up["packed"].shape or n % 8:
                raise ValueError(f"pack {pack.name}: gate / up shapes differ or rows are not a multiple of 8")
            buf, w, s = w4_empty(2 * n, k8 * 8, self.device)
            for dst, key in ((w, "packed"), (s, "scale")):
                v = dst.view(n // 8, 2, 8, dst.shape[1])
                v[:, 0].copy_(gate[key].view(n // 8, 8, -1))
                v[:, 1].copy_(up[key].view(n // 8, 8, -1))
            repack_w4(w, s)
            return buf
        if kind == "head_i4r":
            d = self._decl(parts[0], "dense")
            n, k = d.shape
            buf, w, s = w4_empty(n, k, self.device)
            host = self._dense(parts[0])
            for r0 in range(0, n, _ROWS):
                rows = host[r0 : r0 + _ROWS].to(self.device)
                quant_head_i4(rows, w[r0 : r0 + _ROWS], s[r0 : r0 + _ROWS])
            repack_w4(w, s)
            return buf
        raise ValueError(f"pack {pack.name}: kind {kind!r} is not one of {KINDS}")
