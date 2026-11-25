"""A Source wrapper that re-quantises float tensors to W4G128 on demand: the MTP head of the Qwen3.5 27B.

The checkpoint stores the MTP layer in BF16. The hand engine (v0.12.0, engine/pack_mtp.py) quantises it at load to its
W4G128 format, and `Requantized.mtp` reproduces that byte for byte: per row and 128-column group
`scale = bf16(amax * r / 7.5)`, `q = clamp(round(g / scale), -8, 7)`, nibble `q + 8`, with `r` chosen per group by
minimum squared error from {1.0, 0.95, ..., 0.5} (the clip search; the scale is rounded to bf16 before quantising, so
the kernel's `(nibble - 8) * scale` is exactly what was optimised). The draft head is the rows of `lm_head.weight`
listed by `draft_vocab()`, quantised round to nearest (no clip search), exposed as the pseudo-tensor
`mtp.draft_head.weight`.

The vocabulary list (data/draft_vocab_ext_68864.npy, byte-identical to the hand engine's file at v0.12.0) was built
offline from a token-frequency list over English / code text plus 3,314 extension tokens and 14 pad ids; the frequency
list is not part of this repository, so the list cannot be regenerated here (see the comment in the hand engine's
pack_mtp.py).
"""

from __future__ import annotations

from collections.abc import Collection
from importlib import resources
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..errors import UnsupportedModel

if TYPE_CHECKING:
    from . import Source

GROUP = 128
DRAFT_HEAD = "mtp.draft_head.weight"
_HEAD = "lm_head.weight"
_LAYER = "mtp.layers.0."
MTP_MATRICES = (
    "mtp.fc.weight",
    *(f"{_LAYER}self_attn.{p}_proj.weight" for p in "qkvo"),
    *(f"{_LAYER}mlp.{p}_proj.weight" for p in ("gate", "up", "down")),
)
_CLIP_RATIOS = tuple(1.0 - 0.05 * i for i in range(11))


def quantize_w4(w: torch.Tensor, *, clip: bool, device: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """w: bf16 / fp32 [N, K] -> (packed int32 [N, K/8], scale bf16 [N, K/128]) on the CPU; nibble e of word j is
    column 8j+e. Computed on `device` (default: the GPU if there is one); the chunk size is the hand engine's, so the
    GPU result is its result bit for bit. On the CPU the error sums add in another order than the GPU's, so a clip
    ratio whose errors tie to the last float bit can be chosen differently (still a valid quantisation)."""
    if w.dim() != 2 or w.shape[1] % GROUP:
        raise ValueError(f"W4G128 needs a matrix [N, K] with K a multiple of {GROUP}, got {tuple(w.shape)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_rows, k = w.shape
    packed = torch.empty(n_rows, k // 8, dtype=torch.int32)
    scale = torch.empty(n_rows, k // GROUP, dtype=torch.bfloat16)
    rows = max(16, 4_000_000 // k)
    shifts = (4 * torch.arange(8, device=device, dtype=torch.int64)).view(1, 1, 8)
    ratios = _CLIP_RATIOS if clip else (1.0,)
    for r0 in range(0, n_rows, rows):
        g = w[r0 : r0 + rows].to(device).float().view(-1, k // GROUP, GROUP)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_err = best_q = best_s = None
        for ratio in ratios:
            s = (amax * ratio / 7.5).to(torch.bfloat16).float()
            q = torch.round(g / s).clamp(-8, 7)
            err = (q * s - g).pow(2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_q, best_s = err, q, s
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_q = torch.where(m, q, best_q)
                best_s = torch.where(m, s, best_s)
        n = g.shape[0]
        nib = (best_q.to(torch.int64) + 8).view(n, k // 8, 8)
        word = (nib << shifts).sum(-1)  # < 2**32
        word = torch.where(word >= 2**31, word - 2**32, word)
        packed[r0 : r0 + n] = word.to(torch.int32).cpu()
        scale[r0 : r0 + n] = best_s.squeeze(-1).to(torch.bfloat16).cpu()
    return packed, scale


def draft_vocab() -> np.ndarray:
    """The 68,864 ascending token ids of the draft head (int64)."""
    with resources.files(__package__).joinpath("data/draft_vocab_ext_68864.npy").open("rb") as f:
        return np.load(f, allow_pickle=False)


class Requantized:
    """`base` with the float tensors `matrices` served as W4G128 (clip search on when `clip`), plus the pseudo-tensor
    "mtp.draft_head.weight" = base's lm_head rows `draft_ids`, quantised without the clip search. Quantisation runs in
    `raw`, once per call."""

    def __init__(self, base: Source, *, matrices: Collection[str], draft_ids: np.ndarray, clip: bool = True):
        draft_ids = np.asarray(draft_ids)
        if draft_ids.ndim != 1 or draft_ids.dtype != np.int64 or not bool(np.all(np.diff(draft_ids) > 0)):
            raise ValueError("draft_ids must be a strictly ascending int64 vector")
        if draft_ids.size == 0 or draft_ids[0] < 0 or draft_ids[-1] >= base.config.vocab_size:
            raise ValueError(f"draft_ids outside the vocabulary of {base.config.vocab_size} tokens")
        missing = [n for n in (*matrices, _HEAD) if n not in base]
        if missing:
            raise UnsupportedModel(f"nothing to re-quantise: the checkpoint lacks {', '.join(missing)}")
        stored = [n for n in (*matrices, _HEAD) if base.quant(n) is not None]
        if stored:
            raise UnsupportedModel(f"already quantised in the checkpoint: {', '.join(stored)}")
        self.base = base
        self.config = base.config
        self.matrices = frozenset(matrices)
        self.draft_ids = draft_ids
        self.clip = clip

    @classmethod
    def mtp(cls, base: Source) -> Requantized:
        """The hand engine's configuration: the MTP layer and fc with the clip search, the 68,864-row draft head."""
        absent = [n for n in MTP_MATRICES if n not in base]
        if absent:
            raise UnsupportedModel(f"the checkpoint has no MTP head (missing {', '.join(absent[:3])}...)")
        return cls(base, matrices=MTP_MATRICES, draft_ids=draft_vocab(), clip=True)

    def _ours(self, name: str) -> bool:
        return name == DRAFT_HEAD or name in self.matrices

    def __contains__(self, name: str) -> bool:
        return self._ours(name) or name in self.base

    def get(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if self._ours(name):
            raise ValueError(f"{name}: a W4G128 tensor is not a float tensor; read it with raw()")
        return self.base.get(name, dtype)

    def quant(self, name: str) -> str | None:
        return "W4G128" if self._ours(name) else self.base.quant(name)

    def raw(self, name: str) -> dict[str, torch.Tensor]:
        if name == DRAFT_HEAD:
            rows = self.base.get(_HEAD)[torch.from_numpy(self.draft_ids)]
            packed, scale = quantize_w4(rows, clip=False)
        elif name in self.matrices:
            packed, scale = quantize_w4(self.base.get(name), clip=self.clip)
        else:
            return self.base.raw(name)
        return {"packed": packed, "scale": scale}
