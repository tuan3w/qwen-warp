"""Checkpoint sources: tensors addressed by HF names plus a ModelConfig, whatever the file format."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from .config import Gemma4Text, LinearAttention, ModelConfig
from .gguf import GGUFSource
from .safetensors import SafetensorsSource


class Source(Protocol):
    config: ModelConfig

    def __contains__(self, name: str) -> bool:
        """Logical names: "X.weight" is present if X.weight or its quantised form (X.weight_packed) is."""
        ...

    def get(self, name: str) -> torch.Tensor:
        """Float values, CPU bf16: 2-D [out, in] (HF layout), 1-D [n]. Raises ValueError for a W4G128 tensor, whose
        values bf16 does not represent."""
        ...

    def quant(self, name: str) -> str | None:
        """The on-disk quantised format ("Q4_0", "Q8_0", ..., "W4G128"), or None for float tensors."""
        ...

    def raw(self, name: str) -> dict[str, torch.Tensor]:
        """The on-disk form of a quantised 2-D tensor, CPU, rows in HF order. GGUF: {"blocks": uint8 [out, row_bytes]};
        W4G128: {"packed": int32 [out, in / 8], "scale": bf16 [out, in / 128]}."""
        ...


def open(path: str | Path) -> Source:
    """*.gguf -> GGUF file; a directory with config.json -> HF safetensors checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path}: no such file or directory")
    if path.is_file() and path.suffix == ".gguf":
        return GGUFSource(path)
    if path.is_dir() and (path / "config.json").is_file():
        return SafetensorsSource(path)
    raise ValueError(f"{path}: neither a .gguf file nor a directory with config.json")


__all__ = ["GGUFSource", "Gemma4Text", "LinearAttention", "ModelConfig", "SafetensorsSource", "Source", "open"]
