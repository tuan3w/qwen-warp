"""Checkpoint sources: tensors addressed by HF names plus a ModelConfig, whatever the file format."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from .config import ModelConfig
from .gguf import GGUFSource
from .safetensors import SafetensorsSource


class Source(Protocol):
    config: ModelConfig

    def __contains__(self, name: str) -> bool: ...

    def get(self, name: str) -> torch.Tensor:
        """CPU bf16: 2-D [out, in] (HF layout), 1-D [n]."""
        ...

    def quant(self, name: str) -> str | None:
        """The on-disk block format ("Q4_0", "Q8_0", ...), or None for float tensors."""
        ...

    def raw(self, name: str) -> np.ndarray:
        """The on-disk blocks of a block-quantised 2-D tensor as uint8 [out, row_bytes], rows in HF order."""
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


__all__ = ["GGUFSource", "ModelConfig", "SafetensorsSource", "Source", "open"]
