"""HF checkpoint directory: config.json plus *.safetensors files (sharded through model.safetensors.index.json)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .config import ModelConfig


class SafetensorsSource:
    def __init__(self, directory: str | Path):
        self.path = Path(directory)
        self.config = ModelConfig.from_hf(json.loads((self.path / "config.json").read_text()))
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            self._file_of: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:
            self._file_of = {}
            for f in sorted(self.path.glob("*.safetensors")):
                with safe_open(str(f), framework="pt") as h:
                    self._file_of.update(dict.fromkeys(h.keys(), f.name))
        if not self._file_of:
            raise FileNotFoundError(f"{self.path}: no safetensors tensors")
        self._handles: dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._file_of

    def get(self, name: str) -> torch.Tensor:
        file = self._file_of[name]
        h = self._handles.get(file)
        if h is None:
            h = self._handles[file] = safe_open(str(self.path / file), framework="pt", device="cpu")
        return h.get_tensor(name).to(torch.bfloat16)

    def quant(self, name: str) -> str | None:
        if name not in self._file_of:
            raise KeyError(name)
        return None

    def raw(self, name: str) -> np.ndarray:
        raise ValueError(f"{name}: safetensors tensors are float, there are no on-disk blocks")
