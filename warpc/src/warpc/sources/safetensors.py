"""HF checkpoint directory: config.json plus *.safetensors files (sharded through model.safetensors.index.json).

Float tensors are read as bf16 (`get` refuses values bf16 does not represent; `get(name, torch.float32)` returns the
stored values exactly). A compressed-tensors W4G128 linear `X` stores X.weight_packed / X.weight_scale (and, in
checkpoints written by compressed-tensors, X.weight_shape) instead of X.weight; it is addressed by its logical name
"X.weight" (`quant` is "W4G128", `raw` gives the on-disk codes and scales). A rank-3 packed tensor [E, N, K / 8] is a
stack of E matrices; it needs the config's `quantization_config.warpc_layout == "stacked-experts"`, since
compressed-tensors does not define stacked tensors.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from ..errors import UnsupportedModel
from .config import ModelConfig

_PACKED = ("weight_packed", "weight_scale")
_REFUSED = ("weight_zero_point", "weight_g_idx")
_STACKED = "stacked-experts"


class SafetensorsSource:
    def __init__(self, directory: str | Path):
        self.path = Path(directory)
        hf = json.loads((self.path / "config.json").read_text())
        self.config = ModelConfig.from_hf(hf)
        self._stacked = (hf.get("quantization_config") or {}).get("warpc_layout") == _STACKED
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
        refused = sorted(n for n in self._file_of if n.rsplit(".", 1)[-1] in _REFUSED)
        if refused:
            raise UnsupportedModel(
                f"quantised tensors with a zero point or group permutation: {', '.join(refused[:4])}"
            )
        # logical "X.weight" -> "X." of its packed form
        self._packed = {
            n[: -len("weight_packed")] + "weight": n[: -len("weight_packed")]
            for n in self._file_of
            if n.endswith(".weight_packed")
        }
        for name, base in self._packed.items():
            lacking = [base + p for p in _PACKED if base + p not in self._file_of]
            if lacking or name in self._file_of:
                raise UnsupportedModel(f"{name}: an incomplete or ambiguous W4G128 tensor ({', '.join(lacking)})")
        self._handles: dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._file_of or name in self._packed

    def _tensor(self, name: str) -> torch.Tensor:
        file = self._file_of[name]
        h = self._handles.get(file)
        if h is None:
            h = self._handles[file] = safe_open(str(self.path / file), framework="pt", device="cpu")
        return h.get_tensor(name)

    def get(self, name: str, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if name in self._packed:
            raise ValueError(f"{name}: a W4G128 tensor is not a float tensor; read it with raw()")
        t = self._tensor(name)
        if not t.is_floating_point():
            raise ValueError(f"{name}: a {t.dtype} tensor is not a float weight")
        if dtype == torch.float32:
            return t.to(torch.float32)
        if dtype != torch.bfloat16:
            raise ValueError(f"{name}: get() returns bf16 or float32, not {dtype}")
        b = t.to(torch.bfloat16)
        if t.dtype != torch.bfloat16 and not torch.equal(b.to(t.dtype), t):
            raise UnsupportedModel(f"{name}: {t.dtype} values that bf16 does not represent (would be rounded)")
        return b

    def quant(self, name: str) -> str | None:
        if name in self._packed:
            return "W4G128"
        if name not in self._file_of:
            raise KeyError(name)
        return None

    def raw(self, name: str) -> dict[str, torch.Tensor]:
        """W4G128: {"packed": int32 [..., N, K / 8], "scale": bf16 [..., N, K / 128]}."""
        if name not in self._packed:
            raise ValueError(f"{name}: a float tensor has no on-disk codes")
        base = self._packed[name]
        packed, scale = self._tensor(base + "weight_packed"), self._tensor(base + "weight_scale")
        if packed.dim() == 3 and not self._stacked:
            raise UnsupportedModel(
                f"{name}: a stacked rank-3 tensor without quantization_config.warpc_layout={_STACKED!r}"
            )
        if packed.dim() not in (2, 3):
            raise UnsupportedModel(f"{name}: packed codes of rank {packed.dim()}")
        *lead, n, words = packed.shape
        k = 8 * words
        if base + "weight_shape" in self._file_of:
            shape = self._tensor(base + "weight_shape").tolist()
            if [n, k] != shape:
                raise UnsupportedModel(f"{name}: packed codes {tuple(packed.shape)} for a {shape} matrix")
        if packed.dtype != torch.int32 or k % 128:
            raise UnsupportedModel(f"{name}: packed codes {packed.dtype} {tuple(packed.shape)}")
        if scale.dtype != torch.bfloat16 or tuple(scale.shape) != (*lead, n, k // 128):
            raise UnsupportedModel(f"{name}: scales {scale.dtype} {tuple(scale.shape)} for codes {tuple(packed.shape)}")
        return {"packed": packed, "scale": scale}
