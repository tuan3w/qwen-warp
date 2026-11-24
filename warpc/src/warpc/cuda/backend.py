"""The CUDA backend as the compiler sees it: a name, the capability table, the KV state sizes."""

from __future__ import annotations

from ..ir import StateDecl
from ..kvcache import FORMATS
from .table import TABLE


class CudaBackend:
    name = "cuda"
    caps = TABLE

    @staticmethod
    def kv_bytes(state: StateDecl, rows_max: int) -> int:
        """Bytes of one layer's KV buffer in the kernels' layout, for steps of up to `rows_max` rows (the quantised
        formats' exact-row ring holds the window plus one step)."""
        _, ctx, nkv, hd = state.shape
        return FORMATS[state.cache].nbytes(ctx, nkv, hd, rows_max)
