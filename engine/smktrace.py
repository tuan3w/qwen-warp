"""Read-only activation / attention trace hooks for the fast prefill (sparsity research tooling, env-gated by SMK_TRACE).

  SMK_TRACE=1 python ...   -> FastPrefill.trace = Tracer() (otherwise None and every hook is a single `is not None` test)

The hooks copy tensors OUT of the prefill's scratch buffers; nothing the prefill reads or writes is modified (same contract as the DFlash tap capture, prefill.py `tap`).
A Tracer records, for the layers in `layers`, the rows whose absolute positions are in `want` (sorted list of absolute positions; the driver sets it per run):
  x(kind, tensor, row0)   kind in {"qkv" (normed hidden into qkv(z)), "o" (core output into o/out_proj), "gu" (normed hidden into gate|up), "down" (SiLU(gate)*up into down_proj)}
  attn(...)               full-attention layers in `attn_layers`: rope'd q of the wanted rows [n][24][256], and (once per segment, last wins) the KV history views.
Rows are bf16 on the GPU while a segment runs and are moved to host by `fetch()`.
"""
import os

import torch


class Tracer:
    KINDS = ("qkv", "o", "gu", "down")

    def __init__(self, layers=(), attn_layers=(), want=()):
        self.layers = set(layers)
        self.attn_layers = set(attn_layers)
        self.want = sorted(want)
        self.li = -1
        self.seg = None  # (pos0, T, local row index tensor on cuda, sorted wanted absolute positions in this segment)
        self.rows = {}  # (li, kind) -> list of [n][K] bf16 cuda tensors (one per segment / row block), in segment order
        self.q = {}  # li -> list of [n][24][256] bf16 cuda tensors
        self.kv = {}  # li -> dict(k=[Ltot][4][256] bf16, v=..., pos0, T) host copies (last segment)
        self.keep_kv = False

    @classmethod
    def from_env(cls):
        return cls()

    # ---- segment bookkeeping (called by FastPrefill._segment)
    def begin(self, pos0, T):
        import bisect

        a, b = bisect.bisect_left(self.want, pos0), bisect.bisect_left(self.want, pos0 + T)
        pos = self.want[a:b]
        idx = torch.tensor([p - pos0 for p in pos], dtype=torch.long, device="cuda")
        self.seg = (pos0, T, idx, pos)

    def layer(self, li):
        self.li = li

    # ---- hooks
    def x(self, kind, t, row0=0):
        """t: [n][K] bf16 cuda rows (segment rows row0 .. row0 + n)."""
        li = self.li
        if li not in self.layers or self.seg is None:
            return
        idx = self.seg[2]
        n = t.shape[0]
        sel = idx[(idx >= row0) & (idx < row0 + n)] - row0
        if sel.numel() == 0:
            return
        self.rows.setdefault((li, kind), []).append(t.index_select(0, sel).clone())

    def attn(self, qatt, kall, vall, Ltot, pos0, T):
        """qatt [T][24][256] bf16 (rope'd q), kall / vall [ctx_max][4][256] bf16 token-major (first Ltot rows valid; rows < pos0 = the dequantised cache, rows >= pos0 = this segment's exact rows)."""
        li = self.li
        if li not in self.attn_layers or self.seg is None:
            return
        idx = self.seg[2]
        if idx.numel() == 0:
            return
        self.q.setdefault(li, []).append(qatt.index_select(0, idx).clone())
        if self.keep_kv:
            self.kv[li] = dict(k=kall[:Ltot].cpu(), v=vall[:Ltot].cpu(), pos0=pos0, T=T)

    def fetch(self):
        """-> (rows {(li, kind): bf16 [R][K] cpu}, q {li: bf16 [R][24][256] cpu}); clears the buffers."""
        rows = {k: torch.cat(v).cpu() for k, v in self.rows.items()}
        q = {k: torch.cat(v).cpu() for k, v in self.q.items()}
        self.rows, self.q = {}, {}
        return rows, q


def enabled():
    return bool(os.environ.get("SMK_TRACE"))
