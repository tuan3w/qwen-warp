"""KV cache formats: what a KV state stores, its byte layout, and the reference cache (the numerics).

"bf16" stores the K and V rows as given, [2 (K, V)][Hkv][ctx][hd] bf16. The quantised formats store, per (token, KV
head) and separately for K and V:

  write  x = the row the bf16 cache would hold (K after qk-norm and rope), fp32; y = H x / sqrt(hd) (orthonormal
         Walsh-Hadamard, butterfly stages h = 1, 2, 4, ...); scale s = fp16(amax(y) / qmax) (qmax 127 for 8 bits, 7 for
         4 bits); code = clamp(round_half_even(y / s), -qmax, qmax) + offset (128 for 8 bits, 8 for 4 bits); value =
         (code - offset) * s.
  read   scores of the rotated query (rounded to the activation dtype) against the dequantised K in fp32; softmax in
         fp32 with the probabilities rounded to the activation dtype; PV in fp32 over the dequantised V, un-rotated once
         and rounded. Formats with `sink` / `window` read positions < sink and the newest `window` positions of each
         query row from exact rows (fp16 in the rotated basis) instead.

Byte layout (one uint8 buffer per layer, each section 16-byte aligned): kc [Hkv][ctx][hd * k_bits / 8], vc, ks fp16
[Hkv][ctx], vs, then kr / vr fp16 [Hkv][ring][hd] (slot = position % ring) if window, then kx / vx fp16 [Hkv][sink][hd]
if sink. A 4-bit row holds 8-element words: byte b (0..3) of word w has element 8w + b in its low nibble and 8w + 4 + b
in its high nibble.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _wht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Walsh-Hadamard transform over the last dim (a power of two), fp32, butterfly stages h = 1, 2, 4..."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    y = x.float()
    h = 1
    while h < n:
        y = y.reshape(*lead, n // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
        h *= 2
    return y.reshape(*lead, n) * (1.0 / n**0.5)


def _qmax(bits: int) -> int:
    return 127 if bits == 8 else 7


def _offset(bits: int) -> int:
    return 128 if bits == 8 else 8


def _quantize(bits: int, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y [..., hd] fp32 rotated rows -> (codes uint8 [..., hd] unpacked, scales fp32 [...] holding fp16 values)."""
    amax = y.abs().amax(-1, keepdim=True)
    # a tensor divisor: true division, as the kernel computes it, not a reciprocal multiply
    s = (amax / torch.full_like(amax, float(_qmax(bits)))).to(torch.float16).float()
    u = torch.where(s > 0, y / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(y))
    q = _qmax(bits)
    codes = torch.round(u).clamp(-q, q) + _offset(bits)
    return codes.to(torch.uint8), s[..., 0]


def _dequantize(bits: int, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (codes.float() - _offset(bits)) * scales.float()[..., None]


class ExactCache:
    """Reference storage of a bf16-format KV state, in the interpreter's activation dtype."""

    def __init__(self, shape: tuple[int, ...], *, device, dtype: torch.dtype):
        self.rows = torch.zeros(shape, dtype=dtype, device=device)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        self.rows[0].index_copy_(0, pos, k)
        self.rows[1].index_copy_(0, pos, v)

    def attend(self, q: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
        length = int(pos.max()) + 1
        hq, hkv = q.shape[1], self.rows.shape[2]
        kk = self.rows[0, :length].repeat_interleave(hq // hkv, 1).float()
        vv = self.rows[1, :length].repeat_interleave(hq // hkv, 1).float()
        sc = torch.einsum("thd,lhd->htl", q.float(), kk) * scale
        mask = torch.arange(length, device=q.device)[None, :] > pos[:, None]
        pr = torch.softmax(sc.masked_fill(mask[None], float("-inf")), -1).to(q.dtype).float()
        out = torch.einsum("htl,lhd->thd", pr, vv).to(q.dtype)
        return out.reshape(q.shape[0], hq * q.shape[2])


class QuantizedCache:
    """Reference storage of a quantised KV state: codes, scales and (sink / window formats) the exact rotated rows."""

    def __init__(self, fmt: KVFormat, shape: tuple[int, ...], *, device):
        _, ctx, hkv, hd = shape
        self.fmt = fmt
        self.codes = torch.zeros(2, ctx, hkv, hd, dtype=torch.uint8, device=device)
        self.scales = torch.zeros(2, ctx, hkv, dtype=torch.float32, device=device)
        self.exact = torch.zeros(2, ctx, hkv, hd, dtype=torch.float32, device=device) if fmt.exact_rows else None

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        for i, (bits, x) in enumerate(((self.fmt.k_bits, k), (self.fmt.v_bits, v))):
            y = _wht(x)
            codes, scales = _quantize(bits, y)
            self.codes[i].index_copy_(0, pos, codes)
            self.scales[i].index_copy_(0, pos, scales)
            if self.exact is not None:
                self.exact[i].index_copy_(0, pos, y.to(torch.float16).float())

    def attend(self, q: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
        fmt = self.fmt
        length = int(pos.max()) + 1
        hq, hkv, d = q.shape[1], self.codes.shape[2], q.shape[2]
        g = hq // hkv
        qq = _wht(q).to(q.dtype).float()
        kk = _dequantize(fmt.k_bits, self.codes[0, :length], self.scales[0, :length]).repeat_interleave(g, 1)
        vv = _dequantize(fmt.v_bits, self.codes[1, :length], self.scales[1, :length]).repeat_interleave(g, 1)
        sc = torch.einsum("thd,lhd->htl", qq, kk) * scale
        j = torch.arange(length, device=q.device)[None, :]
        use_exact = None
        if self.exact is not None:
            use_exact = (j < fmt.sink) | (pos[:, None] - j < fmt.window)
            ke = self.exact[0, :length].repeat_interleave(g, 1)
            sc = torch.where(use_exact[None], torch.einsum("thd,lhd->htl", qq, ke) * scale, sc)
        sc = sc.masked_fill((j > pos[:, None])[None], float("-inf"))
        pr = torch.softmax(sc, -1).to(q.dtype).float()
        if use_exact is None:
            out = torch.einsum("htl,lhd->thd", pr, vv)
        else:
            ve = self.exact[1, :length].repeat_interleave(g, 1)
            pe = pr * use_exact[None]
            out = torch.einsum("htl,lhd->thd", pr - pe, vv) + torch.einsum("htl,lhd->thd", pe, ve)
        return _wht(out).to(q.dtype).reshape(q.shape[0], hq * d)


@dataclass(frozen=True)
class KVFormat:
    """A KV cache format; bits 16 = bf16 rows as given, 8 / 4 = rotated symmetric codes with one fp16 scale per row."""

    name: str
    k_bits: int = 16
    v_bits: int = 16
    sink: int = 0
    window: int = 0

    @property
    def quantized(self) -> bool:
        return self.k_bits != 16

    @property
    def exact_rows(self) -> bool:
        return self.sink > 0 or self.window > 0

    def ring(self, rows_max: int) -> int:
        """Slots of the exact-row ring: the window plus the rows of one step of up to `rows_max` rows (a step never
        evicts a row it reads)."""
        return self.window + rows_max if self.window else 0

    def layout(self, ctx: int, nkv: int, hd: int, rows_max: int) -> dict[str, int]:
        """Byte offsets of the sections of a quantised format's per-layer buffer, plus "total"."""
        if not self.quantized:
            raise ValueError(f"KV format {self.name} has no sectioned layout")
        offset, sections = 0, {}

        def add(name: str, n: int) -> None:
            nonlocal offset
            sections[name] = offset
            offset += (n + 15) // 16 * 16

        add("kc", nkv * ctx * hd * self.k_bits // 8)
        add("vc", nkv * ctx * hd * self.v_bits // 8)
        add("ks", nkv * ctx * 2)
        add("vs", nkv * ctx * 2)
        if self.window:
            add("kr", nkv * self.ring(rows_max) * hd * 2)
            add("vr", nkv * self.ring(rows_max) * hd * 2)
        if self.sink:
            add("kx", nkv * self.sink * hd * 2)
            add("vx", nkv * self.sink * hd * 2)
        sections["total"] = offset
        return sections

    def nbytes(self, ctx: int, nkv: int, hd: int, rows_max: int) -> int:
        """Bytes of one layer's cache buffer for steps of up to `rows_max` rows."""
        if not self.quantized:
            return 2 * nkv * ctx * hd * 2
        return self.layout(ctx, nkv, hd, rows_max)["total"]

    def reference_cache(self, shape: tuple[int, ...], *, device, dtype: torch.dtype) -> ExactCache | QuantizedCache:
        """The interpreter's storage for a state of this format; dtype is the activation dtype (bf16 format only)."""
        if not self.quantized:
            return ExactCache(shape, device=device, dtype=dtype)
        return QuantizedCache(self, shape, device=device)


FORMATS: dict[str, KVFormat] = {
    f.name: f
    for f in (
        KVFormat("bf16"),
        KVFormat("k8v8", 8, 8),
        KVFormat("k8v8w", 8, 8, sink=4, window=256),
        KVFormat("k8v4w", 8, 4, sink=4, window=256),
    )
}
