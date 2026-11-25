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

"int4_rot" is the hand engine's rotated 4-bit cache (engine/src/kvrot.cuh, written by attn.cu attn_prep_i4_kernel and
prefill.cu k_attn_prep, read by attn.cu attn_split_i4_kernel), byte for byte at head dim 256 and generalised to 64 and
128 with the same 64-element groups. Per (token, KV head), separately for K and V, the lane layout of the engine's warp
writer is d = lane + 32 i (lane 0..31, i < hd / 32):

  rotate y = H x * fp32(1 / sqrt(hd)) in fp32, butterfly strides 32, 64, .., hd / 2 (within a lane) then 1, 2, 4, 8, 16
         (across lanes; kvrot::wht256). Engine: hd 256, the factor 1/16 is exact.
  group  G(d) = 2 (d >> 7) + ((d & 15) >> 3) for hd >= 128 (4 groups of 64 at hd 256, 2 at hd 128), G = 0 at hd 64.
  grid   L(n) = u fmaf(B, u^2, 1), u = 2n - 15, n = 0..15, B = 0.0030002594 (fp16 of 0.003); code n = the number of the
         15 fp32 midpoints 0.5 (L(k) + L(k+1)) below y * (1 / s) (kvrot::quant_code).
  scale  8-bit code e per group, s(e) = ex2.approx((e - 176) * fp32(1/12)) (kvrot::scale_of; _SCALE). e0 = rint(12
         log2f(max(amax, 1e-30) / L(15)) + 176) clamped to 0..255 (_E0_STEPS: the amax where e0 steps up); candidates
         e0 + (1, 0, -1, -2, -3, -4, -6, -8), clamped, scored by the group's sum of fmaf(r, r, .) with r = fmaf(-s,
         L(n), y), summed per lane over its values in i order, then across the group's lanes in xor order 1, 2, 4,
         (8 at hd 64,) 16; the first strict minimum wins (kvrot::quant_store).
  bytes  codes: hd / 2 bytes per row, byte d / 2 = n(d) | n(d + 1) << 4 (even d low); meta: hd / 64 bytes, byte G = e_G
         (the engine's little-endian u32 at hd 256).
  read   value = s_read(e) * L(n), s_read(e) = ex2.approx(fmaf(e, fp32(1/12), 0.010779 - 176/12)) (_SCALE_READ: the
         reader's 1.0075 gain, kvrot::scale_dec); the split reader evaluates L(n) in fp16 (attn.cu i4_pair), the
         prefill gather (kvrot::dequant_row) in fp32. q is rotated the same way and rounded to fp16; the output is
         un-rotated once. Positions < 4 (the sink) and, for a step whose last row is at position p, positions > p - 496
         (the exact ring) are read from fp16 rotated rows instead of codes.

int4_rot layout, each section 16-byte aligned (no padding at the engine's 4 KV heads x hd 256): kc / vc codes
[Hkv][ctx][hd / 2], ks / vs scale codes [Hkv][ctx][hd / 64], kx / vx sink fp16 [Hkv][4][hd] (positions 0..3), kr / vr
ring fp16 [Hkv][512][hd] (slot = position % 512). The ring has 512 slots for a 496-row window so that a rejected
speculative row of an 8-row step never overwrites a row the next step reads (kvrot.cuh RING_W / RING_R).

"fp8" is the hand engine's e4m3 cache with the checkpoint's per-tensor k_scale / v_scale (`Fp8Format`); the CUDA
backend has no kernel for it.
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
    """Reference storage of a bf16-format KV state, in the interpreter's activation dtype. A state of R rows holds
    position p in slot p mod R, so a state with R >= every position is positional and a smaller one is a ring: a step
    writes its rows before it reads, so it must fit the ring together with the window it reads."""

    def __init__(self, shape: tuple[int, ...], *, device, dtype: torch.dtype):
        self.rows = torch.zeros(shape, dtype=dtype, device=device)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        slots = pos % self.rows.shape[1]
        if slots.unique().numel() != slots.numel():
            raise ValueError(f"{slots.numel()} rows do not fit a ring of {self.rows.shape[1]} rows")
        self.rows[0].index_copy_(0, slots, k)
        self.rows[1].index_copy_(0, slots, v)

    def attend(self, q: torch.Tensor, pos: torch.Tensor, scale: float, window: int | None = None) -> torch.Tensor:
        """Row t attends to the positions pos_t - window + 1 .. pos_t (0 .. pos_t without a window)."""
        ring = self.rows.shape[1]
        last = int(pos.max())
        first = 0 if window is None else max(0, int(pos.min()) - window + 1)
        if last - first + 1 > ring:
            raise ValueError(f"a ring of {ring} rows cannot hold positions {first}..{last} (window {window})")
        where = torch.arange(first, last + 1, device=q.device)
        slots = where % ring
        hq, hkv = q.shape[1], self.rows.shape[2]
        kk = self.rows[0, slots].repeat_interleave(hq // hkv, 1).float()
        vv = self.rows[1, slots].repeat_interleave(hq // hkv, 1).float()
        sc = torch.einsum("thd,lhd->htl", q.float(), kk) * scale
        dist = pos[:, None] - where[None, :]
        mask = dist < 0 if window is None else (dist < 0) | (dist >= window)
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


def _f32_table(text: str) -> torch.Tensor:
    return torch.tensor([int(w, 16) for w in text.split()], dtype=torch.int32).view(torch.float32)


# Measured on an RTX 3090 from the engine's own device code (kvrot.cuh), as fp32 bit patterns: they hold the results of
# the hardware log2f / ex2.approx, which no host library reproduces (torch's CPU exp2 differs from ex2.approx in 96
# of the 256 writer scales). _E0_STEPS[e - 1] is the smallest group amax whose first scale code e0 is >= e (e0 is
# monotone in amax, checked over every positive fp32).
_E0_STEPS = _f32_table(
    """
    3a825644 3a8a1651 3a924c59 3a9aff5f 3aa436d9 3aadfa99 3ab85306 3ac348e5 3acee59f 3adb331a 3ae83be8 3af60b17
    3b025644 3b0a1651 3b124c59 3b1aff5f 3b2436d9 3b2dfa99 3b385306 3b4348e5 3b4ee59f 3b5b331a 3b683be8 3b760b17
    3b825644 3b8a1651 3b924c59 3b9aff5f 3ba436d9 3badfa99 3bb85306 3bc348e5 3bcee59f 3bdb331a 3be83be8 3bf60b17
    3c025644 3c0a1651 3c124c59 3c1aff5f 3c2436d9 3c2dfa99 3c385306 3c4348e5 3c4ee59f 3c5b331a 3c683be8 3c760b17
    3c825644 3c8a1651 3c924c59 3c9aff5f 3ca436d9 3cadfa99 3cb85306 3cc348e5 3ccee59f 3cdb331a 3ce83be8 3cf60b17
    3d025644 3d0a1651 3d124c59 3d1aff5f 3d2436d9 3d2dfa99 3d385306 3d4348dd 3d4ee5a7 3d5b331a 3d683be8 3d760b17
    3d825644 3d8a164b 3d924c5f 3d9aff5f 3da436d9 3dadfa99 3db85306 3dc348dd 3dcee5a5 3ddb331d 3de83beb 3df60b14
    3e025642 3e0a164c 3e124c5e 3e1aff60 3e2436da 3e2dfa97 3e385304 3e4348df 3e4ee5a5 3e5b331d 3e683beb 3e760b14
    3e825642 3e8a164c 3e924c5e 3e9aff60 3ea436da 3eadfa97 3eb85304 3ec348df 3ecee5a5 3edb331d 3ee83beb 3ef60b14
    3f025642 3f0a164c 3f124c5e 3f1aff60 3f2436da 3f2dfa97 3f385304 3f4348df 3f4ee5a5 3f5b331d 3f683beb 3f760b14
    3f825642 3f8a164c 3f924c5e 3f9aff60 3fa436da 3fadfa97 3fb85304 3fc348df 3fcee5a7 3fdb331a 3fe83bf0 3ff60b0f
    40025644 400a164c 40124c5e 401aff5e 402436de 402dfa95 40385307 404348de 404ee5a7 405b331a 40683bf0 40760b0f
    40825644 408a164c 40924c5e 409aff5e 40a436de 40adfa95 40b85307 40c348de 40cee5a5 40db3319 40e83bee 40f60b11
    41025644 410a164c 41124c5e 411aff5e 412436dd 412dfa95 41385307 414348df 414ee5a5 415b3319 41683bee 41760b11
    41825644 418a164c 41924c5e 419aff5e 41a436dd 41adfa95 41b85307 41c348df 41cee5a5 41db3319 41e83bee 41f60b11
    42025644 420a164c 42124c5e 421aff5e 422436dd 422dfa95 42385307 424348de 424ee5a5 425b3319 42683bee 42760b11
    42825644 428a164c 42924c5e 429aff5e 42a436dd 42adfa95 42b85307 42c348df 42cee5a7 42db331a 42e83bf0 42f60b0f
    43025644 430a164c 43124c5e 431aff5e 432436de 432dfa95 43385307 434348de 434ee5a7 435b331a 43683bf0 43760b0f
    43825644 438a164c 43924c5e 439aff5e 43a436de 43adfa95 43b85307 43c348de 43cee5a5 43db3319 43e83bf0 43f60b0f
    44025644 440a164c 44124c5e 441aff5d 442436de 442dfa93 44385308 444348df 444ee5a5 445b3319 44683bf0 44760b0f
    44825644 448a164c 44924c5e 449aff5d 44a436de 44adfa93 44b85308 44c348df 44cee5a5 44db3319 44e83bf0 44f60b0f
    45025644 450a164c 45124c5e
    """
)
_SCALE = _f32_table(
    """
    38214515 382adc03 383504f2 383fc883 384b2fef 385744fc 386411ec 3871a1b8 38800000 38879c7a 388facd2 389837f0
    38a14515 38aadc03 38b504f2 38bfc883 38cb2fef 38d744fc 38e411ec 38f1a1b8 39000000 39079c7a 390facd2 391837f0
    39214515 392adc03 393504f2 393fc883 394b2fef 395744fc 396411ec 3971a1b8 39800000 39879c7a 398facd2 399837f0
    39a14515 39aadc03 39b504f2 39bfc883 39cb2fef 39d744fc 39e411ec 39f1a1b8 3a000000 3a079c7a 3a0facd2 3a1837f0
    3a214515 3a2adc03 3a3504f2 3a3fc883 3a4b2fef 3a5744fc 3a6411ec 3a71a1b8 3a800000 3a879c7a 3a8facd2 3a9837f0
    3aa14515 3aaadc03 3ab504f2 3abfc883 3acb2fef 3ad744fc 3ae411ec 3af1a1b8 3b000000 3b079c7a 3b0facd2 3b1837f0
    3b214515 3b2adc03 3b3504f2 3b3fc883 3b4b2fef 3b5744fc 3b6411ec 3b71a1b8 3b800000 3b879c7a 3b8facd5 3b9837f0
    3ba14515 3baadc07 3bb504f2 3bbfc883 3bcb2ff3 3bd744fc 3be411ec 3bf1a1bd 3c000000 3c079c7a 3c0facd5 3c1837f0
    3c214515 3c2adc07 3c3504f2 3c3fc883 3c4b2ff3 3c5744fc 3c6411ec 3c71a1bd 3c800000 3c879c7a 3c8facd5 3c9837f0
    3ca14515 3caadc07 3cb504f2 3cbfc883 3ccb2ff3 3cd744fc 3ce411ec 3cf1a1bd 3d000000 3d079c7a 3d0facd5 3d1837f0
    3d214515 3d2adc07 3d3504f2 3d3fc883 3d4b2ff3 3d5744fc 3d6411ec 3d71a1bd 3d800000 3d879c7c 3d8facd5 3d9837f0
    3da14517 3daadc07 3db504f2 3dbfc885 3dcb2ff3 3dd744fc 3de411ef 3df1a1bd 3e000000 3e079c7c 3e0facd5 3e1837f0
    3e214517 3e2adc07 3e3504f2 3e3fc885 3e4b2ff3 3e5744fc 3e6411ef 3e71a1bd 3e800000 3e879c7c 3e8facd6 3e9837f0
    3ea14517 3eaadc07 3eb504f2 3ebfc885 3ecb2ff4 3ed744fc 3ee411ef 3ef1a1be 3f000000 3f079c7d 3f0facd6 3f1837f0
    3f214518 3f2adc07 3f3504f2 3f3fc886 3f4b2ff5 3f5744fc 3f6411f0 3f71a1bf 3f800000 3f879c7d 3f8facd6 3f9837f1
    3fa14518 3faadc08 3fb504f3 3fbfc887 3fcb2ff5 3fd744fd 3fe411f1 3ff1a1bf 40000000 40079c7d 400facd7 401837f1
    40214518 402adc09 403504f3 403fc887 404b2ff6 405744fd 406411f1 4071a1c1 40800000 40879c7e 408facd7 409837f1
    40a14519 40aadc09 40b504f3 40bfc888 40cb2ff6 40d744fd 40e411f3 40f1a1c1 41000000 41079c7e 410facd7 411837f1
    41214519 412adc09 413504f3 413fc888 414b2ff6 415744fd 416411f3 4171a1c1 41800000 41879c7e 418facd9 419837f1
    41a14519 41aadc0b 41b504f3 41bfc888 41cb2ff9 41d744fd 41e411f3 41f1a1c3 42000000 42079c7e 420facd9 421837f1
    42214519 422adc0b 423504f3 423fc888 424b2ff9 425744fd 426411f3 4271a1c3 42800000 42879c7e 428facd9 429837f1
    42a14519 42aadc0b 42b504f3 42bfc888
    """
)
_SCALE_READ = _f32_table(
    """
    38227ab5 382c240c 3836607e 384138b9 384cb609 3858e249 3865c7cc 387371a1 3880f5c0 3888a0d7 3890c0a9 38995c2f
    38a27ab5 38ac240c 38b6607e 38c138b9 38ccb609 38d8e249 38e5c7cc 38f371a1 3900f5c0 3908a0d7 3910c0a9 39195c2f
    39227ab5 392c240c 3936607e 394138b9 394cb609 3958e249 3965c7cc 397371a1 3980f5c0 3988a0d7 3990c0a9 39995c2f
    39a27ab5 39ac240c 39b6607e 39c138b9 39ccb609 39d8e249 39e5c7cc 39f371a1 3a00f5c0 3a08a0d7 3a10c0a9 3a195c2f
    3a227ab5 3a2c240c 3a36607e 3a4138b9 3a4cb609 3a58e249 3a65c7cc 3a7371a1 3a80f5c0 3a88a0d7 3a90c0a9 3a995c2f
    3aa27ab5 3aac240c 3ab6607e 3ac138b9 3accb612 3ad8e249 3ae5c7cc 3af371ab 3b00f5c0 3b08a0d7 3b10c0b0 3b195c2f
    3b227ab5 3b2c2414 3b36607e 3b4138b9 3b4cb612 3b58e249 3b65c7cc 3b7371ab 3b80f5c0 3b88a0d7 3b90c0ad 3b995c2f
    3ba27ab5 3bac2410 3bb6607e 3bc138b9 3bccb60d 3bd8e249 3be5c7cc 3bf371a6 3c00f5c0 3c08a0d7 3c10c0ad 3c195c2f
    3c227ab5 3c2c2410 3c36607e 3c4138bd 3c4cb60d 3c58e249 3c65c7d1 3c7371a6 3c80f5c0 3c88a0da 3c90c0ad 3c995c2f
    3ca27ab8 3cac2410 3cb6607e 3cc138bd 3cccb60d 3cd8e249 3ce5c7d1 3cf371a6 3d00f5c0 3d08a0da 3d10c0ad 3d195c2f
    3d227ab8 3d2c2410 3d36607e 3d4138bd 3d4cb60d 3d58e249 3d65c7d1 3d7371a6 3d80f5c0 3d88a0d8 3d90c0ae 3d995c2f
    3da27ab6 3dac2412 3db6607e 3dc138bb 3dccb610 3dd8e249 3de5c7cf 3df371a9 3e00f5c0 3e08a0d8 3e10c0ae 3e195c2f
    3e227ab8 3e2c2412 3e36607e 3e4138bd 3e4cb610 3e58e249 3e65c7d1 3e7371a9 3e80f5c0 3e88a0d9 3e90c0ae 3e995c30
    3ea27ab7 3eac2412 3eb6607f 3ec138bc 3eccb610 3ed8e24a 3ee5c7d0 3ef371a9 3f00f5c1 3f08a0da 3f10c0ae 3f195c30
    3f227ab8 3f2c2412 3f36607f 3f4138bd 3f4cb611 3f58e24a 3f65c7d1 3f7371aa 3f80f5c1 3f88a0da 3f90c0af 3f995c31
    3fa27ab8 3fac2413 3fb66080 3fc138be 3fccb611 3fd8e24b 3fe5c7d3 3ff371aa 4000f5c1 4008a0db 4010c0b0 40195c31
    40227ab9 402c2414 40366080 404138be 404cb612 4058e24b 4065c7d3 407371ab 4080f5c2 4088a0db 4090c0af 40995c32
    40a27ab9 40ac2413 40b66081 40c138be 40ccb613 40d8e24c 40e5c7d3 40f371ad 4100f5c2 4108a0db 4110c0b1 41195c32
    41227ab9 412c2415 41366081 414138be 414cb613 4158e24c 4165c7d3 417371ad 4180f5c1 4188a0db 4190c0b1 41995c33
    41a27ab9 41ac2415 41b66083 41c138be 41ccb613 41d8e24f 41e5c7d3 41f371ad 4200f5c3 4208a0db 4210c0b1 42195c33
    42227ab9 422c2415 42366083 424138be 424cb613 4258e24f 4265c7d3 427371ad 4280f5c3 4288a0db 4290c0b1 42995c33
    42a27ab9 42ac2415 42b66083 42c138be
    """
)

_ROT_SINK = 4  # kvrot::NSINK
_ROT_WINDOW = 496  # kvrot::RING_W
_ROT_RING = 512  # kvrot::RING_R
_ROT_CANDIDATES = (1, 0, -1, -2, -3, -4, -6, -8)  # kvrot::c_cand_off


def _fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fmaf(a, b, c) of fp32 tensors, correctly rounded: the f64 product is exact, and the f64 sum's rounding error
    (TwoSum) settles the one case the f64 -> f32 rounding can get wrong, a sum that rounded onto an f32 tie."""
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bb = s - p
    err = (p - (s - bb)) + (cd - bb)
    r = s.float()
    up = torch.nextafter(r, torch.full_like(r, float("inf")))
    down = torch.nextafter(r, torch.full_like(r, float("-inf")))
    r = torch.where((err > 0) & (s == (r.double() + up.double()) / 2), up, r)
    return torch.where((err < 0) & (s == (r.double() + down.double()) / 2), down, r)


def _rot_levels() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(L(n) fp32 as the writer and kvrot::dequant_row compute it, the 15 decision midpoints, L(n) fp16 as the split
    reader's i4_pair computes it), n = 0..15."""
    u = torch.arange(-15, 16, 2, dtype=torch.float32)
    b32 = torch.tensor(0.0030002594, dtype=torch.float32)
    levels = u * _fma32(b32.expand(16), u * u, torch.ones(16))
    mids = (levels[:-1] + levels[1:]) * 0.5
    # fp16: u * u is exact, fma.rn.f16(uu, B, 1) and the product round once each (exact in fp32 before the rounding)
    b16 = b32.half().float()
    f = (u * u * b16 + 1).half().float()
    return levels, mids, (u * f).half().float()


_ROT_LEVELS, _ROT_MIDS, _ROT_LEVELS16 = _rot_levels()


def _rot_check(hd: int) -> None:
    if hd < 64 or hd & (hd - 1):
        raise ValueError(f"int4_rot needs a power-of-two head dim >= 64, got {hd}")


def rotate(x: torch.Tensor) -> torch.Tensor:
    """int4_rot's orthonormal Walsh-Hadamard transform over the last dim (its own inverse up to rounding), fp32, in the
    engine's butterfly order."""
    hd = x.shape[-1]
    _rot_check(hd)
    lead = x.shape[:-1]
    y = x.float()
    for h in [32 << k for k in range((hd // 64).bit_length())] + [1, 2, 4, 8, 16]:
        y = y.reshape(*lead, hd // (2 * h), 2, h)
        y = torch.stack((y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]), -2)
    return y.reshape(*lead, hd) * torch.tensor(hd**-0.5, dtype=torch.float32)


def _rot_groups(y: torch.Tensor) -> torch.Tensor:
    """[..., hd] -> [..., hd / 64 groups, values per lane (in i order), lanes (in index order)]: the order the writer
    sums a group's squared errors in."""
    hd = y.shape[-1]
    lead = y.shape[:-1]
    if hd == 64:
        return y.reshape(*lead, 1, 2, 32)
    # [chunk c = i >> 2][i & 3][lane bit 4][lane bit 3 = h][lane bits 0..2] -> [group 2 c + h][i & 3][lanes]
    g = y.reshape(*lead, hd // 128, 4, 2, 2, 8).movedim(-2, -4)
    return g.reshape(*lead, hd // 64, 4, 16)


def _rot_ungroup(g: torch.Tensor, hd: int) -> torch.Tensor:
    lead = g.shape[:-3]
    if hd == 64:
        return g.reshape(*lead, 64)
    return g.reshape(*lead, hd // 128, 2, 4, 2, 8).movedim(-4, -2).reshape(*lead, hd)


def _tree_sum(v: torch.Tensor) -> torch.Tensor:
    """Sum over the last dim (lanes in index order) as the xor-shuffle butterfly adds: pairs, then pairs of pairs."""
    while v.shape[-1] > 1:
        v = v[..., 0::2] + v[..., 1::2]
    return v[..., 0]


def _rot_codes(g: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    return ((g * inv)[..., None] > _ROT_MIDS.to(g.device)).sum(-1)


def int4_rot_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rows x [..., hd] (fp32 values) -> (codes uint8 [..., hd / 2], scale codes uint8 [..., hd / 64], the rotated row
    as fp16 [..., hd]: what the sink and ring hold), the bytes the engine's writers store."""
    dev = x.device
    y = rotate(x)
    g = _rot_groups(y)  # [..., G, q, lanes]
    amax = g.abs().amax((-2, -1))
    e0 = torch.searchsorted(_E0_STEPS.to(dev), amax.contiguous(), right=True)
    cand = (e0[..., None] + torch.tensor(_ROT_CANDIDATES, device=dev)).clamp(0, 255)  # [..., G, 8]
    s = _SCALE.to(dev)[cand][..., None, None]  # [..., G, 8, 1, 1]
    inv = 1 / s
    gc = g[..., None, :, :]
    levels = _ROT_LEVELS.to(dev)
    r = _fma32(-s, levels[_rot_codes(gc, inv)], gc)  # y - s * L(n): nvcc contracts it to one fmaf
    err = torch.zeros_like(r[..., 0, :])
    for q in range(r.shape[-2]):
        err = _fma32(r[..., q, :], r[..., q, :], err)
    best = _tree_sum(err).argmin(-1, keepdim=True)  # first minimum: the writer keeps a candidate only if strictly lower
    e = cand.gather(-1, best)[..., 0]  # [..., G]
    n = _rot_ungroup(_rot_codes(g, 1 / _SCALE.to(dev)[e][..., None, None]), x.shape[-1]).to(torch.uint8)
    return n[..., 0::2] | (n[..., 1::2] << 4), e.to(torch.uint8), y.half()


def _rot_unpack(codes: torch.Tensor, meta: torch.Tensor, table: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    n = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    hd = n.shape[-1]
    s = table.to(codes.device)[meta.long()]  # [..., G]
    s = _rot_ungroup(s[..., None, None].expand(*s.shape, *((2, 32) if hd == 64 else (4, 16))), hd)
    return s * levels.to(codes.device)[n]


def int4_rot_dequantize(codes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """The rotated rows the engine's prefill gather decodes (kvrot::dequant_row): s_read(e) * L(n) in fp32."""
    return _rot_unpack(codes, meta, _SCALE_READ, _ROT_LEVELS)


class RotatedInt4Cache:
    """Reference storage of an int4_rot KV state: every row as the split reader decodes its codes, and as the fp16
    rotated row the sink and ring hold (a 512-slot ring always still holds the 496 rows a step reads)."""

    def __init__(self, shape: tuple[int, ...], *, device):
        _, ctx, hkv, hd = shape
        _rot_check(hd)
        self.read = torch.zeros(2, ctx, hkv, hd, dtype=torch.float32, device=device)
        self.exact = torch.zeros(2, ctx, hkv, hd, dtype=torch.float16, device=device)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        for i, x in enumerate((k, v)):
            codes, meta, exact = int4_rot_quantize(x.float())
            self.read[i].index_copy_(0, pos, _rot_unpack(codes, meta, _SCALE_READ, _ROT_LEVELS16))
            self.exact[i].index_copy_(0, pos, exact)

    def attend(self, q: torch.Tensor, pos: torch.Tensor, scale: float) -> torch.Tensor:
        """The rows of one step: the exact ring covers the 496 positions before the step's end, for every row."""
        length = int(pos.max()) + 1
        T, hq, d = q.shape
        hkv = self.read.shape[2]
        lo = min(_ROT_SINK, length)
        cut = max(length - _ROT_WINDOW, lo)
        k, v = (
            torch.cat((self.exact[i, :lo].float(), self.read[i, lo:cut], self.exact[i, cut:length].float()))
            for i in range(2)
        )
        qq = rotate(q).half().float().reshape(T, hkv, hq // hkv, d)
        sc = torch.einsum("tkgd,lkd->kgtl", qq, k).reshape(hq, T, length) * scale
        sc = sc.masked_fill((torch.arange(length, device=q.device)[None, :] > pos[:, None])[None], float("-inf"))
        pr = torch.softmax(sc, -1).reshape(hkv, hq // hkv, T, length)
        out = torch.einsum("kgtl,lkd->tkgd", pr, v)
        return rotate(out).to(q.dtype).reshape(T, hq * d)


@dataclass(frozen=True)
class RotatedInt4Format(KVFormat):
    """int4_rot (module docstring): `window` is per step, not per row, and the ring has a fixed 512 slots."""

    def ring(self, rows_max: int) -> int:
        return _ROT_RING

    def layout(self, ctx: int, nkv: int, hd: int, rows_max: int) -> dict[str, int]:
        _rot_check(hd)
        offset, sections = 0, {}
        for name, n in (
            ("kc", nkv * ctx * hd // 2),
            ("vc", nkv * ctx * hd // 2),
            ("ks", nkv * ctx * hd // 64),
            ("vs", nkv * ctx * hd // 64),
            ("kx", nkv * _ROT_SINK * hd * 2),
            ("vx", nkv * _ROT_SINK * hd * 2),
            ("kr", nkv * _ROT_RING * hd * 2),
            ("vr", nkv * _ROT_RING * hd * 2),
        ):
            sections[name] = offset
            offset += (n + 15) // 16 * 16
        sections["total"] = offset
        return sections

    def reference_cache(self, shape: tuple[int, ...], *, device, dtype: torch.dtype) -> RotatedInt4Cache:
        return RotatedInt4Cache(shape, device=device)


class Fp8Cache:
    """Reference storage of an fp8 KV state: the rows as the reader decodes them (e4m3 value * scale), fp32."""

    def __init__(self, shape: tuple[int, ...], *, device):
        self.read = torch.zeros(shape, dtype=torch.float32, device=device)

    def write(self, pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, k_scale, v_scale) -> None:
        for i, (x, s) in enumerate(((k, k_scale), (v, v_scale))):
            s = s.float().reshape(())
            q = (x.float() * (1.0 / s)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            self.read[i].index_copy_(0, pos, q.float() * s)

    def attend(self, q: torch.Tensor, pos: torch.Tensor, scale: float, *, k_scale, v_scale) -> torch.Tensor:
        """q and the probabilities meet the rows as fp16 (the reader's mma operand type)."""
        length = int(pos.max()) + 1
        hq, hkv = q.shape[1], self.read.shape[2]
        kk = self.read[0, :length].repeat_interleave(hq // hkv, 1)
        vv = self.read[1, :length].repeat_interleave(hq // hkv, 1)
        sc = torch.einsum("thd,lhd->htl", q.half().float(), kk) * scale
        mask = torch.arange(length, device=q.device)[None, :] > pos[:, None]
        pr = torch.softmax(sc.masked_fill(mask[None], float("-inf")), -1).half().float()
        out = torch.einsum("htl,lhd->thd", pr, vv).to(q.dtype)
        return out.reshape(q.shape[0], hq * q.shape[2])


@dataclass(frozen=True)
class Fp8Format(KVFormat):
    """fp8: per (token, KV head) row, x * fp32(1 / scale) saturated to e4m3 (+-448), round to nearest even; read as
    e4m3 value * scale. scale: the checkpoint's per-layer, per-tensor k_scale / v_scale (the attention op's weights).
    Layout [2 (K, V)][Hkv][ctx][hd] bytes."""

    def layout(self, ctx: int, nkv: int, hd: int, rows_max: int) -> dict[str, int]:
        raise ValueError("KV format fp8 has no sectioned layout")

    def nbytes(self, ctx: int, nkv: int, hd: int, rows_max: int) -> int:
        return 2 * nkv * ctx * hd

    def reference_cache(self, shape: tuple[int, ...], *, device, dtype: torch.dtype) -> Fp8Cache:
        return Fp8Cache(shape, device=device)


FORMATS: dict[str, KVFormat] = {
    f.name: f
    for f in (
        KVFormat("bf16"),
        KVFormat("k8v8", 8, 8),
        KVFormat("k8v8w", 8, 8, sink=4, window=256),
        KVFormat("k8v4w", 8, 4, sink=4, window=256),
        RotatedInt4Format("int4_rot", 4, 4, sink=_ROT_SINK, window=_ROT_WINDOW),
        Fp8Format("fp8", 8, 8),
    )
}
