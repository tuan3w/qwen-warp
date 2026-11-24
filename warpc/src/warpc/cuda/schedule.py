"""Launch-configuration rules and kernel ABI facts, shared by the capability table, the launchers and the runtime. The
constants mirror #defines of kernels/dense_common.cuh and kernels/dense_attn_common.cuh (a test compares them)."""

from __future__ import annotations

from ..ir import BLOCK_FORMATS
from ..kvcache import Fp8Format, KVFormat, RotatedInt4Format

STEP_ROWS = 8  # rows one launch handles at most: the decode row or a prompt chunk
GEMV_WARPS = 4  # warps per GEMV CTA (128 threads)
ATTN_CHUNK = 64  # keys per attention chunk (CH)
ATTN_ONE_ROW_CHUNK = 32  # keys per block of the one-row decode of head dims 256 / 512 (DK)
ATTN_MAX_SPLITS = 128  # split blocks per head of the single-chunk bf16 kernels (MAX_SPLITS)
ATTN_MAX_SPLITS_CHUNKED = 1024  # of the chunked bf16 and every quantised kernel (MAX_SPLITS2)
ATTN_HEAD_DIMS = (64, 128, 256, 512)  # head dims the bf16 attention kernels are instantiated for
ATTN_QUANT_HEAD_DIMS = (64, 128, 256)  # the quantised-KV kernels': no head dim 512
PACKED_ROW_ALIGN = 16  # bytes: repacked Q4_0 / Q4_1 / Q6_K rows start 16-byte aligned
KV_META_SECTIONS = ("kc", "vc", "ks", "vs", "kr", "vr", "kx", "vx")


def packed_row_bytes(scheme: str, K: int) -> int:
    """Bytes of one row of a repacked block-format matrix (q4_0, q4_1, q6_k) of K columns: the planes of
    kernels/dense_q4_gemv.cuh / dense_q6_gemv.cuh, rounded up to PACKED_ROW_ALIGN."""
    elems, nbytes = BLOCK_FORMATS[scheme]
    return -(-(K // elems * nbytes) // PACKED_ROW_ALIGN) * PACKED_ROW_ALIGN


def gemv_rows_per_warp(N: int, K: int, mode: int) -> int:
    """Rows per warp of k_gemv. In-flight 16-byte loads per lane = rows per warp x K iterations: the gate|up
    pair (mode 1) already streams two rows, a small K needs more rows to hide the DRAM latency, a wide N has the CTAs to
    spare."""
    if mode == 1:
        return 1
    return 4 if N >= 16384 or K <= 1024 else 2


def gemv_ctas(N: int, K: int, mode: int) -> int:
    return -(-N // (GEMV_WARPS * gemv_rows_per_warp(N, K, mode)))


def q4_rows_per_warp(N: int, K: int, mode: int) -> int:
    """Rows per warp of k_q4_gemv (1 or 2). Measured on an RTX 3090 at T = 1 over the Llama-3.1-8B, Qwen3-4B and
    Qwen3-1.7B shapes: one row per warp is the fastest for the qkv and gate|up kernels (up to 846 GB/s against a 850
    GB/s read-only ceiling), two for the wide residual-add kernels (down 787 against 688 GB/s, o 691 against 644)."""
    return 2 if mode == 3 and N >= 4096 else 1


def q4_gemv_ctas(N: int, K: int, mode: int) -> int:
    return -(-N // (GEMV_WARPS * q4_rows_per_warp(N, K, mode)))


def q6_unroll(K: int) -> int:
    """Blocks of 256 columns a lane of k_q6_gemv loads before using them (1 or 2). Measured on an RTX 3090 over the
    heads of Llama-3.1-8B, Qwen3-4B, Qwen3-1.7B and Mistral-7B: two at K >= 4096 (at 2048 / 2560 the second block costs
    4 to 15%)."""
    return 2 if K >= 4096 else 1


def q6_gemv_ctas(N: int) -> int:
    return -(-N // GEMV_WARPS)


KS_SPLITS = (3, 4, 6, 8)  # K-split kernel's warps per row: K = 2048 x ks (kernels/dense_ks_gemv.cuh)


def gemv_k_split(K: int, mode: int, norm: bool) -> int:
    """Warps that share one row of the segmented GEMV (mode 0, no norm prologue) of a row too long for k_dg's loads in
    flight (K above 4096): 1 is k_gemv; K = 2048 x ks takes the K-split kernel, whose reduction order differs from
    k_gemv's (a row's bits depend on K and ks only). Gemma's down projections: K = 6144 -> 3, 12288 -> 6."""
    ks = K // 2048
    return ks if mode == 0 and not norm and K % 2048 == 0 and ks in KS_SPLITS else 1


def attn_chunks(ctx: int, window: int | None = None) -> int:
    """64-key chunks per split block: 1 (the single-chunk kernels) while the context fits ATTN_MAX_SPLITS blocks per
    head, else the smallest power of two that does."""
    nc = 1
    while attn_splits(ctx, nc, window) > ATTN_MAX_SPLITS:
        nc *= 2
    return nc


def attn_splits(ctx: int, nc: int, window: int | None = None, chunk: int = ATTN_CHUNK) -> int:
    """Split blocks per head: the state's 64-key chunks in groups of nc; with a window, the chunks the window's keys
    touch in position space, (window - 1 + 63) // 64 + 1 wherever it starts (a ring holds fewer rows than the positions
    its window spans; the kernel's attn_nsplit)."""
    if window is None:
        return -(-ctx // (chunk * nc))
    return -(-((window - 1 + chunk - 1) // chunk + 1) // nc)


def kv_rows(ctx: int) -> int:
    """Rows of a bf16 KV state in the attention kernels' layout: rounded up to whole chunks, so that a 64-key chunk of
    a ring (slot = position mod rows) never wraps. A positional state keeps every position in its own slot."""
    return -(-ctx // ATTN_CHUNK) * ATTN_CHUNK


def attn_split_cap(cache: str, nc: int) -> int:
    """Split blocks per head the kernel that runs (cache, nc) accepts."""
    return ATTN_MAX_SPLITS if cache == "bf16" and nc == 1 else ATTN_MAX_SPLITS_CHUNKED


def kv_kernel_reason(fmt: KVFormat, hd: int, window: int | None = None) -> str | None:
    """Why the attention kernels cannot run KV format `fmt` at head dim `hd` (with a sliding window, which is also
    what makes a state a ring), or None."""
    if isinstance(fmt, Fp8Format):
        return f"KV format {fmt.name} has no kernel here"
    if fmt.quantized and hd not in ATTN_QUANT_HEAD_DIMS:
        return f"KV format {fmt.name} has no kernel for head dim {hd} (only {ATTN_QUANT_HEAD_DIMS})"
    if fmt.quantized and window is not None:
        return f"KV format {fmt.name} has no sliding-window / ring kernel"
    return None


def kv_meta(fmt: KVFormat, ctx: int, nkv: int, hd: int, rows_max: int) -> tuple[int, ...]:
    """The 13 int64s the quantised attention kernels take: section byte offsets (0 if absent), sink, window, ring
    slots, V bits, 1 for int4_rot."""
    sections = fmt.layout(ctx, nkv, hd, rows_max)
    r4 = int(isinstance(fmt, RotatedInt4Format))
    return (*(sections.get(n, 0) for n in KV_META_SECTIONS), fmt.sink, fmt.window, fmt.ring(rows_max), fmt.v_bits, r4)
