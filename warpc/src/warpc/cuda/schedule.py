"""Launch-configuration rules and kernel ABI facts, shared by the capability table, the launchers and the runtime. The
constants mirror #defines of kernels/dense_common.cuh and kernels/dense_attn_common.cuh (a test compares them)."""

from __future__ import annotations

from ..kvcache import KVFormat

STEP_ROWS = 8  # rows one launch handles at most: the decode row or a prompt chunk
GEMV_WARPS = 4  # warps per GEMV CTA (128 threads)
ATTN_CHUNK = 64  # keys per attention chunk (CH)
ATTN_MAX_SPLITS = 128  # split blocks per head of the single-chunk bf16 kernels (MAX_SPLITS)
ATTN_MAX_SPLITS_CHUNKED = 1024  # of the chunked bf16 and every quantised kernel (MAX_SPLITS2)
ATTN_HEAD_DIMS = (64, 128, 256)  # head dims the attention kernels are instantiated for
KV_META_SECTIONS = ("kc", "vc", "ks", "vs", "kr", "vr", "kx", "vx")


def gemv_rows_per_warp(N: int, K: int, mode: int) -> int:
    """Rows per warp of k_gemv / k_q4_gemv. In-flight 16-byte loads per lane = rows per warp x K iterations: the gate|up
    pair (mode 1) already streams two rows, a small K needs more rows to hide the DRAM latency, a wide N has the CTAs to
    spare."""
    if mode == 1:
        return 1
    return 4 if N >= 16384 or K <= 1024 else 2


def gemv_ctas(N: int, K: int, mode: int) -> int:
    return -(-N // (GEMV_WARPS * gemv_rows_per_warp(N, K, mode)))


def attn_chunks(ctx: int) -> int:
    """64-key chunks per split block: 1 (the single-chunk kernels) while the context fits ATTN_MAX_SPLITS blocks per
    head, else the smallest power of two that does."""
    nc = 1
    while attn_splits(ctx, nc) > ATTN_MAX_SPLITS:
        nc *= 2
    return nc


def attn_splits(ctx: int, nc: int) -> int:
    return -(-ctx // (ATTN_CHUNK * nc))


def attn_split_cap(cache: str, nc: int) -> int:
    """Split blocks per head the kernel that runs (cache, nc) accepts."""
    return ATTN_MAX_SPLITS if cache == "bf16" and nc == 1 else ATTN_MAX_SPLITS_CHUNKED


def kv_kernel_reason(fmt: KVFormat, hd: int) -> str | None:
    """Why the attention kernels cannot run KV format `fmt` at head dim `hd`, or None."""
    if fmt.quantized and hd not in ATTN_HEAD_DIMS:
        return f"KV format {fmt.name} has no kernel for head dim {hd} (only {ATTN_HEAD_DIMS})"
    return None


def kv_meta(fmt: KVFormat, ctx: int, nkv: int, hd: int, rows_max: int) -> tuple[int, ...]:
    """The 12 int64s the quantised attention kernels take: section byte offsets (0 if absent), sink, window, ring
    slots, V bits."""
    sections = fmt.layout(ctx, nkv, hd, rows_max)
    return (*(sections.get(n, 0) for n in KV_META_SECTIONS), fmt.sink, fmt.window, fmt.ring(rows_max), fmt.v_bits)
