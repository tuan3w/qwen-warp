"""Pinned host memory (warpc.pinned): exact size, zero-filled, pinned, a kernel's copy round trip, and released when the
last reference (a view included) is dropped, by refcount and not through torch's host pool.

  uv run --package warpc python warpc/tests/test_pinned.py

Needs a CUDA device; the tests skip without one."""

import gc

import torch

from harness import Skip, run
from warpc import pinned


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


def pool() -> int:
    return torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)


def test_exact_size_pinned_zeroed():
    require_gpu()
    torch.zeros(1, device="cuda")
    base, pool0 = pinned.live_bytes(), pool()
    t = pinned.empty((3, 1000), torch.bfloat16)
    assert t.shape == (3, 1000) and t.dtype == torch.bfloat16 and t.is_pinned() and not t.any()
    assert pinned.live_bytes() - base == 8192  # 6,000 bytes: two pages, no power-of-two rounding
    big = pinned.empty(600 << 20, torch.uint8)  # torch's pool would hold 1 GiB
    assert pinned.live_bytes() - base == 8192 + (600 << 20)
    del t, big
    assert pinned.live_bytes() == base and pool() == pool0


def test_copy_round_trip_and_views_keep_the_block():
    require_gpu()
    base = pinned.live_bytes()
    src = torch.arange(1 << 16, dtype=torch.float32, device="cuda")
    host = pinned.like(src, non_blocking=True)
    torch.cuda.synchronize()
    assert host.is_pinned() and torch.equal(host.cuda(), src)
    view = host[100:200]
    del host
    gc.collect()
    assert pinned.live_bytes() > base and view[0] == 100  # a view keeps the whole block
    back = torch.zeros(100, device="cuda")
    back.copy_(view, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(back, src[100:200])
    del view
    assert pinned.live_bytes() == base


def test_empty_shapes():
    require_gpu()
    base = pinned.live_bytes()
    assert pinned.empty(0, torch.int32).numel() == 0 and pinned.empty((2, 0), torch.uint8).shape == (2, 0)
    assert pinned.empty(5, torch.int32).shape == (5,)
    assert pinned.live_bytes() == base


if __name__ == "__main__":
    run(globals())
