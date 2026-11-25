"""Pinned host memory of the exact size asked for, returned to the OS the moment its last reference is dropped.

Why not `pin_memory=True`: torch's caching host allocator rounds every block up to a power of two (a 2.4 GiB embedding
table holds 4 GiB) and keeps a freed block for reuse, so the pool only grows: a 27B build with host tables holds 8 GiB
for two tables of 4.7 GiB, a dropped model leaves them behind, and a process that builds a second model, or fills a
`PrefixCache` store, runs into the kernel's out-of-memory killer, silently. Here a block is an anonymous mapping
registered with `cudaHostRegister` and unregistered, then unmapped, when the tensor that owns it (and every view of it)
is garbage: `live_bytes()` is what the process pins through warpc, and torch's own host pool is never touched.
Without a CUDA device the tensors are ordinary ones."""

from __future__ import annotations

import math
import mmap
import weakref

import numpy as np
import torch

_PAGE = mmap.PAGESIZE
_PORTABLE_MAPPED = 3  # cudaHostRegisterPortable | cudaHostRegisterMapped: a kernel reads the host pointer (UVA)
_live = 0


def live_bytes() -> int:
    """Bytes registered by `empty` / `like` and not yet released."""
    return _live


def _release(ptr: int, nbytes: int) -> None:
    global _live
    _live -= nbytes
    torch.cuda.cudart().cudaHostUnregister(ptr)  # the mapping is unmapped by the array's base after this callback


def empty(shape: tuple[int, ...] | int, dtype: torch.dtype) -> torch.Tensor:
    """An uninitialised pinned tensor (zero-filled pages: a fresh mapping)."""
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if not nbytes or not torch.cuda.is_available():
        return torch.empty(shape, dtype=dtype)
    global _live
    size = -(-nbytes // _PAGE) * _PAGE
    arr = np.frombuffer(mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS), dtype=np.uint8)
    t = torch.from_numpy(arr)
    err = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), size, _PORTABLE_MAPPED)
    if err != torch.cuda.cudart().cudaError.success:
        raise RuntimeError(f"cudaHostRegister of {size} bytes failed: {err}")
    _live += size
    weakref.finalize(arr, _release, t.data_ptr(), size).atexit = False
    return t[:nbytes].view(dtype).view(shape)


def like(t: torch.Tensor, *, non_blocking: bool = False) -> torch.Tensor:
    """A pinned copy of `t` (any device). non_blocking: the copy is enqueued on the current stream, the caller
    synchronises before it reads the result."""
    return empty(t.shape, t.dtype).copy_(t, non_blocking=non_blocking)
