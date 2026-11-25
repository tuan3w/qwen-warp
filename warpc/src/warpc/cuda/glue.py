"""The kernel library: built on demand from kernels/ into the cache directory, loaded with the ctypes signatures the
launchers use.

The library is keyed by a digest of the kernel sources and the GPU architecture, so an edited kernel or another GPU gets
its own build; a finished build is published with an atomic rename, so concurrent processes never load a half-written
library.
"""

from __future__ import annotations

import ctypes as C
import functools
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import attn_gated, deltanet, head_refine, hybrid, inkling, w4

KERNELS = Path(__file__).resolve().parent / "kernels"

_PTR, _INT, _FLOAT, _SIZE = C.c_void_p, C.c_int, C.c_float, C.c_size_t
SIGNATURES: dict[str, tuple[list, type]] = {
    "warpc_embed": ([_PTR, _PTR, _PTR, _INT, _INT, _FLOAT, _PTR], _INT),
    "warpc_rmsnorm": ([_PTR, _PTR, _PTR, _PTR, _INT, _INT, _FLOAT, _PTR], _INT),
    "warpc_rmsnorm_post": ([_PTR, _PTR, _PTR, _PTR, _INT, _INT, _FLOAT, _PTR], _INT),
    "warpc_ple": ([_PTR, _PTR, _PTR, _PTR, _PTR, _INT, _INT, _INT, _FLOAT, _FLOAT, _FLOAT, _FLOAT, _PTR], _INT),
    "warpc_gemv": (
        [
            _PTR,
            _PTR,
            _PTR,
            _INT,
            _INT,
            _INT,
            _INT,
            _PTR,
            _PTR,
            _PTR,
            _FLOAT,
            _INT,
            _PTR,
            _INT,
            _FLOAT,
            _INT,
            _INT,
            _PTR,
        ],
        _INT,
    ),
    "warpc_q4_gemv": ([_PTR, _PTR, _PTR, _INT, _INT, _INT, _INT, _PTR, _PTR, _INT, _PTR], _INT),
    "warpc_attn_prep": ([_INT, *[_PTR] * 8, _INT, _INT, _INT, _INT, _PTR, _FLOAT, _INT, _INT, _INT, _PTR], _INT),
    "warpc_attn_decode": ([_INT, *[_PTR] * 5, _INT, _INT, _INT, _INT, _FLOAT, _INT, _PTR], _INT),
    "warpc_attn_splits": ([_INT, _INT, _INT, _INT], _INT),
    "warpc_attn_decode_fused": ([*[_PTR] * 9, _INT, _INT, _INT, _INT, _PTR, _FLOAT, _INT, _INT, _FLOAT, _PTR], _INT),
    "warpc_attn_decode1": (
        [*[_PTR] * 10, _INT, _INT, _INT, _INT, _PTR, _FLOAT, _INT, _INT, _INT, _FLOAT, _INT, _INT, _PTR],
        _INT,
    ),
    "warpc_attn_decode_x": ([_INT, *[_PTR] * 5, _INT, _INT, _INT, _INT, _FLOAT, _INT, _INT, _PTR], _INT),
    "warpc_attn_prep_q": ([_INT, *[_PTR] * 9, _INT, _INT, _INT, _INT, _PTR, _FLOAT, _INT, _INT, _PTR], _INT),
    "warpc_attn_decode_q": ([_INT, *[_PTR] * 6, _INT, _INT, _INT, _INT, _FLOAT, _INT, _PTR], _INT),
    "warpc_attn_ws_bytes": ([_INT, _INT, _INT, _INT, _INT], _SIZE),
    "warpc_argmax": ([_PTR, _INT, _INT, _PTR, _PTR, _PTR], _INT),
    "warpc_argmax_ws_bytes": ([], _SIZE),
    "warpc_prefetch": ([_PTR, _SIZE, _INT, _INT, _PTR], _INT),
    "warpc_error_string": ([_INT], C.c_char_p),
}

SIGNATURES.update(inkling.SIGNATURES)


for _family in (hybrid, w4, attn_gated, deltanet, head_refine):
    SIGNATURES.update(_family.SIGNATURES)


def nvcc() -> str:
    """$NVCC, else nvcc on PATH, else $CUDA_HOME/bin/nvcc."""
    path = os.environ.get("NVCC") or shutil.which("nvcc")
    if not path and os.environ.get("CUDA_HOME"):
        cand = Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"
        path = str(cand) if cand.exists() else None
    if not path:
        raise RuntimeError("no nvcc to build the kernel library: set NVCC or CUDA_HOME, or put nvcc on PATH")
    return path


def cache_dir() -> Path:
    """$WARPC_CACHE_DIR, else $XDG_CACHE_HOME/warpc, else ~/.cache/warpc."""
    if os.environ.get("WARPC_CACHE_DIR"):
        return Path(os.environ["WARPC_CACHE_DIR"])
    base = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(base) / "warpc"


def source_digest() -> str:
    """sha256 of every file the library is built from (names and bytes; kernels/tests is not part of the library)."""
    h = hashlib.sha256()
    for p in sorted(KERNELS.iterdir()):
        if p.is_file():
            h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def build(sm: int) -> Path:
    """The library for compute capability `sm` (86 = sm_86), compiled first if the cache does not have it."""
    out = cache_dir() / f"libwarpc_glue-{source_digest()[:16]}-sm{sm}.so"
    if out.exists():
        return out
    make = shutil.which("make")
    if make is None:
        raise RuntimeError("no make on PATH to build the kernel library")
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="build-", dir=cache))
    try:
        dirs = [f"BUILD={scratch}", f"CACHE={cache / 'objcache'}"]
        cmd = [make, "-C", str(KERNELS), f"NVCC={nvcc()}", f"SM={sm}", *dirs]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            log = (r.stdout + r.stderr)[-4000:]
            raise RuntimeError(f"building the kernel library failed ({' '.join(cmd)}):\n{log}")
        os.replace(scratch / "libwarpc_glue.so", out)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out


@functools.cache
def load(sm: int) -> C.CDLL:
    """The kernel library for compute capability `sm`, built on demand, with every entry point's signature set."""
    lib = C.CDLL(str(build(sm)))
    for name, (args, res) in SIGNATURES.items():
        fn = getattr(lib, name, None)  # an entry of a skipped file stays unbound
        if fn is not None:
            fn.argtypes, fn.restype = args, res
    return lib
