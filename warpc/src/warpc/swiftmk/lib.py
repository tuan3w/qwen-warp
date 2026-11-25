"""ctypes binding of the hand engine's kernel library (libswiftmk.so, built by `make engine`), loaded from
$WARPC_SWIFTMK_LIB only: warpc shares no code with engine/ and bakes in no path.

The library is refused when its size queries disagree with the layouts this package was written against (ABI_SIZES, the
KV cache sizes per format): a changed workspace or state layout would otherwise corrupt memory silently. A signature
change that keeps every size is not detectable from the binary; `load` warns when the headers next to the library
(<lib>/../../include) differ from HEADER_SHA256, the digest of the headers these argtypes transcribe.
"""

from __future__ import annotations

import ctypes as C
import functools
import hashlib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from ..kvcache import FORMATS

ENV = "WARPC_SWIFTMK_LIB"

# engine/include/engine.h smk_epi kinds
EPI_SEGS, EPI_SILU, EPI_F32 = 0, 1, 2
HEADQ_I4R = 3  # head_q.h SMK_HEADQ_I4R: int4 g128 head in the repacked W4 layout
KV_DTYPE = {"bf16": 0, "fp8": 1, "int4_rot": 2}  # attn.h kv_dtype
KV_HEADS, KV_HEAD_DIM = 4, 256  # attn.h: the cache layout [2][4][ctx][256] is fixed

ABI_SIZES = {
    "swiftmk_deltanet_ws_bytes": 397312,
    "swiftmk_deltanet_dbuf_floats": 263680,
    "swiftmk_attn_ws_bytes": 8122368,
    "smk_argmax_ws_bytes": 8256,
    "smk_headq_ws_bytes": 1065088,
}
HEADERS = ("attn.h", "deltanet.h", "engine.h", "head_q.h")
HEADER_SHA256 = "2fedec56e110abbfc106b55bbf1a2419de0b918ea3032444fd7443ccb047d0d7"


class Epi(C.Structure):
    """engine.h smk_epi: the W4 GEMV epilogue (kind, up to three segment ends and outputs)."""

    _fields_ = [("kind", C.c_int), ("end", C.c_int * 3), ("out", C.c_void_p * 3)]


_vp, _i, _f, _sz = C.c_void_p, C.c_int, C.c_float, C.c_size_t
_W4 = [_vp, _vp, _vp, _i, _i, _i, C.POINTER(Epi), _vp]
# name -> (argtypes, restype); every launch returns cudaError_t
_SIGNATURES: dict[str, tuple[list, type]] = {
    "smk_w4_gemv": (_W4, _i),
    "smk_w4r_gemv": (_W4, _i),
    "smk_bf16_gemv_f32": ([_vp, _vp, _i, _i, _i, _vp, _vp], _i),
    "smk_bf16_gemv_small": ([_vp, _vp, _i, _i, _i, _i, _vp, _vp, _vp], _i),
    "smk_rmsnorm": ([_vp, _vp, _vp, _vp, _vp, _i, _i, _f, _vp], _i),
    "smk_embed": ([_vp, _vp, _vp, _i, _i, _vp], _i),
    "smk_argmax_ws_bytes": ([], _sz),
    "smk_argmax": ([_vp, _i, _i, _vp, _vp, _vp, _vp], _i),
    "swiftmk_deltanet_ws_bytes": ([], _sz),
    "swiftmk_deltanet_dbuf_floats": ([], _sz),
    "swiftmk_deltanet_step_lazy2": ([_i, *[_vp] * 10, _vp, _i, _i, _vp, _vp, _vp, _vp, _vp, _vp], _i),
    "swiftmk_attn_ws_bytes": ([], _sz),
    "swiftmk_attn_kv_bytes": ([_i, _i], _sz),
    "swiftmk_attn_step": ([_i, *[_vp] * 6, _i, _i, _f, _f, _vp, _vp, _vp, _vp, _vp], _i),
    "smk_headq_ws_bytes": ([], _sz),
    "smk_headq_gemv": ([_i, _vp, _vp, _vp, _i, _i, _i, _vp, _vp], _i),
    "smk_headq_select_rescore": ([_vp, _vp, _vp, _i, _i, _i, _i, _vp, _vp, _vp, _vp], _i),
    "cudaGetErrorString": ([_i], C.c_char_p),
}


@dataclass(frozen=True)
class Library:
    """The loaded library: the bound entry points as attributes by C name, `sha256` the digest of the loaded file."""

    path: Path
    sha256: str
    cdll: C.CDLL

    def __getattr__(self, name: str):
        if name in _SIGNATURES:
            return getattr(self.cdll, name)
        raise AttributeError(name)

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what}: {self.cdll.cudaGetErrorString(rc).decode()} (CUDA error {rc})")

    def kv_bytes(self, ctx: int, cache: str) -> int:
        """Bytes of one layer's [2][4][ctx][256] cache in format `cache`."""
        return int(self.cdll.swiftmk_attn_kv_bytes(ctx, KV_DTYPE[cache]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def header_digest(include: Path) -> str:
    h = hashlib.sha256()
    for name in HEADERS:
        h.update((include / name).read_bytes())
    return h.hexdigest()


def abi_mismatches(cdll: C.CDLL) -> list[str]:
    """Every size query whose answer differs from the layouts this binding was written against."""
    bad = [f"{n}() = {v} (expected {ABI_SIZES[n]})" for n in ABI_SIZES if (v := getattr(cdll, n)()) != ABI_SIZES[n]]
    for ctx in (1, 4096, 65536):
        dense = 2 * KV_HEADS * ctx * KV_HEAD_DIM
        want = {
            "bf16": 2 * dense,
            "fp8": dense,
            # rows_max 8: the hand engine's ring is fixed (512 slots), so any step bound gives the same bytes
            "int4_rot": FORMATS["int4_rot"].nbytes(ctx, KV_HEADS, KV_HEAD_DIM, 8),
        }
        for cache, n in want.items():
            got = cdll.swiftmk_attn_kv_bytes(ctx, KV_DTYPE[cache])
            if got != n:
                bad.append(f"swiftmk_attn_kv_bytes({ctx}, {cache}) = {got} (expected {n})")
    return bad


@functools.cache
def _load(path: Path) -> Library:
    cdll = C.CDLL(str(path))
    for name, (args, res) in _SIGNATURES.items():
        fn = getattr(cdll, name)
        fn.argtypes, fn.restype = args, res
    bad = abi_mismatches(cdll)
    if bad:
        raise RuntimeError(f"{path}: not the libswiftmk ABI warpc binds: " + "; ".join(bad))
    include = path.parent.parent / "include"
    if all((include / n).is_file() for n in HEADERS) and header_digest(include) != HEADER_SHA256:
        warnings.warn(
            f"{include}: the engine headers differ from the ones warpc.swiftmk transcribes; check the argtypes",
            stacklevel=3,
        )
    return Library(path, sha256(path), cdll)


def load() -> Library:
    """The library named by $WARPC_SWIFTMK_LIB (one load per process and path)."""
    p = os.environ.get(ENV)
    if not p:
        raise RuntimeError(f"${ENV} is not set: point it at libswiftmk.so (built by `make engine`)")
    path = Path(p).resolve()
    if not path.is_file():
        raise RuntimeError(f"${ENV}={p}: no such file")
    return _load(path)
