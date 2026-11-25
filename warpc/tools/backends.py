"""The warpc backends the 27B tools run on, by command-line name."""

from __future__ import annotations

BACKENDS = ("composite", "cuda", "swiftmk")


def make_backend(name: str, head: str, kernels: str | None):
    """`composite`: warpc kernels first, the hand engine's as fallback; `kernels` names the warpc kernels admitted,
    comma-separated (None: all). `head`: the lm_head kernel of every backend (bf16 or int4_refine)."""
    from warpc.composite import CompositeBackend
    from warpc.cuda import CudaBackend
    from warpc.swiftmk import SwiftmkBackend

    if name == "swiftmk":
        return SwiftmkBackend(head)
    if name == "cuda":
        return CudaBackend(head)
    admitted = [k for k in kernels.split(",") if k] if kernels is not None else None
    return CompositeBackend(CudaBackend(head), SwiftmkBackend(head), kernels=admitted)
