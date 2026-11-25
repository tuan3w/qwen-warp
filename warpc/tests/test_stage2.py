"""Stage 2 of the Qwen3.5 hybrid, CPU only (the 27B checkpoint's headers and config): the exit test, `lower(graph,
CudaBackend())` with no swiftmk capability, and the claim that the warpc plan is task for task the hand engine's.

  WARPC_TEST_SWIFT27=<27B checkpoint dir> uv run --package warpc python warpc/tests/test_stage2.py

The end-to-end numbers (ids and logits against the hand engine) are tools/gate27.py's, on the GPU.
"""

import os
from collections import Counter
from pathlib import Path

from harness import Skip, run
from warpc import frontend, sources
from warpc.compiler import lower
from warpc.cuda import CudaBackend
from warpc.errors import LoweringError
from warpc.swiftmk import SwiftmkBackend

# the hand engine's launches of one step, by the warpc kernel that replaces each
HAND_TO_WARPC = {
    "s_embed": "g_embed",
    "s_rmsnorm": "g_rmsnorm_zc",
    "s_rmsnorm_add": "g_rmsnorm_zc_add",
    "s_w4_gemv": "g_w4_gemv",
    "s_w4_gemv_silu": "g_w4_gemv_silu",
    "s_ab_gemv": "g_gemv_small",
    "s_deltanet": "g_deltanet",
    "s_attn": "g_attn_gated",
    "s_head": "g_head",
    "s_head_refine": "g_head_refine",
}


def _buffers(t):
    """The buffers a task reads and writes (the roles are per kernel)."""
    return sorted(t.ins.values()), sorted(t.outs.values())


def _source():
    p = os.environ.get("WARPC_TEST_SWIFT27")
    if not p:
        raise Skip("WARPC_TEST_SWIFT27 (the 27B checkpoint directory) not set")
    return sources.open(Path(p))


def test_the_27b_lowers_onto_cuda_kernels_alone():
    src = _source()
    for kv in ("bf16", "int4_rot"):
        g = frontend.build(src, ctx_max=4608, kv=kv)
        for head in ("bf16", "int4_refine"):
            mine = lower(g, CudaBackend(head), rows_max=8)
            hand = lower(g, SwiftmkBackend(head), rows_max=8)
            assert [t.kernel for t in mine.tasks] == [HAND_TO_WARPC[t.kernel] for t in hand.tasks], (kv, head)
            for a, b in zip(mine.tasks, hand.tasks, strict=True):
                assert a.covers == b.covers and _buffers(a) == _buffers(b), (a.kernel, b.kernel)
            assert Counter(t.kernel for t in mine.tasks)["g_w4_gemv"] == 192


def test_fp8_cache_stays_with_the_hand_kernels():
    g = frontend.build(_source(), ctx_max=4608, kv="fp8")
    try:
        lower(g, CudaBackend(), rows_max=8)
    except LoweringError as e:
        assert "cache=fp8" in str(e), str(e)
    else:
        raise AssertionError("the CUDA table claimed an fp8 KV cache")


if __name__ == "__main__":
    run(globals())
