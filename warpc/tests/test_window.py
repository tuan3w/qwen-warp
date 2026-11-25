"""The window table of the swiftmk backend: the 27B graph lowered at rows_max 4096 claims the hand's fast-prefill
launches and nothing of the decode table, and the decode plan claims no window cap.

  WARPC_TEST_SWIFT27=<27B checkpoint directory> uv run --package warpc python warpc/tests/test_window.py

CPU only (the checkpoint's headers and config).
"""

import os
from collections import Counter
from pathlib import Path

from harness import Skip, run
from warpc import frontend, sources
from warpc.compiler import lower
from warpc.errors import LoweringError
from warpc.swiftmk import SwiftmkBackend
from warpc.swiftmk.window import WINDOW_LAUNCHERS

# per layer: the input norm (+ residual add), qkv|z or q|k|v, a|b (DeltaNet layers), the mixer, o, the post norm,
# gate|up, down
WINDOW_27B = {
    "s_embed": 1,
    "w_rmsnorm": 1,
    "w_rmsnorm_add": 128,
    "w_w4_gemv": 192,
    "w_w4_gemv_silu": 64,
    "w_ab_gemv": 48,
    "w_deltanet": 48,
    "w_attn": 16,
}


def _swift27() -> Path:
    p = os.environ.get("WARPC_TEST_SWIFT27")
    if not p:
        raise Skip("WARPC_TEST_SWIFT27 (the 27B checkpoint directory) not set")
    return Path(p)


def _backend(head: str = "bf16"):
    return SwiftmkBackend(head, window=True)


def test_27b_window_plan_claims_the_fast_prefill_launches():
    src = sources.open(_swift27())
    for kv in ("bf16", "int4_rot"):
        g = frontend.build(src, ctx_max=4608, kv=kv)
        w = g.pruned(["hidden"])
        assert "ids" not in w.outputs and len(w.nodes) < len(g.nodes)
        plan = lower(w, _backend(), rows_max=4096)
        assert Counter(t.kernel for t in plan.tasks) == WINDOW_27B, plan.summary()
        assert all(t.kernel in WINDOW_LAUNCHERS for t in plan.tasks)
        down = [t for t in plan.tasks if t.kernel == "w_w4_gemv" and t.attrs["mlp_down"]]
        assert len(down) == 64 and all(t.attrs["K"] == 17408 for t in down)
        assert not any(t.attrs.get("side") for t in plan.tasks)
        assert (
            plan.tasks[-1].kernel == "w_rmsnorm_add"
            and plan.tasks[-1].outs["xn"] == plan.buffer_of[w.outputs["hidden"]]
        )
        # the decode plan of the same table claims no window cap, and a window plan's states are the decode plan's
        decode = lower(g, _backend(), rows_max=8)
        assert not any(t.kernel.startswith("w_") for t in decode.tasks)


def test_window_row_limits():
    g = frontend.build(sources.open(_swift27()), ctx_max=4608, kv="bf16").pruned(["hidden"])
    assert {t.kernel for t in lower(g, _backend(), rows_max=17).tasks} <= set(WINDOW_27B)
    assert not any(t.kernel.startswith("w_") for t in lower(g, _backend(), rows_max=8).tasks)
    try:
        lower(g, _backend(), rows_max=4097)
    except LoweringError:
        pass
    else:
        raise AssertionError("4097 rows exceed the window kernels")


if __name__ == "__main__":
    run(globals())
