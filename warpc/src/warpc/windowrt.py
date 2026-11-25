"""The window variant of a Runtime: the same model run as eager steps of up to N rows.

A `Runtime` executes its decode plan (steps of up to 8 rows) as CUDA graphs. A `WindowVariant` is the same graph lowered
a second time at rows_max = N against a table with window kernels (`swiftmk.window`), pruned to the final-norm rows
(`hidden`): it executes a prompt chunk of T <= N rows in one eager pass over the 64 layers. It owns only its activation
buffers and its token input; the packs, the states (KV caches, DeltaNet and conv state), the position scalar, the
backend's cursors and the kernel scratch are the runtime's (a pack the window kernels read is the pack the decode
kernels read), so a window continues exactly where a decode step stopped and the other way round.

The window kernels read the recurrent state in place of the ring: before a window the backend's pending lazy rows are
materialised (`Executor.materialize`), and a window leaves the cursors where they were."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .compiler import Plan
from .ir import Graph
from .memory import _ITEMSIZE
from .runtime import _DTYPES, Memory, _live_tasks

if TYPE_CHECKING:
    from .runtime import Runtime


class WindowVariant:
    def __init__(self, runtime: Runtime, graph: Graph, plan: Plan):
        """`graph`: the runtime's graph pruned to the `hidden` output; `plan`: its lowering at rows_max = N."""
        self.graph, self.plan = graph, plan
        self.rows = rows = plan.rows_max
        if "hidden" not in graph.outputs or "ids" in graph.outputs:
            raise ValueError("a window graph outputs the final-norm rows and nothing of the head")
        ex = runtime._executor
        zeros = lambda n, dtype: torch.zeros(n, dtype=dtype, device="cuda")  # noqa: E731
        self._buffers = {b.name: zeros(rows * b.dim, _DTYPES[b.dtype]) for b in plan.buffers.values()}
        self.tokens = zeros(rows, torch.int32)
        self._packs = {p.name: runtime.arena.pack(ex, p) for p in plan.packs.values()}
        self._constants = {n: torch.tensor(v, dtype=torch.float32, device="cuda") for n, v in plan.constants.items()}
        self.memory = Memory(
            rows_max=rows,
            pos=runtime.pos.data_ptr(),
            inputs={"@tokens": self.tokens.data_ptr()},
            buffers={k: v.data_ptr() for k, v in self._buffers.items()},
            packs={k: v.data_ptr() for k, v in self._packs.items()},
            states=runtime._memory.states,
            constants={k: v.data_ptr() for k, v in self._constants.items()},
            scratch=runtime.arena.scratch,
        )
        hidden = graph.outputs["hidden"]
        self.hidden = self._buffers[plan.buffer_of[hidden]]
        self._dim = graph.types[hidden].shape[-1]
        self._tasks = _live_tasks(graph, plan, {hidden})

    def run(self, rt: Runtime, ids: Sequence[int]) -> torch.Tensor:
        """Prefill `ids` (1 to N tokens) at the position of `rt`, the runtime this variant was built on, states only;
        returns the final-norm rows [T, H] (a view that the next window overwrites). Enqueued on the current stream; the
        runtime's position moves by T. The runtime owns the variant, so the variant keeps no reference back: a dropped
        runtime must be freed by refcount (see `_Scratch`)."""
        T = len(ids)
        if not 0 < T <= self.rows:
            raise ValueError(f"a window takes 1 to {self.rows} rows, not {T}")
        rt._require_room(T)
        self.tokens[:T].copy_(torch.tensor(list(ids), dtype=torch.int32))
        ex, stream = rt._executor, torch.cuda.current_stream().cuda_stream
        ex.materialize(rt._states)
        for launch in ex.schedule(self.memory, self._tasks, T):
            launch.main(stream)
        rt.pos.add_(T)
        rt.position += T
        return self.hidden[: T * self._dim].view(T, self._dim)


WINDOW_TRANSIENT = 512 << 20  # fla's chunk states and FlashAttention-2's temporaries, held by torch's allocator


def estimate_bytes(plan: Plan, ctx_max: int) -> int:
    """Device bytes of a window variant of `plan` at a context of `ctx_max`: its activation buffers, the kernels'
    scratch (`swiftmk.window`: the dequantised weight, the segment split of a fused GEMM, the attention and DeltaNet
    temporaries, the token-major bf16 K / V of the whole context) and WINDOW_TRANSIENT."""
    rows = plan.rows_max
    buffers = sum(rows * b.dim * _ITEMSIZE[b.dtype] for b in plan.buffers.values())
    gemms = [t for t in plan.tasks if t.kernel in ("w_w4_gemv", "w_w4_gemv_silu")]
    dequant = max((t.attrs["N"] * t.attrs["K"] * 2 for t in gemms), default=0)
    split = max((t.attrs["N"] * rows * 2 for t in gemms if len(t.attrs.get("ends", ())) > 1), default=0)
    attn = next((t for t in plan.tasks if t.kernel == "w_attn"), None)
    kv = 2 * attn.attrs["ctx"] * 4 * 256 * 2 if attn else 0
    return buffers + dequant + split + rows * 12288 * 2 + kv + WINDOW_TRANSIENT
