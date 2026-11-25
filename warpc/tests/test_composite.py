"""The composite backend and the side-launch rule, CPU only: the primary's kernel wins wherever both tables match the
same nodes, `kernels` admits a subset of the primary, and a side task rides beside the task on its input.

  uv run --package warpc python warpc/tests/test_composite.py
"""

from collections import Counter
from dataclasses import replace

from harness import run
from test_compiler import QWEN3_06B, build
from warpc.compiler import CapsTable, Task, lower
from warpc.composite import CompositeBackend
from warpc.cuda import CudaBackend
from warpc.runtime import Launch, merge_sides


def _rename(cap):
    name = "x_" + cap.kernel[2:]

    def emit(m, A):
        r = cap.emit(m, A)
        r.kernel = name
        return r

    return replace(cap, kernel=name, emit=emit)


class _Renamed:
    """A backend with the CUDA table's capabilities under other kernel names (a second backend that matches the same
    nodes)."""

    name = "other"

    def __init__(self):
        t = CudaBackend().caps
        self.caps = CapsTable([_rename(c) for c in t.caps], t.target, t.unfused_ctas, t.unfused_launches)


def _kernels(backend) -> Counter:
    g = build({**QWEN3_06B, "num_layers": 2})
    return Counter(t.kernel for t in lower(g, backend, rows_max=8).tasks)


def test_the_primary_wins_where_both_match():
    got = _kernels(CompositeBackend(CudaBackend(), _Renamed()))
    assert got == _kernels(CudaBackend()), got


def test_kernels_admits_a_subset_of_the_primary():
    only = {"g_gemv", "g_attn", "g_embed", "g_head"}
    got = _kernels(CompositeBackend(CudaBackend(), _Renamed(), kernels=only))
    assert {k for k in got if k.startswith("g_")} <= only and any(k.startswith("x_") for k in got), got
    none = _kernels(CompositeBackend(CudaBackend(), _Renamed(), kernels=[]))
    assert all(k.startswith("x_") for k in none), none


def test_unknown_kernels_are_refused():
    try:
        CompositeBackend(CudaBackend(), _Renamed(), kernels=["g_nope"])
    except ValueError as e:
        assert "g_nope" in str(e), str(e)
    else:
        raise AssertionError("an unknown kernel was admitted")


def _task(kernel: str, ins: dict, outs: dict, **attrs) -> Task:
    return Task(kernel, ins, outs, attrs=attrs)


def test_a_side_task_rides_beside_the_task_on_its_input():
    mark = lambda k: Launch(k)  # noqa: E731
    tasks = [
        _task("norm", {"x": "b0"}, {"xn": "b1"}),
        _task("big", {"x": "b1"}, {"o": "b2"}),
        _task("small", {"x": "b1"}, {"o": "b3"}, side=True),
        _task("next", {"a": "b2", "b": "b3"}, {"y": "b4"}),
    ]
    launches = merge_sides(tasks, [mark(t.kernel) for t in tasks])
    assert [(ln.main, ln.side) for ln in launches] == [("norm", ()), ("big", ("small",)), ("next", ())]
    # not on its input: stays a launch of its own
    tasks[2] = _task("small", {"x": "b0"}, {"o": "b3"}, side=True)
    assert len(merge_sides(tasks, [mark(t.kernel) for t in tasks])) == 4
    # racing on a buffer is refused
    tasks[2] = _task("small", {"x": "b1"}, {"o": "b2"}, side=True)
    try:
        merge_sides(tasks, [mark(t.kernel) for t in tasks])
    except AssertionError as e:
        assert "share a buffer" in str(e), str(e)
    else:
        raise AssertionError("a side task writing the main task's output was merged")


if __name__ == "__main__":
    run(globals())
