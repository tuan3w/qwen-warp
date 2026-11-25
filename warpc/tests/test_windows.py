"""plan_windows against the hand engine's planner (`server/smk_runner.py` Runner._plan, fold off) on random prompts
chat-turn headers, and the properties a resume relies on.

  uv run --package warpc python warpc/tests/test_windows.py

The hand planner needs the engine's Python side (torch, fla, vllm) but no GPU work: it is called on a stub."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from harness import Skip, run
from warpc.windows import IM_START, plan_windows

SERVER = Path(__file__).resolve().parents[2] / "server"


def hand_plan():
    sys.path.insert(0, str(SERVER))
    try:
        import smk_runner
    except Exception as e:  # the hand engine's imports (fla, vllm) are not installed
        raise Skip(f"cannot import the hand runner: {e}") from e
    stub = SimpleNamespace(fp=SimpleNamespace(fold=False, seg=4096, w4_max=240))
    return lambda ids: smk_runner.Runner._plan(stub, ids)[0]


def prompts(seed: int, count: int):
    g = np.random.default_rng(seed)
    for _ in range(count):
        n = int(g.choice([1, 2, 3, 5, 100, 241, 242, 480, 4095, 4096, 4097, 9000, 20000]))
        ids = g.integers(0, 1000, size=n)
        marks = g.random(n) < g.choice([0.0, 0.001, 0.005, 0.02])
        ids[marks] = IM_START
        yield ids.tolist()


def test_equals_the_hand_planner():
    hand = hand_plan()
    n = 0
    for ids in prompts(0, 300):
        assert plan_windows(ids) == hand(ids), len(ids)
        n += 1
    assert n == 300


def test_edges():
    assert plan_windows([]) == [] and plan_windows([7]) == []
    assert plan_windows([7, 8]) == [1]
    # the headers at the very end are visible only up to len - 2
    ids = [1] * 1000 + [IM_START, 3, 4]
    assert plan_windows(ids) == [1002]


def test_plan_is_prefix_consistent():
    """The decision at a position looks only at the tokens before it: a prompt that extends another keeps every end but
    the last one (the artificial end at len - 1)."""
    g = np.random.default_rng(1)
    base = g.integers(0, 1000, size=30000)
    base[g.random(30000) < 0.003] = IM_START
    for n in (500, 4097, 10000, 19999):
        a, b = base[:n].tolist(), base[: n + 3000].tolist()
        pa, pb = plan_windows(a), plan_windows(b)
        assert pa[:-1] == pb[: len(pa) - 1], n
        assert pb[len(pa) - 1] >= pa[-1], n


def test_segments_are_bounded_and_cut_at_turn_headers():
    g = np.random.default_rng(2)
    ids = g.integers(0, 1000, size=30000)
    ids[g.random(30000) < 0.002] = IM_START
    ends = plan_windows(ids.tolist())
    prev = 0
    for e in ends:
        assert 0 < e - prev <= 4096
        prev = e
    assert ends[-1] == len(ids) - 1
    marks = set((np.flatnonzero(ids[:-2] == IM_START) + 2).tolist())
    cuts = [e for e, p in zip(ends, [0, *ends], strict=False) if e - p < 4096 and e != ends[-1]]
    assert cuts and all(c in marks for c in cuts)


if __name__ == "__main__":
    run(globals())
