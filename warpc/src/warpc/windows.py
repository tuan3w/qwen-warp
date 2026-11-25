"""The window plan: how a prompt is cut into prefill segments.

A window prefill is exact only for a fixed segmentation (cuBLAS picks its kernel by the row count M, the DeltaNet chunk
scan is bit-stable only across cuts that are multiples of its 64-row chunk), so the segmentation is a pure function of
the prompt tokens: a cold prefill and a prefill resumed at a segment end of the same plan run the same segments from
there on, and their states are bit-equal. `plan_windows` is the port of the hand engine's `Runner._plan` (fold off:
the last prompt token is its own head step, the class-D step of the decode path)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

IM_START = (
    248045  # Qwen3.5 <|im_start|>: p + 2 is a chat-turn boundary (the template re-renders earlier turns after it)
)
WINDOW_ROWS = 4096  # most rows of a window
W4_MAX = (
    240  # windows of at most this many rows multiply through the W4 mma kernel, larger ones through dequant + cuBLAS
)


def plan_windows(
    ids: Sequence[int], *, rows: int = WINDOW_ROWS, w4_max: int = W4_MAX, turn_token: int | None = IM_START
) -> list[int]:
    """End positions of the prefill segments of ids[:-1] (the last token is the head step): segments of at most `rows`
    rows, cut earlier at a chat-turn boundary p + 2 (ids[p] == turn_token) more than `w4_max` rows after the previous
    cut, because the next turn re-renders the assistant message after the header and resumes there; a cut that saves
    fewer rows than a weight pass costs is not made. The decision at a position looks only at the tokens before it, so
    a prompt that extends another keeps its cuts (except the last end, `len(ids) - 1`)."""
    n = len(ids) - 1
    if n <= 0:
        return []
    marks = np.empty(0, dtype=np.int64)
    if turn_token is not None and n > 1:
        marks = np.flatnonzero(np.asarray(ids[: n - 1], dtype=np.int64) == turn_token) + 2
    ends, i = [], 0
    while i < n:
        j = min(n, i + rows)
        k = int(np.searchsorted(marks, i + w4_max, side="right"))
        if k < len(marks) and marks[k] < j:
            j = int(marks[k])
        ends.append(j)
        i = j
    return ends
