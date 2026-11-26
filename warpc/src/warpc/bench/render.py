"""Rendering result files: a plain table, the markdown block a pull request or a dated CHANGELOG entry carries, JSON.

Every figure comes from the cells; a cell that did not produce numbers prints `n/s [k]` with the reason as footnote k,
never an empty field."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence

from .schema import Cell, Header

COLUMNS = (
    ("decode_tok_s", "decode tok/s", "{:.1f}"),
    ("ttft_s", "TTFT ms", None),
    ("device_gib_peak", "GiB", "{:.2f}"),
)


def _fmt(cell_values: list[Cell], metric: str, template: str | None) -> str:
    values = [c.metric(metric) for c in cell_values if c.metric(metric) is not None]
    if not values:
        return ""
    v = statistics.median(values)
    return f"{1000 * v:.2f}" if metric == "ttft_s" else (template or "{:.3g}").format(v)


def _groups(cells: Iterable[Cell]) -> dict[tuple[str, str | None], list[Cell]]:
    """Visits of one cell and arm: the cells that share a key (and an `ab` arm)."""
    groups: dict[tuple[str, str | None], list[Cell]] = defaultdict(list)
    for c in cells:
        if c.pair != 0:  # pair 0 is the discarded warm-up pair of an `ab` session
            groups[c.key, c.arm].append(c)
    return groups


def table(cells: Sequence[Cell]) -> str:
    notes: list[str] = []

    def ref(text: str) -> str:
        if text not in notes:
            notes.append(text)
        return f"[{notes.index(text) + 1}]"

    head = ["cell", "arm", "n", *[name for _, name, _ in COLUMNS], "clock MHz", "status"]
    rows = [head]
    for (key, arm), group in _groups(cells).items():
        ok = [c for c in group if c.status == "ok" and not c.extra.get("contended")]
        status = ",".join(sorted({c.status for c in group}))
        flags = [f for f in ("contended", "throttled") if any(c.extra.get(f) for c in group)]
        status = ",".join([status, *flags])
        if not ok:
            first = group[0]
            what = ref(f"{first.status}({first.code}): {first.reason}") if first.reason else first.status
            numbers = [f"n/s {what}", *[""] * (len(COLUMNS) - 1)]
            rows.append([key, arm or "", str(len(group)), *numbers, "", status])
            continue
        sm = [c.clocks["sm_mhz"][1] for c in ok if c.clocks.get("sm_mhz")]
        clock = f"{statistics.median(sm):.0f}" if sm else ""
        rows.append([key, arm or "", str(len(ok)), *[_fmt(ok, m, t) for m, _, t in COLUMNS], clock, status])
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip() for r in rows]
    lines += [f"[{i + 1}] {text}" for i, text in enumerate(notes)]
    return "\n".join(lines)


def markdown(header: Header, cells: Sequence[Cell], verdict_text: str | None = None) -> str:
    """The block AGENTS.md asks of a measurement: hardware, power limit, software versions, the command, how many
    runs, and what was not measured."""
    s = header.stamp
    gpu = s.gpu
    visits = defaultdict(int)
    for c in cells:
        if c.status == "ok" and c.pair != 0:
            visits[c.key, c.arm] += 1
    n = sorted(set(visits.values()))
    versions = ", ".join(
        f"{name} {v}"
        for name, v in (
            ("torch", s.torch),
            ("CUDA", s.cuda),
            ("driver", s.driver),
            ("vLLM", s.vllm),
            ("llama.cpp", s.llama_cpp),
            ("Python", s.python),
        )
        if v
    )
    commit = s.warpc.get("commit")
    dirty = "dirty" if s.warpc.get("dirty") else "clean"
    lines = [
        f"**Hardware**: {gpu.get('name')} at {gpu.get('power_limit_w'):.0f} W power limit"
        if gpu.get("power_limit_w") is not None
        else f"**Hardware**: {gpu.get('name')}",
        f"**Software**: warpc {commit} ({dirty}), {versions}",
        f"**Command**: `python -m warpc.bench {' '.join(header.argv)}`",
        f"**Runs**: session {header.session}, run {header.run}; visits per cell: {', '.join(map(str, n)) or 'none'}",
        "",
        "```",
        table(cells),
        "```",
    ]
    unmeasured = [c for c in cells if c.status in ("cannot", "skip", "error", "fail")]
    seen = []
    for c in unmeasured:
        line = f"- `{c.key}`: {c.status}({c.code}) {c.reason}"
        if line not in seen:
            seen.append(line)
    lines += ["", "**Not measured**: " + ("" if seen else "nothing"), *seen]
    if verdict_text:
        lines += ["", "```", verdict_text, "```"]
    return "\n".join(lines)


def as_json(header: Header, cells: Sequence[Cell]) -> str:
    return json.dumps({"header": header.to_dict(), "cells": [c.to_dict() for c in cells]}, indent=1)
