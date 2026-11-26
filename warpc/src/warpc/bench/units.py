"""Units: the GPU processes a run is made of, and how they are planned, wrapped, spawned and interleaved.

A unit is one child process `python -m warpc.bench unit SPEC.json`: it loads an engine, measures its cells and writes
them line by line. Processes are the isolation boundary (a crash or an out-of-memory loses one unit's cells, never the
session) and the lease granule: `--wrap CMD ...` prefixes every child command (a lease wrapper), and the planner keeps
a unit under `UNIT_CAP_S`, below the 12 minute lease. A `--wrap` command is an argument, never an environment variable.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import protocol, schema
from .engines import Cannot, EngineSpec
from .schema import Cell

UNIT_CAP_S = 600  # below the 12 minute lease, with room for the lease's own start
PROMPT_HEADROOM = 512  # n_new = 256 and a speculative round's overshoot of 8 (k + 1)
ASSET_ENV = {
    "WARPC_MODELS": "set WARPC_MODELS to the model directory (`python -m warpc.bench fetch`)",
    "WARPC_TEST_SWIFT27": "set WARPC_TEST_SWIFT27 to the Qwen3.5 27B checkpoint directory",
    "WARPC_LEGACY_DIR": "WARPC_LEGACY_DIR unset: point it at a checkout of release tag v0.12.0 (warpc/README.md)",
    "WARPC_SWIFTMK_LIB": "set WARPC_SWIFTMK_LIB to libswiftmk.so built from the v0.12.0 checkout (warpc/README.md)",
}


def round_up_256(n: int) -> int:
    return -(-n // 256) * 256


def ctx_max(prompt_len: int) -> int:
    """`round_up_256(prompt_len + 512)` for every cell that has a prompt: 4,096 -> 4,608, the shipped 27B gate value."""
    return round_up_256(prompt_len + PROMPT_HEADROOM)


@dataclass(frozen=True)
class Unit:
    """One GPU process: the lease granule."""

    suite: str
    cells: tuple[Cell, ...]  # status="planned"
    engine: EngineSpec | None
    est_s: int
    needs: tuple[str, ...] = ()  # assets (env vars): unmet -> the unit's cells are `skip`, never a pass
    params: tuple[tuple[str, Any], ...] = ()  # what the suite's runner takes besides the engine: n_new, ctx_max, ...

    def param(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "cells": [c.to_dict() for c in self.cells],
            "engine": None if self.engine is None else self.engine.to_json(),
            "est_s": self.est_s,
            "needs": list(self.needs),
            "params": [list(p) for p in self.params],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Unit:
        return cls(
            d["suite"],
            tuple(Cell.from_dict(c) for c in d["cells"]),
            None if d["engine"] is None else EngineSpec.from_json(d["engine"]),
            d["est_s"],
            tuple(d["needs"]),
            tuple((k, v) for k, v in d["params"]),
        )


@dataclass
class Plan:
    units: list[Unit] = field(default_factory=list)
    decided: list[Cell] = field(default_factory=list)  # cannot / skip decided on the CPU, with reasons

    @property
    def est_s(self) -> int:
        return sum(u.est_s for u in self.units)

    @property
    def cells(self) -> list[Cell]:
        return [c for u in self.units for c in u.cells]

    def render(self) -> str:
        lines = []
        for i, u in enumerate(self.units):
            lines.append(
                f"unit {i}: {u.suite} {u.engine.text if u.engine else ''} ~{u.est_s} s, {len(u.cells)} cell(s)"
            )
            for c in u.cells:
                note = f"  [{c.extra['preflight']}]" if c.extra.get("preflight") else ""
                est = f"  est {c.extra['estimate_gib']:.1f} GiB" if c.extra.get("estimate_gib") else ""
                lines.append(f"    {c.key}  workload {c.workload.get('ids_sha', '')[:8]}{est}{note}")
        for c in self.decided:
            lines.append(f"{c.status:>7}  {c.key}  {c.code}: {c.reason}")
        minutes = self.est_s / 60
        lines.append(
            f"{len(self.units)} unit(s), ~{minutes:.1f} min of GPU; {len(self.decided)} cell(s) decided on the CPU"
        )
        return "\n".join(lines)


def pack_cells(
    items: Sequence[tuple[Cell, int]], *, fixed_s: int, cap_s: int = UNIT_CAP_S
) -> list[tuple[list[Cell], int]]:
    """Group cells (each with its estimated seconds) into units of at most `cap_s`, `fixed_s` (load and warmup) per
    unit, in the order given. A cell that alone exceeds the cap cannot be split and is an error."""
    units: list[tuple[list[Cell], int]] = []
    for cell, est in items:
        if fixed_s + est > cap_s:
            raise ValueError(f"cell {cell.key} needs ~{fixed_s + est} s, over the {cap_s} s unit cap")
        if units and units[-1][1] + est <= cap_s:
            units[-1][0].append(cell)
            units[-1] = (units[-1][0], units[-1][1] + est)
        else:
            units.append(([cell], fixed_s + est))
    return units


def missing_assets(needs: Iterable[str]) -> list[str]:
    return [ASSET_ENV[n] for n in needs if not os.environ.get(n)]


def decide(unit: Unit, status: str, code: str, reason: str) -> list[Cell]:
    return [Cell(c.key, status, code, reason, c.workload, extra=c.extra) for c in unit.cells]  # type: ignore[arg-type]


# ---- arms and worktrees


@dataclass(frozen=True)
class Arm:
    """One side of an A/B: an engine spec on a commit (a temporary `git worktree`, removed after) or on the working
    tree. The measuring code is always this tree's: a worktree runs the arm's library under this tree's harness."""

    spec: EngineSpec
    ref: str | None = None


def repo_root() -> Path:
    here = Path(__file__).resolve().parent
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=here, check=False
    ).stdout.strip()
    if not out:
        raise RuntimeError("an arm on a commit needs a git checkout")
    return Path(out)


class Worktree:
    """`git worktree add --detach` of `ref` under the system temp dir with this tree's `bench` package copied over its
    own (a commit that predates the suite has none)."""

    def __init__(self, ref: str):
        self.ref = ref
        self.path: Path | None = None
        self.commit: str | None = None

    def __enter__(self) -> Worktree:
        root = repo_root()
        self.path = Path(tempfile.mkdtemp(prefix="warpc-arm-")) / "tree"
        subprocess.run(["git", "worktree", "add", "--detach", str(self.path), self.ref], cwd=root, check=True,
                       capture_output=True)  # fmt: skip
        self.commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=self.path, capture_output=True,
                                     text=True, check=True).stdout.strip()  # fmt: skip
        shutil.copytree(Path(__file__).resolve().parent, self.path / "warpc/src/warpc/bench", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__"))  # fmt: skip
        return self

    def __exit__(self, *exc: object) -> None:
        if self.path is not None:
            subprocess.run(["git", "worktree", "remove", "--force", str(self.path)], cwd=repo_root(), check=False,
                           capture_output=True)  # fmt: skip
            shutil.rmtree(self.path.parent, ignore_errors=True)

    @property
    def pythonpath(self) -> str:
        return str(self.path / "warpc/src")


# ---- spawning a unit

_EXCEPTION = re.compile(r"^(?:\([^)]*\)\s+)?((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit|Interrupt)\b.*)$")


def _failure_reason(returncode: int, stderr_lines: list[str]) -> str:
    """The last exception line a child printed, else its last line, else its exit status."""
    exceptions = [m.group(1) for line in stderr_lines if (m := _EXCEPTION.match(line.strip()))]
    last = next((ln.strip() for ln in reversed(stderr_lines) if ln.strip()), None)
    return (exceptions[-1] if exceptions else last) or f"exit status {returncode}"


@dataclass
class Visit:
    """What identifies one process of a session: for the result cells and the log line."""

    run: str
    index: int
    arm: str | None = None
    pair: int | None = None
    arm_extra: dict[str, Any] = field(default_factory=dict)


def spawn(unit: Unit, visit: Visit, *, wrap: Sequence[str] = (), pythonpath: str | None = None) -> list[Cell]:
    """Run `unit` in a child process and return its cells: the ones the child wrote, and an `error` cell (with the last
    exception line of its stderr) for every planned cell it did not finish. Unmet `needs` give `skip` cells without a
    process."""
    if missing := missing_assets(unit.needs):
        cells = decide(unit, "skip", "assets", "; ".join(missing))
        return _tag(cells, visit)
    with tempfile.TemporaryDirectory(prefix="warpc-unit-") as tmp:
        spec_path, out_path = Path(tmp) / "unit.json", Path(tmp) / "cells.jsonl"
        idle = 30 if wrap else protocol.BUSY_WAIT_S  # a lease already serialises the GPU; waiting must fit inside it
        spec_path.write_text(json.dumps({"unit": unit.to_json(), "out": str(out_path), "idle_wait_s": idle}))
        env_prefix = ["env", f"PYTHONPATH={pythonpath}"] if pythonpath else []
        command = [*wrap, *env_prefix, sys.executable, "-m", "warpc.bench", "unit", str(spec_path)]
        print(f"unit {visit.index}: {unit.suite} {unit.engine.text if unit.engine else ''}", flush=True)
        stderr: list[str] = []
        with subprocess.Popen(command, stderr=subprocess.PIPE, text=True, errors="replace") as process:
            for line in process.stderr:
                sys.stderr.write(line)
                stderr.append(line)
                del stderr[:-200]
        done = _read_cells(out_path)
    finished = {c.key for c in done}
    reason = _failure_reason(process.returncode, stderr)
    for c in unit.cells:
        if c.key not in finished:
            done.append(Cell(c.key, "error", "crash", reason, c.workload, extra=c.extra))
    return _tag(done, visit)


def _read_cells(path: Path) -> list[Cell]:
    if not path.exists():
        return []
    cells = []
    for line in path.read_text().splitlines():
        if line.strip():
            item = schema.loads(line)
            if isinstance(item, Cell):
                cells.append(item)
    return cells


def _tag(cells: list[Cell], visit: Visit) -> list[Cell]:
    for c in cells:
        c.run, c.unit, c.arm, c.pair = visit.run, visit.index, visit.arm, visit.pair
        c.extra = {**c.extra, **visit.arm_extra}
    return cells


# ---- the child process


def child_main(spec_path: Path) -> int:
    """The `unit` verb: measure the unit's cells and write each as it is finished. A `Cannot` and every other exception
    become the cells' status (`cannot`, `error`): results are in the file, the exit status is 0 unless the interpreter
    itself failed."""
    from . import engines
    from .suites import SUITES

    spec = json.loads(Path(spec_path).read_text())
    unit, out = Unit.from_json(spec["unit"]), Path(spec["out"])
    suite = SUITES[unit.suite]
    with schema.Writer(out) as writer:

        def give(status: str, code: str, reason: str) -> None:
            for c in decide(unit, status, code, reason):
                writer.write(c)

        busy = protocol.wait_idle(spec["idle_wait_s"])
        if busy:
            give("skip", "gpu_busy", f"foreign GPU process(es) {sorted(busy)} for {spec['idle_wait_s']} s")
            return 0
        engine = None
        try:
            with protocol.Sampler() as sampler:
                engine = engines.make(unit.engine)
                for cell in suite.run_unit(unit, engine, sampler):
                    writer.write(cell)
        except Cannot as e:
            give("cannot", e.code, str(e))
        except Exception as e:
            import traceback

            traceback.print_exc()
            give("error", "exception", f"{type(e).__name__}: {e}")
        finally:
            if engine is not None:
                engine.close()
    return 0


# ---- sessions


def new_run_id() -> tuple[str, str]:
    """(run id, session id): a session is the visits that were interleaved with each other."""
    session = time.strftime("%Y%m%dT%H%MZ", time.gmtime())
    return f"{session}-{secrets.token_hex(2)}", session


def run_plan(
    plan: Plan,
    *,
    run: str,
    visits: int = 1,
    wrap: Sequence[str] = (),
    on_cells: Callable[[list[Cell]], None] = lambda cells: None,
) -> list[Cell]:
    """Every unit of `plan` `visits` times, visit-major (each round runs the units in plan order), so that the visits
    of one cell are spread over the session. The cells decided on the CPU come first. Cells are passed to `on_cells`
    as each process ends."""
    cells = [Cell(**{**vars(c), "run": run}) for c in plan.decided]
    on_cells(cells)
    for v in range(visits):
        for i, unit in enumerate(plan.units):
            got = spawn(unit, Visit(run, i + v * len(plan.units)), wrap=wrap)
            on_cells(got)
            cells.extend(got)
    return cells


def ab_order(pair: int) -> tuple[str, str]:
    """Pair 0 (the discarded warm-up pair) runs A then B, then the order alternates, so that a linear drift cancels to
    first order: A B | B A | A B ..."""
    return ("A", "B") if pair % 2 == 0 else ("B", "A")


def run_ab(
    arms: dict[str, Arm],
    units_for: Callable[[Arm], list[Unit]],
    *,
    run: str,
    pairs: int,
    wrap: Sequence[str] = (),
    on_cells: Callable[[list[Cell]], None] = lambda cells: None,
) -> list[Cell]:
    """The interleaved session: a discarded warm-up pair and `pairs` measured pairs of one visit per arm for each unit.
    `units_for(arm)`: the arm's units (the same cells as the other arm's, differing in engine spec)."""
    from contextlib import ExitStack

    cells: list[Cell] = []
    with ExitStack() as stack:
        trees: dict[str, Worktree | None] = {
            name: stack.enter_context(Worktree(arm.ref)) if arm.ref else None for name, arm in arms.items()
        }
        units = {name: units_for(arm) for name, arm in arms.items()}
        if len({len(u) for u in units.values()}) != 1:
            raise ValueError("the arms plan different numbers of units")
        n_units = len(units["A"])
        index = 0
        for pair in range(pairs + 1):
            for u in range(n_units):
                for name in ab_order(pair):
                    tree = trees[name]
                    extra = {"arm_ref": arms[name].ref, "arm_commit": tree.commit if tree else None}
                    visit = Visit(run, index, name, pair, extra)
                    got = spawn(units[name][u], visit, wrap=wrap, pythonpath=tree.pythonpath if tree else None)
                    index += 1
                    on_cells(got)
                    cells.extend(got)
    return cells


def iter_ok(cells: Iterable[Cell]) -> Iterator[Cell]:
    return (c for c in cells if c.status == "ok")
