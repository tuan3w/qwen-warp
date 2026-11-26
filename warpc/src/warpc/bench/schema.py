"""The result file: JSON Lines, one `Header` line (the run's stamp) then one `Cell` line per measured or decided cell.

One shape for every suite. A `Cell` is both the planned and the measured thing (`status="planned"` until a unit
fills it in); `cannot` and `skip` cells are lines of the file too, so a table prints `n/s (reason)` from the same data.
`read` rejects a file written by a newer schema; the version bumps only with a migration note in CHANGELOG.md.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, get_args

SCHEMA = 1

Status = Literal["planned", "ok", "fail", "cannot", "skip", "error"]
STATUSES: tuple[str, ...] = get_args(Status)
# no metrics on these: the cell says why there is nothing to measure or why the measurement is not a result
NO_METRICS = ("planned", "cannot", "skip")
# a code is the machine-readable half of the reason; planned and ok cells have none
CODED = ("fail", "cannot", "skip", "error")

_ABSOLUTE_PATH = re.compile(r"(?<![\w:/.])(/(?:[\w.+@=~-]+/)*[\w.+@=~-]+)")


def scrub(text: str) -> str:
    """`text` without where this machine keeps things: every absolute path shrinks to `.../<basename>`, the home
    directory and the working directory to nothing. Results files and baselines are shared, paths are not."""
    for root in (str(Path.home()), os.getcwd()):
        if len(root) > 1:
            text = text.replace(root + "/", "").replace(root, "")
    return _ABSOLUTE_PATH.sub(lambda m: ".../" + m.group(1).rsplit("/", 1)[-1], text)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SchemaError(ValueError):
    """A result file that cannot be read: wrong version, unknown field, malformed line."""


@dataclass
class Metric:
    """One measured quantity. `v` is the median of `samples` unless `unit` says otherwise (a peak, a count); `samples`
    are raw and in measurement order so that a rank or bootstrap test can run on the file alone."""

    v: float
    unit: str
    samples: list[float] = field(default_factory=list)

    @classmethod
    def of(cls, samples: Sequence[float], unit: str) -> Metric:
        """The metric whose value is the median of `samples`."""
        if not samples:
            raise ValueError("a metric needs at least one sample")
        return cls(statistics.median(samples), unit, list(samples))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"v": self.v, "unit": self.unit}
        if self.samples:
            out["samples"] = list(self.samples)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Metric:
        _only(d, {"v", "unit", "samples"}, "metric")
        return cls(d["v"], d["unit"], list(d.get("samples", ())))


@dataclass
class Cell:
    """One line of the result file. `key` is the baseline key, built only from what defines comparability (suite,
    model and format, engine spec with its opts, context rung, cache mode, workload id); nothing volatile."""

    key: str
    status: Status
    code: str | None = None  # cannot / skip / error / fail: machine code
    reason: str | None = None  # one line, paths scrubbed
    workload: dict[str, Any] = field(default_factory=dict)  # id, ids_sha, n_prompt, n_new, ignore_eos, ctx_max, forged
    metrics: dict[str, Metric] = field(default_factory=dict)  # empty unless measured
    clocks: dict[str, Any] = field(default_factory=dict)  # protocol.Sampler.summary()
    extra: dict[str, Any] = field(default_factory=dict)
    run: str | None = None
    unit: int | None = None  # index of the GPU process of the run that produced it
    t: str | None = None  # UTC time the cell was written
    arm: str | None = None  # "A" / "B" in an `ab` session
    pair: int | None = None  # `ab` pair index; 0 is the discarded warm-up pair

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise SchemaError(f"cell {self.key!r}: status {self.status!r} is not one of {', '.join(STATUSES)}")
        if self.status in NO_METRICS and self.metrics:
            raise SchemaError(f"cell {self.key!r}: a {self.status} cell carries no metrics")
        if self.status in CODED and self.code is None:
            raise SchemaError(f"cell {self.key!r}: a {self.status} cell needs a code")
        if self.reason is not None:
            self.reason = scrub(self.reason)

    def metric(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return None if m is None else m.v

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"schema": SCHEMA, "kind": "cell", "run": self.run, "unit": self.unit, "t": self.t}
        if self.arm is not None:
            d["arm"], d["pair"] = self.arm, self.pair
        d |= {"key": self.key, "status": self.status, "code": self.code, "reason": self.reason}
        d |= {"workload": self.workload, "metrics": {k: m.to_dict() for k, m in self.metrics.items()}}
        d |= {"clocks": self.clocks, "extra": self.extra}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Cell:
        _only(d, {"schema", "kind", "run", "unit", "t", "arm", "pair", "key", "status", "code", "reason", "workload",
                  "metrics", "clocks", "extra"}, "cell")  # fmt: skip
        return cls(
            key=d["key"],
            status=d["status"],
            code=d.get("code"),
            reason=d.get("reason"),
            workload=dict(d.get("workload", {})),
            metrics={k: Metric.from_dict(m) for k, m in d.get("metrics", {}).items()},
            clocks=dict(d.get("clocks", {})),
            extra=dict(d.get("extra", {})),
            run=d.get("run"),
            unit=d.get("unit"),
            t=d.get("t"),
            arm=d.get("arm"),
            pair=d.get("pair"),
        )


@dataclass
class Stamp:
    """What a run depends on, collected once by `protocol.stamp()`. No hostnames, home directories or mount points."""

    warpc: dict[str, Any] = field(default_factory=dict)  # commit, dirty, lock_sha256
    libswiftmk_sha256: str | None = None
    legacy_commit: str | None = None
    torch: str | None = None
    cuda: str | None = None
    driver: str | None = None
    python: str | None = None
    vllm: str | None = None
    llama_cpp: str | None = None
    transformers: str | None = None
    gpu: dict[str, Any] = field(default_factory=dict)  # name, power_limit_w, persistence, machine
    models: dict[str, dict[str, Any]] = field(default_factory=dict)  # name -> revision, fingerprint
    corpus_sha256: str | None = None
    tokenizer_sha256: str | None = None

    @property
    def machine(self) -> str | None:
        """The baseline file key of the GPU class, e.g. `rtx3090-350w`."""
        return self.gpu.get("machine")

    @property
    def dirty(self) -> bool:
        return bool(self.warpc.get("dirty", True))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Stamp:
        _only(d, {f.name for f in fields(cls)}, "stamp")
        return cls(**d)


@dataclass
class Header:
    run: str  # run id, `<session>-<4 hex>`
    session: str  # a session is the visits that were interleaved with each other; ratios only form inside one
    argv: list[str] = field(default_factory=list)
    stamp: Stamp = field(default_factory=Stamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "kind": "run",
            "run": self.run,
            "argv": list(self.argv),
            "session": self.session,
            "stamp": self.stamp.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Header:
        _only(d, {"schema", "kind", "run", "argv", "session", "stamp"}, "run header")
        return cls(d["run"], d["session"], list(d.get("argv", ())), Stamp.from_dict(d.get("stamp", {})))


@dataclass
class Run:
    """One header and the cells that follow it."""

    header: Header
    cells: list[Cell] = field(default_factory=list)


def _only(d: Mapping[str, Any], known: set[str], what: str) -> None:
    if extra := set(d) - known:
        raise SchemaError(f"{what}: unknown field(s) {', '.join(sorted(extra))}")


def dumps(item: Header | Cell) -> str:
    """One JSON line (no trailing newline). NaN and infinities are refused: a metric that is not a number is a bug."""
    try:
        return json.dumps(item.to_dict(), allow_nan=False)
    except ValueError as e:
        raise SchemaError(f"not representable: {e}") from e


def loads(line: str) -> Header | Cell:
    try:
        d = json.loads(line)
    except json.JSONDecodeError as e:
        raise SchemaError(f"not JSON: {e}") from e
    if not isinstance(d, dict) or d.get("kind") not in ("run", "cell"):
        raise SchemaError("a line is an object with kind 'run' or 'cell'")
    if d.get("schema") != SCHEMA:
        version = d.get("schema")
        newer = isinstance(version, int) and version > SCHEMA
        raise SchemaError(
            f"schema {version!r}: this code reads schema {SCHEMA}" + (" (newer: upgrade)" if newer else "")
        )
    return Header.from_dict(d) if d["kind"] == "run" else Cell.from_dict(d)


class Writer:
    """Appends lines to a result file, flushed per line: a unit killed by its lease leaves every cell it finished."""

    def __init__(self, path: Path, header: Header | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        if header is not None:
            self.write(header)

    def write(self, item: Header | Cell) -> None:
        if isinstance(item, Cell) and item.t is None:
            item.t = utc_now()
        self._file.write(dumps(item) + "\n")
        self._file.flush()

    def __enter__(self) -> Writer:
        return self

    def __exit__(self, *exc: object) -> None:
        self._file.close()


def write(path: Path, header: Header, cells: Iterable[Cell]) -> None:
    """A whole file: replaces `path`."""
    Path(path).unlink(missing_ok=True)
    with Writer(path, header) as w:
        for cell in cells:
            w.write(cell)


def read_runs(path: Path) -> list[Run]:
    """Every run of a file in order (a history file holds many). A cell before the first header is an error."""
    runs: list[Run] = []
    with Path(path).open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = loads(line)
            except SchemaError as e:
                raise SchemaError(f"{path.name}:{n}: {e}") from e
            if isinstance(item, Header):
                runs.append(Run(item))
            elif runs:
                runs[-1].cells.append(item)
            else:
                raise SchemaError(f"{path.name}:{n}: a cell before the run header")
    return runs


def read(path: Path) -> Run:
    """The single run of a result file."""
    runs = read_runs(path)
    if len(runs) != 1:
        raise SchemaError(f"{Path(path).name}: {len(runs)} runs, expected 1")
    return runs[0]


def iter_cells(runs: Iterable[Run]) -> Iterator[Cell]:
    for run in runs:
        yield from run.cells


KEY_SEPARATOR = "|"
ENGINE_SEGMENT = 2


def make_key(suite: str, model: str, engine: str, *parts: str) -> str:
    """`suite|model|engine-spec|part|...`: the engine spec is a segment of its own so that `cell_id` can name the
    cell without it, which is what two arms of an `ab` with different engine specs share."""
    for segment in (suite, model, engine, *parts):
        if KEY_SEPARATOR in segment:
            raise ValueError(f"key segment {segment!r} contains {KEY_SEPARATOR!r}")
    return KEY_SEPARATOR.join((suite, model, engine, *parts))


def cell_id(key: str) -> str:
    """The key without its engine spec: the cell two arms measure."""
    segments = key.split(KEY_SEPARATOR)
    del segments[ENGINE_SEGMENT]
    return KEY_SEPARATOR.join(segments)


def key_engine(key: str) -> str:
    """The engine spec segment of a key (`warpc:cuda,kv=bf16,head=bf16`)."""
    return key.split(KEY_SEPARATOR)[ENGINE_SEGMENT]


def key_engine_name(key: str) -> str:
    """The engine of a key: `warpc` for `warpc:cuda,kv=bf16`."""
    return re.split(r"[:,]", key_engine(key), maxsplit=1)[0]
