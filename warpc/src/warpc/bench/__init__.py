"""warpc's benchmark, quality and regression suite: `python -m warpc.bench --help`.

Three functions and the data they return: `plan` (the dry run), `run` (a plan, or two arms interleaved), `report`.
`warpc/__init__.py` never imports this package; vllm, transformers and huggingface_hub are imported only inside the
adapters and verbs that need them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from . import gates as gate_module
from . import protocol, schema, suites, units
from .gates import Baseline, Gate, Verdict
from .schema import Cell, Header, Metric
from .units import Arm, Plan, Unit

__all__ = [
    "Arm",
    "Baseline",
    "Cell",
    "Gate",
    "Header",
    "Metric",
    "Plan",
    "Result",
    "Unit",
    "Verdict",
    "plan",
    "report",
    "run",
]

PAIRS = 9  # the bootstrap interval's floor (gates.MIN_PAIRS_CI)
VISITS = 3  # the stored baseline's floor (gates.MIN_VISITS_BASELINE)
BASELINES = Path(__file__).resolve().parent / "baselines"


@dataclass
class Result:
    cells: list[Cell]
    path: Path
    header: Header = field(default_factory=lambda: Header("", ""))

    def judge(
        self,
        reference: Baseline | Iterable[Cell] | Path | Literal["baseline"] | None = None,
        *,
        gates: Mapping[str, Gate] | None = None,
        required: Iterable[str] | None = None,
    ) -> Verdict:
        """`reference`: a `Baseline`, the cells of another run, the path of either file, or "baseline" for the
        committed file of this GPU class (none yet: the cells are `new`)."""

        if reference == "baseline":
            reference = load_baseline(self.header.stamp.machine)
        elif isinstance(reference, Path):
            reference = (
                Baseline.load(reference)
                if reference.suffix == ".json"
                else list(schema.iter_cells(schema.read_runs(reference)))
            )
        return gate_module.judge(self.cells, reference, gates=gates, stamp=self.header.stamp, required=required)


def load_baseline(machine: str | None) -> Baseline | None:
    """The committed baseline of a GPU class (`baselines/<machine>.json`), or None when there is none."""
    path = BASELINES / f"{machine}.json" if machine else None
    return Baseline.load(path) if path and path.is_file() else None


def plan(suite: str, **select) -> Plan:
    """The dry run: units with their estimated seconds, the cells decided `cannot` / `skip` on the CPU with reasons.
    No weights are read and no GPU process is started (`select`: the fields of `suites.Selection`)."""
    if suite not in suites.SUITES:
        raise ValueError(f"unknown suite {suite!r} (known: {', '.join(suites.SUITES)})")
    return suites.SUITES[suite].plan(suites.Selection(**select))


def _header(argv: Sequence[str], run_id: str, session: str, planned: Plan) -> Header:
    engines = {u.engine.name for u in planned.units if u.engine}
    models = {
        suites.model_label(u.engine.model): u.engine.model
        for u in planned.units
        if u.engine and u.engine.model.exists()
    }
    return Header(run_id, session, list(argv), protocol.stamp(engines=engines or ("warpc",), models=models))


def run(
    what: Plan | tuple[Arm, Arm],
    *,
    suite: str = "speed",
    select: suites.Selection | None = None,
    out: Path | None = None,
    wrap: Sequence[str] = (),
    pairs: int | None = None,
    visits: int | None = None,
    argv: Sequence[str] = (),
) -> Result:
    """Run a plan (`visits` processes per unit, default 3) or two arms interleaved (`pairs` measured pairs after a
    discarded warm-up pair, default 9); the cells are written to `out` as they arrive. `select`: with arms, the
    selection of the suite (models, contexts) shared by both; default the arms' model at the suite's own contexts."""
    run_id, session = units.new_run_id()
    if isinstance(what, Plan):
        planned, arms = what, None
    else:
        arms = {"A": what[0], "B": what[1]}
        planned = _plan_arm(suite, select, what[0])
    if out is None:
        from warpc.cuda import glue

        out = glue.cache_dir() / "bench" / f"{run_id}.jsonl"
    header = _header(argv, run_id, session, planned)
    out = Path(out)
    out.unlink(missing_ok=True)
    writer = schema.Writer(out, header)
    with writer:
        if arms is None:
            cells = units.run_plan(
                what, run=run_id, visits=visits or VISITS, wrap=wrap, on_cells=lambda c: [writer.write(x) for x in c]
            )
        else:
            cells = units.run_ab(
                arms,
                lambda arm: _plan_arm(suite, select, arm).units,
                run=run_id,
                pairs=pairs or PAIRS,
                wrap=wrap,
                on_cells=lambda c: [writer.write(x) for x in c],
            )
    return Result(cells, out, header)


def _plan_arm(suite: str, select: suites.Selection | None, arm: Arm) -> Plan:
    base = select or suites.Selection(model=[str(arm.spec.model)])
    return suites.SUITES[suite].plan(replace(base, engine=[arm.spec.text]))


def report(
    cells: Iterable[Cell], fmt: Literal["table", "markdown", "json"] = "table", header: Header | None = None
) -> str:
    """The cells as a table, the markdown block of a pull request (needs `header`) or JSON."""
    from . import render as rendering

    cells = list(cells)
    if fmt == "table":
        return rendering.table(cells)
    if header is None:
        raise ValueError(f"report format {fmt!r} needs the run header")
    return rendering.markdown(header, cells) if fmt == "markdown" else rendering.as_json(header, cells)
