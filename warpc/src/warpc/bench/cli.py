"""The command line: a thin argparse over `plan`, `run`, `Result.judge` and `report`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from . import PAIRS, VISITS, Arm, Result, gates, load_baseline, plan, protocol, report, run, schema, suites, units
from .engines import EngineSpec

EXAMPLES = """\
  python -m warpc.bench plan speed --model Qwen3-0.6B --fmt Q4_0,bf16
  python -m warpc.bench run speed --model Qwen3-0.6B --fmt Q4_0 [--wrap lease-command ...]
  python -m warpc.bench ab speed --model Qwen3-0.6B --fmt Q4_0 --a warpc:cuda --b warpc:cuda@HEAD~1
  python -m warpc.bench compare NEW.jsonl [REF.jsonl | baseline.json]
  python -m warpc.bench report FILE... [--markdown | --json]
  python -m warpc.bench explain metrics | gates | suites
exit status: 0 pass, 1 a gate failed or a regression is confirmed, 3 incomplete (a required cell skipped, underpowered,
suspect or inconclusive); 3 is never a pass"""


def _list(values: Sequence[str] | None) -> list[str] | None:
    return None if values is None else [x for v in values for x in v.split(",") if x]


def _selection(args: argparse.Namespace) -> suites.Selection:
    sel = suites.Selection(model=_list(args.model) or [], n_new=args.n_new, repeats=args.repeats)
    if args.fmt:
        sel.fmt = _list(args.fmt)
    if args.engine:
        sel.engine = _list(args.engine)
    sel.kv = _list(args.kv)
    sel.ctx = None if args.ctx is None else [int(x) for x in _list(args.ctx)]
    sel.prompt_len = None if args.prompt_len is None else [int(x) for x in _list(args.prompt_len)]
    return sel


def _add_selectors(p: argparse.ArgumentParser) -> None:
    p.add_argument("suite", choices=list(suites.SUITES))
    p.add_argument(
        "--model", nargs="+", help="a checkpoint path, or a name under $WARPC_MODELS (qwen35-27b: $WARPC_TEST_SWIFT27)"
    )
    p.add_argument("--fmt", nargs="+", help="bf16 | Q4_0 | Q8_0 (default bf16)")
    p.add_argument("--engine", nargs="+", help="engine spec: warpc:cuda,kv=int4_rot,head=bf16 (default warpc)")
    p.add_argument("--kv", nargs="+", help="KV cache format(s), overriding the engine spec's")
    p.add_argument("--ctx", nargs="+", help="ctx_max rung(s) of the 5-token prompt cells (default 512 4096)")
    p.add_argument("--prompt-len", nargs="+", help="corpus prompt length(s); ctx_max = round_up_256(length + 512)")
    p.add_argument("--n-new", type=int, default=suites.N_NEW, help="greedy tokens per round")
    p.add_argument("--repeats", type=int, default=protocol.TIMED_ROUNDS, help="timed rounds per visit")


def _add_wrap(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, help="result file (default: the bench directory of $WARPC_CACHE_DIR)")
    p.add_argument(
        "--wrap", nargs="*", default=[], help="a command every GPU process runs under (a lease); last option"
    )


def _arm(text: str, model: Path) -> Arm:
    spec, _, ref = text.partition("@")
    return Arm(EngineSpec.parse(spec, model), ref or None)


def _finish(result: Result, verdict: gates.Verdict, *, markdown: bool = False) -> int:
    print(report(result.cells, "markdown" if markdown else "table", result.header))
    print()
    print(verdict.render())
    print(f"results: {result.path.name}")
    return verdict.exit_code


def cmd_plan(args: argparse.Namespace) -> int:
    print(plan(args.suite, **vars(_selection(args))).render())
    return 0


def _argv(args: argparse.Namespace) -> list[str]:
    return sys.argv[1:] if not hasattr(args, "_argv") else args._argv


def cmd_run(args: argparse.Namespace) -> int:
    planned = plan(args.suite, **vars(_selection(args)))
    result = run(planned, out=args.out, wrap=args.wrap, visits=args.visits, argv=_argv(args))
    reference = args.baseline if args.baseline else "baseline"
    verdict = result.judge(reference, required=[c.key for c in (*planned.cells, *planned.decided)])
    if args.confirm:
        verdict = _confirm(args, planned, result, verdict, reference)
    return _finish(result, verdict)


def _previous_drifts(path: Path, key: str) -> list[bool]:
    """Whether each earlier run file next to `path` that measured `key` ended in a drift, oldest first."""
    earlier = sorted(
        (p for p in path.parent.glob("*.jsonl") if p != path and not p.name.endswith(".confirm.jsonl")),
        key=lambda p: p.stat().st_mtime,
    )
    out = []
    for p in earlier:
        if p.stat().st_mtime > path.stat().st_mtime:
            continue
        try:
            cells = [c for c in schema.read(p).cells if c.key == key and c.arm is None]
        except (schema.SchemaError, OSError):
            continue
        if cells:
            out.append(any(c.extra.get("drift") is not None for c in cells))
    return out


def _confirm(
    args: argparse.Namespace, planned: units.Plan, result: Result, verdict: gates.Verdict, reference
) -> gates.Verdict:
    """Re-run only the `suspect` cells as a paired `ab` of the baseline's commit against HEAD, in one session, and put
    the end state in place of `suspect` (`gates.resolve_confirm`)."""
    suspects = [f for f in verdict.findings if f.state == gates.SUSPECT]
    if not suspects:
        return verdict
    baseline = (
        gates.Baseline.load(reference) if isinstance(reference, Path) else load_baseline(result.header.stamp.machine)
    )
    if baseline is None or not baseline.origin:
        verdict.notes.append("--confirm: the baseline names no commit to compare against")
        return verdict
    keys = {f.key for f in suspects}
    todo = units.Plan([u for u in planned.units if u.cells[0].key in keys])
    arms = {"A": Arm(todo.units[0].engine, baseline.origin), "B": Arm(todo.units[0].engine, None)}
    run_id, session = units.new_run_id()
    cells = units.run_ab(arms, lambda arm: todo.units, run=run_id, pairs=args.pairs or PAIRS, wrap=args.wrap)
    confirm_path = result.path.with_suffix(".confirm.jsonl")
    confirm_path.unlink(missing_ok=True)
    schema.write(confirm_path, schema.Header(run_id, session, _argv(args), result.header.stamp), cells)
    paired = gates.judge(cells, stamp=result.header.stamp)
    by = {(f.key, f.metric): f for f in paired.findings}
    resolved = []
    for f in verdict.findings:
        if f.state == gates.SUSPECT:
            ab = by.get((schema.cell_id(f.key), f.metric))
            f = gates.resolve_confirm(f, ab, baseline_commit=baseline.origin) if ab else f
        resolved.append(f)
    drifts = {f.key: f.drift for f in resolved if f.drift is not None}
    notes = [*verdict.notes, *paired.notes, f"--confirm: the paired session is {confirm_path.name}"]
    for key in drifts:
        if gates.rebaseline_recommended([*_previous_drifts(result.path, key), True]):
            notes.append(f"REBASELINE RECOMMENDED: {key} ended in pass(drift) in two consecutive runs")
    if drifts:  # the run file records `drift = X` on the cells it concerns
        for c in result.cells:
            if c.key in drifts:
                c.extra["drift"] = drifts[c.key]
        schema.write(result.path, result.header, result.cells)
    return gates.Verdict(resolved, notes)


def cmd_ab(args: argparse.Namespace) -> int:
    sel = _selection(args)
    planned = plan(args.suite, **vars(sel))
    if not planned.units:
        print(planned.render())
        return 3
    model = planned.units[0].engine.model
    arms = (_arm(args.a, model), _arm(args.b, model))
    result = run(arms, suite=args.suite, select=sel, out=args.out, wrap=args.wrap, pairs=args.pairs, argv=_argv(args))
    verdict = result.judge(None, required=[c.key for c in (*planned.cells, *planned.decided)])
    return _finish(result, verdict)


def cmd_compare(args: argparse.Namespace) -> int:
    run_ = schema.read(args.new)
    result = Result(run_.cells, args.new, run_.header)
    reference = args.ref if args.ref else "baseline"
    verdict = result.judge(reference)
    print(verdict.render())
    return verdict.exit_code


def cmd_report(args: argparse.Namespace) -> int:
    for path in args.files:
        run_ = schema.read(path)
        fmt = "markdown" if args.markdown else "json" if args.json else "table"
        print(report(run_.cells, fmt, run_.header))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    if args.what == "metrics":
        print(protocol.__doc__)
    elif args.what == "suites":
        print(suites.table())
    else:
        print(gates.__doc__)
        print(suites.gate_table())
    return 0


def cmd_unit(args: argparse.Namespace) -> int:
    return units.child_main(args.spec)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m warpc.bench",
        description="warpc's benchmark, quality and regression suite",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="verb", required=True)

    p = sub.add_parser("plan", help="the dry run: units, estimated minutes, cells decided cannot / skip, workload shas")
    _add_selectors(p)
    p.set_defaults(fn=cmd_plan)

    p = sub.add_parser("run", help="run the plan's units, judge, set the exit code")
    _add_selectors(p)
    p.add_argument("--visits", type=int, default=VISITS, help="processes per unit (the baseline tripwire needs 3)")
    p.add_argument("--baseline", type=Path, help="a baseline file (default: the committed one of this GPU class)")
    p.add_argument(
        "--confirm", action="store_true", help="re-run suspect cells as a paired ab against the baseline's commit"
    )
    p.add_argument("--pairs", type=int, help=f"pairs of the --confirm ab (default {PAIRS})")
    _add_wrap(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ab", help="interleaved A B | B A | A B ...: ARM is SPEC[@GIT_REF], e.g. warpc:cuda@HEAD~1")
    _add_selectors(p)
    p.add_argument("--a", required=True, metavar="ARM")
    p.add_argument("--b", required=True, metavar="ARM")
    p.add_argument("--pairs", type=int, help=f"measured pairs after the discarded warm-up pair (default {PAIRS})")
    _add_wrap(p)
    p.set_defaults(fn=cmd_ab)

    p = sub.add_parser("compare", help="re-judge a stored result file against another or a baseline")
    p.add_argument("new", type=Path)
    p.add_argument("ref", nargs="?", type=Path)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("report", help="print result files")
    p.add_argument("files", nargs="+", type=Path)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--markdown", action="store_true")
    group.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("explain", help="print the metric definitions, the gate table or the suite table")
    p.add_argument("what", choices=("metrics", "gates", "suites"))
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("unit", help="one GPU process of a run (started by run and ab, not by hand)")
    p.add_argument("spec", type=Path)
    p.set_defaults(fn=cmd_unit)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if argv is not None:
        args._argv = list(argv)
    return args.fn(args)
