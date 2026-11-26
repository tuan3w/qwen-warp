"""Gates and verdicts: how a measured cell becomes pass, regressed or incomplete.

Two comparisons exist. A PAIRED one (`ab`: visits of two arms taken alternately in one session) resolves 1-2%: the
pair ratio `B / A` per adjacent pair, a bootstrap interval of its median from nine kept pairs, an exact sign test from
five. A STORED one (a run against the committed baseline) can only be a tripwire at ~2x the cross-session noise; a
tripped wire is `suspect` until `--confirm` re-runs just that cell as a paired `ab` against the baseline's commit.

Exit codes: 1 when a gate failed or a regression is confirmed, 3 when the run is incomplete (a required cell skipped,
`underpowered`, `suspect`, `inconclusive`), 0 otherwise. 3 is never a pass. `cannot` cells are results, not
incompleteness. There is no waiver: an accepted regression is a reviewed baseline update.

Power, stated once: at ~0.4% noise per pair a true 2.5% slowdown is detected at n = 9 with probability ~1 and a 2%
rule has essentially no A/A false positives; a true 1.5% slowdown is only ever `regressed(minor)`, which does not block.
The gate does not block a 1-2% slowdown; the committed baseline (a 3% tripwire only a reviewed commit moves) is what
catches those that accumulate.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from . import protocol, stats
from .schema import Cell, Stamp, cell_id, key_engine_name

PASS = "pass"
NEW = "new"
INSIDE_NOISE = "inside_noise"
IMPROVED = "improved"
REGRESSED = "regressed"
MINOR = "regressed(minor)"
PASS_DRIFT = "pass(drift)"
UNDERPOWERED = "underpowered"
UNPAIRED = "unpaired"
SUSPECT = "suspect"
INCONCLUSIVE_CLOCK = "inconclusive(clock)"
INCONCLUSIVE_ENV = "inconclusive(env)"
WORKLOAD_CHANGED = "workload changed"
FAIL = "fail"
ERROR = "error"
SKIP = "skip"
CANNOT = "cannot"

BLOCKING = {REGRESSED, FAIL, ERROR}
INCOMPLETE = {UNDERPOWERED, UNPAIRED, SUSPECT, INCONCLUSIVE_CLOCK, INCONCLUSIVE_ENV, WORKLOAD_CHANGED, SKIP}

MIN_PAIRS_CI = 9  # Google Benchmark compare.py's UTEST_OPTIMAL_REPETITIONS: the bootstrap interval starts here
MIN_PAIRS_SIGN = 5  # one-sided p = 1/32 when all five pairs fall on one side
MIN_VISITS_BASELINE = 3
ENV_FIELDS = ("torch", "cuda", "driver", "vllm", "llama_cpp", "libswiftmk_sha256")  # what a speed number depends on


@dataclass(frozen=True)
class Gate:
    """The rule of one metric. `paired`: a ratio of visits (`t_fail`, `t_warn` as fractions) and, against a stored
    baseline, the `tripwire` fraction. `absolute`: worse than the baseline by more than `t_fail` (units) fails, more
    than `t_warn` is minor. `ceiling`: fails above `mult * baseline + slack`. `floor`: fails below `baseline - slack`.
    `exact`: any difference fails (identity gates: no tolerance, no baseline, no waiver)."""

    metric: str
    kind: Literal["paired", "absolute", "ceiling", "floor", "exact"]
    higher_is_better: bool = True
    t_fail: float = 0.0
    t_warn: float = 0.0
    tripwire: float = 0.03
    mult: float = 1.0
    slack: float = 0.0
    basis: str = ""


GATES: dict[str, Gate] = {
    g.metric: g
    for g in (
        Gate("decode_tok_s", "paired", True, 0.02, 0.01, basis="5x same-session noise, 1.5-2x cross-session"),
        Gate("ttft_s", "paired", False, 0.02, 0.01, basis="as decode_tok_s"),
        Gate("decode_tok_s_median", "paired", True, 0.03, 0.02, basis="agent turns: spill/restore I/O adds variance"),
        Gate("ttft_median_s", "paired", False, 0.05, 0.02, basis="agent turns, calibrate in step 2"),
        Gate("device_gib_peak", "absolute", False, 0.25, 0.10, basis="~1% of the card, near-deterministic"),
        Gate("host_gib", "absolute", False, 0.25, 0.10, basis="pinned store bytes"),
        Gate("kl_mean", "ceiling", False, mult=1.25, slack=5e-5, basis="2x the bf16-vs-fp16 floor, +25%"),
        Gate("top1", "floor", True, slack=0.005, basis="0.5 pp"),
        Gate("probes_ok", "floor", True, slack=0.0, basis="deterministic: any lost probe is signal"),
        Gate("resumed_equal", "exact", True, basis="a resumed turn is bit-equal to a cold prefill"),
        Gate("ids_equal", "exact", True, basis="bit identity: no tolerance"),
    )
}


@dataclass
class Finding:
    """The verdict on one metric of one cell (or on a cell: `metric` None)."""

    key: str
    metric: str | None
    state: str
    detail: str = ""
    ratio: float | None = None  # B / A (paired) or new / baseline, read so that above 1 is better
    ci: tuple[float, float] | None = None
    n: int | None = None  # pairs kept (paired) or visits (stored)
    dropped: dict[str, int] = field(default_factory=dict)
    required: bool = True
    drift: float | None = None

    @property
    def blocking(self) -> bool:
        return self.state in BLOCKING

    @property
    def incomplete(self) -> bool:
        return self.required and self.state in INCOMPLETE


@dataclass
class Verdict:
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if any(f.blocking for f in self.findings):
            return 1
        return 3 if any(f.incomplete for f in self.findings) else 0

    @property
    def state(self) -> str:
        """The worst state of the run: a blocking one, else an incomplete one, else `pass`."""
        for pick in (lambda f: f.blocking, lambda f: f.incomplete):
            for f in self.findings:
                if pick(f):
                    return f.state
        return PASS

    def render(self) -> str:
        lines = []
        for f in self.findings:
            what = f.key if f.metric is None else f"{f.key} {f.metric}"
            lines.append(f"{f.state:<20} {what}" + (f": {f.detail}" if f.detail else ""))
        lines += self.notes
        lines.append(f"verdict: {self.state} (exit {self.exit_code})")
        return "\n".join(lines)


# ---- the ratio rule


def judge_ratios(
    ratios: Sequence[float], gate: Gate, *, dropped: Mapping[str, int] | None = None
) -> tuple[str, dict[str, Any]]:
    """The state of a set of pair ratios (B over A, above 1 better) under `gate`, and what it was decided from."""
    n = len(ratios)
    info: dict[str, Any] = {"n": n, "dropped": dict(dropped or {})}
    if n < MIN_PAIRS_SIGN:
        info["min"], info["max"] = (min(ratios), max(ratios)) if ratios else (None, None)
        return UNDERPOWERED, info
    centre = stats.median(ratios)
    info["ratio"] = centre
    below, above = sum(r < 1 for r in ratios), sum(r > 1 for r in ratios)
    if n >= MIN_PAIRS_CI:
        lo, hi = stats.bootstrap_ci(ratios)
        info["ci"] = (lo, hi)
        worse = lambda t: centre < 1 - t and hi < 1  # noqa: E731
        better = lambda t: centre > 1 + t and lo > 1  # noqa: E731
    else:
        info["p"] = stats.sign_test(n, n)
        worse = lambda t: below == n and centre < 1 - t  # noqa: E731
        better = lambda t: above == n and centre > 1 + t  # noqa: E731
    if worse(gate.t_fail):
        return REGRESSED, info
    if better(gate.t_fail):
        return IMPROVED, info
    if worse(gate.t_warn):
        return MINOR, info
    return INSIDE_NOISE, info


def _describe(state: str, info: Mapping[str, Any]) -> str:
    parts = []
    if "ratio" in info:
        parts.append(f"median B/A {info['ratio']:.4f}")
    if "ci" in info:
        parts.append(f"95% CI [{info['ci'][0]:.4f}, {info['ci'][1]:.4f}]")
    if "p" in info and state != INSIDE_NOISE:
        parts.append(f"sign test p = {info['p']:.4f}")
    parts.append(f"n = {info['n']} pairs")
    if info.get("dropped"):
        parts.append("dropped " + ", ".join(f"{k} {v}" for k, v in sorted(info["dropped"].items())))
    if state == UNDERPOWERED and info.get("min") is not None:
        parts.append(f"min..max {info['min']:.4f}..{info['max']:.4f}")
    if state == MINOR:
        parts.append("a 1-2% slowdown: reported, not blocking")
    return ", ".join(parts)


# ---- pair hygiene


def contended(cell: Cell) -> bool:
    return bool(cell.clocks.get("foreign_pids")) or bool(cell.extra.get("contended"))


def _sm(cell: Cell) -> float | None:
    sm = cell.clocks.get("sm_mhz")
    return sm[1] if sm else None


@dataclass
class Pair:
    a: Cell
    b: Cell
    index: int


def pairs_of(cells: Iterable[Cell]) -> dict[str, list[Pair]]:
    """The adjacent visits of an `ab` session by cell id, pair 0 (the discarded warm-up pair) left out."""
    arms: dict[str, dict[int, dict[str, Cell]]] = defaultdict(lambda: defaultdict(dict))
    for c in cells:
        if c.arm is not None and c.pair:
            arms[cell_id(c.key)][c.pair][c.arm] = c
    return {
        cid: [Pair(v["A"], v["B"], i) for i, v in sorted(by_pair.items()) if "A" in v and "B" in v]
        for cid, by_pair in arms.items()
    }


def judge_pairs(
    pairs: Sequence[Pair], gate: Gate, *, clock_bound: float = protocol.CLOCK_BOUND_SAME_ENGINE
) -> tuple[str, dict[str, Any]]:
    """Hygiene, then the ratio rule. A pair is dropped when a visit was not measured (`visit`), a foreign process was
    on the GPU (`contended`), a same-engine pair's median SM clocks differ by more than `clock_bound` (`clock`), or both
    visits are outliers of their arm (`spike`)."""
    dropped: dict[str, int] = defaultdict(int)
    kept: list[Pair] = []
    for p in pairs:
        if p.a.metric(gate.metric) is None or p.b.metric(gate.metric) is None:
            dropped["visit"] += 1
        elif contended(p.a) or contended(p.b):
            dropped["contended"] += 1
        elif key_engine_name(p.a.key) == key_engine_name(p.b.key) and _clock_apart(p, clock_bound):
            dropped["clock"] += 1
        else:
            kept.append(p)
    if len(kept) >= 3:
        spike_a = stats.mad_outliers([p.a.metric(gate.metric) for p in kept])
        spike_b = stats.mad_outliers([p.b.metric(gate.metric) for p in kept])
        both = [x and y for x, y in zip(spike_a, spike_b, strict=True)]
        if spikes := sum(both):
            dropped["spike"] = spikes
        kept = [p for p, drop in zip(kept, both, strict=True) if not drop]
    ratios = stats.paired_ratios(
        [p.a.metric(gate.metric) for p in kept],
        [p.b.metric(gate.metric) for p in kept],
        higher_is_better=gate.higher_is_better,
    )
    return judge_ratios(ratios, gate, dropped=dropped)


def _clock_apart(p: Pair, bound: float) -> bool:
    a, b = _sm(p.a), _sm(p.b)
    return a is not None and b is not None and abs(b - a) / a > bound


# ---- stored baselines


@dataclass
class Baseline:
    """Per gated cell `{metric: median}`, the quartiles, visits, median SM clock and the workload, under the stamp
    subset the numbers depend on. No raw samples: the file is a reviewable diff, written only by `baseline update`."""

    machine: str | None = None
    stamp: dict[str, Any] = field(default_factory=dict)
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)
    reason: str | None = None
    origin: str | None = None  # the commit the numbers were taken at

    @classmethod
    def from_cells(cls, cells: Iterable[Cell], stamp: Stamp | None = None) -> Baseline:
        by_key: dict[str, list[Cell]] = defaultdict(list)
        for c in cells:
            if c.status == "ok" and not contended(c):
                by_key[c.key].append(c)
        rows = {}
        for key, visits in by_key.items():
            row: dict[str, Any] = {"median": {}, "iqr": {}, "n": len(visits), "workload": visits[0].workload}
            for name in sorted({m for c in visits for m in c.metrics}):
                values = [c.metric(name) for c in visits if c.metric(name) is not None]
                row["median"][name] = statistics.median(values)
                row["iqr"][name] = [stats.percentile(values, 0.25), stats.percentile(values, 0.75)]
            sm = [s for c in visits if (s := _sm(c)) is not None]
            row["sm_mhz"] = statistics.median(sm) if sm else None
            rows[key] = row
        s = stamp.to_dict() if stamp else {}
        return cls(stamp.machine if stamp else None, {k: s.get(k) for k in ("warpc", *ENV_FIELDS)}, rows,
                   origin=(stamp.warpc.get("commit") if stamp else None))  # fmt: skip

    @classmethod
    def load(cls, path: Path) -> Baseline:
        d = json.loads(Path(path).read_text())
        return cls(d.get("machine"), d.get("stamp", {}), d.get("rows", {}), d.get("reason"), d.get("from"))

    def dumps(self) -> str:
        return json.dumps(
            {
                "machine": self.machine,
                "stamp": self.stamp,
                "reason": self.reason,
                "from": self.origin,
                "rows": self.rows,
            },
            indent=1,
            sort_keys=True,
        )


def _env_differs(stamp: Stamp | None, baseline: Baseline) -> list[str]:
    if stamp is None:
        return []
    have = stamp.to_dict()
    return [f for f in ENV_FIELDS if baseline.stamp.get(f) is not None and have.get(f) != baseline.stamp[f]]


def judge_stored(
    key: str,
    visits: Sequence[Cell],
    baseline: Baseline,
    gates: Mapping[str, Gate],
    stamp: Stamp | None = None,
    clock_bound: float = protocol.CLOCK_BOUND_CROSS_SESSION,
) -> list[Finding]:
    """The measured visits of one key against its baseline row."""
    row = baseline.rows.get(key)
    if row is None:
        return [Finding(key, None, NEW, "no baseline row: report only until `baseline update`")]
    if row.get("workload") != visits[0].workload:
        return [Finding(key, None, WORKLOAD_CHANGED, "the workload differs from the baseline's: re-baseline")]
    out = []
    env = _env_differs(stamp, baseline)
    for name, gate in gates.items():
        values = [v.metric(name) for v in visits if v.metric(name) is not None]
        reference = row["median"].get(name)
        if not values or reference is None:
            continue
        new = statistics.median(values)
        if gate.kind == "paired":
            out.append(_tripwire(key, name, gate, values, new, reference, row, visits, env, clock_bound))
        else:
            out.append(_deterministic(key, name, gate, new, reference))
    return out


def _tripwire(key, name, gate, values, new, reference, row, visits, env, clock_bound) -> Finding:
    ratio = new / reference if gate.higher_is_better else reference / new
    base = {"ratio": ratio, "n": len(values)}
    sm = [s for v in visits if (s := _sm(v)) is not None]
    if env:
        return Finding(key, name, INCONCLUSIVE_ENV, f"differs from the baseline's: {', '.join(env)}", **base)
    if sm and row.get("sm_mhz") and abs(statistics.median(sm) - row["sm_mhz"]) / row["sm_mhz"] > clock_bound:
        return Finding(key, name, INCONCLUSIVE_CLOCK, "median SM clock apart from the baseline's", **base)
    if len(values) < MIN_VISITS_BASELINE:
        return Finding(key, name, UNDERPOWERED, f"{len(values)} visit(s), {MIN_VISITS_BASELINE} needed", **base)
    if ratio < 1 - gate.tripwire:
        detail = f"{new:.4g} against the baseline's {reference:.4g}: {100 * (1 - ratio):.1f}% worse; run with --confirm"
        return Finding(key, name, SUSPECT, detail, **base)
    return Finding(key, name, PASS, f"{100 * (ratio - 1):+.1f}% against the baseline", **base)


def _deterministic(key: str, name: str, gate: Gate, new: float, reference: float) -> Finding:
    worse_is_higher = not gate.higher_is_better
    if gate.kind == "exact":
        ok, rule = new == reference, "differs"
    elif gate.kind == "floor":
        ok, rule = new >= reference - gate.slack, f"below {reference - gate.slack:.6g}"
    elif gate.kind == "ceiling":
        ok, rule = new <= gate.mult * reference + gate.slack, f"above {gate.mult * reference + gate.slack:.6g}"
    else:  # absolute
        delta = new - reference if worse_is_higher else reference - new
        if delta > gate.t_fail:
            return Finding(key, name, FAIL, f"{new:.4g} against the baseline's {reference:.4g}")
        state = MINOR if delta > gate.t_warn else PASS
        return Finding(key, name, state, f"{new:.4g} against the baseline's {reference:.4g}")
    return Finding(key, name, PASS if ok else FAIL, f"{new:.6g} {'holds' if ok else rule} (baseline {reference:.6g})")


# ---- the whole judgement


def judge(
    cells: Iterable[Cell],
    reference: Baseline | Iterable[Cell] | None = None,
    *,
    gates: Mapping[str, Gate] | None = None,
    stamp: Stamp | None = None,
    required: Iterable[str] | None = None,
    clock_bound: float = protocol.CLOCK_BOUND_SAME_ENGINE,
) -> Verdict:
    """The verdict of a result. Cells of an `ab` session (they carry `arm` and `pair`) are judged by pairs; the others
    are judged against `reference` (a baseline, or the cells of another run) or are `new`. `required`: the cell keys
    the profile lists (default: every cell given): only a required cell can make the run incomplete."""
    gates = GATES if gates is None else gates
    cells = list(cells)
    wanted = None if required is None else set(required)
    verdict = Verdict()
    ab = [c for c in cells if c.arm is not None]
    for c in cells:
        if c.status in (FAIL, ERROR, SKIP, CANNOT):
            verdict.findings.append(Finding(c.key, None, c.status, f"{c.code}: {c.reason}"))
        elif c.status == "planned":
            verdict.findings.append(Finding(c.key, None, ERROR, "planned, never measured"))
    ok = [c for c in cells if c.status == "ok"]
    if ab:
        _judge_ab(verdict, [c for c in ok if c.arm is not None], gates, clock_bound)
    if baseline_cells := [c for c in ok if c.arm is None]:
        if reference is not None and not isinstance(reference, Baseline):
            reference = Baseline.from_cells(reference)
        by_key: dict[str, list[Cell]] = defaultdict(list)
        for c in baseline_cells:
            by_key[c.key].append(c)
        for key, all_visits in by_key.items():
            visits = [c for c in all_visits if not contended(c)]
            if not visits:
                verdict.findings.append(
                    Finding(key, None, SKIP, "every visit contended: a foreign process was on the GPU")
                )
            elif reference is None:
                verdict.findings.append(Finding(key, None, NEW, f"{len(visits)} visit(s); no baseline to compare"))
            else:
                verdict.findings.extend(judge_stored(key, visits, reference, gates, stamp))
    if wanted is not None:
        wanted_ids = {cell_id(k) for k in wanted}
        for f in verdict.findings:
            f.required = f.key in wanted or f.key in wanted_ids  # an `ab` finding is keyed by the cell id
        measured = {f.key for f in verdict.findings}
        for key in sorted(k for k in wanted if k not in measured and cell_id(k) not in measured):
            verdict.findings.append(Finding(key, None, ERROR, "required but absent from the result"))
    return verdict


def _judge_ab(verdict: Verdict, ok: list[Cell], gates: Mapping[str, Gate], clock_bound: float) -> None:
    for cid, pairs in pairs_of(ok).items():
        metrics = sorted({m for p in pairs for m in p.a.metrics if m in gates and gates[m].kind == "paired"})
        for name in metrics:
            state, info = judge_pairs(pairs, gates[name], clock_bound=clock_bound)
            verdict.findings.append(
                Finding(
                    cid,
                    name,
                    state,
                    _describe(state, info),
                    ratio=info.get("ratio"),
                    ci=info.get("ci"),
                    n=info["n"],
                    dropped=info["dropped"],
                )
            )


# ---- --confirm


def resolve_confirm(suspect: Finding, ab: Finding, *, baseline_commit: str | None = None) -> Finding:
    """The end state of a `suspect` cell after its paired re-run (A = the baseline's commit, B = HEAD):

    the pair says `regressed`        -> `regressed`, "slower by X% cumulative since <commit>" (several small steps
                                        surface on the pull request that crosses the tripwire; `git bisect` with `ab`
                                        finds the step)
    the pair says noise or better    -> `pass(drift)`: the stored number is stale by X%, re-baseline with
                                        `baseline update --env-change`; the cell records `drift = X`
    the pair is underpowered, unpaired or otherwise undecided -> still `suspect` (exit 3)."""
    stored = 100 * (1 - suspect.ratio) if suspect.ratio is not None else None
    since = f" since {baseline_commit}" if baseline_commit else ""
    if ab.state == REGRESSED:
        slower = 100 * (1 - ab.ratio) if ab.ratio is not None else None
        text = f"slower by {slower:.1f}% cumulative{since}; `git bisect` with `ab` finds the step"
        return Finding(suspect.key, suspect.metric, REGRESSED, text, ratio=ab.ratio, ci=ab.ci, n=ab.n)
    if ab.state in (INSIDE_NOISE, IMPROVED, MINOR):
        text = (
            f"baseline stale by {stored:.1f}% (driver, thermals, library or env drift): "
            "re-baseline with `baseline update --env-change`"
        )
        return Finding(suspect.key, suspect.metric, PASS_DRIFT, text, ratio=ab.ratio, ci=ab.ci, n=ab.n, drift=stored)
    return Finding(suspect.key, suspect.metric, SUSPECT, f"confirm was {ab.state}: {ab.detail}", n=ab.n)


def rebaseline_recommended(previous: Sequence[bool]) -> bool:
    """After two consecutive `pass(drift)` runs of the same cell the report's headline becomes REBASELINE
    RECOMMENDED. `previous`: whether each earlier run of the cell ended in `pass(drift)`, oldest first."""
    return len(previous) >= 2 and previous[-1] and previous[-2]
