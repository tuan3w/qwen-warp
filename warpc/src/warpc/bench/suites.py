"""The suite table: what each suite measures, how it plans its units and how a unit measures its cells.

`explain suites` prints this table, so the numbers (estimated seconds, gated metrics) live only here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import protocol, units
from .engines import Cannot, Engine, EngineSpec, Unchecked, make
from .gates import GATES
from .schema import Cell, Metric, make_key
from .units import Plan, Unit

# the 5-token prompt of the shipped matrix cells ("The capital of France is" in Qwen3's vocabulary; valid ids in every
# vocabulary) and a different one of the same length for the discarded cold prefill
PROMPT = (785, 6722, 315, 9625, 374)
OTHER_PROMPT = (785, 5944, 315, 9625, 374)
DENSE_CTX = (512, 4096)  # the matrix's own variable: ctx_max, explicit
N_NEW = 256
HOST_TABLES_ABOVE = 80_000  # the 27B's embedding and head tables on the host beyond about 80k positions (bench.py docs)
WINDOW_ABOVE = 16_384  # the window prefill pays above this many positions
TTFT_REPEATS_UP_TO = 4096  # cold prefills above this many tokens are timed once (a 100k prefill takes ~100 s)
LOAD_S = 6  # seconds of process start and compile of a dense visit: 9.6 s a Q4_0 visit (4.9 s compile), 5.7 s bf16


@dataclass
class Selection:
    """What the command line selects."""

    model: list[str] = field(default_factory=list)  # a path, or a name under $WARPC_MODELS
    fmt: list[str] = field(default_factory=lambda: ["bf16"])  # bf16 | Q4_0 | Q8_0 ...
    engine: list[str] = field(default_factory=lambda: ["warpc"])  # engine spec texts (EngineSpec.parse)
    kv: list[str] | None = None
    ctx: list[int] | None = None
    prompt_len: list[int] | None = None
    n_new: int = N_NEW
    repeats: int = protocol.TIMED_ROUNDS


@dataclass(frozen=True)
class Suite:
    name: str
    about: str
    metrics: tuple[str, ...]  # the metrics it gates (the others it reports)
    plan: Callable[[Selection], Plan]
    run_unit: Callable[[Unit, Engine, protocol.Sampler], Iterable[Cell]]
    cadence: str = ""


# ---- model resolution


def model_label(path: Path) -> str:
    """`Qwen3-0.6B.Q4_0` for `.../Qwen3-0.6B/Qwen3-0.6B-Q4_0.gguf`, `Qwen3-0.6B.bf16` for `.../Qwen3-0.6B/hf`."""
    if path.suffix == ".gguf":
        m = re.match(r"(.+?)[-.]((?:I?Q\d\w*)|F16|BF16)$", path.stem)
        return f"{m.group(1)}.{m.group(2)}" if m else path.stem
    name = path.parent.name if path.name == "hf" else path.name
    return f"{name}.bf16"


def resolve_model(name: str, fmt: str) -> Path | str:
    """The checkpoint of `name` in `fmt`: an existing path as given, else `$WARPC_MODELS/<name>/hf` (bf16) or the GGUF
    file of that quantisation in `$WARPC_MODELS/<name>/`; `qwen35-27b` is `$WARPC_TEST_SWIFT27`. A string is the reason
    it cannot be found."""
    if Path(name).exists():
        return Path(name)
    if name == "qwen35-27b":
        env = os.environ.get("WARPC_TEST_SWIFT27")
        return Path(env) if env else units.ASSET_ENV["WARPC_TEST_SWIFT27"]
    root = os.environ.get("WARPC_MODELS")
    if not root:
        return units.ASSET_ENV["WARPC_MODELS"]
    directory = Path(root) / name
    if fmt == "bf16":
        found = [directory / "hf"] if (directory / "hf").is_dir() else []
    else:
        found = sorted(directory.glob(f"*{fmt}*.gguf"))
    if not found:
        return f"no {fmt} checkpoint of {name} under $WARPC_MODELS (python -m warpc.bench fetch)"
    return found[0]


def ids_sha(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


# ---- the dense speed suite


def _spec_for(text: str, model: Path, kv: str | None, ctx: int) -> EngineSpec:
    """The spec of a cell with every ctx_max-dependent default written out: host tables above `HOST_TABLES_ABOVE`
    positions, the window prefill above `WINDOW_ABOVE` on the swiftmk backend."""
    spec = EngineSpec.parse(text, model)
    if kv is not None:
        spec = spec.with_(kv=kv)
    if spec.name in ("warpc", "hand"):
        if spec.host_tables is None and ctx > HOST_TABLES_ABOVE:
            spec = spec.with_(host_tables=True)
        if spec.window is None and ctx > WINDOW_ABOVE and spec.backend == "swiftmk":
            spec = spec.with_(window=True)
    return spec


def plan_speed(sel: Selection) -> Plan:
    """Decode speed, cold TTFT and device memory of `sel.model` x `sel.fmt` x `sel.kv` x `sel.engine` x ctx_max: a
    unit per (spec, ctx_max), one process, one cell. Prompt: the 5-token matrix prompt at the ctx_max rungs of the
    matrix (512, 4096), or with `--prompt-len` a corpus prompt at `ctx_max(prompt_len)`."""
    plan = Plan()
    if not sel.model:
        raise ValueError("speed needs --model")
    cases: list[tuple[int, int]] = (  # (prompt_len, ctx_max)
        [(n, units.ctx_max(n)) for n in sel.prompt_len]
        if sel.prompt_len
        else [(len(PROMPT), c) for c in sel.ctx or DENSE_CTX]
    )
    for name in sel.model:
        for fmt in sel.fmt:
            path = resolve_model(name, fmt)
            for engine_text in sel.engine:
                for kv in sel.kv or [None]:
                    for n_prompt, ctx in cases:
                        _plan_cell(plan, sel, name, fmt, path, engine_text, kv, n_prompt, ctx)
    return plan


def _plan_cell(plan: Plan, sel: Selection, name: str, fmt: str, path: Path | str, engine_text: str, kv: str | None,
               n_prompt: int, ctx: int) -> None:  # fmt: skip
    label = name if isinstance(path, str) else model_label(path)
    workload = {
        "id": "decode-v1",
        "n_prompt": n_prompt,
        "n_new": sel.n_new,
        "ignore_eos": True,
        "ctx_max": ctx,
        "forged": False,
    }
    if isinstance(path, str):
        spec_text = _spec_for(engine_text, Path(name), kv, ctx).text
        key = make_key("speed.decode", f"{label}.{fmt}", spec_text, f"ctx={ctx}", "reuse=1")
        plan.decided.append(Cell(key, "skip", "assets", path, workload))
        return
    spec = _spec_for(engine_text, path, kv, ctx)
    key = make_key("speed.decode", label, spec.text, f"ctx={ctx}", "reuse=1")
    try:
        workload["ids_sha"] = ids_sha(prompt_ids(path, n_prompt)[0])
    except Cannot as e:
        plan.decided.append(Cell(key, "cannot", e.code, str(e), workload))
        return
    cell = Cell(key, "planned", workload=workload)
    try:
        estimate = make(spec).check(ctx)
        if estimate is not None:
            cell.extra["estimate_gib"] = estimate
    except Cannot as e:
        plan.decided.append(Cell(key, "cannot", e.code, str(e), workload))
        return
    except Unchecked as e:
        cell.extra["preflight"] = f"unchecked: {e}"
    ttft_repeats = sel.repeats if n_prompt <= TTFT_REPEATS_UP_TO else 1
    est = estimate_s(spec, n_prompt, sel.n_new, sel.repeats, ttft_repeats)
    plan.units.append(
        Unit(
            "speed",
            (cell,),
            spec,
            est,
            params=(
                ("n_new", sel.n_new),
                ("repeats", sel.repeats),
                ("ttft_repeats", ttft_repeats),
                ("ctx_max", ctx),
                ("n_prompt", n_prompt),
            ),
        )
    )


# Seconds of a visit: [INFERENCE] for the 27B from the shipped numbers (load 30-45 s, window prefill ~1.1k tok/s at
# 100k and 2k tok/s at 16k, decode 41-52 tok/s); the dense row is measured (Qwen3-0.6B, Q4_0 and bf16).
COSTS: dict[str, dict[str, float]] = {
    "dense": {"load_s": LOAD_S, "decode_tok_s": 700.0, "prefill_tok_s": 1000.0},
    "27b": {"load_s": 45.0, "decode_tok_s": 45.0, "prefill_tok_s": 1100.0},
}


def estimate_s(spec: EngineSpec, n_prompt: int, n_new: int, repeats: int, ttft_repeats: int) -> int:
    """The seconds a visit takes: load, one discarded round, `ttft_repeats` cold prefills and `1 + repeats` rounds
    (the first of them a cold prefill that takes the snapshot)."""
    c = COSTS["27b" if spec.backend == "swiftmk" else "dense"]
    prefill = n_prompt / c["prefill_tok_s"] + 0.05
    decode = n_new / c["decode_tok_s"]
    return int(c["load_s"] + prefill + decode + ttft_repeats * prefill + (1 + repeats) * decode + prefill)


def prompt_ids(model: Path, n: int) -> tuple[list[int], list[int]]:
    """The prompt of `n` tokens of a cell and a different one of the same length (the discarded cold prefill's). The
    5-token matrix prompt, else the corpus (`workload.corpus_ids`) through the model's tokenizer."""
    if n == len(PROMPT):
        return list(PROMPT), list(OTHER_PROMPT)
    from . import workload

    directory = model if model.is_dir() else model.parent / "hf"
    if not (directory / "tokenizer.json").is_file():
        raise Cannot("env", f"a {n}-token corpus prompt needs the model's tokenizer.json (an HF directory)")
    try:
        tok = workload.FileTokenizer(directory)
        ids = workload.corpus_ids(tok, n, 0)
    except (workload.CorpusUnavailable, workload.CorpusMismatch) as e:
        raise Cannot("env", str(e)) from e
    try:
        other = workload.corpus_ids(tok, n, 1)
    except ValueError:  # one slice only at this length
        other = ids[::-1]
    return ids, other


def run_speed_unit(unit: Unit, engine: Engine, sampler: protocol.Sampler) -> Iterable[Cell]:
    """One visit: load, a discarded full-length round (graph capture, allocator pool, clock ramp), a discarded cold
    prefill of a different prompt of the same length, `repeats` cold prefills for `ttft_s`, then a discarded and
    `repeats` timed reuse=True rounds of the identical prompt for `decode_tok_s`."""
    (cell,) = unit.cells
    ids, other = prompt_ids(unit.engine.model, unit.param("n_prompt"))
    n_new, repeats = unit.param("n_new"), unit.param("repeats")
    start = sampler.mark()
    info = engine.load(unit.param("ctx_max"))
    engine.turn(other, n_new, reuse=False)  # the process's discarded round and the discarded cold prefill in one
    t0 = sampler.mark()
    ttft = []
    for _ in range(unit.param("ttft_repeats")):
        t = engine.turn(ids, 2, reuse=False)
        if t.cached:
            raise RuntimeError(f"a cold turn served {t.cached} cached tokens")
        ttft.append(t.first_s)
    timed = protocol.rounds(lambda: engine.turn(ids, n_new, reuse=True), repeats)
    end = sampler.mark()
    extra: dict[str, Any] = {
        **cell.extra,
        **{k: v for k, v in info.items() if k != "plan"},
        "plan": info.get("plan"),
        "n_first": timed.n_first,
        "cycles_per_sync": None,
        "ids_stable": timed.stable,
        "cached_fraction": timed.cached[0] / len(ids),
    }
    if (gpu_seconds := getattr(engine, "gpu_seconds", None)) is not None:
        wall_per_token = statistics.median(1 / rate for rate in timed.decode_tok_s)
        extra["host_overhead_us"] = protocol.host_overhead_us(wall_per_token, gpu_seconds(ids, n_new), n_new - 1)
    clocks = sampler.summary(t0, end)
    rss = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    ) / 2**20
    metrics = {
        "decode_tok_s": Metric.of(timed.decode_tok_s, "tok/s"),
        "ttft_s": Metric.of(ttft, "s"),
        "prefill_tok_s": Metric.of([len(ids) / s for s in ttft], "tok/s"),
    }
    if (peak := sampler.device_gib_peak(start, end)) is not None:
        metrics["device_gib_peak"] = Metric(peak, "GiB")
    metrics["host_rss_gib"] = Metric(rss, "GiB")
    workload = dict(cell.workload)
    if not timed.stable:
        yield Cell(cell.key, "fail", "nondeterministic", "rounds from one state emitted different ids", workload,
                   metrics, clocks, extra)  # fmt: skip
        return
    extra["contended"] = bool(clocks.get("foreign_pids"))
    extra["throttled"] = (clocks.get("throttle_frac") or 0.0) > protocol.THROTTLE_FLAG_FRAC
    yield Cell(cell.key, "ok", None, None, workload, metrics, clocks, extra)


SUITES: dict[str, Suite] = {
    s.name: s
    for s in (
        Suite(
            "speed",
            "decode speed, cold TTFT and device memory of the dense matrix: a 5-token prompt at ctx_max 512 and 4,096",
            ("decode_tok_s", "ttft_s", "device_gib_peak"),
            plan_speed,
            run_speed_unit,
            "PR: baseline tripwire; `ab` when kernel or runtime paths changed",
        ),
    )
}


def table() -> str:
    lines = [f"{'suite':<10} {'gated metrics':<46} what"]
    for s in SUITES.values():
        lines.append(f"{s.name:<10} {', '.join(s.metrics):<46} {s.about}" + (f" [{s.cadence}]" if s.cadence else ""))
    return "\n".join(lines)


def gate_table() -> str:
    lines = []
    for g in GATES.values():
        rule = {
            "paired": f"paired ratio fail {g.t_fail:.0%} / warn {g.t_warn:.0%}, baseline tripwire {g.tripwire:.0%}",
            "absolute": f"baseline + {g.t_fail} fail / + {g.t_warn} warn",
            "ceiling": f"<= {g.mult} x baseline + {g.slack}",
            "floor": f">= baseline - {g.slack}",
            "exact": "exact",
        }[g.kind]
        lines.append(f"  {g.metric:<22} {rule}" + (f"  ({g.basis})" if g.basis else ""))
    return "\n".join(lines)
