"""The trust core of warpc.bench on the CPU: schema, statistics, metric definitions, sampler accounting, gates and exit
codes, planning, the unit runner against a fake engine.

  uv run --package warpc python warpc/tests/test_bench.py

Nothing here needs a GPU, a checkpoint or a library; the GPU acceptance (A/A sessions) is the suite's own `ab`."""

import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from harness import Skip, run
from warpc.bench import cli, gates, protocol, schema, stats, suites, units
from warpc.bench.engines import Cannot, EngineSpec, Turn, Unchecked
from warpc.bench.schema import Cell, Header, Metric, Stamp, make_key

KEY = make_key("speed.decode", "Qwen3-0.6B.Q4_0", "warpc:cuda,kv=bf16,head=bf16", "ctx=512", "reuse=1")
RATE = 100.0


def metrics(rate: float, ttft: float = 0.01) -> dict:
    return {"decode_tok_s": Metric.of([rate] * 5, "tok/s"), "ttft_s": Metric.of([ttft], "s")}


def visit(key: str, rate: float, *, arm=None, pair=None, sm=1755.0, foreign=0, **extra) -> Cell:
    clocks = {"sm_mhz": [sm - 15, sm, sm + 15], "foreign_pids": foreign, "samples": 30}
    return Cell(key, "ok", workload={"id": "decode-v1", "ctx_max": 512}, metrics=metrics(rate), clocks=clocks,
                extra=extra, arm=arm, pair=pair)  # fmt: skip


def ab_cells(ratios, *, key_a=KEY, key_b=None, clock_b=1755.0, warmup=0.5) -> list[Cell]:
    """Visits of an `ab` session: the discarded warm-up pair (pair 0, deliberately far off) and one pair per ratio."""
    key_b = key_b or key_a
    cells = [visit(key_a, RATE, arm="A", pair=0), visit(key_b, RATE * warmup, arm="B", pair=0)]
    for i, r in enumerate(ratios, 1):
        a = RATE * (1 + 0.002 * (i % 3))  # a little spread, so that the MAD of an arm is not zero
        cells += [visit(key_a, a, arm="A", pair=i), visit(key_b, a * r, arm="B", pair=i, sm=clock_b)]
    return cells


def noisy(shift: float, n: int, seed: int, sigma: float = 0.005) -> list[float]:
    rng = random.Random(seed)
    return [(1 + shift) * (1 + rng.gauss(0, sigma)) for _ in range(n)]


# ---- schema


def test_schema_round_trip():
    stamp = Stamp(
        warpc={"commit": "abc1234", "dirty": False}, torch="2.14.0", gpu={"name": "GPU", "machine": "gpu-350w"}
    )
    header = Header("20261125T0312Z-ab12", "20261125T0312Z", ["run", "speed"], stamp)
    cells = [
        Cell(KEY, "ok", workload={"n_new": 256}, metrics=metrics(698.9), clocks={"sm_mhz": [1, 2, 3]}, extra={"a": [1]},
             run=header.run, unit=0, t="2026-11-25T03:12:41Z", arm="B", pair=3),
        Cell(KEY + "x", "cannot", "capacity", "needs 24 GiB"),
        Cell(KEY + "y", "skip", "assets", "WARPC_MODELS unset"),
        Cell(KEY + "z", "planned"),
    ]  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "r.jsonl"
        schema.write(path, header, cells)
        run_ = schema.read(path)
        assert run_.header == header and run_.cells == cells
        assert path.read_text().count("\n") == 5
        with schema.Writer(path) as w:  # a second run appended: read_runs sees both, read refuses
            w.write(header)
        assert len(schema.read_runs(path)) == 2
        with contextlib.suppress(schema.SchemaError):
            schema.read(path)
            raise AssertionError("two runs read as one")


def test_schema_refuses():
    cell = Cell(KEY, "ok", metrics=metrics(1.0)).to_dict()
    newer = {**cell, "schema": schema.SCHEMA + 1}
    for bad, why in (
        (newer, "newer schema"),
        ({**cell, "mystery": 1}, "unknown field"),
        ({**cell, "status": "pass"}, "unknown status"),
    ):
        try:
            schema.loads(json.dumps(bad))
        except schema.SchemaError:
            continue
        raise AssertionError(f"accepted: {why}")
    for make in (
        lambda: Cell(KEY, "skip", "assets", metrics=metrics(1.0)),  # a skipped cell has no numbers
        lambda: Cell(KEY, "cannot"),  # a status without a code
        lambda: schema.dumps(Cell(KEY, "ok", metrics={"x": Metric(float("nan"), "s")})),
    ):
        try:
            make()
        except schema.SchemaError:
            continue
        raise AssertionError("an invalid cell was accepted")


def test_reasons_are_scrubbed_of_paths():
    reason = Cell(KEY, "cannot", "env", "no such file /srv/someone/models/Qwen3-0.6B/model.gguf here").reason
    assert "/srv" not in reason and "model.gguf" in reason, reason
    home = str(Path.home())
    assert home not in Cell(KEY, "error", "crash", f"failed in {home}/x/y.py").reason


def test_keys():
    other = make_key("speed.decode", "Qwen3-0.6B.Q4_0", "warpc:cuda,kv=int4_rot,head=bf16", "ctx=512", "reuse=1")
    assert schema.cell_id(KEY) == schema.cell_id(other) != KEY
    assert schema.key_engine_name(KEY) == "warpc"
    assert schema.key_engine_name(make_key("s", "m", "vllm,kv=auto", "c")) == "vllm"
    try:
        make_key("a|b", "m", "e")
    except ValueError:
        return
    raise AssertionError("a separator inside a segment")


def test_engine_spec():
    model = Path("m.gguf")
    spec = EngineSpec.parse("warpc:cuda,kv=int4_rot,head=bf16", model)
    assert (spec.kv, spec.head, spec.backend, spec.mtp) == ("int4_rot", "bf16", "cuda", 0)
    assert spec.text == "warpc:cuda,kv=int4_rot,head=bf16" and EngineSpec.parse(spec.text, model) == spec
    vllm = EngineSpec.parse("vllm,opts=max_num_batched_tokens:8192+gpu_memory_utilization:0.90", model)
    assert vllm.opt("max_num_batched_tokens") == "8192" and EngineSpec.parse(vllm.text, model) == vllm
    assert "opts=" in vllm.text and "|" not in vllm.text  # `|` separates key segments
    assert EngineSpec.from_json(vllm.to_json()) == vllm
    for bad in ("warpc,opts=ubatch:1", "vllm,opts=nonsense:1", "nonsense", "warpc,bogus=1"):
        try:
            EngineSpec.parse(bad, model)
        except ValueError:
            continue
        raise AssertionError(f"parsed {bad!r}")


# ---- statistics


def test_bootstrap_detects_a_3_percent_shift_at_n9():
    gate = gates.GATES["decode_tok_s"]
    for sigma in (0.005, 0.01):
        caught = sum(
            gates.judge_ratios(noisy(-0.03, 9, seed, sigma), gate)[0] == gates.REGRESSED for seed in range(100)
        )
        flagged = sum(
            gates.judge_ratios(noisy(0.0, 9, 1000 + seed, sigma), gate)[0] == gates.REGRESSED for seed in range(100)
        )
        assert caught >= 95, f"sigma {sigma}: caught {caught}/100"
        assert flagged <= 5, f"sigma {sigma}: false alarms {flagged}/100"
    state, info = gates.judge_ratios(noisy(0.03, 9, 7), gate)
    assert state == gates.IMPROVED and info["ci"][0] > 1


def test_bootstrap_interval_and_sign_test():
    values = [0.99, 1.0, 1.01, 1.0, 0.995, 1.005, 1.0, 1.002, 0.998]
    lo, hi = stats.bootstrap_ci(values)
    assert lo <= stats.median(values) <= hi and hi - lo < 0.02
    assert (lo, hi) == stats.bootstrap_ci(values)  # seeded: the same file gives the same interval everywhere
    assert stats.sign_test(5, 5) == 1 / 32 and stats.sign_test(0, 5) == 1.0
    assert stats.paired_ratios([10, 10], [11, 9]) == [1.1, 0.9]
    assert stats.paired_ratios([10, 10], [11, 9], higher_is_better=False) == [
        10 / 11,
        10 / 9,
    ]  # a time: lower is better
    assert stats.mad_outliers([1.0, 1.01, 0.99, 1.0, 1.4]) == [False, False, False, False, True]
    assert stats.mad_outliers([1.0, 1.0, 1.0]) == [False] * 3
    assert stats.percentile([1, 2, 3, 4], 0.5) == 2.5


def test_ratio_rule_tiers():
    gate = gates.GATES["decode_tok_s"]
    assert gates.judge_ratios([0.96] * 9, gate)[0] == gates.REGRESSED
    assert gates.judge_ratios([0.985] * 9, gate)[0] == gates.MINOR  # 1-2% slower: reported, not blocking
    assert gates.judge_ratios([1.0] * 9, gate)[0] == gates.INSIDE_NOISE
    assert gates.judge_ratios([0.97] * 6, gate)[0] == gates.REGRESSED  # sign test tier: all six below 1
    assert gates.judge_ratios([0.97] * 5 + [1.001], gate)[0] == gates.INSIDE_NOISE  # one pair on the other side
    assert gates.judge_ratios([0.97] * 4, gate)[0] == gates.UNDERPOWERED  # never blocks, never passes
    # a time (ttft_s) is read the other way round: the ratios come out of paired_ratios already as B-better-above-1
    assert (
        gates.judge_ratios(stats.paired_ratios([1.0] * 9, [1.04] * 9, higher_is_better=False), gates.GATES["ttft_s"])[0]
        == gates.REGRESSED
    )


# ---- metric definitions


def test_decode_rate_counts_only_tokens_after_the_first_delivery():
    # plain decode: the first delivery is one token; 256 tokens, the last 255 take 0.36 s
    plain = [(10.0, 1), (10.36, 255)]
    assert abs(protocol.decode_rate(plain) - 255 / 0.36) < 1e-9
    # MTP syncing once per group of cycles: the first group makes 24 tokens visible at t_first at no measured cost
    grouped = [(10.0, 24), (10.1, 24), (10.2, 24), (10.3, 24), (10.4, 24), (10.5, 24), (10.6, 24), (10.7, 24),
               (10.8, 24), (10.9, 24), (11.0, 16)]  # fmt: skip
    n_new = sum(n for _, n in grouped)
    assert n_new == 256
    rate = protocol.decode_rate(grouped)
    assert abs(rate - (n_new - 24) / 1.0) < 1e-9
    naive = (n_new - 1) / 1.0  # `n_new - 1` over the interval: credits the 23 tokens that cost no measured time
    assert abs(naive / rate - 1) > 0.09, "the over-credit the n_first rule removes is ~10%"
    turn = Turn(list(range(256)), 0.5, 1.0, 0, n_first=24)
    assert abs(turn.decode_tok_s - 232.0) < 1e-9
    assert abs(Turn(list(range(256)), 0.5, 1.0, 0).decode_tok_s - 255.0) < 1e-9
    for bad in (lambda: protocol.decode_tok_s(10, 10, 0.0, 1.0), lambda: protocol.decode_tok_s(10, 1, 1.0, 1.0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("a degenerate rate was computed")
    assert protocol.cycle_ms(3.0, 150.0) == 20.0 and protocol.prefill_tok_s(4096, 2.0) == 2048.0


class Scripted:
    """An engine whose rounds follow a script, for the round and unit runners."""

    def __init__(self, *, flaky=False, cached_cold=0):
        self.flaky, self.cached_cold, self.calls = flaky, cached_cold, []
        self.cold = 0

    def check(self, ctx_max):
        return 1.5

    def load(self, ctx_max):
        return {"compile_s": 0.25, "plan": "3 tasks"}

    def turn(self, ids, n_new, *, reuse):
        self.calls.append((len(ids), n_new, reuse))
        base = list(range(n_new))
        if self.flaky and reuse and len([c for c in self.calls if c[2]]) == 4:
            base[7] += 1
        return Turn(
            base, 0.01 if reuse else 0.04, (n_new - 1) / 700.0, 0 if not reuse else len(ids) - 1 + self.cached_cold
        )

    def close(self):
        pass


class FakeSampler:
    def __init__(self, foreign=0):
        self.foreign = foreign

    def mark(self):
        return 0.0

    def summary(self, since=0.0, until=float("inf")):
        return {"sm_mhz": [1740.0, 1755.0, 1770.0], "foreign_pids": self.foreign, "samples": 12}

    def device_gib_peak(self, since=0.0, until=float("inf")):
        return 2.25


def speed_unit(n_prompt=5, repeats=5):
    spec = EngineSpec.parse("warpc:cuda", Path("m.gguf"))
    cell = Cell(KEY, "planned", workload={"id": "decode-v1", "ctx_max": 512, "n_prompt": n_prompt})
    params = (("n_new", 256), ("repeats", repeats), ("ttft_repeats", repeats), ("ctx_max", 512), ("n_prompt", n_prompt))
    return units.Unit("speed", (cell,), spec, 40, params=params)


def test_rounds_discard_the_warmup_and_check_identical_ids():
    engine = Scripted()
    got = protocol.rounds(lambda: engine.turn(range(5), 256, reuse=True), repeats=5)
    assert len(engine.calls) == 6 and len(got.decode_tok_s) == 5 and got.stable and got.n_first == 1
    assert abs(got.decode_tok_s[0] - 700.0) < 1e-9
    flaky = Scripted(flaky=True)
    assert not protocol.rounds(lambda: flaky.turn(range(5), 256, reuse=True), repeats=5).stable


def test_speed_unit_on_a_fake_engine():
    engine = Scripted()
    (cell,) = suites.run_speed_unit(speed_unit(), engine, FakeSampler())
    assert cell.status == "ok" and cell.key == KEY and cell.extra["ids_stable"] is True and not cell.extra["contended"]
    assert set(cell.metrics) >= {"decode_tok_s", "ttft_s", "prefill_tok_s", "device_gib_peak", "host_rss_gib"}
    assert len(cell.metrics["decode_tok_s"].samples) == 5 and len(cell.metrics["ttft_s"].samples) == 5
    assert abs(cell.metrics["ttft_s"].v - 0.04) < 1e-12 and cell.metrics["device_gib_peak"].v == 2.25
    cold = [c for c in engine.calls if not c[2]]
    assert cold[0][1] == 256 and all(c[1] == 2 for c in cold[1:]), (
        "one discarded full-length round, then cold ttft turns"
    )
    assert sum(c[2] for c in engine.calls) == 6, "a discarded and five timed reuse=True rounds"
    (bad,) = suites.run_speed_unit(speed_unit(), Scripted(flaky=True), FakeSampler())
    assert bad.status == "fail" and bad.code == "nondeterministic"
    (hit,) = suites.run_speed_unit(speed_unit(), Scripted(), FakeSampler(foreign=1))
    assert hit.extra["contended"] is True


def test_a_cold_turn_that_hits_a_cache_is_refused():
    class Leaky(Scripted):
        def turn(self, ids, n_new, *, reuse):
            t = super().turn(ids, n_new, reuse=reuse)
            t.cached = 3 if not reuse and n_new == 2 else t.cached
            return t

    try:
        list(suites.run_speed_unit(speed_unit(), Leaky(), FakeSampler()))
    except RuntimeError as e:
        assert "cached" in str(e)
        return
    raise AssertionError("a cold turn served from a cache was accepted")


# ---- the sampler's accounting


def test_foreign_pid_detection():
    own, desktop, intruder = 100, 5939, 4242
    acc = protocol.Accumulator(lambda pid: pid == own)
    # the unit alone with two desktop clients (25 and 163 MiB measured): nothing foreign
    acc.feed_apps(1.0, {own: 1200, desktop: 25, desktop + 1: 163})
    acc.feed_apps(2.0, {own: 2300, desktop: 25, desktop + 1: 163})
    assert acc.summary()["foreign_pids"] == 0 and acc.device_gib_peak() == 2300 / 1024
    # a compute job appears at t = 3: it is foreign from then on, and only in windows that contain t = 3
    acc.feed_apps(3.0, {own: 2300, intruder: 4000})
    assert acc.summary()["foreign_pids"] == 1
    assert acc.summary(since=0.0, until=2.5)["foreign_pids"] == 0
    assert acc.summary(since=2.5)["foreign_pids"] == 1
    assert acc.device_gib_peak() == 2300 / 1024, "the intruder's memory is not the unit's"
    acc.feed_apps(4.0, {own: 100, intruder: 4000})  # the same pid again counts once
    assert acc.summary()["foreign_pids"] == 1
    assert protocol.foreign({1: 255, 2: 256, own: 9000}, lambda p: p == own) == {2: 256}


def test_in_tree_follows_children():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert protocol.in_tree(child.pid, os.getpid()) and protocol.in_tree(os.getpid(), os.getpid())
        assert not protocol.in_tree(1, os.getpid())
    finally:
        child.kill()
        child.wait()


def test_sampler_clocks_throttle_and_fallback_memory():
    acc = protocol.Accumulator(lambda pid: False)
    lines = [
        "2026/11/26 06:16:55.605, 1740, 9501, 250.5, 60, 0x0000000000000000, 2000",
        "2026/11/26 06:16:55.805, 1755, 9501, 251.5, 61, 0x0000000000000004, 3100",
        "2026/11/26 06:16:56.005, 1770, 9501, 252.5, 62, 0x0000000000000001, 4100",
        "2026/11/26 06:16:56.205, [N/A], 9501, [N/A], 62, 0x0000000000000040, 4100",
        "not a sample",
    ]
    for line in lines:
        acc.feed_gpu(line)
    s = acc.summary()
    assert s["samples"] == 4 and s["sm_mhz"][1] == 1755 and s["mem_mhz"] == 9501
    assert s["throttle_frac"] == 0.25, "the thermal sample throttles; the power cap and idle (0x1) do not"
    assert s["reasons"] == {"sw_power_cap": 0.25, "idle": 0.25, "hw_thermal": 0.25}
    assert acc.device_gib_peak() == (4100 - 2000) / 1024, "no per-process data: the rise of memory.used"


def test_machine_key_and_fingerprint():
    assert protocol.machine_key("NVIDIA GeForce RTX 3090", 350.0) == "rtx3090-350w"
    assert protocol.machine_key(None, 350.0) is None
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x" * 10)
        a = protocol.fingerprint(d)
        (d / "model.safetensors").write_bytes(b"x" * 11)
        assert protocol.fingerprint(d) != a, "a weight file's size is part of the identity"
        g = d / "m.gguf"
        g.write_bytes(b"GGUF" + b"\0" * 100)
        assert protocol.fingerprint(g).startswith("sha256:")


def test_stamp_has_no_machine_paths():
    s = protocol.stamp()
    text = str(s.to_dict())
    for forbidden in ("/home/", "/media/", "/dev/shm", str(Path.home())):
        assert forbidden not in text, forbidden
    assert s.warpc["dirty"] in (True, False) and "python" in s.to_dict()


# ---- gates and exit codes


def test_exit_codes():
    ok = visit(KEY, RATE)
    assert gates.judge([ok]).exit_code == 0 and gates.judge([ok]).findings[0].state == gates.NEW
    skip = Cell(KEY + "s", "skip", "assets", "WARPC_TEST_SWIFT27 unset")
    cannot = Cell(KEY + "c", "cannot", "capacity", "not enough memory")
    v = gates.judge([ok, cannot])
    assert v.exit_code == 0, "cannot is a result, not incompleteness"
    v = gates.judge([ok, skip], required=[KEY, KEY + "s"])
    assert v.exit_code == 3 and v.state == gates.SKIP, "a skipped required cell is incomplete, never a pass"
    assert gates.judge([ok, skip], required=[KEY]).exit_code == 0, "a skip outside the required list does not block"
    assert gates.judge([ok, Cell(KEY + "f", "fail", "nondeterministic", "ids differ")]).exit_code == 1
    assert gates.judge([ok, Cell(KEY + "e", "error", "crash", "boom")]).exit_code == 1
    assert gates.judge([ok], required=[KEY, KEY + "gone"]).exit_code == 1, "a required cell absent from the result"
    assert gates.judge([visit(KEY, RATE, foreign=1)]).exit_code == 3, "a contended visit is dropped: nothing measured"
    # underpowered ab: 3 pairs
    assert gates.judge(ab_cells([1.0] * 3)).exit_code == 3
    assert gates.judge(ab_cells([1.0] * 9)).exit_code == 0
    assert gates.judge(ab_cells([1.0] * 9), required=[KEY]).exit_code == 0, "an ab finding is keyed by the cell id"
    assert gates.judge(ab_cells([1.0] * 9), required=[KEY, KEY + "gone"]).exit_code == 1
    assert gates.judge(ab_cells([0.95] * 9)).exit_code == 1
    assert gates.judge(ab_cells([1.05] * 9)).exit_code == 0
    minor = gates.judge(ab_cells([0.985] * 9))
    assert minor.exit_code == 0 and minor.findings[0].state == gates.MINOR


def test_cli_exit_codes_without_a_gpu():
    env = {k: os.environ.pop(k) for k in ("WARPC_TEST_SWIFT27", "WARPC_MODELS") if k in os.environ}
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), tempfile.TemporaryDirectory() as tmp:
            code = cli.main(["run", "speed", "--model", "qwen35-27b", "--prompt-len", "385", "--out", f"{tmp}/r.jsonl"])
            result = schema.read(Path(tmp) / "r.jsonl")
        assert code == 3, f"a skipped required cell must not exit 0: {code}\n{out.getvalue()}"
        assert [c.status for c in result.cells] == ["skip"] and "WARPC_TEST_SWIFT27" in result.cells[0].reason
        assert cli.main(["explain", "gates"]) == 0
    finally:
        os.environ.update(env)


def test_pair_hygiene():
    base = ab_cells([1.0] * 12)
    assert len(gates.pairs_of(base)["speed.decode|Qwen3-0.6B.Q4_0|ctx=512|reuse=1"]) == 12, "pair 0 is the warm-up"
    state, info = gates.judge_pairs(gates.pairs_of(base)[schema.cell_id(KEY)], gates.GATES["decode_tok_s"])
    assert state == gates.INSIDE_NOISE and info["n"] == 12 and info["dropped"] == {}, "the warm-up pair never counts"
    # same engine, clocks 2.7% apart -> every pair dropped; a looser bound keeps them
    slow = ab_cells([1.0] * 9, clock_b=1755 * 1.027)
    state, info = gates.judge_pairs(gates.pairs_of(slow)[schema.cell_id(KEY)], gates.GATES["decode_tok_s"])
    assert state == gates.UNDERPOWERED and info["dropped"]["clock"] == 9
    state, _ = gates.judge_pairs(
        gates.pairs_of(slow)[schema.cell_id(KEY)], gates.GATES["decode_tok_s"], clock_bound=0.03
    )
    assert state == gates.INSIDE_NOISE
    # cross-engine: the clock is a logged variable, never a drop reason
    vllm = make_key("speed.decode", "Qwen3-0.6B.Q4_0", "vllm,kv=auto,head=bf16", "ctx=512", "reuse=1")
    cross = ab_cells([1.0] * 9, key_b=vllm, clock_b=1755 * 1.06)
    state, info = gates.judge_pairs(gates.pairs_of(cross)[schema.cell_id(KEY)], gates.GATES["decode_tok_s"])
    assert state == gates.INSIDE_NOISE and info["n"] == 9 and not info["dropped"]
    # a contended visit and a both-sides spike are dropped
    cells = ab_cells([1.0] * 11)
    contended = next(c for c in cells if c.arm == "B" and c.pair == 3)
    contended.clocks["foreign_pids"] = 1
    for c in cells:
        if c.pair == 5:
            c.metrics["decode_tok_s"] = Metric.of([RATE * (1.4 if c.arm == "A" else 1.5)] * 5, "tok/s")
    state, info = gates.judge_pairs(gates.pairs_of(cells)[schema.cell_id(KEY)], gates.GATES["decode_tok_s"])
    assert info["dropped"] == {"contended": 1, "spike": 1} and info["n"] == 9, info


def test_baseline_tripwire_and_confirm_end_states():
    visits = [visit(KEY, RATE, arm=None) for _ in range(3)]
    baseline = gates.Baseline.from_cells(
        visits, Stamp(warpc={"commit": "base123"}, torch="2.14.0", gpu={"machine": "g-350w"})
    )
    assert baseline.origin == "base123" and baseline.rows[KEY]["n"] == 3
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "b.json").write_text(baseline.dumps())
        assert gates.Baseline.load(Path(tmp) / "b.json") == baseline
    same = gates.judge([visit(KEY, RATE * 0.99) for _ in range(3)], baseline, stamp=Stamp(torch="2.14.0"))
    assert same.findings[0].state == gates.PASS and same.exit_code == 0
    slow = gates.judge([visit(KEY, RATE * 0.95) for _ in range(3)], baseline, stamp=Stamp(torch="2.14.0"))
    suspect = next(f for f in slow.findings if f.metric == "decode_tok_s")
    assert suspect.state == gates.SUSPECT and slow.exit_code == 3, "a tripped wire is incomplete until --confirm"
    assert gates.judge([visit(KEY, RATE * 0.95)], baseline, stamp=Stamp(torch="2.14.0")).exit_code == 3, "one visit"
    env = gates.judge([visit(KEY, RATE * 0.95) for _ in range(3)], baseline, stamp=Stamp(torch="2.15.0"))
    assert {f.state for f in env.findings if f.metric == "decode_tok_s"} == {gates.INCONCLUSIVE_ENV}
    clock = gates.judge([visit(KEY, RATE, sm=1755 * 1.05) for _ in range(3)], baseline, stamp=Stamp(torch="2.14.0"))
    assert {f.state for f in clock.findings if f.metric == "decode_tok_s"} == {gates.INCONCLUSIVE_CLOCK}
    other = visit(KEY, RATE)
    other.workload = {"id": "decode-v2"}
    assert gates.judge([other] * 3, baseline).findings[0].state == gates.WORKLOAD_CHANGED
    assert gates.judge([visit(KEY + "n", RATE)], baseline).findings[0].state == gates.NEW
    # the three end states of --confirm
    regressed = gates.resolve_confirm(suspect, gates.Finding(KEY, "decode_tok_s", gates.REGRESSED, ratio=0.94, n=9),
                                      baseline_commit="base123")  # fmt: skip
    assert regressed.state == gates.REGRESSED and "cumulative since base123" in regressed.detail
    drift = gates.resolve_confirm(suspect, gates.Finding(KEY, "decode_tok_s", gates.INSIDE_NOISE, ratio=1.0, n=9))
    assert drift.state == gates.PASS_DRIFT and abs(drift.drift - 5.0) < 1e-6 and "re-baseline" in drift.detail
    assert gates.Verdict([drift]).exit_code == 0
    undecided = gates.resolve_confirm(suspect, gates.Finding(KEY, "decode_tok_s", gates.UNDERPOWERED, "n = 3", n=3))
    assert undecided.state == gates.SUSPECT and gates.Verdict([undecided]).exit_code == 3
    assert gates.Verdict([regressed]).exit_code == 1
    assert not gates.rebaseline_recommended([True]) and gates.rebaseline_recommended([False, True, True])


def test_confirm_reruns_only_suspect_cells_and_records_the_end_state():
    """`run --confirm` on a stored-baseline tripwire: a paired session of the baseline's commit against HEAD, stored
    next to the run file; the end state replaces `suspect`, a drift is written on the run's cells."""
    import argparse

    stamp = Stamp(warpc={"commit": "head999"}, gpu={"machine": "g-350w"})
    other_key = KEY.replace("ctx=512", "ctx=4096")
    steady = [visit(other_key, RATE) for _ in range(3)]
    base_cells = [visit(KEY, RATE) for _ in range(3)] + steady
    baseline = gates.Baseline.from_cells(base_cells, Stamp(warpc={"commit": "base123"}, gpu={"machine": "g-350w"}))
    plan = units.Plan([speed_unit(), units.Unit("speed", (Cell(other_key, "planned"),), None, 10)])
    asked = []

    def fake_ab(arms, units_for, *, run, pairs, wrap, on_cells=None):
        asked.append((arms["A"].ref, arms["B"].ref, [u.cells[0].key for u in units_for(arms["A"])], pairs))
        return ab_cells(fake_ab.ratios)

    original = units.run_ab
    units.run_ab = fake_ab
    try:
        for ratios, state, code in (([1.0] * 9, gates.PASS_DRIFT, 0), ([0.94] * 9, gates.REGRESSED, 1)):
            slow = [visit(KEY, RATE * 0.95) for _ in range(3)]
            fake_ab.ratios = ratios
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "base.json").write_text(baseline.dumps())
                path = Path(tmp) / "run.jsonl"
                header = Header("r", "s", ["run"], stamp)
                schema.write(path, header, slow + steady)
                earlier = Path(tmp) / "earlier.jsonl"  # the previous run of this cell also ended in a drift
                schema.write(earlier, header, [visit(KEY, RATE * 0.95, drift=4.0)])
                os.utime(earlier, (1.0, 1.0))
                result = cli.Result(slow + steady, path, header)
                verdict = result.judge(Path(tmp) / "base.json")
                assert verdict.exit_code == 3 and verdict.state == gates.SUSPECT
                args = argparse.Namespace(pairs=9, wrap=[])
                final = cli._confirm(args, plan, result, verdict, Path(tmp) / "base.json")
                decode = next(f for f in final.findings if f.key == KEY and f.metric == "decode_tok_s")
                assert decode.state == state and final.exit_code == code, final.render()
                assert (Path(tmp) / "run.confirm.jsonl").is_file()
                recommended = any("REBASELINE RECOMMENDED" in note for note in final.notes)
                assert recommended == (state == gates.PASS_DRIFT), final.notes
                recorded = [c.extra.get("drift") for c in schema.read(path).cells if c.key == KEY]
                assert (recorded[0] is not None) == (state == gates.PASS_DRIFT), recorded
    finally:
        units.run_ab = original
    assert asked[0] == ("base123", None, [KEY], 9), "only the suspect cell, the baseline's commit against HEAD"


def test_deterministic_gates():
    base = gates.Baseline(
        rows={KEY: {"median": {"device_gib_peak": 2.0, "top1": 0.99, "kl_mean": 1e-3}, "workload": {}, "n": 3}}
    )

    def cell(**m):
        return Cell(KEY, "ok", metrics={k: Metric(v, "x") for k, v in m.items()})

    def judged(**m):
        rules = {k: gates.GATES[k] for k in m}
        return {f.metric: f.state for f in gates.judge([cell(**m)], base, gates=rules).findings}

    assert judged(device_gib_peak=2.05)["device_gib_peak"] == gates.PASS
    assert judged(device_gib_peak=2.2)["device_gib_peak"] == gates.MINOR  # over the 0.1 GiB warning, under 0.25
    assert judged(device_gib_peak=2.3)["device_gib_peak"] == gates.FAIL
    assert judged(top1=0.986)["top1"] == gates.PASS and judged(top1=0.98)["top1"] == gates.FAIL
    assert judged(kl_mean=1.29e-3)["kl_mean"] == gates.PASS and judged(kl_mean=1.31e-3)["kl_mean"] == gates.FAIL


# ---- planning and units


def test_ctx_max_rule():
    assert units.ctx_max(4096) == 4608, "the shipped 27B gate value"
    assert units.ctx_max(5) == 768 and units.ctx_max(385) == 1024 and units.ctx_max(100_000) == 100_608
    assert units.round_up_256(256) == 256 and units.round_up_256(257) == 512


def test_planner_caps_units_at_the_lease():
    cells = [(Cell(f"k{i}", "planned"), 90) for i in range(10)]
    packed = units.pack_cells(cells, fixed_s=60)
    assert all(est <= units.UNIT_CAP_S for _, est in packed) and sum(len(c) for c, _ in packed) == 10
    assert len(packed) == 2 and [len(c) for c, _ in packed] == [6, 4]
    try:
        units.pack_cells([(Cell("big", "planned"), 600)], fixed_s=60)
    except ValueError:
        return
    raise AssertionError("a cell over the cap was packed")


def test_unmet_assets_skip_never_pass():
    unit = speed_unit()
    needy = units.Unit(unit.suite, unit.cells, unit.engine, unit.est_s, needs=("WARPC_BENCH_TEST_ASSET",))
    assert "WARPC_BENCH_TEST_ASSET" not in os.environ
    units.ASSET_ENV["WARPC_BENCH_TEST_ASSET"] = "WARPC_BENCH_TEST_ASSET unset"
    try:
        cells = units.spawn(needy, units.Visit("r", 0))
    finally:
        del units.ASSET_ENV["WARPC_BENCH_TEST_ASSET"]
    assert [(c.status, c.code) for c in cells] == [("skip", "assets")]
    assert gates.judge(cells, required=[c.key for c in cells]).exit_code == 3


def test_a_wrapped_child_that_dies_leaves_error_cells():
    """`--wrap` prefixes the child's command: this wrapper is the process that runs, and it dies with a message."""
    code = "import sys; print('RuntimeError: lease refused', file=sys.stderr); sys.exit(7)"
    cells = units.spawn(speed_unit(), units.Visit("run1", 4), wrap=[sys.executable, "-c", code])
    (cell,) = cells
    assert cell.status == "error" and cell.code == "crash" and "lease refused" in cell.reason, cell
    assert (cell.run, cell.unit) == ("run1", 4)
    assert gates.judge(cells).exit_code == 1


def test_ab_visit_order_counterbalances():
    order = [units.ab_order(p) for p in range(5)]
    assert order == [("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"), ("A", "B")]


def test_a_worktree_arm_runs_this_trees_harness_on_a_commit():
    """An arm on a commit is a temporary worktree of it with this tree's `bench` package copied over its own (a commit
    that predates the suite has none); it is removed afterwards. Tried on a scratch repository, not on this one."""

    def git(*args, cwd):
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        (repo / "warpc/src/warpc").mkdir(parents=True)
        (repo / "warpc/src/warpc/__init__.py").write_text("OLD = True\n")
        git("init", "-q", cwd=repo)
        git("-c", "user.name=t", "-c", "user.email=t@t", "add", "-A", cwd=repo)
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "old", cwd=repo)
        original = units.repo_root
        units.repo_root = lambda: repo
        try:
            with units.Worktree("HEAD") as tree:
                root = Path(tree.pythonpath).parent.parent
                assert (root / "warpc/src/warpc/__init__.py").read_text() == "OLD = True\n"
                assert (root / "warpc/src/warpc/bench/units.py").is_file(), "this tree's harness is in the arm"
                assert not list((root / "warpc/src/warpc/bench").rglob("__pycache__"))
                assert tree.commit and len(tree.commit) >= 7
            assert not root.exists(), "the worktree is removed"
        finally:
            units.repo_root = original


def test_wait_idle_backs_off_and_gives_up():
    slept = []
    calls = iter([{7: 500}, {7: 500}, {}])
    original = protocol.gpu_busy
    protocol.gpu_busy = lambda: next(calls)
    try:
        assert protocol.wait_idle(600, sleep=slept.append) == {} and slept == [5, 10]
        protocol.gpu_busy = lambda: {9: 4000}
        slept.clear()
        assert protocol.wait_idle(30, sleep=slept.append) == {9: 4000} and sum(slept) == 30
    finally:
        protocol.gpu_busy = original


def test_report_prints_not_measured_with_reasons():
    cells = [
        visit(KEY, RATE),
        Cell(KEY + "c", "cannot", "capacity", "needs 24 GiB"),
        Cell(KEY + "s", "skip", "assets", "WARPC_MODELS unset"),
    ]
    from warpc.bench import report

    table = report(cells)
    assert "n/s [1]" in table and "needs 24 GiB" in table and "WARPC_MODELS unset" in table
    block = report(
        cells, "markdown", Header("r", "s", ["run", "speed"], Stamp(gpu={"name": "GPU", "power_limit_w": 350.0}))
    )
    assert "Hardware" in block and "Not measured" in block and "capacity" in block


def test_report_leaves_out_the_warm_up_pair():
    from warpc.bench import report

    table = report(ab_cells([1.0] * 3))
    arm_a = next(line for line in table.splitlines() if "  A  " in line)
    assert " 3 " in arm_a, "three measured visits of arm A, the warm-up pair not among them"
    assert f"{RATE * 0.5:.1f}" not in table, "the warm-up pair's (deliberately off) rate is not in any median"


# ---- the planner on a real checkpoint (CPU only)


def test_dense_check_estimates_on_the_cpu():
    gguf = os.environ.get("WARPC_TEST_GGUF")
    if not gguf:
        raise Skip("WARPC_TEST_GGUF unset")
    from warpc.bench.engines import make

    spec = EngineSpec.parse("warpc:cuda,kv=bf16,head=bf16", Path(gguf))
    small, large = make(spec).check(512), make(spec).check(8192)
    assert 0.1 < small < large < 60, (small, large)
    try:
        make(spec.with_(kv="fp8")).check(512)
    except Cannot as e:
        assert e.code in ("kernel_missing", "model_limit") and "fp8" in str(e), (e.code, str(e))
    else:
        raise AssertionError(
            "fp8 on a dense checkpoint: no CUDA-backend kernel and no scales; the preflight must say so"
        )


def test_swiftmk_without_its_library_is_unchecked():
    checkpoint = os.environ.get("WARPC_TEST_SWIFT27")
    if not checkpoint:
        raise Skip("WARPC_TEST_SWIFT27 unset")
    from warpc.bench.engines import make

    saved = os.environ.pop("WARPC_SWIFTMK_LIB", None)
    try:
        spec = EngineSpec.parse("warpc:swiftmk,kv=int4_rot,head=bf16", Path(checkpoint))
        try:
            make(spec).check(4608)
        except Unchecked as e:
            assert "WARPC_SWIFTMK_LIB" in str(e)
        else:
            raise AssertionError("the swiftmk layouts come from its library: without it the check cannot estimate")
        try:
            make(spec.with_(kv="k8v8")).check(4608)
        except Cannot as e:
            assert e.code == "kernel_missing"  # lowering is CPU-only and runs before the library is needed
    finally:
        if saved is not None:
            os.environ["WARPC_SWIFTMK_LIB"] = saved


# ---- import discipline


def test_import_discipline():
    package = Path(__file__).resolve().parents[1] / "src" / "warpc"
    assert "bench" not in (package / "__init__.py").read_text(), "warpc/__init__ must never import the suite"
    probe = (
        "import sys, warpc; assert 'warpc.bench' not in sys.modules, 'import warpc pulled the suite';"
        "before = set(sys.modules); import warpc.bench, warpc.bench.cli;"
        "heavy = [m for m in ('vllm', 'transformers', 'huggingface_hub', 'fla', 'tokenizers')"
        " if m in sys.modules and m not in before];"
        "assert not heavy, f'the suite imported {heavy}'"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]


run(globals())
