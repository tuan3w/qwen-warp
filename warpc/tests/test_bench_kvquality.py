"""bench.kvquality: the sparse reference, the lower-bound KL, the validity rule, the ring check and the gate, on
synthetic distributions (CPU), the driver on a fake model, and (a CUDA device and WARPC_TEST_GGUF) on a real model.

  uv run --package warpc python warpc/tests/test_bench_kvquality.py
"""

import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from harness import Skip, run
from warpc.bench import kvquality as kq

V = 2000


def peaked(n: int, vocab: int = V, decay: float = 0.35, seed: int = 0) -> torch.Tensor:
    """fp32 [n, vocab]: logit -decay * rank with every row's own permutation, so the mass sits in a few ids."""
    g = torch.Generator().manual_seed(seed)
    rank = torch.stack([torch.randperm(vocab, generator=g) for _ in range(n)]).float()
    return -decay * rank + 0.001 * torch.randn(n, vocab, generator=g)


def noisy(x: torch.Tensor, sigma: float, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return x + sigma * torch.randn(x.shape, generator=g)


def corpus(n: int, vocab: int = V) -> list[int]:
    return torch.randint(0, vocab, (n,), generator=torch.Generator().manual_seed(7)).tolist()


def test_sparse_equals_full_when_k_covers_the_mass():
    p = peaked(96)
    q = noisy(p, 0.1)
    ref = kq.reference_of(p, corpus(96))
    assert ref.k == 64 and ref.p_tail.mean() <= kq.TAIL_BUDGET, (ref.k, ref.p_tail.mean())
    s = kq.score(ref, q)
    full = kq.kl_full(p, q)
    err = np.abs(s.kl - full).max()
    assert err < 1e-6 and full.mean() > 1e-3, (err, full.mean())
    assert kq.summarize(s, "k8v8")["valid"]


def test_sparse_is_a_lower_bound():
    # a K that does not cover the mass: the coarsened KL never exceeds the full one
    p = peaked(64, decay=0.05)
    q = noisy(p, 0.5)
    ref = kq.reference_of(p, corpus(64), k=64)
    assert ref.p_tail.min() > 1e-3
    s = kq.score(ref, q)
    full = kq.kl_full(p, q)
    assert (s.kl <= full + 1e-12).all() and s.kl.mean() > 0.5 * full.mean(), (s.kl.mean(), full.mean())


def test_adaptive_k():
    ids = corpus(32, 5000)
    assert kq.reference_of(peaked(32, 5000), ids).k == 64
    mid = kq.reference_of(peaked(32, 5000, decay=0.01), ids)
    assert mid.k == 1024 and mid.p_tail.mean() <= kq.TAIL_BUDGET, (mid.k, mid.p_tail.mean())
    flat = kq.reference_of(torch.randn(32, 5000) * 0.5, ids)
    assert flat.k == 4096 and flat.p_tail.mean() > kq.TAIL_BUDGET  # no rung is enough: reported, not hidden
    small = kq.reference_of(peaked(8, 50), corpus(8, 50))
    assert small.k == 50 and (small.p_tail == 0).all()  # the vocabulary is the support: exact


def test_self_check_is_exactly_zero():
    p = torch.randn(70, V) * 3
    ref = kq.reference_of(p, corpus(70))
    s = kq.score(ref, p)
    assert (s.kl == 0).all() and s.top1.all() and (s.q_tail == s.p_tail).all()
    m = kq.summarize(s, "bf16")
    assert m["kl_mean"] == 0 and m["top1"] == 1 and m["dppl"] == 0 and m["rms_dp"] == 0
    assert m["valid"] and kq.judge(m, "bf16").ok
    sharp = kq.reference_of(peaked(70), corpus(70))
    m = kq.summarize(kq.score(sharp, peaked(70)), "bf16")
    assert m["valid"] and kq.judge(m, "bf16").ok
    # chunking of the rows does not change a bit
    one = kq.Scorer(sharp)
    for i in range(0, 70, 7):
        one.add(peaked(70)[i : i + 7])
    assert (one.finish().kl == 0).all()


def test_perturbed_distribution_trips_the_validity_rule():
    p = peaked(64)
    ref = kq.reference_of(p, corpus(64))
    assert ref.k == 64
    # a format that scatters mass into the reference's tail: 500 ids it ranks outside the top 64 get the top logit - 3
    tail_ids = torch.stack([torch.tensor(sorted(set(range(V)) - set(r.tolist()))[:500]) for r in ref.ids])
    q = p.clone()
    q.scatter_(1, tail_ids, (p.max(1).values - 3)[:, None].expand(-1, 500).contiguous())
    s = kq.score(ref, q)
    assert s.q_tail.min() > 0.1 and s.p_tail.max() < kq.TAIL_BUDGET
    m = kq.summarize(s, "k8v8")
    j = kq.judge(m, "k8v8")
    assert not m["valid"] and not j.valid and not j.ok
    assert next(c.name for c in j.failed) == "tail_lost_est" and m["tail_lost_est"] > 0.1
    # a reference whose own tail is large is invalid for any format that differs from it ...
    wide = peaked(64, decay=0.05)
    coarse = kq.reference_of(wide, corpus(64), k=64)
    m = kq.summarize(kq.score(coarse, noisy(wide, 0.3)), "k8v8")
    assert coarse.p_tail.min() > 0.03 and not m["valid"] and not kq.judge(m, "k8v8").ok
    # ... and a valid self-check when it does not (nothing is dropped from a KL of exactly 0)
    assert kq.judge(kq.summarize(kq.score(coarse, wide), "bf16"), "bf16").ok


def test_layer1_bars_catch_gross_failure_and_pass_a_good_format():
    p = peaked(128)
    ref = kq.reference_of(p, corpus(128))
    good = kq.summarize(kq.score(ref, noisy(p, 0.01)), "k8v8")
    assert good["valid"] and kq.judge(good, "k8v8").ok, kq.judge(good, "k8v8").checks
    bad = kq.summarize(kq.score(ref, noisy(p, 2.0)), "k8v8")
    j = kq.judge(bad, "k8v8")
    assert bad["valid"] and not j.ok and {c.name for c in j.failed} == {"kl_mean", "top1"}, j.failed
    assert all(c.layer == "class" for c in j.failed)
    # the 4-bit class tolerates what the 8-bit class does not
    mid = kq.summarize(kq.score(ref, noisy(p, 0.15)), "int4_rot")
    verdict = lambda fmt: {c.name: c.status for c in kq.judge(mid, fmt).checks if c.layer == "class"}  # noqa: E731
    assert verdict("int4_rot") == {"kl_mean": "pass", "top1": "pass"}, (mid["kl_mean"], mid["top1"])
    assert verdict("k8v8") == {"kl_mean": "fail", "top1": "pass"}


def test_calibration_first_and_layer2():
    m = {"tail_lost_est": 1e-3, "kl_mean": 3e-4, "top1": 0.985, "kl_exact_region": 1e-6, "kl_boundary_spike": 1.0}
    first = kq.judge(m, "int4_rot")
    assert first.ok and {c.status for c in first.checks if c.layer == "baseline"} == {"report"}
    base = {"kl_mean": 3e-4, "top1": 0.985}
    assert kq.judge(m, "int4_rot", base).ok
    assert {c.status for c in kq.judge(m, "int4_rot", base).checks if c.layer == "baseline"} == {"pass"}
    worse = m | {"kl_mean": 1.25 * 3e-4 + 6e-5}  # past 1.25 x baseline + 5e-5
    j = kq.judge(worse, "int4_rot", base)
    assert not j.ok and [(c.layer, c.name) for c in j.failed] == [("baseline", "kl_mean")]
    edge = m | {"kl_mean": 1.25 * 3e-4 + 4e-5, "top1": 0.985 - 0.01 + 1e-9}
    assert kq.judge(edge, "int4_rot", base).ok
    flipped = m | {"top1": 0.985 - 0.011}
    assert [(c.layer, c.name) for c in kq.judge(flipped, "int4_rot", base).failed] == [("baseline", "top1")]
    # a baseline never loosens layer 1
    assert not kq.judge(m | {"kl_mean": 0.06}, "int4_rot", {"kl_mean": 0.06, "top1": 0.985}).ok


def test_formats_windows_and_classes():
    assert {f: kq.exact_end(f) for f in ("bf16", "k8v8", "k8v8w", "k8v4w", "int4_rot", "fp8")} == {
        "bf16": 0,
        "k8v8": 0,
        "k8v8w": 260,
        "k8v4w": 260,
        "int4_rot": 500,
        "fp8": 0,
    }
    assert [kq.kv_class(f) for f in ("bf16", "k8v8", "k8v8w", "fp8", "k8v4w", "int4_rot")] == [
        "bf16",
        "8bit",
        "8bit",
        "8bit",
        "4bit",
        "4bit",
    ]


def kl_curve(n: int = 4096, exact: int = 500, floor: float = 1e-6, plateau: float = 3e-4, seed: int = 3):
    """KL per position p: the rounding floor while every key is exact, a ramp over 200 positions, then the plateau."""
    rng = np.random.default_rng(seed)
    p = np.arange(n)
    level = np.clip((p - exact) / 200, 0, 1)
    return p, (floor + (plateau - floor) * level) * rng.uniform(0.7, 1.3, n)


def test_ring_check_passes_on_a_flat_exact_region():
    pos, kl = kl_curve()
    r = kq.ring_check(pos, kl, 500)
    assert r["kl_exact_region"] < 2e-6, r
    assert 0.8 < r["kl_boundary_spike"] < 1.3, r


def test_ring_check_fires_on_a_spike_at_the_boundary():
    pos, kl = kl_curve()
    kl[500:540] += 3e-3  # the first coded keys enter badly: ten plateaus
    r = kq.ring_check(pos, kl, 500)
    assert r["kl_boundary_spike"] > 4, r
    pos, kl = kl_curve()
    kl[:500] += 5e-4  # a ring that was never exact
    r = kq.ring_check(pos, kl, 500)
    assert r["kl_exact_region"] > 4e-4, r
    # the same spike elsewhere does not count: it is not at the boundary
    pos, kl = kl_curve()
    kl[2000:2040] *= 8
    assert kq.ring_check(pos, kl, 500)["kl_boundary_spike"] < 1.3


def test_ring_check_needs_its_positions():
    pos, kl = kl_curve()
    assert kq.ring_check(pos, kl, 0) == {"kl_exact_region": None, "kl_boundary_spike": None}
    far = kq.ring_check(np.arange(31744, 32768), kl[:1024], 500)  # a window of a long context
    assert far == {"kl_exact_region": None, "kl_boundary_spike": None}
    short = kq.ring_check(pos[:900], kl[:900], 500)  # no [1024, 4096) to normalise by
    assert short["kl_boundary_spike"] is None and short["kl_exact_region"] is not None
    z = np.zeros(4096)
    assert kq.ring_check(np.arange(4096), z, 500)["kl_boundary_spike"] == 0.0
    z[520] = 1.0
    assert kq.ring_check(np.arange(4096), z, 500)["kl_boundary_spike"] == math.inf


def test_ring_bounds_gate_a_spike():
    old = dict(kq.RING_BOUNDS)
    try:
        kq.RING_BOUNDS["int4_rot"] = (1e-4, 1.5)
        m = {"tail_lost_est": 1e-3, "kl_mean": 3e-4, "top1": 0.99, "kl_exact_region": 1e-6, "kl_boundary_spike": 1.1}
        assert kq.judge(m, "int4_rot").ok
        j = kq.judge(m | {"kl_boundary_spike": 2.0}, "int4_rot")
        assert [(c.layer, c.name) for c in j.failed] == [("ring", "kl_boundary_spike")]
        assert not kq.judge(m | {"kl_exact_region": 1e-3}, "int4_rot").ok
        assert kq.judge(m | {"kl_boundary_spike": None}, "int4_rot").ok  # not covered: report, not a pass or a fail
    finally:
        kq.RING_BOUNDS.clear()
        kq.RING_BOUNDS.update(old)


def test_buckets():
    pos = np.arange(0, 4096)
    kl = np.where(pos < 500, 1.0, np.where(pos < 1024, 2.0, np.where(pos < 2048, 3.0, 4.0)))
    b = kq.kl_by_bucket(pos, kl, 500)
    assert b == {"0-500": 1.0, "500-1024": 2.0, "1024-2048": 3.0, "2048-4096": 4.0}, b
    assert list(kq.kl_by_bucket(pos, kl, 0)) == ["0-1024", "1024-2048", "2048-4096"]
    tail = kq.kl_by_bucket(np.arange(31744, 32768), np.ones(1024), 500)
    assert tail == {"16384-32768": 1.0}, tail


def test_bootstrap_se():
    rng = np.random.default_rng(0)
    kl = rng.exponential(1e-3, 4096)
    se = kq.block_bootstrap_se(kl)
    assert se == kq.block_bootstrap_se(kl)  # a fixed seed
    assert 0.5 * kl.std() / 64 < se < 2 * kl.std() / 64, (se, kl.std() / 64)
    assert kq.block_bootstrap_se(kl[:100]) > 0  # a short run: one full and one partial block


def test_save_load_roundtrip():
    ref = kq.reference_of(peaked(40), corpus(40), start=3)
    ref.meta = {"model": "x", "kv": "bf16"}
    ref.greedy = [5, 6, 7]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ref.npz"
        ref.save(path)
        back = kq.Reference.load(path)
    assert (back.start, back.vocab, back.meta, back.greedy) == (3, V, ref.meta, [5, 6, 7])
    for name in ("ids", "logits", "lse", "target", "target_logit", "argmax"):
        assert torch.equal(getattr(back, name), getattr(ref, name)), name
    assert kq.score(back, peaked(40)).kl.max() == 0
    assert ref.nbytes() < 40 * ref.k * 8 + 40 * 16


def test_greedy_first_diff():
    assert kq.greedy_first_diff([1, 2, 3], [1, 2, 3]) is None
    assert kq.greedy_first_diff([1, 2, 3], [1, 9, 3]) == 1
    assert kq.greedy_first_diff([1, 2, 3], [1, 2]) == 2
    assert kq.greedy_first_diff([], []) is None
    assert kq.greedy_first_diff([4], [5]) == 0


def test_perplexity_and_top1():
    p = peaked(50)
    ids = corpus(50)
    ref = kq.reference_of(p, ids)
    q = p.clone()
    top = ref.argmax.long()
    q[torch.arange(50), top] -= 1.0  # demote the argmax: another token may take over
    s = kq.score(ref, q)
    m = kq.summarize(s, "k8v8")
    assert m["top1"] < 1 and m["kl_mean"] > 0 and m["ppl_ref"] > 0 and m["rms_dp"] > 0
    assert math.isclose(
        m["ppl_ref"],
        math.exp(-torch.log_softmax(p.double(), -1)[torch.arange(50), torch.tensor(ids)].mean().item()),
        rel_tol=1e-9,
    )


class FakeModel:
    """The Model surface the driver uses; logits are a pure function of the token history."""

    def __init__(self, vocab: int = V, noise: float = 0.0):
        self.vocab, self.noise, self.tokens = vocab, noise, []
        self.calls = []

    def reset(self):
        self.tokens = []
        self.calls.append("reset")

    def append(self, ids):
        self.tokens += list(ids)
        self.calls.append(f"append{len(ids)}")
        return 0

    def step(self, token):
        self.tokens.append(token)
        self.calls.append("step")
        return 0

    @property
    def logits(self):
        g = torch.Generator().manual_seed(hash(tuple(self.tokens[-3:])) % 2**31 + len(self.tokens))
        base = -0.35 * torch.randperm(self.vocab, generator=g).float()
        return base + self.noise * torch.randn(self.vocab, generator=torch.Generator().manual_seed(len(self.tokens)))

    def generate(self, ids, n):
        return [(sum(ids) + i) % 17 for i in range(n)]


def test_driver_streams_the_teacher_forced_rows():
    ids = corpus(100)
    model = FakeModel()
    rows = kq.teacher_forced_logits(model, ids)
    assert rows.shape == (99, V) and model.calls[:2] == ["reset", "append1"]
    sub = kq.teacher_forced_logits(model, ids, start=40)
    assert sub.shape == (59, V) and model.calls.count("step") == 98 + 58 and "append41" in model.calls
    assert torch.equal(sub, rows[40:])
    ref = kq.reference_run(model, ids, start=40)
    assert ref.n == 59 and ref.start == 40 and ref.target.tolist() == ids[41:]
    assert (kq.score_run(model, ids, ref).kl == 0).all()
    other = FakeModel(noise=0.1)
    s = kq.score_run(other, ids, ref)
    assert s.kl.mean() > 0 and s.position[0] == 40 and s.position[-1] == 98
    try:
        kq.score_run(model, [*ids[:-1], ids[-1] ^ 1], ref)
        raise AssertionError("a different corpus must be refused")
    except ValueError:
        pass
    for bad in (-1, 99):
        try:
            next(kq.iter_logits(model, ids, bad))
            raise AssertionError("start out of range")
        except ValueError:
            pass


def test_paired_run_is_the_exact_kl():
    ids = corpus(100)
    ref_model, cand = FakeModel(), FakeModel(noise=0.1)
    s = kq.paired_run(ref_model, cand, ids, start=10)
    want = kq.kl_full(kq.teacher_forced_logits(ref_model, ids, 10), kq.teacher_forced_logits(cand, ids, 10))
    assert s.k is None and s.position.tolist() == list(range(10, 99)) and (s.kl == want).all() and s.kl.mean() > 0
    assert (s.p_tail == 0).all() and (s.q_tail == 0).all()
    m = kq.summarize(s, "k8v8")
    assert m["valid"] and m["k"] is None and m["tail_lost_est"] == 0
    zero = kq.paired_run(ref_model, FakeModel(), ids)
    assert (zero.kl == 0).all() and zero.top1.all() and kq.summarize(zero, "bf16")["dppl"] == 0
    ref = kq.reference_run(ref_model, ids, 10)  # the sparse form of the same pair: a lower bound, the same top-1
    sparse = kq.score_run(cand, ids, ref)
    assert (sparse.kl <= s.kl + 1e-12).all() and (sparse.top1 == s.top1).all()
    assert np.array_equal(sparse.logq_target, s.logq_target) is False or np.allclose(sparse.logq_target, s.logq_target)


def test_cell_through_the_schema():
    from warpc.bench import schema

    p = peaked(128)
    ref = kq.reference_of(p, corpus(128))
    workload = {"id": "kv-v1", "n_prompt": 129, "ctx_max": 768}
    key = schema.make_key("quality.kv", "Qwen3-0.6B", "warpc:cuda,kv=k8v8,head=bf16", "ctx=129", "ref=sparse")
    good = kq.summarize(kq.score(ref, noisy(p, 0.01)), "k8v8")
    cell = kq.to_cell(key, good, kq.judge(good, "k8v8"), workload=workload)
    assert cell.status == "ok" and cell.metric("kl_mean") == good["kl_mean"] and cell.extra["mode"] == "sparse"
    assert "kl_exact_region" not in cell.metrics and cell.extra["kl_exact_region"] is None
    assert schema.loads(schema.dumps(cell)) == cell
    bad = kq.summarize(kq.score(ref, noisy(p, 2.0)), "k8v8")
    cell = kq.to_cell(key, bad, kq.judge(bad, "k8v8"), workload=workload)
    assert cell.status == "fail" and cell.code == "quality_gate" and "kl_mean" in cell.reason, cell
    assert schema.loads(schema.dumps(cell)) == cell
    spike = good | {"kl_boundary_spike": math.inf}
    cell = kq.to_cell(key, spike, kq.judge(spike, "k8v8"), workload=workload)
    assert cell.extra["kl_boundary_spike"] == "inf" and schema.loads(schema.dumps(cell)) == cell


def test_real_model_both_modes_agree_and_self_check():
    path = os.environ.get("WARPC_TEST_GGUF")
    if not path:
        raise Skip("WARPC_TEST_GGUF unset")
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    import warpc

    source = warpc.open(path)
    ids = (torch.arange(513) * 7919 % 20000 + 1000).tolist()  # a fixed token stream: the maths needs no real text
    exact = kq.measure(source, ids, ["bf16", "k8v8"], greedy=16)
    sparse = kq.measure(source, ids, ["bf16", "k8v8"], mode="sparse", greedy=16)
    for m in (exact["bf16"], sparse["bf16"]):
        assert m["kl_mean"] == 0 and m["top1"] == 1 and m["dppl"] == 0 and m["greedy_first_diff_256"] is None
    e, s = exact["k8v8"], sparse["k8v8"]
    assert e["k"] is None and s["k"] > 0 and e["positions"] == s["positions"] == 512
    assert e["top1"] == s["top1"] and math.isclose(e["ppl_fmt"], s["ppl_fmt"], rel_tol=1e-9)
    # the sparse form is a lower bound, short of the exact KL by at most what its own validity estimate says
    lost = e["kl_mean"] - s["kl_mean"]
    assert 0 <= lost <= 1.5 * s["tail_lost_est"] * e["kl_mean"], (e["kl_mean"], s["kl_mean"], s["tail_lost_est"])


run(globals())
