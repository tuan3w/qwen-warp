"""KV-cache format quality against bf16 KV of the same backend (bench design section 5).

Two ways to get the KL of a format against the bf16 KV of the same backend:

- `paired_run`: both models resident, stepped in lockstep, the exact full-vocabulary KL of every row, nothing stored.
  For a model that fits twice (the dense PR gate).
- `reference_run` + `score_run`: a reference run (kv=bf16) stores per scored position the top-K ids and fp32 logits of
  its next-token distribution plus the full-vocabulary log-sum-exp (`Reference`, tens of MB, one .npz); a candidate
  run in another process scores every position against it with `Scorer`: the KL of the two distributions coarsened to
  {the K ids, one lumped tail}, `sum_S P log(P/Q) + p_tail log(p_tail/q_tail)`. Coarsening cannot raise a KL, so it
  is a lower bound. For the 27B, whose two runs cannot share the card.

What the lumping drops is `p_tail KL(P_tail || Q_tail)` per position. Measured on Qwen3-0.6B (code text, 4,096
positions, four formats, K = 64 .. 4096, against the exact KL): the dropped share of the total KL is estimated by
`sum_i max(p_tail_i, q_tail_i) kl_i / sum_i kl_i`, which needs no exact run; the estimate was between 0.82x and 3.8x the
true dropped share in all 16 (format, K) cases (conservative where it matters: at K = 4096, the K the reference picks
there, 0.6% to 1.3% estimated against 0.3% to 0.6% dropped), so it is the validity measure (`tail_lost_est`, bound
`TAIL_LOST_MAX`). A per-position tail bound (the design's `max(p_tail, q_tail) <= 1e-3`) is not usable: on real text a
third of the positions have a tail above 1e-3 at K = 4096, and the bound would reject every cell while the sparse KL
is within 1% of the exact one.

A position is the model position `p` of the fed token: its row predicts `ids[p + 1]` and its step reads keys 0 .. p.
The scored positions are `start .. len(ids) - 2`.

All row maths runs on the host in float64 from the fp32 logits (the device only selects top-K): the same function
of the same fp32 values, so bf16 against bf16 gives kl == 0 and top1 == 1 exactly, on any device.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from warpc.kvcache import FORMATS

K_LADDER = (64, 256, 1024, 4096)  # K is the smallest rung whose mean reference tail is within TAIL_BUDGET
TAIL_BUDGET = 5e-3  # K = 4096 gives a mean tail of 3.5e-3 on Qwen3-0.6B code text, 1024 gives 1.2e-2
TAIL_LOST_MAX = 0.02  # estimated share of the KL the lumped tail drops, above which a cell is invalid (see above)
TAIL_FLOOR = 1e-15  # below the resolution of `lse_S - lse` in float64: a tail mass is never taken smaller
BATCH = 32  # rows converted per device-to-host copy
BUCKET_EDGES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
SE_BLOCK, SE_RESAMPLES, SE_SEED = 64, 1000, 0

# Calibrated on Qwen3-0.6B, code text of the v0.12.0 tag, 4,096 and 16,384 positions, against the exact paired KL.
# Layer 1 is a catastrophe net: the 8-bit bars are 2.3x the worst measured (k8v8 at 4,096: kl 2.2e-3, top1 0.973); the
# 4-bit bars are the design's (k8v4w measures 1.0e-3 / 0.980 and passes; int4_rot measures 0.24 / 0.81 and fails).
CLASS_BARS = {"8bit": (5e-3, 0.95), "4bit": (0.05, 0.95)}  # (kl_mean max, top1 min)
# Layer 2: a run is bit-reproducible across processes, so these only absorb a numerics change; changing only ctx_max
# (another split geometry, the same positions) moved kl_mean by up to 2.2% and top1 by up to 0.42 pp (70-118 flips).
REL_TOLERANCE, KL_SLACK, TOP1_SLACK = 1.25, 5e-5, 0.01
# Ring check: every position below the first coded one is exact, yet the floor there is 5.5e-4 .. 7.1e-4 (fp16
# rotated rows against bf16), the same as the coded 8-bit region, so an 8-bit ring edge is invisible and only the
# exact-region bar and the spike bar below guard it; int4_rot's coded region is 100x its exact one.
RING_BOUNDS: dict[str, tuple[float, float]] = {
    "k8v8w": (2e-3, 3.0),  # measured (kl_exact_region, kl_boundary_spike): 6.9e-4..7.1e-4, 1.6
    "k8v4w": (2e-3, 3.0),  # 7.1e-4, 1.5
    "int4_rot": (2e-3, 3.0),  # 5.5e-4..5.8e-4, 0.27 (KL rises 100x past the edge, so the spike ratio is far below 1)
}
RING_WIDTH = 64  # moving mean width of the spike detector: 32 gave 2.0-2.2 on the 8-bit floor, 64 gives 1.5-1.6
RING_SPAN = (-64, 256)  # scanned around the first coded position
RING_NORM = (1024, 4096)  # positions whose median moving mean normalises the spike


def kv_class(fmt: str) -> str:
    """'bf16' (the self-check), '8bit' (k8v8, k8v8w, fp8) or '4bit' (any 4-bit side)."""
    f = FORMATS[fmt]
    if not f.quantized:
        return "bf16"
    return "8bit" if min(f.k_bits, f.v_bits) == 8 else "4bit"


def exact_end(fmt: str) -> int:
    """The first position whose step reads a coded key (0 for a format with no exact region): below it the sink and
    the ring of `window` newest rows cover every key. Key j is coded iff j >= sink and p - j >= window."""
    f = FORMATS[fmt]
    return f.sink + f.window if f.window else 0


# ---- the sparse reference --------------------------------------------------------------------------------------------


def _host(rows: torch.Tensor) -> torch.Tensor:
    return rows.detach().to("cpu", torch.float32).contiguous()


def _lse(rows: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(rows.double(), -1)


def _tail(lse: torch.Tensor, logits_s: torch.Tensor, vocab: int) -> torch.Tensor:
    """Probability mass outside the ids whose logits are `logits_s` [n, K], float64: 1 - exp(lse_S - lse) by expm1, so
    it keeps its relative accuracy down to TAIL_FLOOR; exactly 0 when S is the whole vocabulary."""
    if logits_s.shape[-1] >= vocab:
        return torch.zeros_like(lse)
    return (-torch.expm1(_lse(logits_s) - lse)).clamp_min(TAIL_FLOOR)


@dataclass
class Reference:
    """The reference distribution of positions `start .. start + n - 1`, sparse: [n, K] ids (descending logit) and fp32
    logits, the full-vocabulary log-sum-exp, the corpus token each row predicts and its logit, and the argmax."""

    start: int
    vocab: int
    ids: torch.Tensor  # int32 [n, K]
    logits: torch.Tensor  # float32 [n, K]
    lse: torch.Tensor  # float64 [n]
    target: torch.Tensor  # int32 [n]
    target_logit: torch.Tensor  # float32 [n]
    argmax: torch.Tensor  # int32 [n]
    meta: dict = field(default_factory=dict)  # free-form provenance (model, corpus slice, ...), JSON values
    greedy: list[int] | None = None  # the reference's greedy continuation after the same prefill, if recorded

    @property
    def n(self) -> int:
        return len(self.lse)

    @property
    def k(self) -> int:
        return self.ids.shape[1]

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.start, self.start + self.n)

    @property
    def p_tail(self) -> np.ndarray:
        return _tail(self.lse, self.logits, self.vocab).numpy()

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.ids, self.logits, self.lse, self.target))

    def save(self, path: Path) -> None:
        """One .npz (no pickle): the hand-off between the per-format processes of a model that cannot share the GPU."""
        meta = json.dumps({"start": self.start, "vocab": self.vocab, "meta": self.meta})
        arrays = {
            "ids": self.ids.numpy(),
            "logits": self.logits.numpy(),
            "lse": self.lse.numpy(),
            "target": self.target.numpy(),
            "target_logit": self.target_logit.numpy(),
            "argmax": self.argmax.numpy(),
            "meta": np.frombuffer(meta.encode(), dtype=np.uint8),
        }
        if self.greedy is not None:
            arrays["greedy"] = np.asarray(self.greedy, dtype=np.int64)
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path: Path) -> Reference:
        with np.load(path, allow_pickle=False) as z:
            head = json.loads(bytes(z["meta"]).decode())
            return cls(
                head["start"],
                head["vocab"],
                torch.from_numpy(z["ids"]),
                torch.from_numpy(z["logits"]),
                torch.from_numpy(z["lse"]),
                torch.from_numpy(z["target"]),
                torch.from_numpy(z["target_logit"]),
                torch.from_numpy(z["argmax"]),
                head["meta"],
                z["greedy"].tolist() if "greedy" in z else None,
            )


class ReferenceBuilder:
    """Accumulates the rows of a reference run (consecutive positions from `start`, `add` in order) and picks K at
    `finish`: the smallest rung of K_LADDER whose mean `p_tail` is within `budget` (the largest if none is)."""

    def __init__(self, start: int, target: Sequence[int], *, budget: float = TAIL_BUDGET, k: int | None = None):
        self.start, self.budget, self.k = start, budget, k
        self.target = torch.as_tensor(list(target), dtype=torch.int64)
        self.vocab = 0
        self._parts: list[tuple[torch.Tensor, ...]] = []
        self._rows = 0

    def add(self, rows: torch.Tensor) -> None:
        n, vocab = rows.shape
        if self._rows + n > len(self.target):
            raise ValueError("more rows than targets")
        self.vocab = vocab
        kmax = min(max(K_LADDER) if self.k is None else self.k, vocab)
        host = _host(rows)
        vals, idx = rows.detach().to(torch.float32).topk(kmax, -1)
        target = self.target[self._rows : self._rows + n]
        self._parts.append(
            (
                idx.to("cpu", torch.int32),
                vals.to("cpu"),
                _lse(host),
                target.to(torch.int32),
                host.gather(1, target[:, None])[:, 0],
                host.argmax(-1).to(torch.int32),
            )
        )
        self._rows += n

    def finish(self, meta: dict | None = None) -> Reference:
        if self._rows != len(self.target):
            raise ValueError(f"{self._rows} rows for {len(self.target)} targets")
        ids, logits, lse, target, target_logit, argmax = (torch.cat(c) for c in zip(*self._parts, strict=True))
        rungs = sorted({min(k, self.vocab) for k in K_LADDER}) if self.k is None else [min(self.k, self.vocab)]
        rungs = [k for k in rungs if k <= ids.shape[1]]
        k = next((k for k in rungs if float(_tail(lse, logits[:, :k], self.vocab).mean()) <= self.budget), rungs[-1])
        return Reference(
            self.start,
            self.vocab,
            ids[:, :k].contiguous(),
            logits[:, :k].contiguous(),
            lse,
            target,
            target_logit,
            argmax,
            meta or {},
        )


def reference_of(
    rows: torch.Tensor, target: Sequence[int], *, start: int = 0, budget: float = TAIL_BUDGET, k: int | None = None
) -> Reference:
    """A Reference from the full logits [n, V] of positions start .. start + n - 1 (tests and small runs)."""
    b = ReferenceBuilder(start, target, budget=budget, k=k)
    for i in range(0, len(rows), BATCH):
        b.add(rows[i : i + BATCH])
    return b.finish()


# ---- scoring a candidate ---------------------------------------------------------------------------------------------


@dataclass
class Scores:
    """Per scored position, aligned with the reference."""

    position: np.ndarray  # int64
    kl: np.ndarray  # float64, nats: KL(P_ref || Q), coarsened to {S, tail} (a lower bound) unless `k` is None
    p_tail: np.ndarray
    q_tail: np.ndarray
    top1: np.ndarray  # bool: argmax equal
    logp_target: np.ndarray  # the corpus token's log-probability under the reference / the candidate
    logq_target: np.ndarray
    k: int | None = None  # the reference's K; None: the exact full-vocabulary KL of a paired run (no tails)


class Scorer:
    """Scores the rows of a candidate run (consecutive positions from the reference's `start`, `add` in order)."""

    def __init__(self, ref: Reference):
        self.ref, self._at = ref, 0
        lp = ref.logits.double() - ref.lse[:, None]
        self._lp, self._p = lp, lp.exp()
        self._tp = _tail(ref.lse, ref.logits, ref.vocab)
        self._out: list[tuple[torch.Tensor, ...]] = []

    def add(self, rows: torch.Tensor) -> None:
        ref, a = self.ref, self._at
        n = len(rows)
        if a + n > ref.n:
            raise ValueError("more rows than the reference has positions")
        if rows.shape[1] != ref.vocab:
            raise ValueError(f"vocabulary {rows.shape[1]} against the reference's {ref.vocab}")
        host = _host(rows)
        lse = _lse(host)
        sl = slice(a, a + n)
        qs = host.gather(1, ref.ids[sl].long())
        lq = qs.double() - lse[:, None]
        kl = (self._p[sl] * (self._lp[sl] - lq)).sum(-1)
        tp, tq = self._tp[sl], _tail(lse, qs, ref.vocab)
        if ref.k < ref.vocab:
            kl = kl + tp * (tp.log() - tq.log())
        target = ref.target[sl].long()
        self._out.append(
            (
                kl,
                tp,
                tq,
                host.argmax(-1).to(torch.int32) == ref.argmax[sl],
                ref.target_logit[sl].double() - ref.lse[sl],
                host.gather(1, target[:, None])[:, 0].double() - lse,
            )
        )
        self._at += n

    def finish(self) -> Scores:
        if self._at != self.ref.n:
            raise ValueError(f"{self._at} rows scored of {self.ref.n}")
        kl, tp, tq, top1, lpt, lqt = (torch.cat(c).numpy() for c in zip(*self._out, strict=True))
        return Scores(self.ref.positions, kl, tp, tq, top1, lpt, lqt, self.ref.k)


def score(ref: Reference, rows: torch.Tensor) -> Scores:
    """Scores the full logits [n, V] of the reference's positions (tests and small runs)."""
    s = Scorer(ref)
    for i in range(0, len(rows), BATCH):
        s.add(rows[i : i + BATCH])
    return s.finish()


def exact_rows(p_rows: torch.Tensor, q_rows: torch.Tensor, target: Sequence[int], position: Sequence[int]) -> Scores:
    """The exact scores of full logit rows [n, V] of two runs (reference first), float64 on the host: KL over the whole
    vocabulary, no tails. What the sparse form bounds from below."""
    p, q = _host(p_rows), _host(q_rows)
    lp, lq = torch.log_softmax(p.double(), -1), torch.log_softmax(q.double(), -1)
    t = torch.as_tensor(list(target), dtype=torch.int64)[:, None]
    zero = np.zeros(len(p))
    return Scores(
        np.asarray(position, dtype=np.int64),
        (lp.exp() * (lp - lq)).sum(-1).numpy(),
        zero,
        zero,
        (p.argmax(-1) == q.argmax(-1)).numpy(),
        lp.gather(1, t)[:, 0].numpy(),
        lq.gather(1, t)[:, 0].numpy(),
    )


def kl_full(p_logits: torch.Tensor, q_logits: torch.Tensor) -> np.ndarray:
    return exact_rows(p_logits, q_logits, [0] * len(p_logits), range(len(p_logits))).kl


# ---- metrics ---------------------------------------------------------------------------------------------------------


def bucket_edges(exact: int, top: int) -> list[int]:
    """[0, exact), [exact, 1k), [1k, 2k), [2k, 4k), ... up to the first edge above `top` (no [0, exact) without one)."""
    edges = {0, *(e for e in BUCKET_EDGES if e > exact)}
    if exact:
        edges.add(exact)
    edges = sorted(edges)
    return [e for i, e in enumerate(edges) if i == 0 or edges[i - 1] <= top]


def kl_by_bucket(position: np.ndarray, kl: np.ndarray, exact: int) -> dict[str, float]:
    edges = bucket_edges(exact, int(position.max()))
    out: dict[str, float] = {}
    for lo, hi in zip(edges, [*edges[1:], None], strict=True):
        m = (position >= lo) & (True if hi is None else position < hi)
        if m.any():
            out[f"{lo}-{'' if hi is None else hi}"] = float(kl[m].mean())
    return out


def moving_mean(x: np.ndarray, width: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    return (c[width:] - c[:-width]) / width


def ring_check(position: np.ndarray, kl: np.ndarray, exact: int) -> dict[str, float | None]:
    """The ring-boundary check of a format whose first coded key enters at position `exact` (design 5.5):

    kl_exact_region: mean KL over positions < exact, where every key is read exact (rounding only);
    kl_boundary_spike: the largest moving mean (RING_WIDTH positions) over [exact + RING_SPAN[0], exact + RING_SPAN[1])
    divided by the median moving mean over RING_NORM. The mean below `exact` is the rounding floor (~0), so it cannot
    be the denominator. None where the scored positions do not cover the range (a window of a long context) or the
    format has no exact region; inf when the normaliser is 0 and the numerator is not."""
    none = {"kl_exact_region": None, "kl_boundary_spike": None}
    if not exact or len(position) == 0 or np.any(np.diff(position) != 1):
        return none
    first = int(position[0])

    def windows(lo: int, hi: int) -> np.ndarray | None:
        lo, hi = max(lo, first), min(hi, int(position[-1]) + 1)
        return None if hi - lo < RING_WIDTH else moving_mean(kl[lo - first : hi - first], RING_WIDTH)

    below = kl[position < exact]
    out = {"kl_exact_region": float(below.mean()) if len(below) else None, "kl_boundary_spike": None}
    near = windows(exact + RING_SPAN[0], exact + RING_SPAN[1])
    norm = windows(*RING_NORM)
    if near is not None and norm is not None:
        num, den = float(near.max()), float(np.median(norm))
        out["kl_boundary_spike"] = num / den if den > 0 else (0.0 if num <= 0 else math.inf)
    return out


def block_bootstrap_se(kl: np.ndarray) -> float:
    """Standard error of the mean KL: SE_BLOCK-position blocks resampled SE_RESAMPLES times, fixed seed."""
    nb = -(-len(kl) // SE_BLOCK)
    sums = np.add.reduceat(kl, np.arange(0, len(kl), SE_BLOCK))
    counts = np.diff([*range(0, len(kl), SE_BLOCK), len(kl)])
    pick = np.random.default_rng(SE_SEED).integers(0, nb, size=(SE_RESAMPLES, nb))
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(means.std(ddof=1))


def summarize(scores: Scores, fmt: str) -> dict:
    """The cell's quality metrics of format `fmt` (design 5.1); plain JSON values."""
    kl, pos, exact = scores.kl, scores.position, exact_end(fmt)
    tail = np.maximum(scores.p_tail, scores.q_tail)
    lost = float((tail * kl).sum() / kl.sum()) if kl.sum() > 0 else 0.0
    ppl_ref, ppl_fmt = math.exp(-scores.logp_target.mean()), math.exp(-scores.logq_target.mean())
    rms_dp = float(np.sqrt(np.mean((np.exp(scores.logq_target) - np.exp(scores.logp_target)) ** 2)))
    return {
        "format": fmt,
        "positions": len(kl),
        "k": scores.k,
        "kl_mean": float(kl.mean()),
        "kl_p95": float(np.percentile(kl, 95)),
        "kl_p99": float(np.percentile(kl, 99)),
        "kl_max": float(kl.max()),
        "top1": float(scores.top1.mean()),
        "kl_by_bucket": kl_by_bucket(pos, kl, exact),
        **ring_check(pos, kl, exact),
        "ppl_ref": ppl_ref,
        "ppl_fmt": ppl_fmt,
        "dppl": ppl_fmt - ppl_ref,
        "rms_dp": rms_dp,
        "se_kl": block_bootstrap_se(kl),
        "p_tail_mean": float(scores.p_tail.mean()),
        "q_tail_mean": float(scores.q_tail.mean()),
        "p_tail_max": float(scores.p_tail.max()),
        "q_tail_max": float(scores.q_tail.max()),
        "tail_lost_est": lost,
        "valid": lost <= TAIL_LOST_MAX,
    }


def greedy_first_diff(a: Sequence[int], b: Sequence[int]) -> int | None:
    """Index of the first token where two greedy continuations differ (the shorter's length if one is a prefix of the
    other); None when identical."""
    n = min(len(a), len(b))
    k = next((i for i in range(n) if a[i] != b[i]), n)
    return None if k == n and len(a) == len(b) else k


# ---- the gate: class bars, baseline, calibration ---------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    value: float | None
    bound: float | None
    op: str  # "<=" | ">=" | "=="
    layer: str  # "validity" | "class" | "baseline" | "ring"
    status: str  # "pass" | "fail" | "report": a report check is shown and never decides

    def __str__(self) -> str:
        return f"{self.status:6} {self.layer:8} {self.name} {self.value} {self.op} {self.bound}"


@dataclass(frozen=True)
class Judgement:
    valid: bool  # False: the sparse lower bound is unsupported, no other check of the cell means anything
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return self.valid and all(c.status != "fail" for c in self.checks)

    @property
    def failed(self) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.status == "fail")


def _check(name, value, bound, op, layer) -> Check:
    if value is None or bound is None:
        return Check(name, value, bound, op, layer, "report")
    ok = {"<=": value <= bound, ">=": value >= bound, "==": value == bound}[op]
    return Check(name, value, bound, op, layer, "pass" if ok else "fail")


def judge(metrics: dict, fmt: str, baseline: dict | None = None) -> Judgement:
    """Gate vs report (design 5.4). Gated: validity (tail mass), Layer 1 (the class bars; for bf16 the exact self-check
    kl == 0, top1 == 1), the ring bounds, and, with a stored `baseline` ({"kl_mean", "top1"}), Layer 2 (kl_mean <=
    REL_TOLERANCE x baseline + KL_SLACK, top1 >= baseline - TOP1_SLACK). Without a baseline the cell is the first
    measurement of its (model, format, context): Layer 2 is shown as report and Layer 1 still decides. Everything else
    (kl_p95/p99/max, dppl, rms_dp, se_kl, buckets, greedy first diff) is never judged."""
    checks = [_check("tail_lost_est", metrics["tail_lost_est"], TAIL_LOST_MAX, "<=", "validity")]
    cls = kv_class(fmt)
    if cls == "bf16":
        checks += [
            _check("kl_mean", metrics["kl_mean"], 0.0, "==", "class"),
            _check("top1", metrics["top1"], 1.0, "==", "class"),
        ]
    else:
        kl_max, top1_min = CLASS_BARS[cls]
        checks += [
            _check("kl_mean", metrics["kl_mean"], kl_max, "<=", "class"),
            _check("top1", metrics["top1"], top1_min, ">=", "class"),
        ]
        have = baseline is not None
        base_kl = REL_TOLERANCE * baseline["kl_mean"] + KL_SLACK if have else None
        base_top1 = baseline["top1"] - TOP1_SLACK if have else None
        checks += [
            _check("kl_mean", metrics["kl_mean"], base_kl, "<=", "baseline"),
            _check("top1", metrics["top1"], base_top1, ">=", "baseline"),
        ]
        bound = RING_BOUNDS.get(fmt)
        if bound is not None:
            checks += [
                _check("kl_exact_region", metrics.get("kl_exact_region"), bound[0], "<=", "ring"),
                _check("kl_boundary_spike", metrics.get("kl_boundary_spike"), bound[1], "<=", "ring"),
            ]
    return Judgement(checks[0].status != "fail", tuple(checks))


# ---- the engine side: logits through `Model` -------------------------------------------------------------------------


def iter_logits(model, ids: Sequence[int], start: int = 0) -> Iterator[torch.Tensor]:
    """Teacher-forced logits of positions start .. len(ids) - 2 in chunks of up to BATCH rows (float32 [n, V] on the
    device, valid until the next chunk). The cache is rebuilt from empty: `ids[: start + 1]` is one `append` (the
    model's own prefill plan, windowed where the runtime has window kernels, and its head step gives the first
    row), then one T = 1 step per further position, which is how a decode reads the cache."""
    if not 0 <= start < len(ids) - 1:
        raise ValueError(f"start {start} outside 0 .. {len(ids) - 2}")
    model.reset()
    model.append(list(ids[: start + 1]))
    first = model.logits
    buf = torch.empty((BATCH, first.shape[0]), dtype=torch.float32, device=first.device)
    buf[0].copy_(first)
    n = 1
    for p in range(start + 1, len(ids) - 1):
        model.step(ids[p])
        buf[n].copy_(model.logits)
        n += 1
        if n == BATCH:
            yield buf[:n]
            n = 0
    if n:
        yield buf[:n]


def teacher_forced_logits(model, ids: Sequence[int], start: int = 0) -> torch.Tensor:
    """fp32 [len(ids) - 1 - start, V] on the host: the whole matrix, 0.6 MB a position (a 4,096-position run on a
    152k vocabulary is 2.5 GB). `reference_run` and `score_run` stream instead."""
    return torch.cat([c.to("cpu", copy=True) for c in iter_logits(model, ids, start)])


def reference_run(model, ids: Sequence[int], start: int = 0, *, meta: dict | None = None, **kw) -> Reference:
    """The reference pass (model compiled with kv=bf16): positions start .. len(ids) - 2 against corpus `ids`."""
    b = ReferenceBuilder(start, ids[start + 1 :], **kw)
    for chunk in iter_logits(model, ids, start):
        b.add(chunk)
    return b.finish(meta)


def score_run(model, ids: Sequence[int], ref: Reference) -> Scores:
    """The candidate pass (model compiled with another kv format, same ctx_max and prompt) against `ref`."""
    if list(ids[ref.start + 1 :]) != ref.target.tolist():
        raise ValueError("the corpus ids differ from the reference's")
    s = Scorer(ref)
    for chunk in iter_logits(model, ids, ref.start):
        s.add(chunk)
    return s.finish()


def paired_run(ref_model, cand_model, ids: Sequence[int], start: int = 0) -> Scores:
    """The exact full-vocabulary KL of two models run in lockstep over the same ids (both on the device, nothing
    stored): for a model that fits twice, whatever its tail mass."""
    out, at = [], start
    for p, q in zip(iter_logits(ref_model, ids, start), iter_logits(cand_model, ids, start), strict=True):
        n = len(p)
        out.append(exact_rows(p, q, ids[at + 1 : at + 1 + n], range(at, at + n)))
        at += n
    return Scores(*(np.concatenate([getattr(o, f) for o in out]) for f in Scores.__dataclass_fields__ if f != "k"))


def greedy_run(model, ids: Sequence[int], n: int = 256) -> list[int]:
    """The first n greedy tokens after prefilling `ids` (the reference's `greedy`, and a candidate's to compare)."""
    return model.generate(list(ids), n)


def ctx_max_of(n_ids: int) -> int:
    """`round_up_256(prompt + 512)` (bench design 1.2.5): room for the 256 greedy tokens of the first-diff check."""
    return -(-(n_ids + 512) // 256) * 256


def measure(
    source, ids: Sequence[int], formats: Sequence[str], *, start: int = 0, mode: str = "paired", greedy: int = 256
) -> dict[str, dict]:
    """Quality metrics of every format in `formats` against bf16 KV of the same source, backend and ctx_max, over the
    corpus `ids` (positions start .. len(ids) - 2), one model alive at a time in "sparse" mode and the reference plus
    one candidate in "paired" mode (exact KL; a model that fits twice). `greedy_first_diff_256` is the first
    differing token of a `greedy`-token continuation of all of `ids`; "bf16" among the formats is the self-check
    (kl == 0, top1 == 1). Each result is `summarize`d, ready for `judge`."""
    import gc

    import warpc

    if mode not in ("paired", "sparse"):
        raise ValueError(f"mode {mode!r}: paired | sparse")
    ctx_max = ctx_max_of(len(ids))

    def compiled(fmt):
        return warpc.compile(source, ctx_max=ctx_max, kv=fmt)

    def free():
        gc.collect()
        torch.cuda.empty_cache()

    out: dict[str, dict] = {}
    ref_model = compiled("bf16")
    ref_greedy = greedy_run(ref_model, ids, greedy) if greedy else None
    ref = None
    if mode == "sparse":
        ref = reference_run(ref_model, ids, start)
        del ref_model
        free()
    for fmt in formats:
        model = compiled(fmt)
        scores = paired_run(ref_model, model, ids, start) if mode == "paired" else score_run(model, ids, ref)
        m = summarize(scores, fmt)
        if greedy:
            m["greedy_first_diff_256"] = greedy_first_diff(ref_greedy, greedy_run(model, ids, greedy))
        out[fmt] = m | {"ctx_max": ctx_max, "start": start}
        del model
        free()
    return out


# ---- the result cell -------------------------------------------------------------------------------------------------

UNITS = {
    "kl_mean": "nats",
    "kl_p95": "nats",
    "kl_p99": "nats",
    "kl_max": "nats",
    "se_kl": "nats",
    "kl_exact_region": "nats",
    "kl_boundary_spike": "ratio",
    "top1": "fraction",
    "tail_lost_est": "fraction",
    "p_tail_mean": "fraction",
    "q_tail_mean": "fraction",
    "p_tail_max": "fraction",
    "q_tail_max": "fraction",
    "ppl_ref": "ppl",
    "ppl_fmt": "ppl",
    "dppl": "ppl",
    "rms_dp": "prob",
    "greedy_first_diff_256": "tokens",
}


def to_cell(key: str, metrics: dict, judgement: Judgement, *, workload: dict, extra: dict | None = None):
    """The result `schema.Cell` of a scored format: `ok`, or `fail` with the code `invalid_tail` (the sparse bound is
    unsupported) or `quality_gate` and the failed checks as the reason. Scalars become metrics (deterministic: one
    sample); what has no finite scalar value (a ring check outside its positions, a first diff of None, an infinite
    spike) and the structures go to `extra`."""
    from . import schema

    out = {
        n: schema.Metric(float(metrics[n]), u, [float(metrics[n])]) for n, u in UNITS.items() if _finite(metrics.get(n))
    }
    rest = {k: _json(v) for k, v in metrics.items() if k not in out and k not in ("format", "valid")}
    rest["mode"] = "exact" if metrics["k"] is None else "sparse"
    rest["checks"] = [str(c) for c in judgement.checks]
    if judgement.ok:
        return schema.Cell(key, "ok", workload=workload, metrics=out, extra=rest | (extra or {}))
    code = "quality_gate" if judgement.valid else "invalid_tail"
    reason = "; ".join(f"{c.name} {c.value:.4g} {c.op} {c.bound:.4g} ({c.layer})" for c in judgement.failed)
    return schema.Cell(key, "fail", code, reason, workload, out, extra=rest | (extra or {}))


def _finite(v) -> bool:
    return isinstance(v, int | float) and math.isfinite(v)


def _json(v):
    return str(v) if isinstance(v, float) and not math.isfinite(v) else v
