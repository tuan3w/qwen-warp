"""Statistics of paired, interleaved visits. Stdlib only: the visits of an `ab` session are paired (adjacent in time, so
a drift cancels inside the pair), which is why the tests here are on pair ratios and not rank tests of two samples."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

RESAMPLES = 10_000
CONFIDENCE = 0.95
SEED = 0


def median(values: Sequence[float]) -> float:
    return statistics.median(values)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-quantile (0..1) by linear interpolation between order statistics."""
    if not values:
        raise ValueError("percentile of nothing")
    ordered = sorted(values)
    x = q * (len(ordered) - 1)
    lo = math.floor(x)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (x - lo)


def paired_ratios(a: Sequence[float], b: Sequence[float], *, higher_is_better: bool = True) -> list[float]:
    """B against A for every pair, read so that a value above 1 means B is better: `b / a` for a rate, `a / b` for a
    time."""
    if len(a) != len(b):
        raise ValueError(f"{len(a)} A visits against {len(b)} B visits")
    if any(x <= 0 for x in (*a, *b)):
        raise ValueError("a ratio needs positive values")
    return [y / x if higher_is_better else x / y for x, y in zip(a, b, strict=True)]


def bootstrap_ci(
    values: Sequence[float], *, resamples: int = RESAMPLES, confidence: float = CONFIDENCE, seed: int = SEED
) -> tuple[float, float]:
    """Percentile bootstrap interval of the median of `values`. Seeded stdlib `random`: the same file gives the same
    interval everywhere. With few values the support is the order statistics, so the interval is coarse but never
    narrower than the data."""
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap of nothing")
    rng = random.Random(seed)
    pick = rng.choices
    mid, odd = n // 2, n % 2
    medians = []
    for _ in range(resamples):
        s = sorted(pick(values, k=n))
        medians.append(s[mid] if odd else (s[mid - 1] + s[mid]) / 2)
    medians.sort()
    tail = (1 - confidence) / 2
    return medians[min(int(tail * resamples), resamples - 1)], medians[min(int((1 - tail) * resamples), resamples - 1)]


def sign_test(below: int, n: int) -> float:
    """One-sided exact p of `below` or more of `n` pairs on the same side when each side has probability 1/2: 1/32 for
    5 of 5."""
    if not 0 <= below <= n:
        raise ValueError(f"{below} of {n}")
    return sum(math.comb(n, k) for k in range(below, n + 1)) / 2**n


def mad_outliers(values: Sequence[float], k: float = 3.0) -> list[bool]:
    """Which values lie more than `k` MADs (median absolute deviations, unscaled) from the median. A zero MAD
    (identical values) flags nothing."""
    centre = statistics.median(values)
    mad = statistics.median(abs(v - centre) for v in values)
    if mad == 0:
        return [False] * len(values)
    return [abs(v - centre) > k * mad for v in values]
