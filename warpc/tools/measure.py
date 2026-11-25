"""The decode-rate protocol shared by the warpc and hand-engine sides of bench.py.

One prefill per prompt length, then one warm-up round and `repeats` timed rounds of n = tokens - 1 greedy tokens, each
round continuing the previous one (a 64k-token prefill takes minutes, so it is not repeated per round; round r decodes
at context L + 1 + r * n). The rate of a round is n / (t_last - t_first): t_first when the previous token is on the host
(prefill or the previous round returned), t_last when the round's last token is. This is the inter-arrival metric vLLM's
engine-core timestamps give (vllm_worker.py bench).

While the rounds run, nvidia-smi samples SM clock, power and temperature every 200 ms. A last round times every
single-token replay with CUDA events: the round's wall time minus the GPU time of its replays is the host overhead of
the decode loop.
"""

from __future__ import annotations

import statistics
import subprocess
import time
from collections.abc import Callable


class Sampler:
    """nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu -lms 200 for the duration of a `with` block."""

    def __enter__(self):
        query = "--query-gpu=clocks.sm,power.draw,temperature.gpu"
        command = ["nvidia-smi", query, "--format=csv,noheader,nounits", "-lms", "200"]
        self._proc = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self._proc.terminate()
        out, _ = self._proc.communicate()
        rows = []
        for line in out.splitlines():
            try:
                rows.append([float(v) for v in line.split(",")])
            except ValueError:
                continue
        cols = list(zip(*rows, strict=True)) if rows else [(), (), ()]
        self.result = {
            name: statistics.median(c) if c else None
            for name, c in zip(("sm_mhz", "power_w", "temp_c"), cols, strict=True)
        }
        self.result["samples"] = len(rows)


def rounds(decode: Callable[[], int], repeats: int, replay_events: Callable[[], tuple[float, int]] | None) -> dict:
    """`decode()` decodes a round of tokens and returns how many, with the last one on the host; `replay_events()`
    decodes a round timing each replay with CUDA events and returns (their summed GPU seconds, tokens)."""
    rates, walls, counts = [], [], []
    with Sampler() as s:
        for _ in range(repeats + 1):
            t0 = time.perf_counter()
            counts.append(decode())
            walls.append(time.perf_counter() - t0)
            rates.append(counts[-1] / walls[-1])
    result = {"rates": rates[1:], "tok_s": statistics.median(rates[1:]), "clocks": s.result}
    if replay_events is not None:
        gpu_s, n = replay_events()
        per_token = statistics.median(w / c for w, c in zip(walls[1:], counts[1:], strict=True))
        result["host_overhead_us"] = (per_token - gpu_s / n) * 1e6
    return result


def replay_seconds(step: Callable[[], None], n: int) -> float:
    """n calls of `step` (one graph replay each) as a decode loop runs them, with a CUDA event pair around each and
    one synchronisation at the end: the summed GPU time of the replays, without the host gaps between them."""
    import torch

    pairs = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        step()
        b.record()
        pairs.append((a, b))
    torch.cuda.synchronize()
    return sum(a.elapsed_time(b) for a, b in pairs) / 1e3
