"""The fairness contract: what every number this suite emits means and which comparisons are refused.

It is enforced here and in `gates.py`, not by convention: three clocks behind one "tok/s" column, a baseline row
measured at 1800-1875 MHz against a warpc row at 1755 MHz, and 1.3% of drift on an unchanged path are what convention
produced.

Scope. Single stream only: one sequence, concurrency 1. Cold compile is recorded (`compile_s`), never gated. KV-format
quality is measured on warpc against warpc-bf16 only.

Metrics (everything else is derived or diagnostic). Two layers: ENGINE (the compute loop with the host launch and sync
gaps the engine's design incurs) and PROCESS (memory). A client-layer figure (HTTP wall, `LLM.generate` wall) is stored
as `extra.*_client` and never in a headline column or a ratio.

  ttft_s (cold)       Wall time from the adapter call `turn(ids, n, reuse=False)` to the first generated token being
                      host-visible, after every cache was dropped (`cached == 0`). Includes the whole prefill,
                      restore/spill/snapshot work and the head step; queue time is in. Excludes load, compile, graph
                      capture, tokenisation, detokenisation, HTTP/IPC. Under MTP it includes the first delivery's extra
                      cycles, so MTP TTFTs are compared only with MTP TTFTs.
  ttft_s (resumed)    The same call with `reuse=True` and `cached > 0`; reported with `cached / len(ids)`.
  prefill_tok_s       `len(ids) / ttft_s(cold)`.
  decode_tok_s        `(n_new - n_first) / (t_last - t_first)`: `t_first` is the host-visible time of the first
                      delivery (the first host sync that returns any generated token), `n_first` the number of tokens
                      that delivery made visible, `t_last` the host-visible time of the last delivery. Plain decode has
                      `n_first = 1`. An engine that syncs once per group of cycles makes ~24 tokens visible at
                      `t_first` at no measured cost; counting them would credit +10% at `n_new = 256`.
  tpot_s              `1 / decode_tok_s`, never stored.
  tokens_per_cycle    `emitted / cycles` of the timed rounds (the first delivery excluded); a cycle is one verify step
                      of a draft chain. `cycle_ms = 1000 * tokens_per_cycle / decode_tok_s`, for every engine alike.
  device_gib_peak     Peak of the per-process device memory (NVML / nvidia-smi `used_memory`) summed over the unit's
                      process tree, sampled every 200 ms from process start. CUDA context, allocator pool and graphs
                      are in; other users' processes are out. Without per-process data: whole-GPU `memory.used` minus
                      the idle baseline taken before launch.
  host_rss_gib        Peak resident set of the process tree.
  torch_peak_gib, estimate_gib   Diagnostics, never compared across engines.

Like-with-like rules (`compare` refuses a comparison that violates one):

  1. Same bytes in: every cell records `workload.ids_sha`; two cells whose workload differs are not compared.
  2. Same weights: engines in one comparison load the same checkpoint files; the stamp holds a cheap fingerprint.
  3. Same decode: greedy, ignore-EOS, fixed `n_new`. Rounds of one cell start from an identical state and must emit
     identical ids on engines that claim determinism (warpc, hand), else the cell fails `nondeterministic`.
  4. Same context at every round: a round restores the engine to the end of the prompt, so round r decodes at the
     positions of round 0 (decode step time grows ~0.4% per 1k context; carrying a round's tokens into the next
     would bias the median).
  5. `ctx_max = round_up_256(prompt_len + 512)` for every cell that has a prompt (matrix cells take it explicitly).
  6. Head, KV format, backend, host tables, window, MTP k, cache mode and engine `opts` are in the cell key; no
     tool-level defaults.
  7. Versions are recorded; a baseline from a dirty tree is refused.
  8. Warmup before timing: per process one full-length discarded round; per cell one discarded plus
     `TIMED_ROUNDS` timed rounds; an `ab` session starts with one discarded pair (a visit per arm).
  9. A speed claim "X is faster than Y" is a paired comparison of visits taken alternately in one session
     (A B | B A | A B ..., counterbalanced); a single-visit cross-engine number is `unpaired` and enters no ratio.
 10. Clocks are logged (SM and memory MHz p10/p50/p90, power, temperature, throttle reasons) and gated only where the
     comparison is same-engine: the two visits of a pair must agree within `CLOCK_BOUND_SAME_ENGINE`, else the pair
     is dropped and counted. Cross-engine pairs are never dropped for their clocks (each engine's clock is a
     consequence of its own power draw). A baseline compare is `inconclusive(clock)` beyond
     `CLOCK_BOUND_CROSS_SESSION`. A cell with hardware-slowdown, thermal or power-brake throttling on more than
     `THROTTLE_FLAG_FRAC` of its samples is flagged `throttled` (the power cap alone is logged: see `_REASON_NAMES`).
 11. A shared GPU is a measured variable: a foreign process of at least `CONTEND_MIB` before a unit is retried with
     backoff and then `skip(gpu busy)`; one appearing during a unit makes the cell `contended` and drops it from every
     statistic. Ratios only form from visits of one session.
 12. Forged positions (timing over a zero cache) are a labelled cell family, never compared with real-prefill cells.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import stats
from .schema import Stamp

if TYPE_CHECKING:
    from .engines import Turn

WARMUP_ROUNDS = 1
TIMED_ROUNDS = 5  # the repo's shipped 1 + 5
SAMPLE_PERIOD_S = 0.2
CONTEND_MIB = 256  # a foreign process below this is a desktop client (25 and 163 MiB measured), not a competing job
THROTTLE_FLAG_FRAC = 0.10
# Measured on seven A/A sessions of Qwen3-0.6B (Q4_0 x5, bf16 x2; 126 pairs): |SM_B - SM_A| / SM_A of a pair's median
# SM clocks has p99 1.65% (Q4_0 alone 0.81%, bf16 alone 1.65%; the clock moves in 15 MHz steps), x 1.5 = 2.5%.
CLOCK_BOUND_SAME_ENGINE = 0.025
CLOCK_BOUND_CROSS_SESSION = (
    0.03  # a placeholder: no cross-session variation was seen (every session's median 1830-1860)
)
BUSY_WAIT_S = 600
BUSY_POLL_S = (5, 10, 20, 30, 60)  # backoff between idle checks; the last repeats

# nvidia-smi clocks_event_reasons.active bits. `throttle_frac` counts the hardware slowdown, thermal and power-brake
# reasons only: a power-limited card asserts `sw_power_cap` on every sample of a load (100% of the samples of a
# 269 W decode under a 350 W limit, measured), so it is logged per reason and flags nothing.
_REASON_NAMES = {
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x20: "sw_thermal",
    0x40: "hw_thermal",
    0x80: "power_brake",
    0x1: "idle",
    0x2: "applications_clocks",
    0x10: "sync_boost",
}
_THROTTLE_MASK = 0x8 | 0x20 | 0x40 | 0x80


def decode_tok_s(n_new: int, n_first: int, t_first: float, t_last: float) -> float:
    """`(n_new - n_first) / (t_last - t_first)`; see the module docstring."""
    if not 0 < n_first < n_new:
        raise ValueError(f"n_first {n_first} must be in 1 .. n_new - 1 = {n_new - 1}")
    if t_last <= t_first:
        raise ValueError(f"t_last {t_last} is not after t_first {t_first}")
    return (n_new - n_first) / (t_last - t_first)


def decode_rate(deliveries: Sequence[tuple[float, int]]) -> float:
    """`decode_tok_s` of a stream of deliveries (host-visible time, tokens made visible), the first one included."""
    if len(deliveries) < 2:
        raise ValueError("a decode rate needs the first delivery and at least one more")
    n_first = deliveries[0][1]
    return decode_tok_s(sum(n for _, n in deliveries), n_first, deliveries[0][0], deliveries[-1][0])


def cycle_ms(tokens_per_cycle: float, rate: float) -> float:
    return 1000 * tokens_per_cycle / rate


def prefill_tok_s(n_prompt: int, ttft_s: float) -> float:
    return n_prompt / ttft_s


# ---- the GPU sampler


def _parse_time(text: str) -> float:
    return datetime.strptime(text.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # "[N/A]", "[Not Supported]"


def _parent(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None


def in_tree(pid: int, root: int) -> bool:
    """Whether `pid` is `root` or one of its descendants (a vLLM engine core is a child of the unit)."""
    seen = 0
    while pid and seen < 64:
        if pid == root:
            return True
        pid = _parent(pid) or 0
        seen += 1
    return False


def foreign(apps: Mapping[int, int], is_own: Callable[[int], bool]) -> dict[int, int]:
    """The processes of `apps` (pid -> MiB) that are not the unit's and are big enough to compete for the GPU."""
    return {pid: mib for pid, mib in apps.items() if mib >= CONTEND_MIB and not is_own(pid)}


class Accumulator:
    """What the sampler collects, fed line by line so that it can be tested without a GPU. `is_own(pid)`: whether a
    compute process belongs to the unit's process tree."""

    def __init__(self, is_own: Callable[[int], bool]):
        self.is_own = is_own
        self.gpu: list[tuple[float, float | None, float | None, float | None, float | None, int, float | None]] = []
        self.own_mib: list[tuple[float, float]] = []  # (time, summed MiB of the unit's processes)
        self.foreign_pids: dict[int, float] = {}  # pid -> first time seen
        self._lock = threading.Lock()

    def feed_gpu(self, line: str) -> None:
        """`timestamp, clocks.sm, clocks.mem, power.draw, temperature.gpu, clocks_event_reasons.active, memory.used`"""
        f = [x.strip() for x in line.split(",")]
        if len(f) != 7:
            return
        try:
            t, reasons = _parse_time(f[0]), int(f[5], 16)
        except ValueError:
            return
        with self._lock:
            self.gpu.append((t, _number(f[1]), _number(f[2]), _number(f[3]), _number(f[4]), reasons, _number(f[6])))

    def feed_apps(self, t: float, apps: Mapping[int, int]) -> None:
        """One sample of `nvidia-smi --query-compute-apps=pid,used_memory`: pid -> MiB."""
        own = sum(mib for pid, mib in apps.items() if self.is_own(pid))
        with self._lock:
            if own:
                self.own_mib.append((t, own))
            for pid in foreign(apps, self.is_own):
                self.foreign_pids.setdefault(pid, t)

    def summary(self, since: float = 0.0, until: float = float("inf")) -> dict[str, Any]:
        """The `clocks` dict of a cell for the window [since, until]: SM MHz [p10, p50, p90], memory MHz, power,
        temperature, the fraction of samples throttled (`_THROTTLE_MASK`), the fraction per reason, foreign pids that
        appeared in the window, the number of samples."""
        with self._lock:
            rows = [r for r in self.gpu if since <= r[0] <= until]
            foreigners = sorted(p for p, t in self.foreign_pids.items() if since <= t <= until)
        sm = [r[1] for r in rows if r[1] is not None]
        throttled = sum(1 for r in rows if r[5] & _THROTTLE_MASK)
        reasons = {name: sum(1 for r in rows if r[5] & bit) / len(rows) for bit, name in _REASON_NAMES.items() if rows}

        def med(i: int) -> float | None:
            xs = [r[i] for r in rows if r[i] is not None]
            return statistics.median(xs) if xs else None

        return {
            "sm_mhz": [stats.percentile(sm, q) for q in (0.1, 0.5, 0.9)] if sm else None,
            "mem_mhz": med(2),
            "power_w": med(3),
            "temp_c": med(4),
            "throttle_frac": throttled / len(rows) if rows else None,
            "reasons": {name: frac for name, frac in reasons.items() if frac},
            "foreign_pids": len(foreigners),
            "samples": len(rows),
        }

    def device_gib_peak(self, since: float = 0.0, until: float = float("inf")) -> float | None:
        """Peak of the unit's process-tree device memory; without per-process data the rise of the whole GPU's
        `memory.used` over its first sample."""
        with self._lock:
            own = [m for t, m in self.own_mib if since <= t <= until]
            used = [r[6] for r in self.gpu if r[6] is not None]
            window = [r[6] for r in self.gpu if since <= r[0] <= until and r[6] is not None]
        if own:
            return max(own) / 1024
        if used and window:
            return (max(window) - used[0]) / 1024
        return None


_GPU_QUERY = "timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,{reasons},memory.used"
_REASONS_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")  # driver-dependent name


@cache
def _reasons_field() -> str | None:
    for name in _REASONS_FIELDS:
        probe = subprocess.run(
            ["nvidia-smi", f"--query-gpu={name}", "--format=csv,noheader"], capture_output=True, text=True
        )
        if probe.returncode == 0:
            return name
    return None


def compute_apps() -> dict[int, int] | None:
    """pid -> MiB of every compute process on the GPU now; None without nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    apps = {}
    for line in out.splitlines():
        pid, _, mib = line.partition(",")
        if pid.strip().isdigit() and _number(mib.strip()) is not None:
            apps[int(pid)] = int(float(mib))
    return apps


class Sampler:
    """nvidia-smi streaming at `SAMPLE_PERIOD_S` for the duration of a `with` block: SM and memory clock, power,
    temperature, throttle reasons, and the per-process device memory of every compute process, split into the unit's
    process tree (`device_gib_peak`) and foreign processes (`clocks.foreign_pids`, contention).

        with Sampler() as s:
            t0 = s.mark()
            ...
            cell.clocks = s.summary(t0)

    A host without nvidia-smi gets an empty summary (samples = 0)."""

    def __init__(self, pid: int | None = None):
        root = os.getpid() if pid is None else pid
        self.acc = Accumulator(lambda p: in_tree(p, root))
        self._procs: list[subprocess.Popen] = []
        self._threads: list[threading.Thread] = []

    @staticmethod
    def mark() -> float:
        return time.time()

    def __enter__(self) -> Sampler:
        reasons = _reasons_field()
        if reasons is None:
            return self
        period = str(int(SAMPLE_PERIOD_S * 1000))
        gpu = ["nvidia-smi", f"--query-gpu={_GPU_QUERY.format(reasons=reasons)}", "--format=csv,noheader,nounits"]
        apps = ["nvidia-smi", "--query-compute-apps=timestamp,pid,used_memory", "--format=csv,noheader,nounits"]
        for command, reader in ((gpu, self._read_gpu), (apps, self._read_apps)):
            proc = subprocess.Popen([*command, "-lms", period], stdout=subprocess.PIPE, text=True)
            thread = threading.Thread(target=reader, args=(proc,), daemon=True)
            thread.start()
            self._procs.append(proc)
            self._threads.append(thread)
        return self

    def __exit__(self, *exc: object) -> None:
        for proc in self._procs:
            proc.terminate()
        for proc, thread in zip(self._procs, self._threads, strict=True):
            proc.wait()
            thread.join(timeout=2)

    def _read_gpu(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            self.acc.feed_gpu(line)

    def _read_apps(self, proc: subprocess.Popen) -> None:
        """Rows with the same timestamp are one sample; a sample with no compute process prints no row."""
        stamp, rows = None, {}
        for line in proc.stdout:
            f = [x.strip() for x in line.split(",")]
            if len(f) != 3:
                continue
            try:
                t, pid, mib = _parse_time(f[0]), int(f[1]), int(float(f[2]))
            except ValueError:
                continue
            if stamp is not None and t != stamp:
                self.acc.feed_apps(stamp, rows)
                rows = {}
            stamp = t
            rows[pid] = mib
        if rows and stamp is not None:
            self.acc.feed_apps(stamp, rows)

    def summary(self, since: float = 0.0, until: float = float("inf")) -> dict[str, Any]:
        return self.acc.summary(since, until)

    def device_gib_peak(self, since: float = 0.0, until: float = float("inf")) -> float | None:
        return self.acc.device_gib_peak(since, until)

    @property
    def foreign_pids(self) -> list[int]:
        return sorted(self.acc.foreign_pids)


def gpu_busy() -> dict[int, int]:
    """The competing processes on the GPU now (see `foreign`; this process and its children are not foreign)."""
    root = os.getpid()
    return foreign(compute_apps() or {}, lambda p: in_tree(p, root))


def wait_idle(timeout_s: float = BUSY_WAIT_S, sleep: Callable[[float], None] = time.sleep) -> dict[int, int]:
    """Block with backoff until no competing process is on the GPU; the busy set that is left after `timeout_s`
    (empty: the GPU is idle)."""
    waited, step = 0.0, 0
    while busy := gpu_busy():
        if waited >= timeout_s:
            return busy
        delay = min(BUSY_POLL_S[min(step, len(BUSY_POLL_S) - 1)], timeout_s - waited)
        sleep(delay)
        waited += delay
        step += 1
    return {}


# ---- rounds


@dataclass
class Rounds:
    """The timed rounds of one cell: per-round samples in measurement order."""

    decode_tok_s: list[float] = field(default_factory=list)
    ttft_s: list[float] = field(default_factory=list)
    cached: list[int] = field(default_factory=list)
    ids: list[int] = field(default_factory=list)  # the first timed round's
    stable: bool = True  # every round emitted the same ids
    n_first: int = 1
    extra: list[dict[str, Any]] = field(default_factory=list)


def rounds(turn: Callable[[], Turn], repeats: int = TIMED_ROUNDS, warmup: int = WARMUP_ROUNDS) -> Rounds:
    """`warmup` discarded rounds then `repeats` timed ones of `turn()` (each a reuse=True turn of the identical prompt,
    so round r starts from the state round 0 did)."""
    for _ in range(warmup):
        turn()
    out = Rounds()
    for r in range(repeats):
        t = turn()
        if r == 0:
            out.ids, out.n_first = list(t.ids), t.n_first
        out.stable &= t.ids == out.ids and t.n_first == out.n_first
        out.decode_tok_s.append(t.decode_tok_s)
        out.ttft_s.append(t.first_s)
        out.cached.append(t.cached)
        out.extra.append(t.extra)
    return out


def replay_seconds(step: Callable[[], None], n: int) -> float:
    """n calls of `step` (one graph replay each) as a decode loop runs them, with a CUDA event pair around each and
    one synchronisation at the end: the summed GPU time of the replays without the host gaps between them."""
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


def host_overhead_us(wall_per_token_s: float, gpu_s: float, n: int) -> float:
    """`extra.host_overhead_us = (median wall per token - GPU seconds per token) * 1e6`: the host's share of a step."""
    return (wall_per_token_s - gpu_s / n) * 1e6


# ---- the stamp


def _run(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        r = subprocess.run(command, capture_output=True, text=True, cwd=cwd, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def machine_key(name: str | None, power_limit_w: float | None) -> str | None:
    """`NVIDIA GeForce RTX 3090` at 350 W -> `rtx3090-350w`: the baseline file of the GPU class."""
    if not name or power_limit_w is None:
        return None
    short = re.sub(r"\W+", "", re.sub(r"(?i)\b(nvidia|geforce)\b", "", name)).lower()
    return f"{short}-{round(power_limit_w)}w"


def fingerprint(path: Path) -> str:
    """A cheap identity of a checkpoint: `config.json` plus the relative path and size of every weight file for an HF
    directory (multi-GB files are not hashed), the name, size and first MiB (the header) for a GGUF file."""
    path = Path(path)
    h = hashlib.sha256()
    if path.is_dir():
        config = path / "config.json"
        if config.is_file():
            h.update(config.read_bytes())
        weights = sorted(p for p in path.rglob("*") if p.suffix in (".safetensors", ".gguf", ".bin"))
        for p in weights:
            h.update(f"{p.relative_to(path)}:{p.stat().st_size}\n".encode())
    else:
        h.update(f"{path.name}:{path.stat().st_size}\n".encode())
        with path.open("rb") as f:
            h.update(f.read(1 << 20))
    return "sha256:" + h.hexdigest()


def _llama_version() -> str | None:
    binary = os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")
    if not binary:
        return None
    r = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60)
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", r.stdout + r.stderr)
    return f"b{m.group(1)} ({m.group(2)})" if m else None


def stamp(
    *,
    engines: Iterable[str] = ("warpc",),
    models: Mapping[str, Path] | None = None,
    corpus_sha256: str | None = None,
    tokenizer_sha256: str | None = None,
) -> Stamp:
    """Everything a run depends on: `git describe --always` of the tree this file lives in and whether `warpc/`,
    `uv.lock` or `pyproject.toml` differ from it (a new untracked file counts: `--dirty` would not see it), sha256 of
    the swiftmk library and `uv.lock`, versions of torch / vllm / transformers, llama-server's, and the GPU. No
    hostnames, home directories or mount points enter a value."""
    here = Path(__file__).resolve().parent
    top = _run(["git", "rev-parse", "--show-toplevel"], here)
    describe = _run(["git", "describe", "--always"], here)
    changes = (
        _run(["git", "status", "--porcelain", "--", "warpc", "uv.lock", "pyproject.toml"], Path(top)) if top else None
    )
    lock = Path(top) / "uv.lock" if top else None
    library = os.environ.get("WARPC_SWIFTMK_LIB")
    legacy = os.environ.get("WARPC_LEGACY_DIR")
    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,power.limit,persistence_mode,driver_version", "--format=csv,noheader,nounits"]
    )
    gpu_name = power = persistence = driver = None
    if gpu:
        gpu_name, power_text, persistence_text, driver = (x.strip() for x in gpu.splitlines()[0].split(","))
        power = _number(power_text)
        persistence = persistence_text.lower() == "enabled"
    cuda = sys.modules["torch"].version.cuda if "torch" in sys.modules else None
    return Stamp(
        warpc={
            "commit": describe,
            "dirty": changes is None or changes != "",
            "lock_sha256": _sha256(lock) if lock and lock.is_file() else None,
        },
        libswiftmk_sha256=_sha256(Path(library)) if library and Path(library).is_file() else None,
        legacy_commit=_run(["git", "describe", "--tags", "--always"], Path(legacy)) if legacy else None,
        torch=_version("torch"),
        cuda=cuda,
        driver=driver,
        python=".".join(map(str, sys.version_info[:3])),
        vllm=_version("vllm"),
        llama_cpp=_llama_version() if "llama" in engines else None,
        transformers=_version("transformers"),
        gpu={
            "name": gpu_name,
            "power_limit_w": power,
            "persistence": persistence,
            "machine": machine_key(gpu_name, power),
        },
        models={name: {"revision": None, "fingerprint": fingerprint(p)} for name, p in (models or {}).items()},
        corpus_sha256=corpus_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )
