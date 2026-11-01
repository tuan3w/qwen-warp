#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""2026-10-30-dataflow-fused-phases (E2) — see README.md.

Builds bench.cu (nvcc -arch=sm_86; kernels.cuh = E1 kernels + L2-prefetch cursor, df.cuh = fused/dataflow/balanced persistent kernel),
takes the GPU lease only around each bench invocation, logs nvidia-smi clocks/power and writes results/*.json.
Stages: check sweep main ablate power  (default: all).  `python3 report.py` renders results/*.json into results/tables.md.

Run:  uv run experiments/2026-10-30-dataflow-fused-phases/run.py [--stages main] [--holder NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]
RES = HERE / "results"
RUNS = HERE / "runs"  # gitignored
LAB = REPO / "projects/llm-lab/bin/lab"
NVCC = "/usr/local/cuda/bin/nvcc"
BENCH = RUNS / "bench"
ALL_STAGES = ["check", "sweep", "main", "ablate", "power"]

# candidate families for the sweep (2 repeats x 100 steps each); `main` re-measures the best of each family at 5 x 200
CFGS_BAR = ["w8r2d4", "w8r2d5", "w8r2d6"]
CFGS_DF = ["w8r2d3", "w8r2d4", "w8r2d5"]


# ------------------------------------------------------------------ lease (taken per bench invocation, never held while idle)
def lease() -> dict | None:
    out = subprocess.run([str(LAB), "gpu", "status"], capture_output=True, text=True, check=True).stdout
    return json.loads(out).get("lease")


def acquire(holder: str) -> None:
    while True:
        cur = lease()
        if cur is None:
            r = subprocess.run([str(LAB), "gpu", "acquire", "--holder", holder], capture_output=True, text=True)
            if r.returncode == 0:
                return
        print(f"lease held by {cur.get('holder') if cur else '?'}; waiting 20s", file=sys.stderr)
        time.sleep(20)


def release() -> None:
    subprocess.run([str(LAB), "gpu", "release"], capture_output=True, text=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build() -> None:
    RUNS.mkdir(exist_ok=True)
    r = sh([NVCC, "-O3", "-std=c++17", "-arch=sm_86", "-Xptxas", "-v", str(HERE / "bench.cu"), "-o", str(BENCH)])
    (RUNS / "build.log").write_text(r.stderr)
    if r.returncode != 0:
        sys.exit("build failed:\n" + r.stderr[-4000:])


class Smi:
    """nvidia-smi sampler (5 Hz) so each timed window can be matched to clocks/power/throttle reasons."""

    def __init__(self):
        self.path = RUNS / "smi.csv"
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "200"], stdout=self.f)

    def stop(self):
        self.p.terminate(); self.p.wait(); self.f.close()

    def rows(self):
        out = []
        for row in csv.reader(open(self.path)):
            if len(row) < 7:
                continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append((ts, float(row[1]), float(row[3]), float(row[4]), row[5].strip(), float(row[6])))
            except ValueError:
                pass
        return out


def window_stats(rows, marks):
    wins: dict[str, list[tuple[float, float]]] = {}
    open_: dict[tuple[str, str], float] = {}
    for line in marks.splitlines():
        m = re.match(r"MARK (\S+) rep=(\d+) (start|end) ([\d.]+)", line)
        if not m:
            continue
        name, rep, kind, ts = m.group(1), m.group(2), m.group(3), float(m.group(4))
        if kind == "start":
            open_[(name, rep)] = ts
        else:
            wins.setdefault(name, []).append((open_[(name, rep)], ts))
    res = {}
    for name, ws in wins.items():
        sm, pw, tmp, thr = [], [], [], set()
        for ts, smc, p, t, reason, _u in rows:
            if any(a <= ts <= b for a, b in ws):
                sm.append(smc); pw.append(p); tmp.append(t); thr.add(reason)
        if sm:
            res[name] = {"samples": len(sm), "sm_mhz_median": statistics.median(sm), "sm_mhz_min": min(sm), "sm_mhz_max": max(sm),
                         "power_w_median": statistics.median(pw), "power_w_max": max(pw), "temp_c_max": max(tmp),
                         "throttle_reason_masks_seen": sorted(thr)}
    return res


def bench(args: list[str], out_name: str, log_name: str, holder: str, timeout: int = 3000):
    """one bench invocation under the GPU lease (released immediately after)"""
    out = RES / out_name
    acquire(holder)
    smi = Smi()
    t = time.time()
    try:
        r = subprocess.run([str(BENCH), *args, "--out", str(out)], capture_output=True, text=True, timeout=timeout)
    finally:
        smi.stop()
        release()
    (RUNS / log_name).write_text(r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f"bench {args} failed rc={r.returncode}")
    print(f"[{out_name}] {time.time() - t:.0f}s", file=sys.stderr)
    return json.loads(out.read_text()), r.stderr, window_stats(smi.rows(), r.stderr)


def best_of(sweep: dict, prefix: str) -> str:
    cands = [r for r in sweep["results"] if r["name"].startswith(prefix)]
    return min(cands, key=lambda r: r["median_ms"])["name"]


def families() -> dict[str, list[str]]:
    fam: dict[str, list[str]] = {}
    fam["v1b"] = [f"v1b:w8r2d2:pf={p}" if p else "v1b:w8r2d2" for p in (0, 2)]
    fam["v4"] = [f"v4:w8r2d5:pf={p}" if p else "v4:w8r2d5" for p in (0, 2)]
    for c in CFGS_BAR:
        for fuse in (0, 1):
            for bal in (0, 1):
                fam.setdefault(f"bar_fuse{fuse}_bal{bal}", []).append(f"e2:{c}:fuse={fuse}:bal={bal}")
    for c in CFGS_DF:
        for fuse in (0, 1):
            for bal in (0, 1):
                fam.setdefault(f"df_fuse{fuse}_bal{bal}", []).append(f"e2:{c}:df=1:fuse={fuse}:bal={bal}")
    return fam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--holder", default="e2-dataflow-fused")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    stages = a.stages.split(",")
    RES.mkdir(exist_ok=True); RUNS.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    build()
    summary: dict = {"started": started.isoformat(),
                     "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"]).stdout.strip()}
    fam = families()
    sweep_path = RES / "sweep.json"
    if "check" in stages:
        specs = ["e2:w8r2d5:bal=1", "e2:w8r2d5:fuse=1", "e2:w8r2d5:fuse=1:bal=1:pf=2", "e2:w8r2d5:df=1", "e2:w8r2d5:df=1:bal=1", "e2:w8r2d5:df=1:fuse=1:bal=1"]
        bench(["check", "--layers", "4", "--cfg", "w8r2d5", *specs], "check4.json", "check4.stderr", a.holder)
        bench(["check", "--layers", "64", "--cfg", "w8r2d5", *specs], "check64.json", "check64.stderr", a.holder)
    if "sweep" in stages:
        allspecs = ["v0"] + [s for v in fam.values() for s in v]
        _, err, _ = bench(["full", "--repeats", "2", "--steps", "100", "--warmup", "10", *allspecs], "sweep.json", "sweep.stderr", a.holder)
    if {"main", "ablate", "power"} & set(stages):
        sweep = json.loads(sweep_path.read_text())
        best = {k: best_of(sweep, "x") if False else min((r for r in sweep["results"] if r["name"] in v), key=lambda r: r["median_ms"])["name"] for k, v in fam.items()}
        summary["best_per_family_from_sweep"] = best
    if "main" in stages:
        specs = ["v0", "v1a:w8r2d2", "v1b:w8r2d2", "v4:w8r2d5"] + [best[k] for k in best if k not in ("v1b", "v4")]
        specs = list(dict.fromkeys(specs))
        _, err, win = bench(["full", "--repeats", str(a.repeats), "--steps", str(a.steps), "--warmup", str(a.warmup), *specs], "main.json", "main.stderr", a.holder)
        summary["clocks_per_variant"] = win
    if "ablate" in stages:
        b = best["df_fuse1_bal1"].split(":")
        base = ":".join(b)
        specs = ["v0", base] + [f"{base}:abl={x}" for x in (1,)]
        bench(["full", "--repeats", "3", "--steps", "100", "--warmup", "10", *specs], "ablation.json", "ablation.stderr", a.holder)
    if "power" in stages:
        base = best["df_fuse1_bal1"]
        specs = ["v0", base, base + ":cheap=1", best["bar_fuse0_bal0"], best["bar_fuse0_bal0"] + ":cheap=1", "v1b:w8r2d2", "v4:w8r2d5"]
        _, err, win = bench(["full", "--repeats", "3", "--steps", "100", "--warmup", "10", *specs], "power.json", "power.stderr", a.holder)
        summary["power_windows"] = win
    summary["duration_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    (RES / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
