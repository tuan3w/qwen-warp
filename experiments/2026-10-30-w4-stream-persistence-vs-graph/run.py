#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""2026-10-30-w4-stream-persistence-vs-graph — see README.md for the question this is answering.

Builds bench.cu (nvcc -arch=sm_86; no torch), takes the GPU lease, logs nvidia-smi clocks/power, runs the stages and
writes results/*.json.  Stages: pdl unit barrier check gemv grid sweep main ablate profile  (default: all).

Run:  uv run experiments/2026-10-30-w4-stream-persistence-vs-graph/run.py [--stages main,profile] [--holder NAME]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
BENCH = RUNS / "bench"  # built binary stays out of git
ALL_STAGES = ["pdl", "unit", "barrier", "check", "gemv", "grid", "sweep", "main", "ablate", "profile"]


# ------------------------------------------------------------------ lease
def lease() -> dict | None:
    out = subprocess.run([str(LAB), "gpu", "status"], capture_output=True, text=True, check=True).stdout
    return json.loads(out).get("lease")


def acquire(holder: str) -> bool:
    """Take the GPU lease (manual, around runs). Waits if someone else holds it; never steals. Returns True if we took it."""
    while True:
        cur = lease()
        if cur is None:
            r = subprocess.run([str(LAB), "gpu", "acquire", "--holder", holder], capture_output=True, text=True)
            if r.returncode == 0:
                return True
        elif cur.get("holder") == holder:
            return False  # already ours (e.g. acquired manually before launching)
        print(f"lease held by {cur.get('holder') if cur else '?'}; waiting 30s", file=sys.stderr)
        time.sleep(30)


def release() -> None:
    subprocess.run([str(LAB), "gpu", "release"], capture_output=True, text=True)


# ------------------------------------------------------------------ helpers
def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def build() -> str:
    RUNS.mkdir(exist_ok=True)
    r = sh([NVCC, "-O3", "-std=c++17", "-arch=sm_86", "-Xptxas", "-v", str(HERE / "bench.cu"), "-o", str(BENCH)])
    RUNS.mkdir(exist_ok=True)
    (RUNS / "build.log").write_text(r.stderr)
    if r.returncode != 0:
        sys.exit("build failed:\n" + r.stderr[-4000:])
    return r.stderr


def cfg_names() -> list[str]:
    return [f"w{a}r{b}d{c}" for a, b, c in re.findall(r"reg<(\d+),\s*(\d+),\s*(\d+)>", (HERE / "cfgs.inc").read_text())]


def run_bench(args: list[str], out_name: str, log_name: str | None = None) -> dict:
    out = RES / out_name
    t = time.time()
    r = sh([str(BENCH), *args, "--out", str(out)])
    if log_name:
        (RUNS / log_name).write_text(r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f"bench {args} failed rc={r.returncode}")
    print(f"[{out_name}] {time.time() - t:.0f}s", file=sys.stderr)
    return json.loads(out.read_text()), r.stderr


class Smi:
    """nvidia-smi sampler (5 Hz) so each timed window can be matched to clocks/power/throttle reasons."""

    def __init__(self):
        RUNS.mkdir(exist_ok=True)
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
    """marks: stderr of bench with 'MARK name rep=k start|end ts' lines -> per-variant clock/power summary."""
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


def best_per_style(sweep: dict) -> dict[str, str]:
    best: dict[str, tuple[float, str]] = {}
    for r in sweep["results"]:
        if ":" not in r["name"]:
            continue
        style, cfg = r["name"].split(":", 1)
        if style not in best or r["median_ms"] < best[style][0]:
            best[style] = (r["median_ms"], cfg)
    return {k: v[1] for k, v in best.items()}


# ------------------------------------------------------------------ stages
def stage_pdl():
    """Programmatic dependent launch (griddepcontrol) is sm_90+; record the exact ptxas error on sm_86."""
    src = HERE / "runs" / "pdl_probe.cu"
    RUNS.mkdir(exist_ok=True)
    src.write_text('__global__ void k(int* p){ asm volatile("griddepcontrol.wait;" ::: "memory");'
                   ' asm volatile("griddepcontrol.launch_dependents;"); p[0]=1; }\nint main(){return 0;}\n')
    r86 = sh([NVCC, "-arch=sm_86", "-c", str(src), "-o", str(RUNS / "pdl86.o")])
    r90 = sh([NVCC, "-arch=sm_90", "-c", str(src), "-o", str(RUNS / "pdl90.o")])
    (RES / "pdl_attempt.json").write_text(json.dumps({
        "nvcc": sh([NVCC, "--version"]).stdout.strip().splitlines()[-1],
        "sm_86_returncode": r86.returncode, "sm_86_stderr": r86.stderr.strip().replace(str(src), "pdl_probe.cu"),
        "sm_90_returncode": r90.returncode}, indent=2))


def stage_simple(name: str, args: list[str]):
    return run_bench(args, f"{name}.json", f"{name}.stderr")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(ALL_STAGES))
    ap.add_argument("--holder", default="w4-stream-persist-exp")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    a = ap.parse_args()
    stages = a.stages.split(",")
    RES.mkdir(exist_ok=True); RUNS.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    build()
    took = acquire(a.holder)
    smi = Smi()
    summary: dict = {"started": started.isoformat(), "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"]).stdout.strip()}
    marks_all = ""
    try:
        if "pdl" in stages:
            stage_pdl()
        if "unit" in stages:
            stage_simple("unit", ["unit"])
        if "barrier" in stages:
            stage_simple("barrier", ["barrier"])
        cfgs = cfg_names()
        sweep_path = RES / "sweep.json"
        if "gemv" in stages:
            stage_simple("gemv_sweep", ["gemv"])
        if "grid" in stages:
            grids = {}
            for g in (41, 64, 82):
                grids[str(g)] = run_bench(["gemv", "--grid", str(g), "--cfgs", "w8r4d1,w16r4d1"], f"gemv_grid{g}.json")[0]
            (RES / "gemv_grid_study.json").write_text(json.dumps(grids))
            for g in (41, 64, 82):
                (RES / f"gemv_grid{g}.json").unlink()
        if "sweep" in stages:
            specs = [f"{s}:{c}" for c in cfgs for s in ("v1a", "v1b", "v3", "v4")]
            specs += [f"v1c:{c}" for c in ("w16r4d1", "w8r4d1", "w12r4d1", "w8r4d2", "w4r4d1", "w8r2d1")]
            _, err = run_bench(["full", "--repeats", "2", "--steps", "100", "--warmup", "10", "v0", *specs], "sweep.json", "sweep.stderr")
            marks_all += err
        if "check" in stages or "main" in stages or "ablate" in stages or "profile" in stages:
            sweep = json.loads(sweep_path.read_text())
            best = best_per_style(sweep)
            summary["best_cfg_per_style_from_sweep"] = best
        if "check" in stages:
            stage_simple("check", ["check", "--cfg", best["v4"]])
        if "main" in stages:
            specs = ["memcpy", "v0", "v0g", f"v1a:{best['v1a']}", f"v1b:{best['v1b']}", f"v1c:{best['v1c']}", f"v3:{best['v3']}", f"v4:{best['v4']}"]
            # same-config comparison too: every style at the V4-best config and at the V1b-best config
            for c in {best["v4"], best["v1b"]}:
                for s in ("v1a", "v1b", "v3", "v4"):
                    sp = f"{s}:{c}"
                    if sp not in specs:
                        specs.append(sp)
            main_res, err = run_bench(["full", "--repeats", str(a.repeats), "--steps", str(a.steps), "--warmup", str(a.warmup), *specs], "main.json", "main.stderr")
            marks_all += err
            summary["clocks_per_variant"] = window_stats(smi.rows(), err)
        if "ablate" in stages:
            c3, c4 = best["v3"], best["v4"]
            specs = ["v0", f"v3:{c3}"] + [f"v3:{c3}:abl={b}" for b in (1, 2, 4, 7)] + [f"v4:{c4}"] + [f"v4:{c4}:abl={b}" for b in (1, 2, 4, 7)]
            run_bench(["full", "--repeats", "3", "--steps", "100", "--warmup", "10", *specs], "ablation.json", "ablation.stderr")
        if "profile" in stages:
            prof = {}
            for cross, cfg in ((0, best["v3"]), (1, best["v4"]), (1, "w8r4d1"), (0, "w8r4d1")):
                key = f"{'v4' if cross else 'v3'}:{cfg}"
                prof[key] = run_bench(["profile", "--cfg", cfg, "--cross", str(cross)], "_prof.json")[0]["profile"]
            (RES / "_prof.json").unlink()
            (RES / "profile.json").write_text(json.dumps(prof, indent=1))
    finally:
        smi.stop()
        if took:
            release()
    summary["duration_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    sm = [r[1] for r in smi.rows()]
    if sm:
        summary["smi_overall"] = {"sm_mhz_median": statistics.median(sm), "sm_mhz_min": min(sm), "sm_mhz_max": max(sm),
                                  "power_w_max": max(r[2] for r in smi.rows()), "temp_c_max": max(r[3] for r in smi.rows()),
                                  "throttle_reason_masks_seen": sorted({r[4] for r in smi.rows()})}
    (RES / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
