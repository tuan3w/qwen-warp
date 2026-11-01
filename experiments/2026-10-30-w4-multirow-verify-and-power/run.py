#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""2026-10-30-w4-multirow-verify-and-power — see README.md.

Builds bench.cu (nvcc -arch=sm_86), takes the GPU lease around the runs, samples nvidia-smi at 10 Hz (clocks, power, throttle reasons) and
writes results/*.json: check.json (correctness), q1.json (T sweep), q2.json (sustained power windows), clocks_q1.json / clocks_q2.json,
nvidia_smi_limits.txt.  `python3 report.py` renders results/tables.md + results/mtp.json from them.

Run:  uv run experiments/2026-10-30-w4-multirow-verify-and-power/run.py [--stages check,q1,q2] [--holder NAME]
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
BENCH = RUNS / "bench"
SMI_FIELDS = ("timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active,"
              "clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,"
              "clocks_throttle_reasons.hw_power_brake_slowdown,utilization.gpu,power.limit")


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def lease():
    return json.loads(sh([str(LAB), "gpu", "status"]).stdout).get("lease")


def acquire(holder: str) -> bool:
    while True:
        cur = lease()
        if cur is None:
            if sh([str(LAB), "gpu", "acquire", "--holder", holder]).returncode == 0:
                return True
        elif cur.get("holder") == holder:
            return False
        print(f"lease held by {cur.get('holder') if cur else '?'}; waiting 30s", file=sys.stderr)
        time.sleep(30)


def release() -> None:
    sh([str(LAB), "gpu", "release"])


def build() -> None:
    RUNS.mkdir(exist_ok=True)
    r = sh([NVCC, "-O3", "-std=c++17", "-arch=sm_86", "-Xptxas", "-v", str(HERE / "bench.cu"), "-o", str(BENCH)])
    (RUNS / "build.log").write_text(r.stderr)
    if r.returncode != 0:
        sys.exit("build failed:\n" + r.stderr[-4000:])


class Smi:
    """nvidia-smi sampler at 10 Hz."""

    def __init__(self):
        self.path = RUNS / "smi.csv"
        self.f = open(self.path, "w")
        self.p = subprocess.Popen(["nvidia-smi", f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits", "-lms", "100"], stdout=self.f)

    def stop(self):
        self.p.terminate(); self.p.wait(); self.f.close()

    def rows(self):
        out = []
        for row in csv.reader(open(self.path)):
            if len(row) < 12:
                continue
            try:
                ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
                out.append({"ts": ts, "sm": float(row[1]), "mem": float(row[2]), "w": float(row[3]), "temp": float(row[4]),
                            "mask": row[5].strip(), "swcap": row[6].strip() == "Active", "hw": row[7].strip() == "Active",
                            "therm": row[8].strip() == "Active", "brake": row[9].strip() == "Active", "util": float(row[10]),
                            "limit": float(row[11])})
            except ValueError:
                pass
        return out


def window_stats(rows, marks: str) -> dict:
    opened: dict[str, float] = {}
    wins: dict[str, tuple[float, float]] = {}
    for line in marks.splitlines():
        m = re.match(r"MARK (\S+) (start|end) ([\d.]+)", line)
        if not m:
            continue
        if m.group(2) == "start":
            opened[m.group(1)] = float(m.group(3))
        else:
            wins[m.group(1)] = (opened[m.group(1)], float(m.group(3)))
    res = {}
    for name, (a, b) in wins.items():
        s = [r for r in rows if a <= r["ts"] <= b]
        if not s:
            continue
        res[name] = {"samples": len(s), "window_s": round(b - a, 2),
                     "sm_mhz_median": statistics.median(r["sm"] for r in s), "sm_mhz_min": min(r["sm"] for r in s),
                     "sm_mhz_max": max(r["sm"] for r in s), "mem_mhz_median": statistics.median(r["mem"] for r in s),
                     "power_w_median": statistics.median(r["w"] for r in s), "power_w_max": max(r["w"] for r in s),
                     "temp_c_max": max(r["temp"] for r in s), "util_median": statistics.median(r["util"] for r in s),
                     "swcap_frac": sum(r["swcap"] for r in s) / len(s), "hw_slowdown_frac": sum(r["hw"] for r in s) / len(s),
                     "thermal_frac": sum(r["therm"] for r in s) / len(s), "power_limit_w": s[0]["limit"],
                     "masks": sorted({r["mask"] for r in s})}
    return res


def run_bench(args: list[str], out: str) -> str:
    t = time.time()
    r = sh([str(BENCH), *args, "--out", str(RES / out)])
    (RUNS / (out + ".stderr")).write_text(r.stderr)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        sys.exit(f"bench {args} failed rc={r.returncode}")
    print(f"[{out}] {time.time() - t:.0f}s", file=sys.stderr)
    return r.stderr


NCU_METRICS = ",".join([
    "gpu__time_duration.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram__bytes_read.sum", "lts__t_bytes.sum",
    "l1tex__t_sector_hit_rate.pct", "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active", "smsp__inst_executed.sum", "sm__inst_executed_pipe_alu.sum",
    "sm__inst_executed_pipe_fma.sum", "sm__inst_executed_pipe_lsu.sum", "sm__inst_executed_pipe_tensor_op_hmma.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio", "sm__cycles_elapsed.avg.per_second"])


def stage_ncu() -> dict:
    """ncu counters (natural clocks) for the best configs: where does T growth come from?  Writes results/ncu.json."""
    q1 = json.loads((RES / "q1.json").read_text())["final"]

    def pick(shape, T, kind, mode):
        c = [r for r in q1 if r["shape"] == shape and r["T"] == T and r["kind"] == kind and r["mode"] == mode]
        return f"{kind}/{mode}/{c[0]['cfg']}" if c else None

    plan = []
    for shape, rows in (("gateup", [(1, "mma", "sub"), (1, "cc", "e1"), (1, "read", "read"), (8, "mma", "sub"), (8, "mma", "abl"), (8, "mma", "ablb"),
                                     (16, "mma", "sub"), (16, "mma", "ablb")]),
                        ("down", [(1, "mma", "sub"), (8, "mma", "sub"), (16, "mma", "sub")]),
                        ("lmhead", [(1, "cc", "bf16"), (1, "mma", "bf16"), (16, "mma", "bf16")])):
        for T, kind, mode in rows:
            v = pick(shape, T, kind, mode)
            if v:
                plan.append((shape, T, v))
    res = {}
    for shape, T, v in plan:
        r = sh(["ncu", "--csv", "--clock-control", "none", "--metrics", NCU_METRICS, "-k", "regex:k_mma|k_cc|k_read", "--launch-skip", "2",
                "--launch-count", "1", str(BENCH), "one", "--shapes", shape, "--T", str(T), "--var", v], env={**os.environ, "LC_ALL": "C"})  # ncu CSV uses the locale decimal separator (en_US: "86,20", "92.037.248")
        rows = {}
        for line in r.stdout.splitlines():
            if not line.startswith('"'):
                continue
            f = next(csv.reader([line]))
            if len(f) >= 15 and f[-3] != "Metric Name":
                try:
                    rows[f[-3]] = float(f[-1].replace(",", ""))
                except ValueError:
                    pass
        res[f"{shape}/T{T}/{v}"] = rows
        print(f"[ncu] {shape} T={T} {v}: {len(rows)} metrics", file=sys.stderr)
    (RES / "ncu.json").write_text(json.dumps(res, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="check,q1,q2")
    ap.add_argument("--holder", default="w4-multirow-e3")
    a = ap.parse_args()
    stages = a.stages.split(",")
    RES.mkdir(exist_ok=True); RUNS.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc)
    build()
    (RES / "nvidia_smi_limits.txt").write_text(sh(["nvidia-smi", "-q", "-d", "POWER,CLOCK,PERFORMANCE"]).stdout)
    took = acquire(a.holder)
    smi = Smi()
    summary = {"started": started.isoformat(),
               "gpu": sh(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,power.limit", "--format=csv,noheader"]).stdout.strip(),
               "nvcc": sh([NVCC, "--version"]).stdout.strip().splitlines()[-1]}
    try:
        if "check" in stages:
            run_bench(["check"], "check.json")
        if "q1" in stages:
            err = run_bench(["q1"], "q1.json")
            (RES / "clocks_q1.json").write_text(json.dumps(window_stats(smi.rows(), err), indent=1))
        if "q2" in stages:
            err = run_bench(["q2"], "q2.json")
            (RES / "clocks_q2.json").write_text(json.dumps(window_stats(smi.rows(), err), indent=1))
        if "ncu" in stages:
            stage_ncu()
    finally:
        smi.stop()
        if took:
            release()
    rows = smi.rows()
    summary["duration_s"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    if rows:
        summary["smi_samples"] = len(rows)
        summary["smi_overall"] = {"sm_mhz_median": statistics.median(r["sm"] for r in rows), "power_w_max": max(r["w"] for r in rows),
                                  "temp_c_max": max(r["temp"] for r in rows), "masks_seen": sorted({r["mask"] for r in rows})}
    (RES / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
