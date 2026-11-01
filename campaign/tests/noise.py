#!/usr/bin/env python3
"""noise.py <root> [rounds | chain N]: run-to-run noise of the three metrics on the pristine base/ (GPU lease must be held).

Round-robin dev / held / plain with 30 s idle gaps (the gate's control, candidate, control legs are also separated by engine load / build idle time), `rounds` rounds (default 4) + one
extra dev.  Reports per metric: scores, mean, SD (%), SM clock / power at the end of each run; and the SD of the gate statistic ratio = mean(c1,c2)/mean(A,B) over all windows of
four consecutive dev runs, the minimal detectable speedup (one-sided alpha 5%, power 80% = 2.49 x SD of that ratio) and the false-keep rate at gate.keep.
Output: <root>/calib/noise.json
`noise.py <root> chain N`: N back-to-back pristine dev legs in the gate's own pattern (full, full, light, light, ...: g-ctrlA / g-dev1 carry the correctness data, g-dev2 / g-ctrlB do not; no idle gaps) and the
SD of the gate ratio over all sliding windows of four (-> calib/noise_chain.json); this is the statistic the keep threshold and the MDE are derived from."""
import json
import os
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
chain = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "chain" else 0
rounds = int(sys.argv[2]) if len(sys.argv) > 2 and not chain else 4
sys.path.insert(0, str(root / "adapter"))
os.environ["CAMPAIGN_ROOT"] = str(root)
import hlib  # noqa: E402

out = root / "calib"
out.mkdir(exist_ok=True)
runs = {"dev": [], "held": [], "plain": []}


def one(kind, i):
    p = out / f"noise-{kind}{i}.json"
    t0 = time.time()
    r = subprocess.run([hlib.PY, str(root / "adapter/metric.py"), kind, str(root / "base"), str(p)], env=dict(os.environ, CAMPAIGN_MODE="gate"), capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{kind}{i} failed:\n{r.stderr[-1500:]}")
    j = json.loads(p.read_text())
    clocks = j.get("clocks") or [None]
    runs[kind].append(dict(score=j["score"], sm_mhz_end=clocks[-1][0] if clocks[-1] else None, power_w=clocks[-1][2] if clocks[-1] else None, wall_s=time.time() - t0, cells={k: v["rate"] for k, v in j["cells"].items()}))
    print(kind, i, f"{j['score']:.3f}", f"{time.time() - t0:.0f}s", clocks[-1], flush=True)


if chain:
    LEGS = ["g-ctrlA", "g-dev1", "g-dev2", "g-ctrlB"]
    cd = out / "chain"
    cd.mkdir(exist_ok=True)
    sc = []
    for i in range(chain):
        p = cd / f"{LEGS[i % 4]}.json"  # the leg name selects the correctness-data tier (metric.LIGHT_LEGS)
        t0 = time.time()
        r = subprocess.run([hlib.PY, str(root / "adapter/metric.py"), "dev", str(root / "base"), str(p)], env=dict(os.environ, CAMPAIGN_MODE="gate"), capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"chain leg {i} failed:\n{r.stderr[-1500:]}")
        j = json.loads(p.read_text())
        sc.append(j["score"])
        print("leg", i, LEGS[i % 4], f"{j['score']:.3f}", f"{time.time() - t0:.0f}s", (j.get("clocks") or [[None]])[-1], flush=True)
    ratios = [(sc[i + 1] + sc[i + 2]) / (sc[i] + sc[i + 3]) for i in range(len(sc) - 3)]
    sdr = st.pstdev(ratios)
    rep = dict(scores=sc, ratios=ratios, ratio_mean=st.mean(ratios), ratio_sd_pct=100 * sdr, leg_sd_pct=100 * st.stdev(sc) / st.mean(sc), mde_pct=100 * 2.49 * sdr, drift_ab_pct=[100 * abs(sc[i] / sc[i + 3] - 1) for i in range(len(sc) - 3)])
    (out / "noise_chain.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0)
for i in range(rounds):
    for kind in ("dev", "held", "plain"):
        one(kind, i)
        time.sleep(30)
one("dev", rounds)

rep = {}
for kind, rs in runs.items():
    s = [r["score"] for r in rs]
    rep[kind] = dict(scores=s, mean=st.mean(s), sd_pct=100 * st.stdev(s) / st.mean(s), range_pct=100 * (max(s) / min(s) - 1), wall_s=st.mean(r["wall_s"] for r in rs), runs=rs)
s = [r["score"] for r in runs["dev"]]
ratios = []
for i in range(len(s) - 3):  # gate pattern: ctrlA, c1, c2, ctrlB all pristine
    A, c1, c2, B = s[i : i + 4]
    ratios.append((c1 + c2) / (A + B))
if len(ratios) > 1:
    sdr = st.pstdev(ratios)
    rep["gate_ratio"] = dict(ratios=ratios, sd_pct=100 * sdr, mde_pct=100 * 2.49 * st.mean([rep["dev"]["sd_pct"] / 100, sdr]))
d = {k: rep[k]["sd_pct"] for k in runs}
rep["note"] = "gate ratio SD ~ SD of one leg (A,B,c1,c2 independent); MDE = 2.49 x SD(ratio) (alpha 0.05 one-sided, power 0.8)"
(out / "noise.json").write_text(json.dumps(rep, indent=1))
print(json.dumps({k: {x: v for x, v in rep[k].items() if x != "runs"} for k in rep if isinstance(rep[k], dict)}, indent=1))
