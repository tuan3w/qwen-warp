#!/usr/bin/env python3
"""calibrate.py <root> [--variants ...]: derive the check limits.  GPU lease must be held (runs metric.py dev on base/ repeatedly).

Measures, with the pristine tree:
  * bit-reproducibility: two pristine dev runs -> max |dlogit| of the teacher-forced data (0 = deterministic)
  * the engine's own noise: T=1 plain steps vs T=4 verify rows on the same teacher-forced tokens (BENCH.md K7 follow-up: top-20 |d| median 0.028, p99 0.16)
  * legit rounding-level changes: the same pristine source with other W4 GEMV configurations (SMK_CFG env, different K-split / accumulation order) that the check MUST accept
and prints the statistics `limits.json` was derived from.  Output: <root>/calib/report.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser()
sys.path.insert(0, str(root / "adapter"))
os.environ["CAMPAIGN_ROOT"] = str(root)
import numpy as np  # noqa: E402

import check  # noqa: E402
import hlib  # noqa: E402

PY = hlib.PY
VARIANTS = {  # name -> SMK_CFG (N:cfg or 100000+K:cfg; cfg ids in gemv.cu: 0 w8k4u2, 1 w8k8u2, 2 w8k2u3, 3 w4k4u3, 4 w16k8u2, 5 w8k1u3)
    "gateup_k8": "34816:1",
    "down_k2u3": "117408:2",
    "oproj_k8": "106144:1",
    "qkvz_k8": "16384:1",
    "attnqkv_k4": "14336:0",
    "all_alt": "34816:1,117408:2,106144:1,16384:1,14336:0",
}
out = root / "calib"
out.mkdir(exist_ok=True)


def run(name, env=None, mode="gate"):
    p = out / f"{name}.json"
    e = dict(os.environ, CAMPAIGN_ROOT=str(root), CAMPAIGN_MODE=mode, **(env or {}))
    if env:
        e["SMKH_KEEP_ENV"] = "1"
    r = subprocess.run([PY, str(root / "adapter/metric.py"), "dev", str(root / "base"), str(p)], env=e, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{name} failed:\n{r.stderr[-2000:]}")
    return p


def stats_vs(a, b, near):
    """candidate npz a vs control npz b."""
    A, B = np.load(hlib.tf_path(a)), np.load(hlib.tf_path(b))
    rep = {}
    for mode, pre, names in [(m, "", hlib.TF_NAMES) for m, _ in hlib.TF_MODES] + [("t4", "m4_", hlib.ORACLE_NAMES), ("t1", "m4_", hlib.ORACLE_NAMES)]:
        rows = [check.devstats(A[f"{pre}{n}_{mode}_id"], A[f"{pre}{n}_{mode}_val"], B[f"{pre}{n}_{mode}_id"], B[f"{pre}{n}_{mode}_val"], near) for n in names]
        rep[pre + mode] = dict(agree=min(r["agree"] for r in rows), unexcused=sum(r["unexcused"] for r in rows), med=max(r["med"] for r in rows), p99=max(r["p99"] for r in rows), max=max(r["max"] for r in rows),
                               per_prompt={n: dict(agree=r["agree"], med=r["med"], max=r["max"]) for n, r in zip(names, rows)})
    ptv = [check.probe_tv(A, B, i) for i in range(len(check.hlib_probes(A)))]
    rep["probe_self_tv"], rep["probe_ctrl_tv"] = max(x[0] for x in ptv), max(x[1] for x in ptv)
    ja, jb = json.load(open(a)), json.load(open(b))
    rep["trials"], rep["trials_pld"] = ja["trials"], ja["trials_pld"]
    gens = dict(ja["spec_gens"])
    gens.update({"m4:" + k: v for k, v in ja["spec_gens_m4"].items()})
    div = [k for k, t in gens.items() if t != hlib.jload(hlib.PRIV / "tf_tokens.json")[k.removeprefix("m4:").split("/")[0]]["cont"][: len(t)]]
    rep["spec_divergent"] = div
    ta, tb = ja["tokens"], jb["tokens"]
    rep["compare"] = float(np.mean([next((i for i, (x, y) in enumerate(zip(ta[n], tb[n])) if x != y), len(ta[n])) / len(ta[n]) for n in ta]))
    rep["score"] = (ja["score"], jb["score"])
    rep["accept_ratio"] = {n: c["tokens_per_cycle"] / jb["cells"][n]["tokens_per_cycle"] for n, c in ja["cells"].items()}  # candidate / control tokens per cycle per prompt (limits.json accept_band)
    rep["cyc_rate_ratio"] = {n: c["cyc_rate"] / jb["cells"][n]["cyc_rate"] for n, c in ja["cells"].items()}
    return rep


def t1_vs_t4(a, near, other="t4"):
    """the engine's own width noise on the same teacher-forced tokens: plain T=1 rows vs T=4 (chain verify) / T=8 (hybrid verify) rows of the same engine."""
    A = np.load(hlib.tf_path(a))
    rows = [check.devstats(A[f"{n}_t1_id"], A[f"{n}_t1_val"], A[f"{n}_{other}_id"], A[f"{n}_{other}_val"], near) for n in hlib.TF_NAMES]
    return dict(agree=min(r["agree"] for r in rows), unexcused=sum(r["unexcused"] for r in rows), med=max(r["med"] for r in rows), p99=max(r["p99"] for r in rows), max=max(r["max"] for r in rows))


near = hlib.limits()["near_tie"]
rep = {}
ctrl1, ctrl2 = run("ctrl1"), run("ctrl2")
rep["reproducibility_ctrl2_vs_ctrl1"] = stats_vs(ctrl2, ctrl1, near)
rep["t1_vs_t4_pristine"] = t1_vs_t4(ctrl1, near)
rep["t1_vs_t8_pristine"] = t1_vs_t4(ctrl1, near, "t8")
for name, cfg in VARIANTS.items():
    if len(sys.argv) > 2 and name not in sys.argv[2:]:
        continue
    print("variant", name, cfg, flush=True)
    rep[name] = stats_vs(run(name, {"SMK_CFG": cfg}), ctrl1, near)
(out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
print(json.dumps(rep, indent=1, default=float))
