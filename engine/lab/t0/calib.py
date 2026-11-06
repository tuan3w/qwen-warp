"""Calibrate T0 relative output error -> real-model logit KL (kl_eval.py runs) and restate B1 as 'weight-byte fraction at KL <= budget'.  python calib.py -> results/b1/B1_kl.md

Model: KL_kind(f) = alpha_kind * err_kind(f)^2   (err = B1 mean per-row relative output error over the 8 traced layers and the docs that kl_eval ran; KL = real-model mean KL(dense||sparse)
over the 128 decode rows when ONE linear kind is sparsified in all 64 layers).  alpha absorbs the number of layers and their sensitivity.  The fit is checked on held-out rules / block sizes."""
import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

import sp

R = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results")
KLDOCS = ["tr32_notes", "tr32_code_cuda", "tr32_vendor_cu"]
BUDGETS = (0.0005, 0.0015, 0.005)


def load_b1():
    E = collections.defaultdict(list)  # (doc, kind, mode, rule, b, f) -> [(bytes, err)] over layers
    with open(R.parent / "runs/b1_raw.csv") as fh:
        for r in csv.DictReader(fh):
            if r["doc"] in KLDOCS:
                E[(r["kind"], r["mode"], r["rule"], int(r["b"]), float(r["f"]))].append((float(r["bytes"]), float(r["err"])))
    return {k: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs])) for k, vs in E.items()}


def load_kl():
    K = collections.defaultdict(list)
    for fn in glob.glob(str(R / "kl/*calib_act*.jsonl")) + glob.glob(str(R / "kl/*calib_act_blocks.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            n = r["name"]
            if r["doc"] not in KLDOCS or not (n.startswith("act_") or n.startswith("actb_") or n.startswith("actu_")):
                continue
            K[n].append(r["kl"])
    return {k: float(np.mean(v)) for k, v in K.items()}


def main():
    B, K = load_b1(), load_kl()
    L = ["# B1 restated at the real-model KL budget (T0 error calibrated against kl_eval.py)\n"]
    alpha = {}
    L.append("## fit: KL = alpha * err^2  (single kind sparsified in ALL 64 layers; per-channel top-|x|, T=1; points f = 0.9..0.4)\n")
    L.append("| kind | alpha | R^2 (log) | points (f: err% -> KL% measured / fitted) |")
    L.append("|---|---|---|---|")
    for kind in sp.KINDS:
        pts = []
        for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            n = f"act_{kind}_f{f}_b1"
            if n in K and (kind, "T1", "abs_topk", 1, f) in B:
                pts.append((f, B[(kind, "T1", "abs_topk", 1, f)][1], K[n]))
        e2 = np.array([p[1] ** 2 for p in pts])
        kl = np.array([p[2] for p in pts])
        a = float(np.exp(np.mean(np.log(kl / e2))))  # geometric fit
        alpha[kind] = a
        pred = a * e2
        r2 = 1 - np.sum((np.log(kl) - np.log(pred)) ** 2) / np.sum((np.log(kl) - np.log(kl).mean()) ** 2)
        L.append(f"| {kind} | {a:.1f} | {r2:.2f} | " + "; ".join(f"{p[0]}: {p[1] * 100:.1f} -> {p[2] * 100:.3f}/{a * p[1] ** 2 * 100:.3f}" for p in pts) + " |")
    # held-out check on block sizes
    L.append("\n## check on block sizes (b > 1, per-row top-k blocks): measured KL% vs fitted alpha*err^2\n")
    L.append("| kind | b | f | err% (T0) | KL% measured | KL% predicted |")
    L.append("|---|---|---|---|---|---|")
    for kind in sp.KINDS:
        for b in (8, 32, 128):
            for f in (0.9, 0.7, 0.5):
                n = f"actb_{kind}_f{f}_b{b}"
                key = (kind, "T1", "abs_topk", b, f)
                if n in K and key in B:
                    e = B[key][1]
                    L.append(f"| {kind} | {b} | {f} | {e * 100:.1f} | {K[n] * 100:.3f} | {alpha[kind] * e * e * 100:.3f} |")
    # byte fraction at the KL budget, per kind / rule / b / mode
    L.append("\n## weight-byte fraction (of the sparsified kind's rows) at which the PREDICTED KL (kind in all layers) reaches the budget; mean over layers; '-' = never below 1.0 within f in [0.2, 0.98]\n")
    for bud in BUDGETS:
        L.append(f"\n### KL budget {bud * 100:g}% (per kind, in isolation)\n")
        L.append("| kind | mode | rule | b=1 | b=8 | b=16 | b=32 | b=64 | b=128 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for kind in sp.KINDS:
            thr = (bud / alpha[kind]) ** 0.5
            for mode, rules in (("T1", ("abs_topk", "abs_thr", "static", "seq_own")), ("U8_union", ("abs_topk",)), ("U8_max", ("abs_topk",))):
                for rule in rules:
                    cells = []
                    for b in (1, 8, 16, 32, 64, 128):
                        pts = sorted((B[(kind, mode, rule, b, f)] for f in (0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2) if (kind, mode, rule, b, f) in B))
                        if not pts:
                            cells.append("n/a")
                            continue
                        bb, ee = zip(*pts)
                        c = sp.crossing(None, ee, bb, thr)
                        cells.append("-" if np.isnan(c) else f"{c:.2f}")
                    if any(c != "n/a" for c in cells):
                        L.append(f"| {kind} (err<={thr * 100:.1f}%) | {mode} | {rule} | " + " | ".join(cells) + " |")
    (R / "b1/B1_kl.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
