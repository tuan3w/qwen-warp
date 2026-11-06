"""B1: activation sparsity on real linear inputs.  Error-vs-bytes curves for the rules in sp.make_mask, at T=1 and T=8 (union / max-over-rows), all trace docs.

  python b1_act.py [--docs a,b] [--layers 2,7] [--kinds qkv,down] [--quick] [--dev cuda]     -> results/b1/b1_raw.csv  (one row per doc/layer/kind/mode/rule/block/f)
  python b1_act.py --summary                                                               -> results/b1/B1.md  (byte fraction at <= 0.5 / 1 / 2 % relative output error)

Rows: the 128 true decode positions of the model's own greedy continuation ('dec'); T=1: each row separately, T=8: 16 groups of 8 consecutive rows.  Calibration rows (abs_thr, static) = the
document rows of the OTHER docs for the same layer / kind (disjoint documents)."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import sp

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b1")
RUNS = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/runs")  # raw rows (5 MB, gitignored)
FS = [0.98, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
THR = (0.01, 0.02, 0.05, 0.10)
DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]


def run(docs, layers, kinds, quick, dev):
    sp.DEV = dev
    RES.mkdir(parents=True, exist_ok=True)
    T = {d: sp.Trace(d, dev) for d in docs}
    fs = [0.8, 0.6, 0.5, 0.4, 0.3, 0.2] if quick else FS
    rows = []
    t0 = time.time()
    for li in layers:
        Wc = sp.Wcache(li)
        for kind in kinds:
            W = Wc[kind]
            Wd = type("W", (), dict(w=W.w.to(dev), cn=W.cn.to(dev)))
            for d in docs:
                x = T[d].lin(li, kind, "dec")
                calib = torch.cat([T[o].lin(li, kind, "doc") for o in docs if o != d])
                y = x @ Wd.w.t()
                yn = y.norm(dim=1).clamp_min(1e-9)
                yg = (x[: 128].reshape(16, 8, -1) @ Wd.w.t()).reshape(128, -1)

                def ev(mask, b, mode, rule, f):
                    dd = (x * mask) @ Wd.w.t() - y
                    err = (dd.norm(dim=1) / yn).mean().item()
                    if mode == "T1":
                        byt = sp.bytes_frac(mask, b).mean().item()
                    else:
                        byt = sp.bytes_frac(mask[::8], b).mean().item()
                    rows.append((d, li, kind, mode, rule, b, f, round(byt, 5), round(err, 6)))

                for f in fs:
                    for rule in ("abs_topk", "wina_topk", "abs_thr", "static"):
                        ev(sp.make_mask(x, rule, f, 1, Wd.cn, calib), 1, "T1", rule, f)
                    own = T[d].lin(li, kind, "doc")  # 32 random rows of the SAME document's prefill (= sequence-level statistics, GRIFFIN-style, whole-prompt window)
                    for b in (1, 8, 64, 128):
                        ev(sp.make_mask(x, "static", f, b, None, own), b, "T1", "seq_own", f)
                    for b in (8, 16, 32, 64, 128):
                        for rule in ("abs_topk", "wina_topk", "max_topk"):
                            ev(sp.make_mask(x, rule, f, b, Wd.cn, calib), b, "T1", rule, f)
                        ev(sp.make_mask(x, "static", f, b, Wd.cn, calib), b, "T1", "static", f)
                    for b in (1, 8, 32, 128):
                        for rule in ("abs_topk", "wina_topk"):
                            for mode in ("union", "max"):
                                ev(sp.union_mask(x, rule, f, b, Wd.cn, 8, mode), b, "U8_" + mode, rule, f)
        print(f"layer {li} done  {time.time() - t0:.1f}s  rows={len(rows)}", flush=True)
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / ("b1_raw_quick.csv" if quick else "b1_raw.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc", "layer", "kind", "mode", "rule", "b", "f", "bytes", "err"])
        w.writerows(rows)
    print(f"wrote {len(rows)} rows, {time.time() - t0:.1f}s")


def summary():
    import collections

    R = collections.defaultdict(list)
    with open(RUNS / "b1_raw.csv") as fh:
        for r in csv.DictReader(fh):
            R[(r["doc"], int(r["layer"]), r["kind"], r["mode"], r["rule"], int(r["b"]))].append((float(r["bytes"]), float(r["err"])))
    # byte fraction at err <= thr per (layer, kind, mode, rule, b), mean over docs
    out = collections.defaultdict(list)
    for (d, li, kind, mode, rule, b), pts in R.items():
        bb, ee = zip(*pts)
        for thr in THR:
            out[(li, kind, mode, rule, b, thr)].append(sp.crossing(None, ee, bb, thr))
    S = {k: float(np.nanmean(v)) if not all(np.isnan(v)) else float("nan") for k, v in out.items()}
    lines = ["# B1 summary: weight-byte fraction needed for mean per-row relative output error <= thr (mean over 6 docs; 1.0 = no saving; '-' = not reached with kept fraction f >= 0.2 / never below 1.0 at f=0.98)\n"]
    layers = sorted({k[0] for k in S})
    cls = {"DN": [2, 14, 30, 46], "ATT": [7, 23, 39, 63]}

    def agg(kind, mode, rule, b, thr, ls):
        v = [S[(li, kind, mode, rule, b, thr)] for li in ls if (li, kind, mode, rule, b, thr) in S]
        return float(np.nanmean(v)) if v and not all(np.isnan(v)) else float("nan")

    for thr in (0.05, 0.02, 0.10, 0.01):
        lines.append(f"\n## err <= {thr * 100:g}%  (mean over layers of the class)\n")
        lines.append("| kind | layers | mode | rule | b=1 | b=8 | b=16 | b=32 | b=64 | b=128 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for kind in sp.KINDS:
            for cname, ls in (("DeltaNet", cls["DN"]), ("Attn", cls["ATT"])):
                for mode, rules in (("T1", ("abs_topk", "wina_topk", "abs_thr", "static", "seq_own", "max_topk")), ("U8_union", ("abs_topk", "wina_topk")), ("U8_max", ("abs_topk", "wina_topk"))):
                    for rule in rules:
                        vals = []
                        for b in (1, 8, 16, 32, 64, 128):
                            vals.append(agg(kind, mode, rule, b, thr, ls))
                        if all(np.isnan(v) for v in vals):
                            continue
                        lines.append(f"| {kind} | {cname} | {mode} | {rule} | " + " | ".join("-" if np.isnan(v) else f"{v:.2f}" for v in vals) + " |")
    (RES / "B1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--layers", default=",".join(map(str, sp.LIN_LAYERS)))
    ap.add_argument("--kinds", default=",".join(sp.KINDS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        summary()
    else:
        run(a.docs.split(","), [int(x) for x in a.layers.split(",")], a.kinds.split(","), a.quick, a.dev)
