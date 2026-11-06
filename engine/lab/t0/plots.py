"""Small PNGs from the result CSV/JSONL.  uv run --no-project --with matplotlib --with numpy python plots.py   (no torch needed)"""
import collections
import csv
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results")
KINDS = ("qkv", "o", "gu", "down")


def b1():
    E = collections.defaultdict(list)
    for r in csv.DictReader(open(R / "b1/b1_raw.csv")):
        if r["mode"] == "T1" and r["rule"] in ("abs_topk",):
            E[(r["kind"], int(r["b"]), float(r["f"]))].append((float(r["bytes"]), float(r["err"])))
        if r["mode"] == "U8_union" and r["rule"] == "abs_topk" and int(r["b"]) == 1:
            E[(r["kind"], "U8", float(r["f"]))].append((float(r["bytes"]), float(r["err"])))
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
    for a, kind in zip(ax, KINDS):
        for b, st in ((1, "-"), (8, "--"), (32, "-."), (128, ":"), ("U8", "-")):
            pts = sorted((np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for (k, bb, f), v in E.items() if k == kind and bb == b)
            if pts:
                a.plot([p[0] for p in pts], [p[1] * 100 for p in pts], st, marker="o", ms=3, label=f"b={b}" if b != "U8" else "T=8 union, b=1")
        a.set_title(kind)
        a.set_xlabel("weight-byte fraction read")
        a.set_yscale("log")
        a.axhline(2.4, color="gray", lw=0.5)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("mean per-row rel. output error (%)")
    ax[0].legend(fontsize=7)
    fig.suptitle("B1: |x| top-k activation sparsity, real traces (6 docs x 8 layers)")
    fig.tight_layout()
    fig.savefig(R / "b1/b1_curves.png", dpi=80)


def b2():
    D = collections.defaultdict(list)
    for r in csv.DictReader(open(R / "b2/b2_raw.csv")):
        if r["scorer"] in ("tok_oracle", "oracle", "quest", "maxlogit") and r["pool"] == "head" and r["b"] in ("1", "64"):
            D[(r["scorer"], int(r["layer"]), int(r["budget"]))].append(float(r["kl"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for sc, st in (("tok_oracle", "-"), ("oracle", "--"), ("maxlogit", ":"), ("quest", "-.")):
        buds = sorted({k[2] for k in D if k[0] == sc})
        ax[0].plot(buds, [np.mean([np.mean(v) for k, v in D.items() if k[0] == sc and k[2] == b]) for b in buds], st, marker="o", label=sc + (" b=64" if sc != "tok_oracle" else ""))
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("tokens read per (row, head) incl. sink+ring")
    ax[0].set_ylabel("attention KL (nats), 32k")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    for l in (7, 23, 39, 63):
        buds = sorted({k[2] for k in D if k[0] == "tok_oracle" and k[1] == l})
        ax[1].plot(buds, [np.mean(D[("tok_oracle", l, b)]) for b in buds], marker="o", label=f"layer {l}")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("budget (tokens)")
    ax[1].set_title("token oracle by depth")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "b2/b2_curves.png", dpi=80)


def kl():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = [json.loads(l) for f in glob.glob(str(R / "kl/*calib_attn.jsonl")) for l in open(f)]
    for pre, st in (("attn_tok_oracle_", "-"), ("attn_oracle_b64_", "--"), ("attn_quest_b64_", "-."), ("attn_quest_rot_b64_", ":")):
        d = collections.defaultdict(list)
        for r in rows:
            if r["name"].startswith(pre) and r["name"][len(pre):].isdigit():
                d[int(r["name"][len(pre):])].append(r["kl"] * 100)
        if d:
            ax[0].plot(sorted(d), [np.mean(d[b]) for b in sorted(d)], st, marker="o", label=pre[5:-1])
    ax[0].axhspan(0.05, 0.15, color="green", alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("tokens read per (row, head), all 16 attention layers")
    ax[0].set_ylabel("real-model logit KL (%), 32k, mean of 3 docs")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    rows = [json.loads(l) for f in glob.glob(str(R / "kl/*calib_act.jsonl")) for l in open(f)]
    d = collections.defaultdict(list)
    for r in rows:
        if r["name"].startswith("act_") and r["name"].endswith("_b1"):
            kind, f = r["name"].split("_")[1], float(r["name"].split("_f")[1].split("_")[0])
            d[(kind, f)].append(r["kl"] * 100)
    for kind in ("qkv", "o", "gu", "down", "all"):
        fs = sorted({k[1] for k in d if k[0] == kind})
        ax[1].plot(fs, [np.mean(d[(kind, f)]) for f in fs], marker="o", label=kind)
    ax[1].axhspan(0.05, 0.15, color="green", alpha=0.15)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("kept fraction of K channels (per-channel |x| top-k, T=1)")
    ax[1].set_ylabel("real-model logit KL (%)")
    ax[1].invert_xaxis()
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "kl/kl_curves.png", dpi=80)


if __name__ == "__main__":
    b1()
    b2()
    kl()
    print("ok")
