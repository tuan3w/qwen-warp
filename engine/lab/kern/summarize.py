#!/usr/bin/env python3
"""Markdown tables from results/kern/*.json (what run.sh --full wrote).  usage: python3 summarize.py [results_dir]"""
import json
import sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/kern"
g = json.load(open(f"{D}/kern_gemv.json"))
a = json.load(open(f"{D}/kern_attn.json"))
m = json.load(open(f"{D}/kern_misc.json"))

print("### floors\n")
for r in m:
    if r["what"] == "floors":
        print({k: v for k, v in r.items() if k != "what"})
print()

print("### GEMV: dense baseline (engine smk_w4r_gemv, default cfg) and fit  time = fixed + bytes / marginal\n")
print("| shape | M | dense MB | dense us | dense GB/s | best-of-6 cfg us | fit fixed us | fit marginal GB/s | dense K=128 floor us |")
print("|---|---|---|---|---|---|---|---|---|")
fit = {(r["shape"], r["M"]): r for r in g if r["what"] == "fit"}
for r in g:
    if r["what"] == "dense":
        f = fit.get((r["shape"], r["M"]), {})
        print(f"| {r['shape']} {r['N']}x{r['K']} | {r['M']} | {r['bytes']/1e6:.1f} | {r['default_us']:.1f} | {r['default_GBps']:.0f} | {r['best_us']:.1f} ({r['best_cfg']}) | {f.get('fixed_us', float('nan')):.2f} | {f.get('marginal_GBps', float('nan')):.0f} | {f.get('dense_K128_floor_us', float('nan')):.2f} |")
print()

print("### GEMV sparse: fraction of dense GB/s (sector bytes) / speedup vs dense (ideal bytes ratio)\n")
rows = [r for r in g if r["what"] == "sparse"]
shapes = []
for r in rows:
    if r["shape"] not in shapes:
        shapes.append(r["shape"])
for M in (1, 8):
    print(f"\n**M = {M}**\n")
    cols = []
    for r in rows:
        key = (r["mode"], r["B"])
        if r["M"] == M and key not in cols:
            cols.append(key)
    fs = sorted({r["f"] for r in rows if r["M"] == M}, reverse=True)
    print("| shape | mode | B | " + " | ".join(f"{int(f*100)}%" if f > 0 else "floor" for f in fs) + " |")
    print("|---|---|---|" + "---|" * len(fs))
    for s in shapes:
        for mode, B in cols:
            cells = []
            for f in fs:
                c = [r for r in rows if r["shape"] == s and r["M"] == M and r["mode"] == mode and r["B"] == B and abs(r["f"] - f) < 1e-9]
                if c:
                    r = c[0]
                    cells.append(f"{r['frac_dense_GBps_sector']:.2f} / {r['speedup']:.2f}x ({r['ideal_bytes_ratio']:.2f}x; union {r['union_frac']*100:.0f}%)")
                else:
                    cells.append("")
            if any(cells):
                print(f"| {s} | {mode} | {B} | " + " | ".join(cells) + " |")
print()

print("### mask build (k_select) cost\n")
print("| shape | M | mode | f | union | select us | select+gemv us | gemv us | dense us | speedup incl. select |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in m:
    if r["what"] == "select":
        print(f"| {r['shape']} | {r['M']} | {r['mode']} | {int(r['f']*100)}% | {r['union_frac']*100:.0f}% | {r['select_us']:.1f} | {r['select_plus_gemv_us']:.1f} | {r['gemv_us']:.1f} | {r['dense_us']:.1f} | {r['speedup_incl_select']:.2f} |")
print()

print("### attention (per layer; x16 for the step)\n")
own100 = {(r["ctx"], r["T"]): r["sparse_us"] for r in a if r["what"] == "sparse" and r["f"] >= 1.0}
print("| ctx | T | mode | f | union | dense us (engine split+reduce) | dense GB/s | sparse us | GB/s on needed bytes | frac of dense GB/s | speedup vs engine dense | speedup vs own 100% | ideal | tuned us (bps) | 16-layer saving ms | max err/rms vs fp32 ref |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in a:
    if r["what"] == "sparse":
        print(f"| {r['ctx']} | {r['T']} | {r['mode']} | {int(r['f']*100) if r['f'] > 0.005 else 'floor'}% | {r['union_frac']*100:.0f}% | {r['dense_us']:.1f} | {r['dense_GBps']:.0f} | {r['sparse_us']:.1f} | {r['GBps_needed']:.0f} | {r['frac_dense_GBps']:.2f} | {r['speedup']:.2f}x | {own100[(r['ctx'], r['T'])]/r['sparse_us']:.2f}x | {r['ideal_bytes_ratio']:.2f}x | {r['tuned_us']:.1f} ({r['tuned_bps']}) | {r['saving_16_layers_ms']:.2f} | {r['err_max_over_rms']:.1e} |")
print()
print("### attention fixed cost (block 0 only)\n")
for r in a:
    if r["what"] == "floor":
        print({k: v for k, v in r.items() if k != "what"})
for r in a:
    if r["what"] in ("ref_vs_engine", "dense"):
        print(r)
