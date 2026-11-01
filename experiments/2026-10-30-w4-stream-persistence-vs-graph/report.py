#!/usr/bin/env python3
"""Render results/*.json into markdown tables (results/tables.md) so README numbers are copied, not retyped."""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
out = []
def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None

main = load("main.json")
if main:
    res = {r["name"]: r for r in main["results"]}
    v0 = res["v0"]["median_ms"]
    peak = max(res["v0"]["GBps"], res["memcpy"]["GBps"])
    out.append(f"### Main table (median of {main['repeats']} interleaved repeats x {main['steps']} steps, {main['warmup']} warmup; {main['bytes_per_step_weights']/1e9:.3f} GB streamed/step, {main['phases']} phases / {main['gemv_phases']} GEMV)\n")
    out.append("| variant | ms/step | rep medians (ms) | GB/s | % of V0 (read-only) | % of 936 GB/s | step p10-p90 (ms) |\n|---|---|---|---|---|---|---|")
    for r in main["results"]:
        out.append(f"| {r['name']} | {r['median_ms']:.3f} | {', '.join(f'{x:.3f}' for x in r['rep_medians_ms'])} | {r['GBps']:.1f} | {100*r['GBps']/res['v0']['GBps']:.1f} | {100*r['GBps']/936:.1f} | {r['step_p10_ms']:.3f}-{r['step_p90_ms']:.3f} |")
    out.append("\n(memcpy row: bytes = read+write of a 1 GiB D2D copy.)\n")
for name, title in (("sweep.json", "Config sweep (2 repeats x 100 steps; info column = grid / smem)"), ("ablation.json", "Ablation of the persistent kernel (results invalid when abl!=0: bit0 no grid barrier, bit1 no activation prologue, bit2 no core phases)")):
    d = load(name)
    if d:
        out.append(f"### {title}\n\n| spec | ms/step | GB/s | info |\n|---|---|---|---|")
        for r in d["results"]:
            out.append(f"| {r['name']} | {r['median_ms']:.3f} | {r['GBps']:.1f} | {r['info']} |")
        out.append("")
d = load("barrier.json")
if d:
    out.append("### Grid barrier microbenchmark (empty phases, 20000 barriers, median of 5)\n\n| mode | blocks | threads | us/barrier |\n|---|---|---|---|")
    for r in d["barrier_us"]:
        out.append(f"| {r['mode']} | {r['blocks']} | {r['threads']} | {r['us_per_barrier_median']:.3f} |")
    out.append("")
d = load("profile.json")
if d:
    out.append("### Per-phase timeline of the persistent kernel (globaltimer, mean over phases of a kind and 20 steps; us)\n")
    for key, p in d.items():
        out.append(f"**{key}** (grid {p['grid']})\n\n| kind | phases/step | phase wall | start skew | prologue | stream (median block) | tail (last-median arrival) | barrier (last arrival->first release) | ms/step |\n|---|---|---|---|---|---|---|---|---|")
        for g in p["groups"]:
            out.append(f"| {g['kind']} | {g['phases_per_step']:.0f} | {g['phase_wall_us']:.2f} | {g['start_skew_us']:.2f} | {g['prologue_us']:.2f} | {g['stream_median_block_us']:.2f} | {g['tail_us(last_arrival-median_arrival)']:.2f} | {g['barrier_us(last_arrival->first_release)']:.2f} | {g['ms_per_step_total']:.3f} |")
        out.append("")
d = load("gemv_sweep.json")
if d:
    out.append("### Single-shape GEMV bandwidth (graph of back-to-back launches over distinct matrices, incl. launch gaps); best 3 per shape + read-only\n\n| shape | kernel | cfg | GB/s |\n|---|---|---|---|")
    shapes = {}
    for r in d["gemv"]:
        shapes.setdefault(r["shape"], []).append(r)
    for s, rs in shapes.items():
        for r in sorted(rs, key=lambda r: -r["GBps"])[:4] + [x for x in rs if x["kernel"].startswith("read") and x not in sorted(rs, key=lambda r: -r["GBps"])[:4]]:
            out.append(f"| {s} | {r['kernel']} | {r['cfg']} | {r['GBps']:.1f} |")
    out.append("")
d = load("gemv_grid_study.json")
if d:
    out.append("### GEMV bandwidth vs CTA count (V1a, GB/s)\n\n| shape | cfg | " + " | ".join(f"grid {g}" for g in d) + " |\n|---|---|---|---|---|")
    keys = {}
    for g, v in d.items():
        for r in v["gemv"]:
            if r["kernel"] == "V1a":
                keys.setdefault((r["shape"], r["cfg"]), {})[g] = r["GBps"]
    for (s, c), row in keys.items():
        out.append(f"| {s} | {c} | " + " | ".join(f"{row.get(g, float('nan')):.1f}" for g in d) + " |")
    out.append("")
d = load("check.json")
if d:
    out.append("### Correctness (full model vs naive fp32 reference)\n\n" + json.dumps(d, indent=1) + "\n")
(R / "tables.md").write_text("\n".join(out))
print("\n".join(out))
