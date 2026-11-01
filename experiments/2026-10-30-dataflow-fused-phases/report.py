#!/usr/bin/env python3
"""Render results/*.json into markdown tables (results/tables.md) so README numbers are copied, not retyped."""
import json
import re
from pathlib import Path

R = Path(__file__).parent / "results"
out = []


def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else None


def phases(info: str) -> str:
    m = re.search(r"dependent_phases/step=(\d+)", info)
    return m.group(1) if m else ("338" if info.startswith(("grid", "single", "cudaMemcpy", "257")) and "single" not in info else "-")


summ = load("run_summary.json") or {}
clk = summ.get("clocks_per_variant", {})
main = load("main.json")
if main:
    res = {r["name"]: r for r in main["results"]}
    v0 = res["v0"]["GBps"]
    v1b = next((r for n, r in res.items() if n.startswith("v1b:") and ":pf" not in n), None)
    out.append(f"### Main table (median of {main['repeats']} interleaved repeats x {main['steps']} steps, {main['warmup']} warmup; {main['bytes_per_step_weights']/1e9:.3f} GB streamed/step)\n")
    out.append("| variant | ms/step | rep medians (ms) | GB/s | % of V0 | V1b/this | dependent phases/step | SM MHz (median) | power W (median) | throttle masks | info |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in main["results"]:
        c = clk.get(r["name"], {})
        ratio = f"{v1b['median_ms']/r['median_ms']:.3f}" if v1b else "-"
        out.append(f"| {r['name']} | {r['median_ms']:.3f} | {', '.join(f'{x:.3f}' for x in r['rep_medians_ms'])} | {r['GBps']:.1f} | {100*r['GBps']/v0:.1f} | {ratio} | {phases(r['info'])} | "
                   f"{c.get('sm_mhz_median', float('nan')):.0f} | {c.get('power_w_median', float('nan')):.0f} | {','.join(c.get('throttle_reason_masks_seen', []))} | {r['info']} |")
    out.append("")
for name, title in (("sweep.json", "Sweep (2 repeats x 100 steps)"), ("ablation.json", "Ablation (results invalid when abl!=0: bit0 no grid barrier / no counter waits)"),
                    ("power.json", "Power: ALU ablation (cheap=1: dequant ALU removed, results invalid) vs normal dequant, 3 x 100 steps")):
    d = load(name)
    if d:
        out.append(f"### {title}\n\n| spec | ms/step | GB/s | info |\n|---|---|---|---|")
        for r in d["results"]:
            out.append(f"| {r['name']} | {r['median_ms']:.3f} | {r['GBps']:.1f} | {r['info']} |")
        out.append("")
if clk and (R / "power.json").exists():
    out.append("### Clocks/power during the power-stage windows\n\n| variant | SM MHz median | power W median | throttle masks |\n|---|---|---|---|")
    for k, c in (summ.get("power_windows") or {}).items():
        out.append(f"| {k} | {c['sm_mhz_median']:.0f} | {c['power_w_median']:.0f} | {','.join(c['throttle_reason_masks_seen'])} |")
    out.append("")
for name in ("check4.json", "check64.json"):
    d = load(name)
    if d:
        out.append(f"### Correctness {name} (naive fp32 reference; late-scale reference = RMS scale applied to the dot product)\n")
        out.append("```json\n" + json.dumps(d, indent=1) + "\n```\n")
(R / "tables.md").write_text("\n".join(out))
print("\n".join(out))
