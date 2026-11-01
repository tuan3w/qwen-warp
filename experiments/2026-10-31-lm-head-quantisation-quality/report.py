"""results/summary.json + runs/bench_head.log -> results/tables.md  (CPU only).  python3 report.py"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.load(open(HERE / "results" / "summary.json"))
ARMS = [a for a in S["arms"] if a in S["weights"]]
RESC = [a for a in S["arms"] if "+rescore" in a]
STEP_T1, STEP_T4, CYCLE_MTP = 20.9, 22.5, 24.2  # engine/BENCH.md: plain step, T=4 verify step, k=3 cycle (verify 22.5 + draft path 1.6)


def ci(e, k, fmt="{:.4f}", pct=False):
    m = e[k]
    f = (lambda v: f"{100 * v:.2f}") if pct else (lambda v: fmt.format(v))
    return f"{f(m['mean'])} [{f(m['lo'])}, {f(m['hi'])}]"


def verdict(e):
    c1 = e["top1"]["mean"] >= 0.995
    d = e["dnll"]
    c2 = (d["lo"] <= 0 <= d["hi"]) or d["mean"] < 0.002
    c3 = e["tv_k20p95_T1"]["mean"] < 0.005
    return c1, c2, c3


L = []
L.append(f"Positions: {S['arms'][ARMS[0]]['all']['n']} evaluated (docs by category: {S['n_docs']}); CIs = document-cluster bootstrap (2000 resamples).\n")
L.append("### Table 1. Pooled over all evaluated positions (code + prose + math + greedy chat)\n")
L.append("| head arm | MB | weight rel-L2 | top-1 agree % [95% CI] | #flips | KL full (nats) | KL top-20 | top-20 mass moved | dNLL nats [CI] | TV k20/p.95 T=1 | TV k20/p.95 T=0.7 | crit top1 / dNLL / TV |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for a in ARMS + ["bf16_logits_rounded"]:
    e = S["arms"][a]["all"]
    w = S["weights"].get(a)
    c = verdict(e)
    L.append(f"| {a} | {w['MB']:.0f} | {w['rel_l2']:.4f} | {ci(e, 'top1', pct=True)} | {e['n_flips']} | {e['kl_full']['mean']:.2e} | {e['kl_top20']['mean']:.2e} | {e['mass_top20']['mean']:.2e} | {ci(e, 'dnll', '{:+.5f}')} | {ci(e, 'tv_k20p95_T1')} | {e['tv_k20p95_T07']['mean']:.4f} | "
             + " / ".join("pass" if x else "FAIL" for x in c) + " |" if w else
             f"| {a} (floor: ref logits rounded to bf16) | 2543 | - | {ci(e, 'top1', pct=True)} | {e['n_flips']} | {e['kl_full']['mean']:.2e} | {e['kl_top20']['mean']:.2e} | {e['mass_top20']['mean']:.2e} | {ci(e, 'dnll', '{:+.5f}')} | {e['tv_k20p95_T1']['mean']:.4f} | {e['tv_k20p95_T07']['mean']:.4f} | - |")
L.append("\n(bf16 reference head = 2543 MB; dNLL = NLL_quant - NLL_ref of the true next token, > 0 = worse.  Scale format: int8/int6 fp32 per row, int4 bf16 per 128 columns; see Method.)\n")

L.append("### Table 2. Noise floors (same metric code, same hidden states unless stated)\n")
L.append("| floor | n | top-1 agree % [CI] | #flips | KL full | dNLL [CI] | TV k20/p.95 T=1 | TV T=0.7 |")
L.append("|---|---|---|---|---|---|---|---|")
for n, r in S["floor"].items():
    e = r["all"]
    L.append(f"| {n} | {e['n']} | {ci(e, 'top1', pct=True)} | {e['n_flips']} | {e['kl_full']['mean']:.2e} | {ci(e, 'dnll', '{:+.5f}')} | {e['tv_k20p95_T1']['mean']:.4f} | {e['tv_k20p95_T07']['mean']:.4f} |")
rel = S.get("floor_hidden_rel_l2_T1_vs_T4")
if rel:
    L.append(f"\nFinal-norm hidden rel-L2 T=1 vs T=4 rows per floor doc: {', '.join(f'{r:.4f}' for r in rel)}.\n")

L.append("### Table 3. Per category (top-1 agree % / dNLL / TV k20p.95 T=1)\n")
cats = ["code", "prose", "math", "greedy", "natural"]
L.append("| head arm | " + " | ".join(cats) + " |")
L.append("|---|" + "---|" * len(cats))
for a in ARMS + ["bf16_logits_rounded"]:
    cells = []
    for c in cats:
        e = S["arms"][a].get(c)
        cells.append(f"{100 * e['top1']['mean']:.2f} / {e['dnll']['mean']:+.4f} / {e['tv_k20p95_T1']['mean']:.4f}" if e else "-")
    L.append(f"| {a} | " + " | ".join(cells) + " |")

L.append("\n### Table 4. Flips and the MTP-verify proxy (aligned 4-row blocks; ideal drafter = reference argmax chain)\n")
L.append("| head arm | n flips | median ref top1-top2 logit margin at flips | mean margin | blocks | P(any argmax change in 4 rows) % | P(change in rows 0-2) % | accepted drafts lost per cycle (of 3) |")
L.append("|---|---|---|---|---|---|---|---|")
for a in ARMS + ["bf16_logits_rounded"] + RESC:
    e, b = S["arms"][a]["all"], S["arms"][a]["blocks_all"]
    mm, fm = e["flip_ref_margin_median"], e["flip_ref_margin_mean"]
    L.append(f"| {a} | {e['n_flips']} | {'-' if mm is None else f'{mm:.3f}'} | {'-' if fm is None else f'{fm:.3f}'} | {b['blocks']} | {100 * b['p_any4']:.2f} | {100 * b['p_any_rows012']:.2f} | {b['accepted_drafts_lost']:.4f} |")


if RESC:
    L.append("\n### Table 1b. Post-hoc (exploratory, added after Table 1): quantised head + exact rescoring of the top-M candidates (their logits replaced by the bf16-weight logits)\n")
    L.append("| arm | top-1 agree % [CI] | #flips | KL full | dNLL [CI] | TV k20/p.95 T=1 [CI] | TV T=0.7 | crit top1 / dNLL / TV |")
    L.append("|---|---|---|---|---|---|---|---|")
    for a in RESC:
        e = S["arms"][a]["all"]
        L.append(f"| {a} | {ci(e, 'top1', pct=True)} | {e['n_flips']} | {e['kl_full']['mean']:.2e} | {ci(e, 'dnll', '{:+.5f}')} | {ci(e, 'tv_k20p95_T1')} | {e['tv_k20p95_T07']['mean']:.4f} | " + " / ".join("pass" if x else "FAIL" for x in verdict(e)) + " |")

import numpy as np
npz = HERE / "runs" / "per_position.npz"
if npz.exists():
    Z = np.load(npz)
    m = Z["int8_rtn/ref_margin"]
    edges = [0, 0.01, 0.03, 0.1, 0.3, 1.0, 1e9]
    L.append("\n### Table 7. Why top-1 agreement is brittle: reference top1-top2 logit margin of the position vs argmax flip rate\n")
    arms7 = ["int8_rtn", "int8_g128_rtn", "int6_rtn", "int4_g128_awclip", "bf16_logits_rounded", "int8_rtn+rescore8", "int4_g128_awclip+rescore16"]
    arms7 = [a for a in arms7 if f"{a}/top1" in Z.files]
    L.append("| margin bin (logits) | positions | share % | flip rate % per arm: " + " / ".join(arms7) + " |")
    L.append("|---|---|---|---|")
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (m >= lo) & (m < hi)
        if mk.sum() == 0:
            continue
        L.append(f"| [{lo}, {hi if hi < 1e8 else 'inf'}) | {int(mk.sum())} | {100 * mk.mean():.2f} | " + " / ".join(f"{100 * (1 - Z[a + '/top1'][mk].mean()):.2f}" for a in arms7) + " |")

# ---- speed
bl = HERE / "runs" / "bench_head.log"
if bl.exists():
    rows = {}
    for ln in bl.read_text().splitlines():
        m = re.match(r"^(.*?)\s+((?:\d+\.\d+\(\s*\d+\)\s*){4})\s*$", ln)
        if m:
            vals = re.findall(r"(\d+\.\d+)\(\s*(\d+)\)", m.group(2))
            rows[m.group(1).strip()] = [(float(a), int(b)) for a, b in vals]
    L.append("\n### Table 5. lm_head GEMV, N=248320 K=5120, L2-cold, ms (GB/s)\n")
    L.append("| kernel | T=1 | T=2 | T=3 | T=4 |")
    L.append("|---|---|---|---|---|")
    for k, v in rows.items():
        L.append(f"| {k} | " + " | ".join(f"{a:.3f} ({b})" for a, b in v) + " |")
    bf = rows.get("bf16 smk_bf16_gemv_f32 (shipped)")
    i8 = rows.get("int8 per-row CUDA-core (this)")
    w4 = [v for k, v in rows.items() if k.startswith("int4 g128")]
    best4 = [min(v[t][0] for v in w4) for t in range(4)] if w4 else None
    if bf and i8:
        L.append("\n### Table 6. Modelled gain (measured head times above; step = engine/BENCH.md: plain 20.9 ms, T=4 verify 22.5 ms, k=3 cycle 24.2 ms)\n")
        L.append("| head | T=1 head ms | T=1 step ms | gain ms / % | tok/s plain (47.8 base) | T=4 head ms | T=4 verify ms | gain ms / % | MTP k=3 cycle speedup (draft path unchanged) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for name, t in (("int8 per-row", [x[0] for x in i8]), ("int4 g128 (best cfg per T)", best4)):
            if not t:
                continue
            g1, g4 = bf[0][0] - t[0], bf[3][0] - t[3]
            L.append(f"| {name} | {t[0]:.3f} | {STEP_T1 - g1:.2f} | {g1:.2f} / {100 * g1 / STEP_T1:.1f}% | {1000 / (STEP_T1 - g1):.1f} | {t[3]:.3f} | {STEP_T4 - g4:.2f} | {g4:.2f} / {100 * g4 / STEP_T4:.1f}% | x{CYCLE_MTP / (CYCLE_MTP - g4):.3f} (code 145 -> {145 * CYCLE_MTP / (CYCLE_MTP - g4):.0f} tok/s) |")

    ov = {}
    for ln in bl.read_text().splitlines():
        mo = re.match(r"^T=(\d) +select\+rescore ([\d.]+) ms", ln)
        if mo:
            ov[int(mo.group(1))] = float(mo.group(2))
    if ov and bf and i8 and best4:
        L.append("\n### Table 8. Net throughput model with exact rescoring (M=16): head ms + measured select+rescore overhead; tokens/cycle lost = Table 4 'accepted drafts lost' (ideal-drafter upper bound)\n")
        L.append("| head | T=4 head ms + overhead | verify gain ms | workload (tok/s, tokens/cycle at 24.2 ms) | cycle speedup | accepted-length factor (upper-bound loss) | net tok/s |")
        L.append("|---|---|---|---|---|---|---|")
        wl = {"code": 145.0, "prose": 94.7, "math": 143.0, "8k": 115.0}
        for name, headms, oh, arm in (("int8 per-row", i8[3][0], 0.0, "int8_rtn"), ("int8 per-row + rescore16", i8[3][0], ov.get(4, 0.0), "int8_rtn+rescore16"),
                                      ("int4 g128 awclip + rescore16", best4[3], ov.get(4, 0.0), "int4_g128_awclip+rescore16"), ("int4 g128 awclip (no rescore)", best4[3], 0.0, "int4_g128_awclip")):
            if arm not in S["arms"]:
                continue
            g = bf[3][0] - headms - oh
            lost = S["arms"][arm]["blocks_all"]["accepted_drafts_lost"]
            for wn, tps in wl.items():
                E = tps * CYCLE_MTP / 1000
                f1 = CYCLE_MTP / (CYCLE_MTP - g)
                f2 = (E - lost) / E
                L.append(f"| {name} | {headms:.3f} + {oh:.3f} | {g:.2f} | {wn} ({tps}, {E:.2f}) | x{f1:.3f} | x{f2:.3f} | {tps * f1 * f2:.1f} |")

(HERE / "results" / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
