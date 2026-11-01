"""Tables from results/arms.json -> results/summary.json + results/tables.md (CPU only, stdlib + numpy).

reproduce: ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/report.py

Mean accepted length of a set of prompts = 1 + sum(accepted)/sum(drafts) (token-weighted pooled ratio, as E0 reports).
Relative = arm / base on the same prompts; 95% CI = paired bootstrap over prompts (stratified by category), 4000 resamples.
Noise = base2/base (identical config run twice in the same engine).
Free rule (README): pooled relative accepted length >= 0.97 in both samplers AND the loss is not distinguishable from 0 within the
base2/base noise (|loss| beyond noise only counts if > 3%); per-category cells below 0.97 are flagged.
Modelled k=3 tok/s ratio vs stock vLLM: E3 step times (experiments/2026-10-30-w4-multirow-verify-and-power/results/e2e_tables.md
Q3c/Q3d): stock vLLM step = E_k / measured tok/s of E0, int4-MTP+32k-head fused engine step (b) = 21.78/22.78/23.85 ms for k=1/2/3
(2K ctx) -> ratio = rel x (stock_step / b_step) at the same acceptance.
"""
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).parent
CATS = ["code", "math", "prose", "agent", "long"]
B_STEP = {1: 21.78, 2: 22.78, 3: 23.85}  # ms, E3 Q3c (b) at 2K
STOCK_STEP3 = 37.2  # ms, E0 k=3 code step = 3.589/96.4 tok/s
rng = np.random.default_rng(0)


def pooled(rows, keys):
    d = sum(rows[k]["drafts"] for k in keys)
    a = sum(rows[k]["accepted"] for k in keys)
    pos = np.sum([rows[k]["accepted_per_pos"] for k in keys], axis=0) / d
    return 1 + a / d, pos


def prompts_by_cat(rows, sampler):
    """{cat: [[row keys of prompt 1], [row keys of prompt 2], ...]} -- T=1 has two seed replicates per prompt; the prompt is the
    resampling unit of the bootstrap."""
    per = {}
    for k, r in rows.items():
        if r["sampler"] == sampler:
            per.setdefault((r["cat"], r["prompt"]), []).append(k)
    out = {c: [] for c in CATS}
    for (c, _), ks in per.items():
        out[c].append(ks)
    return out


def flat(groups):
    return [k for g in groups for k in g]


def rel_boot(rows_a, rows_0, cat_keys, n=4000):
    """Paired bootstrap of L_a/L_0 resampling prompts within each category; returns (point, lo, hi)."""
    def ratio(sel):
        return pooled(rows_a, sel)[0] / pooled(rows_0, sel)[0]
    point = ratio([k for c in cat_keys.values() for k in flat(c)])
    reps = []
    for _ in range(n):
        sel = []
        for c in cat_keys.values():
            if c:
                idx = rng.integers(0, len(c), size=len(c))
                sel += flat([c[i] for i in idx])
        reps.append(ratio(sel))
    return point, float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def main():
    data = json.loads((EXP / "results" / "arms.json").read_text())
    arms = data["arms"]
    base = arms["base"]["rows"]
    out = {"arms": {}, "rule": "free if pooled rel >= 0.97 in greedy and t1 (and within noise)"}
    md = []
    for sampler in ("greedy", "t1"):
        cats = prompts_by_cat(base, sampler)
        md.append(f"\n### {sampler}: mean accepted length (E) per category, k=3; relative to base in brackets; 95% paired-bootstrap CI over prompts for 'all'\n")
        md.append("| arm | code | math | prose | agent | long | all (rel, CI) | per-pos all |")
        md.append("|---|---|---|---|---|---|---|---|")
        for name, arm in arms.items():
            rows = arm["rows"]
            cells, rec = [], {"cat": {}}
            for c in CATS:
                keys = flat(cats[c])
                L, _ = pooled(rows, keys)
                L0, _ = pooled(base, keys)
                rec["cat"][c] = {"E": round(L, 3), "rel": round(L / L0, 4), "n_prompts": len(cats[c])}
                cells.append(f"{L:.2f} ({L / L0:.3f})")
            allk = [k for c in cats.values() for k in flat(c)]
            L, pos = pooled(rows, allk)
            pt, lo, hi = rel_boot(rows, base, cats)
            rec["all"] = {"E": round(L, 3), "rel": round(pt, 4), "ci": [round(lo, 4), round(hi, 4)],
                          "per_pos": [round(float(x), 3) for x in pos]}
            out["arms"].setdefault(name, {"spec": arm["spec"]})[sampler] = rec
            md.append(f"| {name} | " + " | ".join(cells) + f" | {L:.2f} ({pt:.3f} [{lo:.3f},{hi:.3f}]) | "
                      + "/".join(f"{x:.2f}" for x in pos) + " |")

    # noise floor: (a) base2 vs base is an exact replay (same engine, same seeds -> bitwise-identical draft/accept counts), so it
    # measures nothing but determinism; (b) the realistic floor is how far near-no-op arms (int8 MTP, int8 head: <0.5% weight
    # error) move pooled acceptance, plus the paired-bootstrap CI half-width.
    out["noise_base2_over_base"] = {s: out["arms"]["base2"][s]["all"]["rel"] for s in ("greedy", "t1")} if "base2" in arms else None
    nofx = [a for a in ("mtp-int8", "head-int8") if a in out["arms"]]
    floor = max([abs(1 - out["arms"][a][s]["all"]["rel"]) for a in nofx for s in ("greedy", "t1")] or [0.0])
    half = max([(out["arms"][a][s]["all"]["ci"][1] - out["arms"][a][s]["all"]["ci"][0]) / 2 for a in nofx for s in ("greedy", "t1")] or [0.0])
    out["noise_floor"] = {"near_noop_arms": nofx, "max_abs_pooled_dev": round(floor, 4), "max_ci_halfwidth": round(half, 4)}
    # token-stream identity vs base per sampler family (target numerics differ slightly with different accepted prefixes)
    same = {}
    for name, arm in arms.items():
        same[name] = {}
        for fam in ("greedy", "t1"):
            eq = [arm["rows"][k]["sha"] == base[k]["sha"] for k in base if base[k]["sampler"] == fam and k in arm["rows"]]
            same[name][fam] = f"{sum(eq)}/{len(eq)}"
    out["output_identical_to_base"] = same

    # verdicts
    md.append("\n### verdicts (FREE = pooled rel >= 0.97 in both samplers; 'CI' = also the lower 95%% CI bound >= 0.97 in both; noise floor: max pooled |1-rel| of near-no-op arms = %.4f, max CI half-width %.4f)\n" % (floor, half))
    md.append("| arm | greedy rel [CI] | t1 rel [CI] | worst category cell | verdict |")
    md.append("|---|---|---|---|---|")
    for name, rec in out["arms"].items():
        if name in ("base", "base2", "base3"):
            continue
        g, t = rec["greedy"]["all"], rec["t1"]["all"]
        worst = min(((rec[s]["cat"][c]["rel"], f"{c}/{s}") for s in ("greedy", "t1") for c in CATS))
        free = g["rel"] >= 0.97 and t["rel"] >= 0.97
        sure = g["ci"][0] >= 0.97 and t["ci"][0] >= 0.97
        flag = "" if worst[0] >= 0.97 else f" (cell <0.97: {worst[1]})"
        rec["verdict"] = ("FREE" + (" (CI)" if sure else " (CI straddles 0.97)") if free else "NOT free") + flag
        md.append(f"| {name} | {g['rel']:.3f} [{g['ci'][0]:.3f},{g['ci'][1]:.3f}] | {t['rel']:.3f} [{t['ci'][0]:.3f},{t['ci'][1]:.3f}] "
                  f"| {worst[0]:.3f} ({worst[1]}) | {rec['verdict']} |")

    # modelled fused-engine tok/s per arm: step(k) = verify(T=k+1) + k x draft_ms(arm); verify from E3 Q3c (2K ctx, FP8 KV);
    # draft_ms = 0.1 (fixed) + MTP-layer bytes / BW + head bytes / BW with the arm's own byte counts: MTP layer 0.85 GB bf16 / 0.43 int8 /
    # 0.22 int4 (E3 e2e_tables.md Q3b), head = rows x 5120 x {2, 1, 0.53} B, rows = V (restricted) or 248320, BW 883 GB/s (measured
    # lm_head GEMV) except the int4 layer 813 GB/s. Arm 'base' (bf16 layer + full bf16 head) reproduces E3's 3.94 ms draft step (a).
    # Stock vLLM reference = base acceptance at E0's measured k=3 step (37.2 ms).
    md.append("\n### modelled fused-engine tok/s per arm (E3 step-time model with each arm's own draft bytes; acceptance = this experiment, all prompts pooled)\n")
    md.append("| arm | draft ms | greedy: best k, tok/s, x stock k=3 | t1: best k, tok/s, x stock k=3 |")
    md.append("|---|---|---|---|")
    VER = {1: 20.42, 2: 21.03, 3: 21.28, 4: 21.60}
    for name, rec in out["arms"].items():
        sp = rec["spec"]
        layer = {None: (0.85e9, 883e9), "int8": (0.43e9, 883e9), "int4": (0.22e9, 813e9), "int4clip": (0.22e9, 813e9)}[sp.get("mtp")]
        rows = sp.get("vocab", 248320)
        bpp = {None: 2.0, "int8": 1.0, "int4": 0.53}[sp.get("head")]
        draft_ms = 0.1 + layer[0] / layer[1] * 1e3 + rows * 5120 * bpp / 883e9 * 1e3
        rec["draft_ms"] = round(draft_ms, 3)
        cells = []
        for s in ("greedy", "t1"):
            pos = rec[s]["all"]["per_pos"]
            e_base3 = out["arms"]["base"][s]["all"]["E"]
            best = max(range(1, 4), key=lambda k: (1 + sum(pos[:k])) / (VER[k + 1] + k * draft_ms))
            toks = (1 + sum(pos[:best])) / (VER[best + 1] + best * draft_ms) * 1000
            ratio = toks / ((e_base3 / STOCK_STEP3) * 1000)
            rec[s]["model"] = {"best_k": best, "tok_s": round(toks, 1), "ratio_vs_stock_k3": round(ratio, 3)}
            cells.append(f"k={best}, {toks:.0f}, {ratio:.2f}x")
        md.append(f"| {name} | {draft_ms:.2f} | " + " | ".join(cells) + " |")
    # independence check: do the effects multiply?
    if all(a in out["arms"] for a in ("v32k+ctx", "mtp-int4", "head-int8", "comb-head8")):
        md.append("\n### independence: rel(comb-head8) vs rel(v32k+ctx) x rel(mtp-int4) x rel(head-int8)\n")
        for s in ("greedy", "t1"):
            pr = out["arms"]["v32k+ctx"][s]["all"]["rel"] * out["arms"]["mtp-int4"][s]["all"]["rel"] * out["arms"]["head-int8"][s]["all"]["rel"]
            md.append(f"- {s}: product {pr:.3f}, measured {out['arms']['comb-head8'][s]['all']['rel']:.3f}")
    # NOT MEASURED: predicted combined 48k+gen / MTP int4 / head int8 arm by multiplying per-position acceptance ratios
    if all(a in out["arms"] for a in ("v48k+gen", "mtp-int4", "head-int8")):
        md.append("\n### predicted (independence of the three effects; NOT measured): V=49152+prompt+gen, MTP int4, head int8\n")
        draft_ms = 0.1 + 0.22e9 / 813e9 * 1e3 + 49152 * 5120 * 1.0 / 883e9 * 1e3
        for s in ("greedy", "t1"):
            b = out["arms"]["base"][s]["all"]["per_pos"]
            pos = [b[i] * np.prod([out["arms"][a][s]["all"]["per_pos"][i] / b[i] for a in ("v48k+gen", "mtp-int4", "head-int8")]) for i in range(3)]
            e = 1 + sum(pos)
            toks = e / (VER[4] + 3 * draft_ms) * 1000
            ratio = toks / ((out["arms"]["base"][s]["all"]["E"] / STOCK_STEP3) * 1000)
            md.append(f"- {s}: E {e:.3f} = {e / out['arms']['base'][s]['all']['E']:.3f} x base, draft {draft_ms:.2f} ms, k=3 {toks:.0f} tok/s = {ratio:.2f}x stock vLLM k=3")
    # mechanism: OOV rate of target-generated tokens vs measured position-1 acceptance
    oovp = EXP / "results" / "oov.json"
    if oovp.exists():
        oov = json.loads(oovp.read_text())
        md.append("\n### mechanism: share of the target's generated tokens outside the draft vocabulary (base3 generations, greedy+T=1) vs measured relative position-1 acceptance (greedy, all prompts)\n")
        md.append("| V | OOV all (code/math/prose/agent/long) | OOV after + prompt tokens | pos-1 acc rel, V only | pos-1 acc rel, V+prompt(+gen) | 1 - OOV |")
        md.append("|---|---|---|---|---|---|")
        for V, tag in ((8192, "v8k"), (16384, "v16k"), (32768, "v32k"), (49152, "v48k"), (65536, "v64k")):
            o = oov.get(str(V))
            if not o or tag not in out["arms"]:
                continue
            p1 = out["arms"][tag]["greedy"]["all"]["per_pos"][0] / out["arms"]["base"]["greedy"]["all"]["per_pos"][0]
            ctxarm = next((a for a in (tag + "+ctx", tag + "+gen") if a in out["arms"]), None)
            p1c = (out["arms"][ctxarm]["greedy"]["all"]["per_pos"][0] / out["arms"]["base"]["greedy"]["all"]["per_pos"][0]) if ctxarm else float("nan")
            md.append(f"| {V} | {o['all']['oov']:.3f} ({'/'.join(f'{o[c]['oov']:.3f}' for c in CATS)}) | {o['all']['oov_after_ctx']:.3f} | {p1:.3f} | {p1c:.3f} ({ctxarm}) | {1 - o['all']['oov']:.3f} |")
    out["output_identical_to_base"] = same
    md.append("\nOutputs byte-identical to base (sha of generated token ids, per prompt/sampler): " + json.dumps(same))
    (EXP / "results" / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    (EXP / "results" / "tables.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
