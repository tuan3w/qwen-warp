#!/usr/bin/env python3
"""Renders results/*.json into results/tables.md and results/mtp.json (MTP break-even model).  No deps."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).parent / "results"
q1 = json.loads((RES / "q1.json").read_text())
chk = json.loads((RES / "check.json").read_text())["results"]
q2 = json.loads((RES / "q2.json").read_text())["results"]
clk1 = json.loads((RES / "clocks_q1.json").read_text()) if (RES / "clocks_q1.json").exists() else {}
clk2 = json.loads((RES / "clocks_q2.json").read_text()) if (RES / "clocks_q2.json").exists() else {}
SHAPES = ["gateup", "down", "qkvz", "oproj", "lmhead"]
DIM = {"gateup": "5120->34816", "down": "17408->5120", "qkvz": "5120->16384", "oproj": "6144->5120", "lmhead": "5120->248320 (bf16)"}
COUNT = {"gateup": 64, "down": 64, "qkvz": 48 + 16 * 14336 / 16384, "oproj": 64, "lmhead": 1}  # 16 attention qkv GEMVs are 5120->14336
BYTES = {"gateup": 34816 * 5120 * (0.5 + 1 / 64), "down": 5120 * 17408 * (0.5 + 1 / 64), "qkvz": 16384 * 5120 * (0.5 + 1 / 64),
         "oproj": 5120 * 6144 * (0.5 + 1 / 64), "lmhead": 248320 * 5120 * 2.0}
TS = [1, 2, 3, 4, 5, 6, 8, 12, 16]
out = []
P = out.append

fin = defaultdict(dict)  # (shape, kind/mode) -> T -> record
for r in q1["final"]:
    fin[(r["shape"], f"{r['kind']}/{r['mode']}")][r["T"]] = r


def best_valid(shape, T):
    """best exact (non-ablated) kernel at this T: mma sub/ones/bf16 (T>=1) and cc e1/naive/bf16 (T=1 only)."""
    c = []
    for key in (["mma/sub", "mma/ones", "cc/e1", "cc/bf16", "mma/bf16"]):
        rec = fin.get((shape, key), {}).get(T)
        if rec:
            c.append((rec["ms"], key, rec))
    return min(c) if c else None


# ------------------------------------------------------------------ correctness
P("### Correctness (fp32 GPU reference, warp-per-row; rel L2 of the bf16 output; max over all configs/T; `vs bf16(ref)` isolates the kernel error from output rounding)\n")
P("| shape | kernel | rel L2 vs fp32 ref | bf16 rounding floor | rel L2 vs bf16(ref) | n |")
P("|---|---|---|---|---|---|")
agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for r in chk:
    a = agg[(r["shape"], f"{r['kind']}/{r['mode']}")]
    a[0] = max(a[0], r["rel_l2"]); a[1] = max(a[1], r["bf16_round_floor"]); a[2] = max(a[2], r.get("rel_vs_bf16_ref", 0)); a[3] += 1
for (s, k), a in agg.items():
    P(f"| {s} | {k} | {a[0]:.2e} | {a[1]:.2e} | {a[2]:.2e} | {a[3]} |")
P("")

# ------------------------------------------------------------------ Q1 tables
ratio_tab = {}
for shape in SHAPES:
    P(f"### Q1 {shape} {DIM[shape]}: ms (best config per T) and ratio to T=1 of the same kernel family\n")
    keys = [k for (s, k) in fin if s == shape and k.startswith("mma")]
    base1 = best_valid(shape, 1)
    P("| T | " + " | ".join(f"{k} ms (cfg)" for k in keys) + " | " + " | ".join(f"{k} x" for k in keys) + " | best exact GB/s | x vs best-T=1 (cc or mma) |")
    P("|---|" + "---|" * (2 * len(keys) + 2))
    for T in TS:
        cells, ratios = [], []
        for k in keys:
            rec = fin[(shape, k)].get(T); b = fin[(shape, k)].get(1)
            cells.append(f"{rec['ms']:.4f} ({rec['cfg']})" if rec else "-")
            ratios.append(f"{rec['ms'] / b['ms']:.3f}" if rec and b else "-")
        bv = best_valid(shape, T)
        gb = BYTES[shape] / (bv[0] * 1e-3) / 1e9 if bv else 0
        P(f"| {T} | " + " | ".join(cells) + " | " + " | ".join(ratios) + f" | {gb:.0f} | {bv[0] / base1[0]:.3f} |")
        if bv:
            ratio_tab.setdefault(shape, {})[T] = {"ms": bv[0], "kernel": bv[1], "cfg": bv[2]["cfg"], "gbs": gb,
                                                  "x_vs_own_T1": bv[0] / fin[(shape, bv[1])][1]["ms"] if 1 in fin[(shape, bv[1])] else None,
                                                  "x_vs_best_T1": bv[0] / base1[0]}
    cc = [(k, fin[(shape, k)][1]) for (s, k) in fin if s == shape and k.startswith("cc") and 1 in fin[(shape, k)]]
    rd = fin.get((shape, "read/read"), {}).get(1)
    P("\nT=1 CUDA-core / read-only references: " + "; ".join(f"{k} {v['ms']:.4f} ms ({v['cfg']}, {v['gbs']:.0f} GB/s)" for k, v in cc)
      + (f"; read-only {rd['ms']:.4f} ms ({rd['gbs']:.0f} GB/s)" if rd else "") + "\n")

# T* table
P("### T* = largest T in {1,2,3,4,5,6,8,12,16} whose time stays within x of T=1 (best exact kernel at each T; base = best exact kernel at T=1, cc or mma)\n")
P("| shape | T*(1.05x) | T*(1.10x) | T*(1.15x) | T*(1.25x) | ratio@T=2 | @4 | @8 | @16 |")
P("|---|---|---|---|---|---|---|---|---|")
tstar = {}
for shape in SHAPES:
    rt = ratio_tab[shape]
    row = {}
    for th in (1.05, 1.10, 1.15, 1.25):
        ok = [T for T in TS if T in rt and rt[T]["x_vs_best_T1"] <= th]
        # contiguous from T=1
        best = 1
        for T in TS:
            if T in rt and rt[T]["x_vs_best_T1"] <= th: best = T
            else: break
        row[th] = best
    tstar[shape] = row
    P(f"| {shape} | {row[1.05]} | {row[1.10]} | {row[1.15]} | {row[1.25]} | " + " | ".join(f"{rt[T]['x_vs_best_T1']:.3f}" for T in (2, 4, 8, 16)) + " |")
P("")

# whole step
def step_ms(T, F=0.0):
    tot = 0.0
    for s in SHAPES:
        tot += COUNT[s] * ratio_tab[s][T]["ms"]
    return tot + F

P("### Whole-step GEMV time (sum of measured per-shape kernels x layer counts; 64 gate|up, 64 down, 48 qkv|z + 16 attn qkv scaled 14336/16384, 64 o_proj, 1 lm_head); excludes non-GEMV work\n")
P("| T | GEMV sum ms | x vs T=1 |")
P("|---|---|---|")
s1 = step_ms(1)
for T in TS:
    P(f"| {T} | {step_ms(T):.3f} | {step_ms(T) / s1:.3f} |")
P("")

# ------------------------------------------------------------------ MTP model
bf = ratio_tab["lmhead"][1]
GBS_BF16 = bf["gbs"]
OV_DRAFT = 0.10  # ms fixed overhead per draft step (about 10 dependent phases x ~6-10 us, E1)
d_restr = (0.85 + 32768 * 5120 * 2 / 1e9) / GBS_BF16 * 1e3 + OV_DRAFT
d_full = (0.85 + 248320 * 5120 * 2 / 1e9) / GBS_BF16 * 1e3 + OV_DRAFT


def speed(k, p, d, F, delta):
    E = sum(p ** i for i in range(k + 1))
    tv = step_ms(k + 1, F) + delta * k  # delta: extra DeltaNet snapshot/rollback ms per extra verify token
    base = step_ms(1, F)
    return E * base / (k * d + tv), E, tv, base


mtp = {"bf16_gbs_used": GBS_BF16, "draft_ms_restricted_32k": d_restr, "draft_ms_full_lmhead": d_full, "step_ms": {}, "F_default_ms": 1.5,
       "delta_default_ms": 0.17, "tables": {}}
for T in TS:
    mtp["step_ms"][T] = step_ms(T, 1.5)
P(f"### MTP break-even model (draft step = MTP layer 0.85 GB BF16 + head, at the measured BF16 GB/s {GBS_BF16:.0f}: restricted 32k head 0.336 GB -> {d_restr:.2f} ms/step, full lm_head 2.54 GB -> {d_full:.2f} ms/step; chain acceptance p^i, tokens/iter = sum_(i<=k) p^i; verify step = measured GEMV(T=k+1) + F non-GEMV + delta*k DeltaNet snapshot cost)\n")
for F, delta in ((1.5, 0.17), (1.5, 0.0), (1.5, 0.34), (0.5, 0.17), (3.0, 0.17)):
    for name, d in (("restricted 32k head", d_restr), ("full lm_head", d_full)):
        P(f"**F={F} ms, delta={delta} ms/extra token, draft = {name}** (plain step {step_ms(1, F):.2f} ms = {1e3 / step_ms(1, F):.1f} tok/s)\n")
        P("| k | verify T | verify ms | x plain step | " + " | ".join(f"p1={p}" for p in (0.6, 0.7, 0.8, 0.9)) + " |")
        P("|---|---|---|---|" + "---|" * 4)
        tab = {}
        for k in (1, 2, 3, 4, 5):
            row = []
            for p in (0.6, 0.7, 0.8, 0.9):
                sp, E, tv, base = speed(k, p, d, F, delta)
                row.append(sp)
            tab[k] = row
            tv = step_ms(k + 1, F) + delta * k
            P(f"| {k} | {k + 1} | {tv:.2f} | {tv / step_ms(1, F):.3f} | " + " | ".join(f"{x:.2f}x" for x in row) + " |")
        mtp["tables"][f"F{F}_delta{delta}_{name}"] = tab
        P("")
# break-even p per k (speedup = 1), main setting
P("Break-even p1 (speed-up = 1.0) at F=1.5, delta=0.17:\n")
P("| draft | k=1 | k=2 | k=3 |")
P("|---|---|---|---|")
for name, d in (("restricted 32k", d_restr), ("full lm_head", d_full)):
    cells = []
    for k in (1, 2, 3):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if speed(k, mid, d, 1.5, 0.17)[0] >= 1: hi = mid
            else: lo = mid
        cells.append(f"{hi:.2f}")
    P(f"| {name} | " + " | ".join(cells) + " |")
P("")
# ideal (flat verify) upper bound
P("Upper bound with a perfectly flat verify (verify(T)=plain step, delta=0, F=1.5, restricted draft), p1=0.7: " +
  ", ".join(f"k={k}: {sum(0.7 ** i for i in range(k + 1)) * step_ms(1, 1.5) / (k * d_restr + step_ms(1, 1.5)):.2f}x" for k in (1, 2, 3)) + "\n")

# ------------------------------------------------------------------ Q2 tables
P("### Q2 sustained windows (3 s after 2 s warm, interleaved 2 rounds; T=1; best config per kernel; nvidia-smi 10 Hz inside the window; * = ablated, results invalid)\n")
P("| shape | kernel | cfg | GB/s (r0 / r1) | ms | SM MHz median | power W median | SW-cap frac | mem MHz | samples |")
P("|---|---|---|---|---|---|---|---|---|---|")
q2by = defaultdict(dict)
for r in q2:
    q2by[(r["shape"], f"{r['kind']}/{r['mode']}")][r["round"]] = r
best_gemv = {}
for shape in SHAPES:
    rows = [(k, v) for (s, k), v in q2by.items() if s == shape]
    order = ["read/read", "cc/ablload", "cc/ablfma", "cc/naive", "cc/e1", "cc/bf16", "mma/abl", "mma/sub", "mma/ones", "mma/bf16"]
    rows.sort(key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    for k, v in rows:
        c = [clk2.get(f"{shape}/{k}/r{i}") for i in (0, 1)]
        c = [x for x in c if x]
        sm = statistics.median(x["sm_mhz_median"] for x in c) if c else float("nan")
        pw = statistics.median(x["power_w_median"] for x in c) if c else float("nan")
        cap = statistics.median(x["swcap_frac"] for x in c) if c else float("nan")
        mem = c[0]["mem_mhz_median"] if c else float("nan")
        ns = sum(x["samples"] for x in c) if c else 0
        star = "*" if not v[0]["valid"] and k != "read/read" else ""
        P(f"| {shape} | {k}{star} | {v[0]['cfg']} | {v[0]['gbs']:.0f} / {v[1]['gbs']:.0f} | {v[0]['ms']:.4f} | {sm:.0f} | {pw:.0f} | {cap:.2f} | {mem:.0f} | {ns} |")
        if v[0]["valid"]:
            g = max(v[0]["gbs"], v[1]["gbs"])
            if shape not in best_gemv or g > best_gemv[shape][0]:
                best_gemv[shape] = (g, k, v[0]["cfg"], sm, pw)
P("")
P("### Best exact (valid) GEMV per shape at T=1 under sustained load\n")
P("| shape | best kernel | cfg | GB/s | SM MHz | power W | read-only GB/s same shape | % of read-only |")
P("|---|---|---|---|---|---|---|---|")
for shape in SHAPES:
    g = best_gemv[shape]
    rd = max(q2by[(shape, "read/read")][i]["gbs"] for i in (0, 1))
    P(f"| {shape} | {g[1]} | {g[2]} | {g[0]:.0f} | {g[3]:.0f} | {g[4]:.0f} | {rd:.0f} | {100 * g[0] / rd:.1f} |")
P("")

# clocks along the T sweep (q1 final windows)
P("### Q1 clocks along the T sweep (nvidia-smi 10 Hz over each 0.6 s final window; best exact mma/sub or ones per T)\n")
P("| shape | T | kernel | GB/s | SM MHz | power W | samples |")
P("|---|---|---|---|---|---|---|")
for shape in SHAPES:
    for T in (1, 4, 8, 16):
        bv = best_valid(shape, T)
        if not bv:
            continue
        c = clk1.get(f"{shape}/{bv[1]}|{T}")
        P(f"| {shape} | {T} | {bv[1]} {bv[2]['cfg']} | {BYTES[shape] / (bv[0] * 1e-3) / 1e9:.0f} | " +
          (f"{c['sm_mhz_median']:.0f} | {c['power_w_median']:.0f} | {c['samples']} |" if c else "- | - | 0 |"))
P("")

# ablation: where does T growth come from (gateup)
P("### Ablation of T growth: ms ratio to T=1, mma modes (sub = exact bf16x2 subtract, ones = lop3 only + table correction, abl = no dequant)\n")
P("| shape | mode | " + " | ".join(f"T={T}" for T in TS) + " |")
P("|---|---|" + "---|" * len(TS))
for shape in SHAPES:
    for mode in ("mma/sub", "mma/ones", "mma/abl", "mma/bf16"):
        d = fin.get((shape, mode))
        if d and 1 in d:
            P(f"| {shape} | {mode} | " + " | ".join(f"{d[T]['ms'] / d[1]['ms']:.3f}" if T in d else "-" for T in TS) + " |")
P("")

# ------------------------------------------------------------------ ncu (what limits growth)
if (RES / "ncu.json").exists():
    ncu = json.loads((RES / "ncu.json").read_text())
    cols = [("gpu__time_duration.sum", "us", 1e-3), ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "DRAM %peak", 1),
            ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "L1 %peak", 1), ("l1tex__t_sector_hit_rate.pct", "L1 hit %", 1),
            ("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "HMMA pipe %", 1),
            ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue %", 1), ("smsp__inst_executed.sum", "warp-inst M", 1e-6),
            ("sm__inst_executed_pipe_lsu.sum", "LSU inst M", 1e-6), ("sm__cycles_elapsed.avg.per_second", "SM GHz", 1e-9)]
    P("### ncu counters (natural clocks, one launch each; `run.py --stages ncu`; abl/ablb results invalid, timing only)\n")
    P("| shape/T/kernel/cfg | " + " | ".join(c[1] for c in cols) + " |")
    P("|---|" + "---|" * len(cols))
    for k, v in ncu.items():
        P(f"| {k} | " + " | ".join(f"{v[m] * sc:.2f}" if m in v else "-" for m, _, sc in cols) + " |")
    P("")

(RES / "tables.md").write_text("\n".join(out))
(RES / "mtp.json").write_text(json.dumps({"mtp": mtp, "tstar": {s: {str(k): v for k, v in d.items()} for s, d in tstar.items()},
                                          "ratio_tab": {s: {str(T): v for T, v in d.items()} for s, d in ratio_tab.items()},
                                          "best_gemv_T1": {s: {"gbs": v[0], "kernel": v[1], "cfg": v[2], "sm_mhz": v[3], "power_w": v[4]} for s, v in best_gemv.items()}},
                                         indent=1))
print("\n".join(out)[:200])
