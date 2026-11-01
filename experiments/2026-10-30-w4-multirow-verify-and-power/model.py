#!/usr/bin/env python3
"""Q3 (end-to-end verify-step model) and Q4 (best W4 GEMV vs E1) -> results/e2e_model.json + results/e2e_tables.md.  Stdlib only.

Everything here is MEASURED (m) or MODELLED (M); the tables say which.  Inputs:
  m  results/q1.json   per-shape best-exact-kernel ms at T=1..16 (L2-cold, graph of distinct buffers)           (this experiment)
  m  results/q2.json   sustained-window GB/s per kernel at T=1                                                    (this experiment)
  m  E1 chain 18.958 ms/step (V1a: 338 dependent phases incl. prologue, launch + dependency gaps, no DeltaNet/KV) (experiments/2026-10-30-w4-stream-persistence-vs-graph/results/main.json)
  m  E0 stock vLLM tok/s + per-position acceptance                                                                 (experiments/2026-10-30-vllm-swift-int4-bs1-baseline/README.md Results)
  M  DeltaNet state traffic, KV traffic, other non-GEMV work F, draft-step cost, everything combined.
"""
import json
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).parent / "results"
q1 = json.loads((RES / "q1.json").read_text())
q2 = json.loads((RES / "q2.json").read_text())["results"]

SHAPES = ["gateup", "down", "qkvz", "oproj", "lmhead"]
# layer counts per step (E1 phase table): 64 gate|up, 64 down, 64 o_proj, 48 GDN qkv|z (5120->16384), 16 attention q+gate|k|v (5120->14336), 1 lm_head
COUNT = {"gateup": 64, "down": 64, "qkvz": 48 + 16 * 14336 / 16384, "oproj": 64, "lmhead": 1}
W4B = lambda n, k: n * k * (0.5 + 1 / 64)
BYTES = {"gateup": W4B(34816, 5120), "down": W4B(5120, 17408), "qkvz": W4B(16384, 5120), "oproj": W4B(5120, 6144), "lmhead": 248320 * 5120 * 2.0}

E1_CHAIN_MS = 18.958          # m: E1 V1a graph chain, T=1, GEMVs + per-phase prologues/gaps
E1_BEST_MS = 18.643           # m: E1 V4 persistent
E1_GBS = {"gateup": 846, "qkvz": 808, "down": 806, "oproj": 748, "lmhead": 888}   # m: E1 gemv_sweep (plain kernel, graph, distinct matrices)
VLLM_PLAIN_TPS = 46.8         # m: E0 k=0 greedy (all text types 45.9-46.8)
VLLM_PLAIN_MS = 21.4
VLLM_TPS = {"code": {1: 70.0, 2: 86.8, 3: 96.4}, "prose": {1: 62.6, 2: 62.6, 3: 63.0},
            "math": {1: 70.6, 2: 86.2, 3: 93.5}, "8K": {1: 64.4, 2: 73.8, 3: 75.5}}   # m: E0 table, greedy
# m: E0 per-position acceptance, k=3 run.  These are already the MARGINAL accept rates P(draft i accepted): E0's reported mean accepted lengths
# (3.59/2.33/3.49/2.87) equal 1 + sum of the three numbers exactly, whereas 1 + cumulative products would give 3.42/2.13/3.32/2.51.
ACC = {"code": [0.94, 0.90, 0.75], "prose": [0.69, 0.44, 0.21], "math": [0.96, 0.83, 0.70], "8K": [0.78, 0.59, 0.50]}
CTX = {"code": 2048, "prose": 2048, "math": 2048, "8K": 8192}   # M: E0 "short" runs were ~100 tokens + 384 generated; 2K is a conservative stand-in

STATE_BYTES = 48 * 3.15e6     # M: DeltaNet fp32 state, 48 layers x 3.15 MB (ideas/fastest-single-3090-...md:29)
BW = 800.0                    # GB/s used for the modelled state / KV traffic (M)
KV_PER_TOK = {"fp8": 32768, "bf16": 65536}   # 16 layers x 2 x 4 x 256 x {1,2} B (ideas note line 29)
F_OTHER = 1.0                 # M: non-GEMV, non-state work (conv1d, DeltaNet recurrence maths, attention maths, RoPE, norms not in E1, argmax/sampling), ms
D_OV = 0.10                   # M: fixed overhead of one draft step (~10 dependent phases x 6-10 us, E1)

# ---------------------------------------------------------------- measured per-shape tables
fin = defaultdict(dict)
for r in q1["final"]:
    fin[(r["shape"], f"{r['kind']}/{r['mode']}")][r["T"]] = r


def best(shape, T):
    c = [(fin[(shape, k)][T]["ms"], k, fin[(shape, k)][T]) for k in ("mma/sub", "mma/ones", "cc/e1", "cc/bf16", "mma/bf16") if T in fin.get((shape, k), {})]
    return min(c)


def gemv_sum(T):
    return sum(COUNT[s] * best(s, T)[0] for s in SHAPES)


def extras(T, ctx, kv="fp8"):
    """M: DeltaNet state (read once + T snapshot writes) + KV read (once for all T rows) at BW GB/s, ms."""
    state = (1 + T) * STATE_BYTES / (BW * 1e9) * 1e3
    kv_ms = ctx * KV_PER_TOK[kv] / (BW * 1e9) * 1e3
    return state, kv_ms


def step_ms(T, ctx, kv="fp8", F=F_OTHER, base="e1"):
    st, kvm = extras(T, ctx, kv)
    # e1: anchor T=1 to the measured dependent chain, add the measured T-growth; own: this bench's GEMV sum; flat: verify GEMV(T) = GEMV(1) (bound: B-operand delivery free, cf. mma/ablb)
    g = {"e1": E1_CHAIN_MS + (gemv_sum(T) - gemv_sum(1)), "own": gemv_sum(T), "flat": E1_CHAIN_MS}[base]
    return g + st + kvm + F


# draft step cost (M, from measured GB/s): (a) vLLM-today: BF16 MTP layer 0.85 GB + full BF16 lm_head 2.54 GB; (b) int4 MTP layer 0.22 GB + 32k-token BF16 head 0.335 GB
bf_gbs = best("lmhead", 1)[2]["gbs"]
w4_gbs = sum(best(s, 1)[2]["gbs"] * BYTES[s] for s in SHAPES[:4]) / sum(BYTES[s] for s in SHAPES[:4])   # byte-weighted mean of the 4 W4 shapes
D = {"a_vllm_like": (0.85e9 + 248320 * 5120 * 2) / (bf_gbs * 1e9) * 1e3 + D_OV,
     "b_int4_layer_32k_head": 0.22e9 / (w4_gbs * 1e9) * 1e3 + 32768 * 5120 * 2 / (bf_gbs * 1e9) * 1e3 + D_OV}

out, P = [], None
P = out.append
res = {"inputs": {"E1_chain_ms": E1_CHAIN_MS, "state_bytes": STATE_BYTES, "BW_GBs": BW, "F_other_ms": F_OTHER, "draft_overhead_ms": D_OV,
                  "bf16_gbs_T1": bf_gbs, "w4_gbs_T1_weighted": w4_gbs, "draft_ms": D, "ACC": ACC, "CTX": CTX}}

P("### Q3a measured per-shape best-exact-kernel ms (L2-cold graph) at the T that matter, and layer counts [measured]\n")
P("| shape | count/step | T=1 ms | kernel | T=2 | T=3 | T=4 | T=4/T=1 |")
P("|---|---|---|---|---|---|---|---|")
for s in SHAPES:
    b1 = best(s, 1)
    P(f"| {s} | {COUNT[s]:.2f} | {b1[0]:.4f} | {b1[1]} {b1[2]['cfg']} | " + " | ".join(f"{best(s, T)[0]:.4f} ({best(s, T)[1]})" for T in (2, 3, 4)) + f" | {best(s, 4)[0] / b1[0]:.3f} |")
P(f"| **sum x count** | | **{gemv_sum(1):.3f}** | | {gemv_sum(2):.3f} | {gemv_sum(3):.3f} | {gemv_sum(4):.3f} | {gemv_sum(4) / gemv_sum(1):.3f} |")
P(f"\nThis bench's own GEMV sum at T=1 is {gemv_sum(1):.2f} ms (independent launches in a graph, no dependency/prologue); E1's measured dependent chain for the same GEMVs plus norms/prologues is {E1_CHAIN_MS} ms (V1a), "
  f"{E1_BEST_MS} (V4). The model therefore anchors T=1 to E1's chain and adds the measured T-growth (gemv_sum(T) - gemv_sum(1)).\n")

P("### Q3b modelled traffic not in the GEMV benchmarks [modelled]\n")
P("| T rows | DeltaNet state ms ((1+T) x 151 MB at 800 GB/s) | KV 2K FP8 / BF16 ms | KV 8K FP8 / BF16 ms |")
P("|---|---|---|---|")
for T in (1, 2, 3, 4):
    st = extras(T, 2048)[0]
    P(f"| {T} | {st:.3f} | {extras(T, 2048, 'fp8')[1]:.3f} / {extras(T, 2048, 'bf16')[1]:.3f} | {extras(T, 8192, 'fp8')[1]:.3f} / {extras(T, 8192, 'bf16')[1]:.3f} |")
P(f"\nOther non-GEMV work F = {F_OTHER} ms per step (guess, unmeasured), draft-step fixed overhead {D_OV} ms. Draft step (a) vLLM-like = BF16 MTP layer 0.85 GB + full lm_head 2.54 GB at "
  f"{bf_gbs:.0f} GB/s (measured T=1 lm_head) = **{D['a_vllm_like']:.2f} ms**; (b) int4 MTP layer 0.22 GB at {w4_gbs:.0f} GB/s (byte-weighted measured W4 T=1) + 32k-token BF16 head 0.335 GB at {bf_gbs:.0f} GB/s = **{D['b_int4_layer_32k_head']:.2f} ms**.\n")

P("### Q3c modelled step times (ms), E1-anchored, KV FP8 [modelled from measured parts]\n")
P("| ctx | plain T=1 | verify T=2 | verify T=3 | verify T=4 | + draft (a) k=1 / 2 / 3 | + draft (b) k=1 / 2 / 3 |")
P("|---|---|---|---|---|---|---|")
steps = {}
for name, ctx in (("2K", 2048), ("8K", 8192)):
    v = {T: step_ms(T, ctx) for T in (1, 2, 3, 4)}
    steps[name] = v
    P(f"| {name} | {v[1]:.2f} | {v[2]:.2f} | {v[3]:.2f} | {v[4]:.2f} | " + " / ".join(f"{v[k + 1] + k * D['a_vllm_like']:.2f}" for k in (1, 2, 3)) + " | "
      + " / ".join(f"{v[k + 1] + k * D['b_int4_layer_32k_head']:.2f}" for k in (1, 2, 3)) + " |")
res["step_ms"] = steps
P(f"\nModel plain step at 2K = {steps['2K'][1]:.2f} ms ({1e3 / steps['2K'][1]:.1f} tok/s) vs stock vLLM measured {VLLM_PLAIN_MS} ms ({VLLM_PLAIN_TPS} tok/s).\n")

P("### Q3d predicted tok/s and ratio to stock vLLM (measured E0) per text type [modelled; acceptance = E0 vLLM per-position, full-vocab, BF16 MTP layer: UNMEASURED for restricted head / int4 MTP layer]\n")
P("Accepted length E_k = 1 + sum_{i<=k} a_i (a_i = E0 per-position marginal acceptance).  Draft options: (a) as vLLM today, (b) int4 MTP layer + 32k head.  ratio = model tok/s / stock vLLM tok/s at the same k (plain: 46.8).\n")
P("| text | k | E_k | vLLM measured tok/s | model (a) tok/s | ratio (a) | model (b) tok/s | ratio (b) |")
P("|---|---|---|---|---|---|---|---|")
res["pred"] = {}
for txt in ACC:
    ctx = CTX[txt]
    plain = 1e3 / step_ms(1, ctx)
    vplain = 45.9 if txt == "8K" else VLLM_PLAIN_TPS
    P(f"| {txt} | 0 (plain) | 1.00 | {vplain} | {plain:.1f} | {plain / vplain:.2f}x | {plain:.1f} | {plain / vplain:.2f}x |")
    best_a = best_b = (plain, 0)
    rec = {"plain_tps": plain}
    for k in (1, 2, 3):
        E = 1 + sum(ACC[txt][:k])
        sv = step_ms(k + 1, ctx)
        ta = 1e3 * E / (sv + k * D["a_vllm_like"]); tb = 1e3 * E / (sv + k * D["b_int4_layer_32k_head"])
        vt = VLLM_TPS[txt][k]
        P(f"| {txt} | {k} | {E:.2f} | {vt} | {ta:.1f} | {ta / vt:.2f}x | {tb:.1f} | {tb / vt:.2f}x |")
        rec[f"k{k}"] = {"E": E, "verify_ms": sv, "tps_a": ta, "tps_b": tb, "vllm_tps": vt}
        best_a = max(best_a, (ta, k)); best_b = max(best_b, (tb, k))
    vbest = max([vplain] + list(VLLM_TPS[txt].values()))
    P(f"| **{txt} best k** | a: k={best_a[1]}, b: k={best_b[1]} | | best stock {vbest} | **{best_a[0]:.1f}** | **{best_a[0] / vbest:.2f}x** | **{best_b[0]:.1f}** | **{best_b[0] / vbest:.2f}x** |")
    rec["best_a"], rec["best_b"], rec["vllm_best"] = best_a, best_b, vbest
    res["pred"][txt] = rec
P("")

P("### Q3e sensitivity of the k=3 code ratio (b) to the modelled inputs [modelled]\n")
P("| variation | plain tok/s (2K) | k=3 code (b) tok/s | ratio vs vLLM 96.4 |")
P("|---|---|---|---|")
for label, kw in (("default (F=1.0 ms, state 800 GB/s)", {}), ("F=0.5 ms", {"F": 0.5}), ("F=2.0 ms", {"F": 2.0}), ("F=3.0 ms", {"F": 3.0}),
                  ("base = this bench's own GEMV sum (no dependency gaps)", {"base": "own"}), ("verify GEMV flat in T (upper bound)", {"base": "flat"}), ("KV BF16", {"kv": "bf16"})):
    E = 1 + sum(ACC["code"])
    tb = 1e3 * E / (step_ms(4, 2048, **kw) + 3 * D["b_int4_layer_32k_head"])
    P(f"| {label} | {1e3 / step_ms(1, 2048, **kw):.1f} | {tb:.1f} | {tb / 96.4:.2f}x |")
P("")

# ---------------------------------------------------------------- Q4
P("### Q4 best exact W4 g128 GEMV at T=1 vs E1's GEMV [measured: Q1 short graph windows, Q2 3 s sustained windows; E1 gemv_sweep]\n")
P("| shape | E1 GB/s | Q1 best T=1 | kernel/cfg | Q1 vs E1 | Q2 best exact | kernel/cfg | Q2 vs E1 | Q2 read-only same shape |")
P("|---|---|---|---|---|---|---|---|---|")
q2by = defaultdict(dict)
for r in q2:
    q2by[(r["shape"], f"{r['kind']}/{r['mode']}")][r["round"]] = r
res["q4"] = {}
for s in SHAPES:
    b = best(s, 1)
    ex = [(max(v[i]["gbs"] for i in v), k, v[0]["cfg"]) for (sh, k), v in q2by.items() if sh == s and k in ("mma/sub", "mma/ones", "cc/e1", "cc/naive", "cc/bf16", "mma/bf16")]
    g2 = max(ex)
    rd = max(v["gbs"] for v in q2by[(s, "read/read")].values())
    P(f"| {s} | {E1_GBS[s]} | {b[2]['gbs']:.0f} | {b[1]} {b[2]['cfg']} | {b[2]['gbs'] / E1_GBS[s]:.3f} | {g2[0]:.0f} | {g2[1]} {g2[2]} | {g2[0] / E1_GBS[s]:.3f} | {rd:.0f} |")
    res["q4"][s] = {"e1": E1_GBS[s], "q1": b[2]["gbs"], "q1_kernel": b[1], "q2": g2[0], "q2_kernel": g2[1], "read_only": rd}
P("")

(RES / "e2e_tables.md").write_text("\n".join(out))
(RES / "e2e_model.json").write_text(json.dumps(res, indent=1, default=str))
print("\n".join(out))
