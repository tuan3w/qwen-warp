---
title: w4-multirow-verify-and-power
type: experiment
tags: []
status: done
created: 2026-10-30
author: agent
model: claude-sonnet-5-5
question: "On one RTX 3090, with clean-room tensor-core (mma.sync m16n8k16 bf16) W4A16 g128 kernels that dequantise int4->bf16 once in registers and reuse it across T activation rows, how does the time of the real big matvec shapes (gate|up 5120->34816, down 17408->5120, qkv|z 5120->16384, o_proj 6144->5120, BF16 lm_head 5120->248320) grow with T in {1..16} with an L2-cold weight stream; what is the largest T within 1.05x/1.15x of T=1; is the mma path at T=1 as fast as a CUDA-core GEMV; what limits growth; what MTP speed-up does that imply (k=1..3, p1 0.6-0.9, incl. BF16 MTP-layer + restricted-vocab draft cost); and is the GEMV at the software power cap (SM 1455-1545 MHz vs 1710 read-only) held below the 891 GB/s read-only peak by dequant ALU/power, i.e. does a cheaper dequant raise sustained GB/s?"
prediction: "Pre-registered 2026-10-30 before any code ran. (Q1) The mma path with A=weights (M=16 rows), B=activations (n8 = up to 8 tokens) is DRAM-bound for T<=8 because one n8 tile serves T=1..8 with identical mma/dequant work: time(T)/time(1) <= 1.03 for T=1..8 on every shape (so T*(1.05x)=8 and T*(1.10x)=8, bet range 5-8); for T=12 and 16 (two n8 tiles, tensor pipe ~85-90% busy at 1.5 GHz: 1 mma per 8 cycles/SM, one per 16x16 weights needs ~1 per 9 cycles) time grows to 1.10-1.18x at T=12 and 1.20-1.35x at T=16 (T*(1.15x)=12, range 8-16); the lm_head BF16 shape (no dequant, 2x bytes per weight so half the mma rate) stays flat to T=16 (<=1.08x). mma at T=1 is within +-5% of the CUDA-core GEMV (equal or faster) on the large shapes. Growth with T is limited by tensor-pipe issue and clock/power, not by dequant ALU or registers. MTP break-even with chain acceptance p^i, draft step = MTP layer 0.85 GB + 32k head 0.33 GB (~1.4 ms) and verify step cost = measured gemv(T) sum + the E1 non-GEMV overhead: speed-up over plain decode at p1=0.7 is k=1: 1.50x, k=2: 1.72x, k=3: 1.80x (each +-0.15); k=3 stops paying below p1~0.5; with the full 2.54 GB lm_head on the draft (~3.9 ms/step) the k=1..3 speed-ups at p1=0.7 fall to 1.30/1.31/1.27. (Q2) Ablating the dequant ALU (loads only) lifts SM clock to ~1700 MHz and reaches >=97% of the 891 GB/s read-only peak on gate|up; E1-style CUDA-core dequant sits at ~1500 MHz and 846 GB/s (gate|up); the naive shift+I2F dequant is slower (<=780 GB/s, clock ~1400); the cheaper lop3/bf16x2 mma dequant reaches a clock >=1600 MHz but gains only +0..3% GB/s over E1-style (best GEMV gate|up ~855 GB/s, qkv|z ~830, down ~830, o_proj ~770, lm_head ~885 GB/s), i.e. power is a symptom (the card sits at the ~325 W SW cap whatever we do, throttle mask 0x4) and bytes in flight/CTA tails are the real limit, not ALU per byte. Kill/shape criteria: if T*(1.10x) < 6 on gate|up the MTP-verify plan needs a different kernel; if cheaper dequant gains >5% GB/s, ALU/power is the lever."
answer: "mma W4 verify is NOT flat to T=8: time x vs T=1 at T=4/8/16 = gate|up 1.02/1.12/1.45, down 1.04/1.12/1.46, qkv|z 1.05/1.15/1.46, o_proj 1.01/1.05/1.30 (T*(1.05x)=5/4/4/8, T*(1.15x)=8; lm_head 1.014x at T=16; T=4 whole-step GEMV 1.033x), caused by per-warp activation (B) re-fetch through L1 (HMMA pipe only 16-31% busy; B-once ablation flat 1.006x), not tensor pipe or dequant ALU; mma at T=1 is 4-7% faster than the CUDA-core GEMV on W4 shapes (within 2% of E1 GB/s) but 5% slower on BF16 lm_head; every kernel incl. read-only sits at the SW power cap (305 W / ~1720 MHz read-only vs 319-332 W / 1448-1600 MHz GEMV) and cheaper dequant gains only 0.4-1.2% GB/s (no dequant at all <=4.6%); modelled (measured GEMV + modelled state/KV/other + E0 acceptance): k=3 with int4 MTP layer + 32k head = 150/98/146/119 tok/s (code/prose/math/8K) = 1.56-1.58x stock vLLM, only 1.11-1.16x with vLLM's draft cost; plain decode 1.05x"
confidence: medium
confidence_note: "Q1/Q2 are direct measurements (3 interleaved 0.6 s repeats, Q2 3 s windows x2, ncu on 14 launches); one GPU, synthetic weights, one kernel family; Q3 mixes measured GEMV times with modelled state/KV/other work and vLLM acceptance of the unchanged full-vocab BF16 draft, so its tok/s are bounds not measurements"
answers: [docs/PROGRAM.md]
builds_on: [experiments/2026-10-30-w4-stream-persistence-vs-graph, experiments/2026-10-30-vllm-swift-int4-bs1-baseline]
spawned: [ideas/make-the-w4-mma-verify-kernel-flat-to-t-8-fetch-the-activati.md, ideas/measure-mtp-acceptance-with-a-restricted-32k-token-draft-hea.md]
reproduce: uv run experiments/2026-10-30-w4-multirow-verify-and-power/run.py
---

# w4-multirow-verify-and-power

<!-- Procedure: .agents/skills/run-experiment/SKILL.md
     Fill `question` and `prediction` in frontmatter BEFORE running. Never edit `prediction` afterwards;
     if your expectation changes, add a dated line under Prediction below.
     When done: set status: done, write `answer` (one line other agents will read) and `confidence`. -->

## Question

See frontmatter `question`.

## Prediction

See frontmatter `prediction` (pre-registered and committed before any code ran).

## Method

```bash
uv run experiments/2026-10-30-w4-multirow-verify-and-power/run.py --stages check,q1,q2   # ~22 min; --stages ncu for the counters (~4 min); python3 report.py; python3 model.py
```

Files: `kernels.cuh` (clean-room kernels), `bench.cu` (driver: `check | q1 | q2 | one`), `run.py` (build, GPU lease, 10 Hz `nvidia-smi` sampler, writes `results/*.json`), `report.py` (-> `results/tables.md`, `results/mtp.json`), `model.py` (Q3/Q4 -> `results/e2e_tables.md`, `results/e2e_model.json`).

- **Kernels.** W4 layout = compressed-tensors natural (uint32 `[N][K/8]`, nibble e = k mod 8, value `(n-8)*scale`, bf16 scale per 128 k), same as E1. `k_mma_w4`: `mma.sync.m16n8k16` bf16, A = weights (16 output rows per tile), B = activations (n8 = up to 8 tokens; T in 9..16 uses 2 n8 tiles that reuse the same dequantised A registers); dequant = one `lop3` (nibble pair -> bf16x2 `128+n`) then either an exact `fma.rn.bf16x2` (`sub`, -136) or nothing with a fp32 per-group correction `-136*sum_k(x)` (`ones`). fp32 accumulate per 128-k group, fp32 scale after. `k_cc` = E1-style CUDA-core GEMV (`e1`: lop3+fadd+ffma; `naive`: shift+I2F). `k_mma_bf16`/`cc/bf16` for the BF16 lm_head. Ablations (results INVALID, timing only): `abl` = packed words fed to mma without dequant, `ablb` = additionally activations loaded once and reused forever (isolates mma issue from B delivery), `cc/ablfma` (no dequant, FFMAs kept), `cc/ablload` (loads only), `read` (xor-reduce of the same bytes = roofline).
- **L2-cold.** Each shape has `nb` distinct weight buffers (>= 400 MB total, 2-16 buffers; lm_head 2 x 2.54 GB); one CUDA graph launches the kernel once per buffer and the reported ms is graph time / nb, so every launch streams weights that cannot be in the 6 MB L2 and includes launch gaps (same method as E1 `gemv_sweep`).
- **Q1 protocol.** T in {1,2,3,4,5,6,8,12,16} x {gate|up 34816x5120, down 5120x17408, qkv|z 16384x5120, o_proj 5120x6144, lm_head 248320x5120 BF16}. For every (shape, T) all variants (24-41 configs: warps/CTA, K-split warps, prefetch depth, resident CTAs) are timed 0.25 s, then the best config per (kernel, mode, T) is re-timed in 3 interleaved 0.6 s repeats (median reported). "Best exact" = min over valid (non-ablated) kernels at that T (CUDA-core only exists at T=1).
- **Correctness (`results/check.json`, 295 cases, T in {1,3,8,12,16}).** fp32 warp-per-row reference. mma kernels: rel-L2 <= 1.69e-3, which is exactly the bf16-output rounding floor (1.66-1.69e-3); rel-L2 against bf16(ref) <= 1.7e-4. CUDA-core kernels: <= 1.9e-7.
- **Q2.** best config per (shape, kernel) at T=1, then 2 interleaved rounds x (2 s warm + 3 s window), `nvidia-smi` at 10 Hz (clocks, power, throttle reasons) matched to the windows by wall-clock MARK lines. No clock locking (needs root; not attempted).
- **Q3** (`model.py`): measured / modelled split is stated in the tables. **Measured**: this experiment's per-shape ms at T=1..4, E1 chain 18.958 ms (`experiments/2026-10-30-w4-stream-persistence-vs-graph/README.md` Results table V1a), E0 vLLM tok/s and per-position acceptance (`experiments/2026-10-30-vllm-swift-int4-bs1-baseline/README.md` Results). **Modelled**: DeltaNet state traffic `(1+T) x 48 x 3.15 MB` at 800 GB/s (1 read + T per-row snapshot writes; state size from `docs/PROGRAM.md:29`), KV read `ctx x 32 KB` (FP8; 64 KB BF16 reported) at 800 GB/s read once for all T rows, other non-GEMV work F = 1.0 ms (a guess), draft-step costs from measured GB/s, and everything combined: `step(T) = E1_chain + (GEMV_sum(T) - GEMV_sum(1)) + state(T) + KV + F`, `tok/s = E_k / (step(k+1) + k * draft)`.

## Results

2026-10-31, RTX 3090 (sm_86, 82 SMs, driver 580.178.04, power limit 350 W), `nvcc` CUDA 13.0 `-arch=sm_86 -O3`, GPU lease held by `RunE3b`. Run time: check 1 s, Q1 815 s, Q2 451 s (`results/run_summary_q1q2.json`; `run_summary.json` is the later ncu-only run). Full tables: `results/tables.md` (Q1/Q2/ncu), `results/e2e_tables.md` (Q3/Q4), raw `results/q1.json q2.json clocks_q*.json ncu.json check.json`. An earlier kernel generation's results are kept in `results/v1_contiguous_ldcs/` (see Dead ends).

### Q1: time vs T (L2-cold, best config per T; `results/q1.json` `final`)

| shape | row | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=8 | T=12 | T=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| gateup | best mma ms | 0.1106 | 0.1128 | 0.1126 | 0.1130 | 0.1141 | 0.1182 | 0.1241 | 0.1440 | 0.1604 |
| gateup | x mma T=1 | 1.000 | 1.020 | 1.019 | 1.022 | 1.031 | 1.069 | 1.122 | 1.302 | 1.450 |
| gateup | x CUDA-core T=1 (0.1157 ms) | 0.956 | 0.975 | 0.973 | 0.977 | 0.986 | 1.021 | 1.073 | 1.244 | 1.386 |
| down | best mma ms | 0.0566 | 0.0567 | 0.0580 | 0.0587 | 0.0595 | 0.0610 | 0.0635 | 0.0746 | 0.0825 |
| down | x mma T=1 | 1.000 | 1.002 | 1.023 | 1.036 | 1.050 | 1.078 | 1.121 | 1.316 | 1.457 |
| down | x CUDA-core T=1 (0.0605 ms) | 0.936 | 0.937 | 0.958 | 0.970 | 0.983 | 1.008 | 1.049 | 1.232 | 1.364 |
| qkvz | best mma ms | 0.0538 | 0.0557 | 0.0555 | 0.0563 | 0.0566 | 0.0590 | 0.0617 | 0.0708 | 0.0783 |
| qkvz | x mma T=1 | 1.000 | 1.035 | 1.030 | 1.046 | 1.052 | 1.095 | 1.147 | 1.316 | 1.455 |
| qkvz | x CUDA-core T=1 (0.0580 ms) | 0.929 | 0.961 | 0.957 | 0.972 | 0.977 | 1.018 | 1.065 | 1.222 | 1.352 |
| oproj | best mma ms | 0.0219 | 0.0220 | 0.0221 | 0.0220 | 0.0223 | 0.0228 | 0.0230 | 0.0260 | 0.0284 |
| oproj | x mma T=1 | 1.000 | 1.005 | 1.009 | 1.006 | 1.021 | 1.041 | 1.050 | 1.187 | 1.295 |
| oproj | x CUDA-core T=1 (0.0235 ms) | 0.930 | 0.935 | 0.938 | 0.935 | 0.949 | 0.968 | 0.976 | 1.104 | 1.204 |
| lmhead | best mma ms | 3.0280 | 3.0280 | 3.0321 | 3.0418 | 3.0397 | 3.0408 | 3.0474 | 3.0679 | 3.0714 |
| lmhead | x mma T=1 | 1.000 | 1.000 | 1.001 | 1.005 | 1.004 | 1.004 | 1.006 | 1.013 | 1.014 |
| lmhead | x CUDA-core T=1 (2.8784 ms) | 1.052 | 1.052 | 1.053 | 1.057 | 1.056 | 1.056 | 1.059 | 1.066 | 1.067 |

Largest T whose best-exact-kernel time stays within x of T=1 (base = best exact T=1 kernel, which is mma on every W4 shape; `results/tables.md` "T*" table):

| shape | T*(1.05x) | T*(1.10x) | T*(1.15x) | T*(1.25x) | x@T=2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|---|---|
| gate\|up | 5 | 6 | 8 | 8 | 1.020 | 1.022 | 1.122 | 1.450 |
| down | 4 | 6 | 8 | 8 | 1.002 | 1.036 | 1.121 | 1.457 |
| qkv\|z | 4 | 6 | 8 | 8 | 1.035 | 1.046 | 1.147 | 1.455 |
| o_proj | 8 | 8 | 8 | 12 | 1.005 | 1.006 | 1.050 | 1.295 |
| lm_head (mma bf16; vs its own T=1) | 16 | 16 | 16 | 16 | 1.000 | 1.005 | 1.006 | 1.014 |

lm_head against the CUDA-core T=1 kernel (2.878 ms, 883 GB/s) the mma kernel starts 5.2% slower (3.028 ms, 840 GB/s), so vs that base T*(1.05x) = 1 and T*(1.10x) = 16.

Whole-step GEMV time (sum of measured per-shape best kernels x 64 gate|up, 64 down, 48 qkv|z + 16 attention qkv scaled by 14336/16384, 64 o_proj, 1 lm_head; no non-GEMV work): T=1 18.32 ms, T=2 18.74 (1.023x), T=3 18.80, **T=4 18.93 (1.033x)**, T=5 19.09, T=6 19.62, T=8 20.35 (1.111x), T=12 23.11 (1.261x), T=16 25.29 (1.380x).

**T=1: is mma as fast as the CUDA-core GEMV?** Yes, faster on every W4 shape (best mma vs best `cc/e1`, same harness): gate|up 0.1106 vs 0.1157 ms (-4.4%), down 0.0566 vs 0.0605 (-6.4%), qkv|z 0.0538 vs 0.0580 (-7.2%), o_proj 0.0219 vs 0.0235 (-6.8%); `cc/naive` (shift+I2F) is 11-24% slower than `cc/e1`. Not for BF16 lm_head: mma 3.028 vs cc 2.878 ms (+5.2%; ncu: DRAM 93.3% vs 96.9% of peak, 195 vs 171 M warp-instructions).

**What limits growth with T (ncu, natural clocks, `results/ncu.json`; gate|up unless noted).**

| shape/T/kernel/cfg | us | DRAM %peak | L1 %peak | L1 hit % | HMMA pipe % | issue % | warp-inst M | LSU inst M | SM GHz |
|---|---|---|---|---|---|---|---|---|---|
| gateup/T1/mma/sub/w8k8u2 | 112.86 | 90.30 | 18.93 | 47.81 | 19.09 | 40.96 | 23.88 | 0.73 | 1.65 |
| gateup/T1/cc/e1/w8r4u2 | 114.85 | 89.17 | 16.19 | 49.81 | 0.00 | 42.55 | 28.68 | 0.74 | 1.90 |
| gateup/T1/read/read/g246x512 | 109.47 | 93.41 | 16.81 | 0.00 | 0.00 | 4.52 | 2.95 | 0.36 | 1.90 |
| gateup/T8/mma/sub/w4k4u3 | 118.27 | 86.38 | 45.61 | 71.91 | 18.49 | 39.01 | 23.48 | 0.72 | 1.67 |
| gateup/T8/mma/abl/w4k4u3 | 115.39 | 88.63 | 49.44 | 72.82 | 18.62 | 26.97 | 16.12 | 0.89 | 1.65 |
| gateup/T8/mma/ablb/w8k8u2 | 111.62 | 91.31 | 16.65 | 32.48 | 16.42 | 13.49 | 9.14 | 0.39 | 1.91 |
| gateup/T16/mma/sub/w4k4u3 | 127.87 | 80.33 | 65.41 | 74.76 | 30.87 | 32.59 | 23.51 | 1.08 | 1.90 |
| gateup/T16/mma/ablb/w8k8u2 | 115.14 | 88.87 | 17.24 | 34.08 | 34.06 | 18.51 | 12.09 | 0.44 | 1.82 |
| down/T1/mma/sub/w8k4u2 | 55.94 | 90.35 | 17.59 | 45.23 | 16.90 | 35.62 | 11.71 | 0.35 | 1.89 |
| down/T8/mma/sub/w4k4u3 | 61.28 | 84.00 | 40.99 | 71.52 | 16.08 | 33.74 | 11.67 | 0.35 | 1.81 |
| down/T16/mma/sub/w4k4u3 | 77.66 | 66.77 | 61.27 | 75.83 | 28.08 | 29.58 | 11.72 | 0.53 | 1.68 |
| lmhead/T1/cc/bf16/w8r4u4 | 2882.82 | 96.88 | 17.89 | 20.16 | 0.00 | 9.74 | 170.85 | 7.70 | 1.88 |
| lmhead/T1/mma/bf16/w8k4u4 | 2997.44 | 93.26 | 16.68 | 5.81 | 4.19 | 10.31 | 195.35 | 7.60 | 1.95 |
| lmhead/T16/mma/bf16/w8k8u4 | 3053.38 | 91.69 | 30.23 | 6.34 | 8.28 | 12.02 | 230.84 | 10.47 | 1.93 |

- The tensor pipe is NOT the limit: HMMA pipe active 16-19% at T<=8 and 31% at T=16 (pre-registered 85-90%).
- Growth tracks L1 traffic: L1 throughput 19% (T=1) -> 46% (T=8) -> 65% (T=16) of peak, DRAM throughput 90.3% -> 86.4% -> 80.3% of peak (down: 90.4 -> 84.0 -> 66.8). Every warp re-reads the activation (B) tile (T x 256 B per 128 k-group, from L1/L2, 72-76% L1 hit) while it streams only 1 KB of weights per group; at T=8 that is 2 KB of B per 1 KB of W per warp-group, and NT=1 kernels only issue B loads for valid token lanes (g < T), so B traffic grows linearly T=1..8 although mma/dequant work is constant.
- Removing B re-fetch removes the growth: `mma/ablb` (B loaded once; invalid numerics) is 1.006x at T=8 and 1.063/1.072x at T=12/16 on gate|up (1.013/1.054/1.075 on down), with L1 at 17% and HMMA 16-34%. `mma/abl` (same B traffic as the real kernel, no dequant) shows the same growth as `sub` (1.088 / 1.385 / 1.547 at T=8/12/16), so the dequant ALU is not the cause either.
- Config confound: from T=6 the best config switches from w8k8u2 to w4k4u3 (smaller CTA, K-split 4, deeper prefetch); NT=2 (T=9..16) kernels spill (up to 316 B spill stores per thread, `runs/build.log`, not in git) and cannot prefetch B one group ahead. So T=12/16 numbers are for kernels that are known to be non-optimal; ncu rows for `ablb` use a different config than `sub` at the same T.

### Q2: power, clocks, dequant ALU (sustained 3 s windows, `results/q2.json`, `results/clocks_q2.json`)

gate|up, T=1 (windows 100% in SW-power-cap throttle state, every kernel incl. read-only; mem clock 9501 MHz in all):

| kernel | GB/s (2 rounds) | % of read-only | SM MHz median | power W median |
|---|---|---|---|---|
| read-only (same bytes) | 873 / 873 | 100 | 1718 | 305 |
| cc/ablload (loads only) | 863 / 863 | 98.9 | 1695 | 308 |
| cc/ablfma (no dequant, FFMA kept) | 847 / 848 | 97.0 | 1590 | 317 |
| mma/ablb (no dequant, B once) | 850 / 850 | 97.4 | 1616 | 313 |
| mma/abl (no dequant) | 847 / 845 | 96.9 | 1564 | 317 |
| **mma/ones** (lop3 only, fp32 correction) | **831 / 833** | 95.3 | 1496 | 319 |
| mma/sub (exact bf16x2 sub) | 826 / 825 | 94.6 | 1448 | 320 |
| cc/e1 (E1-style CUDA-core) | 793 / 794 | 90.9 | 1560 | 326 |
| cc/naive (shift + I2F) | 716 / 716 | 82.0 | 1590 | 332 |

Best exact kernel per shape vs read-only on the same shape (T=1): gate|up 833 vs 873 GB/s (95.4%), down 818 vs 858 (95.3%), qkv|z 796 vs 858 (92.7%), o_proj 741 vs 818 (90.7%), lm_head 883 (cc) vs 891 (99.2%). SM clock under load 1448-1605 MHz (W4 mma), 1740 (lm_head) vs 1718-1770 read-only; power 311-332 W vs 305 W read-only. Cheaper dequant (`ones` over `sub`) gains +0.4% (o_proj) to +1.2% (down) GB/s at T=1 (but is slower at T>=5: it needs per-token group-sum loads); removing dequant entirely (`abl`) gains +2.4% (down) to +4.6% (qkv|z) over `sub`, and the ablated kernels still end 2.6-3% short of read-only on gate|up. The Q1 per-window clocks (`results/clocks_q1.json`, 6-12 samples per 0.6 s window) are too short to interpret and are not used.

### Q3: end-to-end verify-step model (measured parts + modelled parts; `model.py`, `results/e2e_model.json`)

Note on acceptance: E0's per-position acceptance numbers (code 0.94/0.90/0.75, ...) are already marginal rates, `P(draft i accepted)`: E0's reported mean accepted lengths (3.59 / 2.33 / 3.49 / 2.87) equal `1 + a1 + a2 + a3` exactly, whereas `1 + cumulative products` would give 3.42 / 2.13 / 3.32 / 2.51. The model therefore uses `E_k = 1 + sum_{i<=k} a_i`.

#### Q3a measured per-shape best-exact-kernel ms (L2-cold graph) at the T that matter, and layer counts [measured]

| shape | count/step | T=1 ms | kernel | T=2 | T=3 | T=4 | T=4/T=1 |
|---|---|---|---|---|---|---|---|
| gateup | 64.00 | 0.1106 | mma/ones w8k4u2 | 0.1128 (mma/ones) | 0.1126 (mma/sub) | 0.1130 (mma/sub) | 1.022 |
| down | 64.00 | 0.0566 | mma/sub w8k4u2 | 0.0567 (mma/ones) | 0.0580 (mma/ones) | 0.0587 (mma/ones) | 1.036 |
| qkvz | 62.00 | 0.0538 | mma/ones w8k4u2 | 0.0557 (mma/ones) | 0.0555 (mma/sub) | 0.0563 (mma/ones) | 1.046 |
| oproj | 64.00 | 0.0219 | mma/ones w8k8u2 | 0.0220 (mma/ones) | 0.0221 (mma/ones) | 0.0220 (mma/sub) | 1.006 |
| lmhead | 1.00 | 2.8784 | cc/bf16 w8r4u4 | 3.0280 (mma/bf16) | 3.0321 (mma/bf16) | 3.0418 (mma/bf16) | 1.057 |
| **sum x count** | | **18.320** | | 18.742 | 18.803 | 18.930 | 1.033 |

This bench's own GEMV sum at T=1 is 18.32 ms (independent launches in a graph, no dependency/prologue); E1's measured dependent chain for the same GEMVs plus norms/prologues is 18.958 ms (V1a), 18.643 (V4). The model therefore anchors T=1 to E1's chain and adds the measured T-growth (gemv_sum(T) - gemv_sum(1)).

#### Q3b modelled traffic not in the GEMV benchmarks [modelled]

| T rows | DeltaNet state ms ((1+T) x 151 MB at 800 GB/s) | KV 2K FP8 / BF16 ms | KV 8K FP8 / BF16 ms |
|---|---|---|---|
| 1 | 0.378 | 0.084 / 0.168 | 0.336 / 0.671 |
| 2 | 0.567 | 0.084 / 0.168 | 0.336 / 0.671 |
| 3 | 0.756 | 0.084 / 0.168 | 0.336 / 0.671 |
| 4 | 0.945 | 0.084 / 0.168 | 0.336 / 0.671 |

Other non-GEMV work F = 1.0 ms per step (guess, unmeasured), draft-step fixed overhead 0.1 ms. Draft step (a) vLLM-like = BF16 MTP layer 0.85 GB + full lm_head 2.54 GB at 883 GB/s (measured T=1 lm_head) = **3.94 ms**; (b) int4 MTP layer 0.22 GB at 813 GB/s (byte-weighted measured W4 T=1) + 32k-token BF16 head 0.335 GB at 883 GB/s = **0.75 ms**.

#### Q3c modelled step times (ms), E1-anchored, KV FP8 [modelled from measured parts]

| ctx | plain T=1 | verify T=2 | verify T=3 | verify T=4 | + draft (a) k=1 / 2 / 3 | + draft (b) k=1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 2K | 20.42 | 21.03 | 21.28 | 21.60 | 24.97 / 29.16 / 33.42 | 21.78 / 22.78 / 23.85 |
| 8K | 20.67 | 21.28 | 21.53 | 21.85 | 25.22 / 29.41 / 33.67 | 22.03 / 23.03 / 24.10 |

Model plain step at 2K = 20.42 ms (49.0 tok/s) vs stock vLLM measured 21.4 ms (46.8 tok/s).

#### Q3d predicted tok/s and ratio to stock vLLM (measured E0) per text type [modelled; acceptance = E0 vLLM per-position, full-vocab, BF16 MTP layer: UNMEASURED for restricted head / int4 MTP layer]

Accepted length E_k = 1 + sum_{i<=k} a_i (a_i = E0 per-position marginal acceptance).  Draft options: (a) as vLLM today, (b) int4 MTP layer + 32k head.  ratio = model tok/s / stock vLLM tok/s at the same k (plain: 46.8).

| text | k | E_k | vLLM measured tok/s | model (a) tok/s | ratio (a) | model (b) tok/s | ratio (b) |
|---|---|---|---|---|---|---|---|
| code | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| code | 1 | 1.94 | 70.0 | 77.7 | 1.11x | 89.1 | 1.27x |
| code | 2 | 2.84 | 86.8 | 97.4 | 1.12x | 124.7 | 1.44x |
| code | 3 | 3.59 | 96.4 | 107.4 | 1.11x | 150.5 | 1.56x |
| **code best k** | a: k=3, b: k=3 | | best stock 96.4 | **107.4** | **1.11x** | **150.5** | **1.56x** |
| prose | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| prose | 1 | 1.69 | 62.6 | 67.7 | 1.08x | 77.6 | 1.24x |
| prose | 2 | 2.13 | 62.6 | 73.0 | 1.17x | 93.5 | 1.49x |
| prose | 3 | 2.34 | 63.0 | 70.0 | 1.11x | 98.1 | 1.56x |
| **prose best k** | a: k=2, b: k=3 | | best stock 63.0 | **73.0** | **1.16x** | **98.1** | **1.56x** |
| math | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| math | 1 | 1.96 | 70.6 | 78.5 | 1.11x | 90.0 | 1.27x |
| math | 2 | 2.79 | 86.2 | 95.7 | 1.11x | 122.5 | 1.42x |
| math | 3 | 3.49 | 93.5 | 104.4 | 1.12x | 146.3 | 1.57x |
| **math best k** | a: k=3, b: k=3 | | best stock 93.5 | **104.4** | **1.12x** | **146.3** | **1.57x** |
| 8K | 0 (plain) | 1.00 | 45.9 | 48.4 | 1.05x | 48.4 | 1.05x |
| 8K | 1 | 1.78 | 64.4 | 70.6 | 1.10x | 80.8 | 1.25x |
| 8K | 2 | 2.37 | 73.8 | 80.6 | 1.09x | 102.9 | 1.39x |
| 8K | 3 | 2.87 | 75.5 | 85.2 | 1.13x | 119.1 | 1.58x |
| **8K best k** | a: k=3, b: k=3 | | best stock 75.5 | **85.2** | **1.13x** | **119.1** | **1.58x** |

#### Q3e sensitivity of the k=3 code ratio (b) to the modelled inputs [modelled]

| variation | plain tok/s (2K) | k=3 code (b) tok/s | ratio vs vLLM 96.4 |
|---|---|---|---|
| default (F=1.0 ms, state 800 GB/s) | 49.0 | 150.5 | 1.56x |
| F=0.5 ms | 50.2 | 153.8 | 1.59x |
| F=2.0 ms | 46.7 | 144.5 | 1.50x |
| F=3.0 ms | 44.6 | 138.9 | 1.44x |
| base = this bench's own GEMV sum (no dependency gaps) | 50.6 | 154.7 | 1.60x |
| verify GEMV flat in T (upper bound) | 49.0 | 154.5 | 1.60x |
| KV BF16 | 48.8 | 150.0 | 1.56x |

The earlier break-even grid (`results/tables.md`, `results/mtp.json`; chain acceptance p^i, MTP layer kept BF16 + 32k head = 1.44 ms/draft, F=1.5 ms, delta=0.17 ms/token) gives speed-up over our own plain step at p1 = 0.7 of k=1/2/3 = 1.54x / 1.84x / 1.99x (restricted head) and 1.38x / 1.52x / 1.53x (full 2.54 GB head, 3.9 ms/draft); break-even p1 (speed-up 1.0, F=1.5) 0.10/0.16/0.22 (restricted) and 0.23/0.33/0.41 (full head) for k=1/2/3.

### Q4: best W4 g128 GEMV at T=1 vs E1's GEMV GB/s (measured)

#### Q4 best exact W4 g128 GEMV at T=1 vs E1's GEMV [measured: Q1 short graph windows, Q2 3 s sustained windows; E1 gemv_sweep]

| shape | E1 GB/s | Q1 best T=1 | kernel/cfg | Q1 vs E1 | Q2 best exact | kernel/cfg | Q2 vs E1 | Q2 read-only same shape |
|---|---|---|---|---|---|---|---|---|
| gateup | 846 | 831 | mma/ones w8k4u2 | 0.982 | 833 | mma/ones w8k4u2 | 0.984 | 873 |
| down | 806 | 812 | mma/sub w8k4u2 | 1.007 | 818 | mma/ones w8k4u2 | 1.014 | 858 |
| qkvz | 808 | 803 | mma/ones w8k4u2 | 0.994 | 796 | mma/ones w8k8u2 | 0.985 | 858 |
| oproj | 748 | 741 | mma/ones w8k8u2 | 0.991 | 741 | mma/ones w8k8u2 | 0.991 | 818 |
| lmhead | 888 | 883 | cc/bf16 w8r4u4 | 0.995 | 883 | cc/bf16 w8r4u4 | 0.995 | 891 |

## Conclusion

**Q1 (growth with T).** The pre-registered "flat to T=8" is falsified on the W4 shapes: measured x vs T=1 at T=8 is 1.12 (gate|up), 1.12 (down), 1.15 (qkv|z), 1.05 (o_proj) against a bet of <= 1.03; T=12 is 1.30-1.32 (bet 1.10-1.18; o_proj 1.19) and T=16 1.45-1.46 (bet 1.20-1.35; o_proj 1.30). Largest T within 1.05x/1.15x: gate|up 5/8, down 4/8, qkv|z 4/8, o_proj 8/8 (bet: 5-8 for 1.05x/1.10x - met by 3 of 4 shapes on the low side; 12 for 1.15x - measured 8). What the engine needs is T = k+1 <= 4 for k <= 3: at T=4 the five-shape GEMV sum is 1.033x of T=1 (gate|up 1.022, down 1.036, qkv|z 1.046, o_proj 1.006, lm_head 1.057 vs CUDA-core / 1.005 vs mma), i.e. **four-row verification costs +0.6 ms GEMV on an 18.3 ms plain step** - that part of the plan holds. The lm_head bet (<= 1.08x to T=16) holds: 1.014x.
**Mechanism.** The prediction's cause (tensor-pipe issue ~85-90% busy, clock/power) is wrong: HMMA pipe is 16-31% busy, and dequant ALU is not it either (`abl` grows like `sub`). Growth is activation (B-operand) delivery through L1 (L1 throughput 19% -> 46% -> 65%, DRAM efficiency 90% -> 86% -> 80%); the `ablb` ablation that loads B once is flat to <= 1.02x at T=8 and <= 1.10x at T=16 on every shape, so a kernel that stages the B tile once per CTA (smem/`ldmatrix`) instead of per warp per group should push T*(1.05x) to 8 and T*(1.15x) beyond 12 (not built: [[ideas/make-the-w4-mma-verify-kernel-flat-to-t-8-fetch-the-activati.md]]).
**Q1b (mma vs CUDA-core at T=1).** The mma path is 4-7% faster than the best CUDA-core GEMV on all four W4 shapes (prediction: within +-5%, equal or faster: met), but 5% slower on BF16 lm_head, where the CUDA-core kernel (883 GB/s, 99.2% of read-only) is the one to keep at T=1.
**Q2 (power).** Every kernel, including the read-only copy, runs at the software power cap (mask 0x4, fraction 1.00 in all windows); the cap is not what separates the GEMVs from the roofline: W4 GEMVs draw +14-27 W (319-332 vs 305 W) and run 9-16% lower SM clock (1448-1604 vs 1718-1770 MHz), and still reach 90.7-95.4% of the read-only GB/s. Clock is not the controlling variable either: mma/sub at 1448 MHz beats cc/e1 at 1560 and cc/naive at 1590 (826 vs 793 vs 716 GB/s); instruction count and load structure are (warp instructions per gate|up launch: read 2.95 M, mma 23.9 M, cc/e1 28.7 M). Pre-registered E1-style CUDA-core at 846 GB/s and mma/lop3 at ~855 GB/s: measured 794 and 833 in this harness (gate|up; both lower, ordering preserved; E1's 846 was measured with its own kernel, +6%). Cheaper dequant (`ones`) raises sustained GB/s by only 0.4-1.2% at T=1 (kill criterion "> 5% => ALU/power is the lever": not met), and the best possible case, no dequant at all, +2.4-4.6% over `sub` and still 2.6% below read-only on gate|up: ALU/dequant is worth at most ~2.5 points of the ~4.6 point gap to read-only; the other ~2.6 points are present with dequant removed (load structure, CTA tails, mma issue). Pre-registered Q2 checks: loads-only (`cc/ablload`) at ~1700 MHz and >= 97% of read-only: met (1695 MHz, 98.9%); mma dequant at clock >= 1600 MHz: not met (1448-1496 MHz for sub/ones on gate|up); naive dequant <= 780 GB/s at ~1400 MHz: GB/s met (716), clock was 1590. Not separated: power cap vs instruction issue as the cause of the clock drop (clocks cannot be locked without root).
**Q3 (modelled).** With the measured T=4 verify cost (+0.6 ms GEMV, +0.57 ms modelled DeltaNet snapshots; the 0.08-0.34 ms KV read is the same as in the plain step) a verify step is 21.6 ms vs a 20.4 ms plain step (+5.8%); draft cost then dominates: 3.94 ms per draft step as in vLLM today (BF16 MTP layer + full lm_head), 0.75 ms with an int4 MTP layer and a 32k-token head. Modelled tok/s (code / prose / math / 8K at k=3, option b): 150.5 / 98.1 / 146.3 / 119.1 = **1.56x / 1.56x / 1.57x / 1.58x of stock vLLM's best** (96.4 / 63.0 / 93.5 / 75.5), plain decode 49.0 tok/s = 1.05x of 46.8; with vLLM's own draft cost (option a) only 1.11-1.16x. The pre-registered break-even model (p1 = 0.7: k=1 1.50x, k=2 1.72x, k=3 1.80x +-0.15, vs our plain step) came out at 1.54 / 1.84 / 1.99 (k=1 and k=2 inside the band, k=3 just above it); with the full-vocab draft head the bet was 1.30/1.31/1.27 and the model gives 1.38/1.52/1.53. Sensitivity of the k=3 code ratio to F in 0.5-3 ms: 1.59x-1.44x; a perfectly flat verify would add only +3% (1.60x). So the 1.5x-class outcome is carried by (i) the cheap draft and (ii) high acceptance on code/math/8K; **both are assumptions here**: restricted-vocab/int4-MTP acceptance was not measured, F and the state/KV traffic are modelled, and the E0 acceptance is from single prompts.
**Q4.** The best exact W4 GEMV at T=1 (mma, Q1 graph windows) matches E1's plain GEMV within -1.8..+0.7% on every shape (gate|up 831 vs 846 GB/s, down 812 vs 806, qkv|z 803 vs 808, o_proj 741 vs 748, lm_head 883 vs 888; Q2 sustained: 833/818/796/741/883). So the mma kernel is a drop-in replacement at T=1 (no GEMV-speed regression) rather than a T=1 speed-up beyond E1's already-97%-of-read-only GEMV; the gain (4-7%) shown against `cc/e1` above is against the E1-style kernel as rebuilt in this harness, not against E1's own tuned numbers.

## Dead ends

- **Cancelled predecessor.** The first agent was cancelled mid-q1 (stderr rc -15 = SIGTERM at gate|up T=5 final windows); no code defect. My first relaunch also died (a backgrounded `sleep` poll killed the `nohup` child); relaunched with `setsid`. `run.err` / `q1.json.stderr` from that attempt were overwritten.
- **Kernel v1 (`results/v1_contiguous_ldcs/`, kept as evidence).** The first full run (22:47-23:05) used contiguous per-warp K slices and `__ldcs` weight loads (`E3_LDW_CS` still in `kernels.cuh`; directory name + header comment, v1 source not retained [INFERENCE on the exact differences]). mma/sub at T=1: down 0.0744 ms (759 GB/s, 23% *slower* than cc/e1 0.0605), gate|up 0.1133, qkv|z 0.0547; lm_head mma/bf16 5.36 ms (474 GB/s) in Q2; T=8 gate|up 1.155x, T=16 1.51x. The current kernels (K-split warps read adjacent 64 B pieces of a row, `ld.global.nc.L1::no_allocate`) fixed T=1 (down 0.0566, lm_head 3.03 ms) and slightly improved growth (T=8 1.12x, T=16 1.45x), but did not reach the predicted flatness.
- **Tensor-pipe-bound hypothesis (pre-registered cause) is dead**: ncu HMMA pipe 16-19% (T<=8), 31% (T=16); the real cause is B-operand L1 traffic (see Q1).
- **First ncu.json was garbage**: the CSV uses the `en_US` locale (decimal comma, `.` thousands: `86,20`, `92.037.248`), and `run.py` stripped commas, turning 89.61 % into 8961. Fixed by running ncu with `LC_ALL=C`; the table above is from the second run.
- **`ones` dequant is not a free win**: 0.4-1.2% at T=1, but at T=5..8 it is 2-4% slower than `sub` (per-token group-sum table loads add LSU traffic exactly where B delivery already limits).
- **NT=2 (T 9..16) kernels spill registers** (loading the next group's B early needs a second 32-register copy; disabled for NT=2). T=12/16 are therefore upper bounds on cost for this kernel family, not hardware limits.
- Q1 per-window nvidia-smi clocks (6-12 samples) are not usable; Q2 3 s windows carry the power evidence. Clock locking (`-lgc`) was not attempted (root), so power-vs-issue is not separated.
- Not done: Marlin comparison, real DeltaNet / attention kernels (their cost is modelled), end-to-end run with real weights, acceptance with restricted head / int4 MTP layer.

## Follow-ups

- [[ideas/make-the-w4-mma-verify-kernel-flat-to-t-8-fetch-the-activati.md]] - stage the activation tile once per CTA (smem + `ldmatrix`) or block 2 row tiles per B fetch; `ablb` bound says <= 1.006x at T=8.
- [[ideas/measure-mtp-acceptance-with-a-restricted-32k-token-draft-hea.md]] - the Q3 1.5x result needs acceptance of a restricted 32k head and an int4 MTP layer; unmeasured.
- [[ideas/ncu-check-of-w4-gemv-dequant-is-it-power-cap-alu-limited-sm.md]] - partly answered here (ncu: issue slots 41%, long-scoreboard stalls dominate, dequant worth <= ~2.5 points).
- [[ideas/cut-the-5-6-us-phase-fixed-cost-of-a-dependent-w4-gemv-chain.md]] - still the largest non-draft item once verify is cheap.
