---
title: w4-stream-persistence-vs-graph
type: experiment
tags: []
status: done
created: 2026-10-30
author: agent
model: claude-sonnet-5-5
question: "On the 3090, for the real per-token sequence of Swift-1.5-Qwen3.8-27b-INT4 matvecs (W4 g128 + BF16 lm_head, ~15.1 GB, ~257 dependent GEMV phases), how close to the copy-bandwidth roofline does each execution style get: per-op kernels in a CUDA graph, same with PDL, persistent cooperative kernel with grid barriers, persistent kernel with cross-barrier weight prefetch? What does one grid barrier cost?"
prediction: "Copy peak ~880 GB/s (94% of 936). Graph per-op kernels ~72% of peak (~610 GB/s, ~25 ms/step); graph+PDL ~79%; persistent no-prefetch ~76% (barrier ~3 us x 257-338 phases ~ 1 ms); persistent with cross-barrier prefetch ~88% of peak. Persistent-with-prefetch / graph+PDL = 1.07 (range 1.00-1.15). Kill criterion: ratio < 1.05 means persistence per se is not worth its complexity; the program becomes a fused kernel set in a CUDA graph."
answer: "Persistence is worth 1.5-4%, not 7%: V4 (persistent + cross-barrier prefetch) 18.64 ms/step = 809 GB/s = 90.8% of the 891 GB/s read-only peak vs best per-op CUDA graph (V1b) 19.12 ms = 789 GB/s (88.6%): V4/V1b = 1.025 (1.039 same config) < 1.05, kill criterion triggered; PDL impossible on sm_86; bare grid barrier 0.70-0.74 us, but each phase pays ~5-6.5 us (act gather ~2, tail 2-4, in-situ barrier ~1)."
confidence: medium
confidence_note: "5 interleaved repeats x 200 steps agree within 0.2% and ratios hold across a 34-config sweep (median V4/V1b 1.021, max 1.057), but one GPU, one clean-room GEMV, synthetic weights, static work split, stand-in core phases without DeltaNet/KV traffic, and the GPU ran at the software power cap (SM clock 1455-1545 MHz in GEMV variants)."
spawned: [ideas/cut-the-5-6-us-phase-fixed-cost-of-a-dependent-w4-gemv-chain.md, ideas/ncu-check-of-w4-gemv-dequant-is-it-power-cap-alu-limited-sm.md]
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: uv run experiments/2026-10-30-w4-stream-persistence-vs-graph/run.py
---

# w4-stream-persistence-vs-graph

<!-- Procedure: .agents/skills/run-experiment/SKILL.md
     Fill `question` and `prediction` in frontmatter BEFORE running. Never edit `prediction` afterwards;
     if your expectation changes, add a dated line under Prediction below.
     When done: set status: done, write `answer` (one line other agents will read) and `confidence`. -->

## Question

See frontmatter `question`: how close to the copy-bandwidth roofline does each execution style (per-op kernels in a CUDA graph, persistent cooperative kernel with grid barriers, persistent kernel with cross-barrier weight prefetch; PDL was in the plan but is sm_90+) get on the 3090 for the real 338-phase W4 matvec sequence, and what does one grid barrier cost?

## Prediction

See frontmatter `prediction` (pre-registered 2026-10-30, before any code ran). Reasoning: prior art (msaroufim/megakernels-vs-cuda-graphs: 0.7-4.6% for megakernel over graph+PDL on H100/H200; bonsai-turbo: graph 151 vs megakernel 150 tok/s) says persistence is a small win; the one thing a persistent kernel can do that PDL cannot is keep a deep weight prefetch (smem/register ring) alive across the dependency barrier, which matters more on a 6 MB-L2 GPU where PDL's early-launched CTAs cannot hold much prefetched data.

2026-10-30 (added before results; frontmatter `prediction` untouched): **V2 (graph + programmatic dependent launch) is impossible on sm_86** — `griddepcontrol.{wait,launch_dependents}` needs `.target sm_90`; ptxas rejects it under `nvcc -arch=sm_86` (exact error in `results/pdl_attempt.json`). The comparison that stands in for V2 is **V1b** = per-phase kernels in a CUDA graph that issue their weight loads (cp.async ring) *before* the activation prologue (the dependency only gates the activation vector, not the weights), i.e. the best non-persistent baseline available on this GPU. The kill criterion is therefore re-stated as **V4/V1b < 1.05 kills persistence** (and V3/V1b is reported as well). Also added: V1a (same kernel, weight loads start only after the activation prologue), V1c (classic launch: one task per warp, grid sized per phase, dynamic smem so several CTAs/SM; hardware load balancing).

## Method

Synthetic weights (random int32-packed int4 + bf16 scales, distinct allocation per matrix so L2 never helps), real shapes from `config.json`. Per GDN layer: [in_proj_qkv|z fused 5120->16384] -> [out_proj 6144->5120] -> [gate|up fused 5120->34816] -> [down 17408->5120]; per attention layer: [q+gate|k|v fused 5120->14336] -> [o_proj 6144->5120] -> gate|up -> down; then lm_head BF16 5120->248320. Each phase consumes the previous phase's output (true dependency: residual-style add + RMS-like scale between phases). Same arithmetic in every variant (dequant int4->fp32 FMA, bf16 activations); variants differ only in execution style: V0 read-only copy/reduce of the same total bytes (roofline); V1 per-phase kernels in a CUDA graph; V2 = V1 + programmatic dependent launch; V3 persistent cooperative kernel, one block per SM, grid barrier per phase; V4 = V3 + weight tiles for phase i+1 prefetched (cp.async into smem ring) before the barrier of phase i. Also: barrier microbench (empty phases). Metric: median ms/step over 200 steps after warmup, 5 interleaved repeats, GB/s = bytes/step / time; GPU clocks and power logged. Budget: 1 GPU-hour. Correctness: all variants produce the same final vector within fp32 reassociation tolerance.

```bash
projects/llm-lab/bin/lab gpu status   # lease first
```

```bash
uv run experiments/2026-10-30-w4-stream-persistence-vs-graph/run.py
```

## Results

Run 2026-10-30, RTX 3090 (sm_86, 82 SMs, driver 580.178.04), `nvcc` CUDA 13.0 `-arch=sm_86 -O3`, GPU lease taken via `lab gpu acquire` around the runs. `uv run run.py` (1084 s end to end) builds `bench.cu`/`kernels.cuh`, runs stages `pdl unit barrier check gemv grid sweep main ablate profile`, and writes `results/*.json`; `python3 report.py` renders the same files into `results/tables.md` (all tables below are copied from it). Raw nvidia-smi log: `runs/smi.csv` (ignored). Synthetic weights, 338 phases (embed + 48x5 GDN + 16x6 attention + lm_head), 257 GEMV matrices, **15.086 GB streamed per step** (W4 weights + bf16 scales + bf16 lm_head, distinct `cudaMalloc` per matrix).

**What was built (clean-room; `kernels.cuh`, `bench.cu`, `cfgs.inc`).** W4 layout = compressed-tensors natural `[N][K/8]` int32 (nibble i = k mod 8) + bf16 scales `[N][K/128]`, value `(n-8)*scale`. A warp owns a task of R rows x full K and streams 512 B row-tiles (+16 B of scales) through a private smem ring with `cp.async.cg` (L2 evict-first hint); the bf16 activation vector sits in smem in a lane-conflict-free permuted layout, pre-multiplied by powers of two so that the nibble can be turned into an exact fp32 `(n-8)*2^(P-23)` with one LOP3 + one FADD (mantissa-window trick), then one FFMA per weight, one scale FMA per 32 weights (`kernels.cuh:compute_w4`). Every phase = prologue (RMS-norm of the residual `x`, or copy of the bf16 vector `c`/`h`) -> GEMV tasks -> epilogue (store / `x += y` / SwiGLU on interleaved gate,up rows). Core phases (DeltaNet / attention stand-ins) are 6144-element elementwise kernels that add the extra barriers (5 per GDN layer, 6 per attention layer) but no state/KV traffic (the real 0.3 GB/token DeltaNet state is not streamed here). Variants (identical device code except where noted): **V0** one read-only kernel over all 513 weight/scale buffers (`__ldcs`, 4 loads in flight; the roofline) and **V0g** 257 independent read kernels in a CUDA graph; **V1a** one kernel per phase in a graph, weight loads start after the activation prologue; **V1b** same but the first D weight tiles are issued *before* the prologue (stand-in for the impossible V2); **V1c** classic launch (one task per warp, grid sized per phase, dynamic smem so several CTAs fit per SM); **V3** cooperative persistent kernel, one CTA per SM, own grid barrier (monotone counter, `red.release.gpu` + `ld.acquire.gpu` spin, `__syncthreads` around it), ring primed only after the barrier; **V4** = V3 but the ring is a flat tile stream over the *next GEMV phases* (cursor skips core phases), so the first D tiles of phase i+1 are in smem before the CTA arrives at barrier i and refill continues through the tiny core phases. Ring configs are named `w<warps/CTA>r<rows/task>d<tiles in flight per warp>`; 35 configs in `cfgs.inc` were swept (ring up to the 99 KB smem limit, `results/sweep.json`; w8r4d4 is infeasible).

**Correctness (`results/unit.json`, `results/check.json`).** Dequant/GEMV unit test against a CPU double reference on small matrices (W4 store, W4 gate|up with bf16 output, BF16, W4 with RMS prologue) for 4 configs x {per-phase kernels, V3, V4} = 12 combos, all pass: max rel-L2 error 7.6e-8 (fp32 level; gate|up bf16 output exact). Full model vs a naive one-warp-per-row fp32 reference: logits rel-L2 3.8e-3 and x rel-L2 3.1e-3 for every variant; the same naive reference with its reduction order reversed differs from itself by 3.77e-3 / 3.09e-3, i.e. the gap is the reassociation noise floor (amplified by bf16 rounding of activations over 64 layers), not a kernel difference. V1a/V1b/V3/V4 at the same config are bit-identical to each other (rel-L2 0.0), all finite, and a second back-to-back step reproduces (checks barrier-counter reuse).

**Main result (`results/main.json`; median of 5 interleaved repeats x 200 steps after 20 warmups, repeats agree within 0.2 %; "% of V0" uses V0's 890.9 GB/s as the copy peak):**

| variant (cfg) | ms/step | GB/s | % of V0 | % of 936 | ms above V0 per phase |
|---|---|---|---|---|---|
| cudaMemcpy D2D 1 GiB (bytes = read+write) | 2.543 | 844.4 | 94.8 | 90.2 | - |
| **V0** read-only sum, one kernel | 16.934 | 890.9 | 100 | 95.2 | 0 |
| V0g 257 independent read kernels in a graph | 17.359 | 869.1 | 97.6 | 92.9 | 1.65 per kernel boundary (no dependency) |
| V1a graph, per-phase kernels (w8r2d2) | 18.958 | 795.8 | 89.3 | 85.0 | 5.99 |
| V1b graph + weight prefetch before prologue (w8r2d2) | 19.116 | 789.2 | 88.6 | 84.3 | 6.45 |
| V1c classic grid-per-phase (w16r4d1) | 20.016 | 753.7 | 84.6 | 80.5 | 9.12 |
| V3 persistent + barrier (w8r2d2) | 18.922 | 797.3 | 89.5 | 85.2 | 5.88 |
| **V4** persistent + cross-barrier prefetch (w8r2d5) | **18.643** | **809.2** | **90.8** | **86.5** | 5.06 |

Same config for all styles (w8r2d5, V4's best): V1a 19.293, V1b 19.365, V3 19.168, V4 18.643 ms. V4 at w8r2d2: 18.814. Sweep-wide (`results/sweep.json`, 34 configs feasible for V1b, V3 and V4, 2 x 100 steps): V4/V1b same-config median 1.021, max 1.057 (at w4r2d15, where V4 is itself 20.5 ms, 10 % slower than the best config); V3/V1b median 1.008, max 1.040; V4/V3 median 1.015, max 1.039.

**Ratios: V4/V1b = 19.116/18.643 = 1.025 (each style at its sweep-best config); 1.039 at the identical config w8r2d5; V4/V1a = 1.017; V3/V1b = 1.010.** Pre-registered prediction was 1.07 (range 1.00-1.15) for V4/PDL; the kill criterion (< 1.05) **triggered**.

**Prediction vs measurement:** copy peak ~880 GB/s -> V0 890.9 GB/s (memcpy 844 r+w). Graph per-op ~72 % of peak (~610 GB/s, ~25 ms) -> V1b 789 GB/s, 19.1 ms (88.6 % of V0). Persistent no-prefetch ~76 % -> V3 89.5 % of V0. Persistent + prefetch ~88 % of peak -> V4 90.8 % of V0 (86.5 % of 936). Barrier ~3 us x 338 phases -> bare barrier 0.70-0.74 us (below). Our per-op kernels are far closer to the roofline than the prediction assumed; the predicted ordering V1 < V3 < V4 holds but with 1-2.5 % steps instead of 4-12 %.

**GEMV efficiency (plain kernel, `results/gemv_sweep.json`, graph of back-to-back launches over distinct matrices, launch gaps included; read-only kernel of the same shape in brackets):** gate|up 34816x5120 846 GB/s (873), qkv|z 16384x5120 808 (860), down 5120x17408 806 (860), o_proj 5120x6144 748 (816), lm_head BF16 248320x5120 888 (891) => 97 / 94 / 94 / 92 / 99.7 % of the read-only kernel on the same shape, i.e. above the 85 % target on every large shape. As a share of the whole-step peak (890.9) they are 95 / 91 / 90 / 84 / 99.7 %.

**Barrier (`results/barrier.json`, 20000 empty back-to-back barriers, median of 5; 82 CTAs):** monotone-counter spin barrier **0.70 us (256 thr) / 0.74 us (512 thr) / 0.80 us (128 thr) / 0.84 us (1024 thr)**; `cg::grid::sync()` 0.69-0.73 us (same); with `__nanosleep(32)` in the spin 1.02 us; classic sense-reversing 1.13-1.29 us; 164 CTAs x 512 thr (2/SM) 1.04 us / cg 0.83 us. So 338 barriers cost 0.25 ms (1.4 % of a step) *as a bare primitive*.

**Where the per-phase overhead actually goes (`results/profile.json`, `%globaltimer` stamps per CTA and phase, 20 steps; `results/ablation.json`).** V4 w8r2d5, mean over the 338 phases (us): phase wall 54.9 = prologue 1.95 + streaming (median CTA) 49.8 + tail (last-minus-median CTA arrival) 1.80 + barrier (last arrival -> first release, in situ) 1.02 + start skew 0.18 (these overlap slightly). V3 w8r2d2: prologue 2.81, tail 2.15, barrier 1.31. V4 w8r4d1 (shallow ring): tail 3.68 (qkv|z 6.73, gate|up 7.67, o_proj 1.35). So the per-phase cost over V0 (5-6.5 us) is dominated by activation gather + tail imbalance + the in-situ barrier wait (1.0-1.3 us vs 0.7 us bare), not by the barrier primitive. Ablating the persistent kernel (results then invalid, timing only): no barrier 18.916 -> 17.839 ms (V3), 18.629 -> 17.822 (V4); no activation prologue 18.407 (V3), 18.100 (V4); no core phases 18.877 / 18.612; all three removed 17.767 / 17.786 ms - still 0.83 ms (4.9 %) above V0 (GEMV compute + task structure). So even a free inter-phase barrier plus free dependencies bounds V4 at 1.07x over V1b; the achieved 1.025 is about a third of that headroom. Deeper rings help V4 but not V1/V3: V4/V3 at w8r2d5 = 19.168/18.643 = 1.028, and the timeline shows the prefetch shortens the post-barrier prologue (2.81 -> 1.95 us; the x loads no longer queue behind the primed weight loads) and the tail (2.15 -> 1.80 us).

**Clocks / power (`results/run_summary.json`, `runs/smi.csv`, 5 Hz, no clocks locked).** Every timed window ran at the software power cap (throttle-reason mask 0x4); temperature <= 71 C. SM clock median: V0 1710 MHz @ 306 W, memcpy 1890 MHz, V0g 1718 MHz, GEMV variants 1455-1545 MHz @ 322-325 W (V4 w8r2d5 1455 MHz, V1b w8r2d2 1545 MHz). The GEMV variants therefore run ~10-15 % lower SM clocks than the read-only baseline (the dequant/FMA work draws the power); the V4 vs V1b ratio is measured despite V4 having the *lower* clock.

## Conclusion

**On the 3090, persistence with cross-barrier prefetch buys 1.5-4 % over the best per-op graph, not the predicted 7 %: V4 18.64 ms (809 GB/s, 90.8 % of the 891 GB/s read-only peak) vs V1b 19.12 ms (789 GB/s, 88.6 %) -> V4/V1b = 1.025 (1.039 at an identical ring config); the kill criterion (< 1.05) triggers, so the program should be a fused kernel set in a CUDA graph, with persistence an optional later 2-3 % at best.** Reasons, all measured: (1) PDL does not exist on sm_86, but per-op kernels that prefetch weights before the dependent activation read already reach 89 % of the read roofline, leaving only 6.5 us per phase (2.2 ms/step) to attack; (2) the bare grid barrier is only 0.7 us, but each phase still pays ~2 us activation gather + ~2-4 us CTA-arrival tail + ~1 us in-situ barrier wait, which a persistent kernel shrinks only partly (V3 -1.0 %, V4 -2.5 % step time vs V1b); (3) a kernel boundary alone costs 1.65 us with no dependency at all (V0g vs V0: 0.43 ms over 257 boundaries), the same order as the 0.47 ms V4 gains over V1b, while the remaining 1.7 ms between V4 and V0 is dependency latency that persistence does not remove. Practical numbers for the engine: at 15.09 GB/step the best measured style gives 53.6 steps/s (V4) vs 52.3 (V1b) vs 59.1 for the read-only roofline, before KV/DeltaNet state, MTP and sampling.

Caveats: static warp-task partition (a dynamic/guided tail was not built); stand-in core phases carry no DeltaNet/KV bytes; weights are synthetic; our own GEMV rather than Marlin; the ring config of each style was chosen on the 2 x 100-step sweep and then re-measured at 5 x 200 steps (V4 w8r2d5 was best in both); `ncu` was installed but not used - the timeline comes from in-kernel `%globaltimer` stamps (which add one `__syncthreads` per phase) and ablations. Every style was swept over the same 35 configs, so V1b is as tuned as V4; a still better V1 may exist (e.g. fewer, fatter phases).

## Dead ends

- **V2 (graph + PDL) is impossible on sm_86.** `griddepcontrol.wait` / `.launch_dependents` fail in ptxas with `Modifier '.wait' requires .target sm_90 or higher` / `Instruction 'griddepcontrol' requires .target sm_90 or higher` under `nvcc -arch=sm_86` (CUDA 13.0.88); the same probe compiles for sm_90 (`results/pdl_attempt.json`). Replaced by V1b (weights prefetched before the dependent activation read; kernel boundary still drains).
- **V1b did not beat V1a.** Issuing the first weight tiles before the activation prologue made the graph kernels 0.8 % *slower* in the main run (19.116 vs 18.958 ms) and was faster in only 19 of 34 sweep configs (e.g. w16r4d1 19.55 vs 19.64, but w8r4d1 19.18 vs 19.02). [INFERENCE, from the V3 timeline where the prologue is 2.81 us vs 1.95 us in V4: the early cp.async traffic queues in front of the prologue's x/h loads.]
- **V1c (classic launch, hardware load balancing, dynamic smem so 2-3 CTAs/SM) is the slowest style**: best 20.0 ms (w16r4d1: sweep 20.049, main 20.016); sweep w12r4d1 21.90, w8r4d2 22.83, w8r4d1 23.00, w4r4d1 24.55, w8r2d1 24.68 ms [INFERENCE: every CTA repeats the activation prologue and a warp executes a single task, so no warp tail overlaps the next task; not separately measured].
- **Bigger rings are not monotonically better and warp counts that are not a multiple of 4 are bad.** In the sweep, 5/6/10/15-warp CTAs are 6-28 % slower than 4/8/12/16 for V1b (best V1b per warp count: w4 19.23, w8 19.15, w16 19.21, w12 19.29, w10 20.33, w6 20.49, w15 22.26, w5 24.52 ms) [INFERENCE: uneven warps per scheduler; not checked with ncu]; w24/w32 configs spill registers (`runs/build.log`: 12-88 B spill stores in the persistent kernels) and are slow; V1b/V3 get worse with ring depth at w8r2 while V4 improves up to d5 and is flat after (w8r2: V4 d2 18.814, d5 18.643, d6 18.724, d7 18.720; V1b d2 19.116, d5 19.365, d6 19.581, d7 19.605 ms; sweep and main runs).
- **Uniform random nibbles made the residual stream explode** (mean of (n-8) is -0.5, a common-mode term that feeds back through 64 layers: x rms 6850 at the end, a first check showed a deceptively small 1e-4 error vs the reference [INFERENCE: one late layer dominated x]). Fixed by remapping nibble 0 to 8 (zero-mean values -7..7, `k_fill_w4`); x rms is now 1.8.
- **Static imbalance is a real but partial explanation of the tail.** Shrinking the grid to 64 CTAs so that tasks divide evenly raised qkv|z with w8r4d1 from 785 to 820 GB/s and gate|up 825 -> 847, but o_proj/down (2.5 tasks per warp) fell (743 -> 684, 791 -> 733); at 41 CTAs every W4 shape fell to 610-690 GB/s (lm_head stayed 867-883; `results/gemv_grid_study.json`), i.e. fewer SMs cannot saturate DRAM. No dynamic work distribution was built.
- **Barrier variants were not better than the plain monotone counter**: `__nanosleep(32)` in the spin 1.02 us, classic sense reversal 1.13-1.29 us, cooperative-groups `grid.sync()` the same as ours (0.69-0.73 us).
- The first correctness check (before the zero-mean fix) was discarded; all timed sweeps were run after it. The grid=64/41 and V1c experiments were exploratory first and are reproduced by `run.py` stages `grid` / `sweep`.

## Follow-ups

- [[ideas/cut-the-5-6-us-phase-fixed-cost-of-a-dependent-w4-gemv-chain.md]] - the per-phase cost (activation gather ~2 us + tail 2-4 us + in-situ barrier ~1 us) is the whole remaining gap; levers are balanced tails, fewer phases, early activation broadcast.
- [[ideas/ncu-check-of-w4-gemv-dequant-is-it-power-cap-alu-limited-sm.md]] - even with every barrier/prologue removed the GEMV is 4.9 % above the read-only kernel, at 10-15 % lower SM clock under the software power cap.
- Not done here: dynamic/guided tail scheduling, Marlin/vLLM comparison, MTP verify (T=1+k rows), real DeltaNet/KV traffic in the core phases, `ncu` counters.
