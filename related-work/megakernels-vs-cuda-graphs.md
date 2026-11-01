---
title: megakernels-vs-cuda-graphs
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/msaroufim/megakernels-vs-cuda-graphs
path: vendor/megakernels-vs-cuda-graphs
commit: 1fe5dad
branch: main
commit_date: 2026-10-23
---

# megakernels-vs-cuda-graphs

One-liner: two head-to-head studies (Llama-3.2-1B on H100/H200 against the *unmodified* HazyResearch ThunderKittens megakernel; DSpark DeepSeek-V4 proposal kernel on GB300 against an SGLang graph) showing a hand-fused multi-kernel CUDA-graph + PDL decode within 0.7-4.6% of a megakernel (Hazy faster on Llama, graph faster on DSpark). Licences: Hazy sources MIT (`llama/megakernel/ORIGIN.md:3-11`); dspark subtree keeps DeepSpec licence (`README.md:50-53`).

Pointer legend: root-relative at `1fe5dad`; `[INF]` = my arithmetic/inference. Read only; no GPU run (H100/H200 + NGC image required, `llama/README.md:6-16`); all timings are the authors'.

## Why I cloned it

- How exactly is the graph+PDL baseline built, what is the 0.7-4.6% gap measured on, is the comparison fair, what "numerical criteria" are used, and which tricks matter (fused projections, packed weights, split FP32 reductions, PDL placement)? Prior for our single-3090 W4 decode plan.

## Orientation

- `README.md:5-24` headline table; `llama/README.md` run recipe; `llama/docs/RESULTS.md` (results+method), `llama/docs/PROTOCOL.md` (frozen protocol), `llama/PROTOCOL.md` (H200), `llama/docs/H100_PROTOCOL.md`.
- Graph baseline = **3 files**: `llama/kernels/candidate.py` (179 lines, builds the step), `llama/kernels/projection.cu` (205 lines, all GEMV/fused kernels), `llama/kernels/attention.py` (45 lines, Triton). Contract in `llama/AGENTS.md:16-28`.
- Harness: `llama/benchmarks/matched_final.py` (timing+checks), `llama/reproduce.py`, `llama/tools/stage_runtime.py`. DSpark: `dspark/README.md`, `dspark/deepspec/megakernel/csrc/dspark_v4_kernel.cu` (7,665 lines, persistent scheduler).

## Mechanisms

### 1. Graph + PDL baseline construction
**What:** a step = 98 ordinary launches + glue (README.md:22-24 says "100 kernels"): per layer `qkv`(mode 0), Triton `attention`, `o`(1), `upgate`(2), `down_split` = 2 kernels, i.e. 6 x 16 + `head_norm` + `lmhead`(5) (`llama/kernels/candidate.py:126-162`; docstring says 81 "named calls" - counts down_split as one). Launches via `cudaLaunchKernelEx` with `cudaLaunchAttributeProgrammaticStreamSerialization` (`llama/kernels/projection.cu:101-103,150-152,192-199`), then **stream-captured with `torch.cuda.CUDAGraph`**: `llama/benchmarks/matched_final.py:215-224`. One graph covers **all 127 decode steps** including embedding + 2-stage argmax + next-token embed (`:203-211`, `llama/docs/PROTOCOL.md:17`), so graph launch overhead is amortised over 127 steps; this differs from the pdl-reconstruction repo's graph-per-position. **Verified:** read.
**PDL placement (the key trick):** every kernel calls `cudaTriggerProgrammaticLaunchCompletion()` as its *first statement* (`llama/kernels/projection.cu:28,118,164,185`; Triton `griddepcontrol.launch_dependents` at `llama/kernels/attention.py:13-14`), issues its **weight loads into registers before** `cudaGridDependencySynchronize()` (`llama/kernels/projection.cu:30-40`, wait at :40; `asm volatile("" ::: "memory")` with register operands at :35-39 stops the compiler sinking the loads below the wait), and reads activations only after the wait. Triton attention loads past K/V before the wait and current-slot K/V/Q after it (`llama/kernels/attention.py:17-24`). Safety is transitive: every kernel waits, so N+2 cannot run past N (`down_reduce_kernel` waits immediately, `llama/kernels/projection.cu:185-186`). CTAs are small (128 threads, ~8 KB of register-resident weights each), so successor CTAs can co-reside with the predecessor's tail - unlike TK's 208 KB-smem CTAs ([pdl-megakernel-reconstruction](pdl-megakernel-reconstruction.md) mech. 5) [INF].

### 2. Fused projections (`llama/kernels/projection.cu:21-98`, template `MODE`)
- mode 0 = RMSNorm + QKV GEMV + RoPE(FP32 cos/sin) + KV-cache append in one kernel (`:44-57,85-97`); 1 = O-proj + residual add (`:81`); 2 = RMSNorm + up **and** gate GEMV + SiLU*up (`:44,69,80`, w2 = second weight); 5 = LM head writes FP32 logits (`:82`); mode 3/4/6 exist but the candidate does not use them (`llama/kernels/candidate.py:150-161`).
- RMSNorm is *recomputed per CTA* from the activation vector (sum of squares over K=2048 reading `x` per CTA, `:45-56`) instead of a separate norm kernel; saves ~32 launches/step [INF: 16 layers x 2 norms collapse into modes 0/2; final norm stays a kernel, `head_norm`].
### 3. Packed weights (`llama/kernels/candidate.py:38-45,79-87`)
`pack(weight, rows=2)`: rows interleaved at 16 B granularity -> `[output_block, k_vector, row_in_block, 8]` so a CTA's two rows are contiguous 32 B per K-vector (`llama/kernels/projection.cu:19,32`); each 16 B `uint4` load = 8 BF16. Lossless, done once in `prepare`, timed outside the loop (`llama/docs/RESULTS.md:19-22`).
### 4. Split FP32 reductions
Down projection (K=8192): grid `(N/2, 4)`, each CTA a 2048-wide K chunk, FP32 partials to scratch `[4][2048]`, then a tiny reduce kernel that adds the residual in FP32 (`llama/kernels/projection.cu:159-204`; scratch alloc `llama/kernels/candidate.py:112`). Deterministic (no atomics), unlike Hazy's atomic reductions. Cost: one extra kernel/layer. All other GEMVs: 64-lane x 2-row CTAs, warp-shuffle + 4-way smem reduce (`:52-56,72-78`). FP32 accumulation after exact BF16->FP32 (`mixed_dot2`, `:8-18`); fast-math on (`llama/kernels/candidate.py:66`). The Blackwell-native `fma.rn.f32.bf16` path is disabled on Hopper (`llama/tools/stage_runtime.py:47`; `llama/kernels/candidate.py:64-65`).
### 5. Attention
Triton, grid `(32 heads, 64/D tiles)`, `POS` and `T=next_pow2(pos+1)` are `constexpr`, so a new kernel per position (compile cost excluded from timing); optional eager-matching BF16 rounding of scores (`llama/kernels/attention.py:27-36`). Hard-coded 64 head dim / 512 KV width (`:17-26`). Verified read only.

### 6. What the gap is measured on
| GPU | Graph+PDL us/step | Hazy us/step | Hazy advantage | Source |
|---|---:|---:|---|---|
| H100 SXM | 979.51 | 972.78 | +0.69% [0.67,0.72] | `llama/docs/RESULTS.md:9` |
| H200 | 794.74 | 759.83 | +4.59% [4.50,4.71] | `llama/docs/RESULTS.md:10` |
| GB300 DSpark | 803.968 p50 | 828.544 p50 | **graph** faster: megakernel +3.06% latency, loses 0/96 pairs | `dspark/README.md:5-11` |
- Workload: batch 1, BF16 Llama-3.2-1B-Instruct, 32-token prompt, 127 timed dependent steps, KV capacity 16,384; prefill/packing/compile excluded; 8 prompts x 3 fresh processes + 3 native diagnostics, 5 warmups + 10 timed sequences, balanced AB/BA order, hierarchical bootstrap on paired log-latency ratios (`llama/docs/RESULTS.md:17-27`; `llama/docs/PROTOCOL.md:17-21`). Native-loop (no graph) diagnostics: H100 1010.01 vs 971.75 us (3.9%), H200 817.83 vs 766.19 (6.7%) (`llama/docs/RESULTS.md:12-15`).
- [INF] Weight bytes ~2.47 GB/step (1.236 B BF16 params) => graph reaches ~2.5 TB/s on H100 (~75% of 3.35) and ~3.1 TB/s on H200 (~65% of 4.8): the *faster* the memory, the bigger the megakernel edge (fixed per-boundary cost fraction grows). Matches the 4.59% vs 0.69%.
- DSpark: one GB300 (152 SMs), one session, one fixed input, p50 over 3 blocks x 8 pairs; megakernel 152 CTAs/100 phases/5,793 task chunks without PDL vs SGLang 176 launches with PDL; target-verify excluded (`dspark/README.md:5-22`).

### 7. Fairness caveats (the authors list most themselves)
- **Not identical arithmetic:** conventional arm keeps FP32 scratch/logits; Hazy's logits are BF16 (`llama/docs/RESULTS.md:53-54`). **Hazy fails the numerical gate** (below) so it is not an accuracy-matched win either way (`llama/docs/RESULTS.md:44-47,51-52`).
- Both arms share embedding, argmax selector and prefix KV (`llama/docs/RESULTS.md:19-22`; `llama/docs/PROTOCOL.md:27-32`); the shared fused argmax+embed was introduced after development runs showed a large generic argmax cost.
- Hazy scheduler is compiled for exactly 132 SMs; H100 PCIe etc. rejected (`llama/README.md:8-9`; `llama/docs/H100_PROTOCOL.md:6-9`). Hazy built unmodified `GPU=H100` (sm_90a); the graph arm is a purpose-built 3-file hand-tuned candidate, i.e. effort asymmetry in the graph's favour [INF].
- Known prompt suite = replication workload, not holdout; drivers differ H100 580.159 vs H200 570.133 (`llama/docs/RESULTS.md:55,60-63`); within-process samples are not independent replicates. **No per-feature ablation** of fused projections / packing / split reductions / PDL (`llama/docs/RESULTS.md:54`: "Per-feature speedups need ablations") -> *which trick matters most is not measured here*. The only measured PDL-on/off evidence is the pdl-reconstruction repo (default-dependency 1169 -> PDL 1010 us on TK kernels).
- DSpark: greedy IDs match but head/base logit rel-L2 differ 4.16%/3.44%, confidence by up to 0.039; not a scheduler-only ablation; "native parent" byte-exact outputs match the megakernel, not the graph (`dspark/README.md:19-24`).

### 8. "Numerical criteria" (`llama/docs/PROTOCOL.md:9-13`; results `llama/docs/RESULTS.md:31-52`)
Reference = HF FP32 eager execution of the BF16 checkpoint promoted to FP32; calibration = native HF BF16. Metrics per step: finite, KL(FP32||arm), canonical-token NLL difference, centred logit error, top-1 margin/loss, greedy agreement. **Accept iff** all outputs finite and the upper bound of a paired-prompt-bootstrap 95% interval of `(arm mean KL) - (native BF16 mean KL)` is <= 0, and the same for absolute canonical-token NLL; applied to canonical eager/graph *and* to each arm's actual saved outputs at post-capture/warmup/timed stages. Measured delta vs native BF16: H100 conventional KL -2.4e-4 (passes), Hazy +1.07e-4 (fails); H200 Hazy +8.3e-3 KL, +0.035 NLL (`llama/docs/RESULTS.md:39-46`). Eager vs graph vs bitwise equality kept as reproducibility *diagnostics* (`llama/docs/PROTOCOL.md:20`).

## Worth stealing
- The **PDL kernel prologue**: trigger first, register-prefetch weights, `asm volatile` anchor, wait, then read activations (`llama/kernels/projection.cu:28-40`). Not usable on sm_86 (see Gotchas) but the register-prefetch-before-dependency idea survives as explicit prefetch in a persistent kernel.
- Evaluation protocol worth copying for our gate: KL-vs-FP32 *minus* KL of the native-BF16 baseline with a one-sided bootstrap bound; arms judged on their own realised histories; verify graph replay vs eager every warmup (`llama/benchmarks/matched_final.py:221-224`, `llama/docs/PROTOCOL.md:20`). For us, replace "native BF16" with *vLLM W4A16 stock* and "FP32 ref" with a dequantised-weight FP32/torch reference.
- Packing rows into `[block][k][row][8]` so one warp-row of CTAs reads contiguous 16 B vectors; per-CTA RMSNorm recompute; deterministic split-K with FP32 partials.
- Their repo is a PR-driven benchmark harness (`llama/AGENTS.md`, CI on Modal) - the layout (`prepare/build/output_logits` interface, `llama/AGENTS.md:16-23`) is a good template for our own kernel-vs-baseline arena.

## Gotchas
- **Hopper-only.** `llama/kernels/candidate.py:62-63` raises "PDL requires Hopper or newer"; we verified `griddepcontrol` does not assemble for sm_86 with CUDA 13.0 (`ptxas: requires .target sm_90 or higher`, see pdl note Gotchas). The graph+PDL recipe cannot be built as-is on a 3090.
- Model hard-coded: K=2048/8192, 64-wide heads, 16 layers, 512 KV width (`llama/kernels/projection.cu:112,202`; `llama/kernels/attention.py:17-26`).
- BF16 weights only: no dequant ALU cost, no scales; the GEMV is bandwidth-trivial relative to our W4 case. The 1B model fits fast memory behaviour (L2 50 MB) differently from a 27B model on a 6 MB L2 [INF].
- "100 kernels" vs "81 named calls" vs "98 launches" - the numbers differ by what is counted; do not quote one without the other.

## Transferable to our 3090/W4/MTP megakernel
- **Prior for the thesis:** at 1B/BF16 on 132-SM Hopper a good graph+PDL baseline is within 0.7% (H100) to 4.6% (H200) of a megakernel, and in DSpark it won by 3%. But that prior depends on PDL, which sm_86 lacks; our relevant graph baseline is *plain graph without PDL*, whose boundary cost is the 2.4 us/edge default-dependency figure [INF, from pdl-reconstruction].
- Fusion list to copy into any design (graph or persistent): norm into the next GEMV prologue; QKV/gate-up stacked GEMV; RoPE+KV-append in the QKV epilogue; residual-add in the O/down epilogue; SiLU*up in the gate/up epilogue; FP32 logits + 2-stage argmax; deterministic split-K when K is large (our down proj K=17408).
- Numerical gate: adopt the "KL minus baseline-KL upper bound <= 0" rule, with stock vLLM W4A16 as the calibration baseline, plus finite-output and eager-vs-graph replay checks.
- Expect the megakernel premium to be larger on faster-memory/smaller-boundary-cost regimes and smaller on our 936 GB/s part where the 17 ms step swamps ~1-2 ms of boundary loss [INF].

## Dead ends / lessons
- None reported by the authors as failures, but: no ablation => cannot attribute the 0.7-4.6% to any individual trick; megakernel "wins" are only +0.69% where the arms were numerically different; both arms must pass the gate for a "speed win" and neither did (`llama/docs/RESULTS.md:53-54`).
- DSpark persistent scheduler (7.7 kLOC, 152 CTAs, atomics + `__nanosleep(32)` pollers, `dspark/deepspec/megakernel/csrc/dspark_v4_kernel.cu:995,1033-1037`) *lost* by 3% to graph+PDL: complexity does not buy speed by itself.

## Verdict
- Well-designed, pre-registered, honest benchmark: graph+PDL (hand-fused, register-prefetch before `griddepcontrol.wait`) lands within 0.7-4.6% of megakernels on Llama-1B H100/H200 and slightly ahead on DSpark GB300; with caveats (different arithmetic, 132/152-SM-specific, no ablation).
- Its recipe is Hopper-only; on sm_86 it degrades to plain graph, so it neither supports nor refutes a 3090 megakernel - it only tells us the benefit must come from explicit overlap/fusion, not from persistence per se.
- Steal the fusion list, packing layout, prologue idea, and the KL-delta numerical gate; do not steal the headline.

---

Pinned at `1fe5dad` (main, 2026-10-23) · 192 files · 6,2M
Update with: `git -C vendor/megakernels-vs-cuda-graphs pull` then refresh `commit:` above.
