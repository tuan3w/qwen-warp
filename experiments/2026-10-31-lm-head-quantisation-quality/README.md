---
title: lm-head-quantisation-quality
type: experiment
tags: []
status: done
created: 2026-10-31
author: agent
model: claude-sonnet-5-5
question: "Can Swift-1.5-Qwen3.8-27b's main-model BF16 lm_head (checkpoint lm_head.weight bf16 [248320,5120], 2.54 GB = 2.87 ms of the 20.9 ms engine step) be stored as int8 per-row (RTN or clip-searched), int6 per-row, or int4 g128 (RTN, clip-searched, activation-weighted clip-searched) without hurting output quality (>= 99.5% top-1 agreement vs the fp32-accumulate BF16 reference logits, teacher-forced log-loss delta CI includes 0 or < 0.002 nats, top-k20/top-p0.95 filtered-distribution TV < 0.005), measured on >= 4000 final-norm hidden states from the real engine over text that is not the vocab corpus, relative to a noise floor (engine bf16 kernel vs fp32 reference; T=1 vs T=4 hidden rows; bf16-rounded logits); and what would the CUDA-core int8 / smk_w4_gemv int4 head gain on the 3090 for the T=1 plain step and the T=4 verify step?"
prediction: "Frozen before any run. Decision rule (task): ADOPT iff top-1 agreement >= 99.5% AND log-loss-delta 95% CI includes 0 or point delta < 0.002 nats AND sampling TV < 0.005. Bets (pooled over all eval positions): (1) int8 per-row RTN: top-1 agreement 99.9% (>= 99.8%), full-vocab KL(ref||q) ~2e-5 nats, log-loss delta +0.0001 (|delta| < 0.0005, CI includes 0), sampling TV (top_k20+top_p0.95, T=1) 0.001 (<0.003), argmax changes on the 4 verify rows in < 0.5% of 4-row blocks: ADOPT. int8 clip-search is within 10% of RTN on every metric (no meaningful gain; int8 absmax rows have no outliers worth clipping). (2) int6 per-row: top-1 99.7%, ADOPT. (3) int4 g128 RTN (scale amax/7, q in [-8,7], bf16 scale): top-1 agreement 97.5% (range 96.0-98.8), KL ~0.012, log-loss delta +0.010 nats (CI excludes 0), TV 0.02 (range 0.01-0.04): FAIL on all three criteria. int4 g128 per-group clip search: KL 25% lower than RTN, top-1 98.2% (range 97-99.2), still FAIL (P(passes top-1 >= 99.5) = 10%, P(passes TV < 0.005) = 5%). Activation-weighted clip search (calibrated on disjoint docs) is within +-0.3 pt top-1 of plain clip search (bet 98.4%), still FAIL. Closest int4 arm to the rule = activation-weighted clip search on every metric. (4) Noise floor: engine's own bf16 head kernel vs my fp32-accumulate reference on the same hidden: top-1 agreement >= 99.97%, log-loss delta |.| < 1e-4; hidden-state T=1 vs T=4 rows through the whole engine: top-1 agreement 99.6% (range 99.0-99.9), log-loss delta CI includes 0; bf16-rounded reference logits: top-1 agreement 99.7%. The pass threshold 99.5% is therefore close to the T=1-vs-T=4 floor and int8 is indistinguishable from the floor. (5) Speed (3090, L2-cold, 886 GB/s class): int8 per-row CUDA-core GEMV (my kernel, fp32 logits out) T=1: 1.50 ms (range 1.45-1.65), T=4 (4 accumulators per thread, same read): 1.70 ms (1.5-2.0); smk_w4_gemv EPI_F32 N=248320 K=5120 (0.656 GB incl. scales) T=1: 0.88 ms (0.78-1.0), T=4: 0.95 ms (0.85-1.1). Modelled step gain vs the shipped 2.87 ms (T=1) / ~3.0 ms (T=4 mma): int8 T=1 -1.37 ms = 6.5% of the 20.9 ms step (+7.0% tok/s), T=4 verify -1.3 ms = 5.8% of 22.5 ms; int4 T=1 -2.0 ms = 9.6%, T=4 -2.05 ms = 9.1%. Net recommendation bet: adopt int8 per-row for the verify head (about +6% end-to-end); int4 g128 lm_head does not pass the rule."
answer: "No pre-registered head passes the rule (15,280 teacher-forced positions / 43 docs, engine final-norm hidden states): int8 per-row is closest (top-1 99.19% [98.90, 99.43] vs 99.5 needed, dNLL +0.00012 [-0.00009, +0.00033], TV 0.0038) and its flips are all near-ties (median margin 0.013 logits); int6 97.3%, int4 g128 RTN/clip/act-weighted 93.8/93.2/94.7% (TV 0.035-0.048): int4 alone is not usable. Post-hoc: int8 with a bf16 scale per 128 cols passes (99.503%, 1231 MB), and int8 or int4 + exact bf16 rescoring of the top-16 candidates gives 0 argmax flips in 15,280 positions (TV 0.0003 / 0.0032; int4 dNLL +0.0018). Head GEMV 2.875 ms bf16 -> int8 1.445 ms (881 GB/s, T=1; 1.546 at T=4) -> int4 0.775 ms: modelled step -6.8% (int8) / -10.0% (int4) at T=1, verify -6.4% / -9.8%; with 65 us rescoring MTP k=3 cycle x1.060 (int8) / x1.097 (int4), code 145 -> ~154 / ~159 tok/s."
confidence: medium
confidence_note: "Quality: 15k positions but only 43 documents (cluster bootstrap CIs), one model, offline logits not live sampling, 'ideal drafter' MTP proxy; int4/int8 rescoring arms and int8_g128 are post-hoc/exploratory and not integrated in the engine (g128 int8 kernel unwritten). Speed: measured microbenchmarks (L2-cold, median of 40, one GPU session), step gains are a model, not an end-to-end run."
models: [swift-1.5-int4 (Swift-1.5-Qwen3.8-27b-INT4, ~/models/swift-1.5-int4; lm_head.weight bf16 [248320,5120])]
gpu_hours: 0.15
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: "projects/llm-lab/bin/lab gpu acquire; ~/llm-lab/venv/bin/python experiments/2026-10-31-lm-head-quantisation-quality/run.py collect; (same) analyze; python3 experiments/2026-10-31-lm-head-quantisation-quality/report.py; nvcc bench_head.cu (command in its header); lab gpu release"
---

# lm-head-quantisation-quality

<!-- Procedure: .agents/skills/run-experiment/SKILL.md
     Fill `question` and `prediction` in frontmatter BEFORE running. Never edit `prediction` afterwards;
     if your expectation changes, add a dated line under Prediction below.
     When done: set status: done, write `answer` (one line other agents will read) and `confidence`. -->

## Question

Is the BF16 lm_head (2.54 GB, 2.87 ms of the 20.9 ms step) quantisable to int8 / int6 / int4 without hurting the output distribution, and what does the byte reduction buy on the 3090 at T=1 and at the T=4 verify step? Success criterion per head arm = the decision rule in the frontmatter `prediction`.

## Prediction

Frozen in the frontmatter (committed in `bcbfc1a` before any run). Summary of the bets: int8 per-row and int6 per-row PASS the rule (top-1 ~99.9% / 99.7%, log-loss delta ~0, TV ~0.001); int4 g128 RTN / clip-search / act-weighted clip-search FAIL (top-1 97.5% / 98.2% / 98.4%, TV ~0.02, log-loss +0.01 nats); the T=1-vs-T=4 hidden-state floor sits at ~99.6% top-1, close to the 99.5% threshold. Speed: int8 1.50 ms (T=1), int4 0.88 ms; modelled step gain int8 6.5%, int4 9.6%. Recommendation bet: int8 per-row verify head.

2026-10-31 (after Table 1 of the pre-registered arms was seen; not predicted): added post-hoc arms `int8_g128_rtn` (int8 with a bf16 scale per 128 columns) and "quantised head + exact rescoring of the top-M candidates" (M = 4, 8, 16) for int8_rtn / int4_g128_rtn / int4_g128_awclip, plus a select+rescore kernel timing. They are exploratory (no bet) and are labelled as such in the tables.

## Method

**Hidden states** (`run.py collect`, engine = `engine` `Engine`, ctx 2048, bf16 KV, GPU lease). 53 documents: 17 code (vendor/ files > 120 KB, window starting past the first 100 KB = beyond the per-file cap of the draft-vocab corpus of `experiments/2026-10-31-mtp-draft-head-ablation`: py 6, c 3, cu 2, rs 3, ts 3), 12 prose (notes/, sources/, papers/ markdown; not topics/), 12 math/CoT-like (experiments/*/README.md), 12 greedy chat (12 hand-written chat-template prompts in `lmq.CHAT_PROMPTS` (code, math, prose, agent/JSON, translation); plain T=1 greedy decode, up to 320 tokens, cut at EOS; the doc = prompt + generation). Natural docs are 400 tokens. Each doc is teacher-forced through the engine in T=4 head steps and the final-norm output (`dbg` tap slot 65, bf16 [5120]) of every position is stored (`runs/hidden.pt`); 8 "floor" docs (2 per category) are additionally teacher-forced in T=1 steps, and the engine's own fp32 logits are kept at every 10th position of those (292 positions, T=1 and T=4). 10 docs (code 3, prose 3, math 2, greedy 2; 3436 positions) are calibration docs used only for the activation statistics of the act-weighted clip search and are never evaluated. Evaluated positions: 16..n-2 of natural docs, prompt_len-1..n-2 of greedy docs (next token = target): **15,280 positions, 43 docs** (code 14, prose 9, math 10, greedy 10).

**Reference** = `lm_head.weight` bf16 from the checkpoint, fp32 matmul (no TF32; bf16 x bf16 products are exact in fp32), fp32 logits. The engine's own head kernel agrees with it (Table 2: 0 argmax flips in 584 rows, KL 1e-9), so the reference is the shipped head up to accumulation order.

**Arms** (`lmq.py`): `int8_rtn` symmetric per-row, scale = amax/127 fp32; `int8_clip` per-row clip ratio searched on 26 values in [0.5, 1] minimising the row's squared weight error; `int6_rtn` per-row levels +-31; `int4_g128_rtn` codes [-8,7] (engine W4 format, zero-point 8; with scale = amax/7 the used range is [-7,7]), scale per 128 columns rounded to bf16; `int4_g128_clip` per-group scale ratio searched on 31 values in [0.5, 1.1] x amax/7 (bf16-rounded before coding) minimising weight MSE; `int4_g128_awclip` same search minimising the activation-weighted error sum_j E[x_j^2] (w_j - what_j)^2 with E[x^2] from the calibration docs. Post-hoc: `int8_g128_rtn` (amax/127 per 128 columns, bf16), `+rescoreM` (the M largest quantised logits are replaced by the reference logits, the other logits keep their quantised value). `bf16_logits_rounded` = reference logits rounded to bf16 (what a bf16-output head such as stock vLLM produces), reported as a floor.

**Metrics** (per position, `lmq.metrics`; pooled mean with a document-cluster bootstrap 95% CI, 2000 resamples): top-1 agreement (argmax of quantised == reference); KL(ref||quant) over the full vocab and over the reference top-20 (renormalised); "top-20 mass moved" = 0.5 sum_{top-20 ids} |p_ref - p_q|; dNLL = NLL_quant - NLL_ref of the true next token (> 0 = worse); sampling TV = total variation between the filtered distributions after temperature (T=1 and 0.7), top-k 20, then top-p 0.95 on the renormalised top-k (vLLM order); verify-block proxy over aligned 4-row blocks (3788 blocks): P(any argmax change in the 4 rows), and "accepted drafts lost" = mean over blocks of 3 - (number of leading rows among rows 0-2 whose argmax is unchanged), which is the loss of accepted draft tokens for an ideal drafter whose drafts equal the reference argmax chain (upper bound on the loss with a real drafter). Decision rule as in the frontmatter: top-1 >= 99.5% AND (dNLL CI includes 0 OR mean < 0.002) AND TV(T=1) < 0.005.

**Floors** (Table 2): engine T=1 / T=4 head kernel logits vs the fp32 reference on the same hidden states; hidden T=1 vs T=4 rows (both through the fp32 head; this is the plain-vs-verify-step discrepancy of the engine itself); bf16-rounded logits.

**Speed** (`bench_head.cu`, built with nvcc -O3 -arch=sm_86 against `libswiftmk.so`; header has the command): N=248320 K=5120, random data, 256 MB memset between calls (L2-cold), CUDA events, median of 40, GB/s counts weight + scale bytes; int8 per-row CUDA-core GEMV written here (T = 1..4, fp32 logits out, correctness vs a naive double-accumulate kernel: rel-L2 7.6e-8); baselines `smk_bf16_gemv_f32` and `smk_w4_gemv` with `SMK_EPI_F32` (default + the 6 `_cfg` variants); select+rescore overhead (top-16 two-stage select over fp32 logits + 16 exact bf16 dots per row; select validated against a host sort, 16/16). `src/gemv.cu` untouched. Step model: engine/BENCH.md plain step 20.9 ms (bf16 head 2.87 ms), T=4 verify step 22.5 ms, MTP k=3 cycle 24.2 ms; gain = measured bf16 head time - measured quantised head time (- overhead).

**Budget**: ~4.5 min engine collect, 1 min analysis, 1 min bench on the GPU lease (3090); lease wait was the cost (~50 min).

## Results

Run 2026-10-31, RTX 3090 (82 SMs, SM clock 1695 MHz / 319 W sampled right after the bench), repo HEAD `c1fa55b` plus uncommitted engine edits by other agents (K5 gemv / K6 prefill; the bench links `build/libswiftmk.so` as built at that time), full tables in `results/tables.md`, raw summary `results/summary.json`, per-position metrics `runs/per_position.npz` (ignored), docs `results/docs.json`, bench log `results/bench_head.log`.

### Table 1. Pooled, 15,280 positions (rule: top-1 >= 99.5, dNLL CI contains 0 or < 0.002, TV < 0.005)

| head arm | MB | weight rel-L2 | top-1 agree % [95% CI] | flips | KL full | dNLL nats [CI] | TV k20/p.95 T=1 [CI] | TV T=0.7 | top1 / dNLL / TV |
|---|---|---|---|---|---|---|---|---|---|
| int8_rtn | 1213 | 0.0120 | 99.19 [98.90, 99.43] | 124 | 8.0e-05 | +0.00012 [-0.00009, +0.00033] | 0.0038 [0.0032, 0.0044] | 0.0041 | FAIL / pass / pass |
| int8_clip | 1213 | 0.0118 | 99.26 [99.01, 99.49] | 113 | 1.0e-04 | +0.00002 [-0.00027, +0.00032] | 0.0043 [0.0036, 0.0049] | 0.0046 | FAIL / pass / pass |
| int6_rtn | 910 | 0.0490 | 97.30 [96.72, 97.87] | 412 | 1.45e-03 | +0.00110 [+0.00031, +0.00191] | 0.0158 [0.0134, 0.0182] | 0.0173 | FAIL / pass / FAIL |
| int4_g128_rtn | 625 | 0.1249 | 93.83 [92.59, 94.99] | 943 | 1.02e-02 | +0.00940 [+0.00669, +0.01231] | 0.0417 [0.0353, 0.0480] | 0.0449 | FAIL / FAIL / FAIL |
| int4_g128_clip | 625 | 0.1069 | 93.18 [91.79, 94.50] | 1042 | 1.42e-02 | +0.01250 [+0.00843, +0.01665] | 0.0480 [0.0409, 0.0551] | 0.0512 | FAIL / FAIL / FAIL |
| int4_g128_awclip | 625 | 0.1113 | 94.74 [93.77, 95.65] | 803 | 6.9e-03 | +0.00734 [+0.00508, +0.00973] | 0.0349 [0.0298, 0.0399] | 0.0375 | FAIL / FAIL / FAIL |
| *post-hoc* int8_g128_rtn | 1231 | 0.0069 | 99.50 [99.34, 99.65] | 76 | 4.0e-05 | +0.00004 [-0.00010, +0.00018] | 0.0026 [0.0022, 0.0030] | 0.0028 | pass / pass / pass (99.503%) |
| floor: bf16-rounded ref logits | 2543 | - | 98.28 [97.80, 98.71] | 263 | 1.9e-04 | +0.00052 [+0.00024, +0.00079] | 0.0065 | 0.0071 | - |

Per category (top-1 % / TV T=1, `tables.md` Table 3): int8_rtn code 99.61 / 0.0025, prose 98.90 / 0.0057, math 98.36 / 0.0056, greedy chat 99.92 / 0.0016 (prose and math individually exceed TV 0.005); int4 awclip code 97.0 / 0.024, prose 92.6 / 0.052, math 91.2 / 0.048, greedy 98.1 / 0.016.

### Table 2. Noise floors

| floor | n | top-1 agree % [CI] | flips | KL full | dNLL [CI] | TV T=1 |
|---|---|---|---|---|---|---|
| engine T=1 head kernel vs fp32 ref (same hidden) | 292 | 100.00 | 0 | 3e-10 | 0 | 0 |
| engine T=4 head kernel (mma) vs fp32 ref | 292 | 100.00 | 0 | -7e-9 | 0 | 0 |
| hidden T=1 vs T=4 rows (whole engine, fp32 head) | 2938 | 99.52 [99.31, 99.75] | 14 | 1.5e-04 | +0.00028 [-0.00031, +0.00102] | 0.0045 |
| ref logits rounded to bf16 | 15280 | 98.28 [97.80, 98.71] | 263 | 1.9e-04 | +0.00052 | 0.0065 |

The final-norm hidden T=1 vs T=4 relative L2 per floor doc is 0.012-0.037 except one code doc at 0.110.

### Table 3. Why top-1 is brittle: reference top1-top2 logit margin vs flip rate % (`tables.md` Table 7)

| margin bin | positions | share % | int8_rtn | int8_g128 | int6 | int4 awclip | bf16-rounded | int8_rtn+rescore8 | int4 awclip+rescore16 |
|---|---|---|---|---|---|---|---|---|---|
| [0, 0.01) | 98 | 0.64 | 52.0 | 37.8 | 57.1 | 64.3 | 55.1 | 0 | 0 |
| [0.01, 0.03) | 184 | 1.20 | 31.0 | 17.9 | 47.8 | 52.7 | 45.1 | 0 | 0 |
| [0.03, 0.1) | 536 | 3.51 | 3.0 | 1.1 | 30.4 | 48.7 | 21.8 | 0 | 0 |
| [0.1, 0.3) | 1198 | 7.84 | 0 | 0 | 8.5 | 24.5 | 0.75 | 0 | 0 |
| [0.3, 1.0) | 2860 | 18.72 | 0 | 0 | 0.1 | 3.1 | 0 | 0 | 0 |
| [1.0, inf) | 10404 | 68.09 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

1.84% of positions have a reference margin < 0.03 logits and 5.4% < 0.1: a coin-flip there costs nothing in loss (dNLL +0.0001 for int8) but counts as a top-1 miss. int8 flips have median margin 0.013; int4 flips 0.09-0.13.

### Table 4. Post-hoc rescoring arms (exploratory): quantised head + exact bf16 rescoring of the top-M candidates

| arm | top-1 agree % [CI] | flips | KL full | dNLL [CI] | TV T=1 [CI] | top1 / dNLL / TV |
|---|---|---|---|---|---|---|
| int8_rtn + rescore4 | 99.98 [99.94, 100.00] | 3 | 2.7e-05 | +0.00001 [-0.00016, +0.00016] | 0.0014 [0.0011, 0.0017] | pass / pass / pass |
| int8_rtn + rescore8 | 100.00 | 0 | 1.7e-05 | +0.00000 [-0.00011, +0.00012] | 0.0008 [0.0007, 0.0010] | pass / pass / pass |
| int8_rtn + rescore16 | 100.00 | 0 | 1.2e-05 | +0.00000 [-0.00009, +0.00009] | 0.0003 [0.0002, 0.0004] | pass / pass / pass |
| int4_g128_rtn + rescore8 | 99.93 [99.87, 99.98] | 11 | 2.6e-03 | +0.00247 [+0.00121, +0.00381] | 0.0089 [0.0070, 0.0107] | pass / FAIL / FAIL |
| int4_g128_rtn + rescore16 | 100.00 | 0 | 1.8e-03 | +0.00180 [+0.00085, +0.00276] | 0.0032 [0.0024, 0.0039] | pass / pass (0.0018 < 0.002) / pass |
| int4_g128_awclip + rescore16 | 100.00 | 0 | 1.2e-03 | +0.00256 [+0.00164, +0.00352] | 0.0029 [0.0022, 0.0035] | pass / FAIL (CI excludes 0, 0.0026 >= 0.002) / pass |

The residual dNLL of the int4 + rescore arms (+0.0018..+0.0026, CI excludes 0) is [INFERENCE, not isolated] the quantisation noise of the ~248k non-rescored tail logits inflating the softmax normaliser (Jensen bias); rescoring only removes the top-M errors.

### Table 5. lm_head GEMV, N=248320 K=5120, L2-cold, ms (GB/s of weight+scale bytes)

| kernel | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| bf16 `smk_bf16_gemv_f32` (shipped; CUDA-core T=1, mma T>=2) | 2.875 (884) | 2.945 (863) | 2.969 (857) | 2.985 (852) |
| int8 per-row CUDA-core (`bench_head.cu`) | 1.445 (881) | 1.464 (869) | 1.507 (844) | 1.546 (823) |
| int4 g128 `smk_w4_gemv` EPI_F32 default | 0.778 (842) | 0.785 (835) | 0.798 (822) | 0.824 (795) |
| int4 g128 `smk_w4_gemv_cfg` 1 (best at every T) | 0.775 (846) | 0.779 (841) | 0.783 (837) | 0.786 (834) |
| select top-16 + exact rescore of 16 rows per verify row (M=16) | 0.053 | 0.056 | 0.059 | 0.065 |

(cfg 0/2/3/4/5: 0.776-0.812 / 1.03-1.07 / 0.81-0.83 / 0.776-0.787 / 1.28-1.30 ms; full list in `results/bench_head.log`.)

### Table 6. Modelled gain (step = engine/BENCH.md: plain 20.9 ms, T=4 verify 22.5 ms, MTP k=3 cycle 24.2 ms with the draft path unchanged)

| head | T=1 step ms | T=1 gain | T=4 verify ms | verify gain | MTP cycle speedup | code tok/s (base 145) | accepted-length factor, code / prose (upper-bound loss, `tables.md` Table 8) | net code / prose tok/s (base 145 / 94.7) |
|---|---|---|---|---|---|---|---|---|
| int8 per-row | 19.47 | 1.43 ms = 6.8% (plain 47.8 -> 51.4 tok/s) | 21.06 | 1.44 ms = 6.4% | x1.063 | 154 | 0.988 / 0.982 | 152.3 / 98.9 |
| int8 per-row + rescore16 | 19.52 | 1.37 ms = 6.6% | 21.13 | 1.37 ms = 6.1% | x1.060 | 154 | 1.000 / 1.000 | 153.7 / 100.4 |
| int4 g128 + rescore16 | 18.85 | 2.04 ms = 9.8% | 20.37 | 2.13 ms = 9.5% | x1.097 | 159 | 1.000 / 1.000 | 159.0 / 103.9 |
| int4 g128 awclip, no rescore | 18.80 | 2.10 ms = 10.0% | 20.30 | 2.20 ms = 9.8% | x1.100 | 159 | 0.915 / 0.870 | 146.0 / 90.6 |

(T=1 rows with rescore add the measured 0.053 ms; int8 g128 (not written) would read +1.5% bytes: modelled ~1.47 ms, [INFERENCE].)

## Conclusion

**Answer.** Under the pre-registered rule no head is adoptable on its own. The closest is int8 per-row (RTN 99.19%, clip-search 99.26%): it passes the log-loss criterion (+0.00012 nats, CI contains 0) and the sampling criterion (TV 0.0038) and misses only the top-1 criterion by 0.3 points, because the 124 flips are all near-ties (median logit margin 0.013, none above 0.1) that cost nothing in likelihood. The criterion itself is near the noise floor: the engine's plain-vs-verify (T=1 vs T=4) discrepancy already gives 99.52%, and bf16-rounded logits 98.28%. int6 (97.3%) and every int4 g128 variant (93.2-94.7%, TV 0.035-0.048, dNLL +0.007..+0.0125, 18-22% of 4-row verify blocks change an argmax) clearly fail; plain int4 would also cancel its speed gain (Table 6 last row: code 146 vs 145 tok/s). Passing configurations found post-hoc: **int8 with a bf16 scale per 128 columns** (99.503% [99.34, 99.65], all three criteria; the point estimate is 0.003 pt above the threshold), and **any of int8 / int4 + exact bf16 rescoring of the top-16 candidates** (0 argmax flips in 15,280 positions and in 3788 verify blocks, TV 0.0003 / 0.0032; int4 dNLL +0.0018 passes the < 0.002 clause, awclip's +0.0026 does not).

**Speed.** The head is purely bandwidth-bound: int8 per-row CUDA-core reaches 881 GB/s (1.445 ms, T=1) and 823 GB/s at T=4 (1.546 ms); `smk_w4_gemv` int4 reaches 846 GB/s (0.775 ms with cfg 1, the default config is 0.778 / 0.824 at T=1 / 4). Against the shipped bf16 head (2.875 / 2.985 ms): plain step -1.43 ms (6.8%) int8, -2.10 ms (10.0%) int4; T=4 verify -1.44 ms (6.4%) / -2.20 ms (9.8%); k=3 MTP cycle x1.063 / x1.100 (code 145 -> ~154 / ~159 tok/s) before any acceptance loss.

**Recommendation.** (1) Do not ship plain int4 or int6 heads. (2) Implement **int8 per-row head + top-M exact rescoring against the retained bf16 head** for the verify rows (M = 16, 65 us, kernel prototype validated in `bench_head.cu`): +6.0% MTP-cycle throughput (code ~153.7, prose ~100.4 tok/s), argmax identical to the bf16 head on every measured position (keeps "greedy MTP == plain greedy" intact) and sampling TV 0.0003; cost +1.21 GB VRAM (bf16 head kept for rescoring). (3) The upgrade **int4 g128 RTN + rescore16** adds ~3.5 more points (x1.097 total, code ~159 tok/s, +0.63 GB) and passes all three criteria, but its dNLL is significantly positive (+0.0018 [+0.0009, +0.0028]) and its TV is 0.0032 (T=1) / 0.0012 (T=0.7), so take it only if a small systematic tail bias is acceptable for sampling; use RTN, not clip-search (clip-search made int4 worse). (4) If rescoring is not wanted: int8 g128 passes outright but needs a g128-scale int8 kernel (not written; modelled 1.47 ms, +6.5% step). For sampling, M must be >= the top_k used (16 < 20 measured only through the TV, which was fine; M = 32 untested). The rescoring numbers are offline hybrid-logit metrics, not yet an engine run.

**Against the prediction (Prediction section / frontmatter).** Right: int8 dNLL ~0 with CI containing 0 (+0.00012); int4 fails all criteria and dNLL ~ +0.009 (bet +0.010); int4 KL 1.0e-2 (bet 0.012); T=1-vs-T=4 floor 99.52% (bet 99.6%, range 99.0-99.9); engine kernel vs fp32 floor 100% (bet >= 99.97%); act-weighted clip is the best int4 arm; speeds in/near range (int8 1.445 vs 1.50 [1.45-1.65], T=4 1.546 vs 1.70; int4 0.775 vs 0.88 [0.78-1.0], T=4 0.786 vs 0.95 [0.85-1.1] - faster than the bet at T=4); modelled gains int8 6.8/6.4% (bet 6.5/5.8), int4 10.0/9.8% (bet 9.6/9.1). Wrong: every top-1 / TV level - int8 99.19% (bet 99.9%), int8 KL 8.0e-5 (bet 2e-5), TV 0.0038 (bet 0.001), block change 3.0% (bet < 0.5%), so int8 per-row did NOT pass (bet: ADOPT); int6 97.3% (bet 99.7%, ADOPT); int4 RTN 93.8% (bet 97.5%, range 96.0-98.8), TV 0.042 (bet 0.02); clip-search is *worse* than RTN for int4 (93.2 vs 93.8%, KL +40%; bet: 25% better) and not better for int8 (KL +29%); act-weighted 94.7% (bet 98.4%); bf16-rounded floor 98.28% (bet 99.7%). Cause of the miss: I underestimated the density of near-ties (1.8% of positions < 0.03 logits margin) and the logit noise of a ~1% weight error (~0.013 logits rms).

## Dead ends

- First `collect` run crashed after saving the hidden states (NameError in the engine-logits save, 4.5 min of GPU lease lost); fixed and rerun in full, only the second run's outputs were analysed.
- Weight-MSE clip search is the wrong objective for the head: it lowers weight error (int4 rel-L2 0.125 -> 0.107) but raises logit KL by 40%; only activation-weighted search helps (0.111 rel-L2, KL 0.0069, the best int4 arm). Why is not isolated [INFERENCE: clipped outlier weights/columns carry large logit contributions]. A full GPTQ/Hessian arm was not tried (int4 + rescore already meets the criteria, so not needed for the decision).
- Top-1 agreement >= 99.5% is a poor criterion for this head: 0.65% of positions are essentially ties (margin < 0.01), so even bf16 output rounding fails it (98.28%) and the engine's own T=1/T=4 pair is at 99.52%; dNLL and TV are the informative metrics, top-1 only matters for greedy exactness, which rescoring restores.
- Int8 kernel design notes (no failures, first version already at 881 GB/s; the shipped design was chosen up front, the alternatives below were reasoned, not benchmarked): one warp per row with 16 B per lane and x in smem in natural layout would have 32-way LDS bank conflicts and 16 B of smem traffic per weight byte at T=4; used a lane-permuted smem layout and 2 rows per warp (x fragments reused); int8->fp32 by xor 0x80 + PRMT into the mantissa of 2^23 + FADD instead of I2F.

## Follow-ups

- [[ideas/swift-mk-engine-quantised-lm-head-int8-per-row-or-int4-g128.md]]: integrate int8 (or int4) head + exact top-16 rescoring in the engine verify path and measure end-to-end tok/s and greedy parity (needs select kernel + int8 packer + keeping the bf16 head).
- [[ideas/swift-mk-engine-t-1-vs-t-4-final-norm-hidden-rel-l2-reaches.md]]: one code doc has T=1 vs T=4 final-norm hidden rel-L2 0.110 (median 0.02): find the diverging layer/kernel.
- int8 g128-scale CUDA-core kernel (modelled 1.47 ms) if rescoring is rejected; M = 32 rescoring for top_k > 16; GPTQ-style int4 head.
