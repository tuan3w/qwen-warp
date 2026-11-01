---
title: spec-sampling-exactness
type: experiment
tags: []
status: done
created: 2026-10-31
author: agent
model: claude-sonnet-5-5
question: "Can the swift-mk on-device MTP speculative cycle (one CUDA graph; int4 MTP layer + 64k int4 draft head; BF16 verify of 1+k rows) do sampling at the Qwen settings (T=1.0 / top_p 0.95 / top_k 20, and T=0.6 / top_k 20) with standard speculative-sampling acceptance (accept d with prob min(1,p(d)/q(d)), residual max(0,p-q)/Z on rejection, bonus from p) so that the emitted-token distribution is EXACTLY the autoregressive target distribution (chi-square on the first emitted token and the joint of the first two emitted tokens, 20000 plain samples vs the cycle, plus a deliberately broken no-residual variant that must fail), what does it cost per cycle (top-k over 248k fp32 logits x (1+k) rows + draft-head top-k + RNG, all in-graph), and which draft proposal gives the longer accepted length at T=1: q = the draft head's own filtered distribution (sampled drafts) or the greedy-draft delta (accept w.p. p(d))?"
prediction: "Frozen before any run. (1) Exactness: for the exact variant every chi-square / two-sample test on the top-20 candidate cells (pooled cells with expected count < 5) gives p > 0.01 (bet: all p-values look uniform on (0,1); the smallest of ~10 tests > 0.005), total variation vs the reference < 0.02 at N=20000 (sampling noise floor ~0.01), and no emitted token ever lies outside the target top-20 support. The broken variant (accept w.p. min(1,p/q) but on rejection resample from p instead of from the residual) FAILS: p < 1e-6 on the first-token test for the low-acceptance prose prompt (TV ~0.05-0.10) and p < 0.001 on the other prompts. top_k/top_p semantics equal transformers' Temperature->TopK->TopP warpers exactly on random and peaked logit vectors (same support, probabilities within 1e-6). (2) Sampling kernel cost per cycle (target rows 1+k=4: two-stage block top-k over 4 x 248320 fp32 + softmax/top-p + accept/residual/philox kernel): 0.05-0.12 ms (< 0.15 ms bet 0.08 ms); each draft-head top-k+sample over the 64k logits 10-20 us. So the cycle grows by < 0.5% (24.2 -> ~24.5 ms at k=3). (3) Accepted length at k=3, T=1/top_p0.95/top_k20 with sampled drafts (q = draft head filtered with the same T/top_k/top_p): code 3.00 (2.7-3.3), prose 1.95 (1.7-2.2), math 3.10 (2.8-3.4), 8K 2.65 (2.4-2.9); greedy-draft delta variant is lower on every prompt: code -4%, prose -9%, math -4%, 8K -6% relative (sampled q wins because sum min(p,q) >= p(argmax q) whenever q ~ p). tok/s at k=3 (sampled drafts): code 123, prose 80, math 126, 8K 105 = 1.43x / 1.41x / 1.43x / 1.38x of the E0 vLLM T=1 references (86.0 / 56.9 / 88.1 / 76.0); k=2 and k=1 are lower in the same order as greedy (k=1 ~86 / 74 / 87 / 79 tok/s minus 5%). T=0.6 raises accepted length ~8% over T=1."
answer: "Yes: temperature/top-k/top-p sampling with exact speculative verification runs inside the CUDA graph at +0.11 ms/cycle (0.5%): T=1 top_p .95 top_k 20 k=3 = 144.5 / 90.5 / 150.3 / 124.9 tok/s on code/prose/math/8K = 1.68 / 1.59 / 1.71 / 1.64x the E0 vLLM T=1 numbers (86.0 / 56.9 / 88.1 / 76.0), accepted length 3.34 / 2.09 / 3.47 / 2.99; kernel-level (2e6 cycles) and end-to-end (20000 samples, first two tokens) chi-square tests pass (p 0.04-0.96 / 0.12-0.44) and the no-residual variant fails them (p < 1e-57); sampled drafts beat greedy drafts by 0-4% of E at k=3 (9% of first-draft acceptance on prose); T=0.6 does not lengthen acceptance."
confidence: medium
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: uv run experiments/2026-10-31-spec-sampling-exactness/run.py
---

# spec-sampling-exactness

## Question

Can the swift-mk on-device MTP cycle sample (T=1.0 top_p 0.95 top_k 20, and T=0.6) with standard speculative-sampling acceptance so that the emitted distribution is exactly the target's autoregressive one, at what cost, and does sampling the drafts from the draft head's own distribution beat greedy drafts? (Full question in the frontmatter.)

## Prediction

Frozen in the frontmatter (commit `c1fa55b`, before any run). Summary: exact variant passes every test at p > 0.01, the no-residual variant fails (p < 1e-6 on the first token of a low-acceptance prompt); sampling kernels 0.05-0.12 ms/cycle (< 0.15 ms), +<0.5% cycle; E at k=3 T=1: code 3.0, prose 1.95, math 3.1, 8K 2.65; delta drafts 4-9% lower; tok/s 123 / 80 / 126 / 105 = 1.38-1.43x vLLM T=1; T=0.6 +8% E.

## Method

Code: `engine/` (`include/mtp.h` section K7, `src/mtp_sample.cu|h`, `src/mtp.cu`, `spec.py`, `tests/test_sampling.py`, `tests/bench_spec.py`; design + numbers in `engine/BENCH.md` "K7"). Build `make BUILD=build/k7`, lab venv, GPU lease around every run, bf16 KV, Swift-1.5-Qwen3.8-27b-INT4, the engine tree with K5's GEMV rewrite (8fcbc1e).

* Target distribution per verify row: logits/T -> top-k (<= 64) -> top-p -> softmax over the survivors, computed in the graph by a 3-kernel device top-k (chunk maxima, thresholded block top-k, shared-memory K-way merge: 30 us for 4 x 248320 logits). Drafts: q = the same processing of the restricted-vocab draft head (`draft_sample = 1`) or delta at its argmax (`0`). Accept w.p. min(1, p/q), residual max(0, p-q)/Z, bonus from p. philox4x32-10, counter = device cycle counter.
* Tests: (1) philox known answers; (2) device top-k/top-p vs transformers' Temperature/TopK/TopP warpers and vLLM `apply_top_k_top_p`; (3) kernel-level: the graph's draft-pick + accept functions on fixed rows, 2e6 independent k=3 cycles x 3 configs, chi-square of the emitted length, y0, (y0,y1 | a>=2), (y0,y1,y2 | a>=3) against the analytic speculative-sampling distributions, broken variant (`test_flags` bit 0: on rejection resample from p) must fail; (4) end to end: snapshot after the prefill, N x (restore, start, cycles until 2 tokens) in the real graph vs plain autoregressive sampling from the same snapshot (device sampler on the engine's own logits, T-row numerics) and vs analytic HF probabilities; positions chosen by a scan for rich distributions (the first attempt at `--extra 6` / `math` landed on single-token supports and was discarded and rerun at informative positions).
* Bench: `tests/bench_spec.py --temperature ... --fast-prefill` (E0 prompts, 384 tokens, median of 3).

## Results

All outputs in `results/`: `test_sampling_kernel.md`, `e2e_analysis_{A,A_broken,B,B_broken}.md` (+ raw samples `e2e_*.json`), `bench_*.log|json`, `greedy_regression*.log`, `bench_draft_temperature.log`. Headline numbers and all tables are in `engine/BENCH.md` (K7 section):

* tok/s T=1 top_p .95 top_k 20 k=3: 144.5 / 90.5 / 150.3 / 124.9 (code/prose/math/8K) vs vLLM T=1 86.0 / 56.9 / 88.1 / 76.0 = 1.68x / 1.59x / 1.71x / 1.64x; k=1: 89.9 / 80.4 / 92.5 / 83.8, k=2: 125.2 / 90.7 / 125.1 / 103.6; accepted length 3.34 / 2.09 / 3.47 / 2.99 (greedy same build 3.50 / 2.20 / 3.43 / 2.90; delta drafts 3.27 / 2.09 / 3.33 / 2.87).
* Sampling kernels per cycle at k=3: target rows 0.045 ms, drafts 0.068 ms (0.5% of the 23.1 ms cycle).
* Exactness: kernel-level p-values 0.044-0.96 (all statistics, 3 configs, 2e6 cycles), broken variant p = 0; e2e A (20000 samples, k=3, T=1): first token p = 0.164, joint p = 0.441 (broken: 1e-104 / 2.5e-128); e2e B (6000 samples, k=2, T=0.6, delta drafts): 0.143 / 0.23 (broken: joint 2e-58).

## Conclusion

Exact on-device speculative sampling works and is almost free; the spec cycle at the Qwen sampling settings is 1.6-1.7x stock vLLM at T=1 (vLLM's own greedy-draft rejection sampler), with greedy and sampled throughput within 5%.
Against the prediction: exactness, power (broken fails with TV 0.13-0.19 rather than 0.05-0.10), HF/vLLM semantics (max |dp| 2.7e-7), kernel cost (0.045 ms target rows < 0.15; drafts 18-23 us each, slightly above 10-20 us; cycle +0.49%) hold. Wrong: accepted length and speed were underestimated (E 3.34 / 2.09 / 3.47 / 2.99 vs 3.0 / 1.95 / 3.1 / 2.65; ratios 1.59-1.71x vs 1.38-1.43x; the engine also got ~5% faster from K5's GEMV work between measurements); delta drafts are 0-4% of E lower, not 4-9% (prose: equal, only the first-position acceptance differs, 0.61 vs 0.70); T=0.6 did not raise E (it lowered it by 1-4%). The statement "all p > 0.01" held only after fixing the reference: the plain T=1 engine step differs from the T=1+k row verify step by ~0.05 in logits (TV 0.007 / 0.024 first token at T=1 / T=0.6), which made the e2e test fail at T=0.6 (p = 2e-7) against the T=1-step reference; compared against the same T-row numerics the sampler passes (see BENCH.md).

## Dead ends

* e2e at `--extra 6` (prose) and `math --extra 10`: the tested position had a single-token support (p1 = 1.0), so the test had no information (20 min of GPU time); positions are now picked from an entropy scan.
* Reference from the T=1 engine step: invalid for T=0.6 (numerics of T-row vs T=1 kernels, above); kept as a diagnostic line in the analysis.
* First top-k kernel (block-argmax K times per 2048-logit chunk + 1024-thread merge): 41 + 50 us per call (two 32-warp reductions per extracted element); thresholded stage 1 + 128-thread merge: 30 us.
* Lane-0-only shared stores inside the warp-level accept loop deadlocked the kernel (shuffles with divergent lanes) after 4 correct cycles; all lanes now store.
* Draft proposal temperature 0.85 / 0.7 instead of T: no consistent gain (prose +6%, code/math -1..-2%, single seed).

## Follow-ups

* min_p, top_k > 64 / disabled, per-request sampling params in a batch (needed for the server), repetition penalties.
* Seed-averaged E for the delta vs sampled draft comparison (single seed here; SE of E ~0.09 per cell).
