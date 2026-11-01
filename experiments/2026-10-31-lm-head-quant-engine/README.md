---
title: lm-head-quant-engine
type: experiment
tags: []
status: done
created: 2026-10-31
author: agent
model: claude-sonnet-5-5
question: "Does the E5 recommendation (quantised lm_head over all 248320 rows + exact bf16 rescoring of the top-R candidates, R=16 greedy / 64 sampling) deliver its modelled gain inside the swift-mk engine (plain step, T=4 verify step, k=3 MTP tok/s on the 4 E0 prompts) while keeping greedy argmax identical to the bf16 head and sampling exactness not degraded, and which of int8 per-row / int4 g128 RTN should be the default?"
prediction: "Frozen before any measurement. (1) Speed (engine, lease held, 5 interleaved A/B repeats, bf16-head path of the same build = baseline): plain T=1 step 20.20 -> 18.85 ms int8 (-1.35 ms, +7.2% tok/s, 49.5 -> 53.0), -> 18.25 ms int4 (-1.95 ms, +10.7%, 54.8 tok/s); the select+rescore tail costs 25-60 us per step (4 small kernels, R=16). T=4 verify step 21.55 -> 20.15 (int8) / 19.5 ms (int4). Greedy k=3 MTP tok/s (bench_spec, T=1 greedy, 4 E0 prompts, engine-direct numbers in BENCH.md as baseline): int8 x1.055-1.065 (code ~160), int4 x1.085-1.10 (code ~165); acceptance lengths unchanged (identical argmax). (2) Parity: argmax of (quantised head + rescoring R=16) == bf16-head argmax on every one of >= 4000 teacher-forced positions and the 3 dump prompts for BOTH int8 and int4 (0 mismatches; candidate-set misses 0; E5 saw 0/15,280); e2e greedy token streams (plain and spec) identical to the bf16-head build except documented near-tie rows (rule: existing tolerance). (3) Sampling: first-token chi-square (20000 samples, top_k 20, T=1) plain vs spec arm with the new head both arms: p > 0.01 for int8 and int4; TV of the sampled distribution vs the bf16-head distribution on 5 fixed positions: int8 < 0.001, int4 < 0.003 (R=64 rescoring for top_k<=64, better than E5 rescore16 TV 0.0032/0.0003); int4 passes. (4) Default: int4 if all of (2)(3) pass and it is >= 3 points faster than int8 on code; else int8. (5) VRAM: +1.27 GB (int8 incl. fp32 scales) / +0.64 GB (int4); max context at a fixed VRAM budget drops by 1.27 GB / 69.6 KB per token (bf16 KV incl. MTP KV) ~ 18k tokens (int8) / 9k (int4); fp8 KV 34k / 17k tokens. Wrong-if: int4 flips an argmax or fails TV; speed gain < 4% for int8 (then the select tail is mis-modelled)."
answer: "Yes, and int4 is the default: quantised head + exact bf16 rescoring of the top-16 (64 when sampling) candidates inside the engine graphs gives plain step 20.15 -> 18.80 ms (int8, x1.073) / 18.105 ms (int4, x1.113), greedy k=3 tok/s code/prose/math/8K 152.1/97.4/150.0/137.7 -> 162.0/103.9/159.8/146.6 (int8) / 167.8/107.4/165.5/151.9 (int4, +10.3%), T=1 sampling +6.1% / +9.9%, with the BF16-head argmax on every one of 19,976 real positions (0 real flips, 1 exact fp32 tie), identical e2e greedy streams, sampled distribution TV <= 1e-5 (top_k 20) and the K7 chi-square unchanged; cost +0.59 GiB (int4) / +1.18 GiB (int8): max-model-len 40960 -> 36864 (bf16 KV), 81920 -> 65536 (fp8), int8 does not fit the 32768 default."
confidence: high
confidence_note: "high for speed / greedy parity (same-process interleaved A/B, +-0.3% spread, 19,976 real positions + 12 stream pairs); medium for sampling exactness at top_k 64 on int4 (mean TV 0.0015 but p99 0.018 / max 0.068 on flat positions) and for the VRAM limits (4096-token grid, one card)"
answers: [ideas/swift-mk-engine-quantised-lm-head-int8-per-row-or-int4-g128.md]
builds_on: []
reproduce: uv run experiments/2026-10-31-lm-head-quant-engine/run.py
---

# lm-head-quant-engine

<!-- Procedure: .agents/skills/run-experiment/SKILL.md
     Fill `question` and `prediction` in frontmatter BEFORE running. Never edit `prediction` afterwards;
     if your expectation changes, add a dated line under Prediction below.
     When done: set status: done, write `answer` (one line other agents will read) and `confidence`. -->

## Question

Does the E5 recommendation (quantised lm_head over all 248320 rows + exact bf16 rescoring of the top-R candidates, R=16 greedy / 64 sampling) deliver its modelled gain inside the swift-mk engine (plain step, T=4 verify step, k=3 MTP tok/s on the 4 E0 prompts) while keeping greedy argmax identical to the bf16 head and sampling exactness not degraded, and which of int8 per-row / int4 g128 RTN should be the default?

## Prediction

Frozen in the frontmatter before any measurement (committed before the first GPU run).

<!-- What you expect and why, including the number you'd bet on. -->

## Method

Code: `engine/{include/head_q.h, src/head_q.cu, src/engine.cu (head_tail), src/mtp.cu, pack.py, engine.py}`, tests `tests/test_headq.py`, `tests/bench_head_ab.py` (commit f306273); build `make BUILD=build/k9`.  Design: approximate logits of all 248320 rows from a quantised copy of the head (int8 per-row fp32 scale, CUDA-core kernel from E5's `bench_head.cu`; int4 g128 RTN = E5's `int4_g128_rtn` in the W4 layout through `smk_w4_gemv_cfg` 1 + `SMK_EPI_F32`; both quantised on the GPU at load, 0.1-0.2 s) -> exact top-R by approximate logit (chunk maxima, per-chunk top-R survivors above the R-th chunk maximum, rank) -> R exact fp32 dots against the resident BF16 head, written over the approximate logits (the buffer the K7 sampler reads) -> argmax over the exact candidates -> `ds[OUT]`.  R = 16 in the greedy graphs, 64 once `smk_spec_set_sampling` has temperature > 0 (the device top-k is <= 64).  Hooks: the head step of `smk_run_step` (plain, verify T=2..4, sampling plain, first-token step), `Engine(head="int4"|"int8"|"bf16")` / `SMK_HEAD`, `Engine.set_head("bf16"|"q")` (graphs re-captured, the spec driver's too).  The 64k draft head is unchanged.
Baseline = the BF16-head path of the same build (`set_head("bf16")`: the pre-K9 kernels, 20.15-20.18 ms/step vs 20.2 committed in K5).  Hardware: RTX 3090 (23.56 GiB), lease held for every run (the lease was contended for ~2 h by a long campaign job, hence the wall time), single process per measurement, 5 interleaved repeats alternating mode order.
Measurements (all raw logs in `results/`): (a) unit + pipeline parity on the 19,979 real final-norm hidden states of E5 (`runs/hidden.pt`, 53 docs) against `smk_bf16_gemv_f32` + `smk_argmax`: GEMV vs fp64 with the dequantised head, candidates vs `torch.topk`, exact logits vs fp64, argmax mismatches / top-1 misses / top-20 and top-64 coverage / processed-distribution TV / teacher-forced log-loss (15,278 eval positions); (b) e2e greedy streams quantised vs BF16 head, 192 tokens, plain and k=3, 3 dump + 3 E0 prompts; (c) sampled distribution: 5 fixed positions x 3 sampling configs, 1e6 draws each; (d) K7 chi-square rerun (configuration A, 20,000 + 20,000 samples); (e) `tests/bench_head_ab.py`: plain step, T=2..4 head steps, k=3 greedy and T=1 sampling tok/s on the 4 E0 prompts (8K prompt rebuilt from the current topics/); (f) `test_e2e.py`, `test_mtp.py`, server `smoke_test.py` / `runner_check.py`, `server/maxctx.py` VRAM probes.

```bash
uv run experiments/2026-10-31-lm-head-quant-engine/run.py
```

## Results

Run 2026-11-01, RTX 3090, repo HEAD `f306273` (K9 code) on top of K5 GEMV / K6+K10 prefill / K7 sampling.  Full BENCH.md section "K9": `engine/BENCH.md`.

| | BF16 head (committed path) | int8 per-row + rescore | int4 g128 + rescore (default) |
|---|---|---|---|
| plain T=1 step, ctx 32 | 20.154-20.181 ms (49.6 tok/s) | 18.800 ms (53.2 tok/s), x1.073 | **18.105 ms (55.2 tok/s), x1.113** |
| head step T=4 (verify rows) | 21.49-21.52 ms | 20.146 ms (x1.068) | 19.366 ms (x1.110) |
| greedy k=3 code / prose / math / 8K | 152.1 / 97.4 / 150.0 / 137.7 | 162.0 / 103.9 / 159.8 / 146.6 | **167.8 / 107.4 / 165.5 / 151.9** |
| T=1 sampling k=3 code / prose / math / 8K | 143.8 / 96.7 / 150.3 / 123.8 | 152.6 / 102.5 / 159.4 / 131.4 | **157.9 / 106.1 / 165.3 / 136.3** |
| argmax != BF16 head (19,976 pos T=4, 1,200 T=1; R=16) | - | 1 (exact fp32 tie), 0 | 1 (same tie), 0 |
| BF16 top-1 outside the candidate set | - | 0 / 0 | 0 / 0 |
| e2e greedy streams (plain + k=3, 6 prompts) | - | 12/12 identical | 12/12 identical |
| top-20 in the 64 candidates (sampling set) | - | 19,976 / 19,976 | 19,968 / 19,976 |
| TV vs BF16 head, T=1 top_k 20 p .95 (4,994 pos) | - | 0.00000 | 0.00000 (max 0.019) |
| TV vs BF16 head, T=1 top_k 64 (4,994 pos) | - | mean 0.00013 (p99 0.0031) | mean 0.00154 (p99 0.018, max 0.068) |
| log-loss dNLL, full softmax / sampler top-64 (15,278 pos) | - | -0.00001 +- 0.00003 / +0.00000 | +0.00080 +- 0.00035 / -0.00012 +- 0.00003 |
| device used at max-model-len 32768 bf16 KV (card 23.56 GiB) | 22.61 GiB | OOM | 23.22 GiB (max measured 36864: 23.50) |
| device used at 65536 fp8 KV | 22.86 GiB (max 81920) | OOM | 23.47 GiB (max measured) |

Sampled distribution (1e6 draws/cell, 15 cells): analytic TV(quantised head, BF16 head) <= 0.00001 (int8 0.00000), empirical <= 0.0012 = BF16 sampling noise.  K7 chi-square rerun with int4 in both arms: first token p = 0.211, joint (101 cells) chi2 82.2 df 81 p = 0.441, two-sample p = 0.123 / 0.192, 0 stray tokens: PASS (statistics equal to K7's BF16-head run).  `test_e2e.py` ALL PARITY OK; `test_mtp.py` all OK (the documented prose near-tie at index 104, margin 0.0655, unchanged); server `smoke_test.py` 22/22; `runner_check.py` 2 failures (thinking-on snapshot reuse) identical with `SMK_HEAD=bf16`, so not caused by the head.  Quantisation at load 0.11 s (int8) / 0.16-0.19 s (int4): no cache file.

## Conclusion

**The E5 recommendation transfers to the engine nearly exactly.**  Measured gain vs E5's model: plain step -1.38 / -2.05 ms (model -1.37 / -2.04), k=3 tok/s +6.0-6.5% (int8, model +6.0%) / +9.8-10.4% (int4, model +9.7%); the select + rescore tail costs ~50 us.  Greedy output is the BF16 head's on all 19,976 positions (the BF16 top-1 was always among the 16 candidates; the one differing argmax, position 12527 at T=4, is an exact fp32 tie between tokens 279 and 264 (margin 0.0) resolved the other way by the accumulation order of the rescoring dot vs the mma kernel), so spec == plain greedy parity and the committed test suites hold, and the sampling distribution is unchanged within 1e-5 at top_k <= 20.
**Default = int4** (rule: int4 only if sampling TV and log-loss pass: they do; it is +3.7 points faster than int8 and needs 0.59 instead of 1.18 GiB, which matters because the BF16 head must stay resident: int8 does not fit the server's default 32768 bf16 context).  Caveat: with int4 and top_k > 20 the rescoring set (64) misses part of the true top-64 on flat positions (mean TV 0.0015, p99 0.018 at top_k 64, which is what the server uses when the request has no top_k <= 64); `SMK_HEAD=int8` (TV 0.00013) or `SMK_HEAD=bf16` restore exactness at a cost of ~4 / ~10 points of tok/s.
**Against the prediction (frozen in the frontmatter)**: right: int8 18.85 -> 18.80 ms, int4 18.25 -> 18.105 ms (slightly better), T=4 verify 20.15 / 19.5 -> 20.146 / 19.366, int8 +5.5-6.5% (+6.0-6.5%), parity, TV (int8 < 0.001, int4 < 0.003 at 5 positions: <= 1e-5), chi-square p > 0.01, default int4, tail 25-60 us, VRAM cost +0.59 / +1.27 GB (int4 measured +0.61 GiB at max-model-len 32768; int8 +1.18 GiB from the tensor sizes, it OOMs at 32768), max context drop (bf16 40960 -> 36864 on a 4096 grid; fp8 81920 -> 65536).  Wrong: int4 k=3 speedup x1.104 vs the bet x1.085-1.10 (code 167.8 vs ~165); "0 mismatches" was 0 real + 1 exact fp32 tie (T=4, position 12527); the VRAM bet "~9k / 17k tokens" is not resolved at the 4096-token probe grid (fp8 lost 16-24k tokens, bf16 4-8k: the old maxima had unknown headroom).

## Dead ends

- First select kernel (every logit >= the R-th largest chunk maximum into a 4096-entry list) lost the true top-1 at 94 / 19,976 positions (0.47%, R = 64; R = 16 never overflowed): a flat cluster of near-identical logits at the threshold overflows the list and the atomic append order decides what is dropped.  Found by the pipeline-parity test (not by the 8-row unit test); fixed by letting each chunk contribute at most its own top-R (list <= 128 x 64, deterministic, ties by lower index).  The first parity/sampling numbers (identical streams, TV <= 1e-5) were taken with the flawed kernel and rerun after the fix.
- The unit test's R=64 "BF16 top-1 inside the candidate set" check initially read the candidate table through a host `from_address` on a device pointer (segfault): read through the ws tensor instead.
- First e2e chi-square used `--extra 6` (copied from K7's run.py): that position is a point mass (support 1, degenerate test); K7's committed configuration A is `--extra 3` (16-token support), rerun with it.
- int8 g128 (E5 post-hoc passing arm) not built: per-row int8 + rescoring already has 0 flips.  int4 awclip not built (needs calibration, int4 RTN + exact rescoring passes).
- `runner_check.py` fails 2 checks (thinking-on snapshot reuse) both with the int4 and with the BF16 head: a K8/K10 server issue, not examined here.

## Follow-ups

<!-- New ideas go through `./rsrch idea "..."`; list them here as links. -->
- top_k > 20 on int4: R = 128 (needs RMAX / CAP / smem changes) or a g128 int8 head would close the 0.0015 mean TV at top_k 64 if that case matters.
- Fuse the chunk-maximum into the head GEMV epilogue and the argmax/select kernels into one (4 launches, ~50 us = 0.25% of the step; not worth it yet).
- The hard remaining head cost is the BF16 copy that must stay resident (2.4 GiB): rescoring from an int8 g128 copy that is exact enough (not bitwise BF16) would free 1.2 GiB of VRAM = ~17k context tokens, but then the argmax is no longer the BF16 head's.
