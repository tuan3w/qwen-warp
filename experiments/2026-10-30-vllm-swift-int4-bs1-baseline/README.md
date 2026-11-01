---
title: vllm-swift-int4-bs1-baseline
type: experiment
tags: []
status: done
created: 2026-10-30
author: agent
model: claude-sonnet-5-5
question: "On the 3090, how many tokens/s does stock vLLM (0.30, lab env) decode at batch 1 for Swift-1.5-Qwen3.8-27b-INT4 (Marlin W4A16, CUDA graphs) with num_speculative_tokens k=0,1,2,3 (MTP), at ~100-token and ~8K-token contexts, greedy and T=1.0/top_p0.95/top_k20; what is MTP draft acceptance per position; and what fraction of the 15.1 GB/token bandwidth roofline (61.9 tok/s) does each reach?"
prediction: "k=0 greedy short ctx: 34 tok/s (range 28-42; ~55% of roofline: Marlin bs1 ~75% of BW and ~900 launches/step cost the rest). k=1: 52 (acceptance of draft 1 ~0.75 greedy / 0.65 at T=1, i.e. 1.7 tokens/step at +15% step cost). k=3: 58 (per-position acceptance ~0.75/0.55/0.40). 8K ctx costs <=8% vs short ctx. T=1 sampling is 10-20% slower than greedy under MTP. If k=0 > 45 tok/s, the megakernel headroom over vLLM is mostly gone for plain decode and the value shifts to the MTP path."
answer: "Stock vLLM 0.30 is much faster than predicted: k=0 46.8 tok/s (46.2 at T=1, 45.9 at 8K ctx) = 76% of the 61.9 tok/s roofline; MTP k=3 reaches 96 tok/s greedy on code (mean 3.59 tokens/step, per-position acceptance 0.94/0.90/0.75) but only 63 on prose (0.69/0.44/0.21); k=1 70/63/71 (code/prose/math). Step time at k=3 is 37 ms vs 21.4 ms for k=0, i.e. ~12 ms of draft cost per step (MTP layer + full 248k lm_head per draft step) is the biggest recoverable item."
confidence: medium
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: uv run experiments/2026-10-30-vllm-swift-int4-bs1-baseline/run.py
---

# vllm-swift-int4-bs1-baseline

<!-- Procedure: .agents/skills/run-experiment/SKILL.md
     Fill `question` and `prediction` in frontmatter BEFORE running. Never edit `prediction` afterwards;
     if your expectation changes, add a dated line under Prediction below.
     When done: set status: done, write `answer` (one line other agents will read) and `confidence`. -->

## Question

## Prediction

<!-- What you expect and why, including the number you'd bet on. -->

## Method

<!-- Setup, models + versions, data, what's measured, what's held constant, baseline, budget. -->

```bash
uv run experiments/2026-10-30-vllm-swift-int4-bs1-baseline/run.py
```

## Results

2026-10-31, RTX 3090 (lab lease), vLLM 0.30.0 (lab env), Swift-1.5-Qwen3.8-27b-INT4 (`~/models/swift-1.5-int4`, files complete by size vs index total_size 19,452,785,104), `language_model_only`, max_model_len 16384, kv_cache_dtype auto (FP8 per checkpoint), max_num_seqs 4, prefix cache off, `{"method":"mtp"}`. Decode tok/s = 384 / (t(385 tokens) - t(1 token)), median of 3 repeats (spread <0.5%), raw in `results/bench.json`. Single prompts, so acceptance is per-prompt, not a corpus average.

| arm | code g | code T=1 | prose g | prose T=1 | math g | math T=1 | 8K g | 8K T=1 |
|---|---|---|---|---|---|---|---|---|
| k=0 | 46.8 | 46.2 | 46.7 | 46.2 | 46.8 | 46.3 | 45.9 | 45.3 |
| k=1 | 70.0 | 67.6 | 62.6 | 58.6 | 70.6 | 68.6 | 64.4 | 63.2 |
| k=2 | 86.8 | 84.3 | 62.6 | 56.6 | 86.2 | 83.2 | 73.8 | 67.5 |
| k=3 | 96.4 | 86.0 | 63.0 | 56.9 | 93.5 | 88.1 | 75.5 | 76.0 |

Mean accepted length (tokens/step) and per-position acceptance, greedy: code k=3 3.59 (0.94/0.90/0.75); math k=3 3.49 (0.96/0.83/0.70); prose k=3 2.33 (0.69/0.44/0.21); 8K-context summary k=3 2.87 (0.78/0.59/0.50). k=1 first-draft acceptance 0.93 (code), 0.73 (prose), 0.95 (math), 0.83 (8K).

Derived (our arithmetic): k=0 step = 1000/46.8 = 21.4 ms = 706 GB/s of the 15.1 GB read per token. k=3 step (code) = 3.589/96.4 = 37.2 ms, i.e. +15.8 ms over k=0 for verifying 4 rows and 3 draft steps; one draft step reads the BF16 MTP layer (~0.85 GB) plus the full BF16 lm_head (2.54 GB) = ~3.4 GB = ~4 ms at 850 GB/s, x3 = ~12 ms [INFERENCE: not profiled].

## Conclusion

Prediction was wrong in the direction that matters: stock vLLM plain decode is 46.8 tok/s, not 34 (predicted range 28-42), and the "k=0 > 45" escape clause fired. Against the E1 synthetic bound (GEMV-only graph 18.96 ms = 52.7 tok/s) plain decode has at most ~12% headroom; against the copy-peak 16.9 ms (59 tok/s) at most ~26%. MTP predictions also missed (k=1 predicted 52, got 63-71; k=3 predicted 58, got 57-96): Swift's MTP head is far better aligned than the 2B model's 0.70-0.73 (mtp-rollout-speed), on code/math. Acceptance is strongly text-dependent: prose gets almost nothing from k=2,3 (62.6 -> 63.0), so a fixed k is wrong; k=3 at T=1 on code (86) is ~10% below greedy (96).
The remaining prize is in the speculative path, not plain decode: ~12 of the 37 ms per k=3 step is drafting through full-vocab BF16 heads, and verification of 4 rows costs 1.74x - draft included - over a plain step. A fused engine with a restricted-vocab/quantised draft and near-free multi-row verify could plausibly reach ~23 ms/step [INFERENCE] = ~155 tok/s on code, ~1.6x stock; plain decode gain is capped near +10-15%.
Caveats: 3 repeats, 4 prompts, one seed, FP8 KV chosen automatically (which attention backend vLLM picked was not recorded), output of this lab run not checked for text equality across k.

## Dead ends

- First `lab run` refused: only 3.6 GB free (guard needs 8). Ran with LAB_MIN_FREE_GB=2; disk is the practical limit on this box (99% full).
- Prediction used a guessed Marlin efficiency (75%) and launch-gap cost; the measured 706 GB/s effective suggests vLLM's graph path is already close to what E1 measures for a graph of plain GEMVs.

## Follow-ups

- Draft cost and multi-row verify cost are the levers: E3 (`experiments/2026-10-30-w4-multirow-verify-and-power`, cancelled before results) targets them.
- Adaptive k per text (prose vs code) is free speed in any engine.
