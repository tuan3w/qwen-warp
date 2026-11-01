---
title: dataflow-fused-phases
type: experiment
tags: []
status: abandoned
created: 2026-10-30
author: agent
model: claude-sonnet-5-5
question: "E1 (w4-stream-persistence-vs-graph) found the persistent barrier+ring kernel V4 only 2.5% faster than the best per-op CUDA graph V1b (18.64 vs 19.12 ms for the 15.09 GB, 338-phase synthetic Swift-1.5 step; V0 read-only 16.93 ms) with 5-6.5 us/phase still above V0 (activation gather ~2 us + CTA-arrival tail 2-4 us + in-situ barrier ~1 us) and GEMV variants power-capped at 1455-1545 MHz. Does a genuinely different persistent design beat per-op graphs by >5% on the same step: (A) fewer phases by fusion (gate|up+SiLU*mul+down in ONE phase with fp32 atomics into the residual; phases/step before vs after), (B) dataflow readiness counters + a dedicated gather warp + K-slice-ordered consumption instead of a grid barrier (RMSNorm applied late as a scalar on the dot product), (C) balanced (equal-bytes / split-K) work assignment, (D) a cheaper activation gather and an L2 prefetch buffer that extends cross-phase weight prefetch beyond the smem ring? How close to V0 does the best variant get, where does the rest of the gap go (ablation), and does lowering dequant ALU per byte raise the power-capped clock?"
prediction: "Pre-registered 2026-10-30 before any code of this experiment ran (E1 numbers as priors: V0 16.93, V1b 19.12, V4 18.64 ms). Mechanism: V4's remaining overhead over V0 is the per-phase stall (gather 2 + tail 2-4 + barrier ~1 = 5-7 us) exceeding what the smem ring (D=5, ~3.2 MB = ~3.6 us of DRAM time) can buffer; anything that shrinks the stall or adds buffering recovers part of the 0.85 ms that E1's ablation (abl=7: 17.79 ms) says is sync+gather cost; the other ~0.85 ms (17.79 vs 16.93) is GEMV ALU/power-cap and cannot be removed by scheduling. Bets (median of 5 interleaved repeats x 200 steps, same session as V1a/V1b/V4 re-runs, which I expect within 0.3% of E1): L2 prefetch buffer on V4 (V4+pf) 18.35 ms (range 18.15-18.60); same trick on per-op graph kernels (V1b+pf) 18.95 (18.75-19.10); fusion of gate|up+down on the barrier kernel (64 fewer phases) 18.45 (18.30-18.70), atomics cost ~0.1 ms; balanced assignment alone <=0.1 ms gain; dataflow w/o fusion 18.40 (18.1-18.8); best combined variant (dataflow+fusion+balance+L2 prefetch) 18.10 ms (17.80-18.50) = 833 GB/s = 93.5% of V0, 338 -> 274 phases/step by fusion (or 193 counting the 81 tiny core phases as non-phases). Ratio best/V1b(E1 definition, plain V1b) 1.056 (1.03-1.08), P(>=1.05) = 0.5; against a prefetch-matched V1b+pf 1.035 (1.01-1.06), P(>=1.05) = 0.25. Power: every GEMV-style variant stays at the software cap (throttle mask 0x4, SM clock 1450-1600 MHz vs 1710 read-only); an ALU-ablated dequant (results invalid, ceiling test) lifts the SM clock by >=5% and GB/s by 1-3% for all execution styles equally, so it does not change ratios by more than 0.5%. Remaining gap of the best variant to V0 (~1.2 ms): ~0.8 ms GEMV ALU/clock, ~0.3 ms residual phase dependency (first-tile latency ~1.5 us x ~200 dependent edges), ~0.1 ms launch/other. Correctness: every variant within 1.5x of E1's reassociation noise floor (logits rel-L2 <= 5.7e-3 vs E1's naive reference; late-scale RMS changes only bf16 rounding points) and unit tests <1e-5 vs fp64. KILL CRITERION (as set for this task): if the best persistent/dataflow variant is < 1.05x V1b (E1 definition) after (A)-(D), the persistence idea is dead (three genuinely different attempts: E1 barrier+ring, this dataflow, this fusion); I will additionally report the prefetch-matched ratio because a fair V1 gets the same L2 trick, and the verdict must hold on both."
answer: "Abandoned before any timed result: the human cancelled the run mid-implementation (2026-10-30). Code (df.cuh, bench.cu) is partial and unvalidated. Question (can fused phases / dataflow readiness counters beat per-op CUDA graph by >=5%?) remains OPEN; E3 showed the plain-decode GEMV chain is already within ~10% of copy peak, so it is parked behind the engine build."
confidence: low
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: uv run experiments/2026-10-30-dataflow-fused-phases/run.py
---

# dataflow-fused-phases

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
uv run experiments/2026-10-30-dataflow-fused-phases/run.py
```

## Results

<!-- Run date, hardware, model versions. Small outputs → results/ (tracked). Raw → runs/ (ignored). -->

## Conclusion

<!-- Answer the question. Compare against the prediction. A negative result is a result. -->

## Dead ends

<!-- Everything that failed, and why. Highest-value section in the repo. -->

## Follow-ups

<!-- New ideas go through `./rsrch idea "..."`; list them here as links. -->
