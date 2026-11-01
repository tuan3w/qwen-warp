---
title: pld-tail-engine
type: experiment
tags: []
status: done
created: 2026-11-01
author: agent
model: claude-sonnet-5-5
question: "Does the MTP chain (k=3) + prompt-lookup tail (up to 4 n-gram tokens, verify T<=8 rows) pay off IN THE ENGINE, as E6 (ngram-draft-potential) modelled: (1) what does the real verify step cost V(T) for T=1..8 (K9 int4 head + exact rescoring, DeltaNet ring of 9 slots, attention T>4, W4 mma GEMV T<=8) at ctx ~64 / 2K / 8K, i.e. is E6's modelled V(8) ~ 21.7 ms (V(4)+2.3) right and below its no-loss limit (~27 ms) and the dead-idea threshold 30 ms; (2) what do the host-chosen per-width graphs cost per cycle (host sync + lookup + launch gap); (3) end-to-end greedy and T=1-sampling tok/s per workload class (E0 prompts and E6's 32-task workload, classes a-e) with the tail policy vs the committed engine, accepted lengths vs E6's offline simulation, VRAM / max-model-len cost of the 9-slot ring; (4) is greedy output equal to plain greedy (near-tie rule) and is sampling still exact (chi-square incl. the broken-variant power check) when tail tokens are delta proposals?"
prediction: "Frozen before any engine measurement of T>4 (the engine at HEAD supports T<=4; this prediction is committed before the first V(T>4) run). (1) Verify step V(T), ms, plain step incl. K9 int4 head + R=16 rescoring, ctx ~64: V(1..4) = 18.1 / 18.55 / 18.9 / 19.4 (K9 numbers, reproduced within 2%), V(5) 19.9, V(6) 20.5, V(7) 21.1, V(8) 21.7 (each +-0.7; E6 model = V(4) + E3 GEMV growth + 0.2 ms/row DeltaNet snapshots + 0.02 ms/row attention); the growth is split ~55% W4 GEMVs, ~35% DeltaNet snapshot writes ((1+T) x 151 MB per layer set), ~10% head/attention/other. ctx 2K: +0.2 ms, ctx 8K (bf16 KV): +0.8 ms at every T. P(V(8) at ctx 64 < 25 ms) = 0.85, P(V(8) < 30 ms: idea alive) = 0.97; P(V(8)-V(4) <= 3.0 ms) = 0.8. DeltaNet kernel T=8 (ring 9): 2.0 +- 0.3 ms per 48-layer step (K1 T=4: 1.15), attention T=8 16 layers at 8K bf16: 0.95 +- 0.1 ms (T=4: 0.885). (2) Host-chosen graph per cycle (sync, n-gram lookup, tail H2D, graph launch with a spin-polled mapped result, C++ loop): GPU idle gap 0.05-0.15 ms per cycle (0.2-0.6% of the cycle), so a device-side lookup is not needed; per-cycle sync costs <= 1% on prose. (3) Greedy tok/s with the tail policy (E6's best cell: n=2 longest suffix match of d2 d3 ending the chain, match length >= 3, tail <= 4, so T = 7 or 8 on a match, else T = 4) over the committed engine (same process, interleaved): E6 workload a / b / c = x1.25 / 1.20 / 1.30 (pooled a-c x1.25, 80% interval 1.10-1.40; E6 in-sample modelled 1.37 / 1.32 / 1.46, CV 1.15), d (prose) >= 0.985, e (math) >= 1.00 with an adaptive back-off; E0 prompts: code +5..+15% (167.8 -> ~185), prose +-1%, math +-2%, 8K prompt +-3%. Tokens/cycle on tail cycles within 5% of E6's offline simulation. Sampling (T=1, top_k 20, top_p .95): same ratios within 0.05. Max-model-len with the 9-slot ring (+0.62 GiB): bf16 KV 36864 -> ~28000 (rounded to what fits), fp8 65536 -> ~49000; default stays at tmax=4 (5 slots) unless the A/B passes. (4) Greedy output with tail equals plain greedy on E0 prompts except documented rounding near-ties (top1-top2 margin < 0.4); forced partial acceptance up to 8 rows passes; sampling chi-square (first-token and joint, 20000 samples) p > 0.01 and the broken variant (resample from p instead of the residual at a tail position) fails with p < 1e-6. Default decision: ON by default for greedy and sampling iff pooled a-c >= +5% and d, e >= -0.5%; my bet is that it is turned on (65%), the miss modes being V(7..8) above 25 ms or a d-class loss from the gate-pass tails on prose."
answer: "MTP chain + prompt-lookup tail (T = 4 + <= 4 rows, one CUDA graph per width, host-chosen per cycle with < 10 us of gap) is implemented, exact (greedy == plain greedy, K7 sampling chi-square incl. broken-variant power) and measured: V(8) = 21.6 ms (V(4) 19.6; E6 modelled +2.3 ms, measured +1.98 at ctx 64, +2.32 at 8K), E6 workload greedy a / b / c / d / e = x1.389 / 1.329 / 1.461 / 1.014 / 1.072 (pooled a-c x1.381 [1.29, 1.49], 240 vs 175 tok/s), T=1 sampling x1.359 / 1.315 / 1.426 / 1.004 / 1.072 = E6's model within 1.5 points on a-c; copy-poor E0 prompts x0.98-1.02 (8K summary prompt -2% at gate m=3, x1.005 at m=4); +0.62 GiB for the 9-slot ring (max-model-len bf16 36864 -> 28672, fp8 65536 -> 53248); server default: on (auto, iff the max-model-len fits)"
confidence: medium
confidence_note: "Measured end to end on the real graph: V(T) at three contexts, A/B in one process (interleaved, 3 reps) on E6's 32 tasks + 6 long-context prompts + E0 prompts, greedy and sampling; exactness by unit, kernel-level 2e6-trial chi-square (with power check) and real-graph tests. Caveats: the tail cell (n=2 longest, m=3, Lt=4) was chosen by E6 on the same 32 tasks, so a-c are in-sample for the policy choice (E6 held-out CV: 1.15) while the cost model (V(T), host gap) is now measured; 6-8 tasks per class (bootstrap CIs 8-30 points wide on a-c); one repo, one model; per-task numbers carry +-3% trajectory noise (greedy streams diverge at rounding near-ties)."
models: [swift-1.5-int4 (Swift-1.5-Qwen3.8-27b-INT4, ~/models/swift-1.5-int4; swift-mk engine HEAD 3a9a7cb + K11, build/k11)]
gpu_hours: 3.5
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: uv run experiments/2026-11-01-pld-tail-engine/run.py
---

# pld-tail-engine

## Question

E6 (`experiments/2026-10-31-ngram-draft-potential`) modelled that keeping the MTP chain (k=3) and appending up to 4 prompt-lookup (n-gram) tokens, verified as T <= 8 rows per cycle, gives x1.15 (held-out) to x1.37 (in-sample) on code-edit / log / agent-turn work with no loss on prose / math, **if** V(T = 5..8) stays under ~27 ms — all modelled. This experiment builds it in the engine (`engine`, K11) and measures: V(T) on the real graph, the host-chosen per-width graphs, the end-to-end A/B per workload class, exactness, VRAM. Dead-idea rule (task): V(8) > 30 ms.

## Prediction

Frozen in the frontmatter (commit `e2adaea`, before the first T>4 measurement). Bets: V(5..8) = 19.9 / 20.5 / 21.1 / 21.7 ms, host gap 0.05-0.15 ms/cycle, E6 a / b / c = x1.25 / 1.20 / 1.30, d >= 0.985, e >= 1.00, E0 code +5..15%, +0.62 GiB -> max-model-len ~28K (bf16), default on (65%). Scorecard below.

## Method

- **Engine changes** (`engine`, `build/k11`; all behind a flag, the default engine is unchanged: `tmax = 4`, 5 DeltaNet slots): step buffers 8 rows (`SMK_TMAX` 8); `Engine(tmax=8)` / env `SMK_TMAX=8` allocates the 9-slot ring (`ds[DS_SLOTS]` is the runtime ring size read by every kernel that advances `cur`); DeltaNet T=5..8 (NS=8 V-slices, state tile in registers: 64-72 registers, no spills), attention T=5..8 (NT=4..6 n-tiles x KS warps), K9 head / rescoring T <= 8 (int4; int8 stays T <= 4), top-k / accept for 8 rows, MTP catch-up of up to 8 rows (`SS_MTOK` moved to 48, `SS_N` 64), `sk_delta_q` (tail proposals q = delta), host proposer (`include/pld.h`, E6's cell), per-width cycle graphs `g_pld[samp][k]` with a mapped mailbox (`PldRes`) written by the last kernel, `smk_spec_run_pld`, `smk_spec_pld_prewarm`.
- **Measurement**: `tests/bench_verify.py` (V(T)), `tests/bench_pld.py` (A/B, same process, interleaved, 3 reps, per-task median, class pooled tokens / pooled time, bootstrap over tasks), E6 workload = its 32 tasks (prompt ids from `experiments/2026-10-31-ngram-draft-potential/runs/trace`, stop at the canonical greedy length, 37-768 tokens), E0 prompts (384 tokens), class f = 6 long-context (<= 6000 document tokens) prompts built from `topics/*.md`, greedy and T=1 sampling (top_k 20, top_p .95, sampled chain drafts). Lease held for every run, no server running (except the server tests), RTX 3090.
- **Offline model** (`policy_sim.py`): E6's replay with the measured cycle costs (`results/cost_*.json`) to compare measured vs modelled.

```bash
projects/llm-lab/bin/lab gpu acquire --holder me
make -C engine BUILD=build/k11            # build into a private dir
SMK_BUILD=build/k11 ~/llm-lab/venv/bin/python experiments/2026-11-01-pld-tail-engine/run.py step0|ab|tests
python3 experiments/2026-11-01-pld-tail-engine/run.py tables
```

## Results

Full tables: `results/tables.md` (generated by `analyze_ab.py`), logs and json in `results/`. Numbers also in `engine/BENCH.md` (K11).

**1. V(T)** (ms, plain verify step incl. K9 head, 9-slot ring; `results/bench_verify.log`): ctx 64: 18.40 / 18.89 / 19.21 / 19.63 / 20.07 / 20.62 / 21.17 / **21.61** (T = 1..8); ctx 2048: ... 19.91 / 20.23 / 20.97 / 21.51 / 21.96; ctx 8192: ... 20.11 / 20.61 / 21.27 / 21.89 / 22.43. V(8) - V(4) = 1.98 / 2.06 / 2.32 ms. Components: DeltaNet 48 layers 0.51 (T=1) -> 1.14 (T=4) -> 2.00 ms (T=8); attention 16 layers bf16 8K 0.78 -> 0.88 -> 1.09 ms. Cycle parts (k = 3..7): 20.73 / 21.21 / 21.85 / 22.54 / 23.25 ms (a tail cycle is +2.5 ms = +12%). **The idea is alive** (V(8) 21.6-22.4 ms vs the 30 ms kill threshold and E6's 27 ms no-loss limit).

**2. Host round trip**: the tail needs the previous cycle's tokens and drafts before the width can be chosen. A mapped-memory mailbox + spin-wait in a C++ loop costs 3-5 us of host work and < 10 us of GPU gap per cycle: `notail` (same host loop, tail gate closed) equals the committed async cycles to 0.1% on all four E0 prompts (166.8 / 108.2 / 165.0 / 134.3 tok/s both). No device-side lookup needed.

**3. A/B per class** (greedy, tok/s base -> hybrid, x ratio [95% bootstrap CI over tasks], E tokens/cycle):

| class | tasks | greedy | T=1 sampling | E (greedy base -> hybrid; E6 offline) | tail cycles |
|---|---|---|---|---|---|
| a code edit | 8 | 172.8 -> 240.1, **x1.389** [1.23, 1.59] | x1.359 [1.18, 1.60] | 3.70 -> 5.50 (5.44) | 63% |
| b log / tool output | 6 | 173.4 -> 230.5, **x1.329** [1.17, 1.47] | x1.315 [1.16, 1.45] | 3.71 -> 5.25 (5.23) | 63% |
| c agent turns | 6 | 185.8 -> 271.5, **x1.461** [1.34, 1.52] | x1.426 [1.29, 1.52] | 3.94 -> 6.31 (6.25) | 83% |
| a-c pooled | 20 | 175.4 -> 242.3, **x1.381** [1.29, 1.49] | x1.356 [1.25, 1.48] | 3.75 -> 5.55 | |
| d prose | 6 | 109.9 -> 111.4, x1.014 [0.98, 1.05] | x1.004 [0.98, 1.03] | 2.30 -> 2.36 (2.34) | 7% |
| e reasoning / math | 6 | 166.3 -> 178.2, x1.072 [1.05, 1.10] | x1.072 [1.05, 1.09] | 3.48 -> 3.86 (3.80) | 32% |
| f long-context prose (6K tokens) | 6 | 135.8 -> 138.6, x1.021 [0.98, 1.06] | | 2.90 -> 3.03 | 12-19% |

E0 prompts (single prompts; +-1-2% trajectory noise; greedy; gate m = 3 / 4 / 6): code x1.007 / 1.007 / 1.001, prose x0.990 / 0.999 / 0.999, math x1.007 / 1.017 / 0.994, 8K x0.980 / 1.005 / 1.003. Gate sweep on d / e: x1.014 / 1.008 / 1.003 and x1.072 / 1.054 / 1.022; f: x1.021 / 1.021 / 1.026; offline a-c x1.363 / 1.340 / 1.315. Per-task spread: a1-a3 x1.75-1.77 (326 tok/s, E 7.7: the output copies the input file), minimum a4 x1.06 / b4 x1.02, d1 / d3 / d4 x0.97-0.98.

**4. Model vs measured** (E6 modelled with V(8) - V(4) = 2.30 ms; the offline replay with the measured cycle costs): a / b / c / d / e modelled 1.372 / 1.318 / 1.455 / 0.991 / 1.045, replay 1.367 / 1.313 / 1.445 / 0.997 / 1.047, measured 1.389 / 1.329 / 1.461 / 1.014 / 1.072; E within 1%, tail share within 2 points. The model was right (V(8) came in 0.3 ms lower than its +2.3 ms); d / e are 1.5-2.5 points better than the replay.

**5. Exactness** (details in BENCH.md K11): greedy hybrid == plain greedy on E0 code / math and E6 a2 / b1 / c3 / c2 (192 tokens), E0 prose = the documented near-tie (margin 0.019); forced partial acceptance with patterns up to 8 rows ([8], [1], [5,2,7,1,8,3], [3,4,1,6,2,8,5,7]) and random 1..8 patterns with ring wrap-around: all equal. K7 sampling with delta tail proposals: 2e6-cycle chi-square on y0, (y0,y1), emitted length, tail rejections (p = 0.3-0.7), bonus, accepted-token identity: pass; both broken variants fail (p = 0 .. 1e-106, d_t re-emitted 263,185 / 57,361 times); real graph with a forced tail: first token chi2 p = 0.78. Existing tests (e2e parity, test_mtp, test_prefill, test_sampling) pass at tmax = 4 and 8; server `runner_check.py --pld` ALL OK, `smoke_test.py` 22/22 with the tail on.

**6. Memory**: +0.62 GiB (9-slot ring). `maxctx.py` (23.56 GiB card): bf16 KV 36864 ok / 40960 OOM -> with the ring 28672 ok / 30720 OOM; fp8 KV 65536 ok -> 53248 ok / 57344 OOM.

### Scorecard (prediction vs outcome)

| bet | outcome | verdict |
|---|---|---|
| V(5..8) at ctx 64 = 19.9 / 20.5 / 21.1 / 21.7 (+-0.7); V(1..4) within 2%; ctx 2K +0.2, 8K +0.8 | 20.07 / 20.62 / 21.17 / 21.61; V(1..4) 1.5% high; 2K +0.16..0.35, 8K +0.58..0.82 | HIT |
| P(V(8) < 25) 0.85, P(< 30) 0.97; V(8) - V(4) <= 3.0 ms (0.8) | 21.6 ms; 1.98-2.32 ms | HIT |
| growth ~55% GEMV / 35% DeltaNet / 10% rest; DeltaNet T=8 2.0 +- 0.3 ms; attention T=8 8K 0.95 +- 0.1 | DeltaNet 44% of the growth, 2.004 ms; attention 1.094 ms | split MISS (DeltaNet share higher), DeltaNet HIT, attention MISS (+0.05) |
| host gap 0.05-0.15 ms per cycle, device lookup not needed | < 0.01 ms | HIT (better) |
| E6 a / b / c = x1.25 / 1.20 / 1.30 (pooled 1.25, 80%: 1.10-1.40) | 1.389 / 1.329 / 1.461 (pooled 1.381) | MISS (all above; pooled at the edge of the interval) |
| d >= 0.985, e >= 1.00 (with an adaptive back-off) | 1.014, 1.072 (the adaptive back-off made both worse) | HIT, back-off clause MISS |
| E0: code +5..15%, prose +-1%, math +-2%, 8K +-3% | +0.7%, -1.0%, +0.7%, -2.0% | code MISS, others HIT |
| tail-cycle tokens / cycle within 5% of E6's simulation; sampling ratios within 0.05 of greedy | within 1%; 0.03 / 0.014 / 0.035 | HIT |
| max-model-len bf16 36864 -> ~28000, fp8 65536 -> ~49000 | 28672, 53248 | HIT / better |
| exactness: parity except near-ties, rollback up to 8 rows, chi-square p > 0.01, broken variant p < 1e-6 | all as predicted | HIT |
| default ON (65%) | server auto-on; library default tmax 4 (flag) | HIT |

## Conclusion

**The hybrid works as E6 modelled and the model's costs were right.** V(8) = 21.6 ms (+2.0..2.3 ms over V(4)), the host round trip is free (< 10 us), and on E6's workload the MTP chain + prompt-lookup tail decodes **x1.38 faster on code edits / log extraction / agent turns (240 vs 175 tok/s; 326 tok/s when the output copies the input)**, x1.01 on prose and x1.07 on math, with exact greedy output and exact sampling (x1.36 / 1.00 / 1.07 at T=1). The speed-up equals E6's in-sample figure, not its held-out one — but note the cell was selected on these same 32 tasks, so the a-c ratios are in-sample for the policy choice (E6's held-out estimate was x1.15); what is newly measured is the cost side (V(T), host gap, catch-up), which is what E6 could not verify. Copy-poor prompts are neutral to slightly negative at gate m = 3 (E0 prose -1%, 8K summary -2%, class f of six 6K-token summary prompts +2.1% overall with one task at -5%); m = 4 removes those losses and costs ~2% on a-c.

**Default decision.** The engine default stays `tmax = 4` (5-slot ring, no VRAM cost, T <= 4 only); the hybrid is a flag (`Engine(tmax=8)` + `enable_pld()`). The server's `--prompt-lookup` is **auto**: on iff `k = 3` and `--max-model-len` fits next to the 9-slot ring (28672 bf16 / 53248 fp8; the server default `--max-model-len` becomes 28672, 32768 with `--prompt-lookup 0`). Criteria (>= +5% on a-c, no loss on d / e) hold on E6's classes; the price is -8.2K tokens (bf16) / -12K (fp8) of context.

## Dead ends

- **Adaptive width** (online EMA of the tail's net gain = accepted tail tokens - 0.3, skip 8..64 cycles when < 0): implemented in the cycle loop, A/B'd on E6 (a x1.344, b x1.327, c x1.455, d x1.001, e x1.047 vs 1.389 / 1.329 / 1.461 / 1.014 / 1.072 without) and removed; a5 loses its gain (x1.04 vs x1.22) because a tail with 20% success is still worth 4 tokens per success. The replay grid (`results/policy_sim.log`: gate x tail length x two back-off families) never beats the plain cell on a-c, and by at most 0.6 points on d or e (a 1-3 token tail).
- **Device-side n-gram lookup / conditional graph nodes**: not built; the measured host round trip is < 10 us per cycle.
- **Attention T=6..8**: 5..6 warps per kv head (NT = 5 / 6, KS = 1) is latency bound (fp8 8K T=5 0.62 ms -> T=6 0.92 ms); KS = 2 would need 320-384 threads and a 170-204 register cap against 238 used. Costs 0.2-0.35 ms of V(8) at 8K; not pursued.
- **Lazy graph capture of the tail widths** failed with cudaErrorMemoryAllocation inside a request at max-model-len 28672 (the torch allocator holds the free memory): the widths are captured when the proposer is enabled (`smk_spec_pld_prewarm`).
- Gate sweep: m = 4 / 6 trade a-c (-2.5% / -4.5% in the replay) for prose / 8K safety; not adopted (m = 3 is E6's cell and the best on the target classes).
- First `trials-pld` run printed `p = nan` for the first tail position at top_p .95 (a single residual outcome, df = 0) and still passed by luck of `min` over a list with nan: fixed to treat df = 0 as "only d_t-never-emitted is testable".

## Follow-ups

- Context-dependent gate (agreement >= 4 above ~4K tokens, or a match-reliability estimate from the MTP draft confidences) to remove the long-context prose loss without giving up a-c; needs a held-out workload (the E6 cell is in-sample here).
- Wider attention split for T = 6..8 (0.2-0.35 ms at 8K), and T=5/6 widths are never used with Lt = 4 (tails are 3 or 4 tokens): a cost-aware tail length (shorter tails for low agreement) is the only use.
- Two-candidate / tree tails are impossible with DeltaNet state without per-branch copies; a cheaper 9-slot ring (bf16 snapshots of rows >= 4) would give the context back.
