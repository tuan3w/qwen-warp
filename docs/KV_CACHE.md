# The 4-bit KV cache

`--kv-cache-dtype int4_per_token_head` (and `--profile long`) stores the attention cache in 4 bits per element. This page describes the format, what each part
of it buys, how that was measured, what it costs, and what is not known. Measured on one RTX 3090 with Swift-1.5-Qwen3.8-27B-INT4 on 2026-11-04/05; one model, one card.

## The format

Per token and KV head, separately for K and V (256 elements):

| Step | What | Why |
|---|---|---|
| 1. Rotate | Walsh-Hadamard transform over the 256-vector (`y = H x / 16`, orthonormal, its own inverse). The query is rotated the same way, so scores are unchanged; the attention output is un-rotated once per layer | makes the coordinates near-Gaussian: one outlier channel no longer sets the scale for everything |
| 2. Quantise | 16 symmetric levels with a slightly stretched ("companded") grid, `L(n) = u (1 + 0.003 u^2)`, `u = 2n - 15`; 4 scales per vector (groups of 64 rotated coefficients), each an 8-bit logarithmic code | the grid fits a Gaussian within about 3% of the optimal 16-level quantiser's error, so there is no gain left in its shape |
| 3. Keep the first 4 tokens exact | fp16 rows (rotated basis) in a small fixed buffer | the attention sinks carry a large share of the attention mass |
| 4. Keep the last 496 tokens exact | fp16 rows in a ring buffer (512 slots); the split kernel stops 496 keys before the end of the context and attends the ring as 31 extra fp16 tiles (no dequantisation); the reduce kernel only merges the splits and un-rotates | the newest tokens are the ones attention uses most |
| 5. Decoder gain | the reader multiplies every decoded scale by 1.0075 (`1 / 0.9926`) | the writer's error-minimising scale makes each group about 0.74% too short, so quantised keys get logits slightly too small next to the exact ring keys; the constant comes from the quantiser's statistics on Gaussian groups, not from any text |

Bytes per token are unchanged: 16,896 bytes (16.5 KiB, 4.125 bits per element), measured on the device for the old and the new format. The exact sink and ring are fixed
buffers, not per-token: **+34 MiB of device memory** (measured with `torch.cuda.mem_get_info`, same at 8k and 64k tokens) and **+32 MB of pinned host RAM per prefix
snapshot** (the ring and sink rows are saved and restored with every snapshot, default 4 slots). The fixed device term is in the start-up memory model (`VRAM_KV_FIXED_MIB`).
The decode writer and the prefill writer produce the same cache; a resumed generation from a prefix snapshot equals a cold prefill (tests below).

## How it was measured

**dKL, not perplexity.** Perplexity of one 32k-token text has a standard error of about 0.07 percentage points, which is larger than most of the differences that matter
here (relative noise about 25% on a small gap). The fidelity numbers below are the mean per-token KL divergence from the bf16-cache distribution to the int4 one, over
the bf16 top-64 tokens, teacher-forced through the real kernels: about 4% relative noise, and it does not depend on which token the text happens to contain. Perplexity
is reported next to it. KL tracks decision flips in the literature (Krishnan et al., [2407.09141](https://arxiv.org/abs/2407.09141); Felicio, [2610.00694](https://arxiv.org/abs/2610.00694): flip rate about 1.05 x total variation), but the dKL here (0.07%) sits far below
the level where ordinary benchmarks resolve differences ([2606.19558](https://arxiv.org/abs/2606.19558)), so average-case benchmarks will not show it. The effect shows in the tails below.

**Fidelity** (32k-token code text, real engine; "held-out" is a second 32k text in other languages; each step was kept only if the held-out text did not regress):

| Format | dKL vs bf16 (dev text) | held-out text | what changed |
|---|---|---|---|
| previous int4 (asymmetric, per token+head) | 0.512% | control | |
| + 4 symmetric scale groups | 0.431% | | |
| + rotation, log2 scales, companded grid | 0.196% | 0.241% | |
| + exact sink (4) and ring (256) | 0.0896% | 0.1054% | |
| **+ decoder gain 1.0075 (this version)** | **0.0691%** (+-0.0031) | 5.67x of its control, in score ratio | dPPL +0.025% (+-0.025) on the dev text |

All-exact K and V (the floor, with ring and sink already exact): 0.0149%. With only K exact: 0.0330%; only V exact: 0.0710%. So after the ring and the sinks, **76% of the remaining loss
is the key logits** and 25% the values (they add to within 1%); it is flat in position beyond 256 tokens. A gain of 1.015 (twice the right value) is worse than none (0.0918%): the optimum is sharp and
the gain should not be fitted per text. A logit offset for the Jensen bias of noisy keys on top of the gain gave 0.0685% against 0.0701% and is not included.

**Perplexity at 98k** (code from several repositories, 98,304 tokens, teacher-forced through the engine, paired per token against the earlier bf16, fp8 and previous-int4 runs on the same text): this format **+0.056% (+-0.017)** over bf16, against +0.26% (+-0.05) for the previous format and +0.03% for fp8; it is 0.20% lower than the previous format in the paired comparison. The gap does not grow with position (by bin, 0-4k / 4-8k / 8-16k / 16-32k / 32-64k / 64-98k: +0.00 / -0.06 / -0.01 / +0.10 / +0.09 / +0.04%).

**End to end, through the server** (`--profile long`, thinking on, default sampling; old = the previous 4-bit format of 0.3.0, same prompts):

| Measure | old | this version |
|---|---|---|
| Tool-call rate, minimal one-parameter schema in the usual key order (`--sort-tool-schema off`) | 46% [37, 56] (n = 100) | **92% [87, 95]** (n = 200, first build); **96% [90, 98]** (n = 100, final build) |
| same, with the default sorted rendering, n = 100 | 94% [88, 97] | 88% [80, 93] (within noise of the old value) |
| Needle in a haystack at 100k tokens, through the server | missed in every run (4 of 4 in the first session, and again in every old run of the final speed session) | found in every run |
| Needle probe at 64k in chat format (8 depths, passphrase negative log-likelihood vs the fp8 cache) | 0.124 nats vs 0.086 for fp8 (+0.038, 1.6 standard errors) | 0.079 (-0.006, -0.6 s.e.): indistinguishable from fp8 |

bf16 gave 91% on the type-first schema in the earlier measurement (0.3.0). Wilson 95% intervals. The server still sorts tool schemas by default; with this cache the rendering order no longer matters for that schema.

## The exact ring: 256 against 496 rows

The ring has 512 slots (the memory is the same either way), but the first version attended only the newest 256 of them. A step of up to 8 rows writes its rows over the slots of the positions 512 back, and speculative decoding may accept only the first of them, so the next step can start one row later: the ring may be attended for at most `RING_R - 8` rows, rounded down to the reader's 16-row tiles, which is 496 (`RING_W` in `engine/src/kvrot.cuh`, with a `static_assert`).

Measured on the real engine (a 16k-token text of Python files and a 32k-token text of headers, prose and C++/JS/Rust source, both from installed packages and not in the repository; teacher-forced through the decode path against the bf16 cache; one RTX 3090, 2026-11-14; `RING_W` is the only change; the method is `scripts/kv_dkl.py`: `ref` once with the bf16 cache, then `run --ref ref.pt --build <engine build dir>` per build):

| `RING_W` | dKL dev | dKL held-out | p99 per-token KL, dev / held-out |
|---|---|---|---|
| 256 (before) | 0.0660% | 0.0509% | 1.142% / 0.992% |
| 384 | 0.0552% | 0.0448% | 0.913% / 0.833% |
| **496 (now)** | **0.0504%** | **0.0431%** | **0.884% / 0.747%** |

That is 24% (dev) and 15% (held-out) less dKL and about a quarter less in the p99 tail, at the same bytes per token and the same fixed memory (a build with `RING_W` = 512 gave 0.0504% / 0.0425%, but it is not safe with rejected speculative rows, see above). Decode time per step (T = 1 and T = 8 rows after a fast prefill to the stated context, old and new build back to back, median of 5 x 40 steps; the harness is not in the repository) was equal within 0.1% at 2k, 32k and 190k context. The extra 15 tiles read about 1 MB per layer, which is 0.3% of a step by bandwidth, so this is expected.

Also run with the 496-row ring: `tests/test_tool_call_rate.py --profile long --n 100 --sort off` gave 96/100 (Wilson 95% [0.90, 0.98]), the same as the 96% above for the 256-row ring; a planted-passphrase probe (a sentence inserted into repeated code, one run per cell, greedy, same text for both builds) found it in 4 of 4 depths at 100k and 2 of 2 at 190k with both rings, with identical output. That probe is easy (one needle, one run) and says only that the longer ring did not break retrieval.

Not measured: sampled repetition with the 496-row ring (the result below depends on the ring length: it is the exact ring that made the 61-token loop prompt differ from bf16), coding pass@1, a multi-needle or multi-turn agent benchmark, and the vision, startup-configuration and server-batch tests of `make test` (10 of the test files were run: `test_kv_dtypes`, `test_spec_procs`, `test_prefill_plan`, `test_group_prefill`, `test_prefix_store`, `test_lineage_cache`, `test_batch_engine`, `test_server_smoke` in both profiles, `test_server_seed`, `test_tool_call_rate`; all passed).

## What it costs

Decode speed through the server on the final build (tokens/s, frozen prompts, mean of two runs per cell, runs interleaved old/previous-port/new; the card drifts about 5% inside a session, and the same old tree measured against itself at the start and end of the session differed by up to 6%, so differences under about 5% are not resolved at this number of runs):

| Arm | Prompt | previous format | this version | difference |
|---|---|---|---|---|
| prompt-lookup tail on | 32k | 130.5 | 126.7 | -2.9% |
| | 100k | 115.0 | 123.3 | +7.2% |
| | 190k | 98.6 | 102.3 | +3.8% |
| tail off | 32k | 129.0 | 128.5 | -0.4% |
| | 100k | 111.6 | 124.6 | +11.6% |
| | 190k | 97.2 | 96.9 | -0.3% |

Short prompts (code / prose / math), tail on: 179 / 111 / 179 against 180 / 110 / 183 before; tail off: 178 / 114 / 172 against 176 / 112 / 167. Prefill is unchanged (1,785 to 1,860 tok/s at 32k, 1,085 to 1,128 at 190k).

In short: level with the previous format on short prompts, at 32k (tail off) and at 190k (tail off); ahead at 100k, and at 190k with the tail on; about 3% behind at 32k with the tail on. This was not the case at first. The first port was 15% to 20% slower at 190k because the attention kernel dequantised the same tile once per query tile in the multi-row verify steps; a verify kernel that dequantises each tile once per group fixed that. A smaller cost remained at every context length, about 0.9 ms per speculative cycle (3.8% of kernel time at 32k): the exact-ring attention in the reduce kernel (33 us against 4.6 us per call) and the write kernel (15 us against 3.5 us). It was removed by attending the ring inside the split kernel and by an 8-warp write kernel (reduce 15 to 3.6 us and write 13.6 to 5.8 us per layer call at 32k); a plain one-row step is still about 1% to 2% over the previous reader. Speculative acceptance (tokens per cycle) of the new format tracks bf16 on four prompts (-3.8%, 0, +0.9%, 0), so the remaining differences are kernel time and text, not drafting.

## What is not known

- **Sampled repetition on one prompt.** On a 61-token loop prompt at temperature 0.7 (320 samples), the share of distinct tokens is 0.24 with this cache against 0.29 to 0.33 with bf16, 0.33 to 0.35 with the previous int4 cache and 0.39 with fp8; on four other prompts it equals bf16 (a loop-prone list prompt 0.173 against 0.163, two ordinary chat prompts 0.624 against 0.624 and 0.745 against 0.742, a degenerate prompt identical). It comes from the exact 256-token ring: with a ring of 64 or 16 tokens the ratio returns to 0.39 and 0.37, and rounding the ring to bf16 changes nothing (0.242). Why a ring window that covers a whole 61-token prompt changes sampled loops is not known. Teacher-forced entropy and maximum probability match bf16 closely, so it is not a per-position sharpening. `tests/test_sampling_params.py` passed 48/48 on both runs of the final build; it failed one statistical repetition check in 2 of 4 runs on the first build.
- **Not re-measured on this format:** coding pass@1 over 283 tasks and repo question-answering at 64k and 100k (the numbers in [PERFORMANCE.md](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache) are for the previous format). Perplexity at 98k was measured (above).
- The speed result is for one model and one card, two runs per cell; the remaining 3% at 32k with the tail on may be real or the old tree's high draw.
- One model, one card, one set of prompts; dKL was measured on two 32k texts. Needle tests: 4 runs and 8 depths.

## Where the ideas come from

None of the pieces is new, and the measurements are the contribution here. Rotating before quantising the KV cache: QuaRot ([2404.00456](https://arxiv.org/abs/2404.00456); its ablation gives 5.51 against 5.47 perplexity for 4-bit K and V). An exact recent window: KIVI ([2402.02750](https://arxiv.org/abs/2402.02750), a full-precision window of 128 recent tokens). First token(s) in full precision: KVQuant ([2401.18079](https://arxiv.org/abs/2401.18079)); StreamingLLM ([2309.17453](https://arxiv.org/abs/2309.17453)) shows four initial tokens are needed. Bias in quantised keys and how to correct it: TurboQuant ([2504.19874](https://arxiv.org/abs/2504.19874), the MSE-optimal quantiser is biased for inner products) and HyperQuant ([2606.23406](https://arxiv.org/abs/2606.23406), random rotation plus subtractive dither; it reports all variants within 0.014 perplexity at 4 bits). A decoder-side constant gain derived from the quantiser's own shrink is not in HyperQuant's paper (its calibration sets an SNR-targeting scale, appendix B), but this was only checked in that paper, and no novelty is claimed.

## Checking it here

`make test` covers the format: `tests/test_kv_dtypes.py` (perplexity of bf16, fp8 and 4-bit caches; fails if 4-bit is more than 3% worse; measured +0.05% to +0.10% on its text), `tests/test_host_tables.py`, `tests/test_server_smoke.py`
(greedy parity with the direct engine; a turn resumed from a snapshot equals a cold re-prefill), `tests/test_server_seed.py` and `tests/test_startup_configs.py` (memory edges, including literal near-limit prompts at 262,144 tokens).
`tests/test_tool_call_rate.py` measures the tool-call rate on a live server.
