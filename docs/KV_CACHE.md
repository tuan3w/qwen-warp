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
| 4. Keep the last 256 tokens exact | fp16 rows in a ring buffer; the split kernels stop 256 keys before the end and the reduce kernel attends the ring exactly | the newest tokens are the ones attention uses most |
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
| Tool-call rate, minimal one-parameter schema in the usual key order (`--sort-tool-schema off`), n = 200 / 100 | 46% [37, 56] | **92% [87, 95]** |
| same, with the default sorted rendering, n = 100 | 94% [88, 97] | 88% [80, 93] (within noise of the old value) |
| Needle in a haystack at 100k tokens, 4 runs per format, both speed arms | missed 4 of 4 | found 4 of 4 |
| Needle probe at 64k in chat format (8 depths, passphrase negative log-likelihood vs the fp8 cache) | 0.124 nats vs 0.086 for fp8 (+0.038, 1.6 standard errors) | 0.079 (-0.006, -0.6 s.e.): indistinguishable from fp8 |

bf16 gave 91% on the type-first schema in the earlier measurement (0.3.0). Wilson 95% intervals. The server still sorts tool schemas by default; with this cache the rendering order no longer matters for that schema.

## What it costs

Decode speed through the server (tokens/s, frozen prompts, same session, runs interleaved old/new; the GPU drifts about 5% inside a session, so only repeated pairs mean anything):

| Arm | 32k | 100k | 190k |
|---|---|---|---|
| prompt-lookup tail on: old | 124.0 | 109.0 | 96.4 |
| prompt-lookup tail on: this version | 121.2 | 112.4 | 100.5 |
| tail off: old | 126.3 | 109.3 | 96.0 |
| tail off: this version | 117.7 | 114.6 | 96.4 |

Short prompts (code / prose / math): 167 / 102 / 167 with the tail and 167 / 108 / 164 without, against 172 / 106 / 176 and 171 / 108 / 162 before. Prefill is unchanged (about 1,780 tok/s at 32k, 1,090 at 190k).

In short: level with the old format at 100k and 190k, 2% to 7% slower at 32k and 2% to 4% on short prompts. A plain one-token step at 160k context is about 3% slower; the multi-token verify steps of speculative decoding
were the problem at first (the attention kernel dequantised the same tile once per query tile; the first port was 15% to 20% slower at 190k) and are now level with, or below, the old reader for 2, 3, 4, 6, 7 and 8 rows (kernel `attn_split_i4_kernel`, one dequantisation per tile and group). A step of 5 rows (three drafts plus the verified token and one
prompt-lookup tail token) is still about 12% slower at 160k.

## What is not known

- **Sampled repetition.** On a 61-token loop prompt at temperature 0.7 (40 batches of 8), the share of distinct tokens is 0.240 with this cache against 0.330 with bf16 and 0.350 with the old int4 cache (0.318 against 0.433 with no speculation), i.e. the sampled text repeats more. Teacher-forced entropy and maximum probability match bf16 closely (top-64 entropy 0.539 / 1.959 / 1.184 against 0.537 / 1.958 / 1.185 over positions 0-60 / 60-260 / 260-1400), so it is not a per-position sharpening. The cause is not known. `tests/test_sampling_params.py` passed 4 of 4 runs.
- **Not re-measured on this format:** coding pass@1 over 283 tasks and repo question-answering at 64k and 100k (the numbers in [PERFORMANCE.md](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache) are for the previous format). Perplexity at 98k was measured (above).
- The 2% to 7% speed difference at 32k was not traced: the acceptance rate of speculative drafts under the two formats was not compared.
- One model, one card, one set of prompts; dKL was measured on two 32k texts. Needle tests: 4 runs and 8 depths.

## Where the ideas come from

None of the pieces is new, and the measurements are the contribution here. Rotating before quantising the KV cache: QuaRot ([2404.00456](https://arxiv.org/abs/2404.00456); its ablation gives 5.51 against 5.47 perplexity for 4-bit K and V). An exact recent window: KIVI ([2402.02750](https://arxiv.org/abs/2402.02750), a full-precision window of 128 recent tokens). First token(s) in full precision: KVQuant ([2401.18079](https://arxiv.org/abs/2401.18079)); StreamingLLM ([2309.17453](https://arxiv.org/abs/2309.17453)) shows four initial tokens are needed. Bias in quantised keys and how to correct it: TurboQuant ([2504.19874](https://arxiv.org/abs/2504.19874), the MSE-optimal quantiser is biased for inner products) and HyperQuant ([2606.23406](https://arxiv.org/abs/2606.23406), random rotation plus subtractive dither; it reports all variants within 0.014 perplexity at 4 bits). A decoder-side constant gain derived from the quantiser's own shrink is not in HyperQuant's paper (its calibration sets an SNR-targeting scale, appendix B), but this was only checked in that paper, and no novelty is claimed.

## Checking it here

`make test` covers the format: `tests/test_kv_dtypes.py` (perplexity of bf16, fp8 and 4-bit caches; fails if 4-bit is more than 3% worse; measured +0.05% to +0.10% on its text), `tests/test_host_tables.py`, `tests/test_server_smoke.py`
(greedy parity with the direct engine; a turn resumed from a snapshot equals a cold re-prefill), `tests/test_server_seed.py` and `tests/test_startup_configs.py` (memory edges, including literal near-limit prompts at 262,144 tokens).
`tests/test_tool_call_rate.py` measures the tool-call rate on a live server.
