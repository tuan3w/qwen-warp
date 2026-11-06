# Ideas to explore

Open questions we found and did not finish. **This is a list of ideas, not a roadmap**: nothing here is promised, scheduled or in progress. If one interests you, take it; if you do,
open an issue so others know. Back to the [README](../README.md).

| Idea | Where it stands | Effort to settle it |
|---|---|---|
| [What is left of the 4-bit cache gap](#what-is-left-of-the-4-bit-cache-gap) | the rotated cache is in; the remaining error is mostly key logits; two speed cases and one sampling difference are open | per-layer and K/V byte allocation, a 5-row verify kernel, a repeat-rate control run |
| [Why the 4-bit cache missed a needle at 100k](#why-the-4-bit-cache-missed-a-needle-at-100k) | the previous format missed it; the rotated one found it 4 of 4, cause of the original miss unknown | more needle runs across positions, a per-layer look |
| [Multi-turn agentic task success per KV dtype](#multi-turn-agentic-task-success-per-kv-dtype) | agent tasks not measured; the tool-call rate of one small schema was ([Usage](USAGE.md#tool-calls)) | an agent benchmark run three times |
| [A non-uniform weight codebook](#a-non-uniform-weight-codebook) | helps accuracy on a 4B model; the extra bytes alone bound the 27B step at about +17% | a kernel; likely a dead end at this size |
| [A better quantized model](#a-better-quantized-model-if-one-appears) | nothing to measure yet | kernel work if the format changes |
| [Faster long prefill](#faster-long-prefill) | 963 tok/s at 262k | profiling first |
| [A block drafter (DFlash2-style)](#a-block-drafter-dflash2-style) | HyperQwen's DFlash2 decoded 1.5x faster on short code (5.8 against 3.5 tokens per step) | train or port a drafter; the verify path already handles 8 rows |
| [Repeating the HyperQwen reference measurement](#repeating-the-hyperqwen-reference-measurement) | one pinned build, two same-protocol runs (2026-11-03 and 2026-11-04) that differ by 5% to 22% | a newer build, repeats per cell, a second power limit |
| [A thinking token budget](#a-thinking-token-budget) | `thinking_token_budget` is HTTP 400; `reasoning_effort` is only a prompt hint; a design sketch fits the existing processors | a kernel rule, an exactness test, a quality measurement at cut-off budgets |
| [Log-probabilities](#log-probabilities-through-an-extra-bf16-pass) | HTTP 400; the head returns exact logits for 64 candidates only | an extra bf16 pass for requests that ask, and its cost |
| [bad_words and structured output](#bad_words-and-structured-output) | HTTP 400; each needs device-side state per verify row | a matcher or a grammar automaton in the cycle |
| [Processors on the draft head](#logit-processors-on-the-draft-head) | the draft proposals ignore them; -15% tok/s with five of them together at T 0.7 | apply them to the draft logits, keep the exactness test green |
| [Why bf16 KV is below a reference service on tool calls](#why-bf16-kv-is-below-a-reference-service-on-tool-calls) | 93% against 98%, cause not isolated | a controlled comparison on identical contexts |
| [Upgrading vLLM](#upgrading-vllm) | pinned to 0.30.0; 0.33.0 exists; the client and the tool-schema hook use internals | a scratch environment and `make test` |

Out of scope: other models, other GPUs, batching. (If you try another GPU, the build takes `ARCH=-arch=sm_XX`, but nothing outside sm_86 has been tried.)

## What is left of the 4-bit cache gap

The rotated format ([KV cache](KV_CACHE.md)) cut the KL divergence from the bf16 cache from 0.51% to 0.069% at the same bytes. What was measured and not finished:

- **Where the remaining error is.** With the ring and sinks exact, making K exact removes 76% of the removable loss and V 25%; making both exact leaves 0.0149% (fp16 numerics). Layer groups and distance bands were not attributed (switches were written, not run). A cheaper split of the 4.125 bits between K and V, or per layer, is untested; so is an 8-bit middle tier between the exact ring and the 4-bit body (about 1.4% of decode bandwidth per 4k bf16 tokens, an estimate).
- **Speed cases.** A one-row step is still about 1% to 2% over the previous reader at every context length (the split kernel's own dequantisation, 5.25 against about 2 instructions per fp16 pair, and the write kernel at 5.8 against 3.7 us per call), and 32k with the prompt-lookup tail is about 3% below the previous format in two runs per cell (possibly the old tree's high draw). A 5-row verify step (3 drafts, the verified token and one tail token) needs all four query tiles in one group, which spills registers at 8 warps; its time was last measured before the ring change.
- **Sampled repetition.** On a 61-token loop prompt at temperature 0.7 the rotated cache gives a distinct-token ratio of 0.24 against 0.29 to 0.33 for bf16, 0.33 to 0.35 for the previous format and 0.39 for fp8. It is caused by the 256-token exact ring (rings of 16 and 64 tokens give 0.37 and 0.39; rounding the ring to bf16 changes nothing) and does not appear on four other prompts. Why an exact window that covers the whole prompt changes sampled loops is not known; useful controls: more loop-prone prompts of different lengths, and the ring at 128 tokens.
- **Not re-measured on this format:** coding pass@1 over 283 tasks, repository question-answering, agent-sized tool sets, more needle depths and lengths. (Perplexity at 98k was: +0.056% over bf16.)
- **Other literature ideas not tried:** a learned or searched Hadamard sign vector (adaptive transforms gave at most 0.04 perplexity at 4 bits in the papers read), lattice quantisation (at most about 2.3x lower mean squared error in theory and no shown gain at 4 bits), extra exact slots for outlier tokens.

## Why the 4-bit cache missed a needle at 100k

The previous 4-bit format missed a planted needle at 100k tokens in the speed sweep of 2026-11-03 and found it at 32k, 64k, 130k and 190k; on 2026-11-04 both 4-bit arms (tail off and on) missed it at 100k again and found it at every other length (16k to 262k). In a separate test it got 7 of 8 needles at about 100k
(fp8: 8 of 8, and fp8 found it at every length it served). On 2026-11-05 the rotated format found it in all 4 runs at 100k while the previous one missed all 4. In a chat-format probe at 64k (8 depths) all formats found the passphrase every time. So the miss belongs to the previous format; why one position range was affected is still unknown. **To settle it:** many more needle prompts across positions and lengths with the rotated format, with fp8 as the control.

## Multi-turn agentic task success per KV dtype

The cache is aimed at long agent sessions, but quality was only measured with single-shot coding and retrieval proxies and, for tool calls, one small schema in single-turn requests ([Performance](PERFORMANCE.md#tool-calls)).
**To settle it:** run an edit-and-test agent benchmark with bf16, fp8 and 4-bit caches, several runs each, and compare task success and the number of tool-call errors; include a six-tool coding-agent set and second turns, which were not measured.

## A non-uniform weight codebook

On Qwen3-4B, replacing the uniform 4-bit grid by a per-group 16-level codebook gave +2.8 points of next-token top-1 fidelity (symmetric version; +5.0 points with a free grid). On this 27B it is
probably not worth it: a codebook adds about 0.25 bytes per weight, which is +26% weight bytes (11.68 to 14.72 GiB for the quantised linears), so even a kernel with no extra compute would
lengthen the 18.4 ms decode step by about 3 ms (+17%), and the extra 1.85 GiB would cut the longest context. [INFERENCE: byte counts from the checkpoint headers, step time from the measured memory-bound GEMVs,
not a built kernel.] An earlier fused prototype ran about 2x slower than int4.

## A better quantized model, if one appears

The W4 kernels assume uniform symmetric int4 with groups of 128 (`uint32 [N][K/8]` plus bf16 scales `[N][K/128]`, see [Architecture](ARCHITECTURE.md#conventions-the-kernels-share)).
A different weight format means kernel work, and a different model means re-checking the MTP head and the DeltaNet layout.

## A block drafter (DFlash2-style)

Done as an experiment in 0.6.0 ([Architecture](ARCHITECTURE.md#dflash2-block-drafter-experimental-greedy-only), [Performance, section 5](PERFORMANCE.md#5-dflash2-block-drafter-experimental)): the drafter runs inside the cycle graph, 4.5 tokens per cycle against 3.0 for MTP, level with HyperQwen's DFlash2 in the same session. Wired into the server for greedy requests as `--drafter dflash2` ([Usage](USAGE.md#the-dflash2-drafter---drafter-dflash2-experimental)): +50% decode on code, +38% on math, +34% on 8 chat prompts, -4% on prose, **-31% on copy-heavy agent turns** where the prompt-lookup tail copies 6.8 tokens per cycle.
**What is not known / not built:** (1) an adaptive route: a request that will quote its prompt wants the MTP route and the first token cannot tell; a cheap predictor (the prompt's tool-result turns, the lineage's last acceptance, a short probe of both drafters) is untested. (2) A ring buffer for the drafter's KV cache (20 KiB per token today, absolute position; `--profile long` does not fit with the drafter: 1.15 GiB + 20 KiB per token on top of the 4-bit engine, about 125k tokens at most on a 24 GB card; a ring would cap the drafter at 42 MiB and needs a change in 3 drafter kernels and in the reset / read sizes). (3) Sampling (the drafts are greedy; the sampled and the processor requests use the MTP path). (4) The window is carried through the host prefix cache only up to the entry's last snapshot (a resume at an earlier snapshot, or with a prompt shorter than the stored window, falls back to the MTP path), and a drafter window is not part of the device snapshots.
**Where the next gain is:** tokens per cycle (4.5 on average, 0.87 acceptance at the first draft position down to 0.26 at the seventh). The drafter's top candidate is right 80% of the time and the true token is among its top 16 in 96%: verifying a second candidate needs per-branch DeltaNet state.

## Faster long prefill

Cold prefill is 1,888 tok/s at 16k, 1,094 at 190k and 963 at 262k (re-measured 2026-11-04), so a 190k prompt takes about 3 minutes; later turns reuse the cached prefix. It is within about 10% of the reference build in the
[comparison](PERFORMANCE.md#4-hyperqwen-as-a-reference-point), so a gain would come from the attention and DeltaNet prefill path (`engine/prefill.py`). No profile has been taken.

## Repeating the HyperQwen reference measurement

The reference numbers describe one HyperQwen build. It was measured twice with the same protocol, and its decode speed differed by 5% to 22% between the two days (the cause was not isolated), so one run per cell is a rough number. A repeat with several runs per cell,
a newer build, its DFlash2 preset if it starts, and a second power limit would be a better reference. This is a measurement exercise, not a contest.

## A thinking token budget

`thinking_token_budget` (a hard cap on the thinking length, then a forced `</think>`) returns HTTP 400 today. `reasoning_effort` does not replace it: it is only a prompt hint, and on a stock vLLM server of the same model the median reasoning length
over 18 requests per setting was 0 (`none`, thinking off), 279 (`low`), 269 (`medium`), 290 (`xhigh`) and 256 (unset) tokens, with no measurable shortening ([Usage](USAGE.md#thinking-and-reasoning-output)).

**Sketch.** It fits the per-request processor state (`engine/src/proc.h`): a generated-token counter and an "end of thinking seen" flag, both updated where `proc_note` updates the penalty counts, and a rule in `proc_logit`:
"counter >= N and the flag is not set: restrict the row to the forced token". A forced multi-token tail would index a small device array by `counter - N`. Under speculative decoding, verify row t uses `counter + t` and scans the draft tokens
for the flag exactly as the penalty counts do. Not implemented, not measured.

**To settle it.** Implement the rule, extend `tests/test_spec_procs.py` (the emitted stream must match the exact autoregressive joint, with a broken variant that must fail), then measure answer quality at cut-off budgets.

## Log-probabilities through an extra bf16 pass

`logprobs`, `prompt_logprobs` and `logprob_token_ids` return HTTP 400. The quantised head gives exact (bf16-rescored) logits only for its top-64 candidates; the other logits are int4 approximations, so a full log-softmax is not available.
**Idea:** for requests that ask for log-probabilities, run an extra bf16 pass over the 248k vocabulary per row. [INFERENCE] It would probably sit outside the CUDA-graph cycle so that other requests keep their speed; its cost was not estimated or measured.
**To settle it.** Build the pass, check it against a torch reference, and measure the per-token cost for a request with `logprobs` on.

## bad_words and structured output

Both return HTTP 400. `bad_words` with multi-token sequences needs a matcher over the generated history inside the verify rows, draft tokens included (single-token words could reuse the `logit_bias` and `allowed_token_ids` machinery).
Structured output needs a grammar state machine per token and per verify row on the device; a host round trip per token would defeat the CUDA-graph cycle. Neither was prototyped. **To settle it.** For `bad_words`, a matcher
in `engine/src/proc.cu` and a test like `tests/test_spec_procs.py`; for structured output, decide the grammar subset first (JSON schema only?) and measure the per-token cost.

## Logit processors on the draft head

The draft proposals ignore the request's processors. That keeps the acceptance exact for any proposal, but it lowers the number of accepted tokens: at T 0.7 and top_k 20, five processors together took the server log from 3.16 to 2.75 tokens per cycle
(160 to 136 tok/s, -15%); `repetition_penalty` 1.1 alone: 2.90 (150 tok/s); the other single ones cost little (8 prompts x 200 tokens). **Idea:** apply the processors to the draft head's logits too, which should recover part of that
[INFERENCE: not tried]. **To settle it.** Do it, keep `tests/test_spec_procs.py` and `tests/test_proc_model.py` green, and repeat the acceptance measurement.

## Why bf16 KV is below a reference service on tool calls

A stock vLLM service behind a gateway (its rendered prompt had `function.parameters` key-sorted) gave a well-formed tool call in 98.3% of 400 requests (pooled over four temperatures); the bf16-KV engine with sorted keys gave 93% (139/150). On identical contexts the exact
log-probability gap between `<` and `<|im_start|>` after `<tool_call>` is 13.7 nats on the service (minimum 8.6, n = 70) and 5.7 on the engine (minimum -1.6; fp8 5.8; fast against chunked prefill differ by 0.4 only): the engine's hidden state is less sharp there.
The sampler is exact ([Performance](PERFORMANCE.md#tool-calls)), so the gap is upstream of it. Candidates, none tested: the W4A16 weights of this checkpoint against the service's unknown weights, and KV or activation numerics.
**To settle it.** Compare the same contexts through a path whose precision you can change (the same checkpoint on stock vLLM, an unquantised head, higher-precision accumulation) and see which one moves the gap.

## Upgrading vLLM

The environment pins `vllm==0.30.0` (`pyproject.toml`, `uv.lock`); 0.33.0 exists (PyPI, 2026-11-04). The engine-core client (`server/smk_core_client.py`: `EngineCoreClient.make_async_mp_client`, `SamplingParams._validate_spec_decode`) and the
tool-schema hook (`server/tool_schema.py`: `vllm.renderers.hf.safe_apply_chat_template`) depend on vLLM internals of 0.30.0; no other version was tried. **To settle it.** In a scratch environment, bump vLLM (and whatever torch it needs), run `make test`, and
fix what breaks.

## How to help

| You can | How |
|---|---|
| Report what happened on your machine | open an issue with `nvidia-smi`, your command and the log tail |
| Take one of the ideas above | say so in an issue, then send a pull request with the command, hardware and a single-run or repeated measurement (see [CONTRIBUTING](../CONTRIBUTING.md)) |
| Check a number | re-run `make test` or the measurement and tell us where it differs |
