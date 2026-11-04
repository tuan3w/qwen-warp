# Ideas to explore

Open questions we found and did not finish. **This is a list of ideas, not a roadmap**: nothing here is promised, scheduled or in progress. If one interests you, take it; if you do,
open an issue so others know. Back to the [README](../README.md).

| Idea | Where it stands | Effort to settle it |
|---|---|---|
| [A rotated 4-bit KV cache](#a-rotated-4-bit-kv-cache) | measured as bf16 fake-quant: about 2x lower perplexity cost on this 27B; no kernel, no speed number | a decode kernel and a speed measurement |
| [Why the 4-bit cache missed a needle at 100k](#why-the-4-bit-cache-missed-a-needle-at-100k) | one miss in two experiments, cause unknown | more needle runs, a per-layer look |
| [Multi-turn agentic task success per KV dtype](#multi-turn-agentic-task-success-per-kv-dtype) | agent tasks not measured; the tool-call rate of one small schema was ([Usage](USAGE.md#tool-calls)) | an agent benchmark run three times |
| [A non-uniform weight codebook](#a-non-uniform-weight-codebook) | helps accuracy on a 4B model; the extra bytes alone bound the 27B step at about +17% | a kernel; likely a dead end at this size |
| [A better quantized model](#a-better-quantized-model-if-one-appears) | nothing to measure yet | kernel work if the format changes |
| [Faster long prefill](#faster-long-prefill) | 963 tok/s at 262k | profiling first |
| [Repeating the HyperQwen reference measurement](#repeating-the-hyperqwen-reference-measurement) | one pinned build, single runs | a few hours of runs |
| [A thinking token budget](#a-thinking-token-budget) | `thinking_token_budget` is HTTP 400; `reasoning_effort` is only a prompt hint; a design sketch fits the existing processors | a kernel rule, an exactness test, a quality measurement at cut-off budgets |
| [Log-probabilities](#log-probabilities-through-an-extra-bf16-pass) | HTTP 400; the head returns exact logits for 64 candidates only | an extra bf16 pass for requests that ask, and its cost |
| [bad_words and structured output](#bad_words-and-structured-output) | HTTP 400; each needs device-side state per verify row | a matcher or a grammar automaton in the cycle |
| [Processors on the draft head](#logit-processors-on-the-draft-head) | the draft proposals ignore them; -15% tok/s with five of them together at T 0.7 | apply them to the draft logits, keep the exactness test green |
| [Why bf16 KV is below a reference service on tool calls](#why-bf16-kv-is-below-a-reference-service-on-tool-calls) | 93% against 98%, cause not isolated | a controlled comparison on identical contexts |
| [Upgrading vLLM](#upgrading-vllm) | pinned to 0.30.0; 0.33.0 exists; the client and the tool-schema hook use internals | a scratch environment and `make test` |

Out of scope: other models, other GPUs, batching. (If you try another GPU, the build takes `ARCH=-arch=sm_XX`, but nothing outside sm_86 has been tried.)

## A rotated 4-bit KV cache

**Idea.** Rotate each K and V head vector with a 256-point Walsh-Hadamard transform (foldable into the query and the output, so it costs almost nothing at decode), use a log2 group
scale per 64 coefficients and a fixed Gaussian 16-level codebook, at the same 4.125 bits per element.

| Setting | Current 4-bit cache | Rotated variant | fp8 |
|---|---|---|---|
| Qwen3.5-2B proxy, 262k fresh tokens (perplexity vs bf16) | +2.015% | +0.194% | not measured |
| This 27B, 32k tokens | +0.27% | +0.10% (+-0.04) | 0.00% |
| This 27B, 98k tokens | +0.26% | +0.13% (+-0.03) | +0.03% |

**What we learned.**

- The proxy's 10x did not carry over to the 27B: about 2x. The current cache's loss on the 27B is already small, and what remains looks like the cost of 16 levels per coefficient. [INFERENCE]
- A per-channel mean table, which was most of the gain on the proxy, added nothing on the 27B (no mean gives the same perplexity).
- A 5-bit variant (group 32, asymmetric, with the rotation) measured +0.05% at 32k in an earlier run, at one more bit per element.

**Caveats.** Measured as quantise-then-dequantise in bf16 storage (the format's numbers, not a kernel's): no decode kernel, no speed measurement, code perplexity only, one run per arm.
The CUDA fake-quant was checked against a torch reference on 65,536 real K/V rows (identical scale codes in 99.9996% of groups).

**To settle it.** Write the decode kernel (lookup-table dequant, rotation folded into q and the output), measure speed against the current cache, then perplexity at 32k and 98k.

**Tool calls.** How a rotated cache behaves on tool calls is not measured. The 4-bit cache's well-formed tool-call rate depends on the key order of the tool schema ([Usage](USAGE.md#tool-calls)); whether rotation changes that is open.

## Why the 4-bit cache missed a needle at 100k

The 4-bit arm missed a planted needle at 100k tokens in the speed sweep and found it at 32k, 64k, 130k and 190k; in a separate test it got 7 of 8 needles at about 100k (fp8: 8 of 8). That could be noise
at this sample size or a real failure mode of one position range. **To settle it:** many more needle prompts across positions, with fp8 as the control, and a look at which layers' keys carry the miss.

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

## Faster long prefill

Cold prefill is 1,888 tok/s at 16k, 1,094 at 190k and 963 at 262k (re-measured 2026-11-04), so a 190k prompt takes about 3 minutes; later turns reuse the cached prefix. It is within about 10% of the reference build in the
[comparison](PERFORMANCE.md#4-hyperqwen-as-a-reference-point), so a gain would come from the attention and DeltaNet prefill path (`engine/prefill.py`). No profile has been taken.

## Repeating the HyperQwen reference measurement

The reference numbers describe one HyperQwen build and single runs. A repeat with repeats per cell, a newer build, its DFlash2 preset if it starts, and a second power limit would be a better reference.
This is a measurement exercise, not a contest.

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
