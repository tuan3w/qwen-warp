# Changelog

## Unreleased

### Changed

- The 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`, `--profile long`) has a new format at the same bytes per token (16.5 KiB): the head vector is rotated with a Walsh-Hadamard transform, quantised with a symmetric companded 16-level grid and 4 logarithmic scales, the first 4 tokens and the last 256 tokens are kept exact (fp16), and the reader multiplies every decoded scale by 1.0075 to undo the shrink of the writer's error-minimising scale. KL divergence from the bf16 cache on a 32k code text: 0.069% against 0.51% for the previous format. Through the server on `--profile long`: a minimal one-parameter tool schema in the usual key order gave a well-formed call in 92% of 200 requests (previous format 46% of 100; bf16 91% earlier), and a planted needle at 100k was found in 4 of 4 runs (previous format 0 of 4). Details, method, speed and what is not known: [docs/KV_CACHE.md](docs/KV_CACHE.md).
- The exact sink and ring are fixed buffers: +34 MiB of device memory (in the start-up memory model, `VRAM_KV_FIXED_MIB`) and +32 MB of pinned host RAM per prefix snapshot (183 MB instead of 151 MB each, 4 by default). They are saved and restored with every snapshot, so a resumed generation equals a cold prefill (`kv_aux_*` in `server/smk_runner.py`; `tests/test_server_smoke.py`, `tests/test_server_seed.py`).
- Speed through the server with the previous format as control, same session: level at 100k and 190k prompt tokens (190k: 100.5 against 96.4 tok/s with the prompt-lookup tail, 96.4 against 96.0 without), 2% to 7% slower at 32k and 2% to 4% on short prompts; prefill unchanged. A new decode attention kernel for 2 to 8 verify rows (`attn_split_i4_kernel`) dequantises each K/V tile once per query-tile group; the first version of the new format lost 15% to 20% at 190k without it. A 5-row verify step is about 12% slower and a 1-row step about 3% slower than the previous reader at 160k.

### Fixed

- `kv_aux_new` returned an unpinned empty tensor for caches without aux state (bf16, fp8), which made the server die at the first snapshot; it now returns `None`.

### Not measured on the new format

Perplexity at 98k, coding pass@1, repository question-answering, agent-sized tool sets. On one 61-token loop prompt at temperature 0.7 the sampled text repeats more than with bf16 (distinct-token ratio 0.240 against 0.330; the previous format 0.350); teacher-forced entropy matches bf16 and the cause is not known.

## 0.3.0 - 2026-11-04

### Fixed

- A request with a `seed` now gives the same tokens whatever the cache state (cold prefill, resumed from an earlier turn, repeated prompt, `--snapshots 0`); before, a resumed prefill and a cold one were not numerically identical, so a seeded answer could change with the cache. The prefill is now cut into segments that depend on the prompt only, and a resume continues that plan. A repeated prompt with 240 tokens or fewer after its last cut is prefilled again instead of resumed. The realisation still depends on `--num-speculative-tokens` and `--prompt-lookup` ([Usage](docs/USAGE.md#seeds-and-reproducibility)).
- `top_k` above 64 was silently clamped to 64; it is now HTTP 400 with the limit in the message. `0` and `-1` still mean "no filter" and behave as 64.
- The documented default (`--max-model-len` 28,672 with the prompt-lookup tail, bf16 KV) did not start on a 24 GB card with a 0.6 GiB desktop: it ran out of memory in the first warm-up prefill. The server now measures the free GPU memory at start and refuses a length that does not fit within about 10 s, before loading the weights, naming the largest lengths that do. The documented memory limits were wrong in both directions and were re-measured.
- Tool calls on the 4-bit cache: a minimal one-parameter schema in the usual key order (`type`, `properties`, `required`) gave a well-formed tool call in 39% of requests (bf16: 91%). The server now renders tool schemas with sorted keys by default (`--sort-tool-schema`): 87% (n = 150), 40/40 through the server on `--profile long`. The sampler was checked and is exact. Not measured: coding-agent tool sets, second turns, fp8 with the sorted order ([Usage](docs/USAGE.md#tool-calls)).
- Docs: the full 262,144-token context was documented as using 22.4 GiB; measured on the server it is 23,227 MiB in use on the card, the desktop included, with `--prompt-lookup 0` (23,899 MiB with the tail). Context limits, the prompt-lookup cost (0.56 GiB, not 0.62), the vLLM field name of the thinking text (`reasoning`), "same seed, same output" (true for a fixed configuration only), and the claim that the 4-bit cache costs almost nothing (not for tool calls) were corrected. The `make test` list and the quick no-GPU subset are documented.

### Added

- Sampling parameters `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `min_p`, `logit_bias`, `min_tokens` and `allowed_token_ids` (vLLM semantics and order), exact under speculative decoding; `min_p` and `logit_bias` also work with speculative decoding, which stock vLLM refuses. Still HTTP 400: `logprobs`, `prompt_logprobs`, `logprob_token_ids`, `bad_words`, `structured_outputs`, `thinking_token_budget`. `allowed_token_ids` with thousands of ids costs about 11% per cycle; all processors together lowered tok/s by 15% at T 0.7; no measurable cost without them ([Usage](docs/USAGE.md#sampling-parameters)).
- `--sort-tool-schema {on,off}` (default on).
- Tests in `make test`: `test_logit_procs`, `test_spec_procs`, `test_proc_model`, `test_sampling_params` (default and long profile), `test_tool_schema`, `test_sampler_dist`, `test_tool_call_rate`, `test_prefill_plan`, `test_server_seed`, `test_startup_configs`; the smoke test's long-document prompt is pinned in `tests/long_doc.txt`.
- Docs: `reasoning_effort` (values, the 400 for others, a prompt hint with no measured shortening; the values and the 400 were checked on this server too), the thinking-text field name, the vLLM version pin (0.30.0; other versions untested), the tool-call measurements and a troubleshooting row, new entries in `docs/IDEAS.md` (thinking budget, log-probabilities, `bad_words`, structured output, processors on the draft head, bf16 tool-call gap, rotated cache and tool calls, upgrading vLLM).
- Docker: `Dockerfile`, `docker-compose.yml`, `.env.example` and an entrypoint (`PROFILE`, `EXTRA_ARGS`, `MODEL_DIR`, `PORT`); the checkpoint is mounted, not baked in. Checked on this version (2026-11-04): the image built, the container was healthy after 189 s with an empty cache volume, `EXTRA_ARGS="--api-key ..."` gave 401 without the key and 200 with it, and 22 of 23 request checks passed (the one failure is the old 600-token example, now 2,000).
- `server/serve.py` honours `$SWIFT_MODEL` as the default model path (the README already documented it).

### Changed

- Defaults: `--max-model-len` 20,480 (was 28,672); 28,672 with `--prompt-lookup 0` (was 32,768); `--profile long` stays 196,608. Measured edges on this card: bf16 26,624 with the tail and 34,816 without; fp8 48,128 and 62,464; 4-bit 79,872 and 104,448; with `--host-tables` bf16 94,208 and 105,472, fp8 172,032 and 184,320, 4-bit the model maximum 262,144.
- `--prompt-lookup auto` turns the tail on only if the length leaves 256 MiB of the free memory next to it (before: a table of lengths); the tail costs 0.56 GiB (0.67 GiB with `--host-tables`). With `auto` the tail now also fits at 262,144 on `--profile long`.
- Unsupported-parameter list: the server rejects 6 names now (it rejected 13 before the new parameters).
- Prefix-reuse snapshots are taken at the last two cut points of the prompt-determined prefill plan (a long system prompt or document is reused across different questions); turn 2 and 3 of a 1,529-token chat: 0.54 s and 0.41 s per request (was 0.60 s and 0.39 s).
- Docs reworked around the reader: a README that leads with why and for whom, tables for scanning, and two quick-start paths (Docker, native); new `docs/GETTING_STARTED.md` (prerequisites, weights, troubleshooting), `docs/USAGE.md` (API, flags, memory and context, thinking mode) and `docs/IDEAS.md`; a file map and a "where to start reading" table in `docs/ARCHITECTURE.md`; a summary table in `docs/PERFORMANCE.md`.
- Docs corrected earlier in this cycle: disk and RAM needs include the environment and the Docker image; the repository layout (now a table in `docs/ARCHITECTURE.md`) includes `smk.py`, `docker/` and the scripts; the model's default thinking mode and its effect on `max_tokens` are explained; the non-uniform weight codebook entry states the byte-count bound instead of a measured kernel slowdown.
- The HyperQwen measurements are a reference point, not a competition, and name the exact build they describe (image digest and source revision). The roadmap became "Ideas to explore", with the 27B result for a rotated 4-bit KV cache (about 2x lower perplexity cost, not the 10x seen on a 2B proxy).
- Speed and prefill tables re-measured on this code on 2026-11-04 (decode 1% to 10% above the 2026-11-03 numbers; prefill 1,888 tok/s at 16k, 1,094 at 190k, 963 at 262k), and the HyperQwen reference re-run the same day on the same pinned image: it decoded 5% to 22% faster than on 2026-11-03, so the ratio against `mtp-long` is 1.2x to 1.3x (it was 1.3x to 1.5x). The README now leads with the numbers and a shorter description; its examples use `max_tokens` 2,000 (a 600-token budget cut off about 1 in 20 answers while thinking).

## 0.2.0 - 2026-11-04

Serving-only rework (commit 6d0e932 and the docs commits after it).

- 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`): per token and KV head, asymmetric int4 with fp16 scale and minimum, 4.125 bits per element, 16.5 KiB per token. Perplexity +0.26% at 98k tokens; coding pass@1 244 -> 241 of 283 tasks (not significant).
- `--profile long`: 4-bit KV plus host-pinned embedding and head tables (bit-identical decode); the full 262,144-token context fits on one RTX 3090 (`--prompt-lookup 0`; the 22.4 GiB stated in this entry at the time was not reproduced, see Unreleased).
- MTP speculative decoding (k <= 3) with an optional prompt-lookup tail; exact speculative sampling.
- OpenAI-compatible server on vLLM's frontend with an in-process engine client; multi-turn prefix reuse via DeltaNet state snapshots.
- Torch stack locked to the +cu130 builds (`uv.lock`).
- Reference measurements next to HyperQwen (`docs/PERFORMANCE.md`): single runs, one pinned HyperQwen build, caveats listed there.

Known limits: RTX 3090 only, one model, batch size 1, text only; the 4-bit cache drifts from bf16 output earlier than fp8 does.
