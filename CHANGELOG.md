# Changelog

## 0.7.0 - 2026-11-06

### Added

- `docker-compose.dflash2.yml`: an opt-in override that mounts the drafter checkpoint (`DFLASH_DIR`, default `~/models/dflash2-w4a16`) and sets `DFLASH_W4`; with `EXTRA_ARGS="--drafter dflash2 --max-model-len 122880"` the long profile serves the block drafter (the 196,608-token default is refused until the drafter cache has a ring buffer).
- `--drafter dflash2` (experimental, default `mtp`, which is byte-identical to 0.6.0): the server decodes greedy requests without logit processors with the in-engine DFlash2 block drafter ([Usage](docs/USAGE.md#the-dflash2-drafter---drafter-dflash2-experimental), [Architecture](docs/ARCHITECTURE.md#in-the-server---drafter-dflash2)). The fast prefill captures the drafter's context window from the residual stream of target layers 6/20/34/48/62 (`FastPrefill.tap`, read-only; `dflash.TapCapture`), so prompts are prefilled once, layer-major, as before: +1.3% on a cold 12k-token prefill, +3 ms on a cached follow-up turn, `cached_tokens` unchanged. A router sends sampled requests, requests with logit processors and requests whose drafter window is not on the GPU to the MTP path. Through the server (bf16 KV, `--host-tables`, same-session, stop at the end token): code 276.7 against 183.9 tok/s (+51%), math 254.0 against 183.9 (+38%), prose 113.4 against 118.0 (-4%), +34% over the 8 chat prompts of `bench_common`, +7% over 12 pooled chat tasks; **31% slower on copy-heavy agent turns** where the prompt-lookup tail copies (230 against 334 tok/s); 4-bit KV: +50% / +10% / +42%. [Performance, section 5](docs/PERFORMANCE.md#5-dflash2-block-drafter-experimental).
- `--dflash-path`, `--dflash-codebooks {host,device}` (host: the two 121 MiB selector codebooks in pinned memory, same speed and ids, 244 MiB less VRAM); memory model terms for the drafter (+1.15 GiB with host codebooks, +20 KiB per context token, measured) and a start-up refusal that names the largest `--max-model-len` per cache layout. `--drafter dflash2` practically needs `--host-tables`; `--profile long` does not fit (no ring buffer for the drafter's KV cache yet).
- The host prefix cache carries the drafter window of a spilled lineage (`DfWindow`, 40 MiB of pinned memory per entry), so a host restore keeps decoding with the block drafter; `smk_dflash_kv_ptr` exposes the drafter's K/V cache.
- `tests/test_dflash_server.py` (tap capture is read-only, ids equal the host-driven T = 8 decode after the same fast prefill, drafter K/V rows against `df_prefill`, window bookkeeping across device resumes, a sampled turn in between and host restores), `tests/test_prefix_store.py` and `tests/test_startup_configs.py` cover the new paths.

## 0.6.0 - 2026-11-06

### Added

- Experimental DFlash2 block drafter inside the engine (`engine/dflash.py`, `engine/src/dflash.cu`, `include/dflash.h`, [Architecture](docs/ARCHITECTURE.md#dflash2-block-drafter-experimental-greedy-only)): a 5-layer W4A16 drafter proposes 7 tokens per cycle, the target verifies 8 rows, and the whole cycle (verify, accept, drafter, selector) is one CUDA graph. Greedy only, not wired into the server. Engine-direct on the README prompts: 276 / 112 / 261 tok/s (code / prose / math) against 185 / 119 / 184 for MTP k=3 with the prompt-lookup tail in the same engine, and level with HyperQwen's DFlash2 preset measured in the same session (254 / 103 / 226); [Performance, section 5](docs/PERFORMANCE.md#5-dflash2-block-drafter-experimental).
- Engine API `smk_engine_verify`, `smk_engine_commit`, `smk_engine_tap_gather` (an external draft verified over T <= 8 rows with the host accepting, tap readout) and `smk_bf16_gemv_f32_cfg` (mma configuration of the bf16 GEMV).
- Optional device n-gram lookup over the token history that overrides the drafts on a long exact match (`SMK_DF_LOOKUP=1`, off by default: it measured no gain once the benchmark stops at the end token).

### Changed

- Kernel work from an autonomous optimisation campaign (bit-identical output): the bf16 lm_head table stays on the device for `DflashEngine` when it fits (`Engine(lm_head_host=False)`; the default and the server keep it in pinned host memory), the bf16 attention split loop issues the next K/V tiles early (31k context: `attn_split` 222 -> 198 us per call), the drafter attention is split across query rows (106 -> 64 us per call), a few drafter kernels are fused, and the DeltaNet kernel's tail loads earlier (21.9 -> 20.7 us per call). Measured against the previous tree, same session, 2 interleaved rounds: +1.1% on short prompts and +2.6% on a 31k-token prompt on the benchmark set, +1.2% and +2.3% on a held-out set.

## 0.5.0 - 2026-11-05

### Added

- Host prefix cache: the GPU still holds one conversation, but a request that does not continue it no longer destroys it. The engine spills the conversation's KV rows and snapshots to pinned host memory (`--prefix-cache-gb`, default 6 GiB, LRU) and restores a stored conversation when a later request continues it. Agents interleave a long main conversation with helper calls and subagents; on a replayed agent trace (12k-token system prompt, 6 turns, helper requests, 2x2 parallel subagents) the trace takes 40 s instead of 121 s and the median first-token delay of the main turns drops from 8.6 s to 1.1 s. A resumed request is bit-identical to a cold one.
- `--system-checkpoint on|off` (default on): one extra snapshot at the end of each long system prompt, so parallel agents sharing a system prompt resume there.
- Engine API `swiftmk_kv_range_bytes/copy` and `smk_spec_mtpkv_range_bytes/copy` (copy a row range of the KV cache to and from pinned host memory for bf16, fp8 and int4).

### Changed

- Prefill of a prompt with many chat-turn boundaries is faster: consecutive prefill segments (the plan cuts one at every turn boundary) now run layer by layer, so each layer's int4 weights are dequantized once per group of segments instead of once per segment (the group's rows share the existing 4,096-row scratch, no extra GPU memory; a group ends at every snapshot position). 12,000 tokens with a turn marker every 900 tokens (15 segments): 7.99 s -> 7.21 s; every 300 tokens (41 segments): 11.79 s -> 8.91 s; a prompt with few turns is unchanged. Every segment runs the same kernels on the same row counts, so the result is bit-identical to the one-segment-at-a-time prefill (`tests/test_group_prefill.py`: all snapshots, main + MTP KV rows and tokens, bf16 / fp8 / int4).
- `--snapshots` defaults to 16 (was 4): it now counts the buffers of the GPU conversation and of the host cache together; they are allocated on demand beyond the first four.

## 0.4.0 - 2026-11-05

### Changed

- The 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`, `--profile long`) has a new format at the same bytes per token (16.5 KiB): the head vector is rotated with a Walsh-Hadamard transform, quantised with a symmetric companded 16-level grid and 4 logarithmic scales, the first 4 tokens and the last 256 tokens are kept exact (fp16), and the reader multiplies every decoded scale by 1.0075 to undo the shrink of the writer's error-minimising scale. KL divergence from the bf16 cache on a 32k code text: 0.069% against 0.51% for the previous format; perplexity at 98k +0.056% over bf16 (previous format +0.26%). Through the server on `--profile long`: a minimal one-parameter tool schema in the usual key order gave a well-formed call in 92% of 200 requests (first build) and 96% of 100 (final build), previous format 46% of 100 (bf16 91% earlier); a planted needle at 100k was found in every run, the previous format missed it in every run. Details, method, speed and what is not known: [docs/KV_CACHE.md](docs/KV_CACHE.md).
- The exact sink and ring are fixed buffers: +34 MiB of device memory (in the start-up memory model, `VRAM_KV_FIXED_MIB`) and +32 MB of pinned host RAM per prefix snapshot (183 MB instead of 151 MB each, 4 by default). They are saved and restored with every snapshot, so a resumed generation equals a cold prefill (`kv_aux_*` in `server/smk_runner.py`; `tests/test_server_smoke.py`, `tests/test_server_seed.py`).
- Speed through the server with the previous format as control, two interleaved runs per cell: short prompts level; 32k level without the prompt-lookup tail and about 3% lower with it; 100k +7% (tail) and +12% (no tail); 190k +4% (tail) and level (no tail); prefill unchanged. Kernels: `attn_split_i4_kernel` serves 1 to 8 verify rows and dequantises each K/V tile once per query-tile group (the first version of the format lost 15% to 20% at 190k without it); the exact ring is attended inside the split kernel as 16 fp16 tiles and an 8-warp write kernel (`attn_prep_i4_kernel`) writes the cache, which removed a fixed cost of about 0.9 ms per speculative cycle that showed on short prompts and at 32k. A 1-row step is about 1% to 2% over the previous reader. The ring change makes the attention output differ from the previous build at rounding level (the written cache bytes are identical).

### Fixed

- `kv_aux_new` returned an unpinned empty tensor for caches without aux state (bf16, fp8), which made the server die at the first snapshot; it now returns `None`.

### Not measured on the new format

Coding pass@1, repository question-answering, agent-sized tool sets. (Perplexity at 98k was measured: +0.056% over bf16, previous format +0.26%.) On one 61-token loop prompt at temperature 0.7 the sampled text repeats more than with bf16 (distinct-token ratio 0.24 against 0.29 to 0.33); it comes from the exact 256-token ring (a 16 or 64 token ring gives 0.37 to 0.39) and four other prompts match bf16; why is not known.

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
