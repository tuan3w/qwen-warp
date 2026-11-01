# swift-mk server — the swift-mk engine behind vLLM's own OpenAI frontend

`vllm serve`-compatible OpenAI server for Swift-1.5-Qwen3.8-27b-INT4 on one RTX 3090, running the clean-room CUDA-graph engine (`../engine/`, ENGINE.md) with greedy and
exact-sampling MTP speculative decoding (k = 3 by default).  Chat / completions / models / health / tokenize / metrics, streaming SSE, reasoning parser (`qwen3`),
tool-call parser (`qwen3_coder`), chat template, usage — all of it is **vLLM 0.30's own frontend code, unmodified**; only the EngineCore behind it is swift-mk.

## Launch

```bash
# 1. the GPU lease (the server holds ~21-23 GiB); release it when you stop the server
projects/llm-lab/bin/lab gpu acquire --holder me

# 2. the server (defaults: model ~/models/swift-1.5-int4, served name "swift", --max-model-len 28672 (K11 prompt-lookup tail on; 32768 with --prompt-lookup 0), bf16 KV, k=3, reasoning parser qwen3, tool parser qwen3_coder, auto tool choice)
~/llm-lab/venv/bin/python server/serve.py --port 8000
#    all flags spelled out:
~/llm-lab/venv/bin/python server/serve.py --model ~/models/swift-1.5-int4 --served-model-name swift --host 0.0.0.0 --port 8000 \
    --max-model-len 28672 --kv-cache-dtype auto --num-speculative-tokens 3 --prompt-lookup auto --stream-cycles 2 --snapshots 4 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice

# 3. use it
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"swift","messages":[{"role":"user","content":"Hello"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

The build is picked from `SMK_BUILD` (default `engine/build/`; `make -C engine` first, K7's sampling kernels need the current `libswiftmk.so`).

**Or through stock `vllm serve`** (the plugin route; same engine, every stock flag is read from vLLM's own config):

```bash
VLLM_SWIFT_MK=1 PYTHONPATH=server/plugin ~/llm-lab/venv/bin/python -m vllm.entrypoints.cli.main serve ~/models/swift-1.5-int4 \
    --served-model-name swift --max-model-len 32768 --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice --port 8000
```

### Flags

| flag | meaning |
|---|---|
| every stock `vllm serve` frontend flag | `--host/--port/--api-key/--served-model-name/--chat-template/--reasoning-parser/--tool-call-parser/--generation-config/...` unchanged |
| `--max-model-len N` | context (prompt + output). **VRAM-bound** (below). Default 28672 (32768 with `--prompt-lookup 0`) |
| `--kv-cache-dtype {auto,bfloat16,fp8}` | `auto`/`bfloat16` = bf16 KV (default; note stock vLLM's `auto` resolves to the checkpoint's fp8 KV), `fp8` = e4m3 with the checkpoint k/v scales (half the KV bytes, ~1.8e-2 attention-output rel. error, see BENCH.md K2) |
| `--num-speculative-tokens {0,1,2,3}` | MTP drafts per cycle (sugar for `--speculative-config '{"method":"mtp","num_speculative_tokens":k}'`; 0 = plain decode). Default 3 |
| `--prompt-lookup {0,1,auto}` | K11: MTP chain + prompt-lookup (n-gram) tail, verify up to 8 rows per cycle: code edits / log extraction / agent tool calls decode **x1.32-1.46** faster (greedy and sampling), prose x1.00, math x1.07 (BENCH.md K11).  Needs `k = 3` and the 9-slot DeltaNet ring (+0.62 GiB: `--max-model-len` <= 28672 with bf16 KV, <= 53248 with fp8 KV).  `auto` (default) = on iff the `--max-model-len` fits, else off with a log line; stream chunks carry up to 8 tokens per cycle |
| `--stream-cycles N` | speculative cycles per host sync = streaming granularity (default 2, see the table) |
| `--snapshots N` | DeltaNet state snapshots kept for multi-turn prefix reuse (151 MB pinned host RAM each; 0 = off). Default 4 |

### Startup, memory, `--max-model-len`

* Startup to `/health` OK: **~20 s** with a warm page cache (K8 text: 17 s with an engine load of 7.2 s = weights + MTP pack 3.5 s + warmup 3.7 s; the m3 port adds the in-place W4 repack of the loaded tensors; the rest is vLLM's config / tokenizer / chat-template warmup).
  There is no prepacked-weight cache: `engine/pack.py` copies the checkpoint's W4 tensors as they are and `Engine` / `SpecEngine` convert them to the tile-contiguous layout in place on the GPU after loading (`pack.repack_w4`), and the MTP int4 quantisation runs on the GPU in < 1 s, so the load is
  page-cache -> device copy bound; a cold page cache adds the disk read of 19.5 GB.  The warmup runs a 4160-token prefill so the fla autotune / prefill scratch are paid at startup, not by the first request.
* VRAM (23.56 GiB card, 0.91 GiB used by other processes; `maxctx.py`, prompt of ctx-512 tokens prefilled in 4096-token segments).  Since K9 the engine keeps an int4 copy of the lm_head (+0.59 GiB, next to the BF16 head it needs for the exact rescoring of the top candidates; `SMK_HEAD=int8` +1.18 GiB, `SMK_HEAD=bf16` = the pre-K9 head and limits):

  | KV | --max-model-len | device used, int4 head (default) | device used, BF16 head (`SMK_HEAD=bf16`) |
  |---|---|---|---|
  | bf16 | 32768 (default) | 23.22 GiB, ok, 0.34 GiB headroom | 22.61 GiB |
  | bf16 | 36864 | 23.50 GiB, ok (max measured), 0.06 GiB headroom | |
  | bf16 | 40960 | OOM at start | 23.17 GiB, ok (max measured) |
  | bf16 | 49152 | | OOM at start |
  | fp8 | 65536 | 23.47 GiB, ok (max measured) | 22.86 GiB |
  | fp8 | 73728 / 81920 | OOM at start | 81920: 23.48 GiB, ok (max measured) |

  K11 prompt-lookup tail (9-slot DeltaNet ring, +48 x 4 x 3.2 MB = +0.62 GiB; `maxctx.py CTX [fp8] pld`): bf16 KV **28672 ok (max measured), 30720 OOM**; fp8 KV **49152 / 53248 ok (max measured), 57344 OOM**.  `--prompt-lookup 0` restores 36864 / 65536 (int4 head).
  | fp8 | 98304 | | OOM at start |

  Cost per context token: ~70 KB (bf16: 16 attention layers x 2 x 4 heads x 256 x 2 B = 64 KB + MTP KV 4 KB + the prefill's K/V gather buffers 4 KB), ~39 KB (fp8).  The int8 head does not fit the default 32768 bf16 context (OOM), which is one reason the default is int4.
  The server refuses to start (clear message) when `--max-model-len` does not fit; the 262144 of the model card cannot be served on this card with this engine.

## How it plugs into vLLM (the seam)

`vllm.entrypoints...run_server` -> `AsyncLLM` (the `EngineClient`) -> `EngineCoreClient`.  Implementing the `EngineClient` protocol directly would mean re-implementing `AsyncLLM`
(input processor, output processor with detokenisation / stop strings / streaming deltas, renderer, 30+ methods, `vllm_config`, `model_config`...); the engine-core boundary below it is **6 calls**:
`add_request_async`, `abort_requests_async`, `get_output_async`, `get_supported_tasks_async`, `reset_*_async`, `shutdown`.  `smk_core_client.SwiftCoreClient` (~210 lines) implements those on an
in-process FIFO queue + one worker thread that owns the engine, and `install()` swaps `EngineCoreClient.make_async_mp_client` (a runtime patch: no vLLM file is modified, no vLLM EngineCore process
or GPU worker is started).  Everything the OpenAI frontend does is therefore stock: chat template + `chat_template_kwargs`, `generation_config.json` defaults (T=1.0, top_p .95, top_k 20 when the
request gives no sampling parameters, exactly like `vllm serve`), the `qwen3` reasoning parser (`reasoning` field / deltas), the `qwen3_coder` tool parser, `usage` (incl. `prompt_tokens_details.cached_tokens`
for resumed turns), `n > 1`, stop strings, `/tokenize`, abort on client disconnect.

* `serve.py` — the CLI: vLLM's arg parser + defaults for this model + `run_server`.
* `smk_core_client.py` — the EngineCore: parameter validation, request queue, stop logic (EOS / `stop_token_ids` / `max_tokens` / `max_model_len`, exactly vLLM's `check_stop`), abort.
* `smk_runner.py` — the engine driver (no vLLM): prefill (fast prefill with the slow fallback), speculative cycles, sampling, exact stop cut, snapshots.
* `plugin/` — the `vllm.general_plugins` entry (dist-info directory on `PYTHONPATH`, nothing is pip-installed; active only with `VLLM_SWIFT_MK=1`): it installs the same core client so a stock
  `vllm serve` command line (above) runs the swift-mk engine.  It was cheap (25 lines) because the settings are read from the stock vLLM config
  (`--max-model-len`, `--kv-cache-dtype`, `--speculative-config`).

### Requests: serialisation, streaming, stop, abort

* bs = 1: requests run one at a time in arrival order (FIFO; `n > 1` and concurrent clients just queue).  A client that disconnects (or a stop string that the frontend detects) aborts the request:
  a queued request is dropped, the running one stops at the next sync (<= one stream group = 2 cycles ~ 46 ms, or between 4096-token prefill segments) and the engine is free for the next request (smoke test: next request 0.2 s later).
* Streaming: each `--stream-cycles` group of speculative cycles is one sync; its accepted tokens (1-8) are emitted immediately, the first token (from the prefill) is emitted alone so TTFT = prefill.  The tokens of the last group beyond EOS / `stop_token_ids` / `max_tokens` /
  `max_model_len` are discarded exactly at the stop (`runner_check.py`: `max_tokens` = 1..13 and stop ids cut exactly at every position of a cycle group); stop *strings* are cut by vLLM's detokenizer as in stock vLLM.
* Prefill: `FastPrefill` (K6 + K10): prompts of <= 240 tokens run through the W4 mma GEMM (`src/prefill_w4.cu`, no bf16 dequant; the last prompt row rides in the batch, no separate 20 ms head step), longer ones through K6's dequant + cuBLAS path (~2000 tok/s), a 1-token prompt through the engine's own head step.  TTFT at 8 / 20 / 40 / 64 / 128 tokens = 34 / 38 / 47 / 75 / 128 ms (was 74 / 130 / 235 / 217 / 227 ms; stock vLLM ~96 ms at 30 tokens), `bench_ttft.py`, BENCH.md K10.
* Sampling (K7, exact speculative sampling; BENCH.md): `temperature`, `top_p`, `top_k`, `seed` (same seed -> same tokens, also after a snapshot resume), greedy for `temperature` 0.  `top_k` <= 0 or > 64 is clamped to 64 (the engine's device top-k
  works on <= 64 candidates: the sampled distribution is the target distribution truncated to its 64 most likely tokens).

### Multi-turn prefix reuse

DeltaNet state cannot be rewound, so the server checkpoints it.  During the prefill of every request the DeltaNet + conv state is copied to host memory (pinned, 13 ms, `--snapshots 4`) after the tokens `[0, len-1)` and after the
tokens `[0, p+2)` where `p` is the last `<|im_start|>` (the latter only when `p+2 > 240`: below that it would split a short prompt into two weight passes; for prompts of <= 240 tokens the snapshot before the last token is copied asynchronously after the first token has left, so it is not part of the TTFT); the device KV caches always hold the lineage of the last prompt.  A new request resumes from the snapshot with the largest position `q <= (common prefix with the lineage) - 1`
(MTP pair `q-1` needs token `q` to be common), restores the state (H2D 13 ms) and prefills only `ids[q:]`; no usable snapshot -> reset + full prefill.
Thinking off, or any agent/tool loop: the next prompt extends the previous one -> resume at its last token.  Thinking on: the chat template re-renders the previous assistant turn without `<think>\n` -> resume at `<|im_start|>assistant\n`
and re-prefill the previous answer + the new user turn.  One lineage only: two interleaved conversations evict each other (full re-prefill, still correct).
The resumed run is numerically the same state as a cold prefill up to the accumulation order of the segmentation (a snapshot splits the fast prefill into segments; K6's noise floor), so a greedy near-tie can in principle resolve differently:
`runner_check.py` / the smoke test compare warm vs cold greedy turns (identical in all runs so far).

## Tests and numbers

* `smoke_test.py` — starts the server and talks to it with the `openai` client (22 checks): chat greedy ids == the direct engine's (SpecEngine + FastPrefill, separate process) for 3 short prompts and a 600-token prompt, streaming == non-streaming + usage,
  stop strings, `max_tokens`/EOS finish reasons, abort frees the engine, concurrent == sequential, seeded sampling reproducible, generation-config defaults, unsupported parameters -> 400, tool-call parse + tool-result round trip, `/v1/completions`,
  3-turn chat resumes (`cached_tokens` > 0) and equals the cold answer.  Needs the GPU lease: `~/llm-lab/venv/bin/python server/smoke_test.py`.
* `runner_check.py` — engine-level checks (k = 0..3 equal greedy ids, exact cuts, prefix reuse in both thinking modes, sampling reproducibility).
* `bench_server.py` — the HTTP benchmark used for the numbers below (same client for swift-mk and stock `vllm serve`); `bench_ttft.py` — short-prompt TTFT sweep (8..512 tokens); `maxctx.py` — the VRAM probe.
* `ab_visit.py` — one visit of the server-vs-stock-vLLM A/B (boot time, E0 decode greedy + T=1, TTFT 20 tokens / 8K); driver loop `engine/results/m3-port/srv.sh`.

### Numbers, m3 port (2026-11-01, `ab_visit.py`, order ours / stock / ours / stock, both servers stopped in between, lease held; median of 2 visits x 3 reps (8K: 2); ours = defaults (prompt-lookup auto, bf16 KV, k=3, max-model-len 28672), stock = `vllm serve` 0.30 `--max-model-len 16384 --max-num-seqs 4 --no-enable-prefix-caching` MTP k=3; raw: `engine/results/m3-port/server/`; `engine/BENCH.md` "K12 m3 port")

| decode tok/s | code | prose | math | 8K ctx |
|---|---|---|---|---|
| **swift-mk server, greedy** | **184.1** | **116.8** | **183.9** | **144.7** |
| stock vLLM, greedy | 96.7 | 63.0 | 94.2 | 77.7 |
| ratio | **x1.90** | **x1.86** | **x1.95** | **x1.86** |
| swift-mk server, T=1 top_p .95 top_k 20 | 174.0 | 116.2 | 178.4 | 147.1 |
| stock vLLM, T=1 | 88.7 | 59.6 | 88.1 | 75.7 |
| ratio | x1.96 | x1.95 | x2.02 | x1.94 |

TTFT 20-token prompt **37 ms** (stock 84 ms), 8K prompt (8,030 tokens) **3.91 s** (stock 4.29 s); boot to `/health` 19.6 / 19.7 s (stock 102 s cold / 34 s warm cache).  The tables below are the older K8 measurements (BF16 head, no m3 kernels).

### Numbers (2026-10-31, RTX 3090, lease held, `bench_server.py`, same HTTP client for both servers; `--max-model-len` 16384 (stock vLLM 12288: 16384 does not fit next to the 0.9 GiB of other processes), greedy, ignore_eos, 385 tokens, median of 3 reps)

(K9, 2026-11-01: the numbers below were measured with the BF16 lm_head.  The engine now defaults to the int4 head + exact BF16 rescoring: direct-engine k=3 tok/s +10% (code 152.1 -> 167.8, prose 97.4 -> 107.4, math 150.0 -> 165.5, 8K 137.7 -> 151.9, `engine/BENCH.md` K9), plain step 20.15 -> 18.10 ms, greedy streams identical; the HTTP table was not re-measured.  `SMK_HEAD=bf16` restores the old behaviour.)

Decode = (tokens after the first chunk) / (last chunk - first chunk time), the E0 definition; TTFT = request sent -> first token chunk, one request per rep with a unique 1-token prefix (no cache can hit).
Stock = `vllm serve` 0.30 with E0's engine settings (`--speculative-config mtp k=3 --max-num-seqs 4 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.93`, fp8 KV `auto`, prefix caching on), started in 55 s.

| | code | prose | math | 8K ctx |
|---|---|---|---|---|
| **swift-mk server, bf16 KV, k=3, N=2: decode tok/s** | **152.2** | **96.1** | **149.0** | **122.1** |
| stock `vllm serve` MTP k=3 (HTTP, same client) | 96.6 | 64.4 | 94.8 | 74.1 |
| ratio | 1.58x | 1.49x | 1.57x | 1.65x |
| E0 offline (vLLM `LLM`, greedy k=3) | 96.4 | 63.0 | 93.5 | 75.5 |
| direct engine (`bench_spec.py`, 8 cycles/sync / 2 cycles/sync) | 152.6 / 151.5 | 95.7 / 95.8 | 149.1 / 148.8 | 121.9 / 121.4 |
| swift-mk server, fp8 KV | 149.0 | 93.6 | 151.3 | 116.1 |
| **TTFT swift-mk** (prompt 32 / 34 / 33 / 8030 tokens) | 216 ms | 235 ms | 218 ms | **3944 ms** |
| TTFT stock vLLM | 97 ms | 96 ms | 95 ms | 4351 ms |

* **HTTP-layer overhead = none measurable**: server decode 152.2 / 96.1 / 149.0 / 122.1 vs the direct engine at the same sync size (2 cycles) 151.5 / 95.8 / 148.8 / 121.4 (+0.5 / +0.3 / +0.1 / +0.6 %; run-to-run spread < 0.3 %); TTFT: the server-side prefill time in the request log (216 ms, 3933 ms) vs the client's TTFT (216 ms, 3944 ms) = 0 ms for short prompts, ~10-20 ms for the 8K prompt (parsing 8K token ids, first SSE chunk).
* Short-prompt TTFT (K10, 2026-11-01): the 216-235 ms of the table above came from prefill-by-decode below 48 tokens and the fast path's fixed ~0.18 s; with the W4 mma GEMM for prompts <= 240 tokens (BENCH.md K10) the same sweep through this server reads 34 / 38 / 47 / 75 / 128 / 258 / 357 ms at 8 / 20 / 40 / 64 / 128 / 256 / 512 tokens (`bench_ttft.py`; K8 code: 74 / 130 / 235 / 217 / 227 / 281 / 379), i.e. 2x faster than stock vLLM's ~96 ms at ~30 tokens; 240-600 tokens still use the K6 dequant + cuBLAS path (~80 ms fixed).  A seeded sampled request is reproducible cold-vs-cold and warm-vs-warm; a cold short prompt (batched prefill) and the same prompt resumed from its snapshot (exact T=1 head step) can sample different continuations (bf16 accumulation order).
* **Streaming granularity** (`--stream-cycles N` = cycles per host sync; one cycle = 23.0-24.0 ms and 1-4 tokens; first token is always emitted alone after the prefill):

  | N | chunk gap p50 / p95 | server tok/s code / prose / math / 8K | direct engine code / prose / math / 8K |
  |---|---|---|---|
  | 1 | 23 / 23 ms | 152.2 / 96.6 / 149.0 / 122.2 | 151.5 / 95.7 / 148.3 / 121.4 |
  | **2 (default)** | 46 / 46 ms | 152.2 / 96.1 / 149.0 / 122.1 | 151.5 / 95.8 / 148.8 / 121.4 |
  | 3 | ~69 ms (3 cycles) | | 152.4 / 96.0 / 148.8 / 122.2 |
  | 4 | 92 / 93 ms | 149.1 / 94.9 / 148.9 / 122.1 | 152.5 / 95.7 / 149.1 / 121.9 |
  | 8 | ~184 ms (8 cycles) | | 152.6 / 95.7 / 149.1 / 121.9 |

  The per-sync cost is below the noise (<= 0.7 %); what N buys is a shorter overshoot after the stop (up to N cycles of wasted compute after EOS / `max_tokens`, visible as -2 % at N=4 on the 385-token code run) and shorter abort latency; N=1 is as fast as N=2 here but doubles the chunks the frontend must serialise (~43/s),
  N=2 keeps one SSE event per ~46 ms (one every ~6 tokens on code).  Default 2.
* **Multi-turn** (3-turn chat over a 1.5K-token document, thinking off, 60 new tokens/turn; TTFT warm = resumed from a snapshot, cold = after an unrelated request replaced the lineage):

  | turn | prompt tokens | resumed at | TTFT swift-mk warm | swift-mk cold | stock vLLM (prefix caching on) |
  |---|---|---|---|---|---|
  | 1 | 1535 | 0 | 841 ms | 836 ms | 849 ms |
  | 2 | 1628 | 1534 | **240 ms** | 912 ms | 899 ms (`cached_tokens` 0) |
  | 3 | 1717 | 1627 | **245 ms** | 962 ms | 945 ms (`cached_tokens` 0) |

  3.8x / 3.9x faster first token on turns 2 / 3; the remaining 240 ms = restore (13 ms) + prefill of the ~90 new tokens through the fast path (0.18 s fixed) + the head step.  Thinking on: resume point is the `<|im_start|>assistant\n` snapshot, `runner_check.py` (first-token logits warm vs cold rel-L2 0.004-0.027 = the prefill's bf16 noise floor, K6).
  Stock vLLM did not reuse anything in this setting (hybrid-model prefix caching reported `cached_tokens` 0 and TTFT grew with the context).
* Startup 16.5 s to `/health` (engine 7.2 s) at K8; **19.6 / 19.7 s with the m3 port** (`ab_visit.py`, 2 visits; the load-time W4 repack is the only new load work), stock vLLM 55 s at K8, **102 s (cold compile cache) / 34 s (warm)** in the 2026-11-01 visits.
* Abort: the next request after a dropped 2000-token stream is served 0.22 s later.


## Unsupported (rejected with HTTP 400 naming the parameter, never silently ignored)

`logprobs` / `prompt_logprobs` / `top_logprobs`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `min_p`, `min_tokens`, `logit_bias`, `allowed_token_ids`, `bad_words`, structured outputs / guided decoding / `response_format` JSON schema,
`thinking_token_budget`, `prompt_embeds`, multimodal inputs (text only: `language_model_only`), LoRA, multiple concurrent sequences (bs = 1 by design), `/metrics` engine stats (the frontend metrics exist, the engine ones do not),
context beyond the VRAM limit above, speculative methods other than the built-in MTP (k <= 3), `top_k` > 64 (clamped).
