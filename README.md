# qwen-warp

**Fast Qwen3.8-27B for one RTX 3090. 200k context, 100+ tok/s.**

I built qwen-warp to get the most out of my own RTX 3090: a custom CUDA decode engine behind an OpenAI-compatible API. With a **190k-token prompt it decodes at 102 tokens/s, 3.6x faster than [HyperQwen](https://github.com/syv-ai/hyperqwen)'s 200k preset (28 tok/s)**, measured on the same card, the same day and the same prompts. It decodes 94 to 133 tok/s with 32k to 262k tokens in the prompt and fits the model's full 262,144-token context in 24 GB.
At 32k to 130k it is 1.2x to 1.3x faster than HyperQwen's `mtp-long`, which does not serve 190k. On short prompts, HyperQwen's DFlash2 preset is faster than the MTP engine (about 1.5x on code); version 0.6.0 adds an experimental in-engine DFlash2 drafter (greedy, not in the server) that is level with it in a same-session comparison ([Performance, section 5](docs/PERFORMANCE.md#5-dflash2-block-drafter-experimental)); neither was measured at long context.
I tested it only with [Swift-1.5-Qwen3.8-27B INT4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4). Other Qwen3.8 27B variants may work; I have not tried them. One GPU (RTX 3090), one request at a time, text only.

## Numbers

Same card at 350 W, greedy, batch size 1, one run per cell. Everything below, HyperQwen included, was measured on 2026-11-04 with the same prompts and client ([method](docs/PERFORMANCE.md#3-speed)).

| Speedup: our decode tok/s divided by HyperQwen's | 32k | 64k | 100k | 130k | 190k |
|---|---|---|---|---|---|
| **qwen-warp `--profile long` vs HyperQwen `mtp-long`** | **1.17x** | **1.23x** | **1.34x** | **1.29x** | - |
| qwen-warp `--profile long` vs HyperQwen `mtp-huge` (4/2-bit KV, the only HyperQwen preset that serves 190k) | 1.94x | 2.10x | 2.82x | 3.31x | 3.58x |


| Decode, tokens/s, by prompt length | 32k | 64k | 100k | 130k | 190k | 262k |
|---|---|---|---|---|---|---|
| **qwen-warp** `--profile long` (4-bit KV cache, prompt-lookup tail) | 124 | 119 | 115 | 108 | 102 | - |
| **qwen-warp** 4-bit KV cache, tail off | 124 | 133 | 109 | 105 | 96 | 94 |
| **qwen-warp** fp8 KV cache | 123 | 123 | 112 | 108 | - | - |
| *reference:* HyperQwen `mtp-long` | 106 | 97 | 86 | 83 | - | - |
| *reference:* HyperQwen `mtp-huge` | 64 | 57 | 41 | 33 | 28 | - |

`-` = not run (fp8 stops at 130k; the tail-on row stops at the profile's 196,608; HyperQwen `mtp-long` does not serve 190k).

| Short prompts, tokens/s | code | prose | math |
|---|---|---|---|
| **qwen-warp** default (bf16 KV cache) | 172 | 110 | 172 |
| **qwen-warp** 4-bit KV cache | 170 | 108 | 162 |
| *reference:* HyperQwen `mtp-long` | 135 | 89 | 134 |
| **speedup**, qwen-warp 4-bit KV vs `mtp-long` | **1.27x** | **1.22x** | **1.21x** |
| *reference:* HyperQwen DFlash2 (earlier session, 2026-11-01, same card and 350 W; 254 / 103 / 226 again in the 2026-11-06 session) | 254 | 103 | 225 |
| qwen-warp 4-bit KV vs DFlash2 (different days) | 0.67x | 1.05x | 0.72x |

| | |
|---|---|
| Context | 262,144 tokens fit in about 22.5 to 22.7 GiB in use, desktop included (`--host-tables` moves 4.8 GiB of tables to host memory; the 4-bit cache costs 16.5 KiB per token, bf16 64 KiB) |
| Prefill | 1,890 tok/s at 16k, 1,100 at 190k (3 minutes), 963 at 262k (4.5 minutes); follow-up turns reuse the cached prefix |
| Ready after | 21 to 30 s |
| Quality of the 4-bit cache | rotated format with an exact sink and recent-token ring: KL divergence from the bf16 cache 0.069% at 32k (the previous format: 0.51%), perplexity +0.056% over bf16 at 98k (previous: +0.26%), a minimal tool schema in the usual key order 92% to 96% well-formed (previous: 46%; bf16 91%), a planted needle at 100k found in every run (previous format: missed in every run). Coding pass@1 was measured on the previous format only (241 against 244 of 283) and not repeated. On one 61-token loop prompt, caused by the exact ring, sampled text repeats more than with bf16; four other prompts match. [KV cache](docs/KV_CACHE.md) |
| Not measured | multi-turn agent task success, other GPUs, other power limits, HyperQwen's DFlash2 at long context, copy-heavy answers (HyperQwen reports 381 tok/s when the answer quotes its own prompt, with its lookup drafter), and many concurrent users (it reports about 1,035 tok/s aggregate at 64 concurrent in batch mode; this engine serves one request at a time) |

HyperQwen is a general serving project; this compares single-stream decode of this model on this card, against its unmodified public image. Gaps under about 10% are noise.

The speed rows above were measured with the previous 4-bit format. With the rotated format of this version, two interleaved runs per cell in one session gave the same speed on short prompts, 7% to 12% more at 100k, level to +4% at 190k, and level to 3% less at 32k ([PERFORMANCE](docs/PERFORMANCE.md#3-speed)).

<details>
<summary>Exact HyperQwen build and caveats</summary>

- Image `ghcr.io/syv-ai/hyperqwen@sha256:00713ba1e04b68db1be9eba2b15b579250f9fb0c29ed61add40c750228fc0cf8`, revision `bd6c5e2f11628483109031d99d1cc4a7d22b1a26`, built 2026-10-04. A newer build may differ.
- It decoded 5% to 22% faster on 2026-11-04 than in an earlier session on 2026-11-03, so one run per cell is a rough number.
- Every table, method and caveat: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

</details>

## How it gets there

- **Fused kernels.** The 4-bit weights decode as one CUDA graph of fused kernels, with no framework work per step.
- **MTP speculative decoding.** The checkpoint's own draft head proposes 3 tokens per step; the engine verifies them in one pass and keeps the output distribution exact, greedy or sampled. About 3 tokens come out per step.
- **A 4-bit KV cache and host-pinned tables.** The cache shrinks 4x; the embedding and rescoring head live in pinned host memory. Together they free the VRAM for 262k tokens.
- **Prompt lookup and prefix reuse.** Repeated text in the prompt speeds up decoding; a chat's earlier turns skip prefill.

The code map is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

You need Linux, an NVIDIA driver 580 or newer, an RTX 3090, and the weights (19 GB; accept the model's licence on Hugging Face first). Prerequisites, disk sizes and troubleshooting: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

```bash
# 0. weights, for both paths
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download ukisai/Swift-1.5-Qwen3.8-27b-INT4 --local-dir ~/models/swift-1.5-int4

# A. Docker (needs the NVIDIA container toolkit; serves the long-context profile on :8000)
docker compose up --build

# B. native (needs the CUDA 13.0 toolkit and uv; `make setup` takes 20 to 30 minutes, mostly downloads)
make setup
uv run server/serve.py --profile long --port 8000
```

Then call it like any OpenAI server:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "swift", "messages": [{"role": "user", "content": "Write a binary search in Go."}], "max_tokens": 2000}'
```

The model thinks before it answers, and the thinking tokens count against `max_tokens` (the text comes back in the `reasoning` field). With 2,000 tokens, 20 of 20 test answers to this prompt finished; at 600 to 1,500 tokens about 1 in 20 ran out
during thinking. Raise `max_tokens` if an answer is cut off, or switch thinking off per request with `"chat_template_kwargs": {"enable_thinking": false}`. More in [docs/USAGE.md](docs/USAGE.md).

| Goal | Flags | Context |
|---|---|---|
| Long context (the Docker default) | `--profile long` | 196,608 |
| The model's full context | `--profile long --max-model-len 262144 --prompt-lookup 0` | 262,144 |
| Fastest at short context | defaults (bf16 KV cache) | 20,480 (28,672 with `--prompt-lookup 0`) |
| fp8 KV cache | `--kv-cache-dtype fp8 --max-model-len 53248` | 53,248 |

The server measures the free GPU memory at start and refuses a context that does not fit, before it loads the weights; the limits are in [USAGE](docs/USAGE.md#memory-and-context). With Docker, set `PROFILE=` and `EXTRA_ARGS=` in a `.env` file ([.env.example](.env.example)).
`make test` checks your install; it needs the GPU and the weights and takes a while ([list](docs/GETTING_STARTED.md#4-verify-it-works)).

## What works and what does not

| | |
|---|---|
| Sampling | `temperature`, `top_p`, `top_k` (up to 64), `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty`, `logit_bias`, `min_tokens`, `allowed_token_ids`, `seed`, `max_tokens`, `stop`, `stop_token_ids`, `ignore_eos`, `n` ([table](docs/USAGE.md#sampling-parameters)) |
| Returns HTTP 400 | `logprobs`, `bad_words`, structured output, `thinking_token_budget`: each is hard here, and [IDEAS](docs/IDEAS.md) says why |
| Tool calls | They work, with one caveat: the previous 4-bit format gave a well-formed call in 39% to 46% of requests for a minimal one-parameter schema in the usual key order (bf16: 91%); with the rotated 4-bit cache of this version it is 92% (n = 200, through the server). The default `--sort-tool-schema` stays on ([details](docs/USAGE.md#tool-calls)) |
| Concurrency | One request decodes at a time; the rest queue |

## Documentation

| I want to... | Read |
|---|---|
| Install it, check it works, fix a problem | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Call the API, tune flags, understand memory and context | [docs/USAGE.md](docs/USAGE.md) |
| Understand the 4-bit KV cache: format, measurements, what it costs, what is not known | [docs/KV_CACHE.md](docs/KV_CACHE.md) |
| See every measurement and its caveats | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Understand the engine and find the code | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Pick up an open question (a list of ideas, not a roadmap) | [docs/IDEAS.md](docs/IDEAS.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md), [changelog](CHANGELOG.md) |

## Credits and license

MIT licensed ([LICENSE](LICENSE)); the model weights are not covered by it. What I took, and did not take, from each third-party project: [NOTICE](NOTICE).
Thanks to the [HyperQwen](https://github.com/syv-ai/hyperqwen) authors for publishing their work and benchmarks.
