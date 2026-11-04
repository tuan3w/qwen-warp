# qwen-warp

**Fast Qwen3.8-27B on one RTX 3090, with a 200k-token context.**

I built qwen-warp to get the most out of my own RTX 3090: a custom CUDA decode engine behind an OpenAI-compatible API. It decodes **94 to 133 tokens/s with 32k to 262k tokens in the prompt** and fits the model's full 262,144-token context in 24 GB.
I tested it only with [Swift-1.5-Qwen3.8-27B INT4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4). Other Qwen3.8 27B variants may work; I have not tried them.
One GPU (RTX 3090), one request at a time, text only.

## Numbers

Same card at 350 W, greedy, batch size 1, one run per cell. Re-measured on this code on 2026-11-04 (client and prompts: [method](docs/PERFORMANCE.md#3-speed)).

| Decode, tokens/s, by prompt length | 32k | 64k | 100k | 130k | 190k | 262k |
|---|---|---|---|---|---|---|
| **qwen-warp** `--profile long` (4-bit KV cache, prompt-lookup tail) | 124 | 119 | 115 | 108 | 102 | - |
| **qwen-warp** 4-bit KV cache, tail off | 124 | 133 | 109 | 105 | 96 | 94 |
| **qwen-warp** fp8 KV cache | 123 | 123 | 112 | 108 | - | - |
| *reference:* HyperQwen `mtp-long` | 94 | 88 | 80 | 68 | - | - |

`-` = not run (fp8 stops at 130k; the tail-on row stops at the profile's 196,608).

| Short prompts, tokens/s | code | prose | math |
|---|---|---|---|
| **qwen-warp** default (bf16 KV cache) | 172 | 110 | 172 |
| **qwen-warp** 4-bit KV cache | 170 | 108 | 162 |
| *reference:* HyperQwen `mtp-long` | 118 | 79 | 121 |

| | |
|---|---|
| Context | 262,144 tokens fit in about 22.5 to 22.7 GiB in use, desktop included (`--host-tables` moves 4.8 GiB of tables to host memory; the 4-bit cache costs 16.5 KiB per token, bf16 64 KiB) |
| Prefill | 1,890 tok/s at 16k, 1,100 at 190k (3 minutes), 963 at 262k (4.5 minutes); follow-up turns reuse the cached prefix |
| Ready after | 21 to 30 s |
| Quality of the 4-bit cache | perplexity +0.26% at 98k; coding pass@1 241 against 244 of 283 tasks (not significant). It missed the planted needle at 100k in the speed sweeps (fp8 found it every time). Tool calls are sensitive to it: [what was measured](docs/USAGE.md#tool-calls) |
| Not measured | multi-turn agent task success, other GPUs, other power limits |

HyperQwen rows are a reference point, not a competitor: another project, measured on 2026-11-03 with the same prompts and client, image
`ghcr.io/syv-ai/hyperqwen@sha256:00713ba1e04b68db1be9eba2b15b579250f9fb0c29ed61add40c750228fc0cf8` (revision `bd6c5e2f11628483109031d99d1cc4a7d22b1a26`, built 2026-10-04). A newer build may differ.
Every table, method and caveat: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

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
  "model": "swift", "messages": [{"role": "user", "content": "Write a binary search in Go."}], "max_tokens": 600}'
```

The model thinks before it answers, and the thinking tokens count against `max_tokens` (the text comes back in the `reasoning` field). Raise `max_tokens`, or switch thinking off per request with
`"chat_template_kwargs": {"enable_thinking": false}`. More in [docs/USAGE.md](docs/USAGE.md).

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
| Tool calls | They work, with one caveat: with the 4-bit cache a minimal one-parameter schema in the usual key order gave a well-formed call in 39% of requests (bf16: 91%). The default `--sort-tool-schema` lifts it to 87% ([details](docs/USAGE.md#tool-calls)) |
| Concurrency | One request decodes at a time; the rest queue |

## Documentation

| I want to... | Read |
|---|---|
| Install it, check it works, fix a problem | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Call the API, tune flags, understand memory and context | [docs/USAGE.md](docs/USAGE.md) |
| See every measurement and its caveats | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Understand the engine and find the code | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Pick up an open question (a list of ideas, not a roadmap) | [docs/IDEAS.md](docs/IDEAS.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md), [changelog](CHANGELOG.md) |

## Credits and license

MIT licensed ([LICENSE](LICENSE)); the model weights are not covered by it. What I took, and did not take, from each third-party project: [NOTICE](NOTICE).
Thanks to the [HyperQwen](https://github.com/syv-ai/hyperqwen) authors for publishing their work and benchmarks.
