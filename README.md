# qwen-warp

**Run Swift-1.5-Qwen3.8-27B with a 262,144-token context on one RTX 3090.**

A single-purpose CUDA decode engine behind an OpenAI-compatible API. A 4-bit KV cache and host-pinned tables fit the model's full context in 24 GB, and MTP
speculative decoding keeps decoding at 91 to 128 tok/s even with 32k to 190k tokens in the prompt. Point any OpenAI client, or a coding agent, at it.

> **Scope, stated up front.** One model, one GPU (RTX 3090, tested only there), one request decoded at a time, text only. The code is shared as a reference to learn
> from and borrow ideas from, not as a general inference stack.

## Why try it

| If you want to... | What you get | Evidence |
|---|---|---|
| Keep a whole repository or a long document in context on a 24 GB card | 262,144 tokens in 22.4 GiB; the 4-bit cache costs 16.5 KiB per token against 64 KiB for bf16 | [Memory](docs/PERFORMANCE.md#1-memory-how-much-context-fits) |
| Stay fast when the prompt is long | 91 to 128 tok/s from 32k to 190k tokens, 105 to 166 tok/s on short prompts | [Speed](#numbers-at-a-glance) |
| Not pay for it in quality | Coding pass@1 244 -> 241 of 283 tasks (not significant); perplexity +0.26% at 98k tokens | [Quality](docs/PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache) |
| Plug in what you already use | vLLM's own OpenAI frontend: chat and text completions, streaming, tool calls, reasoning output, stop strings, seeds | [Usage](docs/USAGE.md) |
| Learn how it is built | A small CUDA engine (graph of fused kernels, W4A16, exact speculative sampling) with a file-by-file map | [Architecture](docs/ARCHITECTURE.md) |

## Is it for you?

| Yes, if | Not (yet), if |
|---|---|
| You have an **RTX 3090** (24 GB), Linux, driver 580 or newer | You have another GPU: the kernels target sm_86 and nothing else was tried |
| You want **Swift-1.5-Qwen3.8-27B INT4** with long prompts: coding agents, big repos, long documents | You want a different model, vision input, or batching: one stream at a time, text only |
| You are happy to read the code and adapt it | You need a drop-in, general-purpose serving stack: use vLLM or a similar project |

## Quick start

You need an NVIDIA driver 580 or newer, the model weights (19 GB; the checkpoint [`ukisai/Swift-1.5-Qwen3.8-27b-INT4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4)
is gated and has its own licence, so you accept it on Hugging Face), and one of the two paths below. Full prerequisites, disk sizes and troubleshooting:
[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

| | **A. Docker** | **B. Native (uv)** |
|---|---|---|
| Needs | Docker with the NVIDIA container toolkit | CUDA 13.0 toolkit (`nvcc`), [uv](https://docs.astral.sh/uv/), a C compiler |
| Disk | image about 24 GB, plus the weights | about 8 GB environment, plus the weights |
| First setup | about 8 GB of wheels download during the build | `make setup`: 27 minutes on a clean machine, almost all downloads |
| Start | `docker compose up --build` | `uv run server/serve.py --profile long` |

```bash
# 0. the weights (both paths); accept the licence on the Hugging Face model page first
uvx --from huggingface_hub hf auth login
uvx --from huggingface_hub hf download ukisai/Swift-1.5-Qwen3.8-27b-INT4 --local-dir ~/models/swift-1.5-int4   # 19 GB; or scripts/fetch_model.sh

# A. Docker
docker compose up --build                    # serves on http://localhost:8000, long-context profile

# B. native
make setup
uv run server/serve.py --profile long --port 8000
```

The server is ready after 23 to 40 s (native, warm caches); the first start in Docker took about 80 s in our test because the compile caches are cold. Then try it:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "swift",
  "messages": [{"role": "user", "content": "Write a binary search in Go."}],
  "max_tokens": 600}'
```

The model **thinks before it answers** by default, and the thinking tokens count against `max_tokens`. If you get a truncated or empty answer, raise `max_tokens` or switch
thinking off per request with `"chat_template_kwargs": {"enable_thinking": false}`. More in [docs/USAGE.md](docs/USAGE.md).

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")   # any key unless you started the server with --api-key
r = client.chat.completions.create(model="swift", messages=[{"role": "user", "content": "Hello"}], max_tokens=600)
print(r.choices[0].message.content)
```

Check your install with `make test` (about 6 minutes on an RTX 3090 with the weights): it tests the KV cache, the host tables and the server end to end.

## Pick a mode

| Goal | Flags | Context |
|---|---|---|
| **Long context** (recommended) | `--profile long` | 196,608 |
| The model's full context | `--profile long --max-model-len 262144 --prompt-lookup 0` | 262,144 |
| Fastest at short context | defaults (bf16 KV cache) | 28,672 |
| fp8 KV cache | `--kv-cache-dtype fp8 --max-model-len 53248` | 53,248 |

`--profile long` means a 4-bit KV cache plus `--host-tables`, which keeps the 4.7 GiB embedding and head tables in pinned host memory (bit-identical decode) so the VRAM goes to
the cache. With Docker, set `PROFILE=` and `EXTRA_ARGS=` in a `.env` file instead of typing flags (see [.env.example](.env.example)). Every flag: [docs/USAGE.md](docs/USAGE.md#server-flags).

## Numbers at a glance

Same RTX 3090 at 350 W, same checkpoint and client, batch size 1, greedy decoding. **Single runs on one machine**; `n/s` = not served in that run.

| Decode, tokens/s by prompt length | 32k | 64k | 100k | 130k | 190k |
|---|---|---|---|---|---|
| qwen-warp, 4-bit KV, `--profile long` | 122 | 117 | 107 | 102 | 94 |
| qwen-warp, 4-bit KV, `--prompt-lookup 0` | 120 | 128 | 103 | 101 | 91 |
| qwen-warp, fp8 KV | 117 | 117 | 108 | 98 | n/s |
| *reference:* HyperQwen `mtp-long` (fp8/int8 KV) | 94 | 88 | 80 | 68 | n/s |
| *reference:* HyperQwen `mtp-huge` (4/2-bit KVarN) | 58 | 53 | 39 | 31 | 27 |

| Short prompts, tokens/s | code | prose | math |
|---|---|---|---|
| qwen-warp, 4-bit KV | 166 | 105 | 156 |
| *reference:* HyperQwen `mtp-long` | 118 | 79 | 121 |

| Quality, 283 coding tasks (pass@1) | passed |
|---|---|
| bf16 KV (reference) | 244 |
| qwen-warp, 4-bit KV | 241 (-1.1 points, p = 0.45, not significant) |

- **HyperQwen is a reference point, not a competitor**: a different, more general project. Its numbers describe one build only: image
  `ghcr.io/syv-ai/hyperqwen@sha256:00713ba1e04b68db1be9eba2b15b579250f9fb0c29ed61add40c750228fc0cf8` (revision `bd6c5e2f11628483109031d99d1cc4a7d22b1a26`, built 2026-10-04).
  Prefill is within about 10% of it. A newer build may be faster or slower. Its `mtp-long` needed `GPU_UTIL=0.88` on this desktop card; its `dflash2-long` preset failed to start here and is not compared.
- Cold prefill: 1,830 tok/s at 16k, about 960 tok/s at 262k, so a 190k prompt takes about 3 minutes. Follow-up turns reuse the cached prefix.
- Our 4-bit arm missed a planted needle at 100k in the speed sweep (fp8 and both HyperQwen presets found it); output text drifts from bf16 earlier than with fp8 (67% token-identical on HumanEval+ against 88%).
- **Not measured:** multi-turn agentic task success (tool calls, edit-and-test loops), other GPUs, power limits other than 350 W.

All tables, methods and caveats: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Known limits

| Limit | Detail |
|---|---|
| Hardware | RTX 3090 only (sm_86 kernels, CUDA 13.0 only); no other GPU was tested |
| Model | Swift-1.5-Qwen3.8-27B INT4, text only |
| Concurrency | One request decodes at a time; the rest queue in order |
| Sampling | `temperature`, `top_p`, `top_k` (up to 64), `seed`, `max_tokens`, `stop`, `stop_token_ids`, `ignore_eos`, `n` work; other sampling and structured-output parameters return HTTP 400 ([list](docs/USAGE.md#sampling-parameters)) |
| Long prompts | Slow to prefill: about 3 minutes at 190k tokens |
| 4-bit cache | Lossy; see the quality caveats above |

## Documentation

| I want to... | Read |
|---|---|
| Install it, check it works, fix a problem | [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) |
| Call the API, tune flags, understand memory and context | [docs/USAGE.md](docs/USAGE.md) |
| See every measurement and its caveats | [docs/PERFORMANCE.md](docs/PERFORMANCE.md) |
| Understand the engine and find the code | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Pick up an open question | [docs/IDEAS.md](docs/IDEAS.md) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) and the [changelog](CHANGELOG.md) |

## Ideas to explore

A list of things we may try, not a roadmap: nothing is promised, scheduled or in progress. Details and evidence: [docs/IDEAS.md](docs/IDEAS.md).

| Idea | Where it stands |
|---|---|
| A rotated 4-bit KV cache | About 2x lower perplexity cost on this 27B (+0.10% and +0.13% against +0.27% and +0.26%), not the 10x seen on a 2B proxy; no kernel, no speed number |
| Why the 4-bit cache missed a needle at 100k | One miss in two experiments; cause unknown |
| A better quantized model, if one appears | The W4 kernels assume uniform symmetric int4, group size 128 |
| A non-uniform weight codebook | +2.8 points top-1 fidelity on Qwen3-4B, but on the 27B the extra weight bytes alone would slow the decode step by about 17% |
| Multi-turn agentic task success per KV dtype | Not measured at all |
| Faster long prefill | About 960 tok/s at 262k |
| Repeating the HyperQwen reference measurement | One pinned build, single runs |

Out of scope: other models, other GPUs, batching.

## Credits and license

MIT licensed ([LICENSE](LICENSE)); the model weights are not covered by it. What was and was not taken from each third-party project: [NOTICE](NOTICE).
Measured next to [HyperQwen](https://github.com/syv-ai/hyperqwen) on identical hardware (pinned build above); we are not competing with it: it is a general serving stack
for several models and GPUs, and this engine is a single-purpose reference for one model on one card. Thanks to its authors for publishing their work and benchmarks.
