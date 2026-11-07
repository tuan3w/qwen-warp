# qwen-warp

**Fast Qwen3.8-27B on one RTX 3090.** An OpenAI-compatible server on a custom CUDA engine: 262k context, several users at once, text and images.

- **Fast for one user:** 120 to 190 tok/s on short prompts, up to 277 with the DFlash2 drafter, and still 100 tok/s with a 190k-token prompt.
- **Fast for many users:** requests decode together, so 4 users get about 2.7x the total throughput of one.
- **Long context:** the model's full 262,144 tokens fit in 24 GB (4-bit KV cache).
- **Exact:** speculative decoding never changes the output distribution.

## Speed

| On one RTX 3090 (350 W, greedy) | tok/s |
|---|---|
| One user, short code prompt | 181 (`--drafter dflash2`: 277) |
| One user, 190k-token prompt | 102 |
| 4 users at once, in total | 425 |
| 8 users at once, in total | 441 |

Against [HyperQwen](https://github.com/syv-ai/hyperqwen) on the same card: 1.2x to 1.3x faster at 32k to 130k context, 3.6x faster at 190k, and level to 13% ahead with the same DFlash2 drafter on short prompts. Every measurement, method and caveat: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Quick start

You need Linux, an RTX 3090, an NVIDIA driver 580 or newer, and the weights (19 GB). Details and troubleshooting: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

```bash
# weights (accept the model's licence on Hugging Face first)
uvx --from huggingface_hub hf download ukisai/Swift-1.5-Qwen3.8-27b-INT4 --local-dir ~/models/swift-1.5-int4

# A. Docker (needs the NVIDIA container toolkit)
docker compose up --build

# B. native (needs the CUDA 13.0 toolkit and uv)
make setup
uv run server/serve.py --profile long --port 8000
```

Then call it like any OpenAI server:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "swift", "messages": [{"role": "user", "content": "Write a binary search in Go."}], "max_tokens": 2000}'
```

The model thinks before it answers, and thinking counts against `max_tokens`. Give it room, or turn thinking off per request with `"chat_template_kwargs": {"enable_thinking": false}`.

## Pick a profile

| You want | Flags (Docker: `PROFILE=` in `.env`) |
|---|---|
| Long context, one user (the Docker default) | `--profile long` |
| Several users at once, with images | `--profile multiuser` |
| Fastest single user on short prompts | `--drafter dflash2` ([setup](docs/USAGE.md#the-dflash2-drafter---drafter-dflash2-experimental)) |
| The model's full 262k context | `--profile long --max-model-len 262144 --prompt-lookup 0` |

The server checks free GPU memory at start and refuses a configuration that does not fit. All flags: [docs/USAGE.md](docs/USAGE.md).

## What it supports

- **API:** chat and text completions, streaming, tool calls, reasoning output, and the common sampling parameters ([list](docs/USAGE.md#sampling-parameters)). `logprobs` and structured output return HTTP 400.
- **Images:** `image_url` parts in chat messages, up to 4 per prompt ([details](docs/USAGE.md#images)). Video is not supported.
- **Hardware:** RTX 3090 only, one GPU, tested with [Swift-1.5-Qwen3.8-27B INT4](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4).

## Documentation

| I want to... | Read |
|---|---|
| Install it, check it works, fix a problem | [GETTING_STARTED](docs/GETTING_STARTED.md) |
| Call the API, tune flags, understand memory | [USAGE](docs/USAGE.md) |
| See every measurement and its caveats | [PERFORMANCE](docs/PERFORMANCE.md) |
| Understand the engine and find the code | [ARCHITECTURE](docs/ARCHITECTURE.md) |
| Learn about the 4-bit KV cache | [KV_CACHE](docs/KV_CACHE.md) |
| Pick up an open question | [IDEAS](docs/IDEAS.md) |
| Contribute, or see what changed | [CONTRIBUTING](CONTRIBUTING.md), [CHANGELOG](CHANGELOG.md) |

## License

MIT ([LICENSE](LICENSE)); the model weights are not covered. Third-party credits: [NOTICE](NOTICE). Thanks to the [HyperQwen](https://github.com/syv-ai/hyperqwen) authors for publishing their work and benchmarks.
