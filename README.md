# qwen-warp

The goal: the fastest LLM serving stack for a single RTX 3090.

## Status: refactoring for new models

qwen-warp started as a hand-written CUDA engine and OpenAI-compatible server for one model, Qwen3.8-27B. That code is
**legacy**. Every new model took weeks of kernel work, so we are replacing it with [`warpc`](warpc/README.md), a compiler
that turns a checkpoint (Hugging Face safetensors or GGUF) into fused CUDA kernels replayed as CUDA graphs. A new model
becomes a graph and a table of kernels instead of a new engine.

- **Legacy engine and server** (`engine/`, `server/`, `Dockerfile`, the tests next to them): frozen at
  [v0.12.0](https://github.com/tuan3w/qwen-warp/releases/tag/v0.12.0), the latest release that has them. Its documentation
  ([README](https://github.com/tuan3w/qwen-warp/blob/v0.12.0/README.md),
  [docs](https://github.com/tuan3w/qwen-warp/tree/v0.12.0/docs)) lives at that tag. The directories are still in this tree
  until warpc replaces them; code comments in them that cite `docs/...` mean the docs at v0.12.0. To run the OpenAI-compatible
  server today, use v0.12.0.
- **warpc** (`warpc/`): the active code. It is a library: it has no HTTP server yet. It reproduces the legacy engine's
  results on the 27B bit for bit, and runs other models on the same machinery.

## Results

A snapshot, November 2026: RTX 3090 (350 W), greedy decoding, one sequence, [vLLM](https://github.com/vllm-project/vllm)
0.34.0, [llama.cpp](https://github.com/ggml-org/llama.cpp) b8889. Numbers go stale; the tools below regenerate them, and
each feature's measurements, method and caveats are in the [CHANGELOG](CHANGELOG.md) and the pull requests that added it.

**GGUF Q4_0 decode, tok/s (context 512)**

| Model | warpc | llama.cpp |
|---|---|---|
| Qwen3-0.6B | 938 | 564 |
| Qwen3-1.7B | 504 | 390 |
| Qwen3-8B | 154 | 145 |
| Llama-3.1-8B | 160 | 153 |

**bf16 checkpoints, tok/s (context 512)**

| Model | warpc | vLLM |
|---|---|---|
| Qwen3-0.6B | 585 | 424 |
| Qwen3-8B | 55 | 51 |
| Llama-3.1-8B | 56 | 53 |

**Qwen3.8-27B INT4 (Swift-1.5), tok/s.** warpc reproduces the legacy engine's tokens and logits bit for bit and matches its
speed. The 64k and MTP rows run the engine's own kernels under warpc; warpc's generated kernels are gated bit-identical
and speed-checked for plain decode up to 16k tokens.

| | warpc | vLLM |
|---|---|---|
| plain decode, 385-token prompt | 51.5 | 47.7 |
| plain decode, 64k prompt, bf16 KV cache | 41.3 | 42.5 |
| plain decode, 64k prompt, 4-bit KV cache | 48.1 | no such cache |
| MTP speculative decoding (3 drafts), 385 tokens | 150 | 64 |
| MTP, 16k tokens | 174 | 98 |
| MTP, 131k tokens | 131 | refuses above about 17.6k |

**Agent loop, 27B, 64k-token prefix, 20 turns.** Median time to first token per turn: 1.34 s (vLLM with prefix caching:
1.93 s). A resumed turn equals a cold prefill token for token; vLLM's diverged at token 217.

**Other models.** Gemma 4 E2B: 6.07 ms per token at a 600-token prompt (vLLM 7.06, llama.cpp 7.70). Inkling-2-Mini:
126.7 tok/s (vLLM cannot load it).

Reproduce with the tools in [`warpc/tools`](warpc/tools) (`matrix.py` for the model tables, `bench.py` for one engine,
`gate27.py` and `compare27.py` for the 27B, `agentloop.py` for the agent loop); each has `--help`.

## Where to go

| | |
|---|---|
| Use or extend warpc | [warpc/README.md](warpc/README.md) |
| Work in this repo (people and coding agents) | [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md) |
| What changed, with measurements | [CHANGELOG.md](CHANGELOG.md) |
| Report a vulnerability | [SECURITY.md](SECURITY.md) |

## License

MIT ([LICENSE](LICENSE)); model weights are not covered. Third-party credits: [NOTICE](NOTICE).
