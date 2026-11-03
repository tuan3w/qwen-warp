# qwen-warp

**Fast long-context inference for Swift-1.5-Qwen3.8-27B on a single RTX 3090.**

A CUDA decode engine with a 4-bit KV cache and MTP speculative decoding, behind an OpenAI-compatible API. It holds the model's full
262,144-token context in 24 GB, and on the same card it decodes 1.3x to 1.5x faster than HyperQwen's `mtp-long` and 2.1x to 3.5x faster than its
long-context preset `mtp-huge` (32k to 190k tokens of context; single runs, caveats in [Performance](#performance)).

> **Goal:** push the performance of Qwen3.8-27B (Swift-1.5) on one RTX 3090. The code is shared as a reference to learn from and borrow ideas from, not as a general
> inference stack: there are no plans to support other models or other GPUs.
> Tuned and tested on one RTX 3090 (24 GB) only. One stream at a time, text only.

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [Choosing a mode](#choosing-a-mode)
- [Performance](#performance)
- [How it works](#how-it-works)
- [Limits](#limits)
- [Roadmap](#roadmap)
- [Development](#development)
- [Credits and license](#credits-and-license)

## What you get

- **Long context on one card.** 262,144 tokens fit in 22.4 GiB. The 4-bit cache costs 16.5 KiB per token where bf16 costs 64 KiB.
- **Fast decode, also when the context is long.** 105 to 166 tok/s on short prompts, 91 to 128 tok/s from 32k to 190k tokens of context.
- **Small quality cost.** The 4-bit cache changed coding pass@1 by -1.1 points (244 to 241 of 283 tasks, not significant) and perplexity by about +0.26% at 98k tokens.
- **A normal API.** vLLM's own OpenAI frontend: chat and text completions, streaming, tool calls, reasoning output, stop strings, seeds.

## Quick start

You need Linux, an RTX 3090 (24 GB) with driver 580 or newer, the CUDA 13.0 toolkit (`nvcc`; only 13.0 was tested), [uv](https://docs.astral.sh/uv/),
about 19 GB of disk for the weights and about 5 GiB of pinned host RAM for the long-context mode. The checkpoint
[`ukisai/Swift-1.5-Qwen3.8-27b-INT4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4) is gated and has its own licence; it is not included here.

```bash
make setup                      # uv sync from uv.lock, then build engine/build/libswiftmk.so
uv run hf auth login            # accept the checkpoint's licence on Hugging Face first
scripts/fetch_model.sh          # downloads to ~/models/swift-1.5-int4 (or pass a directory and set $SWIFT_MODEL)

uv run server/serve.py --profile long --port 8000
```

The server is ready 23 to 40 s after start. Then:

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"swift","messages":[{"role":"user","content":"Write a binary search in Go."}],"max_tokens":300}'
```

torch, torchvision and torchaudio are locked to the `+cu130` builds from PyTorch's own index (configured in `pyproject.toml`); everything else comes from PyPI.
`uv.lock` pins all of it.

## Choosing a mode

```
Goal                       Flags                                                       Context
Fastest at short context   (defaults: bf16 KV)                                          28,672
fp8 KV cache               --kv-cache-dtype fp8 --max-model-len 53248                   53,248
Long context               --profile long                                              196,608
Full model context         --profile long --max-model-len 262144 --prompt-lookup 0     262,144
```

`--profile long` means 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`) plus `--host-tables`, which keeps the 4.7 GiB embedding and head tables in
pinned host memory (bit-identical decode) so that VRAM goes to the cache. Explicit flags win over the profile. Other knobs:

- `--prompt-lookup {0,1,auto}`: an n-gram draft tail that needs 0.62 GiB more VRAM. `auto` turns it on only where that combination was measured to fit,
  which is why 262,144 needs `--prompt-lookup 0`.
- `--num-speculative-tokens {0..3}` (default 3), `--snapshots N` (multi-turn prefix reuse), and every stock `vllm serve` frontend flag.

## Performance

Same RTX 3090 at 350 W, same checkpoint, client and prompts, batch size 1, greedy decoding. Single runs; `n/s` = not served in that run.

```
Decode, tokens/s by prompt length           32k   64k  100k  130k  190k
ours int4 KV, --profile long (+ lookup)     122   117   107   102    94
ours int4 KV, --prompt-lookup 0             120   128   103   101    91
ours fp8 KV                                 117   117   108    98   n/s
HyperQwen mtp-long (fp8/int8 KV)             94    88    80    68   n/s
HyperQwen mtp-huge (4/2-bit KVarN)           58    53    39    31    27
Speedup, --profile long vs mtp-long       1.30x 1.33x 1.34x 1.50x   n/s
Speedup, --profile long vs mtp-huge       2.10x 2.21x 2.74x 3.29x 3.48x

Short prompts, tokens/s                    code prose  math
ours int4 KV, --prompt-lookup 0             166   105   156
HyperQwen mtp-long                          118    79   121
Speedup, int4 vs mtp-long                 1.41x 1.33x 1.29x

Coding pass@1, 283 tasks                 passed
bf16 reference                              244
ours int4 KV, --prompt-lookup 0             241   (-1.1 pts vs bf16, p = 0.45, not significant)
HyperQwen mtp-long                          244
HyperQwen mtp-huge                          242
```

"Faster" here means single-stream decode speed. Prefill is within about 10% of HyperQwen (no win), and coding accuracy is within noise across all arms.
Cold prefill is 1,830 tok/s at 16k falling to about 960 tok/s at 262k, so a 190k prompt takes about 3 minutes; later turns reuse the cached prefix.

Read the numbers with these caveats:

- One run per cell on one machine, no repeats, no 250 W run.
- HyperQwen `mtp-long` needed `GPU_UTIL=0.88` because its default runs out of memory on this desktop card (it also drives a monitor). `mtp-huge` ran at its defaults.
- HyperQwen's `dflash2-long` preset failed to start here and is not compared.
- Our 4-bit arm missed a planted needle at 100k in the speed sweep; fp8 and both HyperQwen presets found it.
- Multi-turn agentic task success (tool calls, edit-and-test loops) has **not** been measured.

More tables (memory limits, cold prefill, perplexity by length) and the methods: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## How it works

- **One CUDA graph per step.** Decode is a graph of fused kernels over W4A16 weights, not a persistent kernel; 16 full-attention layers plus gated-DeltaNet layers.
- **Speculative decoding.** The checkpoint's own MTP head drafts up to 3 tokens per step, and an optional n-gram prompt-lookup tail adds more drafts.
  Sampling uses exact speculative sampling, so drafts change the speed, not the output distribution.
- **4-bit KV cache.** Per token and per KV head, asymmetric int4 with an fp16 scale and minimum: 4.125 bits per element, 16.5 KiB per token, no rotation.
- **Host-pinned tables.** The bf16 embedding and rescoring head live in pinned host memory; only the few candidate rows a step needs cross PCIe.
- **Server.** vLLM's OpenAI frontend on an in-process engine client. One request decodes at a time and the rest queue in order.
  DeltaNet state snapshots let follow-up turns resume from the cached prefix.

Details and file-level pointers: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Limits

- RTX 3090 only (sm_86 kernels); no other GPU was tested.
- One model: Swift-1.5-Qwen3.8-27B INT4. Text only, no vision input.
- Batch size 1. Concurrent requests queue.
- Unsupported sampling parameters (`presence_penalty`, `frequency_penalty`, `logprobs`) return HTTP 400.
- Long prompts are slow to prefill (see Performance).
- The 4-bit cache is lossy: output text drifts from bf16's earlier (67% token-identical on HumanEval+ against 88% for fp8) even though task accuracy held.

## Roadmap

The engine is built around one model, and the next gains are in quantization: a better quantized model is expected, and two quantization ideas are prototyped but not in this engine yet.

Done:

- [x] 4-bit KV cache: 262,144-token context in 24 GB
- [x] Host-pinned embedding and head tables (4.7 GiB of VRAM back)
- [x] MTP speculative decoding with an n-gram prompt-lookup tail
- [x] OpenAI-compatible server with multi-turn prefix reuse

To do:

- [ ] Move to the better quantized model when it is released. The W4 kernels assume uniform symmetric int4 with group size 128, so a different format means kernel work.
- [ ] Rotated 4-bit KV cache (Hadamard rotation plus a calibrated per-channel mean, same 4.125 bits). On a Qwen3.5-2B proxy it cut the perplexity cost about 10x (+2.0% to +0.19%).
  Not yet checked on the 27B, and the rotation's kernel cost is unmeasured.
- [ ] Non-uniform weight codebook (16 levels per group): +2.8 points top-1 fidelity over round-to-nearest on Qwen3-4B, unconfirmed on the 27B.
  The fused kernel built so far is about 2x slower than uniform int4 at decode.
- [ ] Measure multi-turn agentic task success (tool calls, edit-and-test loops) for each KV dtype.
- [ ] Int4 KV for the MTP layer (about 3 KiB per token saved).
- [ ] Faster long prefill (about 960 tok/s at 262k context).
- [ ] Prompt lookup beyond 196,608 context with the 4-bit cache, and an explanation for why decode at 262k costs the same as at 196k.
- [ ] Understand the needle miss at 100k with the 4-bit cache.
- [ ] Repeat the HyperQwen comparison (repeats, a 250 W run) and add its DFlash2 preset if it can be started.

Not planned: other models, other GPUs, batching.

## Development

`make test` runs three tests that need only the checkpoint and the GPU: `tests/test_kv_dtypes.py` (perplexity with bf16, fp8 and int4 KV),
`tests/test_host_tables.py` (host tables are bit-identical, VRAM drops 4.7 GiB) and `tests/test_server_smoke.py` (the server end to end, run with the default and
`--profile long`).

```
engine/    CUDA kernels (src/), C headers (include/), Python drivers (engine.py, spec.py, prefill.py, pack*.py, weights.py).
           Internally named swift-mk: libswiftmk.so, smk_* symbols.
server/    serve.py (entry point), the vLLM engine-core client, the engine driver, a small vLLM plugin
tests/     checkpoint-only tests         docs/  architecture and performance notes         scripts/  model download
```

## Credits and license

MIT licensed ([LICENSE](LICENSE)); the model weights are not covered by it. Third-party credits and what was and was not taken from each: [NOTICE](NOTICE).
It was compared against [HyperQwen](https://github.com/syv-ai/hyperqwen) on identical hardware.
