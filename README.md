# qwen-warp

A single-stream CUDA decode engine and OpenAI-compatible server for **Swift-1.5-Qwen3.8-27B (INT4)** on **one RTX 3090**, with a **4-bit KV cache** that
fits the model's full 262,144-token context in 24 GB. Built for long agentic-coding sessions: one user, long prompts, fast decode.

- 17-19 ms/token plain decode (53-59 tok/s without speculation), 105-166 tok/s with MTP speculative decoding on short prompts (code is fastest).
- 91-128 tok/s decode at 32k-190k tokens of context (int4 KV, MTP + prompt lookup).
- 262,144-token context in 22.4 GiB with `--profile long`; 4-bit KV changed coding pass@1 by -1.1 points (not significant) and perplexity by about +0.26% at 98k.
- vLLM's own OpenAI frontend (chat, completions, streaming, tool calls, reasoning parser) on top of the engine.

Numbers, methods and caveats: [docs/PERFORMANCE.md](docs/PERFORMANCE.md). Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Scope, stated plainly.** Optimized for the **RTX 3090 only** and not tested on any other GPU: the kernels are compiled for sm_86 and tuned (tile shapes, memory budgets,
the measured context limits) on a 24 GB 3090. Other cards may fail to build or run, or run slowly; no other chip was tried. One model (Swift-1.5-Qwen3.8-27B INT4), batch size 1,
text only. It is not a general inference server. Speed and quality figures are single runs on one machine at a 350 W power limit. Multi-turn agentic task success has **not** been measured.

## Performance at a glance

Same RTX 3090 at 350 W, same checkpoint, client and prompts; single runs; `n/s` = not served in that run. Details, caveats and more tables: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

```
Decode, tokens/s by prompt length           32k   64k  100k  130k  190k
ours int4 KV, --profile long (+ lookup)     122   117   107   102    94
ours int4 KV, --prompt-lookup 0             120   128   103   101    91
ours fp8 KV                                 117   117   108    98   n/s
HyperQwen mtp-long (fp8/int8 KV)             94    88    80    68   n/s
HyperQwen mtp-huge (4/2-bit KVarN)           58    53    39    31    27

Short prompts, tokens/s                    code prose  math
ours int4 KV, --prompt-lookup 0             166   105   156
HyperQwen mtp-long                          118    79   121

Coding pass@1, 283 tasks                 passed
bf16 reference                              244
ours int4 KV, --prompt-lookup 0             241   (-1.1 pts vs bf16, p = 0.45, not significant)
HyperQwen mtp-long                          244
HyperQwen mtp-huge                          242
```

Prefill speed is within about 10% of HyperQwen (no win there). Caveats: no repeats and no 250 W run; HyperQwen `mtp-long` needed `GPU_UTIL=0.88` because its default runs out of memory on this
desktop card; its `dflash2-long` preset failed to start here and is not compared; our int4 arm missed a planted needle at 100k in the speed sweep (fp8 and both HyperQwen presets found it).

## Requirements

- Linux, one NVIDIA **RTX 3090** (24 GB), driver >= 580. Other GPUs are untested.
- CUDA 13.0 toolkit (`nvcc` in `/usr/local/cuda/bin`; override with `CUDA_HOME`). Only 13.0 was tested.
- [uv](https://docs.astral.sh/uv/), Python 3.12 (uv fetches it). torch, torchvision and torchaudio are locked to the `+cu130` builds from PyTorch's own index
  (`download.pytorch.org/whl/cu130`, configured in `pyproject.toml`), everything else comes from PyPI; `uv.lock` pins all of it.
- About 19 GB disk for the weights, and about 5 GiB of pinned host RAM with `--profile long`
- The checkpoint [`ukisai/Swift-1.5-Qwen3.8-27b-INT4`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-INT4) (gated, its own licence; not included here)

## Run

```bash
make setup                      # uv sync (pyproject.toml / uv.lock) + build engine/build/libswiftmk.so
uv run hf auth login            # the checkpoint is gated: accept its licence on Hugging Face first
scripts/fetch_model.sh          # -> ~/models/swift-1.5-int4  (or set $SWIFT_MODEL and pass a directory)

uv run server/serve.py --profile long --port 8000    # int4 KV + host tables, 196,608-token context
uv run server/serve.py --port 8000                   # bf16 KV, 28,672-token context (the bf16 baseline)
```

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"swift","messages":[{"role":"user","content":"Write a binary search in Go."}],"max_tokens":300}'
```

Context and memory knobs (any stock `vllm serve` frontend flag also works):

- `--max-model-len N`: up to 262144 with `--profile long --prompt-lookup 0`. The prompt-lookup tail (on by default where measured to fit) needs 0.62 GiB more,
  so `--profile long` defaults to 196,608 with it on.
- `--kv-cache-dtype {auto,bfloat16,fp8,int4_per_token_head}`: `auto` is bf16.
- `--host-tables`: keeps the 4.7 GiB embedding and head tables in pinned host memory (bit-identical decode).
- `--num-speculative-tokens {0..3}` (default 3), `--prompt-lookup {0,1,auto}`, `--snapshots N`.

| goal | flags |
|---|---|
| fastest short context | defaults |
| 53k context, fp8 KV | `--kv-cache-dtype fp8 --max-model-len 53248` |
| long context (up to 196k) | `--profile long` |
| full 262k context | `--profile long --max-model-len 262144 --prompt-lookup 0` |

The server is ready 23-40 s after start (measured 23-26 s default, 27-38 s with `--profile long`, which also loads the tables into pinned memory).

Prompt processing is slow at long lengths (about 1,800 tok/s at 16k falling to about 960 tok/s at 262k, so a 190k prompt takes about 3 minutes);
follow-up turns reuse the cached prefix.

## Tests

`make test` runs `tests/test_kv_dtypes.py` (perplexity bf16 / fp8 / int4), `tests/test_host_tables.py` (host tables are bit-identical) and
`tests/test_server_smoke.py` (the server end to end, default and `--profile long`). They need the checkpoint and the GPU, and nothing else.

## Layout

- `engine/`: CUDA kernels (`src/`), C headers (`include/`), Python drivers (`engine.py`, `spec.py`, `prefill.py`, `pack*.py`, `weights.py`). The engine is internally named
  *swift-mk* (`libswiftmk.so`, `smk_*`).
- `server/`: `serve.py` (entry point), the vLLM engine-core client and the engine driver, and a small vLLM plugin.
- `tests/`, `docs/`, `scripts/`.

## Status and credits

Unofficial: not affiliated with Alibaba, the Qwen team or UkisAI. Most of the code was written by AI coding agents under human direction; the tests and
[measurements](docs/PERFORMANCE.md) are what to trust, not the authorship. MIT licensed (see [LICENSE](LICENSE)); the model weights are not covered by it.
Provenance and third-party credits: [NOTICE](NOTICE). It builds on the ideas in several open projects (see NOTICE) and was compared against
[HyperQwen](https://github.com/syv-ai/hyperqwen) on identical hardware.
