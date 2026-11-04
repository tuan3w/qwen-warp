# Architecture

How the engine is put together and where to read the code. Back to the [README](../README.md); to run it, see [Getting started](GETTING_STARTED.md).

Contents: [In one paragraph](#in-one-paragraph) | [Where to start reading](#where-to-start-reading) | [Repository layout](#repository-layout) | [Layers and files](#layers-and-files) | [Conventions the kernels share](#conventions-the-kernels-share) |
[Speculative decoding](#speculative-decoding) | [The 4-bit KV cache](#the-4-bit-kv-cache) | [Host-pinned tables](#host-pinned-tables) | [Server behaviour](#server-behaviour) | [Design choices](#design-choices-and-what-they-cost)

## In one paragraph

The engine targets one thing: single-stream decode of `Swift-1.5-Qwen3.8-27b-INT4` on one RTX 3090 (sm_86, CUDA 13.0, no PDL, no TMA). It was tuned and tested on a 3090 only;
batch size is 1 by design. It is named **swift-mk** internally (library `engine/build/libswiftmk.so`, C API prefix `smk_`, environment prefix `SMK_`); the repository is `qwen-warp`.
A decode step is one CUDA graph of fused kernels over W4A16 weights; the checkpoint's own MTP head drafts tokens that the same pass verifies; a 4-bit KV cache and host-pinned tables
make room for a 262,144-token context.

## Where to start reading

| If you want to understand... | Read first | Then |
|---|---|---|
| One decode step end to end | `engine/engine.py` (`Engine`) | `engine/src/engine.cu`, `engine/src/gemv.cu` |
| The 4-bit KV cache | [that section](#the-4-bit-kv-cache) | `engine/src/attn.cu`, the writer in `engine/src/prefill.cu` |
| Speculative decoding | `engine/spec.py` (`SpecEngine`) | `engine/src/mtp.cu`, `mtp_sample.cu`, `pld.cu` |
| The gated-DeltaNet layers | `engine/src/deltanet.cu` | the state layout under [conventions](#conventions-the-kernels-share) |
| How a request becomes tokens | `server/serve.py` | `server/smk_core_client.py`, `server/smk_runner.py` |
| Fast prefill | `engine/prefill.py` | vLLM's FlashAttention-2 binding and flash-linear-attention, called at run time |
| What is checked and how | `tests/` | `scripts/check_repo.sh` |

## Repository layout

| Path | What is in it |
|---|---|
| `engine/` | the CUDA kernels (`src/`), C headers (`include/`), Python drivers, the draft-vocabulary lists, and its own `Makefile` |
| `server/` | `serve.py` (entry point), the vLLM engine-core client, the engine driver, the tool-schema hook (`tool_schema.py`), a small vLLM plugin |
| `tests/` | the tests `make test` runs, most of them need the checkpoint and the GPU, a few neither ([what they check](GETTING_STARTED.md#4-verify-it-works)) |
| `scripts/` | `fetch_model.sh` (weights download), `check_repo.sh` (checks that need no GPU, run by CI) |
| `docs/` | this documentation |
| `Dockerfile`, `docker-compose.yml`, `docker/`, `.env.example` | the container image, its compose file, its entrypoint and its settings |
| `Makefile`, `pyproject.toml`, `uv.lock` | `make setup` and `make test`; the pinned Python environment |
| `.github/` | CI (runs `scripts/check_repo.sh`) and the issue template |
| `LICENSE`, `NOTICE`, `CHANGELOG.md`, `CONTRIBUTING.md` | licence, third-party credits, version history, contribution guide |

## Layers and files

| Layer | File | Role |
|---|---|---|
| Server | `server/serve.py` | vLLM 0.30 OpenAI frontend (`run_server`) with our engine-core client |
| | `server/smk_core_client.py` | in-process vLLM `EngineCoreClient`: maps the vLLM config to engine settings, checks the length against the free GPU memory (`vram_need_mib`), rejects unsupported parameters, FIFO queue, abort, streaming |
| | `server/smk_runner.py` | engine driver: fast prefill (a plan that depends on the prompt only), MTP cycles, sampling, DeltaNet snapshots for multi-turn prefix reuse |
| | `server/tool_schema.py` | sorts the keys of every tool's `function.parameters` before vLLM's chat template renders them (`--sort-tool-schema`) |
| | `server/plugin/` | a small vLLM plugin |
| Engine (Python) | `engine/spec.py` | `SpecEngine`: MTP speculative decoding (k <= 3, plus a prompt-lookup tail) on top of `Engine` |
| | `engine/prefill.py` | `FastPrefill`: chunked prefill writing the engine's caches |
| | `engine/engine.py` | `Engine`: weight packing, buffers, the CUDA-graph decode step (ctypes into `libswiftmk.so`) |
| | `engine/pack.py`, `pack_mtp.py`, `weights.py`, `smk.py` | weight and MTP packing, safetensors reader for the compressed-tensors layout, C-API bindings |
| Kernels (CUDA) | `engine/src/gemv.cu`, `prefill_w4.cu` | W4A16 GEMV and GEMM |
| | `engine/src/deltanet.cu` | gated DeltaNet layers |
| | `engine/src/attn.cu` | attention and the KV cache |
| | `engine/src/head_q.cu` | quantised lm_head |
| | `engine/src/mtp.cu`, `mtp_sample.cu`, `pld.cu` | MTP, sampling, prompt lookup |
| | `engine/src/proc.cu`, `proc.h` | request-level logit processors (penalties, `logit_bias`, `min_tokens`, `allowed_token_ids`, `min_p`) inside the sampling graphs |
| | `engine/src/engine.cu`, `misc.cu`, `prefill.cu` | engine glue, norms and misc ops, prefill kernels |

All kernel entry points take a `cudaStream_t` and can be captured in a CUDA graph: no host sync, no dynamic allocation, device-side scalars for position and context length,
so one captured graph serves every step.

## Conventions the kernels share

| Thing | Convention |
|---|---|
| Activations | bf16, row-major `[T][dim]`, T <= 8 rows per step (T = 1 plain decode, 1 + k verification) |
| Weights (W4) | `uint32 [N][K/8]` (nibble e of word j = column 8j + e), bf16 scales `[N][K/128]`, groups of 128 contiguous columns |
| Gated DeltaNet | fp32 state `[slot][48 v-heads][128][128]`, bf16 conv state; a ring of slots lets a rejected speculative row roll back |
| Full attention | 16 layers, 4 KV heads, head dim 256; KV is written positionally (rows write `pos0 + t`; rejected rows are overwritten by the next step, no rollback) |
| RMSNorm | zero-centred: `x * rsqrt(mean(x^2) + 1e-6) * (1 + w)`, except the DeltaNet gated norm |

## Speculative decoding

| Piece | What it does |
|---|---|
| **MTP** | the checkpoint's own MTP layer, packed to int4 offline, plus a restricted-vocabulary int4 draft head (`engine/draft_vocab_*.npy`). One cycle drafts k <= 3 tokens and verifies them in one pass |
| **Exact sampling** | speculative sampling is exact in distribution, so drafts change the speed, not the output distribution. A seeded request is reproducible for a fixed configuration, but another `--num-speculative-tokens` or `--prompt-lookup` draws different tokens ([Usage](USAGE.md#seeds-and-reproducibility)); greedy ids are deterministic for a fixed configuration, and across configurations they can differ at a bf16 near-tie on long prompts |
| **Logit processors** | the processors (`proc.cu`) run on every verify row, with the draft tokens before the row counted as generated, so they stay exact under speculation (stock vLLM refuses `min_p` and `logit_bias` there). Draft proposals ignore them, which lowers acceptance. Order and costs: [Usage](USAGE.md#sampling-parameters) |
| **Prompt-lookup tail** (`pld.cu`) | a host n-gram proposer adds up to 4 more rows to the verification pass (up to 8 rows). It needs a 9-slot DeltaNet ring (+0.56 GiB, 0.67 GiB with `--host-tables`), which is why `--prompt-lookup auto` turns it on only where the length leaves 256 MiB of the free GPU memory next to it (`vram_need_mib` in `server/smk_core_client.py`) |
| **Exact rescoring head** | draft and verify use a quantised lm_head (int4 or int8) whose top-R candidates are rescored with the bf16 head (R = 16 for greedy, 64 for sampling), so the output ids equal what the bf16 head would give. This is also why the bf16 table can live in host memory: only the R candidate rows are read per step |

## The 4-bit KV cache

Each token's 256-element head vector (per KV head, separately for K and V) is quantised asymmetrically to 4 bits: 128 bytes of codes plus one `u32` of metadata
(fp16 scale | fp16 minimum), i.e. 4.125 bits per element, no rotation. Layout: codes `[K|V][4][ctx][128]`, then meta `[K|V][4][ctx]`.

| Part | How |
|---|---|
| K scores | `S = s_k * (q . code) + min_k * sum(q)`, with `sum(q)` precomputed once per query (workspace `WS_QSUM`) |
| V | the scale is folded into the softmax probabilities; `sum(p * min)` is accumulated next to the softmax denominator |
| Unpack | a `lop3` trick plus `sub.f16x2` |
| Writers | the decode kernel's writer and the prefill writer produce byte-identical caches |

What it costs and gains: [Performance](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache). A rotated variant was measured as an idea, not shipped: [Ideas](IDEAS.md).

## Host-pinned tables

`Engine(host_tables=True)` (`--host-tables`) moves the bf16 embedding and the rescoring lm_head (4.7 GiB) to pinned host memory before the engine is created, so the VRAM they occupied goes
to the KV cache. Decode output is bit-identical (`tests/test_host_tables.py`). It is refused with the plain bf16 head and with MTP `head_mode 1`, which would stream a whole table over PCIe every step.

## Server behaviour

| Behaviour | Detail |
|---|---|
| Concurrency | one request decodes at a time; the rest queue FIFO; a client disconnect aborts the running request |
| Start-up memory check | the server reads the free GPU memory at start, compares it with a fitted model of what the engine needs (fixed part + KiB per token x length) and refuses a length that does not fit before any weight is loaded ([Usage](USAGE.md#memory-and-context)) |
| Prefix reuse | DeltaNet states are snapshotted at the last two cut points of a prefill plan that depends on the prompt only (`--snapshots N`, 151 MB of pinned host RAM each); a follow-up turn reuses the cached prefix and reports `cached_tokens`; a resumed prefill is bit-identical to a cold one |
| API | chat and text completions, streaming, stop strings, seeds and the supported sampling parameters, tool calls (`qwen3_coder` parser), reasoning output (`qwen3` parser, field `reasoning`). Details: [Usage](USAGE.md) |

## Design choices and what they cost

| Choice | Why | Cost |
|---|---|---|
| A graph of fused kernels, not a persistent kernel | within 2.5% of a persistent kernel for the GEMV chain when this was decided, and far easier to maintain | a little headroom on the table |
| Batch size 1 | the goal is single-stream decode speed on one card | no batching, concurrent requests queue |
| One GPU generation (sm_86) | kernels use its instruction set directly (mma.sync, cp.async, lop3) | no portability |
| Exact speculative sampling | drafts never change the output distribution | no approximate-acceptance speedup |
| 4-bit KV cache without rotation | simplest format that makes 262k fit | lossy: [rotation was measured](IDEAS.md#a-rotated-4-bit-kv-cache) and only helps about 2x; the well-formed tool-call rate of a minimal schema fell to 39% as clients send it ([Usage](USAGE.md#tool-calls)) |
| Tool schemas rendered with sorted keys (`--sort-tool-schema`, on by default) | lifts that rate to 87% on the 4-bit cache without touching the engine or the sampler | the prompt differs from what a client sent (not its JSON or the call); `off` renders the schema as sent |
