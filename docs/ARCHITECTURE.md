# Architecture

The engine is internally called **swift-mk** (library `engine/build/libswiftmk.so`, C API prefix `smk_`, env prefix `SMK_`); the repository is `qwen-warp`.
It targets one thing: single-stream decode of `Swift-1.5-Qwen3.8-27b-INT4` on one RTX 3090 (sm_86, CUDA 13.0, no PDL, no TMA). It was tuned and tested on a 3090 only; no other GPU has been tried.
Batch size is 1 by design.

## Layers

```
server/serve.py            vLLM 0.30 OpenAI frontend (run_server) + our engine-core client
server/smk_core_client.py  in-process vLLM EngineCoreClient: maps vLLM config -> engine settings, FIFO queue, abort, streaming
server/smk_runner.py       engine driver: fast prefill, MTP cycles, sampling, DeltaNet snapshots for multi-turn prefix reuse
engine/spec.py             SpecEngine: MTP k<=3 speculative decoding (+ prompt-lookup tail) on top of Engine
engine/prefill.py          FastPrefill: chunked prefill (vLLM's bundled FlashAttention-2 + flash-linear-attention) writing the engine's caches
engine/engine.py           Engine: weight packing, buffers, CUDA-graph decode step (ctypes into libswiftmk.so)
engine/src/*.cu            kernels: W4A16 GEMV / GEMM (gemv.cu, prefill_w4.cu), DeltaNet (deltanet.cu), attention + KV cache (attn.cu),
                           quantised lm_head (head_q.cu), MTP + sampling (mtp.cu, mtp_sample.cu), prompt lookup (pld.cu)
```

All kernel entry points take a `cudaStream_t` and are capturable in a CUDA graph: no host sync, no dynamic allocation, device-side scalars for position and
context length, so one captured graph serves every step. A decode step is a graph of fused kernels, not a persistent kernel; a graph of per-op kernels was
within 2.5% of a persistent kernel for the GEMV chain when this was decided, and it is far easier to maintain.

## Model conventions the kernels share

- Activations bf16, row-major `[T][dim]`, T <= 8 rows per step (T = 1 plain decode, 1 + k for verification).
- Weights W4: `uint32 [N][K/8]` (nibble e of word j = column 8j + e), bf16 scales `[N][K/128]`, groups of 128 contiguous columns.
- Gated DeltaNet layers: fp32 state `[slot][48 v-heads][128][128]`, bf16 conv state; a ring of slots lets a failed speculative row roll back.
- 16 full-attention layers, 4 KV heads, head dim 256; KV is written positionally (rows write `pos0 + t`; rejected rows are overwritten by the next step, no rollback).
- RMSNorm is zero-centred (`x * rsqrt(mean(x^2) + 1e-6) * (1 + w)`) except the DeltaNet gated norm.

## Speculative decoding

- **MTP**: the checkpoint's own MTP layer, packed to int4 offline, plus a restricted-vocabulary int4 draft head (`engine/draft_vocab_*.npy`). One cycle drafts k <= 3
  tokens and verifies them in one pass. Sampling is exact speculative sampling.
- **Prompt-lookup tail** (`pld.cu`): an n-gram proposer on the host adds up to 4 more rows to the verification pass (up to 8 rows). It needs a 9-slot DeltaNet ring
  (+0.62 GiB), which is why the server's `--prompt-lookup auto` only turns it on where the combination was measured to fit (`PLD_MAX_LEN` in `server/smk_core_client.py`).
- **Exact rescoring head**: the draft/verify argmax uses a quantised lm_head (int4 or int8) whose top-R candidates are rescored with the bf16 head, so the output ids
  equal what the bf16 head would give. This is also why the bf16 table can live in host memory (below): only the R candidate rows are read per step.

## 4-bit KV cache (`attn.cu`, `prefill.cu`)

Each token's 256-element head vector (per KV head, separately for K and V) is quantised asymmetrically to 4 bits: 128 bytes of codes plus one `u32` metadata word
(fp16 scale | fp16 minimum), i.e. 4.125 bits per element, no rotation. Layout: codes `[K|V][4][ctx][128]` then meta `[K|V][4][ctx]`.

- K: `S = s_k * (q . code) + min_k * sum(q)`, with `sum(q)` precomputed once per query (workspace `WS_QSUM`).
- V: the scale is folded into the softmax probabilities, and `sum(p * min)` is accumulated next to the softmax denominator.
- Nibble unpack is a `lop3` trick plus `sub.f16x2`; the decode kernel's writer and the prefill writer produce byte-identical caches.

## Host-pinned tables (`Engine(host_tables=True)`, `--host-tables`)

The bf16 embedding and the rescoring lm_head (4.7 GiB) are moved to pinned host memory before the engine is created, so the VRAM they occupied goes to KV cache.
Decode output is bit-identical. Refused with the plain bf16 head and MTP `head_mode 1`.

## Server behaviour

- One request is decoded at a time; the rest queue FIFO. A client disconnect aborts the running request.
- Multi-turn prefix reuse: DeltaNet states are snapshotted at chat-turn boundaries (`--snapshots N`, 151 MB of pinned host RAM each); a follow-up turn
  reuses the cached prefix and reports `cached_tokens`.
- Supported: chat and text completions, streaming, stop strings, seeds and the usual sampling parameters, tool calls (`qwen3_coder` parser), reasoning output (`qwen3` parser).
  Text only: the vision tower is not loaded.

## Tests (`tests/`)

All three need only the checkpoint. Kernel-level parity tests against a reference decoder were part of development and are not shipped, because they depend on
tensor dumps from a CPU reference implementation that is not in this repository.

- `test_kv_dtypes.py`: bf16 / fp8 / int4 perplexity on a fixed text; gates int4 at +3% and fp8 at +1.5%.
- `test_host_tables.py`: host tables bit-identical to GPU tables; prints the VRAM and speed difference.
- `test_server_smoke.py`: starts the server and checks greedy ids against the direct engine, streaming, stop strings, abort, queueing, sampling, tool calls, prefix reuse.
