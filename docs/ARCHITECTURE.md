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
| | `engine/src/dflash.cu`, `engine/dflash.py` | DFlash2 block drafter (experimental, greedy only): 5-layer W4A16 drafter + candidate selector inside the cycle graph |
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

## DFlash2 block drafter (experimental, greedy only)

`engine/dflash.py` (`DflashEngine`, a `SpecEngine` that keeps the MTP path) and `engine/src/dflash.cu` (`include/dflash.h`) run the 7-token block drafter of `incoai/Qwen3.8-27B-DFlash2` inside the engine.
One CUDA graph per cycle, no host work between cycles: the host launches n graphs and reads the output ring, like the MTP cycle.

| Stage | What runs |
|---|---|
| verify | the engine step over T = 8 rows `[anchor, d1..d7]` (quantised head + exact rescoring), same kernels as `smk_engine_verify` |
| accept | one thread: leading drafts equal to the previous row's argmax, `spec_finish` bookkeeping (pos, DeltaNet ring slot, anchor), output ring, per-cycle log, the drafter's new context rows (`a` rows at the positions of the accepted verify rows) |
| ctx append | the debug taps of target layers 5, 19, 33, 47, 61 of the new rows -> `fc` (W4 GEMV, K = 25600) -> `hidden_norm` -> K/V of all 5 drafter layers in ONE fused W4 GEMV (the context path does not depend on the layer input) -> per-head norm + RoPE -> positional write into the drafter KV cache (`[5][8 heads][ctx_max + 8][128]` bf16 K and V; rows past the committed length are overwritten by the next cycle, no rollback) |
| noise block | `embed([anchor] + [mask] * 7)` with the target's table, then per layer: norm, conv-kernel projection (bf16 GEMV), dynamic grouped conv, q/k/v W4 GEMV, qk-norm + RoPE, attention over cache rows `[max(0, start - 2047), start + 8)` (the noise rows' K/V are written at `start..start + 7`, so one contiguous range; non-causal), o GEMV, conv + residual, norm, conv, gate|up (SiLU epilogue) + down, conv + residual; final norm |
| head + selector | quantised lm_head over rows 1..7 + exact bf16 rescoring of the top-R (R = 32), top-16 per row in `torch.topk(sorted=False)` order, selector hidden projection, the greedy path `score = unary + <pred_codebook[pred] * hproj, succ_codebook[cand]>` over 7 sequential positions (one kernel) -> `ds[TOK + 1 ..]` |

Weights: the W4A16 compressed-tensors checkpoint (pack-quantized, group 128; fp16 scales are rounded to bf16 because the engine kernels take bf16) is repacked with `pack.repack_w4`; every GEMM is `smk_w4r_gemv`.
Device memory (measured on the server's Runner, [Usage](USAGE.md#the-dflash2-drafter---drafter-dflash2-experimental)): the drafter takes 1.4 GiB (1.15 GiB with `codebooks_host=True`: the two 121 MiB selector codebooks stay in pinned host memory) on top of the engine with the 9-slot ring, plus 20 KiB per context position of drafter KV; the W4 weights are 1.0 GiB, the rest is the output ring and log, the selector scratch, the tap staging buffers of the fast prefill (105 MiB) and the captured graphs of both draft routes.
Limits: needs `tmax = 8` and the int4 head; greedy only (no sampling, no logit processors); `DflashEngine` keeps the bf16 lm_head table on the device when the free VRAM covers the rest of the engine (2.5 GiB, about 0.15 GiB spare at a 31k context on bf16 KV; it falls back to the pinned host table; the server's `Runner` passes `lm_head_host=True` so that its memory model holds); an optional device n-gram lookup over the token history (`SMK_DF_LOOKUP=1`) is off by default; the drafter KV is indexed by absolute position, so long contexts pay 20 KiB/token although the window is 2048 (a ring would cap it at 42 MiB; not built: `--profile long` does not fit with the drafter).
`tests/test_dflash.py` checks the ids against the T = 8 greedy decode; `tests/test_dflash_server.py` does the same for the server's path (fast prefill, tap capture, router).

### In the server (`--drafter dflash2`)

The drafter's K/V rows at position p are a function of the target's residual stream at the five tap layers **of row p alone** (`fc` and the K/V projections have no mixing across rows), so any row can be rebuilt from the taps without the
target's KV cache. The fast prefill is layer-major, and the residual entering layer l is the engine's debug tap slot l, so `FastPrefill.tap` (read-only: nothing it writes is touched) copies, after the ln1 norm of layers 6, 20, 34, 48 and 62,
the rows of the drafter's window (the last 2,047 prompt rows, minus what a resume skips) into a 105 MiB staging buffer; `dflash.TapCapture.flush` then appends them in 8-row chunks through the same `smk_dflash_prefill_chunk` the engine-direct
prefill uses (no kernel change). The folded last row of a short final segment gets its taps from the batch (`take_last`), segments of fewer than `min_fast` rows run as 4-row engine steps that leave the taps in place already.
Cost: 1.3% on the cold prefill of a 12k-token prompt and 3 ms on a cached follow-up turn ([Performance](PERFORMANCE.md#5-dflash2-block-drafter-experimental)).

`Runner.run` routes a request to the block drafter iff `drafter == dflash2`, `temperature` is 0, no logit processor is active (`GenParams.procs()` is `None`) **and the drafter window is on the device**; everything else runs the MTP path exactly
as before (`--drafter mtp` is byte-identical to the previous release). The Runner tracks the valid interval `[df_lo, df_hi)` of the drafter rows of the device lineage's prompt: every prefill captures its window whatever route decodes the request
(so a sampled turn in the middle of a conversation does not break the next greedy turn), a device resume keeps the rows below its resume position, a cold prefill or a **host-tier restore** invalidates them (the drafter cache then holds another
conversation's rows, and nothing carries the window of a spilled lineage). A request whose window is not covered after the prefill is decoded by the MTP path; the output is never wrong either way (the greedy accept is an argmax equality of the T = 8
verify, the drafter only changes how many tokens a cycle commits). MTP-layer KV is still computed by the fast prefill for every request, so a later sampled turn on the same lineage reads it.

## Image input

| Piece | How |
|---|---|
| ViT (`engine/vision.py`) | the checkpoint's bf16 Qwen3.5 vision tower (27 blocks, hidden 1152, 16 heads of 72, 2x2 merger to 5120), batched over all patches of an image: the Conv3d patch embed as one GEMM, the 48x48 position table resampled bilinearly, axial 2-D rotary, per-image non-causal FlashAttention-2 (vLLM's binding), the residual stream and LayerNorms in fp32 (transformers keeps them in bf16; against an fp32 forward its bf16 model has tokens at a cosine of 0.96, ours 0.98). Weights on the GPU, or (`weights="host"`) the blocks' GEMM weights in pinned memory copied block by block on a side stream into two buffers, bit-identical. One call per image |
| Placeholder rows | `FastPrefill.mm` (`MMRows`): after the token embedding, the rows of an image's placeholder tokens are overwritten with the ViT rows; the MTP pairs whose next token is a placeholder take the ViT row as their input embedding (vLLM's MTP proposer feeds the merged embeddings) |
| MRoPE | `vision.mrope_positions` == vLLM's `_get_mrope_input_positions`. The prefill writer `k_attn_prep` takes an optional `[T][3]` table: rotary frequency l uses axis T, H or W (`mrope_axis`: section [11, 11, 10], interleaved); the cache row stays the KV position. After the last image every axis is KV position + delta, so decode (target attention prep and the MTP layer's) ropes at `*pos + ds[DS_RDELTA]`; the Runner sets `DS_RDELTA` per request (0 for text, so text is bit-identical to 0.7.0) |
| Prefix cache | lineages and the host store match on `vision.cache_key`: placeholder rows carry a hash of the image's mm hash (negative ids), so a different image diverges at its first row and the same image resumes past it; the ViT runs only for images with rows after the resume point |
| Server | `serve.py` keeps vLLM's multimodal frontend on (processor, chat template, `--limit-mm-per-prompt`, video limit 0); `SwiftCoreClient` fills the frontend's processor-cache holes with vLLM's receiver cache, turns each image feature into `ImageInput(offset, grid, mm hash, pixels)` and refuses video, oversize images and mismatched placeholders with a 400 |
| DFlash2 | image requests take the MTP route: the block drafter's rope has no MRoPE offset and it was trained on text |

Memory (measured, `smk_core_client.VRAM_VISION_*`): +888 MiB with the weights on the GPU, +154 MiB streamed, at 8,100 patches; the ViT's activation peak (333 MiB at 8,160 patches) mostly reuses the fast prefill's transient pool.

## The 4-bit KV cache

Each token's 256-element head vector (per KV head, separately for K and V) is rotated by the orthonormal Walsh-Hadamard transform, then quantised to 4 bits with a symmetric companded 16-level grid:
128 bytes of codes plus one `u32` of metadata (four 8-bit logarithmic scale codes, one per group of 64 rotated coefficients, `s = 2^((e - 176) / 12)`), i.e. 4.125 bits per element. Layout per layer:
codes `[K|V][4][ctx][128]`, meta `[K|V][4][ctx]`, then two small fixed buffers of exact fp16 rows (rotated basis): the first 4 positions `[2][4][4][256]` and a ring of recent positions `[2][4][512][256]` (slot `pos & 511`).
Format, measurements and costs: [KV cache](KV_CACHE.md). Definitions: `engine/src/kvrot.cuh`.

| Part | How |
|---|---|
| Writers | the decode kernel's writer (`attn_prep_i4_kernel`, 8 warps, one scale candidate per warp) and the prefill writer (`k_attn_prep`) rotate (a 256-point transform over one warp with shuffles), pick each group's scale by minimum squared error over a few candidates around the largest magnitude, and write the same bytes; the writers do not know the decoder gain |
| Scores and output | `q` is rotated the same way at prep, so scores equal the unrotated ones; the attention output is un-rotated once in the reduce kernel. The readers decode nibbles with `fp16x2` arithmetic (`L(n) = u (1 + 0.003 u^2)`) and multiply each decoded scale by 1.0075 |
| Exact sinks and ring | positions 0-3 are read from the sink rows; the split kernel stops 256 keys before the end of the context and attends the ring as 16 extra fp16 tiles (no dequantisation), so it runs in parallel with the other splits; the reduce kernel only merges the splits and un-rotates |
| Kernel rows | `attn_split_i4_kernel` serves every int4 launch, 1 to 8 rows (teams of d-split warps, query tiles in groups, so each K/V tile is dequantised once per group) |
| Snapshots | the sink and ring rows are state outside the per-token cache, so every prefix snapshot saves and restores them (`kv_aux_*` in `server/smk_runner.py`, 32 MB of pinned host RAM per slot); without that, a resumed generation diverges |

What it costs and gains: [KV cache](KV_CACHE.md) and [Performance](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache).

## Host-pinned tables

`Engine(host_tables=True)` (`--host-tables`) moves the bf16 embedding and the rescoring lm_head (4.7 GiB) to pinned host memory before the engine is created, so the VRAM they occupied goes
to the KV cache. Decode output is bit-identical (`tests/test_host_tables.py`). It is refused with the plain bf16 head and with MTP `head_mode 1`, which would stream a whole table over PCIe every step.

## Server behaviour

| Behaviour | Detail |
|---|---|
| Concurrency | one request decodes at a time; the rest queue FIFO; a client disconnect aborts the running request |
| Start-up memory check | the server reads the free GPU memory at start, compares it with a fitted model of what the engine needs (fixed part + KiB per token x length) and refuses a length that does not fit before any weight is loaded ([Usage](USAGE.md#memory-and-context)) |
| Prefix reuse | DeltaNet states are snapshotted at the last two cut points of a prefill plan that depends on the prompt only, and at the end of the system prompt (`--snapshots N`, 151 MB of pinned host RAM each, 183 MB with the 4-bit cache, which also saves its exact sink and ring rows); a follow-up turn reuses the cached prefix and reports `cached_tokens`; a resumed prefill is bit-identical to a cold one |
| Layer-major prefill groups | the prefill plan cuts a segment at every chat-turn boundary and each segment dequantizes every layer's int4 weights again; `FastPrefill._group` runs consecutive segments (total rows <= the 4,096-row scratch, a group ends at each snapshot position, `Runner.plan_groups`) layer by layer so each layer is dequantized once per group. Same kernels and row counts per segment, so bit-identical to `_segment` one at a time |
| Host prefix cache | the device caches hold one conversation; `server/prefix_store.py` keeps the others in pinned host memory (KV rows of the main cache and of the MTP layer in chunks that grow with the conversation, plus their snapshots, LRU under `--prefix-cache-gb`). `Runner._spill_device` copies the GPU conversation out before a request that does not continue it overwrites it, `Runner._load_entry` copies a stored one back (engine API `swiftmk_kv_range_*`, `smk_spec_mtpkv_range_*`). Design and the SGLang / vLLM ideas behind it are in the research notes of the repository author |
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
