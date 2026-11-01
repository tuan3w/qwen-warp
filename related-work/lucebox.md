---
title: lucebox
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/Luce-Org/lucebox
path: vendor/lucebox
commit: 502b094
branch: main
commit_date: 2026-10-30
---

# lucebox
# lucebox

Luce-Org/lucebox (Apache-2.0): "speculative inference for consumer GPUs". Two unrelated halves: (a) `optimizations/megakernel/`: a BF16 *Qwen3.5-0.8B* (hybrid Gated-DeltaNet) single-kernel decode on RTX 3090, 413 tok/s; (b) `server/`: a ggml/C++ engine that runs *27B-class* hybrids (Qwen3.5/3.6-27B Q4_K_M) on one 3090 with DFlash block-diffusion drafting + DDTree verify (129.5 tok/s HumanEval) and, on AMD, an MTP chain drafter. Pointers repo-root-relative at `502b094` (squashed history: 1 commit, so no issue/commit archaeology possible). Studied 2026-10-30; nothing built or run (GPU/disk off-limits).

## Why I cloned it

For the 3090 W4A16 Qwen3.5-style 27B + MTP program: (1) a Gated-DeltaNet decode step that already runs on sm_86 inside a persistent kernel (state layout, conv state, fused ops); (2) honest measured fraction of roofline; (3) what speculative decoding looks like on a DeltaNet hybrid (state rollback!) and whether anything applies to a 1-layer MTP head; (4) does anything run a 27B on one 3090 and how fast; (5) license.

## Orientation (repo = 2172 files; these matter)

- `README.md:31-45` results table; `:86-103` single-device / heterogeneous results. License Apache-2.0 (`LICENSE`, `README.md:16,335`) => copying allowed with notice/NOTICE retained (megakernel README credits AlpinDale/qwen_megakernel + Infatoshi/MegaQwen as inspiration, `optimizations/megakernel/README.md:211`; its barrier is byte-identical to MegaQwen's 5090 file, so lineage is shared).
- Megakernel: `optimizations/megakernel/kernel.cu` (1047 l, the whole decode), `model.py` (loader + `step`), `torch_bindings.cpp`, `README.md`, `RESULTS.md`; `prefill.cu` (cuBLAS + WMMA chunked-DeltaNet prefill), `kernel_gb10_nvfp4.cu` (Blackwell-only fp4 decode; excluded from sm_86 builds, `:1-7`).
- Speculative: `server/README.md:1003-1030` (how DFlash works), `server/RESULTS.md`, `server/docs/SPECLA.md` (DeltaNet-aware verify), `server/src/qwen35/qwen35_target_graph.cpp:1496-` (`build_delta_net_block`), `server/deps/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu` (the CUDA DeltaNet kernels used for the 27B), `server/src/common/specla_commit_cuda.cu`, `server/src/common/adaptive_spec_width.h`, `server/src/qwen4exp/qwen4exp_mtp.h` (MTP chain).
- Not relevant to decode: PFlash (speculative *prefill* by prompt compression, lossy, `optimizations/pflash/README.md:35`), KVFlash, Spark (MoE expert placement, `optimizations/spark/README.md`), image-gen, DeepSeek/Gemma/Laguna backends.

## Mechanisms

### M1. Megakernel structure: persistent non-cooperative grid, static phases, atomic barrier
**What:** `decode_kernel` `<<<82, 512>>>` (`NUM_BLOCKS 82`, `BLOCK_SIZE 512`, `optimizations/megakernel/kernel.cu:68-73`, `__launch_bounds__(512,1)` :859, launch :1012). Plain `<<<>>>`, NOT `cudaLaunchCooperativeKernel`; the grid is clamped to `SMs x occupancy` to avoid a barrier deadlock (`:954-1007`, `optimizations/megakernel/README.md:82`). Barrier `AtomicGridSync` (`:140-164`): `__syncthreads`; thread 0 `fence.acq_rel.gpu`, `atomicAdd(counter)`, last arriver resets counter + bumps `generation`, others spin on volatile generation; counters memset before every launch (`:1009-1010`). Layer loop `for layer<24` calls `deltanet_layer` or `full_attention_layer` by `LAYER_TYPE` (`:88-90,908-929`). No weight smem staging: each phase re-reads its activation vector into smem (<=16 KiB static, `:76-77,898`) and streams weights straight from HBM.
**Phases per layer (and barriers):** DeltaNet: [redundant RMSNorm in every block + qkv/z/beta/alpha matvecs] sync | [conv1d+SiLU, L2-norm, recurrence, gated RMSNorm; blocks 0..15 only] sync | out-proj+residual sync | post-norm + fused gate/up/SiLU sync | down+residual sync (`:621-784`; syncs at :652,759,768,776,783) = 5. Full attention: norm+qkv sync | QK-norm+partial RoPE+KV write sync | attention sync | O-proj sync | gate/up sync | down sync (`:453-615`, :481,530,588,597,605,614) = 6. Total 18x5 + 6x6 = **126 barriers/token** [computed]. Final RMSNorm on block 0, then LM head as a **separate 512-block kernel** with last-arriver argmax (`:790-853,1029-1036`); embedding row read in-kernel.
**Matvec:** warp-per-row, lane reads 16 B (`ld.global.L1::no_allocate.v4.b32`, `:186-201`; `matvec_bf16` :310-338), fp32 FMA, `__shfl_down` reduce, activations from smem; rows split `ceil(rows/82)`/block. RMSNorm is Qwen3.5-style `x*rstd*(1+w)` and is recomputed redundantly by every block (`:217-259`, MegaQwen trick) instead of a sync.
**Verified:** read. **Cost:** same as MegaQwen: per-phase tails, un-overlapped LM head (508 MB = 34% of bytes), only 16/82 blocks busy during the DeltaNet recurrence (`:655`, idle-branch comment "could prefetch weights" `:756-758`, i.e. NOT implemented).

### M2. Gated-DeltaNet decode step (the part we need)
**Config (0.8B):** 16 heads, key=value dim 128, conv kernel 4, `DN_CONV_CHANNELS = 2*16*128 + 16*128 = 6144` (`optimizations/megakernel/kernel.cu:57-64`). **State:** fp32 `[layer][head][128 j][128 i]` flat = `state[j*128+i]` (value-row-major, key index contiguous; `:714,727,738`) in *global* memory, read and written every token: 16x128x128x4 = 1 MiB/layer (`optimizations/megakernel/model.py:218`). **Conv state:** fp32 `[6144 ch][4]` per layer, shifted in place each token (`:667-669`, `optimizations/megakernel/model.py:219`). The conv input is the *post-projection* qkv channel, depthwise 4-tap + SiLU (`:662-675`).
**Step (blocks 0-15, one head per block, 16 warps):** conv+SiLU -> `beta=sigmoid(b)`, `decay=exp(-exp(a_log)*softplus(alpha+dt_bias))` (`:678-685`) -> L2-norm q (extra scale 1/sqrt(128)=1/11.3137) and k (`:688-700`) -> per value column j (8 per warp): `stk=S[j,:]·k`, `sqv=S[j,:]·q` via warp shuffle, `err=(v_j - decay*stk)*beta`, `o_j = decay*sqv + err*(k·q)`, `S[j,:] = decay*S[j,:] + k*err` (`:702-740`; algebraically q^T S_new) -> gated RMSNorm per head `rmsnorm(o)*w * SiLU(z)` (`:742-755`) -> out-proj. Fused: conv+SiLU+gates+norm+recurrence in one block; the four input projections are separate matvecs sharing the normed activation.
**The same op for the real 27B (ggml path):** state `[128,128,H_v=48]` f32 per DeltaNet layer = 3 MiB, 48 layers = 151 MB, so read+write = **302 MB/token** (2% of ~16 GB, not free); conv state `[3, 10240]` f32 (`server/src/internal.h:236-238,492-498`, `qwen35_target_graph.cpp:1543,1596-1598`); K heads 16 shared across 48 V heads. ggml kernel parallelises *across value columns*: one warp per (head, column) = 48x128 warps, state transposed `M[col][i]` so a column is contiguous, state held in registers for the whole token chunk, `raw-gate` mode computes `sigmoid(beta)`/`softplus` inside the kernel (`deps/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu:111-145,180,248,284-300`). Contrast: megakernel uses 16 blocks, ggml uses all SMs.

### M3. Results, hardware, roofline (computed from their sizes)
**Megakernel, RTX 3090 (stock 420 W), Qwen3.5-0.8B BF16, pp520/tg128:** 413 tok/s decode (433 at stock clocks/314 W; 411 @220 W, 1.87 tok/J), llama.cpp BF16 267, HF 108 (`optimizations/megakernel/RESULTS.md:12-24,34-41`). Method: Python loop, one `.item()` D2H sync per token (`optimizations/megakernel/model.py:274-296`) => per-token host round-trip is inside the 413.
**Bytes/token (from `optimizations/megakernel/kernel.cu:38-64,96-125`):** DN layer 21.53 M params, FA layer 18.34 M, 18 DN + 6 FA = 497.6 M; LM head 248320x1024 = 254.3 M; total 751.9 M x 2 B = **1.504 GB** (+ ~37.7 MB DN state r/w + 3.5 MB conv r/w). => 413 tok/s = **621 GB/s weights-only = 66% of 936 GB/s**, 638 GB/s (68%) incl. state; 433 tok/s = 70-71%; llama.cpp 267 = 43%. Roofline 606-622 tok/s. [INFERENCE: sm_86 achievable DRAM BW is ~85-90% of 936, so they are at ~75-80% of achievable.] The 2.42 ms/token vs 1.64 ms floor leaves 0.8 ms (125 barriers x ~0.8 us = 0.1 ms, tails, LM head, host sync).
**27B on one 3090 (ggml, not megakernel):** Qwen3.5-27B Q4_K_M (~16 GB) AR **37.78 tok/s** (HumanEval; 37.7/37.65 Math500/GSM8K) = ~604 GB/s = 65% of 936 (roofline 58 tok/s) [INFERENCE: 16 GB]; SGLang AWQ AR 46.6 tok/s on same card (`server/RESULTS.md:1-18`, `server/README.md:1033`). The 5090 (1.9x BW) only gets 42.3 tok/s AR (1.12x; `server/RESULTS.md:408-420`) => ggml AR is launch/latency-bound by ~9-15 ms/token (hundreds of kernels), exactly the overhead a megakernel/graph removes [INFERENCE: solve t = a + 16 GB/BW]. Per-token verify cost 25 ms at N=1 -> 0.97 ms/token at N=128 (`server/RESULTS.md:152`).

### M4. Speculative decoding implemented (server/)
**DFlash + DDTree (Qwen3.5/3.6-27B Q4_K_M, 1x3090):** drafter = 5-layer block-diffusion net (GGUF 1.84 GB, `server/README.md:43`; BF16 3.46 GB, `server/RESULTS.md:5`), input `[last_token, MASK x15]` + hidden states captured from 5 target layers `{1,16,31,46,61}` (`server/src/internal.h:259-262`), outputs 16 tokens in ONE forward (`server/README.md:1005`); DDTree = best-first tree of <=22 nodes over top-K, chain pre-seeded (AL ~4 -> ~9 from `chain_seed`, :1007); one target forward verifies the tree with a mask from parent pointers. Greedy only for the skeleton; full rejection sampling "future work" (`:1049-1052`). Budget sweep 15->40: AL 7.64->8.9, best tok/s at 22 on 3090 (`server/RESULTS.md:105-120`).
**Numbers (3090, n_gen=256, 10 prompts, `server/RESULTS.md:9-17`):** HumanEval AL 8.31, **129.5 tok/s (3.43x)**; Math500 AL 7.04, 110.5 (2.93x); GSM8K AL 6.14, 96.2 (2.55x). Step time = AL/tok/s = ~64 ms [computed] = 3.7x the 17 ms bandwidth floor, so verify of 22 nodes is compute/overhead-bound, not weight-bound [INFERENCE]. Long context (Q4/TQ3 KV, sliding window): 99 tok/s @13K, 35 @32K in the older no-window table (README with `--fa-window 2048`: ~95 @32K, ~91 @64K, `server/README.md:60-67`), ~74 @128K for Qwen3.6 (`server/RESULTS.md:90-103`, `optimizations/pflash/README.md:35`). 2x3090+NVLink Qwen3.8 TP + DFlash2: 79.7 tok/s = 2.16x AR (`README.md:98`; AR ~36.9 implied). 5090: 205 tok/s HE AL 12.9 (budget 40).
**The DeltaNet-specific problem and fix (key for MTP):** verifying k tokens on a recurrent layer needs per-token state for rollback. v1: write a state snapshot per tree node (`ggml_gated_delta_net_tree_persist` writes them directly, saved 9 ms/step; fp16 intermediates to fit budget 22; ceiling ~26 nodes on 24 GB because 3 MiB x 48 layers x nodes; `server/RESULTS.md:122-152`, `gated_delta_net.cu:56-57,200-215` branch reload from parent state; `ssm_conv_tree` gathers conv window by parent chain). v2 **SpecLA** (arXiv:2607.16673, `server/docs/SPECLA.md`): keep state in registers across the chain, store per-node *factors* (normalised k, delta residual, log-decay, raw conv input) instead of states, replay accepted path lazily into the durable state at the NEXT verify (`server/src/common/specla_commit_cuda.cu:19-60`: `S' = e^{gA} S + sum_t w_t k_t ⊗ v_t`); chain accept = swap two banks, no copy; heavy-light decomposition for trees (`server/docs/SPECLA.md:13-75`). Measured on gfx1151 only: tree verify step 191.5 -> 184.1 ms, +4-8% end-to-end (`server/docs/SPECLA.md:176-204`); not tested on 3090.
**MTP:** only the AMD-targeted `qwen4exp` backend (MoE Qwen3.8-Flash-Next, 36 DN + 12 attn): chained MTP head, k=1..7 adaptive, MTP head uses a **64000-token vocab subset** (`server/src/qwen4exp/qwen4exp_mtp.h:37-60`, lowest BPE ids + control tokens), width picked by expected committed tokens / modelled step cost (cost seed `45+20(w-1)`, acceptance modelled by *reached depths* because "acceptance can fall off a cliff at one depth": `server/src/qwen4exp/qwen4exp_mtp.h:17-32`, `server/src/common/adaptive_spec_width.h:1-60`), output bit-identical to MTP off (`server/docs/QWEN4EXP.md:43,169`). Headline: 40 tok/s short reply, "1.8x AR" on Strix Halo; 75.5 tok/s with R9700 (`README.md:63-64,101`). No acceptance-length table is published for MTP (only the log line `server/src/qwen4exp/qwen4exp_seq_engine.cpp:598`).

### M5. Failure lessons recorded
- `grid.sync()` inside the per-token DeltaNet loop => silent hang; sync only between layers (`optimizations/megakernel/README.md:104`). Grid size must be <= resident blocks (`optimizations/megakernel/kernel.cu:954-1007`).
- `S_TILE=16` for ILP => registers spill, "silent crash"/collapse; 8 was the sweet spot (`optimizations/megakernel/README.md:106`; the symbol no longer exists in `kernel.cu`, grep-verified).
- Power: 420->300 W costs 0.2% speed, 220 W 5%, 150 W collapses to 45% (`optimizations/megakernel/README.md:55-62`): the kernel is not power-bound.
- Memory ordering: barrier fences only in thread 0, no acquire fence after the spin (`optimizations/megakernel/kernel.cu:146-163`); cross-block data read with plain loads. It runs; Pascal path substitutes `membar.gl` (`:25-31`). The Blackwell file went back to `cg::this_grid().sync()` (`optimizations/megakernel/kernel_gb10_nvfp4.cu:207-210`) [INFERENCE: for safety]. Do not copy the barrier unmodified.
- DDTree: pure best-first on a quantized target needs the chain pre-seed (AL 4 -> 9); `ggml_cpy` of intermediates cost 9 ms/step; CPU top-K K 32->8 saved 7% of draft time (`server/RESULTS.md:122-137`).

### M6. 3090 / sm_86 specifics
- Megakernel is "primary target" sm_86 (`optimizations/megakernel/README.md:124`); BF16 on sm_80+, FP16 on sm_75 (`half_type.h`); no cp.async/TMA/WMMA in decode. `L1::no_allocate` 128-bit loads for streaming weights (`:188-191`: "bypass L1 to avoid read-for-ownership pollution"). 82x512 threads, 1 block/SM => <=128 regs/thread.
- DFlash tuning is per-GPU: budget 22 on 3090 (82 SMs, 936 GB/s, 6 MB L2) vs 28 on 4090 (72 MB L2) vs 40 on 5090 (`server/RESULTS.md:828,881-895`). Prefill uses cuBLAS + WMMA (`optimizations/megakernel/prefill.cu:1698-1702` raises dynamic smem opt-in).

## Does anything run a 27B-class model on ONE 3090? Yes, but not as a megakernel
`luce_server`/`test_dflash` on Qwen3.5-27B Q4_K_M: **37.8 tok/s AR, 129.5 tok/s HumanEval with DFlash+DDTree (96-130 across tasks)**; Qwen3.6-27B ~74 tok/s at 128K (pflash README). ggml graph, 64 layers x hundreds of launches, Q4_K_M ~16 GB (`server/README.md:43`). No W4A16/compressed-tensors, no MTP, no megakernel at 27B anywhere in the repo; the only megakernel is 0.8B BF16 (and a Blackwell fp4 variant).

## Transferable to our 3090/W4/MTP megakernel
- DeltaNet decode op recipe (M2) and, better, the ggml column-parallel mapping (one warp per value column, state in registers, all SMs) vs the 16-block lucebox mapping; for 27B: 48 V heads x 128 cols, state 3 MiB/layer fp32 (consider bf16/fp16 state only after accuracy check: they use f16 only for *intermediates*).
- Redundant RMSNorm per block (+ `(1+w)` Qwen3.5 variant), 126-barrier structure as a baseline phase list: ours would be 48x5 + 16x6 = 336 barriers (~0.3 ms ~1.5% of a token) [INFERENCE].
- Verify-k-tokens on DeltaNet: factor-capture + lazy replay (SpecLA) so MTP verify of 2-4 tokens needs no state snapshot and ~no extra DRAM; chain case = 2 banks. Direct fit for MTP chain depth 1-3 (no tree).
- MTP head with reduced vocab (64k of 248k) cuts draft LM-head traffic 4x; adaptive depth controller (`adaptive_spec_width.h`) as design reference. Our MTP head is bf16 lm_head-shared? [unknown, check] -- the saving only applies if the draft projection is separate or can be restricted.
- Baseline to beat: 27B AR on 3090 = 37.8 (ggml) / 46.6 (SGLang AWQ) tok/s; floor ~58 tok/s at 16 GB.

## Dead ends / lessons
- 0.8B result says little about 27B: weights/bytes are 20x larger so barrier overhead shrinks and DRAM efficiency matters; their README admits "fusion benefits shrink as model size grows" (`optimizations/megakernel/README.md:167`).
- Idle-SM prefetch left as a comment; DeltaNet recurrence uses 16/82 SMs; state in global memory re-read each token.
- ggml AR is latency-bound (5090 barely faster than 3090); tree verify costs ~3.7x floor; DDTree budget >22 loses on 3090.
- DFlash/PFlash/KVFlash/Spark do not apply to a single-stream W4 megakernel: DFlash needs a separate trained drafter (3.5 GB) and a tree-capable target; PFlash is lossy prefill.

## Verdict
1. Best available evidence that a Qwen3.5-style DeltaNet decode fits in one persistent sm_86 kernel at ~66-70% of DRAM roofline, plus a complete exact-rollback story (SpecLA) for speculative verify on DeltaNet layers.
2. Apache-2.0 so code (barrier, matvec, DN step, SpecLA commit) may be reused with attribution; but fix the memory-ordering gaps and parallelise the recurrence across all SMs.
3. 27B on one 3090 exists only via ggml: 37.8 tok/s AR, ~130 tok/s with a separate DFlash drafter; nothing there answers MTP acceptance, so our own measurement is required.

---

Pinned at `502b094` (main, 2026-10-30) · 2172 files · 107M
Update with: `git -C vendor/lucebox pull` then refresh `commit:` above.
