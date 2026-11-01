---
title: MegaQwen
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/Infatoshi/MegaQwen
path: vendor/MegaQwen
commit: 0bc1fde
branch: main
commit_date: 2026-02-09
---

# MegaQwen
# MegaQwen

Qwen3-0.6B (28 layers, bf16, batch-1) decode as ONE cooperative kernel (82 blocks x 256 thr, `grid.sync()` between phases) + 2 small LM-head kernels; claims 530 tok/s on a 3090. Pointers are repo-root-relative at `0bc1fde`. Studied 2026-10-30.

## Why I cloned it

Questions for the 3090 W4A16 27B hybrid megakernel program: (1) what does a persistent-grid decode structure look like on sm_86 and which phases/syncs does it pay for; (2) what fraction of the 936 GB/s roofline does it really reach (their own README says 5%); (3) which tricks were measured to help/hurt; (4) can we copy code (license).

## Orientation

- `csrc/megakernel/fused_decode_ldg.cu` (1057 l) is the production kernel (`chat.py:40` loads it as "baseline"). Others are failed variants: `_v2` (uint4 loads), `_v3` (cp.async), `_smem`, `_atomic_sync`, `fused_decode_ldg_v4`.
- `experimental/qwen_persistent_kernel.cu` (1212 l): later rewrite for RTX **5090** (header :1-10) with a custom atomic barrier + flag syncs; NOT 3090 and not in README numbers.
- `DEVLOG.md` (496 l) = narrative of what worked/failed; `experiments/RESULTS.md`, `experiments/optimizations/RESULTS.md`, `docs/MEMORY_ANALYSIS.md` = numbers. `config.cuh` is a stale template (BLOCK_SIZE 1024, TILE_ROWS 16; not what `fused_decode_ldg.cu` uses, cf. :14-22).
- Build: JIT via `torch.utils.cpp_extension.load_inline`, flags `-O3 --use_fast_math -maxrregcount=64 -lineinfo` (`chat.py:331-338`). No sm arch flag (inherits torch's). Not run by me (no GPU use allowed in this study).
- License: README says MIT (`README.md:113-115`) but there is **no LICENSE file** in the tree (`git ls-files`). Treat as MIT-by-README; ask before copying verbatim.

## Mechanisms

### Persistent cooperative grid, static phases (no interpreter)
**What:** one `cudaLaunchCooperativeKernel` per token, 82 blocks x 256 threads = 1 block/SM (`csrc/megakernel/fused_decode_ldg.cu:14-16`, `__launch_bounds__(256,1)` :617, launch :947-954). Layer loop is plain C++ (:655-686): `ldg_matvec_qkv -> ldg_qk_norm_rope_cache -> ldg_attention -> ldg_o_proj_postnorm_mlp`. Each phase ends in `cg::grid_group::sync()`. Per layer 8 syncs (QKV :113+:159, qknorm/rope :275, attention :354/:448 (two branches of one divergent call), O-proj :503, post-norm :539, gate/up :582, down :610) = ~225/token (`DEVLOG.md:71`; the other docs say "140+"/"5 per layer", inconsistent: `README.md:85`, `docs/MEMORY_ANALYSIS.md:50`).
**LM head** is NOT in the megakernel: 1184-block non-cooperative kernel + 1-block argmax (:20-22, :956-967). Embedding gather is in-kernel (:646-651).
**Why/cost:** simplest possible structure; every phase boundary is a full-GPU barrier so the slowest SM sets the pace, and weights for phase N+1 are not requested until the barrier releases (the bubble Hazy's interpreter removes, see `related-work/Megakernels.md`). **Verified:** read.

### Weight streaming = warp-per-row, direct `__ldg`, no smem
**What:** rows are split `ceil(rows/82)` per block (QKV :116-157); each warp owns one output row, lane reads `uint2` (4 bf16, 8 B) via `__ldg`, FMA with fp32 activations re-read from global/L1 (:139-157), `__shfl_down` reduce. No cp.async/TMA/smem staging in production; `_v3` cp.async double-buffer gave +1% (`DEVLOG.md:238-257`), smem activation cache 0% (:217-236), producer/consumer warp split 0% or worse: 0:8 509.9, 1:7 510.8, 2:6 489.9, 4:4 478.6 tok/s (:203-213). uint4 loads: +3.5% microbench (:166-186) but `experimental/results/decode_variant_results_tokens100.json` "uint4" = 524.9 vs baseline 531.6 tok/s (slightly worse).
**Attention:** one block per Q head (16 blocks!), each warp strides over cache positions, online softmax, scalar bf16 `__ldg` of K/V (:299-446). Serial in context length -> 525 tok/s @pos1, 472 @100, 422 @200, 382 @300 (`experiments/RESULTS.md:45-56`); 405.8 tok/s @500 tokens (`experimental/results/decode_variant_results_tokens500_baseline_blocks128.json`).
**Verified:** read; numbers are theirs.

### "Block divergence + L2 prefetch" = the only big win (2x)
**What:** during attention the 66 non-attention blocks run `ldg_prefetch_weights_l2` over O/gate/up weights then wait at the barrier (:282-297, :322-355). Claimed 242 -> 525 tok/s at pos 1 (`DEVLOG.md:145-164`, `README.md:92`).
**Read-through [INFERENCE]:** the prefetch loop is `for i=tid; i<n; i+=1024` with 1 bf16 per thread (:290-292) so it touches only 512 B of every 2 KiB = ~25% of lines; and per-layer O+gate+up = 4.2+12.6 MB (bf16, 0.6B dims) > 6 MB L2. So it cannot be "all MLP weights are L2-resident" as claimed; more plausible it keeps HBM busy (extra requests in flight) during the otherwise bandwidth-idle attention phase, i.e. it is a bubble filler. The 5090 rewrite uses `prefetch.global.L2` per element over O+gate+up+down (`experimental/qwen_persistent_kernel.cu:610-635`). An extra "prefetch" variant tested on top is *slower*: 499.4 vs 531.6 tok/s (`experimental/results/decode_variant_results_tokens100.json`).
**Cost:** attention blocks and prefetchers contend for the same HBM/L2; benefit vanishes when attention gets long (spills L2 with KV).

### Redundant RMSNorm (all blocks compute it) and what did NOT work
- Redundant RMSNorm in every block removes 2 syncs/layer: 170 -> 215 tok/s short ctx (+26%), but 0.90x at pos 200 (`experiments/optimizations/RESULTS.md:8-35`). The production file still has block-0-only norm (:76-111); the winning variant lives in `experiments/optimizations/redundant_rmsnorm/kernel.cu`.
- Head-based block assignment (5 blocks per Q head so QKV+attn need no sync): **0.59x** (`experiments/optimizations/RESULTS.md:37-68`): "don't give up block utilization to remove a sync".
- Block count: 96/112/128 blocks on 82 SMs: 521.7/508.8/533.9 vs 531.6 (`..._tokens100.json`): noise-level; non-monotone.

### Barrier alternatives
- cg `grid.sync()` vs CUDA graph of 225 EMPTY kernels vs 225 launches: 167.3 / 186.9 / 347.5 us => 0.73 / 0.83 / 1.54 us per op (`experiments/RESULTS.md:78-97`, `experiments/sync_overhead.py:25-34`). Delta coop-vs-graph is only ~0.1 us/op; for a 1.9 ms token that is 0.2 ms ~ 10%.
- Hand-rolled atomic barrier (`csrc/megakernel/fused_decode_atomic_sync.cu:58-104`) = "inconclusive, compile too slow" (`DEVLOG.md:259-279`). The 5090 file has a working one: thread 0 `fence.acq_rel.gpu` + `atomicAdd(counter)`, last arriver resets and bumps `generation`, others spin on a volatile (`experimental/qwen_persistent_kernel.cu:85-110`), plus *monotone-counter flags* replacing two of the per-layer barriers (KV-ready `:468-491`, attention-done `:602-646`) so only the 16 attention blocks wait where others don't need to.
**Caveat:** that barrier has no acquire fence after the spin (:103-105) and `*counter=0` is a plain store; it relies on later loads being `ld.global.L1::no_allocate` (:166-174) / volatile. [INFERENCE: works in practice, not a proof; compare `related-work/lucebox.md` which copies the same code.]

## Results and roofline (computed from their sizes)

Hardware: RTX 3090, 82 SMs, bf16 weights, batch 1, short context, tied lm_head (`README.md:3-17`, `DEVLOG.md:35-39`).
- Weight bytes/token from `csrc/megakernel/config.cuh:33-40` dims: per layer q 2048x1024 + k,v 1024x1024 + o 2048x1024 + 3x(3072x1024) = 15.73 M params; x28 = 440.4 M; LM head 151936x1024 = 155.6 M; total 596 M x 2 B = **1.192 GB** (non-embedding layers 0.881 GB).
- 530-531.6 tok/s (1.881 ms/tok, `..._tokens100.json`) => **634 GB/s = 67.7% of 936 GB/s** (LM head included). Roofline 785 tok/s. At 405.8 tok/s (pos 500) 484 GB/s = 52%.
- **Their "5% / 47 GB/s" claim (`README.md:84`, `docs/MEMORY_ANALYSIS.md:5,30-36`) is arithmetically impossible**: 47 GB/s x 1.9 ms = 89 MB per token, 13x less than the weights they must read at 530 tok/s. Their sync-bound story (225 x 0.73 us = 167 us = 8.9% of 1.88 ms) also does not explain the 32% gap to roofline [INFERENCE]; the real losses are probably per-phase tails (16-block attention, down-proj 1024 rows over 656 warps, QKV only 4096 rows) plus the un-overlapped LM-head kernel (26% of bytes, separate launches).
- `docs/MEMORY_ANALYSIS.md:30-36` mean latency 1.904 ms = 525 tok/s is consistent with 1.881 ms above; `experiments/optimizations/RESULTS.md` baseline numbers (4.0-5.7 ms) come from a different, slower harness: do not mix them.

## Speculative decoding / DeltaNet
None implemented. `DEVLOG.md:425-427` lists speculative decoding and INT4 ("~4x, 2000+ tok/s") only as future work. No DeltaNet (plain Qwen3 attention).

## Transferable to our 3090/W4/MTP megakernel
- Per-phase `grid.sync` costs 0.73 us (82x256 empty kernel) vs 0.83 us per graph node: scale to ~400-450 barriers/token for 64 layers = ~0.3 ms = ~1.7% of a ~17 ms W4 token [INFERENCE from 16 GB / 936 GB/s]. Syncs are NOT the bottleneck at 27B scale; per-phase *tails* and un-overlapped bytes are.
- Use idle SMs/warps during low-parallelism phases (attention, DeltaNet recurrence) to keep HBM busy: prefetch next-phase weights with `prefetch.global.L2` or cp.async into smem (their 2x). Our DeltaNet recurrence has 48 value heads x 128 columns of parallelism and attention only 4 KV heads, so the same idea applies, with real smem staging instead of 25%-coverage `__ldg` touches.
- Fuse/avoid syncs only if full-GPU row parallelism stays (head-based split lost 41%).
- Keep LM head in the same persistent kernel/stream with the layers: theirs is 26% of bytes and runs as separate launches (ours is 2.54 GB bf16 = ~15% of ~17 GB).
- SASS check (`DEVLOG.md:337-360`): confirm 128-bit `LDG.E.128` actually emitted; `-maxrregcount=64` with 256 thr x 1 blk/SM is far below the 255-reg budget (maybe too tight for dequant).

## Dead ends / lessons
- Warp-specialized producer/consumer inside a block (0:8 best), smem activation caching, cp.async double buffer at row-matvec granularity: all ~0-1% (they were not bandwidth-starved at the *kernel* level but phase-latency-starved).
- Head-based work split (0.59x). Redundant-norm trades sync for L2 pressure, loses at long context.
- Do not trust their utilization narrative; recompute from bytes (done above). Do not trust the 2x attribution to "texture cache".

## Verdict
1. A 67%-of-roofline bf16 baseline on the exact GPU: our floor to beat is ~634 GB/s effective, i.e. a W4 27B kernel at that efficiency = ~39 tok/s [INFERENCE: 634 GB/s / ~16 GB per token = ~13 GB W4g128 body + 2.54 GB bf16 lm_head, sizes estimated not measured]; need >~80% (750 GB/s) for ~47.
2. Structure (static phases + grid.sync) is fine for 0.6B; useful mostly as evidence that overlap during attention (idle SMs) is worth ~2x.
3. MIT-by-README, no LICENSE file; patterns only, write our own code.

---

Pinned at `0bc1fde` (main, 2026-02-09) · 79 files · 1,6M
Update with: `git -C vendor/MegaQwen pull` then refresh `commit:` above.
