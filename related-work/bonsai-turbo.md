---
title: bonsai-turbo
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/RightNow-AI/bonsai-turbo
path: vendor/bonsai-turbo
commit: 7003767
branch: main
commit_date: 2026-07-16
---

# bonsai-turbo

One-liner: batch-1 decode engine (~2.2 kLOC CUDA/C++) for PrismML's Bonsai 27B (Qwen3.6-27B hybrid: 64 layers, GDN everywhere except every 4th = 16 full-attn; README.md:93) with ternary (Q2_0_g128, 2.125 bpw) and 1-bit (Q1_0_g128) GGUF packs, in three modes: eager, one CUDA graph per token, or one cooperative megakernel per token. **Apache-2.0** (LICENSE; README.md:99-101), contains no vendor code. H100 only; no MTP (README.md:84). Our model is the same hybrid shape but W4A16 BF16-activation, sm_86 - the codes are not reusable, the structure is.

Pointer legend: root-relative at `7003767`; `[INF]` = my arithmetic/inference. Read only (no build, no GPU job). Every perf number is the authors' (raw results are not tracked, `results/summary.md:3-5`).

## Why I cloned it

- Closest published batch-1 hybrid (GDN+attn) decode with *both* a graph and a megakernel variant on the same kernels: how does a good GEMV/GDN/attn stack look, what are graph-vs-mega numbers and why, what is the per-token byte budget, how is parity with a reference gated, and what can we lift under Apache-2.0?

## Orientation

- `src/cuda/gemv.cu` (378) GEMV family + activation quant; `src/cuda/mega.cu` (787) the cooperative megakernel; `src/cuda/gdn.cu` (110) delta-net step; `src/cuda/attn.cu` (150) decode attention; `src/cuda/ops.cu` (553) norms/rope/argmax/conv1d; `src/retile.cpp` (138) load-time repack; `src/model.cu` (352) loader+stacking; `tools/bt_run.cu` (894) engine + graph capture + phase probe; `tools/bt_microbench.cu` GEMV bandwidth bench; `scripts/parity.sh`+`compare_logits.py` gate.
- Build `cmake -B build -G Ninja` (arch default **90**, CMakeLists.txt:40-41); run `bt-run --model X.gguf --ids ... --graph|--mega --bench` (README.md:49-60).

## Mechanisms

### 1. Per-token structure and results (H100 SXM 80 GB, CUDA 12.8, tg128, greedy; `results/summary.md:7-18`, README.md:9-18)
| engine | ternary tok/s | 1-bit tok/s |
|---|---:|---:|
| vendor llama.cpp fork (same machine class, 3 reps) | 85.5 +-6.9 | 90.1 +-3.5 |
| bonsai graph mode | 151.1 | 133.0 |
| bonsai megakernel (`--mega`) | 149.6 | **158.7** |
Run-to-run spread 146.8-151.4 (graph, ternary) across containers (`results/summary.md:15-16`); single measurement per cell otherwise. Vendor profile: 3703 graph nodes/token, 97% NVML busy, 10.6 ms/token (`results/summary.md:32-36`; `scripts/analyze_trace.py`).
- Graph mode: eager decode captured once with `cudaStreamBeginCapture` incl. argmax + `step_bump` (device-resident `d_pos/d_step/d_tok/d_ring`), replayed 16 launches at a time then one sync (`tools/bt_run.cu:748-776`). Per layer ~13-14 kernels (`gdn_layer` :355-399, `attn_layer` :401-449, `mlp` :451-462) => ~850-900 kernels/token [INF count]. **No PDL anywhere** (grep: no `griddepcontrol`/programmatic attrs in `src/`, `tools/`).
- Mega mode: `cudaLaunchCooperativeKernel` once per token, CPU queues 16 launches then syncs (`tools/bt_run.cu:695-712`; `src/cuda/mega.cu:746-785`).

### 2. Why graph ~= mega (ternary 151.1 vs 149.6; 1-bit mega +19%)
- Not explained by the authors beyond "kernel boundaries become grid syncs" (`src/cuda/mega.h:3-6`). What the code shows: mega still has **515 `grid.sync()` per token** (1 after embed + 8 per layer x 64 + 2 at head/argmax; `src/cuda/mega.cu:682-740`), each GEMV phase still ends in a tail wave, and the only cross-phase weight prefetch is "idle blocks pull the *next* GEMV's codes into L2" in the narrow phases (`prefetch_l2` `src/cuda/mega.cu:29-46`; used at :296, :410, :534; GDN only 48 of ~1056 blocks work, :409,:413). So mega removes launch gaps/ramp but not dependency tails: the same conclusion as the PDL repo ([pdl-megakernel-reconstruction](pdl-megakernel-reconstruction.md)): persistence alone buys little. [INF]
- Mega gains only where bytes shrink: 1-bit graph 7.52 ms vs mega 6.30 ms/token; yet 1-bit graph is *slower* than ternary graph despite half the bytes (README.md:81: the 1-bit unpack is ALU-bound). Effective bandwidth: ternary 6.83 GB / 6.62 ms = ~1.03 TB/s (~34% of the "measured 3.0 TB/s", README.md:28); 1-bit 3.8 GB / 6.30 ms = ~0.6 TB/s [INF arithmetic]. Per-token cost is ~overhead/latency/ALU-bound, not bandwidth-bound. A BT_PROBE>=3 run prints per-phase times (proj, conv|prep, gdn|attn, gatequant, out_proj, gate_up, swiglu, down; `tools/bt_run.cu:585-620`) but **no numbers are published** - run it before trusting any hot-spot story.

### 3. Loader / retile (`src/retile.cpp`, `src/model.cu`, `src/packs.h`)
- GGUF parse (`src/gguf.cpp`) -> per-tensor retile at load (`src/model.cu:135-152`): GGUF 34-byte Q2_0 blocks (`src/packs.h:41-45`) become a **16 B-aligned code plane + separate fp16 scale plane**; within each 32-bit word code j=4i+b is moved to bit `8b+2i` so `(w >> 2i) & 0x03030303` yields four codes in four bytes = one `dp4a` operand (`src/retile.cpp:11-25,29-59`; Q1: bit `8b+i`, :19-25,61-91). Asserts the vendor ternary pack never emits code 3 (`src/packs.h:35-37`, `src/model.cu:140-144`).
- **Projection stacking at load**: GDN `[qkv|z|alpha|beta]`, attention `[q+gate|k|v]`, MLP `[gate|up]` each concatenated row-wise into one matrix (same K) (`src/model.cu:175-214`, uses :283,:297,:316). Tied/untied lm_head (:267-271); recurrent-layer mask from `attention.recurrent_layers` or `(i+1)%interval` (:250-262).
- Bit-exactness tests: `tests/test_retile.cpp`, `tests/test_dequant.cpp` (compare the permuted layout element-wise to a reference dequant; CTest, README.md:49-50).

### 4. GEMV design (`src/cuda/gemv.cu:62-203`, template `<NBITS, SPLIT, EPI>`)
- **Quantised activations:** every GEMV first runs `quant_acts_kernel` (one warp per 128-group: absmax -> int8, plus 64-group sums for the `(code-1)` / `(2b-1)` correction; :13-60); dot via `dp4a.u32.s32` on (u8 codes, s8 acts) (:7-11), `acc += scale_w*scale_a*(dot - gsum)` per 16 B chunk (:147; Q1: `2*dot - gs`, :173). => A8 math; **this is why vendor parity needs a 1.25 logit tolerance** (see 8). Not W4A16.
- **Tiling:** block 256 thr = 8 warps x 4 rows = 32 rows/block (:62,112); each lane streams 16 B code chunks `c = lane, lane+32, ...` so a warp reads 512 B contiguous per row; activation slice staged once in smem and each lane holds the activation chunk in registers for all 4 rows (:93-135). Per lane per iteration: 4 independent `uint4` loads (one per row, :136/158) = 64 B; per warp 2 KB, per block 16 KB in flight [INF from code]; loop not manually unrolled/pipelined. No barrier between tiles on purpose ("measured 3-5x slowdown" otherwise; `src/cuda/mega.cu:269-271`).
- **Occupancy/plan:** `plan_gemv` aims at ~1024 blocks; if `ceil(M/32) < 256` it splits K over `blockIdx.y` and combines with `atomicAdd` into a pre-zeroed `y` (`cudaMemsetAsync`) (:297-335) -> non-deterministic fp32 order. EPI 0 f32 store, 1 f16 store (single split), 2 residual add into the fp32 residual stream (:68-70,186-200).
- **Claimed efficiency:** "78 to 80% of measured copy peak on the large shapes" (README.md:31). How measured: `tools/bt_microbench.cu`: peak = 8 x 1 GiB `cudaMemcpy` D2D, counting read+write bytes (:41-60); GEMV GB/s = (codes + 2*scales + K + 4M bytes)/time over 100 back-to-back launches of the *same* weights after 20 warmups (:154-177); default shapes `5120x5120,13824x5120,5120x13824,151936x5120` (:240) - note these are not our shapes. **Caveats [INF]:** no L2 flush (grep: none), so shapes under H100's 50 MB L2 (Q2: 6.6, 17.7 MB) are cache-resident; only the 194 MB lm_head shape is HBM-honest; "copy peak" is R+W D2D which differs from a read-only peak. The 78-80% figure itself is not reproducible from the repo (outputs untracked).
- bf16/fp16 fallback `gemv_f16_kernel` is a naive 1-row/warp scalar loop (`src/cuda/ops.cu:526-540`) used only for non-quantised matrices - **not** adequate for our BF16 lm_head (2.54 GB).

### 5. Fused gated-delta-net step (`src/cuda/gdn.cu:15-81`; mega copy `src/cuda/mega.cu:397-457`)
One block (8 warps) per value head, state fp32 **transposed so column c is contiguous** (`S_h + c*S`), warp w handles columns `w, w+8, ...`; per column: lane-strided load of the 128-vector (4 floats/lane), `kv = S^T k` via warp-shuffle reduce, `delta = (v - g*kv)*beta`, `S = g*S + k (x) delta`, store, `out = (S^T q)*scale` (:52-79). `g = exp(A*softplus(alpha+dt_bias))` with `A=-exp(A_log)` pre-baked in the GGUF (:39-42), `beta = sigmoid(beta_raw)` (:42). conv1d(k=4)+SiLU and q/k L2-norm are separate (graph) or `op_conv_l2` (mega :352-395). Grid = H_v blocks only (48 [INF]); the megakernel has the other ~1000 blocks idle/prefetching (:409-411). State traffic per token 48 layers x (H_v x 128 x 128 x 4 B) r+w [INF ~0.30 GB, matches the roofline in `docs/PROGRAM.md` Sharpened]. No chunking needed at batch 1.

### 6. Attention (`src/cuda/attn.cu:120-211`; mega copy :518-598)
"flash-decode" here = one block of 8 warps **per query head**, warps stride context positions with per-warp online softmax (m, l, acc[D/32]) merged in-block (no split-KV across blocks); fp16 KV cache, GQA group re-reads the shared KV per q head; gate (`sigmoid(gate)*out`) fused into the next quant (`SRC_ATTN_GATE`, `src/cuda/mega.cu:114-122`); q/k RMSNorm + NeoX RoPE (`powf/sincosf` per element, :501-512) + KV append in `op_attn_prep` (:459-516). Only H (24 [INF]) blocks active at long context -> ctx-length scaling is poor; fine for short ctx. KV fp16 only (README.md:80).

### 7. Megakernel `mega.cu` (cooperative design)
- **Barrier:** `cooperative_groups::this_grid().sync()` (:667,682-740) - hardware-managed cooperative grid barrier, grid = `SMs x blocks/SM` sized from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` with max smem carveout; 256 threads, smem = max_K int8 acts + scales + group sums (~19 KB for K=17408) (:746-785). Comment: 100% carveout lets threads bound residency at 8/SM instead of 2/SM (:761-764). No custom spin barrier, no cost measurement.
- **Phases per layer (8):** `proj` GEMV(fused norm in staging) | `conv_l2` or `attn_prep` | `gdn` or `attn` | `quant_global` | `out_proj` GEMV(+residual, +next-norm partial) | `gate_up` GEMV | `quant swiglu` | `down` GEMV(+residual) (:685-730). Stamps per phase via `clock64` thread0 when `BT_PROBE>=3` (:658-662).
- **Tricks:** (a) every GEMV block builds its own quantised activation copy in smem from the producer op on the fly (`StageSrc`: norm*x, GDN gate, attn gate, swiglu, :55-143) - removes quant kernels; (b) residual-GEMV epilogue accumulates `sum(x_new^2)` into `red_scratch[slot]` by `atomicAdd`, replacing the next norm's reduction phase (:274-285); slots double-buffered by token parity and zeroed by the next token's embed (:672-679,347-349); (c) idle-block L2 prefetch of next weights, budget-bounded (:29-46); (d) device-resident token/pos/step ring so no host round trip (:640-654); (e) argmax over lm_head by per-block partials + final bump (:600-654).
- Prompt tokens are fed one at a time through the same kernel (`tools/bt_run.cu:558-565`); sampling only in mega/eager.

### 8. Parity gate (`scripts/parity.sh`, `scripts/compare_logits.py`) - reusable
`parity.sh`: build `tools/vendor_logits.cpp` against the vendor libllama; tokenize `scripts/parity_prompts.txt` (32 fixed prompts) with the vendor tokenizer; vendor greedy 64 steps dumps per-step full logits (float32 `n_steps x n_vocab`); `bt-run --ids-file --logits-out` the same; per prompt `compare_logits.py`; exit non-zero if any fail (`scripts/parity.sh:19-57`). Gate (`scripts/compare_logits.py:18-75`): compare until first top-1 flip; fail if max |delta| over the vendor top-20 logits in the first 16 steps >= `MAX_ABS_TOL=1.25`; a flip passes only if vendor top-1/top-2 margin < `TIE_MARGIN=1.0`. `results/summary.md:22-26` claims flips only at margin <= 0.034 and deltas < 1.2; docstring says 0.75 (`scripts/compare_logits.py:10-11`) vs constant 1.25 (`:25`) - inconsistent, trust code. Ran on 4 configs (ternary/1-bit x graph/mega): 32/32 (README.md:20). MATH-500 subset "in progress", 64/66 completed traces correct, only 66/100 finish within 8k tokens (`results/summary.md:27-30`) - weak end-to-end evidence.

## Worth stealing (Apache-2.0: copy with NOTICE/attribution; keep license header)
Directly reusable, with required changes:
- `scripts/compare_logits.py:1-79` + `scripts/parity.sh:1-57` + `parity_prompts.txt` (verbatim). Change: replace `vendor_logits.cpp` by a vLLM/torch dumper writing the same float32 step-major format; tighten tolerance since stock-vLLM W4A16 has no int8-act noise floor (derive from the vLLM-vs-FP32 baseline, cf. KL-delta rule in `related-work/megakernels-vs-cuda-graphs.md`); fix the speed of the pure-Python sort (248,320-vocab x steps).
- `src/cuda/gdn.cu:15-81` GDN recurrence (column-contiguous fp32 state, warp per column, shuffle reductions): math maps 1:1 to Qwen3.5 DeltaNet decode; change: bf16 q/k/v/alpha/beta inputs, **split columns across >=82 blocks instead of 1 block/head** (48 blocks leave 34 SMs idle on a 3090 [INF]), `float4` loads, 2 columns in flight per warp, and convert `A_log` -> `-exp(A_log)` ourselves if the checkpoint stores `A_log` (the GGUF pre-bakes it, :39).
- `src/cuda/mega.cu:352-395` conv1d(k=4)+SiLU+L2-norm with fp32 conv state; `:640-654` device-resident token/pos ring; `:600-654` argmax (but lm_head is 248,320 x 5120 BF16 here).
- `src/cuda/mega.cu:272-285,672-679` fused next-norm sum-of-squares via atomics + parity-banked slots (note: atomics make norm scale order-nondeterministic [INF]); `:29-46` idle-block L2 prefetch (3090 L2 is only 6 MB so prefetch window << H100's 50 MB; budget it).
- `src/model.cu:175-214` stacking concept (we cannot stack `in_proj_a/b`, which are BF16 per the idea file) and `retile.cpp` "permute at load so the in-kernel unpack is one mask+shift" idea; `bt_microbench.cu` structure (add L2 flush + read-only peak).
- Not reusable: `gemv.cu` dp4a/int8-activation path (our weights are INT4 sym g128, activations stay 16-bit), GGUF reader, Q1/Q2 retile bit layouts, `attn.cu` as-is (no split-KV).
What must change for sm_86: default arch 90 -> 86 (CMake:40-41; `griddepcontrol` is not used so no PDL blocker); occupancy: 1536 thr/SM and 100 KB smem => with ~19 KB/block 5 blocks/SM = 410 blocks vs 1056 on H100 [INF]; staging 16-bit activations doubles smem (34.8 KB for K=17408) so stage K-slices or use registers/L1; INT4 g128 GEMV: 64 B of nibbles + 2 B scale per group, unpack via `lop3`/`prmt` magic-number tricks to half2 or fp32 FMA (our own kernel; ALU budget: ~24.5 B weights x ~2-3 ops over 82 SMs x 128 lanes x ~1.7 GHz ~ 3-4 ms vs ~17 ms memory time [INF]).

## Gotchas
- Different regime: ternary/1-bit GGUF, A8 dp4a, H100, no speculative drafting; 6.83 GB/token vs our 15.13 GB (roofline in `docs/PROGRAM.md` Sharpened); our step is ~2.3x more bytes so fixed per-phase overhead matters relatively less (their ~6.5 ms/token floor is mostly overhead) [INF].
- Perf numbers are single cloud runs, spread 3% (graph, ternary) and 7% on the vendor baseline; parity vs vendor only (not vs an FP16 reference); MATH-500 run incomplete.
- `mega.cu` hard-codes `acc[8]` => D <= 256, S_v = 128, `kRingCap=32768` tokens (`tools/bt_run.cu:106`).
- Per-token host work: 16 cooperative launches queued, then sync; sampling not graph-compatible (`tools/bt_run.cu:651-653`).

## Transferable to our 3090/W4/MTP megakernel
- Phase skeleton (8 phases/layer, 515 syncs/token) is a sound starting point for sm_86 using `cg::grid.sync`; first measure the barrier cost on the 3090 before going to a hand-rolled one (open-jet uses a hand-written release/acquire counter, `related-work/open-jet.md` mech. 1).
- Fusion pattern (a)-(e) above, parity gate, load-time stacking/permutation, and the GDN math are the keepers. MTP (draft+verify with 2-4 tokens) is **not** addressed: GEMV per token vs mma-based N>1 must be designed by us.
- Use `BT_PROBE`-style per-phase `clock64` stamps from day one.

## Dead ends / lessons
- Splitting K + atomics only for small matrices; barrier between tiles cost 3-5x on small GEMVs (`src/cuda/mega.cu:269-271`); non-pipelined 16 B-per-row loads still reach ~78-80% of a D2D peak on large shapes (claim) - so the GEMV is *not* where their time goes.
- Hot-spot candidates left on the table (for us to avoid) [INF]: 48-block GDN, H-block attention without split-KV, 1-block (1024-thread) `rmsnorm_quant_f32` in graph mode (`src/cuda/ops.cu:99-113,416-419`), per-call quant kernel, `memset`+atomics split-K.

## Verdict
- A competent, parity-gated Apache-2.0 reference for a hybrid decode stack: graph 151 vs mega 150 tok/s (ternary) corroborates "persistence by itself ~0", but the stack is overhead-bound (~6.5 ms/token floor, 1-bit only +6% over ternary mega), so it is a design reference, not a performance ceiling.
- Lift: parity scripts, GDN recurrence, conv/argmax/ring plumbing, fusion list; rewrite: INT4 W4A16 GEMV, split-column GDN, split-KV attention, MTP verify.
- Biggest unknown: real per-phase breakdown and the cost of `grid.sync` - measure on the 3090 before choosing graph vs megakernel.

---

Pinned at `7003767` (main, 2026-07-16) · 50 files · 1,2M
Update with: `git -C vendor/bonsai-turbo pull` then refresh `commit:` above.
