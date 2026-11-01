---
title: open-jet
type: source
tags: []
status: seed
created: 2026-10-30
author: agent
repo: https://github.com/L-Forster/open-jet
path: vendor/open-jet
commit: 8f4bac4
branch: master
commit_date: 2026-10-29
---

# open-jet

One-liner: OpenJet Engine = single-3090 persistent-cooperative CUDA megakernel for Qwen3.8-27B (GGUF Q4_K_M, arch `qwen35`) that runs a whole MTP speculative cycle in ONE launch; closest existing prior art to our W4A16 megakernel plan. License is **AGPL-3.0** (see §9) so read for design, do not copy code.

Pointer legend: unlabeled `file:line` = `vendor/open-jet/megakernel/<file>` at `8f4bac4`; `mk.cu` = `qwen38_mk.cu`; `README` = `megakernel/README.md`. `[INF]` = my inference/arithmetic, not stated by authors. Nothing here was built or run (no GPU jobs); every performance number is the authors' claim. Repo history for `megakernel/` is one commit (`git log -- megakernel`), so there is no design-history to mine.

## Why I cloned it

- How a persistent fused decode kernel is structured on sm_86 (barriers, phases, GEMV, DeltaNet, attention), how on-device MTP verify works, and what is GGUF-specific vs portable to compressed-tensors W4A16 g128 + `g_idx`.
- Same model shape as ours: `D=5120, FF=17408, VOCAB=248320, 64 layers (48 DeltaNet + 16 attn), 24/4 heads, HD=256, 16 k-heads/48 v-heads x128` (mk.cu:52-55), attn layer iff `(i+1)%4==0`, layer 64 = MTP block (mk.cu:3304, 3303). Dims are drop-in; the quant format and loader are not.

## Orientation

- `qwen38_mk.cu` (5953 lines): constants 50-103, structs 108-220, GEMV 470-717, phases 804-2300, lm_head 2322, sampling 2458-2583, control 2586-2626, forward 2629, MTP 2696, kernel 2811, loader/engine 2934-3760, prefill/slot host code after. Original Q4_K_M engine. `qwen38_mixed.cu` (4166 lines) = same design with extra quant kernels (README:244-247, 276-278; single-conversation only, README:378).
- `bmm.cuh` (2653): *non-persistent* batched kernels, T=8..16 rows, int8 tensor cores, used for `--parallel` slots (bmm.cuh:1-5, TM=16/TS=12 at :10-11; launches at mk.cu:3720-3724, 4404-).
- `prefill*.cuh`, `kv_lowbit.cuh`, `kv_paged.cuh`, `kv_spec.h`: prefill GEMM/attention/DeltaNet-WY, low-bit KV, KV addressing. `capi.cpp`+`server/mk_server.py` (992 lines): tokenizer via libllama + OpenAI server. `tools/vocab_freq.cpp`, `draft_vocab_32k.bin` (131072 B = 32768 int32, `ls -l`).
- Build: `make` (Makefile:6: `-O3 -arch=sm_XX -lineinfo -Xptxas -dlcm=cg -Xptxas -v`), needs llama.cpp build only for tokenizer (README:26-47). Run: `./qwen38_mk -m X.gguf -p ... -d 4 --draft-vocab draft_vocab_32k.bin` (README:262-265). Profiling: `MK_PROFILE=1` (README:286-288).

## Mechanisms

### 1. Persistent kernel structure
- **Grid**: `cudaLaunchCooperativeKernel(mk_cycle, dim3(nsm), dim3(512), args, SM_TOTAL)` (mk.cu:3532); `nsm = multiProcessorCount` (mk.cu:3275; 82 on a 3090), `NT=512` (mk.cu:58), `__launch_bounds__(NT,1)` (mk.cu:2811) => 1 block/SM, 16 warps, <=128 regs/thread [INF]. Dynamic smem <=99 KB (static_assert mk.cu:99,102); occupancy check mk.cu:3500-3503. Cooperative launch is used only for the co-residency guarantee: the barrier is hand-written.
- **Global state**: all weights/pointers live in `__constant__ ModelW cW` / `Bufs cB` (mk.cu:127-134, 160-209), per-layer `LayerW` with `Mat{w,d,type,K,N,rb}` (mk.cu:108-125). Device-resident control block `Ctl` (tokens, n, dn_c, mtp rows, out/n_acc) (mk.cu:147-158); host only copies it back after each launch (mk.cu:5926-5946).
- **Software grid barrier** `gsync` (mk.cu:317-337): `__syncthreads` -> thread 0 `red.release.gpu.global.add.u32` on `cB.bar` -> spin `ld.acquire.gpu` until counter >= `epoch` (epoch += gridDim.x per barrier) -> `__syncthreads`. Monotonic counter (no sense reversal), memset to 0 per launch (mk.cu:3530). Cost is **not quantified anywhere in repo**; profiling hooks record per-barrier timestamps, per-block arrival (first/median/last) and per-phase "fill" time (mk.cu:320-334, 340-342, 3540-3608). `[INF]` typical A100/3090-class one-counter barrier ~2-4 us x ~340 = ~1 ms of ~23 ms/cycle; measure with `MK_PROFILE`.
- **Phases per cycle** (mk.cu:2629-2692): `embed | per layer [attn: attn_proj, attn_core, attn_combine, resid_o(O proj + residual + requant) ; DeltaNet: dn_proj, dn_core, resid_sout ; both: ffn_gu, ffn_down] | lm_head`, each followed by `gsync`. Count: 1 + 48x5 + 16x6 + 1 = **338** barriers for the main pass [INF arithmetic from mk.cu:2637-2691]; README says "~340" (README:120). MTP adds 9 barriers per pass (M0, EH, attn_proj/core/comb, resid_o, ffn_gu, ffn_down, lm_head; mk.cu:2732-2782) => about 374 at `-d 4` (1 catch-up + 3 draft passes) [INF].
- **Phase contents**: `embed_prologue` (embedding row dequant, residual init, first norm+int8 quant; mk.cu:786-799) · `attn_proj` q(+gate)/k/v GEMV (mk.cu:1448-1458) · `attn_core` split-K flash decode + write this cycle's K/V (mk.cu:1524) · `attn_combine` merge partials, sigmoid gate, quant (mk.cu:2238-2278) · `resid32` GEMV + residual add + RMS-norm weight of the *next* consumer folded + requantise (mk.cu:804-909) · `dn_proj` alpha/beta(fp32)/qkv/z (mk.cu:1143-1212) · `dn_core` conv1d, l2norm, delta rule, gated rmsnorm (mk.cu:1217-1358) · `ffn_gu` gate+up GEMV with silu*up fused and requantised for down (mk.cu:914-959) · `lm_head` fused argmax/top-32 (mk.cu:2322-2454).
- **Cycle** (`mk_cycle`, mk.cu:2811-2858): switch on T in 1..5 (`MAXT=5`, mk.cu:56; so `-d 0..4`, qwen38_mk.h comment) -> `main_forward<T>` -> `mtp_pass<R>` catch-up rows in chunks <=5 -> `mtp_pass<1>` for drafts 2..nd. Host-side depth rule: draft one less (min 3) after a rejection (mk.cu:5931-5935).
- **Producer-side requantisation**: every phase writes the next consumer's int8 activation (`QBuf`, per-32 scale, per-16-chunk int sums, partial sum-of-squares for the next RMS norm) so consumers only `cp.async` it into smem and apply the RMS scale from partial sums (mk.cu:136-145, 357-378, 414-443). README:430-436.
- **Under-filled phases** `[INF from code]`: `dn_core` uses only 48 of 82 blocks (`if (h >= NVH) return`, mk.cu:1220); attention items = 4 kv-heads x <=20 chunks <=80 (mk.cu:78, 1429-1431); `ffn_gu` has 1088 warp-items for 1312 warps (mk.cu:927 vs 472-473). These are latency-bound; the authors mask them with L2 warm-up prefetch issued before the barrier (DeltaNet state, conv rows, norm weights, RoPE rows, first K/V tile: mk.cu:1185-1211, 1459-1488; README:455-456).

### 2. W4 (Q4_K) GEMV and the tensor-core batched variant
- **Data path**: weights are never staged in smem. Each lane issues `ld.global.nc.L1::no_allocate.L2::cache_hint` 16 B loads (evict-first policy from `createpolicy`) straight into registers (mk.cu:229-249); activations (T<=5 rows of int8) sit in smem via `cp.async` (mk.cu:400-443). `-dlcm=cg` in Makefile:6.
- **Q4_K math** (mk.cu:540-564, 491-516, 518-538): lane handles "units" u = lane+32i (32 weights = a low-nibble 16-chunk A + high-nibble 16-chunk B, 8 lanes per 144-B super-block, header 16 B re-read by the 8 lanes, qs 16 B each). Nibbles masked with `0x0F0F0F0F`, `__dp4a` against int8 activations, then `acc += fA*dx*sa + fB*dx*sb - gA*dx*sumA - gB*dx*sumB` (min term uses the activation chunk sums carried in `QBuf.meta`). Unsigned nibbles are used directly: no unpack-to-int8 multiply. T tokens share the same weight registers (`acc[R][T]`), so T=5 costs ~T=1 memory time (README:430-433, 148-150).
- **Tiling**: 2 rows per warp (`gemv_pairs`, mk.cu:686-699), warps strided over the whole grid `gwarp()=warp*gridDim+block` (mk.cu:472-473); `ffn_gu` = 16 rows of gate AND up per warp item, 2-4 rows in flight (mk.cu:927-958); `resid32` = 32-row group per block-item (16 warps x 2 rows), K up to 17408 via K-slices when T*K does not fit smem (mk.cu:830-908; smem layout mk.cu:72-76, SM_XU=4*FF).
- **Prefetch/overlap**: only *within* a phase: `pf_rows` L2-prefetches the first 576 B (or `rb/IT`) of each row before waiting for activations (mk.cu:709-717, 809, 920, 1151, 1453), so the DRAM stream starts during the activation fill. No next-phase weight prefetch across the barrier is visible (only small tensors/state are warmed), i.e. each barrier drains DRAM `[INF]`.
- **Other types**: Q5_K same shape +high bits (mk.cu:566-597); Q6_K repacked at load to 224-B blocks (6.7% bigger than 210 B) for aligned 16/8-B loads, scales+d in one 8-B word (mk.cu:599-638, repack 3173-3216); Q8_0 repacked (mk.cu:640-661, 3222-3249). Zero-point trick: Q6_K uses `gA = 32*fA` (mk.cu:624-625) = exactly how a symmetric int4 with offset 8 would be done with the existing sum term.
- **Bandwidth**: no GEMV GB/s figure is recorded in-repo; `bench/gemv_bench.cu` microbenches Q4_K GEMV variants on a ~1 GB matrix (gemv_bench.cu:1). The only DRAM-rate numbers are for attention (mk.cu:1776-1778: 290 -> ~600 GB/s for 3-5 rows, ~740 GB/s for 1-2). End-to-end `-d 0` decode is ~43 tok/s (README:148-149) => ~650-680 GB/s effective, ~70-73% of 936 `[INF, see 6]`.
- **Tensor-core variants**: (a) in-megakernel `ffn_gu_mma` for T>5 because "the dp4a path is compute-bound there" (mk.cu:979-1121, `mma.m16n8k16.s8`, mk.cu:967) - reachable only from `MODE_BENCH` (mk.cu:2805), decode dispatch stops at T=5 (mk.cu:2830-2836). (b) `bmm.cuh`: m16n8k32 s8 MMA on raw nibbles (bmm.cuh:61-69), 4-warp groups on 16-row items, register double-buffered load/step (`B4_LOAD/B4_STEP`, WA/WB ping-pong, bmm.cuh:151-257), up to 16 rows (12 staged in smem), fused 3-matrix launches (bmm.cuh:262-390). These are ordinary kernel launches per phase for the multi-agent path (README:356-378), not part of the persistent kernel; K=6144 projections (attn_o, ssm_out) stay on fp16 GEMM (mk.cu:4402-4403).

### 3. Gated DeltaNet and attention decode
- **DeltaNet state**: `dstate[layer 48][v-head 48][col 128][row 128]` fp32, 64 KB/head => 151 MB total [INF: mk.cu:1235, 2958, alloc 3433-3434]. `dn_core` = 1 block per value head; thread owns state column `col=tid>>2` and rows `32r..32r+31` in `float4 S[8]` registers (mk.cu:1223-1238). Delta rule per event: `kv=S.k; d=(v-g*kv)*beta; S=g*S+k*d; o=S.q` (mk.cu:1310-1341). `g=exp(softplus(alpha+dt)*A)`, `beta=sigmoid` (mk.cu:1265-1268); q,k l2-normalised, q scaled 1/sqrt(128) (mk.cu:1285-1293); `kh = h % NKH` (mk.cu:1221; tiled head order, see 7).
- **conv1d state**: `dn_proj` writes raw qkv projection rows into `conv[layer][CONV_RING=8][10240]` fp32 at slot `(n+t)%8` (mk.cu:1181, 61); `dn_core` computes the kernel-4 conv+silu from the last 3 ring rows + new rows (mk.cu:1250-1256). Rollback is therefore free for conv.
- **Lazy-replay rollback** (README:436-438; mk.cu:12-14): committed state `dn_c` advances by only 1 token per decode cycle (`dn_c=n+1`, mk.cu:2598, `n_commit=1` mk.cu:2634); `(k,v,g,beta)` of every processed token goes to a 16-entry ring `kvgb` (`KVGB = 16*128+48*128+96` floats/entry, mk.cu:62, 1298-1307); next cycle `dn_core` replays the `np=n-dn_c` accepted-but-uncommitted tokens from the ring before the T new ones, saving state after event `np+n_commit-1` (mk.cu:1240, 1310-1346). Cost: no snapshots; price is re-doing up to 4 state updates in registers (cheap, smem inputs, no barriers, mk.cu:1309).
- **Attention KV layout**: `[slot 17][kv head 4][ctx][HD=256]`, fp16 or int8-q8_0 with fp16 scale per 32 (`ksc/vsc [..][HD/32]`) (mk.cu:165-166); slot 16 = MTP block (mk.cu:64). 36,992 B/position at q8 (README:388). Optional paged pool of 256-position pages and shared prefix segments for parallel agents (kv_paged.cuh:5-16); low-bit 4-8 bit K/V + Hadamard rotation + q8 window (README:380-423, credited to another author README:408-409).
- **Split-K decode attention**: per (kv head, chunk) item; the 6 q heads x T tokens of a kv head form MMA rows M=6T (mk.cu:1526-1527); fp16 `mma.m16n8k16`, fp32 accum, online softmax; q is RMS-normed, RoPE'd (first 64 of 256 dims, base 1e7, table fp32 `[ctx][32]` mk.cu:1491-1503, 3470-3481, 59-60) and scaled by 1/16 (mk.cu:1602-1615). Chunk size is a function of context only (`ATT_BUCKET = 20*32` steps, `attn_dims`, mk.cu:78-79, 1429-1444) and the host trims T so a cycle never straddles a bucket (`att_cap_T`, mk.cu:82-86) => greedy output independent of draft depth.
- **Double buffering / pipeline**: base kernel keeps two K/V tiles of 32 positions in flight via `cp.async` (mk.cu:77, 1528, 1550-1570). Past 640 positions the q8 path switches to `attn_core_q8`: 3-deep software pipeline, one block barrier per 32-position tile, warp-specialised (0-3 Q.K^T, 4-11 P.V, 12-15 softmax), K ring 3 stages / V ring 4, q8->fp16 in registers (mk.cu:1770-1795, 93-98, 2219-2235). Claimed 290 -> 600-740 GB/s of KV at 101k context (mk.cu:1776-1778). At 100k ctx q8 KV traffic per cycle ~ 16 layers x 4 x 100k x 2 x 272 B = 3.5 GB `[INF]`, i.e. attention matters for long context.

### 4. MTP speculative cycle on device
- **Verify**: main pass over `T=1+n_draft` rows -> fused lm_head over all T rows in one weight stream -> thread 0 of the last-finishing block runs `sample_targets` (sampling only) then `main_control` (mk.cu:2672-2690). Greedy accept: `a` = longest prefix with `tgt[a]==tok[a+1]`, emit accepted drafts + `tgt[a]` as bonus, `n_acc=a`, `dn_c=n+1` (mk.cu:2590-2598).
- **Sampling** (exact speculative sampling): top-32 candidates per row (mk.cu:2305-2315, 2380-2453); order top_k -> top_p -> min_p -> temperature (mk.cu:2468-2490); counter-based RNG (mk.cu:2458-2464); accept draft d with `min(1,p/q)` (q from the draft head's own top-32 `dcand`, default) or `p(d)` for argmax drafts (`MK_DRAFT_SAMPLING=greedy`); on reject draw from `max(0,p-q)` residual (mk.cu:2512-2583; README:291-294, 452-454). Limits: `top_k<=32` (mk.cu:221; README:83), no penalties.
- **Catch-up rows**: after verify, MTP K/V is valid only below `mtp_valid`; `main_control` lists rows `p=mtp_valid..P` (<=8) with their tokens (accepted drafts + bonus) into `Ctl` (mk.cu:2611-2625); `mk_cycle` feeds them through the MTP block in chunks of <=5 rows (mk.cu:2842-2854). Hidden inputs come from `h_hist` ring (H_RING=16, final-normed hidden per position, written by `write_h_rows`, mk.cu:2281-2301, 2709).
- **MTP block** (layer 64 = normal attention+FFN block, mk.cu:2733): input `eh_proj(concat(enorm(embed(tok)), hnorm(h_prev)))` (Q8_0 GEMV K=2D, mk.cu:2704-2734, loader 3358-3362), own KV slot 16, then `shared_head_norm` -> **draft lm_head** (mk.cu:2749-2756). Only the LAST catch-up row produces logits -> `tok[1]`; drafts 2..nd are further 1-row passes consuming the previous draft token and `hm` (mk.cu:2779, 2855-2857, 2708-2709).
- **Draft-vocab trick**: draft head = rows of the *same* Q6_K lm_head selected by an id list (`draft_ids`, no copy, mk.cu:2329-2356, 3366-3395): 32,768 most frequent tokens (`tools/vocab_freq.cpp` over local code+docs, ~99.4% token coverage, README:447-451) + up to `DV_EXTRA=4096` tokens of the live sequence added on the fly (`dv_add`, mk.cu:3007-3025, 66; +3-8% decode on code, README:449-450). Verification always uses the full vocab, so only acceptance rate is affected. Bytes: 32768/248320 = 13% of lm_head per draft pass.
- **Acceptance/throughput (author numbers, README:136-146)**: reasoning mode 78.0 vs 50.9 tok/s (llama.cpp MTP), greedy code 126-140 tok/s at `-d 3/-d 4`, mean 106.7 over 4 prompts, 42.9 tok/s at `-d 0`. No acceptance-rate table in the repo (stats counters only, mk.cu:5942-5944).

### 5. lm_head (Q6_K, 248,320 rows)
- Hard requirement `output.weight` Q6_K (mk.cu:3364-3365); repacked to 224 B/256 values => 248320x5120/256x224 = **1.11 GB** [INF] (~7% of weight bytes). One pass streams it for all T verify rows at once (`lm_head<T>`, `mrows<TL,5,2,T_Q6K>`, mk.cu:2349-2358). No logits are materialised (except scoring mode, mk.cu:2360-2368): per-warp running argmax/top-32, per-block result, then `atomicAdd` last-block-done reduction + `__threadfence`, the winner reduces all blocks' partials (mk.cu:2380-2453). Followed by its own barrier (PH_LMHEAD, mk.cu:2691).
- Per MTP draft pass the head reads only the 32k-row subset (0.147 GB [INF]) via `draft_ids`; this is the only lm_head pruning.
- `token_embd` (Q4_K, 0.72 GB) is *not* on the GPU by default: mapped pinned host memory, one 2.8 KB row read per token via a single coalesced stage (mk.cu:773-782, 3398-3416; README:349-350).

### 6. Per-token byte budget and roofline `[INF: computed from mk.cu constants and GGUF block sizes; not measured]`
| item | bytes | source |
|---|---|---|
| FFN gate/up/down, 64 layers, Q4_K 0.5625 B/w | 9.63 GB (all-Q4) | mk.cu:52-53, 3352-3356 |
| DeltaNet proj+out (48 L) / attn (16 L) | ~3.0 / ~0.9 GB | mk.cu:3308-3328 |
| lm_head Q6_K repacked | 1.11 GB | mk.cu:3174 |
| trunk+head total | 15.1 GB (all Q4/Q5) .. 17.7 GB (every Q4/Q6-optional tensor Q6) | computed; file is 16.4 GB incl. 0.72 GB embd (docs/models.md:22) => ~15.3-15.8 GB on device |
| per MTP draft pass | block ~0.27 GB + head 0.147 GB | computed |
| DeltaNet state r/w per cycle | 151 MB read + 151 MB write | mk.cu:2958 |
- At 936 GB/s: 15.1-15.8 GB => 16.1-16.9 ms floor for the main pass; reported `-d 0` 42.9 tok/s = 23.3 ms => **~650-680 GB/s effective (~70-73% of peak)**, README itself says decode is "close to the memory-bandwidth limit" (README:148-149). With `-d 4` add ~4x0.42 GB (~1.7 GB, +11%) per cycle.
- `MK_PROFILE=1` prints per-phase total/per-call time, barrier arrival spread, and a "min-composed" launch estimate that filters time-slicing by other GPU users (mk.cu:3573-3608; README:286-288), but **no profile output is committed to the repo**: the phase-time breakdown (where the missing ~28% goes: ~340 barriers, tails, 48-block dn_core, attention) must be measured by us.

### 7. What is hard-wired to GGUF Q4_K_M / these dims, and what a compressed-tensors W4A16 g128 (+g_idx) port must change
- **Dims** are `constexpr` and checked at load: `block_count==65`, `embedding_length==5120` (mk.cu:52-69, 3270-3272). Our dims match, so these stay.
- **Quant type is a template parameter per tensor role** (e.g. `resid32<T,6,T_Q4K>` o-proj, `<T,6,T_Q5K>` ssm_out, `<T,17,Q4K|Q6K>` ffn_down; mk.cu:2649, 2656, 2662-2663); loader `need()` rejects other mixes (mk.cu:3293-3359; README:298-312). Q4_K layout (144-B superblock, 6-bit scales+mins, nibble order) is baked into `rows_q4k`, `q4k_scales`, `dequant_q4k`, prefill GEMMs, bmm kernels.
- **Port W4A16 g128 symmetric**: add a `Mat` type whose block = 64 B nibbles/128 values + fp16 scale; reuse `accR` with `gA=gB=8*scale` (zero-point via the activation chunk sums already in `QBuf.meta`; Q6_K does the same with 32, mk.cu:624-625) and `fA=fB=scale` per group of 128 = 4 units. K in {5120, 6144, 17408} all multiples of 256 so the 256-unit loop structure (b=u>>3) is unchanged. Repack offline in Python (the repo repacks Q6_K/Q8_0 at load, mk.cu:3173-3249).
- **Activation precision**: kernels quantise activations to int8 per-32 (mk.cu:357-378), i.e. W4A8-ish, not W4A16; authors measured KL 0.0009 / 99.08% top-1 agreement vs llama.cpp (README:202); we must re-measure for our checkpoint.
- **`g_idx` actorder**: nothing in repo handles permuted K. Weights in act-order storage are grouped by `g_idx`, not by contiguous 128 columns; a dp4a kernel needs contiguous groups, so options are (a) permute activations per matrix (producer must emit one quantised copy per distinct permutation; q/k/v/gate/up share an input but have different `g_idx`) or (b) undo the permutation offline and treat each column's scale individually (kills the `dp4a` + one-scale-per-chunk shortcut). `[INF]` Check whether `static` actorder in this checkpoint yields identical perms per same-input matrices.
- **Non-quantised tensors**: GGUF has F32 alpha/beta (norm-folded at load, mk.cu:3329-3350), `ssm_a` (already `-exp(A_log)`), conv `[10240][4]`, `ssm_dt`, norms (mk.cu:3325-3328, 3305-3306) - HF safetensors need the same transforms (`[INF]`: llama.cpp's converter does them; verify vs HF tensors). MTP: GGUF `nextn.*` names + Q8_0 `eh_proj` (mk.cu:3358-3363) -> map from `model_mtp.safetensors`; if `eh_proj`/lm_head/embed are bf16 in our checkpoint they need new GEMV types (bf16 lm_head = 2.54 GB vs 1.11 GB Q6_K: +1.4 GB/cycle = +1.5 ms floor `[INF]`, so decide whether to requantise lm_head).
- **Head ordering**: `kh = h % NKH` (mk.cu:1221) and q|k|v channel layout `[q 2048 | k 2048 | v 6144]` (mk.cu:1199) assume the GGUF (llama.cpp) tiled V-head order; HF Qwen3.5 checkpoints use repeat-interleave k-head grouping `[INF]`, so either permute v-head dims of qkv/z/alpha/beta/out/conv/dt/A offline or change to `kh = h/3`.
- **Attention q layout**: q projection output is per-head `[q 256 | gate 256]` stride 512 (mk.cu:1586, 2268); partial RoPE 64/256 dims, base 1e7 hard-coded (mk.cu:59-60, 3470-3481).
- **KV**: engine owns its KV (contiguous or 256-pos paged pool `[page][slot][head][256][HD]`, kv_paged.cuh:5-16), formats fp16/q8_0/low-bit; **no fp8**. Using vLLM's KV blocks/FP8 config means rewriting `kv_own_row`, `kv_stage`, `attn_core_*` K/V conversion (mk.cu:1414-1427, 1797-1811); `[INF]` sm_86 has no hardware fp8->fp16 cvt (sm_89+), so fp8 KV needs a software unpack like `aq_q8x2` (mk.cu:1797-1811).
- **State handoff to/from vLLM prefill**: the engine requires its own state exactly: fp32 DeltaNet `[col][row]`, conv ring, `kvgb` ring, `h_hist`, MTP K/V, `Ctl` (mk.cu:2958-2980, 3433-3470). Its prefill (`prefill.cuh`) is k-quant specific (README:458-494); we would use vLLM prefill + a state-conversion step.
- **Host stack**: tokenizer/chat template/OpenAI server via libllama (`capi.cpp:1-81`, `server/mk_server.py`); not needed under vLLM.

### 8. Limits / open gaps listed by the authors (README:519-532, plus inline)
- Specialised to Qwen3.8-27B shapes + listed quant recipes; loader rejects others; tuned for sm_86; sm_75 experimental (README:521-523, 496-517, int8 m8n8k16 "compiled, not yet run on Turing hardware" README:503-504; no Turing KLD/long-context validation README:514).
- Sampling: temp/top_k<=32/top_p/min_p only, penalties ignored with warning (README:83-85, 524-525). Text only; one sequence and one checkpoint; editing earlier context re-prefills; no slot save/restore (README:526-529).
- `--parallel` max 8 conversations / 16 rows per step, Q4_K_M engine only; ~33K tokens each with 4 agents on 24 GB (README:530-531, 356-378). Mixed-quant engine single conversation only (README:378).
- `MK_KV` low-bit caches: single conversation, sm_80+, errors with `--parallel`, Turing, `MK_ATTN=1`, `--kv f16` (README:420-421, 532).
- Output not bit-identical to llama.cpp; verifying 1 vs 4 tokens per pass rounds differently so `-d` can change near-tie outputs (README:192-195, 233-235). Long-context accuracy measured from a single sequence (README:217).
- Mixed engine compile needs ~11 GiB RAM (README:253-256); original engine build ~2 min (README:262). Decode throughput numbers were measured before later attention optimisations (README:163-164).
- Embedding/head formats hard requirement: Q4_K embeddings + Q6_K output head (README:278).

### 9. License
- **AGPL-3.0-only**: `megakernel/LICENSE:1-2` (GNU AGPL v3 text), `megakernel/pyproject.toml:11` and repo `pyproject.toml:40`, `docs/licensing.md:3`; network-use clause `docs/licensing.md:7-8`; repo root has a CLA (`CLA.md`, `CONTRIBUTING.md`, docs/licensing.md:14-15) and says separate commercial terms may be offered (docs/licensing.md:10-12). No SPDX headers in the megakernel sources (grep). Links `ggml.h`/libllama (MIT, not vendored). The low-bit KV work is a third party's (README:408-409).
- Implication `[INF, not legal advice]`: copying kernels into a vLLM-integrated (Apache-2.0) package or a hosted service pulls in AGPL source-offer obligations; treat as a design reference and write clean-room kernels, or isolate/publish under AGPL deliberately.

## Worth stealing (ideas, not code - see §9) / Reusable pieces vs rewrite
**Reusable as designs** (small, well-specified, with the mk.cu line where the reference lives):
- Hand-rolled monotonic-counter grid barrier + cooperative launch only for residency (mk.cu:317-337, 3532): ~20 lines.
- Phase decomposition and the exact 5/6-phases-per-layer cut; fusing next-norm weight and requantisation into the producer's epilogue (mk.cu:804-909, 914-959) so no separate norm/quant phase.
- `QBuf` activation format with per-16 int sums so asymmetric/zero-point weights cost no extra pass (mk.cu:136-145, 513).
- L2 evict-first weight stream + evict-last prefetch of next latency-bound phase's small data (mk.cu:227-278, 1185-1211, 1459-1488).
- DeltaNet lazy replay with a (k,v,g,beta) ring and committed-state pointer (mk.cu:1240-1346) - removes state snapshots for MTP/spec rollback; conv state derived from a raw-row ring (mk.cu:1181, 1250-1256).
- T-invariant attention partition + `att_cap_T` for depth-independent greedy output (mk.cu:82-86, 1429-1444).
- Pipelined q8 split-K attention with specialised warps (mk.cu:1770-1795).
- Fused lm_head argmax/top-32 with last-block-done reduction (mk.cu:2380-2453); exact speculative sampling on device from top-32 (mk.cu:2547-2583); draft-vocab subset + on-the-fly sequence tokens (mk.cu:3007-3025, 3368-3396); on-device catch-up row bookkeeping (`main_control`, mk.cu:2586-2626).
- Register-held delta-rule state per head with 4-lane row split (mk.cu:1223-1238); strip local-memory spills (stack frame 2,496 -> 624 B, README:405-407; `cudaLimitStackSize=0`, mk.cu:3504-3509).
**Must rewrite for us**: every weight-format-specific path (GEMV/prefill/bmm for W4A16-g128 + `g_idx`), loader (GGUF -> safetensors/compressed-tensors, head reorder, MTP tensors), lm_head/embed handling for our checkpoint dtypes, KV format/layout (fp8 / vLLM paging), state handoff with vLLM prefill, scheduler/server layer, and the 48-block `dn_core` / barrier-tail inefficiencies if we want to beat ~70% of roofline.

## Dead ends / lessons the authors mention
- Old attention at one 32-position tile per SM per ~4.8 us (5 dependent stages, 5 block barriers/tile) reached only ~290 GB/s of KV; the 3-deep pipeline fixed it (mk.cu:1772-1778).
- dp4a GEMV is **compute-bound above 5 rows**, so the persistent kernel stops at T=5 and moves bigger batches to int8 MMA (mk.cu:979-980, 56; bmm.cuh:1-5).
- Contiguous per-warp quarters of a weight row stream 5-10% slower than interleaving the 4 warps' blocks (bmm.cuh:161-162); folding a reduce kernel into the last block "saved nothing: its serial tail cost what the launch did" (mk.cu:4314); fetching scales from global instead of decoding compact ones: 146 -> 139 us on gate/up but 1.5 us slower on smaller launches (bmm.cuh:97-98); more drafts per slot: 4x2 drafts 49 tok/s vs 46 with 1 at short context (mk.cu:5114-5115).
- Conditional arrays in a persistent kernel spill into the local stack frame, which the driver reserves for all 1,536 thread slots of every SM: code comments (mk.cu:1579-1581, 1595, 3504-3506) and README:405-408 report cutting the `mk_cycle` frame 2,496 B -> 624 B and setting stack limit 0 to free VRAM for context. Lesson: check `ptxas -v` per phase function (Makefile:13-15 writes `ptxas.log`).
- Float summation order would depend on T/partition (so greedy output on near-ties would depend on draft depth); the authors therefore made the attention split a function of context only and trim T at bucket edges (mk.cu:82-86, 1429-1435; README:233-235). Residual dependence remains for verify-1-vs-4 rounding (README:233-235).
- Verified tokens per cycle matter more than raw decode: `-d 0` 42.9 vs llama.cpp 40.4 tok/s; the 1.5-1.9x comes from verifying 4-5 drafted tokens "at almost no extra cost" (README:148-150).
- Full-vocab draft head is wasteful: restricting to 32k tokens (+live sequence tokens) changes only acceptance, never output (README:447-451).
- Template explosion: separate `qwen38_mixed.cu` engine because the mixed-quant kernels needed ~11 GiB RAM to compile (README:244-256).
- Prefill is deliberately *not* done in the megakernel (`MK_PREFILL=mk` exists as a fallback, README:288-289); batched tensor-core kernels give ~1500 tok/s (README:458-494, 155-161).

## Gotchas
- Model name "Qwen3.8-27B" in the repo = llama.cpp arch `qwen35` (mk.cu:1-2, 3270); asserts `block_count == 65`.
- All throughput/KLD numbers are the authors' own on one 3090; unverified by us. README says the Q4_K_M numbers predate later attention/draft-vocab changes (README:163-164).
- The engine is one process owning the GPU (single stream, own KV/state, blocking `cudaMemcpy` of `Ctl` after each cycle, mk.cu:5937): integrating with vLLM's scheduler is a design problem, not a drop-in.
- `MAXT=5` (mk.cu:56) caps drafts at 4; `MAXT_BUF=12` buffers exist only for bench (mk.cu:57).

---

Pinned at `8f4bac4` (master, 2026-10-29) · 322 files · 7,1M
Update with: `git -C vendor/open-jet pull` then refresh `commit:` above.
