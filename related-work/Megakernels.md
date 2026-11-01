---
title: Megakernels
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/HazyResearch/Megakernels
path: vendor/Megakernels
commit: 7309cec
branch: main
commit_date: 2025-06-02
---

# Megakernels
# Megakernels

HazyResearch "Look Ma, No Bubbles" low-latency megakernel for Llama-3.2-1B bf16 batch-1 decode: an **on-GPU instruction interpreter** (one persistent block per SM, warp-specialised, paged smem, global-counter dependencies), ThunderKittens (TK) based, Hopper/Blackwell only. Pointers repo-root-relative at `7309cec`. Studied 2026-10-30.

## Why I cloned it

(1) The canonical reference for "interpreter + counters + paged smem" megakernels, vs the static-phase `grid.sync` design of `related-work/MegaQwen.md` / `related-work/lucebox.md`. (2) What would transfer to sm_86 (no TMA, no mbarrier-async bulk copies, 100 KB smem). (3) Where per-instruction overhead goes. (4) License.

## Orientation

- `include/megakernel.cuh:15-156` `mk_internal` = the VM; `include/util.cuh` = state/semaphores/dispatch/`MAKE_WORKER`; `include/controller/*` (`include/controller/controller.cuh:24-143` main loop) = instruction fetch + page allocator + semaphore constructor; `include/config.cuh:7-52` = all knobs.
- `demos/low-latency-llama/` = the 7 ops (`demos/low-latency-llama/llama.cuh:7-13` opcodes): `rms_matvec_rope_append.cu`, `attention_partial.cu`, `attention_reduction.cu`, `matvec_adds.cu` (O-proj/down-proj + residual), `upgate.cu`, `rms_lm_head.cu`, shared `matvec_pipeline.cuh`, `utils.cuh`.
- Python side: `megakernels/scheduler.py` (DAG -> per-SM queues -> int32 tensor), `megakernels/demos/latency/scheduler.py` (Llama DAG), `megakernels/generators.py:95-143` (`MK_Generator`), `megakernels/mk.py`.
- Build: `demos/low-latency-llama/Makefile:19-31` (`GPU=H100|B200`, also `4090`->`-DKITTENS_4090 -arch=sm_89`, `A100`->sm_80 flags exist but the VM needs TMA/mbarrier [INFERENCE: untested, unlikely to work on sm_86]). **`ThunderKittens/` submodule is empty in vendor/** (`.gitmodules:1-4`, branch `bvm-single-ctrl-pre-new-warps`): cannot be compiled from this checkout. Not built by me.
- **License MIT** (`LICENSE:1-3`, (c) 2025 HazyResearch): copying with notice is allowed.
- Results are NOT in the repo (README is build/run only, `README.md:1-46`). Numbers below are from the blog https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles (external, fetched 2026-10-30).

## Mechanisms

### Interpreter: per-SM static instruction queue, 20 warps, 1 block/SM
**What:** grid = `sm_count` blocks (H100 132, B200 148: `demos/low-latency-llama/llama.cuh:25-26,144-146`), `config::NUM_THREADS` = (4 + 16 consumer) warps (`include/config.cuh:24-27`), `__launch_bounds__(NUM_THREADS,1)` (`include/megakernel.cuh:166-171`). Warp roles (`include/megakernel.cuh:118-140`): 16 consumer warps (registers raised to 104, `include/config.cuh:50`) and warpgroup of 4 non-consumer warps (64 regs): loader, storer, launcher (tensor-core issue, Blackwell only), controller. Each op is a C++ struct with `controller/loader/launcher/consumer/storer::run` (`include/util.cuh:260-304` `MAKE_WORKER`, dispatch by opcode through a type list `include/util.cuh:32-55`; unknown opcode -> `trap`).
**Instruction format:** 32 x int32 per instruction (`include/config.cuh:14`, `megakernels/scheduler.py:13,274-278`): opcode, layer, block ranges, etc. Fetched per SM by `smid`, not `blockIdx` (`include/util.cuh:27-30`, `include/controller/instruction_fetch.cuh:11-24`), 2-deep ring (`include/config.cuh:9`) with semaphores `instruction_arrived/finished` (`include/megakernel.cuh:23-33,82-86`).
**Cost:** blog B200 600 us/token breakdown: 250 us storing/awaiting/loading activations per dependency, 200 us real RMSNorm+matvec, 30 us waiting on weights, 40 us warp-barrier overhead ("async barrier ~60 ns even if already passed"), 80 us setup/page marking. **Verified:** read.

### Static host-side schedule, replayed every token
**What:** `schedule_qkv/upgate/downproj/lm_head` split each op over all SMs (e.g. QKV: `round(sm*blocks_per_sm)` ranges of 16-row output blocks, `megakernels/demos/latency/scheduler.py:118-142`; upgate strided `range(sm, n, sm_count)` :145-164; downproj jobs = (col split, row block) so that K=8192 splits into 4 reductions over atomically-added 2048-wide slabs :167-213). `make_dag_layer` (:270-386) wires dependencies; attention is **8 instructions per layer (one per KV head), num_partials=1, no reduction** (:276-277 `assert skip_attn_reduction`) so only 8 of 132 SMs do attention at a time; `pick_num_attention_partitions` (:21-35) is dead for this path. `assign_dag_to_sms` (`megakernels/scheduler.py:94-151`) is list scheduling by cost heap (max cost first onto least-loaded SM); alternatives `rr/zz/wave/pool` :154-271; `tensorize_instructions` pads queues with NoOps (:281-308). Schedule is built once, `pos_id` is a kernel scalar (`megakernels/generators.py:121-135`), barriers zeroed with `fill_(0)` before every token (:115-116,132); embedding and argmax run in torch outside the kernel (:127-130,140-143); LM head (+RMS) is in-kernel (`rms_lm_head.cu`).
**Why:** zero scheduling cost on device. **Cost:** instruction count per SM padded to the max; no dynamic load balancing. **Verified:** read.

### Dependencies: global-memory counters, spin with nanosleep, no grid barrier
**What:** `Bar[layer][opcode-1][idx]` int32 tensor (`demos/low-latency-llama/llama.cuh:103-108`, shape layers x 10 x (heads + 2 kv)). Producer: TMA store -> `tma::store_async_wait()` ("full wait! must be visible in global!") -> `atomicAdd(&Bar[..],n)` (`demos/low-latency-llama/matvec_adds.cu:165-176`, `demos/low-latency-llama/rms_matvec_rope_append.cu:154-163`, `demos/low-latency-llama/upgate.cu:108-121`, `demos/low-latency-llama/attention_partial.cu:651-667`). Consumer: one lane spins `while (*(volatile int*)&Bar[..] < EXPECTED) __nanosleep(..)` (`demos/low-latency-llama/matvec_adds.cu:129-133`, `demos/low-latency-llama/upgate.cu:35-39`, `demos/low-latency-llama/rms_matvec_rope_append.cu:49-60`, `demos/low-latency-llama/attention_partial.cu:308-329`). Granular: QKV counters per 4 x 16-wide blocks = per head, so attention for KV-head h waits only for its K and V blocks (`demos/low-latency-llama/attention_partial.cu:314-326`); MLP intermediate in 4 chunks so down-proj waits for its slab only (blog).
**Memory ordering:** the `fence.acq_rel.gpu` is commented out everywhere (`demos/low-latency-llama/matvec_adds.cu:172` and `demos/low-latency-llama/rms_matvec_rope_append.cu:158` with the note "possible we need sc here but I don't think so"; bare at `demos/low-latency-llama/upgate.cu:114`, `demos/low-latency-llama/attention_partial.cu:642`); correctness relies on TMA-store completion + atomic + volatile read + TMA/`warp::load` of the data. [INFERENCE: works on Hopper's coherence behaviour, not guaranteed by the PTX model; we must use `fence.acq_rel.gpu` / `ld.acquire.gpu` / `red.release`.]
**Bug-in-waiting:** `static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;` (`include/config.cuh:48`) is a `bool`, so every spin sleeps `__nanosleep(1)` (value becomes true) [INFERENCE from the type]. **Verified:** read.

### Weight streaming: paged smem + TMA + cross-instruction prefetch
**What:** H100 smem 227 KB: static control ~10.5 KB (`include/config.cuh:34-37`: 512 + 2 x (4096 scratch + 640 instr/timing + 256 semaphores)), rest = 13 pages x 16 KiB = 208 KiB (`include/config.cuh:38-44`, `static_assert(NUM_PAGES == 13)`). Matvec ops use `matvec_pipeline` (`demos/low-latency-llama/matvec_pipeline.cuh:5-27`): page 0 = activations, then 3 input stages x 4 pages, each stage = a 16-row x 2048-col bf16 tile (64 KiB) loaded by one `tma::load_async<EVICT_FIRST>` per page (`:117-160`, `demos/low-latency-llama/upgate.cu:46-55`); 16 consumer warps each handle a 16 x 128 slice (`REDUCTION_DIM_PER_WARP`=hidden/16, `:14-15`) and write 16 floats to scratch; storer sums 16 warp partials (`demos/low-latency-llama/utils.cuh:103-120`). On Hopper the "matvec" is CUDA-core mul + row_sum, not MMA (`demos/low-latency-llama/utils.cuh:73-100`; blog: "on Hopper better on CUDA cores"); only Blackwell uses `mma_ABt` (:39-71).
**The key trick:** next instruction's loader can start TMA-loading weights as soon as the *previous* instruction releases pages, even while that instruction's consumers/storer are finishing and before its activations are ready: `release_lid` orders the page release (unused pages first, activation page, then weights; `demos/low-latency-llama/matvec_pipeline.cuh:70-102`, `demos/low-latency-llama/attention_partial.cu:276-282`), the controller computes the physical page order for the next instruction (`include/controller/controller.cuh:74-103` Step 2, op hook `include/controller/page_allocator.cuh:16`), and only the *activation* read waits on the dependency counter (`demos/low-latency-llama/matvec_pipeline.cuh:312-334`). So weight HBM traffic is overlapped with the dependency wait. Attention likewise issues TMA loads of old K/V blocks before waiting on the newest KV append (`demos/low-latency-llama/attention_partial.cu:357-383`, `wait_for_kv` only before the last block :370).
**Cost/limits:** 3-stage 64 KiB x 3 does not fit sm_86's 100 KB smem. **Verified:** read.

### Attention (GQA, partial + reduction)
`demos/low-latency-llama/attention_partial.cu:6-15`: GQA ratio 4, 3-stage K/V pipeline of 16-token blocks via TMA, Q via `cp.async.cg` (:268-273), `mma_ABt`/`mma_AB` online softmax (:473,509), per-block LSE/O partials to global for `attention_reduction.cu`. No DeltaNet anywhere.

## Results (blog, not repo) and roofline

H100 SXM 3.35 TB/s, Llama-3.2-1B bf16 = 2.48 GB/token -> memory limit ~1350 fwd/s. Megakernel: "78% of memory bandwidth", <1 ms/token on H100 (~1050 tok/s implied), 1.5x SGLang, ~2.5x vLLM; vLLM/SGLang <=50% BW. B200: <680 us (blog says breakdown total 600 us), ~3000/s theoretical -> ~55%. Dummy-kernel launch cost 2.1 us stream / 1.3 us graph on H100 (blog). PDL rejected as "too coarse" (needs all QKV before attention). [INFERENCE: ~1050 = 0.78 x 1350.]

## Speculative decoding
None. (Their `throughput` demo is a different, batched design: `megakernels/demos/throughput/`.)

## Transferable to our 3090/W4/MTP megakernel
- **Counters instead of grid barriers**: per-head / per-slab counters let dependent work start as soon as its slice exists; the biggest idea. Works on sm_86 with `atom.add.release.gpu` + `ld.acquire.gpu` (do not copy the missing fences).
- **Prefetch next op's weights before its input is ready** (page hand-off). On sm_86 use `cp.async.cg` 16 B (or `cp.async.bulk` is sm_90, unavailable) into a smem ring; budget 99 KB/block: e.g. 2-3 stages of a 16-row x 5120-col W4 tile (= 40 KB + scales) -- Little's law needs only ~8-10 KB/SM in flight [INFERENCE: 936 GB/s x ~0.7 us / 82 SMs].
- Static host-computed per-SM queue + replay every token: good fit for fixed 64-layer hybrid with tree/MTP verify batches; cost-balanced assignment (`megakernels/scheduler.py:94-151`) handles unequal DeltaNet vs attention phases.
- Per-dependency overhead from the blog (~4 x 500 ns per dependent instruction) => 64 layers x ~6 dependent ops x ~2 us = ~0.8 ms = ~4.7% of a ~17 ms token [INFERENCE], same band as msaroufim 0.7-4.6%: the interpreter's win over graph+PDL is *not* the launch cost but weight prefetch across the dependency.
- Split-K with atomic reductions for the K=17408 down-proj (their `reduction_block_idx` + counters).

## Dead ends / lessons
- Hopper-only; TMA, mbarriers, `elect`-style single-lane controllers, `cluster_dims` do not exist on sm_86. Porting = rewriting the pipeline, not a recompile. TK submodule empty.
- Attention is 8 instructions per layer with 1 partition: only 8 SMs busy at a time, fine at 32-token prompts, would cap long context (no partials in the latency path: `megakernels/demos/latency/scheduler.py:276-277`).
- Static schedule hard-codes shapes; `GMEM_SPIN_LOOP_SLEEP_NANOS` type bug; missing fences.

## Verdict
1. Best design reference for dependency counters + cross-op weight prefetch; MIT, so ideas and even small snippets are legal to reuse, but nothing compiles on sm_86 as-is.
2. For a bandwidth-bound W4 27B on 82 SMs the overheads it removes (~4-5% of a token) are real but small; the same overlap can be had in a persistent-block kernel with counters and cp.async.
3. 78% of roofline on H100 is the achievable-efficiency bar to cite; sm_86 DRAM tops out nearer 85-90% of 936 GB/s [INFERENCE].

---

Pinned at `7309cec` (main, 2025-06-02) · 61 files · 776K
Update with: `git -C vendor/Megakernels pull` then refresh `commit:` above.
