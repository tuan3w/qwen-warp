# CuTe DSL + megakernel spike: findings

Hardware: RTX 3090 (sm_86, 82 SMs), **shared with a running `serve.py` (only ~860 MB free)**, CuTe DSL 4.7.1, torch 2.13, CUDA 13.1 nvcc present but not used by the CuTe path.
All weights random (repacked layout), 2 weight sets cycled to defeat the 6 MB L2. Numbers repeat within ~0.05 ms across runs. Prior art: [SURVEY.md](SURVEY.md).

## Verdict

| Question | Answer | Evidence |
|---|---|---|
| Does CuTe DSL work on sm_86 with the instructions the engine uses (mma.sync m16n8k16 bf16, lop3, `ld.global.nc.L1::no_allocate`, acquire/release atomics, cooperative launch)? | **Yes**, through `inline_ptx` and `cooperative=True`. | `s1_smoke.py`, `ptx.py` |
| Can it match the hand-written W4A16 GEMV? | **Yes: bit-identical output, 98.5-100 % of its speed**, 0.15-0.5 s compile. | S2 table |
| Does a table-driven interpreter (flexible, runtime shapes) cost anything? | **Yes, +7 % vs a graph of constexpr kernels** (decode of the stage row sits on the critical path before the first weight load). Prefetching the next stage's weights before the barrier recovers about half. | S3 table |
| Does a persistent "megakernel" beat the CUDA graph on this card? | **No.** Best variant -2.7 % on a GEMV-only chain, and it loses 4-7 % to a graph of constexpr kernels. | S3 table |
| Is the edit-run loop fast? | Compile 0.15-0.5 s per (shape, config) vs **6.0 s for nvcc on `gemv.cu` alone** (measured; plus relink of `libswiftmk.so`, and the bigger files take longer). Full edit -> compile -> bit-exact check -> benchmark in CuTe: ~3 s wall. A ~10-20x compile gain, not minutes vs seconds; the bigger win is shape/config sweeps without templates and rebuilds (`s2_sweep.py`: 48 compiles + benches in 11 s). | `s2_cute.py` wall 3.3 s, `s2_sweep.py` 10.6 s |
| Is the launch overhead we would be removing real? | **No.** The repo's own nsys (docs/PERFORMANCE.md:244): launch idle 0.04 ms per 21 ms cycle, W4 GEMVs at 92-96 % of read peak. My graph of hand CUDA GEMVs sits at the sum of its parts (15.24 vs 15.3 ms). | S3 + PERFORMANCE.md |

So: **CuTe DSL for the kernels: yes. Megakernel as the execution model on sm_86: no.** The vLLM/SGLang designs (survey) agree: they fuse at *boundaries* and keep a stock twin per slot, and their persistent grid exists on Blackwell/MI355 where all-reduce and launch latency dominate, not on one memory-bound Ampere card.

## S2: one W4A16 GEMV, CuTe DSL vs `smk_w4r_gemv` (T=1 / T=8, us, best of 6 configs for CuTe)

| shape (N x K) | CUDA | CuTe | ratio | max diff vs CUDA |
|---|---|---|---|---|
| down 5120 x 17408 | 54.9 / 55.5 | 55.1 / 56.1 | 1.00 / 1.01 | 0 (same KS) |
| gate\|up 34816 x 5120 | 107.0 / 109.7 | 107.4 / 110.3 | 1.00 / 1.01 | 0 |
| qkv 16384 x 5120 | 51.8 / 52.8 | 52.6 / 53.8 | 1.02 / 1.02 | 2e-7 (other K-split order) |
| o 5120 x 6144 | 21.0 / 21.6 | 21.2 / 22.1 | 1.01 / 1.02 | 0.03 of 40 (bf16 ulp, other K-split) |

`cute_w4.py` is a line-for-line port of `k_w4<NW,KS,U,.,RP=1,NT=1>`; the dequant + 8 mma of one 128-k group is one `inline_ptx` block (`ptx.group_mma`). It is Python-metaprogrammed PTX, not CuTe layout algebra; no layouts, TV-partitioning or `cute.copy` were needed. Untried: NT=2 (16-row pass), the SEGS epilogue with 3 outputs (SILU epilogue exists in `mega.py`), the lm_head kernels.

## S3/S4: 64-layer decode-shaped chain (T=1, 7 stages/layer = engine's 7 kernels/layer; norm and the DeltaNet/attention core replaced by stand-ins)

`mega.py`: one stage-code body, run either one stage per launch (graph) or as one cooperative persistent kernel (164 CTAs = 2/SM, split arrive/wait barrier on a monotone counter with a device epoch so nothing is reset between launches, graph-safe). Every variant's output is **bit-identical** to the graph version, so the sync is correct.

| chain, 448 stages, ms per step | with naive CuTe norm | norm stages removed |
|---|---|---|
| sum of isolated GEMVs + mix (floor) | - | ~15.3 |
| graph, **hand CUDA GEMVs** (`smk_w4r_gemv`) | 16.46 | **15.24** |
| graph, **CuTe constexpr GEMVs** (`cute_w4.py`) | 16.69 | 15.45 |
| graph, CuTe table-driven (runtime N,K) | 17.42 | 16.37 |
| persistent, barrier only | 17.51 | 16.29 |
| persistent + next-stage register prefetch | 17.06 | **15.91** |
| persistent + L2 prefetch (2 or 4 groups/thread) | 17.06-17.07 | 15.95-16.03 (no gain) |
| persistent, 82 CTAs / 246 CTAs | - | 16.87 / refused (cooperative limit is 164) |

- Per-stage timeline (persistent kernel, CTA 0): every GEMV stage takes its isolated time (qkv 55 vs 53, o 19 vs 21, gu 107 vs 107, down 55 vs 55 us). **No bubble at stage boundaries to reclaim.**
- Empty-stage cost: graph 1.15-1.19 us/stage, persistent barrier 0.77 (82 CTAs) / 0.89 (164) / 1.12 (246) us. Total possible saving over 448 stages: 0.1-0.2 ms (~1 %).
- The 1.2 ms (7 %) that norm adds is my naive 1-CTA-per-token norm (13 us/stage); an implementation problem, equal in every variant. Fusing norm into the next GEMV's x-load removes the stage.
- Eager host cost per call: CuTe compiled fn 10.4 us, ctypes into libswiftmk 10.7 us, `torch.add` 2.9 us. Kernels launched through CuTe capture into `torch.cuda.CUDAGraph` normally (pass `cuda.CUstream(torch.cuda.current_stream().cuda_stream)` at call time, not at module load).

## What was NOT tested (so do not read it as measured)

- The real DeltaNet decode step and the int4-KV attention (the non-GEMV 3.4 ms of the 18.4 ms step, by subtraction from docs figures; [INFERENCE]). A megakernel could only win by overlapping *low-bandwidth* core stages with streaming the next layer's weights. My stand-in core is 4.5 us and L2 prefetch gave 0; with a 20-40 us DeltaNet core this could differ.
- Real model, real checkpoint, T>1 beyond one T=4 run, accuracy end-to-end, the server, vLLM integration. The GPU has no room for the model while `serve.py` runs; I did not stop it.
- Cooperative persistent kernel under a vLLM server (residency while other kernels/streams run) and with the 4-bit KV / host-pinned tables.

## Proposed architecture (if the rework goes ahead)

1. **Ops as modules, not a megakernel.** Each op = (torch reference twin, CuTe DSL kernel specialised by `constexpr` shape/dtype, lazily compiled and cached by key). Ops needed for Qwen3.5/3.8: `w4_gemv` (done, bit-exact), `rmsnorm(+residual)`, `silu_mul` epilogue (in `mega.py`), `gdn_decode`, `attn_decode_int4kv`, `lm_head`, sampling. A per-model *slot plan* lists them in order; env var swaps a slot back to its twin for bisecting (SGLang pattern). Execution = CUDA graph of those kernels, as today.
2. **Fuse at boundaries where the profile says so** (norm into GEMV x-load; silu*up into down GEMV; residual into the epilogue), not by removing launches.
3. **vLLM compatibility** = a vLLM plugin, not an engine-core replacement: (a) W4A16 compressed-tensors linear method calling the CuTe GEMV for <=16 rows and vLLM's Marlin otherwise; (b) keep vLLM's Qwen3.5 model + GDN backend, swap slots as they are ported; (c) an optional decode executor gated by a flag, covering pure-decode FULL-graph steps only, falling back to the stock forward (vLLM `decode_backend`, SGLang `--enable-nvidia-megakernel` shape). Cost: today's MTP/DFlash-in-graph cycle, the rotated 4-bit KV cache and the host prefix store would have to be re-expressed as vLLM spec-decode proposers and an attention backend. That is the expensive part and where the project's differentiation lives; the spike says nothing about its feasibility.
4. **Keep the persistent/table-driven path as a research branch** only if a profile of the real DeltaNet/attention stages shows idle DRAM windows worth > ~5 %.

## Next steps, in order

1. Port `gdn_decode` and the int4-KV attention split/reduce to CuTe with bit-exact checks against `deltanet.cu` / `attn.cu` (same harness: random state, compare, bench). This is where the 3.4 ms is.
2. Fused norm-into-GEMV; re-measure the chain with the real norm cost.
3. Stop the server for one session and take the real nsys timeline of the engine step to confirm where the non-GEMV time sits before deciding on (4).
4. Kernel tuning (GEMV gap to CUDA is 0-2 %; GEMVs themselves sit at ~838 of 936 GB/s) fits an `autonomous-campaign`: metric = chain GB/s, check = bit-exact vs the CUDA kernel, editable = `cute_w4.py`/`ptx.py`. Not set up yet; needs a held-out shape set and the GPU to itself.

## Files

`ptx.py` inline-PTX helpers · `cute_w4.py` GEMV · `mega.py` stage executor (graph | persistent, prefetch, timing) · `w4_common.py` harness + ctypes baseline · `s1_smoke.py` · `s2_*.py` GEMV bench/sweep/debug · `s3_run.py` chain · `s3_nop.py` boundary cost · `s5_overhead.py` host cost. Run from the repo root with `.venv/bin/python proto_cute/<script>`; set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` while the server holds the GPU.
