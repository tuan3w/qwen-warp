# K5 W4 GEMV tuning log (2026-10-31)

Goal: raise the average W4 GEMV bandwidth of the real 64-layer step (nsys: 756 GB/s average, 79% of the 20.9 ms step) towards the ~880-900 GB/s a
pure streaming read reaches.  Every attempt below is a measurement, kept or not.

## Method
* `tests/gemv_lab.cu` (+ `lab_v2.inc`, `lab_variants.inc`): standalone harness, one CUDA graph of `nb` launches over distinct random weight buffers
  (>= 400 MB, L2-cold), T=1 unless stated.  Kernel variants are compile-time flags (`k_w4x` = the committed `k_w4` with experiment switches).
  Timing: first runs used 20 ms windows and were dominated by DVFS (boost vs the 350 W cap): **never trust < 0.5 s windows**.  Final numbers: 200 ms warm-up +
  400 ms measurement window per variant, 3 interleaved rounds, median (clocks settle at ~1600 MHz / 313 W = what the engine sees: lab gate|up 115.7 us vs engine nsys 114.5 us).
* `ncu --set full` on the engine's own k_w4 instances (locked 1.33 GHz) and on lab kernels; nsys per-kernel table of the real step (`--capture-range=cudaProfilerApi`).
* Read-only reference in the same harness (`readonly-ref`): every warp streams a contiguous slice, 16 B loads, depth 4: **873 GB/s of weight bytes** (gate|up size), 860 (down).

## Findings (what limits the committed kernel)
| finding | evidence |
|---|---|
| Not DRAM locality / layout: a tile-contiguous weight layout (each warp unit = 1 KB contiguous, a CTA streams a contiguous 8 KB per step) is within +-1% of row-major | gate|up 110.0 -> 111.0 us, down 56.7 -> 56.5, qkv|z 54.7 -> 54.8 |
| Not footprint / TLB: 200 distinct down buffers (8.9 GB) = 56.98 us, same as 10 buffers (56.73) | lab7 |
| Not clocks alone: engine k_w4 down is 22% slower than the identical-instruction lab kernel under ncu with locked clocks (73.1 vs 59.9 us, same bytes, same occupancy) | code-generation/schedule dependent; the real `k_w4<8,4,2,2>` down is 67-70 us in lab *and* engine, my restructured copy 56.9 |
| Instruction count matters only mildly: removing dequant+mma altogether (hack) gains 3.5% (gate|up), 4% (down), 7% (o_proj); the loads-only floor (W+S+X loads, same access pattern) is 106.1 us for gate|up vs 115.7 committed | `AB1` variants |
| Scale loads cost exactly their bytes (3.1% of traffic); activation loads cost ~2% | noS / noX hacks |
| The k_w4 main loop is 218-230 SASS instr per 1 KB (ncu: 309 incl. tile epilogue), 64-bit row-pointer arithmetic is recomputed per group | cuobjdump |

## Attempts
| # | change | shape: before -> after (us, T=1, sustained clocks) | verdict |
|---|---|---|---|
| 1 | L2 prefetch (`prefetch.global.L2`) of the next kernel's first-wave weights (2.2 / 1.1 / 0.5 MB) issued when a warp's load stream ends | gate|up 110.98 -> 117.4; down 56.5 -> 57.0..58.3; qkv|z 54.8 -> 58.3; o 22.4 -> 22.7 | **rejected**: always slower (prefetch traffic is not hidden in the tail) |
| 2 | evict-first L2 policy (`createpolicy` + `L2::cache_hint`) on weight loads, with / without prefetch | +-0.3% without prefetch, worse with | rejected |
| 3 | tile-contiguous weight layout (repack) | +-1% | rejected as a goal in itself (needed by 7) |
| 4 | wave-interleaved layout (hack: unit order = time order of the whole grid) | gate|up 118 vs 111 (tile-contiguous) | rejected |
| 5 | packed u32 scale per (tile, group, row) instead of 2 x u16 | gate|up 111.2 -> 116.9 on the old kernel (slower: loses the L1 hits of the row-major u16 sectors) | rejected on the old kernel, fine inside 7 |
| 6 | config sweep: w16 (68 CTAs), KS=1/2, U=3/4 on the old kernel | gate|up w8k8u2 110.0, w8k8u4 112.0, w8k8u3 112.6, w16k8u2 114.1, w8k2u2 115.0, w8k1u3 177.8; down: u3 58.3, u4 59.0, w16 57.5, k2 69.5, k1 123 | rejected (register pressure / more tile-end syncs) |
| 7 | **k_w4 rewritten (kept)**: pointer-increment addressing (no per-group 64-bit row math), activation fragments loaded just in time (no double-buffer registers), unpredicated B loads (lanes g>=T read row T-1: column n of the mma result depends only on B column n, never stored), natural weight layout, same epilogue | lab (sustained clocks, T=1): down real k_w4 67.7 -> 56.8 us (the lab copy of the old kernel: 57.6); gate|up 112.2 (w8k4u2), qkv|z 55.7, q|k|v 49.3 (w8k8u2), o 22.1; outputs rel-L2 0.0 vs old kernel (bit-identical for T=1 and T=4) | **kept** |
| 8 | default config re-sweep inside the full graph (`SMK_CFG`) with the new kernel: all w8k4u2 except q\|k\|v (N=14336) w8k8u2 | step 20.250 (old defaults) / 20.214 / 20.260 / 20.227 / 20.279 ms: all within 0.3% | kept (-0.04 ms) |
| 9 | nibble order permuted inside each word (natural addressing, no PRMT on the activations) | lab: gate|up 112.2 -> 111.2, down 56.8 -> 56.0, qkv|z 55.7 -> 55.3 (~1%) | not pursued: changes K4/K6/pack consumers for ~0.1 ms |
| 10 | full repack (tile-contiguous 1 KB warp units, u32 scale per (tile, group, row), nibble pairs): lab T=1 vs #7: gate|up -3.1%, qkv|z -4.2%, q|k|v -3.6%, down -2.3%, o 0 | timing-only build (kernel reads the natural buffers through the repacked addressing, outputs garbage): whole step T=1 20.192 -> 19.983 ms (-0.21), T=4 21.56 -> 21.11 (-0.45) | **deferred**: below the 0.4 ms/step bar at T=1; would change the weight layout for K4 (MTP packer), K6 (prefill dequant), pack.py and test_ops |

## Result (engine, `tests/bench_e2e.py`, ctx 64, bf16 KV, 5 interleaved A/B repeats, lease held; baseline = committed HEAD built into build_base)
| step | before (ms) | after (ms) |
|---|---|---|
| T=1 plain decode | 20.981 (20.937-20.988) = 47.7 tok/s | **20.195** (20.186-20.223) = 49.5 tok/s |
| T=2 | 21.566 | 20.721 |
| T=3 | 22.053 | 21.077 |
| T=4 verify | 22.582 (22.564-22.675) | **21.545** (21.536-21.624) |

nsys of the real step (20 steps, `--cuda-graph-trace=node`), W4 GEMV kernels (avg us per call, GB/s over weight+scale bytes):
| GEMV | calls/step | before | after |
|---|---|---|---|
| gate\|up (91.9 MB) | 64 | 114.66 us, 801 GB/s | 115.35 us, 797 GB/s |
| down (46.0 MB) | 64 | 69.82 us, 659 GB/s | **56.65 us, 812 GB/s** |
| qkv\|z (43.3 MB) and q\|k\|v (37.8 MB), mixed | 64 | 52.01 us | 52.89 us |
| o_proj (16.3 MB) | 64 | 23.54 us, 691 GB/s | 23.40 us, 695 GB/s |
| all W4 GEMVs | 256 | 16.64 ms, 757 GB/s average | **15.89 ms, 793 GB/s average** |
The whole gain is `down` (-0.85 ms) plus the K-order-independent bookkeeping; in the graph gate|up / qkv|z / o_proj are unchanged (the lab's "old kernel" copy for those shapes was slower than the real k_w4, so lab gains there did not transfer).

## What is left
* Loads-only floor (same access pattern, no dequant/mma) is ~4% under the current kernel for gate|up, 7% for o_proj; a pure streaming read is another ~8% below that (873 vs ~800 GB/s of weight bytes).
* Remaining structural costs per kernel: ~1 us launch gap + first-DRAM-latency ramp + tail (o_proj, 16 MB, is at 695 GB/s: ~2.5 us fixed cost); L2 prefetch of the next layer does not hide it (#1).
* Repack (#10) is the only measured remaining lever (0.2-0.45 ms).
