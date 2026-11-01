### Q3a measured per-shape best-exact-kernel ms (L2-cold graph) at the T that matter, and layer counts [measured]

| shape | count/step | T=1 ms | kernel | T=2 | T=3 | T=4 | T=4/T=1 |
|---|---|---|---|---|---|---|---|
| gateup | 64.00 | 0.1106 | mma/ones w8k4u2 | 0.1128 (mma/ones) | 0.1126 (mma/sub) | 0.1130 (mma/sub) | 1.022 |
| down | 64.00 | 0.0566 | mma/sub w8k4u2 | 0.0567 (mma/ones) | 0.0580 (mma/ones) | 0.0587 (mma/ones) | 1.036 |
| qkvz | 62.00 | 0.0538 | mma/ones w8k4u2 | 0.0557 (mma/ones) | 0.0555 (mma/sub) | 0.0563 (mma/ones) | 1.046 |
| oproj | 64.00 | 0.0219 | mma/ones w8k8u2 | 0.0220 (mma/ones) | 0.0221 (mma/ones) | 0.0220 (mma/sub) | 1.006 |
| lmhead | 1.00 | 2.8784 | cc/bf16 w8r4u4 | 3.0280 (mma/bf16) | 3.0321 (mma/bf16) | 3.0418 (mma/bf16) | 1.057 |
| **sum x count** | | **18.320** | | 18.742 | 18.803 | 18.930 | 1.033 |

This bench's own GEMV sum at T=1 is 18.32 ms (independent launches in a graph, no dependency/prologue); E1's measured dependent chain for the same GEMVs plus norms/prologues is 18.958 ms (V1a), 18.643 (V4). The model therefore anchors T=1 to E1's chain and adds the measured T-growth (gemv_sum(T) - gemv_sum(1)).

### Q3b modelled traffic not in the GEMV benchmarks [modelled]

| T rows | DeltaNet state ms ((1+T) x 151 MB at 800 GB/s) | KV 2K FP8 / BF16 ms | KV 8K FP8 / BF16 ms |
|---|---|---|---|
| 1 | 0.378 | 0.084 / 0.168 | 0.336 / 0.671 |
| 2 | 0.567 | 0.084 / 0.168 | 0.336 / 0.671 |
| 3 | 0.756 | 0.084 / 0.168 | 0.336 / 0.671 |
| 4 | 0.945 | 0.084 / 0.168 | 0.336 / 0.671 |

Other non-GEMV work F = 1.0 ms per step (guess, unmeasured), draft-step fixed overhead 0.1 ms. Draft step (a) vLLM-like = BF16 MTP layer 0.85 GB + full lm_head 2.54 GB at 883 GB/s (measured T=1 lm_head) = **3.94 ms**; (b) int4 MTP layer 0.22 GB at 813 GB/s (byte-weighted measured W4 T=1) + 32k-token BF16 head 0.335 GB at 883 GB/s = **0.75 ms**.

### Q3c modelled step times (ms), E1-anchored, KV FP8 [modelled from measured parts]

| ctx | plain T=1 | verify T=2 | verify T=3 | verify T=4 | + draft (a) k=1 / 2 / 3 | + draft (b) k=1 / 2 / 3 |
|---|---|---|---|---|---|---|
| 2K | 20.42 | 21.03 | 21.28 | 21.60 | 24.97 / 29.16 / 33.42 | 21.78 / 22.78 / 23.85 |
| 8K | 20.67 | 21.28 | 21.53 | 21.85 | 25.22 / 29.41 / 33.67 | 22.03 / 23.03 / 24.10 |

Model plain step at 2K = 20.42 ms (49.0 tok/s) vs stock vLLM measured 21.4 ms (46.8 tok/s).

### Q3d predicted tok/s and ratio to stock vLLM (measured E0) per text type [modelled; acceptance = E0 vLLM per-position, full-vocab, BF16 MTP layer: UNMEASURED for restricted head / int4 MTP layer]

Accepted length E_k = 1 + sum_{i<=k} a_i (a_i = E0 per-position marginal acceptance).  Draft options: (a) as vLLM today, (b) int4 MTP layer + 32k head.  ratio = model tok/s / stock vLLM tok/s at the same k (plain: 46.8).

| text | k | E_k | vLLM measured tok/s | model (a) tok/s | ratio (a) | model (b) tok/s | ratio (b) |
|---|---|---|---|---|---|---|---|
| code | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| code | 1 | 1.94 | 70.0 | 77.7 | 1.11x | 89.1 | 1.27x |
| code | 2 | 2.84 | 86.8 | 97.4 | 1.12x | 124.7 | 1.44x |
| code | 3 | 3.59 | 96.4 | 107.4 | 1.11x | 150.5 | 1.56x |
| **code best k** | a: k=3, b: k=3 | | best stock 96.4 | **107.4** | **1.11x** | **150.5** | **1.56x** |
| prose | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| prose | 1 | 1.69 | 62.6 | 67.7 | 1.08x | 77.6 | 1.24x |
| prose | 2 | 2.13 | 62.6 | 73.0 | 1.17x | 93.5 | 1.49x |
| prose | 3 | 2.34 | 63.0 | 70.0 | 1.11x | 98.1 | 1.56x |
| **prose best k** | a: k=2, b: k=3 | | best stock 63.0 | **73.0** | **1.16x** | **98.1** | **1.56x** |
| math | 0 (plain) | 1.00 | 46.8 | 49.0 | 1.05x | 49.0 | 1.05x |
| math | 1 | 1.96 | 70.6 | 78.5 | 1.11x | 90.0 | 1.27x |
| math | 2 | 2.79 | 86.2 | 95.7 | 1.11x | 122.5 | 1.42x |
| math | 3 | 3.49 | 93.5 | 104.4 | 1.12x | 146.3 | 1.57x |
| **math best k** | a: k=3, b: k=3 | | best stock 93.5 | **104.4** | **1.12x** | **146.3** | **1.57x** |
| 8K | 0 (plain) | 1.00 | 45.9 | 48.4 | 1.05x | 48.4 | 1.05x |
| 8K | 1 | 1.78 | 64.4 | 70.6 | 1.10x | 80.8 | 1.25x |
| 8K | 2 | 2.37 | 73.8 | 80.6 | 1.09x | 102.9 | 1.39x |
| 8K | 3 | 2.87 | 75.5 | 85.2 | 1.13x | 119.1 | 1.58x |
| **8K best k** | a: k=3, b: k=3 | | best stock 75.5 | **85.2** | **1.13x** | **119.1** | **1.58x** |

### Q3e sensitivity of the k=3 code ratio (b) to the modelled inputs [modelled]

| variation | plain tok/s (2K) | k=3 code (b) tok/s | ratio vs vLLM 96.4 |
|---|---|---|---|
| default (F=1.0 ms, state 800 GB/s) | 49.0 | 150.5 | 1.56x |
| F=0.5 ms | 50.2 | 153.8 | 1.59x |
| F=2.0 ms | 46.7 | 144.5 | 1.50x |
| F=3.0 ms | 44.6 | 138.9 | 1.44x |
| base = this bench's own GEMV sum (no dependency gaps) | 50.6 | 154.7 | 1.60x |
| verify GEMV flat in T (upper bound) | 49.0 | 154.5 | 1.60x |
| KV BF16 | 48.8 | 150.0 | 1.56x |

### Q4 best exact W4 g128 GEMV at T=1 vs E1's GEMV [measured: Q1 short graph windows, Q2 3 s sustained windows; E1 gemv_sweep]

| shape | E1 GB/s | Q1 best T=1 | kernel/cfg | Q1 vs E1 | Q2 best exact | kernel/cfg | Q2 vs E1 | Q2 read-only same shape |
|---|---|---|---|---|---|---|---|---|
| gateup | 846 | 831 | mma/ones w8k4u2 | 0.982 | 833 | mma/ones w8k4u2 | 0.984 | 873 |
| down | 806 | 812 | mma/sub w8k4u2 | 1.007 | 818 | mma/ones w8k4u2 | 1.014 | 858 |
| qkvz | 808 | 803 | mma/ones w8k4u2 | 0.994 | 796 | mma/ones w8k8u2 | 0.985 | 858 |
| oproj | 748 | 741 | mma/ones w8k8u2 | 0.991 | 741 | mma/ones w8k8u2 | 0.991 | 818 |
| lmhead | 888 | 883 | cc/bf16 w8r4u4 | 0.995 | 883 | cc/bf16 w8r4u4 | 0.995 | 891 |
