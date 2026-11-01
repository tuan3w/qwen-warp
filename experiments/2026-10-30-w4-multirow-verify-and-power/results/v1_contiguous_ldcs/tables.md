### Correctness (fp32 GPU reference, warp-per-row; rel L2 of the bf16 output; max over all configs/T; `vs bf16(ref)` isolates the kernel error from output rounding)

| shape | kernel | rel L2 vs fp32 ref | bf16 rounding floor | rel L2 vs bf16(ref) | n |
|---|---|---|---|---|---|
| gateup | mma/sub | 1.66e-03 | 1.66e-03 | 3.20e-05 | 30 |
| gateup | mma/ones | 1.66e-03 | 1.66e-03 | 8.27e-05 | 30 |
| gateup | cc/e1 | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| gateup | cc/naive | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| down | mma/sub | 1.69e-03 | 1.69e-03 | 1.59e-05 | 30 |
| down | mma/ones | 1.69e-03 | 1.69e-03 | 6.60e-05 | 30 |
| down | cc/e1 | 1.86e-07 | 0.00e+00 | 0.00e+00 | 4 |
| down | cc/naive | 1.86e-07 | 0.00e+00 | 0.00e+00 | 4 |
| qkvz | mma/sub | 1.67e-03 | 1.67e-03 | 4.27e-05 | 30 |
| qkvz | mma/ones | 1.67e-03 | 1.67e-03 | 1.15e-04 | 30 |
| qkvz | cc/e1 | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| qkvz | cc/naive | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| oproj | mma/sub | 1.66e-03 | 1.66e-03 | 2.85e-05 | 30 |
| oproj | mma/ones | 1.66e-03 | 1.66e-03 | 7.18e-05 | 30 |
| oproj | cc/e1 | 1.17e-07 | 0.00e+00 | 0.00e+00 | 4 |
| oproj | cc/naive | 1.17e-07 | 0.00e+00 | 0.00e+00 | 4 |
| lmhead | mma/bf16 | 1.66e-03 | 1.66e-03 | 1.67e-04 | 15 |
| lmhead | cc/bf16 | 7.99e-08 | 0.00e+00 | 0.00e+00 | 3 |

### Q1 gateup 5120->34816: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/sub x | mma/ones x | mma/abl x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1133 (w8k8u2) | 0.1130 (w8k8u2) | 0.1105 (w8k8u2) | 1.000 | 1.000 | 1.000 | 813 | 1.000 |
| 2 | 0.1163 (w16k8u2) | 0.1149 (w8k8u2) | 0.1108 (w8k8u2) | 1.027 | 1.016 | 1.002 | 800 | 1.016 |
| 3 | 0.1143 (w8k8u2) | 0.1141 (w8k8u2) | 0.1108 (w8k8u2) | 1.009 | 1.009 | 1.002 | 806 | 1.009 |
| 4 | 0.1167 (w8k8u2) | 0.1161 (w8k8u2) | 0.1112 (w8k8u2) | 1.031 | 1.027 | 1.006 | 792 | 1.027 |
| 5 | 0.1182 (w8k8u2) | 0.1173 (w8k8u2) | 0.1120 (w8k8u2) | 1.043 | 1.038 | 1.013 | 783 | 1.038 |
| 6 | 0.1245 (w8k8u2) | 0.1229 (w8k8u2) | 0.1134 (w8k8u2) | 1.100 | 1.087 | 1.026 | 748 | 1.087 |
| 8 | 0.1308 (w8k8u2) | 0.1310 (w8k8u2) | 0.1221 (w8k8u2) | 1.155 | 1.159 | 1.104 | 702 | 1.157 |
| 12 | 0.1548 (w4k4u3) | 0.1609 (w4k4u3) | 0.1618 (w4k4u3) | 1.367 | 1.423 | 1.464 | 594 | 1.369 |
| 16 | 0.1710 (w4k4u3) | 0.1737 (w8k2u3) | 0.1845 (w4k4u3) | 1.510 | 1.536 | 1.669 | 537 | 1.513 |

T=1 CUDA-core / read-only references: cc/e1 0.1155 ms (w8r4u2, 796 GB/s); cc/naive 0.1288 ms (w8r4u2, 714 GB/s); cc/ablfma 0.1083 ms (w8r4u2, 848 GB/s); cc/ablload 0.1063 ms (w8r2u2, 865 GB/s); read-only 0.1057 ms (870 GB/s)

### Q1 down 17408->5120: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/sub x | mma/ones x | mma/abl x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0744 (w8k8u2) | 0.0764 (w16k8u2) | 0.0751 (w8k8u2) | 1.000 | 1.000 | 1.000 | 759 | 1.000 |
| 2 | 0.0741 (w8k8u2) | 0.0758 (w16k8u2) | 0.0754 (w8k8u2) | 0.996 | 0.992 | 1.004 | 620 | 1.225 |
| 3 | 0.0741 (w8k8u2) | 0.0760 (w16k8u2) | 0.0756 (w16k8u2) | 0.996 | 0.995 | 1.007 | 620 | 1.225 |
| 4 | 0.0740 (w8k8u2) | 0.0759 (w16k8u2) | 0.0760 (w16k8u2) | 0.995 | 0.993 | 1.012 | 621 | 1.223 |
| 5 | 0.0750 (w8k8u2) | 0.0765 (w16k8u2) | 0.0760 (w16k8u2) | 1.007 | 1.001 | 1.012 | 613 | 1.239 |
| 6 | 0.0755 (w8k8u2) | 0.0766 (w16k8u2) | 0.0761 (w16k8u2) | 1.014 | 1.003 | 1.014 | 609 | 1.247 |
| 8 | 0.0766 (w8k8u2) | 0.0773 (w8k8u2) | 0.0766 (w16k8u2) | 1.029 | 1.012 | 1.021 | 600 | 1.266 |
| 12 | 0.0804 (w16k8u2) | 0.0815 (w16k8u2) | 0.0805 (w16k8u2) | 1.080 | 1.067 | 1.072 | 572 | 1.328 |
| 16 | 0.0896 (w16k8u2) | 0.0926 (w16k8u2) | 0.0914 (w16k8u2) | 1.204 | 1.212 | 1.218 | 513 | 1.480 |

T=1 CUDA-core / read-only references: cc/e1 0.0605 ms (w8r2u2, 759 GB/s); cc/naive 0.0625 ms (w8r4u2, 736 GB/s); cc/ablfma 0.0555 ms (w8r2u2, 828 GB/s); cc/ablload 0.0543 ms (w8r2u2, 847 GB/s); read-only 0.0537 ms (856 GB/s)

### Q1 qkvz 5120->16384: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/sub x | mma/ones x | mma/abl x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0547 (w8k8u2) | 0.0546 (w8k8u2) | 0.0523 (w8k8u2) | 1.000 | 1.000 | 1.000 | 793 | 1.000 |
| 2 | 0.0553 (w8k8u2) | 0.0555 (w8k8u2) | 0.0525 (w8k8u2) | 1.011 | 1.017 | 1.004 | 782 | 1.013 |
| 3 | 0.0558 (w8k8u2) | 0.0556 (w8k8u2) | 0.0528 (w8k8u2) | 1.020 | 1.019 | 1.010 | 778 | 1.019 |
| 4 | 0.0569 (w8k8u2) | 0.0567 (w8k8u2) | 0.0532 (w8k8u2) | 1.041 | 1.040 | 1.016 | 762 | 1.040 |
| 5 | 0.0576 (w8k8u2) | 0.0576 (w8k8u2) | 0.0544 (w8k8u2) | 1.054 | 1.056 | 1.039 | 750 | 1.056 |
| 6 | 0.0598 (w8k8u2) | 0.0589 (w8k8u2) | 0.0550 (w8k8u2) | 1.094 | 1.079 | 1.051 | 735 | 1.079 |
| 8 | 0.0627 (w8k8u2) | 0.0636 (w8k8u2) | 0.0592 (w8k8u2) | 1.146 | 1.165 | 1.131 | 690 | 1.148 |
| 12 | 0.0737 (w4k4u3) | 0.0749 (w16k8u2) | 0.0719 (w16k8u2) | 1.348 | 1.373 | 1.374 | 587 | 1.351 |
| 16 | 0.0821 (w4k4u3) | 0.0857 (w4k4u3) | 0.0803 (w16k8u2) | 1.502 | 1.571 | 1.534 | 527 | 1.505 |

T=1 CUDA-core / read-only references: cc/e1 0.0577 ms (w8r4u2, 749 GB/s); cc/naive 0.0654 ms (w8r4u2, 661 GB/s); cc/ablfma 0.0525 ms (w8r4u2, 823 GB/s); cc/ablload 0.0507 ms (w8r2u2, 853 GB/s); read-only 0.0506 ms (855 GB/s)

### Q1 oproj 6144->5120: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/sub x | mma/ones x | mma/abl x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0218 (w8k8u2) | 0.0217 (w8k8u2) | 0.0214 (w8k8u2) | 1.000 | 1.000 | 1.000 | 746 | 1.000 |
| 2 | 0.0220 (w8k8u2) | 0.0219 (w8k8u2) | 0.0216 (w8k8u2) | 1.006 | 1.006 | 1.006 | 741 | 1.006 |
| 3 | 0.0220 (w8k8u2) | 0.0220 (w8k8u2) | 0.0218 (w8k8u2) | 1.009 | 1.012 | 1.015 | 737 | 1.012 |
| 4 | 0.0221 (w8k8u2) | 0.0221 (w8k8u2) | 0.0220 (w8k8u2) | 1.015 | 1.018 | 1.024 | 733 | 1.018 |
| 5 | 0.0223 (w8k8u2) | 0.0225 (w8k8u2) | 0.0222 (w8k8u2) | 1.024 | 1.033 | 1.036 | 726 | 1.027 |
| 6 | 0.0227 (w8k8u2) | 0.0228 (w8k8u2) | 0.0225 (w8k8u2) | 1.038 | 1.051 | 1.048 | 716 | 1.042 |
| 8 | 0.0234 (w8k8u2) | 0.0238 (w8k8u2) | 0.0233 (w8k8u2) | 1.071 | 1.095 | 1.087 | 694 | 1.074 |
| 12 | 0.0273 (w8k4u2) | 0.0283 (w4k4u3) | 0.0278 (w8k4u2) | 1.252 | 1.301 | 1.296 | 594 | 1.256 |
| 16 | 0.0300 (w4k4u3) | 0.0315 (w4k4u3) | 0.0303 (w8k4u2) | 1.375 | 1.448 | 1.412 | 541 | 1.379 |

T=1 CUDA-core / read-only references: cc/e1 0.0235 ms (w8r2u2, 691 GB/s); cc/naive 0.0251 ms (w8r2u2, 645 GB/s); cc/ablfma 0.0211 ms (w8r2u2, 768 GB/s); cc/ablload 0.0202 ms (w8r2u2, 805 GB/s); read-only 0.0199 ms (815 GB/s)

### Q1 lmhead 5120->248320 (bf16): ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/bf16 ms (cfg) | mma/bf16 x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|
| 1 | 5.3596 (w8k2u4) | 1.000 | 884 | 1.000 |
| 2 | 5.3683 (w8k2u4) | 1.002 | 474 | 1.867 |
| 3 | 5.3723 (w8k2u4) | 1.002 | 473 | 1.868 |
| 4 | 5.3769 (w8k2u4) | 1.003 | 473 | 1.870 |
| 5 | 5.3796 (w8k2u4) | 1.004 | 473 | 1.871 |
| 6 | 5.3842 (w8k2u4) | 1.005 | 472 | 1.872 |
| 8 | 5.3847 (w8k2u4) | 1.005 | 472 | 1.872 |
| 12 | 5.4087 (w8k2u4) | 1.009 | 470 | 1.881 |
| 16 | 5.4047 (w8k2u4) | 1.008 | 470 | 1.879 |

T=1 CUDA-core / read-only references: cc/bf16 2.8759 ms (w8r4u4, 884 GB/s); read-only 2.8529 ms (891 GB/s)

### T* = largest T in {1,2,3,4,5,6,8,12,16} whose time stays within x of T=1 (best exact kernel at each T; base = best exact kernel at T=1, cc or mma)

| shape | T*(1.05x) | T*(1.10x) | T*(1.15x) | T*(1.25x) | ratio@T=2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|---|---|
| gateup | 5 | 6 | 6 | 8 | 1.016 | 1.027 | 1.157 | 1.513 |
| down | 1 | 1 | 1 | 6 | 1.225 | 1.223 | 1.266 | 1.480 |
| qkvz | 4 | 6 | 8 | 8 | 1.013 | 1.040 | 1.148 | 1.505 |
| oproj | 6 | 8 | 8 | 8 | 1.006 | 1.018 | 1.074 | 1.379 |
| lmhead | 1 | 1 | 1 | 1 | 1.867 | 1.870 | 1.872 | 1.879 |

### Whole-step GEMV time (sum of measured per-shape kernels x layer counts; 64 gate|up, 64 down, 48 qkv|z + 16 attn qkv scaled 14336/16384, 64 o_proj, 1 lm_head); excludes non-GEMV work

| T | GEMV sum ms | x vs T=1 |
|---|---|---|
| 1 | 18.760 | 1.000 |
| 2 | 22.296 | 1.188 |
| 3 | 22.272 | 1.187 |
| 4 | 22.481 | 1.198 |
| 5 | 22.691 | 1.210 |
| 6 | 23.179 | 1.236 |
| 8 | 24.041 | 1.282 |
| 12 | 26.778 | 1.427 |
| 16 | 29.095 | 1.551 |

### MTP break-even model (draft step = MTP layer 0.85 GB BF16 + head, at the measured BF16 GB/s 884: restricted 32k head 0.336 GB -> 1.44 ms/step, full lm_head 2.54 GB -> 3.94 ms/step; chain acceptance p^i, tokens/iter = sum_(i<=k) p^i; verify step = measured GEMV(T=k+1) + F non-GEMV + delta*k DeltaNet snapshot cost)

**F=1.5 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 23.97 | 1.183 | 1.28x | 1.36x | 1.44x | 1.52x |
| 2 | 3 | 24.11 | 1.190 | 1.47x | 1.64x | 1.83x | 2.03x |
| 3 | 4 | 24.49 | 1.209 | 1.53x | 1.78x | 2.08x | 2.42x |
| 4 | 5 | 24.87 | 1.228 | 1.52x | 1.83x | 2.22x | 2.71x |
| 5 | 6 | 25.53 | 1.260 | 1.48x | 1.82x | 2.28x | 2.90x |

**F=1.5 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 23.97 | 1.183 | 1.16x | 1.23x | 1.31x | 1.38x |
| 2 | 3 | 24.11 | 1.190 | 1.24x | 1.39x | 1.55x | 1.72x |
| 3 | 4 | 24.49 | 1.209 | 1.21x | 1.41x | 1.65x | 1.92x |
| 4 | 5 | 24.87 | 1.228 | 1.15x | 1.38x | 1.68x | 2.04x |
| 5 | 6 | 25.53 | 1.260 | 1.07x | 1.32x | 1.65x | 2.10x |

**F=1.5 ms, delta=0.0 ms/extra token, draft = restricted 32k head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 23.80 | 1.175 | 1.28x | 1.36x | 1.45x | 1.53x |
| 2 | 3 | 23.77 | 1.173 | 1.49x | 1.66x | 1.85x | 2.06x |
| 3 | 4 | 23.98 | 1.184 | 1.56x | 1.81x | 2.11x | 2.46x |
| 4 | 5 | 24.19 | 1.194 | 1.56x | 1.88x | 2.27x | 2.77x |
| 5 | 6 | 24.68 | 1.218 | 1.51x | 1.87x | 2.34x | 2.98x |

**F=1.5 ms, delta=0.0 ms/extra token, draft = full lm_head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 23.80 | 1.175 | 1.17x | 1.24x | 1.31x | 1.39x |
| 2 | 3 | 23.77 | 1.173 | 1.25x | 1.40x | 1.56x | 1.73x |
| 3 | 4 | 23.98 | 1.184 | 1.23x | 1.43x | 1.67x | 1.95x |
| 4 | 5 | 24.19 | 1.194 | 1.17x | 1.41x | 1.71x | 2.08x |
| 5 | 6 | 24.68 | 1.218 | 1.09x | 1.34x | 1.68x | 2.14x |

**F=1.5 ms, delta=0.34 ms/extra token, draft = restricted 32k head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 24.14 | 1.191 | 1.27x | 1.35x | 1.43x | 1.51x |
| 2 | 3 | 24.45 | 1.207 | 1.45x | 1.62x | 1.81x | 2.01x |
| 3 | 4 | 25.00 | 1.234 | 1.50x | 1.75x | 2.04x | 2.38x |
| 4 | 5 | 25.55 | 1.261 | 1.49x | 1.79x | 2.17x | 2.65x |
| 5 | 6 | 26.38 | 1.302 | 1.44x | 1.77x | 2.23x | 2.83x |

**F=1.5 ms, delta=0.34 ms/extra token, draft = full lm_head** (plain step 20.26 ms = 49.4 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 24.14 | 1.191 | 1.15x | 1.23x | 1.30x | 1.37x |
| 2 | 3 | 24.45 | 1.207 | 1.23x | 1.37x | 1.53x | 1.70x |
| 3 | 4 | 25.00 | 1.234 | 1.20x | 1.39x | 1.62x | 1.89x |
| 4 | 5 | 25.55 | 1.261 | 1.13x | 1.36x | 1.65x | 2.01x |
| 5 | 6 | 26.38 | 1.302 | 1.05x | 1.29x | 1.62x | 2.06x |

**F=0.5 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 19.26 ms = 51.9 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 22.97 | 1.192 | 1.26x | 1.34x | 1.42x | 1.50x |
| 2 | 3 | 23.11 | 1.200 | 1.45x | 1.62x | 1.81x | 2.01x |
| 3 | 4 | 23.49 | 1.220 | 1.51x | 1.75x | 2.04x | 2.38x |
| 4 | 5 | 23.87 | 1.239 | 1.50x | 1.80x | 2.18x | 2.66x |
| 5 | 6 | 24.53 | 1.274 | 1.45x | 1.79x | 2.24x | 2.84x |

**F=0.5 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 19.26 ms = 51.9 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 22.97 | 1.192 | 1.15x | 1.22x | 1.29x | 1.36x |
| 2 | 3 | 23.11 | 1.200 | 1.22x | 1.36x | 1.52x | 1.68x |
| 3 | 4 | 23.49 | 1.220 | 1.19x | 1.38x | 1.61x | 1.88x |
| 4 | 5 | 23.87 | 1.239 | 1.12x | 1.35x | 1.63x | 1.99x |
| 5 | 6 | 24.53 | 1.274 | 1.04x | 1.28x | 1.61x | 2.04x |

**F=3.0 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 21.76 ms = 46.0 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 25.47 | 1.170 | 1.29x | 1.37x | 1.46x | 1.54x |
| 2 | 3 | 25.61 | 1.177 | 1.50x | 1.67x | 1.86x | 2.07x |
| 3 | 4 | 25.99 | 1.194 | 1.56x | 1.82x | 2.12x | 2.47x |
| 4 | 5 | 26.37 | 1.212 | 1.56x | 1.88x | 2.28x | 2.77x |
| 5 | 6 | 27.03 | 1.242 | 1.51x | 1.87x | 2.34x | 2.98x |

**F=3.0 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 21.76 ms = 46.0 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 25.47 | 1.170 | 1.18x | 1.26x | 1.33x | 1.41x |
| 2 | 3 | 25.61 | 1.177 | 1.27x | 1.42x | 1.59x | 1.76x |
| 3 | 4 | 25.99 | 1.194 | 1.25x | 1.46x | 1.70x | 1.98x |
| 4 | 5 | 26.37 | 1.212 | 1.19x | 1.43x | 1.74x | 2.12x |
| 5 | 6 | 27.03 | 1.242 | 1.11x | 1.37x | 1.72x | 2.18x |

Break-even p1 (speed-up = 1.0) at F=1.5, delta=0.17:

| draft | k=1 | k=2 | k=3 |
|---|---|---|---|
| restricted 32k | 0.25 | 0.26 | 0.30 |
| full lm_head | 0.38 | 0.41 | 0.47 |

Upper bound with a perfectly flat verify (verify(T)=plain step, delta=0, F=1.5, restricted draft), p1=0.7: k=1: 1.59x, k=2: 1.92x, k=3: 2.09x

### Q2 sustained windows (3 s after 2 s warm, interleaved 2 rounds; T=1; best config per kernel; nvidia-smi 10 Hz inside the window; * = ablated, results invalid)

| shape | kernel | cfg | GB/s (r0 / r1) | ms | SM MHz median | power W median | SW-cap frac | mem MHz | samples |
|---|---|---|---|---|---|---|---|---|---|
| gateup | read/read | g246x512 | 870 / 870 | 0.1057 | 1699 | 309 | 1.00 | 9501 | 60 |
| gateup | cc/ablload* | w8r2u2 | 865 / 865 | 0.1063 | 1706 | 307 | 1.00 | 9501 | 59 |
| gateup | cc/ablfma* | w8r4u2 | 848 / 848 | 0.1083 | 1620 | 316 | 1.00 | 9501 | 58 |
| gateup | cc/naive | w8r4u2 | 718 / 719 | 0.1280 | 1605 | 331 | 1.00 | 9501 | 60 |
| gateup | cc/e1 | w8r4u2 | 796 / 797 | 0.1155 | 1564 | 325 | 1.00 | 9501 | 60 |
| gateup | mma/abl* | w8k8u2 | 831 / 831 | 0.1106 | 1579 | 316 | 1.00 | 9501 | 59 |
| gateup | mma/sub | w8k8u2 | 813 / 813 | 0.1130 | 1478 | 320 | 1.00 | 9501 | 60 |
| gateup | mma/ones | w8k8u2 | 812 / 814 | 0.1132 | 1515 | 319 | 1.00 | 9501 | 60 |
| down | read/read | g246x512 | 856 / 856 | 0.0537 | 1740 | 306 | 1.00 | 9501 | 59 |
| down | cc/ablload* | w8r2u2 | 847 / 847 | 0.0543 | 1729 | 306 | 1.00 | 9501 | 60 |
| down | cc/ablfma* | w8r2u2 | 828 / 830 | 0.0555 | 1598 | 321 | 1.00 | 9501 | 59 |
| down | cc/naive | w8r4u2 | 738 / 737 | 0.0623 | 1590 | 329 | 1.00 | 9501 | 60 |
| down | cc/e1 | w8r2u2 | 768 / 766 | 0.0598 | 1519 | 328 | 1.00 | 9501 | 59 |
| down | mma/abl* | w8k8u2 | 612 / 612 | 0.0751 | 1725 | 317 | 1.00 | 9501 | 60 |
| down | mma/sub | w8k8u2 | 616 / 617 | 0.0746 | 1665 | 321 | 1.00 | 9501 | 60 |
| down | mma/ones | w16k8u2 | 602 / 602 | 0.0764 | 1702 | 319 | 1.00 | 9501 | 59 |
| qkvz | read/read | g246x512 | 855 / 855 | 0.0506 | 1732 | 307 | 1.00 | 9501 | 59 |
| qkvz | cc/ablload* | w8r2u2 | 852 / 852 | 0.0508 | 1706 | 306 | 1.00 | 9501 | 59 |
| qkvz | cc/ablfma* | w8r4u2 | 823 / 823 | 0.0525 | 1639 | 316 | 1.00 | 9501 | 60 |
| qkvz | cc/naive | w8r4u2 | 659 / 660 | 0.0656 | 1665 | 331 | 1.00 | 9501 | 60 |
| qkvz | cc/e1 | w8r4u2 | 749 / 749 | 0.0578 | 1612 | 325 | 1.00 | 9501 | 59 |
| qkvz | mma/abl* | w8k8u2 | 827 / 827 | 0.0523 | 1590 | 316 | 1.00 | 9501 | 60 |
| qkvz | mma/sub | w8k8u2 | 790 / 787 | 0.0548 | 1522 | 320 | 1.00 | 9501 | 60 |
| qkvz | mma/ones | w8k8u2 | 788 / 788 | 0.0549 | 1568 | 319 | 1.00 | 9501 | 60 |
| oproj | read/read | g246x512 | 815 / 815 | 0.0199 | 1770 | 305 | 1.00 | 9501 | 59 |
| oproj | cc/ablload* | w8r2u2 | 805 / 805 | 0.0202 | 1762 | 307 | 1.00 | 9501 | 58 |
| oproj | cc/ablfma* | w8r2u2 | 768 / 768 | 0.0211 | 1654 | 320 | 1.00 | 9501 | 60 |
| oproj | cc/naive | w8r2u2 | 645 / 644 | 0.0251 | 1635 | 333 | 1.00 | 9501 | 60 |
| oproj | cc/e1 | w8r2u2 | 693 / 693 | 0.0234 | 1620 | 330 | 1.00 | 9501 | 60 |
| oproj | mma/abl* | w8k8u2 | 757 / 756 | 0.0214 | 1642 | 315 | 1.00 | 9501 | 60 |
| oproj | mma/sub | w8k8u2 | 743 / 743 | 0.0218 | 1601 | 322 | 1.00 | 9501 | 60 |
| oproj | mma/ones | w8k8u2 | 748 / 746 | 0.0217 | 1616 | 318 | 1.00 | 9501 | 60 |
| lmhead | read/read | g246x512 | 891 / 891 | 2.8529 | 1740 | 310 | 1.00 | 9501 | 59 |
| lmhead | cc/bf16 | w8r4u4 | 884 / 884 | 2.8758 | 1759 | 310 | 1.00 | 9501 | 59 |
| lmhead | mma/bf16 | w8k2u4 | 474 / 475 | 5.3586 | 1935 | 315 | 1.00 | 9501 | 59 |

### Best exact (valid) GEMV per shape at T=1 under sustained load

| shape | best kernel | cfg | GB/s | SM MHz | power W | read-only GB/s same shape | % of read-only |
|---|---|---|---|---|---|---|---|
| gateup | mma/ones | w8k8u2 | 814 | 1515 | 319 | 870 | 93.6 |
| down | cc/e1 | w8r2u2 | 768 | 1519 | 328 | 856 | 89.7 |
| qkvz | mma/sub | w8k8u2 | 790 | 1522 | 320 | 855 | 92.3 |
| oproj | mma/ones | w8k8u2 | 748 | 1616 | 318 | 815 | 91.7 |
| lmhead | cc/bf16 | w8r4u4 | 884 | 1759 | 310 | 891 | 99.2 |

### Q1 clocks along the T sweep (nvidia-smi 10 Hz over each 0.6 s final window; best exact mma/sub or ones per T)

| shape | T | kernel | GB/s | SM MHz | power W | samples |
|---|---|---|---|---|---|---|
| gateup | 1 | mma/ones w8k8u2 | 813 | 1470 | 321 | 6 |
| gateup | 4 | mma/ones w8k8u2 | 792 | 1350 | 320 | 6 |
| gateup | 8 | mma/sub w8k8u2 | 702 | 1230 | 320 | 6 |
| gateup | 16 | mma/sub w4k4u3 | 537 | 1380 | 329 | 6 |
| down | 1 | cc/e1 w8r2u2 | 759 | 1440 | 319 | 6 |
| down | 4 | mma/sub w8k8u2 | 621 | 1598 | 319 | 6 |
| down | 8 | mma/sub w8k8u2 | 600 | 1425 | 320 | 6 |
| down | 16 | mma/sub w16k8u2 | 513 | 1470 | 323 | 6 |
| qkvz | 1 | mma/ones w8k8u2 | 793 | 1530 | 322 | 6 |
| qkvz | 4 | mma/ones w8k8u2 | 762 | 1530 | 321 | 6 |
| qkvz | 8 | mma/sub w8k8u2 | 690 | 1320 | 318 | 6 |
| qkvz | 16 | mma/sub w4k4u3 | 527 | 1530 | 323 | 6 |
| oproj | 1 | mma/ones w8k8u2 | 746 | 1575 | 321 | 6 |
| oproj | 4 | mma/ones w8k8u2 | 733 | 1515 | 322 | 6 |
| oproj | 8 | mma/sub w8k8u2 | 694 | 1402 | 318 | 6 |
| oproj | 16 | mma/sub w4k4u3 | 541 | 1492 | 328 | 6 |
| lmhead | 1 | cc/bf16 w8r4u4 | 884 | 1755 | 313 | 6 |
| lmhead | 4 | mma/bf16 w8k2u4 | 473 | 1935 | 315 | 6 |
| lmhead | 8 | mma/bf16 w8k2u4 | 472 | 1920 | 318 | 6 |
| lmhead | 16 | mma/bf16 w8k2u4 | 470 | 1890 | 320 | 6 |

### Ablation of T growth: ms ratio to T=1, mma modes (sub = exact bf16x2 subtract, ones = lop3 only + table correction, abl = no dequant)

| shape | mode | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=8 | T=12 | T=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| gateup | mma/sub | 1.000 | 1.027 | 1.009 | 1.031 | 1.043 | 1.100 | 1.155 | 1.367 | 1.510 |
| gateup | mma/ones | 1.000 | 1.016 | 1.009 | 1.027 | 1.038 | 1.087 | 1.159 | 1.423 | 1.536 |
| gateup | mma/abl | 1.000 | 1.002 | 1.002 | 1.006 | 1.013 | 1.026 | 1.104 | 1.464 | 1.669 |
| down | mma/sub | 1.000 | 0.996 | 0.996 | 0.995 | 1.007 | 1.014 | 1.029 | 1.080 | 1.204 |
| down | mma/ones | 1.000 | 0.992 | 0.995 | 0.993 | 1.001 | 1.003 | 1.012 | 1.067 | 1.212 |
| down | mma/abl | 1.000 | 1.004 | 1.007 | 1.012 | 1.012 | 1.014 | 1.021 | 1.072 | 1.218 |
| qkvz | mma/sub | 1.000 | 1.011 | 1.020 | 1.041 | 1.054 | 1.094 | 1.146 | 1.348 | 1.502 |
| qkvz | mma/ones | 1.000 | 1.017 | 1.019 | 1.040 | 1.056 | 1.079 | 1.165 | 1.373 | 1.571 |
| qkvz | mma/abl | 1.000 | 1.004 | 1.010 | 1.016 | 1.039 | 1.051 | 1.131 | 1.374 | 1.534 |
| oproj | mma/sub | 1.000 | 1.006 | 1.009 | 1.015 | 1.024 | 1.038 | 1.071 | 1.252 | 1.375 |
| oproj | mma/ones | 1.000 | 1.006 | 1.012 | 1.018 | 1.033 | 1.051 | 1.095 | 1.301 | 1.448 |
| oproj | mma/abl | 1.000 | 1.006 | 1.015 | 1.024 | 1.036 | 1.048 | 1.087 | 1.296 | 1.412 |
| lmhead | mma/bf16 | 1.000 | 1.002 | 1.002 | 1.003 | 1.004 | 1.005 | 1.005 | 1.009 | 1.008 |
