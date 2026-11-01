### Correctness (fp32 GPU reference, warp-per-row; rel L2 of the bf16 output; max over all configs/T; `vs bf16(ref)` isolates the kernel error from output rounding)

| shape | kernel | rel L2 vs fp32 ref | bf16 rounding floor | rel L2 vs bf16(ref) | n |
|---|---|---|---|---|---|
| gateup | mma/sub | 1.66e-03 | 1.66e-03 | 2.38e-05 | 30 |
| gateup | mma/ones | 1.66e-03 | 1.66e-03 | 8.27e-05 | 30 |
| gateup | cc/e1 | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| gateup | cc/naive | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| down | mma/sub | 1.69e-03 | 1.69e-03 | 2.63e-05 | 30 |
| down | mma/ones | 1.69e-03 | 1.69e-03 | 5.82e-05 | 30 |
| down | cc/e1 | 1.86e-07 | 0.00e+00 | 0.00e+00 | 4 |
| down | cc/naive | 1.86e-07 | 0.00e+00 | 0.00e+00 | 4 |
| qkvz | mma/sub | 1.67e-03 | 1.67e-03 | 4.24e-05 | 30 |
| qkvz | mma/ones | 1.67e-03 | 1.67e-03 | 1.15e-04 | 30 |
| qkvz | cc/e1 | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| qkvz | cc/naive | 1.09e-07 | 0.00e+00 | 0.00e+00 | 4 |
| oproj | mma/sub | 1.66e-03 | 1.66e-03 | 2.85e-05 | 30 |
| oproj | mma/ones | 1.66e-03 | 1.66e-03 | 7.31e-05 | 30 |
| oproj | cc/e1 | 1.17e-07 | 0.00e+00 | 0.00e+00 | 4 |
| oproj | cc/naive | 1.17e-07 | 0.00e+00 | 0.00e+00 | 4 |
| lmhead | mma/bf16 | 1.66e-03 | 1.66e-03 | 1.67e-04 | 20 |
| lmhead | cc/bf16 | 7.99e-08 | 0.00e+00 | 0.00e+00 | 3 |

### Q1 gateup 5120->34816: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/ablb ms (cfg) | mma/sub x | mma/ones x | mma/abl x | mma/ablb x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1126 (w8k8u2) | 0.1106 (w8k4u2) | 0.1085 (w8k8u2) | 0.1079 (w8k8u2) | 1.000 | 1.000 | 1.000 | 1.000 | 831 | 1.000 |
| 2 | 0.1134 (w8k4u2) | 0.1128 (w8k8u2) | 0.1087 (w8k8u2) | 0.1079 (w8k8u2) | 1.007 | 1.020 | 1.002 | 1.000 | 815 | 1.020 |
| 3 | 0.1126 (w8k8u2) | 0.1135 (w8k8u2) | 0.1090 (w8k8u2) | 0.1081 (w8k8u2) | 1.000 | 1.026 | 1.004 | 1.002 | 816 | 1.019 |
| 4 | 0.1130 (w8k8u2) | 0.1141 (w8k8u2) | 0.1092 (w8k8u2) | 0.1081 (w8k8u2) | 1.003 | 1.031 | 1.006 | 1.002 | 813 | 1.022 |
| 5 | 0.1141 (w8k8u2) | 0.1161 (w16k8u2) | 0.1098 (w8k8u2) | 0.1081 (w8k8u2) | 1.013 | 1.050 | 1.011 | 1.002 | 806 | 1.031 |
| 6 | 0.1188 (w4k4u3) | 0.1182 (w8k8u2) | 0.1122 (w8k8u2) | 0.1083 (w8k8u2) | 1.055 | 1.069 | 1.034 | 1.004 | 778 | 1.069 |
| 8 | 0.1241 (w4k4u3) | 0.1253 (w4k4u3) | 0.1181 (w4k4u3) | 0.1085 (w8k8u2) | 1.102 | 1.133 | 1.088 | 1.006 | 741 | 1.122 |
| 12 | 0.1440 (w8k2u3) | 0.1520 (w4k4u3) | 0.1503 (w8k2u3) | 0.1147 (w8k8u2) | 1.278 | 1.374 | 1.385 | 1.063 | 638 | 1.302 |
| 16 | 0.1604 (w4k4u3) | 0.1710 (w4k4u3) | 0.1679 (w8k2u3) | 0.1157 (w8k8u2) | 1.424 | 1.546 | 1.547 | 1.072 | 573 | 1.450 |

T=1 CUDA-core / read-only references: cc/e1 0.1157 ms (w8r4u2, 794 GB/s); cc/naive 0.1300 ms (w8r4u2, 707 GB/s); cc/ablfma 0.1085 ms (w8r4u2, 847 GB/s); cc/ablload 0.1065 ms (w8r2u2, 863 GB/s); read-only 0.1052 ms (873 GB/s)

### Q1 down 17408->5120: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/ablb ms (cfg) | mma/sub x | mma/ones x | mma/abl x | mma/ablb x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0566 (w8k4u2) | 0.0569 (w8k8u2) | 0.0555 (w8k8u2) | 0.0552 (w8k8u2) | 1.000 | 1.000 | 1.000 | 1.000 | 812 | 1.000 |
| 2 | 0.0572 (w8k4u2) | 0.0567 (w8k4u2) | 0.0558 (w8k8u2) | 0.0551 (w8k8u2) | 1.011 | 0.996 | 1.005 | 0.999 | 810 | 1.002 |
| 3 | 0.0583 (w8k8u2) | 0.0580 (w8k4u2) | 0.0563 (w8k8u2) | 0.0552 (w8k8u2) | 1.029 | 1.018 | 1.015 | 1.000 | 793 | 1.023 |
| 4 | 0.0593 (w8k8u2) | 0.0587 (w8k4u2) | 0.0575 (w8k8u2) | 0.0552 (w8k8u2) | 1.047 | 1.031 | 1.035 | 1.000 | 783 | 1.036 |
| 5 | 0.0595 (w4k4u3) | 0.0624 (w8k4u2) | 0.0609 (w8k8u2) | 0.0552 (w8k8u2) | 1.050 | 1.097 | 1.098 | 1.001 | 773 | 1.050 |
| 6 | 0.0610 (w4k4u3) | 0.0627 (w4k4u3) | 0.0589 (w4k4u3) | 0.0553 (w8k8u2) | 1.078 | 1.101 | 1.061 | 1.003 | 753 | 1.078 |
| 8 | 0.0635 (w4k4u3) | 0.0640 (w4k4u3) | 0.0610 (w4k4u3) | 0.0559 (w8k8u2) | 1.121 | 1.124 | 1.099 | 1.013 | 724 | 1.121 |
| 12 | 0.0746 (w4k4u3) | 0.0761 (w4k4u3) | 0.0739 (w8k4u2) | 0.0581 (w8k4u2) | 1.316 | 1.336 | 1.332 | 1.054 | 616 | 1.316 |
| 16 | 0.0825 (w4k4u3) | 0.0842 (w4k4u3) | 0.0811 (w8k4u2) | 0.0593 (w8k8u2) | 1.457 | 1.478 | 1.461 | 1.075 | 557 | 1.457 |

T=1 CUDA-core / read-only references: cc/e1 0.0605 ms (w8r2u2, 759 GB/s); cc/naive 0.0629 ms (w8r4u2, 731 GB/s); cc/ablfma 0.0555 ms (w8r2u2, 828 GB/s); cc/ablload 0.0544 ms (w8r2u2, 846 GB/s); read-only 0.0535 ms (858 GB/s)

### Q1 qkvz 5120->16384: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/ablb ms (cfg) | mma/sub x | mma/ones x | mma/abl x | mma/ablb x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0552 (w8k8u2) | 0.0538 (w8k4u2) | 0.0524 (w8k8u2) | 0.0521 (w8k8u2) | 1.000 | 1.000 | 1.000 | 1.000 | 803 | 1.000 |
| 2 | 0.0559 (w8k4u2) | 0.0557 (w8k8u2) | 0.0528 (w8k8u2) | 0.0521 (w8k8u2) | 1.013 | 1.035 | 1.007 | 1.000 | 776 | 1.035 |
| 3 | 0.0555 (w8k8u2) | 0.0558 (w8k8u2) | 0.0529 (w8k8u2) | 0.0522 (w8k8u2) | 1.005 | 1.036 | 1.009 | 1.002 | 780 | 1.030 |
| 4 | 0.0565 (w16k8u2) | 0.0563 (w8k8u2) | 0.0532 (w8k8u2) | 0.0522 (w8k8u2) | 1.024 | 1.046 | 1.016 | 1.002 | 768 | 1.046 |
| 5 | 0.0566 (w8k8u2) | 0.0569 (w8k8u2) | 0.0539 (w8k8u2) | 0.0523 (w8k8u2) | 1.026 | 1.057 | 1.027 | 1.004 | 764 | 1.052 |
| 6 | 0.0590 (w8k4u2) | 0.0591 (w8k8u2) | 0.0553 (w16k8u2) | 0.0523 (w8k8u2) | 1.069 | 1.097 | 1.055 | 1.004 | 733 | 1.095 |
| 8 | 0.0617 (w4k4u3) | 0.0619 (w4k4u3) | 0.0589 (w4k4u3) | 0.0526 (w4k4u3) | 1.119 | 1.149 | 1.123 | 1.010 | 700 | 1.147 |
| 12 | 0.0708 (w4k4u3) | 0.0729 (w4k4u3) | 0.0709 (w8k4u2) | 0.0566 (w8k8u2) | 1.284 | 1.354 | 1.352 | 1.087 | 610 | 1.316 |
| 16 | 0.0783 (w4k4u3) | 0.0813 (w4k4u3) | 0.0802 (w8k4u2) | 0.0560 (w8k4u2) | 1.419 | 1.510 | 1.529 | 1.075 | 552 | 1.455 |

T=1 CUDA-core / read-only references: cc/e1 0.0580 ms (w8r4u2, 746 GB/s); cc/naive 0.0656 ms (w8r4u2, 659 GB/s); cc/ablfma 0.0526 ms (w8r4u2, 822 GB/s); cc/ablload 0.0508 ms (w8r2u2, 852 GB/s); read-only 0.0504 ms (858 GB/s)

### Q1 oproj 6144->5120: ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/sub ms (cfg) | mma/ones ms (cfg) | mma/abl ms (cfg) | mma/ablb ms (cfg) | mma/sub x | mma/ones x | mma/abl x | mma/ablb x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0220 (w8k8u2) | 0.0219 (w8k8u2) | 0.0214 (w8k8u2) | 0.0209 (w8k8u2) | 1.000 | 1.000 | 1.000 | 1.000 | 741 | 1.000 |
| 2 | 0.0221 (w8k8u2) | 0.0220 (w8k8u2) | 0.0216 (w8k8u2) | 0.0209 (w8k8u2) | 1.004 | 1.005 | 1.008 | 1.000 | 737 | 1.005 |
| 3 | 0.0221 (w8k8u2) | 0.0221 (w8k8u2) | 0.0216 (w8k8u2) | 0.0209 (w8k8u2) | 1.004 | 1.009 | 1.009 | 1.000 | 735 | 1.009 |
| 4 | 0.0220 (w8k8u2) | 0.0221 (w8k8u2) | 0.0217 (w8k8u2) | 0.0209 (w8k8u2) | 1.001 | 1.011 | 1.015 | 1.000 | 737 | 1.006 |
| 5 | 0.0223 (w8k8u2) | 0.0225 (w8k8u2) | 0.0220 (w8k8u2) | 0.0210 (w8k8u2) | 1.015 | 1.029 | 1.030 | 1.003 | 726 | 1.021 |
| 6 | 0.0228 (w8k8u2) | 0.0231 (w8k8u2) | 0.0223 (w8k8u2) | 0.0210 (w8k8u2) | 1.035 | 1.055 | 1.045 | 1.003 | 712 | 1.041 |
| 8 | 0.0230 (w8k4u2) | 0.0230 (w8k4u2) | 0.0225 (w8k4u2) | 0.0213 (w8k8u2) | 1.047 | 1.050 | 1.054 | 1.018 | 706 | 1.050 |
| 12 | 0.0260 (w4k4u3) | 0.0268 (w4k4u3) | 0.0260 (w8k4u2) | 0.0226 (w8k4u2) | 1.181 | 1.225 | 1.215 | 1.079 | 624 | 1.187 |
| 16 | 0.0284 (w4k4u3) | 0.0295 (w4k4u3) | 0.0288 (w8k4u2) | 0.0230 (w8k4u2) | 1.289 | 1.348 | 1.347 | 1.101 | 572 | 1.295 |

T=1 CUDA-core / read-only references: cc/e1 0.0235 ms (w8r2u2, 689 GB/s); cc/naive 0.0255 ms (w8r2u2, 637 GB/s); cc/ablfma 0.0212 ms (w8r2u2, 766 GB/s); cc/ablload 0.0202 ms (w8r2u2, 805 GB/s); read-only 0.0198 ms (818 GB/s)

### Q1 lmhead 5120->248320 (bf16): ms (best config per T) and ratio to T=1 of the same kernel family

| T | mma/bf16 ms (cfg) | mma/bf16 x | best exact GB/s | x vs best-T=1 (cc or mma) |
|---|---|---|---|---|
| 1 | 3.0280 (w8k4u4) | 1.000 | 883 | 1.000 |
| 2 | 3.0280 (w8k4u4) | 1.000 | 840 | 1.052 |
| 3 | 3.0321 (w8k4u4) | 1.001 | 839 | 1.053 |
| 4 | 3.0418 (w8k4u4) | 1.005 | 836 | 1.057 |
| 5 | 3.0397 (w16k8u3) | 1.004 | 837 | 1.056 |
| 6 | 3.0408 (w16k8u3) | 1.004 | 836 | 1.056 |
| 8 | 3.0474 (w8k4u4) | 1.006 | 834 | 1.059 |
| 12 | 3.0679 (w8k8u4) | 1.013 | 829 | 1.066 |
| 16 | 3.0714 (w8k8u4) | 1.014 | 828 | 1.067 |

T=1 CUDA-core / read-only references: cc/bf16 2.8784 ms (w8r4u4, 883 GB/s); read-only 2.8554 ms (890 GB/s)

### T* = largest T in {1,2,3,4,5,6,8,12,16} whose time stays within x of T=1 (best exact kernel at each T; base = best exact kernel at T=1, cc or mma)

| shape | T*(1.05x) | T*(1.10x) | T*(1.15x) | T*(1.25x) | ratio@T=2 | @4 | @8 | @16 |
|---|---|---|---|---|---|---|---|---|
| gateup | 5 | 6 | 8 | 8 | 1.020 | 1.022 | 1.122 | 1.450 |
| down | 4 | 6 | 8 | 8 | 1.002 | 1.036 | 1.121 | 1.457 |
| qkvz | 4 | 6 | 8 | 8 | 1.035 | 1.046 | 1.147 | 1.455 |
| oproj | 8 | 8 | 8 | 12 | 1.005 | 1.006 | 1.050 | 1.295 |
| lmhead | 1 | 16 | 16 | 16 | 1.052 | 1.057 | 1.059 | 1.067 |

### Whole-step GEMV time (sum of measured per-shape kernels x layer counts; 64 gate|up, 64 down, 48 qkv|z + 16 attn qkv scaled 14336/16384, 64 o_proj, 1 lm_head); excludes non-GEMV work

| T | GEMV sum ms | x vs T=1 |
|---|---|---|
| 1 | 18.320 | 1.000 |
| 2 | 18.742 | 1.023 |
| 3 | 18.803 | 1.026 |
| 4 | 18.930 | 1.033 |
| 5 | 19.088 | 1.042 |
| 6 | 19.624 | 1.071 |
| 8 | 20.353 | 1.111 |
| 12 | 23.109 | 1.261 |
| 16 | 25.288 | 1.380 |

### MTP break-even model (draft step = MTP layer 0.85 GB BF16 + head, at the measured BF16 GB/s 883: restricted 32k head 0.336 GB -> 1.44 ms/step, full lm_head 2.54 GB -> 3.94 ms/step; chain acceptance p^i, tokens/iter = sum_(i<=k) p^i; verify step = measured GEMV(T=k+1) + F non-GEMV + delta*k DeltaNet snapshot cost)

**F=1.5 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.41 | 1.030 | 1.45x | 1.54x | 1.63x | 1.72x |
| 2 | 3 | 20.64 | 1.042 | 1.65x | 1.84x | 2.06x | 2.28x |
| 3 | 4 | 20.94 | 1.057 | 1.71x | 1.99x | 2.32x | 2.70x |
| 4 | 5 | 21.27 | 1.073 | 1.69x | 2.03x | 2.46x | 3.00x |
| 5 | 6 | 21.97 | 1.109 | 1.62x | 2.00x | 2.51x | 3.18x |

**F=1.5 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.41 | 1.030 | 1.30x | 1.38x | 1.46x | 1.55x |
| 2 | 3 | 20.64 | 1.042 | 1.36x | 1.52x | 1.70x | 1.88x |
| 3 | 4 | 20.94 | 1.057 | 1.32x | 1.53x | 1.79x | 2.08x |
| 4 | 5 | 21.27 | 1.073 | 1.23x | 1.48x | 1.80x | 2.19x |
| 5 | 6 | 21.97 | 1.109 | 1.13x | 1.40x | 1.75x | 2.23x |

**F=1.5 ms, delta=0.0 ms/extra token, draft = restricted 32k head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.24 | 1.021 | 1.46x | 1.55x | 1.65x | 1.74x |
| 2 | 3 | 20.30 | 1.024 | 1.68x | 1.87x | 2.09x | 2.32x |
| 3 | 4 | 20.43 | 1.031 | 1.74x | 2.03x | 2.36x | 2.75x |
| 4 | 5 | 20.59 | 1.039 | 1.73x | 2.09x | 2.53x | 3.08x |
| 5 | 6 | 21.12 | 1.066 | 1.67x | 2.06x | 2.58x | 3.28x |

**F=1.5 ms, delta=0.0 ms/extra token, draft = full lm_head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.24 | 1.021 | 1.31x | 1.39x | 1.48x | 1.56x |
| 2 | 3 | 20.30 | 1.024 | 1.38x | 1.54x | 1.72x | 1.91x |
| 3 | 4 | 20.43 | 1.031 | 1.34x | 1.56x | 1.81x | 2.11x |
| 4 | 5 | 20.59 | 1.039 | 1.26x | 1.51x | 1.83x | 2.23x |
| 5 | 6 | 21.12 | 1.066 | 1.16x | 1.43x | 1.79x | 2.27x |

**F=1.5 ms, delta=0.34 ms/extra token, draft = restricted 32k head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.58 | 1.038 | 1.44x | 1.53x | 1.62x | 1.71x |
| 2 | 3 | 20.98 | 1.059 | 1.63x | 1.82x | 2.03x | 2.25x |
| 3 | 4 | 21.45 | 1.082 | 1.67x | 1.95x | 2.27x | 2.64x |
| 4 | 5 | 21.95 | 1.107 | 1.65x | 1.98x | 2.40x | 2.93x |
| 5 | 6 | 22.82 | 1.152 | 1.57x | 1.94x | 2.43x | 3.09x |

**F=1.5 ms, delta=0.34 ms/extra token, draft = full lm_head** (plain step 19.82 ms = 50.5 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 20.58 | 1.038 | 1.29x | 1.37x | 1.45x | 1.54x |
| 2 | 3 | 20.98 | 1.059 | 1.35x | 1.50x | 1.68x | 1.86x |
| 3 | 4 | 21.45 | 1.082 | 1.30x | 1.51x | 1.76x | 2.05x |
| 4 | 5 | 21.95 | 1.107 | 1.21x | 1.46x | 1.77x | 2.15x |
| 5 | 6 | 22.82 | 1.152 | 1.11x | 1.37x | 1.72x | 2.18x |

**F=0.5 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 18.82 ms = 53.1 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 19.41 | 1.032 | 1.44x | 1.53x | 1.62x | 1.71x |
| 2 | 3 | 19.64 | 1.044 | 1.64x | 1.83x | 2.04x | 2.26x |
| 3 | 4 | 19.94 | 1.060 | 1.69x | 1.96x | 2.29x | 2.67x |
| 4 | 5 | 20.27 | 1.077 | 1.67x | 2.00x | 2.43x | 2.96x |
| 5 | 6 | 20.97 | 1.114 | 1.59x | 1.96x | 2.46x | 3.13x |

**F=0.5 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 18.82 ms = 53.1 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 19.41 | 1.032 | 1.29x | 1.37x | 1.45x | 1.53x |
| 2 | 3 | 19.64 | 1.044 | 1.34x | 1.50x | 1.67x | 1.85x |
| 3 | 4 | 19.94 | 1.060 | 1.29x | 1.50x | 1.75x | 2.04x |
| 4 | 5 | 20.27 | 1.077 | 1.20x | 1.45x | 1.76x | 2.14x |
| 5 | 6 | 20.97 | 1.114 | 1.10x | 1.36x | 1.71x | 2.17x |

**F=3.0 ms, delta=0.17 ms/extra token, draft = restricted 32k head** (plain step 21.32 ms = 46.9 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 21.91 | 1.028 | 1.46x | 1.55x | 1.64x | 1.73x |
| 2 | 3 | 22.14 | 1.039 | 1.67x | 1.87x | 2.08x | 2.31x |
| 3 | 4 | 22.44 | 1.053 | 1.73x | 2.02x | 2.35x | 2.74x |
| 4 | 5 | 22.77 | 1.068 | 1.72x | 2.07x | 2.51x | 3.06x |
| 5 | 6 | 23.47 | 1.101 | 1.66x | 2.04x | 2.56x | 3.26x |

**F=3.0 ms, delta=0.17 ms/extra token, draft = full lm_head** (plain step 21.32 ms = 46.9 tok/s)

| k | verify T | verify ms | x plain step | p1=0.6 | p1=0.7 | p1=0.8 | p1=0.9 |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 21.91 | 1.028 | 1.32x | 1.40x | 1.48x | 1.57x |
| 2 | 3 | 22.14 | 1.039 | 1.39x | 1.56x | 1.73x | 1.92x |
| 3 | 4 | 22.44 | 1.053 | 1.35x | 1.58x | 1.84x | 2.14x |
| 4 | 5 | 22.77 | 1.068 | 1.28x | 1.53x | 1.86x | 2.27x |
| 5 | 6 | 23.47 | 1.101 | 1.18x | 1.45x | 1.82x | 2.31x |

Break-even p1 (speed-up = 1.0) at F=1.5, delta=0.17:

| draft | k=1 | k=2 | k=3 |
|---|---|---|---|
| restricted 32k | 0.10 | 0.16 | 0.22 |
| full lm_head | 0.23 | 0.33 | 0.41 |

Upper bound with a perfectly flat verify (verify(T)=plain step, delta=0, F=1.5, restricted draft), p1=0.7: k=1: 1.58x, k=2: 1.91x, k=3: 2.08x

### Q2 sustained windows (3 s after 2 s warm, interleaved 2 rounds; T=1; best config per kernel; nvidia-smi 10 Hz inside the window; * = ablated, results invalid)

| shape | kernel | cfg | GB/s (r0 / r1) | ms | SM MHz median | power W median | SW-cap frac | mem MHz | samples |
|---|---|---|---|---|---|---|---|---|---|
| gateup | read/read | g246x512 | 873 / 873 | 0.1053 | 1718 | 305 | 1.00 | 9501 | 58 |
| gateup | cc/ablload* | w8r2u2 | 863 / 863 | 0.1065 | 1695 | 308 | 1.00 | 9501 | 58 |
| gateup | cc/ablfma* | w8r4u2 | 847 / 848 | 0.1085 | 1590 | 317 | 1.00 | 9501 | 58 |
| gateup | cc/naive | w8r4u2 | 716 / 716 | 0.1284 | 1590 | 332 | 1.00 | 9501 | 59 |
| gateup | cc/e1 | w8r4u2 | 793 / 794 | 0.1159 | 1560 | 326 | 1.00 | 9501 | 60 |
| gateup | mma/abl* | w8k8u2 | 847 / 845 | 0.1085 | 1564 | 317 | 1.00 | 9501 | 60 |
| gateup | mma/sub | w8k8u2 | 826 / 825 | 0.1112 | 1448 | 320 | 1.00 | 9501 | 58 |
| gateup | mma/ones | w8k4u2 | 831 / 833 | 0.1106 | 1496 | 319 | 1.00 | 9501 | 60 |
| gateup | mma/ablb* | w8k8u2 | 850 / 850 | 0.1081 | 1616 | 313 | 1.00 | 9501 | 59 |
| down | read/read | g246x512 | 858 / 858 | 0.0536 | 1748 | 305 | 1.00 | 9501 | 59 |
| down | cc/ablload* | w8r2u2 | 845 / 845 | 0.0544 | 1702 | 307 | 1.00 | 9501 | 58 |
| down | cc/ablfma* | w8r2u2 | 827 / 828 | 0.0556 | 1635 | 322 | 1.00 | 9501 | 59 |
| down | cc/naive | w8r4u2 | 732 / 731 | 0.0628 | 1560 | 329 | 1.00 | 9501 | 60 |
| down | cc/e1 | w8r4u2 | 785 / 783 | 0.0586 | 1541 | 324 | 1.00 | 9501 | 59 |
| down | mma/abl* | w8k8u2 | 826 / 828 | 0.0556 | 1552 | 317 | 1.00 | 9501 | 60 |
| down | mma/sub | w8k8u2 | 809 / 807 | 0.0568 | 1462 | 320 | 1.00 | 9501 | 60 |
| down | mma/ones | w8k4u2 | 818 / 816 | 0.0562 | 1500 | 319 | 1.00 | 9501 | 60 |
| down | mma/ablb* | w8k8u2 | 833 / 833 | 0.0552 | 1628 | 313 | 1.00 | 9501 | 59 |
| qkvz | read/read | g246x512 | 858 / 857 | 0.0504 | 1729 | 305 | 1.00 | 9501 | 117 |
| qkvz | cc/ablload* | w8r2u2 | 852 / 852 | 0.0508 | 1702 | 307 | 1.00 | 9501 | 89 |
| qkvz | cc/ablfma* | w8r4u2 | 823 / 823 | 0.0525 | 1628 | 316 | 1.00 | 9501 | 89 |
| qkvz | cc/naive | w8r4u2 | 655 / 658 | 0.0660 | 1650 | 332 | 1.00 | 9501 | 89 |
| qkvz | cc/e1 | w8r4u2 | 746 / 746 | 0.0580 | 1612 | 327 | 1.00 | 9501 | 89 |
| qkvz | mma/abl* | w8k8u2 | 825 / 825 | 0.0524 | 1579 | 316 | 1.00 | 9501 | 90 |
| qkvz | mma/sub | w8k8u2 | 791 / 787 | 0.0547 | 1500 | 321 | 1.00 | 9501 | 118 |
| qkvz | mma/ones | w8k8u2 | 794 / 796 | 0.0545 | 1564 | 321 | 1.00 | 9501 | 90 |
| qkvz | mma/ablb* | w8k8u2 | 830 / 830 | 0.0521 | 1631 | 313 | 1.00 | 9501 | 88 |
| oproj | read/read | g246x512 | 818 / 818 | 0.0198 | 1762 | 305 | 1.00 | 9501 | 117 |
| oproj | cc/ablload* | w8r2u2 | 805 / 805 | 0.0202 | 1740 | 308 | 1.00 | 9501 | 117 |
| oproj | cc/ablfma* | w8r2u2 | 766 / 766 | 0.0212 | 1628 | 320 | 1.00 | 9501 | 119 |
| oproj | cc/naive | w8r2u2 | 640 / 639 | 0.0253 | 1616 | 332 | 1.00 | 9501 | 118 |
| oproj | cc/e1 | w8r2u2 | 691 / 689 | 0.0235 | 1590 | 329 | 1.00 | 9501 | 119 |
| oproj | mma/abl* | w8k4u2 | 762 / 761 | 0.0213 | 1628 | 317 | 1.00 | 9501 | 120 |
| oproj | mma/sub | w8k8u2 | 739 / 737 | 0.0220 | 1598 | 322 | 1.00 | 9501 | 117 |
| oproj | mma/ones | w8k8u2 | 741 / 741 | 0.0219 | 1605 | 320 | 1.00 | 9501 | 120 |
| oproj | mma/ablb* | w8k8u2 | 775 / 775 | 0.0209 | 1672 | 312 | 1.00 | 9501 | 116 |
| lmhead | read/read | g246x512 | 890 / 891 | 2.8559 | 1770 | 308 | 1.00 | 9501 | 116 |
| lmhead | cc/bf16 | w8r4u4 | 883 / 883 | 2.8803 | 1740 | 311 | 1.00 | 9501 | 118 |
| lmhead | mma/bf16 | w8k4u4 | 840 / 840 | 3.0269 | 1725 | 312 | 1.00 | 9501 | 116 |

### Best exact (valid) GEMV per shape at T=1 under sustained load

| shape | best kernel | cfg | GB/s | SM MHz | power W | read-only GB/s same shape | % of read-only |
|---|---|---|---|---|---|---|---|
| gateup | mma/ones | w8k4u2 | 833 | 1496 | 319 | 873 | 95.4 |
| down | mma/ones | w8k4u2 | 818 | 1500 | 319 | 858 | 95.3 |
| qkvz | mma/ones | w8k8u2 | 796 | 1564 | 321 | 858 | 92.7 |
| oproj | mma/ones | w8k8u2 | 741 | 1605 | 320 | 818 | 90.7 |
| lmhead | cc/bf16 | w8r4u4 | 883 | 1740 | 311 | 891 | 99.2 |

### Q1 clocks along the T sweep (nvidia-smi 10 Hz over each 0.6 s final window; best exact mma/sub or ones per T)

| shape | T | kernel | GB/s | SM MHz | power W | samples |
|---|---|---|---|---|---|---|
| gateup | 1 | mma/ones w8k4u2 | 831 | 1515 | 320 | 6 |
| gateup | 4 | mma/sub w8k8u2 | 813 | 1350 | 319 | 6 |
| gateup | 8 | mma/sub w4k4u3 | 741 | 1125 | 320 | 6 |
| gateup | 16 | mma/sub w4k4u3 | 573 | 1365 | 326 | 6 |
| down | 1 | mma/sub w8k4u2 | 812 | 1245 | 321 | 6 |
| down | 4 | mma/ones w8k4u2 | 783 | 990 | 318 | 6 |
| down | 8 | mma/sub w4k4u3 | 724 | 1192 | 318 | 6 |
| down | 16 | mma/sub w4k4u3 | 557 | 1380 | 323 | 6 |
| qkvz | 1 | mma/ones w8k4u2 | 803 | 1875 | 313 | 6 |
| qkvz | 4 | mma/ones w8k8u2 | 768 | 1440 | 321 | 6 |
| qkvz | 8 | mma/sub w4k4u3 | 700 | 1470 | 319 | 6 |
| qkvz | 16 | mma/sub w4k4u3 | 552 | 1485 | 324 | 6 |
| oproj | 1 | mma/ones w8k8u2 | 741 | 1575 | 323 | 12 |
| oproj | 4 | mma/sub w8k8u2 | 737 | 1425 | 317 | 12 |
| oproj | 8 | mma/ones w8k4u2 | 706 | 1485 | 322 | 11 |
| oproj | 16 | mma/sub w4k4u3 | 572 | 1455 | 324 | 12 |
| lmhead | 1 | cc/bf16 w8r4u4 | 883 | 1710 | 311 | 12 |
| lmhead | 4 | mma/bf16 w8k4u4 | 836 | 1740 | 313 | 12 |
| lmhead | 8 | mma/bf16 w8k4u4 | 834 | 1680 | 313 | 12 |
| lmhead | 16 | mma/bf16 w8k8u4 | 828 | 1470 | 317 | 12 |

### Ablation of T growth: ms ratio to T=1, mma modes (sub = exact bf16x2 subtract, ones = lop3 only + table correction, abl = no dequant)

| shape | mode | T=1 | T=2 | T=3 | T=4 | T=5 | T=6 | T=8 | T=12 | T=16 |
|---|---|---|---|---|---|---|---|---|---|---|
| gateup | mma/sub | 1.000 | 1.007 | 1.000 | 1.003 | 1.013 | 1.055 | 1.102 | 1.278 | 1.424 |
| gateup | mma/ones | 1.000 | 1.020 | 1.026 | 1.031 | 1.050 | 1.069 | 1.133 | 1.374 | 1.546 |
| gateup | mma/abl | 1.000 | 1.002 | 1.004 | 1.006 | 1.011 | 1.034 | 1.088 | 1.385 | 1.547 |
| down | mma/sub | 1.000 | 1.011 | 1.029 | 1.047 | 1.050 | 1.078 | 1.121 | 1.316 | 1.457 |
| down | mma/ones | 1.000 | 0.996 | 1.018 | 1.031 | 1.097 | 1.101 | 1.124 | 1.336 | 1.478 |
| down | mma/abl | 1.000 | 1.005 | 1.015 | 1.035 | 1.098 | 1.061 | 1.099 | 1.332 | 1.461 |
| qkvz | mma/sub | 1.000 | 1.013 | 1.005 | 1.024 | 1.026 | 1.069 | 1.119 | 1.284 | 1.419 |
| qkvz | mma/ones | 1.000 | 1.035 | 1.036 | 1.046 | 1.057 | 1.097 | 1.149 | 1.354 | 1.510 |
| qkvz | mma/abl | 1.000 | 1.007 | 1.009 | 1.016 | 1.027 | 1.055 | 1.123 | 1.352 | 1.529 |
| oproj | mma/sub | 1.000 | 1.004 | 1.004 | 1.001 | 1.015 | 1.035 | 1.047 | 1.181 | 1.289 |
| oproj | mma/ones | 1.000 | 1.005 | 1.009 | 1.011 | 1.029 | 1.055 | 1.050 | 1.225 | 1.348 |
| oproj | mma/abl | 1.000 | 1.008 | 1.009 | 1.015 | 1.030 | 1.045 | 1.054 | 1.215 | 1.347 |
| lmhead | mma/bf16 | 1.000 | 1.000 | 1.001 | 1.005 | 1.004 | 1.004 | 1.006 | 1.013 | 1.014 |

### ncu counters (natural clocks, one launch each; `run.py --stages ncu`; abl/ablb results invalid, timing only)

| shape/T/kernel/cfg | us | DRAM %peak | L1 %peak | L1 hit % | HMMA pipe % | issue % | warp-inst M | LSU inst M | SM GHz |
|---|---|---|---|---|---|---|---|---|---|
| gateup/T1/mma/sub/w8k8u2 | 112.86 | 90.30 | 18.93 | 47.81 | 19.09 | 40.96 | 23.88 | 0.73 | 1.65 |
| gateup/T1/cc/e1/w8r4u2 | 114.85 | 89.17 | 16.19 | 49.81 | 0.00 | 42.55 | 28.68 | 0.74 | 1.90 |
| gateup/T1/read/read/g246x512 | 109.47 | 93.41 | 16.81 | 0.00 | 0.00 | 4.52 | 2.95 | 0.36 | 1.90 |
| gateup/T8/mma/sub/w4k4u3 | 118.27 | 86.38 | 45.61 | 71.91 | 18.49 | 39.01 | 23.48 | 0.72 | 1.67 |
| gateup/T8/mma/abl/w4k4u3 | 115.39 | 88.63 | 49.44 | 72.82 | 18.62 | 26.97 | 16.12 | 0.89 | 1.65 |
| gateup/T8/mma/ablb/w8k8u2 | 111.62 | 91.31 | 16.65 | 32.48 | 16.42 | 13.49 | 9.14 | 0.39 | 1.91 |
| gateup/T16/mma/sub/w4k4u3 | 127.87 | 80.33 | 65.41 | 74.76 | 30.87 | 32.59 | 23.51 | 1.08 | 1.90 |
| gateup/T16/mma/ablb/w8k8u2 | 115.14 | 88.87 | 17.24 | 34.08 | 34.06 | 18.51 | 12.09 | 0.44 | 1.82 |
| down/T1/mma/sub/w8k4u2 | 55.94 | 90.35 | 17.59 | 45.23 | 16.90 | 35.62 | 11.71 | 0.35 | 1.89 |
| down/T8/mma/sub/w4k4u3 | 61.28 | 84.00 | 40.99 | 71.52 | 16.08 | 33.74 | 11.67 | 0.35 | 1.81 |
| down/T16/mma/sub/w4k4u3 | 77.66 | 66.77 | 61.27 | 75.83 | 28.08 | 29.58 | 11.72 | 0.53 | 1.68 |
| lmhead/T1/cc/bf16/w8r4u4 | 2882.82 | 96.88 | 17.89 | 20.16 | 0.00 | 9.74 | 170.85 | 7.70 | 1.88 |
| lmhead/T1/mma/bf16/w8k4u4 | 2997.44 | 93.26 | 16.68 | 5.81 | 4.19 | 10.31 | 195.35 | 7.60 | 1.95 |
| lmhead/T16/mma/bf16/w8k8u4 | 3053.38 | 91.69 | 30.23 | 6.34 | 8.28 | 12.02 | 230.84 | 10.47 | 1.93 |
