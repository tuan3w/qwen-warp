## R1 per-head table (in-sample over all 32 tasks, descriptive; first non-sink pointer of the row that produced the anchor)

rows (cycle starts) = 20000; `cons` = P(token[q+o] == anchor), `d1|cons` = P(token[q+o+1] == next token | consistent), best offset o per head.

| rank | layer(attn idx) | head | o | cons rate | d1 given cons | cons rows | cons&d1 rows |
|---|---|---|---|---|---|---|---|
| 1 | 14 (layer 59) | 21 | +0 | 0.586 | 0.789 | 11712 | 9244 |
| 2 | 11 (layer 47) | 19 | +0 | 0.503 | 0.788 | 10059 | 7925 |
| 3 | 15 (layer 63) | 9 | +0 | 0.473 | 0.832 | 9456 | 7864 |
| 4 | 4 (layer 19) | 10 | +1 | 0.475 | 0.824 | 9504 | 7827 |
| 5 | 0 (layer 3) | 23 | +1 | 0.458 | 0.815 | 9163 | 7464 |
| 6 | 14 (layer 59) | 19 | +0 | 0.482 | 0.773 | 9638 | 7451 |
| 7 | 3 (layer 15) | 19 | +0 | 0.464 | 0.754 | 9289 | 7004 |
| 8 | 2 (layer 11) | 16 | +1 | 0.457 | 0.764 | 9144 | 6985 |
| 9 | 2 (layer 11) | 14 | +1 | 0.445 | 0.768 | 8894 | 6835 |
| 10 | 3 (layer 15) | 21 | +0 | 0.421 | 0.803 | 8422 | 6764 |
| 11 | 2 (layer 11) | 17 | +1 | 0.422 | 0.770 | 8433 | 6494 |
| 12 | 15 (layer 63) | 7 | +0 | 0.449 | 0.722 | 8981 | 6480 |
| 13 | 1 (layer 7) | 8 | +1 | 0.412 | 0.786 | 8239 | 6476 |
| 14 | 0 (layer 3) | 21 | +0 | 0.425 | 0.744 | 8506 | 6328 |
| 15 | 3 (layer 15) | 22 | +0 | 0.401 | 0.782 | 8023 | 6273 |
| 16 | 10 (layer 43) | 16 | +1 | 0.395 | 0.784 | 7891 | 6187 |
| 17 | 2 (layer 11) | 12 | +1 | 0.369 | 0.810 | 7380 | 5978 |
| 18 | 1 (layer 7) | 7 | +1 | 0.399 | 0.744 | 7974 | 5933 |
| 19 | 3 (layer 15) | 1 | +1 | 0.367 | 0.805 | 7332 | 5901 |
| 20 | 0 (layer 3) | 19 | +0 | 0.389 | 0.750 | 7785 | 5841 |

Layer profile (attention layer idx, model layer 4l+3): number of heads with cons rate > 5% and d1|cons > 0.8, best cons rate.

| attn idx | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| good heads | 1 | 0 | 1 | 3 | 1 | 1 | 1 | 0 | 0 | 0 | 1 | 0 | 7 | 3 | 3 | 12 |
| best cons | 0.46 | 0.41 | 0.46 | 0.46 | 0.48 | 0.24 | 0.30 | 0.14 | 0.13 | 0.09 | 0.39 | 0.50 | 0.37 | 0.26 | 0.59 | 0.47 |

Offsets of the good heads (token[q+o] is the anchor): {-2: 1, -1: 9, 0: 12, 1: 12, 2: 0}; of all heads: {-2: 14, -1: 70, 0: 119, 1: 179, 2: 2}.

## R1: anchor-consistent mass-weighted pointer vote, held-out (2-fold), threshold calibrated on the train fold for first-draft precision >= 0.9; `cov` = fraction of cycle starts with a draft, `prec` = P(first draft right | issued), `run` = mean accepted run (draft slots 1..7, no MTP involved) | M = number of heads (best by train score)

| variant | tau (fold 0 / 1) | a cov / prec / run | b cov / prec / run | c cov / prec / run | d cov / prec / run | e cov / prec / run | a-c cov / prec |
|---|---|---|---|---|---|---|---|
| top-1 heads | 0.687 / 0.601 | 0.421 / 0.961 / 5.90 | 0.202 / 0.889 / 4.84 | 0.282 / 0.945 / 4.74 | 0.048 / 0.516 / 0.93 | 0.128 / 0.744 / 2.33 | 0.323 / 0.944 |
| top-4 heads | 1.5 / 1.3 | 0.550 / 0.963 / 6.00 | 0.252 / 0.914 / 5.02 | 0.361 / 0.944 / 4.59 | 0.061 / 0.517 / 0.88 | 0.185 / 0.754 / 2.38 | 0.417 / 0.950 |
| top-16 heads | 8.86 / 6.06 | 0.266 / 0.959 / 6.10 | 0.092 / 0.929 / 5.50 | 0.078 / 0.922 / 4.17 | 0.021 / 0.474 / 0.83 | 0.059 / 0.708 / 2.11 | 0.173 / 0.951 |
| top-64 heads | 22.7 / 15.1 | 0.220 / 0.969 / 6.12 | 0.084 / 0.913 / 5.28 | 0.085 / 0.933 / 4.19 | 0.022 / 0.459 / 0.78 | 0.065 / 0.741 / 2.36 | 0.150 / 0.955 |
| top-384 heads | - / 35 | 0.048 / 0.988 / 6.34 | 0.022 / 0.844 / 4.56 | 0.024 / 0.923 / 2.71 | 0.010 / 0.455 / 0.75 | 0.022 / 0.752 / 2.23 | 0.035 / 0.951 |

## Next-token precision of the pointer / neighbour WITHOUT the anchor filter (what the state 'knows' about the token it just produced), held-out, tau calibrated for precision >= 0.9 on P(pred == anchor)

| predictor | a cov / prec | b cov / prec | c cov / prec | d cov / prec | e cov / prec |
|---|---|---|---|---|---|
| R1 pointer vote top-16 heads | 0.684 / 0.957 | 0.728 / 0.908 | 0.752 / 0.948 | 0.113 / 0.603 | 0.378 / 0.790 |
| R1 pointer vote all heads | 0.273 / 0.966 | 0.349 / 0.923 | 0.332 / 0.972 | 0.057 / 0.469 | 0.182 / 0.822 |
| R2 kNN h65raw (top-1 neighbour's next token) | 0.107 / 0.922 | 0.313 / 0.980 | 0.325 / 0.991 | 0.057 / 0.248 | 0.117 / 0.898 |
| R2 kNN h65 (top-1 neighbour's next token) | 0.122 / 0.924 | 0.323 / 0.973 | 0.311 / 0.992 | 0.063 / 0.268 | 0.161 / 0.909 |
| R2 kNN h65z (top-1 neighbour's next token) | 0.141 / 0.938 | 0.372 / 0.975 | 0.354 / 0.993 | 0.078 / 0.261 | 0.185 / 0.912 |
| R2 kNN h65n (top-1 neighbour's next token) | 0.114 / 0.924 | 0.309 / 0.979 | 0.306 / 0.994 | 0.061 / 0.264 | 0.146 / 0.918 |
| R2 kNN h60 (top-1 neighbour's next token) | 0.032 / 0.930 | 0.211 / 0.960 | 0.188 / 0.975 | 0.027 / 0.242 | 0.053 / 0.893 |
| R2 kNN h60z (top-1 neighbour's next token) | 0.016 / 0.988 | 0.126 / 0.962 | 0.126 / 0.981 | 0.016 / 0.192 | 0.028 / 0.870 |
| R2 kNN h60n (top-1 neighbour's next token) | 0.029 / 0.935 | 0.195 / 0.967 | 0.170 / 0.975 | 0.026 / 0.228 | 0.049 / 0.894 |
| R2 kNN h48 (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN h48z (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN h48n (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN h32 (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN h32z (top-1 neighbour's next token) | 0.002 / 0.667 | 0.017 / 0.967 | 0.009 / 0.900 | 0.002 / 0.250 | 0.002 / 1.000 |
| R2 kNN h32n (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN h16 (top-1 neighbour's next token) | 0.003 / 0.588 | 0.083 / 0.914 | 0.111 / 0.970 | 0.002 / 0.556 | 0.018 / 0.488 |
| R2 kNN k15 (top-1 neighbour's next token) | 0.058 / 0.896 | 0.172 / 0.979 | 0.217 / 0.991 | 0.049 / 0.417 | 0.143 / 0.912 |
| R2 kNN k15z (top-1 neighbour's next token) | 0.071 / 0.904 | 0.211 / 0.983 | 0.252 / 0.983 | 0.054 / 0.433 | 0.147 / 0.909 |
| R2 kNN k11 (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN k11z (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN k7 (top-1 neighbour's next token) | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| R2 kNN k7z (top-1 neighbour's next token) | 0.002 / 0.556 | 0.007 / 1.000 | 0.002 / 1.000 | 0.002 / 0.250 | 0.003 / 0.875 |

## R2: anchor-masked cosine kNN (best neighbour whose next token equals the anchor), held-out, tau calibrated for first-draft precision >= 0.9 (h<slot> = residual after layer slot-1, h65 = final norm out, centred on the prompt mean unless raw; k<i> = K cache of attention layer i, dims 64..255 of the 4 kv heads)

| variant | tau (fold 0 / 1) | a cov / prec / run | b cov / prec / run | c cov / prec / run | d cov / prec / run | e cov / prec / run | a-c cov / prec |
|---|---|---|---|---|---|---|---|
| h65raw | 0.84 / 0.844 | 0.144 / 0.824 / 2.46 | 0.354 / 0.948 / 4.88 | 0.374 / 0.940 / 5.16 | 0.021 / 0.495 / 0.78 | 0.149 / 0.898 / 3.06 | 0.256 / 0.912 |
| h65 | 0.834 / 0.841 | 0.140 / 0.821 / 2.45 | 0.341 / 0.952 / 4.93 | 0.340 / 0.953 / 5.31 | 0.022 / 0.531 / 0.89 | 0.175 / 0.874 / 2.88 | 0.244 / 0.916 |
| h65z | 0.806 / 0.808 | 0.137 / 0.828 / 2.49 | 0.366 / 0.943 / 4.84 | 0.349 / 0.964 / 5.36 | 0.021 / 0.527 / 0.87 | 0.173 / 0.882 / 2.90 | 0.252 / 0.918 |
| h65n | 0.832 / 0.843 | 0.137 / 0.821 / 2.47 | 0.334 / 0.957 / 4.95 | 0.339 / 0.953 / 5.31 | 0.022 / 0.500 / 0.81 | 0.167 / 0.883 / 2.93 | 0.240 / 0.918 |
| h60 | 0.783 / 0.792 | 0.099 / 0.884 / 2.94 | 0.319 / 0.948 / 4.92 | 0.302 / 0.946 / 5.21 | 0.027 / 0.441 / 0.73 | 0.149 / 0.870 / 2.98 | 0.209 / 0.932 |
| h60z | 0.742 / 0.76 | 0.111 / 0.868 / 2.96 | 0.313 / 0.948 / 5.05 | 0.295 / 0.951 / 5.28 | 0.028 / 0.423 / 0.69 | 0.160 / 0.867 / 2.94 | 0.212 / 0.928 |
| h60n | 0.778 / 0.796 | 0.098 / 0.884 / 2.96 | 0.315 / 0.950 / 5.00 | 0.292 / 0.950 / 5.27 | 0.028 / 0.431 / 0.72 | 0.153 / 0.874 / 2.97 | 0.206 / 0.935 |
| h48 | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| h48z | - / 0.881 | 0.001 / 1.000 / 3.25 | 0.009 / 0.968 / 4.90 | 0.001 / 1.000 / 6.33 | 0.002 / 0.400 / 0.70 | 0.004 / 0.895 / 3.16 | 0.003 / 0.974 |
| h48n | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| h32 | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| h32z | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| h32n | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| h16 | 0.992 / 0.97 | 0.005 / 0.571 / 2.00 | 0.164 / 0.874 / 4.35 | 0.209 / 0.964 / 5.85 | 0.005 / 0.381 / 0.90 | 0.030 / 0.633 / 2.68 | 0.096 / 0.904 |
| k15 | 0.892 / 0.885 | 0.048 / 0.878 / 3.36 | 0.169 / 0.977 / 5.71 | 0.223 / 0.964 / 5.21 | 0.020 / 0.478 / 0.76 | 0.129 / 0.838 / 2.85 | 0.121 / 0.953 |
| k15z | 0.881 / 0.874 | 0.056 / 0.865 / 3.23 | 0.194 / 0.972 / 5.53 | 0.237 / 0.954 / 5.17 | 0.022 / 0.464 / 0.72 | 0.125 / 0.845 / 2.90 | 0.136 / 0.945 |
| k11 | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| k11z | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| k7 | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |
| k7z | - / - | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 |

## Modelled tok/s (measured acceptance x K11's MEASURED cycle cost V(T=1..8) per context, interpolated; per-cycle host gap 0.01 ms on tail policies), ratio over MTP-only [task-bootstrap 95% CI] and over MTP + R0

| policy | a tok/s (E) | b tok/s (E) | c tok/s (E) | d tok/s (E) | e tok/s (E) | a-c |
|---|---|---|---|---|---|---|
| MTP-only | 175.1 (3.70) | 175.2 (3.69) | 186.7 (3.92) | 111.5 (2.33) | 167.7 (3.49) | 177.3 (3.74) |
| MTP + R0 (K11: l>=3, tail<=4) | 238.8 (5.44) | 229.5 (5.23) | 269.7 (6.25) | 111.1 (2.34) | 175.7 (3.80) | 241.0 (5.51) |
| MTP + R1@1 (top-1 heads) | 225.7 (5.03) | 219.5 (4.89) | 248.4 (5.62) | 111.4 (2.33) | 172.9 (3.65) | 227.7 (5.09) |
| MTP + R1@4 (top-4 heads) | 232.6 (5.22) | 222.5 (4.97) | 251.8 (5.70) | 111.5 (2.33) | 173.7 (3.67) | 232.6 (5.22) |
| MTP + R1@16 (top-16 heads) | 231.7 (5.19) | 222.3 (4.97) | 254.8 (5.78) | 111.6 (2.33) | 172.5 (3.64) | 232.6 (5.22) |
| MTP + R1@64 (top-64 heads) | 231.0 (5.19) | 222.8 (5.00) | 253.4 (5.78) | 111.4 (2.33) | 173.0 (3.67) | 232.2 (5.23) |
| MTP + R1@384 (top-384 heads) | 231.9 (5.19) | 222.9 (4.98) | 258.0 (5.86) | 111.4 (2.33) | 172.9 (3.65) | 233.4 (5.24) |
| MTP + R2 (h65raw) | 199.8 (4.37) | 211.3 (4.69) | 251.5 (5.70) | 111.5 (2.34) | 171.8 (3.65) | 212.0 (4.69) |
| MTP + best-of fallback (R0 > R1 > R2) | 238.6 (5.45) | 229.8 (5.24) | 269.9 (6.27) | 111.0 (2.34) | 175.2 (3.80) | 241.0 (5.52) |
| MTP + best-of longest tail | 238.6 (5.45) | 229.8 (5.24) | 269.9 (6.27) | 111.0 (2.34) | 175.2 (3.80) | 241.1 (5.52) |
| MTP + best-of longest tail, calibrated precision>=0.9 gates (no MTP-agreement requirement) | 238.6 (5.47) | 230.0 (5.29) | 270.3 (6.29) | 110.8 (2.34) | 175.1 (3.83) | 241.2 (5.55) |
| MTP + oracle of R0/R1/R2 (not implementable) | 239.3 (5.47) | 233.6 (5.32) | 273.6 (6.35) | 111.0 (2.34) | 175.7 (3.81) | 243.3 (5.57) |
| MTP + R0 tail + R1 replacement of the chain (confident and disagreeing with MTP) | 238.5 (5.44) | 229.4 (5.23) | 269.7 (6.25) | 111.1 (2.34) | 175.3 (3.80) | 240.8 (5.50) |
| MTP + R0 tail + R2 replacement of the chain (confident and disagreeing with MTP) | 238.8 (5.44) | 229.5 (5.23) | 269.7 (6.25) | 111.1 (2.34) | 175.7 (3.80) | 241.0 (5.51) |

**ratio over MTP-only** [95% CI]

| policy | a | b | c | d | e | a-c |
|---|---|---|---|---|---|---|
| MTP + R0 (K11: l>=3, tail<=4) | x1.363 [1.198, 1.567] | x1.310 [1.157, 1.445] | x1.444 [1.333, 1.508] | x0.997 [0.993, 1.003] | x1.048 [1.035, 1.062] | x1.359 [1.259, 1.461] |
| MTP + R1@1 (top-1 heads) | x1.289 [1.158, 1.460] | x1.253 [1.124, 1.377] | x1.331 [1.240, 1.372] | x0.999 [0.998, 1.001] | x1.031 [1.019, 1.044] | x1.284 [1.206, 1.372] |
| MTP + R1@4 (top-4 heads) | x1.328 [1.176, 1.520] | x1.271 [1.136, 1.416] | x1.349 [1.262, 1.388] | x1.000 [0.998, 1.004] | x1.036 [1.023, 1.048] | x1.312 [1.221, 1.412] |
| MTP + R1@16 (top-16 heads) | x1.323 [1.175, 1.515] | x1.269 [1.132, 1.400] | x1.365 [1.308, 1.409] | x1.001 [0.998, 1.006] | x1.029 [1.019, 1.039] | x1.312 [1.224, 1.414] |
| MTP + R1@64 (top-64 heads) | x1.319 [1.162, 1.512] | x1.272 [1.137, 1.413] | x1.357 [1.306, 1.399] | x1.000 [0.997, 1.004] | x1.032 [1.018, 1.045] | x1.310 [1.221, 1.414] |
| MTP + R1@384 (top-384 heads) | x1.324 [1.169, 1.513] | x1.272 [1.138, 1.407] | x1.382 [1.316, 1.415] | x1.000 [0.998, 1.002] | x1.031 [1.019, 1.042] | x1.317 [1.227, 1.413] |
| MTP + R2 (h65raw) | x1.141 [1.084, 1.211] | x1.206 [1.091, 1.318] | x1.347 [1.257, 1.413] | x1.000 [0.997, 1.005] | x1.024 [1.013, 1.036] | x1.196 [1.145, 1.251] |
| MTP + best-of fallback (R0 > R1 > R2) | x1.362 [1.202, 1.566] | x1.312 [1.159, 1.470] | x1.446 [1.331, 1.515] | x0.996 [0.991, 1.001] | x1.045 [1.033, 1.058] | x1.360 [1.257, 1.467] |
| MTP + best-of longest tail | x1.362 [1.216, 1.564] | x1.312 [1.156, 1.470] | x1.446 [1.331, 1.518] | x0.996 [0.991, 1.001] | x1.045 [1.033, 1.058] | x1.360 [1.263, 1.470] |
| MTP + best-of longest tail, calibrated precision>=0.9 gates (no MTP-agreement requirement) | x1.363 [1.212, 1.560] | x1.313 [1.168, 1.453] | x1.448 [1.356, 1.504] | x0.994 [0.990, 1.000] | x1.044 [1.033, 1.057] | x1.361 [1.269, 1.466] |
| MTP + oracle of R0/R1/R2 (not implementable) | x1.366 [1.213, 1.573] | x1.334 [1.184, 1.498] | x1.465 [1.331, 1.532] | x0.996 [0.991, 1.001] | x1.048 [1.034, 1.064] | x1.373 [1.273, 1.480] |
| MTP + R0 tail + R1 replacement of the chain (confident and disagreeing with MTP) | x1.362 [1.203, 1.568] | x1.310 [1.161, 1.438] | x1.444 [1.336, 1.508] | x0.997 [0.993, 1.003] | x1.045 [1.032, 1.060] | x1.358 [1.260, 1.462] |
| MTP + R0 tail + R2 replacement of the chain (confident and disagreeing with MTP) | x1.363 [1.209, 1.565] | x1.310 [1.158, 1.444] | x1.444 [1.337, 1.510] | x0.997 [0.993, 1.003] | x1.048 [1.034, 1.062] | x1.359 [1.261, 1.469] |

**ratio over MTP + R0 (K11 tail)** [95% CI]

| policy | a | b | c | d | e | a-c |
|---|---|---|---|---|---|---|
| MTP + R1@1 (top-1 heads) | x0.945 [0.923, 0.963] | x0.956 [0.934, 0.982] | x0.921 [0.879, 0.974] | x1.002 [0.998, 1.007] | x0.984 [0.975, 0.994] | x0.945 [0.931, 0.959] |
| MTP + R1@4 (top-4 heads) | x0.974 [0.952, 0.990] | x0.970 [0.948, 0.993] | x0.934 [0.882, 0.982] | x1.003 [0.999, 1.006] | x0.989 [0.979, 1.000] | x0.965 [0.949, 0.980] |
| MTP + R1@16 (top-16 heads) | x0.970 [0.946, 0.987] | x0.969 [0.949, 0.994] | x0.945 [0.916, 1.005] | x1.004 [1.000, 1.007] | x0.982 [0.972, 0.992] | x0.965 [0.952, 0.980] |
| MTP + R1@64 (top-64 heads) | x0.968 [0.948, 0.983] | x0.971 [0.952, 0.993] | x0.940 [0.892, 1.006] | x1.003 [0.999, 1.006] | x0.985 [0.974, 0.999] | x0.964 [0.949, 0.979] |
| MTP + R1@384 (top-384 heads) | x0.971 [0.953, 0.984] | x0.971 [0.958, 0.988] | x0.957 [0.930, 1.003] | x1.003 [0.999, 1.006] | x0.984 [0.975, 0.996] | x0.969 [0.958, 0.979] |
| MTP + R2 (h65raw) | x0.837 [0.762, 0.904] | x0.921 [0.870, 0.956] | x0.933 [0.906, 0.966] | x1.003 [0.999, 1.007] | x0.978 [0.970, 0.986] | x0.880 [0.840, 0.917] |
| MTP + best-of fallback (R0 > R1 > R2) | x0.999 [0.996, 1.002] | x1.001 [0.997, 1.008] | x1.001 [0.995, 1.008] | x0.998 [0.997, 1.000] | x0.997 [0.995, 0.999] | x1.000 [0.998, 1.003] |
| MTP + best-of longest tail | x0.999 [0.996, 1.002] | x1.001 [0.998, 1.008] | x1.001 [0.995, 1.008] | x0.998 [0.997, 1.000] | x0.997 [0.995, 0.999] | x1.000 [0.998, 1.003] |
| MTP + best-of longest tail, calibrated precision>=0.9 gates (no MTP-agreement requirement) | x1.000 [0.994, 1.007] | x1.002 [0.996, 1.009] | x1.002 [0.997, 1.029] | x0.997 [0.996, 0.998] | x0.997 [0.993, 1.000] | x1.001 [0.997, 1.006] |
| MTP + oracle of R0/R1/R2 (not implementable) | x1.002 [0.999, 1.007] | x1.018 [0.998, 1.039] | x1.015 [0.997, 1.031] | x0.999 [0.998, 1.000] | x1.000 [0.998, 1.003] | x1.010 [1.002, 1.018] |

Gate selection per fold (fitted on the train tasks: min agreement of the retriever's slots 1..3 with the MTP drafts, confidence cut at a train quantile; objective = mean over classes of the modelled ratio over MTP-only with a d/e no-loss guard):

- R1@1: fold 0: agree>=3, tau 0.0301 (q0.0), train ratios 1.261/1.185/1.359/1.000/1.046; fold 1: agree>=3, tau 0.0302 (q0.0), train ratios 1.310/1.330/1.296/0.999/1.016
- R1@4: fold 0: agree>=3, tau 0.0301 (q0.0), train ratios 1.292/1.190/1.361/1.001/1.051; fold 1: agree>=3, tau 0.0301 (q0.0), train ratios 1.355/1.364/1.334/0.999/1.021
- R1@16: fold 0: agree>=3, tau 0.0301 (q0.0), train ratios 1.292/1.197/1.394/1.002/1.045; fold 1: agree>=3, tau 0.0301 (q0.0), train ratios 1.355/1.360/1.324/0.999/1.018
- R1@64: fold 0: agree>=3, tau 0.0302 (q0.0), train ratios 1.289/1.197/1.379/1.003/1.049; fold 1: agree>=2, tau 0.0302 (q0.0), train ratios 1.351/1.397/1.323/0.998/1.015
- R1@384: fold 0: agree>=3, tau 0.0302 (q0.0), train ratios 1.283/1.195/1.397/1.001/1.045; fold 1: agree>=3, tau 0.0304 (q0.0), train ratios 1.347/1.360/1.371/0.999/1.016
- R2: fold 0: agree>=3, tau 0.0256 (q0.0), train ratios 1.111/1.176/1.394/1.003/1.041; fold 1: agree>=2, tau 0.014 (q0.0), train ratios 1.167/1.258/1.296/0.999/1.014
- R1 replacement: fold 0: tau 2.75 (q0.95), train ratios 1.337/1.244/1.477/1.001/1.057; fold 1: tau never (qNone), train ratios 1.382/1.393/1.405/0.993/1.037; used in a 0.55%, b 0.14%, c 0.28%, d 0.11%, e 0.59% of cycles; replaced cycles where the pointer run beats the chain run: a 0.52, b 0.20, c 0.17, d 0.20, e 0.00
- R2 replacement: fold 0: tau 0.976 (q0.99), train ratios 1.338/1.247/1.470/1.001/1.058; fold 1: tau never (qNone), train ratios 1.382/1.393/1.405/0.993/1.037; used in a 0.00%, b 0.03%, c 0.09%, d 0.00%, e 0.00% of cycles; replaced cycles where the pointer run beats the chain run: a 0.00, b 0.00, c 0.00, d 0.00, e 0.00
- best-of / overlap use R1 with the top-4 head subset (best held-out a-c coverage at precision >= 0.9) and R2 = `h65raw`.

## Per-position view, calibrated gates: gates calibrated for first-draft precision >= 0.9 on the train fold (no MTP agreement requirement)

Every position as a cycle start; run = accepted drafts 0..7 of MTP chain + tail (chain-only run when no tail is issued). `P(R0|R1)` = share of R1-issued cycles where R0 also issues; `extra` = mean of max(0, run_X - run_R0) per cycle.

| class | cycles | R0 issued | R1 issued | R2 issued | R1&R0 | R1 only | R2 only | union | P(R0\|R1) | E[run] MTP+R0 | +R1 | +R2 | best-of | extra R1 | extra R2 | extra best-of |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 5296 | 0.756 | 0.550 | 0.144 | 0.495 | 0.055 | 0.014 | 0.825 | 0.900 | 5.241 | 4.557 | 2.893 | 5.292 | +0.074 | +0.015 | +0.050 |
| b | 3538 | 0.743 | 0.252 | 0.354 | 0.202 | 0.051 | 0.028 | 0.822 | 0.800 | 4.848 | 3.394 | 3.619 | 4.884 | +0.037 | +0.136 | +0.036 |
| c | 2126 | 0.882 | 0.361 | 0.374 | 0.327 | 0.034 | 0.011 | 0.927 | 0.906 | 5.636 | 3.714 | 3.944 | 5.650 | +0.022 | +0.059 | +0.014 |
| d | 4438 | 0.094 | 0.061 | 0.021 | 0.010 | 0.051 | 0.011 | 0.156 | 0.164 | 1.466 | 1.432 | 1.429 | 1.469 | +0.003 | +0.001 | +0.003 |
| e | 4602 | 0.357 | 0.185 | 0.149 | 0.091 | 0.095 | 0.030 | 0.481 | 0.490 | 2.990 | 2.700 | 2.684 | 3.035 | +0.048 | +0.027 | +0.045 |

## Per-position view, policy gates: throughput-tuned policy gates (min agreement with the MTP chain + confidence cut, fitted on the train fold)

Every position as a cycle start; run = accepted drafts 0..7 of MTP chain + tail (chain-only run when no tail is issued). `P(R0|R1)` = share of R1-issued cycles where R0 also issues; `extra` = mean of max(0, run_X - run_R0) per cycle.

| class | cycles | R0 issued | R1 issued | R2 issued | R1&R0 | R1 only | R2 only | union | P(R0\|R1) | E[run] MTP+R0 | +R1 | +R2 | best-of | extra R1 | extra R2 | extra best-of |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 5296 | 0.756 | 0.653 | 0.323 | 0.648 | 0.005 | 0.012 | 0.773 | 0.993 | 5.241 | 5.078 | 3.615 | 5.262 | +0.054 | +0.024 | +0.021 |
| b | 3538 | 0.743 | 0.592 | 0.517 | 0.591 | 0.001 | 0.013 | 0.757 | 0.999 | 4.848 | 4.586 | 4.198 | 4.855 | +0.158 | +0.138 | +0.008 |
| c | 2126 | 0.882 | 0.636 | 0.708 | 0.636 | 0.000 | 0.010 | 0.893 | 0.999 | 5.636 | 4.809 | 5.110 | 5.639 | +0.048 | +0.061 | +0.003 |
| d | 4438 | 0.094 | 0.024 | 0.041 | 0.024 | 0.000 | 0.017 | 0.111 | 1.000 | 1.466 | 1.438 | 1.441 | 1.468 | +0.000 | +0.003 | +0.002 |
| e | 4602 | 0.357 | 0.173 | 0.212 | 0.173 | 0.000 | 0.028 | 0.385 | 1.000 | 2.990 | 2.814 | 2.806 | 2.998 | +0.020 | +0.025 | +0.008 |

## Information bound: oracle best realised run (chain + tail) over ALL candidate sources of a family, vs K11's R0 run (held-in, hindsight selection: an upper bound, not implementable)

`ptr` = every (layer, head, top-4 rank) pointer of the producing row with mass >= 0.03, anchor-consistent at its fitted offset (<= 1536 candidates/row); `knn` = the 8 best anchor-masked neighbours of h65raw; `n=1` = every earlier occurrence of the anchor token (any exact-copy source); extra = mean of max(0, run_oracle - run_R0) per cycle, `+modelled` = modelled ratio over MTP + R0 when the oracle's tail replaces R0's wherever it is longer (tail <= 4).

| class | R0 run | extra ptr | extra knn | extra ptr+knn | extra n=1 (any copy) | modelled ptr / R0 | ptr+knn / R0 | n=1 / R0 |
|---|---|---|---|---|---|---|---|---|
| a | 5.241 | +0.172 | +0.134 | +0.175 | +0.184 | x1.018 [1.009, 1.028] | x1.019 | x1.020 |
| b | 4.848 | +0.500 | +0.413 | +0.505 | +0.525 | x1.061 [1.036, 1.085] | x1.059 | x1.064 |
| c | 5.636 | +0.339 | +0.259 | +0.347 | +0.368 | x1.063 [1.049, 1.083] | x1.063 | x1.066 |
| d | 1.466 | +0.020 | +0.020 | +0.022 | +0.024 | x1.005 [1.003, 1.007] | x1.006 | x1.006 |
| e | 2.990 | +0.217 | +0.209 | +0.226 | +0.234 | x1.029 [1.014, 1.047] | x1.031 | x1.031 |
| a-c | | | | | | x1.040 [1.028, 1.051] | x1.040 [1.028, 1.050] | x1.043 [1.030, 1.053] |

## Where the MTP chain is wrong (rm < 3): can a confident pointer / neighbour supply the first missed slot or more?  (held-out gates; replacement view: pointer slots 1..7 instead of the chain)

| class | P(rm<3) | R1 issued among them | R1 right at slot rm+1 \| issued | share of rm<3 cycles where R1 fixes slot rm+1 | R1 run>rm share | R2 fixes share | R0-tail fixes (always 0: tail only after a full chain) |
|---|---|---|---|---|---|---|---|
| a | 0.125 | 140 | 0.386 | 0.081 | 0.081 | 0.012 | 0 |
| b | 0.124 | 96 | 0.427 | 0.094 | 0.087 | 0.140 | 0 |
| c | 0.031 | 20 | 0.250 | 0.077 | 0.077 | 0.046 | 0 |
| d | 0.748 | 184 | 0.076 | 0.004 | 0.003 | 0.002 | 0 |
| e | 0.267 | 165 | 0.200 | 0.027 | 0.019 | 0.022 | 0 |

## R1 catches where R0 issues nothing (calibrated precision >= 0.9 gate; `bl` = backward agreement of the source with the current context incl. the anchor)

| class | R1 issued | R1 issued & R0 absent | share of all cycles | first-draft precision there | mean run (slots 1..7) | with bl < 3 | precision of those | mean run of those |
|---|---|---|---|---|---|---|---|---|
| a | 2913 | 291 | 0.0549 | 0.732 | 2.19 | 62 | 0.645 | 1.13 |
| b | 893 | 179 | 0.0506 | 0.687 | 1.99 | 57 | 0.614 | 1.53 |
| c | 767 | 72 | 0.0339 | 0.792 | 1.31 | 7 | 0.857 | 1.43 |
| d | 269 | 225 | 0.0507 | 0.453 | 0.64 | 137 | 0.423 | 0.61 |
| e | 853 | 435 | 0.0945 | 0.614 | 0.94 | 170 | 0.665 | 0.99 |

