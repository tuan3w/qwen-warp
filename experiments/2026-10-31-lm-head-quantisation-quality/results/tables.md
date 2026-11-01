Positions: 15280 evaluated (docs by category: {'code': 14, 'prose': 9, 'math': 10, 'greedy': 10}); CIs = document-cluster bootstrap (2000 resamples).

### Table 1. Pooled over all evaluated positions (code + prose + math + greedy chat)

| head arm | MB | weight rel-L2 | top-1 agree % [95% CI] | #flips | KL full (nats) | KL top-20 | top-20 mass moved | dNLL nats [CI] | TV k20/p.95 T=1 | TV k20/p.95 T=0.7 | crit top1 / dNLL / TV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| int8_rtn | 1213 | 0.0120 | 99.19 [98.90, 99.43] | 124 | 7.98e-05 | 7.75e-05 | 2.96e-03 | +0.00012 [-0.00009, +0.00033] | 0.0038 [0.0032, 0.0044] | 0.0041 | FAIL / pass / pass |
| int8_clip | 1213 | 0.0118 | 99.26 [99.01, 99.49] | 113 | 1.03e-04 | 1.00e-04 | 3.32e-03 | +0.00002 [-0.00027, +0.00032] | 0.0043 [0.0036, 0.0049] | 0.0046 | FAIL / pass / pass |
| int6_rtn | 910 | 0.0490 | 97.30 [96.72, 97.87] | 412 | 1.45e-03 | 1.41e-03 | 1.26e-02 | +0.00110 [+0.00031, +0.00191] | 0.0158 [0.0134, 0.0182] | 0.0173 | FAIL / pass / FAIL |
| int8_g128_rtn | 1231 | 0.0069 | 99.50 [99.34, 99.65] | 76 | 4.01e-05 | 3.98e-05 | 2.00e-03 | +0.00004 [-0.00010, +0.00018] | 0.0026 [0.0022, 0.0030] | 0.0028 | pass / pass / pass |
| int4_g128_rtn | 625 | 0.1249 | 93.83 [92.59, 94.99] | 943 | 1.02e-02 | 9.74e-03 | 3.30e-02 | +0.00940 [+0.00669, +0.01231] | 0.0417 [0.0353, 0.0480] | 0.0449 | FAIL / FAIL / FAIL |
| int4_g128_clip | 625 | 0.1069 | 93.18 [91.79, 94.50] | 1042 | 1.42e-02 | 1.38e-02 | 3.77e-02 | +0.01250 [+0.00843, +0.01665] | 0.0480 [0.0409, 0.0551] | 0.0512 | FAIL / FAIL / FAIL |
| int4_g128_awclip | 625 | 0.1113 | 94.74 [93.77, 95.65] | 803 | 6.92e-03 | 6.58e-03 | 2.78e-02 | +0.00734 [+0.00508, +0.00973] | 0.0349 [0.0298, 0.0399] | 0.0375 | FAIL / FAIL / FAIL |
| bf16_logits_rounded (floor: ref logits rounded to bf16) | 2543 | - | 98.28 [97.80, 98.71] | 263 | 1.92e-04 | 1.95e-04 | 5.17e-03 | +0.00052 [+0.00024, +0.00079] | 0.0065 | 0.0071 | - |

(bf16 reference head = 2543 MB; dNLL = NLL_quant - NLL_ref of the true next token, > 0 = worse.  Scale format: int8/int6 fp32 per row, int4 bf16 per 128 columns; see Method.)

### Table 2. Noise floors (same metric code, same hidden states unless stated)

| floor | n | top-1 agree % [CI] | #flips | KL full | dNLL [CI] | TV k20/p.95 T=1 | TV T=0.7 |
|---|---|---|---|---|---|---|---|
| engine_T1_vs_fp32 | 292 | 100.00 [100.00, 100.00] | 0 | 3.13e-10 | +0.00000 [-0.00000, +0.00000] | 0.0000 | 0.0000 |
| engine_T4_vs_fp32 | 292 | 100.00 [100.00, 100.00] | 0 | -6.68e-09 | -0.00000 [-0.00000, +0.00000] | 0.0000 | 0.0000 |
| hidden_T1_vs_T4 | 2938 | 99.52 [99.31, 99.75] | 14 | 1.45e-04 | +0.00028 [-0.00031, +0.00102] | 0.0045 | 0.0046 |

Final-norm hidden rel-L2 T=1 vs T=4 rows per floor doc: 0.0143, 0.1103, 0.0124, 0.0177, 0.0293, 0.0121, 0.0367, 0.0289.

### Table 3. Per category (top-1 agree % / dNLL / TV k20p.95 T=1)

| head arm | code | prose | math | greedy | natural |
|---|---|---|---|---|---|
| int8_rtn | 99.61 / +0.0001 / 0.0025 | 98.90 / -0.0001 / 0.0057 | 98.36 / +0.0005 / 0.0056 | 99.92 / +0.0000 / 0.0016 | 99.03 / +0.0001 / 0.0043 |
| int8_clip | 99.68 / -0.0000 / 0.0029 | 98.81 / +0.0002 / 0.0063 | 98.67 / -0.0000 / 0.0060 | 99.85 / -0.0000 / 0.0017 | 99.14 / +0.0000 / 0.0048 |
| int6_rtn | 98.64 / +0.0012 / 0.0107 | 96.34 / +0.0012 / 0.0239 | 94.93 / +0.0014 / 0.0223 | 99.28 / +0.0004 / 0.0062 | 96.89 / +0.0013 / 0.0178 |
| int8_g128_rtn | 99.68 / +0.0002 / 0.0017 | 99.13 / -0.0001 / 0.0040 | 99.24 / -0.0000 / 0.0038 | 100.00 / -0.0000 / 0.0010 | 99.40 / +0.0001 / 0.0030 |
| int4_g128_rtn | 96.62 / +0.0032 / 0.0284 | 90.46 / +0.0188 / 0.0642 | 89.84 / +0.0141 / 0.0568 | 98.33 / +0.0030 / 0.0174 | 92.89 / +0.0107 / 0.0468 |
| int4_g128_clip | 96.55 / +0.0051 / 0.0337 | 89.90 / +0.0185 / 0.0687 | 88.09 / +0.0221 / 0.0685 | 97.99 / +0.0058 / 0.0199 | 92.18 / +0.0139 / 0.0538 |
| int4_g128_awclip | 97.00 / +0.0022 / 0.0239 | 92.60 / +0.0107 / 0.0520 | 91.20 / +0.0135 / 0.0478 | 98.11 / +0.0046 / 0.0159 | 94.04 / +0.0079 / 0.0388 |
| bf16_logits_rounded | 99.24 / +0.0002 / 0.0047 | 97.51 / +0.0011 / 0.0095 | 96.71 / +0.0007 / 0.0087 | 99.62 / +0.0001 / 0.0033 | 98.00 / +0.0006 / 0.0072 |

### Table 4. Flips and the MTP-verify proxy (aligned 4-row blocks; ideal drafter = reference argmax chain)

| head arm | n flips | median ref top1-top2 logit margin at flips | mean margin | blocks | P(any argmax change in 4 rows) % | P(change in rows 0-2) % | accepted drafts lost per cycle (of 3) |
|---|---|---|---|---|---|---|---|
| int8_rtn | 124 | 0.013 | 0.016 | 3788 | 3.04 | 2.09 | 0.0417 |
| int8_clip | 113 | 0.015 | 0.022 | 3788 | 2.85 | 2.01 | 0.0409 |
| int6_rtn | 412 | 0.048 | 0.069 | 3788 | 9.74 | 7.00 | 0.1468 |
| int8_g128_rtn | 76 | 0.010 | 0.013 | 3788 | 1.90 | 1.48 | 0.0280 |
| int4_g128_rtn | 943 | 0.117 | 0.164 | 3788 | 20.72 | 15.97 | 0.3287 |
| int4_g128_clip | 1042 | 0.129 | 0.192 | 3788 | 22.49 | 17.27 | 0.3588 |
| int4_g128_awclip | 803 | 0.094 | 0.134 | 3788 | 18.08 | 14.15 | 0.2975 |
| bf16_logits_rounded | 263 | 0.028 | 0.034 | 3788 | 6.10 | 4.30 | 0.0890 |
| int8_rtn+rescore4 | 3 | 0.032 | 0.022 | 3788 | 0.08 | 0.05 | 0.0011 |
| int8_rtn+rescore8 | 0 | - | - | 3788 | 0.00 | 0.00 | 0.0000 |
| int8_rtn+rescore16 | 0 | - | - | 3788 | 0.00 | 0.00 | 0.0000 |
| int4_g128_rtn+rescore4 | 93 | 0.033 | 0.047 | 3788 | 2.03 | 1.66 | 0.0351 |
| int4_g128_rtn+rescore8 | 11 | 0.007 | 0.036 | 3788 | 0.26 | 0.18 | 0.0032 |
| int4_g128_rtn+rescore16 | 0 | - | - | 3788 | 0.00 | 0.00 | 0.0000 |
| int4_g128_awclip+rescore4 | 61 | 0.032 | 0.046 | 3788 | 1.29 | 1.03 | 0.0230 |
| int4_g128_awclip+rescore8 | 8 | 0.027 | 0.046 | 3788 | 0.21 | 0.16 | 0.0034 |
| int4_g128_awclip+rescore16 | 0 | - | - | 3788 | 0.00 | 0.00 | 0.0000 |

### Table 1b. Post-hoc (exploratory, added after Table 1): quantised head + exact rescoring of the top-M candidates (their logits replaced by the bf16-weight logits)

| arm | top-1 agree % [CI] | #flips | KL full | dNLL [CI] | TV k20/p.95 T=1 [CI] | TV T=0.7 | crit top1 / dNLL / TV |
|---|---|---|---|---|---|---|---|
| int8_rtn+rescore4 | 99.98 [99.94, 100.00] | 3 | 2.67e-05 | +0.00001 [-0.00016, +0.00016] | 0.0014 [0.0011, 0.0017] | 0.0011 | pass / pass / pass |
| int8_rtn+rescore8 | 100.00 [100.00, 100.00] | 0 | 1.74e-05 | +0.00000 [-0.00011, +0.00012] | 0.0008 [0.0007, 0.0010] | 0.0005 | pass / pass / pass |
| int8_rtn+rescore16 | 100.00 [100.00, 100.00] | 0 | 1.16e-05 | +0.00000 [-0.00009, +0.00009] | 0.0003 [0.0002, 0.0004] | 0.0001 | pass / pass / pass |
| int4_g128_rtn+rescore4 | 99.39 [99.07, 99.67] | 93 | 3.68e-03 | +0.00291 [+0.00120, +0.00479] | 0.0148 [0.0117, 0.0178] | 0.0111 | FAIL / FAIL / FAIL |
| int4_g128_rtn+rescore8 | 99.93 [99.87, 99.98] | 11 | 2.60e-03 | +0.00247 [+0.00121, +0.00381] | 0.0089 [0.0070, 0.0107] | 0.0053 | pass / FAIL / FAIL |
| int4_g128_rtn+rescore16 | 100.00 [100.00, 100.00] | 0 | 1.83e-03 | +0.00180 [+0.00085, +0.00276] | 0.0032 [0.0024, 0.0039] | 0.0012 | pass / pass / pass |
| int4_g128_awclip+rescore4 | 99.60 [99.28, 99.81] | 61 | 2.69e-03 | +0.00404 [+0.00239, +0.00577] | 0.0143 [0.0114, 0.0172] | 0.0110 | pass / FAIL / FAIL |
| int4_g128_awclip+rescore8 | 99.95 [99.89, 99.99] | 8 | 1.87e-03 | +0.00339 [+0.00217, +0.00470] | 0.0086 [0.0067, 0.0104] | 0.0054 | pass / FAIL / FAIL |
| int4_g128_awclip+rescore16 | 100.00 [100.00, 100.00] | 0 | 1.24e-03 | +0.00256 [+0.00164, +0.00352] | 0.0029 [0.0022, 0.0035] | 0.0011 | pass / FAIL / pass |

### Table 7. Why top-1 agreement is brittle: reference top1-top2 logit margin of the position vs argmax flip rate

| margin bin (logits) | positions | share % | flip rate % per arm: int8_rtn / int8_g128_rtn / int6_rtn / int4_g128_awclip / bf16_logits_rounded / int8_rtn+rescore8 / int4_g128_awclip+rescore16 |
|---|---|---|---|
| [0, 0.01) | 98 | 0.64 | 52.04 / 37.76 / 57.14 / 64.29 / 55.10 / 0.00 / 0.00 |
| [0.01, 0.03) | 184 | 1.20 | 30.98 / 17.93 / 47.83 / 52.72 / 45.11 / 0.00 / 0.00 |
| [0.03, 0.1) | 536 | 3.51 | 2.99 / 1.12 / 30.41 / 48.69 / 21.83 / 0.00 / 0.00 |
| [0.1, 0.3) | 1198 | 7.84 | 0.00 / 0.00 / 8.51 / 24.54 / 0.75 / 0.00 / 0.00 |
| [0.3, 1.0) | 2860 | 18.72 | 0.00 / 0.00 / 0.10 / 3.08 / 0.00 / 0.00 / 0.00 |
| [1.0, inf) | 10404 | 68.09 | 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 / 0.00 |

### Table 5. lm_head GEMV, N=248320 K=5120, L2-cold, ms (GB/s)

| kernel | T=1 | T=2 | T=3 | T=4 |
|---|---|---|---|---|
| bf16 smk_bf16_gemv_f32 (shipped) | 2.875 (884) | 2.945 (863) | 2.969 (857) | 2.985 (852) |
| int8 per-row CUDA-core (this) | 1.445 (881) | 1.464 (869) | 1.507 (844) | 1.546 (823) |
| int4 g128 smk_w4_gemv EPI_F32 (def) | 0.778 (842) | 0.785 (835) | 0.798 (822) | 0.824 (795) |
| int4 g128 smk_w4_gemv_cfg 0 | 0.794 (826) | 0.813 (806) | 0.822 (797) | 0.810 (809) |
| int4 g128 smk_w4_gemv_cfg 1 | 0.775 (846) | 0.779 (841) | 0.783 (837) | 0.786 (834) |
| int4 g128 smk_w4_gemv_cfg 2 | 1.030 (636) | 1.054 (622) | 1.065 (616) | 1.074 (610) |
| int4 g128 smk_w4_gemv_cfg 3 | 0.812 (807) | 0.831 (788) | 0.825 (794) | 0.822 (797) |
| int4 g128 smk_w4_gemv_cfg 4 | 0.776 (845) | 0.781 (839) | 0.783 (837) | 0.787 (833) |
| int4 g128 smk_w4_gemv_cfg 5 | 1.276 (514) | 1.284 (511) | 1.292 (507) | 1.297 (505) |

### Table 6. Modelled gain (measured head times above; step = engine/BENCH.md: plain 20.9 ms, T=4 verify 22.5 ms, k=3 cycle 24.2 ms)

| head | T=1 head ms | T=1 step ms | gain ms / % | tok/s plain (47.8 base) | T=4 head ms | T=4 verify ms | gain ms / % | MTP k=3 cycle speedup (draft path unchanged) |
|---|---|---|---|---|---|---|---|---|
| int8 per-row | 1.445 | 19.47 | 1.43 / 6.8% | 51.4 | 1.546 | 21.06 | 1.44 / 6.4% | x1.063 (code 145 -> 154 tok/s) |
| int4 g128 (best cfg per T) | 0.775 | 18.80 | 2.10 / 10.0% | 53.2 | 0.786 | 20.30 | 2.20 / 9.8% | x1.100 (code 145 -> 159 tok/s) |

### Table 8. Net throughput model with exact rescoring (M=16): head ms + measured select+rescore overhead; tokens/cycle lost = Table 4 'accepted drafts lost' (ideal-drafter upper bound)

| head | T=4 head ms + overhead | verify gain ms | workload (tok/s, tokens/cycle at 24.2 ms) | cycle speedup | accepted-length factor (upper-bound loss) | net tok/s |
|---|---|---|---|---|---|---|
| int8 per-row | 1.546 + 0.000 | 1.44 | code (145.0, 3.51) | x1.063 | x0.988 | 152.3 |
| int8 per-row | 1.546 + 0.000 | 1.44 | prose (94.7, 2.29) | x1.063 | x0.982 | 98.9 |
| int8 per-row | 1.546 + 0.000 | 1.44 | math (143.0, 3.46) | x1.063 | x0.988 | 150.2 |
| int8 per-row | 1.546 + 0.000 | 1.44 | 8k (115.0, 2.78) | x1.063 | x0.985 | 120.4 |
| int8 per-row + rescore16 | 1.546 + 0.065 | 1.37 | code (145.0, 3.51) | x1.060 | x1.000 | 153.7 |
| int8 per-row + rescore16 | 1.546 + 0.065 | 1.37 | prose (94.7, 2.29) | x1.060 | x1.000 | 100.4 |
| int8 per-row + rescore16 | 1.546 + 0.065 | 1.37 | math (143.0, 3.46) | x1.060 | x1.000 | 151.6 |
| int8 per-row + rescore16 | 1.546 + 0.065 | 1.37 | 8k (115.0, 2.78) | x1.060 | x1.000 | 121.9 |
| int4 g128 awclip + rescore16 | 0.786 + 0.065 | 2.13 | code (145.0, 3.51) | x1.097 | x1.000 | 159.0 |
| int4 g128 awclip + rescore16 | 0.786 + 0.065 | 2.13 | prose (94.7, 2.29) | x1.097 | x1.000 | 103.9 |
| int4 g128 awclip + rescore16 | 0.786 + 0.065 | 2.13 | math (143.0, 3.46) | x1.097 | x1.000 | 156.8 |
| int4 g128 awclip + rescore16 | 0.786 + 0.065 | 2.13 | 8k (115.0, 2.78) | x1.097 | x1.000 | 126.1 |
| int4 g128 awclip (no rescore) | 0.786 + 0.000 | 2.20 | code (145.0, 3.51) | x1.100 | x0.915 | 146.0 |
| int4 g128 awclip (no rescore) | 0.786 + 0.000 | 2.20 | prose (94.7, 2.29) | x1.100 | x0.870 | 90.6 |
| int4 g128 awclip (no rescore) | 0.786 + 0.000 | 2.20 | math (143.0, 3.46) | x1.100 | x0.914 | 143.8 |
| int4 g128 awclip (no rescore) | 0.786 + 0.000 | 2.20 | 8k (115.0, 2.78) | x1.100 | x0.893 | 113.0 |
