### philox
- philox4x32-10 ctr ['0x0', '0x0', '0x0', '0x0'] key ['0x0', '0x0']: ['0x6627e8d5', '0xe169c58d', '0xbc57ac4c', '0x9b00dbd8'] == Random123 KAT
- philox4x32-10 ctr ['0xffffffff', '0xffffffff', '0xffffffff', '0xffffffff'] key ['0xffffffff', '0xffffffff']: ['0x408f276d', '0x41c83b0e', '0xa20bc7c6', '0x6d5451fd'] == Random123 KAT
- philox4x32-10 ctr ['0x243f6a88', '0x85a308d3', '0x13198a2e', '0x3707344'] key ['0xa4093822', '0x299f31d0']: ['0xd16cfe09', '0x94fdcceb', '0x5001e420', '0x24126ea1'] == Random123 KAT
### processors
- device top-k/top-p == transformers Temperature->TopK->TopP and vLLM apply_top_k_top_p on 6 rows x 7 settings (T/top_k/top_p incl. T=1 k=20 p=.95, T=.6, k=1, k=64) + restricted-vocab map: same support, max |dp| 2.7e-07 (fp32 vs fp64 softmax): OK
### trials
- **T=1.0 top_k 20 top_p .95, sampled drafts (q = filtered draft head)** (2000000 cycles, k=3, 0.01 s; beta_t = sum min(p,q) = [0.782, 0.8042, 0.9081], E[a] = 2.9819, observed 2.9826)
  - emitted length a: chi2 7.9 df 3 p = 0.0471
  - y0 vs p_0 (top-11 support): chi2 8.5 df 10 p = 0.579, TV 0.0006 (noise floor ~0.0009), outside support 0
  - (y0,y1 | a>=2) vs min(p0,q0) x p1: 81 cells, chi2 103.9 df 81 p = 0.0439, TV 0.0019, stray 0
  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: 693 cells, chi2 671.8 df 681 p = 0.592, stray 0
  - => PASS (threshold p > 1e-3 for every statistic, no stray tokens)
- **T=0.6 top_k 20 top_p 1.0, sampled drafts** (2000000 cycles, k=3, 0.01 s; beta_t = sum min(p,q) = [0.8133, 0.8861, 0.9782], E[a] = 3.2390, observed 3.2385)
  - emitted length a: chi2 1.2 df 3 p = 0.753
  - y0 vs p_0 (top-20 support): chi2 16.0 df 19 p = 0.658, TV 0.0005 (noise floor ~0.0013), outside support 0
  - (y0,y1 | a>=2) vs min(p0,q0) x p1: 360 cells, chi2 178.0 df 165 p = 0.231, TV 0.0011, stray 0
  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: 6840 cells, chi2 448.6 df 414 p = 0.116, stray 0
  - => PASS (threshold p > 1e-3 for every statistic, no stray tokens)
- **T=1.0 top_k 20 top_p .95, greedy drafts (q = delta)** (2000000 cycles, k=3, 0.01 s; beta_t = sum min(p,q) = [0.2458, 0.5168, 0.7346], E[a] = 1.4662, observed 1.4671)
  - emitted length a: chi2 2.4 df 3 p = 0.497
  - y0 vs p_0 (top-11 support): chi2 16.3 df 10 p = 0.0923, TV 0.0009 (noise floor ~0.0009), outside support 0
  - (y0,y1 | a>=2) vs min(p0,q0) x p1: 9 cells, chi2 10.3 df 9 p = 0.327, TV 0.0005, stray 0
  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: 11 cells, chi2 4.2 df 11 p = 0.962, stray 0
  - => PASS (threshold p > 1e-3 for every statistic, no stray tokens)
- **T=1.0 top_k 20 top_p .95, sampled drafts, BROKEN (no residual)** (2000000 cycles, k=3, 0.01 s; beta_t = sum min(p,q) = [0.782, 0.8042, 0.9081], E[a] = 2.9819, observed 2.9826)
  - emitted length a: chi2 5.6 df 3 p = 0.131
  - y0 vs p_0 (top-11 support): chi2 249267.3 df 10 p = 0, TV 0.1302 (noise floor ~0.0009), outside support 0
  - (y0,y1 | a>=2) vs min(p0,q0) x p1: 81 cells, chi2 203809.7 df 81 p = 0, TV 0.1105, stray 0
  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: 693 cells, chi2 79007.1 df 681 p = 0, stray 0
  - => FAIL (threshold p > 1e-3 for every statistic, no stray tokens) [expected FAIL: power check]
- **T=1.0 top_k 20 top_p .95, greedy drafts, BROKEN (no residual)** (2000000 cycles, k=3, 0.01 s; beta_t = sum min(p,q) = [0.2458, 0.5168, 0.7346], E[a] = 1.4662, observed 1.4663)
  - emitted length a: chi2 1.1 df 3 p = 0.786
  - y0 vs p_0 (top-11 support): chi2 371779.6 df 10 p = 0, TV 0.1856 (noise floor ~0.0009), outside support 0
  - (y0,y1 | a>=2) vs min(p0,q0) x p1: 9 cells, chi2 122615.2 df 9 p = 0, TV 0.0614, stray 0
  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: 11 cells, chi2 49612.0 df 11 p = 0, stray 0
  - => FAIL (threshold p > 1e-3 for every statistic, no stray tokens) [expected FAIL: power check]
ALL OK: True
