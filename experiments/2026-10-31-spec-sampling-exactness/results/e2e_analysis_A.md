### e2e-analyze
e2e A: prompt prose, anchor 279, cfg {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95, 'draft_sample': 1}, spec samples 20000 (4 files), plain reference samples 20000
- tokens outside the target support: first 0, second 0
- first emitted token vs analytic p1 (top-16): chi2 20.2 df 15 p = 0.164, TV 0.0096; vs plain sampling (two-sample): chi2 21.5 df 15 p = 0.123
- numerics floor: TV between the T=1 engine step and the verify-row step distributions: first token 0.0071, joint 0.0106; spec first token vs the T=1-step analytic p1: chi2 29.7 df 15 p = 0.0131
- reference sampler (device philox draws) vs its analytic p1: chi2 20.9 df 15 p = 0.139
- joint (first, second) vs analytic p1 x p2(.|first) (101 cells): chi2 82.2 df 81 p = 0.441, TV 0.0187 (plain sampler vs analytic TV 0.0203); vs plain sampling two-sample: chi2 91.9 df 81 p = 0.192
- => PASS (threshold p > 1e-3 for every statistic, no stray tokens)
ALL OK: True
