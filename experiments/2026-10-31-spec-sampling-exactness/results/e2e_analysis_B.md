### e2e-analyze
e2e B: prompt prose, anchor 760, cfg {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'draft_sample': 0}, spec samples 6000 (1 files), plain reference samples 20000
- tokens outside the target support: first 0, second 0
- first emitted token vs analytic p1 (top-4): chi2 5.4 df 3 p = 0.143, TV 0.0066; vs plain sampling (two-sample): chi2 1.2 df 3 p = 0.75
- numerics floor: TV between the T=1 engine step and the verify-row step distributions: first token 0.0240, joint 0.0396; spec first token vs the T=1-step analytic p1: chi2 33.7 df 3 p = 2.33e-07
- reference sampler (device philox draws) vs its analytic p1: chi2 3.4 df 3 p = 0.337
- joint (first, second) vs analytic p1 x p2(.|first) (12 cells): chi2 14.0 df 11 p = 0.23, TV 0.0124 (plain sampler vs analytic TV 0.0114); vs plain sampling two-sample: chi2 10.9 df 11 p = 0.448
- => PASS (threshold p > 1e-3 for every statistic, no stray tokens)
ALL OK: True
