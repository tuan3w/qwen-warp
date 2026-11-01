### e2e-analyze
e2e B_broken: prompt prose, anchor 760, cfg {'temperature': 0.6, 'top_k': 20, 'top_p': 0.95, 'draft_sample': 0}, spec samples 2000 (1 files), plain reference samples 20000
- tokens outside the target support: first 0, second 0
- first emitted token vs analytic p1 (top-4): chi2 60.1 df 3 p = 5.49e-13, TV 0.0249; vs plain sampling (two-sample): chi2 16.9 df 3 p = 0.000738
- numerics floor: TV between the T=1 engine step and the verify-row step distributions: first token 0.0240, joint 0.0396; spec first token vs the T=1-step analytic p1: chi2 71.6 df 4 p = 1.03e-14
- reference sampler (device philox draws) vs its analytic p1: chi2 3.4 df 3 p = 0.337
- joint (first, second) vs analytic p1 x p2(.|first) (12 cells): chi2 306.3 df 12 p = 2.2e-58, TV 0.1056 (plain sampler vs analytic TV 0.0114); vs plain sampling two-sample: chi2 99.6 df 11 p = 2.17e-16
- => FAIL (threshold p > 1e-3 for every statistic, no stray tokens) [expected FAIL: power check]
ALL OK: True
