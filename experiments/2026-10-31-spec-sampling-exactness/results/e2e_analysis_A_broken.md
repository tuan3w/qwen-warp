### e2e-analyze
e2e A_broken: prompt prose, anchor 279, cfg {'temperature': 1.0, 'top_k': 20, 'top_p': 0.95, 'draft_sample': 1}, spec samples 4000 (1 files), plain reference samples 20000
- tokens outside the target support: first 0, second 0
- first emitted token vs analytic p1 (top-16): chi2 536.5 df 15 p = 1.05e-104, TV 0.1353; vs plain sampling (two-sample): chi2 220.6 df 15 p = 1.31e-38
- numerics floor: TV between the T=1 engine step and the verify-row step distributions: first token 0.0071, joint 0.0106; spec first token vs the T=1-step analytic p1: chi2 498.7 df 15 p = 1.09e-96
- reference sampler (device philox draws) vs its analytic p1: chi2 20.9 df 15 p = 0.139
- joint (first, second) vs analytic p1 x p2(.|first) (101 cells): chi2 778.0 df 55 p = 2.52e-128, TV 0.1584 (plain sampler vs analytic TV 0.0203); vs plain sampling two-sample: chi2 319.2 df 52 p = 4.46e-40
- => FAIL (threshold p > 1e-3 for every statistic, no stray tokens) [expected FAIL: power check]
ALL OK: True
