"""Exactness of speculative sampling WITH request logit processors (penalties, logit_bias, min_tokens, allowed_token_ids, min_p).  Needs a GPU (sm_86) and the built engine, not the checkpoint.

  python tests/test_spec_procs.py [--trials 400000]

A verify cycle checks T = 1 + k rows in one pass; row t must be the processed distribution given the history INCLUDING the draft tokens d1..dt before it (penalty counts, presence and repetition
sets, the min_tokens boundary all move inside the cycle).  The engine hook smk_test_trials_procs runs the cycle's device code (candidate processing with the trial's own drafts, top-k / top-p / min_p, the
speculative accept / residual / bonus kernels) on a small vocabulary.  The emitted stream is then completed to m tokens by sampling the reference conditionals (a trial that emits a < m tokens
continues autoregressively) and the joint distribution of the m tokens is compared with the exact autoregressive joint of the processed model (chi-square, cells with expected count < 5 pooled).
Negative control: the variant that processes every row without the draft tokens MUST fail the same test.  Reference: tests/test_logit_procs.py ref_rows (vLLM v1 sampler semantics).
"""
import argparse
import ctypes as C
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_logit_procs import LIB, check, FAILS, ref_dist, ref_rows  # noqa: E402
from spec import make_procs, make_sampling  # noqa: E402
import smk  # noqa: E402


def chi2_sf(x, dof):
    """Survival function of the chi-square distribution: the regularised upper incomplete gamma Q(dof / 2, x / 2) (Numerical Recipes gammq)."""
    a, x = dof / 2.0, x / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:  # series for P
        term = total = 1.0 / a
        n = a
        for _ in range(100000):
            n += 1
            term *= x / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1.0 - a  # Lentz continued fraction for Q
    c, d = 1e300, 1.0 / b
    h = d
    for i in range(1, 100000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def run_trials(tl, dl, nchain, tail, cfg, procs_kw, gen, ntrials, ignore, ctr0=0):
    N = tl.shape[1]
    k = nchain + len(tail)
    kw = dict(procs_kw)
    p, keep = make_procs(prompt=kw.get("prompt", ()), presence=kw.get("pres", 0.0), frequency=kw.get("freq", 0.0), repetition=kw.get("rep", 1.0), min_tokens=kw.get("min_tokens", 0),
                         bias=kw.get("bias"), allowed=kw.get("allowed"), stop=kw.get("stop", ()))
    y = np.zeros((ntrials, 8), dtype=np.int32)
    a = np.zeros(ntrials, dtype=np.int32)
    g = (C.c_int * max(1, len(gen)))(*gen)
    tt = (C.c_int * max(1, len(tail)))(*tail)
    tlc, dlc = tl.contiguous(), dl.contiguous()
    smk.check(LIB.smk_test_trials_procs(C.c_void_p(tlc.data_ptr()), N, C.c_void_p(dlc.data_ptr()), nchain, tt, len(tail), C.byref(cfg), C.byref(p), g, len(gen), int(ignore), ntrials, ctr0,
                                        y.ctypes.data_as(C.POINTER(C.c_int)), a.ctypes.data_as(C.POINTER(C.c_int))), "test_trials_procs")
    assert keep and k <= 7
    return y, a


def reference_joint(tl, m, procs_kw, gen, T, top_k, top_p, min_p):
    """Exact autoregressive distribution of the first m tokens: row t = the processed distribution given gen + the tokens before it.  Returns (cond(h) -> probs [N], joint array [N] * m)."""
    N = tl.shape[1]
    cache = {}

    def cond(h):
        if h not in cache:
            row = ref_rows(tl[len(h) : len(h) + 1], gen=list(gen) + list(h), **procs_kw)
            cache[h] = ref_dist(row[0], T, top_k, top_p, min_p).numpy()
        return cache[h]

    joint = np.zeros((N,) * m)
    for h in itertools.product(range(N), repeat=m):
        pr = 1.0
        for t in range(m):
            pr *= cond(h[:t])[h[t]]
            if pr == 0.0:
                break
        joint[h] = pr
    return cond, joint


def complete(y, a, m, cond, N, rng):
    """Extend every trial's emitted tokens to m tokens by sampling the reference conditionals (a trial emits a <= k + 1 tokens)."""
    out = np.where(np.arange(m)[None, :] < a[:, None], y[:, :m], -1).astype(np.int64)
    for col in range(m):
        groups = {}
        for i in np.nonzero(out[:, col] < 0)[0]:
            groups.setdefault(tuple(out[i, :col]), []).append(i)
        for h, ii in groups.items():
            p = cond(h).astype(np.float64)
            out[ii, col] = rng.choice(N, size=len(ii), p=p / p.sum())
    return out


def chi_square(counts, probs, n):
    exp = probs * n
    small = exp < 5.0
    obs_p = np.concatenate([counts[~small], [counts[small].sum()]])
    exp_p = np.concatenate([exp[~small], [exp[small].sum()]])
    keep = exp_p > 0
    stat = float(((obs_p[keep] - exp_p[keep]) ** 2 / exp_p[keep]).sum())
    dof = int(keep.sum()) - 1
    return stat, dof, chi2_sf(stat, dof)


def scenario(name, N, nchain, tail, m, trials, cfg_kw, procs_kw, gen, seed, draft_noise=0.8):
    g = torch.Generator().manual_seed(seed)
    k = nchain + len(tail)
    tl = (torch.randn(k + 1, N, generator=g) * 1.5).contiguous()
    dl = (tl[:nchain] + torch.randn(nchain, N, generator=g) * draft_noise).contiguous()  # a related but different draft model: partial acceptance
    cfg = make_sampling(**cfg_kw)
    cond, joint = reference_joint(tl, m, procs_kw, gen, cfg_kw["temperature"], cfg_kw["top_k"], cfg_kw["top_p"], cfg_kw.get("min_p", 0.0))
    probs = joint.reshape(-1)
    assert abs(probs.sum() - 1.0) < 1e-6
    res = {}
    for ignore in (0, 1):
        y, a = run_trials(tl, dl, nchain, tail, cfg, procs_kw, gen, trials, ignore)
        rng = np.random.default_rng(seed + 1000 + ignore)
        out = complete(y, a, m, cond, N, rng)
        flat = np.ravel_multi_index(out.T, (N,) * m)
        counts = np.bincount(flat, minlength=N**m)
        stat, dof, pv = chi_square(counts, probs, trials)
        tv = 0.5 * float(np.abs(counts / trials - probs).sum())
        res[ignore] = (stat, dof, pv, tv, float(a.mean()), (a == 1).mean())
        print(f"    {'broken (drafts ignored)' if ignore else 'engine'}: chi2 {stat:.1f} dof {dof} p {pv:.3g}  TV {tv:.4f}  mean emitted/cycle {a.mean():.2f}", flush=True)
    s, b = res[0], res[1]
    check(f"{name}: engine joint of {m} tokens matches the autoregressive reference (chi-square p > 0.001)", s[2] > 0.001, f"{trials} trials, chi2 {s[0]:.1f}, dof {s[1]}, p {s[2]:.3g}, TV {s[3]:.4f}")
    check(f"{name}: negative control (draft tokens ignored by the later rows) FAILS the same test", b[2] < 1e-6, f"p {b[2]:.3g}, TV {b[3]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=400000)
    args = ap.parse_args()
    # test of the test: chi-square survival function against known values
    ok = abs(chi2_sf(10.0, 5) - 0.0752352) < 1e-6 and abs(chi2_sf(3.84146, 1) - 0.05) < 1e-5 and abs(chi2_sf(101.879, 80) - 0.05) < 1e-4
    check("chi-square survival function matches known values", ok, f"{chi2_sf(10.0, 5):.6f} {chi2_sf(3.84146, 1):.6f} {chi2_sf(101.879, 80):.4f}")
    n = args.trials
    pen = dict(prompt=[1, 2, 5], pres=1.2, freq=0.9, rep=1.3)
    gen = [3, 3, 4]
    scenario("k=3 drafts, penalties (presence 1.2, frequency 0.9, repetition 1.3), top-k 8", 8, 3, [], 4, n,
             dict(temperature=1.0, top_k=8, top_p=1.0), pen, gen, seed=11)
    mix = dict(prompt=[0, 6], pres=0.5, freq=0.5, rep=1.2, min_tokens=len(gen) + 2, stop=[7], bias={2: 1.5, 5: -2.0}, allowed=[0, 1, 2, 3, 4, 5, 7])
    scenario("k=3 drafts, penalties + bias + allowed set + min_tokens boundary inside the cycle, T=0.8 top-p 0.9 min_p 0.05", 8, 3, [], 4, n,
             dict(temperature=0.8, top_k=8, top_p=0.9, min_p=0.05), mix, gen, seed=12)
    scenario("k=4 (2 chain drafts + 2 prompt-lookup tail tokens), penalties", 6, 2, [3, 3], 5, n,
             dict(temperature=1.0, top_k=6, top_p=1.0), dict(prompt=[1, 2], pres=1.5, freq=1.0, rep=1.4), [3, 4], seed=13)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
