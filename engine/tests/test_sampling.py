"""K7 tests: sampling kernels + exact speculative sampling.

  SMK_BUILD=build/k7 ~/llm-lab/venv/bin/python tests/test_sampling.py [philox|processors|trials|e2e-run|e2e-ref|e2e-analyze] ...
Needs the GPU lease.  Sections:
  philox      philox4x32-10 known-answer vectors (Random123)
  processors  device top-k / top-p == transformers' Temperature -> TopK -> TopP warpers (and vLLM's apply_top_k_top_p) on random / peaked logit rows, incl. a restricted vocabulary map
  trials      the device draft-pick + accept/residual/bonus functions (the graph's own code) for 2e6 independent cycles on fixed rows: emitted-length histogram, first token,
              (y0, y1 | a>=2) and (y0, y1, y2 | a>=3) against the analytic speculative-sampling distributions (chi-square, TV); broken variants (no residual) must FAIL
  trials-pld  K11: the same for MTP chain (3) + 4 delta tail proposals (rejections at tail positions, accepted tail tokens, bonus), broken variants must FAIL
  e2e-run     the real graph: fixed prompt, snapshot after the prefill, N x (restore, start, cycle(s) until >= 2 tokens) -> first two emitted tokens (written to results/)
  e2e-ref     plain autoregressive sampling reference from the same snapshot (device sampler on the engine's T=1 logits) + analytic probabilities from the engine logits
  e2e-analyze chi-square / two-sample tests, TV, support check of the e2e samples vs the reference
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ctypes as C  # noqa: E402

from spec import SpecEngine, make_sampling, setup_spec_api, smk  # noqa: E402

V = 248320
RES = Path(os.environ.get("SMK_SAMP_RES") or Path(__file__).resolve().parents[2] / "experiments" / "2026-10-31-spec-sampling-exactness" / "results")  # SMK_SAMP_RES: other output dir (K9 reruns)
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


# ---------------------------------------------------------------------------------------------------------------- statistics
def chi2_sf(x, df):
    if df <= 0:
        return float("nan")
    return float(torch.special.gammaincc(torch.tensor(df / 2.0, dtype=torch.float64), torch.tensor(x / 2.0, dtype=torch.float64)))


def gof(obs, exp, min_exp=5.0):
    """Pearson chi-square of observed counts vs expected counts (same total), cells with expected < min_exp pooled into one."""
    obs, exp = np.asarray(obs, dtype=np.float64), np.asarray(exp, dtype=np.float64)
    big = exp >= min_exp
    o = np.append(obs[big], obs[~big].sum())
    e = np.append(exp[big], exp[~big].sum())
    keep = e > 0
    o, e = o[keep], e[keep]
    stat = float(((o - e) ** 2 / e).sum())
    df = len(e) - 1
    return stat, df, chi2_sf(stat, df)


def two_sample(a, b, min_tot=10.0):
    """Chi-square homogeneity of two count vectors of equal total, cells with a+b < min_tot pooled."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    big = (a + b) >= min_tot
    ao, bo = np.append(a[big], a[~big].sum()), np.append(b[big], b[~big].sum())
    keep = (ao + bo) > 0
    ao, bo = ao[keep], bo[keep]
    stat = float(((ao - bo) ** 2 / (ao + bo)).sum())
    df = len(ao) - 1
    return stat, df, chi2_sf(stat, df)


# ---------------------------------------------------------------------------------------------------------------- reference processors (transformers)
def hf_probs(logits, T, k, p):
    """fp64 probabilities over the vocabulary after transformers' TemperatureLogitsWarper -> TopKLogitsWarper -> TopPLogitsWarper (the vLLM / HF order) + softmax."""
    from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

    s = logits.float().reshape(1, -1).clone()
    ids = torch.zeros(1, 1, dtype=torch.long)
    for w in (TemperatureLogitsWarper(temperature=T), TopKLogitsWarper(top_k=k), TopPLogitsWarper(top_p=p)):
        s = w(ids, s)
    return torch.softmax(s.double(), dim=-1)[0]


def vllm_probs(logits, T, k, p):
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    s = (logits.float().reshape(1, -1) / T).cuda()
    s = apply_top_k_top_p(s, torch.tensor([k], device="cuda"), torch.tensor([p], device="cuda", dtype=torch.float32))
    return torch.softmax(s.double().cpu(), dim=-1)[0]


def dev_dist(lib, logits, cfg, sel=0, idmap=None):
    """device top-k / top-p of host logits [R][N] -> list of (ids, p) per row."""
    logits = logits.float().contiguous()
    if logits.shape[0] > 4:  # the device hook handles <= 4 rows per call (SMK_TMAX)
        return [d for i in range(0, logits.shape[0], 4) for d in dev_dist(lib, logits[i : i + 4], cfg, sel, idmap)]
    R, N = logits.shape
    n = (C.c_int * R)()
    ids = (C.c_int * (R * 64))()
    pp = (C.c_float * (R * 64))()
    im = None if idmap is None else C.c_void_p(idmap.int().contiguous().data_ptr())
    smk.check(lib.smk_test_dist(C.c_void_p(logits.data_ptr()), R, N, im, C.byref(cfg), sel, n, ids, pp), "test_dist")
    out = []
    for r in range(R):
        m = n[r]
        out.append((np.array(ids[r * 64 : r * 64 + m]), np.array(pp[r * 64 : r * 64 + m], dtype=np.float64)))
    return out


# ---------------------------------------------------------------------------------------------------------------- 1 philox
def test_philox(lib):
    kat = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2, (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    ok = True
    for c, k, want in kat:
        out = (C.c_uint * 4)()
        smk.check(lib.smk_test_philox((C.c_uint * 4)(*c), (C.c_uint * 2)(*k), out), "philox")
        got = tuple(out)
        ok &= got == want
        log(f"- philox4x32-10 ctr {[hex(x) for x in c]} key {[hex(x) for x in k]}: {[hex(x) for x in got]} {'==' if got == want else '!='} Random123 KAT")
    return ok


# ---------------------------------------------------------------------------------------------------------------- 2 processors
def test_processors(lib):
    g = torch.Generator().manual_seed(1)
    ok = True
    rows = []
    for sc in (0.7, 1.5, 3.0, 6.0):  # increasingly peaked
        rows.append(torch.randn(V, generator=g) * sc)
    r = torch.randn(V, generator=g) * 2
    r[1234] += 25.0  # one dominant token: top-p keeps 1
    rows.append(r)
    r = torch.randn(V, generator=g) * 0.3
    r[:50] += 3.0  # flat top: top-p cut inside a plateau, top-k cut
    rows.append(r)
    rows = torch.stack(rows)
    vllm_ok = True
    try:
        import vllm  # noqa: F401
    except Exception:
        vllm_ok = False
    worst = 0.0
    for T, k, p in ((1.0, 20, 0.95), (0.6, 20, 0.95), (1.0, 20, 1.0), (1.3, 50, 0.8), (0.8, 64, 0.5), (1.0, 1, 0.95), (2.0, 7, 0.999)):
        cfg = make_sampling(T, k, p)
        dev = dev_dist(lib, rows, cfg, 0)
        for i in range(rows.shape[0]):
            ref = hf_probs(rows[i], T, k, p)
            sup = torch.nonzero(ref > 0).flatten().numpy()
            ids, pr = dev[i]
            same = set(sup.tolist()) == set(ids.tolist())
            err = float(np.abs(ref.numpy()[ids] - pr).max())
            worst = max(worst, err)
            row_ok = same and err < 2e-6 and abs(pr.sum() - 1) < 1e-5 and all(pr[j] >= pr[j + 1] for j in range(len(pr) - 1))
            if vllm_ok:
                rv = vllm_probs(rows[i], T, k, p)
                supv = set(torch.nonzero(rv > 0).flatten().tolist())
                row_ok &= supv == set(ids.tolist()) and float(np.abs(rv.numpy()[ids] - pr).max()) < 2e-6
            if not row_ok:
                log(f"  MISMATCH T={T} k={k} p={p} row {i}: support equal {same}, max|dp| {err:.2e}, n dev {len(ids)} ref {len(sup)}")
            ok &= row_ok
    # restricted vocabulary map (draft head): top-k over the restricted columns, ids mapped to the full vocabulary
    sel = torch.sort(torch.randperm(V, generator=g)[:65536]).values.int()
    cfg = make_sampling(1.0, 20, 0.95)
    dl = rows[:, sel.long()]
    dev = dev_dist(lib, dl, cfg, 1, sel)
    for i in range(rows.shape[0]):
        ref = hf_probs(dl[i], 1.0, 20, 0.95)
        ids, pr = dev[i]
        cols = torch.nonzero(ref > 0).flatten()
        okr = set(sel[cols].tolist()) == set(ids.tolist()) and float(np.abs(ref.numpy()[[int((sel == t).nonzero()) for t in ids]] - pr).max()) < 2e-6
        ok &= okr
    log(f"- device top-k/top-p == transformers Temperature->TopK->TopP{' and vLLM apply_top_k_top_p' if vllm_ok else ''} on {rows.shape[0]} rows x 7 settings "
        f"(T/top_k/top_p incl. T=1 k=20 p=.95, T=.6, k=1, k=64) + restricted-vocab map: same support, max |dp| {worst:.1e} (fp32 vs fp64 softmax): {'OK' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------------------------------------------- 3 trials (kernel level, fixed rows)
def synth_rows(seed, k, nd=65536, sigma=1.0):
    g = torch.Generator().manual_seed(seed)
    S = torch.randperm(V, generator=g)[:40]
    tl = []
    for _ in range(k + 1):
        l = torch.randn(V, generator=g) * 1.5
        l[S] += 8.0 + 2.0 * torch.randn(40, generator=g)
        tl.append(l)
    tl = torch.stack(tl)
    keep = torch.ones(V, dtype=torch.bool)
    keep[S[:4]] = False  # 4 of the 40 likely target tokens are outside the draft vocabulary
    keep[S[4:]] = False
    cand = torch.nonzero(keep).flatten()
    ids = torch.sort(torch.cat([S[4:], cand[torch.randperm(len(cand), generator=g)[: nd - 36]]])).values  # the draft vocabulary: 36 of the 40 likely tokens + random others
    dl = tl[:k][:, ids] + sigma * torch.randn(k, nd, generator=g)
    return tl, dl, ids.int()


def expected_trials(tl, dl, ids, cfg, k):
    """analytic per-row p_t (full vocab, fp64) and q_t (full vocab) for the speculative-sampling model with independent rows."""
    T = cfg.temperature
    dT = cfg.draft_temperature if cfg.draft_temperature > 0 else T
    p = [hf_probs(tl[t], T, cfg.top_k, cfg.top_p).numpy() for t in range(k + 1)]
    q = []
    for t in range(k):
        if cfg.draft_sample:
            qs = hf_probs(dl[t], dT, cfg.draft_top_k or cfg.top_k, cfg.draft_top_p or cfg.top_p).numpy()
        else:
            qs = np.zeros(dl.shape[1])
            qs[int(torch.argmax(dl[t]))] = 1.0
        full = np.zeros(V)
        full[ids.long().numpy()] = qs
        q.append(full)
    return p, q


def test_trials(lib, ntr=2_000_000, k=3):
    allok = True
    configs = [
        ("T=1.0 top_k 20 top_p .95, sampled drafts (q = filtered draft head)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1), 0.3, False),
        ("T=0.6 top_k 20 top_p 1.0, sampled drafts", dict(temperature=0.6, top_k=20, top_p=1.0, draft_sample=1), 0.2, False),
        ("T=1.0 top_k 20 top_p .95, greedy drafts (q = delta)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=0), 0.3, False),
        ("T=1.0 top_k 20 top_p .95, sampled drafts, BROKEN (no residual)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, test_flags=1), 0.3, True),
        ("T=1.0 top_k 20 top_p .95, greedy drafts, BROKEN (no residual)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=0, test_flags=1), 0.3, True),
    ]
    res = []
    for ci, (name, kw, sigma, broken) in enumerate(configs):
        tl, dl, ids = synth_rows(100, k, sigma=sigma)
        cfg = make_sampling(seed=777 + ci, **kw)
        y = (C.c_int * (ntr * 4))()
        a = (C.c_int * ntr)()
        t0 = time.time()
        smk.check(lib.smk_test_trials(C.c_void_p(tl.contiguous().data_ptr()), V, C.c_void_p(dl.contiguous().data_ptr()), dl.shape[1], C.c_void_p(ids.data_ptr()),
                                      C.byref(cfg), k, ntr, 0, y, a), "trials")
        dt = time.time() - t0
        Y = np.frombuffer(y, dtype=np.int32).reshape(ntr, 4)
        A = np.frombuffer(a, dtype=np.int32)
        p, q = expected_trials(tl, dl, ids, cfg, k)
        beta = [float(np.minimum(p[t], q[t]).sum()) for t in range(k)]
        # a histogram
        pa = [1 - beta[0]]
        for m in range(1, k):
            pa.append(np.prod(beta[:m]) * (1 - beta[m]))
        pa.append(np.prod(beta))
        ca = np.bincount(A, minlength=k + 2)[1 : k + 2]
        s_a = gof(ca, np.array(pa) * ntr)
        # y0 marginal (+ support check)
        sup0 = np.nonzero(p[0] > 0)[0]
        c0 = np.array([(Y[:, 0] == t).sum() for t in sup0])
        out0 = ntr - c0.sum()
        s_0 = gof(np.append(c0, out0), np.append(p[0][sup0] * ntr, 0.0))
        tv0 = 0.5 * (np.abs(c0 / ntr - p[0][sup0]).sum() + out0 / ntr)
        # (y0, y1 | a >= 2): expected ntr * min(p0, q0)(u) p1(v)
        m2 = A >= 2
        w0 = np.minimum(p[0], q[0])
        u_s = np.nonzero(w0 > 0)[0]
        v_s = np.nonzero(p[1] > 0)[0]
        key = Y[m2, 0].astype(np.int64) * V + Y[m2, 1]
        uk, cn = np.unique(key, return_counts=True)
        exp_pairs = (w0[u_s][:, None] * p[1][v_s][None, :]) * ntr
        pair_keys = (u_s[:, None].astype(np.int64) * V + v_s[None, :]).ravel()
        obs_pairs = np.zeros(pair_keys.shape[0])
        pos = np.searchsorted(pair_keys, uk)
        pos = np.clip(pos, 0, len(pair_keys) - 1)
        hit = pair_keys[pos] == uk
        obs_pairs[pos[hit]] = cn[hit]
        stray2 = int(cn[~hit].sum())
        rest_obs = ntr - obs_pairs.sum()
        rest_exp = ntr - exp_pairs.sum()
        s_2 = gof(np.append(obs_pairs, rest_obs), np.append(exp_pairs.ravel(), rest_exp))
        tv2 = 0.5 * (np.abs(obs_pairs - exp_pairs.ravel()).sum() + abs(rest_obs - rest_exp)) / ntr
        # (y0, y1, y2 | a >= 3): expected ntr * min(p0,q0)(u) min(p1,q1)(v) p2(w)
        m3 = A >= 3
        w1 = np.minimum(p[1], q[1])
        v3 = np.nonzero(w1 > 0)[0]
        w_s = np.nonzero(p[2] > 0)[0]
        exp3 = (w0[u_s][:, None, None] * w1[v3][None, :, None] * p[2][w_s][None, None, :]) * ntr
        keys3 = ((u_s[:, None, None].astype(np.int64) * V + v3[None, :, None]) * V + w_s[None, None, :]).ravel()
        order = np.argsort(keys3)
        keys3s = keys3[order]
        k3 = (Y[m3, 0].astype(np.int64) * V + Y[m3, 1]) * V + Y[m3, 2]
        uk3, cn3 = np.unique(k3, return_counts=True)
        pos3 = np.clip(np.searchsorted(keys3s, uk3), 0, len(keys3s) - 1)
        hit3 = keys3s[pos3] == uk3
        obs3 = np.zeros(len(keys3s))
        obs3[pos3[hit3]] = cn3[hit3]
        stray3 = int(cn3[~hit3].sum())
        e3 = exp3.ravel()[order]
        s_3 = gof(np.append(obs3, ntr - obs3.sum()), np.append(e3, ntr - e3.sum()))
        passes = min(s_a[2], s_0[2], s_2[2], s_3[2]) > 1e-3 and out0 == 0 and stray2 == 0 and stray3 == 0
        res.append((name, broken, passes, s_a, s_0, s_2, s_3))
        log(f"- **{name}** ({ntr} cycles, k={k}, {dt:.2f} s; beta_t = sum min(p,q) = {[round(b, 4) for b in beta]}, E[a] = {sum((i + 1) * x for i, x in enumerate(pa)):.4f}, "
            f"observed {A.mean():.4f})")
        log(f"  - emitted length a: chi2 {s_a[0]:.1f} df {s_a[1]} p = {s_a[2]:.3g}")
        log(f"  - y0 vs p_0 (top-{len(sup0)} support): chi2 {s_0[0]:.1f} df {s_0[1]} p = {s_0[2]:.3g}, TV {tv0:.4f} (noise floor ~{math.sqrt(len(sup0) / ntr) * 0.4:.4f}), outside support {int(out0)}")
        log(f"  - (y0,y1 | a>=2) vs min(p0,q0) x p1: {len(exp_pairs.ravel())} cells, chi2 {s_2[0]:.1f} df {s_2[1]} p = {s_2[2]:.3g}, TV {tv2:.4f}, stray {stray2}")
        log(f"  - (y0,y1,y2 | a>=3) vs min(p0,q0) x min(p1,q1) x p2: {len(e3)} cells, chi2 {s_3[0]:.1f} df {s_3[1]} p = {s_3[2]:.3g}, stray {stray3}")
        log(f"  - => {'PASS' if passes else 'FAIL'} (threshold p > 1e-3 for every statistic, no stray tokens){' [expected FAIL: power check]' if broken else ''}")
        allok &= (not passes) if broken else passes
    return allok


def test_trials_pld(lib, ntr=2_000_000, ntail=4):
    """K11: MTP chain (3 sampled / delta drafts) + `ntail` prompt-lookup tail tokens (q = delta at fixed tokens) on fixed independent rows, 2e6 cycles through the graph's own device
    functions (draft pick, accept / residual / bonus; the tail proposals are built by the same kernel as in the graph).  With independent rows the accept probability of a chain position
    is beta_t = sum min(p_t, q_t) and of a tail position pi_t = p_t(d_t); checks: emitted-length histogram (k+1 cells), y0 ~ p_0, (y0,y1 | a >= 2), at every tail position t: the tokens emitted
    at a rejection (a = t+1) are exactly p_t restricted to x != d_t (renormalised: the residual of a delta proposal), never d_t; accepted tail tokens (a >= t+2) are d_t; the bonus
    token (a = k+1) ~ p_k.  Broken variants (flags bit0: on rejection resample from p, d_t reappears) must FAIL."""
    nchain, allok = 3, True
    k = nchain + ntail
    configs = [
        ("T=1.0 top_k 20 top_p .95, sampled chain drafts", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1), 0.3, False),
        ("T=0.6 top_k 20 top_p 1.0, sampled chain drafts", dict(temperature=0.6, top_k=20, top_p=1.0, draft_sample=1), 0.2, False),
        ("T=1.0 top_k 20 top_p .95, greedy chain drafts (q = delta)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=0), 0.3, False),
        ("T=1.0 top_k 20 top_p .95, sampled chain drafts, BROKEN (no residual)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, test_flags=1), 0.3, True),
        ("T=1.0 top_k 20 top_p .95, greedy chain drafts, BROKEN (no residual)", dict(temperature=1.0, top_k=20, top_p=0.95, draft_sample=0, test_flags=1), 0.3, True),
    ]
    for ci, (name, kw, sigma, broken) in enumerate(configs):
        tl, dl, ids = synth_rows(100, k, sigma=sigma)
        dl = dl[:nchain].contiguous()
        cfg = make_sampling(seed=1777 + ci, **kw)
        T_, tk, tp = cfg.temperature, cfg.top_k, cfg.top_p
        p = [hf_probs(tl[t], T_, tk, tp).numpy() for t in range(k + 1)]
        _, q = expected_trials(tl[: nchain + 1], dl, ids, cfg, nchain)
        ranks = [0, 1, 0, 2, 1, 0, 3]  # tail token of tail position j = the rank-th most probable token of that row (pi from ~0.6 down to ~0.2)
        tail = [int(np.argsort(-p[nchain + j])[ranks[j]]) for j in range(ntail)]
        pi = [float(p[nchain + j][tail[j]]) for j in range(ntail)]
        y = (C.c_int * (ntr * 8))()
        a = (C.c_int * ntr)()
        tail_arr = (C.c_int * max(1, ntail))(*tail)
        t0 = time.time()
        smk.check(lib.smk_test_trials_pld(C.c_void_p(tl.contiguous().data_ptr()), V, C.c_void_p(dl.data_ptr()), dl.shape[1], C.c_void_p(ids.data_ptr()), C.byref(cfg),
                                          nchain, tail_arr, ntail, ntr, 0, y, a), "trials_pld")
        dt = time.time() - t0
        Y = np.frombuffer(y, dtype=np.int32).reshape(ntr, 8)
        A = np.frombuffer(a, dtype=np.int32)
        beta = [float(np.minimum(p[t], q[t]).sum()) for t in range(nchain)]
        acc = beta + pi
        reach = [1.0]
        for t in range(k):
            reach.append(reach[-1] * acc[t])
        pa = [reach[t] * (1 - acc[t]) for t in range(k)] + [reach[k]]  # P(a = 1 .. k+1)
        ca = np.bincount(A, minlength=k + 2)[1 : k + 2]
        s_a = gof(ca, np.array(pa) * ntr)
        sup0 = np.nonzero(p[0] > 0)[0]
        c0 = np.array([(Y[:, 0] == t).sum() for t in sup0])
        out0 = ntr - c0.sum()
        s_0 = gof(np.append(c0, out0), np.append(p[0][sup0] * ntr, 0.0))
        m2 = A >= 2
        w0 = np.minimum(p[0], q[0])
        u_s, v_s = np.nonzero(w0 > 0)[0], np.nonzero(p[1] > 0)[0]
        key = Y[m2, 0].astype(np.int64) * V + Y[m2, 1]
        uk, cn = np.unique(key, return_counts=True)
        exp_pairs = (w0[u_s][:, None] * p[1][v_s][None, :]) * ntr
        pair_keys = (u_s[:, None].astype(np.int64) * V + v_s[None, :]).ravel()
        pos = np.clip(np.searchsorted(pair_keys, uk), 0, len(pair_keys) - 1)
        hit = pair_keys[pos] == uk
        obs_pairs = np.zeros(pair_keys.shape[0])
        obs_pairs[pos[hit]] = cn[hit]
        stray2 = int(cn[~hit].sum())
        s_2 = gof(np.append(obs_pairs, ntr - obs_pairs.sum()), np.append(exp_pairs.ravel(), ntr - exp_pairs.sum()))
        s_rej, stray_acc, viol_rej, ps = [], 0, 0, []
        for j in range(ntail):
            t = nchain + j
            d = tail[j]
            if j + 1 < ntail or True:  # accepted tail token = d_t whenever the cycle goes on
                cont = A >= t + 2
                stray_acc += int((Y[cont, t] != d).sum())
            rej = A == t + 1  # rejected at tail position t: y_t ~ p_t / (1 - p_t(d_t)) on x != d_t
            yr = Y[rej, t]
            viol_rej += int((yr == d).sum())
            sup = np.nonzero((p[t] > 0) & (np.arange(V) != d))[0]
            nr = int(rej.sum())
            cr = np.array([(yr == x).sum() for x in sup])
            e = p[t][sup] / p[t][sup].sum() * nr
            s = gof(np.append(cr, nr - cr.sum()), np.append(e, 0.0)) if nr > 200 else (0.0, 0, 1.0)
            if s[1] < 1:  # a single residual outcome (top-p left only d_t and one other token): nothing to test beyond "d_t never emitted"
                s = (0.0, 0, 1.0)
            s_rej.append((t, nr, s))
            ps.append(s[2])
        bon = A == k + 1
        yb = Y[bon, k]
        supb = np.nonzero(p[k] > 0)[0]
        nb = int(bon.sum())
        cb = np.array([(yb == x).sum() for x in supb])
        s_b = gof(np.append(cb, nb - cb.sum()), np.append(p[k][supb] / p[k][supb].sum() * nb, 0.0)) if nb > 200 else (0.0, 0, 1.0)
        passes = (min([s_a[2], s_0[2], s_2[2], s_b[2]] + ps) > 1e-3 and out0 == 0 and stray2 == 0 and stray_acc == 0 and viol_rej == 0)
        log(f"- **{name}** ({ntr} cycles, chain 3 + tail {ntail} (tokens {tail}, pi_t = {[round(x, 3) for x in pi]}), {dt:.2f} s; beta_t = {[round(b, 4) for b in beta]}, E[a] = {sum((i + 1) * x for i, x in enumerate(pa)):.4f}, observed {A.mean():.4f})")
        log(f"  - emitted length a (k+1 = {k + 1} cells): chi2 {s_a[0]:.1f} df {s_a[1]} p = {s_a[2]:.3g}; y0 vs p_0: p = {s_0[2]:.3g}; (y0,y1 | a>=2): {len(exp_pairs.ravel())} cells p = {s_2[2]:.3g} stray {stray2}")
        log("  - tail rejections y_t ~ p_t on x != d_t: " + "; ".join(f"t={t}: n={nr} p = {s[2]:.3g}" for t, nr, s in s_rej) + f"; d_t emitted at a rejection: {viol_rej}; accepted tail token != d_t: {stray_acc}; bonus y_k ~ p_k (n={nb}): p = {s_b[2]:.3g}")
        log(f"  - => {'PASS' if passes else 'FAIL'} (threshold p > 1e-3 for every statistic, no stray tokens){' [expected FAIL: power check]' if broken else ''}")
        allok &= (not passes) if broken else passes
    return allok


# ---------------------------------------------------------------------------------------------------------------- e2e on the real engine
def prompt_ids(name):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from bench_spec import prompts

    return prompts([name])[name]


def extend_prompt(e, ids, extra):
    """prompt + `extra` greedily generated tokens (plain engine), so the tested position sits inside a generated text (richer distribution than the first token)."""
    if not extra:
        return ids
    e.reset()
    first = e.prefill(ids)
    return list(ids) + [first] + e.decode(extra - 1)


def e2e_setup(e, ids, k, cfg_kw, flags):
    e.reset()
    e.set_sampling(seed=4242, test_flags=2 | flags, **cfg_kw)  # bit1: the first token stays the engine's argmax (fixed anchor)
    anchor = e.prefill_spec(ids, k, start=False)
    e.snapshot()
    return anchor


def e2e_run(a):
    ids = None
    kw = dict(temperature=a.T, top_k=a.top_k, top_p=a.top_p, draft_sample=a.draft_sample)
    tag = a.tag
    e = SpecEngine(ctx_max=1200)
    ids = extend_prompt(e, prompt_ids(a.prompt), a.extra)
    flags = 1 if a.broken else 0
    anchor = e2e_setup(e, ids, a.k, kw, flags)
    e.set_sampling(seed=a.seed, test_flags=2 | flags, **kw)
    lib, k = e.lib, a.k
    samples = []
    t0 = time.time()
    outn0 = None
    cyc = 0
    ss = e._ss
    for i in range(a.n):
        e.restore()
        smk.check(lib.smk_spec_start(e.sp, k, e.st), "start")
        toks = []
        while len(toks) < 2:
            smk.check(lib.smk_spec_run(e.sp, k, 1, e.st, ss, e._ring), "run")
            outn = ss[12]
            if outn0 is None:
                outn0 = outn - ss[0]
            new = [e._ring[j % 4096] for j in range(outn0, outn)]
            outn0 = outn
            toks += new
            cyc += 1
        samples.append(toks[:2])
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{a.n} samples, {time.time() - t0:.0f} s, cycles/sample {cyc / (i + 1):.3f}")
    RES.mkdir(exist_ok=True)
    out = RES / f"e2e_{tag}_seed{a.seed}.json"
    json.dump(dict(tag=tag, prompt=a.prompt, extra=a.extra, k=k, cfg=kw, broken=a.broken, anchor=anchor, seed=a.seed, n=a.n, cycles_per_sample=cyc / a.n, samples=samples), open(out, "w"))
    log(f"wrote {out}")
    return True


def e2e_ref(a):
    """Plain autoregressive sampling reference from the same snapshot.  The main reference uses the SAME numerics as the verify rows (a T = 1+k row engine step, rows 0 / 1
    of [anchor, t1, ...]; T-row vs T=1 kernels differ by ~0.05 logits = a few % of probability at T=0.6), the secondary one the T=1 engine step (diagnostic: how large that
    numerical difference is)."""
    kw = dict(temperature=a.T, top_k=a.top_k, top_p=a.top_p, draft_sample=a.draft_sample)
    e = SpecEngine(ctx_max=1200)
    ids = extend_prompt(e, prompt_ids(a.prompt), a.extra)
    anchor = e2e_setup(e, ids, a.k, kw, 0)
    R = 1 + a.k

    def batch_logits(prefix, row):
        e.restore()
        e._step(prefix + [prefix[-1]] * (R - len(prefix)), 1)  # T = 1+k rows, rows beyond the prefix are padding (causal: rows <= row do not see them)
        return e.logits(row + 1)[row]

    def t1_logits(prefix):
        e.restore()
        for t in prefix:
            e._step([t], 1)
        return e.logits(1)[0]

    p1 = hf_probs(batch_logits([anchor], 0), a.T, a.top_k, a.top_p)
    batch_logits([anchor], 0)
    draws1 = e.test_draw(a.n, ctr0=10_000_000, row=0).numpy()
    sup1 = torch.nonzero(p1 > 0).flatten().tolist()
    p1_t1 = hf_probs(t1_logits([anchor]), a.T, a.top_k, a.top_p)
    p2, p2_t1, draws2 = {}, {}, {}
    vals, cnts = np.unique(draws1, return_counts=True)
    for t1 in sup1:
        v = hf_probs(batch_logits([anchor, int(t1)], 1), a.T, a.top_k, a.top_p)
        p2[int(t1)] = v
        p2_t1[int(t1)] = hf_probs(t1_logits([anchor, int(t1)]), a.T, a.top_k, a.top_p)
        n1 = int(cnts[vals == t1][0]) if t1 in vals else 0
        if n1:
            batch_logits([anchor, int(t1)], 1)
            draws2[int(t1)] = e.test_draw(n1, ctr0=20_000_000 + (1000 * int(t1)) % 7919, row=1).numpy().tolist()
    RES.mkdir(exist_ok=True)
    out = RES / f"e2e_ref_{a.tag}.json"
    sparse = lambda v: {int(t): float(v[t]) for t in torch.nonzero(v > 0).flatten().tolist()}
    json.dump(dict(tag=a.tag, prompt=a.prompt, extra=a.extra, anchor=anchor, cfg=kw, n=a.n, k=a.k,
                   p1={int(t): float(p1[t]) for t in sup1}, p2={t1: sparse(v) for t1, v in p2.items()},
                   p1_T1=sparse(p1_t1), p2_T1={t1: sparse(v) for t1, v in p2_t1.items()},
                   plain=[[int(t1), int(t2)] for t1, d in draws2.items() for t2 in d]), open(out, "w"))
    log(f"wrote {out}  (top-1 token mass p1 {float(p1.max()):.3f}, support {len(sup1)}; TV(T=1 step, T={R}-row step) of p1 = {0.5 * float((p1 - p1_t1).abs().sum()):.4f})")
    return True


def e2e_analyze(a):
    ref = json.load(open(RES / f"e2e_ref_{a.ref or a.tag}.json"))
    files = sorted(RES.glob(f"e2e_{a.tag}_seed*.json"))
    S = [tuple(s) for f in files for s in json.load(open(f))["samples"]]
    n = len(S)
    p1 = {int(k): v for k, v in ref["p1"].items()}
    p2 = {int(k): {int(t): v for t, v in d.items()} for k, d in ref["p2"].items()}
    plain = [tuple(x) for x in ref["plain"]]
    nr = len(plain)
    log(f"e2e {a.tag}: prompt {ref['prompt']}, anchor {ref['anchor']}, cfg {ref['cfg']}, spec samples {n} ({len(files)} files), plain reference samples {nr}")
    stray1 = sum(1 for s in S if s[0] not in p1)
    stray2 = sum(1 for s in S if s[0] in p2 and s[1] not in p2[s[0]])
    log(f"- tokens outside the target support: first {stray1}, second {stray2}")
    # first token
    sup = sorted(p1)
    obs = np.array([sum(1 for s in S if s[0] == t) for t in sup], dtype=float)
    exp = np.array([p1[t] * n for t in sup])
    s1 = gof(np.append(obs, n - obs.sum()), np.append(exp, n - exp.sum()))
    pr = np.array([sum(1 for s in plain if s[0] == t) for t in sup], dtype=float) * n / nr
    s1b = two_sample(obs, pr)
    tv1 = 0.5 * (np.abs(obs / n - exp / n).sum() + abs(n - obs.sum()) / n)
    log(f"- first emitted token vs analytic p1 (top-{len(sup)}): chi2 {s1[0]:.1f} df {s1[1]} p = {s1[2]:.3g}, TV {tv1:.4f}; vs plain sampling (two-sample): chi2 {s1b[0]:.1f} df {s1b[1]} p = {s1b[2]:.3g}")
    # numerical difference between the T=1 engine step and the T=1+k row verify step (the "plain" reference of the previous version): how much of a bias it would explain
    if "p1_T1" in ref:
        q1 = {int(t): v for t, v in ref["p1_T1"].items()}
        q2 = {int(k_): {int(t): v for t, v in d.items()} for k_, d in ref["p2_T1"].items()}
        tv_a = 0.5 * sum(abs(p1.get(t, 0.0) - q1.get(t, 0.0)) for t in set(p1) | set(q1))
        tv_b = 0.5 * sum(abs(p1[t1] * p2[t1].get(t2, 0.0) - q1.get(t1, 0.0) * q2.get(t1, {}).get(t2, 0.0)) for t1 in p2 for t2 in set(p2[t1]) | set(q2.get(t1, {})))
        exq = np.array([q1.get(t, 0.0) * n for t in sup])
        s1q = gof(np.append(obs, n - obs.sum()), np.append(exq, n - exq.sum()))
        log(f"- numerics floor: TV between the T=1 engine step and the verify-row step distributions: first token {tv_a:.4f}, joint {tv_b:.4f}; spec first token vs the T=1-step "
            f"analytic p1: chi2 {s1q[0]:.1f} df {s1q[1]} p = {s1q[2]:.3g}")
    # the plain reference sampler itself vs its analytic probabilities
    pl_first = np.array([sum(1 for s in plain if s[0] == t) for t in sup], dtype=float)
    s1p = gof(np.append(pl_first, nr - pl_first.sum()), np.append([p1[t] * nr for t in sup], nr - sum(p1[t] * nr for t in sup)))
    log(f"- reference sampler (device philox draws) vs its analytic p1: chi2 {s1p[0]:.1f} df {s1p[1]} p = {s1p[2]:.3g}")
    cells = [(t1, t2) for t1 in sorted(p2) for t2 in sorted(p2[t1])]
    idx = {c: i for i, c in enumerate(cells)}
    o = np.zeros(len(cells))
    r = np.zeros(len(cells))
    for s in S:
        if s in idx:
            o[idx[s]] += 1
    for s in plain:
        if s in idx:
            r[idx[s]] += 1
    ex = np.array([p1[t1] * p2[t1][t2] * n for t1, t2 in cells])
    s2 = gof(np.append(o, n - o.sum()), np.append(ex, n - ex.sum()))
    tv2 = 0.5 * (np.abs(o - ex).sum() + abs(n - o.sum() - (n - ex.sum()))) / n
    rr = r * n / nr
    s2b = two_sample(o, rr)
    # noise floor of the TV: the plain device sampler against the same analytic joint
    tvp = 0.5 * (np.abs(r * n / nr - ex).sum()) / n
    log(f"- joint (first, second) vs analytic p1 x p2(.|first) ({len(cells)} cells): chi2 {s2[0]:.1f} df {s2[1]} p = {s2[2]:.3g}, TV {tv2:.4f} (plain sampler vs analytic TV {tvp:.4f}); "
        f"vs plain sampling two-sample: chi2 {s2b[0]:.1f} df {s2b[1]} p = {s2b[2]:.3g}")
    ok = min(s1[2], s1b[2], s2[2], s2b[2]) > 1e-3 and stray1 == 0 and stray2 == 0
    log(f"- => {'PASS' if ok else 'FAIL'} (threshold p > 1e-3 for every statistic, no stray tokens){' [expected FAIL: power check]' if a.expect_fail else ''}")
    return (not ok) if a.expect_fail else ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="philox,processors,trials")
    ap.add_argument("--ntr", type=int, default=2_000_000)
    ap.add_argument("--prompt", default="prose")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--draft-sample", type=int, default=1)
    ap.add_argument("--broken", action="store_true")
    ap.add_argument("--extra", type=int, default=0, help="append this many greedy tokens to the prompt before the tested position")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="prose_k3")
    ap.add_argument("--ref", default=None, help="reference tag for e2e-analyze (default: --tag)")
    ap.add_argument("--expect-fail", action="store_true", help="e2e-analyze of a deliberately broken run: the test must FAIL")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lib = smk.load("libswiftmk.so")
    setup_spec_api(lib)
    ok = True
    for w in a.what.split(","):
        log(f"### {w}")
        if w == "philox":
            ok &= test_philox(lib)
        elif w == "processors":
            ok &= test_processors(lib)
        elif w == "trials":
            ok &= test_trials(lib, a.ntr, a.k)
        elif w == "trials-pld":
            ok &= test_trials_pld(lib, a.ntr)
        elif w == "e2e-run":
            ok &= e2e_run(a)
        elif w == "e2e-ref":
            ok &= e2e_ref(a)
        elif w == "e2e-analyze":
            ok &= e2e_analyze(a)
    log(f"ALL OK: {ok}")
    if a.out:
        Path(a.out).write_text("\n".join(LOG) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
