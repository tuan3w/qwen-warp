"""T=1 sampling exactness of the deep cycle with a confidence gate / forced truncation / forced tail, end to end in the real CUDA graph.

  ~/llm-lab/venv/bin/python test_sampling_gate.py [--n 4000] [--out results/test_sampling_gate.log]      (lease held)
From ONE snapshot (prompt prefilled, first token sampled and fixed) N independent cycles per config (state restored, fresh drafts and philox counters every sample); the first two emitted tokens
are compared across configs by chi-square homogeneity against config A (K = 5, full width, no gate = the K7-validated path): the emitted stream must not depend on how many drafts are verified.
  A  k5 full width            B  k5 gate (random-looking tau: widths 0..5)       C  k5 fixed width 2
  D  k5 gate + forced tail [t1..t4] (delta proposals after nch chain drafts; widths up to 7)        E  k7 full width
  X  BROKEN variant (test_flags bit0: rejection resamples from p, not the residual) with the gate on: must be REJECTED (power check)
"""
import argparse
import math
import sys
from collections import Counter

import numpy as np

from deep import *  # noqa: F401,F403
import test_pld_engine as T  # noqa: E402

OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def chi2_two_sample(c1, c2, minexp=5.0):
    """homogeneity chi-square of two count dicts, cells with expected < minexp pooled into 'other'; returns (chi2, dof, p)"""
    keys = sorted(set(c1) | set(c2))
    n1, n2 = sum(c1.values()), sum(c2.values())
    cells = []
    oth = [0, 0]
    for k in keys:
        a, b = c1.get(k, 0), c2.get(k, 0)
        if (a + b) * min(n1, n2) / (n1 + n2) < minexp:
            oth[0] += a
            oth[1] += b
        else:
            cells.append((a, b))
    if oth[0] + oth[1] > 0:
        cells.append(tuple(oth))
    chi = 0.0
    for a, b in cells:
        t = a + b
        e1, e2 = t * n1 / (n1 + n2), t * n2 / (n1 + n2)
        if e1 > 0:
            chi += (a - e1) ** 2 / e1 + (b - e2) ** 2 / e2
    dof = max(len(cells) - 1, 1)
    # survival function of chi2 via the regularised upper incomplete gamma (Wilson-Hilferty is enough for a gate at p ~ 0.01)
    z = ((chi / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return chi, dof, p


def collect(e, fp, ids, n, opts, tail_cfg, flags, seed):
    e.reset()
    e.set_force(None)
    e.set_sampling(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, seed=seed, test_flags=flags | 2)  # bit1: keep the engine's argmax first token = identical anchor in every config
    first = fp.prefill_spec(ids, 3, start=False)
    e.set_tail(tail_cfg)
    e.snapshot()
    ctx = list(ids) + [first]
    y0, y01, widths = Counter(), Counter(), Counter()
    outn = 0
    for i in range(n):
        e.restore()
        e.deep_begin(ctx, opts.kc)
        e._emitted = outn
        new, ss = e.cycles_deep(1, opts)
        outn = ss[12]
        a = ss[0]
        y0[new[0]] += 1
        if a >= 2:
            y01[(new[0], new[1])] += 1
        widths[a] += 1
    li, lf = e.deep_log()
    return y0, y01, widths, li


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--out", default="results/test_sampling_gate.log")
    a = ap.parse_args()
    e = DeepEngine(ctx_max=4096, verbose=False)
    fp = FastPrefill(e)
    e.set_sampling(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, seed=1)
    e.prewarm(5, 7, 7)
    prompts = T.e0_prompts(["prose", "code"])
    forced = make_pld(min_len=3, max_tail=4, force_tail=[464, 2068, 1109, 262])
    tau_b = [0.85, 0.7, 0.6, 0.8, 0.5, 0.5, 0.5]
    cfgs = [
        ("A k5 full", make_opts(5, use_q=1), None, 0),
        ("B k5 gate", make_opts(5, gate=1, use_q=1, tau=tau_b), None, 0),
        ("C k5 width 2", make_opts(5, fixed_kv=2), None, 0),
        ("D k5 gate+forced tail", make_opts(5, gate=1, use_q=1, tail=1, tau=tau_b), forced, 0),
        ("E k7 full", make_opts(7, use_q=1), None, 0),
        ("X BROKEN gate", make_opts(5, gate=1, use_q=1, tau=tau_b), None, 1),
    ]
    bad = 0
    for pname, ids in prompts.items():
        res = {}
        for tag, opts, tl, flags in cfgs:
            res[tag] = collect(e, fp, ids, a.n, opts, tl, flags, seed=11)
            y0, y01, widths, li = res[tag]
            w = li[:, 1] + li[:, 2]
            log(f"[{pname}] {tag}: n={a.n}, mean emitted {sum(k * v for k, v in widths.items()) / a.n:.3f}, verify-width hist {np.bincount(w, minlength=8).tolist()}, support y0 {len(y0)}, y01 {len(y01)}")
        A = res["A k5 full"]
        for tag, _, _, flags in cfgs[1:]:
            c1, d1, p1 = chi2_two_sample(A[0], res[tag][0])
            c2, d2, p2 = chi2_two_sample(A[1], res[tag][1])
            ok = (min(p1, p2) > 0.005) if flags == 0 else (min(p1, p2) < 1e-4)
            log(f"[{pname}] {tag} vs A: first token chi2 {c1:.1f}/{d1} p={p1:.3g}; first two tokens chi2 {c2:.1f}/{d2} p={p2:.3g} -> {'OK' if ok else 'FAIL'}{' (broken variant must be rejected)' if flags else ''}")
            bad += (not ok)
    log(f"RESULT {'ALL OK' if bad == 0 else str(bad) + ' FAIL'}")
    open(a.out, "w").write("\n".join(OUT) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
