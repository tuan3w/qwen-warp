"""Exactness / plumbing tests of the deep-chain cycle (private library): greedy == plain T=1 greedy (near-tie rule of tests/test_pld_engine.py), forced partial acceptance, gate / tail / adaptive-depth plumbing.

  ~/llm-lab/venv/bin/python test_deep.py [--n 192] [--out results/test_deep.log]       (lease held)
"""
import argparse
import random
import sys
import time

import numpy as np
import torch

from deep import *  # noqa: F401,F403
import test_pld_engine as T  # noqa: E402

OUT = []


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def gen_deep(e, fp, ids, n, opts, tail=None, force=None, group=2):
    e.reset()
    e.set_force(force)
    e.set_sampling(0.0)
    first = fp.prefill_spec(ids, 3, start=False)
    e.set_tail(make_pld(min_len=3, max_tail=4) if tail else None)
    e.deep_begin(list(ids) + [first], opts.kc)
    toks = [first]
    while len(toks) < n:
        new, ss = e.cycles_deep(group, opts)
        toks += new
    li, lf = e.deep_log()
    return toks[:n], li, lf, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--out", default="results/test_deep.log")
    a = ap.parse_args()
    e = DeepEngine(ctx_max=8192, verbose=False)
    fp = FastPrefill(e)
    e.prewarm(1, 7, 7)
    tasks = {}
    for k, v in T.e0_prompts(["code", "math", "prose"]).items():
        tasks[k] = v
    for tid in ("a2", "c3", "b1"):
        tasks[tid] = T.e6_prompt(tid)
    bad = 0
    rnd = random.Random(7)
    for name, ids in tasks.items():
        ref = T.plain(e, fp, ids, a.n)
        cfgs = [
            ("K3", make_opts(3), False, None),
            ("K5", make_opts(5), False, None),
            ("K7", make_opts(7), False, None),
            ("K4+tail(mode1)", make_opts(4, tail=1), True, None),
            ("K5+tail(mode1)", make_opts(5, tail=1), True, None),
            ("K6+tail(mode2)", make_opts(6, tail=2), True, None),
            ("K7 gate tau=.6,.5,.4,.4,.5,.5,.5", make_opts(7, gate=1, tau=[0.6, 0.5, 0.4, 0.4, 0.5, 0.5, 0.5]), False, None),
            ("K5 gate+tail+adapt", make_opts(5, kc_lo=3, adapt_thr=3, gate=1, tail=1, tau=[0.5, 0.5, 0.5, 0.5, 0.5]), True, None),
            ("K7 random gate", make_opts(7, gate=1, tau=[rnd.random() for _ in range(7)]), False, None),
            ("K6 forced [1,4,2,3,1]", make_opts(6), False, [1, 4, 2, 3, 1]),
            ("K7 forced [8,1,5,2,7,3]", make_opts(7), False, [8, 1, 5, 2, 7, 3]),
            ("K5+tail forced [3,4,1,6,2,8,5,7]", make_opts(5, tail=1), True, [3, 4, 1, 6, 2, 8, 5, 7]),
        ]
        for tag, opts, tl, force in cfgs:
            got, li, lf, ss = gen_deep(e, fp, ids, a.n, opts, tl, force)
            ok, msg = T.compare(e, fp, ids, ref, got, f"{name} {tag}")
            nchs = np.bincount(li[:, 1] + li[:, 2], minlength=8)
            msg += f" | cycles {len(li)} E {sum(li[:, 0]) / max(len(li), 1):.2f} verify-width hist {nchs.tolist()}"
            log(("OK   " if ok else "FAIL ") + msg)
            bad += (not ok)
    # confidence sanity: values in (0, 1], decreasing on average with depth
    got, li, lf, ss = gen_deep(e, fp, tasks["code"], 128, make_opts(7), False)
    log(f"conf raw mean by depth (code, K=7): {np.round(lf[:, :7].mean(0), 3).tolist()}; processed-q (greedy = raw): {np.round(lf[:, 8:15].mean(0), 3).tolist()}; min {lf[:, :7].min():.3f} max {lf[:, :7].max():.3f}")
    # sampling plumbing: runs, counts consistent
    e.reset()
    e.set_sampling(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, seed=5)
    first = fp.prefill_spec(tasks["code"], 3, start=False)
    e.set_tail(make_pld(min_len=3, max_tail=4))
    e.deep_begin(list(tasks["code"]) + [first], 5)
    o = make_opts(5, gate=1, use_q=1, tail=1, tau=[0.5] * 5)
    toks = [first]
    while len(toks) < 192:
        new, ss = e.cycles_deep(2, o)
        toks += new
    li, lf = e.deep_log()
    log(f"sampling K5 gate+tail: tokens {len(toks)}, cycles {len(li)}, E {ss[12] / ss[13]:.2f}, emitted==ring count {ss[12] == len(toks) - 1}, conf_q >= conf_raw {bool((lf[:, 8:13] >= lf[:, :5] - 1e-5).all())}, text head {toks[:6]}")
    log(f"RESULT {'ALL OK' if bad == 0 else str(bad) + ' FAIL'}")
    open(a.out, "w").write("\n".join(OUT) + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
