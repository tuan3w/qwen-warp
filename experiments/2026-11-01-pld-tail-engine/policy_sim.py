"""Offline replay of the hybrid draft policy on E6's saved traces with MEASURED engine costs (replaces E6's modelled V(T>4)).

  python3 experiments/2026-11-01-pld-tail-engine/policy_sim.py --cost results/cost.json [--grid] [--out results/policy_sim.json]

Uses E6's trace (experiments/2026-10-31-ngram-draft-potential/results/trace.npz: prompts, greedy continuations, MTP drafts at every position) and E6's own lookup code
(analyze.lookup) for the proposals, so the proposals are those of the engine's host proposer (tests/test_pld.py proves C++ == analyze.lookup on every position).
cost.json: {"V": {"1": ms, ... "8": ms}, "o_fix": ms (accept + catch-up of 4 rows), "o_cu_row": ms per extra catch-up row, "o_draft": ms per MTP draft, "gap": host round-trip gap ms per cycle}.
Policies: mtp (k=3 chain only, group sync = no gap), tail (chain + tail, gate l >= m, tail <= Lt, optional adaptive back-off).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

E6 = Path(__file__).resolve().parent.parent / "2026-10-31-ngram-draft-potential"
sys.path.insert(0, str(E6))
import analyze as A  # noqa: E402

CLASSES = "abcde"


def prep_tail(task, n=2, strat="longest", cap=64, back=64, L=4):
    """per position: MTP run rm, gate length el, tail length tl, accepted-run table rt[i, Lt] (chain + tail[:Lt] against the real continuation)."""
    g, P, D = task["g"], task["P"], task["D"]
    S = np.concatenate([task["prompt"], g]).astype(np.int64)
    idx = A.build_index(S)
    N = len(g) - 1
    rm = np.zeros(N, np.int8)
    el = np.zeros(N, np.int16)
    tl = np.zeros(N, np.int8)
    rt = np.zeros((N, L + 1), np.int8)
    for i in range(N):
        p = P + i
        rm[i] = A.prefix_match(list(D[i]), g, i)
        K = np.concatenate([S[: p + 1], D[i]])
        l, c = A.lookup(K, idx, n, strat, len(K) - 1, p + 1, L=L, cap=cap, back=back)
        el[i], tl[i] = l, len(c)
        for Lt in range(L + 1):
            rt[i, Lt] = A.prefix_match(list(D[i]) + c[:Lt], g, i)
    return dict(N=N, rm=rm, el=el, tl=tl, rt=rt)


def walk(pt, cost, m=3, Lt=4, wasted=0, skip=0, tail=True, ema=None):
    """-> (tokens, ms, cycles, tail_cycles).  Adaptive back-off: `wasted` consecutive tail cycles whose chain was fully accepted but tail_1 rejected -> skip the tail for `skip` cycles."""
    V, N = cost["V"], pt["N"]
    i, t, cyc, ntail, w, sk = 0, 0.0, 0, 0, 0, 0
    if ema:  # online net-gain estimate of the tail: G = accepted tail tokens - c per tail cycle, EMA with alpha; < 0 -> skip the tail for `skip` eligible cycles, doubling (cap) after each bad re-probe
        alpha, c, s0, smax = ema
        e_, misses = 0.5, 0
    while i < N:
        ln = 0
        if tail and pt["tl"][i] > 0 and pt["el"][i] >= m:
            ln = min(Lt, int(pt["tl"][i]))
            if ln > 0 and sk > 0:
                ln, sk = 0, sk - 1
        T = 4 + ln
        r = int(pt["rt"][i, ln]) if ln else min(int(pt["rm"][i]), 3)
        t += V[T] + cost["o_fix"] + cost["o_cu_row"] * max(0, T - 4) + 3 * cost["o_draft"] + (cost["gap"] if tail else 0.0)
        cyc += 1
        ntail += ln > 0
        if ln and skip:
            w = w + 1 if r == 3 else 0
            if w >= wasted:
                w, sk = 0, skip
        if ema and ln:
            e_ = (1 - alpha) * e_ + alpha * (max(0, r - 3) - c)
            if e_ < 0:
                sk = min(smax, s0 * 2 ** misses)
                misses += 1
                e_ = 0.0
            else:
                misses = 0
        i += min(r + 1, N - i)
    return N, t, cyc, ntail


def ratio(res, base, ids):
    n = sum(res[i][0] for i in ids); t = sum(res[i][1] for i in ids)
    nb = sum(base[i][0] for i in ids); tb = sum(base[i][1] for i in ids)
    return (n / t) / (nb / tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", required=True)
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cost = json.load(open(args.cost))
    cost["V"] = {int(k): v for k, v in cost["V"].items()}
    tasks = A.load()
    cls = defaultdict(list)
    for t in tasks:
        cls[t["cls"]].append(t["id"])
    pts = {t["id"]: prep_tail(t) for t in tasks}
    base = {i: walk(pts[i], cost, tail=False) for i in pts}
    out = {}

    def report(name, **kw):
        res = {i: walk(pts[i], cost, **kw) for i in pts}
        row = {c: ratio(res, base, cls[c]) for c in CLASSES}
        abc = [i for c in "abc" for i in cls[c]]
        row["abc"] = ratio(res, base, abc)
        row["tail_share"] = {c: sum(res[i][3] for i in cls[c]) / sum(res[i][2] for i in cls[c]) for c in CLASSES}
        row["E"] = {c: sum(res[i][0] for i in cls[c]) / sum(res[i][2] for i in cls[c]) for c in CLASSES}
        out[name] = row
        print(f"{name:42s} " + " ".join(f"{c} {row[c]:.3f}" for c in CLASSES) + f"  a-c {row['abc']:.3f}  tail share " + "/".join(f"{row['tail_share'][c]:.2f}" for c in CLASSES))

    print("class MTP-only E:", {c: round(sum(base[i][0] for i in cls[c]) / sum(base[i][2] for i in cls[c]), 2) for c in CLASSES})
    report("E3 cell (m=3, Lt=4)", m=3, Lt=4)
    for al in (0.2, 0.3, 0.5):
        for c in (0.2, 0.3, 0.4, 0.5):
            for s0, smax in ((4, 32), (8, 64), (8, 128)):
                report(f"ema a={al} c={c} skip {s0}/{smax}", ema=(al, c, s0, smax))
    if args.grid:
        for m in (3, 4, 6, 8, 12, 16):
            for Lt in (1, 2, 3, 4):
                for wasted, skip in ((0, 0), (1, 4), (1, 8), (2, 8), (2, 16), (3, 16)):
                    report(f"m={m} Lt={Lt} backoff {wasted}/{skip}", m=m, Lt=Lt, wasted=wasted, skip=skip)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
