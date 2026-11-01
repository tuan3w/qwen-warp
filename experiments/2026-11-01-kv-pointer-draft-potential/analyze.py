"""kv-pointer-draft-potential (E7) offline analysis: R0 exact n-gram tail vs R1 attention pointer vs R2 hidden-state kNN on E6's 32 greedy tasks.

  ~/llm-lab/venv/bin/python experiments/2026-11-01-kv-pointer-draft-potential/analyze.py [--no-examples]
  -> results/summary.json, results/tables.md, results/examples.json

Frame (everything implementable at drafting time).  Cycle i has the anchor g_i = S[P+i] (E6 trace position, S = prompt + greedy continuation).  The row that produced the anchor is r = P+i-1
(input S[r], output S[r+1] = g_i).  Its attention pointers q (top-4 key positions per layer / q head) and its hidden states are what a drafter can read for free when the cycle starts.
A *source* s is the position that holds the copy of the anchor (token[s] == g_i, anchor-consistent, s <= r); draft slot k (k = 1..7, T = 1 + k rows) is token[s + k] (known tokens: the
prompt+generated prefix up to the anchor, then the MTP drafts d1..d3; unknown -> no draft).  Slots 1..3 are the MTP chain's, 4..7 the tail.  For head pointers s = q + o (offset o in -2..2 fixed per head on
the training tasks); for kNN neighbour j of the row's hidden state s = j + 1.  R0 = K11's host proposer (policy_sim.prep_tail: n=2 longest suffix match of the chain end, agreement >= 3, tail <= 4).
Everything tuned (head subset, offsets, thresholds, gates) is fitted on one half of the tasks and evaluated on the other (2-fold, stratified by class, alternating tasks).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
E6 = ROOT / "experiments" / "2026-10-31-ngram-draft-potential"
K11 = ROOT / "experiments" / "2026-11-01-pld-tail-engine"
sys.path.insert(0, str(E6))
sys.path.insert(0, str(K11))
import analyze as A  # noqa: E402
import policy_sim as PS  # noqa: E402

CLASSES = "abcde"
OFFS = (-2, -1, 0, 1, 2)
MMIN = 0.03
KMAX = 7
ORDER = ["h65raw", "h65", "h65z", "h65n", "h60", "h60z", "h60n", "h48", "h48z", "h48n", "h32", "h32z", "h32n", "h16", "k15", "k15z", "k11", "k11z", "k7", "k7z"]
RNG = np.random.default_rng(7)


# ------------------------------------------------------------------------------------------------ data
def load():
    tasks = A.load()
    for t in tasks:
        t["g"] = np.asarray(t["g"]).astype(np.int64)
        t["S"] = np.concatenate([np.asarray(t["prompt"]).astype(np.int64), t["g"]])
        t["N"] = len(t["g"]) - 1
        t["D"] = np.asarray(t["D"]).astype(np.int64)
        z = np.load(Path(os.environ.get("E7_PROBE", HERE / "runs" / "probe")) / f"{t['id']}.npz")
        t["z"] = {k: z[k] for k in z.files}
        i = np.arange(t["N"])
        t["anchor"] = t["g"][:t["N"]]
        t["r"] = t["P"] - 1 + i
        k = np.arange(1, KMAX + 1)[None, :]
        j = i[:, None] + k
        t["truth"] = np.where(j <= t["N"], t["g"][np.minimum(j, t["N"])], -2)
        t["rm"] = prefix_run(t["D"], t["truth"][:, :3])
    # folds: stratified, alternating within class
    for c in CLASSES:
        for n, t in enumerate([t for t in tasks if t["cls"] == c]):
            t["fold"] = n % 2
    return tasks


def prefix_run(tok, truth):
    return np.cumprod(tok == truth, axis=1).sum(1).astype(np.int8)


def slot_tokens(t, s, kmax=KMAX, rows=None, k0=1):
    """token[s+k] for k=k0..kmax per cycle (-1: invalid source or token not yet known).  rows: cycle index of every entry of s (default arange(N))."""
    P, N, S, D = t["P"], t["N"], t["S"], t["D"]
    i = np.arange(N) if rows is None else rows
    x = s[:, None] + np.arange(k0, kmax + 1)[None, :]
    base = (P + i)[:, None]
    out = np.full(x.shape, -1, np.int64)
    ok = s[:, None] >= 0
    inS = ok & (x <= base)
    out[inS] = S[x[inS]]
    inD = ok & (x > base) & (x <= base + 3)
    rr = np.broadcast_to(i[:, None], x.shape)
    out[inD] = D[rr[inD], (x - base - 1)[inD]]
    return out


def backward_agree(t, s, cap=64):
    """number of tokens that agree going backwards from (s, anchor position P+i): token[s-l] == token[P+i-l]."""
    P, N, S = t["P"], t["N"], t["S"]
    i = np.arange(N)
    cur = P + i
    bl = np.zeros(N, np.int32)
    alive = s >= 0
    for l in range(0, cap):
        a = s - l
        b = cur - l
        alive &= (a >= 0) & (b >= 0)
        eq = np.zeros(N, bool)
        eq[alive] = S[a[alive]] == S[b[alive]]
        alive &= eq
        bl += alive
    return bl


# ------------------------------------------------------------------------------------------------ R1: attention pointers
def head_counts(t):
    """per (offset, layer, head): rows where the first non-sink pointer is anchor-consistent, and where in addition the first draft is right; plus rows."""
    idx = t["z"]["idx"].astype(np.int32)  # [N,16,24,4]
    nz = idx > 0
    first = np.argmax(nz, axis=-1)
    q = np.take_along_axis(idx, first[..., None], -1)[..., 0]
    has = nz.any(-1)
    S, r, a = t["S"], t["r"][:, None, None], t["anchor"][:, None, None]
    nxt = t["truth"][:, 0][:, None, None]
    cons = np.zeros((len(OFFS), 16, 24), np.int64)
    d1 = np.zeros_like(cons)
    for oi, o in enumerate(OFFS):
        s = q + o
        valid = has & (s >= 0) & (s <= r)
        hit = valid & (S[np.clip(s, 0, len(S) - 1)] == a)
        cons[oi] = hit.sum(0)
        nx = S[np.clip(s + 1, 0, len(S) - 1)]
        d1[oi] = (hit & (nx == nxt)).sum(0)
    return cons, d1, t["N"]


def fit_heads(train, M):
    """offsets per head (max anchor-consistent hits on the training tasks) and the M best heads by (consistent hits with a right first draft)."""
    cons = sum(head_counts_cache(t)[0] for t in train)
    d1 = sum(head_counts_cache(t)[1] for t in train)
    ostar = np.argmax(cons, axis=0)  # [16,24] index into OFFS
    sc = np.take_along_axis(d1, ostar[None], 0)[0]
    order = np.argsort(-sc.reshape(-1), kind="stable")
    hm = np.zeros(16 * 24, bool)
    hm[order[:M]] = True
    return np.array(OFFS)[ostar], hm.reshape(16, 24)


_hc = {}


def head_counts_cache(t):
    if t["id"] not in _hc:
        _hc[t["id"]] = head_counts(t)
    return _hc[t["id"]]


def pointer_vote(t, ostar, hm):
    """-> (src [N] int, conf [N] float, nvotes [N]) anchor-consistent mass-weighted vote of the selected heads' top-4 pointers; src = -1 if none."""
    idx = t["z"]["idx"].astype(np.int64)
    m = t["z"]["mass"][..., :4].astype(np.float32)
    S, N = t["S"], t["N"]
    s = idx + ostar[None, :, :, None]
    r = t["r"][:, None, None, None]
    ok = (idx > 0) & (s >= 0) & (s <= r) & (m >= MMIN) & hm[None, :, :, None]
    ok &= S[np.clip(s, 0, len(S) - 1)] == t["anchor"][:, None, None, None]
    rows, _, _, _ = np.nonzero(ok)
    keys = rows.astype(np.int64) * 65536 + s[ok]
    w = m[ok]
    src = np.full(N, -1, np.int64)
    conf = np.zeros(N, np.float32)
    nv = np.zeros(N, np.int32)
    if len(keys) == 0:
        return src, conf, nv
    uk, inv = np.unique(keys, return_inverse=True)
    W = np.bincount(inv, weights=w)
    C = np.bincount(inv)
    ur = uk // 65536
    us = uk % 65536
    order = np.lexsort((-W, ur))
    ur_o = ur[order]
    first = np.r_[True, ur_o[1:] != ur_o[:-1]]
    sel = order[first]
    src[ur[sel]] = us[sel]
    conf[ur[sel]] = W[sel]
    nv[ur[sel]] = C[sel]
    return src, conf, nv


def pointer_next_token(t, ostar, hm):
    """pointer as a next-token predictor (no anchor filter): token[q+o] voted by mass -> (pred [N], conf [N])."""
    idx = t["z"]["idx"].astype(np.int64)
    m = t["z"]["mass"][..., :4].astype(np.float32)
    S, N = t["S"], t["N"]
    s = idx + ostar[None, :, :, None]
    r = t["r"][:, None, None, None]
    ok = (idx > 0) & (s >= 0) & (s <= r) & (m >= MMIN) & hm[None, :, :, None]
    rows, _, _, _ = np.nonzero(ok)
    tok = S[s[ok]]
    keys = rows.astype(np.int64) * 262144 + tok
    uk, inv = np.unique(keys, return_inverse=True)
    W = np.bincount(inv, weights=m[ok])
    ur, ut = uk // 262144, uk % 262144
    order = np.lexsort((-W, ur))
    first = np.r_[True, ur[order][1:] != ur[order][:-1]]
    sel = order[first]
    pred = np.full(N, -1, np.int64)
    conf = np.zeros(N, np.float32)
    pred[ur[sel]] = ut[sel]
    conf[ur[sel]] = W[sel]
    return pred, conf


# ------------------------------------------------------------------------------------------------ R2: hidden-state kNN
def knn_src(t, rep):
    z = t["z"]
    j = z[f"{rep}/msk_idx"][:, 0].astype(np.int64)
    sim = z[f"{rep}/msk_sim"][:, 0].astype(np.float32)
    ok = sim > -1.5
    return np.where(ok, j + 1, -1), np.where(ok, sim, -9.0)


def knn_next_token(t, rep):
    z = t["z"]
    j = z[f"{rep}/unc_idx"][:, 0].astype(np.int64)
    return t["S"][np.clip(j + 1, 0, len(t["S"]) - 1)], z[f"{rep}/unc_sim"][:, 0].astype(np.float32)


# ------------------------------------------------------------------------------------------------ generic: confidence calibration, policies
def calibrate(conf, ok, target=0.9, min_n=40):
    """threshold tau (rows with conf >= tau are issued) = the loosest cut whose precision stays >= target on the fitting rows; None if no cut with >= min_n rows qualifies."""
    valid = conf > -5
    c, o = conf[valid], ok[valid].astype(float)
    if len(c) < min_n:
        return None
    order = np.argsort(-c, kind="stable")
    cs = np.cumsum(o[order]) / np.arange(1, len(c) + 1)
    good = np.nonzero(cs >= target)[0]
    good = good[good + 1 >= min_n]
    if len(good) == 0:
        return None
    k = good[-1]
    # do not cut inside a tie
    tau = c[order][k]
    return tau


def tail_table(t, src, gate):
    """pt dict for policy_sim.walk: tail = slots 4..7 of the source, issued where gate; rt[i, Lt] = accepted run of chain + tail[:Lt]."""
    N = t["N"]
    tok = slot_tokens(t, np.where(gate, src, -1))
    tail = tok[:, 3:7]
    known = np.cumprod(tail >= 0, axis=1).sum(1)
    rm = t["rm"]
    tr = t["truth"][:, 3:7]
    rt = np.zeros((N, 5), np.int8)
    for L in range(5):
        if L == 0:
            rt[:, 0] = np.minimum(rm, 3)
        else:
            run = np.cumprod(tail[:, :L] == tr[:, :L], axis=1).sum(1)
            rt[:, L] = np.where(rm < 3, rm, 3 + run)
    tl = np.where(gate, known, 0).astype(np.int8)
    return dict(N=N, rm=rm.astype(np.int8), el=np.where(gate, 99, 0).astype(np.int16), tl=tl, rt=rt)


def r0_table(t):
    return PS.prep_tail(dict(g=t["g"], P=t["P"], D=t["D"], prompt=t["prompt"]))


def cost_for(t, costs):
    ctx = t["P"] + t["N"] / 2
    xs = sorted(costs)
    out = {}
    for T in range(1, 9):
        out[T] = float(np.interp(ctx, xs, [costs[x]["V"][T] for x in xs]))
    return dict(V=out, o_fix=0.0, o_cu_row=0.0, o_draft=0.0, gap=0.01)


def load_costs():
    cs = {}
    for c in (64, 2048, 8192):
        d = json.load(open(K11 / "results" / f"cost_{c}.json"))
        cs[c] = dict(V={int(k): v for k, v in d["V"].items()})
    return cs


def walk_all(tasks, pts, costs, **kw):
    res = {}
    for t in tasks:
        res[t["id"]] = PS.walk(pts[t["id"]], cost_for(t, costs), m=1, Lt=4, **kw)
    return res


def ratio_ci(res, base, tasks, cls, nb=2000):
    ids = [t["id"] for t in tasks if t["cls"] in cls]
    def r(sel):
        n = sum(res[i][0] for i in sel); tt = sum(res[i][1] for i in sel)
        nb_ = sum(base[i][0] for i in sel); tb = sum(base[i][1] for i in sel)
        return (n / tt) / (nb_ / tb)
    point = r(ids)
    by = defaultdict(list)
    for t in tasks:
        if t["cls"] in cls:
            by[t["cls"]].append(t["id"])
    bs = []
    for _ in range(nb):
        sel = []
        for c, v in by.items():
            sel += list(RNG.choice(v, len(v)))
        bs.append(r(sel))
    return point, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tokps(res, tasks, cls):
    ids = [t["id"] for t in tasks if t["cls"] in cls]
    return 1000 * sum(res[i][0] for i in ids) / sum(res[i][1] for i in ids), sum(res[i][0] for i in ids) / sum(res[i][2] for i in ids)


# ------------------------------------------------------------------------------------------------ main analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--quick", action="store_true", help="fewer gate grid cells / bootstrap draws (debug)")
    args = ap.parse_args()
    tasks = load()
    REPRS = [r for r in ORDER if f"{r}/msk_idx" in tasks[0]["z"]]
    costs = load_costs()
    byid = {t["id"]: t for t in tasks}
    out = {"n_tasks": len(tasks), "classes": {c: [t["id"] for t in tasks if t["cls"] == c] for c in CLASSES}}
    L = []  # tables.md lines

    def P(s=""):
        L.append(s)

    folds = {f: ([t for t in tasks if t["fold"] != f], [t for t in tasks if t["fold"] == f]) for f in (0, 1)}  # (train, test)
    pos_cls = {c: sum(t["N"] for t in tasks if t["cls"] == c) for c in CLASSES}
    out["positions"] = pos_cls

    # ---------------------------------------------------------------- 0. R0 baseline tables, MTP-only
    pt0 = {t["id"]: r0_table(t) for t in tasks}
    for t in tasks:
        assert (pt0[t["id"]]["rm"] == t["rm"]).all(), t["id"]
    base = walk_all(tasks, pt0, costs, tail=False)
    r0 = walk_all(tasks, pt0, costs, tail=True, ) if False else {t["id"]: PS.walk(pt0[t["id"]], cost_for(t, costs), m=3, Lt=4, tail=True) for t in tasks}
    out["mtp_only"] = {c: dict(zip(("tok_s", "E"), tokps(base, tasks, c))) for c in CLASSES}
    out["mtp_r0"] = {c: dict(zip(("tok_s", "E"), tokps(r0, tasks, c))) for c in CLASSES}
    out["mtp_r0"]["ratio_over_mtp"] = {c: ratio_ci(r0, base, tasks, c) for c in list(CLASSES) + ["abc"]}
    r0_gate = {t["id"]: (pt0[t["id"]]["tl"] > 0) & (pt0[t["id"]]["el"] >= 3) for t in tasks}
    r0_run = {t["id"]: np.where(r0_gate[t["id"]], pt0[t["id"]]["rt"][:, 4], pt0[t["id"]]["rm"].clip(0, 3)) for t in tasks}

    # ---------------------------------------------------------------- 1. R1: per-head table (in-sample, descriptive)
    cons = sum(head_counts_cache(t)[0] for t in tasks)
    d1 = sum(head_counts_cache(t)[1] for t in tasks)
    nrow = sum(t["N"] for t in tasks)
    ostar = np.argmax(cons, 0)
    cons_b = np.take_along_axis(cons, ostar[None], 0)[0]
    d1_b = np.take_along_axis(d1, ostar[None], 0)[0]
    P("## R1 per-head table (in-sample over all 32 tasks, descriptive; first non-sink pointer of the row that produced the anchor)\n")
    P(f"rows (cycle starts) = {nrow}; `cons` = P(token[q+o] == anchor), `d1|cons` = P(token[q+o+1] == next token | consistent), best offset o per head.\n")
    P("| rank | layer(attn idx) | head | o | cons rate | d1 given cons | cons rows | cons&d1 rows |")
    P("|---|---|---|---|---|---|---|---|")
    sc = d1_b.reshape(-1)
    for rk, f in enumerate(np.argsort(-sc, kind="stable")[:20]):
        l, h = divmod(int(f), 24)
        P(f"| {rk + 1} | {l} (layer {4 * l + 3}) | {h} | {OFFS[ostar[l, h]]:+d} | {cons_b[l, h] / nrow:.3f} | {d1_b[l, h] / max(cons_b[l, h], 1):.3f} | {cons_b[l, h]} | {d1_b[l, h]} |")
    P()
    off_hist = {int(o): int((np.array(OFFS)[ostar] == o).sum()) for o in OFFS}
    heads_good = (d1_b / np.maximum(cons_b, 1) > 0.8) & (cons_b / nrow > 0.05)
    out["heads"] = dict(offset_hist_all=off_hist, offset_hist_good=({int(o): int((np.array(OFFS)[ostar][heads_good] == o).sum()) for o in OFFS}), n_good=int(heads_good.sum()),
                        per_layer_good=[int(heads_good[l].sum()) for l in range(16)],
                        per_layer_best_cons=[float(cons_b[l].max() / nrow) for l in range(16)],
                        cons_rate_by_offset_best=[float(np.sort((cons[oi] / nrow).reshape(-1))[-1]) for oi in range(len(OFFS))],
                        top=[dict(layer=int(f // 24), head=int(f % 24), o=int(OFFS[ostar[f // 24, f % 24]]), cons=float(cons_b[f // 24, f % 24] / nrow), d1_given_cons=float(d1_b[f // 24, f % 24] / max(cons_b[f // 24, f % 24], 1))) for f in np.argsort(-sc, kind="stable")[:20]])
    P("Layer profile (attention layer idx, model layer 4l+3): number of heads with cons rate > 5% and d1|cons > 0.8, best cons rate.\n")
    P("| attn idx | " + " | ".join(str(l) for l in range(16)) + " |")
    P("|---|" + "---|" * 16)
    P("| good heads | " + " | ".join(str(int(heads_good[l].sum())) for l in range(16)) + " |")
    P("| best cons | " + " | ".join(f"{cons_b[l].max() / nrow:.2f}" for l in range(16)) + " |")
    P()
    P(f"Offsets of the good heads (token[q+o] is the anchor): {out['heads']['offset_hist_good']}; of all heads: {off_hist}.\n")

    # ---------------------------------------------------------------- 2. R1 ensembles with CV calibration
    Ms = [1, 4, 16, 64, 384]
    r1_cv = {}
    for M in Ms:
        rec = {}
        for f, (tr, te) in folds.items():
            ost, hm = fit_heads(tr, M)
            for t in tr + te:
                src, conf, nv = pointer_vote(t, ost, hm)
                rec.setdefault(f, {})[t["id"]] = (src, conf, nv)
        r1_cv[M] = rec

    def cv_eval(get, name, store):
        """get(task, fold) -> (src, conf).  Calibrate tau per fold on the training tasks (precision of the first draft >= 0.9), evaluate on the held-out fold."""
        res = {c: dict(n=0, issued=0, d1_ok=0, run_sum=0.0, run_ge3=0) for c in CLASSES}
        taus = {}
        gate_by_task = {}
        for f, (tr, te) in folds.items():
            conf_tr, ok_tr = [], []
            for t in tr:
                src, conf = get(t, f)
                tok = slot_tokens(t, src)
                conf_tr.append(np.where(src >= 0, conf, -9.0))
                ok_tr.append(tok[:, 0] == t["truth"][:, 0])
            tau = calibrate(np.concatenate(conf_tr), np.concatenate(ok_tr))
            taus[f] = None if tau is None else float(tau)
            for t in te:
                src, conf = get(t, f)
                gate = (src >= 0) & (conf >= tau) if tau is not None else np.zeros(t["N"], bool)
                gate_by_task[t["id"]] = (src, gate)
                tok = slot_tokens(t, np.where(gate, src, -1))
                run = prefix_run(tok, t["truth"])
                c = res[t["cls"]]
                c["n"] += t["N"]; c["issued"] += int(gate.sum()); c["d1_ok"] += int((gate & (tok[:, 0] == t["truth"][:, 0])).sum()); c["run_sum"] += float(run[gate].sum()); c["run_ge3"] += int((gate & (run >= 3)).sum())
        store[name] = dict(tau=taus, by_class={c: dict(coverage=r["issued"] / max(r["n"], 1), precision=r["d1_ok"] / max(r["issued"], 1), mean_run=r["run_sum"] / max(r["issued"], 1), run_ge3=r["run_ge3"] / max(r["n"], 1), issued=r["issued"], n=r["n"]) for c, r in res.items()})
        store[name]["pooled_abc"] = dict(coverage=sum(res[c]["issued"] for c in "abc") / sum(res[c]["n"] for c in "abc"), precision=sum(res[c]["d1_ok"] for c in "abc") / max(sum(res[c]["issued"] for c in "abc"), 1))
        return gate_by_task

    out["r1_cv"] = {}
    gates_r1 = {}
    for M in Ms:
        gates_r1[M] = cv_eval(lambda t, f, M=M: r1_cv[M][f][t["id"]][:2], f"M{M}", out["r1_cv"])

    def tab(store, names, title, labels=None):
        P(title + "\n")
        P("| variant | tau (fold 0 / 1) | " + " | ".join(f"{c} cov / prec / run" for c in CLASSES) + " | a-c cov / prec |")
        P("|---|---|" + "---|" * (len(CLASSES) + 1))
        for n in names:
            s = store[n]
            lab = labels[n] if labels else n
            P(f"| {lab} | " + " / ".join("-" if s["tau"][f] is None else f"{s['tau'][f]:.3g}" for f in (0, 1)) + " | " + " | ".join(f"{s['by_class'][c]['coverage']:.3f} / {s['by_class'][c]['precision']:.3f} / {s['by_class'][c]['mean_run']:.2f}" for c in CLASSES) + f" | {s['pooled_abc']['coverage']:.3f} / {s['pooled_abc']['precision']:.3f} |")
        P()

    tab(out["r1_cv"], [f"M{M}" for M in Ms], "## R1: anchor-consistent mass-weighted pointer vote, held-out (2-fold), threshold calibrated on the train fold for first-draft precision >= 0.9; `cov` = fraction of cycle starts with a draft, `prec` = P(first draft right | issued), `run` = mean accepted run (draft slots 1..7, no MTP involved) | M = number of heads (best by train score)",
        labels={f"M{M}": f"top-{M} heads" for M in Ms})

    # pointer as next-token predictor (no anchor filter)
    nt = {}
    for M in (16, 384):
        for f, (tr, te) in folds.items():
            ost, hm = fit_heads(tr, M)
            for t in tr + te:
                nt.setdefault(M, {}).setdefault(f, {})[t["id"]] = pointer_next_token(t, ost, hm)
    P("## Next-token precision of the pointer / neighbour WITHOUT the anchor filter (what the state 'knows' about the token it just produced), held-out, tau calibrated for precision >= 0.9 on P(pred == anchor)\n")
    P("| predictor | " + " | ".join(f"{c} cov / prec" for c in CLASSES) + " |")
    P("|---|" + "---|" * len(CLASSES))
    nt_out = {}

    def nt_eval(name, get):
        res = {c: [0, 0, 0] for c in CLASSES}
        for f, (tr, te) in folds.items():
            cf, ok = [], []
            for t in tr:
                pred, conf = get(t, f)
                cf.append(np.where(pred >= 0, conf, -9.0)); ok.append(pred == t["anchor"])
            tau = calibrate(np.concatenate(cf), np.concatenate(ok))
            for t in te:
                pred, conf = get(t, f)
                g = (pred >= 0) & (conf >= tau) if tau is not None else np.zeros(t["N"], bool)
                res[t["cls"]][0] += t["N"]; res[t["cls"]][1] += int(g.sum()); res[t["cls"]][2] += int((g & (pred == t["anchor"])).sum())
        nt_out[name] = {c: dict(coverage=r[1] / max(r[0], 1), precision=r[2] / max(r[1], 1)) for c, r in res.items()}
        P(f"| {name} | " + " | ".join(f"{nt_out[name][c]['coverage']:.3f} / {nt_out[name][c]['precision']:.3f}" for c in CLASSES) + " |")

    nt_eval("R1 pointer vote top-16 heads", lambda t, f: nt[16][f][t["id"]])
    nt_eval("R1 pointer vote all heads", lambda t, f: nt[384][f][t["id"]])
    for rep in REPRS:
        nt_eval(f"R2 kNN {rep} (top-1 neighbour's next token)", lambda t, f, rep=rep: knn_next_token(t, rep))
    P()
    out["next_token"] = nt_out

    # ---------------------------------------------------------------- 3. R2 kNN with CV calibration
    out["r2_cv"] = {}
    gates_r2 = {}
    for rep in REPRS:
        gates_r2[rep] = cv_eval(lambda t, f, rep=rep: knn_src(t, rep), rep, out["r2_cv"])
    tab(out["r2_cv"], REPRS, "## R2: anchor-masked cosine kNN (best neighbour whose next token equals the anchor), held-out, tau calibrated for first-draft precision >= 0.9 (h<slot> = residual after layer slot-1, h65 = final norm out, centred on the prompt mean unless raw; k<i> = K cache of attention layer i, dims 64..255 of the 4 kv heads)")

    # ---------------------------------------------------------------- 4. policies (MODELLED tok/s with K11's measured cycle costs)
    best_r2 = max(REPRS, key=lambda r: out["r2_cv"][r]["pooled_abc"]["coverage"] * (out["r2_cv"][r]["pooled_abc"]["precision"] >= 0.85))
    out["r2_best_repr"] = best_r2
    best_M = max(Ms, key=lambda M: out["r1_cv"][f"M{M}"]["pooled_abc"]["coverage"] * (out["r1_cv"][f"M{M}"]["pooled_abc"]["precision"] >= 0.88))
    out["r1_best_M"] = best_M

    def retriever(name):
        if name.startswith("R1"):
            M_ = int(name.split("@")[1])

            def get(t, f, M_=M_):
                src, conf, nv = r1_cv[M_][f][t["id"]]
                return src, conf
        else:
            def get(t, f, rep=best_r2):
                return knn_src(t, rep)
        return get

    GATES = [(g, q) for g in (0, 1, 2, 3) for q in ((0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95) if True else ())]  # (min agreement with the MTP chain, conf quantile of the train rows with a source)

    def gate_from(t, src, conf, agree_min, tau):
        gate = (src >= 0) & (conf >= tau)
        if agree_min:
            tok = slot_tokens(t, np.where(gate, src, -1))
            ag = np.cumprod(tok[:, :3] == t["D"], axis=1).sum(1)
            gate &= ag >= agree_min
        return gate

    def policy_pts(name, f, ts, params):
        get = retriever(name)
        pts = {}
        for t in ts:
            src, conf = get(t, f)
            pts[t["id"]] = tail_table(t, src, gate_from(t, src, conf, *params))
        return pts

    sel = {}
    pol = {}
    pol_gate = {}  # id -> (src, gate) for the held-out evaluation of each retriever
    for name in [f"R1@{M}" for M in Ms] + ["R2"]:
        get = retriever(name)
        sel[name] = {}
        for f, (tr, te) in folds.items():
            confs = np.concatenate([get(t, f)[1][get(t, f)[0] >= 0] for t in tr])
            best = None
            for ag in (0, 1, 2, 3):
                for qn in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95):
                    tau = float(np.quantile(confs, qn)) if len(confs) else 0.0
                    pts = policy_pts(name, f, tr, (ag, tau))
                    b = {t["id"]: PS.walk(pt0[t["id"]], cost_for(t, costs), m=3, Lt=4, tail=False) for t in tr}
                    res = {t["id"]: PS.walk(pts[t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tr}
                    # objective: pooled throughput ratio over MTP-only on all five classes (negative controls count) with a 0.5% per-class loss guard on d and e
                    ratios = {c: ratio_ci(res, b, tr, c, nb=1)[0] for c in CLASSES}
                    obj = np.mean([ratios[c] for c in CLASSES]) - 5.0 * sum(max(0.0, 0.995 - ratios[c]) for c in "de")
                    if best is None or obj > best[0]:
                        best = (obj, ag, tau, qn, ratios)
            sel[name][f] = dict(agree_min=best[1], tau=best[2], quantile=best[3], train_ratio=best[4])
            pts = policy_pts(name, f, te, (best[1], best[2]))
            for t in te:
                pol.setdefault(name, {})[t["id"]] = pts[t["id"]]
    out["gate_selection"] = {n: {str(f): v for f, v in d.items()} for n, d in sel.items()}
    policies = {"MTP-only": base}
    policies["MTP + R0 (K11: l>=3, tail<=4)"] = r0
    for name in [f"R1@{M}" for M in Ms] + ["R2"]:
        policies[f"MTP + {name}" + (" (top-%s heads)" % name.split("@")[1] if name.startswith("R1") else f" ({best_r2})")] = {t["id"]: PS.walk(pol[name][t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tasks}

    # best-of / fallback / oracle combos of the three tails
    def combine(mode, pr1=None, pr2=None):
        pr1 = pr1 or pol[f"R1@{best_M}"]
        pr2 = pr2 or pol["R2"]
        pts = {}
        for t in tasks:
            i = t["id"]
            a, b, c = pt0[i], pr1[i], pr2[i]
            ga = (a["tl"] > 0) & (a["el"] >= 3)
            gb = b["tl"] > 0
            gc = c["tl"] > 0
            rt = np.zeros_like(a["rt"]); tl = np.zeros_like(a["tl"]); gate = np.zeros(t["N"], bool)
            if mode == "fallback":  # R0 else R1 else R2
                choose = np.where(ga, 0, np.where(gb, 1, np.where(gc, 2, -1)))
            elif mode == "longest":  # longest tail among the gated retrievers, ties R0 > R1 > R2
                tls = np.stack([np.where(ga, a["tl"], -1), np.where(gb, b["tl"], -1), np.where(gc, c["tl"], -1)], 0)
                choose = np.where(tls.max(0) > 0, tls.argmax(0), -1)
            elif mode == "oracle":  # best realised run among the gated retrievers (not implementable)
                rr = np.stack([np.where(ga, a["rt"][:, 4], -1), np.where(gb, b["rt"][:, 4], -1), np.where(gc, c["rt"][:, 4], -1)], 0)
                choose = np.where(rr.max(0) >= 0, rr.argmax(0), -1)
            src_pt = [a, b, c]
            for k in range(3):
                m_ = choose == k
                rt[m_] = src_pt[k]["rt"][m_]; tl[m_] = src_pt[k]["tl"][m_]; gate |= m_
            pts[i] = dict(N=t["N"], rm=a["rm"], el=np.where(gate, 99, 0).astype(np.int16), tl=tl, rt=rt)
        return pts

    cal_r1 = {t["id"]: tail_table(t, *[gates_r1[best_M][t["id"]][0], gates_r1[best_M][t["id"]][1]]) for t in tasks}
    cal_r2 = {t["id"]: tail_table(t, *[gates_r2[best_r2][t["id"]][0], gates_r2[best_r2][t["id"]][1]]) for t in tasks}
    pts_mode = {m: combine(m) for m in ("fallback", "longest", "oracle")}
    pts_mode["longest_cal"] = combine("longest", cal_r1, cal_r2)
    policies["MTP + best-of fallback (R0 > R1 > R2)"] = {t["id"]: PS.walk(pts_mode["fallback"][t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tasks}
    policies["MTP + best-of longest tail"] = {t["id"]: PS.walk(pts_mode["longest"][t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tasks}
    policies["MTP + best-of longest tail, calibrated precision>=0.9 gates (no MTP-agreement requirement)"] = {t["id"]: PS.walk(pts_mode["longest_cal"][t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tasks}
    policies["MTP + oracle of R0/R1/R2 (not implementable)"] = {t["id"]: PS.walk(pts_mode["oracle"][t["id"]], cost_for(t, costs), m=1, Lt=4, tail=True) for t in tasks}

    # replacement policies: where the retriever is confident AND disagrees with the MTP chain, its slots 1..L (L = known tokens <= 7) replace the chain (T = 1 + L rows); elsewhere MTP + R0 tail
    def repl_pts(t, src, conf, tau):
        tok = slot_tokens(t, np.where(src >= 0, src, -1))
        known = np.cumprod(tok >= 0, axis=1).sum(1)
        agree3 = np.cumprod(tok[:, :3] == t["D"], axis=1).sum(1)
        use = (src >= 0) & (conf >= tau) & (known >= 4) & (agree3 < 3)
        run = np.cumprod(tok == t["truth"], axis=1).sum(1)
        ln = np.clip(np.minimum(known, 7) - 3, 0, 4)
        b = pt0[t["id"]]
        rt = b["rt"].copy(); tl = b["tl"].copy(); el = b["el"].copy()
        rows = np.nonzero(use)[0]
        rt[rows, ln[rows]] = run[rows]; tl[rows] = ln[rows]; el[rows] = 99
        return dict(N=t["N"], rm=b["rm"], el=el, tl=tl, rt=rt), use

    repl_sel = {}
    for name, get in (("R1", retriever(f"R1@{best_M}")), ("R2", retriever("R2"))):
        repl_sel[name] = {}
        rp = {}
        for f, (tr, te) in folds.items():
            confs = np.concatenate([get(t, f)[1][(get(t, f)[0] >= 0)] for t in tr])
            bestv = None
            for qn in (None, 0.5, 0.7, 0.85, 0.95, 0.99):
                tau = np.inf if qn is None else float(np.quantile(confs, qn))
                res = {t["id"]: PS.walk(repl_pts(t, *get(t, f), tau)[0], cost_for(t, costs), m=3, Lt=4, tail=True) for t in tr}
                b_ = {t["id"]: PS.walk(pt0[t["id"]], cost_for(t, costs), m=3, Lt=4, tail=False) for t in tr}
                ratios = {c: ratio_ci(res, b_, tr, c, nb=1)[0] for c in CLASSES}
                obj = np.mean([ratios[c] for c in CLASSES]) - 5.0 * sum(max(0.0, 0.995 - ratios[c]) for c in "de")
                if bestv is None or obj > bestv[0]:
                    bestv = (obj, tau, qn, ratios)
            repl_sel[name][f] = dict(tau=None if bestv[1] == np.inf else bestv[1], quantile=bestv[2], train_ratio=bestv[3])
            for t in te:
                rp[t["id"]] = repl_pts(t, *get(t, f), bestv[1])
        out.setdefault("replacement", {})[name] = dict(sel={str(f): v for f, v in repl_sel[name].items()}, used_share={c: float(sum(rp[t["id"]][1].sum() for t in tasks if t["cls"] == c) / sum(t["N"] for t in tasks if t["cls"] == c)) for c in CLASSES},
                                                       correct_when_used={c: float(sum(((np.cumprod(slot_tokens(t, np.where(rp[t["id"]][1], get(t, t["fold"])[0], -1)) == t["truth"], axis=1).sum(1) > t["rm"]) & rp[t["id"]][1]).sum() for t in tasks if t["cls"] == c) / max(sum(rp[t["id"]][1].sum() for t in tasks if t["cls"] == c), 1)) for c in CLASSES})
        policies[f"MTP + R0 tail + {name} replacement of the chain (confident and disagreeing with MTP)"] = {t["id"]: PS.walk(rp[t["id"]][0], cost_for(t, costs), m=3, Lt=4, tail=True) for t in tasks}

    P("## Modelled tok/s (measured acceptance x K11's MEASURED cycle cost V(T=1..8) per context, interpolated; per-cycle host gap 0.01 ms on tail policies), ratio over MTP-only [task-bootstrap 95% CI] and over MTP + R0\n")
    P("| policy | " + " | ".join(f"{c} tok/s (E)" for c in CLASSES) + " | a-c |")
    P("|---|" + "---|" * 6)
    out["policies"] = {}
    for nm, res in policies.items():
        row = {c: tokps(res, tasks, c) for c in list(CLASSES) + ["abc"]}
        out["policies"][nm] = dict(tokps={c: row[c][0] for c in row}, E={c: row[c][1] for c in row}, over_mtp={c: ratio_ci(res, base, tasks, c) for c in list(CLASSES) + ["abc"]}, over_r0={c: ratio_ci(res, r0, tasks, c) for c in list(CLASSES) + ["abc"]})
        P(f"| {nm} | " + " | ".join(f"{row[c][0]:.1f} ({row[c][1]:.2f})" for c in list(CLASSES) + ["abc"]) + " |")
    P()
    for key, lab in (("over_mtp", "over MTP-only"), ("over_r0", "over MTP + R0 (K11 tail)")):
        P(f"**ratio {lab}** [95% CI]\n")
        P("| policy | " + " | ".join(CLASSES) + " | a-c |")
        P("|---|" + "---|" * 6)
        for nm in policies:
            if nm == "MTP-only" or (key == "over_r0" and nm.startswith("MTP + R0")):
                continue
            o = out["policies"][nm][key]
            P(f"| {nm} | " + " | ".join(f"x{o[c][0]:.3f} [{o[c][1]:.3f}, {o[c][2]:.3f}]" for c in list(CLASSES) + ["abc"]) + " |")
        P()
    P("Gate selection per fold (fitted on the train tasks: min agreement of the retriever's slots 1..3 with the MTP drafts, confidence cut at a train quantile; objective = mean over classes of the modelled ratio over MTP-only with a d/e no-loss guard):\n")
    for nm in sel:
        P(f"- {nm}: " + "; ".join(f"fold {f}: agree>={v['agree_min']}, tau {v['tau']:.3g} (q{v['quantile']}), train ratios " + "/".join(f"{v['train_ratio'][c]:.3f}" for c in CLASSES) for f, v in sel[nm].items()))
    for nm, d in out["replacement"].items():
        P(f"- {nm} replacement: " + "; ".join(f"fold {f}: tau {('never' if v['tau'] is None else format(v['tau'], '.3g'))} (q{v['quantile']}), train ratios " + "/".join(f"{v['train_ratio'][c]:.3f}" for c in CLASSES) for f, v in d["sel"].items()) + f"; used in {', '.join(c + ' ' + format(100 * d['used_share'][c], '.2f') + '%' for c in CLASSES)} of cycles; replaced cycles where the pointer run beats the chain run: {', '.join(c + ' ' + format(d['correct_when_used'][c], '.2f') for c in CLASSES)}")
    P(f"- best-of / overlap use R1 with the top-{best_M} head subset (best held-out a-c coverage at precision >= 0.9) and R2 = `{best_r2}`.\n")

    # ---------------------------------------------------------------- 5. overlap / extra accepted tokens / where MTP is wrong
    views = {"calibrated": (cal_r1, cal_r2, "gates calibrated for first-draft precision >= 0.9 on the train fold (no MTP agreement requirement)"),
             "policy": (pol[f"R1@{best_M}"], pol["R2"], "throughput-tuned policy gates (min agreement with the MTP chain + confidence cut, fitted on the train fold)")}
    ov = {}
    for vn, (pr1, pr2, vdesc) in views.items():
        pbest = combine("longest", pr1, pr2)
        P(f"## Per-position view, {vn} gates: {vdesc}\n")
        P("Every position as a cycle start; run = accepted drafts 0..7 of MTP chain + tail (chain-only run when no tail is issued). `P(R0|R1)` = share of R1-issued cycles where R0 also issues; `extra` = mean of max(0, run_X - run_R0) per cycle.\n")
        P("| class | cycles | R0 issued | R1 issued | R2 issued | R1&R0 | R1 only | R2 only | union | P(R0\\|R1) | E[run] MTP+R0 | +R1 | +R2 | best-of | extra R1 | extra R2 | extra best-of |")
        P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        ov[vn] = {}
        for c in CLASSES:
            ts = [t for t in tasks if t["cls"] == c]
            agg = defaultdict(float)
            for t in ts:
                i = t["id"]
                g0 = r0_gate[i]; g1 = pr1[i]["tl"] > 0; g2 = pr2[i]["tl"] > 0
                base_run = np.minimum(t["rm"], 3)
                ru0 = np.where(g0, pt0[i]["rt"][:, 4], base_run)
                ru1 = np.where(g1, pr1[i]["rt"][:, 4], base_run)
                ru2 = np.where(g2, pr2[i]["rt"][:, 4], base_run)
                rub = np.where(pbest[i]["tl"] > 0, pbest[i]["rt"][:, 4], base_run)
                agg["n"] += t["N"]; agg["g0"] += g0.sum(); agg["g1"] += g1.sum(); agg["g2"] += g2.sum(); agg["g10"] += (g1 & g0).sum(); agg["g1only"] += (g1 & ~g0).sum(); agg["g2only"] += (g2 & ~g0 & ~g1).sum()
                agg["union"] += (g0 | g1 | g2).sum(); agg["r0"] += ru0.sum(); agg["r1"] += ru1.sum(); agg["r2"] += ru2.sum(); agg["rb"] += rub.sum()
                agg["x1"] += np.maximum(ru1 - ru0, 0).sum(); agg["x2"] += np.maximum(ru2 - ru0, 0).sum(); agg["xb"] += np.maximum(rub - ru0, 0).sum()
            n = agg["n"]
            ov[vn][c] = dict(n=int(n), r0=agg["g0"] / n, r1=agg["g1"] / n, r2=agg["g2"] / n, r1_and_r0=agg["g10"] / n, r1_only=agg["g1only"] / n, r2_only=agg["g2only"] / n, union=agg["union"] / n, p_r0_given_r1=agg["g10"] / max(agg["g1"], 1),
                             run_mtp_r0=agg["r0"] / n, run_r1=agg["r1"] / n, run_r2=agg["r2"] / n, run_best=agg["rb"] / n, extra_r1=agg["x1"] / n, extra_r2=agg["x2"] / n, extra_best=agg["xb"] / n)
            o = ov[vn][c]
            P(f"| {c} | {int(n)} | {o['r0']:.3f} | {o['r1']:.3f} | {o['r2']:.3f} | {o['r1_and_r0']:.3f} | {o['r1_only']:.3f} | {o['r2_only']:.3f} | {o['union']:.3f} | {o['p_r0_given_r1']:.3f} | {o['run_mtp_r0']:.3f} | {o['run_r1']:.3f} | {o['run_r2']:.3f} | {o['run_best']:.3f} | {o['extra_r1']:+.3f} | {o['extra_r2']:+.3f} | {o['extra_best']:+.3f} |")
        P()
    out["overlap"] = ov

    # ---------------------------------------------------------------- 5b. information bound: oracle over ALL candidate sources of each family
    def best_tail_runs(t, rows_c, s_c):
        """for every candidate (row, source): realised run of chain + tail from that source; -> best run per row (max over candidates), -1 where no candidate."""
        N = t["N"]
        best = np.full(N, -1, np.int64)
        if len(s_c) == 0:
            return best
        tok = slot_tokens(t, s_c, 7, rows=rows_c, k0=4)
        tr = t["truth"][rows_c, 3:7]
        run = np.cumprod(tok == tr, axis=1).sum(1)
        full = t["rm"][rows_c] == 3
        run = np.where(full, 3 + run, t["rm"][rows_c].astype(np.int64))
        np.maximum.at(best, rows_c, run)
        return best

    ost_all, hm_all = fit_heads(tasks, 384)
    P("## Information bound: oracle best realised run (chain + tail) over ALL candidate sources of a family, vs K11's R0 run (held-in, hindsight selection: an upper bound, not implementable)\n")
    P("`ptr` = every (layer, head, top-4 rank) pointer of the producing row with mass >= 0.03, anchor-consistent at its fitted offset (<= 1536 candidates/row); `knn` = the 8 best anchor-masked neighbours of h65raw; `n=1` = every earlier occurrence of the anchor token (any exact-copy source); extra = mean of max(0, run_oracle - run_R0) per cycle, `+modelled` = modelled ratio over MTP + R0 when the oracle's tail replaces R0's wherever it is longer (tail <= 4).\n")
    P("| class | R0 run | extra ptr | extra knn | extra ptr+knn | extra n=1 (any copy) | modelled ptr / R0 | ptr+knn / R0 | n=1 / R0 |")
    P("|---|---|---|---|---|---|---|---|---|")
    orc = {}
    orc_pts = {k: {} for k in ("ptr", "knn", "ptrknn", "n1")}
    ex_cls = defaultdict(lambda: defaultdict(float))
    for t in tasks:
        i = t["id"]
        N, S = t["N"], t["S"]
        base_run = np.minimum(t["rm"], 3).astype(np.int64)
        ru0 = np.where(r0_gate[i], pt0[i]["rt"][:, 4], base_run)
        # ptr candidates
        idx = t["z"]["idx"].astype(np.int64)
        m = t["z"]["mass"][..., :4].astype(np.float32)
        sp = idx + ost_all[None, :, :, None]
        r = t["r"][:, None, None, None]
        ok = (idx > 0) & (sp >= 0) & (sp <= r) & (m >= MMIN)
        ok &= S[np.clip(sp, 0, len(S) - 1)] == t["anchor"][:, None, None, None]
        rr, _, _, _ = np.nonzero(ok)
        sp_c = sp[ok]
        # knn candidates
        j = t["z"]["h65raw/msk_idx"].astype(np.int64)
        sim = t["z"]["h65raw/msk_sim"].astype(np.float32)
        okk = sim > -1.5
        rk = np.broadcast_to(np.arange(N)[:, None], j.shape)[okk]
        sk_c = (j + 1)[okk]
        # n=1: all earlier occurrences of the anchor
        occ = defaultdict(list)
        for q_, tk in enumerate(S[: t["P"] + N]):
            occ[int(tk)].append(q_)
        rn, sn = [], []
        for ii in range(N):
            a_ = occ.get(int(t["anchor"][ii]), [])
            hi = np.searchsorted(a_, t["r"][ii], side="right") if a_ else 0
            if hi:
                rn.append(np.full(hi, ii)); sn.append(np.asarray(a_[:hi]))
        rn = np.concatenate(rn) if rn else np.zeros(0, int); sn = np.concatenate(sn) if sn else np.zeros(0, int)
        fam = {"ptr": (rr, sp_c), "knn": (rk, sk_c), "ptrknn": (np.concatenate([rr, rk]), np.concatenate([sp_c, sk_c])), "n1": (rn, sn)}
        c = t["cls"]
        ex_cls[c]["n"] += N; ex_cls[c]["r0"] += ru0.sum()
        for k_, (rows_c, s_c) in fam.items():
            best = best_tail_runs(t, rows_c, s_c)
            use = best > ru0
            ru = np.where(use, best, ru0)
            ex_cls[c][k_] += (ru - ru0).sum()
            rt = pt0[i]["rt"].copy(); tl = pt0[i]["tl"].copy(); el = pt0[i]["el"].copy()
            rt[use, 4] = best[use]; tl[use] = 4; el[use] = 99
            orc_pts[k_][i] = dict(N=N, rm=pt0[i]["rm"], el=el, tl=tl, rt=rt)
    for k_ in orc_pts:
        orc[k_] = {t["id"]: PS.walk(orc_pts[k_][t["id"]], cost_for(t, costs), m=3, Lt=4, tail=True) for t in tasks}
    out["oracle"] = {}
    for c in CLASSES:
        e_ = ex_cls[c]
        rat = {k_: ratio_ci(orc[k_], r0, tasks, c, nb=500) for k_ in orc}
        out["oracle"][c] = dict(r0_run=e_["r0"] / e_["n"], extra={k_: e_[k_] / e_["n"] for k_ in orc}, modelled_over_r0=rat)
        P(f"| {c} | {e_['r0'] / e_['n']:.3f} | {e_['ptr'] / e_['n']:+.3f} | {e_['knn'] / e_['n']:+.3f} | {e_['ptrknn'] / e_['n']:+.3f} | {e_['n1'] / e_['n']:+.3f} | x{rat['ptr'][0]:.3f} [{rat['ptr'][1]:.3f}, {rat['ptr'][2]:.3f}] | x{rat['ptrknn'][0]:.3f} | x{rat['n1'][0]:.3f} |")
    abc = {k_: ratio_ci(orc[k_], r0, tasks, "abc", nb=500) for k_ in orc}
    out["oracle"]["abc"] = dict(modelled_over_r0=abc)
    P(f"| a-c | | | | | | x{abc['ptr'][0]:.3f} [{abc['ptr'][1]:.3f}, {abc['ptr'][2]:.3f}] | x{abc['ptrknn'][0]:.3f} [{abc['ptrknn'][1]:.3f}, {abc['ptrknn'][2]:.3f}] | x{abc['n1'][0]:.3f} [{abc['n1'][1]:.3f}, {abc['n1'][2]:.3f}] |")
    P()

    # where MTP is wrong: cycles with rm < 3
    P("## Where the MTP chain is wrong (rm < 3): can a confident pointer / neighbour supply the first missed slot or more?  (held-out gates; replacement view: pointer slots 1..7 instead of the chain)\n")
    P("| class | P(rm<3) | R1 issued among them | R1 right at slot rm+1 \\| issued | share of rm<3 cycles where R1 fixes slot rm+1 | R1 run>rm share | R2 fixes share | R0-tail fixes (always 0: tail only after a full chain) |")
    P("|---|---|---|---|---|---|---|---|")
    mw = {}
    for c in CLASSES:
        ts = [t for t in tasks if t["cls"] == c]
        n = nb = i1 = ok1 = ok2 = gr1 = 0
        for t in ts:
            i = t["id"]
            bad = t["rm"] < 3
            n += t["N"]; nb += bad.sum()
            for nm, ctr in (("R1", 1), ("R2", 2)):  # noqa
                src, gate = gates_r1[best_M][i] if nm == "R1" else gates_r2[best_r2][i]
                tok = slot_tokens(t, np.where(gate, src, -1))
                rows = np.nonzero(bad & gate)[0]
                slot = t["rm"][rows].astype(int)  # index of the first missed slot (0-based)
                hit = tok[rows, slot] == t["truth"][rows, slot]
                known_before = np.cumprod(tok[:, :3] == t["truth"][:, :3], axis=1).sum(1)[rows] >= slot  # pointer also right on the slots the chain got right
                if nm == "R1":
                    i1 += len(rows); ok1 += int(hit.sum())
                    run = prefix_run(tok, t["truth"])
                    gr1 += int((run[rows] > t["rm"][rows]).sum())
                else:
                    ok2 += int(hit.sum())
        mw[c] = dict(p_bad=nb / n, r1_issued=i1, r1_right=ok1 / max(i1, 1), r1_fix_share=ok1 / max(nb, 1), r1_gain_share=gr1 / max(nb, 1), r2_fix_share=ok2 / max(nb, 1))
        o = mw[c]
        P(f"| {c} | {o['p_bad']:.3f} | {i1} | {o['r1_right']:.3f} | {o['r1_fix_share']:.3f} | {o['r1_gain_share']:.3f} | {o['r2_fix_share']:.3f} | 0 |")
    P()
    out["mtp_wrong"] = mw

    # ---------------------------------------------------------------- 6. R1 catches where R0 is absent (fuzzy copies)
    P("## R1 catches where R0 issues nothing (calibrated precision >= 0.9 gate; `bl` = backward agreement of the source with the current context incl. the anchor)\n")
    P("| class | R1 issued | R1 issued & R0 absent | share of all cycles | first-draft precision there | mean run (slots 1..7) | with bl < 3 | precision of those | mean run of those |")
    P("|---|---|---|---|---|---|---|---|---|")
    fz = {}
    ex = []
    for c in CLASSES:
        ts = [t for t in tasks if t["cls"] == c]
        iss = fzz = ok = n = nb3 = ok3 = 0
        runsum = runsum3 = 0.0
        for t in ts:
            i = t["id"]
            src, gate = gates_r1[best_M][i]
            bl = backward_agree(t, np.where(gate, src, -1))
            f_ = gate & ~r0_gate[i]
            tok = slot_tokens(t, np.where(gate, src, -1))
            run = prefix_run(tok, t["truth"])
            n += t["N"]; iss += gate.sum(); fzz += f_.sum(); ok += (f_ & (tok[:, 0] == t["truth"][:, 0])).sum(); runsum += run[f_].sum()
            f3 = f_ & (bl < 3)
            nb3 += f3.sum(); ok3 += (f3 & (tok[:, 0] == t["truth"][:, 0])).sum(); runsum3 += run[f3].sum()
            for k in np.nonzero(f_ & (run >= 3))[0]:
                ex.append(dict(task=i, cls=c, cycle=int(k), P=t["P"], src=int(src[k]), run=int(run[k]), bl=int(bl[k]), rm=int(t["rm"][k])))
        fz[c] = dict(r1_issued=int(iss), catches=int(fzz), share=fzz / n, precision=ok / max(fzz, 1), mean_run=runsum / max(fzz, 1), fuzzy_bl_lt3=int(nb3), fuzzy_precision=ok3 / max(nb3, 1), fuzzy_mean_run=runsum3 / max(nb3, 1))
        o = fz[c]
        P(f"| {c} | {iss} | {fzz} | {o['share']:.4f} | {o['precision']:.3f} | {o['mean_run']:.2f} | {nb3} | {o['fuzzy_precision']:.3f} | {o['fuzzy_mean_run']:.2f} |")
    P()
    out["fuzzy"] = fz
    out["fuzzy_examples_n"] = len(ex)
    if not args.no_examples:
        try:
            from transformers import AutoTokenizer

            tk = AutoTokenizer.from_pretrained(str(Path.home() / "models" / "swift-1.5-int4"))
            dec = lambda ids: tk.decode([int(x) for x in ids if x >= 0])
            ex.sort(key=lambda e: (e["bl"] >= 3, -e["run"], e["bl"]))
            seen, outx = set(), []
            for e_ in ex:
                if (e_["task"], e_["cycle"] // 12) in seen:
                    continue
                seen.add((e_["task"], e_["cycle"] // 12))
                t = byid[e_["task"]]
                p = t["P"] + e_["cycle"]
                s = e_["src"]
                e_["context_tail"] = dec(t["S"][p - 12 : p + 1])
                e_["source_context"] = dec(t["S"][s - 12 : s + 1])
                e_["pointer_continuation"] = dec(t["S"][s + 1 : s + 1 + e_["run"]])
                e_["truth_continuation"] = dec(t["g"][e_["cycle"] + 1 : e_["cycle"] + 1 + e_["run"]])
                outx.append(e_)
                if len(outx) >= 40:
                    break
            json.dump(outx, open(HERE / "results" / "examples.json", "w"), indent=1)
        except Exception as e:  # tokenizer unavailable
            print("examples skipped:", e)

    (HERE / "results").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "results" / "summary.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    (HERE / "results" / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
