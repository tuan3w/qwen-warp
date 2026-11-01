"""Offline analysis of ngram-draft-potential (CPU only).  python3 experiments/2026-10-31-ngram-draft-potential/analyze.py

Input  : results/trace.npz (built from runs/trace/*.json by `pack`): per task prompt ids, the canonical greedy continuation g_0..g_{n-1} (forced single-row MTP cycles),
         the MTP drafts d1..d3 at every position and the natural accepted count, plus the natural k=3 cycle record (accepted drafts per cycle).
Output : results/summary.json, results/tables.md

Notation: position i = a cycle start whose anchor is g_i (already emitted, not yet verified); the cycle must produce g_{i+1}, g_{i+2}, ...
A drafter proposes tokens c_1..c_L for g_{i+1}..g_{i+L}; greedy verification accepts the longest prefix r that equals the real continuation and emits r+1 tokens.
MEASURED: the continuation, the MTP drafts and hence MTP acceptance at every position, PLD matches / accepted runs (exact against the continuation).
MODELLED: the time of a cycle (step-time table V(T), MTP overhead, host-sync penalty).
"""
import bisect
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("NGRAM_RES", HERE / "results"))
TRACE = Path(os.environ.get("NGRAM_TRACE", HERE / "runs" / "trace"))
LMAX = 7
NMAX_GEN = 768  # natural-run wall time is only meaningful for runs that hit the token cap (early-EOS runs kept launching cycles after the EOS)
CLASSES = ["a", "b", "c", "d", "e"]
CNAME = {"a": "a code-edit", "b": "b log/tool-output", "c": "c agent tool-call", "d": "d prose (control)", "e": "e reasoning/math (control)"}

# ------------------------------------------------------------------------------------------------ cost model (ms)
# verify step V(T), T = 1..8 rows incl. the anchor.  T<=4: K5 measured (BENCH.md K5: 20.195 / 20.721 / 21.077 / 21.545).  T=5..8: V(4) + E3 whole-step GEMV growth
# (18.93 -> 19.09 / 19.62 / ~19.985 (interp) / 20.35 ms: experiments/2026-10-30-w4-multirow-verify-and-power/README.md, "Whole-step GEMV time") + DeltaNet ring snapshots
# 0.2 ms/row beyond T=4 (K1: 0.51 ms at T=1, 1.15 ms at T=4 => ~0.2 ms/row) + attention 0.02 ms/row (K2).
V_BASE = {1: 20.195, 2: 20.721, 3: 21.077, 4: 21.545}
_G = {4: 18.93, 5: 19.09, 6: 19.62, 7: 19.985, 8: 20.35}
for _T in range(5, 9):
    V_BASE[_T] = V_BASE[4] + (_G[_T] - _G[4]) + 0.2 * (_T - 4) + 0.02 * (_T - 4)
O_FIX = 0.275  # accept kernel + MTP catch-up of the accepted rows (K4/K7: cycle 23.10 - V(4) 21.545 - 3 drafts 1.28)
O_DRAFT = 0.427  # per MTP draft step (BENCH.md K4: one draft = MTP layer + 64k head = 0.43 ms)
O_CU_ROW = 0.03  # extra catch-up cost per row beyond 4 (MTP layer GEMV T>4; [INFERENCE from E3 growth, ~0.1 ms at T=8])
SYNC_HOST = 0.2  # host picks the graph per cycle (one sync per cycle instead of 2 cycles per sync: BENCH.md K8 per-sync cost <= 0.7% of a cycle)


def make_V(scale=1.0, v8=None):
    """verify-step table; `scale` multiplies the growth beyond T=4, or `v8` pins V(8)."""
    if v8 is not None:
        scale = (v8 - V_BASE[4]) / (V_BASE[8] - V_BASE[4])
    return {T: (V_BASE[T] if T <= 4 else V_BASE[4] + (V_BASE[T] - V_BASE[4]) * scale) for T in V_BASE}


def ofix(T):
    return O_FIX + O_CU_ROW * max(0, T - 4)


# ------------------------------------------------------------------------------------------------ data
def pack():
    out = {}
    for fn in sorted(TRACE.glob("*.json")):
        r = json.loads(fn.read_text())
        i = r["id"]
        out[f"{i}/meta"] = np.array([r["n_prompt"], ord(r["cls"])], dtype=np.int32)
        out[f"{i}/prompt"] = np.array(r["prompt_ids"], dtype=np.int32)
        out[f"{i}/g"] = np.array(r["F"]["tokens"], dtype=np.int32)
        out[f"{i}/D"] = np.array(r["F"]["drafts"], dtype=np.int32).reshape(-1, 3)
        out[f"{i}/m"] = np.array(r["F"]["m"], dtype=np.int8)
        out[f"{i}/Ntok"] = np.array(r["N"]["tokens"], dtype=np.int32)
        out[f"{i}/Nacc"] = np.array(r["N"]["accepted"], dtype=np.int8)
        out[f"{i}/Nsec"] = np.array([r["N"]["decode_s"]], dtype=np.float64)
    np.savez_compressed(RES / "trace.npz", **out)


def load():
    z = np.load(RES / "trace.npz")
    ids = sorted({k.split("/")[0] for k in z.files})
    tasks = []
    for i in ids:
        Nt, Na = z[f"{i}/Ntok"], z[f"{i}/Nacc"]
        # the natural run kept launching cycles after the EOS / token cap inside the last cycle: keep only the cycles that produced Ntok
        cum = 1 + np.cumsum(Na.astype(int) + 1)
        nc = int(np.searchsorted(cum, len(Nt))) + 1
        tasks.append(dict(id=i, cls=chr(int(z[f"{i}/meta"][1])), P=int(z[f"{i}/meta"][0]), prompt=z[f"{i}/prompt"], g=z[f"{i}/g"], D=z[f"{i}/D"], m=z[f"{i}/m"],
                          Ntok=Nt, Nacc=Na[:nc], Nsec=float(z[f"{i}/Nsec"][0])))
    return tasks


# ------------------------------------------------------------------------------------------------ prompt lookup
def build_index(S, nmax=4):
    idx = {}
    for n in range(1, nmax + 1):
        d = defaultdict(list)
        for e in range(n - 1, len(S)):
            d[tuple(S[e - n + 1 : e + 1])].append(e)
        idx[n] = d
    return idx


def lookup(K, idx, n, strat, end, jlim, L=LMAX, cap=64, back=64):
    """n-gram prompt lookup: query = K[end-n+1..end]; candidate end positions j < jlim (known tokens only) from the index of the full sequence S (K == S[:jlim] ... on the known part);
    strat 'recent' = most recent candidate, 'longest' = candidate with the longest backward-extended agreement (ties: most recent; at most `cap` most recent candidates).
    Returns (match length l, proposal tokens K[j+1 : j+1+L]) or (0, [])."""
    if end + 1 < n:
        return 0, []
    occ = idx[n].get(tuple(K[end - n + 1 : end + 1]))
    if not occ:
        return 0, []
    hi = bisect.bisect_left(occ, jlim)
    if hi == 0:
        return 0, []
    cands = occ[max(0, hi - cap) : hi][::-1] if strat == "longest" else [occ[hi - 1]]
    best_j, best_l = -1, -1
    for j in cands:
        l = n
        while l < back and j - l >= 0 and K[j - l] == K[end - l]:
            l += 1
        if l > best_l:
            best_j, best_l = j, l
    return best_l, list(K[best_j + 1 : min(best_j + 1 + L, end + 1)])  # only tokens known at the anchor (K may extend into the real continuation)


def prefix_match(c, g, i):
    """accepted run of proposal c (tokens for g_{i+1}, ...) against the real continuation."""
    r = 0
    for t, x in enumerate(c):
        if i + 1 + t < len(g) and g[i + 1 + t] == x:
            r += 1
        else:
            break
    return r


KEYS = [(n, s) for n in (1, 2, 3, 4) for s in ("recent", "longest")]


def prep(task):
    """per-position arrays for every lookup key + MTP run lengths + MTP-tail extension lookups."""
    g, P, D = task["g"], task["P"], task["D"]
    S = np.concatenate([task["prompt"], g]).astype(np.int64)
    n = len(g)
    idx = build_index(S)
    N = n - 1  # cycle starts i = 0..n-2
    ell = {k: np.zeros(N, np.int16) for k in KEYS}
    plen = {k: np.zeros(N, np.int8) for k in KEYS}
    R = {k: np.zeros((N, LMAX + 1), np.int8) for k in KEYS}  # R[i, L] accepted run of the proposal truncated to L
    ext = {}
    rm = np.zeros(N, np.int8)
    for i in range(N):
        p = P + i
        known = S[: p + 1]
        for k in KEYS:
            l, c = lookup(S, idx, k[0], k[1], p, p)
            ell[k][i], plen[k][i] = l, len(c)
            for L in range(1, LMAX + 1):
                R[k][i, L] = prefix_match(c[:L], g, i)
        rm[i] = prefix_match(list(D[i]), g, i)
    # MTP chain + PLD tail: query ends at d3 (K = known + drafts)
    for k in [(4, "recent"), (3, "recent"), (2, "recent"), (4, "longest"), (2, "longest"), (1, "longest")]:
        el = np.zeros(N, np.int16)
        tl = np.zeros(N, np.int8)
        rt = np.zeros((N, 5), np.int8)  # accepted run of chain [d1,d2,d3] + tail[:Lt], Lt = 1..4
        for i in range(N):
            p = P + i
            K = np.concatenate([S[: p + 1], D[i]])
            l, c = lookup(K, idx, k[0], k[1], len(K) - 1, p + 1, L=4)
            el[i], tl[i] = l, len(c)
            for Lt in range(0, 5):
                rt[i, Lt] = prefix_match(list(D[i]) + c[:Lt], g, i)
        ext[k] = (el, tl, rt)
    return dict(n=n, N=N, ell=ell, plen=plen, R=R, rm=rm, ext=ext)


# ------------------------------------------------------------------------------------------------ policies
def sim(pt, policy, V, **kw):
    """walk the greedy continuation with `policy`; returns (tokens, time_ms, cycles, detail)."""
    N = pt["N"]
    i, t, cyc = 0, 0.0, 0
    nxt_mtp = None
    use_pld_cnt = 0
    acc_hist = np.zeros(LMAX + 2, np.int64)
    while i < N:
        T, r, cost = policy(pt, i, V, **kw)
        adv = min(r + 1, N - i)
        t += cost
        cyc += 1
        use_pld_cnt += T != 4
        acc_hist[min(r, LMAX + 1)] += 1
        i += adv
    return N, t, cyc, acc_hist


def pol_mtp(pt, i, V, k=3, **_):
    r = min(int(pt["rm"][i]), k)
    T = 1 + k
    return T, r, V[T] + ofix(T) + O_DRAFT * k


def sel(pt, i, cfg):
    """longest-n-first key selection: cfg = (nmax, nmin, strat) -> (key, l, plen) or None"""
    nmax, nmin, strat = cfg
    for n in range(nmax, nmin - 1, -1):
        k = (n, strat)
        if pt["plen"][k][i] > 0:
            return k, int(pt["ell"][k][i])
    return None, 0


def pol_hybrid(pt, i, V, cfg=(4, 2, "recent"), m=3, L=7, sync=SYNC_HOST, skip=False, k=3, **_):
    key, l = sel(pt, i, cfg)
    if key is not None and l >= m:
        Ln = min(L, int(pt["plen"][key][i]))
        T = 1 + Ln
        r = int(pt["R"][key][i, Ln])
        cost = V[T] + ofix(T) + sync
        use_pld = True
    else:
        T = 1 + k
        r = min(int(pt["rm"][i]), k)
        cost = V[T] + ofix(T) + sync
        use_pld = False
    if skip:  # variant B: drafts are computed only if the NEXT cycle will use the MTP chain (on-device decision, no host sync)
        # next anchor is unknown before r is known; r is known here (simulation), the device knows it after accept: same information
        nxt = min(i + r + 1, pt["N"] - 1)
        key2, l2 = sel(pt, nxt, cfg)
        nxt_pld = key2 is not None and l2 >= m and i + r + 1 < pt["N"]
        cost += 0.0 if nxt_pld else O_DRAFT * k
    else:
        cost += O_DRAFT * k
    return T, r, cost


def pol_pld_only(pt, i, V, cfg=(4, 2, "recent"), m=2, L=7, sync=SYNC_HOST, **_):
    key, l = sel(pt, i, cfg)
    if key is not None and l >= m:
        Ln = min(L, int(pt["plen"][key][i]))
        T = 1 + Ln
        return T, int(pt["R"][key][i, Ln]), V[T] + sync
    return 1, 0, V[1] + sync


def pol_ext(pt, i, V, ek=(4, "recent"), m=3, Lt=4, sync=SYNC_HOST, k=3, base_cfg=None, bm=3, bL=7, **_):
    """MTP chain + PLD tail (T = 4 + tail <= 8); tail used iff the tail lookup matched with l >= m; optionally combined with PLD-replaces-MTP (base_cfg)."""
    el, tl, rt = pt["ext"][ek]
    if base_cfg is not None:
        key, l = sel(pt, i, base_cfg)
        if key is not None and l >= bm:
            return pol_hybrid(pt, i, V, cfg=base_cfg, m=bm, L=bL, sync=sync)
    if tl[i] > 0 and el[i] >= m:
        Ln = min(Lt, int(tl[i]))
        T = 4 + Ln
        return T, int(rt[i, Ln]), V[T] + ofix(T) + O_DRAFT * k + sync
    return 4, int(pt["rm"][i]), V[4] + ofix(4) + O_DRAFT * k + sync


def pol_oracle(pt, i, V, cfg=(4, 1, "longest"), L=7, k=3, **_):
    """per-position best of {MTP chain, PLD} by tokens/ms (non-implementable upper bound; includes the sync penalty only on the PLD cycles' host decision: none)"""
    Tm, rm_, cm = pol_mtp(pt, i, V)
    best = (Tm, rm_, cm)
    key, l = sel(pt, i, cfg)
    if key is not None:
        Ln = min(L, int(pt["plen"][key][i]))
        T = 1 + Ln
        c = V[T] + ofix(T) + O_DRAFT * k
        r = int(pt["R"][key][i, Ln])
        if (r + 1) / c > (best[1] + 1) / best[2]:
            best = (T, r, c)
    return best


# ------------------------------------------------------------------------------------------------ aggregation
def run_all(tasks, pts, policy, V, **kw):
    """-> {task id: (tokens, time_ms, cycles)}"""
    out = {}
    for t in tasks:
        n, tm, cyc, _ = sim(pts[t["id"]], policy, V, **kw)
        out[t["id"]] = (n, tm, cyc)
    return out


def pooled(res, ids):
    n = sum(res[i][0] for i in ids)
    t = sum(res[i][1] for i in ids)
    c = sum(res[i][2] for i in ids)
    return dict(tok_s=1000.0 * n / t, E=n / c)


def ratio_boot(res, base, ids, nb=10000, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(ids)
    n = np.array([res[i][0] for i in ids], float)
    t = np.array([res[i][1] for i in ids], float)
    nb_ = np.array([base[i][0] for i in ids], float)
    tb = np.array([base[i][1] for i in ids], float)
    pt = (n.sum() / t.sum()) / (nb_.sum() / tb.sum())
    ix = rng.integers(0, len(ids), (nb, len(ids)))
    rs = (n[ix].sum(1) / t[ix].sum(1)) / (nb_[ix].sum(1) / tb[ix].sum(1))
    lo, hi = np.percentile(rs, [2.5, 97.5])
    lo5, hi5 = np.percentile(rs, [5, 95])
    per = (n / t) / (nb_ / tb)
    return dict(ratio=float(pt), lo95=float(lo), hi95=float(hi), lo90=float(lo5), hi90=float(hi5), min_task=float(per.min()), max_task=float(per.max()), mean_task=float(per.mean()))


def by_class(tasks):
    d = defaultdict(list)
    for t in tasks:
        d[t["cls"]].append(t["id"])
    return d


def pctl(a, qs):
    return [float(x) for x in np.percentile(a, qs)] if len(a) else []


def main():
    if "pack" in sys.argv or not (RES / "trace.npz").exists():
        pack()
    tasks = load()
    cls = by_class(tasks)
    print({c: len(v) for c, v in cls.items()}, flush=True)
    pts = {t["id"]: prep(t) for t in tasks}
    V = make_V()
    S = {}  # summary
    S["V"] = V
    S["cost_model"] = dict(O_FIX=O_FIX, O_DRAFT=O_DRAFT, O_CU_ROW=O_CU_ROW, SYNC_HOST=SYNC_HOST)
    S["tasks"] = {t["id"]: dict(cls=t["cls"], prompt=int(t["P"]), gen=int(len(t["g"])), Ntok=int(len(t["Ntok"])), N_equal_F=bool(np.array_equal(t["Ntok"], t["g"][: len(t["Ntok"])]) and len(t["Ntok"]) == len(t["g"])),
                                E_natural_run=float(len(t["Ntok"]) / max(len(t["Nacc"]), 1)), N_decode_toks=float(len(t["Ntok"]) / t["Nsec"]), ms_per_cycle_N=(float(t["Nsec"] * 1000 / len(t["Nacc"])) if len(t["Ntok"]) >= NMAX_GEN else None)) for t in tasks}

    # ---- measured MTP validation: prefix-match vs the engine's own natural accepted count; natural-run E vs simulated E
    agree = [(int(t["m"][i]) == int(pts[t["id"]]["rm"][i])) for t in tasks for i in range(len(t["m"]))]
    S["mtp_m_agreement"] = float(np.mean(agree))
    base = run_all(tasks, pts, pol_mtp, V)
    S["mtp_sim_vs_natural"] = {c: dict(E_sim=pooled(base, cls[c])["E"], E_natural=float(sum(S["tasks"][i]["Ntok"] for i in cls[c]) / sum(len(t["Nacc"]) for t in tasks if t["cls"] == c))) for c in CLASSES}

    # ---- PLD statistics per class (key (1,longest) = longest suffix match of any length >= 1; (3,recent) etc.)
    st = {}
    for c in CLASSES:
        ids = cls[c]
        row = {}
        for key in [(1, "longest"), (2, "recent"), (3, "recent"), (4, "recent")]:
            ell = np.concatenate([pts[i]["ell"][key] for i in ids])
            pl = np.concatenate([pts[i]["plen"][key] for i in ids])
            R7 = np.concatenate([pts[i]["R"][key][:, 7] for i in ids])
            R3 = np.concatenate([pts[i]["R"][key][:, 3] for i in ids])
            rm = np.concatenate([pts[i]["rm"] for i in ids])
            has = pl > 0
            row[f"{key[0]}{key[1][0]}"] = dict(positions=int(len(ell)), match_rate=float(has.mean()), mean_run_L7_given_match=float(R7[has].mean()) if has.any() else 0.0, mean_run_L3_given_match=float(R3[has].mean()) if has.any() else 0.0,
                                               mtp_run_at_matched=float(rm[has].mean()) if has.any() else 0.0, mtp_run_at_unmatched=float(rm[~has].mean()) if (~has).any() else 0.0)
        # distribution of the longest-suffix match length l
        ell = np.concatenate([pts[i]["ell"][(1, "longest")] for i in ids])
        pl = np.concatenate([pts[i]["plen"][(1, "longest")] for i in ids])
        R7 = np.concatenate([pts[i]["R"][(1, "longest")][:, 7] for i in ids])
        rm = np.concatenate([pts[i]["rm"] for i in ids])
        bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 7), (8, 15), (16, 31), (32, 63), (64, 999)]
        dist = []
        for lo, hi in bins:
            msk = (ell >= lo) & (ell <= hi) & (pl > 0)
            dist.append(dict(l=f"{lo}-{hi}" if lo != hi else str(lo), frac_positions=float(msk.mean()), n=int(msk.sum()), pld_run_L7=float(R7[msk].mean()) if msk.any() else None,
                             pld_hit_first=float((R7[msk] >= 1).mean()) if msk.any() else None, mtp_run=float(rm[msk].mean()) if msk.any() else None))
        none = pl == 0
        dist.append(dict(l="none", frac_positions=float(none.mean()), n=int(none.sum()), pld_run_L7=0.0, pld_hit_first=0.0, mtp_run=float(rm[none].mean()) if none.any() else None))
        row["l_dist"] = dist
        # P(run >= j)
        row["pld_run_ge"] = {f"{j}": float((R7[pl > 0] >= j).mean()) if (pl > 0).any() else 0.0 for j in range(1, 8)}
        row["pld_run_ge_l3"] = {f"{j}": float((R7[(ell >= 3) & (pl > 0)] >= j).mean()) if ((ell >= 3) & (pl > 0)).any() else 0.0 for j in range(1, 8)}
        row["mtp_run_ge"] = {f"{j}": float((rm >= j).mean()) for j in range(1, 4)}
        row["mtp_E"] = float(1 + rm.mean())
        row["p_match_l_ge"] = {f"{m}": float(((ell >= m) & (pl > 0)).mean()) for m in (1, 2, 3, 4, 6, 8, 16)}
        row["median_l_matched"] = float(np.median(ell[pl > 0])) if (pl > 0).any() else 0.0
        row["median_l_given_l_ge3"] = float(np.median(ell[(ell >= 3) & (pl > 0)])) if ((ell >= 3) & (pl > 0)).any() else 0.0
        st[c] = row
    S["pld_stats"] = st

    def report(name, res, **extra):
        r = {}
        for c in CLASSES + ["abc", "de", "all"]:
            ids = sum((cls[x] for x in c), []) if c in ("abc", "de") else (cls[c] if c in cls else [t["id"] for t in tasks])
            p = pooled(res, ids)
            p.update(ratio_boot(res, base, ids))
            r[c] = p
        S.setdefault("policies", {})[name] = dict(res=r, **extra)
        return r

    # ---- reference policies
    report("mtp_k3", base)
    for k in (1, 2):
        report(f"mtp_k{k}", run_all(tasks, pts, pol_mtp, V, k=k))
    report("plain_T1", run_all(tasks, pts, lambda pt, i, V, **kw: (1, 0, V[1]), V))
    # ---- primary hybrid H0 (pre-registered)
    H0 = dict(cfg=(4, 2, "recent"), m=3, L=7)
    report("H0_hybrid_4-2recent_m3_L7_A", run_all(tasks, pts, pol_hybrid, V, **H0), params=str(H0))
    report("H0_variantB_skip_drafts_ondevice", run_all(tasks, pts, pol_hybrid, V, skip=True, sync=0.0, **H0))
    report("H0_nosync", run_all(tasks, pts, pol_hybrid, V, sync=0.0, **H0))
    report("H0_L3_T4", run_all(tasks, pts, pol_hybrid, V, cfg=(4, 2, "recent"), m=3, L=3))
    report("H0_L5_T6", run_all(tasks, pts, pol_hybrid, V, cfg=(4, 2, "recent"), m=3, L=5))
    report("pld_only_4-2recent_m2_L7", run_all(tasks, pts, pol_pld_only, V, cfg=(4, 2, "recent"), m=2, L=7))
    report("pld_only_1longest_m1_L7", run_all(tasks, pts, pol_pld_only, V, cfg=(1, 1, "longest"), m=1, L=7))
    report("pld_only_3-3recent_m3_L7", run_all(tasks, pts, pol_pld_only, V, cfg=(3, 3, "recent"), m=3, L=7))
    report("ext_tail_4recent_m3", run_all(tasks, pts, pol_ext, V, ek=(4, "recent"), m=3, Lt=4))
    report("ext_tail_2recent_m2", run_all(tasks, pts, pol_ext, V, ek=(2, "recent"), m=2, Lt=4))
    report("ext_tail+H0", run_all(tasks, pts, pol_ext, V, ek=(4, "recent"), m=3, Lt=4, base_cfg=(4, 2, "recent"), bm=3, bL=7))
    report("oracle_best_of_PLD_or_MTP", run_all(tasks, pts, pol_oracle, V))

    # ---- grid (exploratory): (i) switch hybrids (PLD replaces the MTP chain), (ii) MTP chain + PLD tail
    cfgs = [(4, 2, "recent"), (4, 2, "longest"), (3, 2, "recent"), (3, 3, "recent"), (2, 2, "recent"), (4, 4, "recent"), (1, 1, "longest"), (4, 1, "longest"), (3, 1, "longest"), (2, 1, "longest")]
    ms = [1, 2, 3, 4, 6, 8, 12]
    Ls = [3, 5, 7]
    grid = []

    def add_row(kind, res, params):
        row = dict(kind=kind, **params)
        for c in CLASSES + ["abc", "de"]:
            ids = sum((cls[x] for x in c), []) if c in ("abc", "de") else cls[c]
            pr, pb = pooled(res, ids), pooled(base, ids)
            row[c] = pr["tok_s"] / pb["tok_s"]
            row[c + "_E"] = pr["E"]
        grid.append(dict(row, _res={i: list(v) for i, v in res.items()}))

    add_row("none", base, dict(cfg=[], m=0, L=0, variant="A"))  # MTP-only itself is always a candidate (the CV can decline to use PLD)
    for cfg in cfgs:
        for m in ms:
            for L in Ls:
                for skip in (False, True):
                    res = run_all(tasks, pts, pol_hybrid, V, cfg=cfg, m=m, L=L, skip=skip, sync=0.0 if skip else SYNC_HOST)
                    add_row("switch", res, dict(cfg=list(cfg), m=m, L=L, variant="B" if skip else "A"))
    for ek in [(4, "recent"), (3, "recent"), (2, "recent"), (4, "longest"), (2, "longest"), (1, "longest")]:
        for m in [1, 2, 3, 4, 6, 8]:
            for Lt in (1, 2, 3, 4):
                res = run_all(tasks, pts, pol_ext, V, ek=ek, m=m, Lt=Lt)
                add_row("ext", res, dict(cfg=list(ek), m=m, L=Lt, variant="A"))
    S["grid"] = [{k: v for k, v in r.items() if k != "_res"} for r in grid]

    # ---- 2-fold CV over tasks (tune the policy parameters on one fold by the mean of the 5 class ratios, evaluate on the other fold)
    folds = {}
    for c in CLASSES:
        for j, i in enumerate(cls[c]):
            folds[i] = j % 2

    def cv(rows, constrained=False):
        held, chosen = {}, []
        for f in (0, 1):
            tr = [i for i in folds if folds[i] != f]
            te = [i for i in folds if folds[i] == f]
            best, bv = None, -1
            for r in rows:
                vals = []
                for c in CLASSES:
                    ids = [i for i in cls[c] if i in tr]
                    vals.append(pooled(r["_res"], ids)["tok_s"] / pooled(base, ids)["tok_s"])
                sc = float(np.mean(vals))
                if constrained and (vals[3] < 0.995 or vals[4] < 0.995):  # "no loss" on the d / e classes of the TRAIN fold
                    continue
                if sc > bv:
                    bv, best = sc, r
            chosen.append(dict(fold_train=f, kind=best["kind"], cfg=best["cfg"], m=best["m"], L=best["L"], train_mean_ratio=bv))
            for i in te:
                held[i] = tuple(best["_res"][i])
        rcv = {}
        for c in CLASSES + ["abc", "de", "all"]:
            ids = sum((cls[x] for x in c), []) if c in ("abc", "de") else (cls[c] if c in cls else [t["id"] for t in tasks])
            p = pooled(held, ids)
            p.update(ratio_boot(held, base, ids))
            rcv[c] = p
        return rcv, chosen

    S["cv_chosen"] = {}
    for name, rows, con in [("CV_switch_hybrid_heldout", [r for r in grid if r["kind"] in ("switch", "none") and r["variant"] == "A"], False), ("CV_mtp_plus_pld_tail_heldout", [r for r in grid if r["kind"] in ("ext", "none")], False),
                            ("CV_switch_noloss_heldout", [r for r in grid if r["kind"] in ("switch", "none") and r["variant"] == "A"], True), ("CV_mtp_plus_pld_tail_noloss_heldout", [r for r in grid if r["kind"] in ("ext", "none")], True)]:
        rcv, chosen = cv(rows, con)
        S["policies"][name] = dict(res=rcv, chosen=chosen)
        S["cv_chosen"][name] = chosen

    # ---- post-hoc named cells (in-sample, labelled as such): the best no-loss ext cell of the grid
    E3 = dict(ek=(2, "longest"), m=3, Lt=4)
    report("POSTHOC_ext_tail_2longest_m3_L4", run_all(tasks, pts, pol_ext, V, **E3), params=str(E3))
    H0r = run_all(tasks, pts, pol_hybrid, V, **H0)
    E3r = run_all(tasks, pts, pol_ext, V, **E3)
    S["per_task"] = []
    for t in tasks:
        i = t["id"]
        b = base[i]
        pt = pts[i]
        S["per_task"].append(dict(id=i, cls=t["cls"], prompt=int(t["P"]), gen=int(len(t["g"])), E_mtp=b[0] / b[2], p_l_ge3=float(((pt["ell"][(1, "longest")] >= 3) & (pt["plen"][(1, "longest")] > 0)).mean()),
                                  H0_ratio=(H0r[i][0] / H0r[i][1]) / (b[0] / b[1]), E3_ratio=(E3r[i][0] / E3r[i][1]) / (b[0] / b[1])))
    ext_share = {}
    for c in CLASSES:
        n_tail, n_cyc, hist = 0, 0, np.zeros(LMAX + 2)
        for i in cls[c]:
            pt = pts[i]
            ii = 0
            while ii < pt["N"]:
                T, r, _ = pol_ext(pt, ii, V, **E3)
                n_tail += T != 4
                n_cyc += 1
                hist[min(r, LMAX + 1)] += 1
                ii += min(r + 1, pt["N"] - ii)
        ext_share[c] = dict(tail_cycles=n_tail / n_cyc, cycles=n_cyc, accepted_hist=[float(x) for x in hist / hist.sum()])
    S["ext_E3_share"] = ext_share
    sens_e = {}
    for name, kw in [("V8=24 (default 23.85)", dict(v8=None)), ("V8=26", dict(v8=26.0)), ("V8=28", dict(v8=28.0)), ("V8=30", dict(v8=30.0)), ("V8=34", dict(v8=34.0))]:
        Vs = make_V(**kw)
        b = run_all(tasks, pts, pol_mtp, Vs)
        for sync in (0.0, 0.2, 0.5):
            res = run_all(tasks, pts, pol_ext, Vs, sync=sync, **E3)
            sens_e[f"{name}, sync {sync}"] = {c: pooled(res, sum((cls[x] for x in c), []) if c in ("abc", "de") else cls[c])["tok_s"] / pooled(b, sum((cls[x] for x in c), []) if c in ("abc", "de") else cls[c])["tok_s"] for c in CLASSES + ["abc", "de"]}
    S["sensitivity_ext_E3"] = sens_e
    # PLD vs MTP at the SAME positions (positions with a >=3-token suffix match): P(run >= j)
    for c in CLASSES:
        ids = cls[c]
        ell = np.concatenate([pts[i]["ell"][(1, "longest")] for i in ids])
        pl = np.concatenate([pts[i]["plen"][(1, "longest")] for i in ids])
        R7 = np.concatenate([pts[i]["R"][(1, "longest")][:, 7] for i in ids])
        rm = np.concatenate([pts[i]["rm"] for i in ids])
        msk = (ell >= 3) & (pl > 0)
        S["pld_stats"][c]["same_positions_l_ge3"] = dict(n=int(msk.sum()), frac=float(msk.mean()), mtp_run_ge={str(j): float((rm[msk] >= j).mean()) for j in (1, 2, 3)}, pld_run_ge={str(j): float((R7[msk] >= j).mean()) for j in range(1, 8)},
                                                         mtp_E=float(1 + rm[msk].mean()), pld_E_L7=float(1 + R7[msk].mean()),
                                                         both_ge3_given_pld_ge3=float((rm[msk & (R7 >= 3)] >= 3).mean()) if (msk & (R7 >= 3)).any() else None)

    # ---- sensitivity of H0 (and ext) to the cost model
    sens = {}
    for name, kw in [("V8=24 (default 23.85)", dict(v8=None)), ("V8=26", dict(v8=26.0)), ("V8=28", dict(v8=28.0)), ("V8=30", dict(v8=30.0))]:
        Vs = make_V(**kw)
        b = run_all(tasks, pts, pol_mtp, Vs)
        for sync in (0.0, 0.2, 0.5):
            res = run_all(tasks, pts, pol_hybrid, Vs, sync=sync, **H0)
            sens[f"{name}, sync {sync}"] = {c: pooled(res, sum((cls[x] for x in c), []) if c in ("abc", "de") else cls[c])["tok_s"] / pooled(b, sum((cls[x] for x in c), []) if c in ("abc", "de") else cls[c])["tok_s"] for c in CLASSES + ["abc", "de"]}
    S["sensitivity_H0"] = sens
    # breakeven V(8): scale growth until pooled a-c ratio of H0 hits 1.0 / 1.10
    be = {}
    for target in (1.0, 1.05, 1.10):
        lo_, hi_ = 22.0, 80.0
        for _ in range(40):
            mid = (lo_ + hi_) / 2
            Vs = make_V(v8=mid)
            b = run_all(tasks, pts, pol_mtp, Vs)
            r_ = run_all(tasks, pts, pol_hybrid, Vs, **H0)
            q = pooled(r_, sum((cls[x] for x in "abc"), []))["tok_s"] / pooled(b, sum((cls[x] for x in "abc"), []))["tok_s"]
            if q > target:
                lo_ = mid
            else:
                hi_ = mid
        be[f"V8 where pooled a-c ratio = {target}"] = lo_
    S["breakeven"] = be
    be2 = {}
    abc = sum((cls[x] for x in "abc"), [])
    de = cls["d"] + cls["e"]
    for what, ids, target in [("E3 pooled a-c = 1.10", abc, 1.10), ("E3 pooled a-c = 1.00", abc, 1.00), ("E3 pooled d+e = 0.99", de, 0.99), ("E3 pooled d+e = 0.97", de, 0.97)]:
        lo_, hi_ = 22.0, 120.0
        for _ in range(40):
            mid = (lo_ + hi_) / 2
            Vs = make_V(v8=mid)
            b = run_all(tasks, pts, pol_mtp, Vs)
            r_ = run_all(tasks, pts, pol_ext, Vs, **E3)
            q = pooled(r_, ids)["tok_s"] / pooled(b, ids)["tok_s"]
            if q > target:
                lo_ = mid
            else:
                hi_ = mid
        be2[f"V8 where {what}"] = lo_
    S["breakeven_E3"] = be2
    # tokens/cycle table for H0 and share of cycles using PLD
    share = {}
    for c in CLASSES:
        n_pld, n_cyc, hist = 0, 0, np.zeros(LMAX + 2)
        for i in cls[c]:
            pt = pts[i]
            ii = 0
            while ii < pt["N"]:
                T, r, _ = pol_hybrid(pt, ii, V, **H0)
                n_pld += T != 4
                n_cyc += 1
                hist[min(r, LMAX + 1)] += 1
                ii += min(r + 1, pt["N"] - ii)
        share[c] = dict(pld_cycles=n_pld / n_cyc, cycles=n_cyc, accepted_hist=[float(x) for x in hist / hist.sum()])
    S["H0_pld_share"] = share
    # decision rule
    P = S["policies"]["H0_hybrid_4-2recent_m3_L7_A"]["res"]
    S["decision"] = dict(
        pooled_abc_ratio=P["abc"]["ratio"], pooled_abc_lo95=P["abc"]["lo95"],
        each_abc_ge_1p05={c: P[c]["ratio"] >= 1.05 for c in "abc"},
        d_e=dict(ratio_d=P["d"]["ratio"], lo95_d=P["d"]["lo95"], ratio_e=P["e"]["ratio"], lo95_e=P["e"]["lo95"]),
        recommend=bool(P["abc"]["ratio"] >= 1.10 and P["abc"]["lo95"] > 1.05 and all(P[c]["ratio"] >= 1.05 for c in "abc") and P["d"]["ratio"] >= 0.99 and P["e"]["ratio"] >= 0.99 and P["d"]["lo95"] >= 0.98 and P["e"]["lo95"] >= 0.98))
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_tables(S, cls)


def fmt_pol(S, names, cols=("a", "b", "c", "d", "e", "abc")):
    h = "| policy | " + " | ".join(CNAME.get(c, {"abc": "a-c pooled"}.get(c, c)) for c in cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for n in names:
        r = S["policies"][n]["res"]
        h += f"| {n} | " + " | ".join(f"{r[c]['tok_s']:.0f} tok/s, E {r[c]['E']:.2f}, x{r[c]['ratio']:.3f} [{r[c]['lo95']:.3f}, {r[c]['hi95']:.3f}]" for c in cols) + " |\n"
    return h


def write_tables(S, cls):
    L = ["# ngram-draft-potential: tables (generated by analyze.py; MODELLED tok/s = measured acceptance x modelled cycle time)\n"]
    L.append(f"V(T) ms: " + ", ".join(f"T={t}: {v:.2f}" for t, v in S["V"].items()) + f"; cost model {S['cost_model']}\n")
    L.append(f"MTP accepted count: prefix-match of the recorded drafts against the greedy continuation == the engine's natural histogram count at {100 * S['mtp_m_agreement']:.2f}% of positions\n")
    L.append("## MTP-only: simulated E (walk with per-position acceptance) vs the natural k=3 run (one cycle per sync)\n| class | E simulated from per-position acceptance | E natural run |\n|---|---|---|")
    for c in CLASSES:
        v = S["mtp_sim_vs_natural"][c]
        L.append(f"| {CNAME[c]} | {v['E_sim']:.3f} | {v['E_natural']:.3f} |")
    L.append("\n## Policies (pooled tokens / pooled modelled time per class; ratio over MTP-only k=3 with task-bootstrap 95% CI)\n")
    names = [n for n in S["policies"] if not n.startswith("CV_")]
    L.append(fmt_pol(S, names, ("a", "b", "c")))
    L.append(fmt_pol(S, names, ("d", "e", "abc")))
    cvn = [n for n in S["policies"] if n.startswith("CV_")]
    L.append("\n## Cross-validated tuned policies (2 folds over tasks; each task is evaluated with parameters tuned on the other half only)\n" + fmt_pol(S, cvn, ("a", "b", "c")) + "\n" + fmt_pol(S, cvn, ("d", "e", "abc")) + "\nchosen: " + json.dumps(S["cv_chosen"]) + "\n")
    L.append("\n## PLD statistics per class (longest suffix match of any length = key 1longest; positions = cycle starts of the greedy continuation)\n")
    L.append("| class | positions | P(match) 1L / 2R / 3R / 4R | mean accepted run L=7 given match (1L / 3R) | MTP run at matched / unmatched (1L) | P(PLD run>=1,2,3,4,5,6,7 | match 3R) | P(MTP run>=1,2,3) | MTP E |\n|---|---|---|---|---|---|---|---|")
    for c in CLASSES:
        s = S["pld_stats"][c]
        L.append(f"| {CNAME[c]} | {s['1l']['positions']} | {s['1l']['match_rate']:.2f} / {s['2r']['match_rate']:.2f} / {s['3r']['match_rate']:.2f} / {s['4r']['match_rate']:.2f} | {s['1l']['mean_run_L7_given_match']:.2f} / {s['3r']['mean_run_L7_given_match']:.2f} | {s['1l']['mtp_run_at_matched']:.2f} / {s['1l']['mtp_run_at_unmatched']:.2f} | "
                 + " ".join(f"{s['pld_run_ge'][str(j)]:.2f}" for j in range(1, 8)) + " | " + " ".join(f"{s['mtp_run_ge'][str(j)]:.2f}" for j in (1, 2, 3)) + f" | {s['mtp_E']:.2f} |")
    L.append("\n### MTP vs PLD at the SAME positions (positions where the longest suffix match has l >= 3): P(accepted run >= j)\n| class | share of positions | MTP j=1,2,3 | PLD (L=7) j=1..7 | E MTP / E PLD(L=7) | P(MTP run=3 | PLD run>=3) |\n|---|---|---|---|---|---|")
    for c in CLASSES:
        v = S["pld_stats"][c]["same_positions_l_ge3"]
        bg = v["both_ge3_given_pld_ge3"]
        L.append(f"| {CNAME[c]} | {v['frac']:.3f} | " + " ".join(f"{v['mtp_run_ge'][str(j)]:.2f}" for j in (1, 2, 3)) + " | " + " ".join(f"{v['pld_run_ge'][str(j)]:.2f}" for j in range(1, 8)) + f" | {v['mtp_E']:.2f} / {v['pld_E_L7']:.2f} | " + ("-" if bg is None else f"{bg:.2f}") + " |")
    L.append("\n### Distribution of the longest-suffix match length l (fraction of ALL positions; PLD L=7 mean accepted run and MTP mean accepted run at those positions)\n")
    for c in CLASSES:
        L.append(f"**{CNAME[c]}** (median l among matched {S['pld_stats'][c]['median_l_matched']:.0f}; among l>=3 {S['pld_stats'][c]['median_l_given_l_ge3']:.0f})\n")
        L.append("| l | " + " | ".join(d["l"] for d in S["pld_stats"][c]["l_dist"]) + " |\n|---|" + "---|" * len(S["pld_stats"][c]["l_dist"]))
        L.append("| fraction of positions | " + " | ".join(f"{d['frac_positions']:.3f}" for d in S["pld_stats"][c]["l_dist"]) + " |")
        L.append("| PLD mean run (L=7) | " + " | ".join("-" if d["pld_run_L7"] is None else f"{d['pld_run_L7']:.2f}" for d in S["pld_stats"][c]["l_dist"]) + " |")
        L.append("| MTP mean run (k=3) | " + " | ".join("-" if d["mtp_run"] is None else f"{d['mtp_run']:.2f}" for d in S["pld_stats"][c]["l_dist"]) + " |\n")
    L.append("\n## H0 per-cycle composition\n| class | share of cycles that used PLD | accepted-run histogram r=0..7,8+ |\n|---|---|---|")
    for c in CLASSES:
        s = S["H0_pld_share"][c]
        L.append(f"| {CNAME[c]} | {s['pld_cycles']:.3f} | " + " ".join(f"{x:.2f}" for x in s["accepted_hist"]) + " |")
    L.append("\n## Per task (modelled ratios over MTP-only at the same positions)\n| task | cls | prompt tok | new tok | MTP E (k=3) | P(l>=3 match) | H0 / MTP | E3 / MTP |\n|---|---|---|---|---|---|---|---|")
    for r in S["per_task"]:
        L.append(f"| {r['id']} | {r['cls']} | {r['prompt']} | {r['gen']} | {r['E_mtp']:.2f} | {r['p_l_ge3']:.2f} | {r['H0_ratio']:.3f} | {r['E3_ratio']:.3f} |")
    L.append("\n## Post-hoc cell E3 = MTP chain + PLD tail (n<=2 longest suffix, l>=3, tail <= 4, T<=8): per-cycle composition\n| class | share of cycles with a tail | accepted-run histogram r=0..7,8+ |\n|---|---|---|")
    for c in CLASSES:
        s_ = S["ext_E3_share"][c]
        L.append(f"| {CNAME[c]} | {s_['tail_cycles']:.3f} | " + " ".join(f"{x:.2f}" for x in s_["accepted_hist"]) + " |")
    L.append("\n## Sensitivity of E3 (ratio over MTP-only at the same V(T) table)\n| setting | a | b | c | d | e | a-c | d+e |\n|---|---|---|---|---|---|---|---|")
    for k, v in S["sensitivity_ext_E3"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in ["a", "b", "c", "d", "e", "abc", "de"]) + " |")
    L.append("\n## Sensitivity of H0 (ratio over MTP-only, a-e and a-c pooled / d-e pooled) to V(8) and the host-sync penalty\n| setting | a | b | c | d | e | a-c | d+e |\n|---|---|---|---|---|---|---|---|")
    for k, v in S["sensitivity_H0"].items():
        L.append(f"| {k} | " + " | ".join(f"{v[c]:.3f}" for c in ["a", "b", "c", "d", "e", "abc", "de"]) + " |")
    L.append("\nBreak-even (H0): " + json.dumps(S["breakeven"]) + "\n\nBreak-even (E3): " + json.dumps(S["breakeven_E3"]) + "\n")
    L.append("## Decision rule evaluation\n```\n" + json.dumps(S["decision"], indent=1) + "\n```\n")
    # grid top cells
    for kind in ("switch", "ext"):
        g = sorted([r for r in S["grid"] if r["variant"] == "A" and r["kind"] == kind], key=lambda r: -np.mean([r[c] for c in CLASSES]))[:12]
        L.append(f"\n## Exploratory grid `{kind}`, variant A (host sync 0.2 ms), top 12 cells by mean of the 5 class ratios (IN-SAMPLE, optimistic; see the CV rows for the honest number)\n| cfg (nmax,nmin,strategy) / (n,strategy) | m | L (tail) | a | b | c | d | e | a-c |\n|---|---|---|---|---|---|---|---|---|")
        for r in g:
            L.append(f"| {tuple(r['cfg'])} | {r['m']} | {r['L']} | " + " | ".join(f"{r[c]:.3f}" for c in ["a", "b", "c", "d", "e", "abc"]) + " |")
    (RES / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
