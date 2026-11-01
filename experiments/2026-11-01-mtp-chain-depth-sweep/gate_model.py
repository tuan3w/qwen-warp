"""Offline model of the confidence gate from the per-cycle logs of the full-width (K = 7, no tail) runs + the measured cycle-cost grid.  CPU only.

  python3 gate_model.py --logdir runs/logs --bench results/bench_k7_log.json --grid results/cost_grid.json --T 0 [--variants g5,g6,g7,g5a3,g7a3,g7a4] [--folds 4] [--out results/gate_model_T0]

Model (renewal-reward over the logged cycles; the logged cycles are the natural cycles of the K = 7 run, so the trajectory shift of a gated run is ignored - measured online instead):
  a cycle with confidences c_1..c_7 (raw softmax max-prob of the draft head, 'raw'; or the max prob of the processed proposal q, 'q') and m accepted drafts (m = a - 1, full width 7);
  gate: verify width j* = number of leading depths with c_i >= tau_i (<= K); tokens = min(m, j*) + 1; time = C(Kdraft, j*) = the measured graph of chain depth Kdraft and verify width j*.
  fixed k: j* = k, Kdraft = k.  Adaptive depth a<L>: the chain drafts K next cycle iff the gate kept >= L drafts this cycle, else L (cycles evaluated in log order).
tau is fitted by coordinate descent on the training folds (tasks split into folds, class-balanced round robin) and applied to the held-out fold = cross-fit by task.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

K_ALL = 7
EMA_ALPHA = 0.6
CLASS_ORDER = ["code", "prose", "math", "long8k", "a", "b", "c", "d", "e", "en"]


def load(logdir, bench, cfg, T):
    rows = json.loads(Path(bench).read_text())["rows"]
    seen = {}
    for r in rows:
        seen.setdefault((r["suite"], r["task"]), r)
    tasks = {}
    for (suite, tid), r in seen.items():
        p = Path(logdir) / f"{suite}_{tid}_{cfg}_T{T:g}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        li, lf = z["li"].astype(np.int32), z["lf"].astype(np.float32)
        m_ = li[:, 0] - 1
        ema = {}
        for al in (EMA_ALPHA,):
            e_, prev = np.zeros(len(m_)), 3.0
            for i in range(len(m_)):
                e_[i] = prev  # value before cycle i's launch (m_0..m_{i-1})
                prev = al * prev + (1 - al) * m_[i]
            ema[al] = e_
        tasks[(suite, tid)] = dict(ema=ema, cls=r["cls"], cat=r["cat"], ctx=r["prompt"] + r["n"] / 2, m=li[:, 0] - 1, raw=lf[:, :7], q=lf[:, 8:15], nd=li[:, 3], tail=li[:, 2])
    return tasks


class Cost:
    def __init__(self, grid_path):
        g = json.loads(Path(grid_path).read_text())
        g = g.get("grid", g)
        self.ctx = sorted((int(k[3:]), k) for k in g)
        self.mat = {}
        for c, k in self.ctx:
            M = np.full((8, 8), np.nan)
            for kc in range(1, 8):
                for kv in range(0, 8):
                    M[kc, kv] = g[k][f"cycle_kc{kc}_kv{kv}"]
            self.mat[c] = M

    def at(self, ctx):
        cs = [c for c, _ in self.ctx]
        if ctx <= cs[0]:
            return self.mat[cs[0]]
        if ctx >= cs[-1]:
            return self.mat[cs[-1]]
        for lo, hi in zip(cs[:-1], cs[1:]):
            if lo <= ctx <= hi:
                w = (ctx - lo) / (hi - lo)
                return (1 - w) * self.mat[lo] + w * self.mat[hi]


def gate_width(conf, tau, K):
    """conf [n,7] -> j* [n] = leading depths with conf_i >= tau_i, capped at K."""
    ok = (conf[:, :K] >= np.asarray(tau[:K])[None, :]).astype(np.int32)
    return np.cumprod(ok, axis=1).sum(1)


def sim_task(t, C, policy, use="raw"):
    """-> (tokens, ms) summed over the cycles of one task.  policy: ('fixed', k) | ('gate', K, tau, rule) with rule None (fixed draft depth K) |
    ('kept', L, lo): next cycle drafts K iff the width gate kept >= L drafts this cycle, else lo (<= lo drafts exist, L <= lo) |
    ('rho', rho, L, lo): next cycle drafts K iff the first L confidences of this cycle are all >= rho, else lo"""
    m = t["m"]
    if policy[0] == "oracle":  # upper bound of a width gate: verify exactly the drafts that will be accepted (knows m), drafting depth K
        K = policy[1]
        j = np.minimum(m, K)
        return float((j + 1).sum()), float(C[K, j].sum())
    if policy[0] == "fixed":
        k = policy[1]
        return float((np.minimum(m, k) + 1).sum()), float(C[k, k] * len(m))
    _, K, tau, rule = policy
    conf = t[use]
    j = gate_width(conf, tau, K)
    if rule is None:
        return float((np.minimum(m, j) + 1).sum()), float(C[K, j].sum())
    if rule[0] == "kept":
        _, L, lo = rule
        deep = j >= L
    elif rule[0] == "rho":
        _, rho, L, lo = rule
        deep = (conf[:, :L] >= rho).all(axis=1)
    else:  # ema of the accepted drafts of the previous cycles (known at launch): deep iff ema_before >= theta
        _, theta, alpha, lo = rule
        deep = t["ema"][alpha] >= theta
    kn = np.where(deep, K, lo)
    nd = np.concatenate([[K], kn[:-1]])  # drafts available to cycle i = the depth chosen by cycle i-1
    j = np.minimum(j, nd)
    return float((np.minimum(m, j) + 1).sum()), float(C[kn, j].sum())


def class_rates(tasks, mats, policy, use, keys):
    acc = {}
    for k in keys:
        t = tasks[k]
        tok, ms = sim_task(t, mats[k], policy, use)
        a = acc.setdefault(t["cls"], [0.0, 0.0])
        a[0] += tok
        a[1] += ms
    return {c: v[0] / v[1] * 1e3 for c, v in acc.items()}


def objective(rates, base, mode):
    lr = [math.log(rates[c] / base[c]) for c in rates]
    if mode == "mean":
        return sum(lr) / len(lr)
    return sum(lr) / len(lr) + 5.0 * min(0.0, min(lr))  # 'safe': a loss in any class costs 5x


def make_policy(K, p, rule):
    """p = tau_1..tau_7 (+ rho at index 7 for the rho rule)"""
    if rule is None:
        return ("gate", K, p[:7], None)
    if rule[0] == "kept":
        return ("gate", K, p[:7], rule)
    if rule[0] == "ema":
        return ("gate", K, p[:7], ("ema", p[7], EMA_ALPHA, rule[2]))
    return ("gate", K, p[:7], ("rho", p[7], rule[1], rule[2]))


def fit_tau(tasks, mats, keys, K, rule, use, mode, grid=None):
    """coordinate descent on tau_1..tau_K (and rho) over the training tasks `keys`; objective = mean over classes of log(rate / rate of fixed k = 3)."""
    grid = grid if grid is not None else [0.0] + [round(x, 3) for x in np.arange(0.1, 1.0, 0.05)] + [0.97, 0.99, 1.01]
    base = class_rates(tasks, mats, ("fixed", 3), use, keys)
    ema_grid = [0.5 * i for i in range(0, 15)]
    p = [0.0] * 7 + [0.5]
    idxs = list(range(K)) + ([7] if rule is not None and rule[0] in ("rho", "ema") else [])
    if rule is not None and rule[0] == "ema":
        p[7] = 3.0
    best = objective(class_rates(tasks, mats, make_policy(K, p, rule), use, keys), base, mode)
    for sweep in range(3):
        moved = False
        for j in idxs:
            for g in (ema_grid if (j == 7 and rule[0] == "ema") else grid):
                if g == p[j]:
                    continue
                p2 = list(p)
                p2[j] = g
                o = objective(class_rates(tasks, mats, make_policy(K, p2, rule), use, keys), base, mode)
                if o > best + 1e-9:
                    best, p, moved = o, p2, True
        if not moved:
            break
    return p, best


def folds_of(tasks, nf):
    """class-balanced round robin over the sorted task ids"""
    fo = {}
    cnt = {}
    for k in sorted(tasks):
        c = tasks[k]["cls"]
        fo[k] = cnt.get(c, 0) % nf
        cnt[c] = cnt.get(c, 0) + 1
    return fo


def parse_variant(v):
    # g5 : gate, drafting depth 5.  g5a3 : + adaptive depth (5 iff the width gate kept >= 3 drafts, else 3).  g5a3r : + adaptive depth on the first 3 confidences all >= rho (rho fitted).  g5a2r / g7a3r ...
    import re

    m = re.match(r"^g(\d)(a(\d)([re]?))?$", v)
    K = int(m.group(1))
    if not m.group(2):
        return K, None
    L = int(m.group(3))
    return K, ({"r": ("rho", L, L), "e": ("ema", L, L), "": ("kept", L, L)}[m.group(4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default="runs/logs")
    ap.add_argument("--bench", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--T", type=float, default=0.0)
    ap.add_argument("--cfg", default="k7")
    ap.add_argument("--variants", default="g4,g5,g6,g7,g5a3,g7a3,g7a4")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--use", default=None, help="raw | q (default: raw for greedy, q for sampling)")
    ap.add_argument("--modes", default="mean,safe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    use = a.use or ("q" if a.T > 0 else "raw")
    tasks = load(a.logdir, a.bench, a.cfg, a.T)
    C = Cost(a.grid)
    mats = {k: C.at(t["ctx"]) for k, t in tasks.items()}
    fo = folds_of(tasks, a.folds)
    keys_all = sorted(tasks)
    res = dict(T=a.T, use=use, n_tasks=len(tasks), classes={c: sum(1 for t in tasks.values() if t["cls"] == c) for c in CLASS_ORDER})
    # per-position acceptance (unconditional prefix P(m >= j)) and calibration of the confidence
    pos = {}
    for c in CLASS_ORDER:
        m = np.concatenate([t["m"] for t in tasks.values() if t["cls"] == c]) if any(t["cls"] == c for t in tasks.values()) else np.zeros(0, int)
        if len(m):
            pos[c] = dict(cycles=int(len(m)), P=[float((m >= j).mean()) for j in range(1, 8)], E7=float(m.mean() + 1))
    res["position_acceptance_full_chain"] = pos
    # AUROC of conf_j for 'draft j accepted | drafts 1..j-1 accepted'
    def auroc(x, y):
        o = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x))
        ranks[o] = np.arange(1, len(x) + 1)
        n1, n0 = y.sum(), len(y) - y.sum()
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")

    cal = {}
    for j in range(1, 8):
        xs, ys = [], []
        for t in tasks.values():
            sel = t["m"] >= j - 1
            xs.append(t[use][sel, j - 1])
            ys.append((t["m"][sel] >= j).astype(int))
        x, y = np.concatenate(xs), np.concatenate(ys)
        bins = [0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
        cal[j] = dict(n=int(len(x)), base_rate=float(y.mean()), auroc=auroc(x, y), bins=[dict(lo=bins[i], hi=bins[i + 1], n=int(((x >= bins[i]) & (x < bins[i + 1])).sum()),
                      acc=float(y[(x >= bins[i]) & (x < bins[i + 1])].mean()) if ((x >= bins[i]) & (x < bins[i + 1])).any() else None) for i in range(len(bins) - 1)])
    res["calibration"] = cal
    # fixed k modelled from the K=7 log
    res["fixed"] = {}
    for k in range(1, 8):
        r = class_rates(tasks, mats, ("fixed", k), use, keys_all)
        res["fixed"][k] = r
    base = res["fixed"][3]
    res["oracle_width"] = {k: {c: r / base[c] for c, r in class_rates(tasks, mats, ("oracle", k), use, keys_all).items()} for k in (4, 5, 6, 7)}
    # gate variants: cross-fit
    res["gate"] = {}
    for mode in a.modes.split(","):
        for v in a.variants.split(","):
            K, rule = parse_variant(v)
            taus = {}
            pooled = {}
            for f in range(a.folds):
                train = [k for k in keys_all if fo[k] != f]
                test = [k for k in keys_all if fo[k] == f]
                tau, obj = fit_tau(tasks, mats, train, K, rule, use, mode)
                taus[f] = tau
                for k in test:
                    tok, ms = sim_task(tasks[k], mats[k], make_policy(K, tau, rule), use)
                    p = pooled.setdefault(tasks[k]["cls"], [0.0, 0.0])
                    p[0] += tok
                    p[1] += ms
            rates = {c: pooled[c][0] / pooled[c][1] * 1e3 for c in pooled}
            # also the in-sample fit on all tasks (upper bound)
            tau_all, _ = fit_tau(tasks, mats, keys_all, K, rule, use, mode)
            ins = class_rates(tasks, mats, make_policy(K, tau_all, rule), use, keys_all)
            res["gate"][f"{v}/{mode}"] = dict(crossfit_rate=rates, crossfit_ratio={c: rates[c] / base[c] for c in rates}, insample_ratio={c: ins[c] / base[c] for c in ins}, tau_folds=taus, tau_all=tau_all)
            # per-task tau for the online run (the fold model that did NOT see the task)
            Path(f"{a.out}_tau_{v}_{mode}.json").write_text(json.dumps({k[1]: taus[fo[k]] for k in keys_all}))
            print(f"{v}/{mode}: " + " ".join(f"{c} {rates[c] / base[c]:.3f}" for c in CLASS_ORDER if c in rates), flush=True)
    res["fixed_ratio"] = {k: {c: res["fixed"][k][c] / base[c] for c in base} for k in res["fixed"]}
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, default=float))
    # markdown
    L = [f"# gate model (T={a.T:g}, confidence = {use}, {len(tasks)} tasks, {a.folds}-fold cross-fit by task)", "", "## per-position acceptance P(m >= j) of the full chain (K=7 log), j = 1..7", "", "| class | cycles | " + " | ".join(f"a{j}" for j in range(1, 8)) + " | E (k=7) |", "|---|---|" + "---|" * 8]
    for c in CLASS_ORDER:
        if c in pos:
            L.append(f"| {c} | {pos[c]['cycles']} | " + " | ".join(f"{x:.3f}" for x in pos[c]["P"]) + f" | {pos[c]['E7']:.2f} |")
    L += ["", "## confidence calibration: P(draft j accepted | drafts < j accepted) by conf bin, AUROC", "", "| j | n | base rate | AUROC | " + " | ".join(f"[{b['lo']},{b['hi']})" for b in cal[1]["bins"]) + " |", "|---|---|---|---|" + "---|" * 6]
    for j in range(1, 8):
        c = cal[j]
        L.append(f"| {j} | {c['n']} | {c['base_rate']:.3f} | {c['auroc']:.3f} | " + " | ".join((f"{b['acc']:.2f} ({b['n']})" if b["acc"] is not None else "-") for b in c["bins"]) + " |")
    L += ["", "## modelled tok/s ratio vs fixed k = 3 (same cycles), per class", "", "| policy | " + " | ".join(c for c in CLASS_ORDER if c in pos) + " |", "|---|" + "---|" * len(pos)]
    for k in range(4, 8):
        L.append(f"| fixed k={k} | " + " | ".join(f"{res['fixed_ratio'][k][c]:.3f}" for c in CLASS_ORDER if c in pos) + " |")
    for k in (4, 5, 6, 7):
        L.append(f"| oracle width gate, draft depth {k} (upper bound, knows m) | " + " | ".join(f"{res['oracle_width'][k][c]:.3f}" for c in CLASS_ORDER if c in pos) + " |")
    for key, g in res["gate"].items():
        L.append(f"| gate {key} (cross-fit) | " + " | ".join(f"{g['crossfit_ratio'][c]:.3f}" for c in CLASS_ORDER if c in pos) + " |")
        L.append(f"| gate {key} (in-sample) | " + " | ".join(f"{g['insample_ratio'][c]:.3f}" for c in CLASS_ORDER if c in pos) + " |")
    Path(a.out + ".md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
