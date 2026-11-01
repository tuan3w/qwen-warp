#!/usr/bin/env python3
"""check.py <result.json> <tree> <out.json>: correctness of a candidate `dev` result against the pristine control (no GPU; judges the data metric.py recorded).

The data are teacher-forced on the PRISTINE greedy continuation of the 3 oracle prompts (private/tf_tokens.json), so free-run chaos is irrelevant:
  1. per position top-1 agreement and top-20 logit deviation vs the control run of the same mode (T=8 = widest hybrid verify step, T=4 chain verify rows and T=1 plain decode, on the
     tmax=8 shipped engine; T=4 / T=1 again on the tmax=4 library-default engine = 5-slot ring, subprocess); a disagreement is excused only if it is a near-tie of the control
     (top-1 = control's top-2, control margin < near_tie); deviation limits calibrated on legit rounding-level changes (adapter/limits.json, README)
  2. greedy speculative generations == the pristine continuation, near-tie divergences excused: chain k=2,3 natural + forced partial acceptance, chain + prompt-lookup tail natural and with
     forced acceptance of up to 8 rows (rollback of the 9-slot ring, ring wrap), on short / 8K / copy-heavy prompts; chain-only on the tmax=4 engine
  3. sampling: the engine's device sampler (top-k 20, top-p .95, seeded) at 7 fixed positions (rows up to 7 of a T=8 step), TV(empirical, analytic) from its own logits and from the control's logits
  4. kernel-level exact speculative sampling (accept / residual / bonus) vs the analytic law, chain k=3 and chain 3 + tail 4
Control = the gate's bracketing pristine runs (g-ctrlA/B next to the result) or, for `try`, baseline/main-dev1.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hlib  # noqa: E402


def devstats(xi, xv, ci, cv, near):
    """xi/xv candidate, ci/cv control: [pos][32] sorted top ids / logits.  -> dict(agree, unexcused, deltas)."""
    n = len(xi)
    agree = (xi[:, 0] == ci[:, 0])
    marg = cv[:, 0] - cv[:, 1]
    excused = (~agree) & (xi[:, 0] == ci[:, 1]) & (marg < near)
    deltas = []
    for i in range(n):
        m = dict(zip(xi[i].tolist(), xv[i].tolist()))
        for j in range(20):
            t = int(ci[i, j])
            if t in m:
                deltas.append(abs(m[t] - float(cv[i, j])))
    d = np.array(deltas)
    return dict(n=n, agree=float(agree.mean()), unexcused=int((~agree & ~excused).sum()), med=float(np.median(d)), p99=float(np.percentile(d, 99)), max=float(d.max()), nd=len(d))


def probe_tv(X, C, pi):
    sup, pp, cnt = X[f"probe{pi}_ids"], X[f"probe{pi}_p"], X[f"probe{pi}_cnt"]
    n = int(cnt.sum() + X[f"probe{pi}_out"])
    self_tv = hlib.tv(np.append(cnt / n, X[f"probe{pi}_out"] / n), np.append(pp, 0.0))
    cs, cp = C[f"probe{pi}_ids"], C[f"probe{pi}_p"]
    union = sorted(set(sup.tolist()) | set(cs.tolist()))
    em = dict(zip(sup.tolist(), (cnt / n).tolist()))
    pc = dict(zip(cs.tolist(), cp.tolist()))
    ctrl_tv = hlib.tv([em.get(t, 0.0) for t in union], [pc.get(t, 0.0) for t in union]) + 0.5 * float(X[f"probe{pi}_out"]) / n
    return self_tv, ctrl_tv, int(X[f"probe{pi}_out"])


def main():
    result, tree, out = sys.argv[1:4]
    L = hlib.limits()
    facts, bad = [], []
    R = json.load(open(result))
    ctrls = [p for p in hlib.control_results(result) if hlib.tf_path(p).exists()]
    if R.get("kind") == "profile":
        json.dump({"ok": True, "detail": "profile run: no correctness data", "facts": []}, open(out, "w"))
        return
    if R.get("kind") != "dev" or not hlib.tf_path(result).exists() or not ctrls:
        json.dump({"ok": False, "detail": f"no correctness data (result {result}; controls found: {[str(c) for c in ctrls]})", "facts": []}, open(out, "w"))
        return
    X = np.load(hlib.tf_path(result))
    Cs = [np.load(hlib.tf_path(c)) for c in ctrls]
    C, refs = Cs[0], hlib.jload(hlib.PRIV / "tf_tokens.json")
    near = L["near_tie"]
    # 1. teacher-forced logits
    for mode, pre, names in [(m, "", hlib.TF_NAMES) for m, _ in hlib.TF_MODES] + [("t4", "m4_", hlib.ORACLE_NAMES), ("t1", "m4_", hlib.ORACLE_NAMES)]:
        if pre and f"{pre}{names[0]}_{mode}_id" not in X.files:  # `try` skips the tmax=4 subprocess: the gate has it
            facts.append([f"teacher-forced tmax4 {mode}", "not run in `try` (gate only)"])
            continue
        lim = L[mode]
        rows = []
        for name in names:
            if not (np.isfinite(X[f"{pre}{name}_{mode}_val"]).all()):
                bad.append(f"non-finite logits {pre}{name} {mode}")
            rows.append(devstats(X[f"{pre}{name}_{mode}_id"], X[f"{pre}{name}_{mode}_val"], C[f"{pre}{name}_{mode}_id"], C[f"{pre}{name}_{mode}_val"], near))
        agree = sum(r["agree"] * r["n"] for r in rows) / sum(r["n"] for r in rows)
        un = sum(r["unexcused"] for r in rows)
        med, p99, mx = max(r["med"] for r in rows), max(r["p99"] for r in rows), max(r["max"] for r in rows)
        tag = f"{'tmax4 ' if pre else ''}{mode}"
        facts.append([f"teacher-forced {tag}", f"top-1 {agree:.4f} (unexcused flips {un}); top-20 |dlogit| median {med:.4f} p99 {p99:.3f} max {mx:.3f}"])
        if un > lim["unexcused_max"]:
            bad.append(f"{tag}: {un} unexcused top-1 flips (> {lim['unexcused_max']})")
        if agree < lim["agree_min"]:
            bad.append(f"{tag}: top-1 agreement {agree:.4f} < {lim['agree_min']}")
        if med > lim["med_max"] or p99 > lim["p99_max"] or mx > lim["max_max"]:
            bad.append(f"{tag}: logit deviation median {med:.4f}/p99 {p99:.3f}/max {mx:.3f} over limits {lim['med_max']}/{lim['p99_max']}/{lim['max_max']}")
    if len(Cs) > 1:  # control A vs control B = the harness' own noise (0 if the engine is bit-reproducible)
        rr = [devstats(Cs[1][f"{n}_t4_id"], Cs[1][f"{n}_t4_val"], C[f"{n}_t4_id"], C[f"{n}_t4_val"], near) for n in hlib.TF_NAMES]
        facts.append(["control A vs B (T=4)", f"max |dlogit| {max(r['max'] for r in rr):.4f}, flips {sum(int(round((1 - r['agree']) * r['n'])) for r in rr)}"])
    # 2. greedy spec generations
    nbad = 0
    worst = []
    gens = dict(R["spec_gens"])
    gens.update({"m4:" + k: v for k, v in R.get("spec_gens_m4", {}).items()})
    for key, toks in gens.items():
        name = key.removeprefix("m4:").split("/")[0]
        ref = refs[name]["cont"][: len(toks)]
        if toks == ref:
            continue
        i = next(j for j in range(min(len(ref), len(toks))) if ref[j] != toks[j])
        tops = [(C[f"{name}_{m}_id"][i], C[f"{name}_{m}_val"][i]) for m, _ in hlib.TF_MODES]
        ok1 = toks[i] in {int(t[1]) for t, _ in tops} and min(v[0] - v[1] for _, v in tops) < near
        worst.append(f"{key}@{i}{'(near-tie)' if ok1 else '(WRONG)'}")
        nbad += not ok1
    facts.append(["greedy spec == pristine (chain k=2,3; chain + tail; forced rollback up to 8 rows; tmax=4 engine)", f"{len(gens) - len(worst)}/{len(gens)} equal" + (f"; divergences {worst}" if worst else "")])
    if nbad:
        bad.append(f"speculative greedy diverges from the pristine continuation: {worst}")
    # 3. sampler probes
    st_, ct_, outs = [], [], 0
    for pi in range(len(hlib_probes(X))):
        s, c, o = probe_tv(X, C, pi)
        st_.append(s); ct_.append(c); outs += o
    facts.append([f"sampler TV ({len(st_)} positions, 20000 draws)", f"vs own logits max {max(st_):.4f}; vs control logits max {max(ct_):.4f}; outside support {outs}"])
    if max(st_) > L["probe_self_tv_max"] or outs:
        bad.append(f"sampler vs its own logits TV {max(st_):.4f} (limit {L['probe_self_tv_max']}) / outside support {outs}")
    if max(ct_) > L["probe_ctrl_tv_max"]:
        bad.append(f"sampler vs control distribution TV {max(ct_):.4f} > {L['probe_ctrl_tv_max']}")
    # 4. kernel-level speculative sampling
    T_ = R["trials"]
    facts.append(["speculative sampling (300k cycles)", f"TV y0 {T_['tv_y0']:.4f}, TV emitted-length {T_['tv_len']:.4f}, E[len] {T_['mean_len']:.4f} vs {T_['mean_len_expected']:.4f}, outside {T_['outside']}"])
    if T_["tv_y0"] > L["trials_tv_max"] or T_["tv_len"] > L["trials_tv_max"] or T_["outside"] or abs(T_["mean_len"] - T_["mean_len_expected"]) > L["trials_len_tol"]:
        bad.append(f"speculative sampling off its analytic law: {T_}")
    TP = R["trials_pld"]
    facts.append(["speculative sampling, chain 3 + tail 4 (300k cycles)", f"TV y0 {TP['tv_y0']:.4f}, TV emitted-length {TP['tv_len']:.4f}, E[len] {TP['mean_len']:.4f} vs {TP['mean_len_expected']:.4f}, outside {TP['outside']}; tail: accepted != d_t {TP['tail_stray']}, d_t emitted at a rejection {TP['tail_viol']}, rejection TV {TP['tail_rej_tv']:.4f}, bonus TV {TP['bonus_tv']:.4f}"])
    if (TP["tv_y0"] > L["trials_tv_max"] or TP["tv_len"] > L["trials_tv_max"] or TP["outside"] or abs(TP["mean_len"] - TP["mean_len_expected"]) > L["trials_len_tol"] or TP["tail_stray"] or TP["tail_viol"]
            or TP["tail_rej_tv"] > L["trials_pld_tv_max"] or TP["bonus_tv"] > L["trials_pld_tv_max"]):
        bad.append(f"chain + prompt-lookup tail sampling off its analytic law: {TP}")
    # 5. the engine's own op tests (pristine copy of tests/test_ops.py only)
    S = R.get("suites", {})
    to = S.get("test_ops")
    if isinstance(to, dict):
        facts.append(["tests/test_ops.py", to["last"]])
        if not to["ok"]:
            bad.append(f"tests/test_ops.py failed: {to['last']} {to['fails']} {to['stderr']}")
    if S.get("edited"):
        facts.append(["engine tests edited by the candidate (test_ops skipped if edited)", ", ".join(S["edited"])])
    json.dump({"ok": not bad, "detail": "; ".join(bad) if bad else "all correctness checks passed", "facts": facts}, open(out, "w"))


def hlib_probes(X):
    return sorted({int(k[5:].split("_")[0]) for k in X.files if k.startswith("probe")})


if __name__ == "__main__":
    main()
