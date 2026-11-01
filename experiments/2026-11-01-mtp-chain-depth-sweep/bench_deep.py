"""Deep-chain benchmark (engine-direct, one process, configs interleaved per task, lease held).

  SMK_BUILD=runs/lib/build ~/llm-lab/venv/bin/python bench_deep.py --suites e0,e6,e8 --configs base,k3,k4,k5,k6,k7,k3t1,k4t1,k5t1,k6t1,k5t2 [--temperature 0|1.0] [--reps 3 --reps-e8 1]
        [--gate-file results/gate_tau_<m>.json] [--json results/<name>.json] [--logdir runs/logs] [--cost-grid results/cost_grid.json] [--tasks a1,c3] [--limit-e8 N]

Config grammar:  base = the committed async k=3 cycles (smk_spec_run);  k<K> = deep driver, chain depth K, verify all K drafts, no tail;  k<K>t1 = + prompt-lookup tail after the chain (<= 7 - K tokens);
  k<K>t2 = tail priority (try chain 3 + tail 4 first, deep chain only when the proposer has no match);  g<K>[a<L>][t1|t2] = confidence gate (per-depth tau from --gate-file, cross-fit per task),
  a<L> = adaptive draft depth (K if the gate kept >= L drafts, else L), a<L>r = K iff the first L confidences are all >= rho (rho = 8th value of the gate file), a<L>e = K iff the EMA (0.6) of accepted chain drafts >= theta (8th value of the gate file), tail as above.  Tasks: E0 prompts (384 new tokens), E6's 32 traces (stop at the canonical greedy length / EOS),
  E8's 48 English chat tasks (task set not included in this release) (runs/parta of experiments/e8-task-set-not-released; stop at the canonical length / EOS).
Rate = tokens emitted up to the target / decode wall time (prefill + the first MTP stage excluded), as tests/bench_pld.py.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from deep import *  # noqa: F401,F403

REPO = HERE.parents[1]
E6 = REPO / "experiments" / "2026-10-31-ngram-draft-potential" / "runs" / "trace"
E8 = REPO / "experiments" / "e8-task-set-not-released" / "runs" / "parta"
EOS = {248046, 248044}


def load_tasks(suites, only, limit_e8):
    tasks = []
    if "e0" in suites:
        from bench_spec import prompts

        for n, ids in prompts(["code", "prose", "math", "long8k"]).items():
            tasks.append(dict(id=n, cls=n, suite="e0", cat=n, ids=ids, n=384, eos=False))
    if "e6" in suites:
        for fn in sorted(E6.glob("*.json")):
            tid = fn.name.split("_")[0]
            r = json.loads(fn.read_text())
            tasks.append(dict(id=tid, cls=tid[0], suite="e6", cat=tid[0], ids=r["prompt_ids"], n=len(r["F"]["tokens"]), eos=True))
    if "e8" in suites:
        fs = sorted(E8.glob("*.json"))
        if limit_e8:
            fs = fs[: limit_e8]
        for fn in fs:
            r = json.loads(fn.read_text())
            lang = r["lang"]
            tasks.append(dict(id=fn.stem, cls=lang, suite="e8", cat=r["cat"], ids=r["prompt"], n=len(r["F"]["tokens"]), eos=True))
    if only:
        tasks = [t for t in tasks if any(t["id"] == o or t["id"].startswith(o + "_") for o in only)]
    return tasks


CFG_RE = re.compile(r"^(k|g)(\d)(a(\d)([re]?))?(t([12]))?$")


def parse_cfg(name, temp, tau):
    """-> dict(name, base, opts, tail)"""
    if name == "base":
        return dict(name=name, base=True, kc=3, tail=False, opts=None)
    m = CFG_RE.match(name)
    assert m, name
    gate = m.group(1) == "g"
    kc = int(m.group(2))
    kc_lo = int(m.group(4)) if m.group(4) else kc
    rho = m.group(5) in ("r", "e")
    ema = m.group(5) == "e"
    tl = int(m.group(7)) if m.group(7) else 0
    return dict(name=name, base=False, kc=kc, tail=tl > 0, gate=gate, kc_lo=kc_lo, tl=tl, use_q=1 if temp > 0 else 0, rho=rho, ema=ema)


def make_cfg_opts(c, tau):
    return make_opts(c["kc"], kc_lo=c["kc_lo"], adapt_thr=c["kc_lo"] if c["kc_lo"] < c["kc"] else 0, gate=1 if c["gate"] else 0, use_q=c["use_q"], tail=c["tl"], tail_nch=3, tau=tau[:7] if c["gate"] else (),
                     adapt_rho=tau[7] if c["rho"] and len(tau) > 7 else 0.0, adapt_mode=(3 if c["ema"] else 2) if c["rho"] else 0)


def run_one(e, fp, t, c, temp, seed, tau, keep_log):
    e.reset()
    e.set_force(None)
    if temp > 0:
        e.set_sampling(temperature=temp, top_k=20, top_p=0.95, draft_sample=1, seed=seed)
    else:
        e.set_sampling(0.0)
    ids, n = t["ids"], t["n"]
    toks = []
    if c["base"]:
        e.set_tail(None)
        first = fp.prefill_spec(ids, 3)
        toks = [first]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(toks) < n:
            if t["eos"] and toks[-1] in EOS:
                break
            rem = n - len(toks)
            g = max(1, min(8, rem // 4))
            new, ss = e.cycles(3, g)
            toks += new
            if t["eos"] and any(x in EOS for x in new):
                break
        dt = time.perf_counter() - t0
        li = lf = None
    else:
        opts = make_cfg_opts(c, tau)
        first = fp.prefill_spec(ids, 3, start=False)
        toks = [first]
        e.set_tail(make_pld(min_len=3, max_tail=4) if c["tail"] else None)
        e.deep_begin(list(ids) + [first], c["kc"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(toks) < n:
            if t["eos"] and toks[-1] in EOS:
                break
            rem = n - len(toks)
            g = max(1, min(4, rem // 4))
            new, ss = e.cycles_deep(g, opts)
            toks += new
            if t["eos"] and any(x in EOS for x in new):
                break
        dt = time.perf_counter() - t0
        li, lf = e.deep_log()
        st = e.deep_stats()
    cut = len(toks)
    if t["eos"]:
        cut = next((i + 1 for i, x in enumerate(toks) if x in EOS), len(toks))
    cut = min(cut, n)
    r = dict(tokens=cut - 1, emitted=ss[12], sec=dt, cycles=ss[13], hist=ss[16:24])
    if li is not None:
        w = li[:, 1] + li[:, 2]
        r.update(width_hist=np.bincount(w, minlength=8).tolist(), nch_sum=int(li[:, 1].sum()), tail_tokens=int(li[:, 2].sum()), tail_cycles=int((li[:, 2] > 0).sum()),
                 prio_cycles=int(li[:, 5].sum()), kc_next_hist=np.bincount(li[:, 4], minlength=8).tolist(), wait_us=st["wait_ns"] / max(len(li), 1) / 1e3, work_us=st["work_ns"] / max(len(li), 1) / 1e3)
        if keep_log:
            r["log_i"], r["log_f"] = li.astype(np.int16), lf.astype(np.float16)
    return r


def cost_grid(e, fp, t, temp, n=12):
    """ms of the deep cycle graph for chain depth kc = 1..7 x verify width kv = 0..7 (replays on the state after t's prefill); also the verify-only and MTP-stage parts."""
    e.reset()
    e.set_force(None)
    e.set_sampling(temperature=temp, top_k=20, top_p=0.95, draft_sample=1, seed=1) if temp > 0 else e.set_sampling(0.0)
    fp.prefill_spec(t["ids"], 3, start=False)
    e.deep_begin(list(t["ids"]) + [0], 3)
    e.set_force([1])  # every replayed cycle emits 1 token: the context grows by 1 per replay; the state is restored before every cell (snapshot / restore), so a cell is timed at ctx ~ the prompt length
    e.snapshot()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = {}
    for kc in range(1, 8):
        for kv in range(0, 8):
            for which, nm in ((0, "cycle"), (1, "verify"), (2, "stage")):
                if which != 0 and kc not in (3, 5, 7):
                    continue
                e.restore()
                e.deep_launch(kc, kv, which, 3)
                torch.cuda.synchronize()
                ms = []
                for _ in range(3):
                    ev0.record()
                    e.deep_launch(kc, kv, which, n)
                    ev1.record()
                    ev1.synchronize()
                    ms.append(ev0.elapsed_time(ev1) / n)
                out[f"{nm}_kc{kc}_kv{kv}"] = sorted(ms)[len(ms) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="e0")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--configs", default="base,k3,k5")
    ap.add_argument("--reps", type=int, default=3, help="reps for e0 / e6")
    ap.add_argument("--reps-e8", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gate-file", default=None, help="json {task id: [tau_1..tau_7]} (cross-fit: the tau of the fold that does NOT contain the task)")
    ap.add_argument("--gate-dir", default=None, help="directory with <variant>_<mode>.json tau files (variant = config name without the tail suffix), per-task cross-fit tau")
    ap.add_argument("--gate-mode", default="safe")
    ap.add_argument("--gate-default", default="", help="comma list of tau used for tasks missing from --gate-file")
    ap.add_argument("--json", default=None)
    ap.add_argument("--logdir", default=None, help="save the per-cycle logs of rep 0 (npz per task and config)")
    ap.add_argument("--log-cfgs", default="", help="configs whose logs are saved (default: all deep configs when --logdir is set)")
    ap.add_argument("--grid", default=None, help="write the cost grid (cycle ms vs kc x kv) measured at ctx ~ the first task of each of e0 code / e6 a1 / long prompts")
    ap.add_argument("--limit-e8", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue a --json file: tasks that have every (config, rep) row are skipped")
    ap.add_argument("--kc-max", type=int, default=7)
    a = ap.parse_args()
    suites = a.suites.split(",")
    tasks = load_tasks(suites, [x for x in a.tasks.split(",") if x], a.limit_e8)
    cfgs = [parse_cfg(c, a.temperature, None) for c in a.configs.split(",")]
    gate = json.loads(Path(a.gate_file).read_text()) if a.gate_file else {}
    gate_by_cfg = {}
    if a.gate_dir:
        for c in cfgs:
            if c.get("gate"):
                variant = re.sub(r"t[12]$", "", c["name"])
                gate_by_cfg[c["name"]] = json.loads((Path(a.gate_dir) / f"{variant}_{a.gate_mode}.json").read_text())
    gdef = [float(x) for x in a.gate_default.split(",") if x]
    ctx = max(len(t["ids"]) + t["n"] for t in tasks) + 160
    e = DeepEngine(ctx_max=ctx, verbose=False, tmax=8)
    fp = FastPrefill(e)
    print(f"suites {suites}, {len(tasks)} tasks, configs {[c['name'] for c in cfgs]}, T={a.temperature}, build {os.environ.get('SMK_BUILD')}, ctx_max {ctx}", flush=True)
    # warm-up: capture every graph of the swept depths in the right mode (greedy / sampling) before any timed run
    if a.temperature > 0:
        e.set_sampling(temperature=a.temperature, top_k=20, top_p=0.95, draft_sample=1, seed=1)
    else:
        e.set_sampling(0.0)
    kcs = sorted({c["kc"] for c in cfgs if not c["base"]} | {c["kc_lo"] for c in cfgs if not c["base"]} | {3})
    e.prewarm(min(kcs), max(kcs), 7)
    w0 = tasks[0]
    for c in cfgs:
        run_one(e, fp, dict(w0, n=40), c, a.temperature, 1, gdef or [0.5] * 7, False)
    rows = []
    grid = {}
    if a.resume and a.json and Path(a.json).exists():
        prev = json.loads(Path(a.json).read_text())
        rows, grid = prev["rows"], prev.get("grid", {})
        full = {}
        for r in rows:
            full.setdefault((r["suite"], r["task"]), set()).add((r["cfg"], r["rep"]))
        tasks = [t for t in tasks if len(full.get((t["suite"], t["id"]), ())) < len(cfgs) * (a.reps_e8 if t["suite"] == "e8" else a.reps)]
        print(f"resume: {len(rows)} rows, {len(tasks)} tasks left", flush=True)
    if a.grid and not (a.resume and grid):
        grid_tasks = {}
        for t in tasks:
            if t["suite"] == "e0" and t["id"] in ("code", "long8k"):
                grid_tasks[f"ctx{len(t['ids'])}"] = t
            if t["suite"] == "e6" and t["id"].startswith("a1"):
                grid_tasks[f"ctx{len(t['ids'])}"] = t
        for nm, t in sorted(grid_tasks.items()):
            e.prewarm(1, 7, 7)
            grid[nm] = cost_grid(e, fp, t, a.temperature)
            print("grid", nm, {k: round(v, 2) for k, v in grid[nm].items() if k.startswith("cycle_kc3_") or k.startswith("cycle_kc5_kv5") or k.startswith("cycle_kc7_kv7")}, flush=True)
    logdir = Path(a.logdir) if a.logdir else None
    if logdir:
        logdir.mkdir(parents=True, exist_ok=True)
    log_cfgs = set(a.log_cfgs.split(",")) if a.log_cfgs else {c["name"] for c in cfgs}
    t_start = time.time()
    for ti, t in enumerate(tasks):
        nreps = a.reps_e8 if t["suite"] == "e8" else a.reps
        for rep in range(nreps):
            order = cfgs if rep % 2 == 0 else cfgs[::-1]
            for c in order:
                tau = gate_by_cfg.get(c["name"], gate).get(t["id"]) or gdef or [0.5] * 7
                r = run_one(e, fp, t, c, a.temperature, 1000 + rep, tau, logdir is not None and rep == 0 and c["name"] in log_cfgs and not c["base"])
                if "log_i" in r:
                    np.savez_compressed(logdir / f"{t['suite']}_{t['id']}_{c['name']}_T{a.temperature:g}.npz", li=r.pop("log_i"), lf=r.pop("log_f"))
                r.update(task=t["id"], cls=t["cls"], suite=t["suite"], cat=t["cat"], cfg=c["name"], rep=rep, prompt=len(t["ids"]), n=t["n"])
                rows.append(r)
        line = [f"{t['suite']}/{t['id']:>20s} n={t['n']:4d}"]
        for c in cfgs:
            rs = [r for r in rows if r["task"] == t["id"] and r["suite"] == t["suite"] and r["cfg"] == c["name"]]
            med = statistics.median(r["tokens"] / r["sec"] for r in rs)
            E = statistics.median(r["emitted"] / max(r["cycles"], 1) for r in rs)
            line.append(f"{c['name']} {med:6.1f} E {E:.2f}")
        print(f"[{ti + 1}/{len(tasks)} {time.time() - t_start:6.0f}s] " + " | ".join(line), flush=True)
        if a.json and (ti % 10 == 9 or ti == len(tasks) - 1):
            json.dump(dict(args=vars(a), rows=rows, grid=grid), open(a.json, "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
