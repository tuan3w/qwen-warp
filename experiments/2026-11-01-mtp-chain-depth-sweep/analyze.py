"""Tables from the bench json files (rows of bench_deep.py).  CPU only.

  python3 analyze.py --json results/bench_greedy.json[,more.json] --ref k3 [--cfgs k3,k4,k5,k6,k7] [--out results/tables_greedy.md] [--title "..."]
Per class: pooled tokens / pooled decode time (median over reps of the pooled value), E = tokens per cycle, ratio vs --ref with 95% bootstrap CI over tasks (per-task time = median over reps);
single-task classes (E0 prompts) get the min..max over reps instead.  Per-position acceptance a_j = P(>= j drafts accepted) from the accepted-draft histogram of the no-tail configs.
"""
import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_ORDER = ["code", "prose", "math", "long8k", "a", "b", "c", "d", "e", "en"]
CLASS_NAME = dict(code="E0 code", prose="E0 prose", math="E0 math", long8k="E0 long8k", a="E6 a code edit", b="E6 b log/tool", c="E6 c agent turns", d="E6 d prose", e="E6 e reasoning/math",
                  en="E8 English")
GROUPS = dict(qa="prose", ex="prose", em="prose", sm="copy", tr="copy", ag="copy", mt="math", cd="code")


def load(paths):
    rows = []
    for p in paths:
        rows += json.loads(Path(p).read_text())["rows"]
    return rows


def index(rows):
    """{(cls, task, suite): {cfg: [rows]}}"""
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[(r["cls"], r["task"], r["suite"])][r["cfg"]].append(r)
    return d


def pooled(tasks, cfg, how="median"):
    """tasks: list of per-task {cfg: rows}; -> (tokens, sec) using mean tokens and median sec over reps"""
    tok = sec = 0.0
    for t in tasks:
        rs = t[cfg]
        tok += statistics.mean(r["tokens"] for r in rs)
        sec += statistics.median(r["sec"] for r in rs)
    return tok, sec


def boot_ratio(tasks, cfg, ref, B=2000, seed=1):
    rnd = random.Random(seed)
    per = [(statistics.mean(r["tokens"] for r in t[cfg]), statistics.median(r["sec"] for r in t[cfg]), statistics.mean(r["tokens"] for r in t[ref]), statistics.median(r["sec"] for r in t[ref])) for t in tasks]
    out = []
    for _ in range(B):
        s = [per[rnd.randrange(len(per))] for _ in per]
        a = sum(x[0] for x in s) / sum(x[1] for x in s)
        b = sum(x[2] for x in s) / sum(x[3] for x in s)
        out.append(a / b)
    out.sort()
    return out[int(0.025 * B)], out[int(0.975 * B)]


def class_stats(idx, cls, cfgs, ref, filt=None):
    tasks = [v for (c, tid, suite), v in idx.items() if c == cls and all(cf in v for cf in cfgs + [ref]) and (filt is None or filt(tid, suite, v))]
    if not tasks:
        return None
    out = dict(n=len(tasks))
    nreps = min(min(len(t[cf]) for cf in cfgs + [ref]) for t in tasks)
    for cf in cfgs + [ref]:
        reps = []
        for r in range(nreps):
            tok = sum(t[cf][r]["tokens"] for t in tasks)
            sec = sum(t[cf][r]["sec"] for t in tasks)
            reps.append(tok / sec)
        em = sum(r["emitted"] for t in tasks for r in t[cf])
        cy = sum(r["cycles"] for t in tasks for r in t[cf])
        tok, sec = pooled(tasks, cf)
        out[cf] = dict(tok_s=statistics.median(reps), reps=reps, E=em / max(cy, 1), ms_cycle=sum(r["sec"] for t in tasks for r in t[cf]) / max(cy, 1) * 1e3, pooled=tok / sec)
        deep = [r for t in tasks for r in t[cf] if "width_hist" in r]
        if deep:
            cyc = sum(r["cycles"] for r in deep)
            out[cf]["tail_share"] = sum(r["tail_cycles"] for r in deep) / max(cyc, 1)
            out[cf]["width_hist"] = [sum(r["width_hist"][i] for r in deep) / max(cyc, 1) for i in range(8)]
            out[cf]["mean_width"] = sum(i * out[cf]["width_hist"][i] for i in range(8)) + 1
            out[cf]["wait_us"] = statistics.mean(r["wait_us"] for r in deep)
            out[cf]["work_us"] = statistics.mean(r["work_us"] for r in deep)
            out[cf]["hist"] = [sum(r["hist"][i] for r in deep) for i in range(8)]
    for cf in cfgs:
        out[cf]["ratio"] = out[cf]["pooled"] / out[ref]["pooled"]
        ratios_rep = [out[cf]["reps"][i] / out[ref]["reps"][i] for i in range(nreps)]
        out[cf]["ratio_rep_min"], out[cf]["ratio_rep_max"] = min(ratios_rep), max(ratios_rep)
        if len(tasks) > 1:
            out[cf]["ci"] = boot_ratio(tasks, cf, ref)
    return out


def fmt_ratio(s, cf):
    c = s[cf]
    if "ci" in c:
        return f"{c['ratio']:.3f} [{c['ci'][0]:.3f}, {c['ci'][1]:.3f}]"
    return f"{c['ratio']:.3f} ({c['ratio_rep_min']:.3f}..{c['ratio_rep_max']:.3f})"


def table(idx, cfgs, ref, classes, title, filt=None):
    L = [f"### {title}", "", "tok/s (E = tokens/cycle) pooled over the class; ratio vs `" + ref + "` [95% CI over tasks; single-task classes: min..max over reps]", "",
         "| class (tasks) | " + " | ".join([ref] + cfgs) + " |", "|---|" + "---|" * (len(cfgs) + 1)]
    stats = {}
    for cls in classes:
        s = class_stats(idx, cls, cfgs, ref, filt)
        if not s:
            continue
        stats[cls] = s
        cells = [f"{s[ref]['tok_s']:.1f} ({s[ref]['E']:.2f})"] + [f"{s[cf]['tok_s']:.1f} ({s[cf]['E']:.2f}) **x{fmt_ratio(s, cf)}**" for cf in cfgs]
        L.append(f"| {CLASS_NAME.get(cls, cls)} ({s['n']}) | " + " | ".join(cells) + " |")
    return L, stats


def acceptance_table(idx, cfg, classes, K, title):
    """a_j = P(>= j chain drafts accepted) from the histogram of cfg (a no-tail deep config with chain depth K)."""
    L = [f"### {title}", "", "| class | cycles | " + " | ".join(f"a{j}" for j in range(1, K + 1)) + " | E |", "|---|---|" + "---|" * (K + 1)]
    for cls in classes:
        rs = [r for (c, tid, suite), v in idx.items() if c == cls for r in v.get(cfg, [])]
        if not rs:
            continue
        h = [sum(r["hist"][i] for r in rs) for i in range(8)]
        n = sum(h)
        L.append(f"| {CLASS_NAME.get(cls, cls)} | {n} | " + " | ".join(f"{sum(h[j:]) / n:.3f}" for j in range(1, K + 1)) + f" | {1 + sum(sum(h[j:]) for j in range(1, K + 1)) / n:.2f} |")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--ref", default="k3")
    ap.add_argument("--cfgs", default="")
    ap.add_argument("--classes", default=",".join(CLASS_ORDER))
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--groups", action="store_true", help="E8 by task group instead of language (en x prose / copy / math / code)")
    a = ap.parse_args()
    rows = load(a.json.split(","))
    idx = index(rows)
    cfgs = a.cfgs.split(",") if a.cfgs else sorted({r["cfg"] for r in rows if r["cfg"] != a.ref})
    L, stats = table(idx, cfgs, a.ref, a.classes.split(","), a.title or f"vs {a.ref}")
    if a.groups:
        for lang in ("en",):
            for g in ("prose", "copy", "math", "code"):
                l2, _ = table(idx, cfgs, a.ref, [lang], f"E8 {lang} {g}", filt=lambda tid, suite, v, g=g: suite == "e8" and GROUPS.get(v[a.ref][0]["cat"]) == g)
                L += [""] + l2
    txt = "\n".join(L)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")
        Path(a.out).with_suffix(".json").write_text(json.dumps(stats, indent=1, default=float))


if __name__ == "__main__":
    main()
