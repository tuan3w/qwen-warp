#!/usr/bin/env python3
"""Generate the qwen-warp poster (infographic.svg) and the experiment-history page (history.html). Standard library only.

    python3 gen.py [--out DIR] [--stats]   # default DIR = this folder (infographic.svg, history.html); render the PNG separately (see README.md)

Inputs (all committed next to this file):
  data.json             every number shown, each with a `src` id that resolves in data.json["sources"]  <- the ONE file to edit
  campaign_snapshot.json, lease_snapshot.json, human_prompts.json   snapshots of things that live outside the repo (refresh.py)
  snippets_cache/       code from a campaign lineage repo, for snippets whose source is outside this repo
Derived at generation time from the git repo (committed state, `git show HEAD:...`, never the working tree):
  commit count, wall-clock span, experiment/prediction tallies, CUDA lines, tests, GPU hours, timeline timestamps, snippet text.
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or HERE.parents[3])


def git(*a, check=False):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True)
    return r.stdout


def show(rev, path):
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{rev}:{path}"], capture_output=True, text=True).stdout


def ts(iso):
    return datetime.fromisoformat(iso).timestamp()


def fm(text):
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    d = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            d[k.strip()] = v.strip().strip('"')
    return d


def section(text, name):
    m = re.search(r"^## " + re.escape(name) + r"\s*\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    body = m.group(1).strip() if m else ""
    return re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()


def load():
    D = json.loads((HERE / "data.json").read_text())
    C = json.loads((HERE / "campaign_snapshot.json").read_text())
    L = json.loads((HERE / "lease_snapshot.json").read_text())
    H = json.loads((HERE / "human_prompts.json").read_text())
    return D, C, L, H


def derive(D, C, L, H):
    X = {}
    X["head"] = git("rev-parse", "--short", "HEAD").strip()
    X["head_iso"] = git("log", "-1", "--format=%aI", "HEAD").strip()
    X["generated"] = datetime.now().astimezone().isoformat(timespec="minutes")
    win0, win1 = ts(D["meta"]["window"]["start"]), ts(D["meta"]["window"]["end"])
    X["win0"], X["win1"] = win0, win1
    # --- git: commits of the program (swift-mk + the program's experiment folders), side studies excluded
    dirs = [f"experiments/{e['dir']}" for e in D["experiments"]]
    out = git("log", f"--since={D['meta']['window']['start']}", "--reverse", "--format=%h|%aI|%s", "--", "projects/swift-mk", *dirs)
    commits = []
    for line in out.splitlines():
        h, d, s = line.split("|", 2)
        if h in D["meta"].get("exclude_commits", []):  # commits of studies outside this release
            continue
        commits.append({"hash": h, "iso": d, "msg": re.sub(r"^\[agent:[^\]]+\]\s*", "", s)})
    X["commits"] = commits
    first_commit = git("log", "--reverse", "--format=%aI", "-1", "--since=" + D["meta"]["window"]["start"]).strip()
    X["n_commits"] = len(commits)
    t_first = min(ts(e) for e in [commits[0]["iso"]] + [git("log", "-1", "--format=%aI", e["pre"]).strip() for e in D["experiments"]])
    t_last = ts(X["head_iso"])
    X["t_first"], X["t_last"] = t_first, t_last
    X["wall_h"] = (t_last - t_first) / 3600
    # --- commit times for milestones / experiments
    cache = {}

    def ctime(h):
        if h not in cache:
            cache[h] = ts(git("log", "-1", "--format=%aI", h).strip())
        return cache[h]
    X["ctime"] = ctime
    # --- experiments: outcomes, READMEs at HEAD
    TEXTS = json.loads((HERE / "exp_texts.json").read_text())  # README texts, side-study clauses already removed (see _private/)
    exps = []
    for e in D["experiments"]:
        exps.append(dict(e, **TEXTS[e["dir"]], t_pre=ctime(e["pre"]), t_post=ctime(e["post"])))
    X["exps"] = exps
    X["outcomes"] = {k: sum(1 for e in exps if e["outcome"] == k) for k in ("win", "partial", "negative", "abandoned")}
    X["n_dead_end_sections"] = sum(1 for e in exps if e["dead"])
    # --- predictions
    pr = D["predictions"]
    X["pred_n"] = len(pr)
    X["pred_hit"] = sum(1 for p in pr if p["verdict"] == "hit")
    X["pred_miss"] = sum(1 for p in pr if p["verdict"] == "miss")
    X["pred_mixed"] = sum(1 for p in pr if p["verdict"] == "mixed")
    X["pred_miss_for"] = sum(1 for p in pr if p["verdict"] == "miss" and p["direction"] == "for")
    X["pred_miss_against"] = sum(1 for p in pr if p["verdict"] == "miss" and p["direction"] == "against")
    # --- engine size
    tree = git("ls-tree", "-r", "--name-only", "HEAD", "engine/src/").split()
    cu = [p for p in tree if p.endswith(".cu")]
    X["cu_files"] = len(cu)
    X["cu_lines"] = sum(len(show("HEAD", p).splitlines()) for p in cu)
    tests = git("ls-tree", "-r", "--name-only", "HEAD", "engine/tests/").split()
    X["test_files"] = len([p for p in tests if re.search(r"/(test_|bench_)", p) and p.endswith((".py", ".cu"))])
    X["test_only"] = len([p for p in tests if "/test_" in p])
    # --- ideas
    ideas = git("ls-tree", "--name-only", "HEAD", "ideas/").split()
    n_ideas = 0
    for p in ideas:
        if re.search(r"swift-mk|fastest-single-3090|measure-mtp-acceptance", p):
            c = fm(show("HEAD", p)).get("created", "")
            if c >= "2026-10-30":
                n_ideas += 1
    X["n_ideas"] = n_ideas
    # --- agents: curated work packages in dashboard/gen.py AGENTS at HEAD
    names = 0
    try:
        for n in ast.parse(show("HEAD", "dashboard/gen.py")).body:
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "AGENTS":
                names = sum(len(a[0].split(" / ")) for a in ast.literal_eval(n.value))
    except Exception:
        names = 0
    X["n_packages"] = names
    # --- campaign
    rounds = C["rounds"]
    X["camp_rounds"] = len(rounds)
    X["camp_kept"] = sum(1 for r in rounds if r["decision"] == "keep")
    X["camp_rejected"] = len(rounds) - X["camp_kept"]
    X["camp_tries"] = C["events"].get("try", 0)
    X["camp_sessions"] = C["events"].get("session_start", 0)
    X["camp_wall_h"] = (C["ended"] - C["started"]) / 3600
    X["camp_session_min"] = {r["lineage"]: r["session_s"] / 60 for r in rounds if r["session_s"] > 0}
    # --- GPU lease
    spans = sorted((s["start"], s["end"], s["kind"], s["tag"]) for s in L["spans"])
    union, cur = [], None
    for a, b, k, tg in spans:
        if cur and a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            cur = [a, b]
            union.append(cur)
    X["gpu_union"] = union
    X["gpu_h"] = sum(b - a for a, b in union) / 3600
    X["gpu_span_h"] = (win1 - win0) / 3600
    tagh = {}
    for a, b, k, tg in spans:
        tagh[tg] = tagh.get(tg, 0) + (b - a) / 3600
    X["gpu_tag_h"] = tagh
    X["gpu_spans"] = spans
    # --- human
    ps = H["prompts"]
    X["h_n"] = len(ps)
    X["h_chars"] = sum(p["chars"] for p in ps)
    X["h_words"] = sum(p["words"] for p in ps)
    X["h_class"] = {k: sum(1 for p in ps if p["class"] == k) for k in H["classes"]}
    X["h_times"] = [(ts(p["at"]), p["class"]) for p in ps]
    # --- campaign headline ratios
    cp = D["campaign"]
    v = cp.get("server_verified")
    cp_eff = dict(cp)
    X["campaign"] = cp_eff
    X["camp_x"] = cp["gate_verdict"]
    X["camp_verified"] = bool(v)
    # --- headline ratios
    hd = D["headline"]
    s = {x["id"]: x for x in D["series"]["steps"]}
    X["x_k11_vs_vllm"] = [round(a / b, 2) for a, b in zip(s["k11"]["vals"], s["vllm"]["vals"])]
    X["x_plain"] = hd["vllm_plain_ms"]["v"] / hd["plain_k9_ms"]["v"]
    X["x_plain_camp"] = X["x_plain"] * cp["plain_guard"]
    sv_ = D["server"]
    X["x_srv_greedy"] = sv_["greedy"]["ratio"]
    X["x_srv_t1"] = sv_["t1"]["ratio"]
    X["x_camp_vs_vllm"] = [round(a * cp["gate_verdict"] / b, 2) for a, b in zip(s["k11"]["vals"], s["vllm"]["vals"])]
    return X


def snippet_text(sn):
    a, b = sn["lines"]
    if sn.get("outside_repo"):
        txt = (HERE / "snippets_cache" / f"{sn['id']}.txt").read_text().splitlines()
        assert len(txt) == b - a + 1, sn["id"]
        return txt, sn.get("pointer", f"{sn['file']}:{a}-{b}")
    full = show(sn["rev"], sn["file"]).splitlines()
    txt = full[a - 1:b]
    assert 0 < len(txt) <= 12, (sn["id"], len(txt))
    return txt, f"{sn['file']}:{a}-{b} @{sn['rev'][:7]}"


def validate(D):
    src = D["sources"]
    used = set()

    def walk(o):
        if isinstance(o, dict):
            if "src" in o:
                assert o["src"] in src, f"unknown src {o['src']}"
                used.add(o["src"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk({k: v for k, v in D.items() if k != "sources"})
    unused = set(src) - used
    if unused:
        print("note: sources not referenced by any datum:", sorted(unused), file=sys.stderr)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--stats", action="store_true", help="also write stats.json (the derived numbers)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D, C, L, H = load()
    validate(D)
    X = derive(D, C, L, H)
    snips = []
    for sn in D["snippets"]:
        lines, ptr = snippet_text(sn)
        snips.append(dict(sn, text=lines, ptr=ptr))
    X["snips"] = snips
    sys.path.insert(0, str(HERE))
    import poster
    import history
    svg = poster.build(D, C, X)
    (out / "infographic.svg").write_text(svg)
    (out / "history.html").write_text(history.build(D, C, X))
    if a.stats:
        (out / "stats.json").write_text(json.dumps({k: v for k, v in X.items() if isinstance(v, (int, float, str, dict)) and k not in ("ctime",)}, indent=1, default=str))
    print("wrote", out / "infographic.svg", len(svg) // 1024, "KB;", out / "history.html")


if __name__ == "__main__":
    main()
