#!/usr/bin/env python3
"""Generate dashboard/index.html: a self-contained trace of the megakernel/engine program.

Everything shown is read from the repo (git log, experiment READMEs, idea Attempts, BENCH/GEMV logs) except the two
curated tables at the top of CURATED, whose every row carries a path pointer so it can be re-checked.
Run: python3 dashboard/gen.py   (then serve the directory: python3 -m http.server 8765 -d dashboard)
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Two layouts: the research workspace (dashboard, experiments/ and ideas/ at the repo root, git history)
# and the public release (dashboard/ next to engine/, experiments/, docs/PROGRAM.md, and a trace.json snapshot of the trace).
DASH = Path(__file__).resolve().parent
PUBLIC = not (DASH.parent.name == "swift-mk" and DASH.parents[1].name == "projects")
ROOT = DASH.parent if PUBLIC else DASH.parents[2]
ENGINE = ROOT / "engine" if PUBLIC else ROOT / "engine"
OUT = Path(os.environ.get("SMK_DASH_OUT", DASH / "index.html"))
# SMK_DASH_EXCLUDE: case-insensitive regex; experiments whose name matches are dropped, as are curated rows, commits, log lines,
# idea-attempt lines and experiment-text lines that still match after the PUBLIC_SUBS scrub.  Used to build the public release.
# `gen.py --dump-trace FILE` (research layout) writes the filtered commit/log trace to FILE and exits.
EXCL = re.compile(os.environ["SMK_DASH_EXCLUDE"], re.I) if os.environ.get("SMK_DASH_EXCLUDE") else None
DUMP_TRACE = sys.argv[sys.argv.index("--dump-trace") + 1] if "--dump-trace" in sys.argv else None
PUBLIC_SUBS = []

# ---- curated: measured series (each with a pointer) --------------------------------------------------------------------
STEP_MS = [  # plain decode ms/step at ctx ~64 (T=1)
    ("stock vLLM 0.30", 21.4, "experiments/2026-10-30-vllm-swift-int4-bs1-baseline/README.md (46.8 tok/s)"),
    ("K3 engine first e2e", 21.33, "engine/BENCH.md (K3 full step)"),
    ("K3 a|b on parallel branch", 20.93, "engine/BENCH.md (K3 profile)"),
    ("K5 GEMV rewrite", 20.195, "engine/GEMV_LOG.md, commit 8fcbc1e"),
    ("K9 int8 head + exact rescoring", 18.800, "engine/BENCH.md (K9), experiments/2026-10-31-lm-head-quant-engine/results/bench_ab_int8.log"),
    ("K9 int4 head + exact rescoring (default)", 18.105, "engine/BENCH.md (K9), experiments/2026-10-31-lm-head-quant-engine/results/bench_ab_int4.log"),
]
SPEC_TOKS = {  # k=3 decode tok/s, greedy
    "labels": ["code", "prose", "math", "8K ctx"],
    "series": [
        ("stock vLLM", [96.4, 63.0, 93.5, 75.5], "E0 README"),
        ("engine K4 (int4 draft)", [145.0, 94.7, 143.0, 115.4], "BENCH.md K4"),
        ("engine + K5 GEMV", [151.6, 95.3, 148.5, 121.6], "BENCH.md K7 'same build'"),
        ("stock vllm serve (same HTTP client)", [96.6, 64.4, 94.8, 74.1], "server/README.md"),
        ("engine via HTTP server", [152.2, 96.1, 149.0, 122.1], "server/README.md"),
        ("stock vLLM T=1 sampling", [86.0, 56.9, 88.1, 76.0], "E0 README"),
        ("engine T=1 sampling", [144.5, 90.5, 150.3, 124.9], "BENCH.md K7"),
        ("engine K9 int4 head (greedy; 8K prompt re-built from current topics/)", [167.8, 107.4, 165.5, 151.9], "BENCH.md K9, experiments/2026-10-31-lm-head-quant-engine/results/bench_ab_int4.log"),
        ("engine K9 int4 head T=1 sampling", [157.9, 106.1, 165.3, 136.3], "BENCH.md K9, experiments/2026-10-31-lm-head-quant-engine/results/bench_ab_int4.log"),
        ("engine K11 MTP chain + prompt-lookup tail (greedy; the E0 prompts have no copyable context, same-process base 166.7 / 108.6 / 165.8 / 134.2; E6 code-edit / log / agent-turn classes 240 / 231 / 272 tok/s vs 173 / 173 / 186)", [167.9, 107.5, 166.9, 131.5], "BENCH.md K11, experiments/2026-11-01-pld-tail-engine/results/ab_e0_greedy_m.json, ab_e6_greedy.json"),
        ("HyperQwen MTP `SPEC=mtp CTX=fast` on Swift-1.5, same HTTP client, 350 W (greedy)", [173.9, 104.1, 177.4, 139.2], "experiments/2026-11-01-hyperqwen-vs-swift-mk/results/tables.md"),
        ("HyperQwen DFlash2 (7 drafts + lookup) on Swift-1.5, same client (greedy)", [253.6, 102.9, 225.3, 121.3], "experiments/2026-11-01-hyperqwen-vs-swift-mk/results/tables.md"),
        ("swift-mk server K11, same client, 2 rounds (greedy)", [165.6, 107.2, 165.7, 131.2], "experiments/2026-11-01-hyperqwen-vs-swift-mk/results/tables.md"),
    ],
}
# prediction vs outcome (pre-registered numbers; outcome from the experiment README)
PREDICTIONS = [
    ("E0 vLLM plain decode tok/s", "34 (28-42)", "46.8", "MISS", "experiments/2026-10-30-vllm-swift-int4-bs1-baseline"),
    ("E0 vLLM MTP k=3 code tok/s", "58", "96.4", "MISS", "experiments/2026-10-30-vllm-swift-int4-bs1-baseline"),
    ("E1 persistent+prefetch / graph", "1.07x (1.00-1.15)", "1.025x", "MISS (kill criterion hit)", "experiments/2026-10-30-w4-stream-persistence-vs-graph"),
    ("E1 PDL baseline exists on sm_86", "assumed yes", "impossible (ptxas: griddepcontrol needs sm_90)", "MISS", "experiments/2026-10-30-w4-stream-persistence-vs-graph"),
    ("E3 verify flat to T=8", "<=1.03x", "1.12-1.15x at T=8", "MISS", "experiments/2026-10-30-w4-multirow-verify-and-power"),
    ("E3 cheaper dequant raises GB/s", "'power is a lever'", "+0.4-1.2% only", "MISS", "experiments/2026-10-30-w4-multirow-verify-and-power"),
    ("E4 restricted vocab 32k loses", "<=3%", "6.2%", "MISS", "experiments/2026-10-31-mtp-draft-head-ablation"),
    ("E4 MTP int4 loses", "3-8%", "1.0%", "MISS (better)", "experiments/2026-10-31-mtp-draft-head-ablation"),
    ("E4 64k vocab / head int4", "<=3% / <=3%", "0.6% / 0.0%", "HIT", "experiments/2026-10-31-mtp-draft-head-ablation"),
    ("E5 int8 lm_head passes top-1>=99.5%", "adopt int8", "99.19%", "MISS", "experiments/2026-10-31-lm-head-quantisation-quality"),
    ("K7 spec-sampling speedup vs vLLM T=1", "1.38-1.43x", "1.59-1.71x", "MISS (better)", "experiments/2026-10-31-spec-sampling-exactness"),
    ("N1 PLD+MTP switch hybrid H0 / MTP-only, code edits (modelled)", "1.20x (1.08-1.35)", "1.168x [1.01, 1.40]", "HIT", "experiments/2026-10-31-ngram-draft-potential"),
    ("N1 H0 no loss on prose / math", ">=0.995x / >=0.995x", "0.971x / 0.880x", "MISS", "experiments/2026-10-31-ngram-draft-potential"),
    ("N1 MTP-only tokens/cycle on log extraction", "3.1", "3.69", "MISS (better)", "experiments/2026-10-31-ngram-draft-potential"),
    ("K9 plain step int8 / int4 head + rescore (ms)", "18.85 / 18.25", "18.80 / 18.105", "HIT / HIT (better)", "experiments/2026-10-31-lm-head-quant-engine"),
    ("K9 greedy k=3 code tok/s int8 / int4", "~160 / ~165 (x1.055-1.065 / x1.085-1.10)", "162.0 / 167.8 (x1.065 / x1.104)", "HIT / HIT (int4 slightly above the band)", "experiments/2026-10-31-lm-head-quant-engine"),
    ("K9 greedy argmax == BF16 head on >= 4000 positions", "0 mismatches", "0 real over 19,976 (1 exact fp32 tie)", "HIT", "experiments/2026-10-31-lm-head-quant-engine"),
    ("K11 verify step V(5..8) at ctx 64 (ms)", "19.9 / 20.5 / 21.1 / 21.7", "20.07 / 20.62 / 21.17 / 21.61", "HIT", "experiments/2026-11-01-pld-tail-engine"),
    ("K11 host round trip per hybrid cycle (device-side lookup needed?)", "0.05-0.15 ms, no", "< 0.01 ms (notail == base to 0.1%), no", "HIT (better)", "experiments/2026-11-01-pld-tail-engine"),
    ("K11 hybrid / MTP-only tok/s on E6 code edits / logs / agent turns (greedy)", "x1.25 / 1.20 / 1.30", "x1.389 / 1.329 / 1.461 (pooled a-c 1.381)", "MISS (better)", "experiments/2026-11-01-pld-tail-engine"),
    ("K11 no loss on prose / math (E6 d / e), E0 code +5..15%", ">= 0.985 / >= 1.00, code +5..15%", "x1.014 / 1.072, E0 code +0.7%", "HIT / MISS", "experiments/2026-11-01-pld-tail-engine"),
    ("K11 max-model-len with the 9-slot ring (bf16 / fp8)", "~28000 / ~49000", "28672 / 53248 (from 36864 / 65536)", "HIT", "experiments/2026-11-01-pld-tail-engine"),
    ("K9 int4 sampled-distribution TV vs BF16 head", "< 0.003", "<= 0.00001 (5 positions), mean 0.0015 at top_k 64 (4,994 positions)", "HIT", "experiments/2026-10-31-lm-head-quant-engine"),
    ("MTP chain depth: cycle ms at k = 3 / 5 / 7 (greedy, ctx ~64)", "20.7 / 22.7 / 25.0 (+1.05 per depth)", "20.89 / 22.90 / 25.18 (+1.06 per depth; draft 0.52 ms + verify row ~0.55 ms)", "HIT", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("MTP chain depth: tok/s vs k=3 at k=5, code / math (E0) / E6 e, fixed depth", "+8% (+4..+14%)", "x1.195 / 1.138 / 1.177 [1.142, 1.217]; best fixed k=7 code, 6 math", "MISS (about double)", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("MTP chain depth: prose at k=5 / k=7 vs k=3 (E6 d, E0 prose)", "-2.5% / -5%", "x0.991 [0.961, 1.022] / 0.903 and x0.957 / 0.849", "k=5 HIT, k=7 MISS (2x worse)", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("MTP chain depth: k=4 never loses (E0 / E6 / E8 English, 10 classes)", "+7% code, ~0 prose", "x0.996-1.195 on every class (code 1.121, a-c 1.14-1.20, en 1.058, E6 d 0.996)", "HIT (free win)", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("MTP chain depth: AUROC of the draft head max-prob for acceptance (conf_2)", "0.75-0.82", "0.893 (0.90-0.91 at every depth)", "MISS (better)", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("MTP chain depth: confidence gate recovers prose / E6 d to >= 0 (MODELLED cross-fit; online not run)", "width-only -1.0%, adaptive depth >= -0.3% (p 0.6)", "width-only g5 -3.1% / -1.0%; width + EMA depth controller g7a3e -1.1% / -0.2% (d / e 0.998 / 1.20, a-c x1.41-1.59)", "half (modelled)", "experiments/2026-11-01-mtp-chain-depth-sweep"),
    ("HyperQwen (vLLM 0.30 + patches) on OUR Swift-1.5 weights, E0 greedy tok/s code / prose / math / 8K: MTP `mtp-fast` vs swift-mk", "swift-mk wins >= 3 of 4 (swift-mk 160/103/160/135, HQ MTP 125/88/120/90)", "HQ MTP 174/104/177/139 vs swift-mk 166/107/166/131: swift-mk wins 1 of 4 (prose, +3%)", "MISS (HyperQwen far stronger than predicted)", "experiments/2026-11-01-hyperqwen-vs-swift-mk"),
    ("HyperQwen DFlash2 (7-token block drafter, lookup on) E0 greedy code / prose / math / 8K vs swift-mk", "130 / 92 / 128 / 85", "254 / 103 / 225 / 121 (x1.53 / 0.96 / 1.36 / 0.92 of swift-mk); 5.83 / 2.39 / 5.20 / 2.98 tokens per step at 23 ms vs our 3.50 / 2.24 / 3.50 / 2.87 at 21 ms", "MISS (the finding)", "experiments/2026-11-01-hyperqwen-vs-swift-mk"),
    ("E6 classes a-c pooled greedy tok/s, swift-mk / HQ MTP / HQ DFlash2 (256 tokens, HTTP)", "240 / 140 / (DFLASH_TOKENS=15) 200", "197 / 182 / 225 (DFlash2 x1.14; dflash15 did not serve); class c alone 240 / 185 / 229", "MISS", "experiments/2026-11-01-hyperqwen-vs-swift-mk"),
    ("TTFT 20-token / 8K prompt, swift-mk vs HyperQwen", "38 ms, 3.9 s vs 70 ms (50-130), 5.3 s (4.5-6.5)", "39 ms, 3.92 s vs 53-56 ms, 4.16-4.22 s", "HIT / MISS (HQ 8K faster than predicted)", "experiments/2026-11-01-hyperqwen-vs-swift-mk"),
    ("Greedy parity swift-mk vs HyperQwen, first 128 tokens, 30 E8 prompts identical", "16 (10-24)", "7 (mean common prefix 59 of 128); HQ MTP vs HQ DFlash2 18; swift-mk vs itself 30", "MISS (lower)", "experiments/2026-11-01-hyperqwen-vs-swift-mk"),
]
AGENTS = [  # engineering work packages (parallel agents), outcome one-liners with pointers
    ("StudyOpenJet / StudyTpuMk / StudyGraphVsMega / StudyLuceMega", "source studies", "sources/*.md (8 repos)"),
    ("MapVllm2", "vLLM integration seams", "agent://MapVllm2 (PluggableLayer.register_oot, EngineClient)"),
    ("BuildOracle", "CPU oracle + dumps", "oracle/NOTES.md"),
    ("BuildK1DeltaNet", "fused DeltaNet kernel", "BENCH.md: 0.53 ms/step T=1; T=4 2.1x (snapshot writes)"),
    ("BuildK2Attn", "split-KV attention bf16/fp8", "BENCH.md: 0.48 ms/step @8K fp8"),
    ("BuildK3Core", "packer, GEMV, norms, graph step", "PARITY.md: token parity, 20.93 ms/step"),
    ("BuildK4Spec", "on-device MTP spec loop", "MTP_PARITY.md: 1.5x vLLM greedy"),
    ("BuildK5Gemv", "GEMV rewrite", "GEMV_LOG.md: 20.98 -> 20.20 ms; 'bit-identical' claim later shown false (K7 follow-up)"),
    ("BuildK6Prefill", "fast prefill", "PREFILL_PARITY.md: 2100 tok/s @8K"),
    ("BuildK7Sampling", "exact speculative sampling", "experiments/2026-10-31-spec-sampling-exactness"),
    ("BuildK8Server", "OpenAI-compatible server (stock vLLM AsyncLLM + in-process EngineCoreClient)", "server/README.md: HTTP k=3 152/96/149/122 vs stock vllm serve 96.6/64.4/94.8/74.1 tok/s (1.49-1.65x), 8K TTFT 3.9 s vs 4.35 s, multi-turn TTFT 240 ms vs 912 ms"),
    ("BuildK10ShortPrefill", "short-prompt TTFT: multi-row W4 mma GEMM (T<=64, bit-identical to decode GEMV) + last row folded into the batch", "engine/BENCH.md K10: server TTFT 8/20/40/64/128 tokens 74/130/235/217/227 -> 34/38/47/75/128 ms (stock vLLM ~96 ms at 30 tokens); engine-direct 31/34/44/75/125 ms; 240-600 tokens unchanged (K6 path)"),
    ("BuildK11Pld", "hybrid draft: MTP chain + host prompt-lookup tail, verify T<=8 (DeltaNet 9-slot ring, attention / head / sampling T<=8, per-width graphs with a mapped mailbox), exact greedy + K7 sampling", "BENCH.md K11: V(8) 21.6 ms, host gap < 10 us, E6 workload a/b/c/d/e x1.389/1.329/1.461/1.014/1.072 greedy (x1.359/1.315/1.426/1.004/1.072 T=1), +0.62 GiB (max-model-len 36864 -> 28672), server --prompt-lookup auto"),
    ("BuildK9HeadQ", "quantised lm_head (int4 g128 default / int8) + exact bf16 rescoring of the top-16/64 candidates", "BENCH.md K9: plain step 20.15 -> 18.105 ms, k=3 +10% (167.8/107.4/165.5/151.9), greedy streams identical to the BF16 head, +0.59 GiB (max-model-len 40960 -> 36864)"),
]


def sh(*a):
    return subprocess.run(a, cwd=ROOT, capture_output=True, text=True).stdout


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
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S).strip()
    return body


def txt(s):
    """Apply the public-release scrub to free text (no-op unless SMK_DASH_EXCLUDE is set)."""
    if not EXCL:
        return s
    for pat, rep in PUBLIC_SUBS:
        s = pat.sub(rep, s)
    return s


def clean_lines(s):
    """Scrub + drop the lines that still mention an excluded topic."""
    s = txt(s)
    if not EXCL:
        return s
    return "\n".join(l for l in s.splitlines() if not EXCL.search(l))


def clean_field(s):
    s = txt(s)
    return "(omitted)" if EXCL and EXCL.search(s) else s


def keep_row(row):
    """Curated rows: scrub every string field; drop the row if any still mentions an excluded topic."""
    if not EXCL:
        return row
    new = []
    for f in row:
        if isinstance(f, str):
            f = txt(f)
            if EXCL.search(f):
                return None
        new.append(f)
    return type(row)(new)


def experiments():
    out = []
    if PUBLIC:
        paths = sorted((ROOT / "experiments").glob("*/README.md"))
    else:
        paths = sorted((ROOT / "experiments").glob("2026-10-3[01]-*/README.md")) + sorted((ROOT / "experiments").glob("2026-11-*/README.md"))
    for p in paths:
        if EXCL and EXCL.search(p.parent.name):
            continue
        t = p.read_text()
        f = fm(t)
        out.append({
            "dir": p.parent.name, "status": f.get("status", ""), "question": clean_field(f.get("question", "")),
            "prediction": clean_field(f.get("prediction", "")), "answer": clean_field(f.get("answer", "")), "confidence": f.get("confidence", ""),
            "dead": clean_lines(section(t, "Dead ends")[:2500]), "conclusion": clean_lines(section(t, "Conclusion")[:1800]),
        })
    return out


def ideas():
    out = []
    files = [ROOT / "docs" / "PROGRAM.md"] if PUBLIC else sorted((ROOT / "ideas").glob("*.md"))
    for p in files:
        if not p.exists():
            continue
        t = p.read_text()
        f = fm(t)
        if not PUBLIC and "swift" not in p.name and "megakernel" not in t[:3000].lower() and "mtp" not in p.name and "lm-head" not in p.name:
            continue
        if f.get("created", "") < "2026-10-30":
            continue
        out.append({"file": p.name, "title": f.get("title", ""), "status": f.get("status", ""),
                    "attempts": clean_lines(section(t, "Attempts")[:3500]), "raw": clean_lines(section(t, "Raw")[:300])})
    return out


def commits():
    rows = []
    for line in sh("git", "log", "--since=2026-10-30", "--reverse", "--pretty=%h|%aI|%s").splitlines():
        h, d, s = line.split("|", 2)
        if "swift-mk" in s or "experiment" in s or "exp:" in s or "idea:" in s:
            m = txt(re.sub(r"^\[agent:[^\]]+\]\s*", "", s))
            if EXCL and EXCL.search(m):
                continue
            rows.append({"hash": h, "date": d[:16].replace("T", " "), "msg": m})
    return rows


def log_lines():
    src = sh("git", "show", "HEAD:log.md") if DUMP_TRACE else (ROOT / "log.md").read_text()
    out = []
    for l in src.splitlines():
        if any(k in l for k in ("megakernel", "swift-mk")):
            l = txt(l)
            if not (EXCL and EXCL.search(l)):
                out.append(l)
    return out


if DUMP_TRACE:  # research layout only: snapshot the (filtered) git/log trace so the public tree can regenerate without the research repo
    Path(DUMP_TRACE).write_text(json.dumps({"commits": commits(), "log": log_lines()}, indent=1))
    print("wrote", DUMP_TRACE)
    raise SystemExit(0)

TRACE = DASH / "trace.json"
if PUBLIC and TRACE.exists():
    _t = json.loads(TRACE.read_text())
    _commits, _log = _t["commits"], _t["log"]
elif PUBLIC:
    _commits, _log = [], []
else:
    _commits, _log = commits(), log_lines()

_gemv = ENGINE / "GEMV_LOG.md"
data = {
    "generated": subprocess.run(["date", "+%F %H:%M"], capture_output=True, text=True).stdout.strip(),
    "step_ms": [r for r in map(keep_row, STEP_MS) if r],
    "spec": {"labels": SPEC_TOKS["labels"], "series": [r for r in map(keep_row, SPEC_TOKS["series"]) if r]},
    "predictions": [r for r in map(keep_row, PREDICTIONS) if r], "agents": [r for r in map(keep_row, AGENTS) if r],
    "experiments": experiments(), "ideas": ideas(), "commits": _commits,
    "gemv_log": clean_lines(_gemv.read_text()[:9000]) if _gemv.exists() else "",
    "log": _log,
}

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>swift-mk trace</title>
<style>
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ee;--mut:#8b93a7;--ok:#3ecf8e;--bad:#ff6b6b;--warn:#f5c04a;--acc:#6ea8fe;--bd:#262b36}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
header{padding:14px 22px;border-bottom:1px solid var(--bd);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}small,.mut{color:var(--mut)}nav{display:flex;gap:6px;flex-wrap:wrap;padding:10px 22px;border-bottom:1px solid var(--bd);position:sticky;top:0;background:var(--bg);z-index:5}
nav button{background:var(--card);color:var(--fg);border:1px solid var(--bd);border-radius:6px;padding:5px 11px;cursor:pointer}nav button.on{border-color:var(--acc);color:var(--acc)}
main{padding:18px 22px;max-width:1280px}section{display:none}section.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.card h3{margin:0 0 6px;font-size:14px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid var(--bd);padding:5px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600}.tag{display:inline-block;border-radius:10px;padding:1px 8px;font-size:12px;border:1px solid var(--bd)}
.ok{color:var(--ok);border-color:var(--ok)}.bad{color:var(--bad);border-color:var(--bad)}.warn{color:var(--warn);border-color:var(--warn)}
pre{white-space:pre-wrap;background:#11141a;border:1px solid var(--bd);border-radius:6px;padding:8px;margin:6px 0;max-height:320px;overflow:auto;font:12px/1.4 ui-monospace,monospace}
input[type=search]{background:var(--card);border:1px solid var(--bd);color:var(--fg);padding:6px 10px;border-radius:6px;width:320px;margin-bottom:10px}
details summary{cursor:pointer;color:var(--acc)}.bar{height:14px;border-radius:3px}
.tl{position:relative;margin-left:12px;border-left:2px solid var(--bd);padding-left:14px}.tl .it{margin-bottom:8px;position:relative}
.tl .it:before{content:"";position:absolute;left:-21px;top:6px;width:10px;height:10px;border-radius:50%;background:var(--acc)}
.tl .it.x:before{background:var(--bad)}.tl .it.g:before{background:var(--ok)}
</style></head><body>
<header><h1>swift-mk &mdash; fastest Swift-1.5-Qwen3.8-27b-INT4 decode on one RTX 3090</h1><small id="gen"></small></header>
<nav id="nav"></nav><main id="main"></main>
<script>
const D=__DATA__;
const $=(s,e=document)=>e.querySelector(s);const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
$('#gen').textContent='generated '+D.generated+' from git + experiment READMEs';
const tabs=['Overview','Trace','Experiments','Predictions vs reality','Failures & dead ends','Ideas','Work packages','GEMV log'];
const nav=$('#nav'),main=$('#main');
function bars(title,labels,series,unit){ // grouped horizontal bars (inline SVG-free, CSS)
  const max=Math.max(...series.flatMap(s=>s[1]));const cols=['#6ea8fe','#3ecf8e','#f5c04a','#b39ddb','#ff9f6b'];
  let h=`<div class="card"><h3>${title}</h3>`;
  labels.forEach((l,i)=>{h+=`<div style="margin:8px 0 2px" class="mut">${l}</div>`;series.forEach((s,j)=>{const v=s[1][i];h+=`<div style="display:flex;align-items:center;gap:8px"><div style="width:170px;font-size:12px" class="mut">${esc(s[0])}</div><div class="bar" style="width:${v/max*55}%;background:${cols[j%5]}"></div><div>${v}</div></div>`})});
  return h+`<div class="mut" style="margin-top:6px">${unit}</div></div>`}
function steps(){const mx=Math.max(...D.step_ms.map(r=>r[1])),mn=Math.min(...D.step_ms.map(r=>r[1]))-1;
  let h='<div class="card"><h3>Plain decode step time (ms, lower is better; roofline floor ~16.2 ms at 936 GB/s)</h3><table><tr><th>stage</th><th>ms</th><th>tok/s</th><th></th><th>pointer</th></tr>';
  D.step_ms.forEach(r=>{h+=`<tr><td>${esc(r[0])}</td><td>${r[1]}</td><td>${(1000/r[1]).toFixed(1)}</td><td style="width:35%"><div class="bar" style="width:${(r[1]-mn)/(mx-mn+0.5)*100}%;background:var(--acc)"></div></td><td class="mut">${esc(r[2])}</td></tr>`});return h+'</table></div>'}
const V={
Overview(){let h='<div class="grid">'+bars('MTP k=3 decode tok/s (384 tokens, prefill excluded)',D.spec.labels,D.spec.series.map(s=>[s[0],s[1]]),'greedy unless noted; engine uses bf16 KV, vLLM fp8 KV (auto); different measurement sessions &mdash; see caveats in the README')+steps()+'</div>';
  h+='<div class="card"><h3>Where the program stands</h3><ul><li>Engine = fused kernels in a CUDA graph (not a persistent megakernel: E1 showed persistence +2.5% only).</li><li>Plain decode 20.2 ms/step (49.5 tok/s) vs vLLM 21.4 ms; speculative k=3 &asymp;1.5-1.7x vLLM.</li><li>Prefill ~2100 tok/s at 8K (vLLM ~1900); exact speculative sampling; OpenAI-compatible server done (HTTP overhead ~0%).</li></ul><p class="mut">Every number links to a file in the Experiments / Work packages tabs.</p></div>';return h},
Trace(){let h='<div class="card"><h3>Commit trace (swift-mk / experiments / ideas)</h3><div class="tl">';
  D.commits.forEach(c=>{const x=/dead|abandon|cancel|fail|negative|miss/i.test(c.msg);h+=`<div class="it ${x?'x':''}"><span class="mut">${c.date} &middot; ${c.hash}</span><br>${esc(c.msg)}</div>`});
  h+='</div></div><div class="card"><h3>log.md (this program)</h3><pre>'+esc(D.log.join('\n'))+'</pre></div>';return h},
Experiments(){let h='<input type=search id=q placeholder="filter experiments...">';D.experiments.forEach(e=>{
  const c=e.status==='done'?'ok':(e.status==='abandoned'?'bad':'warn');
  h+=`<div class="card exp"><h3>${esc(e.dir)} <span class="tag ${c}">${esc(e.status)}</span> <span class="tag">${esc(e.confidence||'-')}</span></h3>
  <div><b>Q:</b> ${esc(e.question)}</div><details><summary>pre-registered prediction</summary><pre>${esc(e.prediction)}</pre></details>
  <div style="margin-top:6px"><b>A:</b> ${esc(e.answer)||'<span class=mut>no answer recorded</span>'}</div>
  <details><summary>conclusion</summary><pre>${esc(e.conclusion)}</pre></details></div>`});return h},
'Predictions vs reality'(){let h='<div class="card"><h3>Pre-registered predictions vs measurement</h3><table><tr><th>quantity</th><th>predicted</th><th>measured</th><th>verdict</th><th>experiment</th></tr>';
  D.predictions.forEach(p=>{const c=p[3].startsWith('HIT')?'ok':(p[3].includes('better')?'warn':'bad');h+=`<tr><td>${esc(p[0])}</td><td>${esc(p[1])}</td><td>${esc(p[2])}</td><td><span class="tag ${c}">${esc(p[3])}</span></td><td class="mut">${esc(p[4])}</td></tr>`});
  const miss=D.predictions.filter(p=>p[3].startsWith('MISS')).length;return h+`</table><p class="mut">${miss} of ${D.predictions.length} listed listed predictions missed (curated list, biased toward the ones with a clear number; the full set is in each README): wrong guesses were mostly about the baseline (vLLM is faster than expected) and about what is free in the draft path.</p></div>`},
'Failures & dead ends'(){let h='';D.experiments.forEach(e=>{if(!e.dead.trim())return;h+=`<div class="card"><h3>${esc(e.dir)}</h3><pre>${esc(e.dead)}</pre></div>`});
  h+='<div class="card"><h3>Program-level failures</h3><ul><li><b>Persistent megakernel premise</b>: only +2.5% over per-op CUDA graph (E1); PDL impossible on sm_86; E2 (dataflow/fusion) cancelled, still open.</li><li><b>Fused RMSNorm in GEMV prologue</b>: +5 us/GEMV, worse (21.47 vs 20.93 ms) &mdash; removed.</li><li><b>K5 "bit-identical" claim</b>: false; rounding-level differences found by K7 follow-up (be8661e).</li><li><b>Restricted 32k draft vocab</b>: not free (-6% accepted length).</li><li><b>lm_head int8/int4</b>: none passed the pre-registered rule; rescoring hybrid needed (E5).</li><li><b>Process</b>: parallel agents contending for one GPU/lease corrupted timing and blocked work; one stale lease released by another agent.</li></ul></div>';return h},
Ideas(){let h='';D.ideas.forEach(i=>{h+=`<div class="card"><h3>${esc(i.title)} <span class="tag">${esc(i.status)}</span></h3><div class="mut">${esc(i.file)}</div><details open><summary>attempts</summary><pre>${esc(i.attempts)}</pre></details></div>`});return h},
'Work packages'(){let h='<div class="card"><h3>Agent work packages</h3><table><tr><th>agent</th><th>task</th><th>outcome / pointer</th></tr>';D.agents.forEach(a=>{h+=`<tr><td>${esc(a[0])}</td><td>${esc(a[1])}</td><td class="mut">${esc(a[2])}</td></tr>`});return h+'</table></div>'},
'GEMV log'(){return '<div class="card"><h3>engine/GEMV_LOG.md (every attempt, kept and failed)</h3><pre style="max-height:none">'+esc(D.gemv_log)+'</pre></div>'}
};
function show(t){[...nav.children].forEach(b=>b.classList.toggle('on',b.textContent===t));main.innerHTML='<section class="on">'+V[t]()+'</section>';location.hash=t;
  const q=$('#q');if(q)q.oninput=()=>document.querySelectorAll('.exp').forEach(c=>c.style.display=c.textContent.toLowerCase().includes(q.value.toLowerCase())?'':'none')}
tabs.forEach(t=>{const b=document.createElement('button');b.textContent=t;b.onclick=()=>show(t);nav.appendChild(b)});
show(decodeURIComponent(location.hash.slice(1))||'Overview');
</script></body></html>"""
OUT.write_text(HTML.replace("__DATA__", json.dumps(data)))
print("wrote", OUT, len(HTML) // 1024, "KB")
