"""CPU only: results/visit_*.json (+ runs/hq/visit_*.json for token ids) -> results/tables.md + results/summary.json.   run.py tables  /  python tables.py"""
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RAW = HERE.parents[1] / "runs" / "hq"
ORDER = ["ours", "ours-fp8", "mtp-fast", "mtp-long", "dflash2", "dflash15", "dflash2-long"]


def med(x):
    x = [v for v in x if v == v]
    return statistics.median(x) if x else float("nan")


def rng(x):
    x = [v for v in x if v == v]
    return f"{min(x):.0f}-{max(x):.0f}" if x else "-"


def load(suite):
    V = defaultdict(list)
    for f in sorted(glob.glob(str(RES / f"visit_*_{suite}_r*.json"))):
        d = json.load(open(f))
        if "error" in d and suite == "full" and "e0" not in d:
            continue
        V[d["server"]].append(d)
    return V


def fmt(vs, key):
    return f"{med(vs):.1f} [{rng(vs)}]"


def main():
    out = []
    S = {}
    full = load("full")
    servers = [s for s in ORDER if s in full]
    out.append("## E0 prompts, decode tok/s (HTTP, 385 tokens; median over rounds [min-max]; n = rounds)\n")
    for mode, key in (("greedy", "e0"), ("T=1, top_p .95, top_k 20", "e0_T1")):
        out.append(f"**{mode}**\n")
        out.append("| server | rounds | code | prose | math | 8K summarisation |\n|---|---|---|---|---|---|")
        for s in servers:
            vs = [v for v in full[s] if key in v]
            row = [fmt([v[key][p]["decode"] for v in vs if p in v[key]], p) for p in ("code", "prose", "math", "long8k")]
            out.append(f"| {s} | {len(vs)} | " + " | ".join(row) + " |")
            S.setdefault(s, {})[key] = {p: med([v[key][p]["decode"] for v in vs if p in v[key]]) for p in ("code", "prose", "math", "long8k")}
        out.append("")
    out.append("## TTFT (ms; median over reps and rounds)\n\n| server | 20-token prompt | 8K prompt (8,031 tok) |\n|---|---|---|")
    for s in servers:
        vs = [v for v in full[s] if "ttft" in v]
        a = [x for v in vs for x in v["ttft"]["t20_s"]]
        b = [x for v in vs for x in v["ttft"]["t8k_s"]]
        out.append(f"| {s} | {med(a) * 1e3:.0f} [{min(a) * 1e3:.0f}-{max(a) * 1e3:.0f}] | {med(b) * 1e3:.0f} [{min(b) * 1e3:.0f}-{max(b) * 1e3:.0f}] |" if a and b else f"| {s} | - | - |")
        if a and b:
            S[s]["ttft"] = dict(t20_ms=med(a) * 1e3, t8k_ms=med(b) * 1e3, prefill_8k_tok_s=vs[0]["ttft"]["prompt8k"] / med(b))
    out.append("")
    out.append("## Tokens per engine step and ms per step on the E0 prompts (greedy; HyperQwen: one SSE chunk = one engine step; swift-mk: one chunk = `--stream-cycles 2` = 2 cycles, shown per cycle)\n\n| server | code tok/step | prose | math | 8K | code ms/step | prose | math | 8K |\n|---|---|---|---|---|---|---|---|---|")
    for s in servers:
        vs = [v for v in full[s] if "e0" in v]
        div = 2 if s.startswith("ours") else 1
        tps, mss = [], []
        for p in ("code", "prose", "math", "long8k"):
            t = med([v["e0"][p]["total"] / v["e0"][p]["chunks"] / div for v in vs])
            d_ = med([v["e0"][p]["decode"] for v in vs])
            tps.append(t)
            mss.append(1000 * t / d_)
        out.append(f"| {s} | " + " | ".join(f"{x:.2f}" for x in tps) + " | " + " | ".join(f"{x:.1f}" for x in mss) + " |")
        S[s]["tok_per_step"] = dict(zip(("code", "prose", "math", "long8k"), tps))
        S[s]["ms_per_step"] = dict(zip(("code", "prose", "math", "long8k"), mss))
    out.append("")
    for setname, title in (("e6", "E6 tasks (32 prompts; a code edits, b log extraction, c agent tool-call turns, d prose, e math; 256 new tokens)"), ("e8", "E8 English tasks (48 prompts; 256 new tokens)")):
        for mode in ("greedy", "T1"):
            key = f"{setname}_{mode}"
            out.append(f"## {title}, {mode}: decode tok/s (median over tasks of the class, median over rounds [min-max])\n")
            classes = sorted({x["cls"] for s in servers for v in full[s] if key in v for x in v[key]})
            out.append("| server | rounds | " + " | ".join(classes) + (" | pooled a-c" if setname == "e6" else "") + " | pooled all |\n|---|---|" + "---|" * (len(classes) + (2 if setname == "e6" else 1)))
            for s in servers:
                vs = [v for v in full[s] if key in v]
                if not vs:
                    continue
                row, perc = [], {}
                for c in classes:
                    xs = [med([x["decode"] for x in v[key] if x["cls"] == c]) for v in vs]
                    perc[c] = med(xs)
                    row.append(fmt(xs, c))
                ac = [med([x["decode"] for x in v[key] if x["cls"] in ("a", "b", "c")]) for v in vs]
                al = [med([x["decode"] for x in v[key]]) for v in vs]
                out.append(f"| {s} | {len(vs)} | " + " | ".join(row) + f" | {med(ac):.0f} [{rng(ac)}] | {med(al):.0f} [{rng(al)}] |" if setname == "e6" else
                           f"| {s} | {len(vs)} | " + " | ".join(row) + f" | {med(al):.0f} [{rng(al)}] |")
                S.setdefault(s, {})[key] = dict(per_class=perc, pooled_all=med(al), pooled_abc=med(ac) if setname == "e6" else None)
            out.append("")
    # GPU clocks / power during decode
    out.append("## GPU state during the E0 decode phase (nvidia-smi 1 Hz while the GPU was active; median over rounds)\n\n| server | SM clock MHz | power W (median) | power W (max) | temp C max | power limit W |\n|---|---|---|---|---|---|")
    for s in servers:
        ph = [v["gpu_phases"]["e0"] for v in full[s] if "e0" in v.get("gpu_phases", {})]
        if ph:
            out.append(f"| {s} | {med([p['sm_mhz_med'] for p in ph]):.0f} | {med([p['power_w_med'] for p in ph]):.0f} | {max(p['power_w_max'] for p in ph):.0f} | {max(p['temp_c_max'] for p in ph):.0f} | {ph[0]['power_limit_w']:.0f} |")
    out.append("")
    out.append("boot to /health (s): " + ", ".join(f"{s} {med([v['boot_s'] for v in full[s] if 'boot_s' in v]):.0f}" for s in servers) + "\n")
    # concurrency
    conc = load("conc")
    if conc:
        out.append("## Concurrency (greedy, 192 new tokens per request, 2 rounds per cell; aggregate = all tokens / wall time from first request to last token; per-stream = median decode of each stream)\n")
        for kind in ("short", "4k"):
            out.append(f"**{kind} prompts** ({'E8 chat prompts 30-1100 tokens' if kind == 'short' else 'distinct ~3.9k-token prompts'})\n")
            out.append("| server | C | aggregate e2e tok/s | aggregate decode-window tok/s | per-stream decode tok/s | TTFT med ms | TTFT max ms |\n|---|---|---|---|---|---|---|")
            for s in [x for x in ORDER if x in conc]:
                for v in conc[s]:
                    for C in (1, 2, 4, 8):
                        rr = v.get("conc", {}).get(f"{kind}_{C}")
                        if rr:
                            out.append(f"| {s} | {C} | {med([r['agg_e2e'] for r in rr]):.0f} | {med([r['agg_decode'] for r in rr]):.0f} | {med([r['per_stream_decode'] for r in rr]):.0f} | {med([r['ttft_med'] for r in rr]) * 1e3:.0f} | {max(r['ttft_max'] for r in rr) * 1e3:.0f} |")
                            S.setdefault(s, {}).setdefault("conc", {})[f"{kind}_{C}"] = dict(agg_e2e=med([r["agg_e2e"] for r in rr]), agg_decode=med([r["agg_decode"] for r in rr]), per_stream=med([r["per_stream_decode"] for r in rr]))
            out.append("")
    longs = load("long")
    if longs:
        out.append("## Long context (needle at 40 % depth + 10-item summary, greedy, 160 new tokens, EOS honoured)\n\n| server | prompt tokens | TTFT s | prefill tok/s | decode tok/s | needle | note |\n|---|---|---|---|---|---|---|")
        for s in [x for x in ORDER if x in longs]:
            for v in longs[s]:
                for n, r in v.get("long", {}).items():
                    if "error" in r:
                        out.append(f"| {s} {v.get('extra') or ''} | {r['prompt']} | - | - | - | - | ERROR {r['error'][:120]} |")
                    else:
                        out.append(f"| {s} {v.get('extra') or ''} | {r['prompt']} | {r['ttft']:.1f} | {r['prefill_tok_s']:.0f} | {r['decode']:.1f} | {'found' if r['needle_found'] else 'MISSED'} | |")
                if "error" in v and not v.get("long"):
                    out.append(f"| {s} {v.get('extra') or ''} | - | - | - | - | - | server did not start: {v['error'][:160]} |")
        out.append("")
    # parity from token ids (runs/hq)
    ids = defaultdict(dict)
    for f in sorted(glob.glob(str(RAW / "visit_*_full_r*.json"))):
        d = json.load(open(f))
        ids[d["server"]][d["round"]] = d
    def tokens(d, set_, prompts):
        return {x["id"]: x["ids"] for x in d.get(f"{set_}_greedy", []) if x.get("ids")}
    out.append("## Greedy parity: first 128 generated tokens, 30 E8 prompts (the first 30 tasks) + the 32 E6 prompts, same ids sent to both servers\n")
    out.append("| A vs B | set | prompts | identical 128 | mean common-prefix length (of 128) | median | first divergence at <= 16 tokens |\n|---|---|---|---|---|---|---|")
    pairs = [("ours", s) for s in ORDER[1:] if s in ids and "ours" in ids] + [("mtp-fast", s) for s in ("dflash2", "dflash15") if s in ids and "mtp-fast" in ids] + [("ours", "ours")]
    par = {}
    for A, B in pairs:
        for set_, lim in (("e8", 30), ("e6", 32)):
            if A == B:
                if len(ids.get(A, {})) < 2:
                    continue
                r1, r2 = sorted(ids[A])[:2]
                ta, tb = tokens(ids[A][r1], set_, None), tokens(ids[B][r2], set_, None)
            else:
                rA = sorted(ids[A])[0]
                rB = sorted(ids[B])[0]
                ta, tb = tokens(ids[A][rA], set_, None), tokens(ids[B][rB], set_, None)
            keys = [k for k in ta if k in tb][:lim]
            if not keys:
                continue
            cp = []
            for k in keys:
                a_, b_ = ta[k][:128], tb[k][:128]
                n = 0
                while n < min(len(a_), len(b_)) and a_[n] == b_[n]:
                    n += 1
                cp.append(n if n < min(len(a_), len(b_)) else min(len(a_), len(b_)))
            ident = sum(1 for c in cp if c >= 128)
            par[f"{A}|{B}|{set_}"] = dict(n=len(keys), identical=ident, mean_prefix=statistics.mean(cp), median_prefix=statistics.median(cp), early=sum(1 for c in cp if c <= 16))
            out.append(f"| {A} vs {B}{' (round 1 vs 2)' if A == B else ''} | {set_} | {len(keys)} | {ident} | {statistics.mean(cp):.1f} | {statistics.median(cp):.0f} | {sum(1 for c in cp if c <= 16)} |")
    out.append("")
    S["parity"] = par
    (RES / "tables.md").write_text("\n".join(out))
    json.dump(S, open(RES / "summary.json", "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
