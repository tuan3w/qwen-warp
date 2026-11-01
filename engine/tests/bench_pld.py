"""K11 A/B benchmark: hybrid draft (MTP chain + prompt-lookup tail) vs the committed MTP k=3 cycles, SAME process, interleaved, per task, greedy or sampling.

  SMK_BUILD=build/k11 ~/llm-lab/venv/bin/python tests/bench_pld.py --suite e0|e6|long [--tasks a1,b2] [--reps 3] [--temperature 0|1.0] [--modes base,pld,notail] [--json out.json]
Run from the repo root (E0's 8K prompt comes from topics/*.md).  Needs the GPU lease; hold it for the whole run.  tmax = 8 engine (9-slot DeltaNet ring) for every mode.
Modes (comma list; pld<m> = pld with gate m, e.g. pld4): base = smk_spec_run k=3 (the committed cycle graph, `group` cycles per sync, group sized so the last group overshoots the token target by <= ~1 cycle);
       pld = host-driven hybrid cycles (E6's best cell: n=2 longest suffix of the chain, agreement >= 3, tail <= 4);
       notail = the same host loop with the tail gate closed (min_len huge): isolates the per-cycle host round trip (sync + launch gap) from the width effect.
Rate = tokens emitted up to the target / decode wall time (prefill excluded, torch.cuda.synchronize around).  E6 tasks stop at their canonical greedy length (recorded in
the trace) so a task is the same amount of work in every mode; E0 prompts run 384 new tokens after the prefill's first token.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spec import SpecEngine, make_pld  # noqa: E402
from prefill import FastPrefill  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E6 = REPO / "experiments" / "2026-10-31-ngram-draft-potential" / "runs" / "trace"
EOS = {248046, 248044}


def load_suite(suite, only):
    tasks = []
    if suite == "e0":
        from bench_spec import prompts

        names = only or ["code", "prose", "math", "long8k"]
        for n, ids in prompts(names).items():
            tasks.append(dict(id=n, cls=n, ids=ids, n=384, eos=False))
    elif suite == "long":  # K11: long-context prose / summarisation prompts (class f; E6's d / e prompts are short): 6 topic files x 6 instructions, <= 6000 document tokens, 384 new tokens
        from transformers import AutoTokenizer
        from bench_spec import MODEL

        tok = AutoTokenizer.from_pretrained(str(MODEL))
        instr = ["Summarise the main claims above in a numbered list, citing the section names.", "Write a one-page critique of the weakest argument above.",
                 "Explain the content above to a first-year undergraduate.", "List every open question the text raises, with a sentence on why it matters.",
                 "Rewrite the first section above in a more formal academic tone.", "Extract all numbers above and the claims they support as a table."]
        files = sorted((REPO / "topics").glob("*.md"))
        for i, ins in enumerate(instr):
            ids = tok(files[(i * 3) % len(files)].read_text(errors="ignore"), add_special_tokens=False)["input_ids"][:6000]
            text = tok.apply_chat_template([{"role": "user", "content": tok.decode(ids) + "\n\n" + ins}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            tasks.append(dict(id=f"f{i + 1}", cls="f", ids=tok(text, add_special_tokens=False)["input_ids"], n=384, eos=False))
        tasks = [t for t in tasks if not only or t["id"] in only]
    else:
        for fn in sorted(E6.glob("*.json")):
            tid = fn.name.split("_")[0]
            if only and not any(tid == o or fn.name.startswith(o) for o in only):
                continue
            r = json.loads(fn.read_text())
            tasks.append(dict(id=tid, cls=tid[0], ids=r["prompt_ids"], n=len(r["F"]["tokens"]), eos=True, canon=r["F"]["tokens"]))
    return tasks


def run_mode(e, fp, t, mode, temp, opts, seed):
    """-> dict(tokens, sec, cycles, stats)"""
    e.reset()
    e.set_force(None)
    if temp > 0:
        e.set_sampling(temperature=temp, top_k=20, top_p=0.95, draft_sample=1, seed=seed)
    else:
        e.set_sampling(0.0)
    ids, n = t["ids"], t["n"]
    first = fp.prefill_spec(ids, 3)
    toks = [first]
    pld = mode != "base"
    if pld:
        if mode.startswith("pld") and mode[3:].isdigit():  # pld<m>: tail gate with agreement length >= m
            e.enable_pld(make_pld(min_len=int(mode[3:]), max_tail=opts.cfg.max_tail))
        else:
            e.enable_pld(opts if mode == "pld" else make_pld(min_len=1 << 20))
        e.pld_begin(list(ids) + [first])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cyc0 = 0
    while len(toks) < n:
        if t["eos"] and toks[-1] in EOS:
            break
        rem = n - len(toks)
        g = max(1, min(8, rem // 4)) if not pld else max(1, min(4, rem // 4))
        new, ss = e.cycles_pld(g) if pld else e.cycles(3, g)
        toks += new
        if t["eos"] and any(x in EOS for x in new):
            break
    dt = time.perf_counter() - t0
    cut = len(toks)
    if t["eos"]:
        cut = next((i + 1 for i, x in enumerate(toks) if x in EOS), len(toks))
    cut = min(cut, n)
    st = e.pld_stats() if pld else None
    return dict(tokens=cut - 1, emitted=ss[12], sec=dt, cycles=ss[13], hist=ss[16:24], stats=st, seq=toks[:cut])


def parts_ms(e, fp, t, ks=(3, 4, 5, 6, 7), n=20):
    """ms per cycle part for k drafts (verify rows T = k + 1): full cycle, verify + accept, MTP stage (catch-up k+1 rows + 3 drafts), catch-up only; replays on the current state."""
    e.reset()
    e.set_force(None)
    e.set_sampling(0.0)
    fp.prefill_spec(t["ids"], 3)
    out = {}
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for k in ks:
        for which, name in ((0, "cycle"), (1, "verify"), (2, "mtp_stage"), (3, "catchup")):
            e.launch_part(k, which, 3)
            torch.cuda.synchronize()
            ms = []
            for _ in range(5):
                ev0.record()
                e.launch_part(k, which, n)
                ev1.record()
                ev1.synchronize()
                ms.append(ev0.elapsed_time(ev1) / n)
            out[f"k{k}_{name}"] = sorted(ms)[len(ms) // 2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="e0")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--modes", default="base,pld")
    ap.add_argument("--m", type=int, default=3, help="tail gate: agreement length >= m")
    ap.add_argument("--lt", type=int, default=4, help="max tail tokens")
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-seq", action="store_true")
    ap.add_argument("--parts", action="store_true", help="also time the cycle parts for k = 3..7 (full cycle, verify+accept, MTP stage, catch-up) on the first task's state")
    a = ap.parse_args()
    tasks = load_suite(a.suite, [x for x in a.tasks.split(",") if x])
    ctx = max(len(t["ids"]) + t["n"] for t in tasks) + 160
    e = SpecEngine(ctx_max=ctx, verbose=False, tmax=8)
    fp = FastPrefill(e)
    opts = make_pld(min_len=a.m, max_tail=a.lt)
    modes = a.modes.split(",")
    print(f"suite {a.suite}, {len(tasks)} tasks, modes {modes}, reps {a.reps}, T={a.temperature}, tail gate m={a.m} Lt={a.lt}, build {os.environ.get('SMK_BUILD', 'build')}", flush=True)
    # warm-up: capture every graph once (T = 4..8) so no capture lands in a timed run
    t0 = tasks[0]
    for m in modes:
        run_mode(e, fp, dict(t0, n=40), m, a.temperature, opts, 1)
    e.reset()
    e.enable_pld(make_pld(force_tail=[1, 2, 3, 4]))
    first = fp.prefill_spec(t0["ids"], 3)
    e.pld_begin(list(t0["ids"]) + [first])
    for _ in range(3):
        e.cycles_pld(1)
    rows = []
    parts = None
    if a.parts:
        parts = parts_ms(e, fp, tasks[0])
        print("cycle parts (ms, first task, greedy):", json.dumps({k: round(v, 3) for k, v in parts.items()}), flush=True)
    for t in tasks:
        for rep in range(a.reps):
            order = modes if rep % 2 == 0 else modes[::-1]
            for m in order:
                r = run_mode(e, fp, t, m, a.temperature, opts, 1000 + rep)
                r.update(task=t["id"], cls=t["cls"], mode=m, rep=rep, prompt=len(t["ids"]))
                if not a.keep_seq:
                    r.pop("seq")
                rows.append(r)
        line = [f"{t['id']:>18s} n={t['n']:4d}"]
        for m in modes:
            rs = [r for r in rows if r["task"] == t["id"] and r["mode"] == m]
            med = statistics.median(r["tokens"] / r["sec"] for r in rs)
            E = statistics.median(r["emitted"] / max(r["cycles"], 1) for r in rs)
            line.append(f"{m} {med:6.1f} tok/s E {E:.2f}")
        print(" | ".join(line), flush=True)
    # per-class pooled (tokens / time over the tasks, median over reps of the pooled value)
    classes = sorted({t["cls"] for t in tasks})
    print("\nper class (pooled tokens / pooled time; median over reps):")
    summ = {}
    for c in classes:
        ids = [t["id"] for t in tasks if t["cls"] == c]
        row = {}
        for m in modes:
            per_rep = []
            for rep in range(a.reps):
                rs = [r for r in rows if r["task"] in ids and r["mode"] == m and r["rep"] == rep]
                per_rep.append(sum(r["tokens"] for r in rs) / sum(r["sec"] for r in rs))
            tpc = sum(r["emitted"] for r in rows if r["task"] in ids and r["mode"] == m) / sum(r["cycles"] for r in rows if r["task"] in ids and r["mode"] == m)
            row[m] = dict(tok_s=statistics.median(per_rep), reps=per_rep, E=tpc)
            if m != "base":
                st = [r["stats"] for r in rows if r["task"] in ids and r["mode"] == m]
                cyc = sum(sum(s["cycles_by_k"].values()) for s in st)
                row[m].update(tail_share=sum(s["tail_cycles"] for s in st) / max(cyc, 1), wait_us=sum(s["wait_ns"] for s in st) / max(cyc, 1) / 1e3, work_us=sum(s["work_ns"] for s in st) / max(cyc, 1) / 1e3,
                              T_hist={k + 1: sum(s["cycles_by_k"].get(k, 0) for s in st) for k in range(3, 8)})
        summ[c] = row
        base = row.get("base")
        txt = f"class {c}: " + " | ".join(f"{m} {row[m]['tok_s']:.1f} tok/s (E {row[m]['E']:.2f})" + (f" x{row[m]['tok_s'] / base['tok_s']:.3f}" if base and m != "base" else "") for m in modes)
        if "pld" in row:
            txt += f" | tail cycles {row['pld']['tail_share']:.2f}, T hist {row['pld']['T_hist']}, host wait {row['pld']['wait_us']:.0f} us + work {row['pld']['work_us']:.0f} us per cycle"
        print(txt, flush=True)
    if a.json:
        json.dump(dict(args=vars(a), rows=rows, summary=summ, parts=parts), open(a.json, "w"))


if __name__ == "__main__":
    main()
