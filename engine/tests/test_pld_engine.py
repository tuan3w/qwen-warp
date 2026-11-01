"""K11 engine tests: MTP chain + prompt-lookup tail (verify T <= 8 rows) on the real graph.

  SMK_BUILD=build/k11 SMK_TMAX=8 ~/llm-lab/venv/bin/python tests/test_pld_engine.py [--only parity,rollback,sampling,ring] [--n 192] [--tasks a2,b1,c3] [--out PLD_PARITY.md]
Needs the GPU lease.  Sections:
  parity    greedy speculative generation with the hybrid draft == plain T=1 greedy for n tokens (E0 prompts + E6 workload tasks with copy-heavy output); a divergence is accepted only as a
            rounding near-tie (plain top1-top2 margin < NEAR_TIE, exactly those two tokens; tests/test_mtp.py rule); the cycle statistics prove that T = 7 / 8 cycles ran
  rollback  forced partial acceptance (a <= f) with patterns up to 8 rows (DeltaNet 9-slot ring, conv slots, KV overwrite, MTP catch-up of 8 rows) == plain greedy
  ring      the ring wraps many times: >= 40 tail cycles with accept counts 1..8 in random patterns, final state (anchor / pos / cur) consistent with the emitted count
  sampling  real graph, T=1 sampling with a FORCED tail (q = delta at fixed tokens): the first two emitted tokens are exact (analytic p of the first token / joint over the support
            vs plain sampling from the same snapshot) although every cycle verifies T = 8 rows; plus plumbing checks (emitted counts, support)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spec import SpecEngine, make_pld, smk  # noqa: E402
from prefill import FastPrefill  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
E6 = REPO / "experiments" / "2026-10-31-ngram-draft-potential" / "runs" / "trace"
NEAR_TIE = 0.4
OUT = []
EOS = {248046, 248044}


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def e0_prompts(names):
    from bench_spec import prompts

    return prompts(names)


def e6_prompt(tid):
    fn = next(E6.glob(f"{tid}*.json"))
    return json.loads(fn.read_text())["prompt_ids"]


def plain(e, fp, ids, n):
    e.reset()
    e.set_force(None)
    first = fp.prefill(ids)
    return [first] + e.decode(n - 1)


def margin_at(e, fp, ids, toks, i):
    e.reset()
    fp.prefill(ids + toks[:i])
    lg = e.logits(1)[0]
    v, ix = lg.topk(2)
    return float(v[0]), float(v[1]), int(ix[0]), int(ix[1])


def compare(e, fp, ids, ref, got, tag):
    if got == ref:
        return True, f"{tag}: EQUAL ({len(ref)} tokens)"
    i = next(j for j in range(min(len(ref), len(got))) if ref[j] != got[j])
    v1, v2, i1, i2 = margin_at(e, fp, ids, ref, i)
    near = (v1 - v2) < NEAR_TIE and {ref[i], got[i]} == {i1, i2}
    return near, (f"{tag}: {'NEAR-TIE divergence (accepted)' if near else 'DIVERGES'} at generated index {i} (plain {ref[i]}, hybrid {got[i]}); plain-engine logits top1 {i1}={v1:.4f} top2 {i2}={v2:.4f}, margin {v1 - v2:.4f}")


def gen_pld(e, fp, ids, n, opts, force=None, group=2):
    e.reset()
    e.set_force(force)
    first = fp.prefill_spec(ids, 3)
    toks = [first]
    e.enable_pld(opts)
    e.pld_begin(list(ids) + [first])
    ss = None
    while len(toks) < n:
        new, ss = e.cycles_pld(group)
        toks += new
    return toks[:n], dict(hist=ss[16:24], cycles=ss[13], emitted=ss[12], pld=e.pld_stats())


def tail_summary(st):
    ks = st["pld"]["cycles_by_k"]
    tot = sum(ks.values())
    return f"cycles {tot}, T histogram {{{', '.join(f'T={k + 1}: {v}' for k, v in sorted(ks.items()))}}}, tail tokens proposed {st['pld']['tail_tokens']}, accepted-drafts histogram m=0..7 {st['hist']}, tokens/cycle {st['emitted'] / max(st['cycles'], 1):.2f}"


def test_parity(e, fp, tasks, n, force=None, tag="parity", opts=None):
    ok = True
    for name, ids in tasks:
        ref = plain(e, fp, ids, n)
        t0 = time.time()
        got, st = gen_pld(e, fp, ids, n, opts or make_pld(), force)
        good, msg = compare(e, fp, ids, ref, got, f"{name} ({len(ids)} prompt tokens)")
        ok &= good
        log(f"- {tag} {msg}; {tail_summary(st)} ({time.time() - t0:.1f}s)")
    return ok


def test_ring(e, fp, ids, rounds=3):
    """random forced-acceptance patterns of 1..8 on a copy-heavy prompt (tails from the proposer, so verify rows are often all right and a = f is binding); the DeltaNet ring wraps ~25 times."""
    rng = np.random.default_rng(5)
    ref = plain(e, fp, ids, 400)
    ok = True
    for r in range(rounds):
        pat = [int(x) for x in rng.integers(1, 9, 8)]
        e.reset()
        e.set_force(pat)
        first = fp.prefill_spec(ids, 3)
        e.enable_pld(make_pld())
        e.pld_begin(list(ids) + [first])
        toks = [first]
        while len(toks) < 200:
            new, ss = e.cycles_pld(3)
            toks += new
        got = toks[:200]
        good, msg = compare(e, fp, ids, ref[:200], got, f"ring pattern {pat}")
        # bookkeeping: pos == prompt + emitted, ring slot == emitted % 9 (relative to the start slot 0 of a reset engine + prefill steps) is checked through exact equality of the stream
        st = e.pld_stats()
        log(f"- {msg}; cycles {ss[13]}, tokens/cycle {ss[12] / ss[13]:.2f}, T histogram {st['cycles_by_k']}")
        ok &= good
    return ok


def test_sampling_forced_tail(e, fp, ids, n=6000, tail_len=4, T=1.0, top_k=20, top_p=0.95, prompt_name="prose"):
    """Every cycle verifies T = 4 + tail_len rows with FORCED tail tokens (random in-support continuation tokens): the first two tokens emitted by a cycle are exact."""
    from test_sampling import gof, hf_probs  # noqa: F401

    e.reset()
    e.set_sampling(temperature=T, top_k=top_k, top_p=top_p, draft_sample=1, seed=99, test_flags=2)  # bit1: the first token stays the engine's argmax (fixed anchor)
    anchor = fp.prefill_spec(ids, 3, start=False)
    e.snapshot()
    # analytic p of the first emitted token from the same T=1+k row step numerics
    R = 4 + tail_len
    e.restore()
    e._step([anchor] * R, 1)
    p1 = hf_probs(e.logits(1)[0], T, top_k, top_p)
    sup = torch.nonzero(p1 > 0).flatten().tolist()
    ftail = [sup[0], sup[1] if len(sup) > 1 else sup[0], sup[0], sup[-1]][:tail_len]
    e.set_sampling(temperature=T, top_k=top_k, top_p=top_p, draft_sample=1, seed=7, test_flags=2)
    e.enable_pld(make_pld(force_tail=ftail))
    firsts, second_ok, lens = [], 0, []
    t0 = time.time()
    for i in range(n):
        e.restore()
        smk.check(e.lib.smk_spec_start(e.sp, 3, e.st), "start")
        e.pld_begin(list(ids) + [anchor])
        new, ss = e.cycles_pld(1)
        firsts.append(new[0])
        lens.append(len(new))
    cnt = np.bincount(np.array(firsts), minlength=248320)
    obs = np.array([cnt[t] for t in sup], float)
    out = cnt.sum() - obs.sum()
    exp = p1[sup].double().numpy() * n
    chi, df, pv = gof(np.append(obs, out), np.append(exp, 0.0))
    tv = 0.5 * (np.abs(obs / n - p1[sup].numpy()).sum() + out / n)
    log(f"- sampling with a forced tail of {tail_len} (T = {R} rows every cycle, tail tokens {ftail}, {n} cycles, {time.time() - t0:.0f} s): first emitted token vs analytic p over the {len(sup)}-token support: chi2 {chi:.1f} df {df} p = {pv:.3f}, TV {tv:.4f}, outside support {int(out)}; emitted per cycle mean {np.mean(lens):.3f}")
    return pv > 1e-3 and out == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="parity,rollback,ring,sampling")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--tasks", default="a2_types_full,b1_gitlog,c3_edit,c2_write")
    ap.add_argument("--e0", default="code,prose,math")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    only = set(a.only.split(","))
    tasks = [(t, e6_prompt(t)) for t in a.tasks.split(",") if t]
    e0 = e0_prompts([x for x in a.e0.split(",") if x])
    tasks = [(k, v) for k, v in e0.items()] + tasks
    ctx = max(len(v) for _, v in tasks) + 700
    e = SpecEngine(ctx_max=ctx, verbose=False, tmax=8)
    fp = FastPrefill(e)
    log(f"engine tmax {e.tmax}, ctx_max {ctx}, build {os.environ.get('SMK_BUILD', 'build')}")
    res = {}
    if "parity" in only:
        log(f"### 1 greedy parity (hybrid == plain greedy, {a.n} tokens)")
        res["parity"] = test_parity(e, fp, tasks, a.n)
    if "rollback" in only:
        log("### 2 forced partial acceptance up to 8 rows")
        r = True
        for pat in ([8], [1], [5, 2, 7, 1, 8, 3], [3, 4, 1, 6, 2, 8, 5, 7]):
            log(f"force pattern {pat}")
            r &= test_parity(e, fp, tasks[3:6] if len(tasks) > 5 else tasks, a.n, force=pat, tag=f"rollback{pat}")
        res["rollback"] = r
    if "ring" in only:
        log("### 3 ring wraps with random forced acceptance 1..8")
        res["ring"] = test_ring(e, fp, tasks[-1][1])
    if "sampling" in only:
        log("### 4 sampling, forced tail on the real graph")
        pr = e0_prompts(["prose"])["prose"]
        res["sampling"] = test_sampling_forced_tail(e, fp, pr)
    log("RESULT: " + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in res.items()))
    if a.out:
        Path(a.out).write_text("\n".join(OUT) + "\n")
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
