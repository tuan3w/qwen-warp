"""ngram-draft-potential.

  # 1. CPU: build the 32 prompts (workload.py), 2. GPU (lease held): generate + trace, 3. CPU: offline PLD / hybrid simulation
  ~/llm-lab/venv/bin/python experiments/2026-10-31-ngram-draft-potential/workload.py
  SMK_ROOT=<engine tree> ~/llm-lab/venv/bin/python experiments/2026-10-31-ngram-draft-potential/run.py gen [--only a1,b2]
  python3 experiments/2026-10-31-ngram-draft-potential/analyze.py

`gen` (engine-direct, SpecEngine + FastPrefill, greedy, thinking off, <= 768 new tokens, EOS stops) runs every task twice on the same prompt:
  N  natural MTP k=3 cycles, one cycle per host sync: per cycle the accepted-draft count and the per-position acceptance (measured MTP-only behaviour);
  F  forced single-row cycles (`set_force([1])`: every cycle verifies [anchor, d1, d2, d3] but accepts only the anchor's successor), so EVERY position of the
     greedy continuation is a cycle start: per position i we record the anchor g_i, the MTP drafts d1..d3 of that position and the natural accepted count m_i
     (histogram delta).  F's token stream is the canonical greedy continuation used by the offline simulation.
The engine tree is a private build of HEAD (git archive + make; other agents edit and rebuild the live tree), selected with SMK_ROOT.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENGINE = Path(os.environ.get("SMK_ROOT", ROOT / "engine"))
MODEL = Path.home() / "models" / "swift-1.5-int4"
EOS = {248046, 248044}
NMAX = 768
K = 3


def gen(only):
    import torch

    sys.path.insert(0, str(ENGINE))
    sys.path.insert(0, str(ENGINE / "tests"))
    from spec import SpecEngine
    from prefill import FastPrefill
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    tasks = json.loads((HERE / "runs" / "prompts.json").read_text())
    if only:
        tasks = [t for t in tasks if any(t["id"].startswith(o) for o in only)]
    out_dir = HERE / "runs" / "trace"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_all = {t["id"]: tok(t["text"], add_special_tokens=False)["input_ids"] for t in tasks}
    ctx = max(len(v) for v in ids_all.values()) + NMAX + 96
    e = SpecEngine(ctx_max=ctx, verbose=False)
    fp = FastPrefill(e)
    print(f"engine ready, ctx_max {ctx}, build {os.environ.get('SMK_BUILD', 'build')}", flush=True)
    for t in tasks:
        fn = out_dir / f"{t['id']}.json"
        if fn.exists():
            continue
        ids = ids_all[t["id"]]
        rec = dict(id=t["id"], cls=t["cls"], kind=t["kind"], n_prompt=len(ids))
        # ---- N: natural MTP cycles, one per sync
        e.reset()
        t0 = time.time()
        first = fp.prefill_spec(ids, K)
        torch.cuda.synchronize()
        rec["prefill_s_N"] = time.time() - t0
        toks, cyc_a, cyc_new = [first], [], []
        prev_hist = [0] * 4
        t0 = time.time()
        while len(toks) < NMAX and toks[-1] not in EOS:
            new, ss = e.cycles(K, 1)
            toks += new
            cyc_a.append(len(new) - 1)
        torch.cuda.synchronize()
        dt = time.time() - t0
        toks = toks[:NMAX]
        cut = next((i for i, x in enumerate(toks) if x in EOS), None)
        if cut is not None:
            toks = toks[: cut + 1]
        rec["N"] = dict(tokens=toks, accepted=cyc_a, decode_s=dt, hist=ss[16:20])
        # ---- F: forced single-row cycles, every position a cycle start
        e.reset()
        e.set_force([1])
        first = fp.prefill_spec(ids, K)
        g, D, M = [first], [], []
        hist = [0] * 4
        while len(g) < NMAX and g[-1] not in EOS:
            pos, cur, st = e.draft_state()  # st = [anchor, d1, d2, d3] of the cycle about to run
            assert st[0] == g[-1], (st, g[-3:])
            new, ss = e.cycles(K, 1)
            h = ss[16:20]
            m = [i for i in range(4) if h[i] != hist[i]]
            assert len(m) == 1 and len(new) == 1, (m, new)
            hist = h
            D.append(st[1:4])
            M.append(m[0])
            g += new
        rec["F"] = dict(tokens=g, drafts=D, m=M)
        rec["prompt_ids"] = ids
        fn.write_text(json.dumps(rec))
        eq = rec["N"]["tokens"] == g[: len(rec["N"]["tokens"])] and len(rec["N"]["tokens"]) == len(g)
        print(f"{t['id']:20s} prompt {len(ids):5d} gen N {len(toks):4d} F {len(g):4d} equal={eq} E_N {len(toks) / max(len(cyc_a), 1):.2f} E_F-natural {1 + sum(M) / len(M):.2f}  decode N {len(toks) / dt:.1f} tok/s (1 cycle/sync)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen"])
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    gen([x for x in a.only.split(",") if x])
