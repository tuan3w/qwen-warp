#!/usr/bin/env python
"""Host prefix cache costs (needs the GPU and the checkpoint).

  python scripts/bench_prefix_cache.py --kv-dtype 2 --ctx 70000 --lens 4096,15000,60000 [--reps 3] [--regress]
B2: for each length L: a cold prefill of an L-token conversation, a different conversation (which makes the runner SPILL the first), then the first conversation plus 300 new tokens (a
    RESTORE from the host) and, for comparison, one more turn on that same device lineage.  Reports spill ms, restore ms, prefill s of each, host MB.
B3 (--regress): single conversation, store ON vs OFF, interleaved: cold 12k prefill seconds and decode tok/s (no lineage switching, so the only difference is the code path's own cost).
"""
import argparse
import statistics as S
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import smk_runner as R  # noqa: E402


def conv(seed, n, turn=900):
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        out += [R.IM_START, 872 if len(out) % 2 == 0 else 77091] + [int(x) for x in rng.randint(1000, 30000, size=turn)]
    return out[:n]


def gen(r, ids, max_tokens):
    toks = []
    for out, fin, _ in r.run(ids, R.GenParams(max_tokens=max_tokens)):
        toks += out
        if fin is not None:
            break
    return toks, r.last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dtype", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--ctx", type=int, default=70000)
    ap.add_argument("--lens", default="4096,15000,60000")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--host-tables", action="store_true")
    ap.add_argument("--budget-gb", type=float, default=8.0)
    a = ap.parse_args()
    lens = [int(x) for x in a.lens.split(",")]
    r = R.Runner(ctx_max=a.ctx, kv_fp8=a.kv_dtype, k=3, group=2, max_snaps=16, host_tables=a.host_tables, prefix_cache_gb=a.budget_gb, pld=False)
    other = conv(99, 3000)
    print(f"kv dtype {a.kv_dtype}, ctx {a.ctx}; {'lens ' + a.lens}", flush=True)
    print(f"{'L':>6} {'cold s':>8} {'spill ms':>9} {'host MB':>8} {'restore ms':>11} {'resumed@':>9} {'host-resume prefill s':>22} {'device-continue prefill s':>26}", flush=True)
    for L in lens:
        rows = []
        for rep in range(a.reps):
            base = conv(1000 + L + rep, L)
            r.drop_cache()
            _, st0 = gen(r, base, 4)  # cold
            _, st1 = gen(r, other, 4)  # a different conversation: spills `base`
            mb = r.store.bytes() / 1e6
            _, st2 = gen(r, base + conv(7, 300), 4)  # resume from host
            _, st3 = gen(r, base + conv(7, 300) + conv(8, 300), 4)  # continue the device lineage
            rows.append((st0.prefill_s, st1.spill_s * 1e3, mb, st2.restore_s * 1e3, st2.cached, st2.prefill_s, st3.prefill_s, st2.source, st3.source))
        assert all(x[7] == "host" and x[8] == "device" for x in rows), f"unexpected sources {[(x[7], x[8]) for x in rows]}"
        m = lambda i: S.median([x[i] for x in rows])
        print(f"{L:6d} {m(0):8.2f} {m(1):9.0f} {m(2):8.0f} {m(3):11.0f} {m(4):9.0f} {m(5):22.3f} {m(6):26.3f}", flush=True)
    if a.regress:
        ids = conv(555, 12000)
        res = {"on": [], "off": []}
        store = r.store
        for rep in range(5):
            for mode in (("on", "off") if rep % 2 == 0 else ("off", "on")):
                r.store = store if mode == "on" else None
                r.drop_cache()
                toks, st = gen(r, ids, 200)
                if rep:  # first pair = warm-up
                    res[mode].append((st.prefill_s, (st.tokens - 1) / st.gen_s))
        r.store = store
        for mode in ("on", "off"):
            print(f"store {mode:3s}: cold 12k prefill {S.median([x[0] for x in res[mode]]):.3f} s, decode {S.median([x[1] for x in res[mode]]):.1f} tok/s  (n={len(res[mode])}, ranges "
                  f"{min(x[0] for x in res[mode]):.3f}-{max(x[0] for x in res[mode]):.3f} s, {min(x[1] for x in res[mode]):.1f}-{max(x[1] for x in res[mode]):.1f} tok/s)")


if __name__ == "__main__":
    main()
