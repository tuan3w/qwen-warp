#!/usr/bin/env python
"""Layer-major prefill of several segments (FastPrefill._group) is bit-identical to running the segments one at a time.

  python tests/test_group_prefill.py [--no-gpu] [--kv-dtype 0|1|2] [--ctx 16384]
Part 1 (no GPU): Runner.plan_groups on random plans: every segment covered once in order, rows per group <= seg, a group ends at every snapshot position, a slow-path segment (fewer than
min_fast rows) is alone, enabled=False gives one segment per group.
Part 2 (GPU + checkpoint): the same prompt prefilled with group_prefill off (reference) and on, for prompts with few and many chat-turn cuts: every DeltaNet snapshot (state, conv, int4
aux), the KV rows of the main cache and of the MTP layer up to the last snapshot, and the generated tokens (greedy and seeded sampling) must be EQUAL (torch.equal), not close.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import smk_runner as R  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def part1():
    rng = random.Random(7)
    for case in range(3000):
        seg = rng.choice([512, 4096])
        min_fast = 2
        ends, e = [], rng.randrange(0, 300)
        while len(ends) < rng.randrange(1, 40):
            e += rng.choice([1, 2, 240, 241, 300, 900, 4096]) if rng.random() < 0.9 else rng.randrange(1, 5000)
            ends.append(e)
        q = rng.choice([0] + ends[: len(ends) // 2])
        snap_at = set(rng.sample(ends, k=min(len(ends), rng.randrange(0, 4))))
        for enabled in (True, False):
            gs = R.Runner.plan_groups(ends, q, seg, min_fast, snap_at, enabled)
            flat = [x for g in gs for x in g]
            want, i = [], q
            for j in (x for x in ends if x > q):
                want.append((i, j))
                i = j
            assert flat == want, (case, "coverage / order")
            for g in gs:
                assert all(a2 == b1 for (_, a2), (b1, _) in zip(g, g[1:])), "contiguous"
                if len(g) > 1:
                    assert sum(b - a for a, b in g) <= seg, (case, "rows per group")
                    assert all(b - a >= min_fast for a, b in g), (case, "slow segment inside a group")
                    assert not any(b in snap_at for _, b in g[:-1]), (case, "snapshot position inside a group")
                if any(b - a < min_fast for a, b in g):
                    assert len(g) == 1, (case, "slow segment must be alone")
            if not enabled:
                assert all(len(g) == 1 for g in gs), "disabled = one segment per group"
    check(True, "plan_groups: coverage, rows <= seg, groups end at snapshot positions, slow segments alone, disabled = per segment (3000 random plans x 2)")


def conv(seed, n, turn):
    rng = np.random.RandomState(seed)
    ids = [int(x) for x in rng.randint(1000, 30000, size=n)]
    if turn:
        ids = [R.IM_START if i % turn == 0 else (872 if i % turn == 1 else t) for i, t in enumerate(ids)]
    return ids


def prefill_state(r, ids):
    """Prefill `ids` on a cold runner, return everything the prefill leaves behind (cloned to host) + the generated tokens."""
    r.drop_cache()
    toks = []
    for out, fin, _ in r.run(ids, R.GenParams(max_tokens=24)):
        toks += out
        if fin is not None:
            break
    snaps = [(s.pos, s.dn.clone(), s.conv.clone(), None if s.aux is None else s.aux.clone()) for s in sorted(r.snaps, key=lambda x: x.pos)]
    s_max = max(p for p, *_ in snaps)
    m = s_max
    main, mtp = r.e.kv_chunk_new(m), r.e.mtp_chunk_new(m)
    r.e.kv_chunk_copy(main, m, 0, 0, m, True)
    r.e.mtp_chunk_copy(mtp, m, 0, 0, m, True)
    torch.cuda.synchronize()
    return toks, snaps, main.clone(), mtp.clone(), s_max


def part2(a):
    r = R.Runner(ctx_max=a.ctx, kv_fp8=a.kv_dtype, k=3, group=2, max_snaps=16, host_tables=True, prefix_cache_gb=0, pld=False)
    for name, ids in (("plain 6000 (few cuts)", conv(1, 6000, 0)), ("marker every 900, 9000 tokens", conv(2, 9000, 900)), ("marker every 300, 9000 tokens", conv(3, 9000, 300)),
                      ("marker every 260 (just above w4_max), 5000 tokens", conv(4, 5000, 260)), ("marker every 700, 12000 tokens", conv(5, 12000, 700))):
        ends, _ = r._plan(ids)
        groups = R.Runner.plan_groups(ends, 0, r.fp.seg, r.fp.min_fast, set(ends[-2:]), True)
        r.group_prefill = False
        ref = prefill_state(r, ids)
        r.group_prefill = True
        got = prefill_state(r, ids)
        same_tok = ref[0] == got[0]
        same_snap = len(ref[1]) == len(got[1]) and all(x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) and (x[3] is None or torch.equal(x[3], y[3])) for x, y in zip(ref[1], got[1]))
        same_kv = ref[4] == got[4] and torch.equal(ref[2], got[2]) and torch.equal(ref[3], got[3])
        check(same_tok and same_snap and same_kv, f"{name}: {len(ends) + 1} segments in {len(groups)} groups -> tokens, {len(ref[1])} snapshots, main + MTP KV rows [0, {ref[4]}) all bit-identical to the one-at-a-time prefill")
    # seeded sampling and a repeat through the cache
    ids = conv(6, 9000, 300)
    outs = []
    for g in (False, True):
        r.group_prefill = g
        r.drop_cache()
        toks = []
        for out, fin, _ in r.run(ids, R.GenParams(max_tokens=24, temperature=0.8, top_k=20, top_p=0.95, seed=99)):
            toks += out
            if fin is not None:
                break
        outs.append(toks)
    check(outs[0] == outs[1], "seeded sampling after a grouped prefill equals the one-at-a-time prefill")
    print("ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--kv-dtype", type=int, default=2, choices=[0, 1, 2])
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()
    part1()
    if not a.no_gpu:
        t0 = time.time()
        part2(a)
        print(f"gpu part {time.time() - t0:.0f} s")
