#!/usr/bin/env python
"""The prefill segmentation (Runner._plan) is a function of the prompt, and a resumed prefill continues the plan of the cold one.  No GPU, no checkpoint (a few seconds).

  python tests/test_prefill_plan.py [--cases 20000]
The fast prefill's numerics depend on where its segments start and end, so a request resumed from a snapshot is bit-identical to a cold one only if the snapshot position is a
segment end of the cold plan of the NEW prompt and every earlier segment is the same.  Checked on random token streams with <|im_start|> markers (small seg / w4_max to reach every
branch, and the real 4096 / 240): for two prompts sharing a prefix, each shared segment end that lies inside the shared prefix has identical earlier ends in both plans; the shape
rules of a plan (ends increase, segments <= seg rows, a folded tail is 2..w4_max rows, cuts at turn boundaries only after more than w4_max rows); the plan of a single-turn prompt is
the one the engine used before the plan existed (cut at the last <|im_start|> + 2 when that is beyond w4_max rows, the last row folded).
"""
import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import smk_runner as R  # noqa: E402


def plan(ids, seg, w4_max):
    return R.Runner._plan(SimpleNamespace(fp=SimpleNamespace(fold=True, seg=seg, w4_max=w4_max)), ids)


def stream(rng, n, p_turn):
    out = []
    while len(out) < n:
        out += [R.IM_START, 7] if rng.random() < p_turn else [rng.randrange(100, 200)]
    return out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=20000)
    a = ap.parse_args()
    rng = random.Random(0)
    fails = []
    shared = 0
    for cfg in ((40, 10), (4096, 240)):
        seg, w4 = cfg
        for _ in range(a.cases if cfg[0] == 40 else a.cases // 10):
            hi = 300 if seg == 40 else 12000
            P = stream(rng, rng.randrange(1, hi), 0.08 if seg == 40 else 0.004)
            m = rng.randrange(0, len(P) + 1)
            Q = P[:m] + stream(rng, rng.randrange(0, hi // 2), 0.08 if seg == 40 else 0.004) if rng.random() < 0.85 else P[:]
            if rng.random() < 0.3 and Q:
                Q = Q[: rng.randrange(1, len(Q) + 1)]
            Q = Q or P[:1]
            L = 0
            while L < min(len(P), len(Q)) and P[L] == Q[L]:
                L += 1
            for ids in (P, Q):
                ends, folded = plan(ids, seg, w4)
                n, prev = len(ids) - 1, 0
                ok = all(0 < e <= n for e in ends) and ends == sorted(set(ends))
                for e in ends:
                    ok &= e - prev <= seg and (e - prev == seg or e == n or (e - prev > w4 and ids[e - 2] == R.IM_START))
                    prev = e
                ok &= (n + 1 - prev <= min(seg, w4) and n + 1 - prev >= 2) if folded else (prev == n)
                if not ok:
                    fails.append(("shape", seg, ends, folded, n))
            eP, eQ = plan(P, seg, w4)[0], plan(Q, seg, w4)[0]
            for e in eP:
                if e in eQ and e <= L - 1:
                    shared += 1
                    if [x for x in eP if x <= e] != [x for x in eQ if x <= e]:
                        fails.append(("prefix", seg, e, eP, eQ))
    # single-turn prompts: the last <|im_start|> + 2 when beyond w4_max rows, else one folded batch (what the engine did before the plan existed)
    for _ in range(2000):
        n = rng.randrange(2, 3000)
        body = stream(rng, n, 0.0)
        pos = rng.randrange(1, n - 1)
        ids = body[:pos] + [R.IM_START, 7] + body[pos + 2 :] if pos + 2 < n else body
        pos = next((i for i in range(len(ids) - 2, -1, -1) if ids[i] == R.IM_START), None)
        ends, folded = plan(ids, 4096, 240)
        n = len(ids) - 1
        exp_cut = [pos + 2] if pos is not None and 240 < pos + 2 <= n else []
        start = exp_cut[0] if exp_cut else 0
        if (ends[: len(exp_cut)], folded) != (exp_cut, 2 <= n + 1 - start <= 240) or (not folded and ends[-1] != n):
            fails.append(("single-turn", pos, n, ends, folded))
    for f in fails[:5]:
        print("FAIL", f)
    print(f"{'PASS' if not fails else 'FAIL'}  plan invariants ({shared} shared segment ends checked, {len(fails)} violations)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
