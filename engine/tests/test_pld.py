#!/usr/bin/env python3
"""E6 host PLD tail proposer (include/pld.h, src/pld.cu) vs the offline reference experiments/2026-10-31-ngram-draft-potential/analyze.py.

CPU only (no GPU).  Library: env SMK_PLD_LIB, else compiled here:  g++ -x c++ -std=c++17 -O2 -shared -fPIC -I include src/pld.cu -o /tmp/smk_pld_test.so
Traces: analyze.load() (results/trace.npz; env NGRAM_RES overrides the results dir).
  (1) default cfg (n=2 longest, m=3, Lt=4), K = prompt+g[:i+1] ++ D[i]:  (a) policy walk (pol_ext, N=1 cycles, positions advance by r+1; C++ driven with reset(prompt+g[0]) + per-cycle propose/append),
      (b) every position (reset(prompt), append g[:i] incrementally, propose at each i), (c) short-draft lengths nd=0,1,2; cross-check of the reference arrays vs analyze.prep() on a few tasks.
  (2) config variants n=1,3,4 x longest/recent (every position vs analyze.lookup with the same key).
  (3) micro-benchmark, 8K-token context.
Exit status != 0 on any mismatch."""
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "2026-10-31-ngram-draft-potential"))
import analyze as A  # noqa: E402

COMPILE = ["g++", "-x", "c++", "-std=c++17", "-O2", "-shared", "-fPIC", "-I", str(ENGINE / "include"), str(ENGINE / "src" / "pld.cu")]


def load_lib():
    path = os.environ.get("SMK_PLD_LIB")
    if not path:
        path = "/tmp/smk_pld_test.so"
        cmd = COMPILE + ["-o", path]
        print("compile:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    lib = ctypes.CDLL(path)
    vp = ctypes.c_void_p
    lib.smk_pld_default_cfg.argtypes = [vp]
    lib.smk_pld_create.argtypes = [vp]
    lib.smk_pld_create.restype = vp
    lib.smk_pld_destroy.argtypes = [vp]
    lib.smk_pld_reset.argtypes = [vp, vp, ctypes.c_int]
    lib.smk_pld_append.argtypes = [vp, vp, ctypes.c_int]
    lib.smk_pld_len.argtypes = [vp]
    lib.smk_pld_len.restype = ctypes.c_int
    lib.smk_pld_propose.argtypes = [vp, vp, ctypes.c_int, vp, vp]
    lib.smk_pld_propose.restype = ctypes.c_int
    return lib


class Cfg(ctypes.Structure):
    _fields_ = [(f, ctypes.c_int) for f in ("n", "min_len", "max_tail", "cap", "back", "longest")]


class Pld:
    def __init__(self, lib, **kw):
        self.lib = lib
        c = Cfg()
        lib.smk_pld_default_cfg(ctypes.byref(c))
        for k, v in kw.items():
            setattr(c, k, v)
        self.h = lib.smk_pld_create(ctypes.byref(c))
        self.out = np.zeros(8, np.int32)
        self.l = np.zeros(1, np.int32)
        self.keep = None

    def reset(self, ctx):
        a = np.ascontiguousarray(ctx, dtype=np.int32)
        self.lib.smk_pld_reset(self.h, a.ctypes.data, len(a))

    def append(self, toks):
        a = np.ascontiguousarray(toks, dtype=np.int32)
        self.lib.smk_pld_append(self.h, a.ctypes.data, len(a))

    def __len__(self):
        return self.lib.smk_pld_len(self.h)

    def propose(self, drafts):
        d = np.ascontiguousarray(drafts, dtype=np.int32)
        ln = self.lib.smk_pld_propose(self.h, d.ctypes.data, len(d), self.out.ctypes.data, self.l.ctypes.data)
        return ln, int(self.l[0]), [int(x) for x in self.out[:ln]]

    def __del__(self):
        try:
            self.lib.smk_pld_destroy(self.h)
        except Exception:
            pass


class Checker:
    def __init__(self):
        self.bad = 0
        self.msgs = []

    def __call__(self, ok, msg):
        if not ok:
            self.bad += 1
            if len(self.msgs) < 12:
                self.msgs.append(msg)
                print("MISMATCH:", msg, flush=True)


def ref_ungated(S, idx, D, P, i, n, strat, nd=3):
    p = P + i
    K = np.concatenate([S[: p + 1], D[i][:nd]]).astype(np.int64)
    return A.lookup(K, idx, n, strat, len(K) - 1, p + 1, L=4)


def ref_arrays(task, S, idx, n=2, strat="longest"):
    """el, tl, rt, rm exactly as analyze.prep() computes the ext cell, plus the unpruned proposal tokens."""
    g, P, D = task["g"], task["P"], task["D"]
    N = len(g) - 1
    el = np.zeros(N, np.int16)
    tl = np.zeros(N, np.int8)
    rt = np.zeros((N, 5), np.int8)
    rm = np.zeros(N, np.int8)
    props = []
    for i in range(N):
        l, c = ref_ungated(S, idx, D, P, i, n, strat)
        el[i], tl[i] = l, len(c)
        props.append(c)
        for Lt in range(5):
            rt[i, Lt] = A.prefix_match(list(D[i]) + c[:Lt], g, i)
        rm[i] = A.prefix_match(list(D[i]), g, i)
    return dict(N=N, ext={(n, strat): (el, tl, rt)}, rm=rm), props


def main():
    lib = load_lib()
    tasks = A.load()
    V = A.make_V()
    rows = []  # (config, positions, mismatches)
    chk = Checker()
    ek = (2, "longest")
    prep_ids = {tasks[0]["id"], tasks[len(tasks) // 2]["id"]}
    t0 = time.time()

    # ------------------------------------------------------------------ (1) default cfg
    n_walk = n_cycles = n_every = n_short = n_prep = 0
    bad0 = chk.bad
    cache = {}
    for task in tasks:
        g, P, D = task["g"], task["P"], task["D"]
        S = np.concatenate([task["prompt"], g]).astype(np.int64)
        idx = A.build_index(S)
        pt, props = ref_arrays(task, S, idx)
        cache[task["id"]] = (S, idx)
        N = pt["N"]
        el, tl, rt = pt["ext"][ek]
        gated = (tl > 0) & (el >= 3)

        if task["id"] in prep_ids:  # reference arrays vs analyze.prep()
            ps = A.prep(task)
            pel, ptl, prt = ps["ext"][ek]
            chk(np.array_equal(pel, el) and np.array_equal(ptl, tl) and np.array_equal(prt, rt) and np.array_equal(ps["rm"], pt["rm"]), f"{task['id']}: ref arrays != prep()")
            pt_ref = ps
            n_prep += N
        else:
            pt_ref = pt

        # (b) every position: reset(prompt), append g[:i] incrementally
        pl = Pld(lib)
        pl.reset(task["prompt"])
        pl.append(g[:1])
        for i in range(N):
            if i > 0:
                pl.append(g[i : i + 1])
            chk(len(pl) == P + i + 1, f"{task['id']} i={i}: len {len(pl)} != {P + i + 1}")
            ln, l, tok = pl.propose(D[i])
            want_l, want_c = el[i], props[i]
            ok = l == want_l and ((ln == min(4, len(want_c)) and tok == [int(x) for x in want_c[:4]]) if gated[i] else ln == 0)
            chk(ok, f"{task['id']} every i={i}: l={l} ln={ln} tok={tok} vs l={want_l} c={want_c} gated={gated[i]}")
            n_every += 1

        # (a) policy walk (pol_ext, N=1 cycles)
        pl = Pld(lib)
        pl.reset(S[: P + 1])
        i = 0
        while i < N:
            Tr, rr, _ = A.pol_ext(pt_ref, i, V, ek=ek, m=3, Lt=4)
            ln, l, tok = pl.propose(D[i])
            T = 4 + ln
            r = A.prefix_match(list(D[i]) + tok, g, i)
            chk(T == Tr and r == rr and l == el[i], f"{task['id']} walk i={i}: T={T} r={r} l={l} vs pol_ext T={Tr} r={rr} l={el[i]}")
            n_walk += 1
            adv = min(r + 1, N - i)
            pl.append(g[i + 1 : i + 1 + adv])
            i += adv
            n_cycles += 1
            chk(len(pl) == P + i + 1, f"{task['id']} walk i={i}: len {len(pl)}")

        # (c) shorter draft lists (nd = 0,1,2) on every 5th position: reference lookup on K = known + D[i][:nd]
        pl = Pld(lib, min_len=1)
        pl.reset(task["prompt"])
        pl.append(g[:1])
        for i in range(0, N, 5):
            pl2_len = len(pl)
            if pl2_len != P + i + 1:
                pl.reset(S[: P + i + 1])
            for nd in (0, 1, 2):
                l_ref, c_ref = ref_ungated(S, idx, D, P, i, 2, "longest", nd)
                ln, l, tok = pl.propose(D[i][:nd])
                chk(l == l_ref and ln == min(4, len(c_ref)) and tok == [int(x) for x in c_ref[:4]], f"{task['id']} nd={nd} i={i}: {l},{tok} vs {l_ref},{c_ref}")
                n_short += 1
    rows.append(("n=2 longest m=3 Lt=4: policy walk (pol_ext T,r,l)", n_walk, None))
    rows.append(("n=2 longest: every position (l, tail, gate)", n_every, None))
    rows.append(("n=2 longest: nd=0,1,2 drafts", n_short, None))
    rows.append(("analyze.prep() cross-check (2 tasks)", n_prep, None))
    rows[0] = (rows[0][0], n_walk, None)
    mm1 = chk.bad - bad0
    print(f"(1) done {time.time() - t0:.0f}s: walk cycles {n_cycles}", flush=True)

    # ------------------------------------------------------------------ (2) variants
    var_rows = []
    for n in (1, 2, 3, 4):
        for strat in ("longest", "recent"):
            if (n, strat) == (2, "longest"):
                continue
            bad_v = chk.bad
            cnt = 0
            for task in tasks:
                g, P, D = task["g"], task["P"], task["D"]
                S, idx = cache[task["id"]]
                N = len(g) - 1
                pl = Pld(lib, n=n, longest=int(strat == "longest"), min_len=1)
                pl.reset(task["prompt"])
                pl.append(g[:1])
                for i in range(N):
                    if i > 0:
                        pl.append(g[i : i + 1])
                    l_ref, c_ref = ref_ungated(S, idx, D, P, i, n, strat)
                    ln, l, tok = pl.propose(D[i])
                    chk(l == l_ref and ln == min(4, len(c_ref)) and tok == [int(x) for x in c_ref[:4]], f"n={n} {strat} {task['id']} i={i}: {l},{tok} vs {l_ref},{c_ref}")
                    cnt += 1
            var_rows.append((f"n={n} {strat} (m=1)", cnt, chk.bad - bad_v))
    # gating at other m / Lt / cap / back on n=2 longest: small config grid on 4 tasks
    bad_v = chk.bad
    cnt = 0
    for cap, back, m, Lt in [(1, 64, 3, 4), (8, 16, 4, 7), (64, 4, 2, 2), (64, 64, 5, 7), (4, 1, 1, 1)]:
        for task in tasks[::8]:
            g, P, D = task["g"], task["P"], task["D"]
            S, idx = cache[task["id"]]
            N = len(g) - 1
            pl = Pld(lib, cap=cap, back=back, min_len=m, max_tail=Lt)
            pl.reset(task["prompt"])
            pl.append(g[:1])
            for i in range(0, N, 3):
                if len(pl) != P + i + 1:
                    pl.reset(S[: P + i + 1])
                p = P + i
                K = np.concatenate([S[: p + 1], D[i]]).astype(np.int64)
                l_ref, c_ref = A.lookup(K, idx, 2, "longest", len(K) - 1, p + 1, L=Lt, cap=cap, back=back)
                ln, l, tok = pl.propose(D[i])
                want = [int(x) for x in c_ref] if (l_ref >= m and c_ref) else []
                chk(l == l_ref and tok == want, f"cap={cap} back={back} m={m} Lt={Lt} {task['id']} i={i}: {l},{tok} vs {l_ref},{want}")
                cnt += 1
    var_rows.append(("n=2 longest cap/back/m/Lt grid", cnt, chk.bad - bad_v))

    # ------------------------------------------------------------------ (3) benchmark, 8K context
    big = max(tasks, key=lambda t: t["P"] + len(t["g"]))
    pad = np.concatenate([t["prompt"] for t in tasks if t is not big])
    full = np.concatenate([pad, big["prompt"], big["g"]]).astype(np.int32)
    base = len(pad) + big["P"]  # absolute position of g[0]
    nsteps = min(len(big["g"]) - 1, 700)
    pre = 8192 - nsteps
    lead = base + 1 - pre  # drop leading pad so that the context ends at ~8K after nsteps steps
    lead = max(lead, 0)
    ctx = full[lead:]
    start = base - lead  # index of g[0] in ctx
    pl = Pld(lib)
    pl.reset(ctx[: start + 1])
    ts = np.zeros(nsteps)
    dummy = np.zeros(3, np.int32)
    nop = np.zeros(nsteps)
    for i in range(nsteps):
        if i > 0:
            pl.append(ctx[start + i : start + i + 1])
        d = np.ascontiguousarray(big["D"][i], dtype=np.int32)
        t = time.perf_counter_ns()
        pl.propose(d)
        ts[i] = (time.perf_counter_ns() - t) / 1e3
        t = time.perf_counter_ns()
        lib.smk_pld_len(pl.h)
        nop[i] = (time.perf_counter_ns() - t) / 1e3
    ctx_len = len(pl)
    # adversarial: 200 repetitions of one 40-token block separated by unique tokens -> all 64 candidates agree for exactly 40 tokens (no early exit: cap*40 compares)
    rng = np.random.default_rng(0)
    blk = rng.integers(0, 150000, 40).astype(np.int32)
    ctx_adv = np.concatenate([np.concatenate([[200000 + k], blk]) for k in range(200)]).astype(np.int32)
    pa = Pld(lib)
    pa.reset(ctx_adv)
    adv = np.zeros(200)
    dd = blk[:3].copy()
    for i in range(200):
        t = time.perf_counter_ns()
        ln, l, tok = pa.propose(dd)
        adv[i] = (time.perf_counter_ns() - t) / 1e3
    assert l >= 3 and ln > 0, (ln, l)
    # incremental index maintenance
    pm = Pld(lib)
    t = time.perf_counter_ns()
    pm.reset(ctx)
    t_reset = (time.perf_counter_ns() - t) / 1e3
    t = time.perf_counter_ns()
    for i in range(1000):
        pm.append(np.array([i % 5000], np.int32))
    t_app = (time.perf_counter_ns() - t) / 1e3 / 1000

    total_bad = chk.bad
    # ------------------------------------------------------------------ summary
    print()
    print(f"{'config':58s} {'positions':>10s} {'mismatches':>11s}")
    for name, cnt, _ in rows:
        print(f"{name:58s} {cnt:10d} {'(see total)':>11s}")
    for name, cnt, mm in var_rows:
        print(f"{name:58s} {cnt:10d} {mm:11d}")
    print(f"{'walk-through (1) mismatches':58s} {'':>10s} {mm1:11d}")
    print(f"{'TOTAL mismatches':58s} {'':>10s} {total_bad:11d}")
    print()
    print(f"propose, real {ctx_len}-token context ({nsteps} steps, incl. ctypes call overhead {np.mean(nop):.2f} us): mean {ts.mean():.2f} us, p99 {np.percentile(ts, 99):.2f} us, max {ts.max():.2f} us")
    print(f"propose, adversarial 8.2K-token context (64 candidates x 40 agreeing tokens): mean {adv.mean():.2f} us, max {adv.max():.2f} us")
    print(f"reset({len(ctx)} tokens): {t_reset / 1e3:.2f} ms; append(1 token): {t_app:.2f} us (incl. ctypes)")
    print(f"total time {time.time() - t0:.0f}s")
    if total_bad:
        print("FAIL")
        for m in chk.msgs:
            print("  ", m)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
