"""swift-mk campaign metric: build the tree, then measure it.  Writes {"score": float, ...} to <out> (+ <out stem>.tf.npz for `dev`).

The engine is run AS SHIPPED by the server at K11: SpecEngine(tmax=8) (9-slot DeltaNet ring, verify T <= 8) + enable_pld() (MTP chain k=3 + prompt-lookup tail <= 4, host proposer),
`cycles_pld` with 2 cycles per host sync (server group), exact greedy / exact speculative sampling; k < 3 and plain decode run the plain graphs on the same 9-slot-ring engine.

  metric.py dev   <tree> <out> [--only code,math] [--reps N]   score = geometric mean over 6 prompts (code / prose / math / long8k + 2 copy-heavy E6 tasks `edit`, `table`) of decode tok/s,
                                                               greedy k=3 + prompt-lookup tail (engine-direct, prefill excluded, median of 2 reps (`try`: 1)), adjusted for acceptance luck
                                                               (see hlib.ref_tpc); + the correctness data `check.py` judges (T=8 / T=4 / T=1 teacher forcing, forced partial acceptance up to
                                                               8 rows, sampler / trials / PLD-trials, and the tmax=4 library-default engine in a subprocess)
  metric.py held  <tree> <out>                                 HIDDEN workload: sampled (T=1 top_p .95 top_k 20, seeded) k=2 and k=3+tail on prompts the dev set never uses, PLD-friendly and
                                                               PLD-hostile tasks (k=3+tail), + plain k=0
  metric.py plain <tree> <out>                                 guard: plain greedy decode (k=0) tok/s, short + 8K context
  metric.py tmax4 <tree> <out> / mkrefs <tree> <out>           internal: the tmax=4 correctness subprocess / the pristine teacher-forcing references (base only)

$CAMPAIGN_MODE is `try` (agents' quick loop: fewer reps) or `gate`.  The tree is built with `nice -n 10 make BUILD=build` first (build time is not part of the score).
The GPU must be exclusive (the run waits until no other compute process is on it, then fails).
"""
import argparse
import os
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import hlib  # noqa: E402
from hlib import COPY, LONG, LONGS, ORACLE_NAMES, PRIV, ROOT, TF_MODES, TOPK_SAVED, V, jdump, jload, processed_dist, tf_path, tv  # noqa: E402

N_DEV = 384  # new tokens per timed run (E0)
N_HELD = 256
N_PLAIN = 256
GROUP = 8  # plain k-draft cycles per host sync
GROUP_PLD = 2  # hybrid cycles per host sync (the server's default group)
TMAX = 8  # the shipped K11 engine: 9-slot ring, verify T <= 8
TOK_KEEP = 128  # greedy tokens kept per prompt for `compare`
SPEC_GENS = [(2, None), (3, None), (3, [1]), (3, [2, 1, 3]), (3, [1, 4, 2, 3, 1])]  # (k, forced acceptance pattern), chain only: greedy == plain greedy incl. rollback
PLD_PATS = [None, [8], [1], [5, 2, 7, 1, 8, 3], [3, 4, 1, 6, 2, 8, 5, 7]]  # k=3 + prompt-lookup tail, forced acceptance of up to 8 rows (rollback of the 9-slot ring, ring wrap)
LONG_GENS = {LONG: [(3, None, False), (3, [2, 1, 3], False), (3, None, True), (3, [5, 2, 7, 1, 8, 3], True)],
             COPY: [(3, None, False), (3, None, True), (3, [8], True), (3, [5, 2, 7, 1, 8, 3], True), (3, [3, 4, 1, 6, 2, 8, 5, 7], True)]}  # (k, force, pld) on the long-context prompts
PROBES = [("code", "t4", 3, 1), ("code", "t4", 20, 3), ("prose", "t4", 5, 0), ("prose", "t4", 17, 2), ("math", "t4", 10, 1), ("code", "t8", 4, 6), ("prose", "t8", 7, 7)]  # (prompt, mode, chunk index, row)
HELD_KS = {n: (3,) for n in ("agentedit", "refactor", "docprose", "recmath")}  # held cells per prompt: the 4 original prompts run k=2 and k=3 (+ tail); the PLD-friendly / PLD-hostile E6 tasks k=3 (+ tail)
LIGHT_LEGS = ("g-dev2.json", "g-ctrlB.json")  # gate legs without correctness data (the package checks g-dev1 against g-ctrlA)
N_DRAW = 20000
N_TRIALS = 300000


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def build(tree):
    t0 = time.time()
    p = subprocess.run(["nice", "-n", "10", "make", "-j8", "BUILD=build", "all", "core"], cwd=tree, capture_output=True, text=True)
    if p.returncode:
        log("BUILD FAILED\n" + (p.stdout + p.stderr)[-3500:])
        sys.exit(3)
    return time.time() - t0


def load_engine(tree, ctx_max, tmax=TMAX, prefill=True):
    for k in [k for k in os.environ if k.startswith("SMK_") and k != "SMK_BUILD"]:
        if not os.environ.get("SMKH_KEEP_ENV"):  # tuning overrides (SMK_CFG...) never leak into a measurement; SMKH_KEEP_ENV=1 is for the harness' own calibration
            del os.environ[k]
    os.environ["SMK_BUILD"] = "build"
    sys.path.insert(0, str(tree))
    sys.path.append(str(ROOT / "oracle"))  # the oracle's weights.py (pack.py imports it from ENGINE.parent/oracle, absent in a copied tree)
    os.chdir(tree)
    from prefill import FastPrefill
    from spec import SpecEngine

    e = SpecEngine(ctx_max=ctx_max, verbose=False, tmax=tmax)  # tmax=8: the engine as the server ships it with --prompt-lookup (9-slot DeltaNet ring); tmax=4 = the library default
    return e, (FastPrefill(e) if prefill else None)


def set_pld(e, on):
    """the hybrid draft (MTP chain + prompt-lookup tail) needs tmax >= 5 and k = 3; every other run uses the plain k-draft graphs of the same engine."""
    if on:
        e.enable_pld()  # as the server does per request: config the proposer + (no-op when captured) the T = 4..8 graph widths
    elif e.pld:
        e.disable_pld()


def spec_run(e, fp, ids, k, n, force=None, pld=None):
    """one speculative decode of >= n tokens after the prefill's first token (k=3 uses the prompt-lookup tail on a tmax=8 engine unless pld=False).
    Returns (tok/s, tokens incl. the first, tokens/cycle); e.last_pld = proposer statistics of the run (None without the tail)."""
    import torch

    pld = (k == 3 and e.tmax >= 5) if pld is None else pld
    e.reset()
    e.set_force(force)
    first = fp.prefill_spec(ids, k)
    set_pld(e, pld)
    if pld:
        e.pld_begin(list(ids) + [first])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks = []
    while True:
        new, ss = e.cycles_pld(GROUP_PLD) if pld else e.cycles(k, GROUP)
        toks += new
        if ss[12] >= n:
            break
    dt = time.perf_counter() - t0
    assert len(toks) == ss[12], "emitted-token count mismatch"
    e.last_pld = e.pld_stats() if pld else None
    return len(toks) / dt, [first] + toks, ss[12] / ss[13]


def plain_run(e, fp, ids, n, sampled):
    import torch

    e.reset()
    fp.prefill(ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e.plain_sample(n) if sampled else e.decode(n)
    return n / (time.perf_counter() - t0)


def warmup(e, fp, ids, copy_ids=None):
    """graph captures for every k + clocks up (the first measurement after idle runs ~7% slow); `copy_ids`: a copy-heavy prompt so tail cycles (T = 5..8) run too."""
    import torch

    t0 = time.time()
    for k in (1, 2, 3):
        spec_run(e, fp, ids[:200], k, 24)
    if copy_ids is not None:
        spec_run(e, fp, copy_ids[:1500], 3, 96)
    while time.time() - t0 < 4.0:
        plain_run(e, fp, ids[:200], 64, False)
        spec_run(e, fp, ids[:200], 3, 128)
        if copy_ids is not None:
            spec_run(e, fp, copy_ids[:1500], 3, 128)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------------------------- correctness data (judged by check.py)
def tf_short(e, refs, modes, key=""):
    """teacher-forced pass over the pristine continuation of the 3 oracle prompts, one chunk width per mode (t8 = the widest verify step of the hybrid draft, t4 = chain verify, t1 = plain
    decode): top-k logits per position; in the modes of PROBES also N_DRAW draws of the engine's own device sampler (T=1 top_k 20 top_p .95, seeded) at fixed (chunk, row) positions."""
    arrs, probes = {}, []
    for name in ORACLE_NAMES:
        p, cont = refs[name]["prompt"], refs[name]["cont"]
        seq = [p[-1]] + cont[:-1]  # row i predicts cont[i]
        for mode, T in modes:
            e.reset()
            n, i = len(p) - 1, 0
            while i < n:
                t = min(4, n - i)
                e._step(p[i : i + t], 0)
                i += t
            ids = np.zeros((len(seq), TOPK_SAVED), np.int32)
            vals = np.zeros((len(seq), TOPK_SAVED), np.float32)
            for j in range(0, len(seq), T):
                ch = seq[j : j + T]
                e._step(ch, 1)
                lg = e.logits(len(ch))
                v, ix = lg.topk(TOPK_SAVED, dim=1)
                ids[j : j + len(ch)], vals[j : j + len(ch)] = ix.numpy(), v.numpy()
                if not key:
                    for pi, (pn, pm, pc, pr) in enumerate(PROBES):
                        if pn == name and pm == mode and pc == j // T:
                            row = lg[pr].numpy()
                            e.set_sampling(temperature=1.0, top_k=20, top_p=0.95, draft_sample=1, seed=4242 + pi)
                            sup, pp = processed_dist(row, **hlib.SAMP)
                            cnt = np.bincount(e.test_draw(N_DRAW, 0, pr).numpy(), minlength=V)
                            arrs[f"probe{pi}_ids"], arrs[f"probe{pi}_p"] = sup.astype(np.int32), pp
                            arrs[f"probe{pi}_cnt"] = cnt[sup].astype(np.int64)
                            arrs[f"probe{pi}_out"] = np.int64(N_DRAW - cnt[sup].sum())
                            probes.append(pi)
            arrs[f"{key}{name}_{mode}_id"], arrs[f"{key}{name}_{mode}_val"] = ids, vals
    e.set_sampling(temperature=0.0)
    if not key:
        assert sorted(probes) == list(range(len(PROBES))), "probe positions not reached"
    return arrs


def tf_pass(e, fp, refs):
    arrs = tf_short(e, refs, TF_MODES)
    for name in LONGS:  # long context: fast prefill of the prompt, then the pristine continuation teacher-forced (8K prompt: attention / DeltaNet at long context; edit prompt: tail rows)
        p, cont = refs[name]["prompt"], refs[name]["cont"]
        for mode, T in TF_MODES:
            e.reset()
            fp.prefill(p)
            ids = np.zeros((len(cont), TOPK_SAVED), np.int32)
            vals = np.zeros((len(cont), TOPK_SAVED), np.float32)
            v, ix = e.logits(1).topk(TOPK_SAVED, dim=1)
            ids[0], vals[0] = ix[0].numpy(), v[0].numpy()
            for j in range(0, len(cont) - 1, T):
                ch = cont[j : min(j + T, len(cont) - 1)]
                e._step(ch, 1)
                v, ix = e.logits(len(ch)).topk(TOPK_SAVED, dim=1)
                ids[1 + j : 1 + j + len(ch)], vals[1 + j : 1 + j + len(ch)] = ix.numpy(), v.numpy()
            arrs[f"{name}_{mode}_id"], arrs[f"{name}_{mode}_val"] = ids, vals
    return arrs


def gen_key(name, k, pat, pld):
    return f"{name}/k{k}{'p' if pld else ''}/{'-'.join(map(str, pat)) if pat else 'nat'}"


def spec_gens(e, fp, refs, n=64):
    """greedy speculative generations of the oracle prompts and the long prompts, natural + forced partial acceptance (chain only; chain + prompt-lookup tail with patterns of up to 8
    rows): judged against the pristine continuation."""
    out = {}
    for name in ORACLE_NAMES:
        for k, pat in SPEC_GENS:
            set_pld(e, False)
            out[gen_key(name, k, pat, False)] = e.generate(refs[name]["prompt"], n, k=k, force=pat)[0]
        for pat in PLD_PATS:
            set_pld(e, True)
            out[gen_key(name, 3, pat, True)] = e.generate(refs[name]["prompt"], n, k=3, force=pat)[0]
    for name, gens in LONG_GENS.items():
        for k, pat, pld in gens:
            out[gen_key(name, k, pat, pld)] = spec_run(e, fp, refs[name]["prompt"], k, LONGS[name], force=pat, pld=pld)[1][: LONGS[name]]
    e.set_force(None)
    return out


def gens_short(e, refs, n=64):
    """chain-only generations of the oracle prompts on the tmax=4 library engine (5-slot ring)."""
    out = {}
    for name in ORACLE_NAMES:
        for k, pat in SPEC_GENS:
            out[gen_key(name, k, pat, False)] = e.generate(refs[name]["prompt"], n, k=k, force=pat)[0]
    e.set_force(None)
    return out


def trials_check(e, k=3):
    """the graph's own speculative-sampling accept functions (draft pick, accept / residual / bonus) on fixed synthetic rows, N_TRIALS independent cycles vs the analytic law."""
    import ctypes as C

    import torch
    from spec import make_sampling, smk

    g = torch.Generator().manual_seed(100)
    S = torch.randperm(V, generator=g)[:40]
    tl = []
    for _ in range(k + 1):
        l = torch.randn(V, generator=g) * 1.5
        l[S] += 8.0 + 2.0 * torch.randn(40, generator=g)
        tl.append(l)
    tl = torch.stack(tl)
    keep = torch.ones(V, dtype=torch.bool)
    keep[S[:4]] = False
    keep[S[4:]] = False
    cand = torch.nonzero(keep).flatten()
    ids = torch.sort(torch.cat([S[4:], cand[torch.randperm(len(cand), generator=g)[: 65536 - 36]]])).values.int()
    dl = tl[:k][:, ids.long()] + 0.3 * torch.randn(k, 65536, generator=g)
    cfg = make_sampling(seed=777, draft_sample=1, **hlib.SAMP)
    y, a = (C.c_int * (N_TRIALS * 4))(), (C.c_int * N_TRIALS)()
    smk.check(e.lib.smk_test_trials(C.c_void_p(tl.contiguous().data_ptr()), V, C.c_void_p(dl.contiguous().data_ptr()), 65536, C.c_void_p(ids.data_ptr()), C.byref(cfg), k, N_TRIALS, 0, y, a), "trials")
    Y, A = np.frombuffer(y, dtype=np.int32).reshape(N_TRIALS, 4), np.frombuffer(a, dtype=np.int32)
    p, q = [], []
    for t in range(k + 1):
        s, pp = processed_dist(tl[t].numpy(), **hlib.SAMP)
        full = np.zeros(V)
        full[s] = pp
        p.append(full)
    for t in range(k):
        s, pp = processed_dist(dl[t].numpy(), **hlib.SAMP)
        full = np.zeros(V)
        full[ids.long().numpy()[s]] = pp
        q.append(full)
    beta = [float(np.minimum(p[t], q[t]).sum()) for t in range(k)]
    pa = [1 - beta[0]] + [float(np.prod(beta[:m]) * (1 - beta[m])) for m in range(1, k)] + [float(np.prod(beta))]
    ca = np.bincount(A, minlength=k + 2)[1 : k + 2] / N_TRIALS
    sup0 = np.nonzero(p[0] > 0)[0]
    c0 = np.bincount(Y[:, 0], minlength=V)
    out0 = int(N_TRIALS - c0[sup0].sum())
    return dict(tv_y0=tv(np.append(c0[sup0] / N_TRIALS, out0 / N_TRIALS), np.append(p[0][sup0], 0.0)), tv_len=tv(ca, pa), outside=out0, mean_len=float(A.mean()), mean_len_expected=float(sum((i + 1) * x for i, x in enumerate(pa))))


def trials_pld_check(e, nchain=3, ntail=4):
    """K11: the graph's accept functions with 3 sampled chain drafts + 4 prompt-lookup tail tokens (delta proposals q = 1 at the tail token): N_TRIALS cycles on fixed independent rows.  Chain position t accepts with
    beta_t = sum min(p_t, q_t), tail position t with pi_t = p_t(d_t); checks the emitted-length histogram, the first token, that a tail rejection emits p_t restricted to x != d_t (never d_t),
    that an accepted tail token is d_t, and the bonus token ~ p_k."""
    import ctypes as C

    import torch
    from spec import make_sampling, smk

    k = nchain + ntail
    g = torch.Generator().manual_seed(101)
    S = torch.randperm(V, generator=g)[:40]
    tl = []
    for _ in range(k + 1):
        l = torch.randn(V, generator=g) * 1.5
        l[S] += 8.0 + 2.0 * torch.randn(40, generator=g)
        tl.append(l)
    tl = torch.stack(tl)
    keep = torch.ones(V, dtype=torch.bool)
    keep[S] = False
    cand = torch.nonzero(keep).flatten()
    ids = torch.sort(torch.cat([S[4:], cand[torch.randperm(len(cand), generator=g)[: 65536 - 36]]])).values.int()
    dl = (tl[:nchain][:, ids.long()] + 0.3 * torch.randn(nchain, 65536, generator=g)).contiguous()
    cfg = make_sampling(seed=778, draft_sample=1, **hlib.SAMP)
    p, q = [], []
    for t in range(k + 1):
        s, pp = processed_dist(tl[t].numpy(), **hlib.SAMP)
        full = np.zeros(V)
        full[s] = pp
        p.append(full)
    for t in range(nchain):
        s, pp = processed_dist(dl[t].numpy(), **hlib.SAMP)
        full = np.zeros(V)
        full[ids.long().numpy()[s]] = pp
        q.append(full)
    ranks = [0, 1, 0, 2]  # tail token of tail position j = the rank-th most probable token of that row (pi_t from ~0.6 down to ~0.2)
    tail = [int(np.argsort(-p[nchain + j])[ranks[j]]) for j in range(ntail)]
    y, a = (C.c_int * (N_TRIALS * 8))(), (C.c_int * N_TRIALS)()
    tail_arr = (C.c_int * ntail)(*tail)
    smk.check(e.lib.smk_test_trials_pld(C.c_void_p(tl.contiguous().data_ptr()), V, C.c_void_p(dl.data_ptr()), 65536, C.c_void_p(ids.data_ptr()), C.byref(cfg), nchain, tail_arr, ntail, N_TRIALS, 0, y, a), "trials_pld")
    Y, A = np.frombuffer(y, dtype=np.int32).reshape(N_TRIALS, 8), np.frombuffer(a, dtype=np.int32)
    acc = [float(np.minimum(p[t], q[t]).sum()) for t in range(nchain)] + [float(p[nchain + j][tail[j]]) for j in range(ntail)]
    reach = [1.0]
    for t in range(k):
        reach.append(reach[-1] * acc[t])
    pa = [reach[t] * (1 - acc[t]) for t in range(k)] + [reach[k]]  # P(emitted = 1 .. k+1)
    ca = np.bincount(A, minlength=k + 2)[1 : k + 2] / N_TRIALS
    sup0 = np.nonzero(p[0] > 0)[0]
    c0 = np.bincount(Y[:, 0], minlength=V)
    out0 = int(N_TRIALS - c0[sup0].sum())
    stray, viol, rej_tv, bon_tv = 0, 0, 0.0, 0.0
    for j in range(ntail):
        t, d = nchain + j, tail[j]
        stray += int((Y[A >= t + 2, t] != d).sum())  # accepted tail token must be d_t
        yr = Y[A == t + 1, t]  # rejected at tail position t: y_t ~ p_t on x != d_t
        viol += int((yr == d).sum())
        if len(yr) >= 3000:
            ex = p[t].copy()
            ex[d] = 0.0
            rej_tv = max(rej_tv, tv(np.bincount(yr, minlength=V) / len(yr), ex / ex.sum()))
    yb = Y[A == k + 1, k]
    if len(yb) >= 3000:
        bon_tv = tv(np.bincount(yb, minlength=V) / len(yb), p[k])
    return dict(tv_y0=tv(np.append(c0[sup0] / N_TRIALS, out0 / N_TRIALS), np.append(p[0][sup0], 0.0)), tv_len=tv(ca, pa), outside=out0, mean_len=float(A.mean()), mean_len_expected=float(sum((i + 1) * x for i, x in enumerate(pa))),
                tail_stray=stray, tail_viol=viol, tail_rej_tv=rej_tv, bonus_tv=bon_tv, n_rej=[int((A == nchain + j + 1).sum()) for j in range(ntail)], n_bonus=len(yb))


def run_suites(tree):
    """the engine's own unit test of the GEMV / norm / argmax ops (tests/test_ops.py, ~2 s, small GPU footprint) on the candidate build; run before the engine is loaded.
    The test file is only trusted if it is byte-identical to the pristine one (an agent must not weaken its own test); the speculative-greedy / rollback part of tests/test_mtp.py is
    covered natively by `spec_gens`."""
    now, ref = sha_tests(tree), sha_tests(ROOT / "base")
    edited = [n for n in now if now[n] != ref[n]]
    res = dict(edited=edited)
    if "test_ops.py" in edited:
        res["test_ops"] = "skipped: edited"
        return res
    p = subprocess.run([hlib.PY, "tests/test_ops.py"], cwd=tree, capture_output=True, text=True, env=dict(os.environ, SMK_BUILD="build", PYTHONPATH=str(ROOT / "oracle")))
    last = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    res["test_ops"] = dict(ok=p.returncode == 0 and "passed" in last, last=last, fails=[l[:200] for l in p.stdout.splitlines() if l.startswith("FAIL")][:5], stderr=p.stderr[-300:] if p.returncode else "")
    return res


def sha_tests(tree):
    """hashes of the tests / harness-relevant files so `check` can tell whether the candidate edited the engine's own correctness tests."""
    import hashlib

    return {n: hashlib.sha256((Path(tree) / "tests" / n).read_bytes()).hexdigest() for n in ("test_ops.py", "test_mtp.py", "test_e2e.py")}


# ---------------------------------------------------------------------------------------------------------------- workloads
def run_profile(tree, out, a):
    """`try --profile [--only code]`: nsys kernel table of 32 shipped (k=3 + prompt-lookup tail) greedy cycles (ms per cycle), instead of the benchmark."""
    import csv
    import io
    import shutil
    import tempfile

    name = (a.only or "code").split(",")[0]
    tmp = Path(tempfile.mkdtemp(prefix="smkprof-", dir=ROOT))
    try:
        p = subprocess.run(["nsys", "profile", "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true", "-o", str(tmp / "p"), hlib.PY,
                            str(Path(__file__).resolve().parent / "profile_run.py"), str(tree), name], capture_output=True, text=True, env=dict(os.environ, SMKH_PROFILE="1"))
        line = next((l for l in p.stdout.splitlines() if l.startswith("PROFILED_CYCLES")), None)
        if p.returncode or not line:
            log((p.stdout + p.stderr)[-3000:])
            sys.exit(1)
        s = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--output", "-", str(tmp / "p.nsys-rep")], capture_output=True, text=True).stdout
        rows = list(csv.DictReader(io.StringIO(s[s.index("Time (%)"):])))
        lines = [f"nsys, 32 greedy k=3 + prompt-lookup tail cycles on `{name}` ({line.split()[2]} ms/cycle wall, {line.split()[4]} tokens/cycle); kernel: ms/cycle, calls/cycle, avg us"]
        for r in rows[:22]:
            lines.append(f"  {float(r['Total Time (ns)']) / 32e6:7.3f} ms {int(r['Instances']) / 32:6.1f} x {float(r['Avg (ns)']) / 1e3:8.1f} us  {r['Name'][:90]}")
        e_ref = jload(PRIV / "ref_tpc.json")["dev"].get(name, 1.0) if (PRIV / "ref_tpc.json").exists() else 1.0
        jdump(dict(kind="profile", score=e_ref * 1e3 / float(line.split()[2]), summary="\n".join(lines)), out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def child(kind, tree, out):
    """run one of our own subprocess kinds (tmax4 / mkrefs) in a fresh process (an engine holds ~19 GB: two engines never coexist)."""
    p = subprocess.run([hlib.PY, str(Path(__file__).resolve()), kind, str(tree), str(out)], env=dict(os.environ, SMKH_CHILD="1"), capture_output=True, text=True)
    sys.stderr.write(p.stderr[-1500:])
    if p.returncode:
        raise SystemExit(f"{kind} subprocess failed (rc {p.returncode}):\n{(p.stdout + p.stderr)[-2500:]}")


def run_mkrefs(tree, out, a, gate):
    """pristine control only: greedy plain (T=1) continuation of the oracle prompts and of the long-context prompts = the teacher-forcing reference (private/tf_tokens.json)."""
    P = jload(PRIV / "dev_prompts.json")
    e, fp = load_engine(tree, max(len(P[n]) for n in LONGS) + 700)
    refs = {}
    for n in ORACLE_NAMES:
        pr = jload(PRIV / "oracle_prompts.json")[n]
        e.reset()
        first = e.prefill(pr)
        refs[n] = dict(prompt=pr, cont=[first] + e.decode(95))
    for n, ln in LONGS.items():
        e.reset()
        first = fp.prefill(P[n])
        refs[n] = dict(prompt=P[n], cont=[first] + e.decode(ln - 1))
    jdump(refs, PRIV / "tf_tokens.json")
    jdump(dict(kind="mkrefs", score=0.0), out)


def run_tmax4(tree, out, a, gate):
    """the library default (Engine(tmax=4), 5-slot DeltaNet ring: the server's engine with --prompt-lookup 0 / above the max-model-len the 9-slot ring allows): teacher-forced T=4 / T=1 logits and
    chain-only greedy generations with forced partial acceptance of the oracle prompts."""
    refs = jload(PRIV / "tf_tokens.json")
    e, _ = load_engine(tree, max(len(refs[n]["prompt"]) for n in ORACLE_NAMES) + 700, tmax=4, prefill=False)
    arrs = tf_short(e, refs, (("t4", 4), ("t1", 1)), key="m4_")
    np.savez(Path(out).with_suffix(".npz"), **arrs)
    jdump(dict(kind="tmax4", score=0.0, tmax=e.tmax, gens=gens_short(e, refs)), out)


def run_dev(tree, out, a, gate):
    if a.profile and not gate:
        return run_profile(tree, out, a)
    refs_p = PRIV / "tf_tokens.json"
    P = jload(PRIV / "dev_prompts.json")
    names = a.only.split(",") if a.only else list(P)
    reps = a.reps or (2 if gate else 1)
    mx = max(len(v) for v in P.values())
    # correctness data tiers: gate legs g-dev2 / g-ctrlB carry none (the package checks g-dev1 against g-ctrlA only; the other two legs only repeat the timing), `try` skips the tmax=4
    # subprocess (the gate has it), every other run (gate g-dev1 / g-ctrlA, baseline, calibration, noise) records everything.
    light = Path(out).name in LIGHT_LEGS
    tier = "light" if light else ("full" if gate else "try")
    suites = {} if light else run_suites(tree)
    m4, m4arr = {}, {}
    if not light:
        if not refs_p.exists() or any(n not in jload(refs_p) for n in hlib.TF_NAMES):
            if Path(tree).resolve() != (ROOT / "base").resolve():
                raise SystemExit("private/tf_tokens.json lacks the teacher-forcing references: run `baseline` first")
            mk = Path(out).with_name(Path(out).stem + ".mkrefs.json")
            child("mkrefs", tree, mk)
            mk.unlink()
        if tier == "full":
            m4p = Path(out).with_name(Path(out).stem + ".m4.json")
            child("tmax4", tree, m4p)  # the library-default engine, in its own process (before the shipped-config engine is loaded)
            m4 = jload(m4p)
            m4arr = dict(np.load(m4p.with_suffix(".npz")))
            m4p.with_suffix(".npz").unlink()
            m4p.unlink()
    e, fp = load_engine(tree, mx + N_DEV + 700)
    res = dict(kind="dev", mode="gate" if gate else "try", tier=tier, names=names, reps=reps, suites=suites, spec_gens_m4=m4.get("gens", {}))
    warmup(e, fp, P["code"], P["edit"])
    if not light:
        refs = jload(refs_p)
        t0 = time.time()
        parts = {}

        def timed(name, f, *args):
            t1 = time.time()
            r = f(*args)
            parts[name] = round(time.time() - t1, 1)
            return r

        arrs = timed("tf_pass", tf_pass, e, fp, refs)
        arrs.update(m4arr)
        np.savez(tf_path(out), **arrs)
        res["spec_gens"] = timed("spec_gens", spec_gens, e, fp, refs)
        res["trials"] = timed("trials", trials_check, e)
        res["trials_pld"] = timed("trials_pld", trials_pld_check, e)
        res["check_s"] = time.time() - t0
        res["check_parts_s"] = parts
    cells, tokens, clocks = {}, {}, [hlib.gpu_clocks()]
    for n in names:
        rates, tpcs = [], []
        for r in range(reps):
            rate, toks, tpc = spec_run(e, fp, P[n], 3, N_DEV)
            rates.append(rate)
            tpcs.append(tpc)
            if r == 0:
                tokens[n] = toks[:TOK_KEEP]
        ps = e.last_pld
        tot = sum(ps["cycles_by_k"].values()) or 1
        cells[n] = dict(rate=st.median(rates), reps=rates, tokens_per_cycle=st.mean(tpcs), cyc_rate=st.median(r / t for r, t in zip(rates, tpcs)), prompt_tokens=len(P[n]), tail_cycle_share=ps["tail_cycles"] / tot)
        clocks.append(hlib.gpu_clocks())
        log(f"{n}: {cells[n]['rate']:.1f} tok/s {['%.1f' % x for x in rates]} tpc {cells[n]['tokens_per_cycle']:.3f} tail cycles {cells[n]['tail_cycle_share']:.2f} clocks {clocks[-1]}")
    ref = hlib.ref_tpc("dev", tree, {n: c["tokens_per_cycle"] for n, c in cells.items()}, create=set(names) == set(P))
    band = hlib.limits()["accept_band"]
    for n, c in cells.items():
        c["accept_ratio"] = c["tokens_per_cycle"] / ref[n]
        c["adj_rate"] = c["cyc_rate"] * ref[n] * hlib.band_adjust(c["accept_ratio"], band[n])
    res.update(score=hlib.geomean([c["adj_rate"] for c in cells.values()]), raw_tok_s=hlib.geomean([c["rate"] for c in cells.values()]), cells=cells, tokens=tokens, clocks=clocks)
    res["summary"] = ("  ".join(f"{n} {c['rate']:.1f}" for n, c in cells.items()) + f"  (tok/s, k=3 + prompt-lookup tail greedy, {reps} rep{'s' * (reps > 1)})\n  score = geomean of cycles/s x pristine tokens/cycle = {res['score']:.2f} (raw geomean tok/s {res['raw_tok_s']:.2f}; tokens/cycle vs pristine "
                      + " ".join(f"{n} {c['accept_ratio']:.3f}" for n, c in cells.items()) + f"; only changes beyond the {band} noise band count)  sm clock {clocks[0][0]:.0f}-{clocks[-1][0]:.0f} MHz, {clocks[-1][2]:.0f} W")
    jdump(res, out)


def run_held(tree, out, a, gate):
    P = jload(PRIV / "held_prompts.json")
    mx = max(len(v) for v in P.values())
    e, fp = load_engine(tree, mx + N_HELD + 700)
    reps = 2
    short = next(iter(P.values()))
    e.set_sampling(draft_sample=1, seed=1, **hlib.SAMP)
    warmup(e, fp, short, P["agentedit"])  # greedy warm-up of the cycle graphs; sampling graphs are captured by the first sampled runs below
    for k in (2, 3):
        spec_run(e, fp, short, k, 32)
    spec_run(e, fp, P["agentedit"], 3, 96)
    plain_run(e, fp, short, 32, True)
    cells, clocks = {}, [hlib.gpu_clocks()]
    for n, ids in P.items():
        for k in HELD_KS.get(n, (2, 3)):  # k=2: the plain chain graphs on the 9-slot-ring engine; k=3: chain + prompt-lookup tail
            rates, tpcs = [], []
            for r in range(reps):
                e.set_sampling(draft_sample=1, seed=101 + r, **hlib.SAMP)
                rate, _, tpc = spec_run(e, fp, ids, k, N_HELD)
                rates.append(rate)
                tpcs.append(tpc)
            cells[f"{n}/k{k}"] = dict(rate=st.median(rates), reps=rates, tokens_per_cycle=st.mean(tpcs), cyc_rate=st.median(r / t for r, t in zip(rates, tpcs)), prompt_tokens=len(ids))
            log(f"{n}/k{k}: {cells[f'{n}/k{k}']['rate']:.1f} tok/s tpc {st.mean(tpcs):.2f}")
    first = next(iter(P))
    rates = []
    for r in range(reps):
        e.set_sampling(draft_sample=1, seed=101 + r, **hlib.SAMP)
        rates.append(plain_run(e, fp, P[first], N_HELD, True))
    cells[f"{first}/k0"] = dict(rate=st.median(rates), reps=rates, tokens_per_cycle=1.0, cyc_rate=st.median(rates), prompt_tokens=len(P[first]))
    clocks.append(hlib.gpu_clocks())
    e.set_sampling(temperature=0.0)  # greedy continuations of the held prompts for `compare`
    tokens = {n: spec_run(e, fp, ids, 3, 64)[1][:64] for n, ids in P.items()}
    # sampled trajectories change with any rounding-level change (acceptance varies +-10% per cell), the cost of a cycle does not: score = cycle rate x the pristine tokens/cycle per cell,
    # times a floor factor that only charges a collapse of the mean tokens/cycle (> held_accept_floor below pristine)
    ref = hlib.ref_tpc("held", tree, {c: v["tokens_per_cycle"] for c, v in cells.items()})
    spec = [c for c in cells if not c.endswith("/k0")]
    mE = sum(cells[c]["tokens_per_cycle"] for c in spec) / sum(ref[c] for c in spec)
    f = min(1.0, mE / (1 - hlib.limits()["held_accept_floor"]))
    for c, v in cells.items():
        v["adj_rate"] = v["cyc_rate"] * ref[c]
    res = dict(kind="held", mode="gate" if gate else "try", score=hlib.geomean([v["adj_rate"] for v in cells.values()]) * f, raw_tok_s=hlib.geomean([v["rate"] for v in cells.values()]), accept_ratio=mE, accept_factor=f, cells=cells, tokens=tokens, clocks=clocks,
               plain_ms_per_step=1e3 / cells[f"{first}/k0"]["rate"])
    res["summary"] = "held-out workload (hidden in the gate)"
    jdump(res, out)


def run_plain(tree, out, a, gate):
    P = jload(PRIV / "dev_prompts.json")
    names = [n for n in ("code", "long8k") if n in P]
    mx = max(len(P[n]) for n in names)
    e, fp = load_engine(tree, mx + N_PLAIN + 700)
    warmup(e, fp, P["code"])
    cells = {}
    for n in names:
        rates = [plain_run(e, fp, P[n], N_PLAIN, False) for _ in range(2)]
        cells[n] = dict(rate=st.median(rates), reps=rates, prompt_tokens=len(P[n]))
        log(f"plain {n}: {cells[n]['rate']:.2f} tok/s {['%.2f' % x for x in rates]}")
    res = dict(kind="plain", score=hlib.geomean([c["rate"] for c in cells.values()]), cells=cells, clocks=[hlib.gpu_clocks()])
    res["summary"] = "plain greedy decode " + "  ".join(f"{n} {c['rate']:.2f}" for n, c in cells.items()) + " tok/s"
    jdump(res, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["dev", "held", "plain", "tmax4", "mkrefs"])
    ap.add_argument("tree")
    ap.add_argument("out")
    ap.add_argument("--only", default=None)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    gate = os.environ.get("CAMPAIGN_MODE", "gate") != "try"
    tree = Path(a.tree).resolve()
    a.out = str(Path(a.out).resolve())
    if a.kind != "dev" and not gate:
        a.reps = 0
    try:
        sub_run = bool(os.environ.get("SMKH_CHILD"))
        if not os.environ.get("SMKH_NO_GPU_WAIT") and not sub_run:  # harness self-tests on a fake engine only / our own subprocess (the parent already waited)
            hlib.wait_gpu_idle()
        bs = 0.0 if sub_run else build(tree)
        if not sub_run:
            log(f"build {bs:.0f}s")
        {"dev": run_dev, "held": run_held, "plain": run_plain, "tmax4": run_tmax4, "mkrefs": run_mkrefs}[a.kind](tree, a.out, a, gate)
        r = jload(a.out)
        r["build_s"] = bs
        jdump(r, a.out)
        log(f"{a.kind} score {r['score']:.3f}")
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
