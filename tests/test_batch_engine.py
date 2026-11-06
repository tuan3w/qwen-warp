"""Batched decode against exact solo references, on the GPU (engine level: BatchRunner without the vLLM server).

Tier: 1

Oracle (no tolerance): a request decoded in a batch emits exactly the tokens of the same request decoded ALONE with the same sequence of decode calls (its Task.calls: batched
cycles with (verified drafts, drafts, cycles), single-sequence MTP / prompt-lookup cycles), from a cold prefill.  This holds because every row-wise kernel of a cycle (embedding,
norms, W4 GEMVs, quantised head + rescoring, MTP layer) gives a row the same bits whatever the other rows of the launch are (checked first, part A), and every stateful kernel
(DeltaNet, attention, accept, sampling) runs per sequence with the same rows as alone.  Seeded sampling and logit processors are covered: the philox counter of a sequence is
its own cycle counter.  A resumed prefill (device snapshot / host prefix store) is compared against the cold one at the same time.
  A. row-count invariance of the row-wise kernels (T = 1..16 rows against the same rows inside T = 16: the one-tile launches of up to 8 rows and the two-tile launches of 9..16)
  B. every batch size B = 1..8 x every draft depth k with B (k + 1) <= 16 (9 of these shapes are cycles of 9..16 rows), greedy / seeded sampling / logit_bias + penalties mixed,
     against solo replays
  C. continuous-batching fuzz (fixed seed): ~48 requests with random arrivals, follow-up turns (prefix-cache hits), max_tokens, aborts, through BatchRunner.step, against solo replays
  D. SMK_DEBUG=1 memory guard: no canary after any device buffer of any slot was overwritten
Solo references are cached under ~/.cache/qwen-warp/batch_solo keyed by (engine + runner sources, prompt, parameters, calls).  SMK_RESIDENT=1: resident weights (tests/_engine.py).
  python tests/test_batch_engine.py [--fuzz 48] [--slots 8]
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("SMK_DEBUG", "1")  # canaries (part D); read once by the engine library
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _engine import ROOT, make_engine  # noqa: E402

import torch  # noqa: E402

import smk  # noqa: E402
import smk_batch as SB  # noqa: E402
from smk_runner import GenParams  # noqa: E402

CACHE = Path(os.path.expanduser("~/.cache/qwen-warp/batch_solo"))
SRC_HASH = hashlib.sha1(b"".join(p.read_bytes() for p in sorted([*(ROOT / "engine").glob("*.py"), *(ROOT / "engine" / "src").glob("*"), *(ROOT / "engine" / "include").glob("*.h"),
                                                                  ROOT / "server" / "smk_runner.py", ROOT / "server" / "prefix_store.py"]))).hexdigest()[:16]
EOS = 248046
MSGS = ["Write a Python function that checks whether a string is a palindrome.", "Describe a rainy afternoon in a small library.", "What is 37 * 41? Show the steps.",
        "List four facts about the Moon.", "Explain what a hash map is in two sentences.", "Write a haiku about autumn leaves.", "Give three tips for learning to juggle.",
        "Convert 98.6 Fahrenheit to Celsius and explain.", "Write a SQL query that counts orders per customer.", "Name five prime numbers greater than 100.",
        "Summarise the plot of a heist movie you invent.", "How does a bicycle stay upright?"]


def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4")))


def chat_ids(tk, msgs):
    x = tk.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    return list(x["input_ids"] if hasattr(x, "keys") else x)


# ---------------------------------------------------------------------------------------------------------------------------------------------- A
def part_a(r):
    e = r.seqs[0].e
    lib = e.lib
    vp, ci = C.c_void_p, C.c_int
    lib.smk_w4r_gemv.argtypes = [vp, vp, vp, ci, ci, ci, C.POINTER(smk.Epi), vp]
    lib.smk_w4r_gemv.restype = ci
    lib.smk_headq_gemv.argtypes = [ci, vp, vp, vp, ci, ci, ci, vp, vp]
    lib.smk_headq_gemv.restype = ci
    lib.smk_headq_select_rescore.argtypes = [vp, vp, vp, ci, ci, ci, ci, vp, vp, vp, vp]
    lib.smk_headq_select_rescore.restype = ci
    lib.smk_headq_ws_bytes.restype = C.c_size_t
    g = torch.Generator(device="cuda").manual_seed(0)
    st = e.st
    L = next(L for L in e.Ls if L.kind == "linear")
    A = next(L for L in e.Ls if L.kind != "linear")
    M = e.M
    shapes = [("dn qkvz", L.qkvz, smk.SMK_EPI_SEGS), ("attn qkv", A.qkvz, smk.SMK_EPI_SEGS), ("o", L.o, smk.SMK_EPI_SEGS), ("gate|up", L.gu, smk.SMK_EPI_SILU),
              ("down", L.down, smk.SMK_EPI_SEGS), ("mtp fc", M.fc, smk.SMK_EPI_SEGS), ("mtp head", M.head_r, smk.SMK_EPI_F32)]
    bad = []
    NR = SB.ROWS  # 16
    for name, w4, kind in shapes:
        N, K = w4.s.shape[0], w4.s.shape[1] * 128
        x = (torch.randn(NR, K, device="cuda", generator=g) * 0.5).bfloat16()
        outs = {}
        for T in range(1, NR + 1):
            if kind == smk.SMK_EPI_F32:
                o = torch.empty(NR, N, device="cuda", dtype=torch.float32)
                ep = smk.epi_simple(kind, o)
            elif kind == smk.SMK_EPI_SILU:
                o = torch.empty(NR, N // 2, device="cuda", dtype=torch.bfloat16)
                ep = smk.epi_simple(kind, o)
            else:
                o = torch.empty(NR, N, device="cuda", dtype=torch.bfloat16)
                ep = smk.epi_segs(N, [o])
            smk.check(lib.smk_w4r_gemv(w4.w.data_ptr(), w4.s.data_ptr(), x.data_ptr(), N, K, T, C.byref(ep), st), name)
            outs[T] = o[:T].clone()
        for T in range(1, NR):
            if not torch.equal(outs[T].view(torch.int16) if outs[T].dtype == torch.bfloat16 else outs[T].view(torch.int32), (outs[NR][:T].view(torch.int16) if outs[NR].dtype == torch.bfloat16 else outs[NR][:T].view(torch.int32))):
                bad.append(f"{name} T={T}")
    # quantised head + exact rescoring (R = 16 and 64): logits, argmax, candidate ids
    G = e.G
    H = 5120
    ws = torch.zeros(int(lib.smk_headq_ws_bytes()), dtype=torch.uint8, device="cuda")
    x = (torch.randn(NR, H, device="cuda", generator=g) * 0.5).bfloat16()
    for R_ in (16, 64):
        res = {}
        for T in range(1, NR + 1):
            lg = torch.empty(NR, 248320, device="cuda", dtype=torch.float32)
            out = torch.zeros(NR, dtype=torch.int32, device="cuda")
            smk.check(lib.smk_headq_gemv(e.hq_kind, G.hq.w.data_ptr(), G.hq.s.data_ptr(), x.data_ptr(), 248320, H, T, lg.data_ptr(), st), "headq")
            smk.check(lib.smk_headq_select_rescore(lg.data_ptr(), G.lm_head.data_ptr(), x.data_ptr(), 248320, H, T, R_, out.data_ptr(), None, ws.data_ptr(), st), "rescore")
            res[T] = (lg[:T].clone(), out[:T].clone())
        for T in range(1, NR):
            if not (torch.equal(res[T][0].view(torch.int32), res[NR][0][:T].view(torch.int32)) and torch.equal(res[T][1], res[NR][1][:T])):
                bad.append(f"head R={R_} T={T}")
    # norms and the in_proj_a|b GEMV
    h = (torch.randn(NR, H, device="cuda", generator=g)).bfloat16()
    y = (torch.randn(NR, H, device="cuda", generator=g)).bfloat16()
    nres = {}
    for T in range(1, NR + 1):
        hh, xn = h.clone(), torch.empty(NR, H, device="cuda", dtype=torch.bfloat16)
        smk.check(lib.smk_rmsnorm(hh.data_ptr(), y.data_ptr(), L.ln1.data_ptr(), xn.data_ptr(), None, T, H, 1e-6, st), "rmsnorm")
        a_, b_ = torch.empty(NR, 48, device="cuda", dtype=torch.bfloat16), torch.empty(NR, 48, device="cuda", dtype=torch.bfloat16)
        smk.check(lib.smk_bf16_gemv_small(L.ab.data_ptr(), xn.data_ptr(), 96, 48, H, T, a_.data_ptr(), b_.data_ptr(), st), "ab")
        nres[T] = [t[:T].clone() for t in (hh, xn, a_, b_)]
    for T in range(1, NR):
        if not all(torch.equal(u.view(torch.int16), v[:T].view(torch.int16)) for u, v in zip(nres[T], nres[NR])):
            bad.append(f"rmsnorm/ab T={T}")
    torch.cuda.synchronize()
    assert not bad, f"row-count invariance broken: {bad}"
    print(f"A: row-count invariance: {len(shapes)} W4 GEMV shapes, the quantised head + rescoring (R 16 / 64), rmsnorm, in_proj_a|b: bit-identical for T = 1..{NR}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------- solo replay
def solo(r, ids, p, calls, images=None):
    """The tokens of one request decoded alone on slot 0 from a COLD prefill with the given decode calls (cached on disk).  images: the prompt's vision.ImageInput list (None: text)."""
    ims = [(im.offset, tuple(im.grid), im.key) for im in images or []]
    key = hashlib.sha1(json.dumps([SRC_HASH, r.ctx_max, r.e.kv_fp8, list(ids), repr(p), calls] + ([ims] if ims else []), default=str).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text())
    q = r.seqs[0]
    assert not q.busy
    r.seq = q
    q.lineage = None
    r._free_snaps([])
    store, r.store = r.store, None
    try:
        b = r._begin(ids, p, images=images)
        assert b.q == 0
        try:
            assert r._prefill(b.ids, 0, b.ends, b.folded, lambda: False, False, b.key)
        finally:
            q.fp.mm = None
        e = q.e
        e._emitted = r._ss()[12]
        e.drafted = r.k
        toks = [e.read_out_tok()]
        for c in calls:
            if c[0] == "batch":
                (new,), _ = r.bd.cycles([e], c[1], c[2], c[3])
            elif c[0] == "pld":
                if not e.pld:
                    e.enable_pld(r.pld)
                if c[2]:
                    e.pld_begin(list(b.ids) + toks)
                new, _ = e.cycles_pld(c[1])
            else:
                new, _ = e.cycles(c[1], c[2])
            toks += new
    finally:
        r.store = store
        q.lineage = None
        r._free_snaps([])
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(toks))
    return toks


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def params(rng, i, max_tokens):
    kind = i % 4
    if kind == 0:
        return GenParams(max_tokens=max_tokens, eos=EOS)
    if kind == 1:
        return GenParams(max_tokens=max_tokens, eos=EOS, temperature=0.8, top_k=20, top_p=0.95, seed=1000 + i)
    if kind == 2:
        return GenParams(max_tokens=max_tokens, eos=EOS, temperature=0.6, seed=2000 + i, min_p=0.05)
    return GenParams(max_tokens=max_tokens, eos=EOS, logit_bias={11: 3.0, 13: -2.0}, presence_penalty=0.5, frequency_penalty=0.3, repetition_penalty=1.1)


# ---------------------------------------------------------------------------------------------------------------------------------------------- B
def part_b(r, prompts):
    """prompts: (ids, images | None) entries; consecutive slots of a batch take consecutive entries (a pool that alternates image and text prompts mixes every batch)."""
    rng = random.Random(1)
    S = len(r.seqs)
    n_cmp = n_img = n_wide = 0
    for B in range(1, S + 1):
        for k in range(0, 4):
            if B * (k + 1) > SB.ROWS:
                continue
            slots = r.seqs[:B]
            rec = []
            for i, q in enumerate(slots):
                r.seq = q
                q.lineage = None
                r._free_snaps([])
                ids, ims = prompts[(B * 3 + k + i) % len(prompts)]
                p = params(rng, B * 7 + k + i, 64)
                b = r._begin(ids, p, images=ims)
                try:
                    assert r._prefill(b.ids, b.q, b.ends, b.folded, lambda: False, False, b.key)
                finally:
                    q.fp.mm = None
                q.e._emitted = r._ss()[12]
                q.e.drafted = r.k
                rec.append(dict(ids=b.ids, ims=ims, p=p, toks=[q.e.read_out_tok()], calls=[]))
            es = [q.e for q in slots]
            for call in [(min(k, 3), k, 2), (k, k, 3), (k, k, 2)]:
                news, _ = r.bd.cycles(es, *call)
                for x, new in zip(rec, news):
                    x["toks"] += new
                    x["calls"].append(("batch",) + call)
            for x in rec:
                ref = solo(r, x["ids"], x["p"], x["calls"], x["ims"])
                assert x["toks"] == ref[: len(x["toks"])] and len(ref) >= len(x["toks"]), f"B={B} k={k}: batched != solo (first difference at {first_diff(x['toks'], ref)}, {x['p']}, images {bool(x['ims'])})"
                n_cmp += 1
                n_img += bool(x["ims"])
                n_wide += B * (k + 1) > 8
    print(f"B: {n_cmp} sequences ({n_img} with images, {n_wide} in cycles of 9..{SB.ROWS} rows) in batches B = 1..{S} x every depth: identical to their solo replays", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------- C
def part_c(r, tk, prompts, long_prompts, n_req, seed=7, burst=False):
    """prompts / long_prompts: (ids, images | None) entries.  burst: 2..3 arrivals per step and longer answers, so that more requests run than DECODE_CAP decodes per group (the
    default schedule's round robin).  Returns the request records (task, toks, aborted, ...)."""
    rng = random.Random(seed)
    reqs, waiting, convs = [], deque(), []
    SB.PREFILL_STALL_ROWS = 64  # interleave the (short) prefills with decode groups often

    def add():
        i = len(reqs)
        if convs and rng.random() < 0.4:
            ids, ims, out = rng.choice(convs)
            ids = ids + out + tk.encode("<|im_end|>\n<|im_start|>user\nAnd then? Answer briefly.<|im_end|>\n<|im_start|>assistant\n")
        else:
            ids, ims = (long_prompts if rng.random() < 0.5 else prompts)[rng.randrange(len(prompts))]
        rec = dict(ids=ids, toks=[], fin=None, aborted=False, abort_at=len(reqs) + rng.randint(1, 6) if rng.random() < 0.12 else None, done=0)
        p = params(rng, i, rng.randint(24, 96) if burst else rng.randint(2, 48))

        def emit(t, toks, fin, sr):
            assert not rec["aborted"]
            rec["toks"] += toks
            rec["fin"] = fin

        def done(t):
            rec["done"] += 1

        rec["task"] = SB.Task(i, ids, p, emit, done, cancelled=lambda: rec["aborted"], images=ims)
        reqs.append(rec)
        waiting.append(rec["task"])

    it, t0, maxB = 0, time.time(), 0
    while len(reqs) < n_req or waiting or r.tasks:
        if len(reqs) < n_req and (burst or rng.random() < 0.45):
            for _ in range(rng.randint(2, 3) if burst else rng.randint(1, 3)):
                if len(reqs) < n_req:
                    add()
        for rec in reqs:
            if rec["abort_at"] is not None and it >= rec["abort_at"] and rec["fin"] is None:
                rec["aborted"] = True
        r.step(waiting, lambda t, ex: (_ for _ in ()).throw(ex))
        maxB = max(maxB, len(r.tasks))
        for rec in reqs:
            if rec["fin"] is not None and not rec.get("conv") and not rec["aborted"] and len(rec["ids"]) < 1500:
                rec["conv"] = True
                convs.append((rec["ids"], rec["task"].images, rec["toks"]))
        it += 1
        assert it < 5000
    dt = time.time() - t0
    srcs = {}
    for rec in reqs:
        t = rec["task"]
        assert rec["done"] == 1
        if t.st is None:
            continue
        srcs[t.st.source] = srcs.get(t.st.source, 0) + 1
        ref = solo(r, t.ids, t.p, t.calls, t.images)
        assert rec["toks"] == ref[: len(rec["toks"])], f"request {t.rid} ({t.st.source}, cached {t.st.cached}, images {bool(t.images)}, calls {t.calls}): batched != solo at {first_diff(rec['toks'], ref)}"
    assert all(not q.busy for q in r.seqs)
    n_img = sum(bool(x["task"].images) for x in reqs)
    print(f"C: {len(reqs)} requests ({n_img} with images, {sum(x['aborted'] for x in reqs)} aborted, up to {maxB} running, resumed from {srcs}) in {it} scheduler steps, {dt:.1f} s: "
          "every stream identical to its solo replay", flush=True)
    assert srcs.get("device", 0) + srcs.get("host", 0) > 0, "no prefix-cache hit: the fuzz does not cover resumes"
    assert not burst or maxB > SB.DECODE_CAP, f"burst: at most {maxB} running, the decode cap ({SB.DECODE_CAP}) never applied"
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=48)
    ap.add_argument("--slots", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    r = make_engine(max_num_seqs=a.slots, ctx_max=4096 + 64, host_tables=True, prefix_cache_gb=0.25, max_snaps=12, min_spill=128)
    print(f"engine ({a.slots} slots) in {time.time() - t0:.1f} s", flush=True)
    tk = tok()
    prompts = [(chat_ids(tk, [{"role": "user", "content": m}]), None) for m in MSGS]
    sysmsg = " ".join(f"Rule {i}: answer the user's question accurately, keep it short, and never invent facts about item {i}." for i in range(30))
    long_prompts = [(chat_ids(tk, [{"role": "system", "content": sysmsg}, {"role": "user", "content": m}]), None) for m in MSGS]  # > w4_max rows: plan cuts and snapshots
    part_a(r)
    part_b(r, prompts)
    part_c(r, tk, prompts, long_prompts, a.fuzz)
    part_c(r, tk, prompts, long_prompts, 24, seed=11, burst=True)
    bad = [int(r.e.lib.smk_engine_canary_check(q.e.h)) for q in r.seqs]
    assert all(b == 0 for b in bad), f"D: canaries overwritten per slot {bad} (-1 = SMK_DEBUG off)"
    print(f"D: canaries intact after every device buffer of the {len(r.seqs)} slots and the batch buffers", flush=True)
    print(f"PASS in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
