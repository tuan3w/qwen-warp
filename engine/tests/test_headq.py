"""K9 quantised lm_head + exact BF16 rescoring: unit tests on the real head and real final-norm hidden states, then end-to-end token parity.

  ~/llm-lab/venv/bin/python tests/test_headq.py [--heads int8,int4] [--what unit,parity,e2e] [--hidden <hidden.pt>] [--max-pos N] [--out file.md]
  (SMK_BUILD=build/k9 to pick a private build; needs the GPU lease.  The unit tests load only the 2.5 GB lm_head; e2e loads the full engine.)

unit   : (1) approximate logits vs an fp64 matmul with the DEQUANTISED head (T = 1..4), (2) candidate set == exact top-R of the approximate logits (torch.topk, ties -> lower
         index), (3) rescored logits == fp64 dot of the bf16 head rows, argmax == argmax over the exact candidates, (4) pipeline vs the bf16 head (smk_bf16_gemv_f32 + smk_argmax)
         over every position of the E5 hidden-state file (53 docs, real engine final-norm outputs; T=4 blocks and T=1): argmax mismatches, top-1 outside the candidate set,
         exact logits vs the bf16 head's logits, and the candidate coverage of the true top-20 / top-64 for R = 64 (the sampling set).
parity : engine greedy streams, plain (T=1 graph) and speculative (k = 3), quantised head vs the BF16 head in the SAME engine (Engine.set_head), 3 dump prompts + 3 E0 prompts.
e2e    : alias of parity.
"""
import argparse
import ctypes as C
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smk  # noqa: E402
import pack  # noqa: E402
from weights import Checkpoint  # noqa: E402

N, K = 248320, 5120
HIDDEN = os.path.join(str(smk.ENGINE.parent), "experiments", "2026-10-31-lm-head-quantisation-quality", "runs", "hidden.pt")
LOG = []


def log(s=""):
    print(s, flush=True)
    LOG.append(s)


def setup(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_headq_ws_bytes.restype = C.c_size_t
    lib.smk_headq_gemv.argtypes = [ci, vp, vp, vp, ci, ci, ci, vp, vp]
    lib.smk_headq_gemv.restype = ci
    lib.smk_headq_select_rescore.argtypes = [vp, vp, vp, ci, ci, ci, ci, vp, vp, vp, vp]
    lib.smk_headq_select_rescore.restype = ci
    lib.smk_headq_ws_cand.argtypes = [vp]
    lib.smk_headq_ws_cand.restype = vp
    lib.smk_headq_ws_exact.argtypes = [vp]
    lib.smk_headq_ws_exact.restype = vp


class Rig:
    def __init__(self, lib, head, kind):
        self.lib, self.kind = lib, kind
        self.head, self.hq = head, None
        self.logits = torch.zeros(4, N, dtype=torch.float32, device="cuda")
        self.ref = torch.zeros(4, N, dtype=torch.float32, device="cuda")
        self.tok = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.val = torch.zeros(4, dtype=torch.float32, device="cuda")
        self.tok_ref = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.ws = torch.zeros(lib.smk_headq_ws_bytes(), dtype=torch.uint8, device="cuda")
        self.ws_am = torch.zeros(smk_argmax_ws(lib), dtype=torch.uint8, device="cuda")

    def approx(self, x, T):
        smk.check(self.lib.smk_headq_gemv(self.hq.kind, smk.ptr(self.hq.w), smk.ptr(self.hq.s), smk.ptr(x), N, K, T, smk.ptr(self.logits), None), "headq_gemv")

    def rescore(self, x, T, R):
        smk.check(self.lib.smk_headq_select_rescore(smk.ptr(self.logits), smk.ptr(self.head), smk.ptr(x), N, K, T, R, smk.ptr(self.tok), smk.ptr(self.val), smk.ptr(self.ws), None), "rescore")

    def bf16(self, x, T):
        """the shipped path: bf16 head GEMV + argmax -> (logits [T][N] fp32 in self.ref, argmax in self.tok_ref)"""
        smk.check(self.lib.smk_bf16_gemv_f32(smk.ptr(self.head), smk.ptr(x), N, K, T, smk.ptr(self.ref), None), "bf16 gemv")
        smk.check(self.lib.smk_argmax(smk.ptr(self.ref), T, N, smk.ptr(self.tok_ref), None, smk.ptr(self.ws_am), None), "argmax")

    def cand(self, T, R):
        torch.cuda.synchronize()
        off = self.lib.smk_headq_ws_cand(smk.ptr(self.ws)) - smk.ptr(self.ws)  # device-pointer offset of the candidate table inside ws
        return self.ws[off : off + 4 * 64 * 4].view(torch.int32).view(4, 64)[:T, :R].cpu()


def smk_argmax_ws(lib):
    lib.smk_argmax_ws_bytes.restype = C.c_size_t
    return int(lib.smk_argmax_ws_bytes())


def dequant(hq, r0, r1):
    if hq.kind == 1:
        return hq.w[r0:r1].double() * hq.s[r0:r1].double()[:, None]
    sh = (4 * torch.arange(8, device="cuda", dtype=torch.int64)).view(1, 1, 8)
    nib = ((hq.w[r0:r1].to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> sh) & 0xF
    q = (nib - 8).double().view(r1 - r0, K)
    return (q.view(r1 - r0, -1, 128) * hq.s[r0:r1].double().unsqueeze(-1)).view(r1 - r0, K)


def fp64_logits(W_fn, x):
    """x [T][K] -> fp64 logits [T][N] with W_fn(r0, r1) -> fp64 weights, chunked."""
    out = torch.empty(x.shape[0], N, dtype=torch.float64, device="cuda")
    xd = x.double()
    for r0 in range(0, N, 16384):
        r1 = min(N, r0 + 16384)
        out[:, r0:r1] = xd @ W_fn(r0, r1).T
    return out


def processed(lg, T, k, p):
    """HF/vLLM order: logits / T -> top-k -> softmax -> top-p (keep token i iff the mass before it < p, at least one) -> renormalise; dense fp64 vector [N]."""
    v, i = torch.topk(lg.double() / T, k)
    pr = torch.softmax(v, 0)
    if p < 1.0:
        cum = torch.cumsum(pr, 0) - pr
        keep = cum < p
        keep[0] = True
        pr = pr * keep
        pr = pr / pr.sum()
    out = torch.zeros(lg.shape[0], dtype=torch.float64, device=lg.device)
    out[i] = pr
    return out


TV_CFGS = ((1.0, 20, 0.95), (1.0, 64, 1.0), (0.7, 20, 0.95))


def load_hidden(path, max_pos):
    if os.path.exists(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        H = torch.cat(d["hidden_t4"], 0)
        src = f"{path} ({len(d['hidden_t4'])} docs, real engine final-norm outputs)"
    else:
        raise SystemExit(f"hidden-state file {path} missing: regenerate with experiments/2026-10-31-lm-head-quantisation-quality/run.py collect")
    if max_pos:
        H = H[:max_pos]
    return H.to(torch.bfloat16), src


def logloss(rig, H, docs, R):
    """Teacher-forced log-loss of the true next token (E5 metric; eval docs, positions >= 16) with the logits buffer of the quantised pipeline (exact on the R candidates, approximate
    elsewhere) vs the bf16 head: dNLL of the full-vocabulary softmax, and of the distribution the sampler actually uses (T=1, top-k 64, no top-p: only the top-64 logits matter)."""
    import json

    off, pos = 0, []  # (row in H, target id, doc index)
    for di, d in enumerate(docs):
        n = len(d["ids"])
        if d["role"] == "eval":
            pos += [(off + j, d["ids"][j + 1], di) for j in range(d["eval_from"], n - 1)]
        off += n
    pos = [q for q in pos if q[0] < H.shape[0]]
    rows = {q[0]: q for q in pos}
    full, topk, intk = {}, {}, 0
    for b in range(0, H.shape[0], 4):
        if b + 4 > H.shape[0] or not any((b + t) in rows for t in range(4)):
            continue
        x = H[b : b + 4].cuda().contiguous()
        rig.approx(x, 4)
        rig.rescore(x, 4, R)
        rig.bf16(x, 4)
        for t in range(4):
            q = rows.get(b + t)
            if q is None:
                continue
            tgt = q[1]
            lq, lr = torch.log_softmax(rig.logits[t].double(), 0)[tgt], torch.log_softmax(rig.ref[t].double(), 0)[tgt]
            full.setdefault(q[2], []).append(float(lr - lq))  # NLL_q - NLL_ref = logp_ref - logp_q
            vq, iq = torch.topk(rig.logits[t].double(), 64)
            vr, ir = torch.topk(rig.ref[t].double(), 64)
            inq, inr = bool((iq == tgt).any()), bool((ir == tgt).any())
            if inq and inr:
                topk.setdefault(q[2], []).append(float(torch.log_softmax(vr, 0)[(ir == tgt).nonzero()[0, 0]] - torch.log_softmax(vq, 0)[(iq == tgt).nonzero()[0, 0]]))
                intk += 1
    def stat(dd):
        per = torch.tensor([sum(v) / len(v) for v in dd.values()])
        allv = torch.tensor([x for v in dd.values() for x in v])
        return float(allv.mean()), float(per.std() / len(per) ** 0.5), len(allv)
    return stat(full), stat(topk)


def unit(lib, kind_name, ck, hidden_path, max_pos):
    log(f"## unit: head = {kind_name}")
    G = pack.load_globals(ck, embed=False, head=kind_name)
    hq, head = G.hq, G.lm_head
    t0 = time.time()
    torch.cuda.synchronize()
    rig = Rig(lib, head, kind_name)
    rig.hq = hq
    H, src = load_hidden(hidden_path, max_pos)
    log(f"hidden states: {H.shape[0]} positions from {src}")
    log(f"quantised head bytes {(hq.w.numel() * hq.w.element_size() + hq.s.numel() * hq.s.element_size()) / 2**20:.0f} MiB (bf16 head {head.numel() * 2 / 2**20:.0f} MiB)")
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"- {'PASS' if cond else 'FAIL'}  {name}  {info}")

    # (1) approximate logits vs fp64 with the dequantised head
    xs = H[1000:1008].cuda()
    for T in (1, 2, 3, 4):
        x = xs[:T].contiguous()
        rig.approx(x, T)
        torch.cuda.synchronize()
        ref = fp64_logits(lambda a, b: dequant(hq, a, b), x.float())
        got = rig.logits[:T].double()
        rel = float((got - ref).norm() / ref.norm())
        mx = float((got - ref).abs().max())
        check(f"approx logits T={T} vs fp64 (dequantised head)", rel < 2e-5, f"rel-L2 {rel:.2e}, max abs {mx:.2e}")
    # (2)+(3) candidates / rescoring on 8 positions, R = 16 and 64
    exact_ref = fp64_logits(lambda a, b: head[a:b].double(), xs.float())  # [8][N] exact bf16-head logits in fp64
    for R in (16, 64):
        bad_set = bad_val = bad_arg = 0
        worst = 0.0
        for b in range(0, 8, 4):
            x = xs[b : b + 4].contiguous()
            rig.approx(x, 4)
            appr = rig.logits[:4].clone()
            rig.rescore(x, 4, R)
            cand = rig.cand(4, R)
            torch.cuda.synchronize()
            for t in range(4):
                top = torch.topk(appr[t], R + 8)
                vals, idx = top.values.cpu(), top.indices.cpu()
                order = sorted(range(R + 8), key=lambda i: (-float(vals[i]), int(idx[i])))[:R]
                want = [int(idx[i]) for i in order]
                if cand[t].tolist() != want:
                    bad_set += 1
                ex = rig.logits[t, cand[t].long()].double().cpu()
                rf = exact_ref[b + t, cand[t].long()].cpu()
                worst = max(worst, float(((ex - rf).abs() / rf.abs().clamp_min(1.0)).max()))
                bad_val += int(((ex - rf).abs() > 1e-4 * rf.abs().clamp_min(1.0)).any())
                j = int(rf.argmax())
                bad_arg += int(int(rig.tok[t]) != int(cand[t, j]) and float(rf.max() - rf[j]) > 1e-6)
        check(f"R={R}: candidate ids == exact top-R of the approximate logits (value desc, index asc), 8 rows", bad_set == 0, f"{bad_set} rows differ")
        check(f"R={R}: rescored candidate logits == fp64 dot of the bf16 rows", bad_val == 0, f"max rel err {worst:.2e}")
        check(f"R={R}: argmax == argmax of the exact candidate logits", bad_arg == 0)
    # (4) pipeline vs the bf16 head over all positions
    for R, tag in ((16, "R=16 (greedy)"), (64, "R=64 (sampling)")):
        for T in (4, 1):
            npos = (H.shape[0] // 4) * 4 if T == 4 else min(H.shape[0], 1200)
            mism = miss1 = tie = 0
            tv_ok = {}
            cov20 = cov64 = n20 = 0
            tvs = {c: [] for c in TV_CFGS}
            maxd = 0.0
            first = []
            t0 = time.time()
            for b in range(0, npos, T):
                x = H[b : b + T].cuda().contiguous()
                rig.approx(x, T)
                rig.rescore(x, T, R)
                rig.bf16(x, T)
                a, bb = rig.tok[:T].clone(), rig.tok_ref[:T].clone()
                cand = rig.cand(T, R)
                for t in range(T):
                    isin = bool((cand[t] == bb[t].cpu()).any())
                    miss1 += not isin
                    if int(a[t]) != int(bb[t]):
                        mism += 1
                        top2 = torch.topk(rig.ref[t], 2).values
                        tie += float(top2[0] - top2[1]) < 1e-5
                        if len(first) < 5:
                            first.append((b + t, int(a[t]), int(bb[t]), float(top2[0] - top2[1]), isin))
                    sel = cand[t].cuda().long()
                    maxd = max(maxd, float((rig.logits[t, sel] - rig.ref[t, sel]).abs().max()))
                    if R == 64:
                        t64 = torch.topk(rig.ref[t], 64).indices.cpu()
                        cs = set(cand[t].tolist())
                        cov64 += all(int(i) in cs for i in t64.tolist())
                        cov20 += all(int(i) in cs for i in t64[:20].tolist())
                        n20 += 1
                        if (b + t) % 4 == 0:  # processed-distribution TV (logits buffer = exact on the candidates, approximate elsewhere) vs the bf16 head
                            for c in TV_CFGS:
                                tvs[c].append(0.5 * float((processed(rig.logits[t], *c) - processed(rig.ref[t], *c)).abs().sum()))
            dt = time.time() - t0
            msg = f"{npos} positions, T={T}: argmax mismatches vs bf16 head {mism} (of which fp32 near-ties < 1e-5: {tie}), bf16 top-1 outside the candidate set {miss1}, max |exact - bf16-head logit| on candidates {maxd:.2e} ({dt:.0f} s)"
            if R == 64:
                msg += f"; true top-20 within candidates {cov20}/{n20}, true top-64 within {cov64}/{n20}"
                for c, v in tvs.items():
                    if v:
                        v = torch.tensor(v)
                        msg += f"; TV(T={c[0]},k={c[1]},p={c[2]}) vs bf16 head over {len(v)} positions: mean {float(v.mean()):.5f} p99 {float(v.quantile(0.99)):.5f} max {float(v.max()):.5f}"
                        if T == 4:
                            tv_ok[c] = float(v.mean()) < 0.005
            check(f"{tag}: pipeline vs bf16 head", (mism == tie and miss1 == 0) if R == 16 else (mism == tie and miss1 == 0 and all(tv_ok.values())), msg + (f"  first mismatches (pos, quant, bf16, margin, in_cand) {first}" if first else ""))
    docs_path = os.path.join(os.path.dirname(hidden_path), "docs.json")
    if os.path.exists(docs_path) and not max_pos:
        import json

        t0 = time.time()
        (mf, sf, nf), (mk, sk, nk) = logloss(rig, H, json.load(open(docs_path)), 64)
        log(f"  log-loss (R=64 pipeline logits vs bf16 head, teacher-forced, {nf} eval positions, mean over docs +- s.e. over docs, {time.time() - t0:.0f} s): full-vocab softmax dNLL {mf:+.5f} +- {sf:.5f} nats (E5 int4 rescore16 +0.0018, int8 +0.0000);"
            f" sampler's top-64 distribution dNLL {mk:+.5f} +- {sk:.5f} ({nk} positions with the true token in both top-64)")
        check("log-loss: sampler's top-64 dNLL < 0.002 nats and |full-vocab dNLL| < 0.003", abs(mk) < 0.002 and abs(mf) < 0.003)
    del G, rig, hq, head
    torch.cuda.empty_cache()
    return ok


def parity(args):
    from spec import SpecEngine
    from bench_spec import prompts

    REF = os.path.expanduser("~/llm-lab/swift-ref")
    log(f"## e2e parity: Engine(head={args.heads.split(',')[0]}) quantised head + rescoring vs the BF16 head in the same engine")
    e = SpecEngine(ctx_max=args.ctx, head=args.heads.split(",")[0], verbose=False)
    P = {n: torch.load(f"{REF}/bf16/{n}.pt")["prompt_ids"].tolist() for n in ("code", "prose", "math")}
    P.update({f"e0_{n}": v for n, v in prompts(["code", "prose", "math"]).items()})
    ok = True
    for name, ids in P.items():
        res = {}
        for mode in ("bf16", "q"):
            e.set_head(mode)
            e.reset()
            first = e.prefill(ids)
            plain = [first] + e.decode(args.n - 1)
            spec, st = e.generate(ids, args.n, k=3)
            res[mode] = (plain, spec, st)
        for what, i in (("plain greedy", 0), ("spec k=3 greedy", 1)):
            a, b = res["bf16"][i], res["q"][i]
            same = a == b
            first_bad = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
            ok &= same
            log(f"- {'PASS' if same else 'FAIL'}  {name} {what}: {len(a)} tokens, quantised-head stream {'IDENTICAL to' if same else f'DIVERGES at {first_bad} from'} the bf16-head stream")
        sa, sb = res["bf16"][2], res["q"][2]
        log(f"  spec stats bf16 head: {sa['tokens_per_cycle']:.3f} tok/cycle hist {sa['hist']};  quantised: {sb['tokens_per_cycle']:.3f} tok/cycle hist {sb['hist']}")
    return ok


def sampling(args):
    """Sampled distribution with the quantised head + R=64 rescoring vs the BF16 head, 5 fixed positions x 3 sampling configs: analytic TV between the two processed
    distributions (transformers' temperature -> top-k -> top-p processors on the engine's logits buffer) and the empirical TV of N device draws (smk_test_draw) vs the BF16 analytic one."""
    import numpy as np
    from spec import SpecEngine
    from bench_spec import prompts
    from test_sampling import hf_probs

    head = args.heads.split(",")[0]
    log(f"## sampling: Engine(head={head}) (R=64 rescoring) vs the BF16 head, N={args.nsamp} draws per cell")
    e = SpecEngine(ctx_max=1200, head=head, verbose=False)
    P = prompts(["code", "prose", "math"])
    e.set_head("bf16")
    e.set_sampling(temperature=0.0)
    pos = []
    for name, extra in (("code", 0), ("prose", 0), ("math", 0), ("prose", 60), ("math", 120)):
        ids = P[name]
        if extra:
            e.reset()
            first = e.prefill(ids)
            ids = ids + [first] + e.decode(extra - 1)
        pos.append((f"{name}+{extra}", ids))
    cfgs = [dict(temperature=1.0, top_k=20, top_p=0.95), dict(temperature=1.0, top_k=64, top_p=1.0), dict(temperature=0.7, top_k=20, top_p=0.95)]
    worst_a = worst_e = 0.0
    ok = True
    for cfg in cfgs:
        for name, ids in pos:
            ps, dr = {}, {}
            for mode in ("bf16", "q"):
                e.set_head(mode)
                e.set_sampling(seed=7, **cfg)
                e.reset()
                e.prefill(ids)
                ps[mode] = hf_probs(e.logits(1)[0], cfg["temperature"], cfg["top_k"], cfg["top_p"]).numpy()
                dr[mode] = e.test_draw(args.nsamp, ctr0=1000, row=0).numpy()
            tv_a = 0.5 * float(np.abs(ps["bf16"] - ps["q"]).sum())
            cnt = np.bincount(dr["q"], minlength=ps["bf16"].shape[0]).astype(np.float64) / args.nsamp
            tv_e = 0.5 * float(np.abs(cnt - ps["bf16"]).sum())
            cnt_b = np.bincount(dr["bf16"], minlength=ps["bf16"].shape[0]).astype(np.float64) / args.nsamp
            tv_f = 0.5 * float(np.abs(cnt_b - ps["bf16"]).sum())  # sampling-noise floor: BF16-head draws vs the BF16 analytic distribution
            worst_a, worst_e = max(worst_a, tv_a), max(worst_e, tv_e)
            support = int((ps["bf16"] > 0).sum())
            log(f"  T={cfg['temperature']} k={cfg['top_k']} p={cfg['top_p']} pos {name:9s}: support {support:2d} top-1 p {ps['bf16'].max():.3f}  analytic TV(q-head, bf16-head) {tv_a:.5f}  empirical TV(q draws, bf16 p) {tv_e:.5f}  (bf16 draws vs bf16 p: {tv_f:.5f})")
    ok = worst_a < 0.005
    log(f"- {'PASS' if ok else 'FAIL'}  sampled-distribution TV vs the BF16 head < 0.005 on all {len(pos) * len(cfgs)} cells: worst analytic {worst_a:.5f}, worst empirical {worst_e:.5f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", default="int8,int4")
    ap.add_argument("--what", default="unit,parity,sampling")
    ap.add_argument("--hidden", default=HIDDEN)
    ap.add_argument("--max-pos", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--nsamp", type=int, default=1_000_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    what = set(args.what.replace("e2e", "parity").split(","))
    ok = True
    if "unit" in what:
        lib = smk.load("libswiftmk.so")
        smk.setup_engine_api(lib)
        setup(lib)
        lib.smk_bf16_gemv_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        lib.smk_argmax.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p]
        ck = Checkpoint()
        for h in args.heads.split(","):
            ok &= unit(lib, h, ck, args.hidden, args.max_pos)
    if "parity" in what:
        ok &= parity(args)
    if "sampling" in what:
        ok &= sampling(args)
    log(f"RESULT: {'OK' if ok else 'FAIL'}")
    if args.out:
        open(args.out, "w").write("\n".join(LOG) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
