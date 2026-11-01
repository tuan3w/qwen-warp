"""K4 tests: int4 MTP packer, draft ids vs the oracle's BF16 MTP, speculative greedy == plain greedy, DeltaNet/conv ring rollback under forced partial acceptance.

  SMK_BUILD=build/k4 ~/llm-lab/venv/bin/python tests/test_mtp.py [--prompts code,prose,math] [--n 128] [--ks 1,2,3] [--quant int4clip|int4] [--out MTP_PARITY.md]
Needs the GPU lease.  Sections:
  1 packer     : layout (oracle dequant == packer dequant) and int4 error of the packed MTP fc on random weights
  2 head       : restricted int4 head argmax == full BF16 lm_head argmax on the dump's MTP hidden states and final-norm hidden states
  3 drafts     : engine prefill + forced a=1 cycles: emitted tokens == oracle gen_ids, draft ids == oracle drafts (BF16 MTP, chained k=1..3), MTP hidden rel-L2
  4 greedy     : spec generate(k) == plain engine greedy for n tokens (divergence -> top-1/top-2 logit margin of the plain engine)
  5 rollback   : same with forced partial acceptance patterns (every cycle accepts fewer rows than verified: ring slot / conv slot / KV overwrite)
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spec import SpecEngine, pack_mtp, smk  # noqa: E402

REF = os.path.expanduser("~/llm-lab/swift-ref")
OUT = []
# Logit margin (top1 - top2 of the plain engine) below which a plain-vs-spec divergence between exactly those two tokens is a rounding near-tie (T=1 decode vs T-row verify/prefill
# accumulate order).  Set from data (BENCH.md "K7 follow-up"): over 3 prompts x 128 positions the T=1-vs-T=4 change of that margin has median 0.029, p99 0.127, max 0.178 on the
# current GEMV (old GEMV: p99 0.227, max 0.54); 3 x p99 = 0.38 -> 0.4.  The previous 0.05 was ~ the median delta, so it rejected ordinary rounding flips (prose idx 104, margin 0.0655).
NEAR_TIE = 0.4
TOT = [0, 0, 0.0]  # draft-1 id matches, draft-1..3 id matches, sum of d1 hidden rel-L2 over prompts
SAVED = {}  # engine drafts / MTP hidden per prompt (for the fake-quant oracle cross-check)


def log(s=""):
    print(s, flush=True)
    OUT.append(s)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def test_packer():
    from weights import dequant_w4a16

    torch.manual_seed(0)
    w = torch.randn(256, 1024).bfloat16() * 0.05
    w[3, :128] *= 20  # heavy-tail group
    ok = True
    for clip in (False, True):
        p, s = pack_mtp.quant_w4(w, clip)
        a = dequant_w4a16(p.cpu(), s.cpu())  # oracle / compressed-tensors layout
        b = pack_mtp.dequant_w4(p, s).cpu()
        e = rel(a, b)
        err = rel(b, w.float())
        log(f"- packer clip={clip}: oracle-dequant vs packer-dequant max|diff| {float((a - b).abs().max()):.2e}; int4 rel-L2 vs source {err:.4f}")
        ok &= e == 0.0 and err < 0.12
    return ok


def test_head(e, names):
    M = e.M
    ids = M.ids_t.long()
    n = agree = nm = 0
    for name in names:
        zb = torch.load(f"{REF}/bf16/{name}.pt")
        hs = torch.cat([zb["mtp_hidden"].reshape(-1, 5120), zb["final_norm_out"]]).cuda().bfloat16()
        nh = hs.shape[0]

        def best_over(chunks):
            best = torch.full((nh,), -1e30, device="cuda")
            bi = torch.zeros(nh, dtype=torch.long, device="cuda")
            for r0, wc in chunks:
                v, i = (hs.float() @ wc.T).max(1)
                m = v > best
                best = torch.where(m, v, best)
                bi = torch.where(m, i + r0, bi)
            return bi

        full_i = best_over((r0, e.G.lm_head[r0 : r0 + 8192].float()) for r0 in range(0, 248320, 8192))
        rid = ids[best_over((r0, pack_mtp.dequant_w4(M.head.w[r0 : r0 + 4096], M.head.s[r0 : r0 + 4096])) for r0 in range(0, M.nv, 4096))]
        n += nh
        agree += int((rid == full_i).sum())
        nm += int((~torch.isin(full_i, ids)).sum())
    log(f"- restricted int4 head argmax == full BF16 head argmax: {agree}/{n} (full-head argmax outside the 64k draft vocab: {nm})")
    return agree >= 0.9 * n


def test_drafts(e, names, k=3):
    ok = True
    for name in names:
        zb = torch.load(f"{REF}/bf16/{name}.pt")
        ids = zb["prompt_ids"].tolist()
        gen = zb["gen_ids"].tolist()
        dr = zb["drafts"].tolist()  # [9][3]
        e.reset()
        e.set_force([1])
        first = e.prefill_spec(ids, k)
        pos, cur, tk = e.draft_state()
        got = [tk[1:4]]
        toks = [first]
        hrel = []
        mh = e.mtp_hidden()
        hid = [mh[:3].clone()]
        hrel.append([rel(mh[j], zb["mtp_hidden"][0, j]) for j in range(3)])
        for ev in range(1, 9):
            new, ss = e.cycles(k, 1)
            toks += new
            got.append(e.draft_state()[2][1:4])
            mh = e.mtp_hidden()
            hid.append(mh[:3].clone())
            hrel.append([rel(mh[j], zb["mtp_hidden"][ev, j]) for j in range(3)])
        tok_ok = toks == gen
        SAVED[name] = dict(drafts=torch.tensor(got), hidden=torch.stack(hid))
        m1 = sum(got[ev][0] == dr[ev][0] for ev in range(9))
        m2 = sum(got[ev][:2] == dr[ev][:2] for ev in range(9))
        m3 = sum(got[ev] == dr[ev] for ev in range(9))
        hm = torch.tensor(hrel).mean(0)
        log(f"- {name}: tokens {'EQUAL' if tok_ok else 'DIFFER'} vs oracle gen_ids; draft ids == oracle BF16-MTP drafts: draft1 {m1}/9, drafts1-2 {m2}/9, drafts1-3 {m3}/9; "
            f"MTP hidden rel-L2 vs oracle (mean over 9 events) d1 {hm[0]:.4f} d2 {hm[1]:.4f} d3 {hm[2]:.4f}")
        ok &= tok_ok
        TOT[0] += m1; TOT[1] += m3; TOT[2] += float(hm[0])
        if m3 < 9:
            bad = [(ev, got[ev], dr[ev]) for ev in range(9) if got[ev] != dr[ev]]
            log(f"    mismatches (event, mine, oracle): {bad[:4]}")
    e.set_force(None)
    nn = 9 * len(names)
    log(f"- total vs BF16 oracle: draft1 ids {TOT[0]}/{nn}, drafts1-3 ids {TOT[1]}/{nn}, mean d1 hidden rel-L2 {TOT[2] / len(names):.3f} (int4 MTP layer: expected ~0.2; "
        f"engine vs the SAME int4 weights in the oracle = 0.014-0.016, tests/check_mtp_fq.py)")
    return ok and TOT[0] >= 0.85 * nn and TOT[2] / len(names) < 0.3


def test_eos(e, names, n=64):
    ok = True
    for name in names:
        ids = torch.load(f"{REF}/bf16/{name}.pt")["prompt_ids"].tolist()
        ref = plain(e, ids, n)
        eos = {ref[10]}
        cut = ref.index(ref[10])
        got, st = e.generate(ids, n, k=3, eos=eos)
        good = got == ref[: cut + 1]
        ok &= good
        log(f"- eos {name}: token {ref[10]} first at {cut}; spec stops with {len(got)} tokens, prefix {'EQUAL' if good else 'DIFFERS'}")
    return ok


def plain(e, ids, n):
    e.reset()
    first = e.prefill(ids)
    return [first] + e.decode(n - 1)


def margin_at(e, ids, toks, i):
    """top1 / top2 logit (and the logits of toks[i]) of the plain engine at generated position i (prefix ids + toks[:i])."""
    e.reset()
    e.prefill(ids + toks[:i])
    lg = e.logits(1)[0]
    v, ix = lg.topk(2)
    return float(v[0]), float(v[1]), int(ix[0]), int(ix[1])


def compare(e, ids, ref, got, tag):
    if got == ref:
        return True, f"{tag}: EQUAL ({len(ref)} tokens)"
    i = next(j for j in range(len(ref)) if ref[j] != got[j])
    v1, v2, i1, i2 = margin_at(e, ids, ref, i)
    lg = None
    near = (v1 - v2) < NEAR_TIE and {ref[i], got[i]} == {i1, i2}
    return near, (f"{tag}: {'NEAR-TIE divergence (accepted)' if near else 'DIVERGES'} at generated index {i} (plain {ref[i]}, spec {got[i]}); plain-engine (prefix-prefill) logits top1 {i1}={v1:.4f} top2 {i2}={v2:.4f}, margin {v1 - v2:.4f}")


def test_greedy(e, names, ks, n, force=None, tag="greedy"):
    ok = True
    for name in names:
        zb = torch.load(f"{REF}/bf16/{name}.pt")
        ids = zb["prompt_ids"].tolist()
        ref = plain(e, ids, n)
        for k in ks:
            t0 = time.time()
            got, st = e.generate(ids, n, k=k, force=force)
            good, msg = compare(e, ids, ref, got, f"{name} k={k}")
            ok &= good
            log(f"- {tag} {msg}; mean tokens/cycle {st['tokens_per_cycle']:.2f}, natural accepted-drafts histogram m=0..3 {st['hist']} ({time.time() - t0:.1f}s)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="code,prose,math")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--quant", default="int4clip")
    ap.add_argument("--head-quant", default="int4")
    ap.add_argument("--head-mode", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save", default=None, help="save engine drafts + MTP hidden (section 3) to this .pt")
    a = ap.parse_args()
    names = a.prompts.split(",")
    ks = [int(x) for x in a.ks.split(",")]
    only = set(a.only.split(",")) if a.only else None
    want = lambda s: only is None or s in only
    e = SpecEngine(ctx_max=a.ctx, quant=a.quant, head_quant=a.head_quant, head_mode=a.head_mode)
    log(f"MTP layer quant {a.quant}, draft head {a.head_quant} nv={e.M.nv} head_mode {a.head_mode}; spec buffers {e.lib.smk_spec_bytes(e.sp) / 2**20:.0f} MiB")
    res = {}
    if want("packer"):
        log("### 1 packer"); res["packer"] = test_packer()
    if want("head"):
        log("### 2 draft head vs full head"); res["head"] = test_head(e, names)
    if want("drafts"):
        log("### 3 drafts vs oracle dump (BF16 MTP), forced a=1 cycles"); res["drafts"] = test_drafts(e, names)
    if want("eos"):
        log("### 6 host-side EOS stop"); res["eos"] = test_eos(e, names)
    if want("greedy"):
        log(f"### 4 speculative greedy == plain greedy ({a.n} tokens)"); res["greedy"] = test_greedy(e, names, ks, a.n)
    if want("rollback"):
        log("### 5 forced partial acceptance (rollback)")
        r = True
        for pat in ([1], [2, 1, 3], [1, 4, 2, 3, 1]):
            log(f"force pattern {pat}")
            e.set_force(pat)
            r &= test_greedy(e, names, [3] if True else ks, a.n, force=pat, tag=f"rollback{pat}")
        res["rollback"] = r
    if SAVED and a.save:
        torch.save(SAVED, a.save)
    log("RESULT: " + " ".join(f"{k}={'OK' if v else 'FAIL'}" for k, v in res.items()))
    if a.out:
        open(a.out, "w").write("\n".join(OUT) + "\n")
    sys.exit(0 if all(res.values()) else 1)


if __name__ == "__main__":
    main()
