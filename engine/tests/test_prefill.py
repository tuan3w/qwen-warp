"""K6 fast prefill vs the slow prefill-by-decode path (and the oracle dumps).

  cd <repo root>; SMK_BUILD=build_k6 ~/llm-lab/venv/bin/python engine/tests/test_prefill.py [--fp8 0|1] [--long 1500] [--seg 4096] [--segs 512] [--k 3]
Per prompt (3 oracle dump prompts + a --long token prompt from topics/*.md):
  slow:  Engine.prefill-style T=4 decode chunks (SpecEngine.prefill_spec);  fast: FastPrefill.prefill_spec (+ a small-segment run to exercise segment boundaries).
  checks: final hidden of every layer of the last prompt token (rel-L2 vs the slow path and vs the bf16 oracle dump for the dump prompts), DeltaNet state / conv state of
  every layer (slot cur), full-attention KV of every layer (K and V rel-L2), MTP KV cache, MTP drafts, first token, 32 plain greedy ids and 64 speculative (k) ids equal.
  PASS needs: first token + both continuations EQUAL to the slow path (or a divergence where the slow path's logit gap to the fast path's token is <= 0.15),
  hidden rel-L2 <= 0.03 per layer (fp8 KV: 0.04 - the slow path itself is 0.029 off the bf16 oracle on the `code` prompt there, the K10 fold 0.019), DeltaNet/conv state <= 0.05, KV caches <= 0.10 (fp8 KV: two independently rounded fp8 caches differ by ~0.05-0.08).
  --baseline 1 adds the noise floor: the SAME slow path with T=1 instead of T=4 chunks vs the T=4 slow path (informational, never fails): it shows the size of
  differences between mathematically equal runs (4K tokens: hidden 0.021, V cache 0.040, DeltaNet state 0.026).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

ENG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(ENG / "tests"))
from spec import SpecEngine, smk  # noqa: E402
from prefill import FastPrefill  # noqa: E402

REF = os.path.expanduser("~/llm-lab/swift-ref")
MODEL = Path.home() / "models" / "swift-1.5-int4"


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def snapshot(e, fp, N):
    """clone of everything the decode needs after a prefill of N positions (positions 0..N-1 in the caches)."""
    cur = int(fp.ds[1])
    s = dict(cur=cur, pos=int(fp.ds[0]))
    s["dn"] = fp.dn_state[:, cur].clone()
    s["conv"] = fp.dn_conv[:, cur].clone()
    kvb = e.ctx_max
    kv = []
    for ai in range(16):
        raw = torch.empty(0)
        from prefill import _view

        nb = fp.kv_layer_bytes
        t = _view(fp.kv_base + ai * nb, nb, torch.uint8 if e.kv_fp8 else torch.bfloat16)
        t = t.view(2, 4, kvb, 256)[:, :, :N]
        kv.append(t.clone())
    s["kv"] = kv
    mk = _view(fp.mtp_kv, 2 * 4 * kvb * 256 * 2, torch.bfloat16).view(2, 4, kvb, 256)[:, :, : N - 1]
    s["mtp_kv"] = mk.clone()
    s["hid"] = e.dbg_hidden(1)[:, 0].clone()  # [66][5120] last prompt token
    s["draft"] = e.draft_state()
    return s


def kv_f(t, fp8, scale):
    return t.float() * scale if fp8 else t.float()


def compare(e, fp, a, b, name, lines, ref_hid=None):
    """a = fast, b = slow snapshots."""
    N = a["pos"]
    r_h = max(rel(a["hid"][l + 1], b["hid"][l + 1]) for l in range(64))
    r_f = rel(a["hid"][65], b["hid"][65])
    r_dn = [rel(a["dn"][i], b["dn"][i]) for i in range(48)]
    r_cv = [rel(a["conv"][i], b["conv"][i]) for i in range(48)]
    r_k, r_v = [], []
    for ai in range(16):
        if e.kv_fp8:
            L = [l for l in range(64) if e.Ls[l].kind != "linear"][ai]
            ks, vs = e.Ls[L].k_scale, e.Ls[L].v_scale
        else:
            ks = vs = 1.0
        ka, kb = kv_f(a["kv"][ai][0], e.kv_fp8, ks), kv_f(b["kv"][ai][0], e.kv_fp8, ks)
        va, vb = kv_f(a["kv"][ai][1], e.kv_fp8, vs), kv_f(b["kv"][ai][1], e.kv_fp8, vs)
        r_k.append(rel(ka, kb))
        r_v.append(rel(va, vb))
    r_mk = rel(a["mtp_kv"][0], b["mtp_kv"][0]), rel(a["mtp_kv"][1], b["mtp_kv"][1])
    lines.append(
        f"- {name}: last-token hidden rel-L2 vs slow: max over 64 layers {r_h:.4f}, final-norm {r_f:.4f}; DeltaNet state max/mean {max(r_dn):.4f}/{sum(r_dn)/48:.4f}, "
        f"conv state max {max(r_cv):.4f}; KV K max/mean {max(r_k):.4f}/{sum(r_k)/16:.4f}, V max/mean {max(r_v):.4f}/{sum(r_v)/16:.4f} (attention layer idx of max V: {r_v.index(max(r_v))}, per-layer V: {' '.join(f'{x:.3f}' for x in r_v)}); MTP KV K/V {r_mk[0]:.4f}/{r_mk[1]:.4f}; "
        f"drafts fast {a['draft'][2]} slow {b['draft'][2]}"
    )
    ok = r_h <= (0.04 if e.kv_fp8 else 0.03) and max(r_dn) <= 0.05 and max(r_cv) <= 0.05 and max(r_k) <= 0.10 and max(r_v) <= 0.10 and max(r_mk) <= 0.10
    if ref_hid is not None:
        r_a = max(rel(a["hid"][l + 1], ref_hid[l]) for l in range(64))
        r_b = max(rel(b["hid"][l + 1], ref_hid[l]) for l in range(64))
        lines.append(f"  vs bf16 oracle dump (last prompt token, max over layers): fast {r_a:.4f}, slow {r_b:.4f}")
        ok &= r_a <= 0.03
    return ok


def long_prompt(n_tok):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    ids = tok(raw, add_special_tokens=False)["input_ids"][: n_tok - 40]
    text = tok.apply_chat_template([{"role": "user", "content": tok.decode(ids) + "\n\nSummarise the main claims above in a numbered list."}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", type=int, default=0)
    ap.add_argument("--long", type=int, default=1500)
    ap.add_argument("--seg", type=int, default=4096)
    ap.add_argument("--segs", type=int, default=512, help="small segment size for the boundary test (0 = skip)")
    ap.add_argument("--w4segs", type=int, default=25, help="K10: segment size of the W4-mma-GEMM boundary test (0 = skip)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prompts", default="code,prose,math,long")
    ap.add_argument("--baseline", type=int, default=0, help="also run the slow path with T=1 steps as a noise-floor comparison")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    e = SpecEngine(ctx_max=args.ctx, kv_fp8=args.fp8, verbose=False)
    fp = FastPrefill(e, seg=args.seg, min_fast=0, w4_max=0)  # K6: dequant + cuBLAS
    fps = FastPrefill(e, seg=args.segs, min_fast=0, w4_max=0) if args.segs else None
    fpn = FastPrefill(e, seg=args.seg, min_fast=0, w4_max=args.seg, fold=False)  # K10: every GEMM through the W4 mma kernel (64-row chunks), one segment, last row as a T=1 head step
    fpw = FastPrefill(e, seg=args.seg, min_fast=0, w4_max=args.seg)  # ... and the last row inside the final batch (fold: decode DeltaNet row + engine head tail)
    fpws = FastPrefill(e, seg=args.w4segs, min_fast=0, w4_max=args.seg) if args.w4segs else None  # ... in small segments (state / KV / MTP carried across boundaries)
    fpw.taps = True  # folded head: no step writes the per-layer debug taps of the last row
    if fpws:
        fpws.taps = True
    fps_ = {"fast": fp, "fast_seg": fps, "w4_nofold": fpn, "w4": fpw, "w4_seg": fpws}
    print(f"engine {e.lib.smk_engine_bytes(e.h) / 2**30:.2f} GiB, torch {torch.cuda.memory_allocated() / 2**30:.2f} GiB, kv_fp8={args.fp8}", flush=True)
    lines, allok = [], True
    for name in args.prompts.split(","):
        if name == "long":
            ids = long_prompt(args.long)
            ref = None
        else:
            z = torch.load(f"{REF}/bf16/{name}.pt")
            ids = z["prompt_ids"].tolist()
            ref = z["hid_out"][0]  # [64][5120] event 0 = last prompt token
        N = len(ids)
        hdr = len(lines)
        lines.append(f"### {name} ({N} prompt tokens, kv_fp8={args.fp8})")

        def run(kind):
            e.reset()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if kind == "slow":
                first = e.prefill_spec(ids, args.k)
            elif kind == "slow1":  # noise-floor baseline: the same slow path with T=1 steps (different accumulation order)
                first = e.prefill_spec(ids, args.k, chunk=1)
            else:
                first = fps_[kind].prefill_spec(ids, args.k)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            snap = snapshot(e, fp, N)
            toks, _ = e.cycles(args.k, 8)  # 8 spec cycles (>= 8 tokens)
            spec_toks = [first] + toks
            # plain greedy from a fresh state of the same kind (spec cycles advanced the state)
            e.reset()
            if kind == "slow":
                first2 = e.prefill(ids)
            elif kind == "slow1":
                first2 = e.prefill(ids, chunk=1)
            else:
                first2 = fps_[kind].prefill(ids)
            plain = [first2] + e.decode(32)
            return first, snap, spec_toks, plain, dt

        def near_tie(idx, other_tok):
            """slow-path logits that choose generated token `idx` (plain decode): (logit of the argmax - logit of the other path's token, True)."""
            e.reset()
            e.prefill(ids)
            if idx > 0:
                e.decode(idx)
            lg = e.logits(1)[0]
            return float(lg.max() - lg[other_tok]), True

        kinds = ["slow", "fast"] + (["fast_seg"] if fps else []) + ["w4_nofold", "w4"] + (["w4_seg"] if fpws else []) + (["slow1"] if args.baseline else [])
        res = {"slow": run("slow")}
        sf = res["slow"]
        for k in kinds[1:]:
            res[k] = run(k)  # (snapshots of every kind at once do not fit next to the 18 GB of weights on a long prompt: compare, then drop)
            if k == "fast":
                lines.append(f"- first token slow {sf[0]} fast {res['fast'][0]}; prefill time slow {sf[4]:.2f}s, fast {res['fast'][4]:.3f}s (first call includes triton autotune/compile)")
            ok = compare(e, fp, res[k][1], sf[1], k, lines, ref)
            n = min(len(res[k][2]), len(sf[2]))
            eq_spec = res[k][2][:n] == sf[2][:n]
            eq_plain = res[k][3] == sf[3]
            first_bad = next((i for i, (x, y) in enumerate(zip(res[k][3], sf[3])) if x != y), None)
            lines.append(f"  {k}: first token {'EQUAL' if res[k][0] == sf[0] else 'DIFFER'}, plain greedy 33 ids {'EQUAL' if eq_plain else 'DIFFER first at ' + str(first_bad)}, "
                         f"spec k={args.k} ids ({n}) {'EQUAL' if eq_spec else 'DIFFER'}")
            if name != "long":
                zz = torch.load(f"{REF}/bf16/{name}.pt")
                lines.append(f"  vs oracle gen_ids (9): fast {'EQUAL' if res[k][3][:9] == zz['gen_ids'].tolist() else 'DIFFER'}, slow {'EQUAL' if sf[3][:9] == zz['gen_ids'].tolist() else 'DIFFER'}")
                ok &= res[k][3][:9] == zz["gen_ids"].tolist()
            ok &= res[k][0] == sf[0]
            for what, eq, a_, b_ in (("plain", eq_plain, res[k][3], sf[3]), ("spec", eq_spec, res[k][2][:n], sf[2][:n])):
                if eq:
                    continue
                d = next(i for i, (x, y) in enumerate(zip(a_, b_)) if x != y)
                mg, is2 = near_tie(d, a_[d])
                tie = mg <= 0.15 and is2
                lines.append(f"  {k} {what} divergence at generated index {d}: slow logit gap top1 - other path's token {mg:.4f} -> {'NEAR-TIE (accepted)' if tie else 'REAL MISMATCH'}")
                ok &= tie
            if k != "slow1":
                allok &= ok
            res[k] = res[k][:1] + (None,) + res[k][2:]
            torch.cuda.empty_cache()
        print("\n".join(lines[hdr:]), flush=True)
    # default configuration (min_fast / w4_max: very short prompts fall back to the engine's own prefill, short ones take the W4 mma path, longer ones dequant + cuBLAS)
    fpd = FastPrefill(e)
    for n_ in (4, 6, 32, 200):
        ids_ = (ids * (n_ // len(ids) + 1))[:n_]
        e.reset()
        a_ = (e.prefill_spec(ids_, args.k), e.draft_state()[2])
        e.reset()
        b_ = (fpd.prefill_spec(ids_, args.k), e.draft_state()[2])
        e.reset()
        c_, d_ = e.prefill(ids_), None
        e.reset()
        d_ = fpd.prefill(ids_)
        okd = a_ == b_ and c_ == d_
        path = "fallback" if n_ - 1 < fpd.min_fast else ("W4 mma path" if n_ - 1 <= fpd.w4_max else "dequant + cuBLAS path")
        lines.append(f"- default FastPrefill (min_fast {fpd.min_fast}, w4_max {fpd.w4_max}), {n_}-token prompt ({path}): spec first token + drafts {'EQUAL' if a_ == b_ else 'DIFFER ' + str((a_, b_))}, plain first token {'EQUAL' if c_ == d_ else 'DIFFER'}")
        allok &= okd
    lines.append("RESULT: " + ("PASS" if allok else "FAIL"))
    print(lines[-1])
    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
