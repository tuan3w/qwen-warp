"""Stage-1 unit tests of the K3 core ops on REAL layer-0 (DeltaNet) weights and REAL activations from the bf16 oracle dump.

  ~/llm-lab/venv/bin/python tests/test_ops.py [--lm-head]   (needs the GPU lease; --lm-head also loads the 2.5 GB lm_head + embed)

References: oracle fp32 dequant (weights.py QuantizedLinear.dequant) @ activations in fp32 (CPU), and the bf16-oracle dump taps
(`~/llm-lab/swift-ref/bf16/code.pt`, events 0..3 stacked as T=4 rows).  Tolerances (stated in the asserts):
  W4 GEMV      : rel-L2 vs fp32 reference <= 2.5e-3 (bf16 output rounding floor is ~1.7e-3) and vs bf16(reference) <= 4e-4
  rmsnorm      : bit-equal to the dump except <= 0.1% of elements off by one bf16 ulp
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import smk
from smk import check, epi_segs, epi_simple, ptr, rel_l2, stream

sys.path.insert(0, str(smk.ENGINE))
import pack
from weights import Checkpoint

DEV = "cuda"
TOL_FP32, TOL_BF = 2.5e-3, 4e-4
results = []


def ok(name, cond, msg):
    results.append((name, bool(cond), msg))
    print(("PASS " if cond else "FAIL ") + name + "  " + msg, flush=True)


def bf(x):
    return x.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-head", action="store_true")
    ap.add_argument("--layer", type=int, default=0)
    args = ap.parse_args()
    lib = smk.load("libswiftmk_core.so")
    ck = Checkpoint()
    z = torch.load(os.path.expanduser("~/llm-lab/swift-ref/bf16/code.pt"))
    li = args.layer
    taps = [z["taps"][e][li] for e in range(4)]
    cat = lambda k: torch.cat([t[k] for t in taps]).to(DEV)  # [4, dim] fp32 (bf16-valued)
    T = 4
    L = pack.load_layer(ck, li, DEV)
    st = stream()
    P = f"{pack.LM}layers.{li}."

    # ---------------- rmsnorm: xnorm from h_in, fused residual: h_mid = h_in + mixer_out, xnorm2
    h = bf(cat("h_in")).contiguous()
    xn = torch.empty_like(h)
    check(lib.smk_rmsnorm(ptr(h), None, ptr(L.ln1), ptr(xn), None, T, 5120, 1e-6, st))
    torch.cuda.synchronize()
    ref = cat("xnorm")
    d = (xn.float() - ref).abs()
    ulp = (ref.abs() * 2 ** -8)
    ok("rmsnorm", (d > 0).float().mean() < 1e-3 and bool((d <= ulp + 1e-30).all()), f"mismatch frac {(d > 0).float().mean():.2e}, max |d| {d.max():.2e}")
    y = bf(cat("mixer_out")).contiguous()
    dbg = torch.empty_like(h)
    xn2 = torch.empty_like(h)
    check(lib.smk_rmsnorm(ptr(h), ptr(y), ptr(L.ln2), ptr(xn2), ptr(dbg), T, 5120, 1e-6, st))
    torch.cuda.synchronize()
    d1 = (h.float() - cat("h_mid")).abs()
    d2 = (xn2.float() - cat("xnorm2")).abs()
    ok("rmsnorm+residual", (d1 > 0).float().mean() < 1e-3 and (d2 > 0).float().mean() < 1e-3 and torch.equal(dbg, h),
       f"h_mid mismatch {(d1 > 0).float().mean():.2e}, xnorm2 mismatch {(d2 > 0).float().mean():.2e}")

    # ---------------- W4 GEMV on real weights: qkv|z (segments), T = 1..4
    deq = lambda base: ck.quantized_linear(P + base).dequant()  # fp32 [N, K] CPU, exact
    Wqkv, Wz = deq("linear_attn.in_proj_qkv"), deq("linear_attn.in_proj_z")
    x = bf(cat("xnorm")).contiguous()  # [4, 5120]
    for Tt in (1, 2, 3, 4):
        qkv = torch.empty(Tt, 10240, dtype=torch.bfloat16, device=DEV)
        zz = torch.empty(Tt, 6144, dtype=torch.bfloat16, device=DEV)
        e = epi_segs(16384, [qkv, zz])
        qkv.fill_(float("nan")), zz.fill_(float("nan"))
        check(lib.smk_w4_gemv(ptr(L.qkvz.w), ptr(L.qkvz.s), ptr(x), 16384, 5120, Tt, e, st), "w4 qkvz")
        torch.cuda.synchronize()
        xr = x[:Tt].float().cpu()
        for nm, out, W in (("qkv", qkv, Wqkv), ("z", zz, Wz)):
            r = xr @ W.T
            a, b = rel_l2(out.cpu(), r), rel_l2(out.cpu(), bf(r))
            ok(f"w4 {nm} T={Tt}", a <= TOL_FP32 and b <= TOL_BF, f"rel-L2 vs fp32 {a:.2e}, vs bf16(ref) {b:.2e}")
        if Tt == 4:
            a = rel_l2(qkv.cpu(), torch.cat([t["qkv_proj"] for t in taps])), rel_l2(zz.cpu(), torch.cat([t["z"] for t in taps]))
            ok("w4 qkv|z vs dump taps", max(a) <= TOL_FP32, f"rel-L2 qkv {a[0]:.2e} z {a[1]:.2e}")
    del Wqkv, Wz

    # ---------------- every config id agrees (sanity of the tuning variants)
    xr = x.float().cpu()
    r_all = None
    for cfg in range(6):
        qkv = torch.empty(T, 10240, dtype=torch.bfloat16, device=DEV)
        zz = torch.empty(T, 6144, dtype=torch.bfloat16, device=DEV)
        e = epi_segs(16384, [qkv, zz])
        rc = lib.smk_w4_gemv_cfg(ptr(L.qkvz.w), ptr(L.qkvz.s), ptr(x), 16384, 5120, T, e, cfg, st)
        torch.cuda.synchronize()
        if rc == 0:
            if r_all is None:
                r_all = qkv.clone()
            ok(f"w4 cfg {cfg}", rel_l2(qkv, r_all) <= 2e-5, f"rel-L2 vs first valid cfg {rel_l2(qkv, r_all):.1e} (fp32 K-split order differs)")

    # ---------------- gate|up + SiLU epilogue (T=4), then down, o_proj
    x2 = bf(cat("xnorm2")).contiguous()
    act = torch.empty(T, 17408, dtype=torch.bfloat16, device=DEV)
    check(lib.smk_w4_gemv(ptr(L.gu.w), ptr(L.gu.s), ptr(x2), 34816, 5120, T, epi_simple(smk.SMK_EPI_SILU, act), st), "gu")
    torch.cuda.synchronize()
    Wg, Wu = deq("mlp.gate_proj"), deq("mlp.up_proj")
    xr = x2.float().cpu()
    g, u = bf(xr @ Wg.T).float(), bf(xr @ Wu.T).float()
    ref_act = bf(bf(torch.nn.functional.silu(g)).float() * u)
    a = rel_l2(act.cpu(), ref_act)
    b = rel_l2(act.cpu(), torch.cat([t["mlp_act"] for t in taps]))
    ok("gate|up silu*mul", a <= 6e-3 and b <= 6e-3, f"rel-L2 vs torch bf16 chain {a:.2e}, vs dump mlp_act {b:.2e}")
    del Wg, Wu
    out = torch.empty(T, 5120, dtype=torch.bfloat16, device=DEV)
    xa = bf(cat("mlp_act")).contiguous()
    check(lib.smk_w4_gemv(ptr(L.down.w), ptr(L.down.s), ptr(xa), 5120, 17408, T, epi_segs(5120, [out]), st), "down")
    torch.cuda.synchronize()
    Wd = deq("mlp.down_proj")
    r = xa.float().cpu() @ Wd.T
    a, b = rel_l2(out.cpu(), r), rel_l2(out.cpu(), bf(r))
    ok("w4 down T=4", a <= TOL_FP32 and b <= TOL_BF, f"rel-L2 vs fp32 {a:.2e}, vs bf16(ref) {b:.2e}")
    del Wd
    if L.kind == "linear":
        xg = bf(cat("gated_norm_out")).contiguous()
        out = torch.empty(T, 5120, dtype=torch.bfloat16, device=DEV)
        check(lib.smk_w4_gemv(ptr(L.o.w), ptr(L.o.s), ptr(xg), 5120, 6144, T, epi_segs(5120, [out]), st), "oproj")
        torch.cuda.synchronize()
        Wo = deq("linear_attn.out_proj")
        r = xg.float().cpu() @ Wo.T
        a, b = rel_l2(out.cpu(), r), rel_l2(out.cpu(), bf(r))
        ok("w4 out_proj T=4", a <= TOL_FP32 and b <= TOL_BF, f"rel-L2 vs fp32 {a:.2e}, vs bf16(ref) {b:.2e}")

        # ---------------- small BF16 GEMV a|b
        oa = torch.empty(T, 48, dtype=torch.bfloat16, device=DEV)
        ob = torch.empty(T, 48, dtype=torch.bfloat16, device=DEV)
        check(lib.smk_bf16_gemv_small(ptr(L.ab), ptr(x), 96, 48, 5120, T, ptr(oa), ptr(ob), st), "ab")
        torch.cuda.synchronize()
        ra, rb = cat("a").cpu(), cat("b").cpu()
        ok("bf16 small a|b", rel_l2(oa.cpu(), ra) <= 3e-3 and rel_l2(ob.cpu(), rb) <= 3e-3, f"rel-L2 a {rel_l2(oa.cpu(), ra):.2e} b {rel_l2(ob.cpu(), rb):.2e}")

    # ---------------- lm_head + argmax + embed
    if args.lm_head:
        G = pack.load_globals(ck, DEV)
        toks = z["event_tokens"][:9].int().to(DEV)
        emb = torch.empty(9, 5120, dtype=torch.bfloat16, device=DEV)
        check(lib.smk_embed(ptr(G.embed), ptr(toks), ptr(emb), 9, 5120, st))
        torch.cuda.synchronize()
        ok("embed", torch.equal(emb, G.embed[toks.long()]), "bit-equal to embed rows")
        ok("embed vs dump hid_in[:,0]", rel_l2(emb.cpu(), z["hid_in"][:, 0]) == 0.0, "bf16 dump embedding")
        fin = bf(z["final_norm_out"]).to(DEV).contiguous()  # [9, 5120]
        ws = torch.zeros(lib.smk_argmax_ws_bytes(), dtype=torch.uint8, device=DEV)
        gen = z["gen_ids"][:9].tolist()
        for Tt in (1, 2, 4, 8):
            logit = torch.empty(Tt, 248320, dtype=torch.float32, device=DEV)
            check(lib.smk_bf16_gemv_f32(ptr(G.lm_head), ptr(fin), 248320, 5120, Tt, ptr(logit), st), "lm_head")
            tok = torch.empty(Tt, dtype=torch.int32, device=DEV)
            val = torch.empty(Tt, dtype=torch.float32, device=DEV)
            check(lib.smk_argmax(ptr(logit), Tt, 248320, ptr(tok), ptr(val), ptr(ws), st), "argmax")
            torch.cuda.synchronize()
            rl = fin[:Tt].float() @ G.lm_head.float().T
            ok(f"lm_head T={Tt} logits", rel_l2(logit, rl) <= 2e-6, f"rel-L2 vs fp32 matmul {rel_l2(logit, rl):.2e}")
            ok(f"argmax T={Tt}", tok.tolist() == gen[:Tt] and torch.equal(tok.long(), logit.argmax(1)), f"{tok.tolist()} vs oracle {gen[:Tt]}")
            tv = z["top20_val"][:Tt, 0]
            ok(f"top1 logit T={Tt}", rel_l2(val.cpu(), tv) <= 2e-3, f"rel-L2 vs dump top20_val[:,0] {rel_l2(val.cpu(), tv):.2e}")
    bad = [r for r in results if not r[1]]
    print(f"{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
