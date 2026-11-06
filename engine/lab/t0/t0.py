"""T0 quick lab: evaluate ONE sparsity rule on the real traces in seconds.

  python t0.py act  --kind down --rule abs_topk --b 8 --f 0.9,0.7,0.5 [--mode row|union|max] [--layers 2,23] [--docs a,b] [--dev cpu|cuda]
        -> per f: weight-byte fraction and mean per-row relative output error (mean over docs x layers), plus the error per layer class
  python t0.py attn --scorer oracle|quest|quest_rot|mean|samp4|maxlogit --b 64 --budget 1024,4096 [--pool head|kv_p|row8_p] [--layers 23,63] [--docs a]   (GPU)
        -> output rel. error / attention KL / 8-row union fraction per budget
Rules: sp.make_mask (abs_topk = TEAL oracle, wina_topk, max_topk, abs_thr, static).  Real-model KL of the same rule: lab/traces/kl_eval.py."""
import argparse
import time

import numpy as np
import torch

import sp

DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]


def act(a):
    sp.DEV = a.dev
    docs, layers = a.docs.split(","), [int(x) for x in a.layers.split(",")]
    T = {d: sp.Trace(d, a.dev) for d in docs}
    print(f"{a.kind} rule={a.rule} b={a.b} mode={a.mode}   (bytes fraction of the weight stream / mean per-row rel. output error)")
    for f in [float(x) for x in a.f.split(",")]:
        res = {"DN": [], "ATT": []}
        for li in layers:
            W = sp.Wcache(li)[a.kind]
            w, cn = W.w.to(a.dev), W.cn.to(a.dev)
            for d in docs:
                x = T[d].lin(li, a.kind, "dec")
                calib = torch.cat([T[o].lin(li, a.kind, "doc") for o in docs if o != d]) if a.rule in ("abs_thr", "static") and len(docs) > 1 else x
                m = sp.make_mask(x, a.rule, f, a.b, cn, calib) if a.mode == "row" else sp.union_mask(x, a.rule, f, a.b, cn, 8, a.mode)
                y = x @ w.t()
                err = (((x * m) @ w.t() - y).norm(dim=1) / y.norm(dim=1)).mean().item()
                by = sp.bytes_frac(m if a.mode == "row" else m[::8], a.b).mean().item()
                res["ATT" if li in sp.ATTN_LAYERS else "DN"].append((by, err))
        allr = res["DN"] + res["ATT"]
        s = f"f={f:.2f}: bytes {np.mean([r[0] for r in allr]):.3f} err {np.mean([r[1] for r in allr]) * 100:.2f}%"
        for k, v in res.items():
            if v:
                s += f"   [{k}: bytes {np.mean([r[0] for r in v]):.3f} err {np.mean([r[1] for r in v]) * 100:.2f}%]"
        print(s)


def attn(a):
    import b2_attn

    b2_attn.DEV = "cuda"
    docs, layers = a.docs.split(","), [int(x) for x in a.layers.split(",")]
    gen = torch.Generator(device="cuda").manual_seed(0)
    budgets = [int(x) for x in a.budget.split(",")]
    rows = []
    for d in docs:
        T = sp.Trace(d, "cuda")
        for li in layers:
            out, Ly = b2_attn.run_layer(d, li, T, [a.b], budgets, [a.pool], [a.scorer], {}, None, gen)
            out.pop((li, "_stats", "ent", 0, 0), None)
            rows += [r for r in b2_attn.reduce(out, d) if r[2] == a.scorer and r[3] == a.pool]
            del Ly
    print(f"{a.scorer} b={a.b} pool={a.pool}  (mean over {len(docs)} docs x {len(layers)} layers)")
    for bud in budgets:
        r = np.array([[x[6], x[7], x[10]] for x in rows if x[5] == bud])
        print(f"budget {bud}: out-err {r[:, 0].mean():.4f}  attn-KL {r[:, 1].mean():.4f} nats  union8 {r[:, 2].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["act", "attn"])
    ap.add_argument("--kind", default="down")
    ap.add_argument("--rule", default="abs_topk")
    ap.add_argument("--scorer", default="oracle")
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--f", default="0.9,0.8,0.7,0.6,0.5")
    ap.add_argument("--budget", default="1024,2048,4096")
    ap.add_argument("--pool", default="head")
    ap.add_argument("--mode", default="row")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--dev", default="cpu")
    a = ap.parse_args()
    if a.layers is None:
        a.layers = ",".join(map(str, sp.LIN_LAYERS if a.what == "act" else sp.ATTN_LAYERS))
    t0 = time.time()
    (act if a.what == "act" else attn)(a)
    print(f"[{time.time() - t0:.1f}s]")
