"""B2: attention sparsity on real q / dequantised K,V (32k docs, one 128k layer).  GPU (torch, a few hundred MB), ~5 s per (doc, layer) for the full grid.

  python b2_attn.py [--docs a,b] [--layers 7,23] [--blocks 16,64] [--budgets 512,1024,...] [--quick]        -> results/b2/b2_raw.csv
  python b2_attn.py --summary                                                                              -> results/b2/B2.md

Setting: queries = the 128 true decode positions (rope'd q of all 24 heads, GQA group = 6 heads per KV head), keys = dequantised int4 history + exact window rows.  Every selection is ADDED to the exact
always-read set A = first 4 tokens (sink) + the 256 tokens before / at the query (ring); `budget` = total tokens read per (row, head) = selected + 260.  Quality per (row, head):
  err   = ||O_S - O|| / ||O||   O = dense softmax(q K^T / 16) V, O_S = softmax restricted to S u A (renormalised)
  kl    = -log(mass covered by S u A)  = KL(P || P_S)
  union8 = fraction of history blocks read by the 8-row group after the union over its 8 rows x 6 heads of the (per-head) selections (what the K/V stream really costs at T = 8)
Scorers (block score used for the top-k choice): oracle (sum of the exact probabilities in the block; upper reference), maxlogit (exact max logit), quest (min/max channel bound, unrotated K),
quest_rot (same in the WHT-rotated basis the cache stores), mean (q . mean key), samp4 (max q.k over 4 random keys of the block), prev1 / prev8 (oracle scores of the query 1 / 8 rows earlier = reuse
of the previous step's selection), xlayer (oracle scores of the previous traced attention layer).  Pooling: head (own selection per q head), kv (one selection per KV head: sum over its 6 heads of
softmax-normalised scores), row8 (one selection per 8 consecutive rows and KV head), tok (token-level oracle top-k = no block alignment)."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import sp

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b2")
DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]
RING, SINK = 256, 4
DEV = "cuda"


def wht(x):
    from kvdeq import hadamard

    return hadamard(x)


class Layer:
    """Everything for one (doc, layer) kept on the GPU."""

    def __init__(self, T, li, q=None, K=None, V=None):
        if q is None:
            q = T.q(li)  # [W,24,256]
            K, V = T.kv(li)
        self.q = q
        self.K, self.V = K, V  # [4,L,256]
        self.W = self.q.shape[0]
        self.L = K.shape[1]
        self.n = self.L - self.W
        t = torch.arange(self.L, device=self.q.device)
        p = (self.n + torch.arange(self.W, device=self.q.device)).view(-1, 1)
        self.valid = t.view(1, -1) <= p  # [W,L]
        self.always = (t.view(1, -1) < SINK) | (t.view(1, -1) > p - RING)
        self.always &= self.valid
        self.region = self.valid & ~self.always
        self.Kr = None

    def rotated(self):
        if self.Kr is None:
            self.Kr = wht(self.K)
            self.qr = wht(self.q)
        return self.Kr, self.qr


def pad_blocks(x, b, fill=0.0):
    L = x.shape[-1]
    nb = -(-L // b)
    if nb * b != L:
        x = torch.nn.functional.pad(x, (0, nb * b - L), value=fill)
    return x.reshape(*x.shape[:-1], nb, b)


def block_scores(Ly, g, name, b, lg, P, gen, prev=None):
    """-> [W,6,nb] float32 block scores for kv group g (6 q heads)."""
    W, L = Ly.W, Ly.L
    reg = Ly.region
    if name == "oracle":
        return pad_blocks(P * reg.unsqueeze(1), b).sum(-1)
    if name == "maxlogit":
        return pad_blocks(lg.masked_fill(~reg.unsqueeze(1), float("-inf")), b, float("-inf")).amax(-1)
    q = Ly.q[:, 6 * g : 6 * g + 6]  # [W,6,256]
    if name in ("quest", "quest_rot"):
        if name == "quest_rot":
            Kr, qr = Ly.rotated()
            K, q = Kr[g], qr[:, 6 * g : 6 * g + 6]
        else:
            K = Ly.K[g]
        Kb = pad_blocks(K.t(), b).permute(1, 0, 2)  # [nb,256,b]
        kmax, kmin = Kb.amax(-1), Kb.amin(-1)  # [nb,256]
        qp, qn = q.clamp_min(0), (-q).clamp_min(0)
        return (qp @ kmax.t() - qn @ kmin.t()) / 16
    if name == "mean":
        Kb = pad_blocks(Ly.K[g].t(), b).permute(1, 0, 2).mean(-1)
        return (q @ Kb.t()) / 16
    if name.startswith("samp"):
        s = int(name[4:])
        nb = -(-L // b)
        idx = (torch.arange(nb, device=q.device).view(-1, 1) * b + torch.randint(0, b, (nb, s), generator=gen, device=q.device)).clamp_max(L - 1)
        Ks = Ly.K[g][idx]  # [nb,s,256]
        return torch.einsum("jhd,nsd->jhns", q, Ks).amax(-1) / 16
    raise ValueError(name)


def to_prob(sc):
    """logit-like block scores -> per-head softmax over blocks (so heads / rows can be summed)."""
    return torch.softmax(sc.float(), -1)


def select(sc, bvalid, pool, nsel, gsize=8):
    """sc [W,6,nb] (higher = keep; -inf where invalid) -> bool [W,6,nb] selection, with pooling over heads / rows."""
    W, H, nb = sc.shape
    sc = sc.masked_fill(~bvalid.unsqueeze(1), float("-inf"))
    if pool == "head":
        pass
    else:
        pr = torch.nan_to_num(sc, neginf=-1e30)
        pr = to_prob(pr) if pool.endswith("_p") else pr
        if pool.startswith("kv") or pool.startswith("row8kv") or pool.startswith("row8"):
            sc = pr.sum(1, keepdim=True).expand(-1, H, -1)  # one selection per kv head
        if pool.startswith("row8"):
            sc = sc.reshape(W // gsize, gsize, H, nb).sum(1, keepdim=True).expand(-1, gsize, -1, -1).reshape(W, H, nb)
        sc = sc.masked_fill(~bvalid.unsqueeze(1), float("-inf"))
    k = min(nsel, nb)
    idx = sc.topk(k, dim=-1).indices
    m = torch.zeros(W, H, nb, dtype=torch.bool, device=sc.device)
    m.scatter_(-1, idx, True)
    return m & bvalid.unsqueeze(1)


def evaluate(Ly, g, lg, P, Od, M_blk, b, vg):
    """M_blk [W,6,nb] bool selection -> (err[W,6], kl[W,6], union8 fraction scalar)."""
    W, L = Ly.W, Ly.L
    M = M_blk.repeat_interleave(b, -1)[..., :L]
    allowed = (M | Ly.always.unsqueeze(1)) & Ly.valid.unsqueeze(1)
    cov = (P * allowed).sum(-1)
    Q = torch.softmax(lg.masked_fill(~allowed, float("-inf")), -1)
    O = torch.einsum("jht,td->jhd", Q, vg)
    err = (O - Od).norm(dim=-1) / Od.norm(dim=-1).clamp_min(1e-9)
    kl = -cov.clamp_min(1e-12).log()
    bv = pad_blocks(Ly.region.float(), b).amax(-1) > 0  # [W,nb]
    un = (M_blk.reshape(W // 8, 8, 6, -1).any(1).any(1)).sum(-1).float() / bv.reshape(W // 8, 8, -1).any(1).sum(-1).clamp_min(1)  # [W/8]
    return err, kl, un.mean().item()


def evaluate_bg(Ly, g, lg, P, Od, M_blk, b, vg, kmean, vmean):
    """Selected blocks exact + every UNSELECTED history block replaced by ONE summary entry (logit q.mean_k/16 weighted by its token count, value = mean v).  -> (err, kl, union8) like evaluate()."""
    W, L = Ly.W, Ly.L
    nb = M_blk.shape[-1]
    M = M_blk.repeat_interleave(b, -1)[..., :L]
    allowed = (M | Ly.always.unsqueeze(1)) & Ly.valid.unsqueeze(1)
    q = Ly.q[:, 6 * g : 6 * g + 6]
    cnt = pad_blocks(Ly.region.float(), b).sum(-1)  # [W,nb] region tokens per block
    unsel = (cnt.unsqueeze(1) > 0) & ~M_blk  # [W,6,nb]
    # tokens of an unselected block are NOT 'allowed' (region tokens only; always-tokens are allowed)
    bl = torch.einsum("jhd,nd->jhn", q, kmean) / 16
    bl = bl.masked_fill(~unsel, float("-inf"))
    m = torch.maximum(lg.masked_fill(~allowed, float("-inf")).amax(-1), bl.amax(-1)).unsqueeze(-1)
    e_tok = torch.exp(lg.masked_fill(~allowed, float("-inf")) - m)
    e_blk = cnt.unsqueeze(1) * torch.exp(bl - m)
    den = e_tok.sum(-1) + e_blk.sum(-1)
    num = torch.einsum("jht,td->jhd", e_tok, vg) + torch.einsum("jhn,nd->jhd", e_blk, vmean)
    O = num / den.unsqueeze(-1)
    err = (O - Od).norm(dim=-1) / Od.norm(dim=-1).clamp_min(1e-9)
    # KL(P||Q): Q_t = e_tok/den for allowed tokens, exp(bl_blk - m)/den for tokens of unselected blocks
    lq_blk = (bl - m - den.log().unsqueeze(-1)).repeat_interleave(b, -1)[..., :L]  # log q of a token in an unselected block
    lq_tok = lg - m - den.log().unsqueeze(-1)
    lq = torch.where(allowed, lq_tok, lq_blk)
    kl = (P * (P.clamp_min(1e-30).log() - lq)).masked_fill(~Ly.valid.unsqueeze(1), 0).sum(-1)
    bvu = M_blk.reshape(W // 8, 8, 6, -1).any(1).any(1).sum(-1).float() / 1.0
    return err, kl, float("nan")


def tok_select(P, region, nsel, pool="head"):
    """token-level oracle top-k per (row, head) (no block alignment); pool kv = one selection per KV head (sum over its 6 heads), row8 = one per KV head and 8 consecutive rows."""
    W, H, L = P.shape
    sc = (P * region.unsqueeze(1)).float()
    if pool in ("kv", "row8"):
        sc = sc.sum(1, keepdim=True).expand(-1, H, -1)
    if pool == "row8":
        sc = sc.reshape(W // 8, 8, H, L).sum(1, keepdim=True).expand(-1, 8, -1, -1).reshape(W, H, L)
    idx = sc.topk(nsel, dim=-1).indices
    m = torch.zeros(W, H, L, dtype=torch.bool, device=P.device)
    m.scatter_(-1, idx, True)
    return m & region.unsqueeze(1)


def run_layer(doc, li, T, blocks, budgets, rows, scorers, cur_orc, xl, gen, bg_scorers=()):
    Ly = Layer(T, li)
    W, L = Ly.W, Ly.L
    out = {}
    for g in range(4):
        q = Ly.q[:, 6 * g : 6 * g + 6]
        lg = torch.einsum("jhd,td->jht", q, Ly.K[g]) / 16
        lg = lg.masked_fill(~Ly.valid.unsqueeze(1), float("-inf"))
        P = torch.softmax(lg, -1)
        vg = Ly.V[g]
        Od = torch.einsum("jht,td->jhd", P, vg)
        # dense statistics
        ent = -(P * P.clamp_min(1e-30).log()).sum(-1)
        for bud in budgets:
            ns_tok = bud - (SINK + RING)
            # token-level oracle
            M = tok_select(P, Ly.region, ns_tok)
            allowed = (M | Ly.always.unsqueeze(1)) & Ly.valid.unsqueeze(1)
            cov = (P * allowed).sum(-1)
            Q = torch.softmax(lg.masked_fill(~allowed, float("-inf")), -1)
            O = torch.einsum("jht,td->jhd", Q, vg)
            err = (O - Od).norm(dim=-1) / Od.norm(dim=-1).clamp_min(1e-9)
            kl = -cov.clamp_min(1e-12).log()
            out.setdefault((li, "tok_oracle", "head", 1, bud), []).append((err, kl, float("nan")))
        for b in blocks:
            bv = pad_blocks(Ly.region.float(), b).amax(-1) > 0
            for name in scorers:
                if name in ("prev1", "prev8", "xlayer"):
                    continue
                sc = block_scores(Ly, g, name, b, lg, P, gen)
                if name == "oracle":
                    cur_orc[(b, g)] = sc
                for pool in rows:
                    for bud in budgets:
                        nsel = max(1, (bud - SINK - RING) // b)
                        scs = sc
                        M = select(scs, bv, pool if not (pool.endswith("_p") and name in ("oracle",)) else pool[:-2], nsel)
                        out.setdefault((li, name, pool, b, bud), []).append(evaluate(Ly, g, lg, P, Od, M, b, vg))
            kb = pad_blocks(Ly.K[g].t(), b).permute(1, 0, 2).mean(-1)  # [nb,256]
            vb = pad_blocks(Ly.V[g].t(), b).permute(1, 0, 2).mean(-1)
            for name in bg_scorers:
                sc = block_scores(Ly, g, name, b, lg, P, gen)
                for pool in ("head", "kv_p"):
                    for bud in budgets:
                        nsel = max(1, (bud - SINK - RING) // b)
                        M = select(sc, bv, pool[:-2] if (name == "oracle" and pool.endswith("_p")) else pool, nsel)
                        out.setdefault((li, name + "+bg", pool, b, bud), []).append(evaluate_bg(Ly, g, lg, P, Od, M, b, vg, kb, vb))
            # reuse of earlier selections: oracle scores of the row j-lag (same layer) / previous layer
            osc = cur_orc.get((b, g))
            for lag in (1, 8) if osc is not None else ():
                sc = torch.cat([osc[:lag].clone(), osc[:-lag]])  # row j uses row (j-lag)'s oracle scores (first rows: own)
                for bud in budgets:
                    nsel = max(1, (bud - SINK - RING) // b)
                    M = select(sc, bv, "head", nsel)
                    out.setdefault((li, f"prev{lag}", "head", b, bud), []).append(evaluate(Ly, g, lg, P, Od, M, b, vg))
            pl = xl.get((b, g)) if xl else None
            if pl is not None:
                for bud in budgets:
                    nsel = max(1, (bud - SINK - RING) // b)
                    M = select(pl, bv, "head", nsel)
                    out.setdefault((li, "xlayer", "head", b, bud), []).append(evaluate(Ly, g, lg, P, Od, M, b, vg))
        out.setdefault((li, "_stats", "ent", 0, 0), []).append((ent, ent, 0.0))
        del lg, P, Od
    return out, Ly


def reduce(out, doc):
    rows = []
    for (li, name, pool, b, bud), lst in out.items():
        err = torch.stack([e for e, _, _ in lst], 1).float()  # [W,4,6]
        kl = torch.stack([k for _, k, _ in lst], 1).float()
        un = float(np.mean([u for _, _, u in lst]))
        rows.append((doc, li, name, pool, b, bud, err.mean().item(), kl.mean().item(), torch.quantile(err.flatten(), 0.9).item(), torch.quantile(kl.flatten(), 0.9).item(), un))
    return rows


def main(docs, layers, blocks, budgets, scorers, pools, outname, bg_scorers=()):
    RES.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=DEV).manual_seed(0)
    allrows = []
    t0 = time.time()
    for doc in docs:
        T = sp.Trace(doc, DEV)
        xl = None
        for li in layers:
            if not T.has_kv(li):
                continue
            cur = {}
            out, Ly = run_layer(doc, li, T, blocks, budgets, pools, scorers, cur, xl, gen, bg_scorers)
            xl = cur  # next traced attention layer reuses this layer's oracle block scores
            ent = out.pop((li, "_stats", "ent", 0, 0))
            allrows += reduce(out, doc)
            e = torch.stack([x for x, _, _ in ent], 1).float()
            allrows.append((doc, li, "_stats", "entropy_nats", 0, 0, e.mean().item(), e.median().item(), 0, 0, 0))
            del Ly
            T._c.pop(("kv", li), None)
            torch.cuda.empty_cache()
            print(f"{doc} L{li} {time.time() - t0:.0f}s", flush=True)
    with open(RES / outname, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["doc", "layer", "scorer", "pool", "b", "budget", "err", "kl", "err_p90", "kl_p90", "union8"])
        w.writerows(allrows)
    print(f"wrote {len(allrows)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default=",".join(DOCS))
    ap.add_argument("--layers", default="7,23,39,63")
    ap.add_argument("--blocks", default="16,32,64,128")
    ap.add_argument("--budgets", default="512,1024,2048,4096,8192")
    ap.add_argument("--scorers", default="oracle,maxlogit,quest,quest_rot,mean,samp4")
    ap.add_argument("--pools", default="head,kv_p,row8_p")
    ap.add_argument("--bg", default="")
    ap.add_argument("--out", default="b2_raw.csv")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.blocks, a.budgets, a.scorers, a.pools, a.out = "64", "1024,4096", "oracle,quest", "head,kv_p", "b2_quick.csv"
    main(a.docs.split(","), [int(x) for x in a.layers.split(",")], [int(x) for x in a.blocks.split(",")], [int(x) for x in a.budgets.split(",")], a.scorers.split(","), a.pools.split(","), a.out, [x for x in a.bg.split(",") if x])
