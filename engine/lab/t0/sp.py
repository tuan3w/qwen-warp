"""T0 offline lab: real traces -> sparsity-rule evaluation in seconds.  Library (loaders + rules + metrics); CLI drivers: b1_act.py, b2_attn.py, b3_hybrid.py.

Data (written by traces/dump.py):  /media/fatman/data/sparselab/traces/<doc>/   weights cache (wcache.py): /media/fatman/data/sparselab/t0/w_L{li}.npz

  from sp import *
  T = Trace("tr32_code_cuda")                     # lazy loader
  x = T.lin(23, "down", "dec")                    # float32 [128][17408]: the real down_proj inputs of 128 consecutive decode positions of the model's own greedy continuation
  W = Wcache(23)["down"]                          # .w fp16 [384][K] random output rows of the dequantised W4, .cn fp32 [K] column norms over ALL rows
  e = rel_err(x, mask(x, rule, f, b, ...), W)     # mean per-row relative L2 output error

Conventions: 'rows' = tokens; K-block b in {8,16,32,64,128} columns (8 = one uint32 word of nibbles, 128 = one scale group).  Bytes fraction of one weight row = (kept_cols * 0.5 + touched_scale_groups * 2) /
(K * 0.5 + K/128 * 2) (activation bytes ignored: they are <= 0.3% of the weight bytes at M <= 8)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "traces"))
TR = Path("/media/fatman/data/sparselab/traces")
WC = Path("/media/fatman/data/sparselab/t0")
KINDS = ("qkv", "o", "gu", "down")
KDIM = dict(qkv=5120, o=6144, gu=5120, down=17408)
ATTN_LAYERS = (7, 23, 39, 63)
DN_LAYERS = (2, 14, 30, 46)
LIN_LAYERS = (2, 7, 14, 23, 30, 39, 46, 63)
DEV = "cpu"


def _bf(a):
    return torch.from_numpy(np.ascontiguousarray(a)).view(torch.bfloat16)


class Trace:
    def __init__(self, doc, dev=None):
        self.doc, self.dir, self.dev = doc, TR / doc, dev or DEV
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self._c = {}

    def lin(self, li, kind, part="dec"):
        """float32 [R][K] linear input rows ('dec': the 128 true decode positions; 'doc': ~32 random document positions of the prefill)."""
        k = (li, kind, part)
        if k not in self._c:
            p = self.dir / f"lin_L{li}_{kind}_{part}.npy"
            self._c[k] = _bf(np.load(p)).float().to(self.dev) if p.exists() else None
        return self._c[k]

    def q(self, li):
        """float32 [W][24][256] rope'd q of the window rows."""
        k = ("q", li)
        if k not in self._c:
            self._c[k] = _bf(np.load(self.dir / f"q_L{li}.npy")).float().to(self.dev)
        return self._c[k]

    def has_kv(self, li):
        return (self.dir / f"kv_L{li}.npz").exists()

    def kv(self, li):
        """(K, V) float32 [4][Ltot][256] unrotated: dequantised history (exact sink rows) followed by the exact window rows (Ltot = n_hist + W); K is post-rope."""
        k = ("kv", li)
        if k not in self._c:
            from kvdeq import dequant_rows

            z = np.load(self.dir / f"kv_L{li}.npz")
            out = []
            for iv, w in ((0, "wk"), (1, "wv")):
                per = []
                for h in range(4):
                    hist = dequant_rows(z["codes"][iv, h], z["meta"][iv, h], z["sink"][iv, h], dev=self.dev)
                    win = _bf(z[w][:, h]).float().to(self.dev)
                    per.append(torch.cat([hist, win]))
                out.append(torch.stack(per))
            self._c[k] = tuple(out)
        return self._c[k]

    def logits_ref(self):
        z = np.load(self.dir / "logits_ref.npz")
        return z["topv"], z["topi"], z["lse"]


class Wcache:
    _c = {}

    def __new__(cls, li):
        if li not in cls._c:
            z = np.load(WC / f"w_L{li}.npz")
            o = {}
            for k in KINDS:
                o[k] = type("W", (), dict(w=torch.from_numpy(z[f"{k}_w"]).float(), cn=torch.from_numpy(z[f"{k}_cn"]).float(), nrows=int(z[f"{k}_nrows"])))
            cls._c[li] = o
        return cls._c[li]


# ------------------------------------------------------------------------------------------------ rules
def bytes_frac(mask, b):
    """mask bool [..., K] (kept columns; any pattern) -> weight-byte fraction read per row, with b-column alignment (a block is read iff any of its columns is kept) and 128-column scale groups
    touched.  Returns a float tensor over leading dims."""
    K = mask.shape[-1]
    blk = mask.reshape(*mask.shape[:-1], K // b, b).any(-1)  # [..., K/b]
    kept_cols = blk.sum(-1) * b
    g = blk.reshape(*blk.shape[:-1], K // 128, 128 // b).any(-1) if b <= 128 else blk
    return (kept_cols * 0.5 + g.sum(-1) * 2.0) / (K * 0.5 + K / 128 * 2.0)


def block_score(s, b, how):
    """s float [R][K] per-column score (>=0) -> [R][K/b] block score: 'max' | 'sum' (of s) | 'l2' (sqrt of sum of squares)."""
    R, K = s.shape
    v = s.reshape(R, K // b, b)
    return v.amax(-1) if how == "max" else (v.sum(-1) if how == "sum" else v.pow(2).sum(-1).sqrt())


def topk_mask(score, k):
    """bool [R][n]: the k highest scores per row (k int or per-row tensor)."""
    R, n = score.shape
    if isinstance(k, int):
        idx = score.topk(max(1, min(n, k)), dim=1).indices
        m = torch.zeros(R, n, dtype=torch.bool, device=score.device)
        m.scatter_(1, idx, True)
        return m
    order = score.argsort(1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(n, device=score.device).expand(R, n))
    return rank < k.view(-1, 1)


def make_mask(x, rule, f, b=1, cn=None, calib=None, group=1):
    """Keep-mask bool [R][K] of fraction ~f of the K columns (in blocks of b) per row for:
      abs_topk   TEAL oracle: per-row top-k of |x|                      (b > 1: block score = L2 norm of |x| in the block)
      wina_topk  weight-informed: top-k of |x| * cn   (cn = column norm of W)   (b > 1: L2 of |x| cn in the block)
      max_topk   b > 1: block score = max |x|
      abs_thr    TEAL practical: ONE threshold on the block score (global quantile over `calib` rows -> f kept on average, per-row count varies)
      static     per-column / per-block static mask from `calib` rows' RMS(x) * cn (same mask for every row)
    `group` > 1 pools rows (T = group consecutive rows share the mask): union / max-over-rows scores, see union_mask()."""
    R, K = x.shape
    ax = x.abs()
    base = ax * cn if (rule in ("wina_topk", "static") and cn is not None) else ax
    if rule == "static":
        c = calib.pow(2).mean(0, keepdim=True).sqrt()
        sc = (c * cn if cn is not None else c)
        bs = block_score(sc, b, "l2") if b > 1 else sc
        keep = max(1, int(round(f * K / b)))
        m = topk_mask(bs, keep).expand(R, -1)
        return m.repeat_interleave(b, 1) if b > 1 else m
    if rule == "abs_thr":
        sc = block_score(ax, b, "l2") if b > 1 else ax
        csc = block_score(calib.abs(), b, "l2") if b > 1 else calib.abs()
        thr = torch.quantile(csc.flatten()[:: max(1, csc.numel() // 2_000_000)], 1 - f)
        m = sc >= thr
        return m.repeat_interleave(b, 1) if b > 1 else m
    if b == 1:
        return topk_mask(base, max(1, int(round(f * K))))
    how = "max" if rule == "max_topk" else "l2"
    bs = block_score(base, b, how)
    m = topk_mask(bs, max(1, int(round(f * K / b))))
    return m.repeat_interleave(b, 1)


def union_mask(x, rule, f, b, cn, group=8, mode="union", calib=None):
    """T = `group` consecutive rows verified together.  mode 'union': each row picks its own top-f blocks, all rows of the group use the UNION (bytes = |union|).  mode 'max': one shared mask = top-k
    blocks of the max-over-rows block score (fraction f of K).  Returns a mask [R][K] (identical inside each group)."""
    R, K = x.shape
    G = R // group
    xs = x[: G * group]
    if mode == "max":
        base = xs.abs() * cn if rule == "wina_topk" else xs.abs()
        bs = block_score(base, b, "l2" if rule != "max_topk" else "max") if b > 1 else base
        bs = bs.reshape(G, group, -1).amax(1)
        m = topk_mask(bs, max(1, int(round(f * K / b))))
        m = m.repeat_interleave(b, 1) if b > 1 else m
        return m.repeat_interleave(group, 0)
    m = make_mask(xs, rule, f, b, cn, calib).reshape(G, group, K).any(1)
    return m.repeat_interleave(group, 0)


def rel_err(x, m, W, pooled=False):
    """relative L2 output error of y = x W^T when the columns outside m are zeroed in x (over the W row subset).  Per-row mean (default) or pooled Frobenius."""
    xs = x[: m.shape[0]]
    y = xs @ W.w.t()
    d = (xs * m) @ W.w.t() - y
    if pooled:
        return (d.norm() / y.norm()).item()
    return (d.norm(dim=1) / y.norm(dim=1).clamp_min(1e-9)).mean().item()


def crossing(fs, errs, bytes_, thr):
    """smallest byte fraction at which err <= thr, by linear interpolation along the curve (sorted by bytes); 1.0 if never reached below 1."""
    o = np.argsort(bytes_)
    bb, ee = np.asarray(bytes_)[o], np.asarray(errs)[o]
    ok = np.where(ee <= thr)[0]
    if len(ok) == 0:
        return float("nan")
    i = ok[0]
    if i == 0:
        return float(bb[0])
    # interpolate between i-1 (err > thr) and i
    t = (thr - ee[i - 1]) / (ee[i] - ee[i - 1]) if ee[i] != ee[i - 1] else 1.0
    return float(bb[i - 1] + t * (bb[i] - bb[i - 1]))
