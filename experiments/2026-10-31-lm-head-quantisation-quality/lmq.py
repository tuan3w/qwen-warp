"""lm-head-quantisation-quality: corpus builder, quantisers, metrics.  Lab venv (~/llm-lab/venv/bin/python), torch cu130.

Pure library: run.py drives the stages (corpus -> collect (engine, GPU lease) -> analyze (torch, GPU lease) -> report).
"""
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"  # gitignored (**/runs/), on the data disk
RESULTS = HERE / "results"
MODEL = Path.home() / "models" / "swift-1.5-int4"
V, H = 248320, 5120
SEED = 20261031
SKIP = 16  # first positions of a doc (no context) are not evaluated
DOC_TOKENS = 400

CHAT_PROMPTS = [
    ("code", "Write a Rust function that parses a semantic version string like '1.2.3-beta.4' into a struct and add three unit tests."),
    ("code", "My Python script raises 'RuntimeError: dictionary changed size during iteration' when I remove stale cache entries. Show the bug pattern and two fixes."),
    ("math", "A bag has 5 red and 7 blue marbles. Two are drawn without replacement. Compute the probability they have different colours, step by step."),
    ("math", "Show that the sum of two odd squares is never a perfect square when both are positive and coprime to 2, or give a counterexample. Reason carefully."),
    ("prose", "Write a short, atmospheric scene in which a retired cartographer realises the map she drew fifty years ago has been quietly correcting itself."),
    ("prose", "Explain to a curious teenager why the sky is blue but sunsets are red, using an everyday analogy."),
    ("agent", "You can call tools by emitting JSON like {\"tool\": name, \"args\": {...}}. Available: search(query), read_file(path), run(cmd). The user wants to know which Python files in ./src import torch. Plan and emit the first tool call."),
    ("agent", "Extract the fields name, date and amount from this text as JSON: 'On 12 March 2025 Priya Natarajan paid 1,240.50 EUR to Lumen Labs for the quarterly licence.'"),
    ("chat", "Give me a balanced pros-and-cons list for moving a three-person startup from a monolith to microservices, ending with a one-sentence recommendation."),
    ("chat", "Translate into French and German: 'The meeting has been moved to Thursday afternoon; please bring the revised budget.'"),
    ("code", "Explain what this SQL does and how to make it faster: SELECT user_id, COUNT(*) FROM events WHERE ts > now() - interval '7 days' GROUP BY user_id HAVING COUNT(*) > 100 ORDER BY 2 DESC;"),
    ("math", "Derive the closed form of the sum 1/(1*2) + 1/(2*3) + ... + 1/(n(n+1)) and then use it to evaluate the infinite series."),
]


def _strip_frontmatter(t):
    if t.startswith("---"):
        j = t.find("\n---", 3)
        if j > 0:
            return t[j + 4 :]
    return t


def _window(rng, text, nchar, lo_frac=0.0, hi_frac=0.7, min_off=0):
    """random line-aligned character window of ~nchar chars starting at an offset in [lo, hi] of the text."""
    lo = max(int(len(text) * lo_frac), min_off)
    hi = max(lo + 1, min(int(len(text) * hi_frac), len(text) - nchar))
    s = rng.randint(lo, max(lo, hi))
    nl = text.find("\n", s)
    s = nl + 1 if 0 <= nl < s + 400 else s
    return text[s : s + nchar]


def build_docs(tok):
    """Returns list of dict(cat, name, ids) for the natural-text docs (code / prose / math); chat docs are built in collect (greedy continuation)."""
    rng = random.Random(SEED)
    docs = []

    def add(cat, name, text, nchar_tok=DOC_TOKENS):
        ids = tok(text, add_special_tokens=False)["input_ids"][:nchar_tok]
        if len(ids) >= nchar_tok - 20:
            docs.append(dict(cat=cat, name=name, ids=ids))

    # code: vendor/ files > 120 KB, window past the first 100 KB (the vocab-corpus per-file cap of mtp-draft-head-ablation) where possible
    by_ext = {}
    for p in sorted((REPO / "vendor").rglob("*")):
        if p.suffix in (".py", ".c", ".h", ".cu", ".cuh", ".cpp", ".rs", ".ts") and p.is_file() and not p.is_symlink():
            try:
                sz = p.stat().st_size
            except OSError:
                continue
            if sz > 120_000:
                by_ext.setdefault({".h": ".c", ".cuh": ".cu", ".cpp": ".c"}.get(p.suffix, p.suffix), []).append(p)
    quota = {".py": 6, ".c": 3, ".cu": 2, ".rs": 3, ".ts": 3}
    for ext, n in quota.items():
        files = by_ext.get(ext, [])
        rng.shuffle(files)
        for p in files[:n]:
            text = p.read_text(errors="ignore")
            add("code", str(p.relative_to(REPO)), _window(rng, text, 2600, 0.0, 0.9, min_off=100_000))
    # prose: notes/, sources/, papers/ markdown (not topics/: that is the vocab corpus)
    pool = [p for d in ("notes", "sources", "papers") for p in sorted((REPO / d).rglob("*.md")) if p.stat().st_size > 3500]
    rng.shuffle(pool)
    n = 0
    for p in pool:
        if n >= 12:
            break
        text = _strip_frontmatter(p.read_text(errors="ignore"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) < 2200:
            continue
        before = len(docs)
        add("prose", str(p.relative_to(REPO)), _window(rng, text, 2300, 0.0, 0.8))
        n += len(docs) - before
    # math / CoT-like: experiments READMEs (results sections, derivations, plans)
    pool = sorted((REPO / "experiments").glob("*/README.md"))
    rng.shuffle(pool)
    n = 0
    for p in pool:
        if n >= 12:
            break
        text = _strip_frontmatter(p.read_text(errors="ignore"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) < 2200:
            continue
        before = len(docs)
        add("math", str(p.relative_to(REPO)), _window(rng, text, 2300, 0.2, 0.8))
        n += len(docs) - before
    return docs


def split_calib(docs):
    """every 5th doc (by order) is a calibration doc (activation statistics for the act-weighted clip search); the rest are evaluated."""
    for i, d in enumerate(docs):
        d["role"] = "calib" if i % 5 == 2 else "eval"
    return docs


# --------------------------------------------------------------------------------------------------- quantisers (return codes + scales; dequant on the fly)
class QHead:
    """Quantised head: q int8 [V,H] (codes), scale fp32 [V] (per-row) or bf16 [V,H/128] (g128).  dequant(r0,r1) -> fp32 [r1-r0,H]."""

    def __init__(self, name, bits, group, q, scale):
        self.name, self.bits, self.group, self.q, self.scale = name, bits, group, q, scale

    def dequant(self, r0, r1):
        q = self.q[r0:r1].float()
        if self.group == 0:
            return q * self.scale[r0:r1, None]
        s = self.scale[r0:r1].float()  # [n, H/g]
        return (q.view(r1 - r0, H // self.group, self.group) * s[:, :, None]).view(r1 - r0, H)

    def bytes(self):
        if self.group == 0:
            return V * H * self.bits / 8 + V * 4
        return V * H * self.bits / 8 + V * (H // self.group) * 2


def _q_row(w, bits, clip_search):
    """w fp32 [n,H] -> codes int8, scale fp32 [n]; symmetric, levels +-(2^(bits-1)-1)."""
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().amax(1).clamp_min(1e-12)
    if not clip_search:
        s = amax / qmax
    else:
        best_err = torch.full_like(amax, float("inf"))
        best_s = amax / qmax
        for r in np.linspace(0.5, 1.0, 26):
            s = amax * float(r) / qmax
            q = torch.round(w / s[:, None]).clamp_(-qmax, qmax)
            err = ((w - q * s[:, None]) ** 2).sum(1)
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, s, best_s)
        s = best_s
    q = torch.round(w / s[:, None]).clamp_(-qmax, qmax).to(torch.int8)
    return q, s


def _q_g128(w, mode, colw=None):
    """w fp32 [n,H] -> codes int8 in [-8,7], scale bf16 [n,H/128].  scale = r*amax/7 (bf16-rounded BEFORE coding).  mode: rtn | clip | awclip (colw [H] weights the group error)."""
    n = w.shape[0]
    g = w.view(n, H // 128, 128)
    amax = g.abs().amax(2).clamp_min(1e-12)  # [n,G]
    wcol = None if colw is None else colw.view(1, H // 128, 128)

    def code(s):
        sb = s.bfloat16().float()
        q = torch.round(g / sb[:, :, None]).clamp_(-8, 7)
        return q, sb

    if mode == "rtn":
        q, sb = code(amax / 7)
    else:
        best_err = torch.full_like(amax, float("inf"))
        best_s = amax / 7
        for r in np.linspace(0.5, 1.1, 31):
            q, sb = code(amax * float(r) / 7)
            d = (g - q * sb[:, :, None]) ** 2
            if mode == "awclip":
                d = d * wcol
            err = d.sum(2)
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best_s = torch.where(m, sb, best_s)
        q, sb = code(best_s)
    return q.to(torch.int8).view(n, H), sb.bfloat16()


def _q_g128_int8(w):
    """int8 with a bf16 scale per 128 columns (post-hoc arm): scale = amax/127 (bf16-rounded first), codes in [-127,127]."""
    n = w.shape[0]
    g = w.view(n, H // 128, 128)
    sb = (g.abs().amax(2).clamp_min(1e-12) / 127).bfloat16().float()
    q = torch.round(g / sb[:, :, None]).clamp_(-127, 127)
    return q.to(torch.int8).view(n, H), sb.bfloat16()


ARMS = {
    "int8_rtn": dict(bits=8, group=0, kind="row", clip=False),
    "int8_clip": dict(bits=8, group=0, kind="row", clip=True),
    "int6_rtn": dict(bits=6, group=0, kind="row", clip=False),
    "int8_g128_rtn": dict(bits=8, group=128, kind="g8"),
    "int4_g128_rtn": dict(bits=4, group=128, kind="g", mode="rtn"),
    "int4_g128_clip": dict(bits=4, group=128, kind="g", mode="clip"),
    "int4_g128_awclip": dict(bits=4, group=128, kind="g", mode="awclip"),
}


def quantise(W, arm, colw=None, block=4096):
    """W bf16 [V,H] on the GPU -> QHead (built block by block)."""
    cfg = ARMS[arm]
    q = torch.empty(V, H, dtype=torch.int8, device=W.device)
    if cfg["group"] == 0:
        sc = torch.empty(V, dtype=torch.float32, device=W.device)
    else:
        sc = torch.empty(V, H // 128, dtype=torch.bfloat16, device=W.device)
    for r0 in range(0, V, block):
        r1 = min(V, r0 + block)
        w = W[r0:r1].float()
        if cfg["kind"] == "row":
            q[r0:r1], sc[r0:r1] = _q_row(w, cfg["bits"], cfg["clip"])
        elif cfg["kind"] == "g8":
            q[r0:r1], sc[r0:r1] = _q_g128_int8(w)
        else:
            q[r0:r1], sc[r0:r1] = _q_g128(w, cfg["mode"], colw)
    return QHead(arm, cfg["bits"], cfg["group"], q, sc)


def weight_err(W, qh, block=8192):
    """relative L2 of the dequantised weights vs bf16 (whole matrix)."""
    num = den = 0.0
    for r0 in range(0, V, block):
        r1 = min(V, r0 + block)
        w = W[r0:r1].float()
        num += float(((qh.dequant(r0, r1) - w) ** 2).sum())
        den += float((w**2).sum())
    return (num / den) ** 0.5


# --------------------------------------------------------------------------------------------------- logits + metrics
@torch.no_grad()
def logits_from(X, W=None, qh=None, block=8192):
    """X fp32 [C,H] (exact bf16 values) -> fp32 logits [C,V]; weights bf16 W (reference) or QHead; plain fp32 matmul (no TF32), fp32 accumulate."""
    out = torch.empty(X.shape[0], V, dtype=torch.float32, device=X.device)
    for r0 in range(0, V, block):
        r1 = min(V, r0 + block)
        wb = W[r0:r1].float() if qh is None else qh.dequant(r0, r1)
        out[:, r0:r1] = X @ wb.T
    return out


def _filtered(L, temp, k=20, p=0.95):
    """vLLM order: temperature, top-k, then top-p on the renormalised top-k.  Returns (probs [C,k] sorted desc, ids [C,k])."""
    v, i = torch.topk(L / temp, k, dim=1)
    pr = torch.softmax(v, 1)
    cum = torch.cumsum(pr, 1)
    keep = (cum - pr) < p
    pr = pr * keep
    pr = pr / pr.sum(1, keepdim=True)
    return pr, i


def _tv_filtered(Lr, Lq, temp):
    pr, ir = _filtered(Lr, temp)
    pq, iq = _filtered(Lq, temp)
    eq = (ir[:, :, None] == iq[:, None, :]).float()  # [C,20,20]
    pq_at_r = (eq * pq[:, None, :]).sum(2)  # q's prob at ref's ids
    in_r = eq.sum(1)  # [C,20]: q id j present in ref set
    q_only = (pq * (1 - in_r)).sum(1)
    return 0.5 * ((pr - pq_at_r).abs().sum(1) + q_only)


@torch.no_grad()
def metrics(Lr, Lq, tgt=None):
    """Per-position metrics of quantised logits Lq vs reference logits Lr (both fp32 [C,V]).  tgt: int64 [C] next-token ids or None."""
    lpr = torch.log_softmax(Lr, 1)
    lpq = torch.log_softmax(Lq, 1)
    pr = lpr.exp()
    pq = lpq.exp()
    ar, aq = Lr.argmax(1), Lq.argmax(1)
    out = {"top1": (ar == aq).float()}
    out["kl_full"] = (pr * (lpr - lpq)).sum(1)
    t20 = torch.topk(Lr, 20, 1).indices
    gr, gq = torch.gather(Lr, 1, t20), torch.gather(Lq, 1, t20)
    p20r, p20q = torch.softmax(gr, 1), torch.softmax(gq, 1)
    out["kl_top20"] = (p20r * (torch.log(p20r) - torch.log(p20q))).sum(1)
    out["mass_top20"] = 0.5 * (torch.gather(pr, 1, t20) - torch.gather(pq, 1, t20)).abs().sum(1)  # |dp| over the ref top-20 ids (half = comparable to a TV)
    out["tv_full"] = 0.5 * (pr - pq).abs().sum(1)
    out["tv_k20p95_T1"] = _tv_filtered(Lr, Lq, 1.0)
    out["tv_k20p95_T07"] = _tv_filtered(Lr, Lq, 0.7)
    top2 = torch.topk(Lr, 2, 1).values
    out["ref_margin"] = top2[:, 0] - top2[:, 1]
    if tgt is not None:
        out["dnll"] = torch.gather(lpr, 1, tgt[:, None])[:, 0] - torch.gather(lpq, 1, tgt[:, None])[:, 0]  # nll_q - nll_ref  (> 0 = worse)
        out["nll_ref"] = -torch.gather(lpr, 1, tgt[:, None])[:, 0]
        out["tgt_is_ref_top1"] = (ar == tgt).float()
    out["_aq"] = aq.float()
    out["_ar"] = ar.float()
    return out


def rescore(Lr, Lq, M):
    """exact rescoring of the quantised head's top-M candidates: those M logits are replaced by the reference (bf16-weight) logits, the rest keep the quantised value."""
    idx = torch.topk(Lq, M, 1).indices
    out = Lq.clone()
    out.scatter_(1, idx, torch.gather(Lr, 1, idx))
    return out


# --------------------------------------------------------------------------------------------------- statistics
def doc_bootstrap(vals, doc, stat="mean", n=2000, seed=0, mask=None):
    """cluster (document) bootstrap CI of the mean of `vals` (np [P]); returns (mean, lo, hi)."""
    vals = np.asarray(vals, dtype=np.float64)
    doc = np.asarray(doc)
    if mask is not None:
        vals, doc = vals[mask], doc[mask]
    ud = np.unique(doc)
    s = np.array([vals[doc == d].sum() for d in ud])
    c = np.array([(doc == d).sum() for d in ud], dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ud), size=(n, len(ud)))
    est = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))


def block_stats(aq, ar, doc, pidx):
    """verify-block proxy over aligned 4-row blocks (pos_in_doc // 4) fully inside the evaluated set.
    Returns dict: P(any argmax change in the 4 rows), P(change in rows 0..2) and mean accepted length loss for an ideal drafter (drafts = reference argmax chain):
    accepted length = 1 + number of leading rows (of the first 3) where quantised argmax == reference argmax ... reported as E[len_ref=4] - E[len_q]."""
    key = {}
    for i in range(len(doc)):
        key.setdefault((int(doc[i]), int(pidx[i]) // 4), []).append(i)
    any4 = any3 = 0
    lost = 0.0
    nb = 0
    for (_, _), ix in key.items():
        if len(ix) != 4:
            continue
        ix = sorted(ix, key=lambda j: pidx[j])
        eq = [aq[j] == ar[j] for j in ix]
        nb += 1
        any4 += not all(eq)
        any3 += not all(eq[:3])
        lead = 0
        for e in eq[:3]:
            if not e:
                break
            lead += 1
        lost += 3 - lead  # accepted drafts of an ideal drafter: 3 if no flip in rows 0..2, else rows before the first flip
    return dict(blocks=nb, p_any4=any4 / max(nb, 1), p_any_rows012=any3 / max(nb, 1), accepted_drafts_lost=lost / max(nb, 1))
