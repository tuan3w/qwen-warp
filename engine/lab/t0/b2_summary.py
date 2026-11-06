"""B2 / B3 summary tables from results/b2/b2_raw.csv (+ b2_128k.csv) -> results/b2/B2.md.   python b2_summary.py"""
import csv
import collections
from pathlib import Path

import numpy as np

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b2")


def load(name):
    rows = []
    with open(RES / name) as fh:
        for r in csv.DictReader(fh):
            rows.append(dict(doc=r["doc"], layer=int(r["layer"]), scorer=r["scorer"], pool=r["pool"], b=int(r["b"]), bud=int(r["budget"]), err=float(r["err"]), kl=float(r["kl"]), err90=float(r["err_p90"]), kl90=float(r["kl_p90"]), un=float(r["union8"])))
    return rows


def agg(rows, key, val):
    d = collections.defaultdict(list)
    for r in rows:
        d[key(r)].append(r[val])
    return {k: float(np.mean(v)) for k, v in d.items()}


def table(lines, title, hdr, rows):
    lines.append(f"\n### {title}\n")
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")


def main():
    L = ["# B2 / B3: attention sparsity on real q, K, V (32k docs: 6 docs x layers 7/23/39/63; 128k: 1 doc x layer 63)\n",
         "err = relative L2 error of the attention OUTPUT per (row, head); kl = -log(covered probability mass) = KL(P_dense || P_sparse) of the attention distribution (nats); both include the exact sink 4 + ring 256.",
         "budget = total tokens read per (row, head). union8 = fraction of history blocks actually read by an 8-row verify group (union over its rows and the 6 heads of the KV head).\n"]
    R = load("b2_raw.csv")
    st = [r for r in R if r["scorer"] == "_stats"]
    R = [r for r in R if r["scorer"] != "_stats"]
    BUD = sorted({r["bud"] for r in R})
    # --- B3 entropy by depth
    ent = agg(st, lambda r: r["layer"], "err")
    med = agg(st, lambda r: r["layer"], "kl")
    table(L, "B3: dense attention entropy by layer (nats, mean over 128 rows x 24 heads x 6 docs; max for 32k keys = 10.4)", ["layer", "mean entropy", "median"], [[str(k), f"{ent[k]:.2f}", f"{med[k]:.2f}"] for k in sorted(ent)])
    # --- token oracle per layer
    tok = [r for r in R if r["scorer"] == "tok_oracle"]
    for val, nm in (("err", "output rel. error"), ("kl", "KL (nats) of attention distribution"), ("kl90", "KL p90 over (row, head)")):
        a = agg(tok, lambda r: (r["layer"], r["bud"]), val)
        table(L, f"token-level ORACLE top-k (no block alignment), {nm}", ["layer"] + [str(b) for b in BUD], [[str(l)] + [f"{a[(l, b)]:.4f}" for b in BUD] for l in sorted({k[0] for k in a})])
    # --- block scorers b=64, mean over docs and layers
    for val, nm in (("err", "output rel. error"), ("kl", "KL (nats)"), ("un", "union8 fraction of blocks read at T=8")):
        a = agg([r for r in R if r["scorer"] != "tok_oracle" and r["b"] == 64], lambda r: (r["scorer"], r["pool"], r["bud"]), val)
        keys = sorted({(k[0], k[1]) for k in a})
        table(L, f"block 64, {nm}, mean over docs and layers", ["scorer", "pool"] + [str(b) for b in BUD], [[s, p] + [f"{a[(s, p, b)]:.4f}" if (s, p, b) in a else "-" for b in BUD] for s, p in keys])
    # --- block size
    for sc in ("oracle", "quest"):
        a = agg([r for r in R if r["scorer"] == sc and r["pool"] == "head"], lambda r: (r["b"], r["bud"]), "kl")
        bs = sorted({k[0] for k in a})
        table(L, f"block size effect, scorer {sc}, pool head: KL (nats)", ["block"] + [str(b) for b in BUD], [[str(b0)] + [f"{a[(b0, b)]:.4f}" for b in BUD] for b0 in bs])
    # --- per layer for quest/oracle b=64 budget 2048
    for sc in ("oracle", "quest"):
        a = agg([r for r in R if r["scorer"] == sc and r["pool"] == "head" and r["b"] == 64], lambda r: (r["layer"], r["bud"]), "kl")
        table(L, f"per layer KL, scorer {sc}, b=64, pool head", ["layer"] + [str(b) for b in BUD], [[str(l)] + [f"{a[(l, b)]:.4f}" for b in BUD] for l in sorted({k[0] for k in a})])
    # --- per doc spread tok oracle 2048
    a = agg([r for r in tok if r["bud"] == 2048], lambda r: r["doc"], "kl")
    table(L, "token oracle KL at budget 2048 per document (mean over layers)", ["doc", "kl"], [[d, f"{v:.4f}"] for d, v in sorted(a.items())])
    # --- 128k
    try:
        R2 = [r for r in load("b2_128k.csv") if r["scorer"] != "_stats"]
        B2 = sorted({r["bud"] for r in R2})
        for val, nm in (("kl", "KL (nats)"), ("err", "output rel. error"), ("un", "union8")):
            a = agg([r for r in R2 if r["scorer"] != "tok_oracle"], lambda r: (r["scorer"], r["pool"], r["b"], r["bud"]), val)
            keys = sorted({k[:3] for k in a})
            table(L, f"128k doc, layer 63: block scorers, {nm}", ["scorer", "pool", "b"] + [str(b) for b in B2], [[s, p, str(b0)] + [f"{a[(s, p, b0, b)]:.4f}" if (s, p, b0, b) in a else "-" for b in B2] for s, p, b0 in keys])
        a = agg([r for r in R2 if r["scorer"] == "tok_oracle"], lambda r: r["bud"], "kl")
        table(L, "128k doc, layer 63: token oracle KL", ["budget"] + [str(b) for b in B2], [["kl"] + [f"{a[b]:.4f}" for b in B2]])
    except FileNotFoundError:
        pass
    (RES / "B2.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
