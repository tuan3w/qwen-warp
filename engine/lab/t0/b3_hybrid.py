"""B3: per-layer-class statistics of the real linear inputs (are DeltaNet-layer inputs sparser than MLP ones?) -> results/b3/B3_act.md.  CPU, ~10 s.

For each (layer, kind), over the 128 decode rows x 6 docs: fraction of the row's squared L2 energy held by the top 10 / 25 / 50% of channels, excess kurtosis, and the overlap (Jaccard) of the
top-30% channel sets of two rows at distance 1 / 2 / 4 / 8 positions (how much a union over T=8 rows costs, and how stable a mask is across steps)."""
from pathlib import Path

import numpy as np
import torch

import sp

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b3")
DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]


def stats(x):
    e = x.pow(2)
    es = e.sort(1, descending=True).values
    cs = es.cumsum(1) / es.sum(1, keepdim=True)
    K = x.shape[1]
    top = [cs[:, int(K * p) - 1].mean().item() for p in (0.1, 0.25, 0.5)]
    xc = x - x.mean(1, keepdim=True)
    kurt = ((xc**4).mean(1) / (xc**2).mean(1) ** 2 - 3).mean().item()
    m = sp.topk_mask(x.abs(), int(0.3 * K))
    jac = []
    for d in (1, 2, 4, 8):
        a, b = m[:-d], m[d:]
        jac.append(((a & b).sum(1).float() / (a | b).sum(1).float()).mean().item())
    # union of 8 consecutive rows' top-30% masks: fraction of K covered
    G = x.shape[0] // 8
    un = m[: G * 8].reshape(G, 8, K).any(1).float().mean().item()
    return top, kurt, jac, un


def main():
    RES.mkdir(parents=True, exist_ok=True)
    L = ["# B3 activation statistics of the real linear inputs (128 decode rows x 6 docs)\n", "top10/25/50 = fraction of squared-L2 energy in the top 10/25/50% channels of a row; kurt = excess kurtosis; J1/J2/J4/J8 = Jaccard overlap of the top-30% channel sets of rows d positions apart; U8 = fraction of channels in the union of 8 rows' top-30% sets (T=1 keeps 0.30).\n",
         "| layer | class | kind | top10 | top25 | top50 | kurt | J1 | J2 | J4 | J8 | U8 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for li in sp.LIN_LAYERS:
        for kind in sp.KINDS:
            rs = [stats(sp.Trace(d).lin(li, kind, "dec")) for d in DOCS]
            top = np.mean([r[0] for r in rs], 0)
            kurt = np.mean([r[1] for r in rs])
            jac = np.mean([r[2] for r in rs], 0)
            un = np.mean([r[3] for r in rs])
            cl = "ATT" if li in sp.ATTN_LAYERS else "DN"
            L.append(f"| {li} | {cl} | {kind} | " + " | ".join(f"{v:.3f}" for v in top) + f" | {kurt:.1f} | " + " | ".join(f"{v:.2f}" for v in jac) + f" | {un:.2f} |")
            agg.setdefault((cl, kind), []).append(list(top) + [kurt] + list(jac) + [un])
    L.append("\n## class means\n")
    L.append("| class | kind | top10 | top25 | top50 | kurt | J1 | J2 | J4 | J8 | U8 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for (cl, kind), v in sorted(agg.items(), key=lambda t: (t[0][1], t[0][0])):
        m = np.mean(v, 0)
        L.append(f"| {cl} | {kind} | " + " | ".join(f"{x:.3f}" if i < 3 else (f"{x:.1f}" if i == 3 else f"{x:.2f}") for i, x in enumerate(m)) + " |")
    (RES / "B3_act.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[-20:]))


if __name__ == "__main__":
    main()
