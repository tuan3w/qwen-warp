"""B1e: how stable is the keep-set from one decode row to the next?  Mask for row t built from the RMS of the previous N decode rows of the SAME stream (N = 1, 8, 32 rows; row 0..N-1 skipped)
vs the per-token oracle top-k of row t.  Mean relative output error over the 8 traced layers x 6 docs.  CPU, ~30 s.   -> results/b1/B1e.md"""
from pathlib import Path

import numpy as np
import torch

import sp

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b1")
DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]
sp.DEV = "cpu"


def main():
    L = ["# B1e: previous-rows keep-set vs per-token oracle (b=1, f in 0.9/0.7/0.5), mean per-row relative output error %; rows 32..127 of the decode window\n", "| kind | f | per-token top-k | prev 1 row | prev 8 rows RMS | prev 32 rows RMS | random mask |", "|---|---|---|---|---|---|---|"]
    T = {d: sp.Trace(d, "cpu") for d in DOCS}
    for kind in sp.KINDS:
        for f in (0.9, 0.7, 0.5):
            acc = {k: [] for k in ("tok", 1, 8, 32, "rnd")}
            for li in sp.LIN_LAYERS:
                w = sp.Wcache(li)[kind].w
                for d in DOCS:
                    x = T[d].lin(li, kind, "dec")
                    rows = range(32, 128)
                    y = x[32:] @ w.t()
                    yn = y.norm(dim=1)
                    K = x.shape[1]
                    k = int(round(f * K))
                    masks = {"tok": sp.topk_mask(x[32:].abs(), k)}
                    for N in (1, 8, 32):
                        st = torch.stack([x[t - N : t].pow(2).mean(0).sqrt() for t in rows])
                        masks[N] = sp.topk_mask(st, k)
                    masks["rnd"] = sp.topk_mask(torch.rand(96, K), k)
                    for key, m in masks.items():
                        acc[key].append((((x[32:] * m) @ w.t() - y).norm(dim=1) / yn).mean().item())
            L.append(f"| {kind} | {f} | " + " | ".join(f"{np.mean(acc[k]) * 100:.1f}" for k in ("tok", 1, 8, 32, "rnd")) + " |")
    (RES / "B1e.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
