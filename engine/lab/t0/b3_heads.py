"""B3 (attention side): per-head peakiness of the full-attention layers on the real window (CPU torch, ~1 min).  -> results/b3/B3_heads.md, b3_heads.json

Per (layer, q head): over the 128 decode rows -- mean entropy, median number of tokens needed for 90% of the probability mass (n90), mean mass inside the always-read set A (sink 4 + last 256),
mean max probability.  Classes: 'streaming' = mass(A) >= 0.9 (KV beyond sink+ring is nearly unused: DuoAttention-style static head), 'peaky' = n90 <= 256, 'mid' 256 < n90 <= 2048, 'diffuse' n90 > 2048.
Stability of the classes across the 6 docs (is a streaming head streaming in every doc?)."""
import json
from pathlib import Path

import numpy as np
import torch

import sp

RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/b3")
DOCS = ["tr32_code_cuda", "tr32_notes", "tr32_vendor_py", "tr32_vendor_cu", "tr32_gitlog", "tr32_papers"]


def head_stats(T, li):
    q = T.q(li)
    K, V = T.kv(li)
    W, L = q.shape[0], K.shape[1]
    n = L - W
    t = torch.arange(L)
    p = (n + torch.arange(W)).view(-1, 1)
    valid = t.view(1, -1) <= p
    always = ((t.view(1, -1) < 4) | (t.view(1, -1) > p - 256)) & valid
    out = np.zeros((24, 4))
    for h in range(24):
        lg = (q[:, h] @ K[h // 6].t() / 16).masked_fill(~valid, float("-inf"))
        P = torch.softmax(lg, -1)
        ent = -(P * P.clamp_min(1e-30).log()).sum(-1)
        ps = P.sort(-1, descending=True).values
        n90 = (ps.cumsum(-1) < 0.9).sum(-1) + 1
        mA = (P * always).sum(-1)
        out[h] = [ent.mean().item(), n90.float().median().item(), mA.mean().item(), ps[:, 0].mean().item()]
    return out


def main():
    RES.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    S = {}
    for d in DOCS:
        T = sp.Trace(d, "cpu")
        for li in sp.ATTN_LAYERS:
            if not T.has_kv(li):
                continue
            S[(d, li)] = head_stats(T, li)
            T._c.pop(("kv", li), None)
        print(d, flush=True)
    L = ["# B3: per-head attention peakiness (32k, 128 decode rows, 6 docs)\n", "streaming: mass in sink4+ring256 >= 0.9; peaky: n90 <= 256; mid: <= 2048; diffuse: > 2048 (n90 = tokens for 90% of the mass, median over rows).\n",
         "| layer | streaming | peaky | mid | diffuse | mean entropy | mean mass(A) | mean max-prob | heads streaming in all docs | heads diffuse in all docs |", "|---|---|---|---|---|---|---|---|---|---|"]
    js = {}
    for li in sp.ATTN_LAYERS:
        docs = [d for d in DOCS if (d, li) in S]
        A = np.stack([S[(d, li)] for d in docs])  # [doc,24,4]
        n90, mA = A[:, :, 1], A[:, :, 2]
        stream = mA >= 0.9
        cls = np.where(stream, 0, np.where(n90 <= 256, 1, np.where(n90 <= 2048, 2, 3)))
        fr = [(cls == c).mean() for c in range(4)]
        L.append(f"| {li} | {fr[0]:.2f} | {fr[1]:.2f} | {fr[2]:.2f} | {fr[3]:.2f} | {A[:, :, 0].mean():.2f} | {mA.mean():.2f} | {A[:, :, 3].mean():.2f} | {int(stream.all(0).sum())}/24 ({len(docs)} docs) | {int((n90 > 2048).all(0).sum())}/24 |")
        js[li] = dict(entropy=A[:, :, 0].mean(0).tolist(), n90=n90.mean(0).tolist(), massA=mA.mean(0).tolist(), maxp=A[:, :, 3].mean(0).tolist())
    (RES / "B3_heads.md").write_text("\n".join(L) + "\n")
    (RES / "b3_heads.json").write_text(json.dumps(js))
    print("\n".join(L))


if __name__ == "__main__":
    main()
