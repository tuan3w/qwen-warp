#!/usr/bin/env python
"""Fidelity of the int4 KV cache against bf16: mean KL divergence of the next-token distribution, teacher-forced through the real kernels (decode path, 8-row steps).

  python scripts/kv_dkl.py ref  --text a.py b.py ... [--ids ids.json] --tokens 16384 --out ref.pt     # bf16 cache: top-64 log-probs per position (once per text)
  python scripts/kv_dkl.py run  --ref ref.pt [--build build_other]                     # int4 cache against that reference

dKL = mean over positions of KL(p || q) with p the bf16 distribution restricted to its top 64 tokens (renormalised) and q the int4 log-probs on those tokens (renormalised),
reported in percent, plus the p99 of the per-position KL.  The engine library is engine/<--build or $SMK_BUILD or build>/libswiftmk.so, so two builds can be compared on the same
reference.  The int4 and the bf16 run use the same engine, the same quantised head and the same 8-row steps; only the KV cache differs.  Needs the checkpoint ($SWIFT_MODEL).
The text is the concatenation of the given files (tokenised with the checkpoint's tokenizer, truncated to --tokens); pick files the model has not memorised and that are not repeated.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

ENG = Path(__file__).resolve().parent.parent / "engine"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
TOP = 64


def tokens(files, n):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = []
    for f in files:
        ids += tok(
            f"// {Path(f).name}\n" + Path(f).read_text() + "\n\n",
            add_special_tokens=False,
        )["input_ids"]
        if len(ids) >= n:
            break
    assert len(ids) >= n, f"only {len(ids)} tokens in the given files, need {n}"
    return ids[:n]


def sweep(kv, ids, ref):
    """8-row teacher-forced steps; ref None: return the top-64 (ids, log-probs) per position, else the per-position KL against ref."""
    import torch

    sys.path.insert(0, str(ENG))
    from engine import Engine

    e = Engine(ctx_max=len(ids) + 64, kv_fp8=kv, tmax=8, verbose=False)
    e.reset()
    out_i, out_v, kls, nll = [], [], [], []
    i = 0
    while i < len(ids) - 1:
        T = min(8, len(ids) - 1 - i)
        e._step(ids[i : i + T], 1)
        lp = torch.log_softmax(e.logits(T).cuda(), -1)
        nll.append(
            -lp[torch.arange(T), torch.tensor(ids[i + 1 : i + 1 + T], device="cuda")]
        )
        if ref is None:
            v, ix = lp.topk(TOP, -1)
            out_i.append(ix.cpu())
            out_v.append(v.cpu())
        else:
            rid, rlp = ref["ids"][i : i + T].cuda().long(), ref["lp"][i : i + T].cuda()
            lq = torch.log_softmax(lp.gather(1, rid), -1)
            lpn = torch.log_softmax(rlp, -1)
            kls.append((lpn.exp() * (lpn - lq)).sum(-1))
        i += T
    nll = torch.cat(nll)
    assert torch.isfinite(nll).all(), "non-finite NLL"
    if ref is None:
        return {
            "ids": torch.cat(out_i),
            "lp": torch.cat(out_v),
            "n": len(ids),
            "nll": nll.mean().item(),
        }
    return torch.cat(kls), nll.mean().item()


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ref", "run"])
    ap.add_argument("--text", nargs="+")
    ap.add_argument("--ids", help="JSON list of token ids instead of --text")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--out", default="ref.pt")
    ap.add_argument("--ref", default="ref.pt")
    ap.add_argument(
        "--build",
        default=None,
        help="engine build directory name (default: $SMK_BUILD or build)",
    )
    ap.add_argument(
        "--kv", type=int, default=2, help="KV dtype of the run: 0 bf16, 1 fp8, 2 int4"
    )
    a = ap.parse_args()
    if a.build:
        os.environ["SMK_BUILD"] = a.build
    if a.mode == "ref":
        ids = json.load(open(a.ids))[: a.tokens] if a.ids else tokens(a.text, a.tokens)
        r = sweep(0, ids, None)
        r["ids_text"] = ids
        torch.save(r, a.out)
        print(
            f"reference for {len(ids)} tokens written to {a.out} (bf16 NLL {r['nll']:.4f})"
        )
        return
    ref = torch.load(a.ref)
    kl, nll = sweep(a.kv, ref["ids_text"], ref)
    q99 = torch.quantile(kl, torch.tensor([0.99], device=kl.device)).item()
    print(
        json.dumps(
            {
                "tokens": ref["n"],
                "dkl_pct": 100 * kl.mean().item(),
                "p99_kl_pct": 100 * q99,
                "nll": nll,
                "nll_bf16": ref["nll"],
                "build": os.environ.get("SMK_BUILD", "build"),
            }
        )
    )
    assert math.isfinite(kl.mean().item())


if __name__ == "__main__":
    main()
