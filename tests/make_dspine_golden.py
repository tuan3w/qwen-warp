#!/usr/bin/env python
"""Regenerates tests/data/dspine_golden.npz: the reference outputs tests/test_dspine.py compares the CUDA drafter against.  Run it on purpose, review the diff of the numbers, commit.

  python tests/make_dspine_golden.py [--prompt code_py]
One draft block after the prefill of a fixed prompt: the target's tap rows are read from the engine (so the reference sees exactly the context the CUDA drafter saw), then the independent torch
reference (tests/dspine_ref.py) computes the block with (a) the dequantised W4 weights the CUDA drafter multiplies with and (b) the checkpoint's bf16 weights, plus the refine drafts of both.
Needs the GPU (engine + quantiser), ~8 GB of RAM per reference variant (run one after the other) and about 5 minutes.  Stop the serving container first.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dspine_ref as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="code_py")
    args = ap.parse_args()
    from dspine import DspineEngine
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(D.TARGET)
    p = next(x for x in json.loads((HERE / "data" / "dspine_prompts.json").read_text()) if x["name"] == args.prompt)
    o = tok.apply_chat_template([{"role": "user", "content": p["user"]}], add_generation_prompt=True, tokenize=True)
    ids = list(o["input_ids"] if hasattr(o, "keys") else o)
    e = DspineEngine(ctx_max=4096, kv_fp8=0, verbose=False)
    # target tap rows of every prompt position (the last one comes from the head step)
    taps, n, i = [], len(ids) - 1, 0
    e.reset()
    slots = [l + 1 for l in D.TAP_LAYERS]
    while i < n:
        t = min(8, n - i)
        e._step(ids[i : i + t], 0)
        taps.append(e.dbg_hidden(t)[slots].permute(1, 0, 2).to(torch.bfloat16))
        i += t
    e._step(ids[-1:], 1)
    taps.append(e.dbg_hidden(1)[slots].permute(1, 0, 2).to(torch.bfloat16))
    taps = torch.cat(taps, 0)  # [S, 5, H]
    anchor = e.sp_prefill(ids)
    hid_engine = e.sp_hid().float().numpy()
    drafts_engine = [int(x) for x in e._read(0, 20, torch.int32)[3:10]]
    sig = D.quant_signature()
    S = len(ids)
    block = torch.tensor([[anchor] + [D.MASK_ID] * 7])
    lm = D.target_tensor("lm_head.weight")
    out = {"ids": np.array(ids), "anchor": np.array(anchor), "hid_engine": hid_engine, "drafts_engine": np.array(drafts_engine)}
    for name, quant in (("w4", True), ("bf16", False)):
        ref = D.RefDSpine(quantize=quant)
        ctx = ref.ctx_rows(taps).unsqueeze(0)
        hid, stash, layers = ref.draft(block, ctx, S, return_layers=True)
        cand, _ = ref.candidates(hid, lm)
        out[f"hid_{name}"] = hid.float().numpy()
        out[f"layers_{name}"] = torch.stack(layers).float().numpy()
        out[f"drafts_{name}"] = np.array(ref.refine(cand, stash, anchor, lm))
        print(f"{name}: drafts {out[f'drafts_{name}'].tolist()}  (engine: {drafts_engine})", flush=True)
        del ref, ctx, hid, stash, layers, cand
        gc.collect()
    meta = {"prompt": args.prompt, "quant_signature": sig, "prompt_ids_len": S}
    np.savez_compressed(HERE / "data" / "dspine_golden.npz", meta=json.dumps(meta), **out)
    print("wrote", HERE / "data" / "dspine_golden.npz", meta)


if __name__ == "__main__":
    main()
