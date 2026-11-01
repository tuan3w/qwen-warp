"""Generate the reference dumps (oracle outputs) used to unit-test kernels and the engine.

For each of the 3 E0 prompts (code/prose/math, chat template, thinking off): prefill, then 8 greedy decode steps, in one
batch (layer-major: each layer is dequantised once per step for all 3 sequences). CPU only, ~10-20 min.

    CUDA_VISIBLE_DEVICES= ~/llm-lab/venv/bin/python dump_ref.py [--bf16-rounding] [--out ~/llm-lab/swift-ref]

Output: <out>/<mode>/<prompt>.pt (torch.save of a dict, layout documented in README.md) + <out>/<mode>/manifest.json.
`event e` (e=0..8): e=0 is the last prompt token inside the prefill pass, e=1..8 are the 8 decode steps whose input token is
gen_ids[e-1]. gen_ids[e] is the greedy token produced by event e (9 tokens total, gen_ids[0] comes from the prefill).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch

import reference as R
from reference import STATE_KEYS, Oracle, SeqState
from weights import MODEL_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--bf16-rounding", action="store_true")
ap.add_argument("--out", default=os.path.expanduser("~/llm-lab/swift-ref"))
ap.add_argument("--n", type=int, default=8, help="decode steps")
ap.add_argument("--threads", type=int, default=int(os.environ.get("ORACLE_THREADS", 18)))
a = ap.parse_args()
torch.set_num_threads(a.threads)
assert not os.environ.get("CUDA_VISIBLE_DEVICES", "x"), "CPU only: run with CUDA_VISIBLE_DEVICES=''"

TAP_LAYERS = (0, 1, 2, 3)  # per-event sub-op taps (3 DeltaNet + 1 full attention)
PREFILL_TAP_LAYERS = (0, 3)  # all-token taps during prefill
STATE_LAYERS = (0, 1, 2)  # DeltaNet S/conv state saved after prefill
S_OUT_EVENTS = {0: (0, 1, 2), 1: (0,), 2: (0,)}  # event -> layers whose S_out/conv_state_out are kept (state after that event)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
prompts = R.build_prompts(tok, R.E0_SHORT)
names = list(prompts)
mode = "bf16" if a.bf16_rounding else "fp32"
out_dir = Path(a.out) / mode
out_dir.mkdir(parents=True, exist_ok=True)

t0 = time.time()
o = Oracle(bf16=a.bf16_rounding)
seqs = [SeqState() for _ in names]
events, gen, prefill_final = R.generate(o, seqs, [prompts[n] for n in names], a.n, mtp_k=3, tap_layers=TAP_LAYERS)
print(f"generation total {time.time() - t0:.0f}s")

manifest = {}
for i, name in enumerate(names):
    ev = events[i]
    L = len(prompts[name])
    taps = {}
    for e, per_layer in enumerate(ev["taps"]):
        taps[e] = {}
        for li, d in per_layer.items():
            keep = {k: v for k, v in d.items() if k not in STATE_KEYS}
            if li in S_OUT_EVENTS.get(e, ()):
                keep["S_out"], keep["conv_state_out"] = d["S_out"], d["conv_state_out"]
            taps[e][li] = keep
    # state after prefill = S_out / conv_state_out of event 0 (already in taps); KV of layer 3 after prefill
    kv3 = seqs[i].kv[3]
    kv3_prefill = {"k": kv3[0][:, :L].clone(), "v": kv3[1][:, :L].clone()}
    dump = {
        "meta": {"mode": mode, "prompt": name, "text": R.E0_SHORT[name], "n_prompt": L, "n_decode": a.n, "model": str(MODEL_DIR),
                 "mtp_k": 3, "conventions": "see oracle/README.md; S layout [48,128(K),128(V)], conv [10240,3] oldest first"},
        "prompt_ids": torch.tensor(prompts[name]),
        "gen_ids": torch.tensor(ev["gen"]),
        "event_tokens": torch.tensor([prompts[name][-1]] + ev["gen"][:-1]),
        "event_pos": torch.arange(L - 1, L + a.n),
        "hid_in": torch.stack(ev["hid_in"]),  # [9, 64, 5120] last-token hidden entering each layer
        "hid_out": torch.stack(ev["hid_out"]),  # [9, 64, 5120]
        "final_norm_out": torch.stack(ev["final"]),  # [9, 5120] post model.norm = lm_head input = MTP `hidden`
        "top20_val": torch.stack([t[0] for t in ev["top"]]),
        "top20_idx": torch.stack([t[1] for t in ev["top"]]),
        "drafts": torch.tensor(ev["drafts"]),  # [9, 3]  chained greedy MTP drafts k=1..3 after event e
        "mtp_hidden": torch.stack(ev["mtp_h"]),  # [9, 3, 5120] mtp.norm output of draft k
        "mtp_taps": ev["mtp_taps"],  # per event: fc_out, xnorm, q_proj, mixer_out, h_mid, layer_out, norm_out (+attn taps)
        "taps": taps,  # taps[event][layer][name] last-token (shape [1,...]) sub-op tensors; see README
        "prefill_taps": {li: ev["prefill_taps"][li] for li in PREFILL_TAP_LAYERS},  # all prompt tokens
        "prefill_final": prefill_final[i],  # [L, 5120] post-final-norm hidden of every prompt token
        "kv_layer3_after_prefill": kv3_prefill,  # [4, L, 256] post-rope k / v
    }
    torch.save(dump, out_dir / f"{name}.pt")
    manifest[name] = {"file": f"{name}.pt", "bytes": (out_dir / f"{name}.pt").stat().st_size, "n_prompt": L,
                      "gen_text": tok.decode(ev["gen"]), "drafts": ev["drafts"]}
json.dump(manifest, open(out_dir / "manifest.json", "w"), indent=1)
for n, m in manifest.items():
    print(f"{out_dir / m['file']}  {m['bytes'] / 1e6:.1f} MB  prompt tokens={m['n_prompt']}  gen={m['gen_text']!r}")
