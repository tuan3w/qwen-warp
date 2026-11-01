"""Sanity report on the reference dumps: coherent greedy text + MTP-head agreement + dump integrity.

    CUDA_VISIBLE_DEVICES= ~/llm-lab/venv/bin/python check_sanity.py [--mode fp32|bf16] [--long N]

  default : reads <dir>/<mode>/*.pt written by dump_ref.py, prints prompt + greedy continuation, checks internal
            consistency (gen_ids == argmax of logits, hid_out[l] == hid_in[l+1], final norm recomputed from layer-63 output),
            prints MTP draft agreement (draft k at event e vs greedy token gen[e+k]).
  --long N: regenerates N decode steps for the 3 prompts (CPU, ~15 s/step) and prints the longer text + MTP agreement.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

import reference as R
from weights import MODEL_DIR, Checkpoint

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="fp32")
ap.add_argument("--dir", default=os.path.expanduser("~/llm-lab/swift-ref"))
ap.add_argument("--long", type=int, default=0)
ap.add_argument("--threads", type=int, default=int(os.environ.get("ORACLE_THREADS", 18)))
a = ap.parse_args()
torch.set_num_threads(a.threads)
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL_DIR)


def agreement(gens, drafts, k=3):
    """drafts[e][j-1] (j=1..k) is the draft made after event e; it predicts gens[e+j]."""
    hit, tot = [0] * k, [0] * k
    for g, d in zip(gens, drafts):
        for e in range(len(d)):
            for j in range(1, k + 1):
                if e + j < len(g):
                    tot[j - 1] += 1
                    hit[j - 1] += int(d[e][j - 1] == g[e + j])
    return hit, tot


def report_agreement(label, gens, drafts):
    hit, tot = agreement(gens, drafts)
    print(f"MTP agreement {label}: " + "  ".join(f"k={j + 1}: {hit[j]}/{tot[j]} = {hit[j] / max(tot[j], 1):.0%}" for j in range(3)))


if a.long:
    torch.set_num_threads(a.threads)
    prompts = R.build_prompts(tok, R.E0_SHORT)
    names = list(prompts)
    o = R.Oracle(bf16=(a.mode == "bf16"))
    seqs = [R.SeqState() for _ in names]
    ev, gen, _ = R.generate(o, seqs, [prompts[n] for n in names], a.long, mtp_k=3, tap_layers=())
    out = {}
    for n, g, e in zip(names, gen, ev):
        print(f"\n=== {n}: {R.E0_SHORT[n]}\n{tok.decode(g)}")
        report_agreement(n, [g], [e["drafts"]])
        out[n] = {"gen_ids": g, "drafts": e["drafts"], "text": tok.decode(g)}
    report_agreement("ALL", gen, [e["drafts"] for e in ev])
    json.dump(out, open(Path(a.dir) / a.mode / f"long{a.long}.json", "w"), indent=1)
    sys.exit(0)

d = Path(a.dir) / a.mode
names = list(R.E0_SHORT)
ck = Checkpoint()
fnorm = ck.get("model.language_model.norm.weight").float()
gens, drafts = [], []
for n in names:
    z = torch.load(d / f"{n}.pt")
    gen = z["gen_ids"].tolist()
    print(f"\n=== {n} ({z['meta']['n_prompt']} prompt tokens): {z['meta']['text']}")
    print(f"greedy ({len(gen)} tokens): {tok.decode(gen)!r}")
    print("   ids:", gen)
    print("   drafts after each event (k=1..3):", z["drafts"].tolist())
    # integrity
    assert torch.equal(z["top20_idx"][:, 0], z["gen_ids"]), "gen_ids != argmax(top20)"
    assert torch.equal(z["hid_out"][:, :-1], z["hid_in"][:, 1:]), "hid_out[l] != hid_in[l+1]"
    recomputed = R.rmsnorm0(z["hid_out"][:, -1], fnorm)
    err = float((recomputed - z["final_norm_out"]).abs().max())
    tol = 2e-2 if a.mode == "bf16" else 1e-3  # bf16 mode rounds final_norm_out to bf16 (rel 4e-3)
    assert err < tol * float(z["final_norm_out"].abs().max()), err
    assert torch.equal(z["event_tokens"][1:], z["gen_ids"][:-1])
    print(f"   integrity OK; top-1 logit margin per event: {[round(float(v[0] - v[1]), 2) for v in z['top20_val']]}")
    gens.append(gen)
    drafts.append(z["drafts"].tolist())
    report_agreement(n, [gen], [drafts[-1]])
report_agreement("ALL (9 events/prompt, 8 decode steps)", gens, drafts)
