"""End-to-end parity of the engine vs the oracle dumps (stage 2).

  ~/llm-lab/venv/bin/python tests/test_e2e.py [--ctx 2048] [--prompts code,prose,math] [--out PARITY.md]
For every prompt: prefill (T=4 graph chunks, no lm_head) + 8 decode events; per layer rel-L2 of the last-token hidden state vs the bf16-rounding oracle dump
(and vs the fp32 dump) at each dump event; greedy token ids vs the oracle (bf16 dump 9 ids, fp32 long32.json 33 ids).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Engine, smk  # noqa: E402

REF = os.path.expanduser("~/llm-lab/swift-ref")


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--prompts", default="code,prose,math")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--fp8", type=int, default=0)
    args = ap.parse_args()
    eng = Engine(ctx_max=args.ctx, kv_fp8=args.fp8)
    print(f"engine buffers: {eng.lib.smk_engine_bytes(eng.h) / 2**30:.2f} GiB; torch weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    long32 = json.load(open(f"{REF}/fp32/long32.json"))
    lines = []
    allok = True
    for name in args.prompts.split(","):
        zb = torch.load(f"{REF}/bf16/{name}.pt")
        zf = torch.load(f"{REF}/fp32/{name}.pt")
        ids = zb["prompt_ids"].tolist()
        eng.reset()
        t0 = time.time()
        first = eng.prefill(ids, chunk=args.chunk)
        torch.cuda.synchronize()
        t_pref = time.time() - t0
        toks = [first]
        rb = torch.zeros(9, 64)  # rel L2 vs bf16 dump
        rf = torch.zeros(9, 64)  # vs fp32 dump
        fin_b = torch.zeros(9)
        for ev in range(9):
            if ev > 0:
                toks += eng.decode(1)
            d = eng.dbg_hidden(1)  # [66, 1, 5120]
            for l in range(64):
                rb[ev, l] = rel(d[l + 1, 0], zb["hid_out"][ev, l])
                rf[ev, l] = rel(d[l + 1, 0], zf["hid_out"][ev, l])
            fin_b[ev] = rel(d[65, 0], zb["final_norm_out"][ev])
        ok9 = toks[:9] == zb["gen_ids"].tolist()
        # continue to 32 decode steps for the fp32 long continuation
        toks += eng.decode(32 - 8)
        ref33 = long32[name]["gen_ids"]
        ok33 = toks[:33] == ref33
        allok &= ok9 and ok33
        first_bad = next((i for i, (a, b) in enumerate(zip(toks, ref33)) if a != b), None)
        lines.append(f"### {name}  (prompt {len(ids)} tokens, prefill T={args.chunk} chunks {t_pref:.2f}s)")
        lines.append(f"- greedy ids vs bf16 dump (9): {'EQUAL' if ok9 else 'DIFFER ' + str(toks[:9]) + ' vs ' + str(zb['gen_ids'].tolist())}")
        lines.append(f"- greedy ids vs fp32 long32.json (33): {'EQUAL' if ok33 else 'DIFFER first at ' + str(first_bad)}")
        lines.append(f"- final-norm hidden rel-L2 vs bf16 dump per event: " + " ".join(f"{x:.4f}" for x in fin_b.tolist()))
        lines.append("")
        lines.append("| layer | " + " | ".join(f"e{e}" for e in range(9)) + " | max vs bf16 | e0 vs fp32 |")
        lines.append("|---|" + "---|" * 11)
        for l in list(range(0, 64, 4)) + [3, 7, 63]:
            pass
        shown = sorted(set([0, 1, 2, 3, 7, 15, 23, 31, 39, 47, 55, 62, 63]))
        for l in shown:
            lines.append(f"| {l} | " + " | ".join(f"{rb[e, l]:.4f}" for e in range(9)) + f" | {rb[:, l].max():.4f} | {rf[0, l]:.4f} |")
        lines.append(f"- all 64 layers, event 0, rel-L2 vs bf16 dump: " + " ".join(f"{x:.4f}" for x in rb[0].tolist()))
        lines.append(f"- layer 63, events 0..8 vs fp32 dump: " + " ".join(f"{x:.4f}" for x in rf[:, 63].tolist()))
        lines.append("")
        print("\n".join(lines[-14:]), flush=True)
    if args.out:
        open(args.out, "w").write("\n".join(lines) + "\n")
    print("ALL PARITY OK" if allok else "PARITY MISMATCH")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
