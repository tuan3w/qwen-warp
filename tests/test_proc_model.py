"""Logit processors on REAL logits: the engine's processed distribution (quantised head, candidate selection on the processed approximate logits, exact rescoring, processing of the candidates, top-k / top-p /
min_p) against a full-vocabulary torch reference computed from the engine's own final hidden state with the exact BF16 lm_head.  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL).

  python tests/test_proc_model.py

For every row of a speculative cycle (T = 4 verify rows with their draft tokens, plus the first-token row of the prefill) the engine's distribution (smk_spec_read what = 2) is compared with the reference
(tests/test_logit_procs.py ref_rows / ref_dist; history = the tokens generated before the cycle + the drafts before the row): max abs probability error, total variation, same support.
Greedy mode (temperature 0 with processors = top_k 1 sampling): every emitted token of every cycle (accepted drafts, correction, bonus) must be the argmax of the reference's processed logits.
Negative controls: the reference without the processors must disagree.
"""
import ctypes as C
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import smk  # noqa: E402
from spec import SpecEngine  # noqa: E402
from test_logit_procs import FAILS, check, ref_dist, ref_rows  # noqa: E402
import os  # noqa: E402

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
V = 248320
DIST_BYTES = 4 + 64 * 4 + 64 * 4


def read_dist(e, nrows):
    buf = (C.c_ubyte * (DIST_BYTES * nrows))()
    torch.cuda.synchronize()
    smk.check(e.lib.smk_spec_read(e.sp, 2, buf, DIST_BYTES * nrows), "spec_read")
    a = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, DIST_BYTES)
    n = a[:, :4].copy().view(np.int32)[:, 0]
    ids = a[:, 4:260].copy().view(np.int32)
    p = a[:, 260:].copy().view(np.float32)
    return [{int(ids[r, i]): float(p[r, i]) for i in range(n[r])} for r in range(nrows)]


def full_logits(e, hidden):
    """exact BF16 lm_head logits (fp32 accumulate) of the rows [R][5120] -> [R][V] on the CPU."""
    h = hidden.to("cuda").float()
    out = torch.empty(h.shape[0], V)
    w = e.G.lm_head
    for i in range(0, V, 8192):
        out[:, i : i + 8192] = (w[i : i + 8192].float() @ h.T).T.cpu()
    return out


def compare(d, q):
    full = torch.zeros(V, dtype=torch.float64)
    for i, v in d.items():
        full[i] = v
    return float((full - q).abs().max()), 0.5 * float((full - q).abs().sum()), set(d) == set(torch.nonzero(q).flatten().tolist())


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)

    def enc(text):
        return tok(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]

    prompts = [enc("Write a short story about a lighthouse keeper."), enc("List the numbers from 1 to 30 separated by commas."), enc("Say hi."),
               enc("Explain why the sky is blue, then repeat the explanation in other words."), enc("the cat sat on the mat. " * 12 + "Continue the pattern.")]
    rare = [200000, 150321, 99999, 230001]
    stop = [248046, 248044]
    e = SpecEngine(ctx_max=1024, verbose=False)
    cases = [  # name, procs (ref_rows names), temperature, top_k, top_p, min_p
        ("penalties", dict(pres=0.8, freq=0.6, rep=1.25), 0.8, 20, 0.95, 0.0),
        ("repetition 1.8", dict(rep=1.8), 1.0, 20, 1.0, 0.0),
        ("logit_bias promoting rare tokens", dict(bias={rare[0]: 22.0, rare[1]: 19.0, rare[2]: -5.0}), 1.0, 20, 1.0, 0.0),
        ("allowed set of 2 rare tokens", dict(allowed=rare[:2]), 1.0, 20, 1.0, 0.0),
        ("allowed set of 5000 tokens", dict(allowed=list(range(1000, 6000)) + stop), 1.0, 20, 0.95, 0.0),
        ("min_tokens", dict(min_tokens=40, stop=stop), 1.0, 20, 1.0, 0.0),
        ("all + min_p", dict(pres=0.5, freq=0.5, rep=1.15, bias={rare[3]: 3.0}, min_tokens=3, stop=stop), 0.7, 20, 0.9, 0.05),
    ]
    stats = {}
    for name, pr, T, top_k, top_p, min_p in cases:
        worst_e, worst_tv, rows, bad_sup, neg_tv, worst = 0.0, 0.0, 0, 0, [], None
        for pi, ids in enumerate(prompts):
            procs = dict(prompt=ids, **pr)
            e.set_sampling(temperature=T, top_k=top_k, top_p=top_p, seed=100 + pi, min_p=min_p)
            e.set_procs(prompt=ids, presence=pr.get("pres", 0.0), frequency=pr.get("freq", 0.0), repetition=pr.get("rep", 1.0), min_tokens=pr.get("min_tokens", 0), bias=pr.get("bias"),
                        allowed=pr.get("allowed"), stop=pr.get("stop", ()))
            e.reset()
            first = e.prefill_spec(ids, 3)
            gen = [first]
            # the first-token row: engine Dist row 0 of the prefill (history = prompt only) vs the reference on the same hidden state
            rows_h = [(read_dist(e, 1)[0], e.dbg_hidden(1)[65][0], [], [])]
            for _ in range(4):
                _, _, dd = e.draft_state()
                new, _ = e.cycles(3, 1)
                dists = read_dist(e, 4)
                hid = e.dbg_hidden(4)[65][:4]
                for t in range(4):
                    rows_h.append((dists[t], hid[t], list(gen), dd[1 : 1 + t]))
                gen += new
            lg = full_logits(e, torch.stack([h for _, h, _, _ in rows_h]))
            for r, (d, _, g, drafts) in enumerate(rows_h):
                row = ref_rows(lg[r : r + 1], gen=g, drafts=drafts, **procs)
                q = ref_dist(row[0], T, top_k, top_p, min_p)
                err, tv, sup = compare(d, q)
                if tv > worst_tv:
                    worst = (pi, r, d, q)
                worst_e, worst_tv, rows, bad_sup = max(worst_e, err), max(worst_tv, tv), rows + 1, bad_sup + (not sup)
                q0 = ref_dist(lg[r], T, top_k, top_p, min_p)  # the reference WITHOUT the processors
                neg_tv.append(compare(d, q0)[1])
        stats[name] = (worst_e, worst_tv, rows, bad_sup, float(np.mean(neg_tv)))
        ok = worst_e < 2e-3 and worst_tv < 5e-3
        check(f"real logits, {name}: engine distribution == full-vocabulary reference ({rows} rows from {len(prompts)} prompts)", ok,
              f"max abs err {worst_e:.2e}, max TV {worst_tv:.2e}, support differs in {bad_sup} rows")
        if not ok and worst:
            pi, r, d, q = worst
            top = lambda dd: sorted(dd.items(), key=lambda kv: -kv[1])[:4]
            ref = {int(i): float(q[i]) for i in torch.nonzero(q).flatten()}
            print(f"    worst row: prompt {pi}, row {r}: engine top {top(d)}  reference top {top(ref)}  missing in engine {[(i, round(v, 4)) for i, v in ref.items() if i not in d][:5]}  extra in engine {[(i, round(v, 4)) for i, v in d.items() if i not in ref][:5]}", flush=True)
        check(f"real logits, {name}: negative control (reference without the processors) disagrees on at least one row (max TV > 0.2)", max(neg_tv) > 0.2, f"max TV {max(neg_tv):.3f}, mean {np.mean(neg_tv):.3f}")

    # greedy over the processed logits = top_k 1 sampling: every emitted token is the argmax of the reference's processed row
    tot = ok = 0
    for name, pr in (("penalties", dict(pres=1.0, freq=0.8, rep=1.4)), ("bias + min_tokens", dict(bias={rare[0]: 4.0}, min_tokens=4, stop=stop)), ("allowed 5000", dict(allowed=list(range(1000, 6000)) + stop))):
        for pi, ids in enumerate(prompts):
            e.set_sampling(temperature=1.0, top_k=1, top_p=1.0, seed=0)
            e.set_procs(prompt=ids, presence=pr.get("pres", 0.0), frequency=pr.get("freq", 0.0), repetition=pr.get("rep", 1.0), min_tokens=pr.get("min_tokens", 0), bias=pr.get("bias"),
                        allowed=pr.get("allowed"), stop=pr.get("stop", ()))
            e.reset()
            first = e.prefill_spec(ids, 3)
            gen = [first]
            first_lg = full_logits(e, e.dbg_hidden(1)[65][:1])
            tot += 1
            ok += int(first == int(ref_rows(first_lg, **dict(prompt=ids, **pr))[0].argmax()))
            for _ in range(12):
                _, _, dd = e.draft_state()
                new, _ = e.cycles(3, 1)
                a = len(new)
                lg = full_logits(e, e.dbg_hidden(4)[65][:4])
                ref = ref_rows(lg, gen=gen, drafts=dd[1:4], **dict(prompt=ids, **pr))
                for t in range(a):
                    tot += 1
                    ok += int(new[t] == int(ref[t].argmax()))
                gen += new
    check("greedy with processors: every emitted token is the argmax of the processed reference logits (accepted drafts, corrections and bonus tokens)", ok == tot, f"{ok}/{tot}")
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
