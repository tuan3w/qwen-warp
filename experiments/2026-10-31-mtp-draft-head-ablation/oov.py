"""Out-of-vocabulary rate of the TARGET's generated tokens under each restricted draft vocab (CPU only).

reproduce: ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/oov.py   (needs outputs/base3.json from run.py)

A draft token outside V can never be proposed, so if the target's next token is OOV the draft at that position is wrong for sure.
OOV rate r bounds the position-1 acceptance of an arm at (1 - r) x base acceptance (further positions inherit it). This gives the mechanism
check for the V arms: measured relative position-1 acceptance vs 1 - OOV. Generations: outputs/base3.json (arm 0, greedy + T=1 seeds).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "projects/llm-lab")
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))
import run as R  # noqa: E402
from prompts import AGENT, CODE, MATH, PROSE, long_prompts  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(R.MODEL))
    outs = json.loads((Path.home() / "llm-lab/runs/2026-10-31-mtp-draft-head-ablation/outputs/base3.json").read_text())
    freq = np.load(R.FREQ)
    forced = R.forced_ids(tok)
    cats = {"code": CODE, "math": MATH, "prose": PROSE, "agent": AGENT, "long": long_prompts(tok)}
    ctx = {}
    for cat, d in cats.items():
        for n, t in d.items():
            p = tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
            ctx[n] = set(tok(p, add_special_tokens=False)["input_ids"])
    cat_of = {n: c for c, d in cats.items() for n in d}
    res = {}
    for V in (8192, 16384, 32768, 49152, 65536):
        mask, n = R.allowed_mask(freq, forced, V)
        acc = {}
        for key, toks in outs.items():
            pname = key.split("/")[0]
            c = cat_of[pname]
            arr = np.asarray(toks)
            oov = ~mask[arr]
            oov_ctx = oov & ~np.isin(arr, list(ctx[pname]))
            a = acc.setdefault(c, [0, 0, 0])
            a[0] += len(arr)
            a[1] += int(oov.sum())
            a[2] += int(oov_ctx.sum())
        tot = [sum(v[i] for v in acc.values()) for i in range(3)]
        res[str(V)] = {"allowed": n, **{c: {"tokens": v[0], "oov": round(v[1] / v[0], 4), "oov_after_ctx": round(v[2] / v[0], 4)} for c, v in acc.items()},
                       "all": {"tokens": tot[0], "oov": round(tot[1] / tot[0], 4), "oov_after_ctx": round(tot[2] / tot[0], 4)}}
        print(V, json.dumps({c: (r["oov"], r["oov_after_ctx"]) for c, r in res[str(V)].items() if c != "allowed"}))
    (EXP / "results" / "oov.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
