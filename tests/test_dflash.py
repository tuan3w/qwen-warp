"""DFlash2 block drafter on the device (engine/dflash.py, src/dflash.cu): the greedy output of the cycle graph equals the target's own greedy decode through the T = 8 verify path,
and the drafter earns its keep (tokens per cycle above MTP k = 3 on the same text).

  python tests/test_dflash.py [--n 128]
Needs the target checkpoint ($SWIFT_MODEL), the W4A16 drafter ($DFLASH_W4, default ~/llm-lab/models/dflash2-w4a16) and the GPU.  The device loop and the host-driven T = 8 decode run the same
verify kernels, so the ids must be equal; a near-tie flip between the T = 1 and T = 8 batch shapes is NOT tolerated here (the comparison is T = 8 against T = 8).
"""
import argparse
import os
import sys
from pathlib import Path

ENG = Path(__file__).resolve().parent.parent / "engine"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()
    sys.path.insert(0, str(ENG))
    import torch
    from transformers import AutoTokenizer

    from dflash import DflashEngine

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = "Rewrite this Python with type hints and docstrings, keep the logic identical, then add unit tests:\n\n" + (ENG / "spec.py").read_text()[:900] + "\n\nRewritten:\n"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    e = DflashEngine(ctx_max=len(ids) + args.n + 256, kv_fp8=0, verbose=False, tmax=8, host_tables=True)
    e.disable_pld()
    mask_id = 248070
    # reference: plain greedy through the T = 8 verify path (rows 1..7 are junk, only row 0 is committed), same 8-row prefill chunks as the device loop
    e.reset()
    n, i = len(ids) - 1, 0
    while i < n:
        t = min(8, n - i)
        e._step(ids[i : i + t], 0)
        i += t
    e._step(ids[-1:], 1)
    first = e.read_out_tok()
    ref, anchor = [first], first
    while len(ref) < args.n:
        y = e.verify([anchor] + [mask_id] * 7)
        e.commit(1, 8)
        ref.append(y[0])
        anchor = y[0]
    toks, st = e.generate_df(ids, args.n)
    toks = toks[: args.n]
    assert toks == ref, f"DFlash2 loop differs from the T = 8 greedy decode at token {next(i for i, (a, b) in enumerate(zip(toks, ref)) if a != b)}"
    mtp, mst = e.generate(ids, args.n, k=3)
    print(f"DFlash2: {st['tokens_per_cycle']:.2f} tokens/cycle ({st['cycles']} cycles), MTP k=3: {mst['tokens_per_cycle']:.2f}; ids equal to the T = 8 greedy decode over {args.n} tokens")
    assert st["tokens_per_cycle"] > mst["tokens_per_cycle"] + 0.5, "the drafter does not beat MTP k = 3 on a code prompt"
    # restart on the same engine: the second request (new prefill, drafter cache reset) gives the same ids
    toks2, _ = e.generate_df(ids, args.n)
    assert toks2[: args.n] == toks, "second request differs (drafter state not reset)"
    print("OK")


if __name__ == "__main__":
    main()
