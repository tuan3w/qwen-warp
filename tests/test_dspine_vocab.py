#!/usr/bin/env python
"""The DSpine head scores only the 68,864 draft-vocabulary tokens (engine/draft_vocab_ext_68864.npy) instead of all 248,320: a token outside the list can never be drafted.  The output stays
exactly the target's greedy output (the verify step decides), but every missing token costs an accepted draft.  This test pins WHAT is covered, per kind of text, so a changed list or a
changed tokenizer cannot silently lose coverage, and it states the known gaps instead of hiding them.  CPU only; needs the tokenizer files of $SWIFT_MODEL.

  python tests/test_dspine_vocab.py
Coverage = fraction of the tokens of a fixed corpus (tests/data/dspine_corpora.json, plus this repo's own code and docs) that are in the list.  Floors are the values measured on 2026-11-10 minus
a margin (the corpora are small: +-2 points of sampling noise for the 100-token ones).  Measured: code 0.9992, English 0.9993, Vietnamese 0.987, Chinese 0.81, French 0.61.
"""

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
VOCAB = REPO / "engine" / "draft_vocab_ext_68864.npy"
FULL = 248320
FLOORS = {
    "code": 0.995,
    "english": 0.995,
    "vietnamese": 0.97,
    "chinese": 0.75,
    "french": 0.55,
}
KNOWN_GAPS = {
    "chinese",
    "french",
}  # coverage well below 1: acceptance on these languages is lower with the restricted head (docs/PERFORMANCE.md)
SPECIAL = [
    "<|im_end|>",
    "<|endoftext|>",
    "<|im_start|>",
    "<think>",
    "</think>",
    "\n",
    "\n\n",
    " ",
]


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def main():
    from transformers import AutoTokenizer

    v = np.load(VOCAB)
    check(v.ndim == 1 and len(v) == 68864, f"draft vocabulary has {len(v)} rows")
    check(
        bool((np.diff(v) > 0).all()),
        "ids strictly ascending (sorted, no duplicates): ties in the head select resolve to the lowest id like the full head",
    )
    check(
        int(v.min()) >= 0 and int(v.max()) < FULL,
        f"ids inside the full vocabulary [0, {FULL})",
    )
    check(len(v) % 4 == 0, "row count divisible by 4 (smk_headq_gemv requirement)")
    S = set(int(x) for x in v.tolist())
    tok = AutoTokenizer.from_pretrained(MODEL)
    for t in SPECIAL:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        check(
            all(i in S for i in ids),
            f"special / structural token {t!r} {ids} can be drafted (end of turn, think tags, newlines)",
        )
    corpora = json.loads((REPO / "tests" / "data" / "dspine_corpora.json").read_text())
    corpora["code"] = "".join(
        Path(f).read_text() for f in sorted(glob.glob(str(REPO / "engine" / "*.py")))
    )[:30000]
    corpora["english"] = "".join(
        Path(f).read_text() for f in sorted(glob.glob(str(REPO / "docs" / "*.md")))
    )[:30000]
    for name, floor in FLOORS.items():
        ids = tok(corpora[name], add_special_tokens=False)["input_ids"]
        cov = sum(i in S for i in ids) / len(ids)
        gap = (
            "  [KNOWN GAP: tokens the restricted head can never draft]"
            if name in KNOWN_GAPS
            else ""
        )
        check(
            cov >= floor,
            f"{name}: {cov * 100:.2f}% of {len(ids)} tokens in the draft vocabulary (floor {floor * 100:.1f}%){gap}",
        )
    print("OK")


if __name__ == "__main__":
    sys.exit(main())
