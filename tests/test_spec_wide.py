#!/usr/bin/env python
"""Chain extension and wide verify of the greedy MTP + prompt-lookup cycle (engine/src/mtp.cu, spec.make_pld(ext, wide)), engine level.

Tier: 1

  python tests/test_spec_wide.py [--n 256]
Needs the target checkpoint ($SWIFT_MODEL) and the GPU.

Both features only change WHICH tokens a cycle verifies (extra MTP drafts on tail-less cycles, up to 16 verify rows on a long prompt-lookup match), never the acceptance rule, so the
emitted stream is the target's greedy stream.  It is not bit-identical to the stream of the plain cycle: the attention split depends on the verify width, so a bf16 near-tie can flip
and the text then diverges (a prefix match cascades after the first flip and says nothing about quality).  The oracle is therefore teacher-forced: the emitted stream is fed back through
plain 8-row verify steps and each position counts when the step's argmax equals the next emitted token.  One flip costs one position.
  1. the features engage: a chat prompt has extension cycles and more tokens per cycle than the plain cycle; a prompt that copies a document has wide cycles (T > 8) and at least 1.2x the
     plain cycle's tokens per cycle
  2. FIDELITY  the teacher-forced agreement of the emitted stream is >= 0.98 (measured 0.997 on 8 prompts, the plain cycle against itself 0.997); a negative control, the same stream with every
     7th token replaced, must fall below 0.9, so the check has teeth
  3. DETERMINISM  the same request twice gives identical ids
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from transformers import AutoTokenizer  # noqa: E402

from engine import TMAX_WIDE  # noqa: E402
from prefill import FastPrefill  # noqa: E402
from spec import SpecEngine, make_pld  # noqa: E402

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
FIDELITY_MIN, CONTROL_MAX = 0.98, 0.90


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def fidelity(e, fp, ids, toks):
    """fraction of positions where the plain 8-row verify argmax (teacher-forced on `toks`) equals the next emitted token."""
    e.reset()
    good, n = int(fp.prefill(ids) == toks[0]), 1
    i = 0
    while i < len(toks) - 1:
        rows = toks[i : i + 8]
        r = e.verify(rows)
        for j in range(len(rows)):
            if i + j + 1 < len(toks):
                n += 1
                good += int(r[j] == toks[i + j + 1])
        e.commit(len(rows), len(rows))
        i += len(rows)
    return good / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    doc = (ROOT / "tests" / "long_doc.txt").read_text()
    prompts = {
        "chat": "Write a Python implementation of an LRU cache class with get/put in O(1), then unit tests with pytest.",
        "copy": "Repeat the following text exactly, without changing a word:\n\n" + doc,
    }
    ids = {}
    for k, p in prompts.items():
        o = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=False,
        )
        ids[k] = list(o["input_ids"] if hasattr(o, "keys") else o)
    e = SpecEngine(
        ctx_max=max(map(len, ids.values())) + a.n + 256,
        kv_fp8=2,
        verbose=False,
        tmax=TMAX_WIDE,
        host_tables=True,
    )
    fp = FastPrefill(e)
    arms = {"plain": make_pld(ext=0, wide=0), "new": make_pld()}

    res = {}
    for name, x in ids.items():
        for arm, opts in arms.items():
            e.enable_pld(opts)
            res[name, arm] = e.generate(x, a.n, k=3, eos=eos)
    again, _ = e.generate(
        ids["chat"], a.n, k=3, eos=eos
    )  # the last arm run was 'new' on 'copy': enable the same options and repeat 'chat'
    e.enable_pld(arms["new"])
    again, _ = e.generate(ids["chat"], a.n, k=3, eos=eos)

    tpc = lambda k, arm: res[k, arm][1]["tokens_per_cycle"]  # noqa: E731
    st = res["chat", "new"][1]["pld"]
    check(
        st["ext_cycles"] > 0 and tpc("chat", "new") > tpc("chat", "plain"),
        f"chat: {st['ext_cycles']} extension cycles, {tpc('chat', 'new'):.2f} tokens per cycle against {tpc('chat', 'plain'):.2f} plain",
    )
    st = res["copy", "new"][1]["pld"]
    check(
        st["wide_cycles"] > 0 and tpc("copy", "new") >= 1.2 * tpc("copy", "plain"),
        f"copy: {st['wide_cycles']} wide cycles, {tpc('copy', 'new'):.2f} tokens per cycle against {tpc('copy', 'plain'):.2f} plain",
    )
    check(
        not res["copy", "plain"][1]["pld"]["wide_cycles"]
        and not res["chat", "plain"][1]["pld"]["ext_cycles"],
        "ext=0, wide=0 switches both off",
    )
    for name in ids:
        toks = res[name, "new"][0]
        f = fidelity(e, fp, ids[name], toks)
        bad = [t + 1 if i % 7 == 3 else t for i, t in enumerate(toks)]
        c = fidelity(e, fp, ids[name], bad)
        check(
            f >= FIDELITY_MIN and c < CONTROL_MAX,
            f"{name}: teacher-forced fidelity {f:.4f} >= {FIDELITY_MIN}; negative control (every 7th token replaced) {c:.3f} < {CONTROL_MAX}",
        )
    check(
        again == res["chat", "new"][0],
        "chat: the same request twice gives identical ids",
    )
    print("OK")


if __name__ == "__main__":
    main()
