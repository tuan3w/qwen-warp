#!/usr/bin/env python3
"""Decode speed of the MTP + prompt-lookup cycle with the chain extension and the wide verify switched off (stock policy) and on, interleaved in one process.

  uv run scripts/bench_decode.py --set dev        # 6 short chat prompts + 2 long-context prompts (a code summary, a copy-heavy rewrite)
  uv run scripts/bench_decode.py --set held       # 4 other short prompts + a 12k-token summary and a rewrite of a different file
  uv run scripts/bench_decode.py --reps 5 --tokens 256

Engine level (no server), greedy, int4 KV cache, host tables, `SpecEngine(tmax=16)`, fast prefill; the clock covers decode only (prefill excluded), generation stops at the end token.
Per prompt the median over the repetitions of each arm; the score is the geometric mean over the prompts.  Arms: `stock` = make_pld(ext=0, wide=0), `new` = make_pld() (the defaults).
The two arms produce the same tokens up to rounding-level flips (the attention split depends on the verify width): the table prints the common prefix of the two streams as a fraction of the first 192 tokens (a cascade after the first near-tie flip, not a quality measure).
The long prompts are built from engine/src/*.cu of the checked-out tree, so their length and acceptance move a little with the code.
"""

import argparse
import math
import os
import statistics as S
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from prefill import FastPrefill  # noqa: E402
from engine import TMAX_WIDE  # noqa: E402
from spec import SpecEngine, make_pld  # noqa: E402

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))

SHORT = {
    "dev": [
        (
            "Write a Python implementation of an LRU cache class with get/put in O(1), then unit tests with pytest.",
            False,
        ),
        (
            "Explain how a hash table handles collisions, and when you would choose open addressing over chaining.",
            False,
        ),
        (
            "A train leaves city A at 60 km/h and another leaves city B, 300 km away, at 90 km/h toward A, one hour later. When and where do they meet? Show your steps.",
            True,
        ),
        (
            "Write a SQL query to find the top 3 customers by total order value in each country, given tables customers(id, name, country) and orders(id, customer_id, total).",
            False,
        ),
        (
            "Write a short, friendly email declining a meeting invitation because of a scheduling conflict, and suggest two alternative times.",
            False,
        ),
        (
            "Implement a thread-safe bounded blocking queue in Java with a producer/consumer demo.",
            False,
        ),
    ],
    "held": [
        (
            "Write a Rust function that parses a CSV line with quoted fields and escaped quotes, plus tests.",
            False,
        ),
        (
            "Summarize the main causes of the French Revolution in a few paragraphs.",
            False,
        ),
        (
            "What is the derivative of x^x? Derive it carefully, then check numerically for x=2.",
            True,
        ),
        (
            "Translate to French and then to German: 'The weather is lovely today, so we decided to take a long walk along the river.'",
            False,
        ),
    ],
}
LONG_FILES = {
    "dev": ("gemv.cu", "head_q.cu", "mtp.cu"),
    "held": (
        "attn.cu",
        "batch.cu",
        "deltanet.cu",
        "dflash.cu",
        "dspine.cu",
        "engine.cu",
    ),
}


def prompts(which):
    src = "\n\n".join(
        (ROOT / "engine" / "src" / f).read_text() for f in LONG_FILES[which]
    )
    if which == "dev":
        long = [
            (
                "Here is a CUDA source file:\n\n"
                + src[:15000]
                + "\n\nExplain the structure of this file and list the three most performance-critical functions.",
                False,
            ),
            (
                "Add a short comment above every function in the following code, keep the code itself unchanged:\n\n"
                + src[20000:22500],
                False,
            ),
        ]
    else:
        long = [
            (
                "Here is a CUDA source file:\n\n"
                + src[:36000]
                + "\n\nSummarise what each kernel in this file does and point out one potential performance issue.",
                False,
            ),
            (
                "Rewrite the following code with docstrings added to every function, keeping the logic identical:\n\n"
                + src[1000:3200],
                False,
            ),
        ]
    return SHORT[which] + long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("dev", "held"), default="dev")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=320)
    ap.add_argument("--ngram", type=int, default=2, help="prompt-lookup n-gram length of both arms (2 = the server default make_pld(); 4 = what scripts/bench_t14.py calls shipped)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}

    def enc(p, think):
        o = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=think,
        )
        return list(o["input_ids"] if hasattr(o, "keys") else o)

    ids = [enc(p, t) for p, t in prompts(a.set)]
    e = SpecEngine(
        ctx_max=max(map(len, ids)) + a.tokens + 256,
        kv_fp8=2,
        verbose=False,
        tmax=TMAX_WIDE,
        host_tables=True,
    )
    fp = FastPrefill(e)
    arms = {"stock": make_pld(n=a.ngram, ext=0, wide=0), "new": make_pld(n=a.ngram)}

    def run(arm, x):
        e.enable_pld(arms[arm])
        e.reset()
        e.set_force(None)
        first = fp.prefill_spec(x, 3)
        e.pld_begin(list(x) + [first])
        toks, ss = [first], None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(toks) < a.tokens:
            new, ss = e.cycles_pld(8)
            toks += new
            if any(t in eos for t in new):
                break
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        toks = toks[: a.tokens]
        cut = next((i for i, t in enumerate(toks) if t in eos), None)
        return (toks if cut is None else toks[: cut + 1]), dt, ss[12] / max(ss[13], 1)

    for arm in arms:  # warm-up: graph capture for every verify width
        for x in ids:
            run(arm, x)
    res = {arm: [[] for _ in ids] for arm in arms}
    for r in range(a.reps):
        for i, x in enumerate(ids):
            for arm in arms if r % 2 == 0 else reversed(list(arms)):
                res[arm][i].append(run(arm, x))
    print(
        f"{'prompt':>6} {'ctx':>6} {'stock':>8} {'new':>8} {'ratio':>6} {'tok/cyc stock':>14} {'new':>6} {'agree192':>9}"
    )
    sp = {arm: [] for arm in arms}
    for i, x in enumerate(ids):
        row = {}
        for arm in arms:
            r = res[arm][i]
            row[arm] = (
                (len(r[0][0]) - 1) / S.median(t[1] for t in r),
                S.mean(t[2] for t in r),
                r[0][0],
            )
            sp[arm].append(row[arm][0])
        s, n = row["stock"][2], row["new"][2]
        m = min(len(s), len(n), 192)
        k = next((j for j in range(m) if s[j] != n[j]), m)
        print(
            f"{i:>6} {len(x):>6} {row['stock'][0]:8.1f} {row['new'][0]:8.1f} {row['new'][0] / row['stock'][0]:6.3f} {row['stock'][1]:14.2f} {row['new'][1]:6.2f} {k / m:9.3f}"
        )
    gm = {arm: math.exp(sum(map(math.log, v)) / len(v)) for arm, v in sp.items()}
    print(
        f"geomean tok/s: stock {gm['stock']:.1f}  new {gm['new']:.1f}  ratio {gm['new'] / gm['stock']:.3f}x   ({a.set}, {a.reps} reps, interleaved, median per prompt)"
    )


if __name__ == "__main__":
    main()
