"""G-near-tie: greedy MTP speculation against plain greedy decode on the 27B (design mtp-design.md §2, O2).

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tools/near_tie.py MODEL \
      --mtp 2 --kv bf16|int4_rot --prompt-files engine/src/*.cu [--prompt-len 385 4096] [--slices 3] [--tokens 256] \
      [--ctx-max 4608] [--out FILE]

Speculation verifies a row inside a (k + 1)-row step, and the kernels' arithmetic depends on the row count, so its
tokens can differ from plain decode's at a near-tie. For every cell the tool runs both and either reports the tokens
equal or judges the first divergence: the cycle that emitted it (the device cycle log gives its position P and the
rows it kept, so the verify row t), then on the plain run's state at P the logits of the verify step of k + 1 rows over
[anchor, accepted drafts ...] (row t) against the logits of the plain T = 1 step at P + t. The divergence is a
near-tie iff margin <= d[top1] + d[top2]: margin = the plain top-1 minus top-2 logit, d = |verify - plain| per
vocabulary entry, top1 / top2 the plain run's two best tokens (a flip needs the two logits to cross). The probe state
is the plain run's:
the speculative run's own state differs from it by the rounding history of every earlier verify step, so the tool also
lists the shift of the run's own verify row against the plain logits (`own_delta_top`, `own_delta_max`; by construction
it exceeds the margin, what it shows is its size against the single-step `delta_max`). One that fails the rule is
listed as FAIL: it is either a bug or history beyond the single-step shift; the comparison with the hand engine's MTP
(gate27.py --mtp) tells which, since warpc's cycles are bitwise the hand's.

k = 1 (T = 2) is judged differently: row 0 and row 1 are bitwise the T = 1 step's except where the attention is built
from other keys or another split geometry than the T = 1 step's (`exact_prefix`: a geometry change of the split
kernels, and on int4_rot every cycle from position 495 on, where row 0 reads one more key as int4 that the T = 1 step
reads from the exact ring); the run must be equal through every cycle before the first such cycle, and a divergence
after it is listed. Exit status 0 iff no cell fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

import prompts
import warpc
from backends import make_backend
from gate27 import GLUE_KERNELS


def split_geometry(ctx: int) -> tuple[int, int]:
    """The split-KV geometry (splits, keys per split) the hand attention kernel derives from a step's context length
    (engine/src/attn.cu split_params: >= 128 keys per split, at most 20 splits, chunks of 16 keys)."""
    ctx = max(ctx, 1)
    n = min(max(-(-ctx // 128), 1), 20)
    chunk = (-(-ctx // n) + 15) & ~15
    return -(-ctx // chunk), chunk


def geometry(position: int, rows: int, kv: str) -> tuple[int, ...]:
    """What the attention of the step of `rows` rows at `position` is built from: the split geometry of its last key's
    context and, on int4_rot, which keys the split kernels read (all but the last 496, kvrot.cuh RING_W, which the
    exact ring serves): a step of more rows moves the boundary, so a key the T = 1 step reads from the exact ring is
    read as int4 by row 0 of a T = 2 step."""
    ctx = position + rows
    if kv == "int4_rot":
        ctx = max(ctx - 496, 0)
        return (ctx, *split_geometry(ctx))
    return split_geometry(ctx)


def exact_prefix(kv: str, log: list[tuple[int, int]]) -> int:
    """How many tokens (the append's first included) a k = 1 run emits before its first cycle whose row 0 sees another
    split geometry than the T = 1 step it stands for: a verify row is attended with the geometry of the step's last
    key, so row 0 of that cycle can round differently from plain decode's (the hand engine does the same)."""
    emitted = 1
    for position, a in log:
        if geometry(position, 2, kv) != geometry(position, 1, kv):
            return emitted
        emitted += a
    return emitted


def _cycle_of(log: list[tuple[int, int]], index: int) -> tuple[int, int, int]:
    """(cycle, row t, first token index) of the token with output index `index` (index 0 is the append's)."""
    start = 1
    for c, (_, a) in enumerate(log):
        if index < start + a:
            return c, index - start, start
        start += a
    raise ValueError(f"token {index} is past the {start - 1} tokens of the cycle log")


def judge(m, prompt: list[int], out: list[int], log: list[tuple[int, int]], index: int, k: int) -> dict:
    """The verify row against the plain step for the first divergent token `index` of the speculative run `out`."""
    c, t, start = _cycle_of(log, index)
    position, _ = log[c]
    spec, rt = m.decoder, m._runtime
    spec.k, spec.cap = k, None  # the run's own verify row at P: replay its cycles, probe the next one without a commit
    m.reset()
    m.append(prompt)
    spec.reset_counters()
    spec.cycles(c)
    assert m.position == position and spec.anchor == out[start - 1], (m.position, position)
    rt.enqueue(k + 1, ("ids",), advance=False)
    own = rt.outputs["logits"][t].clone()
    spec.k = 0
    m.reset()
    m.append(prompt)
    if start - 1:
        m.decode(
            start - 1
        )  # the plain run's state at P: the cache holds prompt + out[:start - 1], out[start - 1] is the anchor
    assert m.position == position and int(rt.inputs["tokens"][0]) == out[start - 1], (m.position, position)
    T = k + 1
    toks = out[start - 1 : start + t] + [out[start - 1]] * (T - 1 - t)  # [anchor, accepted drafts ..., filler]
    saved = rt.inputs["tokens"].clone()
    rt.inputs["tokens"][:T].copy_(torch.tensor(toks[:T], dtype=torch.int32))
    rt.enqueue(T, ("ids",), advance=False)
    verify = rt.outputs["logits"][t].clone()
    rt.inputs["tokens"].copy_(saved)
    m.decode(t + 1)  # the plain T = 1 steps: the last one's logits are row t of the plain run
    plain = rt.logits.clone()
    top = plain.topk(2)
    margin = float(top.values[0] - top.values[1])
    delta = (verify - plain).abs()
    bound = float(delta[top.indices[0]] + delta[top.indices[1]])
    return {
        "cycle": c,
        "row": t,
        "position": position,
        "plain_top": top.indices.tolist(),
        "spec_token": out[index],
        "margin": margin,
        "delta_top": [float(delta[i]) for i in top.indices],
        "delta_max": float(delta.max()),
        "own_delta_top": [float((own - plain).abs()[i]) for i in top.indices],
        "own_delta_max": float((own - plain).abs().max()),
        "pass": margin <= bound,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--mtp", type=int, required=True, help="drafts per cycle k")
    ap.add_argument("--kv", default="bf16")
    ap.add_argument("--head", default="int4_refine")
    ap.add_argument("--ctx-max", type=int, default=4608)
    prompts.add_arguments(ap)
    prompts.add_slices(ap)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--host-tables", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    k = args.mtp
    backend = make_backend("composite", args.head, GLUE_KERNELS)
    m = warpc.compile(
        args.model, ctx_max=args.ctx_max, kv=args.kv, backend=backend, speculate=k, host_tables=args.host_tables
    )
    spec, rows, ok = m.decoder, [], True
    for length, s, ids in prompts.cells(args.model, args.prompt_files, args.prompt_len, args.slices):
        spec.k, spec.cap = 0, None
        plain = m.generate(ids, args.tokens)
        spec.k = k
        m.reset()
        first = m.append(ids)
        spec.reset_counters()
        out = [first, *m.decode(args.tokens - 1)]
        log = spec.cycle_log()
        stats = spec.stats
        row = {
            "length": length,
            "slice": s,
            "tokens_per_cycle": stats.emitted / stats.cycles,
            "accepted": stats.accepted,
        }
        index = next((i for i, (a, b) in enumerate(zip(out, plain, strict=True)) if a != b), None)
        if index is None:
            row["result"] = "equal"
        else:
            row |= {"result": "diverges", "index": index} | judge(m, ids, out, log, index, k)
            if k == 1:  # the exact mode: only a divergence before the first geometry change is a bug
                row["exact_prefix"] = exact_prefix(args.kv, log)
                row["pass"] = index >= row["exact_prefix"]
            ok &= row["pass"]
        rows.append(row)
        verdict = row["result"] if index is None else f"diverges at {index} (cycle {row['cycle']} row {row['row']})"
        extra = ""
        if index is not None:
            call = "near-tie" if row["pass"] else "FAIL"
            extra = f": margin {row['margin']:.3g} vs {sum(row['delta_top']):.3g} -> {call}"
            extra += f" (own shift {sum(row['own_delta_top']):.3g}, max {row['own_delta_max']:.3g})"
            if k == 1:
                call = call.replace("near-tie", "geometry")
                extra = f": exact through {row['exact_prefix']} tokens -> {call}" + extra
        print(f"near_tie k={k} kv={args.kv} len {length} slice {s}: {verdict}{extra}", flush=True)
    summary = {"mtp": k, "kv": args.kv, "cells": rows, "pass": ok}
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1))
    equal = sum(r["result"] == "equal" for r in rows)
    print(f"near_tie k={k}: {'PASS' if ok else 'FAIL'} ({equal} of {len(rows)} cells equal)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
