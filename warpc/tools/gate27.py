"""The stage 2 end-to-end gate on the 27B checkpoint (design 5.1 #4): warpc kernels against the hand engine, per family.

  # once per (kv, head): the hand engine's traces of every cell, one process
  WARPC_SWIFTMK_LIB=... uv run --package warpc python warpc/tools/gate27.py reference MODEL --kv bf16 --head bf16 \
      --prompt-files engine/src/*.cu --out DIR
  # per backend configuration: warpc over the same cells, judged against DIR
  WARPC_SWIFTMK_LIB=... uv run --package warpc python warpc/tools/gate27.py check MODEL --kv bf16 --head bf16 \
      --prompt-files engine/src/*.cu --out DIR [--backend composite|cuda|swiftmk] [--kernels g_rmsnorm_zc,...]

Cells: prompt lengths --prompt-len (385 and 4096) x --slices (3) disjoint slices of the text, ctx_max 4608, 8-row steps,
the hand engine as Engine(tmax=8, state_slots=1) (swift_worker.py). A cell passes iff all --tokens (256) ids equal the
hand engine's and, for each of the first --trace-steps (16) head steps, max|d logits| < 0.5 * (top1 - top2) of the hand
logits; a step whose hand margin is below one bf16 ulp of its top logit is a near-tie, listed and not failed. A cell
whose logits are all torch.equal is reported BITWISE. Exit status 0 iff every cell passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import prompts
import swift_worker
from backends import BACKENDS, make_backend


def _reference_file(args) -> Path:
    return args.out / f"hand-{args.kv}-{args.head}.pt"


def reference(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    mode = ["traces", "--prompt-files", *map(str, args.prompt_files), "--prompt-len", *map(str, args.prompt_len)]
    mode += ["--slices", str(args.slices), "--tokens", str(args.tokens), "--trace-steps", str(args.trace_steps)]
    mode += ["--trace-out", str(_reference_file(args))]
    result = swift_worker.spawn(
        args.model, kv=args.kv, head=args.head, ctx_max=args.ctx_max, mode=mode, engine_dir=args.engine_dir
    )
    (args.out / f"hand-{args.kv}-{args.head}.json").write_text(json.dumps(result))
    print(f"gate27: hand traces of {result['cells']} cells -> {_reference_file(args)}, lib {result['lib_sha256'][:12]}")


def judge(hand: dict, ids: list[int], logits: torch.Tensor) -> dict:
    """The gate's verdict on one cell."""
    first = next((i for i, (a, b) in enumerate(zip(ids, hand["ids"], strict=True)) if a != b), None)
    steps = []
    for i, (x, y) in enumerate(zip(logits, hand["logits"], strict=True)):
        top2 = y.topk(2).values
        margin, top = float(top2[0] - top2[1]), float(top2[0])
        diff = float((x - y).abs().max())
        steps.append(
            {
                "step": i,
                "equal": bool(torch.equal(x, y)),
                "max_abs_diff": diff,
                "margin": margin,
                "near_tie": margin < abs(top) * 2**-7,  # one bf16 ulp of the top logit
                "ok": diff < 0.5 * margin,
            }
        )
    failed = [s for s in steps if not s["ok"] and not s["near_tie"]]
    return {
        "ids_equal": first is None,
        "first_differing_id": first,
        "bitwise": first is None and all(s["equal"] for s in steps),
        "worst_ratio": max((s["max_abs_diff"] / s["margin"] if s["margin"] else float("inf")) for s in steps),
        "near_ties": [s["step"] for s in steps if s["near_tie"] and not s["ok"]],
        "failed_steps": [s["step"] for s in failed],
        "pass": first is None and not failed,
    }


def check(args) -> int:
    import warpc

    hand = torch.load(_reference_file(args))
    backend = make_backend(args.backend, args.head, args.kernels)
    t0 = time.perf_counter()
    m = warpc.compile(args.model, ctx_max=args.ctx_max, rows_max=8, kv=args.kv, backend=backend)
    compile_s = time.perf_counter() - t0
    print(f"gate27: {backend.name} {args.kernels or ''}: {m.plan.summary()} (compile {compile_s:.0f} s)")
    rows, ok = [], True
    for cell in hand:
        m.reset()
        out = [m.prefill(cell["prompt"])]
        logits = [m.logits.clone()]
        for _ in range(1, args.trace_steps):
            out += m.decode(1)
            logits.append(m.logits.clone())
        out += m.decode(args.tokens - len(out))
        r = judge(cell, out, torch.stack(logits).cpu()) | {"length": cell["length"], "slice": cell["slice"]}
        rows.append(r)
        ok &= r["pass"]
        verdict = "BITWISE" if r["bitwise"] else ("pass" if r["pass"] else "FAIL")
        ids = "ids equal" if r["ids_equal"] else f"ids differ from {r['first_differing_id']}"
        print(
            f"  len {r['length']} slice {r['slice']}: {ids}, worst |d|/margin {r['worst_ratio']:.3g},"
            f" near-ties {r['near_ties']}, failed steps {r['failed_steps']} -> {verdict}"
        )
    tag = args.tag or f"{args.backend}-{(args.kernels or 'all').replace(',', '+')}"
    summary = {"plan": m.plan.summary(), "kv": args.kv, "head": args.head, "cells": rows, "pass": ok}
    (args.out / f"check-{args.kv}-{args.head}-{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"gate27: {'PASS' if ok else 'FAIL'}" + (" (all cells BITWISE)" if all(r["bitwise"] for r in rows) else ""))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("reference", "check"))
    ap.add_argument("model", type=Path, help="the checkpoint directory")
    ap.add_argument("--kv", choices=swift_worker.KV_FP8, default="bf16")
    ap.add_argument("--head", choices=swift_worker.HEADS, default="bf16")
    ap.add_argument("--ctx-max", type=int, default=4608)
    ap.add_argument("--prompt-files", type=Path, nargs="+", required=True, help="text the prompts are cut from")
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[385, 4096], help="prompt length(s) in tokens")
    prompts.add_slices(ap)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--trace-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True, help="directory of the hand traces and the verdicts")
    ap.add_argument("--engine-dir", type=Path, default=swift_worker.ENGINE_DIR)
    ap.add_argument("--backend", choices=BACKENDS, default="composite")
    ap.add_argument("--kernels", help="composite: the warpc kernels admitted, comma-separated (default: all)")
    ap.add_argument("--tag", help="name of the verdict file (default: from --backend and --kernels)")
    args = ap.parse_args()
    if args.mode == "reference":
        reference(args)
        return
    sys.exit(check(args))


if __name__ == "__main__":
    main()
