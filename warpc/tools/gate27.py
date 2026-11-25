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

--mtp K (greedy MTP speculation, k = K drafts per cycle; the oracle is the hand SpecEngine, `prefill_spec(chunk=8)`,
`cycles(K, 1)`, prompt lookup off): `reference --mtp K` records every cell under the natural acceptance and the force
patterns (--forces: cycle i keeps at most pattern[i % n] rows); `check --mtp K` (backend composite, default kernels
the three data-movement ones, so the hand's kernels do all model compute) judges, BITWISE: (G2) the drafts d1 .. dK and
the head's post-norm hidden rows after the prefill and after each of the first 16 cycles; (G3) on slice 0 the verify
logits and ids of T = 2, 3, 4, 8 rows over [anchor, drafts ...]; (G4) every cycle's emitted tokens and natural accepted
count until --tokens ids exist. A cell passes iff all of them are equal. --window N (MTP): the window-prefill cells: the
oracle prefills with `FastPrefill(seg=4096, rows=2048, fold=False).prefill_spec` and warpc with `compile(window_rows=N)`
cut at `plan_windows`, so that every prompt chunk of more than 8 rows commits its MTP pairs through the window variant
(--ctx-max must hold the longest prompt + 256 tokens).
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
    mtp = f"-mtp{args.mtp}{'-window' if args.window else ''}" if args.mtp else ""
    return args.out / f"hand-{args.kv}-{args.head}{mtp}.pt"


GLUE_KERNELS = "g_select_row,g_take,g_concat"


def reference(args) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    kind = "mtp-traces" if args.mtp else "traces"
    mode = [kind, "--prompt-files", *map(str, args.prompt_files), "--prompt-len", *map(str, args.prompt_len)]
    mode += ["--slices", str(args.slices), "--tokens", str(args.tokens)]
    mode += ["--mtp", str(args.mtp), "--forces", *args.forces] if args.mtp else ["--trace-steps", str(args.trace_steps)]
    mode += ["--fast-prefill"] if args.window else []
    mode += ["--trace-out", str(_reference_file(args))]
    result = swift_worker.spawn(
        args.model,
        kv=args.kv,
        head=args.head,
        ctx_max=args.ctx_max,
        mode=mode,
        engine_dir=args.engine_dir,
        host_tables=args.host_tables,
    )
    _reference_file(args).with_suffix(".json").write_text(json.dumps(result))
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


def _hidden_equal(spec, ref: torch.Tensor, k: int) -> bool:
    return bool(torch.equal(spec.draft_hidden[:k].cpu(), ref))


def _probe(m, probe: dict) -> dict:
    """The target's verify step of T rows over the hand's probe tokens, advance=False (nothing committed)."""
    rt, T = m._runtime, probe["T"]
    saved = rt.inputs["tokens"].clone()
    rt.inputs["tokens"][:T].copy_(torch.tensor(probe["toks"], dtype=torch.int32))
    rt.enqueue(T, ("ids",), advance=False)
    ids, logits = rt.outputs["ids"][:T].tolist(), rt.outputs["logits"][:T].cpu()
    rt.inputs["tokens"].copy_(saved)
    return {"T": T, "ids_equal": ids == probe["ids"], "logits_equal": bool(torch.equal(logits, probe["logits"]))}


def mtp_cell(m, cell: dict, k: int, tokens: int, window_rows: int = 0) -> dict:
    """warpc against one hand MTP cell (swift_worker.mtp_traces): G2, G3 and G4, each bitwise."""
    spec = m.decoder
    m.reset()
    spec.k, spec.cap = k, cell["force"]
    from warpc.windows import plan_windows

    ends = plan_windows(cell["prompt"], rows=window_rows) if window_rows else None
    first = m.append(cell["prompt"], ends=ends)
    r = {"length": cell["length"], "slice": cell["slice"], "force": cell["force"], "first": first == cell["first"]}
    r["start_drafts"] = spec.drafts == cell["start"]["drafts"]
    r["start_hidden"] = _hidden_equal(spec, cell["start"]["hr"], k)
    r["start"] = r["start_drafts"] and r["start_hidden"]
    r["probes"] = [_probe(m, p) for p in cell.get("probes", [])]
    out, last, bad, wrong = [first], [0] * (k + 1), [], []
    for i, ref in enumerate(cell["cycles"]):
        new = spec.cycles(1)
        hist = list(spec.stats.accepted)
        natural = next((j for j in range(k + 1) if hist[j] != last[j]), None)
        last = hist
        tokens_ok = new == ref["new"] and natural == ref["m"]
        state_ok = i >= 16 or (spec.drafts == ref["drafts"] and _hidden_equal(spec, ref["hr"], k))
        if not (tokens_ok and state_ok):
            bad.append(i)
        if not tokens_ok:
            wrong.append(i)
        out += new
    r["cycles"], r["bad_cycles"], r["token_cycles_differ"] = len(cell["cycles"]), bad[:8], len(wrong)
    r["ids_equal"] = len(out) >= tokens
    r["pass"] = (
        r["first"]
        and r["start"]
        and not bad
        and all(p["ids_equal"] and p["logits_equal"] for p in r["probes"])
        and r["ids_equal"]
    )
    return r


def check_mtp(args) -> int:
    import warpc

    k, hand = args.mtp, torch.load(_reference_file(args))
    backend = make_backend(args.backend, args.head, args.kernels or GLUE_KERNELS, bool(args.window))
    t0 = time.perf_counter()
    m = warpc.compile(
        args.model,
        ctx_max=args.ctx_max,
        rows_max=8,
        kv=args.kv,
        backend=backend,
        speculate=k,
        host_tables=args.host_tables,
        window_rows=args.window,
    )
    print(f"gate27: {backend.name} MTP k={k}: {m.plan.summary()} (compile {time.perf_counter() - t0:.0f} s)")
    rows, ok = [], True
    for cell in hand:
        r = mtp_cell(m, cell, k, args.tokens, args.window)
        rows.append(r)
        ok &= r["pass"]
        probes = "".join("=" if p["ids_equal"] and p["logits_equal"] else "X" for p in r["probes"]) or "-"
        verdict = "BITWISE" if r["pass"] else "FAIL"
        print(
            f"  len {r['length']} slice {r['slice']} force {r['force']}: first {r['first']} start {r['start']}"
            f" (drafts {r['start_drafts']}, hidden {r['start_hidden']}) probes[T=2,3,4,8] {probes}"
            f" cycles {r['cycles']} bad {r['bad_cycles']} (emitted tokens or accepted count differ in"
            f" {r['token_cycles_differ']}) -> {verdict}"
        )
    summary = {"plan": m.plan.summary(), "kv": args.kv, "head": args.head, "mtp": k, "cells": rows, "pass": ok}
    tag = f"mtp{k}{'-window' if args.window else ''}"
    (args.out / f"check-{args.kv}-{args.head}-{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"gate27 MTP k={k}: {'PASS (all cells BITWISE)' if ok else 'FAIL'}")
    return 0 if ok else 1


def check(args) -> int:
    if args.mtp:
        return check_mtp(args)
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
        out = [m.append(cell["prompt"])]
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
    ap.add_argument(
        "--kernels", help="composite: the warpc kernels admitted, comma-separated (default: all; MTP: glue)"
    )
    ap.add_argument("--mtp", type=int, default=0, help="speculative MTP gates with K drafts per cycle (0: plain gates)")
    ap.add_argument("--forces", nargs="*", default=["1", "2,1,3", "1,3", "3,1,1,2"], help="MTP: forced acceptance")
    ap.add_argument("--window", type=int, default=0, help="MTP: window prefill of up to N rows (0: 8-row)")
    ap.add_argument("--host-tables", action="store_true", help="MTP check: embedding and rescoring head in host memory")
    ap.add_argument("--tag", help="name of the verdict file (default: from --backend and --kernels)")
    args = ap.parse_args()
    if args.mode == "reference":
        reference(args)
        return
    sys.exit(check(args))


if __name__ == "__main__":
    main()
