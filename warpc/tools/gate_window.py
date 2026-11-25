"""The window-prefill gate on the 27B checkpoint (longctx design G1): warpc's window plan against the hand engine's
`FastPrefill(fold=False)`, bit for bit.

  # WARPC_LEGACY_DIR: a v0.12.0 checkout (hand engine, prompt corpus); WARPC_SWIFTMK_LIB: its libswiftmk.so
  # once per KV format: the hand engine's records of every cell (one process, ~1 minute to load)
  uv run --package warpc python warpc/tools/gate_window.py reference MODEL --kv int4_rot \
      --prompt-files "$WARPC_LEGACY_DIR"/engine/src/*.cu --out DIR
  # warpc over the same cells, judged against DIR
  uv run --package warpc python warpc/tools/gate_window.py check MODEL --kv int4_rot \
      --prompt-files "$WARPC_LEGACY_DIR"/engine/src/*.cu --out DIR

Cells: --prompt-len (385, 4096, 16384) x --slices (3) disjoint slices of the text; ctx_max fixed by --ctx-max so that
both sides have the same KV layout. A cell records, after the prefill and the head step on the last prompt token: sha256
of the raw KV buffer of each of the 16 attention layers (int4_rot: codes, meta, sink and ring), of each DeltaNet layer's
fp32 state and of the conv state at the ring cursor, the fp32 logits of the head step, and --tokens greedy ids. A cell
passes iff every digest, the logits (torch.equal) and the ids are equal. Hand: `FastPrefill(fold=False, seg=4096,
rows=2048)`, no layer-major grouping (bit-identical to it by construction). Exit status 0 iff every cell passes."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import torch

import legacy
import prompts

KV_FP8 = {"bf16": 0, "fp8": 1, "int4_rot": 2}


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def cells(args):
    for length in args.prompt_len:
        for k in range(args.slices):
            yield length, k, prompts.prompt_ids(args.model, args.prompt_files, length, k * length)


def record_path(args) -> Path:
    return args.out / f"window-hand-{args.kv}.pt"


# ---- the hand engine


def reference(args) -> None:
    os.environ["SWIFT_MODEL"] = str(args.model)
    sys.path.insert(0, str(legacy.engine_dir()))
    import engine as hand
    from prefill import FastPrefill

    e = hand.Engine(ctx_max=args.ctx_max, kv_fp8=KV_FP8[args.kv], head="bf16", tmax=8, state_slots=1, verbose=False)
    fp = FastPrefill(e, seg=4096, rows=2048, fold=False, grp_rows=4096)
    out = {}
    for length, k, ids in cells(args):
        e.reset()
        t0 = time.perf_counter()
        first = fp.prefill(ids)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        cur = int(fp.ds[1])
        rec = {"kv": [], "rec": [], "conv": []}
        from prefill import _view

        for ai in range(16):
            rec["kv"].append(digest(_view(fp.kv_base + ai * fp.kv_layer_bytes, fp.kv_layer_bytes, torch.uint8)))
        for di in range(48):
            rec["rec"].append(digest(fp.dn_state[di, cur % fp.nss]))
            rec["conv"].append(digest(fp.dn_conv[di, cur]))
        rec["logits"] = e.logits(1)[0].cpu()
        rec["ids"] = [first, *e.decode(args.tokens - 1)]
        rec["prefill_s"] = prefill_s
        out[f"{length}.{k}"] = rec
        print(f"hand {length}.{k}: prefill {prefill_s:.2f}s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"ctx_max": args.ctx_max, "kv": args.kv, "cells": out}, record_path(args))
    print(f"gate_window: hand records of {len(out)} cells -> {record_path(args)}")


# ---- warpc


def check(args) -> int:
    import warpc
    from warpc.swiftmk import SwiftmkBackend
    from warpc.windows import plan_windows

    ref = torch.load(record_path(args))
    if ref["ctx_max"] != args.ctx_max or ref["kv"] != args.kv:
        raise SystemExit(f"{record_path(args)} was recorded for ctx_max {ref['ctx_max']}, kv {ref['kv']}")
    backend = SwiftmkBackend("bf16", window=True)
    model = warpc.compile(args.model, ctx_max=args.ctx_max, kv=args.kv, backend=backend, window_rows=args.window_rows)
    rt = model.decoder
    failed = 0
    for length, k, ids in cells(args):
        want = ref["cells"][f"{length}.{k}"]
        rt.reset()
        t0 = time.perf_counter()
        first = rt.append(ids, ends=plan_windows(ids, rows=args.window_rows))
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0
        got = {"kv": [], "rec": [], "conv": []}
        cur = int(rt._executor._cur)
        for name, t in rt._states.items():
            decl = rt._graph.states[name]
            if decl.kind == "kv":
                got["kv"].append(digest(t))
            elif decl.kind == "recurrent":
                hv, dk, dv = decl.shape
                got["rec"].append(digest(t[: hv * dk * dv * 4].view(torch.float32)))
            elif decl.kind == "conv":
                slots = rt._executor.slots
                got["conv"].append(digest(t.view(torch.bfloat16).view(slots, -1)[cur % slots]))
        logits = model.logits.cpu()
        ids_out = [first, *model.decode(args.tokens - 1)]
        bad = [
            f"{kind}[{i}]"
            for kind in ("kv", "rec", "conv")
            for i, (a, b) in enumerate(zip(got[kind], want[kind], strict=True))
            if a != b
        ]
        if not torch.equal(logits, want["logits"]):
            bad.append(f"logits (max |d| {float((logits - want['logits']).abs().max()):.3g})")
        if ids_out != want["ids"]:
            diff = next(i for i, (a, b) in enumerate(zip(ids_out, want["ids"], strict=True)) if a != b)
            bad.append(f"ids (first diff at {diff})")
        ok = not bad
        failed += not ok
        print(
            f"{'PASS' if ok else 'FAIL'} {length}.{k}: prefill {prefill_s:.2f}s (hand {want['prefill_s']:.2f}s, "
            f"ratio {prefill_s / want['prefill_s']:.3f}) {', '.join(bad[:6])}{' ...' if len(bad) > 6 else ''}",
            flush=True,
        )
    return 1 if failed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("reference", "check"))
    ap.add_argument("model", type=Path)
    ap.add_argument("--kv", choices=tuple(KV_FP8), default="int4_rot")
    ap.add_argument("--prompt-files", type=Path, nargs="+", required=True, help=prompts.PROMPT_FILES_HELP)
    ap.add_argument("--prompt-len", type=int, nargs="+", default=[385, 4096, 16384])
    ap.add_argument("--slices", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ctx-max", type=int, default=16896)
    ap.add_argument("--window-rows", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.mode == "reference":
        reference(args)
    else:
        sys.exit(check(args))


if __name__ == "__main__":
    main()
