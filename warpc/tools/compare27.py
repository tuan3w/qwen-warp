"""Bit-identity of warpc's swiftmk backend with the hand engine on the 27B checkpoint (design gate §4.1).

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tools/compare27.py MODEL \
      --ctx-max N --kv bf16|fp8|int4_rot --head bf16|int4_refine --prompt-files F... --prompt-len L \
      [--tokens 256] [--trace-steps 16] --out DIR [--engine-dir DIR]

Both sides run the same configuration from this one argument set: the same library file (its sha256 is checked before
anything runs: the warpc side loads $WARPC_SWIFTMK_LIB, the hand side <engine-dir>/<$SMK_BUILD or build>/libswiftmk.so),
the same ctx_max, 8-row prompt steps then a head step (warpc rows_max 8, hand Engine(tmax=8, state_slots=1),
prefill(chunk=8)), the same head and KV format, the same prompt ids, EOS ignored. Each side is a process of its own,
run one after the other (each holds the 17 GB of weights).

Gate: (a) all --tokens ids equal; (b) the fp32 logits of each of the first --trace-steps head steps torch.equal. With
the int4_refine head, (b) compares the logits buffer the refine kernels leave (approximate int4 logits, exact on the
rescored candidates): equal only because both sides run the same kernels. Exit status 0 iff both hold. DIR receives
both traces and summary.json (first differing id, per-step logits equality and max |delta|).

--mtp K [--host-tables]: greedy MTP speculation with K drafts per cycle, the hand's SpecEngine as the oracle (the same
cell check as `gate27.py check --mtp`: drafts, the head's hidden rows, verify logits, every cycle's tokens and accepted
count, bitwise); the hand side runs first and warpc is judged against its trace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

import prompts
import swift_worker


def warpc_trace(args) -> None:
    import warpc
    from warpc.swiftmk import SwiftmkBackend, lib

    (length,) = args.prompt_len
    ids = prompts.prompt_ids(args.model, args.prompt_files, length)
    t0 = time.perf_counter()
    m = warpc.compile(args.model, ctx_max=args.ctx_max, rows_max=8, kv=args.kv, backend=SwiftmkBackend(args.head))
    compile_s = time.perf_counter() - t0
    out = [m.append(ids)]
    logits = [m.logits.clone()]
    for _ in range(1, args.trace_steps):
        out += m.decode(1)
        logits.append(m.logits.clone())
    out += m.decode(args.tokens - len(out))
    torch.save({"prompt": ids, "ids": out, "logits": torch.stack(logits).cpu()}, args.trace_out)
    library = lib.load()
    meta = {"lib": str(library.path), "lib_sha256": library.sha256, "plan": m.plan.summary(), "compile_s": compile_s}
    args.meta_out.write_text(json.dumps(meta))


def warpc_mtp(args) -> None:
    """The warpc child of --mtp: the hand's cell in --trace-out, the verdict and tokens per cycle to --meta-out."""
    import gate27
    import warpc
    from backends import make_backend
    from warpc.swiftmk import lib

    (cell,) = torch.load(args.trace_out)
    backend = make_backend("composite", args.head, gate27.GLUE_KERNELS)
    t0 = time.perf_counter()
    kwargs = {"ctx_max": args.ctx_max, "rows_max": 8, "kv": args.kv, "backend": backend, "speculate": args.mtp}
    m = warpc.compile(args.model, host_tables=args.host_tables, **kwargs)
    compile_s = time.perf_counter() - t0
    r = gate27.mtp_cell(m, cell, args.mtp, args.tokens)
    stats = m.decoder.stats
    meta = {"lib": str(lib.load().path), "lib_sha256": lib.load().sha256, "plan": m.plan.summary(), "cell": r}
    meta |= {"compile_s": compile_s, "tokens_per_cycle": stats.emitted / stats.cycles, "accepted": stats.accepted}
    args.meta_out.write_text(json.dumps(meta))


def compare_mtp(args, prompt: list[str]) -> int:
    th = args.out / "hand-mtp.pt"
    mode = ["mtp-traces", *prompt, "--slices", "1", "--tokens", str(args.tokens), "--mtp", str(args.mtp), "--forces"]
    hand = swift_worker.spawn(
        args.model,
        kv=args.kv,
        head=args.head,
        ctx_max=args.ctx_max,
        mode=[*mode, "--trace-out", str(th)],
        engine_dir=args.engine_dir,
        host_tables=args.host_tables,
    )
    common = [str(args.model), "--ctx-max", str(args.ctx_max), "--kv", args.kv, "--head", args.head, *prompt]
    side = ["--side", "warpc-mtp", "--trace-out", str(th), "--meta-out", str(args.out / "warpc.json")]
    mtp = ["--mtp", str(args.mtp), "--tokens", str(args.tokens), *(["--host-tables"] if args.host_tables else [])]
    subprocess.run([sys.executable, __file__, *common, *mtp, "--out", str(args.out), *side], check=True)
    meta = json.loads((args.out / "warpc.json").read_text())
    if meta["lib_sha256"] != hand["lib_sha256"]:
        sys.exit("compare27: the two sides loaded different libraries")
    r = meta["cell"]
    (args.out / "summary.json").write_text(json.dumps(meta, indent=1))
    probes = "".join("=" if p["ids_equal"] and p["logits_equal"] else "X" for p in r["probes"])
    print(
        f"compare27 MTP k={args.mtp} kv={args.kv} head={args.head} prompt={args.prompt_len[0]}: first {r['first']}"
        f" start {r['start']} verify probes [T=2,3,4,8] {probes} cycles {r['cycles']} bad {r['bad_cycles']},"
        f" {meta['tokens_per_cycle']:.3f} tokens per cycle -> {'BITWISE' if r['pass'] else 'MISMATCH'}"
    )
    return 0 if r["pass"] else 1


def compare(warpc_t: dict, hand_t: dict) -> dict:
    a, b = warpc_t["ids"], hand_t["ids"]
    first = next((i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y), None)
    steps = []
    for i, (x, y) in enumerate(zip(warpc_t["logits"], hand_t["logits"], strict=True)):
        steps.append({"step": i, "equal": bool(torch.equal(x, y)), "max_abs_diff": float((x - y).abs().max())})
    return {
        "prompt_equal": warpc_t["prompt"] == hand_t["prompt"],
        "ids_equal": a == b,
        "first_differing_id": first,
        "logits_equal": all(s["equal"] for s in steps),
        "steps": steps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="the checkpoint directory")
    ap.add_argument("--ctx-max", type=int, required=True)
    ap.add_argument("--kv", choices=swift_worker.KV_FP8, default="bf16")
    ap.add_argument("--head", choices=swift_worker.HEADS, default="bf16")
    prompts.add_arguments(ap)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--trace-steps", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--engine-dir", type=Path, default=swift_worker.ENGINE_DIR)
    ap.add_argument("--mtp", type=int, default=0, help="compare greedy MTP speculation with K drafts per cycle")
    ap.add_argument("--host-tables", action="store_true", help="--mtp: embedding and rescoring head in host memory")
    ap.add_argument("--side", choices=("warpc", "warpc-mtp"), help=argparse.SUPPRESS)  # internal: the warpc child
    ap.add_argument("--trace-out", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--meta-out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if len(args.prompt_len) != 1:
        ap.error("one --prompt-len per comparison")
    if args.side == "warpc":
        warpc_trace(args)
        return
    if args.side == "warpc-mtp":
        warpc_mtp(args)
        return
    from warpc.swiftmk import lib

    ours, theirs = Path(os.environ.get(lib.ENV, "")), swift_worker.library(args.engine_dir)
    if not ours.is_file() or not theirs.is_file():
        sys.exit(f"compare27: missing library: ${lib.ENV}={ours}, hand side {theirs}")
    sha_w, sha_h = lib.sha256(ours.resolve()), swift_worker.sha256(theirs)
    if sha_w != sha_h:
        sys.exit(f"compare27: library mismatch: {ours} {sha_w} vs {theirs} {sha_h} (rebuild or point both at one file)")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = ["--prompt-files", *map(str, args.prompt_files), "--prompt-len", str(args.prompt_len[0])]
    if args.mtp:
        sys.exit(compare_mtp(args, prompt))
    common = [str(args.model), "--ctx-max", str(args.ctx_max), "--kv", args.kv, "--head", args.head]
    steps = ["--tokens", str(args.tokens), "--trace-steps", str(args.trace_steps)]
    tw, th = args.out / "warpc.pt", args.out / "hand.pt"
    side = ["--side", "warpc", "--trace-out", str(tw), "--meta-out", str(args.out / "warpc.json")]
    t0 = time.perf_counter()
    subprocess.run([sys.executable, __file__, *common, *prompt, *steps, "--out", str(args.out), *side], check=True)
    t1 = time.perf_counter()
    hand = swift_worker.spawn(
        args.model,
        kv=args.kv,
        head=args.head,
        ctx_max=args.ctx_max,
        mode=["trace", *prompt, *steps, "--trace-out", str(th)],
        engine_dir=args.engine_dir,
    )
    t2 = time.perf_counter()
    meta = json.loads((args.out / "warpc.json").read_text())
    if meta["lib_sha256"] != hand["lib_sha256"]:
        sys.exit("compare27: the two sides loaded different libraries")
    summary = compare(torch.load(tw), torch.load(th))
    summary |= {
        "kv": args.kv,
        "head": args.head,
        "prompt_len": args.prompt_len[0],
        "ctx_max": args.ctx_max,
        "tokens": args.tokens,
        "lib_sha256": sha_w,
        "warpc": meta,
        "seconds": {"warpc": t1 - t0, "hand": t2 - t1},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    ok = summary["prompt_equal"] and summary["ids_equal"] and summary["logits_equal"]
    worst = max(s["max_abs_diff"] for s in summary["steps"])
    ids = "equal" if summary["ids_equal"] else f"differ from index {summary['first_differing_id']}"
    lg = "equal" if summary["logits_equal"] else f"differ (max |d| {worst:.3g})"
    cell = f"kv={args.kv} head={args.head} prompt={args.prompt_len[0]}"
    print(f"compare27 {cell}: ids {ids}, logits {lg} -> {'BITWISE' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
