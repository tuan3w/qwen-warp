"""The A/B/A/B speed protocol of the speculative gate (design mtp-design.md §4.1 G6, §7.1): warpc against the hand
engine, same prompts, same configuration, alternating processes so that clock and thermal drift hit both sides alike.

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tools/ab.py MODEL \
      --mtp 3 --kv int4_rot --head int4_refine --prompt-files engine/src/*.cu --prompt-len 385 4096 16384 \
      [--visits 2] [--repeats 5] [--tokens 256] [--host-tables] [--forge-position] [--plain] [--wrap CMD ...] --out FILE

One visit = one `bench.py` process per side (the two hold 17 - 19 GB each and cannot share the card), each decoding
every
--prompt-len cell: a warm-up round, then --repeats timed rounds of --tokens - 1 tokens (the median of the rounds is the
visit's figure). Visits alternate warpc, hand, warpc, hand. The ratio of a cell is the mean over the visits of
warpc / hand tok/s. --plain adds the k = 0 rows of each side. --wrap runs every process under that command (a lease
wrapper): the processes still alternate. The speculative stage passes when every cell's ratio is >= 0.98 and at least
5 of 6 cells are >= 0.99 (the gate's rule; the cells of one invocation are those given). --forge-position (timing
only, both sides): each process prefills 385 tokens and moves the position to the cell's length, and the ratio is the
cycle time's, hand / warpc. Prints the table (tok/s of each visit, tokens per cycle, ratio) and writes it as JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "bench.py"


def run_side(args, side: str, mtp: int, tmp: Path, name: str) -> dict:
    """One bench.py process: `side` is the engine (warpc or swift-engine)."""
    out = tmp / f"{name}.json"
    cmd = [sys.executable, str(BENCH), side, str(args.model), "--kv", args.kv, "--head", args.head]
    cmd += ["--tokens", str(args.tokens), "--repeats", str(args.repeats), "--mtp", str(mtp), "--json", str(out)]
    cmd += ["--prompt-files", *map(str, args.prompt_files), "--prompt-len", *map(str, args.prompt_len)]
    if side == "warpc":
        cmd += ["--backend", "composite"]
    if args.host_tables:
        cmd += ["--host-tables"]
    if args.forge_position:
        cmd += ["--forge-position"]
    if args.window:
        cmd += ["--window"]
    subprocess.run([*args.wrap, *cmd], check=True)
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--kv", default="int4_rot")
    ap.add_argument("--head", default="int4_refine")
    ap.add_argument("--prompt-files", type=Path, nargs="+", required=True)
    ap.add_argument("--prompt-len", type=int, nargs="+", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--visits", type=int, default=2)
    ap.add_argument("--host-tables", action="store_true")
    ap.add_argument(
        "--forge-position",
        action="store_true",
        help="timing only, both sides: prefill 385 tokens, move the position to --prompt-len; the ratio is then the"
        " cycle time's (the acceptance over a zero cache means nothing)",
    )
    ap.add_argument(
        "--window", action="store_true", help="both sides prefill with the window plan (warpc) / FastPrefill (hand)"
    )
    ap.add_argument("--plain", action="store_true", help="also the k = 0 rows")
    ap.add_argument("--wrap", nargs="*", default=[], help="a command every process runs under (e.g. a lease)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    modes = [args.mtp, *([0] if args.plain else [])]
    sides = {"warpc": "warpc", "hand": "swift-engine"}
    runs: dict[tuple[int, str], list[dict]] = {(k, s): [] for k in modes for s in sides}
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for visit in range(args.visits):
            for k in modes:
                for s, engine in sides.items():
                    print(f"ab: visit {visit + 1}/{args.visits}: {s} k={k}", flush=True)
                    runs[k, s].append(run_side(args, engine, k, tmp, f"{s}-{k}-{visit}"))
    table = {}
    for k in modes:
        for length in map(str, args.prompt_len):
            cell = {}
            for s in sides:
                visits = [r["cells"][length] for r in runs[k, s]]
                tok_s = [v["tok_s"] for v in visits]
                tpc = [v.get("tokens_per_cycle") for v in visits]
                cell[s] = {
                    "tok_s": tok_s,
                    "tokens_per_cycle": tpc,
                    "cycle_ms": [None if c is None else 1000 * c / r for c, r in zip(tpc, tok_s, strict=True)],
                    "clock_mhz": [v["clocks"]["sm_mhz"] for v in visits],
                }
            w, h = cell["warpc"], cell["hand"]
            ratios = [a / b for a, b in zip(w["tok_s"], h["tok_s"], strict=True)]
            if args.forge_position and k:  # same cycle structure on both sides, acceptance over zeros is not
                ratios = [b / a for a, b in zip(w["cycle_ms"], h["cycle_ms"], strict=True)]
            cell["ratio"], cell["ratio_mean"] = ratios, statistics.mean(ratios)
            table[f"k{k}/{length}"] = cell
    memory = {
        s: [{k: r.get(k) for k in ("device_gib_after_load", "device_gib_after_decode", "peak_gib")} for r in rs]
        for (k, s), rs in runs.items()
        if k == args.mtp
    }
    summary = {
        "mtp": args.mtp,
        "kv": args.kv,
        "head": args.head,
        "visits": args.visits,
        "cells": table,
        "memory": memory,
    }
    spec = [c["ratio_mean"] for key, c in table.items() if key.startswith(f"k{args.mtp}/")]
    summary["pass"] = bool(spec) and min(spec) >= 0.98 and sum(r >= 0.99 for r in spec) >= len(spec) - 1
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"{'cell':>12} {'warpc tok/s':>20} {'hand tok/s':>20} {'tok/cyc w':>10} {'tok/cyc h':>10} {'ratio':>7}")
    for key, c in table.items():
        w, h = c["warpc"], c["hand"]
        tpc = lambda x: "-" if x[0] is None else f"{statistics.mean(x):.3f}"  # noqa: E731
        print(
            f"{key:>12} {'/'.join(f'{v:.1f}' for v in w['tok_s']):>20} {'/'.join(f'{v:.1f}' for v in h['tok_s']):>20}"
            f" {tpc(w['tokens_per_cycle']):>10} {tpc(h['tokens_per_cycle']):>10} {c['ratio_mean']:>7.4f}"
        )
    print(f"ab: G6 {'PASS' if summary['pass'] else 'NOT MET'} (every cell >= 0.98 and all but one >= 0.99)")


if __name__ == "__main__":
    main()
