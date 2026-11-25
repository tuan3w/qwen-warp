"""MTP at long context on the 27B (design mtp-design.md G-long): memory against the budget table, decode speed and
acceptance, and the agent-turn flow on the `Speculator` (snapshot, restore, append after a decode).

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tools/mtp_long.py MODEL \
      --prompt-files engine/src/*.cu --prompt-len 16384 [--kv int4_rot] [--mtp 3] [--host-tables] [--forge-position] \
      [--no-window] [--ctx-max N] [--tokens 256] [--tool 600] [--out FILE]

One process per prompt length (the 27B takes the card). The prompt is prefilled with the window plan (`plan_windows`,
`window_rows` 4096) or, with --forge-position, as 385 real tokens under a moved position (timing and memory only: the
cache below it is zeros, so acceptance and the turn flow mean nothing there; --no-window then also leaves out the
window prefill's buffers, the decode-only memory the design's budget table is about). Reported: the device memory the
driver shows (total - free) after the load, after the prefill and after the decodes, torch's peak, the prefill time,
decode tok/s and tokens per cycle of two rounds of --tokens, and, for a real prompt, the turn flow: snapshot at the turn
boundary, decode 64, restore, decode 64 (must equal the first 64), then append [anchor] + --tool tokens, decode 64,
restore the snapshot again and repeat the append and the decode (must equal): the resume is bitwise on the Speculator.
Exit status 0 iff both equalities hold."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import prompts
import warpc
from backends import make_backend
from gate27 import GLUE_KERNELS
from warpc.windows import plan_windows

GIB = 2**30
TURN = 64


def used_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / GIB


def decode_round(m, n: int) -> dict:
    spec = m.decoder
    before = spec.stats
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    m.decode(n)
    dt = time.perf_counter() - t0
    after = spec.stats
    cycles, emitted = after.cycles - before.cycles, after.emitted - before.emitted
    return {"tok_s": n / dt, "tokens_per_cycle": emitted / cycles, "cycle_ms": 1000 * dt / cycles}


def turn_flow(m, ids: list[int], tool: list[int]) -> dict:
    """The agent turn on the Speculator; the equalities are bitwise."""
    snap = m.snapshot()
    first = m.decode(TURN)
    anchor = m.decoder.anchor
    appended = m.append([anchor, *tool])
    second = m.decode(TURN)
    m.restore(snap)
    again = m.decode(TURN)
    m.append([m.decoder.anchor, *tool])
    again_second = m.decode(TURN)
    return {
        "restore_equal": first == again,
        "append_equal": second == again_second,
        "snapshot_mib": sum(v.numel() * v.element_size() for v in snap.states.values()) / 2**20,
        "position_after_turn": m.position,
        "first_after_append": appended,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path)
    prompts.add_arguments(ap)
    ap.add_argument("--kv", default="int4_rot")
    ap.add_argument("--head", default="int4_refine")
    ap.add_argument("--mtp", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--tool", type=int, default=600, help="tokens of the tool result the turn appends")
    ap.add_argument("--ctx-max", type=int)
    ap.add_argument("--host-tables", action="store_true")
    ap.add_argument("--forge-position", action="store_true")
    ap.add_argument("--no-window", action="store_true", help="with --forge-position: compile without the window plan")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    (length,) = args.prompt_len
    ctx_max = args.ctx_max or length + 1 + 2 * args.tokens + 4 * (TURN + args.tool) + 64
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    m = warpc.compile(
        args.model,
        ctx_max=ctx_max,
        kv=args.kv,
        backend=make_backend("composite", args.head, GLUE_KERNELS, window=not args.no_window),
        speculate=args.mtp,
        host_tables=args.host_tables,
        window_rows=0 if args.no_window else 4096,
    )
    result = {"prompt_len": length, "ctx_max": ctx_max, "compile_s": time.perf_counter() - t0, "kv": args.kv}
    result["device_gib_after_load"] = used_gib()
    spec, rt = m.decoder, m._runtime
    short = prompts.prompt_ids(args.model, args.prompt_files, 385 if args.forge_position else length + args.tool)
    ids, tool = (short, []) if args.forge_position else (short[:length], short[length:])
    t0 = time.perf_counter()
    m.append(ids, ends=None if args.forge_position else plan_windows(ids))
    if args.forge_position:  # timing and memory only: the position moves over the zero cache
        rt.pos.fill_(length)
        rt.position = length
        spec._last_hidden_pos = length
        spec.start(args.mtp)
    torch.cuda.synchronize()
    result["prefill_s"] = time.perf_counter() - t0
    result["device_gib_after_prefill"] = used_gib()
    decode_round(m, 64)  # warm-up
    result["rounds"] = [decode_round(m, args.tokens - 1) for _ in range(2)]
    result["device_gib_after_decode"] = used_gib()
    ok = True
    if not args.forge_position:
        result["turn"] = turn_flow(m, ids, tool)
        ok = result["turn"]["restore_equal"] and result["turn"]["append_equal"]
        result["device_gib_after_turn"] = used_gib()
    result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
    result["pass"] = ok
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
