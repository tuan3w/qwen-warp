"""The hand engine (engine/engine.py of this repository, or --engine-dir) on the 27B checkpoint, in a process of its
own, writing its result to --out. bench.py and compare27.py start it through `spawn`.

  uv run --package warpc python warpc/tools/swift_worker.py MODEL --kv bf16|fp8|int4_rot --head bf16|int4_refine \
      --ctx-max N --out FILE [--engine-dir DIR] MODE ...
    bench --prompt-files F... --prompt-len L... --tokens N --repeats R
    trace --prompt-files F... --prompt-len L --tokens N --trace-steps S
    traces --prompt-files F... --prompt-len L... --slices K --tokens N --trace-steps S  (every length x slice)
    ring

The engine is `Engine(ctx_max, kv_fp8, head, tmax=8, state_slots=1)` with 8-row prefill chunks: the configuration
warpc's swiftmk backend reproduces (rows_max 8, one DeltaNet state slot). The library it loads is
<engine-dir>/<$SMK_BUILD or build>/libswiftmk.so; its sha256 is in every result.

bench: see bench.py (decode rounds continuing one prefill per prompt length). trace: the ids of a greedy continuation
(EOS ignored) and the fp32 logits of its first S head steps, saved with torch.save. ring: the DeltaNet ring cursors
(engine_impl.h ds[DS_CUR], ds[DS_PEND]) after each step of T = 1..8 rows, three times over.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import child
import measure
import prompts

KV_FP8 = {"bf16": 0, "fp8": 1, "int4_rot": 2}
HEADS = {"bf16": "bf16", "int4_refine": "int4"}
KV_BYTES_PER_TOKEN = {"bf16": 64 * 1024, "fp8": 32 * 1024, "int4_rot": 16896}  # 16 attention layers
DS_CUR, DS_PEND, DS_N = 1, 35, 37
SHORT = 385  # the real prompt of a forged-position cell (bench.py --forge-position)
ENGINE_DIR = Path(__file__).resolve().parents[2] / "engine"


def library(engine_dir: Path) -> Path:
    return engine_dir / os.environ.get("SMK_BUILD", "build") / "libswiftmk.so"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spawn(
    model: Path,
    *,
    kv: str,
    head: str,
    ctx_max: int,
    mode: list[str],
    engine_dir: Path = ENGINE_DIR,
    host_tables: bool = False,
) -> dict:
    """Run this file with the mode arguments `mode`: its JSON result, or RuntimeError with why it failed."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        command = [sys.executable, __file__, str(model), "--kv", kv, "--head", head, "--ctx-max", str(ctx_max)]
        command += ["--engine-dir", str(engine_dir), "--out", str(out), *(["--host-tables"] if host_tables else [])]
        command += mode
        if error := child.run(command):
            raise RuntimeError(f"hand engine: {error}")
        return json.loads(out.read_text())


def engine(args):
    os.environ["SWIFT_MODEL"] = str(args.model)  # read by the hand engine's weights.py at import
    sys.path.insert(0, str(args.engine_dir))
    kv = KV_FP8[args.kv]
    if getattr(args, "mtp", 0):
        # the hand engine's speculative decoder: the MTP head drafting k tokens, int4 head, one DeltaNet state slot
        # (bit-identical to the full ring, deltanet.h), the tables in pinned host memory with --host-tables
        from spec import SpecEngine

        e = SpecEngine(
            ctx_max=args.ctx_max, kv_fp8=kv, tmax=8, state_slots=1, host_tables=args.host_tables, verbose=False
        )
        e.disable_pld()
        return e
    import engine as hand

    return hand.Engine(ctx_max=args.ctx_max, kv_fp8=kv, head=HEADS[args.head], tmax=8, state_slots=1, verbose=False)


class _DeviceBuffer:
    """A raw device pointer as a torch tensor (the CUDA array interface)."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 2}


def forge_position(e, position: int, k: int) -> None:
    """Timing only: move the engine's position (the device scalar ds[0]) over the cache it has, whose rows below
    stay as the short prompt left them (zeros beyond), as bench.py --forge-position does for warpc; a speculative
    engine then drafts again from there. Acceptance is meaningless afterwards, the cycle time is not."""
    import ctypes as C

    import smk
    import torch

    e.lib.smkp_engine_ptr.argtypes, e.lib.smkp_engine_ptr.restype = [C.c_void_p, C.c_int], C.c_void_p
    torch.cuda.synchronize()
    ds = torch.as_tensor(_DeviceBuffer(e.lib.smkp_engine_ptr(e.h, 0), DS_N * 4), device="cuda").view(torch.int32)
    ds[0] = position
    torch.cuda.synchronize()
    e.pos = position
    if k:
        smk.check(e.lib.smk_spec_start(e.sp, k, e.st), "spec_start")
        torch.cuda.synchronize()


def fast_prefill(e, args):
    """The hand engine's window prefill (the 8-row steps take minutes beyond 32k), as gate_window.py runs it: None
    without --fast-prefill."""
    if not args.fast_prefill:
        return None
    from prefill import FastPrefill

    return FastPrefill(e, seg=4096, rows=2048, fold=False, grp_rows=4096)


def bench(e, args) -> dict:
    fast = fast_prefill(e, args)
    cells, n, k = {}, args.tokens - 1, args.mtp
    state = {}
    for length in args.prompt_len:
        ids = prompts.prompt_ids(args.model, args.prompt_files, SHORT if args.forge_position else length)
        e.reset()
        t0 = time.perf_counter()
        if fast is not None and not args.forge_position:
            fast.prefill_spec(ids, k=k) if k else fast.prefill(ids)
        elif k:
            e.prefill_spec(ids, k=k, chunk=8)
        else:
            e.prefill(ids, chunk=8)
        if args.forge_position:
            forge_position(e, length, k)
        prefill_s = time.perf_counter() - t0

        def plain() -> int:
            e.decode(n)
            return n

        def spec() -> int:
            got = 0
            while got < n:  # 8 draft + verify cycles per host round trip
                new, state["ss"] = e.cycles(k, 8)
                got += len(new)
            return got

        def replay_events() -> tuple[float, int]:
            """GPU time of n single-token head steps (the graph smk_engine_decode replays)."""
            s = measure.replay_seconds(lambda: e.lib.smk_engine_step(e.h, 1, 1, e.st), n)
            e.pos += n
            return s, n

        cell = measure.rounds(spec if k else plain, args.repeats, None if k else replay_events)
        if k:  # device counters since the prefill (the warm-up round included), as warpc's SpecStats
            ss = state["ss"]
            cell |= {"tokens_per_cycle": ss[12] / ss[13], "cycles": ss[13], "accepted": ss[16 : 17 + k]}
        cells[str(length)] = {**cell, "prefill_s": prefill_s, "ctx_max": args.ctx_max}
    return {"cells": cells, "kv_bytes_per_token": KV_BYTES_PER_TOKEN[args.kv], "mtp": k}


def trace(e, args) -> dict:
    import torch

    (length,) = args.prompt_len
    ids = prompts.prompt_ids(args.model, args.prompt_files, length)
    e.reset()
    out = [e.prefill(ids, chunk=8)]
    logits = [e.logits(1)[0].clone()]
    for _ in range(1, args.trace_steps):
        out += e.decode(1)
        logits.append(e.logits(1)[0].clone())
    out += e.decode(args.tokens - len(out))
    torch.save({"prompt": ids, "ids": out, "logits": torch.stack(logits)}, args.trace_out)
    return {"trace": str(args.trace_out)}


def traces(e, args) -> dict:
    """The first --tokens ids and the first --trace-steps fp32 logits of the greedy continuation of every cell."""
    import torch

    cells = []
    for length, k, ids in prompts.cells(args.model, args.prompt_files, args.prompt_len, args.slices):
        e.reset()
        out = [e.prefill(ids, chunk=8)]
        logits = [e.logits(1)[0].clone()]
        for _ in range(1, args.trace_steps):
            out += e.decode(1)
            logits.append(e.logits(1)[0].clone())
        out += e.decode(args.tokens - len(out))
        cells.append({"length": length, "slice": k, "prompt": ids, "ids": out, "logits": torch.stack(logits).cpu()})
    torch.save(cells, args.trace_out)
    return {"trace": str(args.trace_out), "cells": len(cells)}


def _mtp_state(e, k: int) -> dict:
    """The anchor's drafts d1 .. dk and the head's post-norm hidden of each (bf16 [k, 5120]) after a draft stage."""
    import torch

    _, _, toks = e.draft_state()
    return {"drafts": toks[1 : k + 1], "hr": e.mtp_hidden()[:k].to(torch.bfloat16)}


def mtp_traces(e, args) -> dict:
    """The hand SpecEngine's MTP runs, the oracle of the speculative gates: for every (prompt length, slice) cell and
    force pattern (None: natural acceptance), after `prefill_spec(chunk=8)` (--fast-prefill: FastPrefill's) the first
    token, the drafts and the head's hidden rows; on slice 0 of the natural run the verify probes (the logits of
    T = 2, 3, 4, 8 rows over [anchor, drafts ...]); then one cycle at a time until --tokens ids exist: the tokens it
    emitted, its natural accepted count m (from the histogram) and, for the first 16 cycles, the drafts and head
    hidden after it."""
    import torch

    k = args.mtp
    forces = [None] + [[int(v) for v in f.split(",")] for f in args.forces]
    fast, cells = fast_prefill(e, args), []
    for length, s, ids in prompts.cells(args.model, args.prompt_files, args.prompt_len, args.slices):
        for force in forces:
            e.reset()
            e.set_force(force)
            first = fast.prefill_spec(ids, k=k) if fast else e.prefill_spec(ids, k=k, chunk=8)
            cell = {"length": length, "slice": s, "force": force, "prompt": ids, "first": first}
            cell["start"] = _mtp_state(e, k)
            if force is None and s == 0:
                probes = []
                for T in (2, 3, 4, 8):
                    toks = ([first, *cell["start"]["drafts"]] + [cell["start"]["drafts"][-1]] * 4)[:T]
                    probes.append({"T": T, "toks": toks, "ids": e.verify(toks), "logits": e.logits(T).clone()})
                cell["probes"] = probes
            tokens, cycles, last = [first], [], [0] * 8
            while len(tokens) < args.tokens:
                new, ss = e.cycles(k, 1)
                m = next(i for i in range(8) if ss[16 + i] != last[i])
                last = ss[16:24]
                cycles.append({"new": new, "m": m} | (_mtp_state(e, k) if len(cycles) < 16 else {}))
                tokens += new
            cell["cycles"] = cycles
            cells.append(cell)
    torch.save(cells, args.trace_out)
    return {"trace": str(args.trace_out), "cells": len(cells)}


def ring(e, args) -> dict:
    import torch

    e.reset()
    seq = []
    for T in [*range(1, 9)] * 3:
        e._step([0] * T, 0)
        ds = e._read(0, DS_N, torch.int32)
        seq.append({"T": T, "cur": int(ds[DS_CUR]), "pend": int(ds[DS_PEND])})
    return {"ring": seq}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="the checkpoint directory")
    ap.add_argument("--kv", choices=KV_FP8, default="bf16")
    ap.add_argument("--head", choices=HEADS, default="bf16")
    ap.add_argument("--ctx-max", type=int, required=True)
    ap.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    ap.add_argument("--out", type=Path, required=True, help="where to write the JSON result")
    modes = ap.add_subparsers(dest="mode", required=True)
    b = modes.add_parser("bench")
    prompts.add_arguments(b)
    b.add_argument("--tokens", type=int, default=256)
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--mtp", type=int, default=0, help="MTP speculative decode drafting K tokens (0: plain)")
    b.add_argument("--forge-position", action="store_true", help="prefill SHORT tokens, then move the position")
    b.add_argument(
        "--fast-prefill", action="store_true", help="prefill with FastPrefill (windows) instead of 8-row steps"
    )
    t = modes.add_parser("trace")
    prompts.add_arguments(t)
    t.add_argument("--tokens", type=int, default=256)
    t.add_argument("--trace-steps", type=int, default=16)
    t.add_argument("--trace-out", type=Path, required=True, help="torch.save file of ids and logits")
    ts = modes.add_parser("traces", help="trace for every (prompt length, slice) cell, one file")
    prompts.add_arguments(ts)
    prompts.add_slices(ts)
    ts.add_argument("--tokens", type=int, default=256)
    ts.add_argument("--trace-steps", type=int, default=16)
    ts.add_argument("--trace-out", type=Path, required=True, help="torch.save file: a list of cells")
    modes.add_parser("ring")
    mt = modes.add_parser("mtp-traces", help="the MTP oracle: per cell and force pattern, drafts / hidden / cycles")
    prompts.add_arguments(mt)
    prompts.add_slices(mt)
    mt.add_argument("--tokens", type=int, default=256)
    mt.add_argument("--mtp", type=int, required=True, help="drafts per cycle k")
    mt.add_argument("--forces", nargs="*", default=["1", "2,1,3", "1,3", "3,1,1,2"], help="forced acceptance patterns")
    mt.add_argument(
        "--fast-prefill", action="store_true", help="prefill with FastPrefill (windows) instead of 8-row steps"
    )
    mt.add_argument("--trace-out", type=Path, required=True, help="torch.save file: a list of cells")
    ap.add_argument(
        "--host-tables", action="store_true", help="hand engine: embedding and head tables in pinned memory"
    )
    args = ap.parse_args()
    lib = library(args.engine_dir)
    e = engine(args)
    modes_fn = {"bench": bench, "trace": trace, "traces": traces, "ring": ring, "mtp-traces": mtp_traces}
    result = modes_fn[args.mode](e, args)
    result |= {"lib": str(lib), "lib_sha256": sha256(lib), "engine": "hand", "kv": args.kv, "head": args.head}
    args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
