"""Single-stream greedy decode throughput of one checkpoint on warpc, llama.cpp's llama-server, vLLM or the hand engine.

  uv run --package warpc python warpc/tools/bench.py ENGINE MODEL [--ctx-max N] [--tokens N] [--repeats N] \
      [--kv FORMAT] [--tokenizer DIR] [--json FILE] [--backend cuda|swiftmk|composite] \
      [--kernels K,...] [--head bf16|int4_refine] [--mtp K] [--host-tables] [--window] [--forge-position] \
      [--prompt-files F... --prompt-len L...] [--gpu-memory-utilization U...] [--language-model-only]

ENGINE is warpc, llama-server, vllm or swift-engine (the hand engine of the v0.12.0 checkout $WARPC_LEGACY_DIR, 27B
checkpoint only); MODEL an HF checkpoint directory or a GGUF file. Every engine decodes --tokens tokens greedily, end
of sequence ignored: one warm-up round, then the median tok/s of --repeats rounds. Only decode is timed.

Without --prompt-files, the prompt is the same 5 tokens on every engine:
  warpc: wall time of Model.decode(tokens) after the prefill; decode returns once the last token is on the host. Also
    reports the compile time and the peak device memory torch allocated. --kv picks the KV cache format.
  llama-server (GGUF only): `-ngl 99 -c ctx_max -np 1`; the server's timings.predicted_per_second of the /completion.
  vllm: in a child process, see vllm_worker.py; a GGUF file needs --tokenizer (the HF directory of the same model).

With --prompt-files (the 27B protocol, one result cell per --prompt-len), every engine reports the inter-arrival rate of
the tokens - 1 tokens after the first: warpc and swift-engine by measure.py (one prefill in 8-row steps per prompt
length, decode rounds continuing it, nvidia-smi clock samples, host overhead per token), vllm by its engine-core token
timestamps (a fresh request per round). --ctx-max defaults to the longest prompt plus every round's tokens. Each cell
records the KV cache bytes per token of the 16 attention layers.

--mtp K: greedy MTP speculation with K drafts per cycle (vllm: num_speculative_tokens; swift-engine: the hand's
SpecEngine with the prompt-lookup hybrid off; warpc: backend composite with the three data-movement kernels unless
--kernels says otherwise, so the hand's kernels do all model compute). warpc and the hand engine then also report the
tokens per cycle and the histogram of accepted drafts of the timed rounds; both decode a round as whole groups of 8
cycles until it holds at least --tokens - 1 tokens (the same tokens over the same spans on both sides).
--host-tables keeps the embedding and rescoring head in pinned host memory (warpc: compile(host_tables=True), hand:
Engine(host_tables=True)): needed to fit more than about 80k tokens of context.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

import llama_server
import measure
import prompts
import swift_worker
import vllm_worker
import warpc
from backends import BACKENDS, make_backend
from warpc.windows import plan_windows

PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is" in Qwen3's vocabulary; valid ids in every vocabulary
ENGINES = ("warpc", "llama-server", "vllm", "swift-engine")
ROWS = 8  # prompt step rows: the hand engine's Engine(tmax=8) chunk
SHORT = swift_worker.SHORT  # the real prompt under a forged position


def bench_warpc(model: Path, ctx_max: int, tokens: int, repeats: int, kv: str, backend=None) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    compiled = warpc.compile(model, ctx_max=ctx_max, kv=kv, backend=backend)
    compile_s = time.perf_counter() - t0
    rates = []
    for _ in range(repeats + 1):
        compiled.reset()
        compiled.append(PROMPT)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled.decode(tokens)
        rates.append(tokens / (time.perf_counter() - t0))
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {"rates": rates[1:], "plan": compiled.plan.summary(), "compile_s": compile_s, "peak_gib": peak}


def _backend(args):
    """(backend, the hand engine's library or None) of --backend / --head / --kernels; with --mtp and no --kernels the
    composite takes only the three data-movement kernels (the hand's kernels do all model compute); so does a window
    run (its prefill packs are the hand library's)."""
    glue = "g_select_row,g_take,g_concat" if (args.mtp or args.window) and args.backend == "composite" else None
    backend = make_backend(args.backend, args.head, args.kernels or glue, window=args.window)
    if args.backend == "cuda":
        return backend, None
    from warpc.swiftmk import lib

    return backend, lib.load()


def long_prompt(args, length: int) -> list[int]:
    """The prompt of `length` tokens: from --ids-file (JSON {"speed": {length: ids}}) or cut from --prompt-files."""
    if args.ids_file:
        return json.loads(args.ids_file.read_text())["speed"][str(length)]
    return prompts.prompt_ids(args.tokenizer or args.model, args.prompt_files, length)


def bench_warpc_long(args, ctx_max: int) -> dict:
    backend, library = _backend(args)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    compiled = warpc.compile(
        args.model,
        ctx_max=ctx_max,
        rows_max=ROWS,
        kv=args.kv,
        backend=backend,
        speculate=args.mtp,
        host_tables=args.host_tables,
        window_rows=4096 if args.window else 0,
    )
    compile_s = time.perf_counter() - t0
    rt, n, k = compiled._runtime, args.tokens - 1, args.mtp
    spec = compiled.decoder if k else None
    free, total = torch.cuda.mem_get_info()
    cells = {}
    for length in args.prompt_len:
        ids = long_prompt(args, SHORT if args.forge_position else length)
        compiled.reset()
        t0 = time.perf_counter()
        compiled.append(ids, ends=None if args.forge_position or not args.window else plan_windows(ids))
        if args.forge_position:  # timing only: the position moves, the cache rows below it stay as they are (zeros)
            rt.pos.fill_(length)
            rt.position = length
            if k:
                spec._last_hidden_pos = length
                spec.start(k)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0

        def replay_events() -> tuple[float, int]:
            s = measure.replay_seconds(rt.graph(1, ("ids",)).replay, n)
            rt.position += n
            return s, n

        def decode() -> int:
            if not k:
                compiled.decode(n)
                return n
            got = 0
            while got < n:  # whole groups of 8 cycles as the hand's rounds: the same tokens over the same spans
                got += len(spec.cycles(8))
            return got

        before = spec.stats if k else None
        cell = measure.rounds(decode, args.repeats, None if k else replay_events)
        if k:
            after = spec.stats
            cycles, emitted = after.cycles - before.cycles, after.emitted - before.emitted
            cell |= {
                "tokens_per_cycle": emitted / cycles,
                "cycles": cycles,
                "accepted": [a - b for a, b in zip(after.accepted, before.accepted, strict=True)],
                "cycle_ms": 1000 * emitted / cycles / cell["tok_s"],
            }
        cells[str(length)] = {**cell, "prefill_s": prefill_s, "ctx_max": ctx_max}
    used = torch.cuda.mem_get_info()
    result = {
        "cells": cells,
        "plan": compiled.plan.summary(),
        "compile_s": compile_s,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "device_gib_after_load": (total - free) / 2**30,
        "device_gib_after_decode": (used[1] - used[0]) / 2**30,
        "kv_bytes_per_token": swift_worker.KV_BYTES_PER_TOKEN.get(args.kv),
        "mtp": k,
    }
    if library is not None:
        result |= {"lib": str(library.path), "lib_sha256": library.sha256}
    return result


def bench_swift(args, ctx_max: int) -> dict:
    mode = ["bench", "--prompt-files", *map(str, args.prompt_files), "--prompt-len", *map(str, args.prompt_len)]
    mode += ["--tokens", str(args.tokens), "--repeats", str(args.repeats), "--mtp", str(args.mtp)]
    mode += ["--forge-position"] if args.forge_position else []
    mode += ["--fast-prefill"] if args.window else []
    return swift_worker.spawn(
        args.model, kv=args.kv, head=args.head, ctx_max=ctx_max, mode=mode, host_tables=args.host_tables
    )


def bench_vllm_long(args) -> dict:
    """One vLLM instance for every prompt length (max_model_len = the longest + 512); a length vLLM cannot serve
    makes the whole run n/s, with vLLM's reason."""
    tok = args.tokenizer or args.model
    ids = {str(n): prompts.prompt_ids(tok, args.prompt_files, n) for n in args.prompt_len}
    opts = ["--kv-cache-dtype", "auto" if args.kv == "bf16" else args.kv]
    opts += ["--language-model-only"] if args.language_model_only else []
    opts += ["--mtp", str(args.mtp)] if args.mtp else []
    with tempfile.TemporaryDirectory() as tmp:
        pf = Path(tmp) / "prompts.json"
        pf.write_text(json.dumps(ids))
        mode = ["bench", "--prompts-file", str(pf), "--tokens", str(args.tokens), "--repeats", str(args.repeats)]
        try:
            return vllm_worker.spawn(
                args.model,
                ctx_max=max(args.prompt_len) + 512,
                mode=mode,
                tokenizer=args.tokenizer,
                options=opts,
                utilizations=args.gpu_memory_utilization,
            )
        except RuntimeError as e:
            return {"cells": {n: {"tok_s": None, "n/s": str(e)} for n in ids}}


def bench_llama_server(model: Path, ctx_max: int, tokens: int, repeats: int) -> dict:
    with llama_server.LlamaServer(model, ctx_max) as server:
        rates = [
            server.complete(PROMPT, tokens, ignore_eos=True)["timings"]["predicted_per_second"]
            for _ in range(repeats + 1)
        ]
    return {"rates": rates[1:], "llama.cpp": llama_server.version()}


def bench_vllm(model: Path, ctx_max: int, tokens: int, repeats: int, tokenizer: Path | None) -> dict:
    mode = ["bench", "--prompt", ",".join(map(str, PROMPT)), "--tokens", str(tokens), "--repeats", str(repeats)]
    return vllm_worker.spawn(model, ctx_max=ctx_max, mode=mode, tokenizer=tokenizer)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("engine", choices=ENGINES)
    ap.add_argument("model", type=Path, help="HF checkpoint directory or *.gguf file")
    ap.add_argument("--ctx-max", type=int, help="KV cache positions (default 4096; long prompts: see above)")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kv", default="bf16", help="KV cache format (warpc's names; vllm: bf16 = auto, or fp8)")
    ap.add_argument("--tokenizer", type=Path, help="vllm on a GGUF file: the HF directory of the same model")
    ap.add_argument("--json", type=Path, help="also write the result to this file")
    ap.add_argument("--backend", choices=BACKENDS, default="cuda", help="warpc backend")
    ap.add_argument("--kernels", help="composite: the warpc kernels admitted, comma-separated (default: all)")
    ap.add_argument("--head", choices=("bf16", "int4_refine"), default="bf16", help="swiftmk / swift-engine head")
    ap.add_argument("--prompt-files", type=Path, nargs="+", help=prompts.PROMPT_FILES_HELP)
    ap.add_argument("--ids-file", type=Path, help="warpc: JSON with 'speed': {length: prompt ids}, instead of files")
    ap.add_argument("--prompt-len", type=int, nargs="+", help="long prompt length(s) in tokens")
    ap.add_argument(
        "--gpu-memory-utilization", type=float, nargs="+", default=[0.93, 0.90, 0.87], help="vllm, tried in order"
    )
    ap.add_argument("--language-model-only", action="store_true", help="vllm: skip the vision tower")
    ap.add_argument(
        "--mtp", type=int, default=0, help="warpc / vllm / swift-engine: MTP draft tokens (0: plain decode)"
    )
    ap.add_argument("--host-tables", action="store_true", help="warpc / swift-engine: tables in pinned host memory")
    ap.add_argument(
        "--forge-position",
        action="store_true",
        help="timing only: prefill 385 tokens, then move the position to --prompt-len over the empty cache",
    )
    ap.add_argument(
        "--window",
        action="store_true",
        help="warpc: the window prefill (SwiftmkBackend(window=True), plan_windows ends); swift-engine: FastPrefill",
    )
    args = ap.parse_args()
    if args.prompt_files or args.ids_file:
        if not args.prompt_len:
            ap.error("--prompt-files / --ids-file need --prompt-len")
        if args.ids_file and args.engine != "warpc":
            ap.error("--ids-file is for the warpc engine (the other engines' prompts come from --prompt-files)")
        # every round's tokens (+ the events round), and a speculative round's overshoot of up to 8 cycles x (k + 1)
        rounds = (args.repeats + 2) * (args.tokens - 1 + 8 * (args.mtp + 1) * bool(args.mtp))
        ctx_max = args.ctx_max or max(args.prompt_len) + 1 + rounds
        if args.engine == "warpc":
            result = bench_warpc_long(args, ctx_max)
        elif args.engine == "swift-engine":
            result = bench_swift(args, ctx_max)
        elif args.engine == "vllm":
            result = bench_vllm_long(args)
        else:
            ap.error(f"{args.engine} has no long-prompt mode")
        result |= {"engine": args.engine, "kv": args.kv, "head": args.head, "backend": args.backend}
        for length, cell in result["cells"].items():
            rate = f"{cell['tok_s']:.2f} tok/s" if cell.get("tok_s") else f"n/s ({cell.get('n/s')})"
            print(f"{args.engine} {args.kv} prompt {length}: {rate}")
        if args.json:
            args.json.write_text(json.dumps(result))
        return
    if args.engine == "swift-engine":
        ap.error("swift-engine needs --prompt-files / --prompt-len")
    ctx_max = args.ctx_max or 4096
    if args.engine == "warpc":
        backend = _backend(args)[0]
        result = bench_warpc(args.model, ctx_max, args.tokens, args.repeats, args.kv, backend)
    elif args.engine == "llama-server":
        result = bench_llama_server(args.model, ctx_max, args.tokens, args.repeats)
    else:
        result = bench_vllm(args.model, ctx_max, args.tokens, args.repeats, args.tokenizer)
    result["tok_s"] = statistics.median(result["rates"])
    setup = f"median of {args.repeats} x {args.tokens} tokens, ctx_max {ctx_max}"
    print(f"{args.engine} {args.model.name}: {result['tok_s']:.1f} tok/s ({setup})")
    print(f"all rounds: {', '.join(f'{r:.1f}' for r in result['rates'])}")
    if args.engine == "warpc":
        print(f"{result['plan']}\ncompile {result['compile_s']:.1f} s, peak device memory {result['peak_gib']:.2f} GiB")
    if args.json:
        args.json.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
