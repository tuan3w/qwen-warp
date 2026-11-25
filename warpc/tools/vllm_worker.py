"""vLLM (the workspace's locked version) on one checkpoint, single sequence, in a process of its own so no other CUDA
runtime shares its memory budget. bench.py and check.py start it through `spawn`; it writes its result as JSON to --out.

  uv run --package warpc python warpc/tools/vllm_worker.py MODEL --ctx-max N --out FILE [--tokenizer DIR] \
      [--kv-cache-dtype auto|fp8] [--gpu-memory-utilization U] [--language-model-only] [--mtp K] \
      bench (--prompt IDS | --prompts-file JSON) --tokens N --repeats N | greedy --prompts JSON --tokens N

MODEL is an HF checkpoint directory or a GGUF file (then --tokenizer names the HF directory of the same model, used for
the tokenizer and the config). Everything else is vLLM's default: torch.compile and CUDA graphs, as `vllm serve` runs.
The options of the 27B protocol: --kv-cache-dtype (fp8 takes the checkpoint's scales), --language-model-only (no
vision tower), --mtp K (the checkpoint's MTP head drafting K tokens); a run with any of them also uses max_num_seqs 1
and no prefix caching, and reports the served KV capacity (num_gpu_blocks x block_size). With --mtp every bench cell
also reports the speculation of its timed rounds (tokens_per_cycle = 1 + accepted drafts per draft step, cycles,
accepted_per_position = accepted / cycles at each draft position, all from vLLM's spec-decode counters) and the GiB of
device memory of the engine-core process (device_gib).

bench: greedy decode of --tokens tokens (EOS ignored), one warm-up round, then --repeats rounds, per prompt. The rate of
a round is vLLM's own decode time: (tokens - 1) / (last_token_ts - first_token_ts) from RequestOutput.metrics, the
engine core's timestamps of the first and the last generated token. Prefill (which yields the first token) is
excluded, as is any frontend work (tokenisation, detokenisation, scheduling before the first step).
greedy: the token ids of each prompt's greedy continuation, ending early at an end-of-sequence token (included).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import child

# 8B bf16 weights plus a KV cache for one 4096-token sequence fit in 0.85 x 24 GiB; the remainder covers the CUDA
# context of the process that started the worker
GPU_MEMORY_UTILIZATION = 0.85


def spawn(
    model: Path,
    *,
    ctx_max: int,
    mode: list[str],
    tokenizer: Path | None = None,
    options: list[str] = (),
    utilizations: list[float] = (),
) -> dict:
    """Run this file on `model` with the mode arguments `mode`: its result, or RuntimeError with why it failed. With
    `utilizations`, one attempt per --gpu-memory-utilization value in order until one succeeds."""
    errors = []
    for u in utilizations or [None]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "result.json"
            command = [sys.executable, __file__, str(model), "--ctx-max", str(ctx_max), "--out", str(out)]
            command += ["--tokenizer", str(tokenizer)] if tokenizer else []
            command += [*options, *(["--gpu-memory-utilization", str(u)] if u is not None else [])]
            if error := child.run(command + mode):
                errors.append(f"{u}: {error}" if u is not None else error)
                continue
            return json.loads(out.read_text())
    raise RuntimeError(f"vLLM: {'; '.join(errors)}")


def engine(model: Path, tokenizer: Path | None, ctx_max: int, args=None):
    from vllm import LLM

    extra = {"tokenizer": str(tokenizer), "hf_config_path": str(tokenizer)} if tokenizer else {}
    if args is not None and (args.kv_cache_dtype != "auto" or args.language_model_only or args.mtp):
        extra |= {"kv_cache_dtype": args.kv_cache_dtype, "max_num_seqs": 1, "enable_prefix_caching": False}
        if args.language_model_only:
            extra["language_model_only"] = True
        if args.mtp:
            extra["speculative_config"] = {"method": "mtp", "num_speculative_tokens": args.mtp}
    util = args.gpu_memory_utilization if args is not None and args.gpu_memory_utilization else GPU_MEMORY_UTILIZATION
    return LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=ctx_max,
        gpu_memory_utilization=util,
        disable_log_stats=False,  # RequestOutput.metrics is only filled with stats logging on
        seed=0,
        **extra,
    )


def capacity(llm) -> dict:
    """The KV cache vLLM allocated: blocks x block size = the positions it can serve."""
    cc = llm.llm_engine.vllm_config.cache_config
    blocks = cc.num_gpu_blocks
    return {
        "num_gpu_blocks": blocks,
        "block_size": cc.block_size,
        "kv_positions": blocks * cc.block_size if blocks else None,
        "cache_dtype": cc.cache_dtype,
        "gpu_memory_utilization": cc.gpu_memory_utilization,
    }


def spec_counters(llm) -> list[int]:
    """vLLM's cumulative [draft steps, accepted drafts, accepted at draft position 0, 1, ...] over all requests."""
    drafts = accepted = 0
    per_pos: list[int] = []
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            drafts += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            accepted += m.value
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            per_pos = m.values
    return [drafts, accepted, *per_pos]


def engine_memory(worker) -> dict:
    """Runs inside the engine-core worker process: GiB resident for this process on the device (NVML: CUDA context,
    weights, KV cache, graphs, scratch) and torch's allocator figures; the peak is the allocator's since the start
    of the KV sizing profile run."""
    import os

    import pynvml
    import torch

    gib = 2.0**30
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{torch.cuda.get_device_properties(worker.device).uuid}")
    resident = next(
        p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if p.pid == os.getpid()
    )
    return {
        "resident": resident / gib,
        "torch_allocated": torch.cuda.memory_allocated(worker.device) / gib,
        "torch_peak_reserved": torch.cuda.max_memory_reserved(worker.device) / gib,
    }


def bench(llm, prompt: list[int], tokens: int, repeats: int, mtp: int = 0) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=tokens, temperature=0.0, ignore_eos=True)
    rates = []
    for i in range(repeats + 1):
        (out,) = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        m = out.metrics
        if m is None:
            raise RuntimeError("vLLM returned no request metrics")
        n = len(out.outputs[0].token_ids)
        rates.append((n - 1) / (m.last_token_ts - m.first_token_ts))
        if i == 0 and mtp:
            before = spec_counters(llm)  # the warm-up round is not counted
    result = {"rates": rates[1:]}
    if mtp:
        drafts, accepted, *per_pos = (a - b for a, b in zip(spec_counters(llm), before, strict=True))
        result |= {
            "tokens_per_cycle": 1 + accepted / drafts,
            "cycles": drafts,
            "accepted_per_position": [n / drafts for n in per_pos],
            "device_gib": llm.collective_rpc(engine_memory)[0],
        }
    return result


def greedy(llm, prompts: list[list[int]], tokens: int) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=tokens, temperature=0.0)
    outs = llm.generate([{"prompt_token_ids": ids} for ids in prompts], params, use_tqdm=False)
    return {"tokens": [list(o.outputs[0].token_ids) for o in outs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", type=Path, help="HF checkpoint directory or *.gguf file")
    ap.add_argument("--ctx-max", type=int, required=True, help="vLLM max_model_len")
    ap.add_argument("--out", type=Path, required=True, help="where to write the JSON result")
    ap.add_argument("--tokenizer", type=Path, help="HF directory for the tokenizer and config of a GGUF file")
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--gpu-memory-utilization", type=float)
    ap.add_argument("--language-model-only", action="store_true")
    ap.add_argument("--mtp", type=int, default=0, help="MTP speculative tokens (0: off)")
    modes = ap.add_subparsers(dest="mode", required=True)
    b = modes.add_parser("bench")
    b.add_argument("--prompt", type=lambda s: [int(t) for t in s.split(",")], help="comma-separated ids")
    b.add_argument("--prompts-file", type=Path, help="JSON {name: ids}: one result cell per prompt")
    b.add_argument("--tokens", type=int, required=True)
    b.add_argument("--repeats", type=int, required=True)
    g = modes.add_parser("greedy")
    g.add_argument("--prompts", type=json.loads, required=True, help="JSON list of token id lists")
    g.add_argument("--tokens", type=int, required=True)
    args = ap.parse_args()
    if args.mtp:
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # engine_memory is shipped to the engine-core process
    llm = engine(args.model, args.tokenizer, args.ctx_max, args)
    if args.mode == "bench" and args.prompts_file:
        cells = {}
        for name, ids in json.loads(args.prompts_file.read_text()).items():
            cell = bench(llm, ids, args.tokens, args.repeats, args.mtp)
            cells[name] = {**cell, "tok_s": statistics.median(cell["rates"]), "max_model_len": args.ctx_max}
        result = {"cells": cells, "capacity": capacity(llm)}
    elif args.mode == "bench":
        result = bench(llm, args.prompt, args.tokens, args.repeats, args.mtp)
    else:
        result = greedy(llm, args.prompts, args.tokens)
    args.out.write_text(json.dumps({"vllm": version("vllm"), **result}))


if __name__ == "__main__":
    main()
