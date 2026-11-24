"""vLLM (the workspace's locked version) on one checkpoint, single sequence, in a process of its own so no other CUDA
runtime shares its memory budget. bench.py and check.py start it through `spawn`; it writes its result as JSON to --out.

  uv run --package warpc python warpc/tools/vllm_worker.py MODEL --ctx-max N --out FILE [--tokenizer DIR] \
      bench --prompt IDS --tokens N --repeats N | greedy --prompts JSON --tokens N

MODEL is an HF checkpoint directory or a GGUF file (then --tokenizer names the HF directory of the same model, used for
the tokenizer and the config). Everything else is vLLM's default: torch.compile and CUDA graphs, as `vllm serve` runs.

bench: greedy decode of --tokens tokens (EOS ignored), one warm-up round, then --repeats rounds. The rate of a round is
vLLM's own decode time: (tokens - 1) / (last_token_ts - first_token_ts) from RequestOutput.metrics, the engine core's
timestamps of the first and the last generated token. Prefill (which yields the first token) is excluded, as is any
frontend work (tokenisation, detokenisation, scheduling before the first step).
greedy: the token ids of each prompt's greedy continuation, ending early at an end-of-sequence token (included).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import child

# 8B bf16 weights plus a KV cache for one 4096-token sequence fit in 0.85 x 24 GiB; the remainder covers the CUDA
# context of the process that started the worker
GPU_MEMORY_UTILIZATION = 0.85


def spawn(model: Path, *, ctx_max: int, mode: list[str], tokenizer: Path | None = None) -> dict:
    """Run this file on `model` with the mode arguments `mode`: its result, or RuntimeError with why it failed."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        command = [sys.executable, __file__, str(model), "--ctx-max", str(ctx_max), "--out", str(out)]
        command += ["--tokenizer", str(tokenizer)] if tokenizer else []
        if error := child.run(command + mode):
            raise RuntimeError(f"vLLM: {error}")
        return json.loads(out.read_text())


def engine(model: Path, tokenizer: Path | None, ctx_max: int):
    from vllm import LLM

    extra = {"tokenizer": str(tokenizer), "hf_config_path": str(tokenizer)} if tokenizer else {}
    return LLM(
        model=str(model),
        dtype="bfloat16",
        max_model_len=ctx_max,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        disable_log_stats=False,  # RequestOutput.metrics is only filled with stats logging on
        seed=0,
        **extra,
    )


def bench(llm, prompt: list[int], tokens: int, repeats: int) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=tokens, temperature=0.0, ignore_eos=True)
    rates = []
    for _ in range(repeats + 1):
        (out,) = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)
        m = out.metrics
        if m is None:
            raise RuntimeError("vLLM returned no request metrics")
        n = len(out.outputs[0].token_ids)
        rates.append((n - 1) / (m.last_token_ts - m.first_token_ts))
    return {"rates": rates[1:]}


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
    modes = ap.add_subparsers(dest="mode", required=True)
    b = modes.add_parser("bench")
    b.add_argument("--prompt", type=lambda s: [int(t) for t in s.split(",")], required=True, help="comma-separated ids")
    b.add_argument("--tokens", type=int, required=True)
    b.add_argument("--repeats", type=int, required=True)
    g = modes.add_parser("greedy")
    g.add_argument("--prompts", type=json.loads, required=True, help="JSON list of token id lists")
    g.add_argument("--tokens", type=int, required=True)
    args = ap.parse_args()
    llm = engine(args.model, args.tokenizer, args.ctx_max)
    if args.mode == "bench":
        result = bench(llm, args.prompt, args.tokens, args.repeats)
    else:
        result = greedy(llm, args.prompts, args.tokens)
    args.out.write_text(json.dumps({"vllm": version("vllm"), **result}))


if __name__ == "__main__":
    main()
