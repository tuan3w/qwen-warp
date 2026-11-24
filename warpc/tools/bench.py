"""Single-stream greedy decode throughput of one checkpoint on warpc, llama.cpp's llama-server or vLLM.

  uv run --package warpc python warpc/tools/bench.py ENGINE MODEL [--ctx-max N] [--tokens N] [--repeats N] \
      [--kv FORMAT] [--tokenizer DIR] [--json FILE]

ENGINE is warpc, llama-server or vllm; MODEL an HF checkpoint directory or a GGUF file. Every engine decodes --tokens
tokens greedily after the same 5-token prompt, end of sequence ignored: one warm-up round, then the median tok/s of
--repeats rounds. Only decode is timed:
  warpc: wall time of Model.decode(tokens) after the prefill; decode returns once the last token is on the host. Also
    reports the compile time and the peak device memory torch allocated. --kv picks the KV cache format.
  llama-server (GGUF only): `-ngl 99 -c ctx_max -np 1`; the server's timings.predicted_per_second of the /completion.
  vllm: in a child process, see vllm_worker.py; a GGUF file needs --tokenizer (the HF directory of the same model).
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import llama_server
import vllm_worker
import warpc

PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is" in Qwen3's vocabulary; valid ids in every vocabulary
ENGINES = ("warpc", "llama-server", "vllm")


def bench_warpc(model: Path, ctx_max: int, tokens: int, repeats: int, kv: str) -> dict:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    compiled = warpc.compile(model, ctx_max=ctx_max, kv=kv)
    compile_s = time.perf_counter() - t0
    rates = []
    for _ in range(repeats + 1):
        compiled.reset()
        compiled.prefill(PROMPT)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled.decode(tokens)
        rates.append(tokens / (time.perf_counter() - t0))
    peak = torch.cuda.max_memory_allocated() / 2**30
    return {"rates": rates[1:], "plan": compiled.plan.summary(), "compile_s": compile_s, "peak_gib": peak}


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
    ap.add_argument("--ctx-max", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kv", default="bf16", help="warpc KV cache format")
    ap.add_argument("--tokenizer", type=Path, help="vllm on a GGUF file: the HF directory of the same model")
    ap.add_argument("--json", type=Path, help="also write the result to this file")
    args = ap.parse_args()
    if args.engine == "warpc":
        result = bench_warpc(args.model, args.ctx_max, args.tokens, args.repeats, args.kv)
    elif args.engine == "llama-server":
        result = bench_llama_server(args.model, args.ctx_max, args.tokens, args.repeats)
    else:
        result = bench_vllm(args.model, args.ctx_max, args.tokens, args.repeats, args.tokenizer)
    result["tok_s"] = statistics.median(result["rates"])
    setup = f"median of {args.repeats} x {args.tokens} tokens, ctx_max {args.ctx_max}"
    print(f"{args.engine} {args.model.name}: {result['tok_s']:.1f} tok/s ({setup})")
    print(f"all rounds: {', '.join(f'{r:.1f}' for r in result['rates'])}")
    if args.engine == "warpc":
        print(f"{result['plan']}\ncompile {result['compile_s']:.1f} s, peak device memory {result['peak_gib']:.2f} GiB")
    if args.json:
        args.json.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
