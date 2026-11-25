"""Decode throughput and peak device memory of a compiled model.

  uv run --package warpc python warpc/tests/bench.py MODEL [--ctx-max N] [--kv FORMAT] [--tokens N] [--repeats N]

MODEL is a *.gguf file or an HF checkpoint directory. Reports the median tok/s of `repeats` greedy decodes of `tokens`
tokens after a short prompt (prefill not timed) and the peak memory torch allocated on the device.
"""

import argparse
import statistics
import time

import torch

import warpc


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="*.gguf file or HF checkpoint directory")
    ap.add_argument("--ctx-max", type=int, default=4096)
    ap.add_argument("--kv", default="bf16")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = warpc.compile(args.model, ctx_max=args.ctx_max, kv=args.kv)
    compile_s = time.perf_counter() - t0
    prompt = [t % model.config.vocab_size for t in (785, 6722, 315, 9625, 374)]
    rates = []
    for _ in range(args.repeats + 1):  # the first round warms up
        model.reset()
        model.prefill(prompt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.decode(args.tokens)  # returns after the last token reached the host
        rates.append(args.tokens / (time.perf_counter() - t0))
    rates = rates[1:]
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{args.model}: {model.plan.summary()}")
    rate = statistics.median(rates)
    setup = f"median of {args.repeats} x {args.tokens}, ctx_max {args.ctx_max}, kv {args.kv}"
    print(f"compile {compile_s:.1f} s; decode {rate:.1f} tok/s ({setup})")
    print(f"peak device memory {peak:.2f} GiB; all rounds: {', '.join(f'{r:.1f}' for r in rates)}")


if __name__ == "__main__":
    main()
