"""Decode throughput: DSpine vs DFlash2 vs plain greedy on the same prompts.

  python bench_sp.py dspine [N]     python bench_sp.py dflash [N]     python bench_sp.py plain [N]
Each mode runs in its own process (the drafters do not fit next to each other on the 24 GB card).
"""

import os
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, ".")

TARGET = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
PROMPTS = [
    "Write a Python implementation of an LRU cache class with get/put in O(1), then unit tests with pytest.",
    "Implement a thread-safe bounded blocking queue in Java with a producer/consumer demo.",
    "Write a Rust function that parses a CSV line with quoted fields and escaped quotes, plus tests.",
]


def main():
    mode = sys.argv[1]
    n_new = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    tok = AutoTokenizer.from_pretrained(TARGET)
    if mode == "dspine":
        from dspine import DspineEngine

        e = DspineEngine(ctx_max=2048, kv_fp8=0, verbose=False)
        gen = e.generate_sp
    elif mode == "dflash":
        from dflash import DflashEngine

        e = DflashEngine(
            ctx_max=2048, kv_fp8=0, verbose=False, tmax=8, host_tables=True
        )
        gen = e.generate_df
    elif mode == "plain":
        from spec import SpecEngine

        e = SpecEngine(ctx_max=2048, kv_fp8=0, verbose=False, tmax=8, host_tables=True)
        e.disable_pld()

        def gen(ids, n):
            e.reset()
            first = e.prefill(ids, chunk=8)
            return [first] + e.decode(n - 1), dict(tokens_per_cycle=1.0)
    else:
        raise SystemExit(__doc__)

    def encode(p):
        out = tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True
        )
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    gen(encode(PROMPTS[0]), 32)  # warm-up / graph capture
    torch.cuda.synchronize()
    tot_tok, tot_t = 0, 0.0
    for p in PROMPTS:
        ids = encode(p)
        torch.cuda.synchronize()
        t0 = time.time()
        toks, st = gen(ids, n_new)
        torch.cuda.synchronize()
        dt = time.time() - t0
        tot_tok += len(toks)
        tot_t += dt
        print(
            f"  {mode}: {len(toks)} tok in {dt:.2f}s = {len(toks) / dt:6.1f} tok/s  tok/cycle={st['tokens_per_cycle']:.2f}  hist={st.get('hist')}",
            flush=True,
        )
    print(
        f"{mode}: TOTAL {tot_tok / tot_t:.1f} tok/s over {len(PROMPTS)} prompts, N={n_new}"
    )


if __name__ == "__main__":
    main()
