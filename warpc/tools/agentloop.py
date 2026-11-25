"""Agent-loop benchmark (AL-64k-20): a long shared prefix, then turns that each append a tool result and decode a short
answer. One workload, replayed token for token by every engine.

  uv run --package warpc python warpc/tools/agentloop.py trace --tokenizer DIR --files F... --out trace.json
  uv run --package warpc python warpc/tools/agentloop.py run ENGINE MODEL --trace trace.json --mode chat|linear \
      [--turns N] [--verify-last] [--kv ...] --out result.json

ENGINE: warpc (PrefixCache over warpc.compile; `--cache lineage` appends to the device cache or re-prefills, `none`
re-prefills every turn), hand (server/smk_runner.py Runner with snapshots and the host prefix store; `--mtp K`),
hand-cold (the same Runner with no reuse), vllm (LLM with prefix caching on, chunked prefill, language model only).

The trace (deterministic, token ids only): the corpus is the text of --files in sorted path order tokenised by the
checkpoint's tokenizer; the prefix is `<|im_start|>system\\n` + --prefix-len corpus tokens + the user header and a
200-token task; turn k appends an assistant message (100-299 corpus tokens) and a tool result (500-2,000 tokens) and
asks for n_k greedy tokens (100-299, end of sequence ignored). `render` is the only place prompts are built. Modes:
`chat`: earlier assistant messages are the trace's, rendered without the think tail, so each prompt diverges from the
previous one right after its assistant header (what the Qwen chat template does to a history); `linear`: the history is
the engine's own output and the new prompt extends the cached tokens.

Per turn: `first_s` is request to first token (restore, spill, snapshots and the whole prefill), `decode_s` the rest;
`cached` the tokens the engine did not prefill. The summary reports the medians over turns 1.. and the peak GPU memory
(nvidia-smi every 200 ms over the whole process, the CUDA context and the allocator's cache included). --verify-last
runs the last turn again from an empty cache: the ids must equal the resumed turn's (the hand engine and warpc claim
bit equality, vLLM does not). One process per engine: the GPU is exclusive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IM_START, IM_END = 248045, 248046
SYSTEM, USER, ASSISTANT, NL, THINK = [8678, 198], [846, 198], [74455, 198], [198], [248068, 198]


def header(role: list[int]) -> list[int]:
    return [IM_START, *role]


# ---- the trace


def build_trace(tokenizer_dir, files, prefix_len, turns, seed, tool=(500, 2000), decode=(100, 300), task=200):
    import numpy as np
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    text = "\n\n".join(Path(f).read_text(errors="replace") for f in sorted(files, key=str))
    corpus = tok(text, add_special_tokens=False)["input_ids"]
    rng = np.random.RandomState(seed)
    at = 0

    def take(n):
        nonlocal at
        out = corpus[at : at + n]
        at += n
        if len(out) < n:
            raise ValueError(f"the corpus of {len(corpus)} tokens is too short")
        return out

    prefix = [*header(SYSTEM), *take(prefix_len), IM_END, *NL, *header(USER), *take(task), IM_END, *NL]
    rows = []
    for _ in range(turns):
        assistant = take(int(rng.randint(*decode)))
        tool_result = take(int(rng.randint(tool[0], tool[1] + 1)))
        rows.append({"assistant": assistant, "tool": tool_result, "decode": int(rng.randint(*decode))})
    return {"seed": seed, "prefix": prefix, "turns": rows, "corpus_tokens": len(corpus)}


def render(trace, k, mode, own=None):
    """The prompt of turn k: the prefix, the k earlier (assistant, tool) pairs, the assistant header (+ the think tail
    in chat mode, which the next prompt does not carry)."""
    ids = list(trace["prefix"])
    for j in range(k):
        t = trace["turns"][j]
        a = t["assistant"] if mode == "chat" else own[j]
        ids += [*header(ASSISTANT), *a, IM_END, *NL, *header(USER), *t["tool"], IM_END, *NL]
    ids += header(ASSISTANT)
    ids += THINK if mode == "chat" else []
    return ids


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---- memory


class Mem:
    """Peak GPU memory in use (nvidia-smi), whole process tree."""

    def __enter__(self):
        query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"]
        self.p = subprocess.Popen(query, stdout=subprocess.PIPE, text=True)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        out, _ = self.p.communicate()
        values = [float(x) for x in out.split() if x.strip()]
        self.peak_mib = max(values) if values else None


# ---- engines: turn(prompt_ids, n_new) -> dict(first_s, decode_s, ids, cached, ...); cold() after dropping every cache


class HandEngine:
    def __init__(self, model, ctx_max, args):
        os.environ["SWIFT_MODEL"] = str(model)
        sys.path.insert(0, str(ROOT / "server"))
        import smk_runner as R

        self.R = R
        cold = args.engine == "hand-cold"
        self.r = R.Runner(
            ctx_max=ctx_max,
            kv_fp8=args.kv_hand,
            k=args.mtp,
            group=args.group,
            max_snaps=0 if cold else args.snaps,
            host_tables=args.host_tables,
            prefix_cache_gb=0.0 if cold else args.store_gb,
            pld=False,
            warmup=True,
            log=lambda *a: None,
        )

    def turn(self, ids, n):
        toks = []
        for out, _, _ in self.r.run(ids, self.R.GenParams(max_tokens=n)):
            toks += out
        st = self.r.last
        return {
            "first_s": st.prefill_s,
            "decode_s": st.gen_s,
            "ids": toks,
            "cached": st.cached,
            "source": st.source,
            "spill_s": st.spill_s,
            "restore_s": st.restore_s,
        }

    def cold(self, ids, n):
        self.r.drop_cache()
        return self.turn(ids, n)


class WarpcEngine:
    """warpc.compile + PrefixCache. `--cache lineage`: append when the new prompt extends the device cache, else
    reset and prefill (no snapshots); `none`: reset and prefill every turn."""

    def __init__(self, model, ctx_max, args):
        import torch

        import warpc
        from warpc.prefix import PrefixCache, class_d_plan
        from warpc.windows import plan_windows

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from backends import make_backend

        self.torch, self.args = torch, args
        kw = {}
        if args.backend:
            kw["backend"] = make_backend(args.backend, args.head, args.kernels, bool(args.window_rows))
        if args.mtp:
            kw["speculate"] = args.mtp
        if args.host_tables:
            kw["host_tables"] = True
        if args.window_rows:
            kw["window_rows"] = args.window_rows
        self.m = warpc.compile(model, ctx_max=ctx_max, rows_max=8, kv=args.kv, **kw)
        self.lineage: list[int] = []
        self.cache = None
        if args.cache == "prefix":
            plan = plan_windows if args.window_rows else class_d_plan
            self.cache = PrefixCache(
                self.m.decoder,
                host_bytes=int(args.store_gb * (1 << 30)),
                device_snapshots=args.device_snaps,
                plan=plan,
                exact=not args.inexact,
            )

    def _sync(self):
        self.torch.cuda.synchronize()
        return time.perf_counter()

    def turn(self, ids, n):
        if self.cache is not None:
            t0 = self._sync()
            first = self.cache.run(ids)
            t1 = self._sync()
            rest = self.cache.decode(n - 1) if n > 1 else []
            t2 = self._sync()
            s = self.cache.stats.last
            return {
                "first_s": t1 - t0,
                "decode_s": t2 - t1,
                "ids": [first, *rest],
                "cached": s.cached,
                "source": s.source,
                "spill_s": s.spill_s,
                "restore_s": s.restore_s,
                "snapshot_s": s.snapshot_s,
                "host_gib": self.cache.stats.host_bytes / (1 << 30),
            }
        lin = self.lineage
        use = len(lin) if self.args.cache == "lineage" and lin and ids[: len(lin)] == lin and len(ids) > len(lin) else 0
        t0 = self._sync()
        if use == 0:
            self.m.reset()
        first = self.m.append(ids[use:])
        t1 = self._sync()
        rest = self.m.decode(n - 1) if n > 1 else []
        t2 = time.perf_counter()
        out = [first, *rest]
        self.lineage = [*ids, *out[:-1]]  # the last token is the anchor: not in the cache
        return {"first_s": t1 - t0, "decode_s": t2 - t1, "ids": out, "cached": use}

    def cold(self, ids, n):
        if self.cache is not None:
            self.cache.clear()
        self.lineage = []
        return self.turn(ids, n)


class VllmEngine:
    def __init__(self, model, ctx_max, args):
        from vllm import LLM, SamplingParams

        for _ in range(60):  # the previous process tree may still be releasing its memory
            used = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            ).stdout.split()
            if not used or int(used[0]) < 600:
                break
            time.sleep(3)
        self.SP = SamplingParams
        kw = {"max_num_batched_tokens": args.max_num_batched_tokens} if args.max_num_batched_tokens else {}
        self.llm = LLM(
            model=str(model),
            dtype="bfloat16",
            max_model_len=ctx_max,
            gpu_memory_utilization=args.util,
            kv_cache_dtype=args.vllm_kv,
            language_model_only=True,
            max_num_seqs=1,
            enable_prefix_caching=True,
            disable_log_stats=False,
            seed=0,
            **kw,
        )
        cc = self.llm.llm_engine.vllm_config.cache_config
        self.capacity = {
            "kv_positions": cc.num_gpu_blocks * cc.block_size,
            "block_size": cc.block_size,
            "mamba_cache_mode": cc.mamba_cache_mode,
        }

    def turn(self, ids, n):
        sampling = self.SP(max_tokens=n, temperature=0.0, ignore_eos=True)
        (o,) = self.llm.generate([{"prompt_token_ids": ids}], sampling, use_tqdm=False)
        m = o.metrics
        return {
            "first_s": m.first_token_ts - m.queued_ts,
            "decode_s": m.last_token_ts - m.first_token_ts,
            "ids": list(o.outputs[0].token_ids),
            "cached": o.num_cached_tokens or 0,
        }

    def cold(self, ids, n):
        self.llm.reset_prefix_cache()
        return self.turn(ids, n)


ENGINES = {"hand": HandEngine, "hand-cold": HandEngine, "warpc": WarpcEngine, "vllm": VllmEngine}


# ---- the loop


def run(args):
    trace = json.loads(Path(args.trace).read_text())
    turns = min(args.turns or len(trace["turns"]), len(trace["turns"]))
    own: list[list[int]] = []
    chat = [t["assistant"] for t in trace["turns"]]
    longest = len(render(trace, turns - 1, "chat", chat)) + max(t["decode"] for t in trace["turns"])
    ctx_max = args.ctx_max or longest + 64
    result = {
        "engine": args.engine,
        "mode": args.mode,
        "turns": [],
        "ctx_max": ctx_max,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    t_all = time.perf_counter()
    with Mem() as mem:
        t0 = time.perf_counter()
        eng = ENGINES[args.engine](args.model, ctx_max, args)
        result["load_s"] = time.perf_counter() - t0
        if hasattr(eng, "capacity"):
            result["capacity"] = eng.capacity
        t_loop = time.perf_counter()
        prev = None
        for k in range(turns):
            n = trace["turns"][k]["decode"]
            ids = render(trace, k, args.mode, own)
            r = eng.turn(ids, n)
            own.append(r["ids"])
            row = {
                "turn": k,
                "prompt": len(ids),
                "new": len(ids) - (common_prefix(prev, ids) if prev else 0),
                "decode_n": n,
                "first_s": r["first_s"],
                "decode_s": r["decode_s"],
                "decode_tok_s": (len(r["ids"]) - 1) / r["decode_s"] if r["decode_s"] else None,
                "cached": r["cached"],
                "first_ids": r["ids"][:8],
                "ids_sha": hashlib.sha256(json.dumps(r["ids"]).encode()).hexdigest()[:12],
            }
            row |= {x: r[x] for x in ("source", "spill_s", "restore_s", "snapshot_s", "host_gib") if x in r}
            result["turns"].append(row)
            shown = {x: (round(v, 3) if isinstance(v, float) else v) for x, v in row.items() if x != "first_ids"}
            print(json.dumps(shown), flush=True)
            prev = ids
        result["loop_s"] = time.perf_counter() - t_loop
        if args.verify_last:
            ids = render(trace, turns - 1, args.mode, own[:-1])
            c = eng.cold(ids, trace["turns"][turns - 1]["decode"])
            diff = next((i for i, (a, b) in enumerate(zip(c["ids"], own[-1], strict=False)) if a != b), None)
            result["verify_last"] = {"equal": c["ids"] == own[-1], "first_diff": diff, "cold_first_s": c["first_s"]}
            print("verify_last", result["verify_last"], flush=True)
        if hasattr(eng, "torch"):
            result["torch_peak_gib"] = eng.torch.cuda.max_memory_allocated() / (1 << 30)
    result["peak_gpu_mib"] = mem.peak_mib
    result["wall_s"] = time.perf_counter() - t_all
    summarize(result)
    Path(args.out).write_text(json.dumps(result))


def summarize(result):
    rows = result["turns"]
    later = rows[1:]
    ttft = sorted(r["first_s"] for r in later)
    p95 = ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))] if ttft else None
    turn_s = [r["first_s"] + r["decode_s"] for r in later]
    summary = {
        "turn0_first_s": rows[0]["first_s"],
        "prefill_s_turns_1plus": sum(r["first_s"] for r in later),
        "decode_s_turns_1plus": sum(r["decode_s"] for r in later),
        "ttft_median_s_turns_1plus": statistics.median(ttft) if ttft else None,
        "ttft_p95_s_turns_1plus": p95,
        "turn_median_s_turns_1plus": statistics.median(turn_s) if turn_s else None,
        "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in rows if r["decode_tok_s"]),
        "cached_fraction_turns_1plus": sum(r["cached"] for r in later) / max(1, sum(r["prompt"] for r in later)),
        "loop_s": result["loop_s"],
    }
    result["summary"] = summary
    print(
        "SUMMARY", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace", help="build the deterministic trace (CPU)")
    t.add_argument("--tokenizer", type=Path, required=True)
    t.add_argument("--files", type=Path, nargs="+", required=True)
    t.add_argument("--prefix-len", type=int, default=65536)
    t.add_argument("--turns", type=int, default=20)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("run", help="replay the trace on one engine")
    r.add_argument("engine", choices=ENGINES)
    r.add_argument("model", type=Path)
    r.add_argument("--trace", type=Path, required=True)
    r.add_argument("--mode", choices=("chat", "linear"), default="chat")
    r.add_argument("--turns", type=int)
    r.add_argument("--ctx-max", type=int)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--verify-last", action="store_true")
    r.add_argument("--snaps", type=int, default=4, help="hand: snapshot buffers")
    r.add_argument("--store-gb", type=float, default=8.0, help="host prefix store budget")
    r.add_argument("--host-tables", action="store_true", help="bf16 embedding + lm_head tables in pinned host memory")
    r.add_argument("--mtp", type=int, default=0, help="MTP draft tokens (0 = plain decode)")
    h = r.add_argument_group("hand")
    h.add_argument("--kv-hand", type=int, default=2, help="0 bf16, 1 fp8, 2 int4_rot")
    h.add_argument("--group", type=int, default=8, help="decode steps per host sync")
    w = r.add_argument_group("warpc")
    w.add_argument("--kv", default="int4_rot", help="KV cache format")
    w.add_argument("--cache", choices=("prefix", "lineage", "none"), default="prefix")
    w.add_argument("--device-snaps", type=int, default=2, help="snapshots of the device lineage")
    w.add_argument(
        "--inexact", action="store_true", help="PrefixCache(exact=False): append to a lineage the prompt extends"
    )
    w.add_argument("--window-rows", type=int, default=0, help="window prefill of up to N rows (0: 8-row steps)")
    w.add_argument("--backend", help="warpc backend (cuda, swiftmk, composite); default: compile's")
    w.add_argument("--head", default="bf16", help="lm_head kernel of the backend")
    w.add_argument("--kernels", help="composite: the warpc kernels admitted")
    v = r.add_argument_group("vllm")
    v.add_argument("--vllm-kv", default="auto")
    v.add_argument("--util", type=float, default=0.93)
    v.add_argument("--max-num-batched-tokens", type=int)
    args = ap.parse_args()
    if args.cmd == "trace":
        tr = build_trace(args.tokenizer, args.files, args.prefix_len, args.turns, args.seed)
        args.out.write_text(json.dumps(tr))
        print(f"prefix {len(tr['prefix'])} tokens, {len(tr['turns'])} turns, corpus {tr['corpus_tokens']} tokens")
    else:
        run(args)


if __name__ == "__main__":
    main()
