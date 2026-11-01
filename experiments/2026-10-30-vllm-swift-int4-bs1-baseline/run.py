"""vLLM stock batch-1 decode baseline for Swift-1.5-Qwen3.8-27b-INT4 on the 3090. See README.md.

reproduce: projects/llm-lab/bin/lab run --detach experiments/2026-10-30-vllm-swift-int4-bs1-baseline/run.py [arm ...]

Arm = k<spec tokens>[-kvbf16]. Each arm is a fresh vLLM engine in its own process.
Decode tok/s for a (prompt, sampler) = N_NEW / (t(max_tokens=N_NEW+1) - t(max_tokens=1)): the prefill
(and its first token) cancel, so only decode steps remain. 5 repeats, median. Prompts are fixed;
the 8K prompt is built from repo text so no external data is needed.
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "projects/llm-lab")
from lab.proc import isolated  # noqa: E402
from lab.record import manifest, write_result  # noqa: E402

EXP = Path(__file__).parent
MODEL = Path.home() / "models" / "swift-1.5-int4"
N_NEW = 384
REPEATS = 3
ARMS = ["k0", "k1", "k3", "k2"]

SHORT = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}


def _long_prompt(tok, n_tokens=8000):
    text = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
    ctx = tok.decode(ids)
    return ctx + "\n\nSummarise the main claims above in a numbered list, citing the section names."


def _arm(name: str) -> dict:
    from vllm import LLM, SamplingParams

    k = int(name.split("-")[0][1:])
    kv = "bfloat16" if name.endswith("kvbf16") else "auto"
    spec = {"method": "mtp", "num_speculative_tokens": k} if k else None
    llm = LLM(model=str(MODEL), served_model_name="swift", dtype="bfloat16", language_model_only=True,
              max_model_len=16384, gpu_memory_utilization=0.93, max_num_seqs=4, max_num_batched_tokens=4096,
              enable_prefix_caching=False, kv_cache_dtype=kv, speculative_config=spec, seed=0,
              disable_log_stats=False)
    tok = llm.get_tokenizer()

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       tokenize=False, enable_thinking=False)

    prompts = {**{f"short-{n}": chat(t) for n, t in SHORT.items()}, "long8k": chat(_long_prompt(tok))}
    samplers = {"greedy": dict(temperature=0.0), "t1": dict(temperature=1.0, top_p=0.95, top_k=20)}

    def metrics():
        out = {}
        for m in llm.get_metrics():
            if m.name.startswith("vllm:spec_decode"):
                out[m.name.split(":")[1]] = list(m.values) if hasattr(m, "values") and m.values is not None else m.value
        return out

    def timed(prompt, sp_kw, n):
        sp = SamplingParams(max_tokens=n, ignore_eos=True, seed=0, **sp_kw)
        t = time.perf_counter()
        r = llm.generate([prompt], sp, use_tqdm=False)
        return time.perf_counter() - t, len(r[0].outputs[0].token_ids)

    timed(prompts["short-code"], samplers["greedy"], 64)  # warm-up (graph replay, JIT)
    rows = {}
    for pname, prompt in prompts.items():
        for sname, kw in samplers.items():
            m0 = metrics()
            dec, pre = [], []
            for _ in range(REPEATS):
                t1, _ = timed(prompt, kw, 1)
                tn, got = timed(prompt, kw, N_NEW + 1)
                assert got == N_NEW + 1, got
                pre.append(t1)
                dec.append(N_NEW / (tn - t1))
            m1 = metrics()
            row = {"decode_tok_s_median": round(statistics.median(dec), 2), "decode_tok_s_all": [round(x, 2) for x in dec],
                   "prefill_1tok_s_median": round(statistics.median(pre), 3)}
            if k:
                d = {key: (m1[key] - m0[key]) if isinstance(m1[key], (int, float)) else [b - a for a, b in zip(m0[key], m1[key])]
                     for key in m1 if key in m0}
                row["spec"] = d
                nd, na = d.get("spec_decode_num_drafts"), d.get("spec_decode_num_accepted_tokens")
                if nd:
                    row["mean_accepted_len"] = round(1 + na / nd, 3)
                if d.get("spec_decode_num_accepted_tokens_per_pos") and nd:
                    row["accept_per_pos"] = [round(x / nd, 3) for x in d["spec_decode_num_accepted_tokens_per_pos"]]
            rows[f"{pname}/{sname}"] = row
            print(name, pname, sname, json.dumps(row), flush=True)
    return {"arm": name, "k": k, "kv_cache_dtype": kv, "rows": rows}


def main() -> None:
    arms = sys.argv[1:] or ARMS
    path = EXP / "results" / "bench.json"
    prev = json.loads(path.read_text())["arms"] if path.exists() else {}
    for name in arms:
        try:
            prev[name] = isolated(_arm, name)
        except Exception as e:  # noqa: BLE001 - record and continue
            prev[name] = {"arm": name, "ok": False, "error": str(e)[-1200:]}
            print(f"{name}: {prev[name]['error']}", flush=True)
        write_result(EXP, "bench", {"arms": prev, "manifest": manifest()})


if __name__ == "__main__":
    main()
