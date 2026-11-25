"""Correctness of warpc on a real checkpoint over fixed text prompts.

  uv run --package warpc python warpc/tools/check.py PATH [--tokenizer DIR] [--greedy N] [--json FILE]

PATH is an HF checkpoint directory or a GGUF file.
Teacher-forced: at every prompt position warpc's next-token logits (the prompt fed one step per token) against an
oracle's, as top-1 agreement and the relative L2 error |warpc - oracle| / |oracle|. The oracle of an HF directory is
transformers in bf16 on the GPU; of a GGUF file, warpc.reference with fp32 activations on the GPU, on the same file.
Greedy: the first N tokens of each prompt's greedy continuation against a baseline engine's, as the length of the
agreeing prefix and, where they part, warpc's logit margin of its own token over the baseline's. The baseline of an HF
directory is vLLM (vllm_worker.py), of a GGUF file llama.cpp (llama_server.py; skipped when llama-server is absent). The
baseline stops after an end-of-sequence token, so it can compare fewer tokens.
The prompts are tokenized by the HF tokenizer in --tokenizer (default: PATH when it is a directory).
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import llama_server
import vllm_worker
import warpc
from warpc import frontend
from warpc.reference import Interpreter

PROMPTS = (
    # Dickens, A Tale of Two Cities (1859)
    "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it "
    "was the epoch of belief, it was the epoch of incredulity,",
    # Lincoln, the Gettysburg Address (1863)
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, "
    "and dedicated to the proposition that",
    "Water boils at 100 degrees Celsius at sea level. At higher altitudes the air pressure is lower, so water boils at",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
)


def free_gpu() -> None:
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def transformers_logits(directory: Path, prompts: list[list[int]]) -> list[torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16).to("cuda")
    out = [model(torch.tensor([ids], device="cuda")).logits[0].float().cpu() for ids in prompts]
    del model
    free_gpu()
    return out


def reference_logits(source, prompts: list[list[int]]) -> list[torch.Tensor]:
    graph = frontend.build(source, ctx_max=max(map(len, prompts)), kv="bf16")
    interpreter = Interpreter(graph, source, device="cuda", dtype=torch.float32)
    out = []
    for ids in prompts:
        interpreter.reset()
        window = {"tokens": torch.tensor(ids, dtype=torch.int32), "pos": torch.arange(len(ids), dtype=torch.int32)}
        out.append(interpreter.run(window)["logits"].cpu())
    del interpreter
    free_gpu()
    return out


def teacher_forced(model: warpc.Model, ids: list[int]) -> torch.Tensor:
    """fp32 [len(ids), vocab]: the logits after each prompt token, fed one step at a time."""
    model.reset()
    model.prefill(ids[:1])
    rows = [model.logits.cpu()]
    for token in ids[1:]:
        model.step(token)
        rows.append(model.logits.cpu())
    return torch.stack(rows)


def compare(got: list[torch.Tensor], want: list[torch.Tensor]) -> dict:
    g, w = torch.cat(got), torch.cat(want)
    rel = (g - w).norm(dim=-1) / w.norm(dim=-1)
    top1 = (g.argmax(-1) == w.argmax(-1)).float().mean()
    return {"positions": len(g), "top1": float(top1), "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max())}


def agreement(model: warpc.Model, prompts: list[list[int]], ours: list[list[int]], theirs: list[list[int]]) -> dict:
    """Per prompt, the length of the prefix our and their greedy tokens share; where they part, warpc's logit margin of
    its own token over theirs (a small margin is a near tie that rounding differences decide)."""
    matched, margins = [], []
    for ids, a, b in zip(prompts, ours, theirs, strict=True):
        k = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        matched.append(k)
        if k < min(len(a), len(b)):
            model.reset()
            model.prefill(ids + a[:k])
            margins.append(float(model.logits[a[k]] - model.logits[b[k]]))
    return {"matched": matched, "compared": [len(b) for b in theirs], "margins": margins}


def check(path: Path, tokenizer: Path, greedy: int) -> dict:
    tok = AutoTokenizer.from_pretrained(tokenizer)
    prompts = [tok(text).input_ids for text in PROMPTS]
    ctx_max = max(map(len, prompts)) + greedy
    gguf = path.is_file()
    baseline = None  # {"against": engine, "tokens": its greedy tokens} or {"against": engine, "error": why not}
    if not gguf:  # first, while this process holds no GPU memory
        mode = ["greedy", "--prompts", json.dumps(prompts), "--tokens", str(greedy)]
        try:
            vllm = vllm_worker.spawn(path, ctx_max=ctx_max, mode=mode)
            baseline = {"against": f"vLLM {vllm['vllm']}", "tokens": vllm["tokens"]}
        except RuntimeError as e:
            baseline = {"against": "vLLM", "error": str(e)}
    source = warpc.open(path)
    oracle = reference_logits(source, prompts) if gguf else transformers_logits(path, prompts)
    model = warpc.compile(source, ctx_max=ctx_max)
    result = {"oracle": "warpc.reference fp32" if gguf else "transformers bf16", "plan": model.plan.summary()}
    result |= compare([teacher_forced(model, ids) for ids in prompts], oracle)
    ours = [model.generate(ids, greedy) for ids in prompts]
    if gguf and llama_server.binary():
        with llama_server.LlamaServer(path, ctx_max) as server:
            baseline = {"against": "llama.cpp", "tokens": [server.complete(ids, greedy)["tokens"] for ids in prompts]}
    if baseline is not None:
        theirs = baseline.pop("tokens", None)
        result["greedy"] = baseline if theirs is None else baseline | agreement(model, prompts, ours, theirs)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", type=Path, help="HF checkpoint directory or *.gguf file")
    ap.add_argument("--tokenizer", type=Path, help="HF directory with the tokenizer (default: PATH if a directory)")
    ap.add_argument("--greedy", type=int, default=64, help="greedy tokens compared with the baseline engine")
    ap.add_argument("--json", type=Path, help="also write the result to this file")
    args = ap.parse_args()
    tokenizer = args.tokenizer or (args.path if args.path.is_dir() else None)
    if tokenizer is None:
        raise SystemExit("a GGUF file needs --tokenizer DIR (the HF checkpoint of the same model)")
    result = check(args.path, tokenizer, args.greedy)
    print(f"{args.path.name}: {result['plan']}")
    print(
        f"teacher-forced vs {result['oracle']}: {result['positions']} positions, top-1 {result['top1']:.1%}, "
        f"relative L2 mean {result['rel_l2_mean']:.4f} max {result['rel_l2_max']:.4f}"
    )
    greedy = result.get("greedy")
    if greedy is None:
        print("greedy vs llama.cpp skipped: llama-server not found (set LLAMA_SERVER or put it on PATH)")
    elif "error" in greedy:
        print(f"greedy vs {greedy['against']} failed: {greedy['error']}")
    else:
        pairs = zip(greedy["matched"], greedy["compared"], strict=True)
        margins = ", ".join(f"{m:.3f}" for m in greedy["margins"]) or "none"
        print(f"greedy prefix equal to {greedy['against']}'s: {' '.join(f'{m}/{n}' for m, n in pairs)}")
        print(f"warpc's logit margin where they part: {margins}")
    if args.json:
        args.json.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
