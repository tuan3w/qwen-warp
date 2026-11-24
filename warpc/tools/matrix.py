"""check.py and bench.py over the models of models.toml, merged into one results file and printed as a markdown table.

  uv run --package warpc python warpc/tools/matrix.py [MODEL ...] [--dir DIR] [--results FILE] [--steps STEP,...] \
      [--formats FORMAT,...] [--table]

Steps (default all): check (check.py on every format), warpc (bench.py warpc on every format), llama-server (GGUF
formats), vllm (every format; a GGUF file gets the HF directory as tokenizer and config). Every bench step runs
at ctx_max 512 and 4096. Each measurement is a process of its own; one that fails is recorded with its error and the
others still run. Results (default DIR/matrix.json) are keyed by model, format and measurement, so runs over different
models or steps add up; the table covers everything in the file. --table only prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

import catalog
import child
import llama_server

TOOLS = Path(__file__).parent
STEPS = ("check", "warpc", "llama-server", "vllm")
CTX = (512, 4096)


def measure(tool: str, args: list[str]) -> dict:
    """Run tools/`tool` with `args` and `--json`: its result, or {"error": the reason it failed}."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        command = [sys.executable, str(TOOLS / tool), *args, "--json", str(out)]
        print(f"$ {' '.join(command[1:-2])}", flush=True)
        error = child.run(command)
        return {"error": error} if error else json.loads(out.read_text())


def jobs(entry: catalog.Entry, root: Path, fmt: str, steps: list[str]) -> list[tuple[str, str, list[str]]]:
    """(result key, tool, arguments) of every measurement of one model format."""
    path, hf = str(entry.path(root, fmt)), str(entry.hf_dir(root))
    gguf = fmt != catalog.HF
    tokenizer = ["--tokenizer", hf] if gguf else []
    out = []
    if "check" in steps:
        out.append(("check", "check.py", [path, *tokenizer]))
    for ctx in CTX:
        if "warpc" in steps:
            out.append((f"warpc@{ctx}", "bench.py", ["warpc", path, "--ctx-max", str(ctx)]))
        if "llama-server" in steps and gguf:
            out.append((f"llama-server@{ctx}", "bench.py", ["llama-server", path, "--ctx-max", str(ctx)]))
        if "vllm" in steps:
            out.append((f"vllm@{ctx}", "bench.py", ["vllm", path, "--ctx-max", str(ctx), *tokenizer]))
    return out


class Notes:
    """Footnotes for cells that hold an error instead of a number, one per distinct text."""

    def __init__(self):
        self.texts: list[str] = []

    def ref(self, text: str) -> str:
        if text not in self.texts:
            self.texts.append(text)
        return f"[{self.texts.index(text) + 1}]"


def table(results: dict) -> str:
    entries = {e.name: e for e in catalog.load()}
    vllm = f"vLLM {version('vllm')}"
    llama = f"llama.cpp {llama_server.version() or '(llama-server not found)'}"
    head = ["model", "format", "top-1", "rel. L2", f"greedy = {vllm}", "greedy = llama.cpp"]
    head += [f"{engine} {ctx}" for engine in ("warpc", "llama.cpp", vllm) for ctx in CTX]
    notes = Notes()

    def cell(result: dict | None, render) -> str:
        if result is None:
            return ""
        return notes.ref(result["error"]) if "error" in result else render(result)

    def prefix(greedy: dict) -> str:
        text = " ".join(f"{m}/{n}" for m, n in zip(greedy["matched"], greedy["compared"], strict=True))
        return f"{text}, margin ≤ {max(greedy['margins']):.2f}" if greedy["margins"] else text

    rows = [head, ["---"] * len(head)]
    for name in sorted(results, key=list(entries).index):
        for fmt in catalog.FORMATS:
            r = results[name].get(fmt)
            if r is None:
                continue
            check = r.get("check")
            measured = check is not None and "error" not in check
            greedy = check.get("greedy") if measured else None
            gguf = fmt != catalog.HF
            row = [name, "HF bf16" if fmt == catalog.HF else f"GGUF {fmt}"]
            row.append(cell(check, lambda c: f"{c['top1']:.1%}"))
            row.append(f"{check['rel_l2_mean']:.4f}" if measured else "")
            row += ["–", cell(greedy, prefix)] if gguf else [cell(greedy, prefix), "–"]
            for engine in ("warpc", "llama-server", "vllm"):
                for ctx in CTX:
                    applies = engine != "llama-server" or gguf
                    row.append(cell(r.get(f"{engine}@{ctx}"), lambda b: f"{b['tok_s']:.0f}") if applies else "–")
            rows.append(row)
    lines = [f"| {' | '.join(row)} |" for row in rows]
    lines.append(f"\n{llama}; tok/s columns by ctx_max.")
    lines += [f"{notes.ref(text)} {text}" for text in notes.texts]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    catalog.add_arguments(ap)
    ap.add_argument("--results", type=Path, help="results file (default: DIR/matrix.json)")
    ap.add_argument("--steps", default=",".join(STEPS), help=f"comma-separated subset of {', '.join(STEPS)}")
    ap.add_argument("--formats", default=",".join(catalog.FORMATS), help="comma-separated subset of bf16,Q4_0,Q8_0")
    ap.add_argument("--table", action="store_true", help="only print the table of the results file")
    args = ap.parse_args()
    root = catalog.models_dir(args)
    results_file = args.results or root / "matrix.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    steps, formats = args.steps.split(","), args.formats.split(",")
    if unknown := set(steps) - set(STEPS) | set(formats) - set(catalog.FORMATS):
        raise SystemExit(f"unknown steps or formats: {', '.join(sorted(unknown))}")
    for entry in [] if args.table else catalog.select(args.models):
        for fmt in [f for f in entry.formats if f in formats]:
            if not entry.path(root, fmt).exists():
                print(f"skip {entry.name} {fmt}: {entry.path(root, fmt)} missing, run fetch.py", file=sys.stderr)
                continue
            for key, tool, tool_args in jobs(entry, root, fmt, steps):
                result = measure(tool, tool_args)
                if "error" in result:  # one text per kind of failure, without where this machine keeps its models
                    text = result["error"].replace(str(entry.path(root, fmt)), "MODEL")
                    result["error"] = text.replace(str(root), "$WARPC_MODELS")
                results.setdefault(entry.name, {}).setdefault(fmt, {})[key] = result
                results_file.write_text(json.dumps(results, indent=1))
    print(table(results))


if __name__ == "__main__":
    main()
