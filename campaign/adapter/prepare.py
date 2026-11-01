#!/usr/bin/env python3
"""prepare.py: build a campaign root (run with the lab venv python; CPU only).

  ~/llm-lab/venv/bin/python campaign/adapter/prepare.py [--root ~/llm-lab/swift-campaign] [--src ~/llm-lab/swift-campaign-src] [--rev HEAD]

1. exports the COMMITTED engine (git archive <rev> engine = ~1.3 MB; no build dirs, no untracked/uncommitted edits) to --src (the `campaign setup` source)
2. installs the harness into the root: campaign.json (+ generated `forbidden` strings), task.md, templates/, adapter/ (without the held-out prompt texts), oracle/ (frozen copy of
   oracle/*.py: the engine's python imports `weights` from there; a copied agent tree has no ../oracle)
3. freezes the prompts as token ids under <root>/private (dev: the E0 prompts; the 8K one is built from topics/*.md NOW and never changes again; held: adapter/held_prompts.py;
   oracle: the 3 swift-ref prompts for teacher forcing).  private/ is outside every agent's tree.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAMP = HERE.parent
REPO = CAMP.parent
sys.path.insert(0, str(HERE))
import held_prompts as HP  # noqa: E402

MODEL = Path.home() / "models" / "swift-1.5-int4"
SHORT = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}
DEV_E6 = ("edit", "table")
HELD_E6 = ("agentedit", "refactor", "docprose", "recmath")
STATIC_FORBIDDEN = ["CAMPAIGN", "bench_spec", "dev_prompts", "held_prompts", "oracle_prompts", "tf_tokens", "swift-campaign", "resource.lock", "limits.json", "/private/", "/proc/", "cmdline",
                    "Fibonacci", "lighthouse keeper", "n squared", "memoisation", "Summarise the main claims above"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "llm-lab" / "swift-campaign"))
    ap.add_argument("--src", default=str(Path.home() / "llm-lab" / "swift-campaign-src"))
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--force", action="store_true", help="replace an existing --src export")
    a = ap.parse_args()
    root, src = Path(a.root).expanduser(), Path(a.src).expanduser()
    dirty = subprocess.run(["git", "status", "--porcelain", "engine"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    rev = subprocess.run(["git", "rev-parse", "--short", a.rev], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    print(f"engine commit {rev}" + (f"; WORKING TREE HAS UNCOMMITTED ENGINE CHANGES (not exported; their authors may still be working):\n{dirty}" if dirty else "; engine working tree clean"))
    if src.exists():
        if not a.force:
            sys.exit(f"{src} exists (--force to replace)")
        shutil.rmtree(src)
    with tempfile.TemporaryDirectory() as td:
        arch = subprocess.run(["git", "archive", "--format=tar", a.rev, "engine"], cwd=REPO, capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", td], input=arch, check=True)
        shutil.copytree(Path(td) / "engine", src)
    (src / ".engine_commit").write_text(rev + "\n")
    # ---- harness into the root
    (root / "private").mkdir(parents=True, exist_ok=True)
    for stale in ("tf_tokens.json", "ref_tpc.json"):  # references of the pristine control: re-created by the first run on the new base/
        (root / "private" / stale).unlink(missing_ok=True)
    for sub in ("adapter", "oracle", "templates"):
        shutil.rmtree(root / sub, ignore_errors=True)
    shutil.copytree(HERE, root / "adapter", ignore=shutil.ignore_patterns("__pycache__", "held_prompts.py", "prepare.py"))
    shutil.copytree(CAMP / "templates", root / "templates")
    shutil.copytree(REPO / "oracle", root / "oracle", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(CAMP / "task.md", root / "task.md")
    # ---- prompts
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))

    def chat(text):
        return tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, enable_thinking=False)

    def ids_of(text):
        return tok(chat(text), add_special_tokens=False)["input_ids"]

    def doc_prompt(glob, ntok, suffix):
        raw = "\n\n".join(p.read_text(errors="ignore") for p in sorted((REPO / glob[0]).glob(glob[1])))
        ids = tok(raw, add_special_tokens=False)["input_ids"][:ntok]
        return ids_of(tok.decode(ids) + suffix)

    dev = {n: ids_of(t) for n, t in SHORT.items()}
    dev["long8k"] = doc_prompt(("topics", "*.md"), 8000, "\n\nSummarise the main claims above in a numbered list, citing the section names.")  # = tests/bench_spec.py
    e6 = json.loads((CAMP / "frozen" / "e6_prompts.json").read_text())  # K11 workload: E6 (ngram-draft-potential) tasks as frozen token ids (the trace lives in a gitignored runs/ dir)
    for n in DEV_E6:  # two copy-heavy tasks of typical (not extreme) copy share, a7 add validation + docstrings to two methods of a file (E 4.9 tokens/cycle) and b2 nvcc/ptxas log -> table (E 5.6): the prompt-lookup tail is active on them
        dev[n] = e6[n]["ids"]
    held = {n: ids_of(t) for n, t in HP.TEXTS.items()}
    held["docqa"] = doc_prompt(("papers", "*.md"), HP.DOC_TOKENS, HP.DOC_QUESTION)
    held = {n: held[n] for n in ("jsontool", "sql", "docqa", "reason")}
    for n in HELD_E6:  # PLD-friendly (c3 agent turn edit_file with quoted anchors, a8 refactor into a dataclass) and PLD-hostile (d1 long-document opinion piece, e4 recurrence derivation) tasks the dev set never uses
        held[n] = e6[n]["ids"]
    import torch

    orc = {n: torch.load(Path.home() / "llm-lab/swift-ref/bf16" / f"{n}.pt")["prompt_ids"].tolist() for n in ("code", "prose", "math")}
    P = root / "private"
    for name, obj in (("dev_prompts", dev), ("held_prompts", held), ("oracle_prompts", orc)):
        (P / f"{name}.json").write_text(json.dumps(obj))
    # ---- config with generated forbidden strings: token-id windows of every prompt (both separators) so a hard-coded id list fails
    cfg = json.loads((CAMP / "campaign.json").read_text())
    forb = list(cfg.get("forbidden", [])) + STATIC_FORBIDDEN + HP.MARKERS
    for ids in list(dev.values()) + list(held.values()) + list(orc.values()):
        w = ids[8:16] if len(ids) > 20 else ids[4:12]
        forb += [", ".join(map(str, w)), ",".join(map(str, w))]
    # the gate scans the whole diff (context + removed lines too): a string that already occurs in the pristine tree would fire on innocent edits next to it, so drop those
    blob = "\n".join(f.read_text(errors="ignore") for f in src.rglob("*") if f.is_file() and f.suffix in (".py", ".cu", ".cuh", ".h", ".md", ".json", ".txt", "") or f.name == "Makefile")
    dropped = sorted(f for f in set(forb) if f in blob)
    cfg["forbidden"] = sorted(set(forb) - set(dropped))
    print(f"forbidden strings dropped because the pristine tree contains them: {dropped}")
    (root / "campaign.json").write_text(json.dumps(cfg, indent=1))
    print(f"root {root}: dev prompts {{{', '.join(f'{n}:{len(v)}' for n, v in dev.items())}}} held {{{', '.join(f'{n}:{len(v)}' for n, v in held.items())}}}; {len(cfg['forbidden'])} forbidden strings")
    print(f"next: campaign --root {root} setup {src}")


if __name__ == "__main__":
    main()
