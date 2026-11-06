"""Build the real agent-style documents used for the trace dump (SparseTraces).  Reproducible: `python build_docs.py` writes
/media/fatman/data/sparselab/traces/docs/<name>.npy (int32 token ids of the chat-templated prompt, ending with the assistant generation prompt) and <name>.json (sources).

Each document = system + ONE user message with real files / diffs / logs (disjoint source families per document) and a final question that needs a long answer, so
that the model's own greedy continuation (its 'thinking') gives 128+ true decode tokens.  Targets: 32k tokens (six docs) and one 128k doc."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

R = Path("/media/fatman/data/workspace/research")
Q = Path("/media/fatman/data/workspace/qwen-warp-traces")
OUT = Path("/media/fatman/data/sparselab/traces/docs")
MODEL = "/media/fatman/data/models/Swift-1.5-Qwen3.8-27b-INT4"
tok = AutoTokenizer.from_pretrained(MODEL)


def files(root, pats, maxsize=60_000, skip=()):
    out = []
    for p in pats:
        out += sorted(root.glob(p))
    return [f for f in out if f.is_file() and 500 < f.stat().st_size < maxsize and not any(s in str(f) for s in skip)]


def git_diffs(repo, n):
    return subprocess.run(["git", "-C", str(repo), "log", "-p", f"-{n}", "--no-color", "--stat"], capture_output=True, text=True).stdout


FAMILIES = {
    # name: (list of (label, text) generator)
    "code_cuda": lambda: [(str(f.relative_to(Q)), f.read_text(errors="ignore")) for f in files(Q, ["engine/src/*.cu", "engine/src/*.cuh", "engine/include/*.h"])],
    "notes_md": lambda: [(str(f.relative_to(R)), f.read_text(errors="ignore")) for f in files(R, ["topics/*.md", "notes/*.md", "ideas/*.md"])],
    "vendor_py": lambda: [(str(f.relative_to(R)), f.read_text(errors="ignore")) for f in files(R, ["vendor/Quest/**/*.py", "vendor/MInference/minference/**/*.py", "vendor/SeerAttention/**/*.py", "vendor/ShadowKV/**/*.py"], maxsize=40_000)],
    "vendor_cu": lambda: [(str(f.relative_to(R)), f.read_text(errors="ignore")) for f in files(R, ["vendor/FlashMLA/csrc/**/*.cu", "vendor/FlashMLA/csrc/**/*.h", "vendor/FlashMLA/**/*.py", "vendor/DeepGEMM/**/*.py", "vendor/MegaQwen/**/*.cu"], maxsize=50_000)],
    "gitlog": lambda: [("git log -p qwen-warp", git_diffs(Q, 80))],
    "papers_md": lambda: [(str(f.relative_to(R)), f.read_text(errors="ignore")) for f in files(R, ["papers/*.md", "sources/*.md"], maxsize=30_000)],
    "agent_logs": lambda: [(str(f.relative_to(R)), f.read_text(errors="ignore")) for f in files(R, ["experiments/*/notes.md", "experiments/*/README.md", "experiments/*/results/*.md", "log.md", "projects/*/*/notes.md"], maxsize=40_000)],
}

QUESTIONS = [
    "Study every file above. Then write a detailed technical review: list the five most important design decisions you can find, explain for each how it works step by step, which numbers it relies on, and what could go wrong. Quote concrete identifiers from the files.",
    "Walk through the material above like a senior engineer onboarding a new teammate: summarise the structure, explain the three trickiest parts in depth with small worked examples, and list open questions you would ask the authors.",
]


def build(name, family, target, seed, qi, shuffle=True):
    rnd = random.Random(seed)
    items = FAMILIES[family]()
    if shuffle:
        rnd.shuffle(items)
    sysmsg = "You are a careful software engineering agent. The user shares repository files, diffs and logs; read them fully before answering."
    parts, srcs, ntok = [], [], 0
    qtxt = QUESTIONS[qi]
    budget = target - 300  # template + question
    for label, text in items:
        chunk = f"\n\n### FILE: {label}\n```\n{text}\n```"
        n = len(tok(chunk, add_special_tokens=False).input_ids)
        if ntok + n > budget:
            if budget - ntok < 1500:
                break
            ids = tok(chunk, add_special_tokens=False).input_ids[: budget - ntok]
            chunk = tok.decode(ids)
            n = len(ids)
        parts.append(chunk)
        srcs.append(label)
        ntok += n
        if ntok >= budget:
            break
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": "Here are the files.\n" + "".join(parts) + "\n\n" + qtxt}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = np.array(tok(s, add_special_tokens=False).input_ids, dtype=np.int32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}.npy", ids)
    (OUT / f"{name}.json").write_text(json.dumps(dict(family=family, n_tokens=int(len(ids)), n_files=len(srcs), sources=srcs, question=qtxt), indent=1))
    print(name, family, len(ids), "tokens", len(srcs), "files", flush=True)


if __name__ == "__main__":
    plan = [("tr32_code_cuda", "code_cuda", 32000, 1, 0), ("tr32_notes", "notes_md", 32000, 2, 1), ("tr32_vendor_py", "vendor_py", 32000, 3, 0), ("tr32_vendor_cu", "vendor_cu", 32000, 4, 1),
            ("tr32_gitlog", "gitlog", 32000, 5, 0), ("tr32_papers", "papers_md", 32000, 6, 1), ("tr32_agentlogs", "agent_logs", 32000, 7, 0),
            ("tr128_mixed", "agent_logs", 128000, 8, 1)]
    only = sys.argv[1:]
    for p in plan:
        if not only or p[0] in only:
            build(*p)
