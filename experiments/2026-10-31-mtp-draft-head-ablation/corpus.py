"""Token-frequency corpus for the restricted draft vocabulary (CPU only). See README.md.

Corpus = python / markdown / C-family / rust / ts / go source under vendor/ (per-extension byte caps, per-file cap) plus
topics/*.md. It never touches papers/, sources/, experiments/ (the long-context eval prompts are built from
papers/ + sources/) nor any eval prompt text, so the frequency list is held out from the eval set.

reproduce: nice -n 10 ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/corpus.py
Writes ~/llm-lab/mtp-ablation/freq.npy (token counts, int64[vocab_size 248320]) and results/vocab_corpus.json.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EXP = Path(__file__).parent
REPO = EXP.parents[1]
MODEL = Path.home() / "models" / "swift-1.5-int4"
OUT = Path.home() / "llm-lab" / "mtp-ablation"
CAPS = {  # extension -> max bytes of text taken from vendor/
    ".py": 40e6, ".md": 25e6, ".c": 8e6, ".h": 8e6, ".cu": 6e6, ".cuh": 4e6, ".cpp": 8e6, ".rs": 10e6, ".ts": 10e6, ".go": 10e6,
}
PER_FILE = 100_000
VOCAB = 248320


def files():
    used = defaultdict(float)
    for root, dirs, fs in os.walk(REPO / "vendor"):
        dirs[:] = sorted(d for d in dirs if d not in {".git", "node_modules", "__pycache__", "dist", "build"})
        for f in sorted(fs):
            ext = Path(f).suffix
            if ext not in CAPS or used[ext] >= CAPS[ext]:
                continue
            p = Path(root) / f
            try:
                txt = p.read_text()[:PER_FILE]
            except (UnicodeDecodeError, OSError):
                continue
            used[ext] += len(txt)
            yield ext, txt
    for p in sorted((REPO / "topics").glob("*.md")):
        yield "topics.md", p.read_text()


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL))
    counts = np.zeros(VOCAB, dtype=np.int64)
    stats = defaultdict(lambda: {"files": 0, "bytes": 0, "tokens": 0})
    batch, kinds = [], []

    def flush():
        ids = tok(batch, add_special_tokens=False)["input_ids"]
        for k, row in zip(kinds, ids):
            counts.__iadd__(np.bincount(np.asarray(row, dtype=np.int64), minlength=VOCAB))
            stats[k]["tokens"] += len(row)
        batch.clear()
        kinds.clear()

    for ext, txt in files():
        stats[ext]["files"] += 1
        stats[ext]["bytes"] += len(txt)
        batch.append(txt)
        kinds.append(ext)
        if len(batch) >= 256:
            flush()
    if batch:
        flush()
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "freq.npy", counts)
    (EXP / "results").mkdir(exist_ok=True)
    nz = int((counts > 0).sum())
    top = np.argsort(-counts)
    cum = np.cumsum(counts[top]) / counts.sum()
    summary = {"by_kind": stats, "total_tokens": int(counts.sum()), "distinct_tokens_seen": nz,
               "coverage_of_corpus_by_top_V": {str(v): round(float(cum[v - 1]), 4) for v in (8192, 16384, 32768, 65536)}}
    (EXP / "results" / "vocab_corpus.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
