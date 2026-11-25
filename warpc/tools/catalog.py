"""The model list (models.toml) and where each artifact lives under the models directory:

DIR/<name>/hf/          the transformers checkpoint
DIR/<name>/<file>.gguf  one file per quantisation, named as in its repository
"""

from __future__ import annotations

import argparse
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

LIST = Path(__file__).with_name("models.toml")
HF = "bf16"
FORMATS = (HF, "Q4_0", "Q8_0")


@dataclass(frozen=True)
class Repo:
    repo: str
    revision: str


@dataclass(frozen=True)
class Entry:
    name: str
    family: str
    hf: Repo
    gguf: Repo
    gguf_files: dict[str, str]  # quantisation -> file name in the gguf repository

    @property
    def formats(self) -> list[str]:
        return [f for f in FORMATS if f == HF or f in self.gguf_files]

    def hf_dir(self, root: Path) -> Path:
        return root / self.name / "hf"

    def path(self, root: Path, fmt: str) -> Path:
        """The checkpoint of one format: the hf directory for bf16, else the GGUF file."""
        return self.hf_dir(root) if fmt == HF else root / self.name / self.gguf_files[fmt]


def load() -> list[Entry]:
    with LIST.open("rb") as f:
        models = tomllib.load(f)["model"]
    entries = []
    for m in models:
        gguf = dict(m["gguf"])
        repo = Repo(gguf.pop("repo"), gguf.pop("revision"))
        if unknown := set(gguf) - set(FORMATS):
            raise ValueError(f"{LIST.name}: {m['name']}: unknown quantisation {sorted(unknown)}")
        entries.append(Entry(m["name"], m["family"], Repo(**m["hf"]), repo, gguf))
    return entries


def select(names: list[str]) -> list[Entry]:
    """The named entries in list order, all of them for no names."""
    entries = load()
    if unknown := set(names) - {e.name for e in entries}:
        raise SystemExit(
            f"not in {LIST.name}: {', '.join(sorted(unknown))}; known: {', '.join(e.name for e in entries)}"
        )
    return [e for e in entries if not names or e.name in names]


def add_arguments(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("models", nargs="*", help=f"model names from {LIST.name} (default: all)")
    ap.add_argument("--dir", help="models directory (default: $WARPC_MODELS)")


def models_dir(args: argparse.Namespace) -> Path:
    directory = args.dir or os.environ.get("WARPC_MODELS")
    if not directory:
        raise SystemExit("pass --dir or set WARPC_MODELS")
    return Path(directory)
