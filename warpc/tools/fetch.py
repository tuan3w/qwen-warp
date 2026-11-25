"""Download the checkpoints of models.toml into the models directory. Files already there with the expected size are
skipped; an interrupted download resumes on the next run.

  uv run --package warpc python warpc/tools/fetch.py [MODEL ...] [--dir DIR] [--dry-run]

The Hugging Face token, if one is needed, comes from HF_TOKEN or the huggingface_hub login.
"""

import argparse
from fnmatch import fnmatchcase
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

import catalog

# config, tokenizer and transformers-layout weights; leaves out READMEs and duplicate weight layouts (consolidated.*)
HF_FILES = ("*.json", "*.txt", "model*.safetensors", "tokenizer*")


def artifacts(api: HfApi, entry: catalog.Entry, root: Path) -> list[tuple[str, catalog.Repo, dict[str, int], Path]]:
    """(format, repository, {file: bytes}, local directory) of every artifact of `entry`."""

    def sizes(repo: catalog.Repo) -> dict[str, int]:
        info = api.model_info(repo.repo, revision=repo.revision, files_metadata=True)
        return {s.rfilename: s.size for s in info.siblings}

    hf = {f: n for f, n in sizes(entry.hf).items() if any(fnmatchcase(f, p) for p in HF_FILES)}
    out = [(catalog.HF, entry.hf, hf, entry.hf_dir(root))]
    gguf = sizes(entry.gguf)
    for quant, name in entry.gguf_files.items():
        if name not in gguf:
            raise SystemExit(f"{entry.name}: {quant} file {name} not in {entry.gguf.repo}@{entry.gguf.revision}")
        out.append((quant, entry.gguf, {name: gguf[name]}, root / entry.name))
    return out


def present(path: Path, size: int) -> bool:
    return path.is_file() and path.stat().st_size == size


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    catalog.add_arguments(ap)
    ap.add_argument("--dry-run", action="store_true", help="only list the artifacts and their sizes")
    args = ap.parse_args()
    root = catalog.models_dir(args)
    api = HfApi()
    total = missing = 0
    for entry in catalog.select(args.models):
        for fmt, repo, files, directory in artifacts(api, entry, root):
            absent = {f: n for f, n in files.items() if not present(directory / f, n)}
            size = sum(files.values())
            total += size
            missing += sum(absent.values())
            state = f"from {repo.repo}" if absent else "present"
            print(f"{entry.name:<26} {fmt:<5} {size / 1e9:6.2f} GB  {entry.path(root, fmt).name}  {state}", flush=True)
            for name in [] if args.dry_run else absent:
                hf_hub_download(repo.repo, name, revision=repo.revision, local_dir=directory)
    verb = "to fetch" if args.dry_run else "fetched"
    print(f"total {total / 1e9:.2f} GB, {verb} {missing / 1e9:.2f} GB, in {root}")


if __name__ == "__main__":
    main()
