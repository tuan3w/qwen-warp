"""Long prompts of real text for the 27B comparisons: the checkpoint's tokenizer over files concatenated in sorted
path order, truncated to the wanted number of tokens. Every side of a comparison gets the same ids."""

from __future__ import annotations

from pathlib import Path


def prompt_ids(tokenizer_dir: Path, files: list[Path], length: int) -> list[int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    text = "\n\n".join(Path(f).read_text(errors="replace") for f in sorted(files, key=str))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < length:
        raise ValueError(f"the prompt files give {len(ids)} tokens, fewer than the {length} asked for")
    return ids[:length]


def add_arguments(ap) -> None:
    ap.add_argument("--prompt-files", type=Path, nargs="+", required=True, help="text the prompt is cut from")
    ap.add_argument("--prompt-len", type=int, nargs="+", required=True, help="prompt length(s) in tokens")
