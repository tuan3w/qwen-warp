"""Long prompts of real text for the 27B comparisons: the checkpoint's tokenizer over files concatenated in sorted
path order, truncated to the wanted number of tokens. Every side of a comparison gets the same ids."""

from __future__ import annotations

from pathlib import Path


def prompt_ids(tokenizer_dir: Path, files: list[Path], length: int, offset: int = 0) -> list[int]:
    """`length` token ids starting `offset` tokens into the text (disjoint slices for several prompts of one length)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    text = "\n\n".join(Path(f).read_text(errors="replace") for f in sorted(files, key=str))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) < offset + length:
        raise ValueError(f"the prompt files give {len(ids)} tokens, fewer than the {offset + length} asked for")
    return ids[offset : offset + length]


SLICE_STRIDE = 4608  # tokens between the starts of the slices: disjoint for prompts of up to 4,096 tokens


def cells(tokenizer_dir: Path, files: list[Path], lengths: list[int], slices: int):
    """(length, slice, ids) for every prompt length and slice: slice k starts k * SLICE_STRIDE tokens into the text."""
    for length in lengths:
        for k in range(slices):
            yield length, k, prompt_ids(tokenizer_dir, files, length, k * SLICE_STRIDE)


def add_slices(ap) -> None:
    ap.add_argument("--slices", type=int, default=3, help="disjoint prompts per length")


def add_arguments(ap) -> None:
    ap.add_argument("--prompt-files", type=Path, nargs="+", required=True, help="text the prompt is cut from")
    ap.add_argument("--prompt-len", type=int, nargs="+", required=True, help="prompt length(s) in tokens")
