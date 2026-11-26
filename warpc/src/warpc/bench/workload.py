"""The prompts of every bench workload, built in one place: the text corpus, the ~100k-token agent trace, the probes
and their exact-match scorers.

Nothing is stored: the trace is a pure function of (corpus bytes, tokenizer, seeds, this code) and carries its hashes,
`corpus_sha256`, `tokenizer_sha256` and `trace_sha256`, into every cell that uses it, so a changed input is a
different workload and never a silently different number.

The corpus is `engine/src/*.cu` of the repository's own v0.12.0 tag (the text the legacy gates used), read with
`git show`, so no checkout or build of the legacy tree is needed; `WARPC_LEGACY_DIR` is only the fallback for a clone
without the tag. `corpus_ids` cuts prompts of a given length from it (the same ids as the old `prompts.prompt_ids`).

The agent trace (`build_trace`): a system message holding the haystack with planted facts, a task, 12 turns that each
append an assistant message and a tool result, then 19 probe forks of the final history, each asking one question
about a planted fact:

    prompt_k   = prefix + sum_{j<k}(assistant_j, tool_j) + <|im_start|>assistant\\n <think>\\n
    probe_i    = history + <|im_start|>user\\n Q_i <|im_end|>\\n <|im_start|>assistant\\n <think>\\n\\n</think>\\n\\n

The longest probe prompt is exactly `scale` tokens. Every id of the chat format comes from the tokenizer, so a
tokenizer without ChatML and <think> is `Unsupported`."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from . import schema

if TYPE_CHECKING:
    from .engines import Engine, Turn

CORPUS_TAG = "v0.12.0"
CORPUS_DIR = "engine/src"
CORPUS_BYTES = 409_853  # the 16 files, before joining
CORPUS_SHA256 = "d63dd146abe0d3b20dde243a26bc1a6c1582357cc00eae557697bada42e0663b"  # of the files joined with \n\n
MIN_CORPUS_TOKENS = 100_000  # measured 161,644 (27B tokenizer), 155,113 (Qwen3): a truncating tokenizer gives 4,096
LEGACY_ENV = "WARPC_LEGACY_DIR"

SLICE_STRIDE = 4608  # tokens between the slices of prompts up to 4,096 long (the 385 / 4,096 gates)
SCALES = {"100k": 100_000, "64k": 64_000, "32k": 32_000}
TURNS = 12
TOOL_TOKENS = (500, 2000)
DECODE_TOKENS = (100, 300)  # numpy randint, upper bound exclusive: 100..299
LAST_TOOL_MIN = 900  # R_11: room for the three tail needles at 128 / 380 / 700 tokens before the question
PROBE_NEW = 24
JSON_NEW = 40  # a fenced {"vault": ..., "code": ...} reply is ~16-26 tokens
QUESTION_MAX = 240  # windows.W4_MAX: a longer question would add a plan cut inside the fork
TAIL_DISTANCES = (128, 380, 700)
TAIL_SNAP_TOLERANCE = 32
SNAP_REACH = 64  # tokens a fact moves to land on a line start
INFORMATIVE_MIN = 16  # probes a bf16 run must get right for the workload to say anything about a KV format
MIN_DEVICE_SNAPSHOTS = 2  # the system checkpoint and the fork point
VLLM_MTP_MAX_CTX = 17_600  # vLLM 0.34.0 refuses speculation above this many tokens

NEEDLE_DEPTHS = (0.04, 0.06, 0.24, 0.26, 0.49, 0.51, 0.74, 0.76, 0.94, 0.96)  # 5, 25, 50, 75, 95% +- 1%
VT_DEPTHS = ((0.10, 0.40, 0.85), (0.18, 0.55, 0.92))
JSON_DEPTHS = (0.15, 0.35, 0.65, 0.88)

VAULTS = (
    "Copper", "Cobalt", "Nickel", "Zinc", "Tin", "Lead", "Silver", "Gold", "Platinum", "Iron", "Titanium", "Chrome",
    "Amber", "Garnet", "Opal", "Jade", "Onyx", "Quartz", "Topaz", "Ruby", "Pearl", "Basalt", "Granite", "Marble",
)  # fmt: skip

INSTR = (
    "You are a code review assistant working in a large C++ and CUDA repository. The source files are pasted below. "
    "Read them closely: the comments of the sources hold exact facts, such as vault access codes and variable "
    "assignments, that later questions ask about. Answer only from what is written in this conversation."
)
TASK = (
    "Task: review the kernel launch conventions in the sources above. Work through the tool results as they arrive, "
    "and keep every answer short and exact."
)
# no example code: a small model copies it instead of searching (Qwen3-0.6B answered "12345-6789")
NEEDLE_QUESTION = "What is the access code of vault {name}? Reply with the code only: five digits, a dash, four digits."
VT_QUESTION = (
    "What number is VAR {name}? Follow the VAR assignments in the source comments. Reply with the number only "
    "(five digits)."
)
JSON_QUESTION = (
    "Find the access code of vault {name} in the source comments. Reply with one JSON object "
    '{{"vault": "<name>", "code": "<code>"}} and no code fence.'
)

NEEDLE_RE = re.compile(r"\d{5}-\d{4}")
VT_RE = re.compile(r"\d{5}")


class CorpusUnavailable(RuntimeError):
    """The corpus text cannot be read: neither the v0.12.0 tag nor $WARPC_LEGACY_DIR. A test skips on it, a planner
    turns the cells into `skip`."""


class CorpusMismatch(RuntimeError):
    """The corpus bytes are not the ones every baseline was recorded on."""


class Unsupported(ValueError):
    """The tokenizer has no ChatML format with <think>; the planner decides `cannot(unsupported)`."""


class TraceError(ValueError):
    """A builder assertion failed: the trace would not measure what the design says it measures."""


# ---- the corpus


def _git(root: Path, *args: str) -> bytes | None:
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def _tag_files() -> list[bytes] | None:
    here = Path(__file__).resolve().parent
    top = _git(here, "rev-parse", "--show-toplevel")
    if top is None:
        return None
    root = Path(top.decode().strip())
    listing = _git(root, "ls-tree", "--name-only", CORPUS_TAG, CORPUS_DIR + "/")
    if listing is None:
        return None
    names = sorted(n for n in listing.decode().split("\n") if n.endswith(".cu"))
    blobs = [_git(root, "show", f"{CORPUS_TAG}:{n}") for n in names]
    return None if not names or any(b is None for b in blobs) else blobs  # type: ignore[return-value]


def _legacy_files() -> list[bytes]:
    value = os.environ.get(LEGACY_ENV)
    if not value:
        raise CorpusUnavailable(
            f"{CORPUS_TAG} tag not found: `git fetch --tags` (or set {LEGACY_ENV} to a {CORPUS_TAG} checkout)"
        )
    files = sorted((Path(value) / CORPUS_DIR).glob("*.cu"), key=str)
    if not files:
        raise CorpusUnavailable(f"{LEGACY_ENV} has no {CORPUS_DIR}/*.cu: it is not a {CORPUS_TAG} checkout")
    return [f.read_bytes() for f in files]


@functools.cache
def corpus_text() -> str:
    """engine/src/*.cu of the v0.12.0 tag in sorted path order, joined with a blank line; checked against the bytes
    and hash every baseline was recorded on."""
    blobs = _tag_files()
    if blobs is None:
        blobs = _legacy_files()
    text = "\n\n".join(b.decode("utf-8", errors="replace") for b in blobs)
    size, sha = sum(map(len, blobs)), hashlib.sha256(text.encode()).hexdigest()
    if size != CORPUS_BYTES or sha != CORPUS_SHA256:
        raise CorpusMismatch(
            f"the corpus is {size} bytes, sha256 {sha[:12]}; expected {CORPUS_BYTES}, {CORPUS_SHA256[:12]}"
        )
    return text


# ---- the tokenizer


class Tokenizer(Protocol):
    sha256: str  # of tokenizer.json: the tokenizer's identity in every cell

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Ids and the (start, end) character span of each, special tokens in the text kept whole."""

    def decode(self, ids: Sequence[int]) -> str: ...

    def token_id(self, token: str) -> int | None: ...


class FileTokenizer:
    """A `tokenizer.json` through the `tokenizers` library. The 27B checkpoint's file truncates every encode at 4,096
    ids, so truncation is switched off here."""

    def __init__(self, path: Path | str):
        from tokenizers import Tokenizer as Backend

        file = Path(path)
        file = file / "tokenizer.json" if file.is_dir() else file
        self._backend = Backend.from_file(str(file))
        self._backend.no_truncation()
        self._backend.no_padding()
        self.sha256 = hashlib.sha256(file.read_bytes()).hexdigest()

    def encode(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoding = self._backend.encode(text, add_special_tokens=False)
        return list(encoding.ids), list(encoding.offsets)

    def decode(self, ids: Sequence[int]) -> str:
        return self._backend.decode(list(ids), skip_special_tokens=False)

    def token_id(self, token: str) -> int | None:
        return self._backend.token_to_id(token)


@dataclass(frozen=True)
class Chat:
    """The ChatML ids of a tokenizer: every role header is `<|im_start|>` + the role word and newline."""

    im_start: int
    im_end: int
    nl: tuple[int, ...]
    system: tuple[int, ...]
    user: tuple[int, ...]
    assistant: tuple[int, ...]
    think: tuple[int, ...]  # `<think>\n`: the tail of an assistant header that asks for thinking
    think_empty: tuple[int, ...]  # `<think>\n\n</think>\n\n`: the probes' answers start after it

    @classmethod
    def of(cls, tok: Tokenizer) -> Chat:
        ids: dict[str, int] = {}
        for name in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
            token = tok.token_id(name)
            if token is None:
                raise Unsupported(f"the tokenizer has no {name} token: the agent workload needs ChatML with <think>")
            ids[name] = token

        def enc(text: str) -> tuple[int, ...]:
            return tuple(tok.encode(text)[0])

        return cls(
            im_start=ids["<|im_start|>"],
            im_end=ids["<|im_end|>"],
            nl=enc("\n"),
            system=enc("system\n"),
            user=enc("user\n"),
            assistant=enc("assistant\n"),
            think=enc("<think>\n"),
            think_empty=enc("<think>\n\n</think>\n\n"),
        )

    def header(self, role: tuple[int, ...]) -> list[int]:
        return [self.im_start, *role]

    def close(self) -> list[int]:
        return [self.im_end, *self.nl]


@dataclass
class Corpus:
    """The corpus text and its token ids with the character span of each token. `line_start[i]`: token i begins a
    line (the token before it ends at a newline), the only places a fact is planted."""

    text: str
    ids: list[int]
    line_start: list[bool]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    @classmethod
    def encode(cls, tok: Tokenizer, text: str) -> Corpus:
        ids, spans = tok.encode(text)
        starts = [
            i == 0 or (spans[i][0] > 0 and text[spans[i][0] - 1] == "\n" and spans[i - 1][1] == spans[i][0])
            for i in range(len(ids))
        ]
        return cls(text, ids, starts)


@functools.lru_cache(maxsize=2)
def load_corpus(tok: Tokenizer) -> Corpus:
    """The real corpus through `tok`: at least MIN_CORPUS_TOKENS, which also catches a truncating tokenizer."""
    corpus = Corpus.encode(tok, corpus_text())
    if len(corpus.ids) < MIN_CORPUS_TOKENS:
        raise CorpusMismatch(
            f"the corpus is {len(corpus.ids)} tokens, fewer than {MIN_CORPUS_TOKENS}: a truncating tokenizer?"
        )
    return corpus


def slice_stride(length: int) -> int:
    """Tokens between the starts of the slices of a `length`-token prompt: the shipped stride up to 4,096, disjoint
    slices above (16,384: 9 slices, 32,768: 4, 65,536: 2; 98,304 and up: 1)."""
    return SLICE_STRIDE if length <= 4096 else length


def corpus_ids(tok: Tokenizer, length: int, slice: int = 0, *, corpus: Corpus | None = None) -> list[int]:
    """`length` ids of the corpus, slice `slice` of the disjoint slices of that length."""
    corpus = corpus or load_corpus(tok)
    offset = slice * slice_stride(length)
    if len(corpus.ids) < offset + length:
        raise ValueError(f"the corpus of {len(corpus.ids)} tokens is too short for {offset + length}")
    return corpus.ids[offset : offset + length]


def round_up_256(n: int) -> int:
    return -(-n // 256) * 256


def corpus_workload(
    tok: Tokenizer, length: int, slice: int = 0, *, n_new: int, corpus: Corpus | None = None
) -> tuple[list[int], dict[str, Any]]:
    """A prompt of `length` corpus tokens and the `workload` dict of its cells: `ctx_max` is the prompt plus the 512
    positions of headroom (the largest decode and a speculative round's overshoot), in 256s."""
    corpus = corpus or load_corpus(tok)
    ids = corpus_ids(tok, length, slice, corpus=corpus)
    return ids, {
        "id": "corpus-v1",
        "ids_sha": ids_sha(ids),
        "n_prompt": length,
        "n_new": n_new,
        "ignore_eos": True,
        "ctx_max": round_up_256(length + 512),
        "forged": False,
        "slice": slice,
        "corpus_sha256": corpus.sha256,
        "tokenizer_sha256": tok.sha256,
    }


def stamp_fields(tok: Tokenizer) -> dict[str, str]:
    """The workload part of a run's stamp: the corpus and the tokenizer every trace and prompt of the run came from."""
    return {"corpus_sha256": CORPUS_SHA256, "tokenizer_sha256": tok.sha256}


# ---- the trace


@dataclass(frozen=True)
class TurnSpec:
    assistant: list[int]  # A_j: the trace's assistant message
    tool: list[int]  # R_j
    decode: int  # n_j


@dataclass(frozen=True)
class Probe:
    kind: Literal["needle", "tail", "vt", "json"]
    label: str
    question: list[int]  # Q_i
    answer: dict[str, str]  # needle, tail: {"code"}; vt: {"value"}; json: {"vault", "code"}
    n_new: int
    where: dict[str, Any]  # depths (fractions of the haystack, as placed) or the tail distance, as placed


@dataclass(frozen=True)
class Trace:
    scale: int
    seed: int
    chat: Chat
    prefix: list[int]
    turns: list[TurnSpec]
    probes: list[Probe]
    haystack: int  # P: haystack tokens including the planted facts
    corpus_tokens: int
    corpus_sha256: str
    tokenizer_sha256: str
    trace_sha256: str

    @property
    def ctx_max(self) -> int:
        """The longest probe prompt plus the largest decode and a margin, in 256s (also vLLM's max_model_len)."""
        return round_up_256(self.scale + 64)

    def workload(self, *, workload_id: str = "agent-v1") -> dict[str, Any]:
        return {
            "id": workload_id,
            "ids_sha": self.trace_sha256,
            "n_prompt": self.scale,
            "n_new": PROBE_NEW,
            "ignore_eos": True,
            "ctx_max": self.ctx_max,
            "forged": False,
            "corpus_sha256": self.corpus_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
        }


def render(
    trace: Trace, k: int, mode: Literal["chat", "linear"] = "chat", own: Sequence[Sequence[int]] | None = None
) -> list[int]:
    """The prompt of turn k: the prefix, the k earlier (assistant, tool) pairs, the assistant header. `chat` renders
    the trace's assistant messages and ends in the think tail, which the next prompt does not carry, so consecutive
    prompts diverge right after the assistant header (what the Qwen template does to a history). `linear` renders the
    engine's own earlier outputs `own` and no tail: the next prompt extends the cached tokens."""
    chat = trace.chat
    ids = list(trace.prefix)
    for j in range(k):
        turn = trace.turns[j]
        message = turn.assistant if mode == "chat" else own[j]  # type: ignore[index]
        ids += [
            *chat.header(chat.assistant),
            *message,
            *chat.close(),
            *chat.header(chat.user),
            *turn.tool,
            *chat.close(),
        ]
    ids += chat.header(chat.assistant)
    if mode == "chat":
        ids += chat.think
    return ids


def history(trace: Trace) -> list[int]:
    """H: the prefix and all 12 (assistant, tool) pairs, ending after the last tool result's `<|im_end|>\\n`."""
    chat = trace.chat
    ids = list(trace.prefix)
    for turn in trace.turns:
        ids += [*chat.header(chat.assistant), *turn.assistant, *chat.close(), *chat.header(chat.user), *turn.tool]
        ids += chat.close()
    return ids


def probe_tail(trace: Trace, i: int) -> list[int]:
    chat = trace.chat
    question = trace.probes[i].question
    return [*chat.header(chat.user), *question, *chat.close(), *chat.header(chat.assistant), *chat.think_empty]


def probe_ids(trace: Trace, i: int, *, hist: list[int] | None = None) -> list[int]:
    return [*(history(trace) if hist is None else hist), *probe_tail(trace, i)]


def ids_sha(ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def _trace_sha(trace_fields: dict[str, Any]) -> str:
    canonical = json.dumps(trace_fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---- planting facts


@dataclass(frozen=True)
class _Line:
    text: str  # the comment line, without the newline
    mark: str  # the part of it a retrieval must carry: the code or the number


def _vault_line(name: str, code: str) -> _Line:
    return _Line(f"// vault {name} access code: {code}", code)


def _nearest_line_start(line_start: list[bool], want: int, lo: int, hi: int) -> int:
    """The line start nearest to `want` in [lo, hi), within SNAP_REACH (the lower one on a tie)."""
    best = None
    for i in range(max(lo, want - SNAP_REACH), min(hi, want + SNAP_REACH + 1)):
        if line_start[i] and (best is None or abs(i - want) < abs(best - want)):
            best = i
    if best is None:
        raise TraceError(f"no line start within {SNAP_REACH} tokens of corpus token {want}")
    return best


def _splice(
    corpus: Corpus, start: int, count: int, inserts: list[tuple[int, list[int]]]
) -> tuple[list[int], list[int]]:
    """Corpus tokens [start, start + count) with every insert (final position, ids) placed at the line start nearest
    to its final position; the inserts ascend. Returns the ids and the final position of each insert."""
    out: list[int] = []
    placed: list[int] = []
    at, end, before = start, start + count, 0
    for target, ids in inserts:
        base = _nearest_line_start(corpus.line_start, start + target - before, at, end)
        out += corpus.ids[at:base]
        placed.append(len(out))
        out += ids
        at, before = base, before + len(ids)
    out += corpus.ids[at:end]
    return out, placed


def _mark_offset(tok: Tokenizer, line: _Line) -> tuple[list[int], int]:
    """The ids of `line` + newline and the index of the first token of its mark."""
    text = line.text + "\n"
    ids, spans = tok.encode(text)
    first = text.index(line.mark)
    return ids, next(i for i, (s, _) in enumerate(spans) if s >= first)


def _names(rng: np.random.RandomState, count: int) -> list[str]:
    return [VAULTS[i] for i in rng.permutation(len(VAULTS))[:count]]


def _code(rng: np.random.RandomState) -> str:
    return f"{int(rng.randint(10000, 100000))}-{int(rng.randint(1000, 10000))}"


def _var(rng: np.random.RandomState) -> str:
    return "".join(chr(97 + int(c)) for c in rng.randint(0, 26, 6))


@dataclass
class _Spec:
    kind: Literal["needle", "tail", "vt", "json"]
    label: str
    question: str
    answer: dict[str, str]
    n_new: int
    depths: tuple[float, ...] = ()
    lines: tuple[_Line, ...] = ()
    distance: int = 0


def _specs() -> list[_Spec]:
    """The 19 probes in the order they run. The second RandomState(1) draws the names and values: names first, then
    the codes in probe order, then the variable chains."""
    rng = np.random.RandomState(1)
    names = _names(rng, len(NEEDLE_DEPTHS) + len(TAIL_DISTANCES) + len(JSON_DEPTHS))
    specs: list[_Spec] = []
    for depth in NEEDLE_DEPTHS:
        name, code = names.pop(0), _code(rng)
        line = _vault_line(name, code)
        specs.append(
            _Spec(
                "needle",
                f"needle@{depth:.2f}",
                NEEDLE_QUESTION.format(name=name),
                {"code": code},
                PROBE_NEW,
                (depth,),
                (line,),
            )
        )
    for distance in TAIL_DISTANCES:
        name, code = names.pop(0), _code(rng)
        specs.append(
            _Spec(
                "tail",
                f"tail@{distance}",
                NEEDLE_QUESTION.format(name=name),
                {"code": code},
                PROBE_NEW,
                lines=(_vault_line(name, code),),
                distance=distance,
            )
        )
    for depths in VT_DEPTHS:
        value = str(int(rng.randint(10000, 100000)))
        chain = [_var(rng), _var(rng), _var(rng)]
        lines = (
            _Line(f"// VAR {chain[0]} = {value}", value),
            _Line(f"// VAR {chain[1]} = VAR {chain[0]}", f"VAR {chain[0]}"),
            _Line(f"// VAR {chain[2]} = VAR {chain[1]}", f"VAR {chain[1]}"),
        )
        label = f"vt@{depths[0]:.2f}"
        specs.append(_Spec("vt", label, VT_QUESTION.format(name=chain[2]), {"value": value}, PROBE_NEW, depths, lines))
    for depth in JSON_DEPTHS:
        name, code = names.pop(0), _code(rng)
        specs.append(
            _Spec(
                "json",
                f"json@{depth:.2f}",
                JSON_QUESTION.format(name=name),
                {"vault": name, "code": code},
                JSON_NEW,
                (depth,),
                (_vault_line(name, code),),
            )
        )
    return specs


def build_trace(
    tok: Tokenizer, *, scale: int = SCALES["100k"], seed: int = 0, turns: int = TURNS, corpus: Corpus | None = None
) -> Trace:
    """The agent trace whose longest probe prompt has exactly `scale` tokens. The haystack length P is solved from
    that: everything else (headers, task, the turns' messages, the probe questions) has a size independent of P."""
    chat = Chat.of(tok)
    corpus = corpus or load_corpus(tok)
    specs = _specs()
    rng = np.random.RandomState(seed)
    sizes = []
    for _ in range(turns):
        assistant = int(rng.randint(*DECODE_TOKENS))
        tool = int(rng.randint(TOOL_TOKENS[0], TOOL_TOKENS[1] + 1))
        sizes.append((assistant, tool, int(rng.randint(*DECODE_TOKENS))))
    last = sizes[-1]
    sizes[-1] = (last[0], max(last[1], LAST_TOOL_MIN), last[2])

    instr, task = tok.encode(INSTR)[0], tok.encode(TASK)[0]
    questions = [tok.encode(s.question)[0] for s in specs]
    if too_long := [len(q) for q in questions if len(q) >= QUESTION_MAX]:
        raise TraceError(f"a question of {too_long[0]} tokens: it must stay under {QUESTION_MAX}")
    framing = (
        len(chat.header(chat.system))
        + len(instr)
        + len(chat.close())
        + len(chat.header(chat.user))
        + len(task)
        + len(chat.close())
    )
    pair = len(chat.header(chat.assistant)) + len(chat.close()) + len(chat.header(chat.user)) + len(chat.close())
    overhead = max(
        len(chat.header(chat.user))
        + len(q)
        + len(chat.close())
        + len(chat.header(chat.assistant))
        + len(chat.think_empty)
        for q in questions
    )
    haystack = scale - framing - sum(pair + a + t for a, t, _ in sizes) - overhead

    # facts planted in the haystack, ascending by depth
    planted = sorted((depth, i, j) for i, s in enumerate(specs) if s.kind != "tail" for j, depth in enumerate(s.depths))
    line_ids = {}
    for i, s in enumerate(specs):
        for j, line in enumerate(s.lines):
            line_ids[i, j] = _mark_offset(tok, line)
    fact_tokens = sum(len(line_ids[i, j][0]) for _, i, j in planted)
    if haystack < 2000 + fact_tokens:
        raise TraceError(f"a scale of {scale} leaves {haystack} haystack tokens")
    base = haystack - fact_tokens
    tail_tokens = sum(len(line_ids[i, 0][0]) for i, s in enumerate(specs) if s.kind == "tail")
    if len(corpus.ids) < base + sum(a + t for a, t, _ in sizes) - tail_tokens:
        raise ValueError(f"the corpus of {len(corpus.ids)} tokens is too short")
    inserts = [(round(depth * haystack), line_ids[i, j][0]) for depth, i, j in planted]
    hay, placed = _splice(corpus, 0, base, inserts)
    if len(hay) != haystack:
        raise TraceError("the haystack length differs from the solved one")
    where: dict[int, dict[str, Any]] = {
        i: {"depth": [None] * len(s.depths), "target": list(s.depths)} for i, s in enumerate(specs)
    }
    for (_, i, j), pos in zip(planted, placed, strict=True):
        where[i]["depth"][j] = round((pos + line_ids[i, j][1]) / haystack, 4)

    prefix = [*chat.header(chat.system), *instr, *hay, *chat.close(), *chat.header(chat.user), *task, *chat.close()]
    at = base
    turn_specs: list[TurnSpec] = []
    for k, (a_size, t_size, n) in enumerate(sizes):
        assistant = corpus.ids[at : at + a_size]
        at += a_size
        if k < turns - 1:
            tool = corpus.ids[at : at + t_size]
            at += t_size
        else:
            tool, at = _last_tool(corpus, at, t_size, specs, line_ids, where)
        turn_specs.append(TurnSpec(assistant, tool, n))

    probes = [
        Probe(s.kind, s.label, q, dict(s.answer), s.n_new, where[i])
        for i, (s, q) in enumerate(zip(specs, questions, strict=True))
    ]
    fields = {
        "scale": scale,
        "seed": seed,
        "chat": asdict(chat),
        "prefix": prefix,
        "turns": [asdict(t) for t in turn_specs],
        "probes": [asdict(p) for p in probes],
    }
    trace = Trace(
        scale=scale,
        seed=seed,
        chat=chat,
        prefix=prefix,
        turns=turn_specs,
        probes=probes,
        haystack=haystack,
        corpus_tokens=len(corpus.ids),
        corpus_sha256=corpus.sha256,
        tokenizer_sha256=tok.sha256,
        trace_sha256=_trace_sha(fields),
    )
    check_trace(trace)
    return trace


def _last_tool(
    corpus: Corpus,
    at: int,
    size: int,
    specs: list[_Spec],
    line_ids: dict[tuple[int, int], tuple[list[int], int]],
    where: dict[int, dict[str, Any]],
) -> tuple[list[int], int]:
    """R_11: `size` tokens of corpus from `at` with the three tail needles planted so that each code starts
    `distance` tokens before the question's `<|im_start|>` (two tokens after the result's end)."""
    tails = sorted((i for i, s in enumerate(specs) if s.kind == "tail"), key=lambda i: -specs[i].distance)
    fact_tokens = sum(len(line_ids[i, 0][0]) for i in tails)
    inserts = [(size + 2 - specs[i].distance - line_ids[i, 0][1], line_ids[i, 0][0]) for i in tails]
    ids, placed = _splice(corpus, at, size - fact_tokens, inserts)
    for i, pos in zip(tails, placed, strict=True):
        where[i] = {"distance": size + 2 - (pos + line_ids[i, 0][1]), "target": specs[i].distance}
    return ids, at + size - fact_tokens


# ---- the builder's assertions


def _windows() -> tuple[int, int]:
    """The two exact-row windows of the KV formats, from the format table that defines them: (256, 496) today."""
    from warpc.kvcache import FORMATS

    sizes = sorted({f.window for f in FORMATS.values() if f.window})
    if len(sizes) != 2:
        raise TraceError(f"the KV formats have windows {sizes}: the tail needles are placed for two")
    return sizes[0], sizes[1]


def fork_rows(
    trace: Trace,
    device_snapshots: int = MIN_DEVICE_SNAPSHOTS,
    *,
    plan: Callable[[Sequence[int]], list[int]] | None = None,
) -> list[dict[str, int]]:
    """What each probe fork costs a prefix cache that keeps `device_snapshots` snapshots of the lineage
    (`prefix.PrefixCache._points`) and cuts its prefill with `plan` (default `windows.plan_windows` at the trace's
    turn token): the resume position and the rows it prefills. Raises TraceError unless every fork resumes at the
    history's end + 2, the user header, and no later than the prompt it shares with the previous probe. The 8-row plan
    (`prefix.class_d_plan`) fails this: its snapshot goes to the last 8-row boundary below the assistant header, so a
    fork re-prefills everything above the system checkpoint."""
    from warpc.prefix import PrefixCache, common_prefix
    from warpc.windows import plan_windows

    if device_snapshots < MIN_DEVICE_SNAPSHOTS:
        raise TraceError(f"device_snapshots={device_snapshots}: a fork needs the system checkpoint and the fork point")
    hist = history(trace)
    h = len(hist)
    cache = PrefixCache(
        None,  # type: ignore[arg-type]  # `_points` and the plan touch no decoder
        host_bytes=0,
        device_snapshots=device_snapshots,
        plan=plan or (lambda ids: plan_windows(ids, turn_token=trace.chat.im_start)),
        turn_token=trace.chat.im_start,
    )
    rows = []
    prev: list[int] | None = None
    for i in range(len(trace.probes)):
        ids = probe_ids(trace, i, hist=hist)
        arr = np.asarray(ids, dtype=np.int32)
        points, _ = cache._points(arr, cache.plan(ids))
        resume = max((p for p in points if p <= h + 2), default=0)
        common = len(ids) if prev is None else common_prefix(np.asarray(prev, dtype=np.int32), arr)
        if resume != h + 2:
            raise TraceError(f"probe {i} would resume at {resume}, not at the fork point {h + 2}")
        if prev is not None and common - 1 < resume:
            raise TraceError(f"probe {i} shares only {common} tokens with the previous probe")
        rows.append({"resume": resume, "prefill_rows": len(ids) - resume})
        prev = ids
    return rows


def tail_regimes(trace: Trace) -> list[dict[str, int]]:
    """For each tail needle the distance from its code to the query row of the first and the last answer token.
    Exact rows are the newest `window` positions: 128 must be exact in every format, 380 exact in the larger window
    and coded in the smaller, 700 coded in both."""
    small, large = _windows()
    h = len(history(trace))
    out = []
    for i, probe in enumerate(trace.probes):
        if probe.kind != "tail":
            continue
        code_at = h - probe.where["distance"]
        first = h + len(probe_tail(trace, i)) - 1 - code_at
        out.append({"index": i, "target": probe.where["target"], "first": first, "last": first + probe.n_new - 1})
    for row, regime in zip(out, ("exact", "exact_large_only", "coded"), strict=True):
        ok = {
            "exact": row["last"] < small,
            "exact_large_only": row["first"] >= small and row["last"] < large,
            "coded": row["first"] >= large,
        }[regime]
        if not ok:
            raise TraceError(
                f"tail needle @{row['target']}: answer rows are {row['first']}..{row['last']} tokens after its code, "
                f"the exact windows are {small} and {large}"
            )
    return out


def check_trace(trace: Trace) -> None:
    """The assertions the fork-cost and tail-needle claims depend on."""
    if len(trace.turns[-1].tool) < LAST_TOOL_MIN:
        raise TraceError(f"R_11 has {len(trace.turns[-1].tool)} tokens, fewer than {LAST_TOOL_MIN}")
    lengths = [len(probe_ids(trace, i)) for i in range(len(trace.probes))]
    if max(lengths) != trace.scale:
        raise TraceError(f"the longest probe prompt has {max(lengths)} tokens, not {trace.scale}")
    if (q := max(len(p.question) for p in trace.probes)) >= QUESTION_MAX:
        raise TraceError(f"a question of {q} tokens, not under {QUESTION_MAX}")
    for probe in trace.probes:
        if probe.kind == "tail":
            if abs(probe.where["distance"] - probe.where["target"]) > TAIL_SNAP_TOLERANCE:
                raise TraceError(f"{probe.label} was placed {probe.where['distance']} tokens before the question")
        elif any(abs(d - t) > 0.011 for d, t in zip(probe.where["depth"], probe.where["target"], strict=True)):
            raise TraceError(f"{probe.label} was placed at depths {probe.where['depth']}")
    fork_rows(trace)
    tail_regimes(trace)


# ---- scoring


def strip_fence(text: str) -> str:
    """`text` without one leading and one trailing code fence (``` or ```json)."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rstrip()
        if body.endswith("```"):
            body = body[:-3]
    return body.strip()


def score_probe(probe: Probe, text: str) -> dict[str, Any]:
    """Exact match of one reply: the first code (or 5-digit number) in the text equals the planted one; for `json` the
    reply, fences stripped, parses with the keys `vault` and `code` (`format_ok`) and both values equal (`value_ok`)."""
    if probe.kind in ("needle", "tail"):
        found = NEEDLE_RE.search(text)
        return {"ok": found is not None and found.group() == probe.answer["code"], "found": found and found.group()}
    if probe.kind == "vt":
        found = VT_RE.search(text)
        return {"ok": found is not None and found.group() == probe.answer["value"], "found": found and found.group()}
    try:
        value = json.loads(strip_fence(text))
    except ValueError:
        return {"ok": False, "format_ok": False, "value_ok": False, "found": None}
    format_ok = isinstance(value, dict) and {"vault", "code"} <= value.keys()
    value_ok = format_ok and value["vault"] == probe.answer["vault"] and value["code"] == probe.answer["code"]
    return {"ok": bool(value_ok), "format_ok": bool(format_ok), "value_ok": bool(value_ok), "found": None}


def reply_ids(trace: Trace, ids: Sequence[int]) -> list[int]:
    """The reply: the generated ids up to the first `<|im_end|>`. The decode ignores the end of sequence (so every
    engine times the same number of tokens) and a model that ends its turn goes on writing after it."""
    ids = list(ids)
    return ids[: ids.index(trace.chat.im_end)] if trace.chat.im_end in ids else ids


def score(trace: Trace, decode: Callable[[Sequence[int]], str], outputs: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Every probe's score on the reply (`reply_ids`), the counts per kind, `probes_ok` (of 19) and the json counters.
    Each row keeps the start of the reply to read a failure by."""
    if len(outputs) != len(trace.probes):
        raise ValueError(f"{len(outputs)} outputs for {len(trace.probes)} probes")
    texts = [decode(reply_ids(trace, ids)) for ids in outputs]
    rows = [
        score_probe(p, text) | {"label": p.label, "kind": p.kind, "reply": text[:100]}
        for p, text in zip(trace.probes, texts, strict=True)
    ]
    kinds: dict[str, list[int]] = {}
    for row in rows:
        kinds.setdefault(row["kind"], [0, 0])
        kinds[row["kind"]][0] += row["ok"]
        kinds[row["kind"]][1] += 1
    return {
        "probes": rows,
        "probes_ok": sum(r["ok"] for r in rows),
        "by_kind": {k: {"ok": ok, "n": n} for k, (ok, n) in kinds.items()},
        "json_format_ok": sum(r.get("format_ok", False) for r in rows),
        "json_value_ok": sum(r.get("value_ok", False) for r in rows),
    }


# ---- which cells cannot exist


def agent_cannot(
    *,
    engine: str,
    kv: str,
    ctx_max: int,
    chatml: bool = True,
    positions: int | None = None,
    hybrid: bool = False,
    backend: str = "cuda",
    mtp: int = 0,
    swiftmk_lib: bool = False,
) -> tuple[str, str] | None:
    """(code, reason) when an agent cell of this engine cannot run at `ctx_max`, from what is known before loading
    (the table of the design's section 4.5); None when it may run, an adapter's `load` still decides capacity. `hybrid`:
    the 27B (gated attention, hd 256); `positions`: the model's max_position_embeddings."""
    if not chatml:
        return "unsupported", "the agent workload needs a ChatML chat format with <think> (Qwen3, Qwen3.5)"
    if positions is not None and ctx_max > positions:
        return "model_limit", f"needs {ctx_max} positions, the model has {positions}"
    if engine == "llama" and hybrid:
        return "model_limit", "no GGUF of the 27B in models.toml"
    if engine == "vllm" and mtp and ctx_max > VLLM_MTP_MAX_CTX:
        return "unsupported", f"vLLM does not run speculation above {VLLM_MTP_MAX_CTX} tokens"
    if engine == "warpc" and hybrid and kv in ("k8v8", "k8v8w", "k8v4w"):
        return (
            "kernel_missing",
            f"KV format {kv} has no kernel for the 27B's attention (bf16, int4_rot; fp8 on swiftmk)",
        )
    if engine == "warpc" and kv == "fp8":
        if backend != "swiftmk":
            return "kernel_missing", "KV format fp8 has no kernel on the cuda backend: use backend=swiftmk"
        if not swiftmk_lib:
            return "env", "KV format fp8 needs WARPC_SWIFTMK_LIB"
    if engine == "hand" and kv not in ("bf16", "fp8", "int4_rot"):
        return "unsupported", f"the hand engine has KV bf16, fp8 and int4_rot, not {kv}"
    return None


# ---- running a trace against an engine


def turn_row(k: int, ids: Sequence[int], n: int, turn: Turn, previous: Sequence[int] | None) -> dict[str, Any]:
    """One turn of the loop as the cell stores it: the engine's `Turn` and what the prompt added to the last."""
    shared = 0
    if previous is not None:
        shared = next(
            (i for i, (a, b) in enumerate(zip(previous, ids, strict=False)) if a != b), min(len(previous), len(ids))
        )
    return {
        **{name: value for name, value in turn.extra.items() if isinstance(value, int | float | str)},
        "turn": k,
        "prompt": len(ids),
        "new": len(ids) - shared,
        "decode_n": n,
        "first_s": turn.first_s,
        "decode_s": turn.decode_s,
        "n_first": turn.n_first,
        "decode_tok_s": turn.decode_tok_s if turn.decode_s else None,
        "cached": turn.cached,
        "ids_sha": ids_sha(turn.ids)[:12],
    }


def loop_metrics(rows: Sequence[dict[str, Any]], forks: Sequence[dict[str, Any]]) -> dict[str, schema.Metric]:
    """The summary of one loop: the turns 1.. (turn 0 is the cold prefill of the prefix) and the forks."""
    later = rows[1:]
    ttft = sorted(r["first_s"] for r in later)
    p95 = ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))]
    metrics = {
        "ttft_median_s": schema.Metric(statistics.median(ttft), "s", ttft),
        "ttft_p95_s": schema.Metric(p95, "s", [p95]),
        "turn0_first_s": schema.Metric(rows[0]["first_s"], "s", [rows[0]["first_s"]]),
        "decode_tok_s_median": schema.Metric.of([r["decode_tok_s"] for r in later if r["decode_tok_s"]], "tok/s"),
        "cached_fraction": schema.Metric(
            sum(r["cached"] for r in later) / max(1, sum(r["prompt"] for r in later)), "", []
        ),
    }
    if forks:
        times = [f["first_s"] for f in forks]
        metrics["fork_ttft_median_s"] = schema.Metric(statistics.median(times), "s", times)
        rows_ = [f["prompt"] - f["cached"] for f in forks]
        metrics["fork_rows_median"] = schema.Metric(statistics.median(rows_), "rows", rows_)
    return metrics


def resumed_check(resumed: Sequence[int], cold: Sequence[int]) -> dict[str, Any]:
    """The invariant of a prefix cache: a resumed turn equals a cold prefill of the same prompt."""
    diff = next((i for i, (a, b) in enumerate(zip(resumed, cold, strict=False)) if a != b), None)
    equal = list(resumed) == list(cold)
    return {"equal": equal, "first_diff": diff if diff is not None or equal else min(len(resumed), len(cold))}


def run_trace(
    engine: Engine,
    trace: Trace,
    tok: Tokenizer,
    *,
    key: str,
    cold: bool = False,
    mode: Literal["chat", "linear"] = "chat",
) -> list[schema.Cell]:
    """Unit A of the agent cell: turn 0 cold, turns 1.. resumed, then the 19 probe forks (greedy, EOS ignored), scored.
    `linear` mode is perf-only: the loop on the engine's own outputs, no forks, no score. `cold=True` also runs unit
    B's check in this process (the cold re-prefill of probe 0 against the resumed one); at 100k the two are separate
    units (`run_cold`). Returns one cell: the loop's metrics, `probes_ok` and the json counters, and in `extra` the
    per-turn rows, the per-fork rows and the probe ids."""
    own: list[list[int]] = []
    rows: list[dict[str, Any]] = []
    previous: list[int] | None = None
    for k, spec in enumerate(trace.turns):
        ids = render(trace, k, mode, own)
        turn = engine.turn(ids, spec.decode, reuse=k > 0)
        if k == 0 and turn.cached:
            raise RuntimeError(f"turn 0 with reuse=False found {turn.cached} cached tokens")
        own.append(list(turn.ids))
        rows.append(turn_row(k, ids, spec.decode, turn, previous))
        previous = ids
    forks: list[dict[str, Any]] = []
    outputs: list[list[int]] = []
    if mode == "chat":
        hist = history(trace)
        for i, probe in enumerate(trace.probes):
            ids = probe_ids(trace, i, hist=hist)
            turn = engine.turn(ids, probe.n_new, reuse=True)
            outputs.append(list(turn.ids))
            forks.append(turn_row(i, ids, probe.n_new, turn, None) | {"label": probe.label})
    metrics = loop_metrics(rows, forks)
    extra: dict[str, Any] = {"turns": rows, "forks": forks, "mode": mode}
    if outputs:
        scored = score(trace, tok.decode, outputs)
        metrics["probes_ok"] = schema.Metric(scored["probes_ok"], "count", [])
        metrics["json_format_ok"] = schema.Metric(scored["json_format_ok"], "count", [])
        metrics["json_value_ok"] = schema.Metric(scored["json_value_ok"], "count", [])
        extra |= {"probes": scored["probes"], "by_kind": scored["by_kind"], "probe_ids": outputs}
    if cold and outputs:
        again = run_cold(engine, trace, key=key)[0]
        check = resumed_check(outputs[0], again.extra["probe_ids"][0])
        metrics["resumed_equal"] = schema.Metric(int(check["equal"]), "bool", [])
        metrics["ttft_cold_s"] = again.metrics["ttft_cold_s"]
        extra["resumed"] = check
    return [schema.Cell(key, "ok", workload=trace.workload(), metrics=metrics, extra=extra)]


def run_cold(engine: Engine, trace: Trace, *, key: str) -> list[schema.Cell]:
    """Unit B: probe 0's whole `scale`-token prompt prefilled from nothing (`reuse=False`), decoded greedily. Its ids
    are what `resumed_check` compares with unit A's probe 0, and its first token's time is the cold TTFT at `scale`."""
    probe = trace.probes[0]
    turn = engine.turn(probe_ids(trace, 0), probe.n_new, reuse=False)
    if turn.cached:
        raise RuntimeError(f"a cold prefill found {turn.cached} cached tokens")
    metrics = {"ttft_cold_s": schema.Metric(turn.first_s, "s", [turn.first_s])}
    return [schema.Cell(key, "ok", workload=trace.workload(), metrics=metrics, extra={"probe_ids": [list(turn.ids)]})]
