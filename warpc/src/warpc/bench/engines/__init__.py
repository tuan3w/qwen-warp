"""The only seam of the suite: an `Engine` is anything that can answer `turn(ids, n_new, reuse)` with greedy ids and
two host-visible times. Teacher-forced logits, near-tie judges and hand-trace gates are warpc-against-warpc (or hand)
and are plain functions of their own modules, not engines."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from .. import protocol

NAMES = ("warpc", "llama", "vllm", "hand")
# engine-native knobs that move the numbers: part of the cell key and the stamp, an unknown name is an error
OPTS: dict[str, tuple[str, ...]] = {
    "llama": ("ubatch", "batch", "flash_attn"),
    "vllm": ("max_num_batched_tokens", "gpu_memory_utilization", "language_model_only", "enable_chunked_prefill"),
    "warpc": ("window_rows", "device_snapshots", "store_gb"),
    "hand": ("window_rows", "device_snapshots", "store_gb"),
}
CannotCode = Literal["capacity", "memory", "model_limit", "kernel_missing", "unsupported", "env"]


class Cannot(Exception):
    """The engine or configuration cannot do this cell: a result (`status=cannot`), not an error. The message is the
    one-line reason the table shows."""

    def __init__(self, code: CannotCode, message: str):
        super().__init__(message)
        self.code = code


class Unchecked(Exception):
    """`Engine.check` could not decide on the CPU (it needs a CUDA context or a library that is not here): the cell
    stays planned and `load` raises the `Cannot` if there is one."""


@dataclass(frozen=True)
class EngineSpec:
    name: Literal["warpc", "llama", "vllm", "hand"]
    model: Path  # HF directory or GGUF file (hand: the 27B checkpoint)
    kv: str = "bf16"  # warpc FORMATS names; vllm: bf16 -> "auto", fp8 -> "fp8"; llama: bf16 -> f16
    head: str = "bf16"  # bf16 | int4_refine (27B swiftmk / hand heads)
    backend: str = "cuda"  # warpc: cuda | swiftmk | composite
    mtp: int = 0
    host_tables: bool | None = None  # None: decided by ctx_max
    window: bool | None = None  # None: on for ctx_max > 16k when the backend has it
    opts: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.name not in NAMES:
            raise ValueError(f"unknown engine {self.name!r} (known: {', '.join(NAMES)})")
        unknown = {k for k, _ in self.opts} - set(OPTS[self.name])
        if unknown:
            raise ValueError(f"unknown {self.name} opts {sorted(unknown)} (known: {', '.join(OPTS[self.name])})")
        if list(self.opts) != sorted(self.opts):
            raise ValueError("opts must be sorted: the spec text is a cell key")

    @classmethod
    def parse(cls, text: str, model: Path) -> EngineSpec:
        """`warpc:cuda,kv=int4_rot,head=bf16,mtp=3`; `llama,opts=batch:2048+ubatch:512` (`|` separates key segments)."""
        head, *fields = text.split(",")
        name, _, backend = head.partition(":")
        values: dict[str, Any] = {"name": name, "model": Path(model)}
        if backend:
            values["backend"] = backend
        for field_text in fields:
            key, eq, value = field_text.partition("=")
            if not eq:
                raise ValueError(f"engine spec field {field_text!r} is not key=value")
            if key in ("kv", "head", "backend"):
                values[key] = value
            elif key == "mtp":
                values[key] = int(value)
            elif key in ("host_tables", "window"):
                values[key] = {"1": True, "true": True, "0": False, "false": False}[value.lower()]
            elif key == "opts":
                pairs = [p.partition(":") for p in value.split("+") if p]
                values[key] = tuple(sorted((k, v) for k, _, v in pairs))
            else:
                raise ValueError(f"unknown engine spec field {key!r}")
        return cls(**values)

    def with_(self, **changes: Any) -> EngineSpec:
        return replace(self, **changes)

    @property
    def text(self) -> str:
        """The spec as a cell-key segment; every set field is written, so two specs never share a key."""
        head = f"{self.name}:{self.backend}" if self.name in ("warpc", "hand") else self.name
        parts = [head, f"kv={self.kv}", f"head={self.head}"]
        if self.mtp:
            parts.append(f"mtp={self.mtp}")
        for name in ("host_tables", "window"):
            if (value := getattr(self, name)) is not None:
                parts.append(f"{name}={int(value)}")
        if self.opts:
            parts.append("opts=" + "+".join(f"{k}:{v}" for k, v in self.opts))
        return ",".join(parts)

    def opt(self, name: str, default: str | None = None) -> str | None:
        return dict(self.opts).get(name, default)

    def to_json(self) -> dict[str, Any]:
        return {**{k: getattr(self, k) for k in ("name", "kv", "head", "backend", "mtp", "host_tables", "window")},
                "model": str(self.model), "opts": [list(p) for p in self.opts]}  # fmt: skip

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> EngineSpec:
        return cls(**{**d, "model": Path(d["model"]), "opts": tuple(tuple(p) for p in d["opts"])})


@dataclass
class Turn:
    ids: list[int]  # the n_new generated ids (ignore-EOS)
    first_s: float  # request -> first token host-visible
    decode_s: float  # first delivery -> last delivery host-visible
    cached: int  # prompt tokens the engine did not prefill
    extra: dict[str, Any] = field(default_factory=dict)  # engine counters: spill_s, restore_s, tokens_per_cycle, ...
    n_first: int = 1  # tokens the first delivery made visible (a group of cycles under MTP)

    @property
    def decode_tok_s(self) -> float:
        return protocol.decode_tok_s(len(self.ids), self.n_first, 0.0, self.decode_s)


class Engine(Protocol):
    def check(self, ctx_max: int) -> float | None:
        """CPU preflight: the estimated device GiB (None: no estimate); raises `Cannot` when the cell cannot run,
        `Unchecked` when the CPU cannot tell."""

    def load(self, ctx_max: int) -> dict[str, Any]:
        """Load and compile for `ctx_max` positions; `{load_s, ...}`. Raises `Cannot` on OOM or capacity."""

    def turn(self, ids: Sequence[int], n_new: int, *, reuse: bool) -> Turn:
        """`reuse=True`: the longest cached prefix of `ids` may be served, generated tokens never leak into a later
        prompt unless the prompt contains them. `reuse=False` drops every cache and fails loudly if it cannot."""

    def close(self) -> None: ...


def make(spec: EngineSpec) -> Engine:
    """The adapter of `spec.name`; imports it only now (vllm, llama and hand pull heavy dependencies)."""
    if spec.name == "warpc":
        from .warpc import WarpcEngine

        return WarpcEngine(spec)
    raise Cannot("unsupported", f"the {spec.name} engine is not available in this version of the suite")


def reason_text(exc: BaseException) -> str:
    """One line of an exception's message."""
    text = str(exc).strip().splitlines()
    return re.sub(r"\s+", " ", text[0]) if text else type(exc).__name__
