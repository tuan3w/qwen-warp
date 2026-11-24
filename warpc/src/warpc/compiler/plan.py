"""The plan: what lowering a graph onto a backend produces, torch-free.

A Plan is a list of Tasks (one backend kernel launch each, in a valid order), the Buffers they read and write and the
weight Packs the kernels need (layouts decided by the kernel choice, not by the checkpoint). Row counts stay symbolic: a
buffer holds `rows_max` rows of `dim` elements and a task runs on the first T rows of its `[T][dim]` row-major
activations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

PACK_KINDS = ("dense", "dense_cat", "q4_0", "q4_0_cat")


@dataclass
class Buffer:
    name: str
    dim: int  # elements per row
    dtype: str  # "bf16" | "f32" | "i32"
    values: list[str] = field(default_factory=list)  # graph values living here, in definition order


@dataclass
class Pack:
    """A device weight a kernel reads, built from the graph weights `parts`.

    dense      one weight as the source stores it
    dense_cat  the rows of the dense `parts`, concatenated in order
    q4_0       one GGUF Q4_0 matrix as stored: raw blocks [N, K / 32 * 18] bytes, read in place by the fused
               dequant GEMV
    q4_0_cat   the rows of the Q4_0 `parts`, concatenated in order
    """

    name: str
    kind: str
    parts: list[str]

    def __post_init__(self):
        if self.kind not in PACK_KINDS:
            raise ValueError(f"pack {self.name}: unknown kind {self.kind!r}")


@dataclass
class Task:
    kernel: str
    ins: dict[str, str]  # role -> buffer name ("@<input>" for a graph input the runtime binds)
    outs: dict[str, str]  # role -> buffer name
    weights: dict[str, str] = field(default_factory=dict)  # role -> pack name
    states: dict[str, str] = field(default_factory=dict)  # role -> graph state name
    attrs: dict = field(default_factory=dict)
    covers: list[int] = field(default_factory=list)  # graph nodes this task implements
    numerics: str = ""  # the kernel's numerics, prose
    tolerance: tuple[str, ...] = ()  # reference rounding points ("op.point") the kernel does not reproduce


@dataclass
class Plan:
    """Positions of the rows of a step are pos0 .. pos0 + T - 1, pos0 a device scalar of the runtime (the graph input
    "pos")."""

    graph_name: str
    rows_max: int
    backend: str
    buffers: dict[str, Buffer] = field(default_factory=dict)
    buffer_of: dict[str, str] = field(default_factory=dict)  # graph value -> the buffer holding it
    packs: dict[str, Pack] = field(default_factory=dict)
    tasks: list[Task] = field(default_factory=list)

    def summary(self) -> str:
        c = Counter(t.kernel for t in self.tasks)
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        return f"{len(self.tasks)} tasks, {len(self.buffers)} buffers, {len(self.packs)} packs; {kinds}"
