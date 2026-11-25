"""The plan: what lowering a graph onto a backend produces, torch-free.

A Plan is a list of Tasks (one backend kernel launch each, in a valid order), the Buffers they read and write, the
weight Packs the kernels need (layouts decided by the kernel choice, not by the checkpoint) and the constant fp32 tables
they read (rope inverse frequencies), which a runtime places on the device once. Row counts stay symbolic: a buffer
holds `rows_max` rows of `dim` elements and a task runs on the first T rows of its `[T][dim]` row-major activations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Buffer:
    name: str
    dim: int  # elements per row
    dtype: str  # "bf16" | "f32" | "i32"
    values: list[str] = field(default_factory=list)  # graph values living here, in definition order


@dataclass
class Pack:
    """A device weight a kernel reads, built from the graph weights `parts` in the layout `kind` names. Kinds are the
    backend's vocabulary (its executor builds them and refuses the others); `name` is for humans and unique per plan.
    """

    name: str
    kind: str
    parts: list[str]


@dataclass
class Task:
    kernel: str
    ins: dict[str, str]  # role -> buffer name ("@<input>" for a graph input the runtime binds)
    outs: dict[str, str]  # role -> buffer name
    weights: dict[str, str] = field(default_factory=dict)  # role -> pack name
    states: dict[str, str] = field(default_factory=dict)  # role -> graph state name
    constants: dict[str, str] = field(default_factory=dict)  # role -> plan constant name
    attrs: dict = field(default_factory=dict)
    covers: list[int] = field(default_factory=list)  # graph nodes this task implements
    numerics: str = ""  # the kernel's numerics, prose
    tolerance: tuple[str, ...] = ()  # reference rounding points ("op.point") the kernel does not reproduce
    gather: tuple[str, ...] = ()  # weight roles the kernel reads only by row gather (its Cap's `gather`)


@dataclass
class Plan:
    """Positions of the rows of a step are pos0 .. pos0 + T - 1, pos0 a device scalar of the runtime (the graph input
    "pos")."""

    graph_name: str
    rows_max: int
    backend: str
    buffers: dict[str, Buffer] = field(default_factory=dict)
    buffer_of: dict[str, str] = field(default_factory=dict)  # graph value -> the buffer holding it
    packs: dict[tuple[str, tuple[str, ...]], Pack] = field(default_factory=dict)  # (kind, parts) -> pack
    constants: dict[str, tuple[float, ...]] = field(default_factory=dict)  # name -> fp32 values
    tasks: list[Task] = field(default_factory=list)

    def summary(self) -> str:
        c = Counter(t.kernel for t in self.tasks)
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        return f"{len(self.tasks)} tasks, {len(self.buffers)} buffers, {len(self.packs)} packs; {kinds}"
