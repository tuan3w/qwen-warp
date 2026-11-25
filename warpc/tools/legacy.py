"""The legacy tree: a checkout of release tag v0.12.0, named by $WARPC_LEGACY_DIR.

The hand engine (`engine/`), its server runner (`server/`) and the sources of its kernels are not in this repository
any more; the references of warpc's gates (tools/swift_worker.py, gate27.py, compare27.py, gate_window.py, agentloop.py,
the tests that import the hand engine or read its text corpus) all go through this module. Nothing falls back to a copy
in the tree. Set up with `git worktree add ../qwen-warp-legacy v0.12.0` and build engine/build/libswiftmk.so there
(that tag's README; warpc/README.md has the short version).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV = "WARPC_LEGACY_DIR"


class LegacyError(RuntimeError):
    """$WARPC_LEGACY_DIR is not usable."""


class LegacyUnset(LegacyError):
    """$WARPC_LEGACY_DIR is not set."""


def root() -> Path:
    value = os.environ.get(ENV)
    if not value:
        raise LegacyUnset(
            f"{ENV} unset: point it at a checkout of release tag v0.12.0 (the legacy 27B engine and server), "
            "`git worktree add ../qwen-warp-legacy v0.12.0`, with engine/build/libswiftmk.so built (warpc/README.md)"
        )
    path = Path(value).resolve()
    if not (path / "engine" / "engine.py").is_file() or not (path / "server" / "smk_runner.py").is_file():
        raise LegacyError(f"{ENV}={value} is not a checkout of v0.12.0 (no engine/engine.py or server/smk_runner.py)")
    return path


def engine_dir() -> Path:
    return root() / "engine"


def server_dir() -> Path:
    return root() / "server"


def library() -> Path:
    """The hand engine's library: engine/<$SMK_BUILD or build>/libswiftmk.so of the checkout."""
    return engine_dir() / os.environ.get("SMK_BUILD", "build") / "libswiftmk.so"


def corpus() -> list[Path]:
    """The prompt text of the 27B gates: engine/src/*.cu of the checkout, in sorted path order."""
    files = sorted((engine_dir() / "src").glob("*.cu"), key=str)
    if not files:
        raise LegacyError(f"{ENV}: no engine/src/*.cu under {root()}")
    return files
