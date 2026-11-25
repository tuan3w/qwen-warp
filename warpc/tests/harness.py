"""Runner for the plain-script tests: runs every module-level test_* function in definition order."""

import sys
import traceback
from pathlib import Path


class Skip(Exception):
    """Raised by a test whose prerequisite (GPU, env var) is missing."""


def legacy():
    """tools/legacy.py (the v0.12.0 checkout of $WARPC_LEGACY_DIR); Skip when the variable is unset."""
    tools = str(Path(__file__).resolve().parents[1] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import legacy as module

    try:
        module.root()
    except module.LegacyUnset as e:
        raise Skip(f"{module.ENV} unset") from e
    return module


def run(namespace: dict) -> None:
    failed = 0
    for name, fn in namespace.items():
        if not (name.startswith("test_") and callable(fn)):
            continue
        try:
            fn()
        except Skip as e:
            print(f"skip {name}: {e}")
            continue
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
            continue
        print(f"ok {name}")
    sys.exit(1 if failed else 0)
