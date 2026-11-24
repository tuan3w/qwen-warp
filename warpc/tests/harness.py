"""Runner for the plain-script tests: runs every module-level test_* function in definition order."""

import sys
import traceback


class Skip(Exception):
    """Raised by a test whose prerequisite (GPU, env var) is missing."""


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
