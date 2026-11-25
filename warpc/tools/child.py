"""Run a tool as a child process with its output passed through, keeping the reason it failed."""

import re
import subprocess
import sys

# the last line of a Python traceback, also when vLLM's engine-core process prefixes it with "(EngineCore pid=N) "
_EXCEPTION = re.compile(r"^(?:\([^)]*\)\s+)?((?:[A-Za-z_]\w*\.)*[A-Z]\w*(?:Error|Exception|Exit|Interrupt)\b.*)$")
# vLLM's front end fails with this after the engine-core process printed the exception that caused it
_SEE_ABOVE = "See root cause above"


def run(command: list[str]) -> str | None:
    """None if `command` succeeded, else its last exception line on stderr that is not a pointer to an earlier one (else
    its last stderr line, else its exit status)."""
    exceptions, last = [], None
    with subprocess.Popen(command, stderr=subprocess.PIPE, text=True, errors="replace") as process:
        for line in process.stderr:
            sys.stderr.write(line)
            if stripped := line.strip():
                last = stripped
                if m := _EXCEPTION.match(stripped):
                    exceptions.append(m.group(1))
    if process.returncode == 0:
        return None
    cause = next((e for e in reversed(exceptions) if _SEE_ABOVE not in e), None)
    return cause or last or f"exit status {process.returncode}"
