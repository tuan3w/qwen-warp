"""llama.cpp's `llama-server` (LLAMA_SERVER, else `llama-server` on PATH) serving one GGUF file on the GPU, driven with
token ids: the oracle for greedy tokens and the llama.cpp throughput baseline.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path


def binary() -> str | None:
    return os.environ.get("LLAMA_SERVER") or shutil.which("llama-server")


def version() -> str | None:
    """The build of `binary()` as it reports it, e.g. "b8889 (bcb5eeb64)"."""
    exe = binary()
    if exe is None:
        return None
    run = subprocess.run([exe, "--version"], capture_output=True, text=True, check=False)
    m = re.search(r"version: (\d+) \((\w+)\)", run.stdout + run.stderr)
    return f"b{m[1]} ({m[2]})" if m else None


class LlamaServer:
    """`with LlamaServer(file, ctx) as server:` runs `llama-server -m file -ngl 99 -c ctx -np 1` for the block."""

    def __init__(self, path: Path, ctx: int, *, startup_s: float = 300.0):
        exe = binary()
        if exe is None:
            raise FileNotFoundError("llama-server not found: set LLAMA_SERVER or put it on PATH")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.command = [exe, "-m", str(path), "-ngl", "99", "-c", str(ctx), "-np", "1"]
        self.command += ["--host", "127.0.0.1", "--port", str(self.port)]
        self.startup_s = startup_s

    def __enter__(self) -> LlamaServer:
        self._log = tempfile.TemporaryFile()
        self._process = subprocess.Popen(self.command, stdout=self._log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + self.startup_s
        while True:
            failure = None
            if self._process.poll() is not None:
                failure = RuntimeError(f"llama-server exited with {self._process.returncode}:\n{self._tail()}")
            elif time.monotonic() > deadline:
                failure = TimeoutError(f"llama-server not ready after {self.startup_s:.0f} s:\n{self._tail()}")
            if failure:
                self.__exit__()
                raise failure
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError):
                pass  # loading (503) or not listening yet
            time.sleep(0.25)

    def __exit__(self, *exc) -> None:
        self._process.terminate()
        try:
            self._process.wait(30)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._log.close()

    def _tail(self, lines: int = 20) -> str:
        self._log.seek(0)
        return "\n".join(self._log.read().decode(errors="replace").splitlines()[-lines:])

    def complete(self, ids: Sequence[int], n: int, *, ignore_eos: bool = False) -> dict:
        """Greedy continuation of `ids` (no BOS added, no prompt cache): the /completion response with `tokens`
        (ending at an end-of-generation token if one came first unless `ignore_eos`) and `timings`."""
        body = {
            "prompt": list(ids),
            "n_predict": n,
            "temperature": 0.0,
            "ignore_eos": ignore_eos,
            "cache_prompt": False,
            "return_tokens": True,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as r:
            out = json.load(r)
        if out["tokens_evaluated"] != len(ids):
            raise RuntimeError(f"llama-server evaluated {out['tokens_evaluated']} prompt tokens, sent {len(ids)}")
        return out
