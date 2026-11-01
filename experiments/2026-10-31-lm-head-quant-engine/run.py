#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""2026-10-31-lm-head-quant-engine — quantised lm_head + exact bf16 rescoring inside the swift-mk engine (see README.md).

Thin driver: the work is done by engine/tests/{test_headq.py, bench_head_ab.py, test_e2e.py, test_mtp.py, test_sampling.py} and
server/maxctx.py in the lab venv on the shared 3090.  Build first: `cd engine && make BUILD=build/k9`.
Run (needs the GPU lease: `projects/llm-lab/bin/lab gpu acquire --holder <you>` first, release after; ~45 min):

    uv run experiments/2026-10-31-lm-head-quant-engine/run.py [--quick]

Results go to results/ (tracked).  test_headq.py needs the E5 hidden states (experiments/2026-10-31-lm-head-quantisation-quality/runs/hidden.pt, regenerate with that
experiment's `run.py collect`).  --quick: 5 A/B reps -> 3, 4 x 1250 e2e sampling samples.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENG = REPO / "engine"
PY = os.path.expanduser("~/llm-lab/venv/bin/python")
RES = HERE / "results"
ENV = {**os.environ, "SMK_BUILD": os.environ.get("SMK_BUILD", "build/k9")}


def run(*args: str, env: dict | None = None, log: Path | None = None) -> None:
    print("+", " ".join(args), flush=True)
    out = open(log, "w") if log else None
    subprocess.run([PY, "-u", *args], cwd=ENG, env={**ENV, **(env or {})}, check=False, stdout=out, stderr=subprocess.STDOUT if out else None)


def main() -> None:
    quick = "--quick" in sys.argv
    RES.mkdir(exist_ok=True)
    T = ["tests/test_headq.py"]
    run(*T, "--what", "unit", "--heads", "int8,int4", "--out", str(RES / "test_headq_unit.md"))
    for h in ("int8", "int4"):
        run(*T, "--what", "parity", "--heads", h, "--out", str(RES / f"test_headq_parity_{h}.md"))
        run(*T, "--what", "sampling", "--heads", h, "--out", str(RES / f"test_headq_sampling_{h}.md"))
        run("tests/bench_head_ab.py", "--head", h, "--reps", "3" if quick else "5", "--fast-prefill", "--json", str(RES / f"bench_ab_{h}.json"), log=RES / f"bench_ab_{h}.log")
    run("tests/test_e2e.py", "--out", str(RES / "test_e2e_int4.md"), log=RES / "test_e2e_int4.log")
    run("tests/test_mtp.py", "--out", str(RES / "test_mtp_int4.md"), log=RES / "test_mtp_int4.log")
    srv = {"SMK_HEAD": "int4", "SMK_SAMP_RES": str(RES / "sampling")}
    a = ["--prompt", "prose", "--k", "3", "--T", "1.0", "--top-k", "20", "--top-p", "0.95", "--extra", "3", "--tag", "A"]
    s = "4" if quick else "1"
    run("tests/test_sampling.py", "e2e-ref", *a, "--n", str(20000 // int(s) if quick else 20000), env=srv, log=RES / "sampling_ref_A.log")
    for seed in range(1, 5):
        run("tests/test_sampling.py", "e2e-run", *a, "--n", str(5000 // int(s)), "--seed", str(seed), env=srv, log=RES / f"sampling_run_A_{seed}.log")
    run("tests/test_sampling.py", "e2e-analyze", "--tag", "A", "--out", str(RES / "sampling" / "e2e_analysis_A.md"), env=srv)
    for kv, ctxs in (("", (32768, 36864, 40960)), ("fp8", (65536, 73728, 81920))):
        for c in ctxs:
            run("../server/maxctx.py", str(c), *([kv] if kv else []), env={"SMK_HEAD": "int4"})


if __name__ == "__main__":
    main()
