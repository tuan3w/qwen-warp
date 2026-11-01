#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""2026-10-31-spec-sampling-exactness — exact speculative sampling in the swift-mk on-device MTP cycle (see README.md).

This is a thin driver: the work is done by engine/{tests/test_sampling.py, tests/bench_spec.py} in the lab venv on the shared 3090.
Run (needs the GPU lease: `projects/llm-lab/bin/lab gpu acquire` first, release after; ~35 min with N = 20000 e2e samples):

    uv run experiments/2026-10-31-spec-sampling-exactness/run.py [--quick]

--quick: 1/10 of the e2e samples and 200k kernel trials (smoke).  Results go to results/ (tracked): test_sampling_kernel.md, e2e_analysis_*.md, bench_*.log/json,
greedy_regression*.log, e2e_*.json (raw samples).  Build first: `cd engine && make BUILD=build/k7`.
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
ENV = {**os.environ, "SMK_BUILD": os.environ.get("SMK_BUILD", "build/k7")}


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, "-u", *args], cwd=REPO, env=ENV, check=False)


def main() -> None:
    quick = "--quick" in sys.argv
    T = str(ENG / "tests/test_sampling.py")
    B = str(ENG / "tests/bench_spec.py")
    RES.mkdir(exist_ok=True)
    run(T, "philox,processors,trials", "--ntr", "200000" if quick else "2000000", "--out", str(RES / "test_sampling_kernel.md"))
    run(str(ENG / "tests/test_mtp.py"), "--only", "greedy,rollback", "--prompts", "code,prose,math", "--ks", "1,2,3")
    scale = 10 if quick else 1
    for tag, extra, n_ok, n_bad, args in (
        ("A", 6, 20000 // scale, 4000 // scale, ["--prompt", "prose", "--k", "3", "--T", "1.0", "--top-k", "20", "--top-p", "0.95"]),
        ("B", 10, 6000 // scale, 2000 // scale, ["--prompt", "math", "--k", "2", "--T", "0.6", "--top-k", "20", "--top-p", "0.95", "--draft-sample", "0"]),
    ):
        a = [*args, "--extra", str(extra)]
        run(T, "e2e-ref", *a, "--tag", tag, "--n", str(20000 // scale))
        for s in range(1, 5 if tag == "A" else 2):
            run(T, "e2e-run", *a, "--tag", tag, "--n", str(n_ok // (4 if tag == "A" else 1)), "--seed", str(s))
        run(T, "e2e-run", *a, "--tag", tag + "_broken", "--broken", "--n", str(n_bad), "--seed", "11")
    for tag in ("A", "B"):
        run(T, "e2e-analyze", "--tag", tag, "--out", str(RES / f"e2e_analysis_{tag}.md"))
        run(T, "e2e-analyze", "--tag", tag + "_broken", "--ref", tag, "--expect-fail", "--out", str(RES / f"e2e_analysis_{tag}_broken.md"))
    for name, flags in (("sampled_T1", ["--temperature", "1.0", "--ks", "0,1,2,3"]), ("delta_T1", ["--temperature", "1.0", "--draft-sample", "0", "--ks", "1,2,3"]),
                        ("sampled_T0.6", ["--temperature", "0.6", "--ks", "3"]), ("greedy", ["--ks", "3"])):
        run(B, "--prompts", "code,prose,math,long8k", "--reps-long", "1", "--fast-prefill", *flags, "--json", str(RES / f"bench_{name}.json"))


if __name__ == "__main__":
    main()
