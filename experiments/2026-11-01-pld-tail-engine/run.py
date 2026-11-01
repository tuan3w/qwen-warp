"""pld-tail-engine: reproduce the measurements (GPU lease held for the whole run; `projects/llm-lab/bin/lab gpu acquire` first, `release` after).

  SMK_BUILD=build/k11 ~/llm-lab/venv/bin/python experiments/2026-11-01-pld-tail-engine/run.py step0|ab|tests|tables

step0   verify-step time V(T), T = 1..8, ctx 64 / 2048 / 8192 + the DeltaNet / attention kernel benches + cycle parts (results/bench_verify.log, vt_int4.json, bench_deltanet.log, bench_attn.log)
ab      A/B vs the committed cycles (same process, interleaved, 3 reps): E6 workload greedy + T=1 sampling, E0 prompts, long-context class f, gate sweep m = 3 / 4 / 6
tests   correctness: kernels (test_deltanet / test_attn T <= 8), the proposer vs E6's simulation, hybrid greedy == plain greedy, forced partial acceptance up to 8 rows, ring wraps,
        K7 chi-square with the delta tail incl. the broken-variant power check, server runner_check / smoke_test with the tail, max-model-len
tables  CPU only: results/tables.md (analyze_ab.py) and the offline replay with the measured costs (policy_sim.py)
Logs land in results/; the exact command lines are below.
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ENG = REPO / "engine"
PY = os.path.expanduser("~/llm-lab/venv/bin/python")
R = HERE / "results"
ENV = dict(os.environ, SMK_BUILD=os.environ.get("SMK_BUILD", "build/k11"))

STEPS = {
    "step0": [
        f"{PY} {ENG}/tests/bench_verify.py --ctxs 64,2048,8192 --reps 7 --json {R}/vt_int4.json | tee {R}/bench_verify.log",
        f"{PY} {ENG}/tests/bench_deltanet.py --lib {ENG}/build/k11/libswiftmk_deltanet.so | tee {R}/bench_deltanet.log",
        f"ATTN_SO={ENG}/build/k11/libswiftmk_attn.so {PY} {ENG}/tests/bench_attn.py --ctx 2048 8192 --dtype both --T 1 2 3 4 5 6 7 8 | tee {R}/bench_attn.log",
    ],
    "ab": [
        f"{PY} {ENG}/tests/bench_pld.py --suite e0 --reps 3 --modes base,notail,pld --parts --json {R}/ab_e0_greedy.json | tee {R}/ab_e0_greedy.log",
        f"{PY} {ENG}/tests/bench_pld.py --suite e6 --reps 3 --modes base,pld --json {R}/ab_e6_greedy.json | tee {R}/ab_e6_greedy.log",
        f"{PY} {ENG}/tests/bench_pld.py --suite e6 --reps 3 --modes base,pld --temperature 1.0 --json {R}/ab_e6_t1.json | tee {R}/ab_e6_t1.log",
        f"{PY} {ENG}/tests/bench_pld.py --suite long --reps 3 --modes base,pld3,pld4,pld6 --json {R}/ab_long_greedy.json | tee {R}/ab_long_greedy.log",
        f"{PY} {ENG}/tests/bench_pld.py --suite e0 --reps 3 --modes base,pld3,pld4,pld6 --json {R}/ab_e0_greedy_m.json | tee {R}/ab_e0_greedy_m.log",
        f"{PY} {ENG}/tests/bench_pld.py --suite e6 --tasks d1,d2,d3,d4,d5,d6,e1,e2,e3,e4,e5,e6 --reps 3 --modes base,pld3,pld4,pld6 --json {R}/ab_e6_de_m.json | tee {R}/ab_e6_de_m.log",
    ],
    "tests": [
        f"{PY} {ENG}/tests/test_deltanet.py --lib {ENG}/build/k11/libswiftmk_deltanet.so --slots 9 | tee {R}/test_deltanet_9.log",
        f"ATTN_SO={ENG}/build/k11/libswiftmk_attn.so {PY} {ENG}/tests/test_attn.py | tee {R}/test_attn.log",
        f"{PY} {ENG}/tests/test_pld.py | tee {R}/test_pld.log",
        f"{PY} {ENG}/tests/test_sampling.py trials-pld --ntr 2000000 --out {R}/trials_pld.log",
        f"{PY} {ENG}/tests/test_pld_engine.py --n 192 --out {R}/test_pld_engine.log",
        f"{PY} {REPO}/server/runner_check.py --pld | tee {R}/runner_check_pld.log",
        f"{PY} {REPO}/server/smoke_test.py --pld 1 | tee {R}/smoke_pld.log",
    ],
    "tables": [
        f"python3 {HERE}/analyze_ab.py",
        f"python3 {HERE}/policy_sim.py --cost {R}/cost_2048.json --out {R}/policy_sim.json | tee {R}/policy_sim.log",
    ],
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        raise SystemExit(__doc__)
    for cmd in STEPS[sys.argv[1]]:
        print("$", cmd, flush=True)
        subprocess.run(cmd, shell=True, cwd=REPO, env=ENV, check=True)
