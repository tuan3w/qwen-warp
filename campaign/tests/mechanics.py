#!/usr/bin/env python3
"""mechanics.py <root> [case ...]: end-to-end tests of the harness on a prepared root (prepare.py + `campaign setup`, baseline done).  GPU lease must be held.

Each case = `campaign init`, a source mutation in the lineage's work tree, [`try`], `campaign gate`; asserts the decision the harness must reach:
  noop        comment appended to a .cu file (identical binary)          -> revert (no gain), check PASS            [`try` also run: prints check PASS]
  rounding    gate|up GEMV with another K-split (legit rounding change)  -> check PASS (must not false-reject), revert (no real gain), held / guard fine
  luck        q|k|v GEMV w8k4u2 instead of w8k8u2: raw tok/s was +0.9% in calibration only because a rounding change moved prose tokens/cycle +4% -> revert (the score must not reward it)
  slower      gate|up GEMV config w8k1u3 (1.6x slower kernel)            -> revert / revert-held (speed 0.84x), check PASS
  longctx     attention ignores the newest 64 KV rows at context > 2048  -> revert-output (only the 8K teacher-forced rows / long8k generations see it)
  sign        DeltaNet gate sign flipped (g = +exp(A_log)*sp)            -> the state explodes and the engine faults (CUDA 700): revert-crash (or revert-output if it survives)
  ret         `return` in the DeltaNet kernel for T >= 3 (verify rows)   -> revert-output (check FAIL)
  skipmlp     MLP of layers 20..23 skipped (faster but wrong)            -> revert-output although dev is +1.3% faster (check FAIL)
  forbidden   diff mentions the held-out prompt file                      -> revert-special-casing (no GPU)
  keep        noop with gate.keep lowered to 0.90 (campaign.json patched and restored) -> keep (commit + tag + tree node)
  fp8kv       engine.py / spec.py default kv_fp8=1 (approximate KV): measured -> revert-output (1 unexcused top-1 flip at T=1, no speed gain)
  pldtail     (K11) greedy accept kernel accepts every prompt-lookup tail token once the 3 chain drafts matched, WITHOUT verifying it -> revert-output (faster but wrong: only tail cycles see it)
  ringwrap    (K11) DeltaNet 9-slot ring: `cur` update off by one when it wraps (slots > 5) -> revert-output (rollback / wrap of the T=8 ring)
  ring5       `cur` update off by one when the 5-slot ring (tmax=4 library default) wraps -> revert-output, visible ONLY in the tmax=4 subprocess data of the check
Prints one line per case and exits 1 on any unexpected decision.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).expanduser()
SKILL = Path.home() / ".omp/plugins/node_modules/agent-campaign/skills/autonomous-campaign/scripts/campaign"


def camp(*args, check=True):
    r = subprocess.run([str(SKILL), "--root", str(ROOT), *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"campaign {' '.join(args)} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r


def sub(path, old, new):
    s = path.read_text()
    assert old in s, f"mutation anchor not found in {path}: {old!r}"
    path.write_text(s.replace(old, new, 1))


def m_noop(w):
    with (w / "src/misc.cu").open("a") as f:
        f.write("\n// harness mechanics test: comment only\n")


def m_rounding(w):
    sub(w / "src/gemv.cu", "int cfg = (N == 14336) ? 1 : 0;", "int cfg = (N == 14336 || N == 34816) ? 1 : 0;")


def m_slower(w):
    sub(w / "src/gemv.cu", "int cfg = (N == 14336) ? 1 : 0;", "int cfg = (N == 14336) ? 1 : (N == 34816 ? 5 : 0);")


def m_fp8(w):
    sub(w / "engine.py", "def __init__(self, ctx_max=4096, kv_fp8=0,", "def __init__(self, ctx_max=4096, kv_fp8=1,")
    sub(w / "spec.py", "def __init__(self, ctx_max=4096, kv_fp8=0,", "def __init__(self, ctx_max=4096, kv_fp8=1,")


def m_luck(w):
    sub(w / "src/gemv.cu", "int cfg = (N == 14336) ? 1 : 0;", "int cfg = 0;")


def m_longctx(w):
    """attention drops the newest 64 KV rows once the context exceeds 2048 tokens: invisible to every 130-token context, wrong on the 8K prompt"""
    sub(w / "src/attn.cu", "const int kstart = s * chunk, kend = min(ctx, kstart + chunk);", "const int kstart = s * chunk, kend = min(ctx > 2048 ? ctx - 64 : ctx, kstart + chunk);")


def m_sign(w):
    sub(w / "src/deltanet.cu", "const float g = -expf(bf(p.A_log[h])) * sp;", "const float g = expf(bf(p.A_log[h])) * sp;")


def m_ret(w):
    sub(w / "src/deltanet.cu", "  const int cur = *p.cur;\n", "  const int cur = *p.cur;\n  if (T >= 3 && (blockIdx.x & 7) == 7) return;\n")


def m_skipmlp(w):
    sub(w / "src/engine.cu", "    {\n      smk_epi ep = {};\n      ep.kind = SMK_EPI_SILU;", "    if (l < 20 || l > 23) {\n      smk_epi ep = {};\n      ep.kind = SMK_EPI_SILU;")


def m_pldtail(w):
    sub(w / "src/mtp.cu", "while (m < k && ds[DS_TOK + m + 1] == ds[DS_OUT + m]) m++;", "while (m < k && (m >= 3 || ds[DS_TOK + m + 1] == ds[DS_OUT + m])) m++;")


def m_ringwrap(w):
    sub(w / "src/mtp_sample.h", "  ds[DS_CUR] = (ds[DS_CUR] + a) % ds[DS_SLOTS];", "  { const int c_ = ds[DS_CUR] + a, S_ = ds[DS_SLOTS]; ds[DS_CUR] = c_ >= S_ ? (S_ > 5 ? c_ - S_ + 1 : c_ - S_) : c_; }")


def m_ring5(w):
    sub(w / "src/mtp_sample.h", "  ds[DS_CUR] = (ds[DS_CUR] + a) % ds[DS_SLOTS];", "  { const int c_ = ds[DS_CUR] + a, S_ = ds[DS_SLOTS]; ds[DS_CUR] = c_ >= S_ ? (S_ == 5 ? c_ - S_ + 1 : c_ - S_) : c_; }")


def m_forbidden(w):
    with (w / "engine.py").open("a") as f:
        f.write("\n# tuned on held_prompts\n")


CASES = {  # name: (mutation, expected decision, extra assertion on the ledger row, run try first)
    "noop": (m_noop, "revert", lambda r: r["correct"] and abs(r["speedup_pooled"] - 1) < 0.02, True),
    "rounding": (m_rounding, "revert", lambda r: r["correct"] and 0.99 < r["speedup_pooled"] < 1.003, False),
    "slower": (m_slower, ("revert", "revert-held"), lambda r: r["correct"] and r["speedup_pooled"] < 0.95, False),
    "luck": (m_luck, "revert", lambda r: r["correct"] and r["speedup_pooled"] < 1.003, False),
    "longctx": (m_longctx, "revert-output", lambda r: not r["correct"], False),
    "sign": (m_sign, ("revert-output", "revert-crash"), lambda r: True, False),
    "ret": (m_ret, "revert-output", lambda r: not r["correct"], False),
    "skipmlp": (m_skipmlp, "revert-output", lambda r: not r["correct"] and r["speedup_pooled"] > 1.005, False),
    "forbidden": (m_forbidden, "revert-special-casing", lambda r: True, False),
    "keep": (m_noop, "keep", lambda r: r["correct"], False),
    "fp8kv": (m_fp8, ("revert-output", "revert"), lambda r: True, False),
    "pldtail": (m_pldtail, "revert-output", lambda r: not r["correct"], False),
    "ringwrap": (m_ringwrap, "revert-output", lambda r: not r["correct"], False),
    "ring5": (m_ring5, "revert-output", lambda r: not r["correct"] and ("tmax4" in json.dumps(r) or "m4:" in json.dumps(r)), False),
}


def run_case(name):
    mut, want, extra, do_try = CASES[name]
    lin = f"t_{name}"
    cfgp = ROOT / "campaign.json"
    cfg0 = cfgp.read_text()
    if name == "keep":
        c = json.loads(cfg0)
        c["gate"]["keep"] = 0.90
        cfgp.write_text(json.dumps(c, indent=1))
    try:
        camp("init", lin, f"mechanics test {name}")
        w = ROOT / lin / "work"
        mut(w)
        camp("brief", lin, "5")
        (ROOT / lin / "ASI.json").write_text(json.dumps(dict(headline=f"mechanics test {name}", hypothesis="harness test", change=name, predicted_gain_pct=0)))
        tr = ""
        if do_try:
            r = camp("try", lin, check=False)
            tr = r.stdout
            print(f"[{name}] try:\n{tr}{r.stderr[-800:]}", flush=True)
        r = camp("gate", lin)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        row = [json.loads(l) for l in (ROOT / "campaign/ledger.jsonl").read_text().splitlines() if l.strip()][-1]
    finally:
        cfgp.write_text(cfg0)
    ok = (want is None or res["decision"] in (want if isinstance(want, tuple) else (want,))) and bool(extra(row))
    facts = "; ".join(f"{a}: {b}" for a, b in row.get("facts", []))
    print(f"[{name}] {'PASS' if ok else 'UNEXPECTED'} decision={res['decision']} speedup_pooled={res.get('speedup_pooled')} reasons={res.get('reasons')}\n    facts: {facts}", flush=True)
    return ok


if __name__ == "__main__":
    names = sys.argv[2:] or list(CASES)
    results = {n: run_case(n) for n in names}
    print("RESULT", results)
    sys.exit(0 if all(results.values()) else 1)
