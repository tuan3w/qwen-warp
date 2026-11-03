"""KV-cache dtype quality guard: teacher-forced perplexity of the same text with bf16, fp8 and int4 KV (same engine, same quantised head).

  python tests/test_kv_dtypes.py [--tokens 3072]
Each dtype runs in its own subprocess (the engine owns ~20 GB).  The text is the repo's own Python source (real code, ships with the repo).  Measured on 32k-98k tokens
of other code with the same engine: fp8 +0.03%, int4 +0.26% perplexity vs bf16.  The gates here are looser because 3k tokens are noisy: fp8 <= +1.5%, int4 <= +3.0%.
A broken cache (wrong scale, bad nibble order, stale library) shows up as tens or hundreds of percent.  Needs the checkpoint ($SWIFT_MODEL).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ENG = Path(__file__).resolve().parent.parent / "engine"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
GATES = {1: 1.5, 2: 3.0}  # kv dtype -> max perplexity increase over bf16, percent


def corpus(n):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = []
    for f in sorted(ENG.glob("*.py")) + sorted((ENG.parent / "server").glob("*.py")):
        ids += tok(f.read_text(), add_special_tokens=False)["input_ids"]
        if len(ids) >= n:
            break
    assert len(ids) >= n, f"only {len(ids)} tokens of source text"
    return ids[:n]


def child(kv, n, out):
    import torch

    sys.path.insert(0, str(ENG))
    from engine import Engine

    ids = corpus(n)
    e = Engine(ctx_max=n + 64, kv_fp8=kv, tmax=8, verbose=False)
    e.reset()
    nll = []
    i = 0
    while i < n - 1:
        T = min(8, n - 1 - i)
        e._step(ids[i : i + T], 1)
        lp = torch.log_softmax(e.logits(T).cuda(), -1)
        tgt = torch.tensor(ids[i + 1 : i + 1 + T], device="cuda")
        nll += (-lp[torch.arange(T), tgt]).cpu().tolist()
        i += T
    assert all(math.isfinite(x) for x in nll), "non-finite NLL"
    json.dump(nll, open(out, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=3072)
    ap.add_argument("--child", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.child is not None:
        return child(a.child, a.tokens, a.out)
    ppl = {}
    for kv, name in ((0, "bf16"), (1, "fp8"), (2, "int4")):
        out = os.path.join(tempfile.gettempdir(), f"qwen_warp_kv{kv}.json")
        r = subprocess.run([sys.executable, __file__, "--child", str(kv), "--tokens", str(a.tokens), "--out", out], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        nll = json.load(open(out))
        os.unlink(out)
        ppl[kv] = math.exp(sum(nll) / len(nll))
        print(f"{name:5s} KV: perplexity {ppl[kv]:.4f} over {len(nll)} tokens", flush=True)
    ok = True
    for kv, gate in GATES.items():
        d = 100 * (ppl[kv] / ppl[0] - 1)
        good = d <= gate
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {'fp8' if kv == 1 else 'int4'} perplexity {d:+.2f}% vs bf16 (gate <= +{gate}%)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
