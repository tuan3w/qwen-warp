"""Host-pinned embedding / lm_head tables (Engine(host_tables=True)): bit-identical to the GPU tables, ~4.7 GiB less VRAM, no per-step cost.

  python tests/test_host_tables.py [--kv 2] [--prompt 600]
Runs the same workload twice in subprocesses (GPU tables, host tables) with SpecEngine: prefill, plain greedy decode, one T=1 head step's logits, MTP spec cycles.
Checked: greedy ids and spec ids EQUAL, logits bit-identical.  Reported: device GiB, plain decode ms/token, spec tok/s.  Needs the checkpoint ($SWIFT_MODEL).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ENG = Path(__file__).resolve().parent.parent / "engine"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))


def prompt_ids(n):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    text = (ENG / "spec.py").read_text() + "\n\n" + (ENG / "prefill.py").read_text()  # real text that ships with the repo
    return tok(text, add_special_tokens=False)["input_ids"][:n]


def child(args):
    import torch

    sys.path.insert(0, str(ENG))
    from spec import SpecEngine

    ids = prompt_ids(args.prompt)
    e = SpecEngine(ctx_max=args.prompt + 512, kv_fp8=args.kv, verbose=False, host_tables=bool(args.host))
    f, t = torch.cuda.mem_get_info()
    used = (t - f) / 2**30
    e.reset()
    first = e.prefill(ids)
    plain = e.decode(48)
    lg = e.logits(1)[0].clone()
    e.decode(8)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e.decode(64)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / 64 * 1e3
    e.reset()
    first2 = e.prefill_spec(ids, 3)
    e.cycles(3, 4)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    toks, _ = e.cycles(3, 24)
    torch.cuda.synchronize()
    sp = len(toks) / (time.perf_counter() - t0)
    torch.save({"first": first, "plain": plain, "logits": lg.cpu(), "spec": [first2] + toks}, args.out)
    print(json.dumps({"host": args.host, "device_gib": round(used, 2), "decode_ms": round(ms, 2), "spec_tok_s": round(sp, 1), "on_host": e.tables_on_host}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=int, default=2)
    ap.add_argument("--prompt", type=int, default=600)
    ap.add_argument("--host", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.host is not None:
        return child(args)
    import tempfile

    import torch

    res, outs = {}, {}
    for host in (0, 1):
        out = os.path.join(tempfile.gettempdir(), f"qwen_warp_host_tables_{host}.pt")
        r = subprocess.run([sys.executable, __file__, "--kv", str(args.kv), "--prompt", str(args.prompt), "--host", str(host), "--out", out], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        assert r.returncode == 0 and line, r.stderr[-2000:]
        res[host] = json.loads(line[-1])
        outs[host] = torch.load(out)
        os.unlink(out)
        print(res[host], flush=True)
    a, b = outs[0], outs[1]
    assert a["first"] == b["first"] and a["plain"] == b["plain"], "greedy ids differ"
    assert torch.equal(a["logits"], b["logits"]), "logits differ"
    n = min(len(a["spec"]), len(b["spec"]))
    assert a["spec"][:n] == b["spec"][:n], "spec ids differ"
    print(f"HOST TABLES OK: greedy ids + logits bit-identical, spec ids equal ({n}); VRAM {res[0]['device_gib']:.2f} -> {res[1]['device_gib']:.2f} GiB, "
          f"decode {res[0]['decode_ms']:.2f} -> {res[1]['decode_ms']:.2f} ms/token, spec {res[0]['spec_tok_s']:.0f} -> {res[1]['spec_tok_s']:.0f} tok/s")


if __name__ == "__main__":
    main()
