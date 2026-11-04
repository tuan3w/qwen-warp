#!/usr/bin/env python
"""Every documented way to start the qwen-warp server must start AND serve on a 24 GB RTX 3090 that also drives a desktop (~0.6 GiB in use).  Needs the checkpoint ($SWIFT_MODEL,
default ~/models/swift-1.5-int4); starts and stops its own servers, one after the other (about 6 minutes).

  python tests/test_startup_configs.py [--port 8021] [--full]

For each configuration: start serve.py, wait for /health, send a short chat request, then a /v1/completions request with a prompt of (--max-model-len - 64) tokens and 64 generated tokens,
and report the time to ready and the GPU memory in use afterwards.  --max-model-len defaults come from serve.py, so a default that no longer fits fails here.
  no flags | --prompt-lookup 0 | --kv-cache-dtype fp8 | --profile long | --profile long --max-model-len 262144 --prompt-lookup 0 | bf16 KV at the longest length a 24 GB card takes
Beyond the memory there must be a fast, actionable refusal: two configurations that need more than the card has must exit non-zero within 60 s (not hang, not die with a raw CUDA OOM
after loading the weights) and the message must give the free memory and the largest --max-model-len per cache layout.
A prompt of ~200k tokens takes 3 minutes to prefill: without --full the long profiles get a 32,768-token prompt (their KV cache is allocated up front and the request-time scratch does not
depend on the prompt length: the torch reserved memory after a request is identical at 8k and 98k tokens); --full sends the literal near-limit prompts (adds about 9 minutes).
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "server" / "serve.py"
sys.path.insert(0, str(HERE.parent / "server"))
import serve  # noqa: E402  (only the default constants; no CUDA, no vLLM import at module level)

EDGE_BF16 = 32768  # bf16 KV without the tail: measured to start and serve up to 34,816 on a 24 GB card with a 0.6 GiB desktop
JUST_ABOVE = 36864  # first length above that edge that fails (35,840 does not serve)
ABSURD = 131072  # bf16 KV: 29 GB of cache alone
QUICK_PROMPT = 32768


def smi_used():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


def stop(srv):
    if srv.poll() is None:
        os.killpg(srv.pid, signal.SIGINT)
        try:
            srv.wait(30)
        except subprocess.TimeoutExpired:
            os.killpg(srv.pid, signal.SIGKILL)
            srv.wait()
    for _ in range(30):  # the next configuration must see the same free memory
        if smi_used() < 2000:
            break
        time.sleep(2)


class T:
    def __init__(self):
        self.res = []

    def check(self, name, ok, detail=""):
        self.res.append((name, bool(ok)))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else (f"  ({detail})" if detail else "")), flush=True)


def launch(flags, port, log):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    return subprocess.Popen([sys.executable, str(SERVE), "--port", str(port)] + flags, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True)


def serves(t, name, flags, limit, prompt_len, port):
    import httpx
    import numpy as np

    log = f"/tmp/swift_mk_startup_{port}.log"
    srv = launch(flags, port, log)
    base = f"http://127.0.0.1:{port}"
    t0 = time.time()
    try:
        while srv.poll() is None and time.time() - t0 < 240:
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            tail = "".join(Path(log).read_text().splitlines(True)[-4:]).strip()
            t.check(f"{name}: starts", False, f"exit {srv.poll()} after {time.time() - t0:.0f} s: {tail[-300:]}")
            return
        ready = time.time() - t0
        t.check(f"{name}: starts", True, f"/health after {ready:.1f} s")
        r = httpx.post(base + "/v1/chat/completions", timeout=120, json=dict(model="swift", messages=[{"role": "user", "content": "Say hi."}], max_tokens=16, temperature=0,
                                                                          chat_template_kwargs=dict(enable_thinking=False)))
        ok = r.status_code == 200 and r.json()["choices"][0]["message"]["content"].strip() != ""
        t.check(f"{name}: short chat request", ok, f"HTTP {r.status_code}" + ("" if ok else r.text[:150]))
        ids = np.random.default_rng(1).integers(1000, 100000, size=prompt_len).tolist()
        t1 = time.time()
        r = httpx.post(base + "/v1/completions", timeout=900, json=dict(model="swift", prompt=ids, max_tokens=64, temperature=0, ignore_eos=True))
        n = r.json()["usage"]["completion_tokens"] if r.status_code == 200 else None
        t.check(f"{name}: {prompt_len:,}-token prompt (limit {limit:,}) + 64 tokens", n == 64 and srv.poll() is None, f"HTTP {r.status_code}" + ("" if n == 64 else r.text[:200]))
        print(f"      time to ready {ready:.1f} s, prefill+decode {time.time() - t1:.1f} s, GPU memory in use {smi_used()} MiB", flush=True)
    finally:
        stop(srv)


def refuses(t, name, flags, port, hint):
    import httpx

    log = f"/tmp/swift_mk_startup_{port}.log"
    srv = launch(flags, port, log)
    t0 = time.time()
    try:
        while srv.poll() is None and time.time() - t0 < 60:
            time.sleep(0.5)
        dt = time.time() - t0
        text = Path(log).read_text()
        t.check(f"{name}: exits with an error within 60 s", srv.poll() not in (None, 0), f"exit {srv.poll()} after {dt:.1f} s")
        t.check(f"{name}: message names the free memory and the largest lengths", "MiB are free" in text and "Largest --max-model-len" in text and "--prompt-lookup 0" in text and hint in text,
                "; ".join(text.strip().splitlines()[-3:])[:400])
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=1)
            up = True
        except httpx.HTTPError:
            up = False
        t.check(f"{name}: no server is left listening", not up)
    finally:
        stop(srv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--full", action="store_true", help="send the literal near-limit prompt to the long-context configurations (3 to 5 minutes each)")
    a = ap.parse_args()
    t = T()
    used = smi_used()
    if used > 1500:
        raise SystemExit(f"the GPU already holds {used} MiB: the limits are defined for a card with only a desktop on it (~0.6 GiB)")
    cap = (lambda n: n) if a.full else (lambda n: min(n, QUICK_PROMPT))
    serves(t, "no flags", [], serve.DEFAULT_MAX_MODEL_LEN_PLD, serve.DEFAULT_MAX_MODEL_LEN_PLD - 64, a.port)
    serves(t, "--prompt-lookup 0", ["--prompt-lookup", "0"], serve.DEFAULT_MAX_MODEL_LEN, serve.DEFAULT_MAX_MODEL_LEN - 64, a.port)
    serves(t, "--kv-cache-dtype fp8", ["--kv-cache-dtype", "fp8"], serve.DEFAULT_MAX_MODEL_LEN_PLD, serve.DEFAULT_MAX_MODEL_LEN_PLD - 64, a.port)
    serves(t, "--profile long", ["--profile", "long"], serve.LONG_MAX_MODEL_LEN, cap(serve.LONG_MAX_MODEL_LEN - 64), a.port)
    serves(t, "--profile long --max-model-len 262144 --prompt-lookup 0", ["--profile", "long", "--max-model-len", "262144", "--prompt-lookup", "0"], 262144, cap(262144 - 64), a.port)
    serves(t, f"bf16 --max-model-len {EDGE_BF16:,} --prompt-lookup 0", ["--max-model-len", str(EDGE_BF16), "--prompt-lookup", "0"], EDGE_BF16, EDGE_BF16 - 64, a.port)
    refuses(t, f"--max-model-len {ABSURD:,}", ["--max-model-len", str(ABSURD)], a.port, "--profile long")
    if 450 <= used <= 900:  # the edge is defined for the 0.6 GiB desktop
        refuses(t, f"--max-model-len {JUST_ABOVE:,} --prompt-lookup 0", ["--max-model-len", str(JUST_ABOVE), "--prompt-lookup", "0"], a.port, "--profile long")
    else:
        print(f"SKIP  just-above-the-limit refusal (the GPU holds {used} MiB before the start, the limit is defined for 450 to 900 MiB)")
    bad = [n for n, ok in t.res if not ok]
    print(f"\n{len(t.res) - len(bad)}/{len(t.res)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
