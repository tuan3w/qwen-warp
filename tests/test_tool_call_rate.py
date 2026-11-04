#!/usr/bin/env python
"""Tool-call success rate gate of the qwen-warp OpenAI server.  Needs an RTX 3090 (the server takes ~21 GB) and the checkpoint ($SWIFT_MODEL, default ~/models/swift-1.5-int4).

  python tests/test_tool_call_rate.py [--profile long|default] [--n 40] [--min-rate 0.70] [--sort on|off] [--port 8013]

Starts serve.py, sends N identical requests ("What is the weather in Hanoi?" + a get_weather tool whose schema is written type-first, the way the OpenAI SDK / pydantic
send it; thinking on, default sampling = the generation config's temperature 1 / top_k 20 / top_p 0.95; max_tokens 800) and counts the answers that carry a well-formed
tool call (get_weather with a string `city`).  Gates: (1) the prompt the server rendered has the parameters keys in the order --sort asks for (default on: properties,
required, type); (2) success rate >= --min-rate.  Why N = 40: the int4-KV profile gave 39% with the schema rendered as sent and 87% rendered sorted (n = 150 each), so 70% over
40 requests separates the two by about 3 standard deviations on either side; the bf16-KV profile is above 90%.  A failing request is the model sampling `<|im_start|>` where
`<function` belongs, right after `<tool_call>\\n` (the sampled distribution is the bf16 head's: tests/test_sampler_dist.py), not a parser problem.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVE = HERE.parent / "server" / "serve.py"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def well_formed(msg):
    calls = msg.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != "get_weather":
        return False
    try:
        city = json.loads(calls[0].function.arguments).get("city")
    except ValueError:
        return False
    return isinstance(city, str) and bool(city)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="long", choices=["default", "long"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--min-rate", type=float, default=0.70)
    ap.add_argument("--sort", default=None, choices=["on", "off"], help="server --sort-tool-schema (default: the server's default); the gate (1) expects this rendering (default on)")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--max-model-len", type=int, default=16384)
    a = ap.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    log = open("/tmp/swift_mk_tool_rate_server.log", "w")
    cmd = [sys.executable, str(SERVE), "--port", str(a.port), "--max-model-len", str(a.max_model_len), "--profile", a.profile] + (["--sort-tool-schema", a.sort] if a.sort else [])
    srv = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    import httpx
    from openai import OpenAI
    from transformers import AutoTokenizer

    base = f"http://127.0.0.1:{a.port}"
    fails = []
    try:
        while True:
            if srv.poll() is not None:
                raise SystemExit("server died, see /tmp/swift_mk_tool_rate_server.log")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        cl = OpenAI(base_url=base + "/v1", api_key="x", timeout=300)
        tok = AutoTokenizer.from_pretrained(MODEL)

        def ask(**kw):
            return cl.chat.completions.create(model="swift", messages=[{"role": "user", "content": "What is the weather in Hanoi?"}], tools=TOOLS, max_tokens=800, **kw)

        first = ask(extra_body=dict(return_token_ids=True))
        prompt = tok.decode(first.model_extra["prompt_token_ids"])
        want = '"parameters": {"properties": {"city"' if a.sort != "off" else '"parameters": {"type": "object", "properties": {"city"'
        ok_render = want in prompt
        print(f"{'PASS' if ok_render else 'FAIL'}  rendered prompt has the schema in the {'sorted' if a.sort != 'off' else 'as-sent'} key order", flush=True)
        if not ok_render:
            fails.append("render")
        ok = int(well_formed(first.choices[0].message))
        t0 = time.time()
        for _ in range(a.n - 1):
            ok += int(well_formed(ask().choices[0].message))
        lo, hi = wilson(ok, a.n)
        passed = ok / a.n >= a.min_rate
        print(f"{'PASS' if passed else 'FAIL'}  tool-call rate {ok}/{a.n} = {ok / a.n:.2f} (Wilson 95% [{lo:.2f}, {hi:.2f}]) >= {a.min_rate:.2f}  [profile {a.profile}, {time.time() - t0:.0f} s]", flush=True)
        if not passed:
            fails.append("rate")
    finally:
        srv.send_signal(2)
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill()
    print("TOOL CALL RATE " + ("OK" if not fails else f"FAILED: {fails}"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
