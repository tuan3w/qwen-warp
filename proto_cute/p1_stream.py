"""Decode rate from SSE chunk timestamps: tokens after the first chunk / time after the first chunk (prefill and TTFT excluded)."""
import json, sys, time, statistics as S, urllib.request
PORT, MODEL, N = sys.argv[1], sys.argv[2], 300
sys.argv = sys.argv[:1]
CODE = ("import sqlite3\nfrom typing import Any\n\n def connect(db_path: str) -> sqlite3.Connection:\n    \"\"\"Open a connection.\"\"\"\n    return sqlite3.connect(db_path)\n\n"
        "def close(conn: sqlite3.Connection) -> None:\n    \"\"\"Close a connection.\"\"\"\n    return conn.close()\n\n") * 4 + "Continue the module:\n\n"
PROSE = ("The lighthouse keeper climbed the spiral stairs each evening to trim the lamp. " * 12) + "Continue the story: "
def run(prompt):
    body = dict(model=MODEL, prompt=prompt, max_tokens=N, temperature=0.0, ignore_eos=True, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; n_first = 0; usage = None; last = None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
            d = json.loads(line[6:]); now = time.perf_counter()
            if d.get("usage"): usage = d["usage"]["completion_tokens"]
            if d.get("choices") and d["choices"][0].get("text") is not None and d["choices"][0]["text"] != "":
                if first is None: first = now; n_first = 1
                last = now
    return (usage - n_first) / (last - first), first - t0
for label, p in (("code", CODE), ("prose", PROSE)):
    r = [run(p) for _ in range(4)][1:]
    print(f"{label:6s} decode {S.median(x for x, _ in r):6.1f} tok/s (chunks after the first)   TTFT {S.median(t for _, t in r)*1000:6.0f} ms")
