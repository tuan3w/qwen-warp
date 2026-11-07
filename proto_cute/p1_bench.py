"""P1 benchmark against any OpenAI-compatible server (stock vLLM, this repo's engine): greedy decode tok/s, 1 and N clients, plus spec-decode acceptance from /metrics when available."""
import json, sys, time, re, statistics as S, urllib.request, concurrent.futures as cf
PORT = sys.argv[1] if len(sys.argv) > 1 else "8100"; MODEL = sys.argv[2] if len(sys.argv) > 2 else "swift"; N = 300
CODE = ("import sqlite3\nfrom typing import Any\n\n def connect(db_path: str) -> sqlite3.Connection:\n    \"\"\"Open a connection.\"\"\"\n    return sqlite3.connect(db_path)\n\n"
        "def close(conn: sqlite3.Connection) -> None:\n    \"\"\"Close a connection.\"\"\"\n    return conn.close()\n\n") * 4 + "Continue the module:\n\n"
PROSE = ("The lighthouse keeper climbed the spiral stairs each evening to trim the lamp. " * 12) + "Continue the story: "
CHAT = ["Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few test cases.",
        "Explain how a hash table handles collisions, and when you would choose open addressing over chaining.",
        "Write a SQL query to find the top 3 customers by total order value in each country, given tables customers(id, name, country) and orders(id, customer_id, total).",
        "Summarize the main causes of the French Revolution in a few paragraphs."]
def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)
def comp(prompt):
    t0 = time.perf_counter(); d = post("/v1/completions", dict(model=MODEL, prompt=prompt, max_tokens=N, temperature=0.0, ignore_eos=True))
    return d["usage"]["completion_tokens"], time.perf_counter() - t0
def chat(text):
    t0 = time.perf_counter(); d = post("/v1/chat/completions", dict(model=MODEL, messages=[{"role": "user", "content": text}], max_tokens=N, temperature=0.0,
                                                                     chat_template_kwargs={"enable_thinking": False}))
    return d["usage"]["completion_tokens"], time.perf_counter() - t0
def metrics():
    try: txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=10).read().decode()
    except Exception: return None
    g = lambda k: sum(float(m.group(1)) for m in re.finditer(rf"^{k}(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M))
    return g("vllm:spec_decode_num_drafts_total"), g("vllm:spec_decode_num_accepted_tokens_total")
def tpc(m0, m1):
    if not m0 or not m1 or m1[0] == m0[0]: return float("nan")
    return 1 + (m1[1] - m0[1]) / (m1[0] - m0[0])      # tokens per cycle = 1 + accepted per draft step
def comp_n(prompt, n):
    t0 = time.perf_counter(); d = post("/v1/completions", dict(model=MODEL, prompt=prompt, max_tokens=n, temperature=0.0, ignore_eos=True))
    return d["usage"]["completion_tokens"], time.perf_counter() - t0
def decode_only(prompt, reps=3):
    """differential: (300 - 30 tokens) / (t300 - t30) removes prefill and fixed per-request cost"""
    a = [comp_n(prompt, 30) for _ in range(reps)]; b = [comp_n(prompt, 300) for _ in range(reps)]
    return (S.median(n for n, _ in b) - S.median(n for n, _ in a)) / (S.median(t for _, t in b) - S.median(t for _, t in a))
comp("warm up " * 8); chat("hi"); 
print(f"== server :{PORT} model {MODEL}, {N} tokens, greedy")
m0 = metrics(); print(f"1 client  decode-only code    {decode_only(CODE):6.1f} tok/s   tokens/cycle {tpc(m0, metrics()):.2f}   (prefill excluded)")
m0 = metrics(); print(f"1 client  decode-only prose   {decode_only(PROSE):6.1f} tok/s   tokens/cycle {tpc(m0, metrics()):.2f}   (prefill excluded)", flush=True)
for label, fn, arg in [("code (completions)", comp, CODE), ("prose (completions)", comp, PROSE), ("chat code", chat, CHAT[0]), ("chat explain", chat, CHAT[1])]:
    m0 = metrics(); r = [fn(arg) for _ in range(3)]; m1 = metrics()
    print(f"1 client  {label:22s} {S.median(n / t for n, t in r):6.1f} tok/s   tokens/cycle {tpc(m0, m1):.2f}", flush=True)
for C in (2, 4, 8):
    m0 = metrics(); t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(C) as ex: r = list(ex.map(lambda i: chat(CHAT[i % 4]), range(C)))
    wall = time.perf_counter() - t0; m1 = metrics()
    print(f"{C} clients chat, aggregate {sum(n for n, _ in r) / wall:6.1f} tok/s  (per client {S.median(n / t for n, t in r):5.1f})   tokens/cycle {tpc(m0, m1):.2f}", flush=True)
