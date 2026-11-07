"""Concurrent streams: decode rate per stream after its first chunk, summed (prefill excluded when all requests are admitted together)."""
import json, sys, time, statistics as S, urllib.request, concurrent.futures as cf
PORT, MODEL, C = sys.argv[1], sys.argv[2], int(sys.argv[3]); N = 300
PROMPTS = ["Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few test cases.",
           "Explain how a hash table handles collisions, and when you would choose open addressing over chaining.",
           "Write a SQL query to find the top 3 customers by total order value in each country, given tables customers(id, name, country) and orders(id, customer_id, total).",
           "Summarize the main causes of the French Revolution in a few paragraphs."]
def run(i):
    body = dict(model=MODEL, messages=[{"role": "user", "content": PROMPTS[i % 4]}], max_tokens=N, temperature=0.0, ignore_eos=True, stream=True, stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    first = last = None; usage = None; n_first = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
            d = json.loads(line[6:]); now = time.perf_counter()
            if d.get("usage"): usage = d["usage"]["completion_tokens"]
            if d.get("choices") and (d["choices"][0]["delta"].get("content") or d["choices"][0]["delta"].get("reasoning")):
                if first is None: first = now; n_first = 1
                last = now
    return first, last, usage - n_first
for rep in range(3):
    with cf.ThreadPoolExecutor(C) as ex: res = list(ex.map(run, range(C)))
    t0 = max(f for f, _, _ in res); t1 = min(l for _, l, _ in res)         # window where all streams are decoding
    per = [(n / (l - f)) for f, l, n in res]
    agg = sum(n for _, _, n in res) / (max(l for _, l, _ in res) - min(f for f, _, _ in res))
    print(f"C={C} rep{rep}: per-stream decode {S.median(per):6.1f} tok/s, sum of rates {sum(per):6.1f}, aggregate over the whole decode span {agg:6.1f} tok/s")
