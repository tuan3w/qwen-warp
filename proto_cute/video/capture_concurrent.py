"""Record N simultaneous streamed chat completions with the arrival time of every chunk: capture_concurrent.py <port> <label>.
Writes data/<label>.json = {"label", "streams": [{"prompt", "chunks": [[t_since_common_start, "text delta"], ...], "tokens"}], "t_end", "tokens"}
(keeps the best of 2 runs by aggregate tok/s, after one warm run)."""

import json, sys, threading, time, urllib.request

PORT, LABEL = sys.argv[1], sys.argv[2]
PROMPTS = [
    "Write a Python function that merges overlapping intervals in a list of [start, end] pairs, then a few pytest test cases for it. Keep the explanation short.",
    "Explain how a CPU cache works to a new programmer, in three short paragraphs.",
    "Write a SQL query that returns the top 3 customers by total order value per country, with a short explanation.",
    "A train leaves at 9:40 and travels 210 km at 84 km/h. When does it arrive? Show the steps briefly.",
]


def one(prompt, t0, out):
    body = dict(
        model="swift",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": True},
        chat_template_kwargs={"enable_thinking": False},
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks, usage = [], None
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"):
                continue
            d = json.loads(line[6:])
            now = time.perf_counter() - t0
            if d.get("usage"):
                usage = d["usage"]["completion_tokens"]
            if d.get("choices"):
                delta = d["choices"][0]["delta"]
                txt = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if txt:
                    chunks.append([round(now, 4), txt])
    out.update(prompt=prompt, chunks=chunks, tokens=usage)


def run():
    outs = [{} for _ in PROMPTS]
    t0 = time.perf_counter()
    th = [threading.Thread(target=one, args=(p, t0, o)) for p, o in zip(PROMPTS, outs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    t_end = max(o["chunks"][-1][0] for o in outs)
    tokens = sum(o["tokens"] for o in outs)
    return dict(label=LABEL, streams=outs, t_end=t_end, tokens=tokens)


run()  # warm
best = max((run() for _ in range(2)), key=lambda r: r["tokens"] / r["t_end"])
json.dump(best, open(f"data/{LABEL}.json", "w"))
print(
    f"{LABEL}: {len(PROMPTS)} users, {best['tokens']} tokens in {best['t_end']:.2f} s = {best['tokens'] / best['t_end']:.0f} tok/s aggregate; per user:",
    ", ".join(
        f"{o['tokens']} tok / {o['chunks'][-1][0]:.2f} s" for o in best["streams"]
    ),
)
