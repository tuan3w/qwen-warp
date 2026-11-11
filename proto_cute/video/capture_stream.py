"""Record one streamed chat completion with the arrival time of every chunk: capture_stream.py <port> <label> [prompt-id].
Writes data/<label>.json = {"label", "prompt", "model_tokens", "chunks": [[t_seconds_since_request, "text delta"], ...], "t_first", "t_end", "tokens"}"""
import json, sys, time, urllib.request
PORT, LABEL = sys.argv[1], sys.argv[2]
PROMPT = ("Write a Python function that merges overlapping intervals in a list of [start, end] pairs, then a few pytest test cases for it. Keep the explanation short.")
def run():
    body = dict(model="swift", messages=[{"role": "user", "content": PROMPT}], max_tokens=700, temperature=0.0, stream=True,
                stream_options={"include_usage": True}, chat_template_kwargs={"enable_thinking": False})
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    chunks, usage = [], None; t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.startswith(b"data: [DONE]"): continue
            d = json.loads(line[6:]); now = time.perf_counter() - t0
            if d.get("usage"): usage = d["usage"]["completion_tokens"]
            if d.get("choices"):
                delta = d["choices"][0]["delta"]; txt = (delta.get("content") or "") + (delta.get("reasoning") or "")
                if txt: chunks.append([round(now, 4), txt])
    return chunks, usage
run()                                   # warm
best = None
for _ in range(3):
    chunks, usage = run()
    t_first, t_end = chunks[0][0], chunks[-1][0]
    rate = (usage - 1) / (t_end - t_first)
    if best is None or rate > best[0]: best = (rate, chunks, usage)
rate, chunks, usage = best
json.dump(dict(label=LABEL, prompt=PROMPT, chunks=chunks, tokens=usage, t_first=chunks[0][0], t_end=chunks[-1][0]), open(f"data/{LABEL}.json", "w"))
print(f"{LABEL}: {usage} tokens, first chunk {chunks[0][0]*1000:.0f} ms, end {chunks[-1][0]:.2f} s, decode {rate:.1f} tok/s, {len(chunks)} chunks")
