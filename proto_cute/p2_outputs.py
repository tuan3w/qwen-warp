import json, sys, urllib.request
PORT, OUT = sys.argv[1], sys.argv[2]
P = ["Write a Python function that merges overlapping intervals in a list of [start, end] pairs, with a few test cases.",
     "Explain how a hash table handles collisions, and when you would choose open addressing over chaining.",
     "Write a SQL query to find the top 3 customers by total order value in each country.",
     "A train leaves city A at 60 km/h and another leaves city B, 300 km away, at 90 km/h toward A, one hour later. When and where do they meet?",
     "Translate to French and then to German: 'The weather is lovely today, so we decided to take a long walk along the river.'",
     "Summarize the main causes of the French Revolution in a few paragraphs."]
res = []
for p in P:
    body = dict(model="swift", messages=[{"role": "user", "content": p}], max_tokens=200, temperature=0.0, chat_template_kwargs={"enable_thinking": False})
    d = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=300))
    res.append(d["choices"][0]["message"]["content"])
json.dump(res, open(OUT, "w"), indent=1); print("saved", OUT, [len(r) for r in res])
