"""Held-out eval prompts (hand-written here; none is in the vocab corpus). Long ones are built from papers/ + sources/ notes
(the vocab corpus uses only vendor/ and topics/)."""
from pathlib import Path

REPO = Path(__file__).parents[2]

CODE = {
    "code-lru": "Implement an LRU cache class in Python with O(1) get and put, using only the standard library. Include type hints, a docstring, and a small pytest test module covering eviction order and capacity 1.",
    "code-c-list": "Write a C program that defines a singly linked list of ints, reverses the list in place, prints it before and after, and frees all memory. Comment each function briefly.",
    "code-sql": "Tables: orders(id, customer_id, total, created_at) and customers(id, name, country). Write a SQL query (PostgreSQL) that returns, for each country, the top 3 customers by total spend in 2024 together with their rank, using window functions. Then explain each clause.",
    "code-bugfix": "This Python binary search has a bug. Find it, explain why it fails, and give the corrected version with two test cases.\n\n```python\ndef bsearch(a, x):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x:\n            return mid\n        elif a[mid] < x:\n            lo = mid\n        else:\n            hi = mid - 1\n    return -1\n```",
    "code-explain": "Explain what this code does, line by line, and when you would use it in practice:\n\n```python\nimport functools, time\n\ndef retry(times=3, delay=0.5, exceptions=(Exception,)):\n    def deco(fn):\n        @functools.wraps(fn)\n        def wrapper(*args, **kwargs):\n            for attempt in range(times):\n                try:\n                    return fn(*args, **kwargs)\n                except exceptions as e:\n                    if attempt == times - 1:\n                        raise\n                    time.sleep(delay * 2 ** attempt)\n        return wrapper\n    return deco\n```",
    "code-script": "Write a Python command-line script using argparse that reads a log file where each line looks like '2024-05-01T12:00:03Z GET /api/items 200 37ms', and prints the count, mean, p50, p95 and p99 latency per endpoint as an aligned table. Handle malformed lines gracefully.",
}

MATH = {
    "math-train": "Two trains start 300 km apart and travel toward each other, one at 80 km/h and the other at 70 km/h. A bird flies back and forth between them at 120 km/h starting from the first train. How far does the bird fly before the trains meet? Solve it two different ways and explain.",
    "math-sqrt2": "Prove that the square root of 2 is irrational, then generalise the argument to show that the square root of any prime is irrational. Write the proofs carefully.",
    "math-series": "Find a closed form for the sum of 1/(k(k+1)(k+2)) for k from 1 to n, derive it step by step with partial fractions, and check it for n = 1, 2, 3.",
    "math-count": "How many distinct arrangements are there of the letters of the word MISSISSIPPI? Explain the reasoning, then compute how many of them have no two S's adjacent.",
    "math-cubic": "Solve x^3 - 6x^2 + 11x - 6 = 0. Show the factoring steps, verify each root by substitution, and explain how Vieta's formulas relate the roots to the coefficients.",
    "math-prob": "A bag contains 5 red and 3 blue balls. Two balls are drawn without replacement. Compute the probability that both are red, the probability that they have different colours, and the expected number of red balls drawn. Show every step.",
}

PROSE = {
    "prose-story": "Write a short story (about 400 words) about a retired cartographer who discovers that one of her old maps has started to change by itself.",
    "prose-email": "Draft a polite but firm email to a landlord who has not fixed a broken heater for three weeks in winter. Mention the dates, the earlier messages, and the next steps you will take.",
    "prose-explain": "Explain to a curious ten-year-old why the sky is blue and why sunsets are red. Use an analogy and keep it friendly.",
    "prose-travel": "I have five days in Lisbon in October, I like food markets, old tram rides and hiking but dislike crowds. Suggest an itinerary and some practical tips.",
    "prose-essay": "Write a thoughtful short essay on whether remote work makes teams more or less creative. Give balanced arguments and end with your own view.",
    "prose-dialogue": "Write a dialogue between a sceptical lighthouse keeper and a young sailor who claims to have seen a ship that sails above the water. Keep the voices distinct.",
}

TOOLS_JSON = """[
  {"name": "get_weather", "description": "Get the current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}},
  {"name": "create_event", "description": "Create a calendar event", "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"}, "duration_min": {"type": "integer"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "start"]}},
  {"name": "search_docs", "description": "Full-text search in the internal knowledge base", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}, "required": ["query"]}}
]"""

AGENT = {
    "agent-call": "You have these tools:\n" + TOOLS_JSON + "\n\nUser request: 'What is the weather in Osaka in fahrenheit, and schedule a 45 minute sync called Launch Review with ana@example.com and raj@example.com next Tuesday at 3pm UTC (today is 2025-03-12).' Respond with the JSON tool calls you would make, as a JSON array, followed by a one-paragraph explanation of your choices.",
    "agent-shell": "I am in a Linux shell in a git repository. Give me the exact shell commands, one per step with a short explanation each, to: find the 10 largest files tracked by git, list which authors changed src/ in the last 30 days sorted by number of commits, and create a gzipped tarball of the working tree excluding node_modules and .git.",
    "agent-extract": "Extract structured data from the text below and output a single JSON object with keys vendor, invoice_number, date (ISO), line_items (array of {description, qty, unit_price}), subtotal, tax, total.\n\nText: 'ACME Industrial Supplies Ltd. Invoice #A-20931 dated 4 March 2025. 12 x hex bolt M8 @ 0.35 each, 3 x torque wrench (digital) @ 89.90, 1 x freight @ 24.00. Subtotal 295.... tax 20% ... total due within 30 days.' Compute the missing amounts yourself and state any assumption.",
    "agent-plan": "You are a coding agent with tools read_file(path), run_tests(), and edit_file(path, patch). A failing test says 'AssertionError: expected 3 items, got 2' in tests/test_cart.py::test_remove_item. Write your plan and the first three tool calls you would make (as JSON), with your reasoning before each call.",
}


def _notes(dirname, n_files=None):
    return [(p.name, p.read_text(errors="ignore")) for p in sorted((REPO / dirname).glob("*.md"))][:n_files]


def long_prompts(tok):
    """Three long-context summarise/QA prompts of ~4K / ~6K / ~8K tokens from papers/ and sources/ notes."""
    def build(parts, n_tokens):
        text = "\n\n".join(f"=== {name} ===\n{body}" for name, body in parts)
        ids = tok(text, add_special_tokens=False)["input_ids"][:n_tokens]
        return tok.decode(ids)
    papers = _notes("papers")
    sources = _notes("sources")
    return {
        "long-4k-papers": build(papers[:12], 4000) + "\n\nSummarise the main claims of the notes above in a numbered list, naming the note each claim comes from.",
        "long-6k-sources": build(sources[::-1], 6000) + "\n\nWhich of the projects above are written in CUDA or C++, and which in Python? Answer as two bullet lists with one-line justifications taken from the text.",
        "long-8k-mixed": build(papers[12:] + sources[:12], 8000) + "\n\nWrite a brief report (about 300 words) comparing the approaches described above, and list three open questions they leave unresolved.",
    }
