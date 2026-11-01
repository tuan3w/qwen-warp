#!/usr/bin/env python3
"""compare.py <a.json> <b.json> <out.json>: greedy-continuation agreement of two metric results (control a, candidate b), 0..1.

value = mean over the prompts both results have of (longest common prefix of the greedy token streams) / (stream length): dev = first 128 tokens of the k=3 greedy run,
held = first 64 tokens of its greedy k=3 run.  A rounding-level change flips a near-tie now and then (prose), a wrong kernel diverges within a few tokens;
the gate accepts value >= min(compare_min, control-vs-control - compare_margin)."""
import json
import sys


def lcp(x, y):
    n = 0
    for u, v in zip(x, y):
        if u != v:
            break
        n += 1
    return n


def main():
    a, b, out = (json.load(open(p)) if i < 2 else p for i, p in enumerate(sys.argv[1:4]))
    if b.get("kind") == "profile":
        json.dump({"value": 1.0, "detail": "profile run"}, open(out, "w"))
        return
    ta, tb = a.get("tokens", {}), b.get("tokens", {})
    names = [n for n in ta if n in tb]
    if not names:
        json.dump({"value": 0.0, "detail": "no common prompts"}, open(out, "w"))
        return
    per = {n: lcp(ta[n], tb[n]) / max(1, min(len(ta[n]), len(tb[n]))) for n in names}
    json.dump({"value": sum(per.values()) / len(per), "per_prompt": per}, open(out, "w"))


if __name__ == "__main__":
    main()
