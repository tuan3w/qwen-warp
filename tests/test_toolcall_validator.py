#!/usr/bin/env python
"""CPU test of the tool-call validator (tests/toolcall_check.py) that tests/test_dspine.py applies to real generations: good calls must pass and every kind of broken call must fail.

  python tests/test_toolcall_validator.py
A check that cannot fail checks nothing: each broken case below is a way a slightly-wrong engine produces a tool call (wrong type, missing / unknown argument, unknown function, truncated block,
invalid JSON for an array, not a boolean, the wrong value).  Cases are the prompts of tests/data/dspine_prompts.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from toolcall_check import validate_tool_call  # noqa: E402

P = {p["name"]: p for p in json.loads((HERE / "data" / "dspine_prompts.json").read_text())}


def call(name, **kv):
    return f"<tool_call>\n<function={name}>\n" + "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in kv.items()) + "</function>\n</tool_call>"


CASES = [
    ("good weather", "tool_weather", call("get_weather", city="Hanoi"), True),
    ("good, reasoning text before the call", "tool_weather", "I will look that up.\n\n" + call("get_weather", city="Hanoi, Vietnam"), True),
    ("wrong city", "tool_weather", call("get_weather", city="Paris"), False),
    ("missing required argument", "tool_weather", call("get_weather"), False),
    ("unknown argument", "tool_weather", call("get_weather", city="Hanoi", unit="c"), False),
    ("unknown function", "tool_weather", call("get_forecast", city="Hanoi"), False),
    ("no call at all", "tool_weather", "It is sunny in Hanoi.", False),
    ("truncated block", "tool_weather", "<tool_call>\n<function=get_weather>\n<parameter=city>\nHan", False),
    ("good search", "tool_search", call("search_files", query="rate limit", max_results=5, case_sensitive="false"), True),
    ("integer given as a word", "tool_search", call("search_files", query="rate limit", max_results="five"), False),
    ("wrong integer", "tool_search", call("search_files", query="rate limit", max_results=3), False),
    ("good Vietnamese event", "tool_vi_event", call("create_event", title="Họp nhóm", date="2026-11-12", attendees='["An", "Bình"]', all_day="true"), True),
    ("invalid JSON array", "tool_vi_event", call("create_event", title="Họp nhóm", date="2026-11-12", attendees="[An, Bình", all_day="true"), False),
    ("boolean given as 'yes'", "tool_vi_event", call("create_event", title="Họp nhóm", date="2026-11-12", all_day="yes"), False),
    ("date in another format", "tool_vi_event", call("create_event", title="Họp nhóm", date="12/11/2026", all_day="true"), False),
    ("title with the accents stripped", "tool_vi_event", call("create_event", title="Hop nhom", date="2026-11-12", all_day="true"), False),
]


def main():
    bad = 0
    for name, prompt, text, want in CASES:
        ok, detail = validate_tool_call(P[prompt], text)
        print("PASS" if ok == want else "FAIL", f"{name}: validator says {ok} ({detail[:80]})", flush=True)
        bad += ok != want
    assert not bad, f"{bad} validator cases wrong"
    print("OK")


if __name__ == "__main__":
    main()
