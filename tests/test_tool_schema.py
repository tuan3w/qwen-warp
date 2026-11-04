#!/usr/bin/env python
"""Unit test of server/tool_schema.py (no GPU, no checkpoint, no vLLM): the canonical key order of the tool schemas the chat template renders.

  python tests/test_tool_schema.py

sort_tool_parameters sorts the top-level keys of function.parameters and nothing else: idempotent, input not mutated, list order / nested objects / values (unicode, numbers)
/ other tool fields untouched, odd inputs pass through.  install() wraps vllm.renderers.hf.safe_apply_chat_template (checked against a stand-in module).
"""
import copy
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import tool_schema  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""), flush=True)
    if not ok:
        FAILS.append(name)


def dump(x):
    return json.dumps(x, ensure_ascii=False)  # key order matters here: no sort_keys


TOOL = {"type": "function", "function": {"name": "book", "description": "Réserver \u00e9 vé \u2603", "parameters": {
    "type": "object", "description": "outer", "required": ["to", "n"],
    "properties": {"to": {"type": "string", "enum": ["Hà Nội", "Tokyo", "a"]}, "n": {"type": "integer", "minimum": 1, "maximum": 1e3, "default": 1.0}, "ex": {"type": "object", "properties": {"z": {}, "a": {}}}},
    "$defs": {"b": {"type": "string"}, "a": {"type": "string"}}}}}
TOOLS = [TOOL, {"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}]

out = tool_schema.sort_tool_parameters(copy.deepcopy(TOOLS))
p = out[0]["function"]["parameters"]
check("top-level parameters keys sorted", list(p) == sorted(TOOL["function"]["parameters"]) and list(p)[-1] == "type", list(p))
check("equal as JSON values", out == TOOLS)
check("nested key order untouched", list(p["properties"]) == ["to", "n", "ex"] and list(p["properties"]["to"]) == ["type", "enum"] and list(p["properties"]["ex"]["properties"]) == ["z", "a"] and list(p["$defs"]) == ["b", "a"])
check("list order untouched (tools, required, enum)", [t["function"]["name"] for t in out] == ["book", "ping"] and p["required"] == ["to", "n"] and p["properties"]["to"]["enum"] == ["Hà Nội", "Tokyo", "a"])
check("outer tool / function key order untouched", list(out[0]) == ["type", "function"] and list(out[0]["function"]) == ["name", "description", "parameters"])
check("unicode and numbers untouched", p["properties"]["n"]["maximum"] == 1e3 and p["properties"]["n"]["default"] == 1.0 and type(p["properties"]["n"]["default"]) is float
      and out[0]["function"]["description"] == TOOL["function"]["description"] and dump(p["properties"]) == dump(TOOL["function"]["parameters"]["properties"]))
check("second tool sorted too", list(out[1]["function"]["parameters"]) == ["properties", "type"])
check("idempotent", dump(tool_schema.sort_tool_parameters(out)) == dump(out))
src = copy.deepcopy(TOOLS)
before = dump(src)
res = tool_schema.sort_tool_parameters(src)
check("input not mutated", dump(src) == before and res is not src and res[0] is not src[0])
check("different key orders of the same schema give the identical canonical form",
      dump(tool_schema.sort_tool_parameters([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}}]))
      == dump(tool_schema.sort_tool_parameters([{"type": "function", "function": {"name": "f", "parameters": {"required": ["a"], "type": "object", "properties": {"a": {"type": "string"}}}}}])))
odd = [None, 7, "x", {}, {"type": "function"}, {"type": "function", "function": None}, {"type": "function", "function": {"name": "f"}}, {"type": "function", "function": {"name": "g", "parameters": None}},
       {"type": "function", "function": {"name": "h", "parameters": ["not", "a", "dict"]}}]
check("odd entries pass through", tool_schema.sort_tool_parameters(odd) == odd)
check("None / empty list pass through", tool_schema.sort_tool_parameters(None) is None and tool_schema.sort_tool_parameters([]) == [])

# install(): wraps vllm.renderers.hf.safe_apply_chat_template; the wrapper hands the sorted tools (and everything else unchanged) to the original
seen = {}


def fake_safe_apply(model_config, tokenizer, conversation, *, tools=None, chat_template=None, tokenize=True, **kwargs):
    seen.update(args=(model_config, tokenizer, conversation), tools=tools, chat_template=chat_template, tokenize=tokenize, kwargs=kwargs)
    return "prompt"


hf = types.ModuleType("vllm.renderers.hf")
hf.safe_apply_chat_template = fake_safe_apply
for name, mod in (("vllm", types.ModuleType("vllm")), ("vllm.renderers", types.ModuleType("vllm.renderers")), ("vllm.renderers.hf", hf)):
    sys.modules[name] = mod
sys.modules["vllm"].renderers = sys.modules["vllm.renderers"]
sys.modules["vllm.renderers"].hf = hf
tool_schema.install()
ret = hf.safe_apply_chat_template("cfg", "tok", ["msg"], tools=copy.deepcopy(TOOLS), chat_template="tpl", tokenize=False, add_generation_prompt=True)
check("install: wrapper returns the original's result", ret == "prompt" and hf.safe_apply_chat_template is not fake_safe_apply and hf.safe_apply_chat_template.__name__ == "fake_safe_apply")
check("install: sorted tools + untouched other arguments", list(seen["tools"][0]["function"]["parameters"])[-1] == "type" and seen["args"] == ("cfg", "tok", ["msg"]) and seen["chat_template"] == "tpl"
      and seen["tokenize"] is False and seen["kwargs"] == {"add_generation_prompt": True})
hf.safe_apply_chat_template("cfg", "tok", ["msg"])
check("install: no tools stays None", seen["tools"] is None)

print("TOOL SCHEMA " + ("OK" if not FAILS else f"FAILED: {FAILS}"))
sys.exit(1 if FAILS else 0)
