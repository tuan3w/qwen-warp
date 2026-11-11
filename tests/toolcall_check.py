"""Structural validator for Qwen-format tool calls (<tool_call><function=name><parameter=key>value</parameter></function></tool_call>) against the tool's JSON schema.

Used by tests/test_dspine.py on real generations; tests/test_toolcall_validator.py feeds it good and broken calls so the validator itself is tested (a validator that accepts everything makes
the tool-call check worthless).  Checks, in order: a complete call block exists; the function is declared; every required argument is present and none is unknown; every value parses as the
declared type (integer, number, boolean, array / object as JSON); then the values the request implies (`expect`: eq / contains, case-insensitive for contains).
"""
import json
import re

FUNC = re.compile(
    r"<tool_call>\s*<function=([\w.\-]+)>(.*?)</function>\s*</tool_call>", re.S
)
PARAM = re.compile(r"<parameter=([\w.\-]+)>\n?(.*?)\n?</parameter>", re.S)


def coerce(value, typ):
    """a <parameter> body -> a Python value of the declared JSON-schema type (raises ValueError when it is not one)."""
    v = value.strip()
    if typ == "string":
        return value.strip("\n")
    if typ == "integer":
        return int(v)
    if typ == "number":
        return float(v)
    if typ == "boolean":
        if v.lower() not in ("true", "false"):
            raise ValueError(f"not a boolean: {v!r}")
        return v.lower() == "true"
    return json.loads(v)  # array / object


def validate_tool_call(p, text):
    """-> (ok, detail).  Structural validity against the schema, then the values the request implies."""
    calls = [(n, dict(PARAM.findall(b))) for n, b in FUNC.findall(text)]
    if not calls:
        return (
            False,
            f"no well-formed <tool_call><function=..> block in: {text[:300]!r}",
        )
    name, raw = calls[0]
    fn = {t["function"]["name"]: t["function"] for t in p["tools"]}
    if name not in fn:
        return False, f"unknown function {name!r} (declared: {sorted(fn)})"
    schema = fn[name]["parameters"]
    props, req = schema["properties"], schema.get("required", [])
    miss = [k for k in req if k not in raw]
    extra = [k for k in raw if k not in props]
    if miss or extra:
        return False, f"{name}: missing required {miss}, unknown arguments {extra}"
    args = {}
    for k, v in raw.items():
        try:
            args[k] = coerce(v, props[k].get("type", "string"))
        except (ValueError, json.JSONDecodeError) as exc:
            return (
                False,
                f"{name}.{k}: value {v!r} is not a valid {props[k].get('type')}: {exc}",
            )
    exp = p["expect"]
    if name != exp["name"]:
        return False, f"called {name!r}, expected {exp['name']!r}"
    for k, rule in exp["args"].items():
        if k not in args:
            return False, f"argument {k!r} missing, the request implies it"
        if "eq" in rule and args[k] != rule["eq"]:
            return False, f"{k} = {args[k]!r}, expected {rule['eq']!r}"
        if "contains" in rule and rule["contains"].lower() not in str(args[k]).lower():
            return False, f"{k} = {args[k]!r} does not contain {rule['contains']!r}"
    return True, f"{name}({', '.join(f'{k}={v!r}' for k, v in args.items())})"
