"""Canonical key order of the tool schemas the chat template renders.

The chat template prints every tool with `tool | tojson`, i.e. in the key order the client sent.  The model is sensitive to that order: with the common type-first
schema ({"type": "object", "properties": ..., "required": ...}) the int4-KV engine returns a well-formed tool call far less often than with the parameters dict
key-sorted (properties, required, type); see docs/ for the measurement.  `sort_tool_parameters` sorts the top-level keys of every `function.parameters` dict (a
JSON object's key order carries no meaning); `install()` applies it to every chat prompt vLLM's HF renderer builds.  Nothing else changes: list order, nested objects,
values, the other tool fields and the request itself are untouched.
"""
import functools


def sort_tool_parameters(tools):
    """Copy of `tools` (OpenAI `tools` list) with the top-level keys of each function.parameters dict sorted; None / non-dict entries pass through."""
    if not tools:
        return tools
    out = []
    for tool in tools:
        fn = tool.get("function") if isinstance(tool, dict) else None
        params = fn.get("parameters") if isinstance(fn, dict) else None
        if isinstance(params, dict):
            tool = {**tool, "function": {**fn, "parameters": {k: params[k] for k in sorted(params)}}}
        out.append(tool)
    return out


def install():
    """Make vLLM's HF chat renderer render sorted tool schemas (module-level wrapper of vllm.renderers.hf.safe_apply_chat_template; no vLLM file is modified)."""
    import vllm.renderers.hf as hf

    orig = hf.safe_apply_chat_template

    @functools.wraps(orig)
    def render(model_config, tokenizer, conversation, *, tools=None, **kwargs):
        return orig(model_config, tokenizer, conversation, tools=sort_tool_parameters(tools), **kwargs)

    hf.safe_apply_chat_template = render
