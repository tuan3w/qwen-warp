#!/usr/bin/env python
"""qwen-warp server: vLLM 0.30's own OpenAI frontend (run_server) on top of the swift-mk CUDA-graph decode engine (RTX 3090, Swift-1.5-Qwen3.8-27B INT4).

  python server/serve.py --port 8000                          # default: bf16 KV, 20k context, MTP k=3 + prompt lookup
  python server/serve.py --profile long --port 8000           # int4 KV + host-pinned tables: 196k context (up to 262144 with --max-model-len 262144)
  python server/serve.py --kv-cache-dtype fp8 --max-model-len 53248 --num-speculative-tokens 3

Every stock `vllm serve` frontend flag works (--served-model-name, --host/--port, --api-key, --chat-template, --reasoning-parser, --tool-call-parser, ...).
Engine flags that map onto the engine (read from vLLM's config in smk_core_client.engine_cfg): --max-model-len, --kv-cache-dtype {auto,bfloat16 -> bf16 | fp8 |
int4_per_token_head}, --num-speculative-tokens (k, 0..3; sugar for --speculative-config '{"method":"mtp","num_speculative_tokens":k}'), and the engine's own
--profile / --host-tables / --prompt-lookup / --stream-cycles / --snapshots.  Defaults for this model: reasoning parser qwen3, tool parser qwen3_coder + auto tool choice, language-model only.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
# Defaults leave about 450 MiB of a 24 GB 3090 free after a request at the full length with the desktop holding 0.6 GiB (smk_core_client.vram_need_mib; tests/test_startup_configs.py)
DEFAULT_MAX_MODEL_LEN = 28672
DEFAULT_MAX_MODEL_LEN_PLD = 20480  # the prompt-lookup tail (default on) needs the 9-slot DeltaNet ring: +0.56 GiB
LONG_MAX_MODEL_LEN = 196608  # --profile long: with the prompt-lookup tail (262144 starts too)


def parse_args(argv=None):
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter))
    g = parser.add_argument_group("swift-mk")
    g.add_argument("--num-speculative-tokens", type=int, default=None, choices=[0, 1, 2, 3], help="MTP drafts per cycle (0 = plain decode; default 3 unless --speculative-config is given)")
    g.add_argument("--stream-cycles", type=int, default=2, help="speculative cycles per host sync = streaming granularity (one cycle ~ 24 ms, 1-4 tokens)")
    g.add_argument("--snapshots", type=int, default=4, help="DeltaNet state snapshots kept for multi-turn prefix reuse (151 MB pinned host RAM each, 183 MB with the 4-bit KV cache; 0 = off)")
    g.add_argument("--prompt-lookup", default="auto", choices=["0", "1", "auto"], help="K11: MTP chain + prompt-lookup (n-gram) tail, verify up to 8 rows per cycle (needs k = 3; +0.56 GiB for the 9-slot DeltaNet ring; "
                   "auto (default) = on iff --max-model-len leaves 256 MiB of the free GPU memory next to it, e.g. up to 23k with bf16 KV, 42k with fp8 KV, any length with int4 KV + --host-tables); "
                   "the default --max-model-len is 20480 (28672 with --prompt-lookup 0)")
    g.add_argument("--host-tables", action="store_true", help="keep the bf16 embedding + rescoring lm_head (4.7 GiB) in pinned host memory (bit-identical decode, ~free): VRAM for ~190k more KV tokens with int4 KV")
    g.add_argument("--sort-tool-schema", default="on", choices=["on", "off"], help="render the keys of every tool's function.parameters in sorted order (properties, required, type): the model returns a well-formed tool call "
                   "more often than with the type-first order most clients send (server/tool_schema.py); off = render the schemas exactly as sent")
    g.add_argument("--profile", default="default", choices=["default", "long"], help="long = --kv-cache-dtype int4_per_token_head --host-tables and a default --max-model-len of 196608 (explicit flags win)")
    parser.set_defaults(model=DEFAULT_MODEL, served_model_name=["swift"], reasoning_parser="qwen3", tool_call_parser="qwen3_coder", enable_auto_tool_choice=True,
                        language_model_only=True, disable_log_stats=True, dtype="bfloat16", max_num_seqs=1, enable_prompt_tokens_details=True)
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)
    if args.profile == "long":
        if args.kv_cache_dtype in (None, "auto"):
            args.kv_cache_dtype = "int4_per_token_head"
        args.host_tables = True
        if args.max_model_len is None:
            args.max_model_len = LONG_MAX_MODEL_LEN
    if args.max_model_len is None:
        args.max_model_len = DEFAULT_MAX_MODEL_LEN if args.prompt_lookup == "0" else DEFAULT_MAX_MODEL_LEN_PLD
    if args.num_speculative_tokens is None and args.speculative_config is None:
        args.num_speculative_tokens = 3
    if args.num_speculative_tokens:  # the engine reads k from the stock config; the draft model is the checkpoint's own MTP head
        args.speculative_config = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens}
    return args


def main(argv=None):
    import uvloop

    args = parse_args(argv)
    import smk_core_client

    smk_core_client.install(stream_cycles=args.stream_cycles, snapshots=args.snapshots, prompt_lookup={"0": False, "1": True}.get(args.prompt_lookup, "auto"), host_tables=args.host_tables, sort_tool_schema=args.sort_tool_schema == "on")
    from vllm.entrypoints.launchers.api_server.entry import run_server

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
