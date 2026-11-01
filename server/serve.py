#!/usr/bin/env python
"""`vllm serve` for the swift-mk engine: vLLM 0.30's own OpenAI frontend (run_server) on top of the swift-mk CUDA-graph engine.

  ~/llm-lab/venv/bin/python server/serve.py --port 8000          # defaults below
  ~/llm-lab/venv/bin/python server/serve.py --model ~/models/swift-1.5-int4 --max-model-len 32768 --kv-cache-dtype fp8 --num-speculative-tokens 3

Every stock `vllm serve` frontend flag works (--served-model-name, --host/--port, --api-key, --chat-template, --reasoning-parser, --tool-call-parser, ...).
Engine flags that map onto swift-mk (read from vLLM's config in smk_core_client.engine_cfg): --max-model-len, --kv-cache-dtype {auto,bfloat16 -> bf16 | fp8},
--num-speculative-tokens (k, 0..3; sugar for --speculative-config '{"method":"mtp","num_speculative_tokens":k}'), and swift-mk's own --stream-cycles / --snapshots.  Defaults for this model: reasoning parser qwen3, tool parser qwen3_coder + auto tool choice, language-model only.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = os.path.expanduser("~/models/swift-1.5-int4")
DEFAULT_MAX_MODEL_LEN = 32768
DEFAULT_MAX_MODEL_LEN_PLD = 28672  # the K11 prompt-lookup tail (default on) needs the 9-slot DeltaNet ring: +0.62 GiB, measured limit with bf16 KV


def parse_args(argv=None):
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter))
    g = parser.add_argument_group("swift-mk")
    g.add_argument("--num-speculative-tokens", type=int, default=None, choices=[0, 1, 2, 3], help="MTP drafts per cycle (0 = plain decode; default 3 unless --speculative-config is given)")
    g.add_argument("--stream-cycles", type=int, default=2, help="speculative cycles per host sync = streaming granularity (one cycle ~ 24 ms, 1-4 tokens)")
    g.add_argument("--snapshots", type=int, default=4, help="DeltaNet state snapshots kept for multi-turn prefix reuse (151 MB pinned host RAM each; 0 = off)")
    g.add_argument("--prompt-lookup", default="auto", choices=["0", "1", "auto"], help="K11: MTP chain + prompt-lookup (n-gram) tail, verify up to 8 rows per cycle (needs k = 3; +0.62 GiB for the 9-slot DeltaNet ring: "
                   "--max-model-len <= 28672 with bf16 KV / 53248 with fp8 KV).  auto (default) = on iff --max-model-len fits; the default --max-model-len is 28672 (32768 with --prompt-lookup 0)")
    parser.set_defaults(model=DEFAULT_MODEL, served_model_name=["swift"], reasoning_parser="qwen3", tool_call_parser="qwen3_coder", enable_auto_tool_choice=True,
                        language_model_only=True, disable_log_stats=True, dtype="bfloat16", max_num_seqs=1, enable_prompt_tokens_details=True)
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)
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

    smk_core_client.install(stream_cycles=args.stream_cycles, snapshots=args.snapshots, prompt_lookup={"0": False, "1": True}.get(args.prompt_lookup, "auto"))
    from vllm.entrypoints.launchers.api_server.entry import run_server

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
