#!/usr/bin/env python
"""qwen-warp server: vLLM 0.30's own OpenAI frontend (run_server) on top of the swift-mk CUDA-graph decode engine (RTX 3090, Swift-1.5-Qwen3.8-27B INT4).

  python server/serve.py --port 8000                          # default: bf16 KV, 20k context, MTP k=3 + prompt lookup, one request at a time
  python server/serve.py --profile long --port 8000           # int4 KV + host-pinned tables: 196k context (up to 262144 with --max-model-len 262144)
  python server/serve.py --profile multiuser --port 8000      # 5 requests decoded together (--max-num-seqs 5 --host-tables --vision device), ~14k context each
  python server/serve.py --kv-cache-dtype fp8 --max-model-len 53248 --num-speculative-tokens 3

Every stock `vllm serve` frontend flag works (--served-model-name, --host/--port, --api-key, --chat-template, --reasoning-parser, --tool-call-parser, ...).
Engine flags that map onto the engine (read from vLLM's config in smk_core_client.engine_cfg): --max-model-len, --kv-cache-dtype {auto,bfloat16 -> bf16 | fp8 |
int4_per_token_head}, --num-speculative-tokens (k, 0..3; sugar for --speculative-config '{"method":"mtp","num_speculative_tokens":k}'), --max-num-seqs (1 = one request at a
time, the default; 2..8 = that many requests decoded together, server/smk_batch.py) and the engine's own --profile / --host-tables / --prompt-lookup / --stream-cycles / --snapshots /
--vision / --max-image-pixels.  Defaults for this model: reasoning parser qwen3, tool parser qwen3_coder + auto tool choice, image input on (at most 4 images per prompt, 2 megapixels
each; video is refused).
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
MULTIUSER_SEQS = 5  # --profile multiuser: 5 slots = DECODE_CAP (server/smk_batch.py), 1.36x the one-at-a-time server on a mixed text + image workload (docs/PERFORMANCE.md section 7)
MAX_IMAGE_PIXELS = 2097152  # 2 megapixels = 8192 patches of 16 x 16 = 2048 placeholder tokens per image (1920 x 1080 keeps its size)
MAX_IMAGES = 4  # default --limit-mm-per-prompt image count


def parse_args(argv=None):
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser(description=__doc__, formatter_class=__import__("argparse").RawDescriptionHelpFormatter))
    g = parser.add_argument_group("swift-mk")
    g.add_argument("--num-speculative-tokens", type=int, default=None, choices=[0, 1, 2, 3], help="MTP drafts per cycle (0 = plain decode; default 3 unless --speculative-config is given)")
    g.add_argument("--stream-cycles", type=int, default=2, help="speculative cycles per host sync = streaming granularity (one cycle ~ 24 ms, 1-4 tokens)")
    g.add_argument("--snapshots", type=int, default=16, help="DeltaNet state snapshots kept for multi-turn prefix reuse, in total over the device lineage and the host prefix cache (151 MB pinned host RAM each, 183 MB with the 4-bit KV cache; 0 = off)")
    g.add_argument("--system-checkpoint", default="on", choices=["on", "off"], help="one extra state snapshot per long prompt at the end of its system prompt (the first chat-turn boundary after 1,024 tokens), so parallel agents that share a system prompt and differ after it resume there; costs one 183 MB snapshot per conversation")
    g.add_argument("--prefix-cache-gb", type=float, default=6.0, help="pinned host RAM budget for the KV rows of OTHER conversations (spilled from the GPU when a request does not continue the cached one, restored when one does; agents interleave a main conversation, helper calls and subagents); ~21 KB per token with the 4-bit KV cache, 37 KB with fp8, 70 KB with bf16; 0 = one cached conversation only")
    g.add_argument("--prompt-lookup", default="auto", choices=["0", "1", "auto"], help="K11: MTP chain + prompt-lookup (n-gram) tail, verify up to 8 rows per cycle (needs k = 3; +0.56 GiB for the 9-slot DeltaNet ring; "
                   "auto (default) = on iff --max-model-len leaves 256 MiB of the free GPU memory next to it, e.g. up to 23k with bf16 KV, 42k with fp8 KV, any length with int4 KV + --host-tables); "
                   "the default --max-model-len is 20480 (28672 with --prompt-lookup 0)")
    g.add_argument("--host-tables", action="store_true", help="keep the bf16 embedding + rescoring lm_head (4.7 GiB) in pinned host memory (bit-identical decode, ~free): VRAM for ~190k more KV tokens with int4 KV")
    g.add_argument("--sort-tool-schema", default="on", choices=["on", "off"], help="render the keys of every tool's function.parameters in sorted order (properties, required, type): the model returns a well-formed tool call "
                   "more often than with the type-first order most clients send (server/tool_schema.py); off = render the schemas exactly as sent")
    g.add_argument("--profile", default="default", choices=["default", "long", "multiuser"], help="long = --kv-cache-dtype int4_per_token_head --host-tables and a default --max-model-len of 196608; "
                   f"multiuser = --max-num-seqs {MULTIUSER_SEQS} --host-tables --vision device (several users at once: bf16 KV, the longest --max-model-len per sequence that the free memory "
                   "holds, at most 28,672); explicit flags win")
    g.add_argument("--drafter", default="mtp", choices=["mtp", "dflash2"], help="mtp (default): MTP chain + prompt-lookup tail.  dflash2 (experimental): greedy requests without logit processors are decoded by the in-engine DFlash2 block drafter "
                   "(7 drafts per cycle, W4A16 checkpoint): +50% decode tok/s on code, +38% on math, -4% on prose and +34% over 8 chat prompts with bf16 KV (docs/PERFORMANCE.md section 5); an answer that copies a long stretch of "
                   "its prompt can be slower than with the prompt-lookup tail.  Sampled requests, requests with logit processors and requests whose drafter window is not on the GPU (resumed from the host prefix cache) use the MTP path.  "
                   "Needs the drafter checkpoint (--dflash-path), ~1.2 GiB + 20 KiB per context token of GPU memory on top of the MTP engine (needs --host-tables for any useful length; the --profile long length does not fit)")
    g.add_argument("--dflash-path", default=None, help="directory of the DFlash2 W4A16 checkpoint (compressed-tensors; model.safetensors) for --drafter dflash2; default $DFLASH_W4 or ~/llm-lab/models/dflash2-w4a16")
    g.add_argument("--dflash-codebooks", default="host", choices=["device", "host"], help="where the drafter's two 121 MiB candidate-selector codebooks live: host (default) = pinned memory, gathered by row over PCIe (same decode speed, same ids, 244 MiB less VRAM); device = GPU memory")
    g.add_argument("--vision", default=None, choices=["auto", "device", "host", "off"], help="image input with the checkpoint's vision tower.  device = its 879 MiB of bf16 weights on the GPU "
                   "(fastest encode: 16 ms at 448x448, 65 ms at 1024x768); host = the 27 blocks' weights stay in pinned host memory and are streamed per image (153 MiB on the GPU, +51 ms at "
                   "448x448, +9 ms at 1024x768, level at 1920x1080); auto (default) = device iff it leaves 256 MiB free at --max-model-len, else host, else off (the start log warns and an "
                   "image request gets a 400 naming the settings that would fit; without --max-model-len, --max-num-seqs > 1 picks a length that keeps image input on); off = text only, "
                   "the GPU memory and behaviour of 0.7.0 (an image request gets a 400)")
    g.add_argument("--max-image-pixels", type=int, default=MAX_IMAGE_PIXELS, help="largest image after the processor's resize, in pixels (images are scaled down to fit, aspect kept; "
                   f"256 pixels per ViT patch, 1,024 per placeholder token; default {MAX_IMAGE_PIXELS:,} = 2,048 tokens); the ViT's activation memory grows with it")
    parser.set_defaults(model=DEFAULT_MODEL, served_model_name=["swift"], reasoning_parser="qwen3", tool_call_parser="qwen3_coder", enable_auto_tool_choice=True,
                        disable_log_stats=True, dtype="bfloat16", max_num_seqs=None, enable_prompt_tokens_details=True)
    args = parser.parse_args(argv)
    validate_parsed_serve_args(args)
    if args.profile == "multiuser":
        args.host_tables = True
        if args.max_num_seqs is None:
            args.max_num_seqs = MULTIUSER_SEQS
        if args.vision is None:
            args.vision = "device"
    if args.max_num_seqs is None:
        args.max_num_seqs = 1
    if args.vision is None:
        args.vision = "auto"
    if args.vision == "off":
        args.language_model_only = True
    else:  # images on: at most MAX_IMAGES per prompt unless --limit-mm-per-prompt says otherwise, video always refused; the processor resizes to --max-image-pixels
        if args.mm_processor_cache_type == "shm":
            raise SystemExit("--mm-processor-cache-type shm is not supported by the swift-mk engine (use the default lru)")
        lim = dict(args.limit_mm_per_prompt or {})
        lim.setdefault("image", MAX_IMAGES)
        lim["video"] = 0
        args.limit_mm_per_prompt = lim
        mk = dict(args.mm_processor_kwargs or {})
        mk.setdefault("size", {"shortest_edge": 65536, "longest_edge": args.max_image_pixels})
        args.mm_processor_kwargs = mk
    if args.profile == "long":
        if args.kv_cache_dtype in (None, "auto"):
            args.kv_cache_dtype = "int4_per_token_head"
        args.host_tables = True
        if args.max_model_len is None and args.max_num_seqs == 1:
            args.max_model_len = LONG_MAX_MODEL_LEN
    if args.max_model_len is None and args.max_num_seqs > 1:  # S sequence slots, each with its own KV cache: the longest length per sequence the free GPU memory holds (capped at the profile's)
        import smk_core_client as cc

        # image input stays on (unless --vision off): room for the ViT with its weights in host memory (156 MiB at the default --max-image-pixels, under 1,000 tokens per slot), on
        # the GPU for --vision device; --vision auto still puts them on the GPU where the profile's cap leaves room for that
        vis = 0.0 if args.vision == "off" else cc.vision_mib("device" if args.vision == "device" else "host", args.max_image_pixels)
        cap = LONG_MAX_MODEL_LEN if args.profile == "long" else DEFAULT_MAX_MODEL_LEN
        fit = cc.batch_max_len(cc.free_gpu_mib(), cc.KV_DTYPES.get(args.kv_cache_dtype or "auto", 0), bool(args.host_tables), args.max_num_seqs, cc.AUTO_TAIL_HEADROOM_MIB, vis)
        args.max_model_len = max(1024, min(cap, fit))
    if args.max_model_len is None:
        args.max_model_len = DEFAULT_MAX_MODEL_LEN if args.prompt_lookup == "0" else DEFAULT_MAX_MODEL_LEN_PLD
    if args.num_speculative_tokens is None and args.speculative_config is None:
        args.num_speculative_tokens = 3
    if args.num_speculative_tokens:  # the engine reads k from the stock config; the draft model is the checkpoint's own MTP head
        args.speculative_config = {"method": "mtp", "num_speculative_tokens": args.num_speculative_tokens}
    # vLLM >= 0.32 hands the model raw uint8 pixels and normalises on the device (mm_device_do_normalize, on by default); the engine's ViT (engine/vision.py) takes the frontend's
    # rescaled and normalised floats, as vLLM 0.30 delivered them, so the frontend keeps normalising (a flag value of true would feed it 0..255 and the model would misread images).
    args.mm_device_do_normalize = False
    return args


def main(argv=None):
    import uvloop

    args = parse_args(argv)
    import smk_core_client

    smk_core_client.install(stream_cycles=args.stream_cycles, snapshots=args.snapshots, prefix_cache_gb=args.prefix_cache_gb, system_checkpoint=args.system_checkpoint == "on", prompt_lookup={"0": False, "1": True}.get(args.prompt_lookup, "auto"), host_tables=args.host_tables, sort_tool_schema=args.sort_tool_schema == "on",
                            drafter=args.drafter, dflash_path=args.dflash_path, dflash_codebooks=args.dflash_codebooks, vision=args.vision, max_image_pixels=args.max_image_pixels)
    from vllm.entrypoints.launchers.api_server.entry import run_server

    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
