"""vLLM general plugin: with VLLM_SWIFT_MK=1, stock `vllm serve` runs the swift-mk engine instead of vLLM's own EngineCore (no vLLM file is modified).

  VLLM_SWIFT_MK=1 PYTHONPATH=server/plugin ~/llm-lab/venv/bin/python -m vllm.entrypoints.cli.main serve ~/models/swift-1.5-int4 --max-model-len 32768 \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --kv-cache-dtype auto --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice

The entry point is discovered through the dist-info directory next to this file (put the directory on PYTHONPATH; nothing is pip-installed).  Engine settings come from the stock
flags (--max-model-len, --kv-cache-dtype, --speculative-config); VLLM_SWIFT_MK_STREAM_CYCLES (default 2) and VLLM_SWIFT_MK_SNAPSHOTS (default 4) and VLLM_SWIFT_MK_PLD (K11 prompt-lookup tail: 0 / 1 / auto, default auto) are the swift-mk knobs.
Without VLLM_SWIFT_MK=1 this plugin does nothing.
"""
import os
import sys
from pathlib import Path


def register():
    if os.environ.get("VLLM_SWIFT_MK", "0") not in ("1", "true", "True"):
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import smk_core_client

    smk_core_client.install(stream_cycles=int(os.environ.get("VLLM_SWIFT_MK_STREAM_CYCLES", "2")), snapshots=int(os.environ.get("VLLM_SWIFT_MK_SNAPSHOTS", "4")),
                            prompt_lookup={"0": False, "1": True}.get(os.environ.get("VLLM_SWIFT_MK_PLD", "auto"), "auto"))
