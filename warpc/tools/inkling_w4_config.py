"""Write the config.json of an Inkling-2-Mini W4G128 checkpoint directory: the official config with the quantisation
the tensors actually have.

  uv run --package warpc python warpc/tools/inkling_w4_config.py OFFICIAL_CONFIG W4_DIR [--out DIR]

The directory holds the experts as stacked compressed-tensors pack-quantized int4 (group 128, symmetric, a leading
expert dimension) and no config.json. The official config declares modelopt NVFP4, which is false for these bytes, so
`quantization_config` is replaced; every other key is copied verbatim. `warpc_layout` marks the stacked rank-3 tensors,
which compressed-tensors itself does not define. No tensor is read or rewritten.
"""

import argparse
import json
from pathlib import Path

QUANTIZATION_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "experts": {
            "targets": ["re:.*mlp\\.experts\\.(gate|up|down)_proj$"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "strategy": "group",
                "group_size": 128,
            },
            "input_activations": None,
        }
    },
    "ignore": [],
    "warpc_layout": "stacked-experts",
}


def w4_config(official: dict) -> dict:
    """`official` with its quantization_config replaced; refuses a config that is not Inkling-2 loop."""
    if official.get("model_type") != "inkling2_loop":
        raise SystemExit(f"model_type {official.get('model_type')!r}: not an inkling2_loop config")
    return {
        **{k: v for k, v in official.items() if k != "quantization_config"},
        "quantization_config": QUANTIZATION_CONFIG,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("official_config", type=Path, help="the official Inkling-2-Mini config.json")
    p.add_argument("w4_dir", type=Path, help="the W4 checkpoint directory (read for its index)")
    p.add_argument("--out", type=Path, help="directory to write config.json into (default: W4_DIR)")
    args = p.parse_args()
    index = args.w4_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise SystemExit(f"{args.w4_dir}: no model.safetensors.index.json")
    names = json.loads(index.read_text())["weight_map"]
    packed = [n for n in names if n.endswith(".weight_packed")]
    if not packed or any(".experts." not in n for n in packed):
        raise SystemExit(f"{args.w4_dir}: expected packed int4 tensors for the experts only, found {len(packed)}")
    out = args.out or args.w4_dir
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(w4_config(json.loads(args.official_config.read_text())), indent=2) + "\n"
    (out / "config.json").write_text(text)
    print(f"wrote {out / 'config.json'}")


if __name__ == "__main__":
    main()
