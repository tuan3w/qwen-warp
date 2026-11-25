"""Shared by the Inkling tests: the official config, the W4 checkpoint's config, a source that has only a config (the
real graph without a 15 GB checkpoint) and the real-shape graphs."""

import json
from pathlib import Path

from warpc import frontend
from warpc.ir import Graph
from warpc.sources import ModelConfig

DATA = Path(__file__).resolve().parent / "data"
# the official config.json of Inkling-2-Mini (modelopt NVFP4 quantisation_config) and the W4G128 one the checkpoint
# directory carries (what warpc/tools/inkling_w4_config.py writes)
INKLING_OFFICIAL = json.loads((DATA / "inkling2_mini_config.json").read_text())
INKLING_W4G128 = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "experts": {
            "targets": ["re:.*mlp\\.experts\\.(gate|up|down)_proj$"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 128},
            "input_activations": None,
        }
    },
    "ignore": [],
    "warpc_layout": "stacked-experts",
}


def inkling_w4(**text) -> dict:
    """The config.json of the W4 checkpoint directory, `text` patched into its text_config."""
    return {
        **INKLING_OFFICIAL,
        "text_config": {**INKLING_OFFICIAL["text_config"], **text},
        "quantization_config": INKLING_W4G128,
    }


class InklingShapes:
    """A source that only has a config: every tensor exists and the experts are W4G128."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> str | None:
        return "W4G128" if ".mlp.experts." in name else None


def real_graph(*, ctx_max: int = 4096, rows_max: int = 8, **text) -> Graph:
    """The graph of the real Inkling-2-Mini config (`text` patches its text_config)."""
    source = InklingShapes(ModelConfig.from_hf(inkling_w4(**text)))
    return frontend.build(source, ctx_max=ctx_max, rows_max=rows_max)


class DictSource:
    """A source of named tensors (and W4G128 weights as {"packed", "scale"} dicts) for hand-built kernel-test graphs; it
    has no config."""

    config = None

    def __init__(self, tensors: dict):
        self.tensors = tensors

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str, dtype=None):
        import torch

        return self.tensors[name].to(torch.float32 if dtype == torch.float32 else torch.bfloat16)

    def quant(self, name: str) -> str | None:
        return "W4G128" if isinstance(self.tensors[name], dict) else None

    def raw(self, name: str) -> dict:
        return self.tensors[name]


# A tiny Inkling-2 with the real widths (hidden 2304, 24 / 8 heads of 128, expert width 1152, gate rank 64: what the
# kernels are written for) in 8 layers: loop layers 2-5 run three times (16 visits), blocks of 2 layers, a window of 16
# that 24+ positions wrap, relative extent 16 and log scaling from position 8 on the full layers, 16 experts.
TINY_TEXT = {
    "num_hidden_layers": 8,
    "vocab_size": 2048,
    "unpadded_vocab_size": 2000,
    "local_layer_ids": [0, 1, 3, 4, 7],
    "sliding_window_size": 16,
    "rel_extent": 16,
    "log_scaling_n_floor": 8,
    "log_scaling_alpha": 0.5,
    "n_layers_per_attn_res_block": 2,
    "n_routed_experts": 16,
    "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 64, "is_varloop": False},
}


class ConfigSource(DictSource):
    """A DictSource that has a model config."""

    def __init__(self, tensors: dict, config: ModelConfig):
        super().__init__(tensors)
        self.config = config


def _random_weight(d, gen):
    """A random weight of declaration `d` on the GPU with the scale that keeps the 16 visits' activations O(1): linears
    N(0, 1 / K), norm weights 1 +- 0.1, an unembedding wide enough that the greedy token is not a coin toss."""
    import torch

    def randn(*shape, std):
        return torch.randn(*shape, device="cuda", generator=gen) * std

    name = d.name.split(".", 1)[1] if d.name[0] == "L" and d.name[1].isdigit() else d.name
    if d.scheme == "w4g128":
        e, n, k = d.shape
        # nibbles are uniform: std 4.6 per code, so scale 1 / (4.6 sqrt(K)) gives a unit-gain matrix
        packed = torch.randint(-(2**31), 2**31 - 1, (e, n, k // 8), dtype=torch.int32, device="cuda", generator=gen)
        gain = (0.5 + torch.rand(e, n, k // 128, device="cuda", generator=gen)) * (0.25 if name == "e_down" else 1.0)
        return {"packed": packed, "scale": (gain / (4.6 * k**0.5)).to(torch.bfloat16)}
    if d.scheme == "f32":
        return randn(*d.shape, std=0.1 if name == "rbias" else 1.0)
    k = d.shape[-1]
    if name.endswith("norm") or name in ("qclip", "rclip"):
        w = 1 + randn(*d.shape, std=0.1)
    elif "sconv" in name or "static" in name:
        w = randn(*d.shape, std=0.3)
    elif "taps" in name:
        w = randn(*d.shape, std=0.02)
    elif name == "relproj":
        w = randn(*d.shape, std=0.5)
    elif name == "embed":
        w = randn(*d.shape, std=1.0)
    elif name == "unembed":
        w = randn(*d.shape, std=3 / k**0.5)
    elif name.endswith((".forget", ".gain")):
        w = randn(*d.shape, std=0.15)
    elif name == "router":
        w = randn(*d.shape, std=2 / k**0.5)
    else:
        w = randn(*d.shape, std=1 / k**0.5)
    return w.to(torch.bfloat16)


def random_inkling(seed: int = 0, **text) -> ConfigSource:
    """The source of a random TINY_TEXT model (`text` patched in), weights on the GPU."""
    import torch

    config = ModelConfig.from_hf(inkling_w4(**{**TINY_TEXT, **text}))
    graph = frontend.build(InklingShapes(config), ctx_max=64, rows_max=8)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return ConfigSource({d.source: _random_weight(d, gen) for d in graph.weights.values()}, config)
