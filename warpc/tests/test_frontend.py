"""Frontend + reference interpreter vs transformers on tiny random models of every family, and the config guard.

uv run --package warpc python warpc/tests/test_frontend.py

WARPC_TEST_SWIFT27 (the Qwen3.5 27B W4G128 checkpoint directory) enables the real-checkpoint graph check (headers only).
"""

import contextlib
import inspect
import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralConfig,
    MistralForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import logging as transformers_logging

import warpc.sources
from harness import Skip, run
from inkling_common import INKLING_OFFICIAL, INKLING_W4G128, InklingShapes, inkling_w4
from warpc import frontend
from warpc.errors import UnsupportedModel
from warpc.ir import Graph, TType
from warpc.reference import _IMPL, Interpreter
from warpc.sources import ModelConfig

transformers_logging.disable_progress_bar()

SIZES = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 256,
}
MODELS = {
    "llama": lambda: LlamaForCausalLM(
        LlamaConfig(**SIZES, rope_parameters={"rope_type": "default", "rope_theta": 10000.0}, tie_word_embeddings=False)
    ),
    "mistral": lambda: MistralForCausalLM(
        MistralConfig(**SIZES, sliding_window=None, rope_parameters={"rope_type": "default", "rope_theta": 1e6})
    ),
    "qwen3": lambda: Qwen3ForCausalLM(
        Qwen3Config(
            **SIZES,
            head_dim=32,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0},
            tie_word_embeddings=True,
        )
    ),
    # 8 rotary pairs: wavelengths 6.3 (high band, unscaled), 20 and 63 (smoothed), 199 ... 19869 (divided by 8)
    "llama_rope_llama3": lambda: LlamaForCausalLM(
        LlamaConfig(
            **SIZES,
            rope_parameters={
                "rope_type": "llama3",
                "rope_theta": 10000.0,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 64,
            },
            tie_word_embeddings=False,
        )
    ),
}
LLAMA_3_1_8B = {  # the real Llama-3.1-8B config.json fields the guard and the rope read, with one layer
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 1,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 131072,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "hidden_act": "silu",
    "attention_bias": False,
    "mlp_bias": False,
    "tie_word_embeddings": False,
}
QWEN3_0_6B = {  # the real Qwen3-0.6B config.json fields the guard reads, with one layer
    "model_type": "qwen3",
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 1,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000,
    "rope_scaling": None,
    "hidden_act": "silu",
    "attention_bias": False,
    "use_sliding_window": False,
    "sliding_window": None,
    "tie_word_embeddings": True,
}


def tiny_model(name: str):
    """A tiny random fp32 model whose weights are exactly bf16-representable (the source hands out bf16)."""
    model = MODELS[name]()
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_((torch.randn_like(p) * (0.3 if "norm" in n else p.shape[-1] ** -0.5)).bfloat16().float())
    return model.eval()


class Quantized:
    """A source whose linears claim to be Q4_0 on disk (the scheme rule only reads quant())."""

    def __init__(self, source):
        self.source, self.config = source, source.config

    def __contains__(self, name):
        return name in self.source

    def get(self, name):
        return self.source.get(name)

    def quant(self, name):
        return "Q4_0" if name.endswith("proj.weight") else None


def check_family(family: str, model_name: str | None = None):
    model = tiny_model(model_name or family)
    ids = torch.randint(0, SIZES["vocab_size"], (24,), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model(ids[None]).logits[0]
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        source = warpc.sources.open(tmp)
        assert source.config.family == family
        graph = frontend.build(source, ctx_max=64)
        it = Interpreter(graph, source, dtype=torch.float32)
    full = it.run({"tokens": ids.int(), "pos": torch.arange(24, dtype=torch.int32)})["logits"]
    assert torch.allclose(full, expected, atol=2e-4, rtol=2e-4), (family, (full - expected).abs().max())
    it.reset()  # windows of any size through the KV cache equal the full pass
    parts, at = [], 0
    for w in (10, 1, 1, 4, 8):
        window = {"tokens": ids[at : at + w].int(), "pos": torch.arange(at, at + w, dtype=torch.int32)}
        parts.append(it.run(window)["logits"])
        at += w
    assert torch.allclose(torch.cat(parts), full, atol=1e-4, rtol=1e-4), family
    tokens = it.generate(ids[:5].tolist(), 6)
    assert tokens[0] == int(full[4].argmax()) and len(tokens) == 6, tokens


def test_llama():
    check_family("llama")


def test_mistral():
    check_family("mistral")


def test_qwen3():
    check_family("qwen3")


def test_llama3_rope_scaling():
    check_family("llama", "llama_rope_llama3")
    tiny = tiny_model("llama_rope_llama3")
    real = {k: v for k, v in LLAMA_3_1_8B.items() if k != "model_type"}
    for hf, want, bands in (
        (tiny.config.to_dict(), tiny.model.rotary_emb.inv_freq, (1, 2, 5)),
        (LLAMA_3_1_8B, ROPE_INIT_FUNCTIONS["llama3"](LlamaConfig(**real), "cpu")[0], (29, 6, 29)),
    ):
        config = ModelConfig.from_hf(hf)
        factors = torch.tensor(config.rope_freq_factors)
        high, low = int((factors == 1).sum()), int((factors == 8).sum())
        assert (high, len(factors) - high - low, low) == bands, factors
        # the factors are llama.cpp's divisors while transformers smooths the inverse frequency itself: measured, the
        # smoothed pairs are bit-equal or one fp32 ulp apart (tiny: 7 of 8 equal; Llama 3.1 8B: 61 of 64, 1.1e-7)
        got = torch.tensor(config.rope_inv_freq())
        assert torch.allclose(got, want, rtol=2e-7, atol=0), ((got - want) / want).abs().max()


def test_graph_shape():
    model = tiny_model("qwen3")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        source = warpc.sources.open(tmp)
        graph = frontend.build(source, ctx_max=64)
        assert set(graph.inputs) == {"tokens", "pos"} and set(graph.outputs) == {"hidden", "logits", "ids"}
        assert graph.weights["lm_head"].source == "model.embed_tokens.weight"  # tied
        assert {w.scheme for w in graph.weights.values()} == {"dense"}
        assert [s.shape for s in graph.states.values()] == [(2, 64, 2, 32)] * 3
        quantized = frontend.build(Quantized(source), ctx_max=64)
        schemes = {name: w.scheme for name, w in quantized.weights.items()}
        assert schemes["L0.q"] == schemes["L2.down"] == "q4_0" and schemes["embed"] == schemes["L0.ln1"] == "dense"
        assert schemes["lm_head"] == "dense"
        for bad in ("fp8", "k3v3"):
            try:
                frontend.build(source, ctx_max=64, kv=bad)
            except ValueError as e:
                assert bad in str(e)
            else:
                raise AssertionError(f"KV format {bad} accepted")


def test_quantized_kv_reference():
    model = tiny_model("llama")
    ids = torch.randint(0, SIZES["vocab_size"], (20,), generator=torch.Generator().manual_seed(2))
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        source = warpc.sources.open(tmp)
        inputs = {"tokens": ids.int(), "pos": torch.arange(20, dtype=torch.int32)}
        exact = Interpreter(frontend.build(source, ctx_max=32), source, dtype=torch.float32).run(inputs)["logits"]
        for kv in ("k8v8", "k8v8w", "k8v4w"):
            graph = frontend.build(source, ctx_max=32, kv=kv)
            assert {s.cache for s in graph.states.values()} == {kv}
            logits = Interpreter(graph, source, dtype=torch.float32).run(inputs)["logits"]
            err = (logits - exact).abs().max().item()
            # measured: k8v8 6.7e-3; the "w" formats 3.1e-4 (sink 4 + window 256 cover all 20 positions: exact rows)
            assert err < (1e-3 if kv.endswith("w") else 3e-2), (kv, err)


def test_missing_tensor():
    class Missing(Quantized):
        def __contains__(self, name):
            return name != "model.layers.1.mlp.up_proj.weight" and name in self.source

    model = tiny_model("llama")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        try:
            frontend.build(Missing(warpc.sources.open(tmp)), ctx_max=16)
        except UnsupportedModel as e:
            assert "model.layers.1.mlp.up_proj.weight" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint without a tensor built")


def test_head_choice():
    """An untied config reads lm_head.weight, and a checkpoint without it is an error naming the tensor."""
    model = tiny_model("llama")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        config = Path(tmp) / "config.json"
        hf = json.loads(config.read_text())
        hf.pop("tie_word_embeddings", None)
        config.write_text(json.dumps(hf))
        source = warpc.sources.open(tmp)
        assert not source.config.tie_embeddings
        assert frontend.build(source, ctx_max=16).weights["lm_head"].source == "lm_head.weight"

        class NoHead(Quantized):
            def __contains__(self, name):
                return name != "lm_head.weight" and name in self.source

        try:
            frontend.build(NoHead(source), ctx_max=16)
        except UnsupportedModel as e:
            assert "lm_head.weight" in str(e), str(e)
        else:
            raise AssertionError("an untied config without lm_head.weight built")


def test_real_config_accepted():
    for patch in (
        {},
        {"rope_scaling": {"rope_type": "default"}, "sliding_window": None},
        {"sliding_window": 4096, "use_sliding_window": False},  # carried but switched off: Qwen configs do this
    ):
        c = ModelConfig.from_hf({**QWEN3_0_6B, **patch})
        assert (c.family, c.head_dim, c.num_kv_heads, c.qk_norm, c.tie_embeddings) == ("qwen3", 128, 8, True, True)


def test_unsupported_settings_are_named():
    cases = {
        "missing rope scaling low_freq_factor; missing rope scaling high_freq_factor; missing rope scaling "
        "original_max_position_embeddings": {"rope_scaling": {"rope_type": "llama3", "factor": 8.0}},
        "rope scaling 'yarn'": {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6}},
        "rope scaling 'linear'": {"rope_scaling": {"type": "linear", "factor": 2.0}},
        "sliding_window=4096": {"sliding_window": 4096, "use_sliding_window": True},
        "attention_bias=True": {"attention_bias": True},
        "mlp_bias=True": {"mlp_bias": True},
        "hidden_act='gelu'": {"hidden_act": "gelu"},
        "model_type 'qwen2' (supported: llama, mistral, qwen3, qwen3_5": {"model_type": "qwen2"},
    }
    mistral_style = {k: v for k, v in QWEN3_0_6B.items() if k != "use_sliding_window"}
    for needle, hf in [(n, {**QWEN3_0_6B, **p}) for n, p in cases.items()] + [
        ("sliding_window=4096", {**mistral_style, "sliding_window": 4096}),  # no switch: the window is active
        ("sliding_window=4096", {**mistral_style, "sliding_window": 4096, "use_sliding_window": None}),
    ]:
        try:
            ModelConfig.from_hf(hf)
        except UnsupportedModel as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError(f"{needle}: accepted a config the graph cannot reproduce")
    try:
        ModelConfig.from_hf({**QWEN3_0_6B, "attention_bias": True, "hidden_act": "gelu"})
    except UnsupportedModel as e:
        assert "attention_bias=True; hidden_act='gelu'" in str(e), str(e)
    else:
        raise AssertionError("two unsupported settings accepted")


QWEN35 = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "layer_types": ["linear_attention"] * 3 + ["full_attention"],
    "full_attention_interval": 4,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [1, 1, 2],
    },
}


def tiny_qwen35(**overrides):
    """A tiny random Qwen3_5ForCausalLM (3 Gated DeltaNet layers, then 1 gated full-attention layer), fp32 with
    bf16-representable weights."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**{**QWEN35, **overrides})).float()
    with torch.no_grad():
        for n, p in model.named_parameters():
            scale = 0.3 if "norm" in n or n.endswith(("A_log", "dt_bias")) else p.shape[-1] ** -0.5
            p.copy_((torch.randn_like(p) * scale).bfloat16().float())
    return model.eval()


@contextlib.contextmanager
def hf_torch_paths(*, chunk_is_recurrent: bool):
    """transformers' own torch functions for the conv and the DeltaNet rule, not the fla / causal_conv1d kernels its hub
    wrappers pick when those packages are installed (GPU only, other numerics); with `chunk_is_recurrent`, the chunked
    prefill algorithm (about 3e-4 per layer from the recurrent rule in fp32) is the recurrent rule too."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    names = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    saved = {n: getattr(hf, n) for n in names}
    plain = {n: inspect.unwrap(f) for n, f in saved.items()}
    if chunk_is_recurrent:
        plain["torch_chunk_gated_delta_rule"] = plain["torch_recurrent_gated_delta_rule"]
    for n, f in plain.items():
        setattr(hf, n, f)
    try:
        yield
    finally:
        for n, f in saved.items():
            setattr(hf, n, f)


def hf_stepwise(model, ids: torch.Tensor) -> torch.Tensor:
    """transformers one row at a time through its cache, every row on the recurrent rule the gated_delta_rule op
    defines."""
    from transformers.cache_utils import DynamicCache

    cache, outs = DynamicCache(config=model.config), []
    with hf_torch_paths(chunk_is_recurrent=True), torch.no_grad():
        for t in range(ids.shape[0]):
            outs.append(model(ids[None, t : t + 1], past_key_values=cache, use_cache=True).logits[0])
    return torch.cat(outs)


def check_qwen35(source, expected: torch.Tensor, ids: torch.Tensor) -> Interpreter:
    """The reference against transformers' rows, and windows of any size through the conv, recurrent and KV states
    against the full pass."""
    n = ids.shape[0]
    it = Interpreter(frontend.build(source, ctx_max=64), source, dtype=torch.float32)
    full = it.run({"tokens": ids.int(), "pos": torch.arange(n, dtype=torch.int32)})["logits"]
    err = (full - expected).abs().max().item()
    assert torch.allclose(full, expected, atol=2e-4, rtol=2e-4), err
    it.reset()
    parts, at = [], 0
    for w in (11, 1, 1, 4, 1, n - 18):  # a prompt, decode rows, an 8-row-sized window, a tail
        parts.append(
            it.run({"tokens": ids[at : at + w].int(), "pos": torch.arange(at, at + w, dtype=torch.int32)})["logits"]
        )
        at += w
    assert torch.allclose(torch.cat(parts), full, atol=1e-4, rtol=1e-4), (torch.cat(parts) - full).abs().max()
    return it


def test_qwen35_matches_transformers():
    model = tiny_qwen35()
    ids = torch.randint(0, QWEN35["vocab_size"], (30,), generator=torch.Generator().manual_seed(3))
    expected = hf_stepwise(model, ids)
    with hf_torch_paths(chunk_is_recurrent=False), torch.no_grad():  # transformers' chunked prefill: a loose bound
        assert torch.allclose(model(ids[None]).logits[0], expected, atol=5e-2, rtol=5e-2)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        source = warpc.sources.open(tmp)
        c = source.config
        assert (c.family, c.rotary_dim, c.norm_offset, c.quantization) == ("qwen3_5", 8, 1.0, None)
        assert c.layer_types == ("linear_attention",) * 3 + ("attention",)
        check_qwen35(source, expected, ids)


def test_qwen35_vision_language_prefix():
    """The vision-language checkpoint nests the text config under text_config and the text tensors under
    model.language_model.; the graph reads the same weights."""
    model = tiny_qwen35()
    ids = torch.randint(0, QWEN35["vocab_size"], (24,), generator=torch.Generator().manual_seed(4))
    expected = hf_stepwise(model, ids)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from safetensors.torch import load_file, save_file

        path = Path(tmp) / "model.safetensors"
        tensors = load_file(path)
        renamed = {k.replace("model.", "model.language_model.", 1): v for k, v in tensors.items()}
        renamed["mtp.fc.weight"] = torch.zeros(4, 4)  # carried by the real checkpoint, read by nobody
        save_file(renamed, path)
        text = json.loads((Path(tmp) / "config.json").read_text())
        hf = {"model_type": "qwen3_5", "tie_word_embeddings": False, "text_config": text, "vision_config": {}}
        (Path(tmp) / "config.json").write_text(json.dumps(hf))
        source = warpc.sources.open(tmp)
        graph = frontend.build(source, ctx_max=64)
        assert graph.weights["embed"].source == "model.language_model.embed_tokens.weight"
        assert graph.weights["lm_head"].source == "lm_head.weight"
        check_qwen35(source, expected, ids)


def pack_w4(codes: torch.Tensor) -> torch.Tensor:
    """Nibbles [N, K] in 0..15 -> int32 [N, K / 8], column 8j + e in nibble e of word j."""
    words = (codes.reshape(codes.shape[0], -1, 8).long() << torch.arange(0, 32, 4)).sum(-1)
    return torch.from_numpy(words.numpy().astype("uint32").view("int32"))


def test_qwen35_w4g128_checkpoint():
    """A compressed-tensors W4G128 checkpoint: the packed linears become w4g128 weights (in_proj_a / b stay dense, as in
    the real checkpoint's ignore list), and the reference equals transformers running the dequantised fp32 matrices."""
    # every linear's K a multiple of the 128-column group: hidden 128, FF 256, 4 value heads of 32
    model = tiny_qwen35(hidden_size=128, intermediate_size=256, linear_value_head_dim=32)
    g = torch.Generator().manual_seed(5)
    packed = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() != 2 or "embed" in name or "lm_head" in name or "in_proj_a" in name or "in_proj_b" in name:
                continue
            n, k = p.shape
            codes = torch.randint(0, 16, (n, k), generator=g)
            scale = (torch.rand(n, k // 128, generator=g) * 0.02 + 0.002).bfloat16()
            p.copy_((codes - 8).float() * scale.float().repeat_interleave(128, 1))
            packed[name] = (pack_w4(codes), scale)
    ids = torch.randint(0, QWEN35["vocab_size"], (24,), generator=torch.Generator().manual_seed(6))
    expected = hf_stepwise(model, ids)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from safetensors.torch import load_file, save_file

        path = Path(tmp) / "model.safetensors"
        tensors = load_file(path)
        for name, (words, scale) in packed.items():
            base = name[: -len("weight")]
            del tensors[name]
            tensors[base + "weight_packed"], tensors[base + "weight_scale"] = words, scale
            tensors[base + "weight_shape"] = torch.tensor([words.shape[0], words.shape[1] * 8])
        save_file(tensors, path)
        cfg = json.loads((Path(tmp) / "config.json").read_text())
        cfg["quantization_config"] = W4G128_CONFIG
        (Path(tmp) / "config.json").write_text(json.dumps(cfg))
        source = warpc.sources.open(tmp)
        assert source.config.quantization == "w4g128"
        graph = frontend.build(source, ctx_max=64)
        schemes = {w.name: w.scheme for w in graph.weights.values()}
        assert schemes["L0.qkv"] == schemes["L3.q"] == schemes["L3.down"] == "w4g128", schemes
        assert schemes["L0.a"] == schemes["L0.b"] == schemes["lm_head"] == schemes["L0.conv_w"] == "dense", schemes
        check_qwen35(source, expected, ids)


W4G128_CONFIG = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "format": "pack-quantized",
            "targets": ["Linear"],
            "input_activations": None,
            "output_activations": None,
            "weights": {
                "num_bits": 4,
                "type": "int",
                "strategy": "group",
                "group_size": 128,
                "symmetric": True,
                "dynamic": False,
                "actorder": "static",
            },
        }
    },
    "ignore": ["lm_head"],
}


def test_qwen35_fp8_kv_reads_the_checkpoint_scales():
    """kv="fp8": the attention layers read the checkpoint's per-layer k_scale / v_scale as weights; the e4m3 cache stays
    close to the exact one, and a wrong scale moves it."""
    model = tiny_qwen35()
    ids = torch.randint(0, QWEN35["vocab_size"], (20,), generator=torch.Generator().manual_seed(7))
    inputs = {"tokens": ids.int(), "pos": torch.arange(20, dtype=torch.int32)}
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        from safetensors.torch import load_file, save_file

        path = Path(tmp) / "model.safetensors"
        tensors = load_file(path)
        source = warpc.sources.open(tmp)
        exact = Interpreter(frontend.build(source, ctx_max=32), source, dtype=torch.float32).run(inputs)["logits"]
        errs = []
        for s in (0.01, 1.0):  # |k|, |v| here are about 1 to 3: 0.01 puts them near e4m3's top, 1.0 far below
            scales = {f"model.layers.3.self_attn.{n}": torch.tensor([s]).bfloat16() for n in ("k_scale", "v_scale")}
            save_file({**tensors, **scales}, path)
            source = warpc.sources.open(tmp)
            graph = frontend.build(source, ctx_max=32, kv="fp8")
            att = next(n for n in graph.nodes if n.op == "attention")
            assert graph.weights[att.attrs["k_scale"]].source == "model.layers.3.self_attn.k_scale"
            logits = Interpreter(graph, source, dtype=torch.float32).run(inputs)["logits"]
            errs.append((logits - exact).abs().max().item())
        assert 0 < errs[0] < 0.2 and errs[0] != errs[1], errs


def test_qwen35_graph_shape():
    model = tiny_qwen35()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        source = warpc.sources.open(tmp)
        graph = frontend.build(source, ctx_max=64)
    kinds = {s.name: (s.kind, s.shape) for s in graph.states.values()}
    assert kinds == {
        "L0.conv": ("conv", (128, 3)),
        "L0.rec": ("recurrent", (4, 16, 16)),
        "L1.conv": ("conv", (128, 3)),
        "L1.rec": ("recurrent", (4, 16, 16)),
        "L2.conv": ("conv", (128, 3)),
        "L2.rec": ("recurrent", (4, 16, 16)),
        "L3.kv": ("kv", (2, 64, 2, 32)),
    }, kinds
    ops = [n.op for n in graph.nodes]
    for op, count in (("gated_delta_rule", 3), ("causal_conv1d", 3), ("silu", 3), ("gated_rmsnorm", 3)):
        assert ops.count(op) == count, (op, ops.count(op))
    assert ops.count("split") == 4 and ops.count("sigmoid_mul") == 1 and ops.count("attention") == 1
    assert {n.attrs.get("offset") for n in graph.nodes if n.op == "rmsnorm"} == {1.0}
    assert graph.weights["L0.conv_w"].shape == (128, 4)  # [C, 1, W] on disk
    rope = next(n for n in graph.nodes if n.op == "rope")
    assert rope.attrs["rotary_dim"] == 8 and len(rope.attrs["inv_freq"]) == 4
    for bad in ({"attn_output_gate": False}, {"layer_types": ["linear_attention"] * 3 + ["sliding_attention"]}):
        hf = {**QWEN35, "model_type": "qwen3_5_text", **bad}
        try:
            ModelConfig.from_hf(hf)
        except UnsupportedModel as e:
            assert next(iter(bad)) in str(e), str(e)
        else:
            raise AssertionError(f"{bad} accepted")


def test_real_qwen35_checkpoint():
    """The 27B checkpoint: every weight the graph declares exists with the declared shape (safetensors headers: the
    packed form's weight_shape, the [C, 1, W] conv), and the schemes are the checkpoint's (400 W4G128 linears)."""
    path = os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        raise Skip("WARPC_TEST_SWIFT27 unset")
    from safetensors import safe_open

    source = warpc.sources.open(path)
    graph = frontend.build(source, ctx_max=4096, kv="fp8")
    files = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]

    def shape_of(w) -> tuple[int, ...]:
        """The on-disk shape: the packed form's weight_shape values, else the header's ([C, 1, W] conv as [C, W])."""
        name = w.source[: -len("weight")] + "weight_shape" if w.scheme == "w4g128" else w.source
        with safe_open(str(Path(path) / files[name]), framework="pt") as h:
            if w.scheme == "w4g128":
                return tuple(h.get_tensor(name).tolist())
            shape = tuple(h.get_slice(name).get_shape())
        return (shape[0], shape[2]) if w.name.endswith("conv_w") else shape

    for w in graph.weights.values():
        assert shape_of(w) == w.shape, (w.name, w.source, shape_of(w), w.shape)
    schemes = {s: sum(1 for w in graph.weights.values() if w.scheme == s) for s in ("w4g128", "dense")}
    c = source.config
    print(f"  {len(graph.nodes)} nodes, {len(graph.weights)} weights {schemes}, {len(graph.states)} states")
    assert schemes["w4g128"] == 400 and c.layer_types.count("attention") == 16, schemes


def test_data_movement_ops():
    """concat, select_row and take on the reference interpreter: exact, and ill-typed uses refused."""
    g = Graph("move")
    x = g.input("x", TType(("T", 3), "bf16"))
    y = g.input("y", TType(("T", 2), "bf16"))
    i = g.input("i", TType((1,), "i32"))
    table = (7, 0, 16777215, 42)
    g.output(cat=g.op("concat", [x, y]), row=g.op("select_row", [x, i]), id=g.op("take", [i], table=table))
    g.verify()
    assert [g.types[v] for v in g.outputs.values()] == [
        TType(("T", 5), "bf16"),
        TType((1, 3), "bf16"),
        TType((1,), "i32"),
    ]
    a, b = torch.randn(4, 3).bfloat16(), torch.randn(4, 2).bfloat16()
    it = Interpreter(g, None)
    for k in range(4):
        out = it.run({"x": a, "y": b, "i": torch.tensor([k], dtype=torch.int32)})
        assert torch.equal(out["cat"], torch.cat([a, b], 1)) and torch.equal(out["row"], a[k : k + 1])
        assert out["id"].dtype == torch.int32 and out["id"].tolist() == [table[k]]

    def refused(op, ins, attrs, needle):
        h = Graph("bad")
        for n, t in ins.items():
            h.input(n, t)
        try:
            h.op(op, list(ins), **attrs)
        except ValueError as e:
            assert needle in str(e), str(e)
        else:
            raise AssertionError(f"{op} {attrs} accepted")

    row, idx = TType(("T", 3), "bf16"), TType((1,), "i32")
    refused("concat", {"x": row, "y": TType(("T", 2), "f32")}, {}, "concat")
    refused("concat", {"x": row, "y": TType((2, 2), "bf16")}, {}, "concat")
    refused("select_row", {"x": row, "i": TType((2,), "i32")}, {}, "select_row")
    refused("select_row", {"x": row, "i": TType((1,), "f32")}, {}, "select_row")
    for bad in ((), (-1,), (1 << 24,), [1, 2], (1.5,)):
        refused("take", {"i": idx}, {"table": bad}, "take: table")
    refused("take", {"i": TType((2,), "i32")}, {"table": (1,)}, "take")


def tiny_mtp_source(tmp: str, **overrides):
    """A tiny Qwen3.5 checkpoint in `tmp` plus the 15 `mtp.*` tensors of the real one (random, bf16-representable) and
    mtp_num_hidden_layers = 1; returns (model, the source, the mtp tensors)."""
    from safetensors.torch import load_file, save_file

    model = tiny_qwen35(**overrides)
    model.save_pretrained(tmp)
    c = model.config
    H, FF, hd = c.hidden_size, c.intermediate_size, c.head_dim
    nq, nkv = c.num_attention_heads, c.num_key_value_heads
    shapes = {
        "fc": (H, 2 * H),
        "pre_fc_norm_embedding": (H,),
        "pre_fc_norm_hidden": (H,),
        "norm": (H,),
        "layers.0.input_layernorm": (H,),
        "layers.0.post_attention_layernorm": (H,),
        "layers.0.self_attn.q_proj": (2 * nq * hd, H),
        "layers.0.self_attn.k_proj": (nkv * hd, H),
        "layers.0.self_attn.v_proj": (nkv * hd, H),
        "layers.0.self_attn.o_proj": (H, nq * hd),
        "layers.0.self_attn.q_norm": (hd,),
        "layers.0.self_attn.k_norm": (hd,),
        "layers.0.mlp.gate_proj": (FF, H),
        "layers.0.mlp.up_proj": (FF, H),
        "layers.0.mlp.down_proj": (H, FF),
    }
    gen = torch.Generator().manual_seed(11)
    mtp = {}
    for name, shape in shapes.items():
        scale = 0.3 if "norm" in name else shape[-1] ** -0.5
        mtp[f"mtp.{name}.weight"] = (torch.randn(shape, generator=gen) * scale).bfloat16().float()
    path = Path(tmp) / "model.safetensors"
    save_file({**load_file(path), **mtp}, path)
    cfg = json.loads((Path(tmp) / "config.json").read_text())
    cfg["mtp_num_hidden_layers"] = 1
    (Path(tmp) / "config.json").write_text(json.dumps(cfg))
    return model, warpc.sources.open(tmp), mtp


class MtpOracle:
    """The MTP module as vLLM wraps it (qwen3_5_mtp.py): fc(cat[norm(embed(token)), norm(hidden)]), then a
    transformers `Qwen3_5DecoderLayer` of a full-attention layer loaded with the mtp.layers.0.* weights, then `norm` and
    the target's lm_head, causal attention over the whole sequence of (token, hidden) pairs (no cache: every call
    recomputes the sequence)."""

    def __init__(self, model, mtp: dict[str, torch.Tensor]):
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )

        c = model.config
        self.mtp, self.embed, self.head = mtp, model.model.embed_tokens.weight.detach(), model.lm_head.weight.detach()
        self.layer = Qwen3_5DecoderLayer(c, c.layer_types.index("full_attention")).float().eval()
        prefix = "mtp.layers.0."
        self.layer.load_state_dict(
            {k[len(prefix) : -len(".weight")] + ".weight": v for k, v in mtp.items() if k.startswith(prefix)}
        )
        self.rotary = Qwen3_5TextRotaryEmbedding(c)
        self.norm = Qwen3_5RMSNorm
        self.eps = c.rms_norm_eps

    def _norm(self, x: torch.Tensor, name: str) -> torch.Tensor:
        n = self.norm(x.shape[-1], eps=self.eps)
        n.weight.data = self.mtp[f"mtp.{name}.weight"]
        return n(x)

    @torch.no_grad()
    def rows(self, tokens: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens [S], hidden [S, H] -> post-norm rows [S, H] and their logits [S, V]."""
        S = tokens.shape[0]
        x = torch.cat(
            [self._norm(self.embed[tokens.long()], "pre_fc_norm_embedding"), self._norm(hidden, "pre_fc_norm_hidden")],
            -1,
        )
        x = torch.nn.functional.linear(x, self.mtp["mtp.fc.weight"])[None]
        position_ids = torch.arange(S).view(1, 1, -1).expand(3, 1, -1)
        mask = torch.full((S, S), float("-inf")).triu(1)[None, None]
        x = self.layer(
            x, position_embeddings=self.rotary(x, position_ids), attention_mask=mask, position_ids=position_ids[0]
        )
        x = self._norm(x[0], "norm")
        return x, torch.nn.functional.linear(x, self.head)


def check_mtp_windows(source, oracle: MtpOracle, draft_ids=None, atol: float = 2e-4) -> None:
    """The draft graph on windows of pairs against the oracle: a window of 6 rows at 0 (select a row in the middle), a
    window of 4 at 6 accepting 2 and then the chain's single rows at 8 and 9 over the KV rows the window left."""
    c = source.config
    graph = frontend.build_draft(source, ctx_max=32)
    it = Interpreter(graph, source, dtype=torch.float32)
    gen = torch.Generator().manual_seed(12)
    tok = torch.randint(0, c.vocab_size, (16,), generator=gen)
    hid = torch.randn(16, c.hidden_size, generator=gen)
    seq_tok, seq_hid = [], []  # the pair at each cache position, as the last call to touch it left it

    def step(at: int, tokens: torch.Tensor, hidden: torch.Tensor, sel: int) -> None:
        T = tokens.shape[0]
        del seq_tok[at:], seq_hid[at:]
        seq_tok.extend(tokens.tolist()), seq_hid.extend(hidden)
        out = it.run(
            {
                "tokens": tokens.int(),
                "pos": torch.arange(at, at + T, dtype=torch.int32),
                "hidden": hidden,
                "sel": torch.tensor([sel], dtype=torch.int32),
            }
        )
        norm_rows, logits = oracle.rows(torch.tensor(seq_tok), torch.stack(seq_hid))
        want, want_logits = norm_rows[at + sel], logits[at + sel]
        assert out["hidden"].shape == (1, c.hidden_size) and out["logits"].shape[0] == 1
        err = (out["hidden"][0] - want).abs().max().item()
        assert torch.allclose(out["hidden"][0], want, atol=atol, rtol=atol), (at, sel, err)
        if draft_ids is None:
            assert torch.allclose(out["logits"][0], want_logits, atol=atol, rtol=atol), (at, sel)
            assert out["ids"].tolist() == [int(want_logits.argmax())], (at, sel)
        else:
            assert out["ids"].tolist() == [draft_ids[int(out["logits"][0].argmax())]], (at, sel)

    step(0, tok[:6], hid[:6], 3)
    step(6, tok[6:10], hid[6:10], 1)  # accepts 2 rows: rows 8, 9 of the window are junk
    step(8, tok[10:11], hid[10:11], 0)  # the chain row over the junk row's slot
    step(9, tok[11:12], hid[11:12], 0)
    step(10, tok[12:16], hid[12:16], 3)  # the next window; the junk is long overwritten


def test_mtp_draft_matches_the_vllm_wrapped_layer():
    """O3: reference.py on `build_draft` (fp32) against transformers' decoder layer wrapped as vLLM wraps it: fc over
    the two normed halves, the layer, norm, the full head; select_row, a multi-row window and single rows through the
    KV state."""
    with tempfile.TemporaryDirectory() as tmp:
        model, source, mtp = tiny_mtp_source(tmp)
        graph = frontend.build_draft(source, ctx_max=32)
        assert set(graph.inputs) == {"tokens", "pos", "hidden", "sel"} and set(graph.outputs) == {
            "hidden",
            "logits",
            "ids",
        }
        assert graph.types[graph.outputs["hidden"]] == TType((1, 64), "bf16")
        assert graph.types[graph.outputs["logits"]] == TType((1, 128), "f32")
        assert graph.types[graph.outputs["ids"]] == TType((1,), "i32")
        assert list(graph.states) == ["mtp.kv"] and graph.states["mtp.kv"].shape == (2, 32, 2, 32)
        assert {n.op for n in graph.nodes} >= {"concat", "select_row"} and "take" not in {n.op for n in graph.nodes}
        assert all(w.startswith("mtp.") or w in ("embed", "lm_head") for w in graph.weights), list(graph.weights)
        assert graph.weights["embed"].source == "model.embed_tokens.weight"
        assert graph.weights["lm_head"].source == "lm_head.weight"
        assert {n.attrs.get("offset") for n in graph.nodes if n.op == "rmsnorm"} == {1.0}
        check_mtp_windows(source, MtpOracle(model, mtp))


def test_mtp_draft_requantized_head_maps_ids_back():
    """With a Requantized source the head is the pseudo-tensor mtp.draft_head.weight (w4g128) and the argmax goes
    through `take` to full-vocabulary ids; the target graph is untouched by the wrapper."""
    import numpy as np

    from warpc.sources.requantize import MTP_MATRICES, Requantized

    with tempfile.TemporaryDirectory() as tmp:
        # every linear's K a multiple of the 128-column group, as in test_qwen35_w4g128_checkpoint
        _, base, _ = tiny_mtp_source(tmp, hidden_size=128, intermediate_size=256, linear_value_head_dim=32)
        ids = np.arange(3, 128, 2, dtype=np.int64)
        source = Requantized(base, matrices=MTP_MATRICES, draft_ids=ids)
        graph = frontend.build_draft(source, ctx_max=32)
        head = graph.weights["mtp.draft_head"]
        assert (head.scheme, head.shape, head.source) == ("w4g128", (len(ids), 128), "mtp.draft_head.weight")
        assert "lm_head" not in graph.weights and graph.weights["embed"].source == "model.embed_tokens.weight"
        schemes = {n: w.scheme for n, w in graph.weights.items()}
        assert schemes["mtp.fc"] == schemes["mtp.q"] == schemes["mtp.down"] == "w4g128", schemes
        assert schemes["mtp.ln1"] == schemes["mtp.norm"] == schemes["embed"] == "dense", schemes
        take = next(n for n in graph.nodes if n.op == "take")
        assert take.attrs["table"] == tuple(int(i) for i in ids)
        assert graph.types[graph.outputs["logits"]] == TType((1, len(ids)), "f32")
        assert graph.types[graph.outputs["ids"]] == TType((1,), "i32")
        target = frontend.build(source, ctx_max=32)
        assert not any(n.startswith("mtp.") for n in target.weights) and set(target.outputs) == {
            "hidden",
            "logits",
            "ids",
        }
        it = Interpreter(graph, source, dtype=torch.float32)
        gen = torch.Generator().manual_seed(13)
        for _ in range(3):  # random windows: the id is the table entry of the logits' argmax
            T = 3
            out = it.run(
                {
                    "tokens": torch.randint(0, 128, (T,), generator=gen).int(),
                    "pos": torch.arange(T, dtype=torch.int32),
                    "hidden": torch.randn(T, 128, generator=gen),
                    "sel": torch.tensor([T - 1], dtype=torch.int32),
                }
            )
            assert out["ids"].tolist() == [int(ids[int(out["logits"][0].argmax())])]
            it.reset()


def test_mtp_draft_guards():
    """Only a one-layer MTP head of a Qwen3.5 checkpoint has a draft graph; the KV format is checked."""
    with tempfile.TemporaryDirectory() as tmp:
        _, source, _ = tiny_mtp_source(tmp)
        try:
            frontend.build_draft(source, ctx_max=32, draft_kv="fp4")
        except ValueError as e:
            assert "fp4" in str(e), str(e)
        else:
            raise AssertionError("an unknown draft KV format accepted")
        cfg = json.loads((Path(tmp) / "config.json").read_text())
        for layers in (0, 2):
            (Path(tmp) / "config.json").write_text(json.dumps({**cfg, "mtp_num_hidden_layers": layers}))
            try:
                frontend.build_draft(warpc.sources.open(tmp), ctx_max=32)
            except UnsupportedModel as e:
                assert f"{layers} MTP layers" in str(e), str(e)
            else:
                raise AssertionError(f"{layers} MTP layers accepted")
        (Path(tmp) / "config.json").write_text(json.dumps(cfg))
        from safetensors.torch import load_file, save_file

        path = Path(tmp) / "model.safetensors"
        tensors = load_file(path)
        del tensors["mtp.fc.weight"], tensors["mtp.norm.weight"]
        save_file(tensors, path)
        try:
            frontend.build_draft(warpc.sources.open(tmp), ctx_max=32)
        except UnsupportedModel as e:
            assert "mtp.fc.weight" in str(e) and "mtp.norm.weight" in str(e), str(e)
        else:
            raise AssertionError("a checkpoint without the MTP tensors accepted")
    model = tiny_model("llama")
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        try:
            frontend.build_draft(warpc.sources.open(tmp), ctx_max=32)
        except UnsupportedModel as e:
            assert "llama" in str(e), str(e)
        else:
            raise AssertionError("a llama checkpoint has no draft graph")


def test_real_mtp_head_shapes():
    """The 27B checkpoint: every weight of the draft graph exists with the declared shape (the draft head is the
    lm_head rows of the vocabulary list)."""
    path = os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        raise Skip("WARPC_TEST_SWIFT27 unset")
    from safetensors import safe_open

    from warpc.sources.requantize import Requantized

    source = Requantized.mtp(warpc.sources.open(path))
    graph = frontend.build_draft(source, ctx_max=4096)
    files = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
    for w in graph.weights.values():
        name = "lm_head.weight" if w.source == "mtp.draft_head.weight" else w.source
        with safe_open(str(Path(path) / files[name]), framework="pt") as h:
            shape = tuple(h.get_slice(name).get_shape())
        if w.source == "mtp.draft_head.weight":  # the rows of lm_head named by the vocabulary list
            assert shape[0] >= w.shape[0] and shape[1:] == w.shape[1:], (w.name, shape, w.shape)
        else:
            assert shape == w.shape, (w.name, w.source, shape, w.shape)
    assert len(graph.weights) == 17 and set(graph.states) == {"mtp.kv"}, len(graph.weights)
    print(f"  {len(graph.nodes)} nodes, {len(graph.weights)} weights")


INKLING_ORACLE = Path(__file__).resolve().parent / "oracle" / "inkling_hf.py"
INKLING_WINDOWS = (10, 1, 1, 4, 3, 1, 4)


def test_inkling_graph_shape():
    """The real config: 48 visits of 24 layers, every weight and state counted, rings on the sliding visits."""
    source = InklingShapes(ModelConfig.from_hf(inkling_w4()))
    graph = frontend.build(source, ctx_max=4096, rows_max=8)
    states = graph.states
    assert len(states) == 240 and len(graph.weights) == 718
    assert sum(1 for k in graph.outputs if k.startswith("h")) == 48
    # layers 5 and 23 run once, 11 and 17 three times: 8 full visits; the other 40 are rings of window + rows - 1
    kv = [s.shape[1] for s in states.values() if s.kind == "kv"]
    assert sorted(set(kv)) == [1031, 4096] and kv.count(4096) == 8 and kv.count(1031) == 40, sorted(set(kv))
    ops = {}
    for n in graph.nodes:
        ops[n.op] = ops.get(n.op, 0) + 1
    assert {
        k: ops[k] for k in ("attention", "moe_route", "moe_experts", "message_gate", "dyn_conv1d", "log_scale")
    } == {
        "attention": 48,
        "moe_route": 48,
        "moe_experts": 48,
        "message_gate": 96,  # 24 loop visits in passes 2 and 3 x (attn, k, v, mlp)
        "dyn_conv1d": 16,
        "log_scale": 16,
    }, ops
    # 96 message reads (12 loop layers x 4 queries x 2 passes), one block-residual read per sliding visit but layer 0's
    # (39) and three per full visit (24), one per visit before the MLP (48), the output read
    assert ops["attn_res"] == 96 + 39 + 24 + 48 + 1, ops["attn_res"]
    schemes = {s: sum(1 for w in graph.weights.values() if w.scheme == s) for s in ("w4g128", "f32", "dense")}
    assert schemes == {"w4g128": 72, "f32": 74, "dense": 572}, schemes
    assert graph.weights["L6.e_gate"].shape == (128, 1152, 2304) and graph.weights["L6.e_down"].shape == (
        128,
        2304,
        1152,
    )
    assert graph.weights["L5.aq"].shape == (3, 2304) and graph.weights["L5.kstatic"].shape == (4, 1024)
    assert graph.weights["L0.ksconv"].shape == (1024, 4)  # [C, 1, W] on disk
    assert abs(sum(w.nbytes for w in graph.weights.values()) / 2**30 - 14.74) < 0.01
    assert graph.types[graph.outputs["logits"]].shape == ("T", 200058)
    assert graph.types[graph.outputs["ids"]].dtype == "i32"
    ring = frontend.build(source, ctx_max=512, rows_max=8)  # a window longer than the context is no ring
    assert {s.shape[1] for s in ring.states.values() if s.kind == "kv"} == {512}


def test_inkling_guards():
    cases = {
        "n_shared_experts=1": {"n_shared_experts": 1},
        "q_bias=True": {"q_bias": True},
        "o_bias=True": {"o_bias": True},
        "use_value_norm_no_gamma=False": {"use_value_norm_no_gamma": False},
        "attn_res_normalize_values=False": {"attn_res_normalize_values": False},
        "use_gate_bias=False": {"use_gate_bias": False},
        "loop_config.is_varloop": {
            "loop_config": {**INKLING_OFFICIAL["text_config"]["loop_config"], "is_varloop": True}
        },
        "scoring_func='softmax'": {"scoring_func": "softmax"},
        "missing sconv_kernel_size": {"sconv_kernel_size": None},
        "missing loop_config": {"loop_config": None},
    }
    for message, patch in cases.items():
        try:
            ModelConfig.from_hf(inkling_w4(**patch))
        except UnsupportedModel as e:
            assert message in str(e), (message, str(e))
        else:
            raise AssertionError(f"{patch} accepted")
    for quantization, message in (
        (INKLING_OFFICIAL["quantization_config"], "requantise the experts to W4G128"),
        ({**INKLING_W4G128, "warpc_layout": None}, "warpc_layout"),
        ({**INKLING_W4G128, "format": "int-quantized"}, "pack-quantized"),
    ):
        try:
            ModelConfig.from_hf({**inkling_w4(), "quantization_config": quantization})
        except UnsupportedModel as e:
            assert message in str(e), (message, str(e))
        else:
            raise AssertionError(f"{quantization} accepted")
    c = ModelConfig.from_hf(inkling_w4())
    ik = c.inkling
    assert (c.family, c.rope_theta, c.quantization, c.vocab_size, ik.unpadded_vocab) == (
        "inkling2_loop",
        None,
        "w4g128",
        201024,
        200058,
    )
    assert [i for i, t in enumerate(c.layer_types) if t == "full_attention"] == [5, 11, 17, 23]
    assert (c.sliding_window, ik.d_rel, ik.conv_kernel, ik.experts, ik.top_k, ik.expert_width, ik.gate_rank) == (
        1024,
        16,
        4,
        128,
        8,
        1152,
        64,
    )
    # the checkpoint without a quantisation config has dense experts
    dense = {k: v for k, v in INKLING_OFFICIAL.items() if k != "quantization_config"}
    assert ModelConfig.from_hf(dense).quantization is None


def inkling_oracle(variant: str, directory: Path) -> dict:
    """Run the official model in the reference environment (`WARPC_INKLING_REF_PYTHON`: its python, or its venv)."""
    value = os.environ.get("WARPC_INKLING_REF_PYTHON")
    if not value:
        raise Skip("WARPC_INKLING_REF_PYTHON unset (the isolated environment with the official inkling2_loop)")
    python = Path(value) / "bin" / "python" if Path(value).is_dir() else Path(value)
    done = subprocess.run([str(python), str(INKLING_ORACLE), variant, str(directory)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return torch.load(directory / "ref.pt")


def inkling_windows(it: Interpreter, ids: torch.Tensor, sizes) -> torch.Tensor:
    it.reset()
    parts, at = [], 0
    for w in sizes:
        window = {"tokens": ids[at : at + w], "pos": torch.arange(at, at + w, dtype=torch.int32)}
        parts.append(it.run(window)["logits"])
        at += w
    return torch.cat(parts)


def check_inkling(variant: str):
    """The interpreter on the tiny model against the official one, fp32: every visit's output within a few times the
    official model's own sensitivity to 1e-7 noise, the logits, windows through the states (also against the official
    cache), a state ring that wraps, and six semantic slips that each move the logits."""
    with tempfile.TemporaryDirectory() as tmp:
        ref = inkling_oracle(variant, Path(tmp))
        source = warpc.sources.open(Path(tmp) / "ckpt")
        n = len(ref["ids"])
        graph = frontend.build(source, ctx_max=32, rows_max=n)
        assert (source.config.quantization == "w4g128") == (variant == "b")
        it = Interpreter(graph, source, dtype=torch.float32)
        ids = ref["ids"].int()
        full_run = {"tokens": ids, "pos": torch.arange(n, dtype=torch.int32)}
        outs = it.run(full_run)
        visits = [k for k in outs if k.startswith("h")]
        assert len(visits) == len(ref["visits"]) == 16 and len(graph.states) == 5 * 16
        for v, (want, floor) in enumerate(zip(ref["visits"], ref["floor"], strict=True)):
            err = ((outs[f"h{v:02d}"] - want).norm() / want.norm()).item()
            assert err < max(5 * floor, 2e-6), (variant, v, err, floor)
        full = outs["logits"]
        assert full.shape == ref["logits"].shape == (n, 120)
        d = (full - ref["logits"]).abs().max().item()
        assert torch.allclose(full, ref["logits"], atol=5e-4, rtol=1e-3), d
        assert (full.argmax(-1) == ref["logits"].argmax(-1)).all()
        windows = inkling_windows(it, ids, INKLING_WINDOWS)
        assert torch.allclose(windows, full, atol=1e-4, rtol=1e-4), (windows - full).abs().max()
        assert torch.allclose(windows, ref["cache_logits"], atol=2e-4, rtol=1e-3), (
            (windows - ref["cache_logits"]).abs().max()
        )
        # steps of at most 8 rows: the sliding states are rings of 6 + 8 - 1 = 13 rows, and 24 positions wrap them
        ring = frontend.build(source, ctx_max=32, rows_max=8)
        assert {s.shape[1] for s in ring.states.values() if s.kind == "kv"} == {13, 32}
        wrapped = inkling_windows(Interpreter(ring, source, dtype=torch.float32), ids, (8, 1, 1, 4, 3, 1, 4, 2))
        assert torch.allclose(wrapped, full, atol=1e-4, rtol=1e-4), (wrapped - full).abs().max()

        slips = {
            "no sink mass": (
                "attention",
                lambda f: lambda it, a, q, k, v, p, r, s: f(it, a, q, k, v, p, r, s * 0 - 1e9),
            ),
            "no window": ("attention", lambda f: lambda it, a, *x: f(it, dict(a, window=None), *x)),
            "bias off by one": (
                "attention",
                lambda f: lambda it, a, q, k, v, p, r, s: f(it, a, q, k, v, p, r.roll(1, -1), s),
            ),
            "conv divisor": ("causal_conv1d", lambda f: lambda it, a, x: f(it, dict(a, divisor=a["divisor"] - 0.3), x)),
            "no log scaling": ("log_scale", lambda f: lambda it, a, x, pos: x),
            "route scale": ("moe_route", lambda f: lambda it, a, x: (lambda o: (o[0], o[1] * 1.01))(f(it, a, x))),
        }
        impl = dict(_IMPL)
        try:
            for name, (op, wrap) in slips.items():
                _IMPL[op] = wrap(impl[op])
                bad = (
                    (Interpreter(graph, source, dtype=torch.float32).run(full_run)["logits"] - ref["logits"])
                    .abs()
                    .max()
                )
                _IMPL[op] = impl[op]
                assert bad > 0.05, (variant, name, bad)
        finally:
            _IMPL.update(impl)


def test_inkling_vs_official_dense_experts():
    check_inkling("a")


def test_inkling_vs_official_w4g128_experts():
    check_inkling("b")


if __name__ == "__main__":
    run(globals())
