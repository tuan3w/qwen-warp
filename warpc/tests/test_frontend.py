"""Frontend + reference interpreter vs transformers on tiny random models of every family, and the config guard.

uv run --package warpc python warpc/tests/test_frontend.py
"""

import json
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
from harness import run
from warpc import frontend
from warpc.errors import UnsupportedModel
from warpc.reference import Interpreter
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
        assert set(graph.inputs) == {"tokens", "pos"} and set(graph.outputs) == {"logits", "ids"}
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
        "model_type 'qwen2' (supported: llama, mistral, qwen3)": {"model_type": "qwen2"},
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


if __name__ == "__main__":
    run(globals())
