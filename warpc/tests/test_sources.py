"""Checkpoint sources: GGUF decoders vs ggml C, GGUF reader vs gguf.dequantize, safetensors and compressed-tensors
W4G128 (vs compressed_tensors' own unpacking), config guards.

uv run --package warpc python warpc/tests/test_sources.py

LLAMA_CPP_DIR (a llama.cpp checkout) enables the ggml bit-identity check, WARPC_TEST_GGUF (a GGUF file) the real-file
check; each skips when unset.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import logging as transformers_logging

import warpc.sources
from harness import Skip, run
from warpc import frontend
from warpc.errors import UnsupportedModel
from warpc.ir import Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS
from warpc.sources import ModelConfig, w4g128
from warpc.sources.gguf import DECODERS, GGUFSource, hf_name

transformers_logging.disable_progress_bar()

ORACLE = Path(__file__).resolve().parent / "oracle" / "dequant_oracle.c"
Q = gguf.GGMLQuantizationType
SIZES = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 256,
}
# GGUF format per HF tensor suffix in the synthetic files: the float formats, the three own decoders, one format through
# gguf.dequantize (Q5_0); q / k are float so the llama row permutation is checked exactly against safetensors.
TYPES = {
    "q_proj.weight": Q.F32,
    "k_proj.weight": Q.BF16,
    "v_proj.weight": Q.F16,
    "o_proj.weight": Q.Q8_0,
    "gate_proj.weight": Q.Q4_0,
    "up_proj.weight": Q.Q4_1,
    "down_proj.weight": Q.Q5_0,
    "lm_head.weight": Q.Q4_0,
    "embed_tokens.weight": Q.BF16,
}


def tiny_model(family: str, **overrides):
    """A tiny random HF model whose weights are exactly bf16-representable."""
    if family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**SIZES, tie_word_embeddings=False, **overrides))
    else:
        model = Qwen3ForCausalLM(Qwen3Config(**SIZES, head_dim=32, tie_word_embeddings=False, **overrides))
    torch.manual_seed(0)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_((torch.randn_like(p) * (0.3 if "norm" in n else p.shape[-1] ** -0.5)).bfloat16().float())
    return model.eval()


# Llama 3.1 rope scaling with all three bands in 8 rotary pairs (wavelengths 6.3 | 20, 63 | 199 ... 19869)
LLAMA3_ROPE = {
    "rope_type": "llama3",
    "rope_theta": 10000.0,
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}


def converter_rope_freqs(c) -> np.ndarray:
    """llama.cpp's converter (LlamaModel.generate_extra_tensors), verbatim: Llama 3.1 scaling as rope_freqs."""
    rope_params = c.rope_parameters
    base = rope_params.get("rope_theta", 10000.0)
    dim = c.head_dim
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))

    factor = rope_params.get("factor", 8.0)
    low_freq_factor = rope_params.get("low_freq_factor", 1.0)
    high_freq_factor = rope_params.get("high_freq_factor", 4.0)
    old_context_len = rope_params.get("original_max_position_embeddings", 8192)

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    rope_factors = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            rope_factors.append(1)
        elif wavelen > low_freq_wavelen:
            rope_factors.append(factor)
        else:
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            rope_factors.append(1 / ((1 - smooth) / factor + smooth))
    return torch.tensor(rope_factors, dtype=torch.float32).numpy()


def permute_for_ggml(w: np.ndarray, heads: int) -> np.ndarray:
    """llama.cpp's converter permute of llama q / k rows (half-split pairs -> adjacent pairs)."""
    return w.reshape(heads, 2, w.shape[0] // heads // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def unpermute(w: np.ndarray, heads: int) -> np.ndarray:
    return w.reshape(heads, -1, 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def gguf_name(name: str) -> str:
    table = {
        "model.embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "model.norm.weight": "output_norm.weight",
        "self_attn.q_proj.weight": "attn_q.weight",
        "self_attn.k_proj.weight": "attn_k.weight",
        "self_attn.v_proj.weight": "attn_v.weight",
        "self_attn.o_proj.weight": "attn_output.weight",
        "self_attn.q_norm.weight": "attn_q_norm.weight",
        "self_attn.k_norm.weight": "attn_k_norm.weight",
        "mlp.gate_proj.weight": "ffn_gate.weight",
        "mlp.up_proj.weight": "ffn_up.weight",
        "mlp.down_proj.weight": "ffn_down.weight",
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    if name in table:
        return table[name]
    _, _, i, rest = name.split(".", 3)
    return f"blk.{i}.{table[rest]}"


def write_gguf(path: Path, model, arch: str, *, extra=()) -> None:
    """Convert an HF model the way llama.cpp's converter lays it out (names, metadata, llama q / k permute, Llama 3.1
    rope scaling as rope_freqs)."""
    c = model.config
    w = gguf.GGUFWriter(str(path), arch)
    w.add_block_count(c.num_hidden_layers)
    w.add_embedding_length(c.hidden_size)
    w.add_feed_forward_length(c.intermediate_size)
    w.add_head_count(c.num_attention_heads)
    w.add_head_count_kv(c.num_key_value_heads)
    w.add_layer_norm_rms_eps(c.rms_norm_eps)
    w.add_rope_freq_base(c.rope_parameters["rope_theta"])
    for name, t in model.state_dict().items():
        x = t.float().numpy()
        if arch == "llama" and name.endswith("q_proj.weight"):
            x = permute_for_ggml(x, c.num_attention_heads)
        if arch == "llama" and name.endswith("k_proj.weight"):
            x = permute_for_ggml(x, c.num_key_value_heads)
        qtype = next((q for suffix, q in TYPES.items() if name.endswith(suffix)), Q.F32)
        if qtype == Q.F32:
            w.add_tensor(gguf_name(name), x)
        else:
            w.add_tensor(gguf_name(name), gguf.quantize(x, qtype), raw_dtype=qtype)
    if c.rope_parameters.get("rope_type") == "llama3":
        w.add_tensor("rope_freqs.weight", converter_rope_freqs(c))
    for name, data, qtype in extra:
        w.add_tensor(name, data, raw_dtype=qtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def reference_tensor(t: gguf.ReaderTensor) -> torch.Tensor:
    shape = tuple(int(n) for n in reversed(t.shape))
    return torch.from_numpy(np.array(gguf.dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(shape))


def check_against_gguf_package(source: GGUFSource, path: Path) -> int:
    """Every mapped tensor equals gguf.dequantize of the file's tensor (rows back in HF order), bit for bit in bf16."""
    llama = source.config.family in ("llama", "mistral")
    checked = 0
    for t in gguf.GGUFReader(str(path)).tensors:
        name = hf_name(t.name)
        if name is None:
            continue
        expected = reference_tensor(t)
        if llama and name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = source.config.num_heads if "q_proj" in name else source.config.num_kv_heads
            expected = torch.from_numpy(unpermute(expected.numpy(), heads))
        got = source.get(name)
        assert got.dtype == torch.bfloat16 and got.shape == expected.shape, (name, got.dtype, got.shape)
        assert torch.equal(got, expected.bfloat16()), name
        kind = t.tensor_type.name
        assert source.quant(name) == (None if kind in ("F32", "F16", "BF16") else kind), name
        checked += 1
    return checked


def test_ggml_dequant_bit_identical():
    llama_cpp = os.environ.get("LLAMA_CPP_DIR")
    if not llama_cpp:
        raise Skip("LLAMA_CPP_DIR unset")
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        raise Skip("no C compiler (CC)")
    root = Path(llama_cpp)
    quants = root / "ggml-quants.c"
    includes = [root]
    if not quants.is_file():
        quants = root / "ggml" / "src" / "ggml-quants.c"
        includes = [root / "ggml" / "include", root / "ggml" / "src"]
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "dequant_oracle"
        flags = [f"-I{d}" for d in includes]
        r = subprocess.run(
            [compiler, "-O2", *flags, str(ORACLE), str(quants), "-lm", "-o", str(exe)], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-2000:]
        r = subprocess.run([str(exe), tmp], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        for kind, decode in DECODERS.items():
            raw = np.fromfile(Path(tmp) / f"{kind.lower()}.raw", dtype=np.uint8)
            expected = np.fromfile(Path(tmp) / f"{kind.lower()}.out", dtype=np.float32)
            got = decode(raw.reshape(1, -1)).reshape(-1)
            assert got.dtype == np.float32 and got.shape == expected.shape, (kind, got.shape, expected.shape)
            assert np.array_equal(got.view(np.uint32), expected.view(np.uint32)), (kind, np.abs(got - expected).max())


def test_decoders_match_gguf_package():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, 256)).astype(np.float32)
    for kind, decode in DECODERS.items():
        qtype = Q[kind]
        blocks = gguf.quantize(data, qtype)
        expected = gguf.dequantize(blocks, qtype)
        assert np.array_equal(decode(blocks), expected), kind


def test_gguf_matches_safetensors():
    for family, overrides in (("qwen3", {}), ("llama", {}), ("llama", {"rope_parameters": LLAMA3_ROPE})):
        model = tiny_model(family, **overrides)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            path = Path(tmp) / "model.gguf"
            write_gguf(path, model, family)
            hf = warpc.sources.open(tmp)
            source = warpc.sources.open(path)
            assert isinstance(source, GGUFSource)
            assert source.config == hf.config, (family, source.config, hf.config)
            assert (source.config.rope_freq_factors is not None) == bool(overrides), source.config
            assert check_against_gguf_package(source, path) == len(model.state_dict()), family
            for name in model.state_dict():
                assert name in source, name
                if source.quant(name) is None and not name.endswith("v_proj.weight"):  # F16 rounds bf16 subnormals
                    assert torch.equal(source.get(name), hf.get(name)), (family, name)
            gate = "model.layers.1.mlp.gate_proj.weight"
            raw = source.raw(gate)["blocks"]
            assert raw.dtype == torch.uint8 and raw.shape == (96, 64 // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES), raw.shape
            assert torch.equal(torch.from_numpy(DECODERS["Q4_0"](raw.numpy())).bfloat16(), source.get(gate))
            q = "model.layers.0.self_attn.q_proj.weight"
            try:
                source.raw(q)
            except ValueError:
                pass
            else:
                raise AssertionError("raw() of a float tensor must fail")


def test_gguf_guards():
    model = tiny_model("qwen3")
    iq = ("blk.0.attn_extra.weight", np.zeros((64, 64 // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES), np.uint8), Q.IQ4_NL)
    cases = {
        "IQ* block formats": {"arch": "qwen3", "extra": (iq,)},
        "GGUF architecture 'qwen2' (supported: llama, qwen3)": {"arch": "qwen2"},
        "15 rope frequency factors for 16 rotary pairs": {
            "arch": "qwen3",
            "extra": (("rope_freqs.weight", np.ones(15, np.float32), Q.F32),),
        },
        "rope_freqs.weight of type F16": {
            "arch": "qwen3",
            "extra": (("rope_freqs.weight", np.ones(16, np.float16), Q.F16),),
        },
        "does not read: blk.0.mystery.weight": {
            "arch": "qwen3",
            "extra": (("blk.0.mystery.weight", np.ones(4, np.float32), Q.F32),),
        },
        "GGUF architecture 'gemma'": {"arch": "gemma"},
    }
    for needle, case in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            write_gguf(path, model, case["arch"], extra=case.get("extra", ()))
            try:
                GGUFSource(path)
            except UnsupportedModel as e:
                assert needle in str(e), (needle, str(e))
            else:
                raise AssertionError(f"{needle}: accepted")


def test_real_gguf():
    path = os.environ.get("WARPC_TEST_GGUF")
    if not path:
        raise Skip("WARPC_TEST_GGUF unset")
    source = warpc.sources.open(path)
    c = source.config
    expected = 2 + (not c.tie_embeddings) + c.num_layers * (9 + 2 * c.qk_norm)
    assert check_against_gguf_package(source, Path(path)) == expected, expected
    graph = frontend.build(source, ctx_max=64)
    for decl in graph.weights.values():
        assert tuple(source.get(decl.source).shape) == decl.shape, (decl.name, decl.shape)
        assert (decl.scheme == "q4_0") == (source.quant(decl.source) == "Q4_0"), decl.name
        if decl.scheme == "q4_0":
            row_bytes = decl.shape[1] // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES
            assert source.raw(decl.source)["blocks"].shape == (decl.shape[0], row_bytes), decl.name


def test_safetensors_source():
    model = tiny_model("qwen3")
    state = model.state_dict()
    for shard in (None, "40KB"):
        with tempfile.TemporaryDirectory() as tmp:
            if shard is None:
                model.save_pretrained(tmp)
            else:
                model.save_pretrained(tmp, max_shard_size=shard)
                assert (Path(tmp) / "model.safetensors.index.json").is_file()
            source = warpc.sources.open(tmp)
            assert source.config == ModelConfig.from_hf(model.config.to_dict())
            for name, t in state.items():
                got = source.get(name)
                assert got.dtype == torch.bfloat16 and torch.equal(got, t.bfloat16()), name
                assert source.quant(name) is None
            assert "lm_head.bias" not in source
            try:
                source.raw("lm_head.weight")
            except ValueError:
                pass
            else:
                raise AssertionError("raw() of a safetensors tensor must fail")


def test_config_errors():
    base = {
        "model_type": "qwen3",
        "vocab_size": 1000,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 1,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
    }
    c = ModelConfig.from_hf(base)
    assert (c.family, c.qk_norm, c.tie_embeddings, c.rope_theta) == ("qwen3", True, False, 1e6)
    missing = dict(base)
    del missing["rms_norm_eps"], missing["vocab_size"]
    for patch, needle in (
        (missing, "missing vocab_size; missing rms_norm_eps"),
        ({**base, "model_type": "qwen2"}, "model_type 'qwen2' (supported: llama, mistral, qwen3, qwen3_5"),
        ({**base, "model_type": "gemma"}, "model_type 'gemma'"),
        ({**base, "quantization_config": {"quant_method": "compressed-tensors"}}, "'compressed-tensors'"),
        ({**base, **QWEN35_TEXT, "hidden_act": "gelu"}, "hidden_act='gelu'"),
        ({**base, **QWEN35_TEXT, "rope_parameters": {"rope_type": "yarn"}}, "rope type 'yarn'"),
        ({**base, **QWEN35_TEXT, "mamba_ssm_dtype": "bfloat16"}, "mamba_ssm_dtype='bfloat16'"),
        ({**base, **QWEN35_TEXT, "quantization_config": {**W4G128, "format": "float-quantized"}}, "pack-quantized"),
        ({**base, **QWEN35_TEXT, "quantization_config": w4g128_with(group_size=64)}, "group_size=64"),
        ({**base, **QWEN35_TEXT, "quantization_config": w4g128_with(symmetric=False)}, "symmetric=False"),
        ({**base, **QWEN35_TEXT, "quantization_config": w4g128_with(actorder="group")}, "actorder 'group'"),
        ({**base, **QWEN35_TEXT, "quantization_config": w4g128_activations()}, "input_activations"),
    ):
        try:
            ModelConfig.from_hf(patch)
        except UnsupportedModel as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError(f"{needle}: accepted")


QWEN35_TEXT = {
    "model_type": "qwen3_5_text",
    "layer_types": ["linear_attention", "full_attention"],
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_conv_kernel_dim": 4,
    "num_hidden_layers": 2,
    "attn_output_gate": True,
    "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25, "mrope_section": [1]},
}
W4G128 = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "input_activations": None,
            "weights": {"num_bits": 4, "type": "int", "strategy": "group", "group_size": 128, "symmetric": True},
        }
    },
}


def w4g128_with(**weights) -> dict:
    group = W4G128["config_groups"]["group_0"]
    return {**W4G128, "config_groups": {"group_0": {**group, "weights": {**group["weights"], **weights}}}}


def w4g128_activations() -> dict:
    group = W4G128["config_groups"]["group_0"]
    return {**W4G128, "config_groups": {"group_0": {**group, "input_activations": {"num_bits": 8}}}}


def test_qwen35_config():
    """The 27B's config.json shape: text keys nested under text_config, quantisation and vision at the top."""
    text = {**QWEN35_TEXT, "vocab_size": 1000, "hidden_size": 512, "intermediate_size": 1024, "num_attention_heads": 8,
            "num_key_value_heads": 2, "head_dim": 256, "rms_norm_eps": 1e-6, "mamba_ssm_dtype": "float32"}  # fmt: skip
    hf = {"model_type": "qwen3_5", "text_config": text, "vision_config": {"depth": 1}, "quantization_config": W4G128}
    c = ModelConfig.from_hf(hf)
    assert (c.family, c.head_dim, c.rotary_dim, c.norm_offset, c.quantization) == ("qwen3_5", 256, 64, 1.0, "w4g128")
    assert c.layer_types == ("linear_attention", "attention") and c.linear.num_v_heads == 4 and c.attn_output_gate
    want = 1.0 / (1e7 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    assert c.rope_inv_freq() == tuple(want.tolist())
    assert ModelConfig.from_hf(text).quantization is None  # a Qwen3_5ForCausalLM config.json


def write_w4g128(directory: Path, tensors: dict, n: int = 48, k: int = 256, seed: int = 0) -> dict:
    """A synthetic compressed-tensors checkpoint: one W4G128 linear "lin" packed by compressed_tensors itself, a
    bf16 tensor, and `tensors`; returns the int8 codes and bf16 scales."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(-8, 8, (n, k), generator=g, dtype=torch.int8)
    scale = (torch.rand(n, k // 128, generator=g) * 0.1 + 1e-3).bfloat16()
    save_file(
        {
            "lin.weight_packed": pack_to_int32(codes, 4),
            "lin.weight_scale": scale,
            "lin.weight_shape": torch.tensor([n, k]),
            "norm.weight": torch.randn(k, generator=g).bfloat16(),
            **tensors,
        },
        directory / "model.safetensors",
    )
    (directory / "config.json").write_text(json.dumps({**QWEN35_CONFIG, "quantization_config": W4G128}))
    return {"codes": codes, "scale": scale}


QWEN35_CONFIG = {
    **QWEN35_TEXT,
    "vocab_size": 64,
    "hidden_size": 256,
    "intermediate_size": 256,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
}


def test_w4g128_source_and_dequantize_oracle():
    """W4G128 linears by their logical name: quant / raw / contains, get() refused. The codes equal compressed_tensors'
    own unpacking; dequantize is the exact fp32 (code - 8) * scale, and compressed_tensors' dequantize (which computes
    in the scale's dtype, bf16, whatever output dtype is asked) is it rounded once."""
    try:
        from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
        from compressed_tensors.quantization.lifecycle.forward import dequantize as ct_dequantize
    except ImportError as e:
        raise Skip(f"compressed_tensors not installed: {e}") from e
    with tempfile.TemporaryDirectory() as tmp:
        want = write_w4g128(Path(tmp), {})
        source = warpc.sources.open(tmp)
        assert "lin.weight" in source and "lin.weight_packed" in source and "lin.bias" not in source
        assert source.quant("lin.weight") == "W4G128" and source.quant("norm.weight") is None
        raw = source.raw("lin.weight")
        assert raw["packed"].dtype == torch.int32 and raw["packed"].shape == (48, 32)
        assert torch.equal(raw["scale"], want["scale"])
        assert torch.equal(w4g128.unpack(raw["packed"]) - 8, want["codes"].int())
        ours = w4g128.dequantize(raw["packed"], raw["scale"])
        unpacked = unpack_from_int32(raw["packed"], 4, torch.Size([48, 256]))
        assert torch.equal(w4g128.unpack(raw["packed"]) - 8, unpacked.int())
        assert torch.equal(ours, want["codes"].float() * want["scale"].float().repeat_interleave(128, 1))
        assert torch.equal(ours.bfloat16(), ct_dequantize(unpacked, raw["scale"]))
        assert torch.equal(ours.bfloat16().float(), ct_dequantize(unpacked, raw["scale"], dtype=torch.float32))
        for call, name in ((source.get, "lin.weight"), (source.raw, "norm.weight")):
            try:
                call(name)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{call.__name__}({name}) must be refused")


def test_w4g128_source_guards():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_w4g128(Path(tmp), {})
        except ImportError as e:
            raise Skip(f"compressed_tensors not installed: {e}") from e
    cases = {
        "zero point": {"lin.weight_zero_point": torch.zeros(48, 2, dtype=torch.int32)},
        "ambiguous": {"lin.weight": torch.zeros(48, 256).bfloat16()},
        # 1 + 2^-10 needs 11 significant bits: the source must not round it to bf16 silently
        "bf16 does not represent": {"f32.weight": torch.tensor([1.0, 1.0 + 2**-10])},
    }
    for needle, extra in cases.items():
        with tempfile.TemporaryDirectory() as tmp:
            write_w4g128(Path(tmp), extra)
            try:
                source = warpc.sources.open(tmp)
                source.get("f32.weight")
            except UnsupportedModel as e:
                assert needle in str(e), (needle, str(e))
            else:
                raise AssertionError(f"{needle}: accepted")
    with tempfile.TemporaryDirectory() as tmp:
        write_w4g128(Path(tmp), {"f32.weight": torch.tensor([1.0, 0.5, -3.0])})
        assert torch.equal(warpc.sources.open(tmp).get("f32.weight"), torch.tensor([1.0, 0.5, -3.0]).bfloat16())


INKLING_OFFICIAL = Path(__file__).resolve().parent / "data" / "inkling2_mini_config.json"
INKLING_TOOL = Path(__file__).resolve().parent.parent / "tools" / "inkling_w4_config.py"


def run_inkling_tool(official: Path, directory: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(INKLING_TOOL), str(official), str(directory), *extra], capture_output=True, text=True
    )


def write_stacked(directory: Path, experts: int = 3, n: int = 128, k: int = 256) -> dict:
    """A synthetic Inkling-layout checkpoint: one stacked W4G128 expert projection [E, N, K / 8] packed by
    compressed_tensors per expert (no weight_shape, as in Inkling-2-Mini-W4), an fp32 tensor bf16 cannot represent, a
    bf16 tensor; its index and the config.json the tool writes."""
    from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    codes = torch.randint(-8, 8, (experts, n, k), generator=g, dtype=torch.int8)
    scale = (torch.rand(experts, n, k // 128, generator=g) * 0.1 + 1e-3).bfloat16()
    name = "model.llm.layers.0.mlp.experts.gate_proj"
    tensors = {
        f"{name}.weight_packed": torch.stack([pack_to_int32(c, 4) for c in codes]),
        f"{name}.weight_scale": scale,
        "model.llm.attn_res.output_query": torch.tensor([1.0, 1.0 + 2**-10, -0.1]),
        "model.llm.norm.weight": torch.randn(4, generator=g).bfloat16(),
    }
    save_file(tensors, directory / "model.safetensors")
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, "model.safetensors")}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    done = run_inkling_tool(INKLING_OFFICIAL, directory)
    assert done.returncode == 0, done.stderr
    return {"codes": codes, "scale": scale, "name": f"{name}.weight"}


def test_stacked_w4g128_source():
    """A rank-3 W4G128 tensor is E matrices: raw and dequantize equal the per-expert rank-2 results and the codes times
    the scales; get(name, float32) is exact on fp32 tensors and widens bf16; get() still refuses values bf16 rounds."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            want = write_stacked(Path(tmp))
            source = warpc.sources.open(tmp)
            raw = source.raw(want["name"])
            assert source.quant(want["name"]) == "W4G128" and source.config.quantization == "w4g128"
            assert raw["packed"].shape == (3, 128, 32) and torch.equal(raw["scale"], want["scale"])
            ours = w4g128.dequantize(raw["packed"], raw["scale"])
            per_expert = torch.stack([w4g128.dequantize(raw["packed"][e], raw["scale"][e]) for e in range(3)])
            assert torch.equal(ours, per_expert)
            assert torch.equal(ours, want["codes"].float() * want["scale"].float().repeat_interleave(128, -1))
            query = "model.llm.attn_res.output_query"
            assert torch.equal(source.get(query, torch.float32), torch.tensor([1.0, 1.0 + 2**-10, -0.1]))
            try:
                source.get(query)
            except UnsupportedModel as e:
                assert "bf16 does not represent" in str(e), str(e)
            else:
                raise AssertionError("an fp32 tensor bf16 rounds was handed out as bf16")
            norm = source.get("model.llm.norm.weight")
            assert norm.dtype == torch.bfloat16
            assert torch.equal(source.get("model.llm.norm.weight", torch.float32), norm.float())
    except ImportError as e:
        raise Skip(f"compressed_tensors not installed: {e}") from e
    # a stacked tensor needs the config's warpc_layout marker (a Qwen3.5 config does not carry it)
    with tempfile.TemporaryDirectory() as tmp:
        try:
            write_w4g128(Path(tmp), {"stack.weight_packed": torch.zeros(2, 4, 32, dtype=torch.int32),
                                     "stack.weight_scale": torch.ones(2, 4, 2).bfloat16()})  # fmt: skip
        except ImportError as e:
            raise Skip(f"compressed_tensors not installed: {e}") from e
        try:
            warpc.sources.open(tmp).raw("stack.weight")
        except UnsupportedModel as e:
            assert "warpc_layout" in str(e), str(e)
        else:
            raise AssertionError("a stacked tensor was read without the layout marker")


def test_inkling_w4_config_tool():
    """The config writer copies the official config except quantization_config, which the config guard accepts; it
    refuses a directory without an index, a packed tensor outside the experts and a config of another model."""
    official = json.loads(INKLING_OFFICIAL.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.llm.layers.0.mlp.experts.up_proj.weight_packed": "x"}})
        )
        done = run_inkling_tool(INKLING_OFFICIAL, d, "--out", str(d / "out"))
        assert done.returncode == 0 and not (d / "config.json").exists(), done.stderr
        written = json.loads((d / "out" / "config.json").read_text())
        assert {k: v for k, v in written.items() if k != "quantization_config"} == {
            k: v for k, v in official.items() if k != "quantization_config"
        }
        c = ModelConfig.from_hf(written)
        assert c.family == "inkling2_loop" and c.quantization == "w4g128", c
        assert official["quantization_config"]["quant_method"] == "modelopt"  # what the tool replaced
        other = d / "other.json"
        other.write_text(json.dumps({**official, "model_type": "qwen3"}))
        assert "not an inkling2_loop config" in run_inkling_tool(other, d).stderr
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.llm.layers.0.attn.wq_du.weight_packed": "x"}})
        )
        assert "experts only" in run_inkling_tool(INKLING_OFFICIAL, d).stderr
        (d / "model.safetensors.index.json").unlink()
        assert "no model.safetensors.index.json" in run_inkling_tool(INKLING_OFFICIAL, d).stderr


def test_inkling_checkpoint():
    """The W4 checkpoint directory: the config guard passes, and the graph declares every one of the 790 tensors with
    the dtype and shape the file header has (safetensors headers only)."""
    path = os.environ.get("WARPC_TEST_INKLING")
    if not path:
        raise Skip("WARPC_TEST_INKLING unset")
    from safetensors import safe_open

    directory = Path(path)
    if not (directory / "config.json").is_file():
        raise Skip(f"{path}/config.json missing: write it with warpc/tools/inkling_w4_config.py")
    source = warpc.sources.open(directory)
    graph = frontend.build(source, ctx_max=4096, rows_max=8)
    files = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    assert len(files) == 790
    headers: dict[str, tuple[str, tuple[int, ...]]] = {}
    for f in sorted(set(files.values())):
        with safe_open(str(directory / f), framework="pt") as h:
            for name in h.keys():  # noqa: SIM118
                s = h.get_slice(name)
                headers[name] = (s.get_dtype(), tuple(s.get_shape()))
    assert set(headers) == set(files)
    read: set[str] = set()
    for w in graph.weights.values():
        if w.scheme == "w4g128":
            *lead, n, k = w.shape
            base = w.source[: -len("weight")]
            want = {
                f"{base}weight_packed": ("I32", (*lead, n, k // 8)),
                f"{base}weight_scale": ("BF16", (*lead, n, k // 128)),
            }
        else:
            on_disk = (w.shape[0], 1, w.shape[1]) if w.name.endswith("sconv") else w.shape  # [C, 1, W] convs
            want = {w.source: ("F32" if w.scheme == "f32" else "BF16", on_disk)}
        for name, header in want.items():
            assert headers[name] == header, (w.name, name, headers[name], header)
        read.update(want)
    assert read == set(headers), sorted(set(headers) - read)[:4]
    assert (len(graph.weights), len(headers)) == (718, 790)
    with safe_open(str(directory / files["model.llm.unembed.weight"]), framework="pt") as h:
        padding = h.get_tensor("model.llm.unembed.weight")[source.config.inkling.unpadded_vocab :]
    print(f"  {padding.shape[0]} padding rows of the unembedding, max |value| {padding.float().abs().max():.4f}")
    assert padding.float().abs().max() > 0.01  # not zero: they would win the argmax if the graph kept them


def test_open_errors():
    for path, error in (
        (Path(__file__).parent / "no-such-checkpoint", FileNotFoundError),
        (Path(__file__), ValueError),  # an existing file that is not .gguf
        (Path(__file__).parent, ValueError),  # a directory without config.json
    ):
        try:
            warpc.sources.open(path)
        except error as e:
            assert str(path) in str(e), str(e)
        else:
            raise AssertionError(f"open({path}) accepted")


if __name__ == "__main__":
    run(globals())
