"""Checkpoint sources: GGUF decoders vs ggml C, GGUF reader vs gguf.dequantize, safetensors, config guards.

uv run --package warpc python warpc/tests/test_sources.py

LLAMA_CPP_DIR (a llama.cpp checkout) enables the ggml bit-identity check, WARPC_TEST_GGUF (a GGUF file) the real-file
check; each skips when unset.
"""

import os
import shutil
import subprocess
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
from warpc.sources import ModelConfig
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
    """Convert an HF model the way llama.cpp's converter lays it out (names, metadata, llama q / k permute)."""
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
    for family in ("qwen3", "llama"):
        model = tiny_model(family)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            path = Path(tmp) / "model.gguf"
            write_gguf(path, model, family)
            hf = warpc.sources.open(tmp)
            source = warpc.sources.open(path)
            assert isinstance(source, GGUFSource)
            assert source.config == hf.config, (family, source.config, hf.config)
            assert check_against_gguf_package(source, path) == len(model.state_dict()), family
            for name in model.state_dict():
                assert name in source, name
                if source.quant(name) is None and not name.endswith("v_proj.weight"):  # F16 rounds bf16 subnormals
                    assert torch.equal(source.get(name), hf.get(name)), (family, name)
            gate = "model.layers.1.mlp.gate_proj.weight"
            raw = source.raw(gate)
            assert raw.dtype == np.uint8 and raw.shape == (96, 64 // Q4_0_BLOCK_ELEMS * Q4_0_BLOCK_BYTES), raw.shape
            assert torch.equal(torch.from_numpy(DECODERS["Q4_0"](raw)).bfloat16(), source.get(gate))
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
        "rope scaling 'llama3'": {"arch": "qwen3", "extra": (("rope_freqs.weight", np.ones(16, np.float32), Q.F32),)},
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
            assert source.raw(decl.source).shape == (decl.shape[0], row_bytes), decl.name


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
        ({**base, "model_type": "qwen2"}, "model_type 'qwen2' (supported: llama, mistral, qwen3)"),
        ({**base, "model_type": "gemma"}, "model_type 'gemma'"),
        ({**base, "quantization_config": {"quant_method": "compressed-tensors"}}, "'compressed-tensors'"),
    ):
        try:
            ModelConfig.from_hf(patch)
        except UnsupportedModel as e:
            assert needle in str(e), (needle, str(e))
        else:
            raise AssertionError(f"{needle}: accepted")


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
