"""The CUDA backend end to end: the fused Q4_0 GEMV against its CPU reference (kernels/tests/test_q4_gemv.c), compiled
greedy decode of tiny random models of every family (and every KV format) against the reference interpreter, the
runtime's and the facade's contract, and a real GGUF file.

  uv run --package warpc python warpc/tests/test_cuda.py

Needs a CUDA device and nvcc (NVCC / PATH / CUDA_HOME); the GPU tests skip without a device. WARPC_TEST_GGUF (a GGUF
file) enables the real-file check: the compiled tokens equal the reference interpreter's, and for Qwen3-0.6B Q4_0 the
known 24 tokens. WARPC_TEST_SWIFT27 (the Qwen3.5 27B W4G128 checkpoint) enables its reference interpreter on the GPU
(needs the card alone: about 18 GB).
"""

import gc
import os
import subprocess
import tempfile
import weakref
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import logging as transformers_logging

import warpc
from harness import Skip, run
from warpc import frontend, sources
from warpc.cuda import glue, schedule
from warpc.reference import Interpreter

transformers_logging.disable_progress_bar()

PROMPT = [785, 6722, 315, 9625, 374]  # "The capital of France is" (Qwen3 tokenizer)
# llama.cpp's greedy continuation of PROMPT on Qwen3-0.6B-Q4_0.gguf
QWEN3_06B_Q4_0 = [
    12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279,
    3146, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279,
]  # fmt: skip
TINY_PROMPT = [7, 300, 42, 5, 511, 0, 99, 123, 8, 64, 250]
SIZES = {
    "vocab_size": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6,
    "max_position_embeddings": 32768,
}
MODELS = {
    "llama": lambda: LlamaForCausalLM(
        LlamaConfig(
            **SIZES,
            num_attention_heads=4,
            num_key_value_heads=2,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            tie_word_embeddings=False,
        )
    ),
    "mistral": lambda: MistralForCausalLM(
        MistralConfig(
            **SIZES,
            num_attention_heads=4,
            num_key_value_heads=1,
            sliding_window=None,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
        )
    ),
    "qwen3": lambda: Qwen3ForCausalLM(
        Qwen3Config(
            **SIZES,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
            tie_word_embeddings=True,
        )
    ),
    # head dim 128: the one-row decode runs the fused prep + split kernel; pairs 17 ... 63 are divided by 8, e.g.
    # pair 17 turns 0.28 instead of 2.25 radians at position 26
    "llama_rope_llama3": lambda: LlamaForCausalLM(
        LlamaConfig(
            **SIZES,
            num_attention_heads=2,
            num_key_value_heads=1,
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
    # head dim 256, the hand engine's: KV format int4_rot is its cache byte for byte
    "qwen3_hd256": lambda: Qwen3ForCausalLM(
        Qwen3Config(
            **SIZES,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=256,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
            tie_word_embeddings=True,
        )
    ),
}
_CHECKPOINTS = tempfile.TemporaryDirectory()


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


def tiny_source(family: str):
    """A tiny random model with bf16-representable weights, saved once as an HF safetensors checkpoint."""
    directory = Path(_CHECKPOINTS.name) / family
    if not directory.exists():
        model = MODELS[family]()
        torch.manual_seed(0)
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_((torch.randn_like(p) * (0.3 if "norm" in n else p.shape[-1] ** -0.5)).bfloat16().float())
        model.save_pretrained(directory)
    return sources.open(directory)


def reference_tokens(source, ids, n: int, kv: str = "bf16") -> list[int]:
    graph = frontend.build(source, ctx_max=len(ids) + n, kv=kv)
    return Interpreter(graph, source, dtype=torch.float32).generate(ids, n)


def stepwise_reference(source, ids, n: int, kv: str, ctx: int) -> tuple[list[int], torch.Tensor]:
    """Greedy tokens and the logits after the prompt from the reference interpreter in bf16 on the GPU, fed the prompt
    in the runtime's steps of STEP_ROWS rows: int4_rot's exact window is counted from a step's last row, so the steps
    are part of the result past 496 positions, and bf16 activations give the cache the kernels' K and V rows."""
    it = Interpreter(frontend.build(source, ctx_max=ctx, kv=kv), source, device="cuda", dtype=torch.bfloat16)
    rows = schedule.STEP_ROWS
    for s in range(0, len(ids), rows):
        chunk = ids[s : s + rows]
        pos = torch.arange(s, s + len(chunk), dtype=torch.int32)
        out = it.run({"tokens": torch.tensor(chunk, dtype=torch.int32), "pos": pos})
    tokens, logits = [int(out["ids"][-1])], out["logits"][-1].float().cpu()
    while len(tokens) < n:
        pos = torch.tensor([len(ids) + len(tokens) - 1], dtype=torch.int32)
        tokens.append(int(it.run({"tokens": torch.tensor(tokens[-1:], dtype=torch.int32), "pos": pos})["ids"][-1]))
    return tokens, logits


def expect(error: type[Exception], fn, *args) -> str:
    try:
        fn(*args)
    except error as e:
        return str(e)
    raise AssertionError(f"{fn.__name__}{args} must raise {error.__name__}")


def test_q4_gemv_kernel_is_bit_identical_to_its_cpu_reference():
    require_gpu()
    major, minor = torch.cuda.get_device_capability()
    lib = glue.build(10 * major + minor)
    with tempfile.TemporaryDirectory() as d:
        exe = Path(d) / "test_q4_gemv"
        src = glue.KERNELS / "tests" / "test_q4_gemv.c"
        link = [f"-L{lib.parent}", f"-l:{lib.name}", "-lm"]
        subprocess.run(
            [glue.nvcc(), "-O2", "-Xcompiler", "-ffp-contract=off", "-o", str(exe), str(src), *link], check=True
        )
        env = {**os.environ, "LD_LIBRARY_PATH": str(lib.parent)}
        r = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
    print(r.stdout.strip())
    assert r.returncode == 0 and r.stdout.strip().endswith("PASS"), r.stdout + r.stderr


def test_tiny_models_decode_like_the_reference():
    require_gpu()
    # 11 tokens: an 8-row and a 2-row prompt step, then the head step
    cases = [("llama", "bf16", 256), ("mistral", "bf16", 256), ("qwen3", "bf16", 256)]
    cases += [("qwen3", kv, 256) for kv in ("k8v8", "k8v8w", "k8v4w")] + [("llama", "k8v4w", 512)]
    cases += [("llama_rope_llama3", "bf16", 256), ("llama_rope_llama3", "k8v4w", 512)]
    cases += [("qwen3", "bf16", 16384), ("llama", "bf16", 16384)]  # more than 8192 positions: the chunked kernels
    cases += [(family, "int4_rot", 256) for family in ("llama", "mistral", "qwen3", "qwen3_hd256")]
    for family, kv, ctx in cases:
        source = tiny_source(family)
        model = warpc.compile(source, ctx_max=ctx, kv=kv)
        got = model.generate(TINY_PROMPT, 16)
        want = reference_tokens(source, TINY_PROMPT, 16, kv)
        assert got == want, (family, kv, ctx, got, want)
        assert model.generate(TINY_PROMPT, 16) == got, "generate must start from an empty cache"
        print(f"  {family} kv={kv} ctx={ctx}: {got}")


def test_int4_rot_past_the_exact_window_and_16k_positions():
    """A 16,500-token prompt: past 496 positions the 4-bit codes carry the attention, past 8192 the chunked split
    kernel runs; head dims 64, 128 and 256."""
    require_gpu()
    prompt = torch.randint(0, SIZES["vocab_size"], (16500,), generator=torch.Generator().manual_seed(5)).tolist()
    ctx = len(prompt) + 8
    for family in ("llama", "qwen3", "qwen3_hd256"):
        source = tiny_source(family)
        model = warpc.compile(source, ctx_max=ctx, kv="int4_rot")
        want, want_logits = stepwise_reference(source, prompt, 8, "int4_rot", ctx)
        model.prefill(prompt)
        err = float((model.logits.float().cpu() - want_logits).abs().max())
        # measured 0.0055 (llama) and 0.0040 (hd 256), as close as the bf16 cache's 0.0050 and 0.0040; the same
        # kernels against the fp32 reference: 0.014 and 0.019; int4_rot against the bf16 cache: 0.036 and 0.043
        assert err < 0.01, (family, err)
        got = model.generate(prompt, 8)
        assert got == want, (family, got, want)
        print(f"  {family}: {got}, logits within {err:.4f}")


def test_dropped_runtimes_are_freed_by_refcount():
    """A dropped model frees its runtime (device memory, CUDA graphs) at once: a runtime left to the cyclic collector
    can be torn down inside another runtime's capture, which invalidates that capture. The failing sequence was models
    compiled and dropped back to back, then a new one."""
    require_gpu()
    source = tiny_source("qwen3")
    gc.disable()  # only refcounting may free them
    try:
        for kv in ("bf16", "k8v8", "bf16", "k8v4w"):
            model = warpc.compile(source, ctx_max=256, kv=kv)
            runtime = weakref.ref(model._runtime)
            model.generate(TINY_PROMPT, 4)
            del model
            assert runtime() is None, f"kv={kv}: a dropped runtime outlives its model (a reference cycle)"
    finally:
        gc.enable()
    model = warpc.compile(source, ctx_max=256, kv="k8v8")
    assert model.generate(TINY_PROMPT, 16) == reference_tokens(source, TINY_PROMPT, 16, "k8v8")


def test_prompt_chunks_follow_rows_max():
    require_gpu()
    source = tiny_source("qwen3")
    want = reference_tokens(source, TINY_PROMPT, 8)
    for rows_max in (1, 3):  # 10 prompt rows: 10 single-row steps, or chunks of 3, 3, 3, 1
        model = warpc.compile(source, ctx_max=64, rows_max=rows_max)
        assert model.plan.rows_max == rows_max
        assert model.generate(TINY_PROMPT, 8) == want, rows_max


def test_runtime_and_facade_contract():
    require_gpu()
    model = warpc.compile(tiny_source("llama"), ctx_max=16)
    assert model.generate(TINY_PROMPT, 0) == []
    got = model.generate(TINY_PROMPT, 3)
    assert int(torch.argmax(model.logits)) == got[-1] and model.logits.shape == (SIZES["vocab_size"],)
    assert model.decode(2) == reference_tokens(tiny_source("llama"), TINY_PROMPT, 5)[3:]  # 11 + 2 + 2 cached: 15
    assert "exceed the KV cache of 16 positions" in expect(ValueError, model.decode, 2)
    model.step(got[0])  # the 16th position
    assert "exceed" in expect(ValueError, model.step, got[0])
    model.reset()
    assert "prefill needs at least one token" in expect(ValueError, model.prefill, [])
    assert "cache is empty" in expect(ValueError, model.decode, 1)
    assert "exceed" in expect(ValueError, model.prefill, list(range(17)))
    assert model.prefill(TINY_PROMPT) == got[0] and model.step(got[0]) == got[1]


def test_gguf_file():
    path = os.environ.get("WARPC_TEST_GGUF")
    if not path:
        raise Skip("WARPC_TEST_GGUF unset")
    require_gpu()
    source = sources.open(path)
    model = warpc.compile(source, ctx_max=512)
    got = model.generate(PROMPT, 24)
    print(f"  {Path(path).name}: {model.plan.summary()}\n  {got}")
    assert got == reference_tokens(source, PROMPT, 24), got
    c = source.config
    qwen3_06b = c.family == "qwen3" and (c.hidden_size, c.num_layers, c.vocab_size) == (1024, 28, 151936)
    if qwen3_06b and source.quant("model.layers.0.self_attn.q_proj.weight") == "Q4_0":
        assert got == QWEN3_06B_Q4_0, got
    model.reset()
    first = model.prefill(PROMPT)
    assert first == got[0] and int(torch.argmax(model.logits)) == first and model.logits.shape == (c.vocab_size,)
    assert model.step(first) == got[1] and model.step(got[1]) == got[2]


def test_qwen35_27b_reference_on_the_gpu():
    """The 27B reference fits the GPU (W4G128 linears stay packed, dequantised per use) and continues a factual prompt
    sensibly; it is the tolerance oracle of the compiled 27B (the bit-exact one is the hand engine)."""
    path = os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        raise Skip("WARPC_TEST_SWIFT27 unset")
    require_gpu()
    import time

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    ids = tok("The capital of France is", add_special_tokens=False)["input_ids"]
    source = sources.open(path)
    t0 = time.perf_counter()
    it = Interpreter(frontend.build(source, ctx_max=64), source, device="cuda", dtype=torch.bfloat16)
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = it.generate(ids, 8)
    torch.cuda.synchronize()
    text = tok.decode(got)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(
        f"  load {load:.0f} s, {len(ids)} + 8 tokens in {time.perf_counter() - t0:.1f} s, peak {peak:.1f} GiB: {text!r}"
    )
    assert "Paris" in text, (got, text)


if __name__ == "__main__":
    run(globals())
