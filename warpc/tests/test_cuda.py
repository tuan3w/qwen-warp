"""The CUDA backend end to end: the fused Q4_0 / Q4_1 and Q6_K GEMVs against their CPU references (kernels/tests), the
repacked weight layouts, compiled greedy decode of tiny random models of every family (and every KV format) and of a
tiny GGUF file in the fused block formats against the reference interpreter, the runtime's and the facade's contract,
and a real GGUF file.

  uv run --package warpc python warpc/tests/test_cuda.py

Needs a CUDA device and nvcc (NVCC / PATH / CUDA_HOME); the GPU tests skip without a device. WARPC_TEST_GGUF (a GGUF
file) enables the real-file check: the compiled tokens equal the reference interpreter's, and for Qwen3-0.6B Q4_0 the
known 24 tokens. WARPC_TEST_SWIFT27 (the Qwen3.5 27B W4G128 checkpoint) enables its reference interpreter on the GPU
(needs the card alone: about 18 GB).
"""

import gc
import json
import os
import subprocess
import tempfile
import weakref
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType as Q
from transformers import LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM, Qwen3Config, Qwen3ForCausalLM
from transformers.utils import logging as transformers_logging

import warpc
from harness import Skip, run
from test_sources import write_gguf
from warpc import frontend, sources
from warpc.cuda import glue, pack, schedule
from warpc.reference import Interpreter
from warpc.sources.gguf import DECODERS

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


def test_fused_dequant_kernels_are_bit_identical_to_their_cpu_references():
    require_gpu()
    major, minor = torch.cuda.get_device_capability()
    lib = glue.build(10 * major + minor)
    for name in ("test_q4_gemv", "test_q6_gemv"):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / name
            src = glue.KERNELS / "tests" / f"{name}.c"
            link = [f"-L{lib.parent}", f"-l:{lib.name}", "-lm"]
            subprocess.run(
                [glue.nvcc(), "-O2", "-Xcompiler", "-ffp-contract=off", "-o", str(exe), str(src), *link], check=True
            )
            env = {**os.environ, "LD_LIBRARY_PATH": str(lib.parent)}
            r = subprocess.run([str(exe)], capture_output=True, text=True, env=env)
        print(r.stdout.strip())
        assert r.returncode == 0 and r.stdout.strip().endswith("PASS"), r.stdout + r.stderr


def unpack_q4(planes: torch.Tensor, K: int, minimums: bool) -> torch.Tensor:
    """The fp32 weights [N, K] of rows repacked by pack.repack_q4, read back by the layout of dense_q4_gemv.cuh."""
    n, pairs, kb = planes.shape[0], K // 512, K // 32
    words = planes[:, : K // 2].reshape(n, pairs, 32, 2, 4).permute(0, 1, 3, 2, 4).reshape(n, K // 8, 4)
    codes = torch.stack([words & 15, words >> 4], -1).reshape(n, K).float()  # column order

    def head(h: int) -> torch.Tensor:
        plane = planes[:, K // 2 + h * (K // 16) : K // 2 + (h + 1) * (K // 16)]
        values = plane.reshape(n, pairs, 8, 2, 2).permute(0, 1, 3, 2, 4).reshape(n, kb, 2).contiguous()
        return values.view(torch.float16).float().expand(n, kb, 32).reshape(n, K)

    return head(0) * codes + head(1) if minimums else head(0) * (codes - 8)


def unpack_q6_k(planes: torch.Tensor, K: int) -> torch.Tensor:
    """The fp32 weights [N, K] of rows repacked by pack.repack_q6_k, read back by the layout of dense_q6_gemv.cuh."""
    n, kb = planes.shape[0], K // 256
    ql = planes[:, : 128 * kb].reshape(n, kb, 32, 4)
    low = torch.stack([ql & 15, ql >> 4], -1).reshape(n, kb, 32, 8)
    qh = planes[:, 128 * kb : 192 * kb].reshape(n, kb, 32, 2)
    high = torch.stack([(qh >> 2 * m) & 3 for m in range(4)], -1).reshape(n, kb, 32, 8)
    codes = (low | (high << 4)).reshape(n, kb, 16, 16).float() - 32
    sc = planes[:, 192 * kb : 208 * kb].reshape(n, kb, 16).contiguous().view(torch.int8).float()
    d = planes[:, 208 * kb : 210 * kb].reshape(n, kb, 2).contiguous().view(torch.float16).float()
    return ((d * sc).reshape(n, kb, 16, 1) * codes).reshape(n, K)


def test_repacked_rows_hold_the_gguf_weights():
    rng = np.random.default_rng(0)
    for qtype, scheme in ((Q.Q4_0, "q4_0"), (Q.Q4_1, "q4_1")):
        K = 1536
        raw = gguf.quantize(rng.standard_normal((5, K)).astype(np.float32), qtype)
        planes = pack.repack_q4(torch.from_numpy(raw), scheme)
        assert planes.shape == (5, schedule.packed_row_bytes(scheme, K)) and planes.shape[1] == raw.shape[1]
        got = unpack_q4(planes, K, scheme == "q4_1")
        assert torch.equal(got, torch.from_numpy(DECODERS[qtype.name](raw))), scheme
    K = 2560
    raw = rng.integers(0, 256, (7, K // 256 * 210), dtype=np.uint8)
    raw.reshape(7, -1, 210)[:, :, 208:] = np.array([0x40, 0x1C], dtype=np.uint8)  # a finite d
    planes = pack.repack_q6_k(torch.from_numpy(raw))
    assert planes.shape == (7, schedule.packed_row_bytes("q6_k", K))
    want = torch.from_numpy(np.asarray(gguf.dequantize(raw, Q.Q6_K), dtype=np.float32))
    assert torch.equal(unpack_q6_k(planes, K), want)


def block_format_model():
    """A tiny llama whose linears are Q4_0 (down_proj Q4_1) and whose head is Q6_K, written as a GGUF file; the
    widths are the fused kernels' (K a multiple of 512)."""
    config = LlamaConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        vocab_size=1024,
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_((torch.randn_like(p) * (0.3 if "norm" in n else p.shape[-1] ** -0.5)).bfloat16().float())
    rng = np.random.default_rng(1)
    head = rng.integers(0, 256, (1024, 512 // 256 * 210), dtype=np.uint8)
    head.reshape(1024, -1, 210)[:, :, 208:] = np.array([0x00, 0x14], dtype=np.uint8)  # d ~ 6.1e-4, scales up to 127
    types = dict.fromkeys(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"), Q.Q4_0)
    types = {f"{k}.weight": v for k, v in types.items()} | {"down_proj.weight": Q.Q4_1}
    return model, types, {"lm_head.weight": (head, Q.Q6_K)}


def test_block_formats_decode_like_the_reference():
    require_gpu()
    model, types, blocks = block_format_model()
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.gguf"
        write_gguf(path, model, "llama", types=types, blocks=blocks)
        source = sources.open(path)
        compiled = warpc.compile(source, ctx_max=64)
        summary = compiled.plan.summary()
        got = compiled.generate(TINY_PROMPT, 16)
        want = reference_tokens(source, TINY_PROMPT, 16)
    print(f"  {summary}\n  {got}")
    kinds = {t.kernel for t in compiled.plan.tasks}
    assert {"g_q4gemv", "g_q4gemv_dual", "g_q4gemv_resadd", "g_head_q6k"} <= kinds, summary
    assert {t.attrs["fmt"] for t in compiled.plan.tasks if t.kernel == "g_q4gemv_resadd"} == {0, 1}, summary
    assert got == want, (got, want)


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
        model.append(prompt)
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
    assert "append needs at least one token" in expect(ValueError, model.append, [])
    assert "cache is empty" in expect(ValueError, model.decode, 1)
    assert "exceed" in expect(ValueError, model.append, list(range(17)))
    assert model.append(TINY_PROMPT) == got[0] and model.step(got[0]) == got[1]


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
    first = model.append(PROMPT)
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


INKLING_IDS = json.loads((Path(__file__).resolve().parent / "data" / "inkling_quality_ids.json").read_text())


def inkling_env() -> Path:
    """The W4 checkpoint directory of $WARPC_TEST_INKLING."""
    path = os.environ.get("WARPC_TEST_INKLING")
    if not path:
        raise Skip("WARPC_TEST_INKLING unset")
    require_gpu()
    return Path(path)


def agreement(ref: list[torch.Tensor], got: list[torch.Tensor]) -> dict:
    """Over every predicted position of the quality texts (the logits at positions 0..n-2 predict ids 1..n-1): KL(ref |
    got), top-1 agreement, how many disagreements are near ties of ref (top-2 gap < 0.5), both perplexities."""
    kl, agree, near, nll_ref, nll_got = [], [], [], [], []
    for ids, a, b in zip(INKLING_IDS, ref, got, strict=True):
        a, b = a[:-1].float().cuda(), b[:-1].float().cuda()
        lp, lq = torch.log_softmax(a, -1), torch.log_softmax(b, -1)
        kl.append((lp.exp() * (lp - lq)).sum(-1))
        agree.append(a.argmax(-1) == b.argmax(-1))
        top2 = a.topk(2, -1).values
        near.append(top2[:, 0] - top2[:, 1] < 0.5)
        target = torch.tensor(ids[1:], device="cuda")
        nll_ref.append(-lp.gather(1, target[:, None])[:, 0])
        nll_got.append(-lq.gather(1, target[:, None])[:, 0])
    kl, agree, near = torch.cat(kl), torch.cat(agree), torch.cat(near)
    return {
        "positions": kl.numel(),
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "top1": agree.float().mean().item(),
        "disagreements": int((~agree).sum()),
        "disagreements_not_near_tie": int((~agree & ~near).sum()),
        "ppl_ref": torch.cat(nll_ref).mean().exp().item(),
        "ppl_got": torch.cat(nll_got).mean().exp().item(),
    }


def inkling_reference_logits(path: Path) -> list[torch.Tensor]:
    """The reference interpreter (bf16, on the GPU, the real INT4 weights) on the four quality texts: each text's logits
    [n, 200058] f32 on the host. Needs the card alone (about 14 GB)."""
    source = sources.open(path)
    it = Interpreter(frontend.build(source, ctx_max=256, rows_max=160), source, device="cuda", dtype=torch.bfloat16)
    out = []
    for ids in INKLING_IDS:
        it.reset()
        n = len(ids)
        out.append(
            it.run({"tokens": torch.tensor(ids, dtype=torch.int32), "pos": torch.arange(n, dtype=torch.int32)})[
                "logits"
            ].cpu()
        )
    return out


def test_inkling_reference_real():
    """The reference on the real INT4 weights: the perplexity of the four texts (622 predicted positions) is the model's
    (the old tree measured 3.710 with its own interpreter), and, when $WARPC_INKLING_QUALITY holds that interpreter's
    logit dumps (ref-{0..3}.pt), their agreement is printed. Measured against the dumps: KL mean 0.00288, top-1 98.07%,
    12 disagreements of which 1 not a near tie, perplexity 3.6996. That is a floor of this model, not an error: the old
    store rounded the dequantised expert weights to bf16 and multiplied in bf16, this one multiplies the fp32
    dequantisation. With the old rounding emulated the same code gives KL 0.00195, top-1 99.04%, 0 disagreements that
    are not near ties, and its first visit differs from the dump by 1e-5 (rel. L2), against 3e-3 with fp32 experts: the
    two fp32 / bf16 expert variants of THIS code differ by KL 0.00286. No bound is asserted on the dumps."""
    path = inkling_env()
    import time

    t0 = time.perf_counter()
    got = inkling_reference_logits(path)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    assert [g.shape for g in got] == [(len(ids), 200058) for ids in INKLING_IDS], [g.shape for g in got]
    nll = []
    for ids, g in zip(INKLING_IDS, got, strict=True):
        lp = torch.log_softmax(g[:-1].float().cuda(), -1)
        nll.append(-lp.gather(1, torch.tensor(ids[1:], device="cuda")[:, None])[:, 0])
    ppl = torch.cat(nll).mean().exp().item()
    print(f"  4 texts in {time.perf_counter() - t0:.0f} s, peak {peak:.1f} GiB, perplexity {ppl:.4f}")
    assert 3.65 <= ppl <= 3.75, ppl  # the old tree's 3.710 within 1.1%
    quality = os.environ.get("WARPC_INKLING_QUALITY")
    if quality:
        old = [torch.load(Path(quality) / f"ref-{p}.pt")["logits"][: len(ids)] for p, ids in enumerate(INKLING_IDS)]
        print(f"  against the old interpreter's dumps: {agreement(old, got)}")


def dump_inkling_reference(path: Path, out: Path) -> None:
    torch.save(inkling_reference_logits(path), out)


def prefill_logits(model, ids, rows: int) -> torch.Tensor:
    """The logits after prefilling `ids` in steps of `rows` rows (the plan holds the graphs of 1..rows_max)."""
    rt = model._runtime
    saved, rt._rows_max = rt._rows_max, rows
    try:
        model.reset()
        model.append(ids)
    finally:
        rt._rows_max = saved
    return model.logits.clone()


def test_inkling_tiny():
    """A tiny Inkling-2 (real widths, 8 layers = 16 visits, a window of 16 that the 36 positions wrap, 16 experts)
    compiled onto the CUDA table, one row per step along the reference's greedy continuation of a 13-token prompt. A
    random model of this depth amplifies bf16 rounding (the real one does too: DESIGN 7h) so free-running ids part
    after a few tokens; the gate is the model's own noise instead: at every generated position the compiled logits are
    within 3 times the bf16 reference's distance from the fp32 reference (plus 0.01) of the bf16 reference, on average
    within 1.5 times, and the compiled id is the reference's wherever the reference's top-2 gap is above twice the
    largest logit difference between the two references. Logits after a 33-token prompt are bit-identical whether the
    prompt goes in as steps of 8, 4 or 1 rows."""
    require_gpu()
    from inkling_common import random_inkling

    source = random_inkling(0)
    prompt = torch.randint(0, 2000, (13,), generator=torch.Generator().manual_seed(1)).tolist()
    model = warpc.compile(source, ctx_max=64, rows_max=8)
    graph = frontend.build(source, ctx_max=64, rows_max=36)
    bf = Interpreter(graph, source, device="cuda", dtype=torch.bfloat16)
    fp = Interpreter(graph, source, device="cuda", dtype=torch.float32)
    want = bf.generate(prompt, 24)
    ids = prompt + want[:23]
    window = {"tokens": torch.tensor(ids, dtype=torch.int32), "pos": torch.arange(len(ids), dtype=torch.int32)}
    ref_b, ref_f = (it.reset() or it.run(window)["logits"].float() for it in (bf, fp))
    model.reset()
    rel_c, rel_f, agree, decided = [], [], 0, 0
    for pos, token in enumerate(ids):
        model.step(token)
        if pos < len(prompt) - 1:
            continue
        got, b, f = model.logits.float(), ref_b[pos], ref_f[pos]
        norm = b.norm()
        rel_c.append(((got - b).norm() / norm).item())
        rel_f.append(((b - f).norm() / norm).item())
        assert rel_c[-1] <= 3 * rel_f[-1] + 0.01, (pos, rel_c[-1], rel_f[-1])
        top = b.topk(2).values
        if top[0] - top[1] > 2 * (b - f).abs().max():
            decided += 1
            agree += int(got.argmax()) == int(b.argmax())
    mean_c, mean_f = sum(rel_c) / len(rel_c), sum(rel_f) / len(rel_f)
    print(
        f"  {model.plan.summary()}\n  rel L2 to the bf16 reference {mean_c:.4f} on average (bf16 against fp32 reference"
        f" {mean_f:.4f}); {agree} of {decided} decided ids (of {len(rel_c)}) equal"
    )
    assert mean_c <= 1.5 * mean_f and agree == decided and decided >= 8, (mean_c, mean_f, agree, decided)
    ids = prompt + want[:20]
    windows = {rows: prefill_logits(model, ids, rows) for rows in (8, 4, 1)}
    assert torch.equal(windows[8], windows[4]) and torch.equal(windows[8], windows[1]), {
        r: (windows[r] - windows[8]).abs().max().item() for r in windows
    }


def test_inkling_compiled_real():
    """The compiled Inkling-2-Mini (all 15.8 GB of INT4 weights on the card, 340 tasks per step) against the reference
    interpreter on the four quality texts (622 predicted positions, one row per step): KL mean at most 0.0033, top-1 at
    least 98% with every disagreement a near tie of the reference, and the logits after a 41-token prompt
    bit-identical for steps of 8, 4 and 1 rows. The reference runs in a process of its own first (it needs the card
    alone).

    0.0033 is 1.25 times the old tree's noise floor of this model (0.00263: its interpreter against itself with the
    embedding table perturbed by 2^-9); the design's 0.0026 sat below the floor, and one realisation of any two correct
    bf16 implementations lands anywhere in 0.0019 - 0.0030 (the mean of 622 heavy-tailed positions: one position of
    the first text has KL 0.19). Measured: KL(reference | compiled) 0.00301, top-1 98.23% (11 disagreements, all near
    ties; the old compiled model: 98.23%, 11), perplexity 3.6996 -> 3.7078 (old: 3.7102 -> 3.7093)."""
    path = inkling_env()
    import sys
    import time

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "reference.pt"
        t0 = time.perf_counter()
        done = subprocess.run(
            [sys.executable, __file__, "--inkling-reference", str(path), str(out)], capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr[-2000:]
        ref = torch.load(out)
    print(f"  reference in {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    model = warpc.compile(path, ctx_max=256, rows_max=8)
    print(f"  compiled in {time.perf_counter() - t0:.0f} s: {model.plan.summary()}")
    t0 = time.perf_counter()
    got = []
    for ids in INKLING_IDS:
        model.reset()
        rows = []
        for t in ids:
            model.step(t)
            rows.append(model.logits.clone().cpu())
        got.append(torch.stack(rows))
    r = agreement(ref, got)
    print(f"  {sum(map(len, INKLING_IDS))} steps in {time.perf_counter() - t0:.1f} s: {r}")
    assert r["positions"] == 622 and r["kl_mean"] <= 0.0033 and r["top1"] >= 0.98, r
    assert r["disagreements_not_near_tie"] == 0, r
    windows = {rows: prefill_logits(model, INKLING_IDS[0][:41], rows) for rows in (8, 4, 1)}
    assert torch.equal(windows[8], windows[4]) and torch.equal(windows[8], windows[1]), {
        k: (v - windows[8]).abs().max().item() for k, v in windows.items()
    }


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["--inkling-reference"]:
        dump_inkling_reference(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        run(globals())
