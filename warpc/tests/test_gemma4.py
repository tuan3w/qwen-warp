"""Gemma 4 text decoder: config guard, ring KV reference, frontend + reference interpreter vs transformers on a tiny
random fp32 model (every layer output and the logits, chunks through wrapping rings, 16 injected slips), the state
reads of attention_read in the compiler.

uv run --package warpc python warpc/tests/test_gemma4.py

WARPC_TEST_GEMMA4 (a Gemma 4 E2B text checkpoint directory) enables the real-checkpoint header and graph checks.
"""

import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import DynamicCache, Gemma4ForCausalLM, Gemma4TextConfig
from transformers.utils import logging as transformers_logging

import warpc.sources
from harness import Skip, run
from warpc import frontend
from warpc.compiler import Region, lower
from warpc.compiler.fuse import fuse
from warpc.compiler.lower import _check_state_order
from warpc.cuda import CudaBackend
from warpc.errors import LoweringError, UnsupportedModel
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.kvcache import ExactCache
from warpc.reference import _IMPL, Interpreter
from warpc.runtime import _live_tasks
from warpc.sources import ModelConfig

transformers_logging.disable_progress_bar()

S, F_ = "sliding_attention", "full_attention"
GATE = 1e-4
T = 24
# the old tree's tiny config: window 6 < 24 tokens, 3 sharing layers of both types with the wide MLP, PLE 8, softcap
TINY = {
    "vocab_size": 128,
    "vocab_size_per_layer_input": 128,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "global_head_dim": 32,
    "hidden_size_per_layer_input": 8,
    "num_kv_shared_layers": 3,
    "use_double_wide_mlp": True,
    "sliding_window": 6,
    "layer_types": [S, S, F_, S, S, F_],
    "final_logit_softcapping": 30.0,
    "max_position_embeddings": 256,
}
E2B_ROPE = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1000000.0,
        "rope_type": "proportional",
    },
    "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
}


def e2b_text_config() -> dict:
    """The text_config of google/gemma-4-E2B, the fields the guard and the graph read."""
    return {
        "attention_bias": False,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "final_logit_softcapping": 30.0,
        "global_head_dim": 512,
        "head_dim": 256,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": 1536,
        "hidden_size_per_layer_input": 256,
        "intermediate_size": 6144,
        "layer_types": ([S] * 4 + [F_]) * 7,
        "model_type": "gemma4_text",
        "num_attention_heads": 8,
        "num_global_key_value_heads": None,
        "num_hidden_layers": 35,
        "num_key_value_heads": 1,
        "num_kv_shared_layers": 20,
        "rms_norm_eps": 1e-6,
        "rope_parameters": E2B_ROPE,
        "sliding_window": 512,
        "tie_word_embeddings": True,
        "use_bidirectional_attention": None,
        "use_double_wide_mlp": True,
        "vocab_size": 262144,
        "vocab_size_per_layer_input": 262144,
    }


def make_hf(kw: dict, seed: int = 0):
    """A random fp32 Gemma4ForCausalLM with bf16-representable weights (the source hands out bf16), large logits so the
    softcap is in its nonlinear range, a learned layer scalar that is not 1."""
    cfg = Gemma4TextConfig(**kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = Gemma4ForCausalLM(cfg).float().eval()

    def bf16(t):
        return t.bfloat16().float()

    with torch.no_grad():
        for n, p in model.named_parameters():
            if "embed_tokens_per_layer" in n:
                p.copy_(bf16(torch.randn_like(p) * 0.5))
            elif "embed_tokens" in n:
                p.copy_(bf16(torch.randn_like(p) * 0.3))
            elif n == "model.norm.weight":
                p.copy_(bf16(5.0 + 0.5 * torch.randn_like(p)))
            elif p.dim() == 1:
                p.copy_(bf16(1.0 + 0.2 * torch.randn_like(p)))  # incl. the q / k norms that carry the attention scale
            else:
                p.copy_(bf16(torch.randn_like(p) * p.shape[-1] ** -0.5))
        for n, b in model.named_buffers():
            if n.endswith("layer_scalar"):
                b.copy_(bf16(0.7 + 0.6 * torch.rand_like(b)))
    return cfg, model


def hf_run(model, ids: torch.Tensor):
    """logits [T, V] and the output of every decoder layer [L x [T, H]] of the official model."""
    outs = []
    hooks = [
        layer.register_forward_hook(
            lambda mod, args, kwargs, out: outs.append((out[0] if isinstance(out, tuple) else out)[0].detach().clone()),
            with_kwargs=True,
        )
        for layer in model.model.layers
    ]
    try:
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0]
    finally:
        for h in hooks:
            h.remove()
    return logits, outs


def rel(a, b) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def with_layer_outputs(graph: Graph) -> Graph:
    """The graph with the output of every layer (the layer scalar's result) as an extra graph output h{i:02d}."""
    graph = copy.deepcopy(graph)
    for n in graph.nodes:
        if n.op == "scale" and n.attrs.get("weight", "").endswith(".scalar"):
            graph.output(**{f"h{int(n.attrs['weight'].split('.')[0][1:]):02d}": n.outs[0]})
    return graph


def feed(ids: torch.Tensor, a: int, b: int) -> dict:
    return {"tokens": ids[a:b].int(), "pos": torch.arange(a, b, dtype=torch.int32)}


class Tiny:
    """A tiny model saved as a checkpoint, with its source and the official model."""

    def __init__(self, kw: dict = TINY):
        self.kw = kw
        self.cfg, self.hf = make_hf(kw)
        self._tmp = tempfile.TemporaryDirectory()
        self.hf.save_pretrained(self._tmp.name)
        self.source = warpc.sources.open(self._tmp.name)

    def graph(self, *, ctx_max: int = 64, rows_max: int = 8) -> Graph:
        return with_layer_outputs(frontend.build(self.source, ctx_max=ctx_max, rows_max=rows_max))

    def interpreter(self, graph: Graph) -> Interpreter:
        return Interpreter(graph, self.source, dtype=torch.float32)


def errors(outs: dict, ref_logits, ref_layers) -> tuple[list[float], float]:
    return [rel(outs[f"h{i:02d}"], ref_layers[i]) for i in range(len(ref_layers))], rel(outs["logits"], ref_logits)


# ---------------------------------------------------------------------------------------------------------- config
def test_gemma4_config():
    nested = {
        "model_type": "gemma4",
        "text_config": e2b_text_config(),
        "vision_config": {},
        "audio_config": {},
    }
    for hf in (nested, e2b_text_config()):
        c = ModelConfig.from_hf(hf)
        g = c.gemma
        assert (c.family, c.num_layers, c.head_dim, c.num_heads, c.num_kv_heads) == (
            "gemma4",
            35,
            256,
            8,
            1,
        )
        assert (c.activation, c.logit_softcap, c.sliding_window, c.tie_embeddings) == (
            "gelu_tanh",
            30.0,
            512,
            True,
        )
        assert c.embed_scale == float(1536**0.5)
        assert (
            g.global_head_dim,
            g.global_kv_heads,
            g.kv_shared_layers,
            g.per_layer_hidden,
        ) == (512, 1, 20, 256)
        assert [g.owner(c.layer_types, i) for i in (14, 15, 18, 19, 33, 34)] == [
            14,
            13,
            13,
            14,
            13,
            14,
        ]
        assert c.rope_theta == 10000.0 and len(c.rope_inv_freq()) == 128
        full = c.rope_inv_freq(F_)
        assert len(full) == 256 and full[63] > 0 and not any(full[64:])
        assert full[0] == 1.0 and abs(full[1] - 1e6 ** (-2 / 512)) < 1e-6
    bad = {
        "enable_moe_block": True,
        "attention_k_eq_v": True,
        "attention_bias": True,
        "use_bidirectional_attention": True,
        "hidden_activation": "gelu",
        "attn_logit_softcapping": 50.0,
        "per_layer_config": {"4": {"num_key_value_heads": 2}},
        "quantization_config": {"quant_method": "gptq"},
    }
    for key, value in bad.items():
        try:
            ModelConfig.from_hf({**e2b_text_config(), key: value})
        except UnsupportedModel as e:
            assert key.split("_config")[0] in str(e) or key in str(e), (key, str(e))
        else:
            raise AssertionError(f"{key}={value!r} accepted")
    rope = {
        **E2B_ROPE,
        "full_attention": {**E2B_ROPE["full_attention"], "rope_type": "yarn"},
    }
    try:
        ModelConfig.from_hf({**e2b_text_config(), "rope_parameters": rope})
    except UnsupportedModel as e:
        assert "yarn" in str(e)
    else:
        raise AssertionError("yarn accepted")
    tail = {
        **e2b_text_config(),
        "layer_types": [S] * 34 + [F_],
    }  # the sharing full layer has no full owner
    try:
        ModelConfig.from_hf(tail)
    except UnsupportedModel:
        pass
    else:
        raise AssertionError("a sharing layer without an owner of its type accepted")


# ---------------------------------------------------------------------------------------------------------- ring
def test_ring_window():
    """Chunks (8, 1, 1, 4, 3, 1, 6) through a wrapping ring equal a positional cache; a step needing more raises."""
    torch.manual_seed(0)
    n, window, hkv, hq, d = 24, 6, 2, 4, 16
    q = torch.randn(n, hq, d)
    k, v = torch.randn(n, hkv, d), torch.randn(n, hkv, d)
    pos = torch.arange(n)
    positional = ExactCache((2, 64, hkv, d), device="cpu", dtype=torch.float32)
    positional.write(pos, k, v)
    full = positional.attend(q, pos, 0.25, window)
    ring = ExactCache((2, window + 8 - 1, hkv, d), device="cpu", dtype=torch.float32)
    parts, at = [], 0
    for w in (8, 1, 1, 4, 3, 1, 6):
        p = pos[at : at + w]
        ring.write(p, k[at : at + w], v[at : at + w])
        parts.append(ring.attend(q[at : at + w], p, 0.25, window))
        at += w
    # the same keys, summed over fewer masked-out columns: equal to the last fp32 bits
    assert torch.allclose(torch.cat(parts), full, atol=1e-6, rtol=1e-6), "ring differs from the positional cache"
    # the window really restricts: the same rows without it differ
    assert not torch.equal(positional.attend(q, pos, 0.25), positional.attend(q, pos, 0.25, window))
    ring = ExactCache((2, 12, hkv, d), device="cpu", dtype=torch.float32)
    try:
        ring.write(pos[:16], k[:16], v[:16])
    except ValueError:
        pass
    else:
        raise AssertionError("16 rows written into a 12-row ring")
    ring.write(pos[:12], k[:12], v[:12])
    ring.write(pos[12:14], k[12:14], v[12:14])
    try:
        ring.attend(q[:2], pos[12:14], 0.25, None)  # unwindowed: needs positions 0..13 in 12 slots
    except ValueError:
        pass
    else:
        raise AssertionError("an unwindowed read of a wrapped ring")


# ---------------------------------------------------------------------------------------------------------- frontend
def test_gemma4_matches_transformers():
    m = Tiny()
    ids = torch.randint(0, 120, (T,), generator=torch.Generator().manual_seed(1))
    ref_logits, ref_layers = hf_run(m.hf, ids[None])
    assert len(ref_layers) == 6 and ref_logits.abs().max() > 30 * 0.4, "softcap is in its nonlinear range"
    graph = m.graph(rows_max=T)  # the one-shot pass needs a ring of window + 24 - 1 rows
    ring_graph = m.graph()
    # ring of window 6 + 8 - 1 = 13 rows < 24 positions: the sliding owners wrap; global owners hold 64
    assert {n: s.shape[1] for n, s in ring_graph.states.items()} == {
        "L0.kv": 13,
        "L1.kv": 13,
        "L2.kv": 64,
    }
    full = m.interpreter(graph).run(feed(ids, 0, T))
    it = m.interpreter(ring_graph)
    layer_errs, logit_err = errors(full, ref_logits, ref_layers)
    print(
        "per-layer rel L2:",
        " ".join(f"{e:.1e}" for e in layer_errs),
        f"| logits {logit_err:.1e}",
    )
    assert max(layer_errs) < GATE and logit_err < GATE, (layer_errs, logit_err)
    assert (full["ids"] == ref_logits.argmax(-1)).all()
    plan = (8, 1, 1, 4, 3, 1, 6)
    assert sum(plan) == T
    parts, at = [], 0
    for w in plan:
        parts.append(it.run(feed(ids, at, at + w))["logits"])
        at += w
    win = torch.cat(parts)
    assert rel(win, full["logits"]) < GATE and rel(win, ref_logits) < GATE, (
        rel(win, full["logits"]),
        rel(win, ref_logits),
    )
    cache, hf_parts, at = DynamicCache(config=m.cfg), [], 0
    with torch.no_grad():
        for w in plan:
            hf_parts.append(
                m.hf(
                    input_ids=ids[None, at : at + w],
                    past_key_values=cache,
                    use_cache=True,
                ).logits[0]
            )
            at += w
    assert rel(win, torch.cat(hf_parts)) < GATE
    it.reset()  # a prefill window then single steps, the serving pattern
    dec = torch.cat(
        [
            it.run(feed(ids, 0, 8))["logits"],
            *[it.run(feed(ids, t, t + 1))["logits"] for t in range(8, T)],
        ]
    )
    assert rel(dec, full["logits"]) < GATE
    it.reset()
    try:
        it.run(feed(ids, 0, 16))
    except ValueError:
        pass
    else:
        raise AssertionError("a 16-row window fits a 13-row ring")


def test_gemma4_variants():
    """No PLE, no softcap, no sharing, the default 5:1 layer pattern."""
    kw = {k: v for k, v in TINY.items() if k not in ("layer_types", "final_logit_softcapping")}
    kw.update(
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        final_logit_softcapping=None,
    )
    m = Tiny(kw)
    ids = torch.randint(0, 120, (T,), generator=torch.Generator().manual_seed(2))
    ref_logits, ref_layers = hf_run(m.hf, ids[None])
    outs = m.interpreter(m.graph(rows_max=T)).run(feed(ids, 0, T))
    layer_errs, logit_err = errors(outs, ref_logits, ref_layers)
    assert max(layer_errs) < GATE and logit_err < GATE, (layer_errs, logit_err)


def test_gemma4_graph():
    m = Tiny()
    graph = frontend.build(m.source, ctx_max=64, rows_max=8)
    ops = [n.op for n in graph.nodes]
    assert ops.count("attention") == 3 and ops.count("attention_read") == 3 and ops.count("softcap") == 1
    assert ops.count("gelu_mul") == 12 and ops.count("embed") == 2 and ops.count("split") == 1
    assert {n.attrs["window"] for n in graph.nodes if n.op == "attention"} == {6, None}
    owners = {n.attrs["state"] for n in graph.nodes if n.op == "attention_read"}
    assert owners == {"L1.kv", "L2.kv"}, owners
    assert graph.weights["L5.gate"].shape == (192, 64) and graph.weights["L0.gate"].shape == (96, 64)
    assert graph.weights["L0.scalar"].shape == (1,) and graph.weights["lm_head"].source == "model.embed_tokens.weight"
    try:
        frontend.build(m.source, ctx_max=64, kv="int4_rot")
    except UnsupportedModel:
        pass
    else:
        raise AssertionError("gemma4 with a quantised KV cache accepted")


# ---------------------------------------------------------------------------------------------------------- slips
def patched(graph: Graph, op: str, fn) -> Graph:
    graph = copy.deepcopy(graph)
    for n in graph.nodes:
        if n.op == op:
            fn(n, graph)
    return graph


def graph_slips(graph: Graph, hf_graph_widths: tuple[int, int]) -> dict[str, Graph]:
    """Each plausible implementation bug as a graph with a wrong attribute or wire."""
    sliding_dim, full_dim = hf_graph_widths

    def plain(theta: float, dim: int, angles: int | None = None) -> tuple[float, ...]:
        n = dim // 2 if angles is None else angles
        rotated = [float(1.0 / (theta ** (torch.arange(0, 2 * n, 2, dtype=torch.float32)[j] / dim))) for j in range(n)]
        return tuple(rotated + [0.0] * (dim // 2 - n))

    def rope_where(dim: int, **attrs):
        return lambda n, g: n.attrs.update(attrs) if n.attrs["rotary_dim"] == dim else None

    def weights_to_unscaled(table: str):
        return lambda n, g: n.attrs.update(scale=1.0) if n.attrs["weight"] == table else None

    def window(delta: int):
        return lambda n, g: n.attrs.update(window=n.attrs["window"] + delta) if n.attrs["window"] else None

    def wrong_owner(n, g):
        if n.attrs["window"]:
            n.attrs["state"] = "L0.kv"

    def scalar_ignored(n, g):
        if "weight" in n.attrs:
            n.attrs.pop("weight")
            n.attrs["factor"] = 1.0

    slips = {
        "sliding layers use the global rope theta": patched(
            graph, "rope", rope_where(sliding_dim, inv_freq=plain(1e6, sliding_dim))
        ),
        "global layers use the sliding rope theta (all angles)": patched(
            graph, "rope", rope_where(full_dim, inv_freq=plain(1e4, full_dim))
        ),
        "proportional rope with half the angles": patched(
            graph,
            "rope",
            rope_where(full_dim, inv_freq=plain(1e6, full_dim, full_dim // 16)),
        ),
        "embedding not scaled by sqrt(H)": patched(graph, "embed", weights_to_unscaled("embed")),
        "PLE table not scaled by sqrt(D)": patched(graph, "embed", weights_to_unscaled("ple_embed")),
        "window off by one (+1)": patched(graph, "attention", window(1)),
        "window off by one (-1)": patched(graph, "attention", window(-1)),
        "attention scale head_dim^-0.5 instead of 1": patched(
            graph, "attention", lambda n, g: n.attrs.update(scale=0.25)
        ),
        "shared sliding layer reads the wrong owner": patched(graph, "attention_read", wrong_owner),
        "sharing layers attend without the window": patched(
            graph, "attention_read", lambda n, g: n.attrs.update(window=None)
        ),
        "norm with 1 + weight": patched(
            graph,
            "rmsnorm",
            lambda n, g: n.attrs.update(offset=1.0) if "weight" in n.attrs else None,
        ),
        "softcap missing": patched(graph, "softcap", lambda n, g: n.attrs.update(cap=1e9)),
        "layer_scalar ignored": patched(graph, "scale", scalar_ignored),
    }
    shifted = copy.deepcopy(graph)  # layer i reads PLE slice i + 1 (mod L)
    ple = next(n.outs for n in shifted.nodes if n.op == "split")
    for n in shifted.nodes:
        if n.op == "gelu_mul" and n.ins[1] in ple:
            n.ins[1] = ple[(ple.index(n.ins[1]) + 1) % len(ple)]
    slips["per-layer embedding slice off by one layer"] = shifted
    return slips


def test_gemma4_slips():
    """16 slips, each moves the layer outputs or the logits by at least 10 gates (the tiny model's logits are large,
    so the gate is relative)."""
    m = Tiny()
    ids = torch.randint(0, 120, (T,), generator=torch.Generator().manual_seed(1))
    ref_logits, ref_layers = hf_run(m.hf, ids[None])
    graph = m.graph(rows_max=T)
    table = []
    for name, bad in graph_slips(graph, (16, 32)).items():
        table.append(
            (
                name,
                *errors(m.interpreter(bad).run(feed(ids, 0, T)), ref_logits, ref_layers),
            )
        )
    orig = dict(_IMPL)
    erf_gelu = lambda it, a, g_, u: F.gelu(g_) * u  # noqa: E731
    raw_v = lambda it, a, x: orig["rmsnorm"](it, a, x) if "weight" in a else x  # noqa: E731
    for name, op, impl in (
        ("v_norm dropped (raw V)", "rmsnorm", raw_v),
        ("gelu in its erf form", "gelu_mul", erf_gelu),
    ):
        _IMPL[op] = impl
        try:
            table.append(
                (
                    name,
                    *errors(
                        m.interpreter(graph).run(feed(ids, 0, T)),
                        ref_logits,
                        ref_layers,
                    ),
                )
            )
        finally:
            _IMPL[op] = orig[op]
    assert len(table) == 15 + 1, len(table)
    weakest = min(max(max(layers), logits) for _, layers, logits in table)
    for name, layers, logits in table:
        print(f"  {name:<56} worst layer {max(layers):.2e} logits {logits:.2e}")
        assert max(max(layers), logits) > 10 * GATE, f"slip not detected: {name}"
    print(f"  weakest slip {weakest / GATE:.0f}x the gate")


# ---------------------------------------------------------------------------------------------------------- compiler
def test_gemma4_rejected_by_dense_caps():
    """The dense CUDA table has no kernel for Gemma's ops or attributes: lowering names each. (Its attention kernel
    takes a window since g_attn widened: tests/test_gemma4_attn.py.)"""
    m = Tiny()
    graph = frontend.build(m.source, ctx_max=64, rows_max=8)
    try:
        lower(graph, CudaBackend(), rows_max=8)
    except LoweringError as e:
        text = str(e)
        for needle in ("scale", "softcap", "gelu_mul", "attention_read", "rmsnorm"):
            assert needle in text, (needle, text)
    else:
        raise AssertionError("the dense table lowered a Gemma 4 graph")
    dense = Graph("windowed")
    dense.input("tokens", TType(("T",), "i32"))
    pos = dense.input("pos", TType(("T",), "i32"))
    x = dense.input("x", TType(("T", 1024), "bf16"))
    for name, shape in (("wq", (1024, 1024)), ("wk", (128, 1024)), ("wv", (128, 1024))):
        dense.weight(WeightDecl(name, shape, "dense", name))
    dense.state(StateDecl("kv", "kv", (2, 64, 1, 128)))
    q = dense.op("reshape", [dense.op("linear", [x], weight="wq")], shape=["T", 8, 128])
    k = dense.op("reshape", [dense.op("linear", [x], weight="wk")], shape=["T", 1, 128])
    v = dense.op("reshape", [dense.op("linear", [x], weight="wv")], shape=["T", 1, 128])
    q, k = dense.op_multi("rope", [q, k, pos], rotary_dim=128, inv_freq=(1.0,) * 64)
    att = dense.op("attention", [q, k, v, pos], state="kv", scale=1.0, window=32)
    dense.output(att=att)
    fused = fuse(dense, CudaBackend().caps, 8)
    assert not any(dense.nodes[i].op == "attention" for i in fused.unclaimed), fused.backlog(dense)


def test_state_read_order():
    """attention_read needs its state written earlier in the graph; fused tasks never reorder the two."""
    g = Graph("order")
    g.input("tokens", TType(("T",), "i32"))
    pos = g.input("pos", TType(("T",), "i32"))
    q = g.input("q", TType(("T", 2, 8), "bf16"))
    k = g.input("k", TType(("T", 1, 8), "bf16"))
    g.state(StateDecl("kv", "kv", (2, 16, 1, 8)))
    g.weight(WeightDecl("w", (8,), "dense", "w"))
    g.op("attention_read", [q, pos], state="kv", scale=1.0)
    try:
        g.verify()
    except ValueError as e:
        assert "before any node of the graph writes it" in str(e)
    else:
        raise AssertionError("a read before the write verified")
    h = Graph("order2")
    h.input("tokens", TType(("T",), "i32"))
    pos = h.input("pos", TType(("T",), "i32"))
    q, k = (h.input(n, TType(("T", a, 8), "bf16")) for n, a in (("q", 2), ("k", 1)))
    h.state(StateDecl("kv", "kv", (2, 16, 1, 8)))
    h.op("attention", [q, k, k, pos], state="kv", scale=1.0)
    h.op("attention_read", [q, pos], state="kv", scale=1.0, window=4)
    h.verify()

    plan = SimpleNamespace(tasks=[SimpleNamespace(covers=[0]), SimpleNamespace(covers=[1])])

    assert [t.covers for t in _live_tasks(h, plan, {h.nodes[1].outs[0]})] == [[0], [1]]
    assert [t.covers for t in _live_tasks(h, plan, set())] == [[0]], "only the writer is live without a consumer"
    regions = [Region("w", {0}, {}, {}), Region("r", {1}, {}, {})]
    _check_state_order(h, regions, [0, 1])
    try:
        _check_state_order(h, regions, [1, 0])
    except LoweringError as e:
        assert "reads state kv before" in str(e)
    else:
        raise AssertionError("a read ordered before its write")


# ------------------------------------------------------------------------------------------------ real checkpoint
def test_gemma4_checkpoint():
    """The E2B graph resolves: 541 declarations over 540 tensors (the head reuses embed_tokens) with the declared
    shapes, all bf16; 875 nodes, 15 KV states with the ring and positional shapes, owners 13 / 14."""
    path = os.environ.get("WARPC_TEST_GEMMA4")
    if not path:
        raise Skip("WARPC_TEST_GEMMA4 unset")
    from safetensors import safe_open

    source = warpc.sources.open(path)
    graph = frontend.build(source, ctx_max=4096, rows_max=8)
    with safe_open(str(Path(path) / "model.safetensors"), framework="pt") as h:
        header = {k: (tuple(h.get_slice(k).get_shape()), h.get_slice(k).get_dtype()) for k in list(h.keys())}
    assert len(graph.weights) == 541 and len({w.source for w in graph.weights.values()}) == 540
    for w in graph.weights.values():
        shape, dtype = header[w.source]
        assert shape == w.shape and dtype == "BF16", (w.name, shape, dtype, w.shape)
    ops = {}
    for n in graph.nodes:
        ops[n.op] = ops.get(n.op, 0) + 1
    assert len(graph.nodes) == 875, len(graph.nodes)
    assert ops == {
        "linear": 277, "rmsnorm": 242, "add": 106, "gelu_mul": 70, "reshape": 67, "scale": 37, "rope": 35,
        "attention_read": 20, "attention": 15, "embed": 2, "split": 1, "cast": 1, "softcap": 1, "argmax": 1,
    }, ops  # fmt: skip
    assert len(graph.states) == 15
    ring = 512 + 8 - 1
    for i, s in graph.states.items():
        layer = int(i[1:].split(".")[0])
        full = source.config.layer_types[layer] == F_
        assert s.shape == (2, 4096 if full else ring, 1, 512 if full else 256), (
            i,
            s.shape,
        )
    reads = {n.attrs["state"] for n in graph.nodes if n.op == "attention_read"}
    assert reads == {"L13.kv", "L14.kv"}, reads
    config = json.loads((Path(path) / "config.json").read_text())
    assert ModelConfig.from_hf(config).gemma.per_layer_hidden == 256


class ShapeSource:
    """A source with a config only: every tensor exists, none quantised (the compiler reads shapes, not values)."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> str | None:
        return None


def test_gemma4_lowers():
    """The exit test of the CUDA kernels: the E2B graph lowers with no unclaimed node, 12 launches per layer (427 per
    token: 35 x 12, the embedding, the PLE projection and its row kernel, the final norm and the 3-launch head)."""
    config = ModelConfig.from_hf(e2b_text_config())
    graph = frontend.build(ShapeSource(config), ctx_max=4096, rows_max=8)
    backend = CudaBackend()
    plan = lower(graph, backend, rows_max=8)
    launches = sum(backend.caps.by_kernel[t.kernel].launches for t in plan.tasks)
    assert launches == 427, (launches, plan.summary())


# a tiny model the CUDA kernels accept (K a multiple of 256, head dims 64 and 512) whose sliding rings wrap: window 24,
# rows_max 8 -> 31 rows, rounded up to 64 by the executor, and the 100 positions of the run pass it
GPU_TINY = {
    **TINY,
    "vocab_size": 512,
    "vocab_size_per_layer_input": 512,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 512,
    "hidden_size_per_layer_input": 256,
    "sliding_window": 24,
}


def test_gemma4_tiny_compiled():
    """Compiled greedy decode (windows of 8 rows, then single steps) equals the reference interpreter in bf16 on the
    GPU: both sliding-ring owners and sharers wrap, the global layers have head dim 512."""
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    import warpc

    m = Tiny(GPU_TINY)
    prompt = torch.randint(0, 500, (75,), generator=torch.Generator().manual_seed(5)).tolist()
    new = 25
    model = warpc.compile(m.source, ctx_max=256, rows_max=8)
    got = model.generate(prompt, new)
    graph = frontend.build(m.source, ctx_max=256, rows_max=8)
    it = Interpreter(graph, m.source, device="cuda", dtype=torch.bfloat16)
    at, out = 0, None
    while at < len(prompt):
        chunk = prompt[at : at + 8]
        feed = {
            "tokens": torch.tensor(chunk, dtype=torch.int32),
            "pos": torch.arange(at, at + len(chunk), dtype=torch.int32),
        }
        out = it.run(feed)
        at += len(chunk)
    want = [int(out["ids"][-1])]
    while len(want) < new:
        feed = {
            "tokens": torch.tensor(want[-1:], dtype=torch.int32),
            "pos": torch.tensor([at], dtype=torch.int32),
        }
        want.append(int(it.run(feed)["ids"][-1]))
        at += 1
    assert got == want, (got, want)
    assert model.generate(prompt, new) == got, "generate must start from an empty cache"


if __name__ == "__main__":
    run(globals())
