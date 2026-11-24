"""The compiler on dense graphs, CPU only: fusion against golden regions, legality tiers, cost model, rounding points,
LoweringError, buffer assignment, packs, and the CUDA table / kernel-library contracts that need no GPU.

  uv run --package warpc python warpc/tests/test_compiler.py [--update]

tests/data/fusion_golden.json holds, per case of CASES, the regions the pass claims (kernel, covered ops, operand
values, packs, launch attrs) or the op counts nothing claims; --update rewrites it from the current pass.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from harness import run
from warpc import frontend
from warpc.compiler import Absorb, Cap, CapsTable, Pack, Region, Verdict, lower
from warpc.compiler.caps import ref_points
from warpc.compiler.fuse import Analysis, _candidates, fuse
from warpc.cuda import CudaBackend, glue, schedule
from warpc.cuda.pack import Packer
from warpc.cuda.table import HEAD_NUMERICS, RTX3090, TABLE
from warpc.errors import LoweringError
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.kvcache import FORMATS
from warpc.sources import ModelConfig

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "fusion_golden.json"
BACKEND = CudaBackend()
QWEN3_06B = {
    "vocab_size": 151936,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rms_eps": 1e-6,
    "rope_theta": 1e6,
}
LLAMA3_8B = {
    "vocab_size": 128256,
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_layers": 2,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rms_eps": 1e-5,
    "rope_theta": 500000.0,
}
LAYER = ["g_gemv", "g_attn", "g_gemv_resadd", "g_gemv_dual", "g_gemv_resadd"]


class ShapeSource:
    """A source that only has a config: every tensor exists, linears inside the layers are Q4_0 if `q4`."""

    def __init__(self, config: ModelConfig, q4: bool):
        self.config, self.q4 = config, q4

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> str | None:
        return "Q4_0" if self.q4 and "layers." in name else None


def build(cfg: dict, family: str = "qwen3", ctx: int = 2048, kv: str = "bf16", q4: bool = False) -> Graph:
    config = ModelConfig(family=family, qk_norm=family == "qwen3", tie_embeddings=False, rope_freq_factors=None, **cfg)
    return frontend.build(ShapeSource(config, q4), ctx_max=ctx, kv=kv)


def _cases() -> dict:
    """name -> (graph, rows_max): real shapes, every KV format, Q4_0 linears, and mutations that exercise each decline
    path."""

    def q3(**kw):
        return {**QWEN3_06B, "num_layers": 2, **kw}

    def small():
        return build(q3())

    def of(g, op):
        return [i for i, n in enumerate(g.nodes) if n.op == op]

    c = {
        "qwen3_06b": (build(QWEN3_06B), 8),
        "llama3_8b_2L": (build(LLAMA3_8B, "llama"), 8),
        "qwen3_q4_0": (build(q3(), q4=True), 8),
        "llama_q4_0_k8v4w": (build(LLAMA3_8B, "llama", 512, "k8v4w", True), 8),
    }
    for kv in ("k8v8", "k8v8w", "k8v4w"):
        c[f"kv_{kv}"] = (build(q3(), kv=kv), 8)
    for kv in ("bf16", "k8v8w"):
        c[f"kv_{kv}_32768"] = (build(q3(), ctx=32768, kv=kv), 8)
    for hd in (64, 256, 96):
        c[f"hd{hd}"] = (build(q3(head_dim=hd), ctx=1024), 8)
    c["g8"] = (build(q3(num_kv_heads=2)), 8)
    c["g16"] = (build(q3(num_kv_heads=1)), 8)
    c["rows1"] = (small(), 1)
    c["rows9"] = (small(), 9)
    c["k_odd"] = (build(q3(hidden_size=1032)), 8)
    for name, op, k, side in (
        ("norm_out_is_output", "rmsnorm", 0, "outs"),
        ("q_out_is_output", "linear", 0, "outs"),
        ("add_out_is_output", "add", 0, "outs"),
        ("gate_out_is_output", "silu_mul", 0, "ins"),
        ("rope_q_is_output", "rope", 0, "outs"),
        ("v_reshape_is_output", "reshape", 2, "outs"),
    ):
        g = small()
        g.output(extra=getattr(g.nodes[of(g, op)[k]], side)[0])
        c[name] = (g, 8)
    g = small()
    g.nodes[of(g, "attention")[0]].attrs["scale"] = 0.5
    c["attn_scale"] = (g, 8)
    g = small()
    rope = g.nodes[of(g, "rope")[0]].attrs
    rope.update(rotary_dim=64, inv_freq=rope["inv_freq"][:32])
    c["partial_rope"] = (g, 8)
    g = small()
    g.nodes[of(g, "rmsnorm")[1]].attrs["eps"] = 1e-5
    c["eps_mismatch_qk"] = (g, 8)
    return c


CASES = _cases()


def behaviour(g: Graph, rows_max: int) -> dict:
    """What the pass claims on `g`, independent of node numbering, buffer reuse and pack naming."""
    res = fuse(g, TABLE, rows_max)
    if res.unclaimed:
        return {"unclaimed": dict(sorted(Counter(g.nodes[i].op for i in res.unclaimed).items()))}
    regions = []
    for r in res.regions:
        packs = {p.name: [p.kind, p.parts] for p in r.packs}
        regions.append(
            {
                "kernel": r.kernel,
                "covers": [g.nodes[i].op for i in sorted(r.covers)],
                "ins": r.ins,
                "outs": r.outs,
                "weights": {role: packs[name] for role, name in r.weights.items()},
                "states": r.states,
                # a table's values are the graph's (test_frontend); the claim is which tables a kernel reads
                "constants": {role: len(values) for role, values in r.constants.items()},
                "attrs": r.attrs,
                "tolerance": list(r.tolerance),
            }
        )
    return {"regions": regions}


class RoundingError(AssertionError):
    """A region drops a reference rounding point without declaring it a tolerance."""


@dataclass(frozen=True)
class Finding:
    region: int
    kernel: str
    node: int
    op: str
    point: str
    # preserved | acknowledged (a declared tolerance of the kernel) | tolerated (in the region's tolerance) | GAP
    # | unchecked (no capability of that name)
    status: str

    def __str__(self):
        return f"region {self.region} {self.kernel}: node {self.node} {self.op}.{self.point}: {self.status}"


def check_rounding(g: Graph, regions, caps: CapsTable) -> list[Finding]:
    """Every reference rounding point of every node the regions (Regions or Tasks) cover, against what each region's
    kernel reproduces. Raises RoundingError if a point is dropped without being a declared tolerance of the kernel or of
    the region; returns the findings otherwise."""
    fs = []
    for ri, r in enumerate(regions):
        cap = caps.by_kernel.get(r.kernel)
        if cap is None:
            fs.append(Finding(ri, r.kernel, -1, "*", "*", "unchecked"))
            continue
        for i in sorted(r.covers):
            n = g.nodes[i]
            for p in ref_points(n.op, g.types[n.outs[0]].dtype):
                if p in cap.rounds.get(n.op, ()):
                    st = "preserved"
                elif f"{n.op}.{p}" in cap.tolerance:
                    st = "acknowledged"
                elif f"{n.op}.{p}" in r.tolerance:
                    st = "tolerated"
                else:
                    st = "GAP"
                fs.append(Finding(ri, r.kernel, i, n.op, p, st))
    bad = [f for f in fs if f.status in ("GAP", "unchecked")]
    if bad:
        more = f"\n  ... {len(bad) - 12} more" if len(bad) > 12 else ""
        lines = "\n  ".join(map(str, bad[:12]))
        raise RoundingError(f"rounding points dropped without a declared tolerance:\n  {lines}{more}")
    return fs


def table_with(*caps: Cap) -> CapsTable:
    """TABLE with `caps` replacing the entries of the same kernel, or appended."""
    by = {c.kernel: c for c in caps}
    entries = [by.pop(c.kernel, c) for c in TABLE.caps] + list(by.values())
    return CapsTable(entries, TABLE.target, TABLE.unfused_ctas, TABLE.unfused_launches)


def covers_only(m, A) -> Region:
    """An emitter for test capabilities whose regions are only inspected, never lowered."""
    return Region(m.cap.kernel, set(m.covers), {}, {})


def test_fusion_matches_the_golden_regions():
    golden = json.loads(GOLDEN_PATH.read_text())
    assert sorted(golden) == sorted(CASES), sorted(set(golden) ^ set(CASES))
    for name, (g, rows) in CASES.items():
        got = json.loads(json.dumps(behaviour(g, rows)))
        want = golden[name]
        if "unclaimed" in want:
            assert got == want, (name, got)
            try:
                lower(g, BACKEND, rows_max=rows)
            except LoweringError as e:
                assert all(op in str(e) for op in want["unclaimed"]), (name, str(e))
            else:
                raise AssertionError(f"{name}: must not lower")
            continue
        assert len(got["regions"]) == len(want["regions"]), name
        for i, (a, b) in enumerate(zip(got["regions"], want["regions"], strict=True)):
            assert a == b, (name, i, a, b)


def test_qwen3_06b_plan_structure_and_tolerance():
    g = build(QWEN3_06B)
    plan = lower(g, BACKEND, rows_max=8)
    assert [t.kernel for t in plan.tasks] == ["g_embed"] + LAYER * 28 + ["g_rmsnorm", "g_head"]
    assert plan.backend == "cuda" and plan.rows_max == 8 and plan.graph_name == "qwen3"
    tolerant = [t for t in plan.tasks if t.tolerance]
    assert Counter(t.kernel for t in tolerant) == {"g_attn": 28, "g_head": 1}
    assert set(tolerant[0].tolerance) == {"rope.trig", "rope.prod", "attention.p"}
    assert tolerant[-1].tolerance == ("linear.out",) and tolerant[-1].numerics == HEAD_NUMERICS
    attn = [t for t in plan.tasks if t.kernel == "g_attn"]
    assert all(t.attrs["qk_norm"] and t.attrs["eps"] == 1e-6 for t in attn)
    (table,) = {n.attrs["inv_freq"] for n in g.nodes if n.op == "rope"}
    assert list(plan.constants.values()) == [table], "one rope table for the 28 layers"
    assert all(plan.constants[t.constants["inv_freq"]] == table for t in attn)


def test_llama_plan_folds_the_norm_into_qkv_and_updates_the_stream_in_place():
    plan = lower(build(LLAMA3_8B, "llama"), BACKEND, rows_max=8)
    assert [t.kernel for t in plan.tasks] == ["g_embed"] + LAYER * 2 + ["g_rmsnorm", "g_head"]
    qkv = plan.tasks[1]
    assert "nw" in qkv.weights and qkv.attrs["segs"] == [4096, 1024, 1024]
    assert plan.packs[qkv.weights["w"]].kind == "dense_cat"
    attn = plan.tasks[2]
    assert attn.attrs["qk_norm"] is False and "eps" not in attn.attrs and not attn.weights
    for t in plan.tasks:
        if t.kernel == "g_gemv_resadd":
            assert t.ins["h"] == t.outs["h_out"] and t.ins["x"] != t.ins["h"]


def test_q4_0_linears_lower_to_the_fused_dequant_kernels():
    plan = lower(build({**QWEN3_06B, "num_layers": 2}, q4=True), BACKEND, rows_max=8)
    layer = ["g_rmsnorm", "g_q4gemv", "g_attn", "g_q4gemv_resadd", "g_rmsnorm", "g_q4gemv_dual", "g_q4gemv_resadd"]
    assert [t.kernel for t in plan.tasks] == ["g_embed"] + layer * 2 + ["g_rmsnorm", "g_head"]
    kinds = Counter(p.kind for p in plan.packs.values())
    assert kinds == {"dense": 11, "q4_0_cat": 2, "q4_0": 8}, kinds
    g = build({**QWEN3_06B, "num_layers": 1, "hidden_size": 1040}, q4=True)  # K % 32 != 0: no Q4_0 kernel
    res = fuse(g, TABLE, 8)
    assert any("K=1040 is not a multiple of 32" in r for r in res.unclaimed.values()), res.unclaimed


def test_lowering_error_names_the_ops_and_reasons():
    cfg = {**QWEN3_06B, "vocab_size": 128, "hidden_size": 64, "intermediate_size": 96, "num_layers": 1}
    g = build({**cfg, "num_heads": 4, "num_kv_heads": 2, "head_dim": 32}, ctx=64)
    try:
        lower(g, BACKEND, rows_max=8)
    except LoweringError as e:
        msg = str(e)
        assert msg.startswith("cuda: no kernel for") and "linear" in msg and "attention" in msg, msg
        assert "K=64 is not a multiple of 256" in msg and "hd=32 not in (64, 128, 256)" in msg, msg
    else:
        raise AssertionError("hidden 64 / head dim 32 must not lower")


def test_unknown_kv_format_and_lone_cast_are_lowering_errors():
    g = build({**QWEN3_06B, "num_layers": 1})
    g.states["L0.kv"] = replace(g.states["L0.kv"], cache="nonesuch")
    try:
        lower(g, BACKEND, rows_max=8)
    except LoweringError as e:
        assert "attention" in str(e) and "KV cache format nonesuch has no kernel" in str(e), str(e)
    else:
        raise AssertionError("an unknown cache format must not lower")
    g = Graph("cast")
    g.input("h", TType(("T", 256), "bf16"))
    g.output(o=g.op("cast", ["h"], dtype="f32"))
    try:
        lower(g, BACKEND, rows_max=8)
    except LoweringError as e:
        assert "cast x1: only absorbable as epilogue of g_head" in str(e), str(e)
    else:
        raise AssertionError("a cast alone has no kernel")


def test_attention_split_chunks_and_their_limit():
    assert [schedule.attn_chunks(c) for c in (64, 2048, 8192, 8193, 32768, 65536)] == [1, 1, 1, 2, 4, 8]
    g = build({**QWEN3_06B, "num_layers": 1}, ctx=32768)
    assert lower(g, BACKEND, rows_max=8).tasks[2].attrs["nc"] == 4
    attn = TABLE.by_kernel["g_attn"]

    def one_chunk(m, A):  # 512 blocks per head: the single-chunk kernel has ATTN_MAX_SPLITS = 128
        c = attn.constants(m, A)
        return {**c, "nc": 1, "nsplit": schedule.attn_splits(c["ctx"], 1)}

    res = fuse(g, table_with(replace(attn, constants=one_chunk)), 8)
    assert any("512 split blocks per head" in w for w in res.unclaimed.values()), res.unclaimed


def tiny_graph(residual=False, extra_reader=False, y_output=False, H=512):
    """embed -> rmsnorm -> linear (-> add with the embed output): the smallest graph with a residual epilogue."""
    g = Graph("tiny")
    tokens = g.input("tokens", TType(("T",), "i32"))
    for n, s in (("embed", (64, H)), ("nw", (H,)), ("w", (H, H)), ("w2", (H, H))):
        g.weight(WeightDecl(n, s, "dense", n))
    x = g.op("embed", [tokens], weight="embed", name="x")
    y = g.op("linear", [g.op("rmsnorm", [x], weight="nw", eps=1e-6)], weight="w")
    outs = {}
    if residual:
        outs["x2"] = g.op("add", [x, y], name="x2")
    if y_output or not residual:
        outs["y"] = y
    if extra_reader:
        outs["z"] = g.op("linear", [x], weight="w2", name="z")
    g.output(**outs)
    g.verify()
    return g


def test_structural_single_user_rule_rejects_a_residual_epilogue():
    # the linear output is a graph output: it cannot vanish into the epilogue
    g = tiny_graph(residual=True, y_output=True)
    res = fuse(g, TABLE, 8)
    add = next(i for i, n in enumerate(g.nodes) if n.op == "add")
    assert add in res.unclaimed and "single-user" in res.unclaimed[add], res.unclaimed


def test_structural_no_reduction_in_an_epilogue_unless_declared():
    g = Graph("lnorm")
    g.input("h", TType(("T", 512), "bf16"))
    g.weight(WeightDecl("w", (512, 512), "dense", "w"))
    g.weight(WeightDecl("nw", (512,), "dense", "nw"))
    g.output(o=g.op("rmsnorm", [g.op("linear", ["h"], weight="w")], weight="nw", eps=1e-6))
    rounds = {"linear": ("out",), "rmsnorm": ("out",)}
    cap = Cap("k_lin_norm", "linear", covers_only, post=(Absorb("rmsnorm", min=1),), rounds=rounds)
    alone = CapsTable([cap], RTX3090, TABLE.unfused_ctas)
    res = fuse(g, alone, 8)
    assert "reduction: not legal in an epilogue" in res.unclaimed[0], res.unclaimed
    declared = replace(cap, post=(Absorb("rmsnorm", min=1, reduction_ok=True),))
    res = fuse(g, CapsTable([declared], RTX3090, TABLE.unfused_ctas), 8)
    assert not res.unclaimed and [r.kernel for r in res.regions] == ["k_lin_norm"]


def test_operand_cap_of_sixteen():
    g = Graph("wide")
    g.input("h", TType(("T", 256), "bf16"))
    outs = {}
    for i in range(18):
        g.weight(WeightDecl(f"w{i}", (256, 256), "dense", f"w{i}"))
        outs[f"o{i}"] = g.op("linear", ["h"], weight=f"w{i}")
    g.output(**outs)
    wide = Cap("k_wide", "linear", covers_only, segments=18, rounds={"linear": ("out",)})
    res = fuse(g, CapsTable([wide], RTX3090, TABLE.unfused_ctas), 8)
    assert len(res.unclaimed) == 18 and "more than 16 operands" in res.unclaimed[0], res.unclaimed
    res = fuse(g, CapsTable([replace(wide, kernel="k_three", segments=3)], RTX3090, TABLE.unfused_ctas), 8)
    assert not res.unclaimed and len(res.regions) == 6


def test_memory_tier_refuses_in_place_updates_that_race_a_reader():
    # z = linear(x) reads the stream and does not precede the epilogue region
    g = tiny_graph(residual=True, extra_reader=True)
    res = fuse(g, TABLE, 8)
    add = next(i for i, n in enumerate(g.nodes) if n.op == "add")
    assert add in res.unclaimed and "race" in res.unclaimed[add], res.unclaimed
    ok = fuse(tiny_graph(residual=True), TABLE, 8)
    assert "g_gemv_resadd" in {r.kernel for r in ok.regions} and not ok.unclaimed
    g3 = tiny_graph(residual=True)  # a graph output is never updated in place
    g3.output(raw=next(n.outs[0] for n in g3.nodes if n.op == "embed"))
    assert "g_gemv_resadd" not in {r.kernel for r in fuse(g3, TABLE, 8).regions}


def test_memory_tier_refuses_updating_the_gemvs_own_input():
    g = Graph("selfres")
    g.input("tokens", TType(("T",), "i32"))
    g.weight(WeightDecl("embed", (64, 512), "dense", "embed"))
    g.weight(WeightDecl("w", (512, 512), "dense", "w"))
    x = g.op("embed", ["tokens"], weight="embed", name="x")
    g.output(x2=g.op("add", [x, g.op("linear", [x], weight="w")], name="x2"))
    res = fuse(g, TABLE, 8)
    add = next(i for i, n in enumerate(g.nodes) if n.op == "add")
    assert add in res.unclaimed and "own read" in res.unclaimed[add], res.unclaimed


def test_selection_covers_an_add_with_the_norm_that_reads_it():
    """The stream is a graph input (never updated in place), so the residual GEMV is illegal; coverage puts the add into
    g_rmsnorm_add, which updates the GEMV output in place instead."""
    g = Graph("cover")
    g.input("h", TType(("T", 256), "bf16"))
    g.input("a", TType(("T", 256), "bf16"))
    for n, s in (("nw", (256,)), ("wo", (256, 256)), ("wg", (512, 256)), ("wu", (512, 256))):
        g.weight(WeightDecl(n, s, "dense", n))
    y = g.op("linear", ["a"], weight="wo")
    x2 = g.op("add", ["h", y], name="x2")
    n = g.op("rmsnorm", [x2], weight="nw", eps=1e-6)
    g.output(x2=x2, m=g.op("silu_mul", [g.op("linear", [n], weight="wg"), g.op("linear", [n], weight="wu")]))
    res = fuse(g, TABLE, 8)
    assert not res.unclaimed and sorted(r.kernel for r in res.regions) == ["g_gemv", "g_gemv_dual", "g_rmsnorm_add"]
    add = next(r for r in res.regions if r.kernel == "g_rmsnorm_add")
    assert add.ins["h"] == y and add.ins["y"] == "h" and add.inplace == ("h_out", "h")


def test_a_fusion_that_would_close_a_cycle_is_dropped():
    """k_add_norm absorbs the norm in front of the add, but the linear between them reads the norm: the region would
    run both before and after the linear. The pass picks it (it gains), finds the cycle and re-solves without it."""
    g = Graph("cycle")
    g.input("h", TType(("T", 256), "bf16"))
    g.weight(WeightDecl("nw", (256,), "dense", "nw"))
    g.weight(WeightDecl("w", (256, 256), "dense", "w"))
    a = g.op("rmsnorm", ["h"], weight="nw", eps=1e-6)
    g.output(o=g.op("add", [a, g.op("linear", [a], weight="w")]))
    rounds = {"rmsnorm": ("out",), "add": ("out",), "linear": ("out",)}
    caps = [
        Cap("k_add_norm", "add", covers_only, pre=(Absorb("rmsnorm", min=1, exposes=True),), rounds=rounds),
        Cap("k_add", "add", covers_only, rounds=rounds),
        Cap("k_norm", "rmsnorm", covers_only, rounds=rounds),
        Cap("k_lin", "linear", covers_only, rounds=rounds),
    ]
    res = fuse(g, CapsTable(caps, RTX3090, TABLE.unfused_ctas), 8)
    assert not res.unclaimed and sorted(r.kernel for r in res.regions) == ["k_add", "k_lin", "k_norm"], res.regions


def test_full_width_qk_norm_is_not_absorbed_into_the_attention():
    H, nq, nkv, hd = 512, 4, 2, 128
    g = Graph("fullnorm")
    g.input("tokens", TType(("T",), "i32"))
    pos = g.input("pos", TType(("T",), "i32"))
    for n, s in (("embed", (64, H)), ("wq", (nq * hd, H)), ("wk", (nkv * hd, H)), ("wv", (nkv * hd, H))):
        g.weight(WeightDecl(n, s, "dense", n))
    g.weight(WeightDecl("wo", (H, nq * hd), "dense", "wo"))
    g.weight(WeightDecl("qn", (nq * hd,), "dense", "qn"))
    g.weight(WeightDecl("kn", (nkv * hd,), "dense", "kn"))
    g.state(StateDecl("kv", (2, 256, nkv, hd)))
    x = g.op("embed", ["tokens"], weight="embed", name="x")
    q = g.op("rmsnorm", [g.op("linear", [x], weight="wq")], weight="qn", eps=1e-6)  # full width, before the reshape
    k = g.op("rmsnorm", [g.op("linear", [x], weight="wk")], weight="kn", eps=1e-6)
    q = g.op("reshape", [q], shape=["T", nq, hd])
    k = g.op("reshape", [k], shape=["T", nkv, hd])
    v = g.op("reshape", [g.op("linear", [x], weight="wv")], shape=["T", nkv, hd])
    q, k = g.op_multi("rope", [q, k, pos], rotary_dim=hd, inv_freq=(1.0,) * (hd // 2))
    att = g.op("attention", [q, k, v, pos], state="kv", scale=hd**-0.5)
    g.output(y=g.op("add", [x, g.op("linear", [att], weight="wo")], name="y"))
    g.verify()
    plan = lower(g, BACKEND, rows_max=8)
    assert [t.kernel for t in plan.tasks].count("g_rmsnorm") == 2
    (a,) = [t for t in plan.tasks if t.kernel == "g_attn"]
    produced = {b for t in plan.tasks if t.kernel == "g_rmsnorm" for b in t.outs.values()}
    assert a.attrs["qk_norm"] is False and {a.ins["q"], a.ins["k"]} == produced


def test_cost_model_orders_the_fusions():
    g = build(QWEN3_06B)

    def gains(table):
        """The best gain each kernel offers on the graph (its richest legal variant)."""
        by: dict = {}
        for m in _candidates(Analysis(g, 8), table, False)[0]:
            by[m.cap.kernel] = max(by.get(m.cap.kernel, m.gain), m.gain)
        return by

    base = gains(TABLE)
    # launches removed dominate: qkv + norm, attention + qk-norm + rope and gate|up + silu + norm remove 2-3 launches,
    # the residual GEMV one, the head none (only the logits it does not write twice)
    assert min(base["g_gemv"], base["g_attn"], base["g_gemv_dual"]) > base["g_gemv_resadd"] > base["g_head"] > 0, base
    assert base["g_embed"] == 0 and base["g_rmsnorm"] == 0  # a bare anchor gains nothing
    slow = gains(CapsTable(TABLE.caps, replace(RTX3090, launch_us=40.0), TABLE.unfused_ctas, TABLE.unfused_launches))
    assert slow["g_gemv_dual"] > 5 * base["g_gemv_dual"] and slow["g_head"] == base["g_head"], (slow, base)
    # the same fusion on a kernel that fills one CTA loses against the standalone kernels: the pass leaves it apart
    k_add = Cap("k_add", "add", covers_only, rounds={"add": ("out",)})
    one_cta = replace(TABLE.by_kernel["g_gemv_resadd"], info=lambda c: Verdict(ctas=1))
    kernels = {r.kernel for r in fuse(tiny_graph(residual=True), table_with(k_add, one_cta), 8).regions}
    assert "g_gemv_resadd" not in kernels and "k_add" in kernels, kernels


def test_limits_reject_unknown_names_and_missing_constants():
    try:
        Cap("k_bad", "linear", covers_only, limits={"K_multple": 256})
    except ValueError as e:
        assert "unknown limit K_multple" in str(e)
    else:
        raise AssertionError("a misspelt limit must be refused")
    typo = Cap("k_typo", "linear", covers_only, limits={"Kx_multiple": 256}, constants=lambda m, A: {"K": 512})
    try:
        fuse(tiny_graph(), CapsTable([typo], RTX3090, TABLE.unfused_ctas), 8)
    except ValueError as e:
        assert "'Kx'" in str(e), e
    else:
        raise AssertionError("a limit on an undefined constant must not pass silently")


def test_rounding_points_of_the_dense_plan():
    g = build(QWEN3_06B)
    plan = lower(g, BACKEND, rows_max=8)
    fs = check_rounding(g, plan.tasks, TABLE)
    tolerant = {plan.tasks[f.region].kernel for f in fs if f.status in ("acknowledged", "tolerated")}
    assert tolerant == {"g_attn", "g_head"}
    pres = Counter((f.op, f.point) for f in fs if f.status == "preserved")
    assert pres[("silu_mul", "silu")] == 28 and pres[("silu_mul", "out")] == 28 and pres[("add", "out")] == 56
    assert pres[("linear", "out")] == 28 * 7 and pres[("rmsnorm", "out")] == 28 * 4 + 1
    gaps = Counter((f.op, f.point) for f in fs if f.status in ("acknowledged", "tolerated"))
    assert gaps == {("rope", "trig"): 28, ("rope", "prod"): 28, ("attention", "p"): 28, ("linear", "out"): 1}, gaps


def test_checker_fails_when_a_fusion_drops_the_silu_rounding():
    g = build({**QWEN3_06B, "num_layers": 2})
    bare = [replace(t, tolerance=()) for t in lower(g, BACKEND, rows_max=8).tasks]
    dual = TABLE.by_kernel["g_gemv_dual"]
    broken = table_with(replace(dual, rounds={"linear": ("out",), "rmsnorm": ("out",), "silu_mul": ("out",)}))
    try:
        check_rounding(g, bare, broken)
    except RoundingError as e:
        assert "silu_mul.silu" in str(e) and "g_gemv_dual" in str(e), e
    else:
        raise AssertionError("the dropped silu rounding must be reported")
    regions = fuse(g, broken, 8).regions  # the pass records the gap in the region's tolerance
    duals = [r for r in regions if r.kernel == "g_gemv_dual"]
    assert len(duals) == 2 and all(r.tolerance == ("silu_mul.silu",) for r in duals)
    check_rounding(g, regions, broken)
    for r in duals:
        r.tolerance = ()
    try:
        check_rounding(g, regions, broken)
    except RoundingError:
        pass
    else:
        raise AssertionError("an undeclared drop must fail")
    strict = fuse(g, broken, 8, strict_rounding=True)
    assert "g_gemv_dual" not in {r.kernel for r in strict.regions}
    assert any(g.nodes[i].op == "silu_mul" and "rounding gap silu_mul.silu" in w for i, w in strict.unclaimed.items())


def test_checker_fails_on_a_dropped_cast_and_on_undocumented_gaps():
    g = Graph("cast-roundtrip")
    g.input("h", TType(("T", 512), "bf16"))
    g.weight(WeightDecl("w", (512, 512), "dense", "w"))
    back = g.op("cast", [g.op("cast", [g.op("linear", ["h"], weight="w")], dtype="f32")], dtype="bf16")
    g.output(o=g.op("add", ["h", back]))
    fake = Cap("k_fake", "linear", covers_only, rounds={"linear": ("out",), "add": ("out",)})
    region = Region("k_fake", {0, 1, 2, 3}, ins={"x": "h"}, outs={"o": g.outputs["o"]})
    table = CapsTable([fake], RTX3090, TABLE.unfused_ctas)
    try:
        check_rounding(g, [region], table)
    except RoundingError as e:
        assert "node 2 cast.out" in str(e), e
    else:
        raise AssertionError("dropping the bf16 cast must be reported")
    region.tolerance = ("cast.out",)
    check_rounding(g, [region], table)
    g = build({**QWEN3_06B, "num_layers": 2})
    bare = [replace(t, tolerance=()) for t in lower(g, BACKEND, rows_max=8).tasks]
    for kernel, point in (("g_attn", "rope.trig"), ("g_head", "linear.out")):
        try:
            check_rounding(g, bare, table_with(replace(TABLE.by_kernel[kernel], tolerance=())))
        except RoundingError as e:
            assert point in str(e), e
        else:
            raise AssertionError(f"{kernel}: an undocumented gap must fail")


def _check_buffers(g: Graph, plan) -> None:
    """Every value a task reads still lives in its buffer: the next value of that buffer is written only after the last
    reader ran, and only by that reader if it updates the buffer in place; graph outputs end in buffers of their own."""
    task_of = {i: k for k, t in enumerate(plan.tasks) for i in t.covers}
    producer = {o: i for i, n in enumerate(g.nodes) for o in n.outs}
    writer, readers = {}, {}
    for i, n in enumerate(g.nodes):
        for o in n.outs:
            writer[o] = task_of[i]
        for v in n.ins:
            p = producer.get(v)
            if p is not None and task_of[p] != task_of[i]:
                readers.setdefault(v, set()).add(task_of[i])
    for b in plan.buffers.values():
        for v in b.values:
            assert plan.buffer_of[v] == b.name, (v, b.name)
        for v, w in zip(b.values, b.values[1:], strict=False):
            last = max(readers.get(v, {writer[v]}))
            t = plan.tasks[writer[w]]
            inplace = b.name in t.ins.values() and b.name in t.outs.values()
            assert last < writer[w] or (last == writer[w] and inplace), (b.name, v, w)
    for v in g.outputs.values():
        assert plan.buffers[plan.buffer_of[v]].values[-1] == v


def test_buffer_assignment_invariants():
    for g, rows in CASES.values():
        if fuse(g, TABLE, rows).unclaimed:
            continue
        plan = lower(g, BACKEND, rows_max=rows)
        _check_buffers(g, plan)
        assert all(t.ins.get("h") == t.outs.get("h_out") for t in plan.tasks if "h_out" in t.outs)


DETERMINISM_SCRIPT = """
import dataclasses, hashlib
from warpc import frontend
from warpc.compiler import lower
from warpc.cuda import CudaBackend
from warpc.sources import ModelConfig

class Shapes:
    config = ModelConfig(family="qwen3", qk_norm=True, tie_embeddings=False, vocab_size=4096, hidden_size=1024,
                         intermediate_size=3072, num_layers=3, num_heads=16, num_kv_heads=8, head_dim=128,
                         rms_eps=1e-6, rope_theta=1e6, rope_freq_factors=None)
    def __contains__(self, name): return True
    def quant(self, name): return None

plan = lower(frontend.build(Shapes(), ctx_max=256), CudaBackend(), rows_max=8)
print(hashlib.sha256(repr(dataclasses.asdict(plan)).encode()).hexdigest())
"""


def test_lowering_is_deterministic_across_hash_seeds():
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-c", DETERMINISM_SCRIPT], capture_output=True, text=True, check=True, env=env
        )
        outs.add(r.stdout.strip())
    assert len(outs) == 1 and len(next(iter(outs))) == 64, outs


class TensorSource:
    def __init__(self, tensors: dict, raw: dict):
        self.tensors, self.raw_blocks = tensors, raw

    def get(self, name):
        return self.tensors[name]

    def raw(self, name):
        return self.raw_blocks[name]


def test_packer_builds_the_four_kinds():
    torch.manual_seed(0)
    a, b, e = (torch.randn(s).bfloat16() for s in ((8, 64), (4, 64), (16, 64)))
    qa, qb = (torch.randint(0, 256, (n, 2 * 18), dtype=torch.uint8).numpy() for n in (8, 4))
    weights = {
        "a": WeightDecl("a", (8, 64), "dense", "a.weight"),
        "b": WeightDecl("b", (4, 64), "dense", "b.weight"),
        "embed": WeightDecl("embed", (16, 64), "dense", "embed.weight"),
        "lm_head": WeightDecl("lm_head", (16, 64), "dense", "embed.weight"),
        "qa": WeightDecl("qa", (8, 64), "q4_0", "qa.weight"),
        "qb": WeightDecl("qb", (4, 64), "q4_0", "qb.weight"),
        "norm": WeightDecl("norm", (64,), "dense", "norm.weight"),
    }
    dense = {"a.weight": a, "b.weight": b, "embed.weight": e, "norm.weight": e[0]}
    pk = Packer(weights, TensorSource(dense, {"qa.weight": qa, "qb.weight": qb}), device="cpu")
    assert torch.equal(pk.build(Pack("a|b", "dense_cat", ["a", "b"])), torch.cat([a, b]))
    tied = pk.build(Pack("lm_head", "dense", ["lm_head"]))
    assert pk.build(Pack("embed", "dense", ["embed"])) is tied  # one device tensor
    assert torch.equal(pk.build(Pack("qa|qb", "q4_0_cat", ["qa", "qb"])), torch.from_numpy(np.concatenate([qa, qb])))
    assert torch.equal(pk.build(Pack("qa", "q4_0", ["qa"])), torch.from_numpy(qa))
    assert pk.build(Pack("norm", "dense", ["norm"])).shape == (64,)
    for bad in (Pack("qa", "dense", ["qa"]), Pack("a", "q4_0", ["a"])):
        try:
            pk.build(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} must be refused")
    try:
        Pack("x", "w4r_cat", ["x"])
    except ValueError:
        pass
    else:
        raise AssertionError("only the dense pack kinds exist")


def test_kv_bytes_follow_the_cache_format_and_the_step_rows():
    s = StateDecl("kv", (2, 4096, 8, 128), cache="bf16")
    assert CudaBackend.kv_bytes(s, 8) == CudaBackend.kv_bytes(s, 1) == 2 * 8 * 4096 * 128 * 2
    q = replace(s, cache="k8v4w")
    assert 4096 * 8 * (128 + 64 + 4) < CudaBackend.kv_bytes(q, 8) < CudaBackend.kv_bytes(s, 8)
    assert CudaBackend.kv_bytes(q, 8) - CudaBackend.kv_bytes(q, 1) == 2 * 8 * 7 * 128 * 2  # the ring's extra rows
    meta = schedule.kv_meta(FORMATS["k8v4w"], 4096, 8, 128, 3)
    assert meta[8:] == (4, 256, 259, 4), meta


def test_glue_signatures_and_constants_match_the_kernel_sources():
    """Every ctypes signature names an extern "C" entry point with that many parameters; every local include exists; the
    #defines that schedule.py mirrors agree with it."""
    src = "\n".join(p.read_text() for p in sorted(glue.KERNELS.glob("*.c*")))
    for name, (args, _) in glue.SIGNATURES.items():
        m = re.search(r"\b(?:int|size_t|const char\*) " + name + r"\(([^)]*)\)", src)
        assert m, name
        params = [p for p in m.group(1).split(",") if p.strip()]
        assert len(params) == len(args), (name, len(params), len(args))
    files = {p.name for p in glue.KERNELS.iterdir() if p.is_file()}
    for p in glue.KERNELS.glob("*.c*"):
        for inc in re.findall(r'#include "([^"]+)"', p.read_text()):
            assert inc in files, (p.name, inc)
    defines = {k: int(v) for k, v in re.findall(r"^#define (\w+) (\d+)\b", src, re.MULTILINE)}
    mirrored = {
        "STEP_ROWS": schedule.STEP_ROWS,
        "GEMV_WARPS": schedule.GEMV_WARPS,
        "CH": schedule.ATTN_CHUNK,
        "MAX_SPLITS": schedule.ATTN_MAX_SPLITS,
        "MAX_SPLITS2": schedule.ATTN_MAX_SPLITS_CHUNKED,
    }
    assert {k: defines.get(k) for k in mirrored} == mirrored, defines


def _update_golden() -> None:
    """One region per line: a diff of the file shows which regions a change moved."""
    cases = []
    for name, (g, rows) in sorted(CASES.items()):
        value = behaviour(g, rows)
        if "regions" in value:
            regions = ",\n".join("   " + json.dumps(r, sort_keys=True) for r in value["regions"])
            body = '{"regions": [\n' + regions + "\n ]}"
        else:
            body = json.dumps(value, sort_keys=True)
        cases.append(f" {json.dumps(name)}: {body}")
    GOLDEN_PATH.write_text("{\n" + ",\n".join(cases) + "\n}\n")
    print(f"wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    if "--update" in sys.argv[1:]:
        _update_golden()
    run(globals())
