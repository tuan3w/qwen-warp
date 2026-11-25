"""Family 6, the Gated DeltaNet step `g_deltanet` (cuda/deltanet.py, kernels/qwen35_deltanet*): capability match on
CPU, then the kernel gates against the hand engine's `swiftmk_deltanet_step_lazy2` on the GPU (design 5.2 row 6).

  WARPC_LEGACY_DIR=<v0.12.0> WARPC_SWIFTMK_LIB=<libswiftmk.so> [WARPC_TEST_SWIFT27=<27B checkpoint dir>] \\
      [WARPC_DN_TRACE=<trace.pt>] uv run --package warpc python warpc/tests/test_deltanet.py

Inputs: the real shapes (Hk 16, Hv 48, dk = dv = 128, C 10240, W 4). The layer weights are the checkpoint's when
WARPC_TEST_SWIFT27 is set (layers 0, 20, 46; layer 47 is a full-attention layer), else synthetic. The step gate
runs from random states and from a state advanced over 64 rows. The drift gate runs a prompt of 385 rows in chunks
of 8 and 1,024 single-row steps; the inputs
are captured stage-1 rows if WARPC_DN_TRACE names a file {layer: {qkv, z, a, b: [rows, ...], conv_w, A_log,
dt_bias, norm_w, prompt: int}} (torch.save), else a synthetic AR(1) generator of the same shapes.
"""

from __future__ import annotations

import ctypes as C
import os
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ab_util import graph_us, stream
from harness import Skip, legacy, run
from warpc import frontend
from warpc.compiler.fuse import fuse
from warpc.cuda import deltanet, glue
from warpc.cuda.table import TABLE
from warpc.reference import _causal_conv1d, _gated_delta_rule, _gated_rmsnorm, _silu
from warpc.sources import ModelConfig
from warpc.sources.config import LinearAttention
from warpc.swiftmk import lib as smk

HK, HV, DK, DV, W, CC = 16, 48, 128, 128, 4, 10240
KEYD, VALD = HK * DK, HV * DV
ULP_TOL, ULP_FRAC = 2, 0.999
DRIFT_TOL = 1e-5
KERNELS = Path(deltanet.__file__).resolve().parent / "kernels"


# ------------------------------------------------------------------------------------------------ capability (CPU)


class _ShapeSource:
    """A source that only has a config: every tensor exists, nothing is quantised."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> str | None:
        return None


def _graph(hk=2, hv=4, dk=128, dv=128, w=4, layers=4):
    lin = LinearAttention(num_k_heads=hk, num_v_heads=hv, k_head_dim=dk, v_head_dim=dv, conv_kernel=w)
    cfg = ModelConfig(
        family="qwen3_5",
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_layers=layers,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        rms_eps=1e-6,
        rope_theta=1e7,
        rope_freq_factors=None,
        qk_norm=True,
        tie_embeddings=False,
        rotary_dim=16,
        norm_offset=1.0,
        attn_output_gate=True,
        layer_types=("linear_attention",) * (layers - 1) + ("attention",),
        linear=lin,
    )
    return frontend.build(_ShapeSource(cfg), ctx_max=128, kv="bf16")


def test_config_agrees_with_the_kernel_header():
    """deltanet.CONFIGS lists exactly the (dk, dv, W) the header instantiates."""
    text = (KERNELS / "qwen35_deltanet.cuh").read_text()
    m = re.search(r"#define DN_CONFIGS\(X\)\s*(.*)", text)
    got = tuple(tuple(int(v) for v in t) for t in re.findall(r"X\((\d+),\s*(\d+),\s*(\d+)\)", m.group(1)))
    assert got == deltanet.CONFIGS, (got, deltanet.CONFIGS)
    rows = re.search(r"constexpr int TMAX = (\d+)", (KERNELS / "qwen35_deltanet.cuh").read_text())
    assert int(rows.group(1)) == deltanet.ROWS_MAX
    header = (KERNELS / "qwen35_deltanet.cuh").read_text()
    assert int(re.search(r"constexpr int NW_TUNED = (\d+)", header).group(1)) == deltanet.WARPS
    assert "return T <= 2 ? 16 : 8;" in header  # deltanet._ctas mirrors lpr_for


def test_rejects_unsupported_shapes():
    """The entry refuses (an error code, before any launch) what no instantiated kernel computes."""
    lib, _ = _gpu()

    def call(T, w=W, dk=DK):
        c = 2 * HK * dk + HV * DV
        return lib.warpc_deltanet_step(T, *[None] * 12, c, w, HK, HV, dk, DV, 1e-6, 0.1, 1e-6, None)

    for what, rc in (("T=0", call(0)), ("T=9", call(9)), ("W=3", call(1, w=3)), ("dk=64", call(1, dk=64))):
        assert rc != 0, what


def test_capability_match():
    """The match covers conv, silu, split, the four reshapes, the delta rule, the gated norm and the output reshape,
    binds the roles and constants of the graph, and is the hand engine's task (one per linear layer)."""
    g = _graph(hk=2, hv=4, layers=4)
    res = fuse(g, TABLE, 8)
    dn = [r for r in res.regions if r.kernel == "g_deltanet"]
    assert len(dn) == 3, [r.kernel for r in res.regions]
    r = dn[0]
    ops = ["causal_conv1d", "silu", "split", "gated_delta_rule", "gated_rmsnorm", *["reshape"] * 5]
    assert sorted(g.nodes[i].op for i in r.covers) == sorted(ops), sorted(g.nodes[i].op for i in r.covers)
    assert set(r.ins) == {"qkv", "z", "a", "b"} and set(r.outs) == {"y"}
    assert set(r.weights) == {"conv_w", "A_log", "dt_bias", "norm_w"} and set(r.states) == {"conv", "rec"}
    assert r.attrs == {
        "C": 2 * 2 * 128 + 4 * 128,
        "W": 4,
        "Hk": 2,
        "Hv": 4,
        "dk": 128,
        "dv": 128,
        "eps": 1e-6,
        "norm_eps": 1e-6,
        "scale": 128**-0.5,
    }, r.attrs
    assert [p.kind for p in r.packs] == ["dense"] * 4
    # the hand kernel's gaps, plus the unrounded o (the hand table lists the same points as its tolerance)
    assert set(r.tolerance) == {
        "causal_conv1d.out",
        "silu.out",
        "gated_delta_rule.beta",
        "gated_delta_rule.out",
        "gated_rmsnorm.norm",
        "gated_rmsnorm.scale",
    }, r.tolerance
    assert g.states[r.states["rec"]].shape == (4, 128, 128)  # the plain in-place layout


def test_capability_declines_with_a_reason():
    for kw, text in (
        ({"dk": 64}, "not in"),
        ({"w": 3}, "not in"),
    ):
        g = _graph(**kw, layers=2)
        res = fuse(g, TABLE, 8)
        assert not [r for r in res.regions if r.kernel == "g_deltanet"], kw
        why = " ".join(str(v) for v in res.unclaimed.values())
        assert text in why, (kw, why)
    g = _graph(layers=2)
    assert not [r for r in fuse(g, TABLE, 9).regions if r.kernel == "g_deltanet"], "T_max 8"


# ------------------------------------------------------------------------------------------------ GPU gates


def _gpu():
    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    return glue.load(10 * major + minor), smk.load()


def _bf(x):
    return x.to(torch.bfloat16).cuda()


def _synthetic_weights(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "conv_w": _bf(torch.randn(CC, W, generator=g) * 0.5),
        "A_log": _bf(torch.log(torch.empty(HV).uniform_(1.0, 16.0, generator=g))),
        "dt_bias": _bf(torch.randn(HV, generator=g) * 0.5 + 1.0),
        "norm_w": _bf(torch.randn(DV, generator=g) * 0.1 + 1.0),
    }


def weights(layer: int, path: str | None = None) -> dict:
    """Layer `layer`'s weights: the checkpoint's (`path`, else $WARPC_TEST_SWIFT27) if set, else synthetic."""
    path = path or os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        return _synthetic_weights(layer)
    from warpc import sources

    src = sources.open(path)
    p = "model.language_model.layers." if "model.language_model.embed_tokens.weight" in src else "model.layers."
    p = f"{p}{layer}.linear_attn."
    return {
        "conv_w": _bf(src.get(p + "conv1d.weight").reshape(CC, W)),
        "A_log": _bf(src.get(p + "A_log")),
        "dt_bias": _bf(src.get(p + "dt_bias")),
        "norm_w": _bf(src.get(p + "norm.weight")),
    }


def synthetic_rows(n: int, seed: int) -> dict:
    """AR(1) rows (correlation 0.8 between steps) of the four inputs, bf16 on the GPU."""
    g = torch.Generator().manual_seed(seed)

    def ar(width: int, scale: float) -> torch.Tensor:
        x = torch.empty(n, width)
        x[0] = torch.randn(width, generator=g)
        for i in range(1, n):
            x[i] = 0.8 * x[i - 1] + 0.6 * torch.randn(width, generator=g)
        return _bf(x * scale)

    return {"qkv": ar(CC, 1.0), "z": ar(VALD, 1.0), "a": ar(HV, 1.0), "b": ar(HV, 1.0)}


class Ours:
    """The warpc kernel on one layer: state [Hv, dk, dv] fp32, conv [C, W-1] bf16, both in place."""

    def __init__(self, lib, w):
        self.lib, self.w = lib, w
        self.state = torch.zeros(HV, DK, DV, dtype=torch.float32, device="cuda")
        self.conv = torch.zeros(CC, W - 1, dtype=torch.bfloat16, device="cuda")
        self.ws = torch.zeros(lib.warpc_deltanet_ws_bytes(HK, HV, DV), dtype=torch.uint8, device="cuda")

    def launch(self, T, qkv, z, a, b, y, st):
        w = self.w
        args = (qkv, z, a, b, w["conv_w"], w["A_log"], w["dt_bias"], w["norm_w"], self.state, self.conv, y, self.ws)
        rc = self.lib.warpc_deltanet_step(
            T,
            *(t if isinstance(t, int) else t.data_ptr() for t in args),
            CC,
            W,
            HK,
            HV,
            DK,
            DV,
            1e-6,
            DK**-0.5,
            1e-6,
            st,
        )
        assert rc == 0, rc

    def step(self, qkv, z, a, b):
        y = torch.empty(qkv.shape[0], VALD, dtype=torch.bfloat16, device="cuda")
        self.launch(qkv.shape[0], qkv.data_ptr(), z.data_ptr(), a.data_ptr(), b.data_ptr(), y.data_ptr(), stream())
        return y


class Hand:
    """swiftmk_deltanet_step_lazy2 as the engine runs it: one physical state slot [Hv][dv][dk], a conv ring of 9
    slots, the step reads ring slot cur. A step of T >= 2 rows leaves pending rows: `step` materialises them, so the
    state is always complete (what the all-rows-accepted semantics of the graph need)."""

    SLOTS = 9

    def __init__(self, lib, w):
        self.lib, self.w = lib, w
        self.state = torch.zeros(HV, DV, DK, dtype=torch.float32, device="cuda")
        self.conv = torch.zeros(self.SLOTS, CC, W - 1, dtype=torch.bfloat16, device="cuda")
        self.ws = torch.zeros(lib.swiftmk_deltanet_ws_bytes(), dtype=torch.uint8, device="cuda")
        self.dbuf = torch.zeros(lib.swiftmk_deltanet_dbuf_floats(), dtype=torch.float32, device="cuda")
        self.cur = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.pend = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.cur_i = 0

    def load(self, state, conv):
        """State from the graph layout [Hv, dk, dv] and conv [C, W-1]."""
        self.state.copy_(state.permute(0, 2, 1))
        self.conv.zero_()
        self.conv[0].copy_(conv)
        self.cur.zero_()
        self.pend.zero_()
        self.cur_i = 0

    def graph_state(self):
        return self.state.permute(0, 2, 1)

    def graph_conv(self):
        return self.conv[self.cur_i]

    def launch(self, T, qkv, z, a, b, y, st):
        w = self.w
        args = (qkv, z, a, b, w["conv_w"], w["A_log"], w["dt_bias"], w["norm_w"], self.state, self.conv)
        rc = self.lib.swiftmk_deltanet_step_lazy2(
            T,
            *(t if isinstance(t, int) else t.data_ptr() for t in args),
            self.cur.data_ptr(),
            self.SLOTS,
            1,
            y,
            self.ws.data_ptr(),
            self.dbuf.data_ptr(),
            self.pend.data_ptr(),
            None,
            st,
        )
        assert rc == 0, rc

    def materialize(self, st):
        # not in the swiftmk binding's table (the engine calls it between steps): bound here
        fn = self.lib.cdll.swiftmk_deltanet_materialize
        fn.argtypes, fn.restype = [C.c_void_p] * 2 + [C.c_int] + [C.c_void_p] * 2 + [C.c_int, C.c_void_p], C.c_int
        rc = fn(self.state.data_ptr(), self.dbuf.data_ptr(), 1, self.cur.data_ptr(), self.pend.data_ptr(), 1, st)
        assert rc == 0, rc

    def step(self, qkv, z, a, b):
        T = qkv.shape[0]
        y = torch.empty(T, VALD, dtype=torch.bfloat16, device="cuda")
        self.launch(T, qkv.data_ptr(), z.data_ptr(), a.data_ptr(), b.data_ptr(), y.data_ptr(), stream())
        base, self.cur_i = self.cur_i, (self.cur_i + T) % self.SLOTS
        self.cur.fill_(self.cur_i)
        if T == 1:
            self.pend.zero_()
        else:
            self.pend.fill_(T | base << 8)
            self.materialize(stream())
        return y


def ulp_errors(y, ref):
    """|y - ref| in bf16 ulps of the reference value (per element)."""
    r = ref.float()
    ulp = torch.exp2(torch.floor(torch.log2(r.abs().clamp_min(1e-30))) - 7)
    return (y.float() - r).abs() / ulp


def rel_l2(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def reference_step(w, state, conv, qkv, z, a, b):
    """The interpreter's numerics (reference.py) on the GPU: (y, new state, new conv) without touching the inputs."""
    it = SimpleNamespace(
        states={"c": conv.clone(), "r": state.clone()},
        weights={"cw": w["conv_w"], "al": w["A_log"], "dt": w["dt_bias"], "nw": w["norm_w"]},
    )
    T = qkv.shape[0]
    c = _silu(it, {}, _causal_conv1d(it, {"state": "c", "weight": "cw"}, qkv))
    q, k, v = c.split([KEYD, KEYD, VALD], -1)
    q, k, v = q.reshape(T, HK, DK), k.reshape(T, HK, DK), v.reshape(T, HV, DV)
    attrs = {"state": "r", "a_log": "al", "dt_bias": "dt", "eps": 1e-6, "scale": DK**-0.5}
    o = _gated_delta_rule(it, attrs, q, k, v, a, b)
    o = _gated_rmsnorm(it, {"weight": "nw", "eps": 1e-6}, o, z.reshape(T, HV, DV))
    return o.reshape(T, VALD), it.states["r"], it.states["c"]


def _random_state(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(HV, DK, DV, generator=g) * 0.3).cuda(), _bf(torch.randn(CC, W - 1, generator=g))


def test_step_vs_hand():
    """Gate (i): per step y within 2 bf16 ulp of the hand's on 99.9% of the elements, T = 1..8, from the same initial
    state: random, then the state after 64 rows. The conv state is exact and the fp32 state agrees; the interpreter's
    numerics are measured against both (the decision to follow the hand)."""
    ours_lib, hand_lib = _gpu()
    layer = 20
    w = weights(layer)
    rows = synthetic_rows(64 + 8, 7)
    print(
        "T  init    y>2ulp(ours-hand)  max ulp  state relL2  conv  bitwise y,state"
        " | ref-vs-hand y>2ulp  ref-vs-ours  ref state relL2"
    )
    for init in ("random", "warm"):
        for T in range(1, 9):
            ours, hand = Ours(ours_lib, w), Hand(hand_lib, w)
            if init == "random":
                S, cv = _random_state(100 + T)
            else:
                warm = Hand(hand_lib, w)
                warm.load(
                    torch.zeros(HV, DK, DV, device="cuda"), torch.zeros(CC, W - 1, dtype=torch.bfloat16, device="cuda")
                )
                for i in range(64):
                    warm.step(*(rows[k][i : i + 1] for k in ("qkv", "z", "a", "b")))
                S, cv = warm.graph_state().contiguous(), warm.graph_conv().clone()
            ours.state.copy_(S)
            ours.conv.copy_(cv)
            hand.load(S, cv)
            x = [rows[k][64 : 64 + T].contiguous() for k in ("qkv", "z", "a", "b")]
            y_ref, S_ref, cv_ref = reference_step(w, S, cv, *x)
            y_o, y_h = ours.step(*x), hand.step(*x)
            torch.cuda.synchronize()
            e = ulp_errors(y_o, y_h)
            bad = (e > ULP_TOL).float().mean().item()
            conv_ok = torch.equal(ours.conv, hand.graph_conv())
            sl2 = rel_l2(ours.state, hand.graph_state())
            eh, eo = ulp_errors(y_h, y_ref), ulp_errors(y_o, y_ref)
            same = torch.equal(y_o, y_h), torch.equal(ours.state, hand.graph_state())
            print(
                f"{T}  {init:6s}  {bad:.2e}           {e.max().item():7.1f}  {sl2:.2e}    {conv_ok}  {same} |"
                f" {(eh > 2).float().mean().item():.2e}            {(eo > 2).float().mean().item():.2e}"
                f"      {rel_l2(S_ref, hand.graph_state()):.2e}"
            )
            assert 1 - bad >= ULP_FRAC, (T, init, bad)
            assert conv_ok and torch.equal(ours.conv, cv_ref), (T, init)
            assert sl2 <= DRIFT_TOL, (T, init, sl2)


def _trace(layer: int):
    """(weights, rows, prompt length) of a captured trace, or None."""
    path = os.environ.get("WARPC_DN_TRACE")
    if not path:
        return None
    t = torch.load(path)[layer]
    return (
        {k: _bf(t[k]) for k in ("conv_w", "A_log", "dt_bias", "norm_w")},
        {k: _bf(t[k]) for k in ("qkv", "z", "a", "b")},
        int(t["prompt"]),
    )


def drift(ours_lib, hand_lib, layer: int, w, rows, prompt: int):
    """Both kernels advance their own state over `rows` from zero: the prompt in chunks of 8, then single rows.
    Returns the fp32 state rel-L2 difference after every decode step and the fraction of y elements (over all steps)
    more than 2 bf16 ulp from the hand's."""
    ours, hand = Ours(ours_lib, w), Hand(hand_lib, w)
    hand.load(ours.state.clone(), ours.conv.clone())
    n = rows["qkv"].shape[0]
    chunks = [(i, min(i + 8, prompt)) for i in range(0, prompt, 8)] + [(i, i + 1) for i in range(prompt, n)]
    diffs, bad, total = [], 0, 0
    for lo, hi in chunks:
        x = [rows[k][lo:hi].contiguous() for k in ("qkv", "z", "a", "b")]
        y_o, y_h = ours.step(*x), hand.step(*x)
        e = ulp_errors(y_o, y_h)
        bad, total = bad + (e > ULP_TOL).sum().item(), total + e.numel()
        if lo >= prompt:
            diffs.append(rel_l2(ours.state, hand.graph_state()))
    assert torch.equal(ours.conv, hand.graph_conv())
    return np.array(diffs), bad / total


def test_drift():
    """Gate (ii): the fp32 state rel-L2 difference to the hand kernel is <= 1e-5 at every decode step and its linear fit
    over the last 512 steps does not rise. The difference of two correct fp32 kernels is rounding noise around a level,
    so the sign of the fitted slope of a flat series is a coin flip: the series passes if the slope is <= 0 or the
    fitted rise over the window is under 5% of its level (both are printed)."""
    ours_lib, hand_lib = _gpu()
    for layer in (0, 20, 46):
        got = _trace(layer)
        if got:
            w, rows, prompt = got
            src = "trace"
        else:
            w, rows, prompt, src = weights(layer), synthetic_rows(385 + 1024, 1000 + layer), 385, "synthetic"
        d, bad = drift(ours_lib, hand_lib, layer, w, rows, prompt)
        slope = np.polyfit(np.arange(512), d[-512:], 1)[0]
        rise = slope * 511 / d[-512:].mean() if d[-512:].any() else 0.0
        print(
            f"drift layer {layer:2d} ({src}): relL2 max {d.max():.2e} mean {d.mean():.2e} last {d[-1]:.2e}"
            f" slope(last 512) {slope:+.2e}/step = {rise:+.2%} of the level over the window,"
            f" y > 2 ulp {bad:.1e}"
        )
        assert d.max() <= DRIFT_TOL, (layer, d.max())
        assert slope <= 0.0 or rise < 0.05, (layer, slope, rise)
        assert 1 - bad >= ULP_FRAC, (layer, bad)


LAYERS = 48  # one decode step's DeltaNet launches: 144 MiB of states, far above the 6 MiB L2


def _alternating(f_ours, f_hand, rounds: int = 3):
    """(ours, hand) median microseconds per call of two functions that each make LAYERS launches."""
    o, h = [], []
    for _ in range(rounds):
        o.append(graph_us(f_ours, inner=2, replays=100) / LAYERS)
        h.append(graph_us(f_hand, inner=2, replays=100) / LAYERS)
    return statistics.median(o), statistics.median(h)


def _speed_row(ours_lib, hand_lib, w, rows, T):
    """(ours, hand, hand + materialize or 0) microseconds per launch of a T-row step over LAYERS layers' states."""
    S, cv = _random_state(5)
    ours, hands = [Ours(ours_lib, w) for _ in range(LAYERS)], [Hand(hand_lib, w) for _ in range(LAYERS)]
    for o, h in zip(ours, hands, strict=True):
        o.state.copy_(S)
        o.conv.copy_(cv)
        h.load(S, cv)
    x = [rows[k][:T].contiguous() for k in ("qkv", "z", "a", "b")]
    y = torch.empty(T, VALD, dtype=torch.bfloat16, device="cuda")
    p = [t.data_ptr() for t in (*x, y)]

    def f_ours():
        for o in ours:
            o.launch(T, *p, stream())

    def f_hand():
        for h in hands:
            h.launch(T, *p, stream())

    def f_hand_m():
        # the step finds pend == 0 (the materialize clears it); fill_ is a captured memset node, ~2 us of the number
        for h in hands:
            h.launch(T, *p, stream())
            h.pend.fill_(T)
            h.materialize(stream())

    o, h = _alternating(f_ours, f_hand)
    return o, h, _alternating(f_ours, f_hand_m)[1] if T > 1 else 0.0


def test_speed():
    """Both kernels in CUDA graphs, one launch per layer over 48 layers' states (DRAM-resident, as in a decode step),
    3 alternating rounds: decode (T = 1) and prefill (T = 8) steps. The hand's T >= 2 step is lazy (it writes no state:
    pending rows, the deferred write is its `materialize`): the plain table row is the step alone, the second adds
    the materialize that an all-rows-accepted step needs."""
    ours_lib, hand_lib = _gpu()
    w = weights(20)
    rows = synthetic_rows(8, 3)
    print("T   ours us   hand us  hand/ours   hand+materialize us  hand/ours")
    for T in (1, 2, 3, 4, 8):
        o, h, hm = _speed_row(ours_lib, hand_lib, w, rows, T)
        extra = f"{hm:9.1f}  {hm / o:6.3f}" if hm else ""
        print(f"{T}  {o:8.2f}  {h:8.2f}  {h / o:8.3f}          {extra}")


def capture(model_dir: str, out: str, layers=(0, 20, 46), steps: int = 1024) -> None:
    """Record the real inputs of the hand engine's DeltaNet launches of `layers` over a 385-token prompt and `steps`
    greedy tokens (stage-1 eager decode through ab_util.Tap, the 27B alone on the card) into the WARPC_DN_TRACE file."""
    import warpc
    from ab_util import Tap
    from warpc.swiftmk import SwiftmkBackend

    corpus = legacy().corpus()
    from prompts import prompt_ids

    ids = prompt_ids(Path(model_dir), corpus, 385)
    tap = Tap(SwiftmkBackend("bf16"), ("s_deltanet",), layers=tuple(layers))
    m = warpc.compile(model_dir, ctx_max=2048, rows_max=8, kv="bf16", backend=tap.backend)
    tap.run(m, ids, steps)
    trace = {}
    for layer in layers:
        recs = [r for r in tap.records if r["layer"] == layer]
        prompt = sum(r["T"] for r in recs if r["pos"] < len(ids) - 1)
        trace[layer] = {
            **{k: torch.cat([r["ins"][k][: r["T"]] for r in recs]).cpu() for k in ("qkv", "z", "a", "b")},
            **{k: v.cpu() for k, v in weights(layer, model_dir).items()},
            "prompt": prompt,
        }
        print(f"layer {layer}: {sum(r['T'] for r in recs)} rows, prompt rows {prompt}")
    torch.save(trace, out)


if __name__ == "__main__":
    if sys.argv[1:2] == ["capture"]:
        capture(sys.argv[2], sys.argv[3])
    else:
        run(globals())
