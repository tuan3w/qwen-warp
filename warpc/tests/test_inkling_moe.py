"""The Inkling-2 MoE family (g_moe: moe_route + moe_experts + the MLP conv epilogue in one persistent launch).

  uv run --package warpc python warpc/tests/test_inkling_moe.py

CPU: the capability on the real graph (48 visits, one g_moe each), its limits, the pack layout. GPU (a device and nvcc):
the kernel gate against float64 references with the interpreter's rounding points (idx = the float64 router except
true ties, w within 1e-5, output relative L2 <= 1e-3, >= 99.95% of the activations bit-equal), the ascending-expert-id
sum bit for bit, the conv epilogue through the steps 1, 3, 8, 2, 1, determinism, graph replay = eager; the lowered
pack + launcher + capability against the interpreter. Env: WARPC_TEST_INKLING (the W4 checkpoint directory) adds real
layers 0 and 8 to the gate and the interpreter's `moe_route` / `moe_experts` on a real layer. The SASS size of the
T = 1 kernel is held to the old tree's 4,408 instructions plus 2% (the instruction cache is why: README).
"""

import ctypes as C
import math
import os
import shutil
import subprocess
import types
from pathlib import Path

import torch

from harness import Skip, run
from inkling_common import DictSource, real_graph
from test_compiler import check_rounding
from warpc import sources
from warpc.compiler import Pack, lower
from warpc.compiler.fuse import fuse
from warpc.cuda import CudaBackend, glue
from warpc.cuda.inkling import moe as M
from warpc.cuda.pack import Packer
from warpc.cuda.table import TABLE
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.reference import _IMPL, Interpreter
from warpc.runtime import Memory, _Scratch

E, H, FF, TOPK = 128, M.HIDDEN, M.WIDTH, M.TOP_K
ROUTE_SCALE = 8.0
TIE_TOL = 2e-6
DIV = math.sqrt(5.0)
BF = torch.bfloat16
CKPT = os.environ.get("WARPC_TEST_INKLING")


def need_gpu() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


def sm() -> int:
    major, minor = torch.cuda.get_device_capability()
    return 10 * major + minor


# ---------------------------------------------------------------------------------------------------- CPU


def test_caps_real_graph():
    g = real_graph(rows_max=8)
    res = fuse(g, TABLE, 8)
    moe = [r for r in res.regions if r.kernel.startswith("g_moe")]
    assert [r.kernel for r in moe] == ["g_moe"] * 48, [r.kernel for r in moe]
    for r in moe:
        ops = sorted(g.nodes[i].op for i in r.covers)
        assert ops == ["add", "cast", "cast", "causal_conv1d", "moe_experts", "moe_route"], ops
        assert set(r.ins) == {"x", "res"} and set(r.outs) == {"h"} and set(r.states) == {"conv"}
        assert len(r.weights) == 4 and len(r.packs) == 4
    left = {op for op, _why, _n in res.backlog(g)}
    assert not left & {"moe_route", "moe_experts"}, left
    check_rounding(g, moe, TABLE)  # every reference rounding point of the absorbed nodes is reproduced


def test_caps_limits():
    g = real_graph(rows_max=8, intermediate_size=1024)
    res = fuse(g, TABLE, 8)
    why = " ".join(w for op, w, _n in res.backlog(g) if op == "moe_experts")
    assert "I=1024 not in (1152,)" in why, why
    g = real_graph(rows_max=16)
    why = " ".join(w for op, w, _n in fuse(g, TABLE, 16).backlog(g) if op == "moe_experts")
    assert "T=16 > 8" in why, why


def test_stack_layout():
    lay = M.stack_layout(128)
    assert lay.gate[0] == 0 and lay.total == 525_533_184, lay  # the three projections of one real layer
    assert all(v % 256 == 0 for v in (*lay.gate, *lay.up, *lay.down, lay.total))
    assert lay.gate[0] < lay.gate[1] < lay.up[0] < lay.up[1] < lay.down[0] < lay.down[1] < lay.total


# ---------------------------------------------------------------------------------------------------- weights


def dequant(packed, scale, dtype=torch.float64):
    """int32 [N, K/8] + bf16 [N, K/128] -> [N, K]; nibble e of word j is column 8j+e, value (nibble - 8) * scale."""
    n, k8 = packed.shape
    sh = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    q = ((packed.unsqueeze(-1) >> sh) & 0xF).to(dtype) - 8.0
    return q.reshape(n, k8 * 8) * scale.to(dtype).repeat_interleave(128, dim=1)


def make_layer(parts: dict, gw, bias) -> dict:
    """A layer for the kernel: the stacked pack built by the real pack builder from W4G128 `parts` {gate, up, down}."""
    shapes = {"gate": (E, FF, H), "up": (E, FF, H), "down": (E, H, FF)}
    weights = {n: WeightDecl(n, shapes[n], "w4g128", n) for n in shapes}
    stack = Packer(weights, DictSource(parts), "cuda").build(Pack("stack", "w4g128_experts", list(shapes)))
    lay = {"gw": gw, "bias": bias, "stack": stack, "parts": parts}
    for k, n in (("g", "gate"), ("u", "up"), ("d", "down")):
        lay["p" + k], lay["s" + k] = parts[n]["packed"], parts[n]["scale"]
    return lay


def random_layer(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    parts = {}
    for n, rows, k in (("gate", FF, H), ("up", FF, H), ("down", H, FF)):
        packed = torch.randint(-(2**31), 2**31 - 1, (E, rows, k // 8), dtype=torch.int32, device="cuda", generator=g)
        scale = (0.003 + 0.003 * torch.rand(E, rows, k // 128, device="cuda", generator=g)).to(BF)
        parts[n] = {"packed": packed, "scale": scale}
    gw = (0.0208 * torch.randn(E, H, device="cuda", generator=g)).to(BF).contiguous()
    bias = (0.03 * torch.randn(E, device="cuda", generator=g)).float().contiguous()
    return make_layer(parts, gw, bias)


def real_parts(source, layer: int) -> dict:
    p = f"model.llm.layers.{layer}.mlp."
    parts = {
        n: {k: v.to("cuda") for k, v in source.raw(f"{p}experts.{n}_proj.weight").items()}
        for n in ("gate", "up", "down")
    }
    return parts


def real_layer(source, layer: int) -> dict:
    p = f"model.llm.layers.{layer}.mlp."
    gw = source.get(p + "gate.weight").to("cuda").contiguous()
    bias = source.get(p + "gate.bias", torch.float32).to("cuda").contiguous()
    return make_layer(real_parts(source, layer), gw, bias)


# ---------------------------------------------------------------------------------------------------- the kernel


class Moe:
    """ctypes binding of the family's entry points over torch CUDA tensors."""

    def __init__(self):
        self.lib = glue.load(sm())
        self.ws = torch.zeros(self.lib.warpc_moe_ws_bytes(E), dtype=torch.uint8, device="cuda")
        offs = (C.c_size_t * 5)()
        self.lib.warpc_moe_ws_offsets(E, offs)
        self.offs = list(offs)

    @staticmethod
    def stream():
        return torch.cuda.current_stream().cuda_stream

    def alloc(self, T):
        return {
            "out": torch.empty(T, H, dtype=BF, device="cuda"),
            "idx": torch.empty(T, TOPK, dtype=torch.int32, device="cuda"),
            "w": torch.empty(T, TOPK, dtype=torch.float32, device="cuda"),
        }

    def forward(self, x, lay, bufs, conv=None, route_scale=ROUTE_SCALE):
        """The whole layer; with `conv` {res, cw, cst, h, div} the conv epilogue (bufs["out"] is then not written)."""
        T, base, o = x.shape[0], lay["stack"].data_ptr(), M.stack_layout(E)
        cv = conv or {}
        rc = self.lib.warpc_moe_forward(
            x.data_ptr(),
            lay["gw"].data_ptr(),
            lay["bias"].data_ptr(),
            base + o.gate[0],
            base + o.gate[1],
            base + o.up[0],
            base + o.up[1],
            base + o.down[0],
            base + o.down[1],
            None if conv else bufs["out"].data_ptr(),
            bufs["idx"].data_ptr(),
            bufs["w"].data_ptr(),
            self.ws.data_ptr(),
            cv["res"].data_ptr() if conv else None,
            cv["cw"].data_ptr() if conv else None,
            cv["cst"].data_ptr() if conv else None,
            cv["h"].data_ptr() if conv else None,
            T,
            E,
            route_scale,
            cv.get("div", 1.0),
            self.stream(),
        )
        if rc:
            raise RuntimeError(f"warpc_moe_forward: CUDA error {rc}")

    def _view(self, i, nbytes, dtype):
        return self.ws[self.offs[i] : self.offs[i] + nbytes].view(dtype)

    def logits(self, T):
        return self._view(1, T * E * 4, torch.float32).view(T, E)

    def abuf(self, T):
        """The activations a, bf16 values as fp32, un-permuted from the kernel's layout (xpos<4>, a * 2^(-4 (n & 3)))"""
        raw = self._view(3, M.MAX_ROWS * TOPK * FF * 4, torch.float32).view(-1, FF)[: T * TOPK]
        return raw[:, xpos(FF, 4)] * (16.0 ** (torch.arange(FF, device="cuda") & 3))

    def ybuf(self, T):
        return self._view(4, M.MAX_ROWS * TOPK * H * 2, BF).view(-1, H)[: T * TOPK]

    def counters_zero(self):
        return int(self.ws[self.offs[0] : self.offs[0] + 4096].view(torch.int32).abs().sum()) == 0


def xpos(n, lpr):
    """The kernel's layout of an activation vector for row groups of `lpr` lanes: position of every column."""
    c = torch.arange(n, device="cuda")
    b = 32 * lpr
    i, m = c // b, c % b
    ln, w, h, q = m >> 5, (m >> 3) & 3, (m >> 2) & 1, m & 3
    return i * b + (((w * 2 + h) * lpr + ln) << 2) + q


# ---------------------------------------------------------------------------------------------------- references


def route_ref(x, lay):
    """float64 router: logits, scores, keys and the indices sorted by key descending, lowest index first on ties."""
    logits = x.double() @ lay["gw"].double().T
    sc = torch.sigmoid(logits)
    key = sc + lay["bias"].double()
    return logits, sc, key, torch.sort(key, dim=1, descending=True, stable=True).indices


def softcap_act(g, u):
    return torch.tanh(g / 4) * 4 * torch.sigmoid(g) * (torch.tanh(u / 25) * 25)


def experts_ref(x, idx, w, lay):
    """float64 dequantised weights, the interpreter's rounding points (g, u, a, y to bf16): a [T,8,FF], y [T,8,H], out
    [T,H] (pre-cast), and the first-order sensitivity of a to fp32 summation noise in g and u."""
    T = x.shape[0]
    cache = {}

    def W(e):
        if e not in cache:
            cache[e] = tuple(dequant(lay["p" + k][e], lay["s" + k][e]) for k in "gud")
        return cache[e]

    rnd = lambda v: v.to(BF).double()  # noqa: E731
    A = torch.zeros(T, TOPK, FF, dtype=torch.float64, device="cuda")
    ab = torch.zeros_like(A)
    Y = torch.zeros(T, TOPK, H, dtype=torch.float64, device="cuda")
    out = torch.zeros(T, H, dtype=torch.float64, device="cuda")
    for t in range(T):
        xt = x[t].double()
        for k in range(TOPK):
            Wg, Wu, Wd = W(int(idx[t, k]))
            g0, u0 = Wg @ xt, Wu @ xt
            sg_, sgr = Wg.abs() @ xt.abs(), Wu.abs() @ xt.abs()
            c, sig, d = 4 * torch.tanh(g0 / 4), torch.sigmoid(g0), 25 * torch.tanh(u0 / 25)
            ab[t, k] = (sig + 0.25 * c.abs()) * d.abs() * sg_ + c.abs() * sig * sgr
            a = rnd(softcap_act(rnd(g0), rnd(u0)))
            y = rnd(Wd @ a)
            A[t, k], Y[t, k] = a, y
            out[t] += float(w[t, k]) * y
    return A, Y, out, ab


def check_router(moe, x, lay, name, bufs):
    """idx = the float64 router except where the swapped experts' keys sit on the 8th / 9th boundary (true ties), logits
    within 1e-5, w within 1e-5 relative."""
    T = x.shape[0]
    logits, sc, key, order = route_ref(x, lay)
    assert (moe.logits(T).double() - logits).abs().max() < 1e-5, f"{name}: logits"
    nties = 0
    for t in range(T):
        got, ref = [int(v) for v in bufs["idx"][t]], [int(v) for v in order[t, :TOPK]]
        assert len(set(got)) == TOPK, f"{name} t={t}: duplicate experts {got}"
        if got != ref:
            kt = torch.sort(key[t], descending=True).values
            lo, hi = float(kt[TOPK - 1]), float(kt[TOPK])
            if set(got) != set(ref):
                for e in set(got) ^ set(ref):
                    near = abs(float(key[t, e]) - lo) < TIE_TOL or abs(float(key[t, e]) - hi) < TIE_TOL
                    assert near, (
                        f"{name} t={t}: idx {got} vs {ref}, expert {e} key {float(key[t, e])} boundary {lo}/{hi}"
                    )
                nties += 1
            else:
                ks = [float(key[t, e]) for e in got]
                assert all(ks[i] >= ks[i + 1] - TIE_TOL for i in range(TOPK - 1)), f"{name} t={t}: order {got} vs {ref}"
        s = sc[t, bufs["idx"][t].long()]
        wref = s / (s.sum() + 1e-20) * ROUTE_SCALE
        assert torch.allclose(bufs["w"][t].double(), wref, rtol=1e-5, atol=1e-7), f"{name} t={t}: w"
    return nties


def ordered_ref(idx, w, y):
    """The op contract: the rows' contributions added in ascending expert id, fl(w * y) rounded before each fp32 add."""
    T = idx.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    for t in range(T):
        for k in sorted(range(TOPK), key=lambda k: int(idx[t, k])):
            out[t] = out[t] + w[t, k] * y[t * TOPK + k].float()
    return out.to(BF)


def check_experts(moe, x, lay, name, dup=1):
    T = x.shape[0]
    bufs = moe.alloc(T)
    bufs["out"].fill_(float("nan"))
    moe.forward(x, lay, bufs)
    torch.cuda.synchronize()
    assert moe.counters_zero(), f"{name}: counters not left at zero"
    nties = check_router(moe, x, lay, name, bufs)
    distinct = len({int(e) for e in bufs["idx"].flatten()})
    A, Y, out_r, ab = experts_ref(x, bufs["idx"], bufs["w"], lay)
    ka, ky = moe.abuf(T).view(T, TOPK, FF), moe.ybuf(T).view(T, TOPK, H)
    # a: g and u are rounded to bf16 before the activation, so one flipped rounding of g or u (fp32 sums in another
    # order) moves a by up to ~2 bf16 ulp plus its own rounding (3 ulp), and the cancellation of partial sums adds an
    # absolute allowance of 2e-6 x (da/dg sum |Wg x| + da/du sum |Wu x|)
    a_eq = (ka.double() == A).double().mean().item()
    a_bound = 3 * 2.0**-7 * torch.maximum(ka.double().abs(), A.abs()) + 2e-6 * ab + 1e-30
    assert ((ka.double() - A).abs() <= a_bound).all().item(), f"{name}: a beyond the bound"
    assert a_eq >= 1 - 0.0005 * dup, f"{name}: only {a_eq:.5f} of a bit-equal"
    # y: the propagated bound of the flipped a entries, 1 ulp for its own rounding, 1e-6 |W||a| for the fp32 order
    y_eq = (ky.double() == Y).double().mean().item()
    wd_abs = {}
    for t in range(T):
        for k in range(TOPK):
            e = int(bufs["idx"][t, k])
            if e not in wd_abs:
                wd_abs[e] = dequant(lay["pd"][e], lay["sd"][e]).abs()
            Wd = wd_abs[e]
            da = (ka[t, k].double() - A[t, k]).abs()
            bound = (
                Wd @ da
                + 1e-6 * (Wd @ A[t, k].abs())
                + 2.0**-7 * torch.maximum(ky[t, k].double().abs(), Y[t, k].abs())
                + 1e-30
            )
            assert ((ky[t, k].double() - Y[t, k]).abs() <= bound).all().item(), f"{name}: y[{t},{k}] beyond the bound"
    o = bufs["out"].double()
    assert torch.isfinite(o).all(), f"{name}: non-finite output"
    ref_b = out_r.to(BF).double()
    err = ((o - ref_b).norm() / ref_b.norm()).item()
    assert err <= 1e-3, f"{name}: output rel L2 {err:.2e}"
    # the sum itself: ascending expert id, bit for bit, from the kernel's own y / w / idx
    assert torch.equal(bufs["out"], ordered_ref(bufs["idx"], bufs["w"], moe.ybuf(T))), (
        f"{name}: not the ascending-id sum"
    )
    return {"distinct": distinct, "a_eq": a_eq, "y_eq": y_eq, "err": err, "ties": nties}


def make_x(kind, T, gen):
    base = torch.randn(T, H, device="cuda", generator=gen)
    if kind == "dup":  # rows 2i+1 copy rows 2i: every expert of a duplicated row is shared by two tokens
        base[1::2] = base[0::2][: base[1::2].shape[0]]
    if kind == "same":  # all tokens identical: 8 experts carry all T tokens
        base[:] = base[0]
    return base.to(BF)


def layers():
    out = [("random", random_layer())]
    if CKPT:
        source = sources.open(CKPT)
        out += [(f"layer{n}", real_layer(source, n)) for n in (0, 8)]
    return out


def test_gate_experts():
    need_gpu()
    moe, gen = Moe(), torch.Generator(device="cuda").manual_seed(1234)
    for name, lay in layers():
        for T in range(1, 9) if name == "random" else (1, 3, 8):
            for kind in ("rand", "dup", "same"):
                if kind != "rand" and T == 1:
                    continue
                dup = {"rand": 1, "dup": 2, "same": T}[kind]  # rows that repeat each other repeat their flips
                r = check_experts(moe, make_x(kind, T, gen), lay, f"{name} T={T} {kind}", dup)
                print(
                    f"  [{name}] T={T} {kind:4s} distinct {r['distinct']:2d} ties {r['ties']} | a == {r['a_eq']:.5f}"
                    f" y == {r['y_eq']:.4f} | out rel L2 {r['err']:.2e}",
                    flush=True,
                )
        del lay


def test_grid_resident():
    """The launch grid is the occupancy limit clamped to 288 blocks: all of them fit at once (the kernel's blocks wait
    for each other)."""
    need_gpu()
    lib, sms = glue.load(sm()), torch.cuda.get_device_properties(0).multi_processor_count
    for T in range(1, 9):
        for conv in (0, 1):
            n = lib.warpc_moe_grid(T, conv)
            assert 1 <= n <= min(288, 4 * sms), (T, conv, n)
    print(f"  grid {lib.warpc_moe_grid(1, 0)} blocks on {sms} SMs", flush=True)


def test_gate_ties():
    """Exact tie semantics of the router: T = 1 takes the rank path, T > 1 the warp-round path."""
    need_gpu()
    moe, lay = Moe(), random_layer(2)
    for n in (1, 2, 3, 8):
        z = dict(lay)
        # all logits equal (router weight 0, bias 0): scores all 0.5, the 8 lowest ids win and w = 1
        z["gw"], z["bias"] = torch.zeros_like(lay["gw"]), torch.zeros_like(lay["bias"])
        b = moe.alloc(n)
        moe.forward(torch.randn(n, H, device="cuda").to(BF), z, b)
        assert (b["idx"].cpu() == torch.arange(8, dtype=torch.int32).repeat(n, 1)).all(), b["idx"]
        assert torch.allclose(b["w"], torch.ones_like(b["w"]), atol=1e-6)
        # saturated sigmoids (score == 1.0f exactly): the 8 lowest of nine saturated ids win, in ascending id
        sat = [5, 9, 11, 40, 41, 77, 100, 101, 127]
        coef = torch.zeros(E, device="cuda")
        coef[sat] = 3.0
        coef[3] = 0.2  # score 0.99, below the saturated ones
        z["gw"] = coef[:, None].expand(E, H).contiguous().to(BF)
        x = torch.full((n, H), 0.01, device="cuda").to(BF)
        moe.forward(x, z, b)
        assert [int(v) for v in b["idx"][0]] == sat[:8], (b["idx"][0], sat[:8])
        assert torch.allclose(b["w"][0], torch.ones(8, device="cuda"), atol=1e-6)
        # a bias breaks the tie the other way: expert 127 gets +0.25 and comes first in idx
        z["bias"] = torch.zeros(E, device="cuda")
        z["bias"][127] = 0.25
        moe.forward(x, z, b)
        assert [int(v) for v in b["idx"][0]] == [127, *sat[:7]], b["idx"][0]
    # the layer's output of those routings is still the ascending-id sum
    out = moe.alloc(8)
    moe.forward(make_x("rand", 8, torch.Generator(device="cuda").manual_seed(5)), lay, out)
    assert torch.equal(out["out"], ordered_ref(out["idx"], out["w"], moe.ybuf(8)))


def test_gate_determinism_graph():
    need_gpu()
    moe, lay = Moe(), random_layer(3)
    gen = torch.Generator(device="cuda").manual_seed(9)
    for T in (1, 4, 8):
        gx = make_x("rand", T, gen)
        b = moe.alloc(T)
        moe.forward(gx, lay, b)
        torch.cuda.synchronize()
        ref = b["out"].clone()
        other = make_x("rand", 8 if T != 8 else 3, gen)
        ob = moe.alloc(other.shape[0])
        for i in range(10):
            moe.forward(other, lay, ob)  # another T in between leaves no state behind
            moe.forward(gx, lay, b)
            assert torch.equal(b["out"], ref), f"T={T}: output changed on run {i}"
        assert moe.counters_zero()
        # a CUDA graph on static inputs, replayed on new data, equals the eager launch
        sx, sb = gx.clone(), moe.alloc(T)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            moe.forward(sx, lay, sb)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            moe.forward(sx, lay, sb)
        for _ in range(3):
            nx = make_x("rand", T, gen)
            sx.copy_(nx)
            g.replay()
            torch.cuda.synchronize()
            eb = moe.alloc(T)
            moe.forward(nx, lay, eb)
            torch.cuda.synchronize()
            assert torch.equal(sb["out"], eb["out"]) and torch.equal(sb["idx"], eb["idx"]), f"T={T}: replay != eager"


def test_gate_conv_epilogue():
    """The conv epilogue through the steps 1, 3, 8, 2, 1 against the interpreter's causal_conv1d on the plain launch's
    output: the state bit for bit, h within one bf16 ulp (the conv's fp32 sum order), no counter left behind."""
    need_gpu()
    moe, gen = Moe(), torch.Generator(device="cuda").manual_seed(21)
    for name, lay in layers():
        cw = (torch.randn(H, 4, device="cuda", generator=gen) * 0.5).to(BF)
        st_k = torch.randn(H, 3, device="cuda", generator=gen).to(BF)
        st_r = st_k.clone()
        it = types.SimpleNamespace(states={"s": st_r}, weights={"w": cw}, dtype=BF)
        attrs = {"state": "s", "weight": "w", "residual": True, "divisor": DIV}
        for step, T in enumerate((1, 3, 8, 2, 1)):
            x = make_x("rand", T, gen)
            res = torch.randn(T, H, device="cuda", generator=gen) * 3
            plain = moe.alloc(T)
            moe.forward(x, lay, plain)
            h = torch.zeros(T, H, dtype=BF, device="cuda")
            fused = moe.alloc(T)
            fused["out"].fill_(7.0)
            moe.forward(x, lay, fused, conv={"res": res, "cw": cw, "cst": st_k, "h": h, "div": DIV})
            torch.cuda.synchronize()
            ref = (res + _IMPL["causal_conv1d"](it, attrs, plain["out"]).float()).to(BF)
            tag = f"{name} step {step} T={T}"
            assert torch.equal(fused["idx"], plain["idx"]) and torch.equal(fused["w"], plain["w"]), f"{tag}: routing"
            assert torch.equal(st_k, st_r), f"{tag}: conv state"
            assert (fused["out"] == 7.0).all(), f"{tag}: the conv launch wrote the expert output"
            d = (h.float() - ref.float()).abs()
            ulp = torch.maximum(h.float().abs(), ref.float().abs()) * 2.0**-7
            assert (d <= ulp + 1e-30).all(), f"{tag}: h beyond 1 ulp (max {d.max().item()})"
            eq = (h == ref).float().mean().item()
            assert eq >= 0.99, f"{tag}: only {eq:.4f} of h bit-equal"
            assert moe.counters_zero(), f"{tag}: counters"
        print(f"  [{name}] conv epilogue ok through 1, 3, 8, 2, 1", flush=True)


# ---------------------------------------------------------------------------------------------------- lowered


def probe_graph(conv: bool) -> Graph:
    """The MLP half of a visit at the real shape: route + experts (+ conv, cast f32, add res, cast bf16)."""
    g = Graph("moe_probe")
    x = g.input("@x", TType(("T", H), "bf16"))
    res = g.input("@res", TType(("T", H), "f32"))

    def w(name, shape, scheme="dense"):
        return g.weight(WeightDecl(name, shape, scheme, name))

    idx, wt = g.op_multi(
        "moe_route",
        [x],
        weight=w("router", (E, H)),
        bias=w("rbias", (E,), "f32"),
        top_k=TOPK,
        route_scale=ROUTE_SCALE,
        scoring="sigmoid",
    )
    mo = g.op(
        "moe_experts",
        [x, idx, wt],
        gate=w("e_gate", (E, FF, H), "w4g128"),
        up=w("e_up", (E, FF, H), "w4g128"),
        down=w("e_down", (E, H, FF), "w4g128"),
        gate_cap=4.0,
        up_cap=25.0,
    )
    if not conv:
        g.output(out=mo)
        return g
    state = g.state(StateDecl("mlp_conv", "conv", (H, 3)))
    cv = g.op("causal_conv1d", [mo], weight=w("msconv", (H, 4)), state=state, residual=True, divisor=DIV)
    g.output(h=g.op("cast", [g.op("add", [res, g.op("cast", [cv], dtype="f32")])], dtype="bf16"))
    return g


def run_launches(launches) -> None:
    for launch in launches:
        launch.main(torch.cuda.current_stream().cuda_stream)


def test_lowered_vs_interpreter():
    """lower() claims the chain with g_moe (g_moe_plain without the conv), the pack builder and launcher run it: the
    steps 1, 3, 8, 2, 1 against the reference interpreter (routing equal, h / out rel L2), graph replay = eager."""
    need_gpu()
    lay = random_layer(4)
    parts = lay["parts"]
    cw = (torch.randn(H, 4, device="cuda") * 0.5).to(BF)
    source = DictSource(
        {
            "router": lay["gw"],
            "rbias": lay["bias"],
            "e_gate": parts["gate"],
            "e_up": parts["up"],
            "e_down": parts["down"],
            "msconv": cw,
        }
    )
    gen = torch.Generator(device="cuda").manual_seed(31)
    for conv in (True, False):
        g = probe_graph(conv)
        plan = lower(g, CudaBackend(), rows_max=8)
        assert [t.kernel for t in plan.tasks] == ["g_moe" if conv else "g_moe_plain"], plan.summary()
        ex = CudaBackend().executor(g, plan, source)
        packs = {p.name: ex.pack(p) for p in plan.packs.values()}
        bufs = {
            b.name: torch.zeros(8 * b.dim, dtype={"bf16": BF, "f32": torch.float32}[b.dtype], device="cuda")
            for b in plan.buffers.values()
        }
        x_in, res_in = torch.zeros(8, H, dtype=BF, device="cuda"), torch.zeros(8, H, dtype=torch.float32, device="cuda")
        states = {
            s.name: torch.zeros(ex.state_nbytes(s, 8), dtype=torch.uint8, device="cuda") for s in g.states.values()
        }
        mem = Memory(
            rows_max=8,
            pos=0,
            tokens=0,
            ids=0,
            buffers={**{k: v.data_ptr() for k, v in bufs.items()}, "@x": x_in.data_ptr(), "@res": res_in.data_ptr()},
            packs={k: v.data_ptr() for k, v in packs.items()},
            states={k: v.data_ptr() for k, v in states.items()},
            constants={},
            scratch=_Scratch(),
        )
        out_buf = bufs[plan.buffer_of[g.outputs["h" if conv else "out"]]]
        it = Interpreter(g, source, device="cuda")

        for n, T in enumerate((1, 3, 8, 2, 1)):
            x = make_x("rand", T, gen)
            res = torch.randn(T, H, device="cuda", generator=gen) * 0.2
            x_in[:T], res_in[:T] = x, res
            run_launches(ex.schedule(mem, plan.tasks, T))
            torch.cuda.synchronize()
            got = out_buf[: T * H].view(T, H).clone()
            ref = it.run({"@x": x, "@res": res})["h" if conv else "out"]
            err = ((got.float() - ref.float()).norm() / ref.float().norm()).item()
            print(f"  [lowered conv={conv}] step {n} T={T}: rel L2 vs the interpreter {err:.2e}", flush=True)
            assert err <= 2e-3, f"conv={conv} step {n}: {err:.2e}"
            if conv:
                sk, sr = states["mlp_conv"].view(BF).view(H, 3).float(), it.states["mlp_conv"].float()
                assert ((sk - sr).norm() / sr.norm()).item() <= 2e-3, f"step {n}: conv state"  # the state holds mo
        # graph replay = eager (the state rewound before each run)
        if conv:
            launches = ex.schedule(mem, plan.tasks, 3)
            x_in[:3], res_in[:3] = make_x("rand", 3, gen), torch.randn(3, H, device="cuda", generator=gen)
            states["mlp_conv"].zero_()
            run_launches(launches)
            torch.cuda.synchronize()
            eager, st_eager = out_buf[: 3 * H].clone(), states["mlp_conv"].clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run_launches(launches)
            states["mlp_conv"].zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out_buf[: 3 * H], eager) and torch.equal(states["mlp_conv"], st_eager), "replay != eager"


def test_real_layer_vs_reference():
    """One real layer of the W4 checkpoint against the interpreter's own moe_route / moe_experts (GPU bf16)."""
    need_gpu()
    if not CKPT:
        raise Skip("WARPC_TEST_INKLING is not set")
    source = sources.open(CKPT)
    moe, gen = Moe(), torch.Generator(device="cuda").manual_seed(77)
    for layer in (3, 20):
        lay = real_layer(source, layer)
        it = types.SimpleNamespace(
            weights={"router": lay["gw"], "bias": lay["bias"], **{n: lay["parts"][n] for n in ("gate", "up", "down")}},
            dtype=BF,
        )
        route = {"weight": "router", "bias": "bias", "top_k": TOPK, "route_scale": ROUTE_SCALE, "scoring": "sigmoid"}
        experts = {"gate": "gate", "up": "up", "down": "down", "gate_cap": 4.0, "up_cap": 25.0}
        for T in (1, 5, 8):
            x = make_x("rand", T, gen)
            b = moe.alloc(T)
            moe.forward(x, lay, b)
            torch.cuda.synchronize()
            idx, w = _IMPL["moe_route"](it, route, x)
            _logits, _sc, key, _order = route_ref(x, lay)
            bad = 0
            for t in range(T):
                if not torch.equal(idx[t], b["idx"][t]):
                    kt = torch.sort(key[t], descending=True).values
                    lo, hi = float(kt[TOPK - 1]), float(kt[TOPK])
                    swapped = set(idx[t].tolist()) ^ set(b["idx"][t].tolist())
                    ok = all(
                        abs(float(key[t, e]) - lo) < TIE_TOL or abs(float(key[t, e]) - hi) < TIE_TOL for e in swapped
                    )
                    assert ok or set(idx[t].tolist()) == set(b["idx"][t].tolist()), (
                        f"layer {layer} T={T} t={t}: routing"
                    )
                    bad += 1
            assert torch.allclose(b["w"], w, rtol=1e-5, atol=1e-7) or bad, f"layer {layer} T={T}: w"
            ref = _IMPL["moe_experts"](it, experts, x, b["idx"], b["w"])
            err = ((b["out"].float() - ref.float()).norm() / ref.float().norm()).item()
            print(
                f"  [real layer {layer}] T={T}: routing differs in {bad} rows, out rel L2 vs moe_experts {err:.2e}",
                flush=True,
            )
            assert err <= 1e-3, f"layer {layer} T={T}: out rel L2 {err:.2e}"
        del lay, it


# ------------------------------------------------------------------------------------------------------- code size


def sass_instructions(so: Path, kernel_prefix: str) -> dict[str, int]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise Skip("no cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    counts, cur = {}, None
    for line in text.splitlines():
        if "Function :" in line:
            cur = line.split(":")[1].strip()
            counts[cur] = 0
        elif cur and line.strip().startswith("/*") and "*/" in line.split()[0]:
            counts[cur] += 1
    return {k: v for k, v in counts.items() if k.startswith(kernel_prefix)}


def test_code_size():
    """One shared copy of the unrolled K loop: the T = 1 kernel stays near the old 4,408 SASS instructions."""
    need_gpu()
    counts = sass_instructions(glue.build(sm()), "_ZN3moe5k_moeILi1ELi0E")
    assert len(counts) == 1, counts
    (n,) = counts.values()
    print(f"  k_moe<1, 0>: {n} SASS instructions (old: 4408)", flush=True)
    assert n <= 4500, n


if __name__ == "__main__":
    run(globals())
