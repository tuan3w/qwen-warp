"""The Inkling-2 GEMV family (g_gemv_multi, g_gemv_convres[_add], g_head_softcap) against the reference interpreter.

  uv run --package warpc python warpc/tests/test_inkling_gemv.py

Each case builds a small graph of the family's ops with random bf16 weights, lets the fusion pass claim it against the
CUDA table, packs the weights and runs the launchers on the GPU (the same code the runtime captures), and compares with
`warpc.reference` run on the same graph and weights. The one change to the oracle is its linear: cuBLAS' bf16
`F.linear` at a handful of rows is itself off by single bf16 roundings, so the oracle's linear is the correctly rounded
float64 product (the old tree's `lin64`). A GEMV output is the bf16 rounding of an fp32 sum: another summation order
than float64's flips the rounding of about 2e-4 of the elements by one ulp (error 1e-6 against an ulp of 4e-3), which is
a relative L2 of about 5e-5 over a few thousand elements; the old tree's 2.1e-5 (GEMV) and 4.2e-5 (convolution
epilogue) were single samples of that. The gates here: every bf16 output within one ulp of the oracle's, at most 5e-4
of the elements differ, aggregate relative L2 at most 1e-4 (the convolution epilogue 4.2e-5), and every segment
bit-equal to `warpc_gemv`, which carries the dense models. Stateful cases run steps of 1, 3, 8, 2, 1 rows through the
states. The GPU tests skip without a device; the lowering test of the real Inkling graph needs none.
"""

import ctypes as C
import math
from collections import Counter

import torch

from harness import Skip, run
from inkling_common import DictSource, real_graph
from warpc.compiler import Task
from warpc.compiler.fuse import fuse
from warpc.compiler.lower import _toposort
from warpc.cuda import glue, launch, schedule
from warpc.cuda.pack import Packer
from warpc.cuda.table import TABLE
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.reference import _IMPL, Interpreter
from warpc.runtime import Memory

BF = torch.bfloat16
GEMV_L2 = 1e-4
CONVRES_L2 = 4.2e-5
FLIP_FRACTION = 5e-4
STEPS = (1, 3, 8, 2, 1)
ROWS = 8


def _device() -> str:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    torch.manual_seed(1234)
    return "cuda"


def rnd(*shape, scale=1.0, dtype=BF):
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


def rel_l2(got, ref) -> float:
    got, ref = got.double(), ref.double()
    return ((got - ref).pow(2).sum() / ref.pow(2).sum().clamp_min(1e-30)).sqrt().item()


class Err:
    """Aggregate error of many outputs against the oracle: relative L2, the number of elements that differ (an fp32
    output: by more than 1e-4 relative, which the last bits of tanh and of a division do not reach) and, for bf16
    outputs, the largest error in units of one bf16 ulp of the larger value or, where a cancelling sum is near zero,
    of 1e-5 of the rms (an fp32 sum's rounding error is absolute, about 1e-6 of the terms)."""

    def __init__(self):
        self.sq = self.ref = 0.0
        self.n = self.flips = 0
        self.ulps = 0.0

    def add(self, got, ref) -> None:
        g, r = got.double(), ref.double()
        diff = (g - r).abs()
        self.sq += diff.pow(2).sum().item()
        self.ref += r.pow(2).sum().item()
        self.n += got.numel()
        if got.dtype == BF:
            self.flips += int((got != ref).sum())
            exp = torch.frexp(torch.maximum(g.abs(), r.abs()))[1].double()
            unit = torch.exp2(exp - 8).clamp_min(1e-5 * r.pow(2).mean().sqrt().item())
            self.ulps = max(self.ulps, (diff / unit).max().item())
        else:
            self.flips += int((diff > 1e-4 * r.abs() + 1e-6).sum())

    @property
    def l2(self) -> float:
        return math.sqrt(self.sq / max(self.ref, 1e-30))

    def check(self, what: str, l2: float) -> None:
        print(f"    {what}: rel L2 {self.l2:.2e}, {self.flips} of {self.n} differ, max {self.ulps:.2f} ulp")
        assert self.l2 <= l2, f"{what}: rel L2 {self.l2:.3e} > {l2}"
        assert self.flips <= FLIP_FRACTION * self.n + 1, f"{what}: {self.flips} of {self.n} elements differ"
        assert self.ulps <= 1.0, f"{what}: {self.ulps} ulp"


def linear64(it, a, x):
    """The correctly rounded linear: float64 product, one rounding to the activation dtype."""
    return (x.double() @ it.weights[a["weight"]].double().T).to(x.dtype)


class Oracle(Interpreter):
    """The reference interpreter with the correctly rounded linear."""

    @torch.no_grad()
    def run(self, inputs):
        impl = {**_IMPL, "linear": linear64}
        env = {name: inputs[name].to(self.device) for name in self.graph.inputs}
        for n in self.graph.nodes:
            outs = impl[n.op](self, n.attrs, *[env[v] for v in n.ins])
            for name, value in zip(n.outs, outs if isinstance(outs, tuple) else (outs,), strict=True):
                env[name] = value
        return {alias: env[v] for alias, v in self.graph.outputs.items()}


class Rig:
    """A graph lowered by the fusion pass onto the CUDA table and run task by task on the GPU: the plan's tasks with the
    graph's value names as their buffers, the packs the Packer builds, conv states as bf16 [C, W - 1]."""

    def __init__(self, graph: Graph, tensors: dict):
        self.graph, self.source = graph, DictSource(tensors)
        res = fuse(graph, TABLE, ROWS)
        assert not res.unclaimed, res.backlog(graph)
        regions = [res.regions[i] for i in _toposort(res.regions)]
        self.kernels = [r.kernel for r in regions]
        self.tasks = [
            Task(
                r.kernel,
                dict(r.ins),
                dict(r.outs),
                dict(r.weights),
                dict(r.states),
                {},
                dict(r.attrs),
                sorted(r.covers),
            )
            for r in regions
        ]
        packer = Packer(graph.weights, self.source)
        self.packs = {p.name: packer.build(p) for r in regions for p in r.packs}
        dtypes = {"bf16": BF, "f32": torch.float32, "i32": torch.int32}
        values = {v for t in self.tasks for v in (*t.ins.values(), *t.outs.values())}
        self.bufs = {}
        for v in values:
            ty = graph.types[v]
            self.bufs[v] = torch.zeros(ROWS * math.prod(d for d in ty.shape[1:]), dtype=dtypes[ty.dtype], device="cuda")
        self.states = {n: torch.zeros(s.shape, dtype=BF, device="cuda") for n, s in graph.states.items()}
        self.scratch = {}

        def scratch(name: str, nbytes: int) -> int:
            if name not in self.scratch or self.scratch[name].numel() < nbytes:
                self.scratch[name] = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
            return self.scratch[name].data_ptr()

        zero = torch.zeros(ROWS, dtype=torch.int32, device="cuda")
        self.mem = Memory(
            ROWS,
            zero.data_ptr(),
            zero.data_ptr(),
            zero.data_ptr(),
            {k: v.data_ptr() for k, v in self.bufs.items()},
            {k: v.data_ptr() for k, v in self.packs.items()},
            {k: v.data_ptr() for k, v in self.states.items()},
            {},
            scratch,
        )
        major, minor = torch.cuda.get_device_capability()
        self.lib = glue.load(10 * major + minor)

    def launch(self, T: int) -> None:
        st = torch.cuda.current_stream().cuda_stream
        for t in self.tasks:
            launch.LAUNCHERS[t.kernel](self.lib, self.mem, t, T, st)

    def step(self, T: int, inputs: dict) -> dict:
        """One window of T rows: graph inputs by name -> graph outputs by alias."""
        for name, x in inputs.items():
            self.bufs[name][: x.numel()].copy_(x.reshape(-1))
        self.launch(T)
        torch.cuda.synchronize()
        out = {}
        for alias, v in self.graph.outputs.items():
            out[alias] = self.bufs[v][: T * math.prod(self.graph.types[v].shape[1:])].reshape(
                T, *self.graph.types[v].shape[1:]
            )
        return out

    def reset(self) -> None:
        for s in self.states.values():
            s.zero_()


def _graph_weights(g: Graph, tensors: dict, name: str, w: torch.Tensor) -> str:
    tensors[name] = w
    return g.weight(WeightDecl(name, tuple(w.shape), "dense", name))


# ------------------------------------------------------------------------------------------------------ g_gemv_multi


def _multi_case(widths, which, K, scale=0.02):
    g, tensors = Graph("multi"), {}
    xs = [g.input(f"x{j}", TType(("T", K), "bf16")) for j in range(max(which) + 1)]
    for j, n in enumerate(widths):
        w = _graph_weights(g, tensors, f"w{j}", rnd(n, K, scale=scale))
        g.output(**{f"o{j}": g.op("linear", [xs[which[j]]], weight=w)})
    g.verify()
    return g, tensors


def _gemv_bit_equal(rig: Rig, T: int, x: dict, out: dict, widths, which, K) -> None:
    """Every segment equals warpc_gemv mode 0 on the same weight rows and input, bit for bit."""
    pack = rig.packs["|".join(f"w{j}" for j in range(len(widths)))]
    off, st = 0, torch.cuda.current_stream().cuda_stream
    for j, n in enumerate(widths):
        o = torch.zeros(T, n, dtype=BF, device="cuda")
        outs = (C.c_void_p * 3)(o.data_ptr(), 0, 0)
        rc = rig.lib.warpc_gemv(
            pack.data_ptr() + off * K * 2,
            None,
            x[f"x{which[j]}"].data_ptr(),
            n,
            K,
            T,
            0,
            None,
            outs,
            None,
            C.c_float(0.0),
            schedule.gemv_rows_per_warp(n, K, 0),
            None,
            0,
            C.c_float(0.0),
            0,
            1,
            st,
        )
        assert rc == 0, rc
        torch.cuda.synchronize()
        assert torch.equal(o, out[f"o{j}"]), f"segment {j} (N={n}) differs from warpc_gemv at T={T}"
        off += n


def _run_multi(widths, which, K, steps=STEPS, scale=0.02):
    _device()
    g, tensors = _multi_case(widths, which, K, scale)
    rig, oracle = Rig(g, tensors), Oracle(g, DictSource(tensors), device="cuda", dtype=BF)
    assert rig.kernels == ["g_gemv_multi"], rig.kernels
    err = Err()
    for T in steps:
        x = {f"x{j}": rnd(T, K, scale=1.5) for j in range(max(which) + 1)}
        got, ref = rig.step(T, x), oracle.run(x)
        for j in range(len(widths)):
            err.add(got[f"o{j}"], ref[f"o{j}"])
        _gemv_bit_equal(rig, T, x, got, widths, which, K)
    err.check(f"multi {widths} K={K} (every segment bit-equal to warpc_gemv)", GEMV_L2)


def test_multi_sliding_visit():
    # q, r, k, v, sink, gate of a sliding loop visit: three inputs
    _run_multi((3072, 384, 1024, 1024, 24, 3072), (0, 0, 1, 2, 0, 0), 2304)


def test_multi_full_visit():
    # q, r, k, k taps, v, v taps, sink, gate of a full visit
    _run_multi((3072, 384, 1024, 32, 1024, 32, 24, 3072), (0, 0, 1, 1, 2, 2, 0, 0), 2304)


def test_multi_one_input_and_rows():
    _run_multi((3072, 384, 1024, 1024, 24, 3072), (0,) * 6, 2304, steps=(1, 2, 3, 4, 5, 6, 7, 8))


def test_multi_k_with_decode_kernel():
    # K = 2048: warpc_gemv at T = 1 runs its all-loads-in-flight kernel, which is bit-identical to the loop kernel
    _run_multi((512, 128, 256), (0, 1, 0), 2048, steps=(1, 3, 1))


def test_multi_awkward_widths():
    # a segment width that is not a multiple of 4 (or 2) lowers the rows per warp
    _run_multi((130, 26, 64, 25), (0, 1, 0, 1), 2304, steps=(1, 8))


# ------------------------------------------------------------------------------------------------------ g_gemv_convres


def _convres_case(N, K, residual):
    g, tensors = Graph("convres"), {}
    x = g.input("x", TType(("T", K), "bf16"))
    w = _graph_weights(g, tensors, "wo", rnd(N, K, scale=0.02))
    cw = _graph_weights(g, tensors, "cw", rnd(N, 4, scale=0.5))
    state = g.state(StateDecl("conv", "conv", (N, 3)))
    o = g.op("linear", [x], weight=w)
    ao = g.op(
        "cast",
        [g.op("causal_conv1d", [o], weight=cw, state=state, residual=True, divisor=math.sqrt(5.0))],
        dtype="f32",
    )
    if residual:
        h = g.input("h", TType(("T", N), "bf16"))
        ao = g.op("add", [g.op("cast", [h], dtype="f32"), ao])
    g.output(res=ao)
    g.verify()
    return g, tensors


def _run_convres(residual, N=2304, K=3072, steps=(*STEPS, 1, 5, 4, 6, 7), exact=False):
    _device()
    g, tensors = _convres_case(N, K, residual)
    if exact:  # small integers: every fp32 sum of the GEMV is exact, so a = bf16(acc) is the same in any order
        tensors["wo"] = torch.randint(-2, 3, (N, K), device="cuda").to(BF)
    rig, oracle = Rig(g, tensors), Oracle(g, DictSource(tensors), device="cuda", dtype=BF)
    assert rig.kernels == ["g_gemv_convres_add" if residual else "g_gemv_convres"], rig.kernels
    err, state_err = Err(), Err()
    for T in steps:
        x = torch.randint(-2, 3, (T, K), device="cuda").to(BF) if exact else rnd(T, K)
        inputs = {"x": x} | ({"h": rnd(T, N, scale=2.0)} if residual else {})
        got, ref = rig.step(T, inputs)["res"], oracle.run(inputs)["res"]
        err.add(got, ref)
        state_err.add(rig.states["conv"], oracle.states["conv"])
    err.check(f"convres residual={residual} exact={exact}", CONVRES_L2)
    state_err.check("  conv state", GEMV_L2)
    return err.l2, err.flips


def test_convres_add():
    _run_convres(True)


def test_convres_block_boundary():
    _run_convres(False)


def test_convres_exact_gemv():
    # with the GEMV exact the epilogue is the only difference to the reference: it matches to the fp32 order of the
    # four-tap sum, so at most a handful of bf16 values flip
    for residual in (False, True):
        _, flips = _run_convres(residual, exact=True)
        assert flips <= 8, flips


def test_convres_odd_rows():
    # N not a multiple of the rows per warp
    _run_convres(True, N=2305, K=1024, steps=(1, 3, 8))


# ------------------------------------------------------------------------------------------------------ g_head_softcap


def _head_case(N, Nw, K):
    g, tensors = Graph("head"), {}
    x = g.input("x", TType(("T", K), "bf16"))
    w = _graph_weights(g, tensors, "unembed", rnd(Nw, K, scale=0.03))
    capped = g.op("softcap", [g.op("cast", [g.op("linear", [x], weight=w)], dtype="f32")], cap=30.0, name="logits")
    logits, _padding = g.op_multi("split", [capped], sizes=[N, Nw - N])
    g.output(logits=logits, ids=g.op("argmax", [logits], name="ids"))
    g.verify()
    return g, tensors


def test_head():
    _device()
    N, Nw, K = 20003, 20096, 2304
    g, tensors = _head_case(N, Nw, K)
    rig, oracle = Rig(g, tensors), Oracle(g, DictSource(tensors), device="cuda", dtype=BF)
    assert rig.kernels == ["g_head_softcap"], rig.kernels
    err = Err()
    for T in (*STEPS, 4, 5, 6, 7):
        x = {"x": rnd(T, K, scale=1.5)}
        got, ref = rig.step(T, x), oracle.run(x)
        err.add(got["logits"], ref["logits"])
        top = ref["logits"].topk(2, dim=-1)
        gap = top.values[:, 0] - top.values[:, 1]
        picked = ref["logits"].gather(1, got["ids"].long()[:, None])[:, 0]
        # the id is the oracle's, or a row within the bf16 noise of the best (a near tie the rounding decides)
        assert bool(((got["ids"] == top.indices[:, 0].int()) | (top.values[:, 0] - picked < 0.05)).all()), gap
    err.check("head logits", GEMV_L2)


def test_head_ties_and_padding():
    _device()
    N, Nw, K = 20003, 20096, 2304
    g, tensors = _head_case(N, Nw, K)
    w = tensors["unembed"].clone()
    T = 8
    x = rnd(T, K, scale=1.5)
    # row t's winner is planted twice (identical rows tie exactly): the lower index must win; the padding rows
    # (>= N) hold a row that would beat everything for row 0 and must not be read
    for t in range(T):
        plant = (x[t].float() * 0.05).to(BF)
        w[7 + t] = plant
        w[19000 + t] = plant
    w[N + 3] = (x[0].float() * 0.2).to(BF)
    tensors["unembed"] = w
    rig = Rig(g, tensors)
    got = rig.step(T, {"x": x})
    assert got["ids"].tolist() == [7 + t for t in range(T)], got["ids"].tolist()
    assert got["logits"].shape == (T, N)
    ref = torch.tanh((x.double() @ w[:N].double().T).to(BF).float() / 30.0) * 30.0
    err = Err()
    err.add(got["logits"], ref)
    err.check("head logits with planted rows", GEMV_L2)


# ------------------------------------------------------------------------------------------------ determinism, graphs


def test_determinism_and_graph_replay():
    _device()
    g, tensors = _multi_case((3072, 384, 1024, 1024, 24, 3072), (0, 0, 1, 2, 0, 0), 2304)
    for name, rig in (
        ("multi", Rig(g, tensors)),
        ("convres", Rig(*_convres_case(2304, 3072, True))),
        ("head", Rig(*_head_case(5003, 5120, 2304))),
    ):
        T = 3
        inputs = {
            v: rnd(T, rig.graph.types[v].shape[1], scale=1.0)
            for v in rig.graph.inputs
            if rig.graph.types[v].dtype == "bf16"
        }
        first = {k: v.clone() for k, v in rig.step(T, inputs).items()}
        rig.reset()
        again = {k: v.clone() for k, v in rig.step(T, inputs).items()}
        assert all(torch.equal(first[k], again[k]) for k in first), f"{name}: a repeated launch differs"
        # the launches captured into a CUDA graph and replayed equal the eager ones
        rig.reset()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            rig.launch(T)
        torch.cuda.current_stream().wait_stream(s)
        rig.reset()
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph):
            rig.launch(T)
        rig.reset()
        cuda_graph.replay()
        torch.cuda.synchronize()
        for alias, v in rig.graph.outputs.items():
            n = T * math.prod(rig.graph.types[v].shape[1:])
            assert torch.equal(rig.bufs[v][:n].reshape(first[alias].shape), first[alias]), f"{name}: graph != eager"


# ------------------------------------------------------------------------------------------------------ lowering (CPU)


def test_real_graph_lowering():
    g = real_graph(rows_max=ROWS)
    res = fuse(g, TABLE, ROWS)
    linears = {i for i, n in enumerate(g.nodes) if n.op == "linear"}
    mine = {"g_gemv_multi", "g_gemv_convres", "g_gemv_convres_add", "g_head_softcap"}
    owner = {i: r.kernel for r in res.regions for i in r.covers if i in linears}
    assert set(owner) == linears and set(owner.values()) <= mine, Counter(owner.values())
    by = Counter(r.kernel for r in res.regions if r.kernel in mine)
    # per visit one multi task (6 segments, 8 on the 8 full visits), one o_proj task, one head
    assert by == {"g_gemv_multi": 48, "g_gemv_convres_add": 40, "g_gemv_convres": 8, "g_head_softcap": 1}, by
    widths = Counter(len(r.attrs["segs"]) for r in res.regions if r.kernel == "g_gemv_multi")
    assert widths == {6: 40, 8: 8}, widths
    assert all(max(len(set(r.ins)), 1) <= 3 for r in res.regions if r.kernel == "g_gemv_multi")
    assert not any(r.tolerance for r in res.regions if r.kernel in mine)
    head = next(r for r in res.regions if r.kernel == "g_head_softcap")
    assert head.attrs["N"] == 200058 and head.attrs["Nw"] == 201024 and head.attrs["cap"] == 30.0


if __name__ == "__main__":
    run(globals())
