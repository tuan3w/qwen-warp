"""Kernel gate of the Inkling `res` family (g_attn_res, g_msg_gate, g_convres) against the reference interpreter.

Every case is a small graph lowered through the CUDA table (so the capability, the region emitter, the pack builders and
the launchers are all exercised), run on the GPU in steps of 1, 3, 8, 2, 1 rows through its states, and compared with
`warpc.reference.Interpreter` on the same weights and tokens (the inputs of the op under test come from embeddings of
the step's tokens or, for f32 data, from attn_res mixes whose kernel values the reference takes as given, so both sides
see identical data). Old bounds: attn_res rel L2 <= 9.3e-5, message_gate <= 8.8e-6 (against an fp64 oracle, see
`test_msg_gate_hidden`), conv bit-equal. GPU and nvcc required.
"""

import math
import sys
from pathlib import Path

import torch

from harness import Skip, run

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inkling_common import DictSource
from warpc.compiler import lower
from warpc.cuda.backend import CudaBackend
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.reference import _IMPL, Interpreter
from warpc.runtime import Memory

ROWS = 8
H = 2304
EPS = 1e-6
VOCAB = 64
STEPS = (1, 3, 8, 2, 1)
_DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32}
# the reference's bf16 linears (message_gate's down projection) run on cuBLAS, whose split-K reductions are bf16 by
# default: at 8 rows the interpreter is then 2.4e-3 and 2.1e-3 off an fp64 oracle (9.5e-6 and 5e-8 without it), the
# kernel 1e-5 and 5e-8
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def _need_gpu():
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


class Build:
    """A graph whose activations come from embeddings of the step's tokens."""

    def __init__(self, seed: int, width: int = H):
        self.g, self.tensors, self.width = Graph("res_test"), {}, width
        self.seeds: list[str] = []  # graph outputs whose kernel values feed the reference (inputs of the op under test)
        self.gen = torch.Generator().manual_seed(seed)
        self.tok = self.g.input("tokens", TType(("T",), "i32"))

    def randn(self, *shape, scale=1.0, shift=0.0):
        return torch.randn(*shape, generator=self.gen) * scale + shift

    def dense(self, name, shape, scale=1.0, shift=0.0) -> str:
        self.tensors[name] = self.randn(*shape, scale=scale, shift=shift).to(torch.bfloat16)
        return self.g.weight(WeightDecl(name, tuple(shape), "dense", name))

    def f32(self, name, shape, scale=1.0) -> str:
        self.tensors[name] = self.randn(*shape, scale=scale)
        return self.g.weight(WeightDecl(name, tuple(shape), "f32", name))

    def rows(self, name, scale=1.0, shift=0.0) -> str:
        """bf16 [T, width]: the embedding rows of the step's tokens."""
        return self.g.op("embed", [self.tok], weight=self.dense(name, (VOCAB, self.width), scale, shift))

    def f32_rows(self, name, scale=1.0, shift=0.0) -> str:
        """f32 [T, width] (not bf16-representable): an attn_res mix of two embeddings. The kernel's values are the
        reference's inputs for everything downstream (`seeds`), so only the op under test differs."""
        mix = self.g.op(
            "attn_res",
            [self.rows(name + "_a", scale, shift), self.rows(name + "_b", scale, -shift)],
            query=self.f32(name + "_q", (self.width,), 0.5),
            eps=EPS,
        )
        self.g.output(**{f"seed{len(self.seeds)}": mix})
        self.seeds.append(f"seed{len(self.seeds)}")
        return mix

    def norm_cast(self, x: str, name: str) -> str:
        w = self.dense(name, (self.width,), 0.3, 1.0)
        return self.g.op("cast", [self.g.op("rmsnorm", [x], weight=w, eps=EPS)], dtype="bf16")


class _Scratch:
    def __init__(self):
        self.t, self.frozen = {}, False

    def __call__(self, name, nbytes):
        t = self.t.get(name)
        if t is None or t.numel() < nbytes:
            assert not self.frozen, f"scratch {name} grows after the warm-up"
            t = self.t[name] = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        return t.data_ptr()


class Rig:
    """The plan of a graph on the GPU: packs, buffers, states, and launches of its tasks (eager or as a CUDA graph)."""

    def __init__(self, graph: Graph, tensors: dict):
        self.graph, self.source = graph, DictSource(tensors)
        backend = CudaBackend()
        self.plan = lower(graph, backend, rows_max=ROWS)
        self.ex = backend.executor(graph, self.plan, self.source)
        self.packs = {p.name: self.ex.pack(p) for p in self.plan.packs.values()}
        self.bufs = {
            b.name: torch.zeros(ROWS * b.dim, dtype=_DT[b.dtype], device="cuda") for b in self.plan.buffers.values()
        }
        self.states = {
            s.name: torch.zeros(self.ex.state_nbytes(s, ROWS), dtype=torch.uint8, device="cuda")
            for s in graph.states.values()
        }
        self.tokens = torch.zeros(ROWS, dtype=torch.int32, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.scratch = _Scratch()
        self.mem = Memory(
            rows_max=ROWS,
            pos=self.pos.data_ptr(),
            inputs={"@tokens": self.tokens.data_ptr()},
            buffers={k: v.data_ptr() for k, v in self.bufs.items()},
            packs={k: v.data_ptr() for k, v in self.packs.items()},
            states={k: v.data_ptr() for k, v in self.states.items()},
            constants={},
            scratch=self.scratch,
        )

    def kernels(self) -> list[str]:
        return [t.kernel for t in self.plan.tasks]

    def _launch(self, T: int) -> None:
        st = torch.cuda.current_stream().cuda_stream
        for launch in self.ex.schedule(self.mem, self.plan.tasks, T):
            launch.main(st)

    def step(self, tokens: torch.Tensor) -> None:
        self.tokens[: len(tokens)] = tokens.to("cuda", torch.int32)
        self._launch(len(tokens))
        torch.cuda.synchronize()

    def capture(self, T: int) -> torch.cuda.CUDAGraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._launch(T)
        return g

    def reset(self) -> None:
        for s in self.states.values():
            s.zero_()

    def out(self, alias: str, T: int) -> torch.Tensor:
        v = self.graph.outputs[alias]
        b = self.plan.buffers[self.plan.buffer_of[v]]
        return self.bufs[b.name][: T * b.dim].view(T, b.dim).clone()


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().cpu(), b.double().cpu()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def reference(ref: Interpreter, toks: torch.Tensor, seeds: dict) -> dict:
    """The interpreter's run of the graph with the values in `seeds` (value name -> tensor) taken as given: the
    nodes that produce them are not run."""
    env = {"tokens": toks.to("cuda"), **seeds}
    for n in ref.graph.nodes:
        if all(o in env for o in n.outs):
            continue
        outs = _IMPL[n.op](ref, n.attrs, *[env[v] for v in n.ins])
        for name, value in zip(n.outs, outs if isinstance(outs, tuple) else (outs,), strict=True):
            env[name] = value
    return {alias: env[v] for alias, v in ref.graph.outputs.items()}


def gate(b: Build, steps=STEPS, *, f32_bound: float, bf16_bound: float, exact=False, replay=True) -> dict:
    """Run the graph of `b` through `steps` on the kernels and the reference; returns the worst rel L2 per output."""
    g = b.g
    g.verify()
    rig = Rig(g, b.tensors)
    ref = Interpreter(g, rig.source, device="cuda", dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(1234)
    checked = [a for a in g.outputs if a not in b.seeds]
    worst = dict.fromkeys(checked, 0.0)
    batches = [torch.randint(0, VOCAB, (T,), generator=gen, dtype=torch.int32) for T in steps]
    for toks in batches:
        T = len(toks)
        rig.step(toks)
        want = reference(ref, toks, {g.outputs[s]: rig.out(s, T) for s in b.seeds})
        for alias in checked:
            got, exp = rig.out(alias, T), want[alias]
            if exact:
                assert got.dtype == exp.dtype and torch.equal(got, exp), f"{alias} (T={T}) is not bit-equal"
                continue
            bound = f32_bound if exp.dtype == torch.float32 else bf16_bound
            e = rel_l2(got, exp)
            worst[alias] = max(worst[alias], e)
            assert e <= bound, f"{alias} (T={T}): rel L2 {e:.3e} > {bound:.1e}"
    if replay:
        # determinism (the same steps again from empty states, bit for bit) and graph replay = eager
        rig.scratch.frozen = True
        runs = []
        for _ in range(2):
            rig.reset()
            runs.append([])
            for toks in batches:
                rig.step(toks)
                runs[-1].append({a: rig.out(a, len(toks)) for a in g.outputs})
        for x, y in zip(*runs, strict=True):
            assert all(torch.equal(x[a], y[a]) for a in g.outputs), "two eager runs differ"
        graphs = {T: rig.capture(T) for T in sorted(set(steps))}
        rig.reset()
        for toks, want in zip(batches, runs[0], strict=True):
            T = len(toks)
            rig.tokens[:T] = toks.to("cuda")
            graphs[T].replay()
            torch.cuda.synchronize()
            for a in g.outputs:
                assert torch.equal(rig.out(a, T), want[a]), f"{a} (T={T}): graph replay differs from eager"
    return worst


# ---------------------------------------------------------------- attn_res
def attn_res_build(seed, nrows, segs, *, norm, partial_f32=True, width=H, stacked=None) -> Build:
    b = Build(seed, width)
    rows = [b.rows(f"blk{i}", 0.5 + 0.4 * i, shift=0.1 * i) for i in range(nrows - 1)]
    rows.append(b.f32_rows("partial", 2.0) if partial_f32 else b.rows("partial", 2.0))
    stacked = segs > 1 if stacked is None else stacked
    q = b.f32("query", (segs, width) if stacked else (width,), 0.6)
    for j in range(segs):
        extra = {"row": j} if stacked else {}
        a = b.g.op("attn_res", rows, query=q, eps=EPS, **extra)
        b.g.output(**{f"o{j}": b.norm_cast(a, f"nw{j}") if norm else a})
    return b


def _run_attn_res(label, b, **kw):
    kernels = Rig(b.g, b.tensors).kernels()
    assert kernels.count("g_attn_res") == 1 + len(b.seeds), f"{label}: {kernels}"
    worst = gate(b, f32_bound=9.3e-5, bf16_bound=9.3e-5, **kw)
    print(f"  attn_res {label}: worst rel L2 {max(worst.values()):.2e}")


def test_attn_res_rows():
    _need_gpu()
    for nrows in range(1, 7):
        for partial_f32 in (True, False):
            _run_attn_res(
                f"rows={nrows} partial={'f32' if partial_f32 else 'bf16'} + norm",
                attn_res_build(10 + nrows, nrows, 1, norm=True, partial_f32=partial_f32),
                replay=nrows == 5,
            )
    _run_attn_res("rows=3 f32 out", attn_res_build(3, 3, 1, norm=False), replay=False)


def test_attn_res_segments():
    _need_gpu()
    _run_attn_res("full layer: 3 segments, norms", attn_res_build(21, 4, 3, norm=True))
    _run_attn_res("3 segments, f32", attn_res_build(22, 5, 3, norm=False))
    _run_attn_res("5 segments (a dead slot), f32", attn_res_build(23, 2, 5, norm=False), replay=False)
    _run_attn_res("7 segments, norms", attn_res_build(24, 3, 7, norm=True), replay=False)
    _run_attn_res("48 segments: a pass of messages", attn_res_build(25, 2, 48, norm=False))
    _run_attn_res("H=520, 2 segments", attn_res_build(26, 3, 2, norm=True, width=520), replay=False)


# ---------------------------------------------------------------- message_gate
def msg_gate_build(seed, segs, *, shared_x, norm, width=H) -> Build:
    b = Build(seed, width)
    xs = [b.f32_rows(f"x{i}", 3.0, 0.5) for i in range(1 if shared_x else segs)]
    for j in range(segs):
        m = b.f32_rows(f"m{j}", 1.0)
        w = {
            "down": b.dense(f"down{j}", (64, 2 * width), 0.03),
            "forget": b.dense(f"forget{j}", (width, 64), 0.1),
            "gain": b.dense(f"gain{j}", (width, 64), 0.1),
        }
        gt = b.g.op("message_gate", [xs[0 if shared_x else j], m], eps=EPS, **w)
        b.g.output(**{f"o{j}": b.norm_cast(gt, f"nw{j}") if norm else gt})
    return b


# A bf16 rounding point of the reference (the rank-64 hidden vector) flips for one element in a few percent of the rows
# when the accumulation order differs from cuBLAS's: the output of that row then moves by 5e-4 to 2e-3 (measured). The
# comparison therefore only has the noise bound; the old 8.8e-6 holds against an fp64 oracle fed the kernel's own hidden
# vector (`test_msg_gate_hidden`), which checks the hidden vector itself to one bf16 ulp.
NOISE = 3e-3


def _run_msg_gate(label, b, **kw):
    kernels = Rig(b.g, b.tensors).kernels()
    assert kernels.count("g_msg_gate") == 1, f"{label}: {kernels}"
    worst = gate(b, f32_bound=NOISE, bf16_bound=NOISE, **kw)
    print(f"  message_gate {label}: worst rel L2 vs the interpreter {max(worst.values()):.2e}")


def test_msg_gate():
    _need_gpu()
    _run_msg_gate("1 gate, f32 out", msg_gate_build(31, 1, shared_x=True, norm=False))
    _run_msg_gate("1 gate + norm + cast", msg_gate_build(32, 1, shared_x=True, norm=True))
    _run_msg_gate("3 gates, one x, norms", msg_gate_build(33, 3, shared_x=True, norm=True))
    _run_msg_gate("3 gates, own x, f32", msg_gate_build(34, 3, shared_x=False, norm=False))
    _run_msg_gate("H=520", msg_gate_build(35, 2, shared_x=False, norm=True, width=520), replay=False)


def _nrm(v: torch.Tensor, eps: float) -> torch.Tensor:
    v = v.double()
    return v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)


def test_msg_gate_hidden():
    """The old bound 8.8e-6 against an fp64 oracle: the hidden vector to one bf16 ulp of the exact fp64 one (equal to
    its rounding but for a few flips), the gate's output from the kernel's own hidden vector, and the norm epilogue from
    the kernel's own f32 output."""
    _need_gpu()
    ws_hid, ws_y = 32 * 4, 32 * 4 + 2 * 32 * 64 * 4 * 1  # the workspace layout of kernels/inkling_res.cu
    for segs, shared, norm, width in (
        (1, True, False, H),
        (3, True, True, H),
        (3, False, False, H),
        (2, False, True, 520),
    ):
        b = msg_gate_build(60 + segs, segs, shared_x=shared, norm=norm, width=width)
        rig = Rig(b.g, b.tensors)
        ref = Interpreter(b.g, rig.source, device="cuda", dtype=torch.bfloat16)
        gates = [n for n in b.g.nodes if n.op == "message_gate"]
        nx = 1 if shared else segs
        gen = torch.Generator().manual_seed(7)
        flips = total = 0
        for T in STEPS:
            rig.step(torch.randint(0, VOCAB, (T,), generator=gen, dtype=torch.int32))
            ws = rig.scratch.t["ink_msg_gate"]
            hid = ws[ws_hid : ws_hid + segs * T * 64 * 4].view(torch.float32).view(segs, T, 64)
            ybuf = ws[ws_y : ws_y + segs * ROWS * width * 4].view(torch.float32).view(segs, ROWS, width)
            for c, n in enumerate(gates):
                x, m = rig.out(f"seed{0 if shared else c}", T), rig.out(f"seed{nx + c}", T)
                eps, w = n.attrs["eps"], {k: ref.weights[n.attrs[k]].double().cpu() for k in ("down", "forget", "gain")}
                pair = torch.cat([_nrm(x.cpu(), eps), _nrm(m.cpu(), eps)], -1).to(torch.bfloat16)
                exact = pair.double() @ w["down"].T
                scale = (
                    pair.double().abs() @ w["down"].abs().T
                )  # sum of the |terms|: what the fp32 accumulation errs by
                got = hid[c].cpu().double()
                # at most one bf16 ulp from the exact value, plus the fp32 accumulation error of the sum
                bad = (got - exact).abs() - 2**-8 * exact.abs() - 1e-6 * scale
                assert float(bad.max()) <= 0, f"gate {c} T={T}: hidden vector off by {float(bad.max()):.2e}"
                flips += int((got != exact.to(torch.bfloat16).double()).sum())
                total += got.numel()
                f, g = torch.tanh(got @ w["forget"].T), torch.tanh(got @ w["gain"].T)
                want = (1 - f) * x.cpu().double() + g * m.cpu().double()
                if norm:
                    wn = ref.weights[f"nw{c}"]
                    y = ybuf[c, :T].cpu()
                    ex = (_nrm(y, eps) * wn.double().cpu()).float().to(torch.bfloat16)
                    out = rig.out(f"o{c}", T).cpu()
                    off = (out.double() - ex.double()).abs() / ex.double().abs().clamp_min(1e-30)
                    assert float(off.max()) <= 2**-7 and float((out != ex).float().mean()) <= 0.005, (
                        f"gate {c} norm epilogue"
                    )
                    e = rel_l2(y, want)
                else:
                    e = rel_l2(rig.out(f"o{c}", T), want)
                assert e <= 8.8e-6, (
                    f"gate {c} (T={T}): rel L2 {e:.2e} > 8.8e-6 against the fp64 oracle on its own hidden vector"
                )
        assert flips <= 0.02 * total, f"{flips} of {total} hidden values differ from the correctly rounded ones"
        print(f"  message_gate hidden ({segs} gates, norm={norm}, H={width}): {flips} of {total} hidden one ulp off")


# ---------------------------------------------------------------- conv
def conv_build(seed, variant, width=H) -> Build:
    """variant 0: conv -> bf16; 1: + cast f32; 2: + add (f32 residual stream); 3: + cast bf16."""
    b = Build(seed, width)
    x = b.rows("x", 1.5)
    w = b.dense("w", (width, 4), 0.5)
    b.g.state(StateDecl("conv", "conv", (width, 3)))
    y = b.g.op("causal_conv1d", [x], weight=w, state="conv", residual=True, divisor=math.sqrt(5))
    if variant >= 1:
        y = b.g.op("cast", [y], dtype="f32")
    if variant >= 2:
        y = b.g.op("add", [b.f32_rows("res", 2.0), y])
    if variant >= 3:
        y = b.g.op("cast", [y], dtype="bf16")
    b.g.output(out=y)
    return b


def test_convres():
    _need_gpu()
    for variant in range(4):
        b = conv_build(40 + variant, variant)
        assert Rig(b.g, b.tensors).kernels().count("g_convres") == 1
        gate(b, steps=(*STEPS, 8, 1, 3, 1, 1, 2), f32_bound=0.0, bf16_bound=0.0, exact=True)
        print(f"  convres variant {variant}: bit-equal")
    gate(conv_build(49, 3, 520), f32_bound=0.0, bf16_bound=0.0, exact=True, replay=False)


# ---------------------------------------------------------------- the real graph
def test_real_graph_claims():
    """Every attn_res and message_gate of the real graph is claimed here, with the designed segments and epilogues."""
    from collections import Counter

    from inkling_common import real_graph
    from warpc.compiler.fuse import fuse
    from warpc.cuda.table import TABLE

    g = real_graph(rows_max=ROWS)
    res = fuse(g, TABLE, ROWS)
    mine = {"g_attn_res", "g_msg_gate", "g_convres"}
    regions = [r for r in res.regions if r.kernel in mine]
    covered = Counter(g.nodes[i].op for r in regions for i in r.covers)
    total = Counter(n.op for n in g.nodes)
    assert covered["attn_res"] == total["attn_res"] == 208, (covered, total)
    assert covered["message_gate"] == total["message_gate"] == 96, (covered, total)
    segs = Counter(r.attrs["n"] for r in regions if r.kernel == "g_attn_res")
    assert segs[48] == 2, segs  # one launch of 48 message reads per looped pass
    assert max(segs) == 48 and set(segs) <= {1, 3, 48}, segs
    gates = Counter(r.attrs["n"] for r in regions if r.kernel == "g_msg_gate")
    assert gates == Counter({1: 24, 3: 24}), gates  # a loop layer's q / k / v merges, then its mlp merge
    for r in regions:
        if r.kernel == "g_msg_gate":
            assert all(k.startswith("n") for k in r.outs), "a gate's norm and cast are in its launch"
    print(f"  attn_res regions by segments {dict(segs)}, message_gate {dict(gates)}")


if __name__ == "__main__":
    run(globals())
