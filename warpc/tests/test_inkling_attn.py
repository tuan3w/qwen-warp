"""Inkling attention block (`g_attn_rel`: prep + split + merge of kernels/inkling_attn.cu) against the reference.

  uv run --package warpc python warpc/tests/test_inkling_attn.py

CPU: on the real Inkling graph the fusion pass claims every attention node (48 visits: 40 sliding, 8 full) with the
whole block absorbed, and the rounding check accepts the regions with `attention.p` as the only open point.

GPU (a device and nvcc): a hand-built one-visit graph (the projections' outputs are graph inputs "@q", "@r", ... that
the test fills: they are what the GEMV family writes) is lowered by the real pass onto the CUDA backend; the task's
launcher runs on a hand-built `Memory`, and the reference `Interpreter` (bf16 on the GPU) runs the same graph on the
same random states.
  steps         1, 3, 8, 2, 1 rows through the states (the old protocol), 8-row steps through positions >= 8,990 (log
                scaling active) and a ring that wraps, ctx up to 33,000 (chunked split blocks), group sizes 1, 2, 3, 5
  attention     |kernel - reference| <= 2 bf16 ulp + 2^-7 * max|attention| of the (row, head), rel L2 <= 3.3e-3 over
                all cases (the reference rounds P to bf16, the kernel does not); and against a float64 attention
                without that rounding on the kernel's own prep outputs: <= 3 ulp + 2^-12 * max
  prep          conv states bit-equal; clipped q / r and the K / V rows written to the state: at most 1 bf16 ulp apart,
                rel L2 <= 2e-3 (the clip norms sum in another order than torch's)
  windows       8 rows at once = 4 + 4 = 8 x 1: outputs, KV rows and conv states bit-identical
  determinism   a rerun from the same states with the workspace poisoned is bit-identical
  graph         the launches captured in a CUDA graph at one position and replayed at others = the eager launches
"""

import math

import torch

from harness import Skip, run
from inkling_common import DictSource, real_graph
from warpc.compiler import lower
from warpc.compiler.fuse import fuse
from warpc.cuda.backend import CudaBackend
from warpc.cuda.table import TABLE
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.reference import Interpreter
from warpc.runtime import Memory, _Scratch

D, DREL, W = 128, 16, 4
BF = torch.bfloat16
L2_CASE = 6e-3  # one step: a few keys carry all the probability, so the reference's bf16 P rounding is not averaged out
L2_ALL = 3.3e-3  # over all cases (the old bound)
FLOOR_REL = (
    2.0**-7
)  # allowance against the reference's bf16-rounded probabilities, relative to the largest |attention| of the row
ROWS = 8
FAILS: list[str] = []


# --- CPU: the real graph ---
def test_real_graph_claims_every_attention_block():
    g = real_graph(rows_max=ROWS)
    res = fuse(g, TABLE, ROWS)
    attn = [i for i, n in enumerate(g.nodes) if n.op == "attention"]
    assert len(attn) == 48
    regions = [r for r in res.regions if r.kernel == "g_attn_rel"]
    assert len(regions) == 48, res.backlog(g)
    assert {next(i for i in r.covers if g.nodes[i].op == "attention") for r in regions} == set(attn)
    full = [r for r in regions if r.attrs["dyn"]]
    sliding = [r for r in regions if not r.attrs["dyn"]]
    assert (len(sliding), len(full)) == (40, 8)
    for r in sliding:  # attention, 4 reshapes, 3 clips, 2 convs, the v norm, the gate cast, the 3-node output chain
        assert len(r.covers) == 15 and r.attrs["window"] == 1024 and r.attrs["R"] == 1031 and not r.attrs["logs"]
        assert set(r.ins) == {"q", "r", "k", "v", "sink", "gate"} and r.attrs["nsplit"] == 16
    for r in full:  # plus the 2 log scalings; the taps of the dynamic convs are inputs
        assert len(r.covers) == 17 and r.attrs["window"] == 0 and r.attrs["logs"] and r.attrs["alpha"] == 0.1
        assert set(r.ins) == {"q", "r", "k", "v", "sink", "gate", "ktap", "vtap"}
    for r in regions:
        assert r.tolerance == ("attention.p",), r.tolerance
        assert len(r.outs) == 1 and set(r.states) == {"kv", "kconv", "vconv"} and len(r.packs) == 5
    from test_compiler import check_rounding

    check_rounding(g, regions, TABLE)


# --- one visit's graph ---
def build_graph(kind, Hq, Hkv, window, E, R, expose=False):
    """The attention block of frontend/inkling.py for one visit, the projection outputs as inputs "@name"."""
    full = kind == "full"
    C_ = Hkv * D
    g = Graph("inkling-attn")
    pos = g.input("pos", TType(("T",), "i32"))
    ins = {
        n: g.input(f"@{n}", TType(("T", dim), "bf16"))
        for n, dim in (("q", Hq * D), ("r", Hq * DREL), ("k", C_), ("v", C_), ("sink", Hq), ("gate", Hq * D))
    }
    if full:
        ins |= {n: g.input(f"@{n}", TType(("T", Hkv * W), "bf16")) for n in ("ktap", "vtap")}
    wts = {
        "qs": (Hq, D),
        "rs": (Hq, DREL),
        "rel": (DREL, E),
        "kw": (W, C_) if full else (C_, W),
        "vw": (W, C_) if full else (C_, W),
    }
    for n, shape in wts.items():
        g.weight(WeightDecl(n, shape, "dense", n))
    g.state(StateDecl("kv", "kv", (2, R, Hkv, D)))
    for n in ("kc", "vc"):
        g.state(StateDecl(n, "conv", (C_, W - 1)))

    def clipped(x, heads, dim, scale, thr):
        x = g.op("l2_clip", [g.op("reshape", [x], shape=["T", heads, dim])], scale=scale, threshold=thr)
        return g.op("log_scale", [x, pos], alpha=0.1, floor=8192) if full else x

    q = clipped(ins["q"], Hq, D, "qs", 22.627416997969522)
    r = clipped(ins["r"], Hq, DREL, "rs", 8.0)

    def conv(x, w, state, which):
        if full:
            return g.op("dyn_conv1d", [x, ins[which + "tap"]], weight=w, state=state, heads=Hkv, divisor=3.0)
        return g.op("causal_conv1d", [x], weight=w, state=state, residual=True, divisor=math.sqrt(5.0))

    k = g.op(
        "l2_clip",
        [g.op("reshape", [conv(ins["k"], "kw", "kc", "k")], shape=["T", Hkv, D])],
        threshold=22.627416997969522,
    )
    vv = g.op("rmsnorm", [g.op("reshape", [conv(ins["v"], "vw", "vc", "v")], shape=["T", Hkv, D])], eps=1e-6)
    att = g.op(
        "attention",
        [q, k, vv, pos, r, ins["sink"]],
        state="kv",
        scale=D**-0.5,
        window=window or None,
        rel_proj="rel",
        sinks=True,
    )
    gate = g.op("cast", [ins["gate"]], dtype="f32")
    y = g.op("cast", [g.op("sigmoid_mul", [g.op("cast", [att], dtype="f32"), gate])], dtype="bf16")
    g.output(y=y)
    if expose:
        g.output(att=att, q=q, r=r)
    return g


class Rig:
    """One configuration: the kernel path (lowered plan, launcher, states) and the reference path on the same states."""

    def __init__(self, kind, *, Hq=24, Hkv=8, window=0, E=1024, R, seed=0):
        self.kind, self.Hq, self.Hkv, self.window, self.E, self.R = kind, Hq, Hkv, window, E, R
        self.full = kind == "full"
        self.C = Hkv * D
        gen = torch.Generator(device="cuda").manual_seed(seed)
        self.gen = gen

        def rnd(*shape, std=1.0):
            return (torch.randn(*shape, generator=gen, device="cuda") * std).to(BF)

        self.rnd = rnd
        wt = {
            "qs": (1 + 0.3 * rnd(Hq, D)).to(BF),
            "rs": (1 + 0.3 * rnd(Hq, DREL)).to(BF),
            "rel": rnd(DREL, E, std=0.3),
            "kw": rnd(W, self.C, std=0.5) if self.full else rnd(self.C, W, std=0.5),
            "vw": rnd(W, self.C, std=0.5) if self.full else rnd(self.C, W, std=0.5),
        }
        self.source = DictSource(wt)
        self.gk = build_graph(kind, Hq, Hkv, window, E, R)
        gr = build_graph(kind, Hq, Hkv, window, E, R, expose=True)
        self.ref = Interpreter(gr, self.source, device="cuda", dtype=BF)
        backend = CudaBackend()
        res = fuse(self.gk, TABLE, ROWS)
        assert [r.kernel for r in res.regions] == ["g_attn_rel"] and not res.unclaimed, res.backlog(self.gk)
        self.plan = lower(self.gk, backend, rows_max=ROWS)
        self.task = self.plan.tasks[0]
        self.ex = backend.executor(self.gk, self.plan, self.source)
        self.packs = {p.name: self.ex.pack(p) for p in self.plan.packs.values()}
        self.states = {
            s.name: torch.zeros(self.ex.state_nbytes(s, ROWS), dtype=torch.uint8, device="cuda")
            for s in self.gk.states.values()
        }
        self.inputs = {
            f"@{n}": torch.zeros(ROWS, dim, dtype=BF, device="cuda")
            for n, dim in (
                ("q", Hq * D),
                ("r", Hq * DREL),
                ("k", self.C),
                ("v", self.C),
                ("sink", Hq),
                ("gate", Hq * D),
            )
        }
        if self.full:
            self.inputs |= {f"@{n}": torch.zeros(ROWS, Hkv * W, dtype=BF, device="cuda") for n in ("ktap", "vtap")}
        self.buffers = {b.name: torch.zeros(ROWS * b.dim, dtype=BF, device="cuda") for b in self.plan.buffers.values()}
        self.out = self.buffers[self.plan.buffer_of[self.gk.outputs["y"]]]
        self.pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.scratch = _Scratch()
        self.mem = Memory(
            rows_max=ROWS,
            pos=self.pos.data_ptr(),
            tokens=0,
            ids=0,
            buffers={
                **{k: v.data_ptr() for k, v in self.buffers.items()},
                **{k: v.data_ptr() for k, v in self.inputs.items()},
            },
            packs={k: v.data_ptr() for k, v in self.packs.items()},
            states={k: v.data_ptr() for k, v in self.states.items()},
            constants={},
            scratch=self.scratch,
        )
        # the executor rounds a bf16 state up to whole attention chunks (schedule.kv_rows); the kernel reads R rows
        self.kv = self.states["kv"].view(BF)[: 2 * Hkv * R * D].view(2, Hkv, R, D)
        self.conv = {n: self.states[n].view(BF).view(self.C, W - 1) for n in ("kc", "vc")}

    def launch_step(self, T, stream=None):
        sched = self.ex.schedule(self.mem, self.plan.tasks, T)
        stream = stream or torch.cuda.current_stream().cuda_stream
        for launch in sched:
            launch.main(stream)

    def randomize_states(self):
        """The same random history in both paths: every KV slot (positions above a row's own must stay invisible) and
        the conv states."""
        rows = (self.rnd(2, self.R, self.Hkv, D, std=2.0)).clone()
        rows[1] *= 0.5
        self.ref.states["kv"].rows.copy_(rows)
        self.kv.copy_(rows.permute(0, 2, 1, 3))
        for n, rn in (("kc", "kc"), ("vc", "vc")):
            s = self.rnd(self.C, W - 1, std=2.0)
            self.ref.states[rn].copy_(s)
            self.conv[n].copy_(s)

    def snapshot(self):
        return {k: v.clone() for k, v in self.states.items()}

    def restore(self, snap):
        for k, v in snap.items():
            self.states[k].copy_(v)

    def draw(self, T, scale=1.0):
        """Random projection outputs of T rows (bf16): in the kernel's input buffers, returned for the reference."""
        rnd = self.rnd
        vals = {
            "@q": rnd(T, self.Hq * D, std=1.2 * scale),
            "@r": rnd(T, self.Hq * DREL, std=2.0 * scale),
            "@k": rnd(T, self.C, std=2.0 * scale),
            "@v": rnd(T, self.C, std=2.0),
            "@sink": rnd(T, self.Hq, std=2.0),
            "@gate": rnd(T, self.Hq * D, std=2.0),
        }
        if self.full:
            vals |= {"@ktap": rnd(T, self.Hkv * W, std=0.3), "@vtap": rnd(T, self.Hkv * W, std=0.3)}
        for k, v in vals.items():
            self.inputs[k][:T].copy_(v)
        return vals

    def reference(self, vals, pos0, T):
        pos = torch.arange(pos0, pos0 + T, dtype=torch.int32, device="cuda")
        return self.ref.run({"pos": pos, **vals})

    def run_kernel(self, pos0, T, poison=False):
        self.pos.fill_(pos0)
        if poison:
            self.poison()
        self.out.fill_(float("nan"))
        self.launch_step(T)
        torch.cuda.synchronize()
        return self.out.view(ROWS, -1)[:T].clone()

    def poison(self):
        """NaN into the split / merge workspace: a partial read before it is written shows."""
        self.scratch._tensors["ink_attn_ws"].view(torch.float32).fill_(float("nan"))

    def scratch_view(self, name, dtype=BF):
        return self.scratch._tensors[name].view(dtype)


# --- comparisons ---
def ulp_bf16(x):
    e = torch.floor(torch.log2(x.abs().clamp_min(2.0**-100)))
    return torch.exp2(e - 7)


def row_max(a, T, Hq):
    return a.float().abs().reshape(T, Hq, D).amax(-1, keepdim=True).expand(-1, -1, D).reshape(T, Hq * D)


def exact_attention(rig, T, pos0, vals):
    """Attention of the kernel's own prep outputs in float64 without the bf16 rounding of P: what the split / merge
    compute up to summation order. Returns the gated output and the attention, bf16 [T, Hq*128]."""
    Hq, Hkv, R = rig.Hq, rig.Hkv, rig.R
    G = Hq // Hkv
    qo = rig.scratch_view("ink_attn_q")[: T * Hq * D].view(T, Hq, D)
    ro = rig.scratch_view("ink_attn_r")[: T * Hq * DREL].view(T, Hq, DREL)
    proj = rig.source.tensors["rel"].to("cuda")
    att = []
    for t in range(T):
        p = pos0 + t
        lo = max(0, p - rig.window + 1) if rig.window else 0
        pos = torch.arange(lo, p + 1, device="cuda")
        slots = pos % R
        K = rig.kv[0][:, slots].repeat_interleave(G, 0).double()
        V = rig.kv[1][:, slots].repeat_interleave(G, 0).double()
        sc = torch.einsum("hd,hld->hl", qo[t].double(), K) * D**-0.5
        bias = (ro[t] @ proj).float().double()  # [Hq, E], rounded to bf16 once
        dist = p - pos
        sc = sc + bias[:, dist.clamp(max=rig.E - 1)].masked_fill((dist >= rig.E)[None], 0.0)
        pr = torch.softmax(torch.cat([sc, vals["@sink"][t].double()[:, None]], -1), -1)[:, :-1]
        att.append(torch.einsum("hl,hld->hd", pr, V).to(BF).reshape(-1))
    att = torch.stack(att)
    gate = vals["@gate"].float()
    return (att.float() * torch.sigmoid(gate)).to(BF), att


def compare_attention(rig, T, got, want, att_ref, exact_out, exact_att):
    """-> (ok, stats). Two bounds: against the reference (bf16-rounded P: 2 ulp + 2^-7 row max, rel L2) and against the
    float64 math without that rounding (3 ulp + 2^-12 row max)."""
    assert torch.isfinite(got.float()).all(), "non-finite output"
    g, w, x = got.float(), want.float(), exact_out.float()
    err, errx = (g - w).abs(), (g - x).abs()
    bound = 2 * ulp_bf16(w) + FLOOR_REL * row_max(att_ref, T, rig.Hq)
    boundx = 3 * ulp_bf16(x) + 2.0**-12 * row_max(exact_att, T, rig.Hq)
    l2 = (err.norm() / w.norm().clamp_min(1e-30)).item()
    ok = bool((err <= bound).all()) and bool((errx <= boundx).all()) and l2 <= L2_CASE
    return ok, {
        "l2": l2,
        "e2": (err.norm() ** 2).item(),
        "w2": (w.norm() ** 2).item(),
        "l2x": (errx.norm() / x.norm().clamp_min(1e-30)).item(),
        "ulps": (err / ulp_bf16(w)).max().item(),
        "ratio": (err / bound).max().item(),
        "ratiox": (errx / boundx).max().item(),
    }


def compare_prep(rig, T, pos0, ref_out):
    """Prep outputs against the reference: conv states bit-equal, clipped q / r and the K / V rows written to the state
    at most 1 bf16 ulp apart (rel L2 <= 2e-3). Returns (mismatching elements, elements)."""
    for n in ("kc", "vc"):
        assert torch.equal(rig.conv[n], rig.ref.states[n]), f"conv state {n} differs"
    bad = total = 0
    slots = torch.arange(pos0, pos0 + T, device="cuda") % rig.R
    refrows = rig.ref.states["kv"].rows
    pairs = {
        "q": (rig.scratch_view("ink_attn_q")[: T * rig.Hq * D].view(T, rig.Hq, D), ref_out["q"]),
        "r": (rig.scratch_view("ink_attn_r")[: T * rig.Hq * DREL].view(T, rig.Hq, DREL), ref_out["r"]),
        "K": (rig.kv[0][:, slots].transpose(0, 1), refrows[0][slots]),
        "V": (rig.kv[1][:, slots].transpose(0, 1), refrows[1][slots]),
    }
    for name, (got, want) in pairs.items():
        g, w = got.float(), want.float()
        err = (g - w).abs()
        ulps = err / ulp_bf16(torch.maximum(g.abs(), w.abs()))
        assert ulps.max().item() <= 1.0001, f"{name}: {ulps.max().item():.2f} ulp apart"
        l2 = (err.norm() / w.norm().clamp_min(1e-30)).item()
        assert l2 <= 2e-3, f"{name}: rel L2 {l2:.2e}"
        bad += int((err > 0).sum())
        total += err.numel()
    return bad, total


# --- GPU tests ---
def need_gpu():
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


STEPS = (1, 3, 8, 2, 1)


def run_case(rig, T, pos0, stats, label, scale=1.0, prep_stats=None):
    vals = rig.draw(T, scale)
    ref_out = rig.reference(vals, pos0, T)
    got = rig.run_kernel(pos0, T)
    ex_out, ex_att = exact_attention(rig, T, pos0, vals)
    ok, s = compare_attention(rig, T, got, ref_out["y"], ref_out["att"], ex_out, ex_att)
    bad, total = compare_prep(rig, T, pos0, ref_out)
    s["prep_bad"], s["prep_n"] = bad, total
    stats.append((label, s))
    if not ok:
        FAILS.append(f"{label}: {s}")


def sweep(rig, starts, label, stats, cycles=2, scales=(1.0, 0.4)):
    n = 0
    for start in starts:
        rig.randomize_states()
        pos0 = start
        for _ in range(cycles):
            for T in STEPS:
                run_case(rig, T, pos0, stats, f"{label} pos0={pos0} T={T}", scale=scales[(n // 5) % len(scales)])
                pos0 += T
                n += 1


def test_attention_block_against_the_reference():
    need_gpu()
    stats: list = []
    configs = [  # (label, rig kwargs, start positions)
        ("sliding w1024 R1031", {"kind": "sliding", "window": 1024, "R": 1031}, (0, 1019, 5000, 100000)),
        ("sliding w40 R47 E100", {"kind": "sliding", "window": 40, "E": 100, "R": 47}, (0, 30, 46, 1000)),
        ("sliding w64 G2", {"kind": "sliding", "Hq": 16, "window": 64, "R": 71}, (10, 500)),
        ("sliding w40 G1", {"kind": "sliding", "Hq": 8, "window": 40, "E": 100, "R": 47}, (0, 60)),
        ("sliding w100 G5", {"kind": "sliding", "Hq": 20, "Hkv": 4, "window": 100, "R": 107}, (0, 90, 4000)),
        ("full R4096", {"kind": "full", "R": 4096}, (0, 4050)),
        ("full R9600 chunked", {"kind": "full", "R": 9600}, (0, 8180, 8990, 9500)),
        ("full R2048 E100", {"kind": "full", "E": 100, "R": 2048}, (0, 1000)),
        ("full R20000 chunked", {"kind": "full", "R": 20000}, (8190, 19900)),
        ("full G3 R300", {"kind": "full", "Hq": 12, "Hkv": 4, "R": 300}, (0, 270)),
    ]
    for label, kw, starts in configs:
        rig = Rig(seed=len(stats), **kw)
        sweep(rig, starts, label, stats)
        del rig
        torch.cuda.empty_cache()
    rig = Rig("full", R=33000, seed=99)  # nc = 8
    sweep(rig, (32940,), "full R33000 chunked", stats, cycles=1)
    assert not FAILS, "\n".join(FAILS[:10])
    mx = lambda k: max(s[k] for _, s in stats)  # noqa: E731
    bad = sum(s["prep_bad"] for _, s in stats)
    total = sum(s["prep_n"] for _, s in stats)
    worst = max(stats, key=lambda s: s[1]["ratio"])
    l2_all = (sum(s["e2"] for _, s in stats) / sum(s["w2"] for _, s in stats)) ** 0.5
    assert l2_all <= L2_ALL, f"rel L2 over all cases {l2_all:.2e}"
    print(
        f"{len(stats)} cases within bound: rel L2 over all {l2_all:.2e}, max per case {mx('l2'):.2e}"
        f" (vs fp64 without P rounding {mx('l2x'):.2e}), max {mx('ulps'):.2f} ulp,"
    )
    print(
        f"  worst err/bound {worst[1]['ratio']:.2f} ({worst[0]}), vs fp64 {mx('ratiox'):.2f};"
        f" prep elements differing from the reference: {bad} of {total}"
    )
    assert len(stats) >= 160


def test_rows_per_step_do_not_change_a_bit():
    need_gpu()
    for kind, kw in (("sliding", {"window": 40, "E": 100, "R": 47}), ("full", {"R": 9600})):
        rig = Rig(kind, seed=5, **kw)
        pos0 = 8990 if kind == "full" else 1000
        rig.randomize_states()
        snap = rig.snapshot()
        vals = rig.draw(ROWS)
        results = {}
        for split in ((8,), (4, 4), (1,) * 8, (3, 5)):
            rig.restore(snap)
            outs, off = [], 0
            for T in split:
                for k, v in vals.items():
                    rig.inputs[k][:T].copy_(v[off : off + T])
                outs.append(rig.run_kernel(pos0 + off, T))
                off += T
            results[split] = (torch.cat(outs), rig.snapshot())
        base_out, base_states = results[(8,)]
        for split, (out, states) in results.items():
            assert torch.equal(out, base_out), f"{kind}: output of steps {split} differs from one 8-row step"
            for k in base_states:
                assert torch.equal(states[k], base_states[k]), f"{kind}: state {k} after steps {split} differs"
    print("steps of 8 = 4 + 4 = 3 + 5 = 8 x 1 rows: outputs, KV rows and conv states bit-identical (ring, chunked)")


def test_deterministic_and_graph_replay_equals_eager():
    need_gpu()
    for kind, kw, p_cap, p_replay in (
        ("sliding", {"window": 1024, "R": 1031}, 130, (1500, 5, 1030, 100000)),
        ("full", {"R": 9600}, 130, (8990, 9500, 5)),
    ):
        rig = Rig(kind, seed=11, **kw)
        rig.randomize_states()
        T = 3
        rig.draw(T)
        snap = rig.snapshot()
        first = rig.run_kernel(p_cap, T)  # also sizes the scratch before the capture
        for _ in range(3):
            rig.restore(snap)
            again = rig.run_kernel(p_cap, T, poison=True)
            assert torch.equal(first, again), f"{kind}: not deterministic"
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rig.launch_step(T)
        for pos0 in p_replay:
            rig.draw(T)
            rig.restore(snap)
            rig.pos.fill_(pos0)
            eager = rig.run_kernel(pos0, T)
            eager_states = rig.snapshot()
            rig.restore(snap)
            rig.out.fill_(float("nan"))
            rig.poison()
            graph.replay()
            torch.cuda.synchronize()
            replay = rig.out.view(ROWS, -1)[:T].clone()
            assert torch.equal(eager, replay), f"{kind}: replay at pos0={pos0} differs from the eager launches"
            for k, v in eager_states.items():
                assert torch.equal(v, rig.states[k]), f"{kind}: state {k} after the replay at pos0={pos0} differs"
    print(
        "reruns with a poisoned workspace and graph replays at other positions are bit-identical to the eager launches"
    )


if __name__ == "__main__":
    run(globals())
