"""The row / GEMV kernels of the Gemma 4 E2B plan, one by one, on the GPU (kernels/gemma_rows.cu, dense_norm.cu,
dense_gemv*.cuh): the scaled embedding, the PLE prelude, the post-norm residual add, the GELU pair and row epilogues,
the softcapped head, and the plain GEMV shapes of the model; and, on the CPU, the layout the fusion pass gives the E2B
graph and the two generic extensions of the pass (a capability's operand cap, in-place updates after earlier readers).

  uv run --package warpc python warpc/tests/test_gemma4_rows.py

The oracles. The interpreter's functions (warpc.reference._IMPL, torch on the GPU) define the semantics. The kernels
differ from them only where a reduction order differs (torch's mean / matmul against the kernels' fixed orders) and,
for the GEMV-class kernels, in nothing else, so every case is checked three ways:
  - bitwise against a torch emulation of the kernel's own arithmetic (reduction order and fma contraction included; the
    fma is done in float64, which differs from a fp32 fma only on a double rounding that did not occur here), or, for
    epilogues, bitwise against the interpreter's function applied to the kernel's own linear outputs;
  - against the interpreter on the same inputs: the number of differing elements and the relative L2 error;
  - against float64 accumulation (GEMV shapes): the relative L2 error.
Exact errors are printed. Needs a CUDA device and nvcc.
"""

import ctypes as C
from dataclasses import replace
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from harness import Skip, run
from test_compiler import covers_only
from test_gemma4 import ShapeSource, e2b_text_config
from warpc import frontend
from warpc.compiler import Cap, CapsTable, Region, lower
from warpc.compiler.fuse import fuse
from warpc.compiler.lower import _toposort
from warpc.cuda import CudaBackend, glue, schedule
from warpc.cuda.table import RTX3090, TABLE
from warpc.ir import Graph, TType, WeightDecl
from warpc.reference import _IMPL
from warpc.sources import ModelConfig

DEV = "cuda"
BF = torch.bfloat16
EPS = 1e-6
GEN = torch.Generator().manual_seed(7)


def lib() -> C.CDLL:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    major, minor = torch.cuda.get_device_capability()
    return glue.load(10 * major + minor)


def rand(*shape, std: float = 1.0, dtype=BF) -> torch.Tensor:
    return (torch.randn(*shape, generator=GEN) * std).to(dtype).to(DEV)


def zeros(*shape, dtype=BF) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device=DEV)


def call(L: C.CDLL, name: str, *args) -> None:
    rc = getattr(L, name)(*args, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert rc == 0, f"{name}: {L.warpc_error_string(rc).decode()} ({rc})"


def rel(a, b) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def differing(a, b) -> int:
    return int((a != b).sum())


def rb(x: torch.Tensor) -> torch.Tensor:
    return x.to(BF).float()


def fma32(a, b, c):
    """fp32 fma by float64: the product is exact in float64, the sum rounds once to float64 and once to float32."""
    return (a.double() * b.double() + c.double()).float()


def butterfly(v: torch.Tensor) -> torch.Tensor:
    """warp_sum: xor 16, 8, 4, 2, 1 over the last dim of 32 lanes (every lane ends equal); returns lane 0."""
    lane = torch.arange(32, device=v.device)
    for o in (16, 8, 4, 2, 1):
        v = v + v[..., lane ^ o]
    return v[..., 0]


# ------------------------------------------------------------------------------- the pass's generic extensions (CPU)
def test_operand_cap_of_a_capability():
    """Cap.operand_cap replaces the pass's 16-operand default for a kernel that binds more (the PLE kernel's 35)."""
    g = Graph("wide")
    g.input("h", TType(("T", 256), "bf16"))
    outs = {}
    for i in range(18):
        g.weight(WeightDecl(f"w{i}", (256, 256), "dense", f"w{i}"))
        outs[f"o{i}"] = g.op("linear", ["h"], weight=f"w{i}")
    g.output(**outs)
    wide = Cap("k_wide", "linear", covers_only, segments=18, rounds={"linear": ("out",)})
    refused = fuse(g, CapsTable([wide], RTX3090, TABLE.unfused_ctas), 8)
    assert len(refused.unclaimed) == 18 and "more than 16 operands" in refused.unclaimed[0]
    claimed = fuse(g, CapsTable([replace(wide, operand_cap=40)], RTX3090, TABLE.unfused_ctas), 8)  # 1 + 18 + 18
    assert not claimed.unclaimed and len(claimed.regions) == 1
    refused = fuse(g, CapsTable([replace(wide, operand_cap=36)], RTX3090, TABLE.unfused_ctas), 8)
    assert "more than 36 operands" in refused.unclaimed[0]


def residual_graph(reader_first: bool) -> Graph:
    """embed -> x; y = linear(norm(x)); add(x, y), and z = linear(x) a graph output either before or after the add."""
    g = Graph("residual")
    tokens = g.input("tokens", TType(("T",), "i32"))
    for n, s in (("embed", (64, 512)), ("nw", (512,)), ("w", (512, 512)), ("w2", (512, 512))):
        g.weight(WeightDecl(n, s, "dense", n))
    x = g.op("embed", [tokens], weight="embed", name="x")
    outs = {}
    if reader_first:
        outs["z"] = g.op("linear", [x], weight="w2", name="z")
    y = g.op("linear", [g.op("rmsnorm", [x], weight="nw", eps=1e-6)], weight="w")
    outs["x2"] = g.op("add", [x, y], name="x2")
    if not reader_first:
        outs["z"] = g.op("linear", [x], weight="w2", name="z")
    g.output(**outs)
    g.verify()
    return g


def test_in_place_update_is_ordered_after_an_earlier_reader():
    """A reader of the stream that comes before the residual add in the graph but is no ancestor of it (Gemma's PLE
    projection of the embedding, read before layer 0 updates it) no longer blocks the in-place epilogue: lowering runs
    it first. A reader after the add still races."""
    late = fuse(residual_graph(reader_first=False), TABLE, 8)
    assert "g_gemv_resadd" not in {r.kernel for r in late.regions}
    g = residual_graph(reader_first=True)
    assert "g_gemv_resadd" in {r.kernel for r in fuse(g, TABLE, 8).regions}
    tasks = lower(g, CudaBackend(), rows_max=8).tasks
    at = {t.weights["w"]: i for i, t in enumerate(tasks) if "w" in t.weights}
    assert at["w2"] < at["w"], [t.kernel for t in tasks]
    # a reader that is ready last still runs before the update: the order is the toposort's, not the node order's
    p = Region("p", {0}, {}, {"v": "v"})
    w = Region("w", {1}, {"h": "v"}, {"h_out": "v2"}, inplace=("h_out", "h"))
    r = Region("r", {2}, {"x": "v"}, {"y": "y"})
    assert _toposort([p, w, r]) == [0, 2, 1]


def test_e2b_plan_layout():
    """The pass's choice on the real E2B graph is the design's: the prelude (embed, PLE projection, PLE kernel), then
    ten tasks per layer (attention counts three launches: 12), the final norm and the softcapped head."""
    config = ModelConfig.from_hf(e2b_text_config())
    graph = frontend.build(ShapeSource(config), ctx_max=4096, rows_max=8)
    plan = lower(graph, CudaBackend(), rows_max=8)
    kernels = [t.kernel.replace("g_attn_read", "g_attn") for t in plan.tasks]
    layer = ["g_gemv", "g_attn", "g_gemv", "g_rmsnorm_post", "g_gemv_dual", "g_gemv"]
    layer += ["g_rmsnorm_post", "g_gemv_row", "g_gemv", "g_rmsnorm_post"]
    assert kernels == ["g_embed", "g_gemv", "g_ple"] + layer * 35 + ["g_rmsnorm", "g_head"], plan.summary()
    head = plan.tasks[-1]
    assert head.attrs["cap"] == 30.0 and set(head.outs) == {"logits", "ids"}
    ple = plan.tasks[2]
    assert (ple.attrs["L"], ple.attrs["D"]) == (35, 256) and len(ple.outs) == 35
    assert sum(t.kernel == "g_rmsnorm_post" and "s" in t.weights for t in plan.tasks) == 35


def emu_block_sumsq(x: torch.Tensor, threads: int) -> torch.Tensor:
    """x [R, n] -> [R], k_rmsnorm / k_rmsnorm_post / k_ple statistics: thread t sums the squares of its 8-element groups
    t, t + threads, ... in order (fma), a butterfly per warp, then the warp sums added in order from 0."""
    R, n = x.shape
    iters = -(-(n // 8) // threads)
    xp = torch.zeros(R, iters * threads * 8, device=x.device)
    xp[:, :n] = x.float()
    xv = xp.view(R, iters, threads, 8)
    acc = torch.zeros(R, threads, device=x.device)
    for i in range(iters):
        for e in range(8):
            acc = fma32(xv[:, i, :, e], xv[:, i, :, e], acc)
    warps = butterfly(acc.view(R, threads // 32, 32))
    r = torch.zeros(R, device=x.device)
    for k in range(warps.shape[1]):
        r = r + warps[:, k]
    return r


def emu_gemv(
    w: torch.Tensor, x: torch.Tensor, nw: torch.Tensor | None = None, eps: float = 0.0, ks: int = 1
) -> torch.Tensor:
    """k_gemv / k_dg / k_ks: [N, T] fp32 accumulators. Lane l of slice s (K is cut into ks slices, one warp each) sums
    the 8-element groups s * K / (8 ks) + l + 32 j in order (fma, element order), the 32 lanes are added with the
    butterfly, the ks slice sums in slice order from 0; with `nw` the rows of x are normalised first exactly as the
    prologue does (statistics over the whole row, one butterfly)."""
    N, K = w.shape
    T, J = x.shape[0], K // 256
    xf = x.float()
    if nw is not None:
        xv = xf.view(T, J, 32, 8)
        acc = torch.zeros(T, 32, device=x.device)
        for j in range(J):
            for e in range(8):
                acc = fma32(xv[:, j, :, e], xv[:, j, :, e], acc)
        rs = torch.rsqrt(butterfly(acc) / K + eps)
        xf = rb(xf * rs[:, None] * nw.float())
    nit = J // ks
    wv, xv = w.float().view(N, ks, nit, 32, 8), xf.view(T, ks, nit, 32, 8)
    acc = torch.zeros(N, T, ks, 32, device=x.device)
    for j in range(nit):
        for e in range(8):
            acc = fma32(wv[:, None, :, j, :, e], xv[None, :, :, j, :, e], acc)
    parts = butterfly(acc)
    out = torch.zeros(N, T, device=x.device)
    for s in range(ks):
        out = out + parts[:, :, s]
    return out


def gemv(L, w, x, N, K, T, mode, outs, segs=(), w2=None, nw=None, eps=0.0, row=None, act=0, cap=0.0, rnd=0):
    o = (C.c_void_p * 3)(*[t.data_ptr() for t in outs])
    ends = (C.c_int * 3)(*[sum(segs[: j + 1]) for j in range(3)]) if segs else None
    call(
        L,
        "warpc_gemv",
        w.data_ptr(),
        w2.data_ptr() if w2 is not None else None,
        x.data_ptr(),
        N,
        K,
        T,
        mode,
        ends,
        o,
        nw.data_ptr() if nw is not None else None,
        C.c_float(eps),
        schedule.gemv_rows_per_warp(N, K, mode),
        row.data_ptr() if row is not None else None,
        act,
        C.c_float(cap),
        rnd,
        schedule.gemv_k_split(K, mode, nw is not None),
    )


def ref(op: str, attrs: dict, *args, weights: dict | None = None):
    return _IMPL[op](SimpleNamespace(dtype=BF, weights=weights or {}), attrs, *args)


# ------------------------------------------------------------------------------------------------------ embedding
def test_embed_scale():
    """bf16(row * bf16(scale)) bitwise against the interpreter: sqrt(1536) (rounded to 39.25 first), the PLE table's 16,
    and scale 1 (a plain gather)."""
    L = lib()
    for H, scale, V in ((1536, 1536**0.5, 4096), (8960, 16.0, 700), (1536, 1.0, 64)):
        table = rand(V, H, std=0.5)
        for T in (1, 3, 8):
            tok = torch.randint(0, V, (T,), generator=GEN).int().to(DEV)
            out = zeros(8, H)
            call(L, "warpc_embed", table.data_ptr(), tok.data_ptr(), out.data_ptr(), T, H, C.c_float(scale))
            want = ref("embed", {"weight": "t", "scale": scale}, tok, weights={"t": table})
            assert torch.equal(out[:T], want), (H, scale, T, differing(out[:T], want))
        if scale == 1536**0.5:
            unrounded = (table[tok.long()].float() * scale).to(BF)
            assert not torch.equal(want, unrounded), "the scale is rounded to bf16 first"
        print(f"  embed H={H} scale={scale:.6g}: bitwise equal to the interpreter (T = 1, 3, 8)")


# ------------------------------------------------------------------------------------------------------ post norm
def test_rmsnorm_post():
    """h = bf16(h + bf16(rmsnorm(x) * w)) [then bf16(h * s)] in place. Bitwise against the emulation of the block
    reduction (256 / 512 threads, ragged H included); against the interpreter: differing elements and relative L2."""
    L = lib()
    for H in (1536, 1544, 2048, 4096, 6144):
        threads = 512 if H >= 4096 else 256
        w, s = rand(H, std=0.3) + 1.0, rand(1, std=0.2) + 0.9
        for T in (1, 3, 8):
            for with_scale in (False, True):
                x, h0 = rand(T, H, std=3.0), rand(T, H, std=5.0)
                h = zeros(8, H)
                h[:T] = h0
                call(
                    L,
                    "warpc_rmsnorm_post",
                    h.data_ptr(),
                    x.data_ptr(),
                    w.data_ptr(),
                    s.data_ptr() if with_scale else None,
                    T,
                    H,
                    C.c_float(EPS),
                )
                r = torch.rsqrt(emu_block_sumsq(x.float(), threads) / H + EPS)
                want = rb(h0.float() + rb(x.float() * r[:, None] * w.float()))
                if with_scale:
                    want = rb(want * s.float())
                assert torch.equal(h[:T], want.to(BF)), (H, T, with_scale, differing(h[:T], want.to(BF)))
                ws = {"w": w, "s": s}
                y = ref("rmsnorm", {"weight": "w", "eps": EPS}, x, weights=ws)
                got = ref("add", {}, h0, y)
                if with_scale:
                    got = ref("scale", {"weight": "s"}, got, weights=ws)
                bad, e = differing(h[:T], got), rel(h[:T], got)
                assert e < 5e-3, (H, T, e)
                if T == 3:
                    print(
                        f"  rmsnorm_post H={H} scale={with_scale}: emulation bitwise;"
                        f" interpreter {bad}/{got.numel()} differ, rel L2 {e:.2e}"
                    )


# ------------------------------------------------------------------------------------------------------ PLE prelude
def ple_reference(ctx, tok, w, table, L_, D, f1, f2, escale):
    ws = {"w": w, "t": table}
    x = ref("scale", {"factor": f1}, ctx)
    x = ref("reshape", {"shape": ["T", L_, D]}, x)
    x = ref("rmsnorm", {"weight": "w", "eps": EPS}, x, weights=ws)
    x = ref("reshape", {"shape": ["T", L_ * D]}, x)
    e = ref("embed", {"weight": "t", "scale": escale}, tok, weights=ws)
    return ref("split", {"sizes": [D] * L_}, ref("scale", {"factor": f2}, ref("add", {}, x, e)))


def test_ple_prelude():
    """The PLE prelude kernel against the interpreter's chain (scale, reshape, rmsnorm rank 3, reshape, embed, add,
    scale, split) at the E2B geometry (35 slices of 256) and at D = 512, 1024: bitwise against the emulation of the
    256-element norm reduction (D / 8 thread partials, one butterfly per warp, warp sums in order)."""
    L = lib()
    for L_, D, V in ((35, 256, 600), (3, 512, 64), (2, 1024, 64)):
        w = rand(D, std=0.2) + 1.0
        table = rand(V, L_ * D, std=0.5)
        f1, f2, escale = 1536**-0.5, 2.0**-0.5, 16.0
        for T in (1, 5, 8):
            ctx = rand(T, L_ * D, std=20.0)
            tok = torch.randint(0, V, (T,), generator=GEN).int().to(DEV)
            outs = [zeros(8, D) for _ in range(L_)]
            ptrs = (C.c_void_p * L_)(*[o.data_ptr() for o in outs])
            call(
                L,
                "warpc_ple",
                ctx.data_ptr(),
                tok.data_ptr(),
                table.data_ptr(),
                w.data_ptr(),
                ptrs,
                T,
                L_,
                D,
                C.c_float(f1),
                C.c_float(f2),
                C.c_float(EPS),
                C.c_float(escale),
            )
            c = rb(ctx.float() * f1).view(T * L_, D)
            r = torch.rsqrt(emu_block_sumsq(c, D // 8) / D + EPS)
            n = rb(c * r[:, None] * w.float()).view(T, L_ * D)
            e = rb(table[tok.long()].float() * rb(torch.tensor(escale)).item())
            want = (rb(n + e) * f2).to(BF)
            got = torch.cat([o[:T] for o in outs], 1)
            assert torch.equal(got, want), (L_, D, T, differing(got, want))
            if T == 5:
                want_ref = torch.cat(ple_reference(ctx, tok, w, table, L_, D, f1, f2, escale), 1)
                bad, err = differing(got, want_ref), rel(got, want_ref)
                assert err < 5e-3, (L_, D, err)
                print(
                    f"  ple L={L_} D={D}: emulation bitwise; interpreter {bad}/{got.numel()} differ, rel L2 {err:.2e}"
                )


# ------------------------------------------------------------------------------------------------------ GEMVs
def test_gemv_shapes_of_the_model():
    """The plain GEMVs of Gemma 4 E2B (q|k|v with the ln1 prologue for both head dims and q alone for the sharers, o,
    down at both FFN widths, pproj at K = 256, the PLE projection) at T = 1 (k_dg where it exists), 2 and 8: bitwise
    against the emulation of the kernel's order (k_gemv / k_dg: one lane order over K; the K-split kernel of the
    down projections, K = 6144 / 12288: 3 / 6 slices, one warp each); relative L2 against float64 accumulation."""
    L = lib()
    cases = [
        ("q|k|v ln1 hd256", 1536, [2048, 256, 256], True),
        ("q|k|v ln1 hd512", 1536, [4096, 512, 512], True),
        ("q ln1 (sharer)", 1536, [2048], True),
        ("o hd256", 2048, [1536], False),
        ("o hd512", 4096, [1536], False),
        ("down 6144", 6144, [1536], False),
        ("down 12288", 12288, [1536], False),
        ("pproj K=256", 256, [1536], False),
        ("ple_proj", 1536, [8960], False),
    ]
    for name, K, segs, norm in cases:
        N = sum(segs)
        w = rand(N, K, std=K**-0.5)
        nw = rand(K, std=0.3) + 1.0 if norm else None
        for T in (1, 2, 8):
            x = rand(T, K, std=2.0)
            outs = [zeros(8, s) for s in segs]
            gemv(L, w, x, N, K, T, 0, outs, segs=segs, nw=nw, eps=EPS)
            got = torch.cat([o[:T] for o in outs], 1)
            ks = schedule.gemv_k_split(K, 0, norm)
            assert ks == {6144: 3, 12288: 6}.get(K, 1), (name, ks)
            acc = emu_gemv(w, x, nw, EPS, ks)
            assert torch.equal(got, acc.T.to(BF)), (name, T, ks, differing(got, acc.T.to(BF)))
            if T == 2:
                xin = (
                    rb(x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS) * nw.float())
                    if norm
                    else x.float()
                )
                exact = xin.double() @ w.double().T
                print(
                    f"  {name:<18} N={N:<5} K={K:<5} emulation bitwise (T=1,2,8);"
                    f" vs float64 rel L2 {rel(got, exact):.2e}"
                )


def test_gemv_gelu_pair():
    """gate|up GEMV with the ln3 prologue and the GELU epilogue, K = 1536, N = 6144 and 12288, at T = 1 (k_dg), 3 and 8:
    bitwise equal to the interpreter's gelu_mul applied to the same kernel's gate and up outputs (mode 0), whose own
    accumulators are checked against the emulation. act = 0 (silu) is the existing epilogue."""
    L = lib()
    K = 1536
    for N in (6144, 12288):
        wg, wu = rand(N, K, std=K**-0.5), rand(N, K, std=K**-0.5)
        nw = rand(K, std=0.3) + 1.0
        both = torch.cat([wg, wu])
        for T in (1, 3, 8):
            x = rand(T, K, std=2.0)
            out = zeros(8, N)
            gemv(L, wg, x, N, K, T, 1, [out], w2=wu, nw=nw, eps=EPS, act=1)
            g, u = zeros(8, N), zeros(8, N)
            gemv(L, both, x, 2 * N, K, T, 0, [g, u], segs=[N, N], nw=nw, eps=EPS)
            want = ref("gelu_mul", {}, g[:T], u[:T])
            assert torch.equal(out[:T], want), (N, T, differing(out[:T], want))
            if T == 1:
                acc = emu_gemv(both[:512], x, nw, EPS).T
                assert torch.equal(g[:T, :512], acc.to(BF)), "gate accumulators"
                silu = zeros(8, N)
                gemv(L, wg, x, N, K, T, 1, [silu], w2=wu, nw=nw, eps=EPS, act=0)
                want_silu = ref("silu_mul", {}, g[:T], u[:T])
                near = (silu[:T].float() - want_silu.float()).abs().max().item()
                exact = F.gelu(g[:T].double(), approximate="tanh") * u[:T].double()
                print(
                    f"  gelu pair N={N}: bitwise vs gelu_mul(g, u); vs float64 gelu {rel(out[:T], exact):.2e};"
                    f" silu epilogue {differing(silu[:T], want_silu)} elements off torch's silu (max {near:.1e})"
                )


def test_gemv_row():
    """pgate: K = 1536, N = 256, out = bf16(bf16(gelu(g)) * row) with a [T, 256] row operand, at T = 1 (k_dg), 2, 8:
    bitwise equal to the interpreter's gelu_mul on the kernel's own linear output, which is checked against the
    emulation."""
    L = lib()
    K, N = 1536, 256
    w = rand(N, K, std=K**-0.5)
    for T in (1, 2, 8):
        x = rand(T, K, std=2.0)
        row = rand(T, N, std=1.0)
        out, g = zeros(8, N), zeros(8, N)
        gemv(L, w, x, N, K, T, 4, [out], row=row, act=1)
        gemv(L, w, x, N, K, T, 0, [g])
        assert torch.equal(g[:T], emu_gemv(w, x).T.to(BF))
        want = ref("gelu_mul", {}, g[:T], row)
        assert torch.equal(out[:T], want), (T, differing(out[:T], want))
        exact = F.gelu((x.double() @ w.double().T), approximate="tanh") * row.double()
        print(f"  gemv_row T={T}: bitwise vs gelu_mul(g, row); vs float64 {rel(out[:T], exact):.2e}")


def test_head_softcap():
    """The head GEMV: fp32 logits = the accumulator (cap 0, the dense head) and cap * tanhf(acc / cap) with a softcap,
    bitwise against the emulation (true division, CUDA's tanhf); with rnd the accumulator is rounded to bf16 first
    (the interpreter's linear output: Inkling's head), capped or not. The interpreter's softcap is counted against
    the unrounded one (torch on the GPU divides by a Python scalar as a multiply by the reciprocal; on the CPU tanh is
    the C library's). Argmax over the capped values, lowest index on a tie: two equal rows, and two rows whose
    accumulators differ but whose capped values are both exactly cap."""
    L = lib()
    K, N, cap = 1536, 262144, 30.0
    w = rand(N, K, std=K**-0.5) * 8
    x = rand(1, K, std=2.0)
    plain, capped = zeros(1, N, dtype=torch.float32), zeros(1, N, dtype=torch.float32)
    rounded_plain, rounded_capped = zeros(1, N, dtype=torch.float32), zeros(1, N, dtype=torch.float32)
    gemv(L, w, x, N, K, 1, 2, [plain])
    gemv(L, w, x, N, K, 1, 2, [capped], cap=cap)
    gemv(L, w, x, N, K, 1, 2, [rounded_plain], rnd=1)
    gemv(L, w, x, N, K, 1, 2, [rounded_capped], cap=cap, rnd=1)
    acc = emu_gemv(w, x).T
    assert torch.equal(plain, acc), differing(plain, acc)
    cap_t = torch.tensor(cap, device=DEV)  # a tensor divisor: a true division
    want = cap_t * torch.tanh(acc / cap_t)
    assert torch.equal(capped, want), differing(capped, want)
    rounded = acc.to(BF).float()
    assert torch.equal(rounded_plain, rounded), differing(rounded_plain, rounded)
    want_rounded = cap_t * torch.tanh(rounded / cap_t)
    assert torch.equal(rounded_capped, want_rounded), differing(rounded_capped, want_rounded)
    gpu, cpu = ref("softcap", {"cap": cap}, acc), ref("softcap", {"cap": cap}, acc.cpu()).to(DEV)
    print(
        f"  head N={N}: dense logits bitwise = accumulator; softcapped logits bitwise = cap * tanh(acc / cap);"
        f" with rnd bitwise = bf16(acc) and cap * tanh(bf16(acc) / cap);"
        f" interpreter softcap differs in {differing(capped, gpu)} (GPU) / {differing(capped, cpu)} (CPU) of {N},"
        f" rel L2 {rel(capped, cpu):.1e}, max |logit| {capped.abs().max():.3f}"
    )
    ws = zeros(int(L.warpc_argmax_ws_bytes()), dtype=torch.uint8)

    def argmax(logits, n):
        ids = zeros(1, dtype=torch.int32)
        call(L, "warpc_argmax", logits.data_ptr(), 1, n, ids.data_ptr(), ws.data_ptr())
        return int(ids[0])

    assert argmax(capped, N) == int((capped[0] == capped.max()).nonzero()[0])
    # forced ties on a small head: x = ones; rows 5 and 9 sum to ~460 / ~614 (capped: exactly 30), rows 70, 71 are equal
    n = 4096
    x1 = torch.ones(1, K, dtype=BF, device=DEV)
    w1 = rand(n, K, std=K**-0.5)
    w1[5], w1[9] = 0.3, 0.4
    w1[70] = w1[71] = 0.0175
    w1[60:70] = 0.0
    lo = zeros(1, n, dtype=torch.float32)
    gemv(L, w1, x1, n, K, 1, 2, [lo], cap=cap)
    raw = zeros(1, n, dtype=torch.float32)
    gemv(L, w1, x1, n, K, 1, 2, [raw])
    assert capped_tie(lo, 5, 9) and raw[0, 9] > raw[0, 5], (raw[0, 5].item(), raw[0, 9].item())
    assert argmax(lo, n) == 5 and argmax(raw, n) == 9, (argmax(lo, n), argmax(raw, n))
    w1[5] = w1[9] = 0.0
    w1[70] = w1[71] = 0.3
    gemv(L, w1, x1, n, K, 1, 2, [lo], cap=cap)
    assert lo[0, 70] == lo[0, 71] == lo.max() and argmax(lo, n) == 70, argmax(lo, n)
    print(
        "  head argmax: capped ties (rows 5, 9: accumulators differ, logits both exactly cap) and equal rows (70, 71)"
        " -> lowest index"
    )


def capped_tie(logits, a, b) -> bool:
    return logits[0, a].item() == logits[0, b].item() == 30.0 == logits.max().item()


if __name__ == "__main__":
    run(globals())
