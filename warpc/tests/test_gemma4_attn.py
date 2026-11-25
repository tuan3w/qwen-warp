"""Gemma 4's attention kernels (g_attn widened, g_attn_read) against the reference interpreter, and their capability
entries, one kernel at a time.

  uv run --package warpc python warpc/tests/test_gemma4_attn.py

The GPU tests lower a small graph of exactly the nodes the frontend builds around one attention (reshape, q / k norm,
weightless v norm, rope, attention [, attention_read of the same state]), take the regions the fusion pass claims, and
launch them through the executor's own launcher (`launch.LAUNCHERS`) on device buffers, a ring state allocated by
`state_nbytes` and the plan's constant tables. The oracle is the reference interpreter running the same nodes on the
same inputs with a POSITIONAL cache (every position in its own slot), so a ring that wraps, a window that skips
chunks and the chunk rounding of the ring are all checked against the definition of the op. Tolerance: rel L2 <=
6e-3 per step (the kernel keeps the softmax probabilities in fp32 and rotates in fp32: the declared `attention.p`,
`rope.trig`, `rope.prod` gaps; the old tree measured 1.2-3.0e-3); the measured errors are printed.
"""

import math

import torch

from harness import Skip, run
from warpc.compiler.fuse import fuse
from warpc.compiler.plan import Task
from warpc.cuda import CudaBackend, glue, launch, schedule
from warpc.cuda.backend import state_nbytes
from warpc.ir import Graph, StateDecl, TType, WeightDecl
from warpc.kvcache import FORMATS
from warpc.reference import Interpreter
from warpc.runtime import Memory, _Scratch

GATE = 6e-3
EPS = 1e-6
# q / k norm weights: scores (scale 1.0) of order sqrt(hd) * NORM_WEIGHT^2 ~ 0.5, spread over many keys; the kernels'
# error against the bf16 reference grows with the score magnitude (rope and P rounding): 2.7e-3 at 0.12, 4.2e-3 at 0.18
NORM_WEIGHT = 0.18
ROWS_MAX = 8
CHUNKS = (8, 1, 1, 4, 3, 1, 6)  # rows per step: prompt windows and decode steps through a ring that wraps
BACKEND = CudaBackend()


def require_gpu() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


def rope_table(rd: int, theta: float, live: int | None = None) -> tuple[float, ...]:
    """The rd / 2 inverse frequencies of a plain rope; with `live` < rd / 2 the rest are zero (proportional rope: the
    angles of the other pairs are 0, the identity)."""
    n = rd // 2
    live = n if live is None else live
    return tuple([theta ** (-2.0 * i / rd) for i in range(live)] + [0.0] * (n - live))


SLIDING = {"hq": 8, "hkv": 1, "hd": 256, "window": 512, "inv_freq": rope_table(256, 1e4)}
GLOBAL = {"hq": 8, "hkv": 1, "hd": 512, "window": None, "inv_freq": rope_table(512, 1e6, 64)}


def attn_weights(hd: int, seed: int = 0) -> dict[str, torch.Tensor]:
    """q / k norm weights scaled so that the scores (scale 1.0) spread over many keys instead of one."""
    g = torch.Generator().manual_seed(seed)
    return {n: (NORM_WEIGHT * (1 + 0.3 * torch.randn(hd, generator=g))).bfloat16() for n in ("qn", "kn", "qr")}


def attn_graph(*, hq, hkv, hd, window, inv_freq, rows, reader=True, qk_norm=True, v_norm=True, cache="bf16") -> Graph:
    """The nodes of one Gemma owner layer's attention (and of a layer that shares its state), as the frontend builds
    them, with the projection outputs as graph inputs."""
    g = Graph("attn")
    pos = g.input("pos", TType(("T",), "i32"))
    g.state(StateDecl("kv", "kv", (2, rows, hkv, hd), cache))
    for n in ("qn", "kn", "qr"):
        g.weight(WeightDecl(n, (hd,), "dense", n))

    def heads(name: str, n: int, norm: str | None) -> str:
        x = g.op("reshape", [g.input(name, TType(("T", n * hd), "bf16"))], shape=["T", n, hd])
        if norm == "plain":
            return g.op("rmsnorm", [x], eps=EPS)
        return x if norm is None else g.op("rmsnorm", [x], weight=norm, eps=EPS)

    rd = len(inv_freq) * 2
    q = heads("qp", hq, "qn" if qk_norm else None)
    k = heads("kp", hkv, "kn" if qk_norm else None)
    v = heads("vp", hkv, "plain" if v_norm else None)
    q, k = g.op_multi("rope", [q, k, pos], rotary_dim=rd, inv_freq=inv_freq)
    outs = {"y": g.op("attention", [q, k, v, pos], state="kv", scale=1.0, window=window)}
    if reader:
        q = g.op("rope", [heads("qs", hq, "qr" if qk_norm else None), pos], rotary_dim=rd, inv_freq=inv_freq)
        outs["yr"] = g.op("attention_read", [q, pos], state="kv", scale=1.0, window=window)
    g.output(**outs)
    return g


class DictSource:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors

    def get(self, name: str) -> torch.Tensor:
        return self.tensors[name]


class Rig:
    """The tasks of `graph` as the executor launches them, on device buffers: one `step` writes the inputs, sets the
    position, runs every task through `launch.LAUNCHERS` and returns the graph outputs."""

    def __init__(self, graph: Graph, weights: dict[str, torch.Tensor], shape: dict):
        require_gpu()
        res = fuse(graph, BACKEND.caps, ROWS_MAX)
        assert not res.unclaimed, res.backlog(graph)
        major, minor = torch.cuda.get_device_capability()
        self.graph, self.shape, self.lib = graph, shape, glue.load(10 * major + minor)
        constants: dict[str, torch.Tensor] = {}
        self.tasks: list[Task] = []
        for r in sorted(res.regions, key=lambda r: min(r.covers)):
            names = {}
            for role, values in r.constants.items():
                names[role] = f"{role}.{len(constants)}"
                constants[names[role]] = torch.tensor(values, dtype=torch.float32, device="cuda")
            self.tasks.append(
                Task(r.kernel, dict(r.ins), dict(r.outs), dict(r.weights), dict(r.states), names, dict(r.attrs))
            )
        dims = {v: math.prod(t.shape[1:]) for v, t in graph.types.items()}
        wanted = [v for v in graph.inputs if v != "pos"] + [v for r in res.regions for v in r.outs.values()]
        self.buffers = {v: torch.zeros(ROWS_MAX * dims[v], dtype=torch.bfloat16, device="cuda") for v in wanted}
        self.dims = dims
        self.states = {
            s.name: torch.zeros(state_nbytes(s, ROWS_MAX), dtype=torch.uint8, device="cuda")
            for s in graph.states.values()
        }
        self.packs = {n: w.to("cuda").contiguous() for n, w in weights.items()}
        self.pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.constants = constants
        self.scratch = _Scratch()
        self.mem = Memory(
            rows_max=ROWS_MAX,
            pos=self.pos.data_ptr(),
            tokens=0,
            ids=0,
            buffers={k: v.data_ptr() for k, v in self.buffers.items()},
            packs={k: v.data_ptr() for k, v in self.packs.items()},
            states={k: v.data_ptr() for k, v in self.states.items()},
            constants={k: v.data_ptr() for k, v in constants.items()},
            scratch=self.scratch,
        )

    def cache(self, name: str = "kv") -> torch.Tensor:
        """The state as the kernels lay it out: bf16 [2 (K, V)][Hkv][rows rounded to chunks][hd]."""
        _, rows, hkv, hd = self.graph.states[name].shape
        return self.states[name].view(torch.bfloat16).view(2, hkv, schedule.kv_rows(rows), hd)

    def step(self, p0: int, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rows = next(iter(inputs.values())).shape[0]
        for name, x in inputs.items():
            self.buffers[name][: x.numel()].copy_(x.reshape(-1))
        self.pos.fill_(p0)
        stream = torch.cuda.current_stream().cuda_stream
        for t in self.tasks:
            launch.LAUNCHERS[t.kernel](self.lib, self.mem, t, rows, stream)
        torch.cuda.synchronize()
        return {a: self.buffers[v][: rows * self.dims[v]].view(rows, -1).clone() for a, v in self.graph.outputs.items()}


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def step_inputs(shape: dict, rows: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    def rnd(n: int, scale: float = 1.0) -> torch.Tensor:
        return (scale * torch.randn(rows, n, generator=gen)).bfloat16().cuda()

    hq, hkv, hd = shape["hq"], shape["hkv"], shape["hd"]
    return {"qp": rnd(hq * hd), "kp": rnd(hkv * hd), "vp": rnd(hkv * hd, 3.0), "qs": rnd(hq * hd)}


def preload(rig: Rig, ref: Interpreter, p0: int, gen: torch.Generator) -> None:
    """Positions 0 .. p0 - 1 of random K / V rows in the kernel's state (slot = position mod its rows) and the
    reference's, as if p0 tokens had been decoded."""
    k = (
        (0.25 * torch.randn(p0, rig.graph.states["kv"].shape[2], rig.graph.states["kv"].shape[3], generator=gen))
        .bfloat16()
        .cuda()
    )
    v = torch.randn(k.shape, generator=gen).bfloat16().cuda()
    ref.states["kv"].rows[0, :p0], ref.states["kv"].rows[1, :p0] = k, v
    cache = rig.cache()
    lo = max(0, p0 - cache.shape[2])
    idx = torch.arange(lo, p0, device="cuda")
    slots = idx % cache.shape[2]
    cache[0][:, slots], cache[1][:, slots] = k[idx].transpose(0, 1), v[idx].transpose(0, 1)


def pair(shape: dict, ring: int, ctx: int, **kw) -> tuple[Rig, Interpreter]:
    """The kernels' rig on a state of `ring` rows and the reference on a positional state of `ctx` rows."""
    weights = attn_weights(shape["hd"])
    kernel = Rig(attn_graph(**shape, rows=ring, **kw), weights, shape)
    ref = Interpreter(attn_graph(**shape, rows=ctx, **kw), DictSource(weights), device="cuda")
    return kernel, ref


def compare(rig: Rig, ref: Interpreter, p0: int, rows: int, gen: torch.Generator) -> dict[str, float]:
    ins = step_inputs(rig.shape, rows, gen)
    want = ref.run(
        {
            **{k: v for k, v in ins.items() if k in ref.graph.inputs},
            "pos": torch.arange(p0, p0 + rows, dtype=torch.int32),
        }
    )
    got = rig.step(p0, {k: v for k, v in ins.items() if k in rig.graph.inputs})
    return {a: rel_l2(got[a], want[a]) for a in want}


def sequence(name: str, shape: dict, ring: int, ctx: int, upto: int) -> None:
    """Chunks of 8 / 1 / 1 / 4 / 3 / 1 / 6 rows up to position `upto`, every step against the positional reference."""
    rig, ref = pair(shape, ring, ctx)
    gen = torch.Generator().manual_seed(1)
    errs: dict[str, list[float]] = {"y": [], "yr": []}
    p, k = 0, 0
    while p < upto:
        rows = CHUNKS[k % len(CHUNKS)]
        for a, e in compare(rig, ref, p, rows, gen).items():
            errs[a].append(e)
        p, k = p + rows, k + 1
    for a, e in errs.items():
        assert max(e) <= GATE, (name, a, max(e))
    # what the prep wrote: the newest window of K (rope, fp32 vs bf16 reference) and V (weightless norm)
    cache, rows_ref = rig.cache(), ref.states["kv"].rows
    keep = torch.arange(p - min(p, shape["window"] or p), p, device="cuda")
    kv_err = [rel_l2(cache[i][:, keep % cache.shape[2]], rows_ref[i, keep].transpose(0, 1)) for i in (0, 1)]
    assert max(kv_err) <= GATE, (name, kv_err)
    print(
        f"  {name}: {k} steps to position {p}, ring {cache.shape[2]} rows, "
        f"rel L2 owner max {max(errs['y']):.2e} mean {sum(errs['y']) / k:.2e}, "
        f"reader max {max(errs['yr']):.2e} mean {sum(errs['yr']) / k:.2e}, cache K {kv_err[0]:.2e} V {kv_err[1]:.2e}"
    )


def test_sliding_hd256_ring_wraps_against_a_positional_reference():
    """Window 512, ring 519 (576 allocated): 7 step shapes through positions 0 .. 3000, owner and reader."""
    sequence("sliding hd 256 window 512", SLIDING, 519, 3072, 3000)


def test_global_hd512_with_weightless_v_norm():
    """Head dim 512, positional state of 3072 rows, rotary_dim 512 over the zero-padded 256-entry table (64 live)."""
    sequence("global hd 512", GLOBAL, 3072, 3072, 3000)


def sweep(name: str, shape: dict, ring: int, ctx: int, positions: list[int], rows: range) -> None:
    gen = torch.Generator().manual_seed(2)
    worst = {"y": 0.0, "yr": 0.0}
    n = 0
    for p0 in positions:
        rig, ref = pair(shape, ring, ctx)
        for rows_ in rows:
            ref.reset()
            rig.states["kv"].zero_()
            preload(rig, ref, p0, gen)
            for a, e in compare(rig, ref, p0, rows_, gen).items():
                worst[a] = max(worst[a], e)
                assert e <= GATE, (name, a, p0, rows_, e)
            n += 1
    print(f"  {name}: {n} (position, rows) cases, worst rel L2 owner {worst['y']:.2e} reader {worst['yr']:.2e}")


def test_chunk_and_wrap_boundaries_of_the_ring():
    """Steps of 1 .. 8 rows starting at the positions where the window edge, the 64-key chunks and the 576-row wrap
    meet, on a preloaded cache."""
    edges = [0, 1, 63, 64, 65, 510, 511, 512, 513, 575, 576, 577, 1023, 1024, 1151, 1152, 1153, 2047, 2999]
    sweep("sliding hd 256 boundaries", SLIDING, 519, 3100, edges, range(1, 9))
    sweep("global hd 512 boundaries", GLOBAL, 3072, 3072, [0, 1, 63, 64, 65, 2999], range(1, 9))
    # 1024 rows: 32 chunks of 32 keys, which the one-row kernel merges itself (more chunks: k_attn_reduce2)
    sweep("global hd 512 ctx 1024", GLOBAL, 1024, 1024, [0, 31, 32, 33, 500, 991, 1015], range(1, 9))


def test_small_windows_and_two_kv_heads():
    """Windows of 1, 2, 8, 64, 65 and 100 keys (rings of 8 .. 107 rows, 64 .. 128 allocated), 2 KV heads of 4 query
    heads, hd 64: the window edge inside, on and across chunk boundaries."""
    for window in (1, 2, 8, 64, 65, 100):
        shape = {"hq": 4, "hkv": 2, "hd": 64, "window": window, "inv_freq": rope_table(64, 1e4)}
        sweep(f"hd 64 window {window}", shape, window + ROWS_MAX - 1, 700, [0, 5, 63, 64, 130, 400, 640], range(1, 9))


def test_chunked_split_blocks_of_long_contexts():
    """nc = 2 (more than 128 split blocks per head): the chunked kernel with a window and without."""
    long_window = {**SLIDING, "window": 9000, "inv_freq": rope_table(256, 1e4)}
    assert schedule.attn_chunks(9007, 9000) == 2  # the ring is 141 chunks: 141 split blocks per head > 128
    assert schedule.attn_chunks(12000, 9000) == 2
    sweep("sliding hd 256 window 9000", long_window, 9007, 12100, [8200, 11990], [1, 3, 8])
    sweep("global hd 512 ctx 8256", GLOBAL, 8256, 8256, [8000, 8240], [1, 3, 8])


def test_dense_head_dims_still_match_the_reference():
    """Qwen3-0.6B's shape (16 query heads over 8 KV heads of 128) and hd 64 / 256, no window, no v norm, qk-norm on: the
    path every dense graph takes (the fused one-row kernel included)."""
    for shape in (
        {"hq": 16, "hkv": 8, "hd": 128, "window": None, "inv_freq": rope_table(128, 1e6)},
        {"hq": 4, "hkv": 2, "hd": 64, "window": None, "inv_freq": rope_table(64, 1e4)},
        {"hq": 4, "hkv": 2, "hd": 256, "window": None, "inv_freq": rope_table(256, 1e4)},
    ):
        rig, ref = pair(shape, 1024, 1024, reader=False, v_norm=False)
        gen = torch.Generator().manual_seed(3)
        worst, p = 0.0, 0
        for k in range(40):
            rows = CHUNKS[k % len(CHUNKS)] if k < 21 else 1  # prompt chunks, then decode steps
            worst = max(worst, compare(rig, ref, p, rows, gen)["y"])
            p += rows
        assert worst <= GATE, (shape["hd"], worst)
        print(f"  dense hd {shape['hd']} ({shape['hq']}/{shape['hkv']} heads): {p} positions, worst rel L2 {worst:.2e}")


def test_split_counts_agree_with_the_kernel():
    require_gpu()
    major, minor = torch.cuda.get_device_capability()
    lib = glue.load(10 * major + minor)
    for ctx in (64, 519, 576, 3072, 8256, 12000, 20000):
        for window in (None, 1, 64, 65, 512, 9000):
            for nc in (1, 2, 4):
                for ch in (schedule.ATTN_CHUNK, schedule.ATTN_ONE_ROW_CHUNK):
                    want = schedule.attn_splits(ctx, nc, window, ch)
                    assert lib.warpc_attn_splits(ctx, window or 0, nc, ch) == want, (ctx, window, nc, ch)


# ---------------------------------------------------------------------------------------------------- CPU: the table
def test_table_claims_the_gemma_attention_nodes():
    """g_attn takes reshape x3, q / k norms, the weightless v norm, rope and the attention; g_attn_read the q reshape,
    q norm, rope and attention_read; the regions carry the window and the v norm as attributes."""
    g = attn_graph(**SLIDING, rows=519)
    res = fuse(g, BACKEND.caps, ROWS_MAX)
    assert not res.unclaimed, res.backlog(g)
    own, read = sorted(res.regions, key=lambda r: min(r.covers))
    assert (own.kernel, read.kernel) == ("g_attn", "g_attn_read")
    assert [g.nodes[i].op for i in sorted(own.covers)].count("rmsnorm") == 3 and len(own.covers) == 8
    assert len(read.covers) == 4 and set(read.ins) == {"q"} and read.states == {"kv": "kv"}
    assert own.attrs["window"] == 512 and own.attrs["v_norm"] and read.attrs["window"] == 512
    assert "v_norm" not in read.attrs and set(read.weights) == {"q_norm"}
    assert BACKEND.caps.by_kernel["g_attn_read"].launches == 3
    plain = fuse(attn_graph(**GLOBAL, rows=4096, reader=False, v_norm=False), BACKEND.caps, ROWS_MAX).regions
    assert "window" not in plain[0].attrs and "v_norm" not in plain[0].attrs  # a dense region is what it was


def test_table_refuses_what_the_kernels_do_not_have():
    def why(**kw) -> str:
        g = attn_graph(**{**SLIDING, "rows": 519, "reader": False, **kw})
        return "; ".join(sorted(set(fuse(g, BACKEND.caps, ROWS_MAX).unclaimed.values())))

    # the IR already refuses a window on a quantised state; the kernel table says so too
    assert "no sliding-window / ring kernel" in schedule.kv_kernel_reason(FORMATS["k8v4w"], 128, 512)
    assert schedule.kv_kernel_reason(FORMATS["bf16"], 512, 512) is None
    assert "has no kernel for head dim 512" in why(**{**GLOBAL, "rows": 4096, "cache": "int4_rot", "v_norm": False})
    assert "has no weightless v-norm kernel" in why(**{**GLOBAL, "window": None, "rows": 4096, "cache": "k8v8"})
    g = attn_graph(**SLIDING, rows=300, reader=False)
    g.state(StateDecl("big", "kv", (2, 4096, 1, 256)))
    assert "cannot hold a window of 512 keys and a step of 8 rows" in str(fuse(g, BACKEND.caps, ROWS_MAX).unclaimed)


def test_state_bytes_round_a_bf16_ring_to_whole_chunks():
    ring = StateDecl("r", "kv", (2, 519, 1, 256))
    assert state_nbytes(ring, 8) == 2 * 576 * 256 * 2 == 2 * schedule.kv_rows(519) * 256 * 2
    assert state_nbytes(StateDecl("p", "kv", (2, 4096, 1, 512)), 8) == 2 * 4096 * 512 * 2  # a multiple of 64: as before
    assert state_nbytes(StateDecl("p", "kv", (2, 100, 8, 128)), 8) == 2 * 128 * 8 * 128 * 2
    q = StateDecl("q", "kv", (2, 100, 8, 128), cache="k8v4w")
    assert state_nbytes(q, 8) == FORMATS["k8v4w"].nbytes(100, 8, 128, 8)  # a quantised format keeps its own layout


def test_split_counts_of_windows():
    assert schedule.attn_splits(576, 1, 512) == 9 and schedule.attn_splits(519, 1, 512) == 9
    assert [schedule.attn_splits(64 * 40, 1, w) for w in (1, 64, 65, 512, 513)] == [1, 2, 2, 9, 9]
    assert schedule.attn_splits(4096, 1, None) == 64 and schedule.attn_chunks(4096, 512) == 1


def test_prefetch_pairs_an_attention_with_its_o_projection():
    def task(kernel: str, **kw) -> Task:
        return Task(kernel, kw.pop("ins", {}), kw.pop("outs", {}), attrs=kw)

    for kernel in ("g_attn", "g_attn_read"):
        attn = task(kernel, outs={"y": "b3"}, Hq=8, hd=256)
        o = task("g_gemv", ins={"x": "b3"}, N=1536, K=2048)
        assert launch.prefetch_bytes(attn, o) == 3 << 20  # half of the 6 MiB of o
        assert launch.prefetch_bytes(attn, task("g_gemv", ins={"x": "b4"}, N=1536, K=2048)) == 0  # not its output
        assert launch.prefetch_bytes(attn, task("g_gemv", ins={"x": "b3"}, N=3072, K=1024)) == 0  # not an o projection
        assert launch.prefetch_bytes(attn, task("g_q4gemv", ins={"x": "b3"}, N=1536, K=2048)) == 0
        assert launch.prefetch_bytes(attn, task("g_gemv_resadd", N=1024, K=2048)) == 2 << 20  # dense boundary as before
    wide = task("g_attn", outs={"y": "b3"}, Hq=8, hd=512)
    assert launch.prefetch_bytes(wide, task("g_gemv", ins={"x": "b3"}, N=1536, K=4096)) == 4 << 20  # the cap


if __name__ == "__main__":
    run(globals())
