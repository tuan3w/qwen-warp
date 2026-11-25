"""The gated hd-256 attention family (warpc/cuda/attn_gated.py, kernels/qwen35_attn*.cu) against the hand engine's
swiftmk_attn_step.

  match:   the capability claims the Qwen3.5 attention (bf16 and int4_rot, not fp8), declines what it cannot run (CPU)
  rope:    the rope angle needs fp64 from the theta table: the fp32 plan table alone flips bf16 roundings of the
           cached rows (CPU)
  gate i:  the layer's whole KV buffer equals the hand engine's byte for byte after a run of steps over the same rows
  gate ii: the output is bit for bit the hand's, for both caches (T = 1 and 8 at ctx 385 / 4096 / 16384, random and
           real-like rows; T = 1..8 at ctx 385 and 4096)
  speed:   one decode step of both, CUDA-graph timed over a ring of layer caches (only with $WARPC_ATTN_SPEED)

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> uv run --package warpc python warpc/tests/test_attn_gated.py

The GPU tests need a CUDA device and $WARPC_SWIFTMK_LIB (the hand kernel); they skip without.
"""

import ctypes as C
import os
import time

import torch

from harness import Skip, run
from warpc import frontend
from warpc.compiler import CapsTable
from warpc.compiler.fuse import fuse
from warpc.cuda import attn_gated, glue, schedule
from warpc.cuda.table import RTX3090
from warpc.kvcache import FORMATS
from warpc.sources import ModelConfig

HQ, HKV, HD = 24, 4, 256
KV_DTYPE = {"bf16": 0, "int4_rot": 2}
CACHES = tuple(KV_DTYPE)

QWEN35_27B = {
    "model_type": "qwen3_5",
    "tie_word_embeddings": False,
    "vision_config": {},
    "text_config": {
        "vocab_size": 512,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 4,
        "num_attention_heads": HQ,
        "num_key_value_heads": HKV,
        "head_dim": HD,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4608,
        "tie_word_embeddings": False,
        "hidden_act": "silu",
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    },
}


class _ShapeSource:
    def __init__(self, config: ModelConfig):
        self.config = config

    def __contains__(self, name: str) -> bool:
        return True

    def quant(self, name: str) -> None:
        return None


def _graph(kv: str, **text):
    hf = {**QWEN35_27B, "text_config": {**QWEN35_27B["text_config"], **text}}
    return frontend.build(_ShapeSource(ModelConfig.from_hf(hf)), ctx_max=4608, kv=kv)


def _regions(g):
    return fuse(g, CapsTable(attn_gated.CAPS, RTX3090, lambda n, A: 1), 8)


def test_capability_claims_the_gated_attention():
    for kv in CACHES:
        res = _regions(_graph(kv))
        assert [r.kernel for r in res.regions] == ["g_attn_gated"], res.regions
        r = res.regions[0]
        assert r.attrs == {
            "Hq": HQ,
            "Hkv": HKV,
            "hd": HD,
            "ctx": 4608,
            "scale": 1 / 16,
            "eps": 1e-6,
            "offset": 1.0,
            "cache": kv,
        }, r.attrs
        assert set(r.ins) == {"q", "k", "v"} and set(r.outs) == {"y"} and set(r.weights) == {"q_norm", "k_norm"}
        assert r.states == {"kv": "L3.kv"} and len(r.covers) == 10
        assert "sigmoid_mul.sigmoid" in r.tolerance and "attention.p" in r.tolerance
    assert not _regions(_graph("fp8")).regions, "the fp8 cache has no kernel here"


def test_capability_limits_name_what_they_decline():
    cases = {
        "hd=128": {"head_dim": 128, "partial_rotary_factor": 0.5},
        "G=": {"num_attention_heads": 40},
        "rope=0": {"rope_parameters": {**QWEN35_27B["text_config"]["rope_parameters"], "rope_theta": 5e5}},
    }
    for name, text in cases.items():
        res = _regions(_graph("bf16", **text))
        assert not res.regions, name
        why = " ".join(w for _, ws, _ in res.backlog(_graph("bf16", **text)) for w in [ws])
        assert "g_attn_gated" in why, (name, why)


def test_rope_angle_needs_fp64():
    """The plan carries the fp32 table of theta 1e7; an angle from it (even multiplied in fp64) differs from the hand's
    fp64 theta^(-l/32) table by ~pos * 6e-8 rad, enough to flip bf16 roundings of the rotated rows."""
    gen = torch.Generator().manual_seed(0)
    exact = torch.tensor([1e7 ** (-j / 32) for j in range(32)], dtype=torch.float64)
    table = torch.tensor(attn_gated.ROPE_TABLE, dtype=torch.float32).double()
    x = torch.randn(4096, 64, generator=gen).to(torch.bfloat16).float()
    for pos in (385, 4096, 16384):
        out = []
        for freq in (exact, table):
            ang = pos * freq
            q = torch.round(ang / (2 * torch.pi))
            r = (ang - q * 2 * torch.pi).float()
            s, c = torch.sin(r), torch.cos(r)
            a, b = x[:, :32], x[:, 32:]
            out.append(torch.cat([a * c - b * s, b * c + a * s], 1).to(torch.bfloat16))
        flipped = float((out[0] != out[1]).float().mean())
        print(
            f"  rope at pos {pos}: {100 * flipped:.2f}% of the rotated bf16 values differ between the fp32 table"
            " and the fp64 theta table"
        )
        if pos >= 4096:
            assert flipped > 0.001, flipped


# ------------------------------------------------------------------------------------------------ GPU


def _env():
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    from warpc.swiftmk import lib as smk

    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    major, minor = torch.cuda.get_device_capability()
    return smk.load(), glue.load(major * 10 + minor)


def _stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def _ptr(t):
    return None if t is None else t.data_ptr()


class Layer:
    """One layer's attention state and weights, stepped by the hand kernel and the generated one on identical inputs."""

    def __init__(self, hand, gen, cache: str, ctx_max: int, seed: int = 0, real: bool = False):
        self.hand, self.gen, self.cache, self.ctx = hand, gen, cache, ctx_max
        self.fmt = FORMATS[cache]
        g = torch.Generator().manual_seed(seed)
        self.real = real
        # zero-centred norm weights (x * (1 + w)): small
        self.qn = (torch.randn(HD, generator=g) * 0.4).to(torch.bfloat16).cuda()
        self.kn = (torch.randn(HD, generator=g) * 0.4).to(torch.bfloat16).cuda()
        nbytes = hand.kv_bytes(ctx_max, cache)
        assert nbytes == self.fmt.nbytes(ctx_max, HKV, HD, 8), (nbytes, self.fmt.nbytes(ctx_max, HKV, HD, 8))
        self.kv_h = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        self.kv_g = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        self.pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.ws_h = torch.empty(int(hand.swiftmk_attn_ws_bytes()), dtype=torch.uint8, device="cuda")
        self.smax = attn_gated.split_blocks(HKV)
        self.chmin = attn_gated.MIN_SPLIT_KEYS
        self.ws_g = torch.empty(8 * HQ * self.smax * (HD + 4) * 4, dtype=torch.uint8, device="cuda")
        self.qbuf = torch.empty(8 * HQ * HD * 2, dtype=torch.uint8, device="cuda")
        self.ring_slots = self.fmt.ring(8) if self.fmt.quantized else 0
        self.meta = (
            (C.c_longlong * 13)(*schedule.kv_meta(self.fmt, ctx_max, HKV, HD, 8)) if self.fmt.quantized else None
        )
        self.rows = g
        self.chan = None
        if real:  # a few outlier channels per head, like real projections
            self.chan = (torch.exp(torch.randn(HQ * 2 * HD, generator=g) * 0.5)).cuda()
            self.chan[torch.randint(0, HQ * 2 * HD, (96,), generator=g).cuda()] *= 6.0

    def rows_for(self, T: int):
        g = self.rows
        qp = torch.randn(T, HQ * 2 * HD, generator=g)
        k = torch.randn(T, HKV * HD, generator=g)
        v = torch.randn(T, HKV * HD, generator=g) * 0.5
        if self.real:
            qp = qp * self.chan.cpu()
            k = k * 2.0
        qp, k, v = (x.to(torch.bfloat16).cuda().contiguous() for x in (qp, k, v))
        return qp, k, v

    def step_hand(self, qp, k, v, pos0: int, out, fill: bool = True):
        T = qp.shape[0]
        if fill:
            self.pos.fill_(pos0)
        rc = self.hand.swiftmk_attn_step(
            T,
            qp.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            self.qn.data_ptr(),
            self.kn.data_ptr(),
            self.kv_h.data_ptr(),
            self.ctx,
            KV_DTYPE[self.cache],
            1.0,
            1.0,
            self.pos.data_ptr(),
            None,
            out.data_ptr(),
            self.ws_h.data_ptr(),
            _stream(),
        )
        self.hand.check(rc, "swiftmk_attn_step")

    def step_gen(self, qp, k, v, pos0: int, out, fill: bool = True):
        T = qp.shape[0]
        if fill:
            self.pos.fill_(pos0)
        gen, st = self.gen, _stream()
        rc = gen.warpc_attn_g_prep(
            T,
            qp.data_ptr(),
            k.data_ptr(),
            v.data_ptr(),
            self.qn.data_ptr(),
            self.kn.data_ptr(),
            self.qbuf.data_ptr(),
            self.kv_g.data_ptr(),
            self.meta,
            self.pos.data_ptr(),
            HQ,
            HKV,
            self.ctx,
            C.c_float(1e-6),
            C.c_float(1.0),
            st,
        )
        assert rc == 0, f"prep: CUDA error {rc}"
        rc = gen.warpc_attn_g_decode(
            T,
            self.qbuf.data_ptr(),
            self.kv_g.data_ptr(),
            self.meta,
            self.pos.data_ptr(),
            qp.data_ptr(),
            out.data_ptr(),
            self.ws_g.data_ptr(),
            HQ,
            HKV,
            self.ctx,
            C.c_float(1 / 16),
            self.smax,
            self.chmin,
            st,
        )
        assert rc == 0, f"decode: CUDA error {rc}"

    def sections(self):
        """(name, byte offset, bytes) of the layer buffer's sections."""
        if not self.fmt.quantized:
            return [("K", 0, self.kv_h.numel() // 2), ("V", self.kv_h.numel() // 2, self.kv_h.numel() // 2)]
        lay = self.fmt.layout(self.ctx, HKV, HD, 8)
        names = list(lay)[:-1]
        return [
            (n, lay[n], (lay[names[i + 1]] if i + 1 < len(names) else lay["total"]) - lay[n])
            for i, n in enumerate(names)
        ]

    def diff_bytes(self) -> dict[str, int]:
        d = {}
        for name, off, n in self.sections():
            a, b = self.kv_h[off : off + n], self.kv_g[off : off + n]
            bad = int((a != b).sum())
            if bad:
                d[name] = bad
        return d


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    def ordered(x):
        u = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(u < 0, -(u & 0x7FFF), u)

    return (ordered(a) - ordered(b)).abs()


def reference_bf16(L: Layer, T: int, pos0: int, qp) -> torch.Tensor:
    """Truth for the bf16 cache: fp32 softmax(q K^T / 16) V * sigmoid(gate) over the cache rows the generated kernel
    wrote (byte-equal to the hand's), q as prepped by the generated prep (bf16), P not rounded."""
    half = L.kv_g.numel() // 2
    kk = L.kv_g[:half].view(torch.bfloat16).view(HKV, L.ctx, HD).float()
    vv = L.kv_g[half:].view(torch.bfloat16).view(HKV, L.ctx, HD).float()
    q = L.qbuf[: T * HQ * HD * 2].view(torch.bfloat16).view(T, HQ, HD).float()
    gate = qp.float().view(T, HQ, 2 * HD)[:, :, HD:]
    out = torch.empty(T, HQ, HD, device="cuda")
    for t in range(T):
        live = pos0 + t + 1
        for h in range(HQ):
            kvh = h // (HQ // HKV)
            p = torch.softmax(kk[kvh, :live] @ q[t, h] / 16, 0)
            out[t, h] = (p @ vv[kvh, :live]) * torch.sigmoid(gate[t, h])
    return out.to(torch.bfloat16).view(T, HQ * HD)


def compare(oh: torch.Tensor, og: torch.Tensor) -> dict:
    """hand vs generated: the share of elements within 2 bf16 ulp (all elements, and those at least 1/16 of their row's
    largest, where the ulp is a meaningful unit: an element near zero is a cancellation of the sum, its error is the
    row's), and the relative L2 distance."""
    h, g = oh.float(), og.float()
    u = ulp_distance(oh, og)
    big = h.abs() >= h.abs().amax(1, keepdim=True) / 16
    return {
        "all": float((u <= 2).float().mean()),
        "same": float((oh == og).float().mean()),
        "big": float((u[big] <= 2).float().mean()),
        "nbig": int(big.sum()),
        "max_ulp_big": int(u[big].max()),
        "rel_l2": float((h - g).norm() / h.norm()),
    }


def judge(c: dict) -> str | None:
    """Why an output is not acceptable, or None: bit for bit the hand's, for both caches (same keys per online softmax,
    same reduction orders, the hand's fp16 score accumulation for int4_rot)."""
    return None if c["same"] == 1.0 else f"not bitwise the hand's: {c}"


def _check_run(cache: str, real: bool, ctxs=(385, 4096, 16384)):
    hand, gen = _env()
    ctx_max = ctxs[-1] + 16
    L = Layer(hand, gen, cache, ctx_max, seed=1 if real else 0, real=real)
    pos = 0
    tag = f"{cache}{' real-like' if real else ''}"
    failures = []
    for ctx in ctxs:
        # history up to the test step's first row, then T = 8 and T = 1 steps ending at live length ctx
        while pos < ctx - 8:
            T = min(8, ctx - 8 - pos)
            qp, k, v = L.rows_for(T)
            oh = torch.empty(T, HQ * HD, dtype=torch.bfloat16, device="cuda")
            og = torch.empty_like(oh)
            L.step_hand(qp, k, v, pos, oh)
            L.step_gen(qp, k, v, pos, og)
            pos += T
        for T, pos0 in ((8, ctx - 8), (1, ctx - 1)):
            qp, k, v = L.rows_for(T)
            oh = torch.empty(T, HQ * HD, dtype=torch.bfloat16, device="cuda")
            og = torch.empty_like(oh)
            L.step_hand(qp, k, v, pos0, oh)
            L.step_gen(qp, k, v, pos0, og)
            torch.cuda.synchronize()
            bad = L.diff_bytes()
            c = compare(oh, og)
            line = (
                f"  {tag} ctx {ctx} T={T}: KV bytes differing {bad or 0}; output within 2 ulp:"
                f" {100 * c['all']:.2f}% of all, {100 * c['big']:.3f}% of the {c['nbig']} elements >= 1/16 of the row"
                f" max (max {c['max_ulp_big']} ulp); identical {100 * c['same']:.2f}%,"
                f" rel L2 hand-vs-generated {c['rel_l2']:.2e}"
            )
            print(line)
            if bad:
                failures.append(f"{tag} ctx {ctx} T={T}: KV buffer differs from the hand's in {bad}")
            if why := judge(c):
                failures.append(f"{tag} ctx {ctx} T={T}: {why}")
        pos = max(pos, ctx)
    assert not failures, "; ".join(failures)


def test_bf16_cache_and_output_match_the_hand_kernel():
    _check_run("bf16", real=False)
    _check_run("bf16", real=True)


def test_int4_rot_cache_and_output_match_the_hand_kernel():
    _check_run("int4_rot", real=False)
    _check_run("int4_rot", real=True)


def test_every_row_count_matches_the_hand_kernel():
    """T = 1..8 rows (n-tiles 1..6 of the split kernel) at a short and a long context, both caches."""
    hand, gen = _env()
    for cache in CACHES:
        L = Layer(hand, gen, cache, 4608)
        pos = 0
        failures = []
        for ctx in (385, 4096):
            while pos < ctx - 8:
                T = min(8, ctx - 8 - pos)
                qp, k, v = L.rows_for(T)
                oh = torch.empty(T, HQ * HD, dtype=torch.bfloat16, device="cuda")
                og = torch.empty_like(oh)
                L.step_hand(qp, k, v, pos, oh)
                L.step_gen(qp, k, v, pos, og)
                pos += T
            for T in range(8, 0, -1):
                qp, k, v = L.rows_for(T)
                oh = torch.empty(T, HQ * HD, dtype=torch.bfloat16, device="cuda")
                og = torch.empty_like(oh)
                L.step_hand(qp, k, v, ctx - T, oh)
                L.step_gen(qp, k, v, ctx - T, og)
                torch.cuda.synchronize()
                bad, c = L.diff_bytes(), compare(oh, og)
                print(
                    f"  {cache} ctx {ctx} T={T}: KV bytes differing {bad or 0}; {100 * c['big']:.3f}% of the big"
                    f" elements within 2 ulp (max {c['max_ulp_big']}), rel L2 {c['rel_l2']:.2e}"
                )
                if bad or (why := judge(c)):
                    failures.append((cache, ctx, T, bad, why))
            pos = max(pos, ctx)
        assert not failures, failures


# ------------------------------------------------------------------------------------------------ speed


def _graph_us(fn, replays=200, rounds=5) -> float:
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / replays * 1e6)
    return best


def _speed_row(hand, gen, cache: str, ctx: int) -> tuple[float, float, int, int]:
    """(hand us, generated us, layers, ctx_max) of one decode step at live length `ctx`, per layer, over a ring of layer
    caches larger than the L2 and its prefetch."""
    ctx_max = max(4608, ctx + 8)
    nbytes = hand.kv_bytes(ctx_max, cache)
    n = int(min(16, max(1, -(-(160 << 20) // nbytes))))
    L = Layer(hand, gen, cache, ctx_max)
    kvs = []
    for _ in range(n):
        kv = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
        if cache == "bf16":
            kv.view(torch.bfloat16).copy_(torch.randn(nbytes // 2, device="cuda").to(torch.bfloat16))
        else:
            lay = FORMATS[cache].layout(ctx_max, HKV, HD, 8)
            kv[: lay["kx"]] = torch.randint(0, 256, (lay["kx"],), dtype=torch.uint8, device="cuda")
        kvs.append(kv)
    qp, k, v = L.rows_for(1)
    out = torch.empty(1, HQ * HD, dtype=torch.bfloat16, device="cuda")
    L.pos.fill_(ctx - 1)  # set once: a fill inside the graph would be a memset node of the timing

    def run_hand():
        for kv in kvs:
            L.kv_h = kv
            L.step_hand(qp, k, v, ctx - 1, out, fill=False)

    def run_gen():
        for kv in kvs:
            L.kv_g = kv
            L.step_gen(qp, k, v, ctx - 1, out, fill=False)

    h = min(_graph_us(run_hand) for _ in range(2)) / n
    g = min(_graph_us(run_gen) for _ in range(2)) / n
    return h, g, n, ctx_max


def test_speed_against_the_hand_step():
    if not os.environ.get("WARPC_ATTN_SPEED"):
        raise Skip("$WARPC_ATTN_SPEED not set (the speed A/B takes the card for a few minutes)")
    hand, gen = _env()
    for cache in CACHES:
        for ctx in (385, 4096, 16384, 65536):
            h, g, n, ctx_max = _speed_row(hand, gen, cache, ctx)
            print(
                f"  {cache:9s} ctx {ctx:6d} (ctx_max {ctx_max}, {n} layers): hand {h:7.1f} us  generated {g:7.1f} us"
                f"  ({100 * h / g:5.1f}% of the hand speed)"
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    run(globals())
