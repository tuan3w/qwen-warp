"""K2 attention tests (lab venv).  Needs the GPU (take the lease around the run).

    tests/build_attn.sh && ~/llm-lab/venv/bin/python tests/test_attn.py

Reference = fp32 torch transcription of oracle/reference.py::Oracle._attn (rope angle in fp64), first validated against the oracle's own
`gated` tap on the real dump (layer 3, real weights/activations), then used for synthetic long contexts built from real K/V rows.
Tolerances: rel L2 of the bf16 output vs the fp32 reference (reference q is rounded like the kernel's q so only kernel error is measured).
"""
import ctypes as C
import os
import sys
from pathlib import Path

import torch

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE.parent / "oracle"))
from weights import Checkpoint  # noqa: E402

DEV = os.environ.get("ATTN_DEV", "cuda")
DUMP = os.path.expanduser("~/llm-lab/swift-ref/fp32")
TOL = 4e-3  # bf16 output rounding is ~1.1e-3 rms; bf16/fp16 P and q add a little
LIB = None


def lib():
    global LIB
    if LIB is None:
        so = os.environ.get("ATTN_SO", str(ENGINE / "build" / "libswiftmk_attn.so"))
        LIB = C.CDLL(so)
        vp, ci = C.c_void_p, C.c_int
        LIB.swiftmk_attn_ws_bytes.restype = C.c_size_t
        LIB.swiftmk_attn_kv_bytes.restype = C.c_size_t
        LIB.swiftmk_attn_kv_bytes.argtypes = [ci, ci]
        LIB.swiftmk_attn_step.restype = ci
        LIB.swiftmk_attn_step.argtypes = [ci, vp, vp, vp, vp, vp, vp, ci, ci, C.c_float, C.c_float, vp, vp, vp, vp]
    return LIB


def stream():
    return torch.cuda.current_stream().cuda_stream


class Layer:
    """One attention layer's weights + KV cache (bf16 or fp8) + workspace."""

    def __init__(self, qn, kn, ks, vs, ctx_max, fp8, ws=None):
        self.qn, self.kn = qn.to(DEV, torch.bfloat16), kn.to(DEV, torch.bfloat16)
        self.ks, self.vs, self.ctx_max, self.fp8 = float(ks), float(vs), ctx_max, fp8
        dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
        self.kv = torch.zeros(2, 4, ctx_max, 256, dtype=dt, device=DEV)
        assert self.kv.numel() * self.kv.element_size() == lib().swiftmk_attn_kv_bytes(ctx_max, int(fp8))
        self.ws = ws if ws is not None else torch.empty(lib().swiftmk_attn_ws_bytes(), dtype=torch.uint8, device=DEV)
        self.pos0 = torch.zeros(1, dtype=torch.int32, device=DEV)

    def load(self, K, V, n):
        """Fill positions 0..n-1 from fp32 K,V [4,n,256] (quantised like the kernel would store them)."""
        if self.fp8:
            self.kv[0, :, :n] = (K / self.ks).clamp(-448, 448).to(torch.float8_e4m3fn)
            self.kv[1, :, :n] = (V / self.vs).clamp(-448, 448).to(torch.float8_e4m3fn)
        else:
            self.kv[0, :, :n] = K.to(torch.bfloat16)
            self.kv[1, :, :n] = V.to(torch.bfloat16)

    def cache_f32(self, n):
        """Dequantised cache rows 0..n-1 as fp32 [2][4,n,256]."""
        x = self.kv[:, :, :n].float()
        if self.fp8:
            x = x * torch.tensor([self.ks, self.vs], device=DEV).view(2, 1, 1, 1)
        return x[0], x[1]

    def step(self, qg, k, v, pos0, T=None, out=None):
        """qg [T,12288] k,v [T,1024] bf16 cuda tensors."""
        T = qg.shape[0] if T is None else T
        self.pos0.fill_(pos0)
        out = torch.empty(T, 6144, dtype=torch.bfloat16, device=DEV) if out is None else out
        err = lib().swiftmk_attn_step(T, qg.data_ptr(), k.data_ptr(), v.data_ptr(), self.qn.data_ptr(), self.kn.data_ptr(), self.kv.data_ptr(),
                                      self.ctx_max, int(self.fp8), self.ks, self.vs, self.pos0.data_ptr(), out.data_ptr(), self.ws.data_ptr(), stream())
        assert err == 0, f"swiftmk_attn_step error {err}"
        return out


# --------------------------------------------------------------------------------------------- reference
def ref_step(qg, k, v, qn, kn, K, V, pos0, fp8_scales=None, quant="auto"):
    """fp32 oracle semantics.  qg [T,12288], k,v [T,1024] fp32; K,V [4,pos0,256] cache (post-rope, dequantised).
    quant=None: no roundings at all (pure fp32, used to validate against the oracle).
    fp8_scales=(ks,vs): the new K/V rows go through the fp8 store/dequant round trip and q through fp16 (as the kernel's fp8 path), else bf16.
    Returns out [T,6144], Knew, Vnew [4,T,256] (as stored/dequantised)."""
    T = qg.shape[0]
    qgv = qg.float().view(T, 24, 512)
    q, gate = qgv[..., :256], qgv[..., 256:]

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + w.float())

    q = rms(q, qn)
    kk = rms(k.float().view(T, 4, 256), kn)
    pos = torch.arange(pos0, pos0 + T, dtype=torch.float64, device=qg.device)
    inv = 1e7 ** (-torch.arange(32, dtype=torch.float64, device=qg.device) / 32.0)
    ang = pos[:, None] * inv[None]
    c, s = ang.cos().float()[:, None, :], ang.sin().float()[:, None, :]

    def rope(x):
        x1, x2 = x[..., :32], x[..., 32:64]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, x[..., 64:]], -1)

    q, kk = rope(q), rope(kk)
    vv = v.float().view(T, 4, 256)
    if quant is None:
        pass
    elif fp8_scales is None:
        q = q.to(torch.bfloat16).float()
        kk, vv = kk.to(torch.bfloat16).float(), vv.to(torch.bfloat16).float()
    else:
        ks, vs = fp8_scales
        q = q.to(torch.float16).float()
        kk = (kk / ks).clamp(-448, 448).to(torch.float8_e4m3fn).float() * ks
        vv = (vv / vs).clamp(-448, 448).to(torch.float8_e4m3fn).float() * vs
    Kn, Vn = kk.transpose(0, 1), vv.transpose(0, 1)
    Ka, Va = torch.cat([K[:, :pos0], Kn], 1), torch.cat([V[:, :pos0], Vn], 1)
    S = Ka.shape[1]
    qh = q.view(T, 4, 6, 256)
    sc = torch.einsum("tgjd,gsd->tgjs", qh, Ka) / 16.0
    keypos = torch.arange(S, device=qg.device)[None, None, None, :]
    qpos = (pos0 + torch.arange(T, device=qg.device))[:, None, None, None]
    sc = sc.masked_fill(keypos > qpos, float("-inf"))
    p = torch.softmax(sc, -1)
    o = torch.einsum("tgjs,gsd->tgjd", p, Va).reshape(T, 24, 256)
    out = (o * torch.sigmoid(gate)).reshape(T, 6144)
    return out, Kn, Vn


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def maxrel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------------------------- data
_cache = {}


def real():
    """layer-3 weights (real) + dump (3 prompts)."""
    if "r" not in _cache:
        ck = Checkpoint()
        p = "model.language_model.layers.3.self_attn."
        w = dict(qn=ck.get(p + "q_norm.weight"), kn=ck.get(p + "k_norm.weight"), ks=ck.get(p + "k_scale").float().item(), vs=ck.get(p + "v_scale").float().item())
        zs = {n: torch.load(os.path.join(DUMP, f"{n}.pt")) for n in ("code", "prose", "math")}
        _cache["r"] = (w, zs)
    return _cache["r"]


def bf(x):
    return x.to(DEV).to(torch.bfloat16)


def pad_rows(x, T):  # [n,*] -> [T,*]
    return x[:T]


def test_reference_vs_oracle():
    """Validate ref_step against the oracle's own `gated` tap (fp32, real layer-3 weights, real decode events)."""
    w, zs = real()
    worst = 0
    for name, z in zs.items():
        L = len(z["prompt_ids"])
        Kp, Vp = z["kv_layer3_after_prefill"]["k"].to(DEV), z["kv_layer3_after_prefill"]["v"].to(DEV)
        K, V = Kp[:, : L - 1].clone(), Vp[:, : L - 1].clone()
        for e in range(9):
            t = z["taps"][e][3]
            qg, k, v = (t[n].to(DEV).float() for n in ("q_proj", "k_proj", "v_proj"))
            # fp32 reference with exact fp32 inputs vs oracle gated (no bf16 rounding anywhere)
            out, Kn, Vn = ref_step(qg, k, v, w["qn"].to(DEV), w["kn"].to(DEV), K, V, L - 1 + e, quant=None)
            r = rel(out, t["gated"].to(DEV).reshape(1, 6144))
            worst = max(worst, r)
            assert torch.allclose(Kn[:, 0], t["k_rope"][0].to(DEV), atol=1e-4, rtol=1e-4), f"{name} e{e} k_rope mismatch"
            K, V = torch.cat([K, Kn], 1), torch.cat([V, Vn], 1)
    print(f"[ref vs oracle gated] worst rel L2 {worst:.2e}")
    assert worst < 2e-5


def real_events(z, L):
    """Per-event bf16 inputs [9,*] for layer 3."""
    t = [z["taps"][e][3] for e in range(9)]
    return [torch.cat([x[n] for x in t], 0).to(DEV).to(torch.bfloat16) for n in ("q_proj", "k_proj", "v_proj")], torch.cat([x["gated"].reshape(1, 6144) for x in t], 0).to(DEV)


def test_real_layer3():
    """Kernel on real weights/activations/KV (ctx 32..41) vs the oracle `gated` tap, bf16 and fp8 KV, T=1 sequential and T=4 blocks."""
    w, zs = real()
    qn, kn = w["qn"], w["kn"]
    res = {}
    for fp8 in (False, True):
        for name, z in zs.items():
            L = len(z["prompt_ids"])
            (QG, KK, VV), gated = real_events(z, L)
            Kp, Vp = z["kv_layer3_after_prefill"]["k"].to(DEV), z["kv_layer3_after_prefill"]["v"].to(DEV)
            ly = Layer(qn, kn, w["ks"], w["vs"], 128, fp8)
            # T=1 sequential: events 0..8 (event 0 = last prompt token at pos L-1, cache holds 0..L-2 before it)
            ly.load(Kp[:, : L - 1], Vp[:, : L - 1], L - 1)
            outs = torch.cat([ly.step(QG[e : e + 1], KK[e : e + 1], VV[e : e + 1], L - 1 + e) for e in range(9)], 0)
            r1 = rel(outs, gated)
            # T=4 blocks: events 0-3 then 4-7, in-block causality
            ly2 = Layer(qn, kn, w["ks"], w["vs"], 128, fp8)
            ly2.load(Kp[:, : L - 1], Vp[:, : L - 1], L - 1)
            o4 = torch.cat([ly2.step(QG[0:4], KK[0:4], VV[0:4], L - 1), ly2.step(QG[4:8], KK[4:8], VV[4:8], L + 3)], 0)
            r4 = rel(o4, gated[:8])
            # T=4 must equal 4 sequential steps (same cache bytes, outputs within kernel noise)
            same_kv = torch.equal(ly.kv[:, :, : L + 8].view(torch.uint8 if fp8 else torch.int16)[:, :, : L + 7], ly2.kv[:, :, : L + 8].view(torch.uint8 if fp8 else torch.int16)[:, :, : L + 7])
            r41 = rel(o4, outs[:8])
            res[(fp8, name)] = (r1, r4, r41)
            print(f"[real L3 {'fp8' if fp8 else 'bf16'} {name:5s} ctx {L}..{L+8}] vs oracle: T=1 {r1:.2e}  T=4 {r4:.2e}   T4-vs-4xT1 {r41:.2e} cache-equal={same_kv}")
            assert same_kv
            tol = 6e-2 if fp8 else TOL  # fp8 KV: quantisation error vs the fp32-KV oracle dominates, reported not tight
            assert r1 < tol and r4 < tol and r41 < TOL, (r1, r4, r41)
    return res


def synth_cache(z, ctx, seed=0):
    """Realistic-scale long KV [4,ctx,256]: resample the real post-rope K / V rows (+5% per-channel noise)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    Kp, Vp = z["kv_layer3_after_prefill"]["k"].to(DEV), z["kv_layer3_after_prefill"]["v"].to(DEV)
    L = Kp.shape[1]
    idx = torch.randint(0, L, (4, ctx), device=DEV, generator=g)
    K = torch.gather(Kp, 1, idx[..., None].expand(-1, -1, 256))
    V = torch.gather(Vp, 1, idx[..., None].expand(-1, -1, 256))
    K = K + 0.05 * K.std(1, keepdim=True) * torch.randn(K.shape, device=DEV, generator=g)
    V = V + 0.05 * V.std(1, keepdim=True) * torch.randn(V.shape, device=DEV, generator=g)
    return K, V


def run_synth(ctx, T, fp8, name="code", seed=0, layer=None):
    w, zs = real()
    z = zs[name]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = real_events(z, L)
    pos0 = max(ctx - T, 0)
    K, V = synth_cache(z, pos0, seed)
    ly = layer or Layer(w["qn"], w["kn"], w["ks"], w["vs"], max(ctx, 64) + 8, fp8)
    ly.load(K, V, pos0)
    Kd, Vd = ly.cache_f32(pos0)
    rows = slice(1, 1 + T)
    out = ly.step(QG[rows], KK[rows], VV[rows], pos0)
    ref, Kn, Vn = ref_step(QG[rows], KK[rows], VV[rows], w["qn"].to(DEV), w["kn"].to(DEV), Kd, Vd, pos0, (w["ks"], w["vs"]) if fp8 else None)
    # new rows written by the kernel
    Kw, Vw = ly.cache_f32(pos0 + T)
    kerr = rel(Kw[:, pos0:], Kn)
    verr = rel(Vw[:, pos0:], Vn)
    return rel(out, ref), maxrel(out, ref), kerr, verr


def test_synthetic_long():
    for fp8 in (False, True):
        for ctx in (1, 33, 1000, 8000, 32000):
            for T in (1, 4, 8):
                r, m, ke, ve = run_synth(ctx, T, fp8)
                eff = max(ctx, T)
                print(f"[synth {'fp8' if fp8 else 'bf16'} ctx {eff:5d} T={T}] vs ref: rel L2 {r:.2e} max/maxabs {m:.2e}  new-K row rel {ke:.1e} V {ve:.1e}")
                assert r < TOL and m < 8 * TOL and ke < 3e-3 and ve < 1e-3 + (0 if not fp8 else 3e-2), (r, m, ke, ve)


def test_t2_t3():
    """T=2, 3 (NT=2 / NT=3 configs) and K11's T=5..7 (NT=4 / 5 / 6) at a few contexts, bf16 and fp8."""
    for fp8 in (False, True):
        for ctx in (5, 100, 5000):
            for T in (2, 3, 5, 6, 7):
                r, m, ke, ve = run_synth(ctx, T, fp8)
                print(f"[synth {'fp8' if fp8 else 'bf16'} ctx {ctx:5d} T={T}] vs ref: rel L2 {r:.2e} max/maxabs {m:.2e}")
                assert r < TOL and m < 8 * TOL and ke < 3e-3


def test_t4_equals_4_sequential_long():
    """A T=4 step must equal 4 sequential T=1 steps (same cache bytes; outputs within kernel noise) with split-KV active (ctx 8000)."""
    w, zs = real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = real_events(z, L)
    for fp8, TT in [(f, t) for f in (False, True) for t in (4, 8)]:
        pos0 = 7992
        K, V = synth_cache(z, pos0, 9)
        a = Layer(w["qn"], w["kn"], w["ks"], w["vs"], 8064, fp8)
        b = Layer(w["qn"], w["kn"], w["ks"], w["vs"], 8064, fp8)
        a.load(K, V, pos0)
        b.load(K, V, pos0)
        o4 = a.step(QG[0:TT], KK[0:TT], VV[0:TT], pos0)
        o1 = torch.cat([b.step(QG[t : t + 1], KK[t : t + 1], VV[t : t + 1], pos0 + t) for t in range(TT)], 0)
        v = torch.uint8 if fp8 else torch.int16
        same = torch.equal(a.kv.view(v), b.kv.view(v))
        r = rel(o4, o1)
        print(f"[T={TT} vs {TT}xT=1 ctx 8000 {'fp8' if fp8 else 'bf16'}] rel L2 {r:.2e} cache-equal={same}")
        assert same and r < 2e-3


def test_fp8_vs_bf16():
    """Quantisation error of fp8 KV (real k_scale/v_scale) on the attention output, same underlying real-valued KV."""
    w, zs = real()
    rows = []
    for name, z in zs.items():
        L = len(z["prompt_ids"])
        (QG, KK, VV), gated = real_events(z, L)
        Kp, Vp = z["kv_layer3_after_prefill"]["k"].to(DEV), z["kv_layer3_after_prefill"]["v"].to(DEV)
        outs = {}
        for fp8 in (False, True):
            ly = Layer(w["qn"], w["kn"], w["ks"], w["vs"], 128, fp8)
            ly.load(Kp[:, : L - 1], Vp[:, : L - 1], L - 1)
            outs[fp8] = torch.cat([ly.step(QG[e : e + 1], KK[e : e + 1], VV[e : e + 1], L - 1 + e) for e in range(9)], 0)
        r = rel(outs[True], outs[False])
        rb = rel(outs[False], gated)
        rows.append(r)
        print(f"[fp8 vs bf16 real {name} ctx {L}..{L+8}] rel L2 fp8-vs-bf16 {r:.2e}   (bf16-vs-oracle {rb:.2e})")
    for ctx in (2000, 8000, 32000):
        r_syn = []
        for T in (1, 4):
            a = run_synth_pair(ctx, T)
            r_syn.append(a)
        print(f"[fp8 vs bf16 synth ctx {ctx}] rel L2 T=1 {r_syn[0]:.2e} T=4 {r_syn[1]:.2e}")
    return rows


def run_synth_pair(ctx, T):
    w, zs = real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = real_events(z, L)
    pos0 = ctx - T
    K, V = synth_cache(z, pos0, 1)
    o = {}
    for fp8 in (False, True):
        ly = Layer(w["qn"], w["kn"], w["ks"], w["vs"], ctx + 8, fp8)
        ly.load(K, V, pos0)
        o[fp8] = ly.step(QG[1 : 1 + T], KK[1 : 1 + T], VV[1 : 1 + T], pos0)
    return rel(o[True], o[False])


def test_mtp_layer():
    """MTP attention layer (bf16 weights, bf16 KV): real q_norm/k_norm, real xnorm -> raw q/k/v with the real MTP projections.
    The kernel's new K row must equal the oracle tap k_rope (validates norm+rope for the MTP weights); attention over a synthetic context."""
    ck = Checkpoint()
    p = "mtp.layers.0.self_attn."
    qn, kn = ck.get(p + "q_norm.weight"), ck.get(p + "k_norm.weight")
    Wq, Wk, Wv = (ck.get(p + n + "_proj.weight").float().to(DEV) for n in "qkv")
    _, zs = real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    for e in (0, 3):
        tp = z["mtp_taps"][e]
        x = tp["xnorm"].to(DEV).float()
        qg, k, v = (x @ W.T for W in (Wq, Wk, Wv))
        qg_b, k_b, v_b = (t.to(torch.bfloat16) for t in (qg, k, v))
        pos = L - 1 + e
        # tap q_proj sanity (our projection from xnorm reproduces the oracle's q_proj)
        assert rel(qg, tp["q_proj"].to(DEV)) < 1e-4
        K, V = synth_cache(z, pos, 5)
        ly = Layer(qn, kn, 1.0, 1.0, pos + 16, False)
        ly.load(K, V, pos)
        out = ly.step(qg_b, k_b, v_b, pos)
        Kw, Vw = ly.cache_f32(pos + 1)
        kerr = rel(Kw[:, pos], tp["k_rope"][0].to(DEV))
        verr = rel(Vw[:, pos], tp["v"][0].to(DEV))
        ref, _, _ = ref_step(qg_b, k_b, v_b, qn.to(DEV), kn.to(DEV), ly.cache_f32(pos)[0], ly.cache_f32(pos)[1], pos)
        r = rel(out, ref)
        print(f"[mtp attn event {e} pos {pos}] new K vs oracle k_rope {kerr:.2e}  V {verr:.2e}  out vs ref {r:.2e}")
        assert kerr < 5e-3 and verr < 5e-3 and r < TOL


def test_graph_device_scalar():
    """One captured graph (T=4, fp8 and bf16) replayed for several pos0 values written into the device scalar: equals eager each time."""
    w, zs = real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = real_events(z, L)
    for fp8 in (False, True):
        ctx_max = 4096
        K, V = synth_cache(z, ctx_max - 8, 3)
        ly = Layer(w["qn"], w["kn"], w["ks"], w["vs"], ctx_max, fp8)
        ly.load(K, V, ctx_max - 8)
        T = 4
        q, k, v = QG[1 : 1 + T].contiguous(), KK[1 : 1 + T].contiguous(), VV[1 : 1 + T].contiguous()
        out = torch.empty(T, 6144, dtype=torch.bfloat16, device=DEV)
        ly.step(q, k, v, 100, out=out)  # warm-up (sets smem attribute outside capture)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            err = lib().swiftmk_attn_step(T, q.data_ptr(), k.data_ptr(), v.data_ptr(), ly.qn.data_ptr(), ly.kn.data_ptr(), ly.kv.data_ptr(), ctx_max, int(fp8),
                                          ly.ks, ly.vs, ly.pos0.data_ptr(), out.data_ptr(), ly.ws.data_ptr(), torch.cuda.current_stream().cuda_stream)
            assert err == 0
        worst = 0
        for pos0 in (1, 17, 129, 1000, 4000, 4088):
            if pos0 + T > ctx_max:
                continue
            ly.pos0.fill_(pos0)
            g.replay()
            torch.cuda.synchronize()
            got = out.clone()
            eager = ly.step(q, k, v, pos0)
            worst = max(worst, rel(got, eager))
            Kd, Vd = ly.cache_f32(pos0)
            ref, _, _ = ref_step(q, k, v, w["qn"].to(DEV), w["kn"].to(DEV), Kd, Vd, pos0, (w["ks"], w["vs"]) if fp8 else None)
            assert rel(got, ref) < TOL, (pos0, rel(got, ref))
        print(f"[graph {'fp8' if fp8 else 'bf16'}] one captured graph, pos0 in 1..4088: graph-vs-eager worst rel {worst:.1e}")
        assert worst < 1e-6


def test_graph_first_call_under_capture():
    """First-ever call of a config inside stream capture (cudaFuncSetAttribute for the dynamic smem happens there)."""
    w, zs = real()
    z = zs["code"]
    L = len(z["prompt_ids"])
    (QG, KK, VV), _ = real_events(z, L)
    ctx_max = 2048
    for fp8 in (False, True):
        K, V = synth_cache(z, 1000, 4)
        ly = Layer(w["qn"], w["kn"], w["ks"], w["vs"], ctx_max, fp8)
        ly.load(K, V, 1000)
        T = 2
        q, k, v = QG[1 : 1 + T].contiguous(), KK[1 : 1 + T].contiguous(), VV[1 : 1 + T].contiguous()
        out = torch.empty(T, 6144, dtype=torch.bfloat16, device=DEV)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            err = lib().swiftmk_attn_step(T, q.data_ptr(), k.data_ptr(), v.data_ptr(), ly.qn.data_ptr(), ly.kn.data_ptr(), ly.kv.data_ptr(), ctx_max, int(fp8),
                                          ly.ks, ly.vs, ly.pos0.data_ptr(), out.data_ptr(), ly.ws.data_ptr(), torch.cuda.current_stream().cuda_stream)
            assert err == 0
        ly.pos0.fill_(1000)
        g.replay()
        torch.cuda.synchronize()
        Kd, Vd = ly.cache_f32(1000)
        ref, _, _ = ref_step(q, k, v, w["qn"].to(DEV), w["kn"].to(DEV), Kd, Vd, 1000, (w["ks"], w["vs"]) if fp8 else None)
        r = rel(out, ref)
        print(f"[graph first-call-in-capture T=2 {'fp8' if fp8 else 'bf16'}] rel L2 vs ref {r:.2e}")
        assert r < TOL


if __name__ == "__main__":
    assert torch.cuda.is_available()
    test_reference_vs_oracle()
    test_graph_first_call_under_capture()  # must run before anything else touches the T=2 (NT=2) kernels
    test_real_layer3()
    test_synthetic_long()
    test_t2_t3()
    test_t4_equals_4_sequential_long()
    test_fp8_vs_bf16()
    test_mtp_layer()
    test_graph_device_scalar()
    print("ALL PASSED")
