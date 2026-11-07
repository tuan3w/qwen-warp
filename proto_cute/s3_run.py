import sys, time, torch, numpy as np
import cutlass, cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import mega as M

L = int(sys.argv[1]) if len(sys.argv) > 1 else 8          # virtual layers per step
T = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NSET = 2                                                  # distinct weight sets, cycled (GPU shared: 800 MB free)
dev = "cuda"
shapes = dict(qkv=(16384, 5120), o=(5120, 6144), gu=(34816, 5120), dn=(5120, 17408))
woff, soff, wn, sn = {}, {}, 0, 0
for st in range(NSET):
    for k, (N, K) in shapes.items():
        woff[st, k] = wn; wn += N * K // 8
        soff[st, k] = sn; sn += N * (K // 128) // 2        # u32 words holding bf16 pairs
torch.manual_seed(0)
Wh = torch.randint(-2**31, 2**31 - 1, (wn,), dtype=torch.int32, device=dev)
Sh = ((torch.rand(sn * 2, device=dev) * 0.004 + 0.001).to(torch.bfloat16)).view(torch.int32)
def al(n): return (n + 63) // 64 * 64
offs, a = {}, 0
for name, sz in dict(h=H if False else 5120 * T, xn=5120 * T, qkv=16384 * T, y6=6144 * T, mix=5120 * T, act=17408 * T, dn=5120 * T, nw1=5120, nw2=5120).items():
    offs[name] = a; a += al(sz)
A0 = (torch.randn(a, device=dev) * 0.5).to(torch.bfloat16)
for k in ("nw1", "nw2"): A0[offs[k]:offs[k] + 5120] = (torch.randn(5120, device=dev) * 0.1).to(torch.bfloat16)
rows = []
def R(op, N=0, K=0, w=0, s=0, x=0, o=0, epi=0, a0=0, a1=0, a2=0, a3=0): rows.append([op, N, K, w, s, x, o, epi, a0, a1, a2, a3])
for l in range(L):
    st = l % NSET
    R(M.OP_NORM, epi=int(l > 0), a0=offs["h"], a1=offs["dn"], a2=offs["xn"], a3=offs["nw1"])
    R(M.OP_GEMV, 16384, 5120, woff[st, "qkv"], soff[st, "qkv"], offs["xn"], offs["qkv"], 0)
    R(M.OP_MIX, a0=offs["qkv"], a2=offs["y6"])
    R(M.OP_GEMV, 5120, 6144, woff[st, "o"], soff[st, "o"], offs["y6"], offs["mix"], 0)
    R(M.OP_NORM, epi=1, a0=offs["h"], a1=offs["mix"], a2=offs["xn"], a3=offs["nw2"])
    R(M.OP_GEMV, 34816, 5120, woff[st, "gu"], soff[st, "gu"], offs["xn"], offs["act"], 1)
    R(M.OP_GEMV, 5120, 17408, woff[st, "dn"], soff[st, "dn"], offs["act"], offs["dn"], 0)
import os
if os.environ.get("NONORM"):
    for r in rows:
        if r[0] == M.OP_NORM: r[0] = 9
tbl = torch.tensor(rows, dtype=torch.int32, device=dev).flatten()
S = len(rows)
sync = torch.zeros(16 + len(rows) + 8, dtype=torch.int32, device=dev)
A = A0.clone()
mk = lambda t: from_dlpack(t, assumed_align=16)
args = (mk(tbl), mk(Wh), mk(Sh), mk(A), mk(A.view(torch.int32)), mk(sync))
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
bytes_per_layer = sum(N * K // 2 + N * K // 128 * 2 for N, K in shapes.values())
print(f"S={S} stages ({L} layers, T={T}), {bytes_per_layer/1e6:.0f} MB/layer, floor @ 900 GB/s = {L*bytes_per_layer/900e9*1e3:.2f} ms", flush=True)

def build(**kw):
    k = M.Mega(T, S, **kw)
    t0 = time.time(); fn = cute.compile(k, *args, Int32(0), stream); print(f"  compile {kw}: {time.time()-t0:.1f}s", flush=True)
    return fn

def cur(): return cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def run_graph_stages(fn):
    st = cur()
    for s in range(S): fn(*args, Int32(s), st)
def run_mega(fn): fn(*args, Int32(0), cur())

def capture(body):
    g = torch.cuda.CUDAGraph()
    body(); torch.cuda.synchronize()
    with torch.cuda.graph(g): body()
    return g

def timeit(g, n=20):
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n

results = {}
fng = build(mode="graph")
A.copy_(A0); run_graph_stages(fng); torch.cuda.synchronize(); ref = A.clone(); print('output changed vs init:', not torch.equal(ref, A0), ' finite:', bool(torch.isfinite(ref.float()).all()))
gg = capture(lambda: run_graph_stages(fng)); ms = timeit(gg); results["graph"] = ms
print(f"graph  (1 kernel/stage, {S} launches): {ms:.3f} ms  = {L*bytes_per_layer/ms/1e6:.0f} GB/s", flush=True)
# existing hand-written CUDA GEMVs (smk_w4r_gemv) for the GEMV stages + the same CuTe norm/mix stages, in one CUDA graph
import ctypes as C
from w4_common import smk
from w4_common import lib
lib()
def run_cuda_gemv():
    st_i = torch.cuda.current_stream().cuda_stream; st = cur()
    for s_, r in enumerate(rows):
        if r[0] == M.OP_GEMV:
            e = smk.Epi(); e.kind = smk.SMK_EPI_SILU if r[7] else smk.SMK_EPI_SEGS
            e.end[0] = e.end[1] = e.end[2] = r[1]; e.out[0] = A.data_ptr() + r[6] * 2
            err = lib().smk_w4r_gemv(Wh.data_ptr() + r[3] * 4, Sh.data_ptr() + r[4] * 4, A.data_ptr() + r[5] * 2, r[1], r[2], T, C.byref(e), st_i)
            assert err == 0, err
        else:
            fng(*args, Int32(s_), st)
A.copy_(A0); run_cuda_gemv(); torch.cuda.synchronize()
print("CUDA-GEMV chain output bit-identical to CuTe-GEMV chain:", torch.equal(A.view(torch.int16), ref.view(torch.int16)),
      " max |diff| =", (A.float() - ref.float()).abs().max().item())
gc_ = capture(run_cuda_gemv); ms = timeit(gc_)
print(f"graph  (hand CUDA GEMVs + CuTe norm/mix): {ms:.3f} ms  = {L*bytes_per_layer/ms/1e6:.0f} GB/s", flush=True)
# graph of CuTe kernels with CONSTEXPR shapes (cute_w4.py, the S2 kernel) for the GEMV stages + the same CuTe norm/mix stage kernels
import cute_w4
scratch_gu = torch.empty(T * 34816, dtype=torch.bfloat16, device=dev)
compiled, call_list = {}, []
for s_, r in enumerate(rows):
    if r[0] != M.OP_GEMV:
        call_list.append(None); continue
    N, K, wo, so, xo, oo, epi = r[1:8]
    Wv = Wh[wo:wo + N * K // 8]; Sv = Sh[so:so + N * (K // 128) // 2]
    xv = A[xo:xo + T * K].view(T, K); ov = (scratch_gu[:T * N] if epi else A[oo:oo + T * N]).view(T, N)
    if (N, K) not in compiled:
        compiled[N, K] = cute.compile(cute_w4.W4Gemv(N, K, T), *cute_w4.wrap(Wv.view(torch.int32), Sv, xv, ov), cur())
    call_list.append((compiled[N, K], cute_w4.wrap(Wv.view(torch.int32), Sv, xv, ov)))
def run_cute_const():
    st = cur()
    for s_, c in enumerate(call_list):
        if c is None: fng(*args, Int32(s_), st)
        else: c[0](*c[1], st)
A.copy_(A0); run_cute_const(); torch.cuda.synchronize()
gcc = capture(run_cute_const); ms = timeit(gcc)
print(f"graph  (CuTe constexpr GEMVs + CuTe norm/mix): {ms:.3f} ms  = {L*bytes_per_layer/ms/1e6:.0f} GB/s", flush=True)
VARS = [("mega  barrier only", dict(mode="mega", prefetch=0)), ("mega  +reg prefetch", dict(mode="mega", prefetch=1)),
        ("mega  +reg +L2pf2", dict(mode="mega", prefetch=1, l2pf=2)), ("mega  +reg +L2pf4", dict(mode="mega", prefetch=1, l2pf=4)), ("mega  +L2pf4 only", dict(mode="mega", prefetch=0, l2pf=4)),
        ("mega  G=82 +reg", dict(mode="mega", prefetch=1, G=82)), ("mega  G=246 +reg", dict(mode="mega", prefetch=1, G=246))]
for name, kw in VARS:
    try:
        fn = build(**kw)
        A.copy_(A0); sync.zero_(); run_mega(fn); torch.cuda.synchronize()
        same = torch.equal(A.view(torch.int16), ref.view(torch.int16))
        gm = capture(lambda: run_mega(fn)); ms = timeit(gm); results[name] = ms
        print(f"{name}: {ms:.3f} ms  = {L*bytes_per_layer/ms/1e6:.0f} GB/s   bit-identical to graph: {same}", flush=True)
    except Exception as e:
        print(name, "FAILED:", repr(e)[:600], flush=True)

# per-stage timeline of the persistent kernel (CTA 0 clock after each wait), grouped by op / shape
fn = build(mode="mega", prefetch=1, timing=True)
A.copy_(A0); sync.zero_(); run_mega(fn); torch.cuda.synchronize()
A.copy_(A0); run_mega(fn); torch.cuda.synchronize()          # second launch: warm
t = sync[16:16 + S].cpu().numpy().astype(np.int64) & 0xFFFFFFFF
d = (np.diff(t) % (1 << 32)).astype(np.float64)                 # SM cycles, stage s start -> stage s+1 start
gm2 = capture(lambda: run_mega(fn)); ms_total = timeit(gm2)
d = d * (ms_total * 1000.0 * (S - 1) / S / d.sum())              # calibrate cycles -> us so that the stages sum to the measured step
tags = []
for r in rows:
    tags.append({M.OP_NORM: "norm", M.OP_MIX: "mix"}.get(r[0]) or {16384: "gemv qkv", 34816: "gemv gu"}.get(r[1]) or ("gemv o" if r[2] == 6144 else "gemv dn"))
from collections import defaultdict
agg = defaultdict(list)
for s_, dd in enumerate(d): agg[tags[s_]].append(dd)
iso = {"gemv qkv": 52.6, "gemv o": 21.3, "gemv gu": 107.4, "gemv dn": 55.1}
print("stage wall time in the chain (start -> next start, incl. barrier skew), mean us   [isolated GEMV us]")
tot = 0
for k, v in agg.items():
    n = len(v); tot += sum(v)
    print(f"  {k:9s} n={n:3d}  {np.mean(v):7.1f}   [{iso.get(k, float('nan')):6.1f}]")
print(f"  sum over stages {tot/1000:.2f} ms")
