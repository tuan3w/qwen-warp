"""Shared harness for the W4 GEMV spike: weights in the engine's repacked layout, torch reference, ctypes baseline, timing."""
import ctypes as C, sys, torch
sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp/engine")
import smk, pack

def make_w4(N, K, seed=0, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    w = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev, generator=g)
    s = (torch.rand(N, K // 128, device=dev, generator=g) * 0.02 + 0.005).to(torch.bfloat16)
    return w, s

def dequant_nat(w, s):
    """natural layout -> fp32 [N,K]: nibble e of word j = column 8j+e, (n-8)*scale per 128-group"""
    N, K8 = w.shape
    sh = (4 * torch.arange(8, device=w.device)).view(1, 1, 8)
    nib = ((w.unsqueeze(-1) >> sh.to(torch.int32)) & 0xF).to(torch.float32) - 8.0   # [N,K8,8]
    W = nib.reshape(N, K8 * 8)
    return W * s.float().repeat_interleave(128, dim=1)

def ref_gemv(x, w, s, chunk=128):
    return torch.cat([x.float() @ dequant_nat(w[i:i+chunk], s[i:i+chunk]).T for i in range(0, w.shape[0], chunk)], dim=1)  # [T,N] fp32

_lib = None
def lib():
    global _lib
    if _lib is None:
        _lib = smk.load()
        vp, ci = C.c_void_p, C.c_int
        _lib.smk_w4r_gemv.argtypes = [vp, vp, vp, ci, ci, ci, C.POINTER(smk.Epi), vp]
        _lib.smk_w4r_gemv.restype = ci
        _lib.smk_w4r_gemv_cfg.argtypes = [vp, vp, vp, ci, ci, ci, C.POINTER(smk.Epi), ci, vp]
        _lib.smk_w4r_gemv_cfg.restype = ci
    return _lib

def baseline_call(w, s, x, out, N, K, T, cfg=None):
    e = smk.Epi(); e.kind = smk.SMK_EPI_SEGS; e.end[0] = e.end[1] = e.end[2] = N
    e.out[0] = out.data_ptr()
    st = torch.cuda.current_stream().cuda_stream
    if cfg is None:
        r = lib().smk_w4r_gemv(w.data_ptr(), s.data_ptr(), x.data_ptr(), N, K, T, C.byref(e), st)
    else:
        r = lib().smk_w4r_gemv_cfg(w.data_ptr(), s.data_ptr(), x.data_ptr(), N, K, T, C.byref(e), cfg, st)
    if r: raise RuntimeError(f"smk_w4r_gemv err {r}")

def bench(fns, iters=200, warm=20):
    """fns: list of callables (rotating weight copies); returns us/call"""
    for i in range(warm): fns[i % len(fns)]()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters): fns[i % len(fns)]()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters

def copies(N, K, n):
    ws = []
    for i in range(n):
        w, s = make_w4(N, K, seed=i); ws.append(pack.repack_w4(w, s, rows=128))
    return ws
