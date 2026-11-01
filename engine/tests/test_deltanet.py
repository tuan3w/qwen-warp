#!/usr/bin/env python
"""K1 DeltaNet decode kernel vs the swift-mk oracle (real layer 0/1/2 weights + real activations from the dumps).

Run (needs the GPU lease):  projects/llm-lab/bin/lab gpu acquire; ~/llm-lab/venv/bin/python engine/tests/test_deltanet.py [--lib build/x.so]
Reference = fp64 torch re-implementation of the oracle's `_deltanet` (reference.py:219-246) fed with the SAME bf16-rounded inputs the kernel
sees, so errors measure the kernel (fp32 math + one bf16 output rounding), not input quantisation.  The oracle's own taps (fp32 inputs) are
compared with a looser bound as a semantic cross-check.
"""
import argparse, ctypes, os, sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))  # projects/swift-mk
sys.path.insert(0, os.path.join(ROOT, "oracle"))
from weights import Checkpoint  # noqa: E402

NV, DV, DK, CD, SL = 48, 128, 128, 10240, 9  # SL = ring slots (K11: tmax + 1 = 9 covers T <= 8; main() --slots 5 re-runs the T <= 4 cases on the K1 ring)
DUMPS = os.path.expanduser("~/llm-lab/swift-ref/fp32/{}.pt")


# ------------------------------------------------------------------------------------------------ library wrapper
class DN:
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.swiftmk_deltanet_ws_bytes.restype = ctypes.c_size_t
        L.swiftmk_deltanet_step.restype = ctypes.c_int
        L.swiftmk_deltanet_step.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 11 + [ctypes.c_int] + [ctypes.c_void_p] * 3
        self.ws = torch.zeros(L.swiftmk_deltanet_ws_bytes(), dtype=torch.uint8, device="cuda")

    def step(self, T, qkv, z, a, b, w, S, conv, cur, y, stream=None):
        st = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        e = self.lib.swiftmk_deltanet_step(
            T, qkv.data_ptr(), z.data_ptr(), a.data_ptr(), b.data_ptr(), w["conv_w"].data_ptr(), w["A_log"].data_ptr(),
            w["dt_bias"].data_ptr(), w["norm_w"].data_ptr(), S.data_ptr(), conv.data_ptr(), cur.data_ptr(), S.shape[0], y.data_ptr(),
            self.ws.data_ptr(), st)
        assert e == 0, f"launch error {e}"


def load_w(ck, layer):
    p = f"model.language_model.layers.{layer}.linear_attn."
    g = lambda n: ck.get(p + n).to(torch.bfloat16).contiguous()
    return {"conv_w": g("conv1d.weight").reshape(CD, 4), "A_log": g("A_log"), "dt_bias": g("dt_bias"), "norm_w": g("norm.weight")}


# ------------------------------------------------------------------------------------------------ fp64 reference
def ref_steps(w, qkv, z, a, b, S, conv, nrows):
    """fp64. qkv [T,10240], z [T,6144], a,b [T,48] (bf16-valued), S [48,V,K] fp64, conv [10240,3] fp64.
    Returns y [T,6144] (fp64), list of post S [48,V,K], list of post conv [10240,3]."""
    d = torch.float64
    cw, A, dt, nw = (w[k].double().cpu() for k in ("conv_w", "A_log", "dt_bias", "norm_w"))
    S, conv = S.clone(), conv.clone()
    ys, Ss, Cs = [], [], []
    for t in range(nrows):
        x = qkv[t].double()
        X = torch.cat([conv, x[:, None]], 1)  # [10240,4]
        c = F.silu((X * cw).sum(1))
        conv = X[:, 1:].clone()
        q, k, v = c[:2048].reshape(16, 128), c[2048:4096].reshape(16, 128), c[4096:].reshape(48, 128)
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK ** -0.5
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q, k = q.repeat_interleave(3, 0), k.repeat_interleave(3, 0)
        beta = torch.sigmoid(b[t].double())
        g = -A.exp() * F.softplus(a[t].double() + dt)
        S = S * g.exp()[:, None, None]
        kv = (S * k[:, None, :]).sum(2)  # [48,V]
        dlt = (v - kv) * beta[:, None]
        S = S + dlt[:, :, None] * k[:, None, :]
        o = (S * q[:, None, :]).sum(2)  # [48,V]
        n = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + 1e-6)
        yy = (nw * n) * F.silu(z[t].double().reshape(48, 128))
        ys.append(yy.reshape(-1)); Ss.append(S); Cs.append(conv)
    return torch.stack(ys), Ss, Cs


# ------------------------------------------------------------------------------------------------ helpers
def relerr(x, ref):
    x, ref = x.double().cpu(), ref.double().cpu()
    return ((x - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item()


FAILS = []


def check(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:58s} max|err|/max|ref| = {err:.3e} (tol {tol:.1e})")
    if not ok:
        FAILS.append(name)


class Case:
    """Device state for one layer, ring layout + helpers."""

    def __init__(self, dn, w, S0_hf, conv0):
        self.dn, self.w = dn, {k: v.cuda() for k, v in w.items()}
        self.S = torch.zeros(SL, NV, DV, DK, device="cuda")
        self.conv = torch.zeros(SL, CD, 3, device="cuda", dtype=torch.bfloat16)
        self.S0 = S0_hf.transpose(1, 2).double().contiguous()  # HF [H,K,V] -> vLLM [H,V,K]
        self.conv0 = conv0.to(torch.bfloat16).double()
        self.cur = torch.zeros(1, dtype=torch.int32, device="cuda")

    def reset(self, cur0):
        self.S.zero_(); self.conv.zero_()
        self.S[cur0] = self.S0.float().cuda(); self.conv[cur0] = self.conv0.to(torch.bfloat16).cuda()
        self.cur.fill_(cur0)
        self.cur0 = cur0

    def run(self, T, qkv, z, a, b):
        y = torch.empty(T, 6144, device="cuda", dtype=torch.bfloat16)
        bf = lambda x: x.to(torch.bfloat16).cuda().contiguous()
        qkv, z, a, b = bf(qkv), bf(z), bf(a), bf(b)
        self.dn.step(T, qkv, z, a, b, self.w, self.S, self.conv, self.cur, y)
        torch.cuda.synchronize()
        assert not self.dn.ws[:256].any(), "counters not reset"
        return y.float().cpu()


def get_inputs(taps, layer, events):
    f = lambda k: torch.cat([taps[e][layer][k] for e in events]).to(torch.bfloat16).float()
    return f("qkv_proj"), f("z"), f("a"), f("b")


def test_real(dn, ck, dump, layer, T, cur0, label):
    z = torch.load(DUMPS.format(dump))
    taps = z["taps"]
    w = load_w(ck, layer)
    C = Case(dn, w, taps[0][layer]["S_out"], taps[0][layer]["conv_state_out"])
    ev = list(range(1, 1 + T))
    qkv, zz, a, b = get_inputs(taps, layer, ev)
    yref, Ss, Cs = ref_steps(w, qkv, zz, a, b, C.S0, C.conv0, T)
    C.reset(cur0)
    pre = C.S[cur0].clone(), C.conv[cur0].clone()
    y = C.run(T, qkv, zz, a, b)
    print(f"[{label}] layer {layer} dump {dump} T={T} cur0={cur0}")
    check("y vs fp64 ref (all rows)", relerr(y, yref), 1.5e-2 if False else 6e-3)
    for t in range(T):
        slot = (cur0 + 1 + t) % SL
        check(f"state slot {slot} (after row {t}) vs fp64 ref", relerr(C.S[slot], Ss[t]), 2e-5)
        check(f"conv  slot {slot} (after row {t}) vs ref", relerr(C.conv[slot], Cs[t]), 0.0)
    check("slot cur untouched (state)", (C.S[cur0] - pre[0]).abs().max().item(), 0.0)
    check("slot cur untouched (conv)", (C.conv[cur0].float() - pre[1].float()).abs().max().item(), 0.0)
    # untouched slots
    for s in range(SL):
        if s != cur0 and s not in [(cur0 + 1 + t) % SL for t in range(T)]:
            check(f"unused slot {s} still zero", C.S[s].abs().max().item(), 0.0)
    # oracle taps (fp32 inputs, loose): gated_norm_out and S_out for e=1,2 (layer 0)
    yor = torch.cat([taps[e][layer]["gated_norm_out"] for e in ev])
    check("y vs ORACLE taps gated_norm_out (fp32 inputs, loose)", relerr(y, yor), 3e-2)
    for t, e in enumerate(ev):
        if "S_out" in taps[e][layer]:
            slot = (cur0 + 1 + t) % SL
            check(f"state row {t} vs ORACLE S_out (loose)", relerr(C.S[slot], taps[e][layer]["S_out"].transpose(1, 2)), 2e-2)
    return C, (qkv, zz, a, b), y


def test_equiv(dn, ck, dump, layer, TT=4):
    """T-row verify == T sequential T=1 steps (bitwise expected: identical arithmetic)."""
    z = torch.load(DUMPS.format(dump)); taps = z["taps"]
    w = load_w(ck, layer)
    C = Case(dn, w, taps[0][layer]["S_out"], taps[0][layer]["conv_state_out"])
    qkv, zz, a, b = get_inputs(taps, layer, list(range(1, TT + 1)))
    C.reset(3)
    y4 = C.run(TT, qkv, zz, a, b); S4 = C.S.clone(); C4 = C.conv.clone()
    C.reset(3)
    ys = []
    for t in range(TT):
        ys.append(C.run(1, qkv[t:t + 1], zz[t:t + 1], a[t:t + 1], b[t:t + 1]))
        C.cur.fill_((3 + t + 1) % SL)  # accepted 1 row
    y1 = torch.cat(ys)
    print(f"[equiv] layer {layer} dump {dump}: T={TT} vs {TT}x T=1 (cur0=3, wraps)")
    check("y equal", (y4 - y1).abs().max().item(), 0.0)
    check(f"all {SL} state slots equal", (S4 - C.S).abs().max().item(), 0.0)
    check(f"all {SL} conv slots equal", (C4.float() - C.conv.float()).abs().max().item(), 0.0)


def test_chain(dn, ck, dump, layer):
    """Multi-step speculative pattern: T=4 steps with partial acceptance (a = 2,3,1,4,...) over 8 real events, wraparound included,
    compared with the fp64 reference run sequentially on the accepted prefix tokens."""
    z = torch.load(DUMPS.format(dump)); taps = z["taps"]
    w = load_w(ck, layer)
    C = Case(dn, w, taps[0][layer]["S_out"], taps[0][layer]["conv_state_out"])
    qkv, zz, a, b = get_inputs(taps, layer, list(range(1, 9)))
    C.reset(4)
    if SL != 5:  # 8 real events only: start the chain over after each commit by cycling the event window
        qkv, zz, a, b = (torch.cat([x, x, x]) for x in (qkv, zz, a, b))
    pos, errs, S_ref, conv_ref = 0, [], C.S0.clone(), C.conv0.clone()
    print(f"[chain] layer {layer} dump {dump}: T=4/3/2 steps with partial acceptance, start cur=4")
    for T, acc in [(4, 2), (3, 3), (4, 1), (2, 2)] if SL == 5 else [(8, 3), (5, 5), (7, 1), (6, 2), (8, 8), (4, 2)]:
        if pos + T > qkv.shape[0]:
            break
        sl = slice(pos, pos + T)
        y = C.run(T, qkv[sl], zz[sl], a[sl], b[sl])
        # reference continues from the reference state after `pos` committed tokens
        yref, Ss, Cs = ref_steps(w, qkv[sl], zz[sl], a[sl], b[sl], S_ref, conv_ref, T)
        errs.append(relerr(y, yref))
        cur = int(C.cur.item())
        for t in range(T):
            errs.append(relerr(C.S[(cur + 1 + t) % SL], Ss[t]))
        S_ref, conv_ref = Ss[acc - 1], Cs[acc - 1]
        C.cur.fill_((cur + acc) % SL)
        pos += acc
        check(f"T={T} accept {acc}: cur->{int(C.cur.item())}; max err y/states", max(errs), 6e-3)
        errs = []
    cur = int(C.cur.item())
    check("committed state == ref after all accepted", relerr(C.S[cur], S_ref), 2e-5)
    check("committed conv == ref", relerr(C.conv[cur], conv_ref), 0.0)


def test_extreme(dn):
    """Random data stress: large q/k magnitude, a such that a+dt_bias > 20 (softplus identity branch), strongly negative / positive b."""
    torch.manual_seed(1)
    ck = Checkpoint()
    w = load_w(ck, 0)
    for case in ["scale3", "softplus>20", "b extreme"]:
        T = SL - 1
        qkv = torch.randn(T, CD) * (3.0 if case == "scale3" else 1.0)
        zz = torch.randn(T, 6144) * 2
        a = torch.randn(T, 48) * 2 + (20.0 if case == "softplus>20" else 0.0)
        b = torch.randn(T, 48) * (30.0 if case == "b extreme" else 2.0)
        qkv, zz, a, b = (x.to(torch.bfloat16).float() for x in (qkv, zz, a, b))
        S0 = torch.randn(48, 128, 128) * 0.3
        conv0 = torch.randn(CD, 3)
        C = Case(dn, w, S0, conv0)
        # S0 above is given as [H,K,V] to Case (transposed internally) -> build ref identically
        yref, Ss, Cs = ref_steps(w, qkv, zz, a, b, C.S0, C.conv0, T)
        C.reset(2)
        y = C.run(T, qkv, zz, a, b)
        print(f"[random:{case}]")
        check("y", relerr(y, yref), 6e-3)
        check("last state", relerr(C.S[(2 + T) % SL], Ss[-1]), 2e-5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default=os.path.join(HERE, "..", "build", "libswiftmk_deltanet.so"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--slots", type=int, default=9)
    args = ap.parse_args()
    SL = args.slots
    dn = DN(args.lib)
    ck = Checkpoint()
    for layer in ([0] if args.quick else [0, 1, 2]):
        for T in range(1, SL):
            test_real(dn, ck, "code", layer, T, cur0=(layer + T) % SL if not args.quick else 0, label="real")
        test_real(dn, ck, "prose", layer, SL - 1, cur0=SL - 1, label="real wrap")
        test_equiv(dn, ck, "math", layer, SL - 1)
        test_chain(dn, ck, "code", layer)
    test_extreme(dn)
    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
