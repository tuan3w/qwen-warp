import torch, sys
from w4_common import *
import cute_w4
N, K = 2048, 6144
w, s = make_w4(N, K, seed=3); Wref = dequant_nat(w, s)
wr, sr = w.clone(), s.clone(); pack.repack_w4(wr, sr, rows=128)
for T in (1, 5, 8, 9, 12, 16):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16); o = torch.zeros(T, N, device="cuda", dtype=torch.bfloat16)
    fn = cute_w4.get(N, K, T, wr, sr, x, o); fn(*cute_w4.wrap(wr, sr, x, o), cute_w4.cur()); torch.cuda.synchronize()
    ref = x.float() @ Wref.T; err = ((o.float() - ref).abs().max() / ref.abs().max()).item()
    # row-count invariance: row t of a T-row launch == the same row computed alone (T=1)
    o1 = torch.zeros(1, N, device="cuda", dtype=torch.bfloat16); f1 = cute_w4.get(N, K, 1, wr, sr, x[:1], o1); f1(*cute_w4.wrap(wr, sr, x[:1].contiguous(), o1), cute_w4.cur())
    print(f"T={T:2d} NT={(T+7)//8} rel err {err:.1e}  row0 identical to a 1-row launch: {torch.equal(o[0], o1[0])}")
