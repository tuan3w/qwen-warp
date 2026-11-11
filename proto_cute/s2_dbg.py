import sys, torch
from w4_common import *
import cute_w4
N, K, T = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
cfg = eval(sys.argv[4]) if len(sys.argv) > 4 else {}
ws = copies(N, K, 1)
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
o_ref = torch.empty(T, N, device="cuda", dtype=torch.bfloat16); o = torch.zeros_like(o_ref)
baseline_call(*ws[0], x, o_ref, N, K, T)
fn = cute_w4.get(N, K, T, ws[0][0], ws[0][1], x, o, **cfg)
fn(*cute_w4.wrap(ws[0][0], ws[0][1], x, o), cute_w4.cur()); torch.cuda.synchronize()
bad = (o.float() - o_ref.float()).abs() > 0
print(cfg, "bad elems", bad.sum().item(), "of", bad.numel())
if bad.any():
    cols = bad.any(0).nonzero().flatten(); rows = bad.any(1).nonzero().flatten()
    print("bad token rows", rows.tolist(), "bad cols n", cols.numel(), "first cols", cols[:20].tolist(), "col%16:", sorted(set((cols % 16).tolist())))
