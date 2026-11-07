import sys, torch
from w4_common import *
import cute_w4
N, K, T = (int(a) for a in sys.argv[1:4]) if len(sys.argv) > 3 else (5120, 17408, 1)
ws = copies(N, K, 2)
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
o_ref = torch.empty(T, N, device="cuda", dtype=torch.bfloat16); o = torch.zeros_like(o_ref)
baseline_call(*ws[0], x, o_ref, N, K, T)
fn = cute_w4.get(N, K, T, ws[0][0], ws[0][1], x, o)
fn(*cute_w4.wrap(ws[0][0], ws[0][1], x, o), cute_w4.cur()); torch.cuda.synchronize()
print("max |cute - cuda| =", (o.float() - o_ref.float()).abs().max().item(), " (ref max", o_ref.float().abs().max().item(), ")")
nb = N * K // 2 + N * K // 128 * 2
fns = []
for w, s in ws:
    a = cute_w4.wrap(w, s, x, o); fns.append(lambda a=a: fn(*a, cute_w4.cur()))
us = bench(fns); print(f"cute    {us:7.1f} us {nb/us/1e3:6.1f} GB/s")
us = bench([lambda w=w, s=s: baseline_call(w, s, x, o_ref, N, K, T) for w, s in ws]); print(f"cuda    {us:7.1f} us {nb/us/1e3:6.1f} GB/s")
