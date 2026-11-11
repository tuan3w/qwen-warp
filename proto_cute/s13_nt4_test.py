import torch, time
from w4_common import *
import cute_w4
N, K = 5120, 17408
ws = copies(N, K, 2)
nb = N * K // 2 + N * K // 128 * 2
for T in (1, 4, 8, 16, 24, 32):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16); o = torch.zeros(T, N, device="cuda", dtype=torch.bfloat16)
    fn = cute_w4.get(N, K, T, ws[0][0], ws[0][1], x, o)
    args = [cute_w4.wrap(w, s, x, o) for w, s in ws]
    for a in args: fn(*a, cute_w4.cur())
    torch.cuda.synchronize()
    us = bench([lambda a=a: fn(*a, cute_w4.cur()) for a in args])
    print(f"T={T:2d} NT={(T+7)//8}  {us:6.1f} us  {nb/us/1e3:6.1f} GB/s")
