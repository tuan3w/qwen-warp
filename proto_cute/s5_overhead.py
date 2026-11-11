"""Host-side cost of calling a compiled CuTe DSL kernel from Python (eager path) and whether a cold process pays the compile again."""
import time, torch, sys
from w4_common import *
import cute_w4
N, K, T = 5120, 6144, 1
ws = copies(N, K, 1); w, s = ws[0]
x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16); o = torch.empty(T, N, device="cuda", dtype=torch.bfloat16)
t0 = time.time(); fn = cute_w4.get(N, K, T, w, s, x, o); print(f"get() cold process: {time.time()-t0:.2f}s")
a = cute_w4.wrap(w, s, x, o); st = cute_w4.cur()
for _ in range(50): fn(*a, st)
torch.cuda.synchronize(); n = 2000
t0 = time.perf_counter()
for _ in range(n): fn(*a, st)
cpu = (time.perf_counter() - t0) / n * 1e6; torch.cuda.synchronize()
print(f"CuTe compiled fn, python call: {cpu:.1f} us CPU / launch (kernel itself ~21 us)")
t0 = time.perf_counter()
for _ in range(n): baseline_call(w, s, x, o, N, K, T)
print(f"ctypes -> libswiftmk launch:   {(time.perf_counter()-t0)/n*1e6:.1f} us CPU / launch"); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n): torch.add(x, x)
print(f"torch.add launch:              {(time.perf_counter()-t0)/n*1e6:.1f} us CPU / launch"); torch.cuda.synchronize()
