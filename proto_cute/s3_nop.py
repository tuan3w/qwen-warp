"""Cost of one stage boundary: S empty stages, one kernel per stage (graph) vs persistent cooperative kernel with arrive/wait barriers."""
import sys, time, torch, cutlass, cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import mega as M
S = 448; T = 1
tbl = torch.zeros(S * M.COLS, dtype=torch.int32, device="cuda"); tbl.view(S, M.COLS)[:, 0] = 9
Wh = torch.zeros(1024, dtype=torch.int32, device="cuda"); Sh = Wh.clone(); A = torch.zeros(1024, dtype=torch.bfloat16, device="cuda"); sync = torch.zeros(4, dtype=torch.int32, device="cuda")
mk = lambda t: from_dlpack(t, assumed_align=16)
args = (mk(tbl), mk(Wh), mk(Sh), mk(A), mk(A.view(torch.int32)), mk(sync))
cur = lambda: cuda.CUstream(torch.cuda.current_stream().cuda_stream)
def timeit(body, n=50):
    g = torch.cuda.CUDAGraph(); body(); torch.cuda.synchronize()
    with torch.cuda.graph(g): body()
    for _ in range(3): g.replay()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(n): g.replay()
    b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) / n
for G in (82, 164, 246):
    kg = M.Mega(T, S, G=G, mode="graph"); fg = cute.compile(kg, *args, Int32(0), cur())
    km = M.Mega(T, S, G=G, mode="mega"); fm = cute.compile(km, *args, Int32(0), cur())
    def bg():
        st = cur()
        for s in range(S): fg(*args, Int32(s), st)
    sync.zero_(); ms_g = timeit(bg); sync.zero_(); torch.cuda.synchronize(); ms_m = timeit(lambda: fm(*args, Int32(0), cur()))
    print(f"G={G:3d}  graph: {ms_g*1000/S:5.2f} us/stage   persistent: {ms_m*1000/S:5.2f} us/stage (barrier)", flush=True)
