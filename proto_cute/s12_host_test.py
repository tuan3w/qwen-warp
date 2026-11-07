import torch, time
import cute_host as H
V, K = 8192, 5120
W = torch.randn(V, K, dtype=torch.bfloat16)
t = H.HostTable(W)
for R in (1, 4, 16, 64, 1000):
    ids = torch.randint(0, V, (R,), dtype=torch.int32, device="cuda")
    out = H.host_gather(ids, t.addr, K); torch.cuda.synchronize()
    ref = W[ids.cpu().long()].cuda()
    print(f"R={R:5d} equal: {torch.equal(out, ref)}")
ids = torch.randint(0, V, (4,), dtype=torch.int32, device="cuda")
for R in (4, 128):
    ids = torch.randint(0, V, (R,), dtype=torch.int32, device="cuda")
    for _ in range(10): H.host_gather(ids, t.addr, K)
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(100): H.host_gather(ids, t.addr, K)
    b.record(); torch.cuda.synchronize(); print(f"R={R}: {a.elapsed_time(b)*10:.0f} us per call (incl. host launch)")
g = torch.cuda.CUDAGraph(); ids = torch.randint(0, V, (4,), dtype=torch.int32, device="cuda")
H.host_gather(ids, t.addr, K)
with torch.cuda.graph(g): o = H.host_gather(ids, t.addr, K)
ids.copy_(torch.tensor([1, 2, 3, 4], dtype=torch.int32)); g.replay(); torch.cuda.synchronize(); print("graph replay correct:", torch.equal(o.cpu(), W[1:5]))
