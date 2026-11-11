import torch, cute_misc as M
torch.manual_seed(0)
N, K = 96, 5120
w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.03
for T in (1, 4, 8):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    y = M._bf16_small_impl(x, w); ref = x.float() @ w.float().T
    print(T, "max rel err", ((y.float() - ref).abs().max() / ref.abs().max()).item())
import time
x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
for f, n in ((lambda: M._bf16_small_impl(x, w), "cute small"), (lambda: torch.nn.functional.linear(x, w), "cublas")):
    for _ in range(20): f()
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(200): f()
    b.record(); torch.cuda.synchronize(); print(f"{n}: {a.elapsed_time(b)*1000/200:.1f} us/call (incl. host)")
