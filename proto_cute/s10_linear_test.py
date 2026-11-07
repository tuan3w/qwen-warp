import torch
from w4_common import make_w4, dequant_nat, pack
import vllm_mp_kernel as VK
VK._register_op()
N, K = 2048, 6144
w, s = make_w4(N, K, seed=2); Wref = dequant_nat(w, s).to(torch.bfloat16)
wr, sr = w.clone(), s.clone(); pack.repack_w4(wr, sr, rows=128)
for T in (1, 8, 16, 17, 100, 600):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    y = torch.ops.vllm.cute_w4a16_linear(x, wr, sr); ref = x.float() @ Wref.float().T
    print(f"T={T:4d} rel err {((y.float()-ref).abs().max()/ref.abs().max()).item():.1e}")
import time
for T in (64, 512):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    for _ in range(3): torch.ops.vllm.cute_w4a16_linear(x, wr, sr)
    torch.cuda.synchronize(); a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
    for _ in range(20): torch.ops.vllm.cute_w4a16_linear(x, wr, sr)
    b.record(); torch.cuda.synchronize(); t = a.elapsed_time(b) / 20
    a.record()
    for _ in range(20): torch.mm(x, Wref.t())
    b.record(); torch.cuda.synchronize(); print(f"T={T}: dequant+mm {t*1000:.0f} us  vs bf16 mm on resident weights {a.elapsed_time(b)/20*1000:.0f} us")
