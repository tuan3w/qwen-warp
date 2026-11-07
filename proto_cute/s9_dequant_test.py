import torch, time
import cute_dequant as D
from w4_common import make_w4, dequant_nat, pack
N, K = 2048, 6144
w, s = make_w4(N, K, seed=1)
ref = dequant_nat(w, s)                                     # (nib-8)*scale in fp32 on the natural layout
ref_bf = ref.to(torch.bfloat16)
wr, sr = w.clone(), s.clone(); pack.repack_w4(wr, sr, rows=128)
rows = D.chunk_rows(N, K, max_bytes=8 << 20); print("chunk rows", rows)
scratch = torch.empty(rows * K, dtype=torch.bfloat16, device="cuda")
out = torch.empty(N, K, dtype=torch.bfloat16, device="cuda")
for r0 in range(0, N, rows):
    D.dequant_chunk(wr, sr, scratch, K, rows, r0); out[r0:r0 + rows] = scratch[: rows * K].view(rows, K)
torch.cuda.synchronize()
print("bit-equal to natural dequant:", torch.equal(out.view(torch.int16), ref_bf.view(torch.int16)), " max abs diff", (out.float() - ref_bf.float()).abs().max().item())
a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); a.record()
for _ in range(20):
    for r0 in range(0, N, rows): D.dequant_chunk(wr, sr, scratch, K, rows, r0)
b.record(); torch.cuda.synchronize(); t = a.elapsed_time(b) / 20
print(f"dequant {N}x{K}: {t*1000:.0f} us/matrix  -> {(N*K*2 + N*K//2)/t/1e6:.0f} GB/s (write+read)")
