import sys, torch, numpy as np
import cute_head as H, pack, cute_host
torch.manual_seed(0)
V, K, nv = 8192, 5120, 1024
Wb = torch.randn(V, K, device="cuda", dtype=torch.bfloat16) * 0.02
w, s = pack.quant_head_i4(Wb, rows=256); pack.repack_w4(w, s, rows=128)
ids = torch.sort(torch.randperm(V, device="cuda")[:nv]).values
ws = torch.empty(nv, K // 8, dtype=torch.int32, device="cuda"); ss = torch.empty(nv, K // 128, dtype=torch.bfloat16, device="cuda")
a, b = pack.quant_head_i4(Wb[ids], rows=256); ws.copy_(a); ss.copy_(b); pack.repack_w4(ws, ss, rows=128)
hostW = cute_host.HostTable(Wb)
for T in (1, 4, 8, 20):
    x = torch.randn(T, K, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, Wb)                        # exact bf16 head
    full = H._head_impl(x, w, s, Wb, ws, ss, ids, False, V, 0, V)
    same = (full.argmax(1) == ref.argmax(1)).float().mean().item()
    cand_ok = torch.isfinite(full).sum(1).tolist()
    exact_vals = torch.where(torch.isfinite(full), (full.float() - ref.float()).abs(), torch.zeros_like(full.float())).max().item()
    full_h = H._head_impl(x, w, s, Wb[:0], ws, ss, ids, False, V, hostW.addr, V)
    print('   host rescoring == device rescoring:', torch.equal(full_h, full))
    dr = H._head_impl(x, w, s, Wb, ws, ss, ids, True, V, 0, V)
    sub_ref = ref[:, ids]; dsame = (dr.argmax(1) == ids[sub_ref.argmax(1)]).float().mean().item()
    print(f"T={T}  target argmax==bf16 head: {same:.2f}  finite per row {cand_ok}  max|exact-ref| {exact_vals:.3g} | draft argmax==subset bf16 argmax: {dsame:.2f}  finite per row {torch.isfinite(dr).sum(1).tolist()}")
