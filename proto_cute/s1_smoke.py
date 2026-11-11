import time, torch
import cutlass, cutlass.cute as cute
from cutlass import Int32, Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.arch import inline_ptx

# S1: (a) compile latency, (b) raw bf16 mma.sync.m16n8k16 through inline PTX, (c) device-scope atomics + smid
@cute.kernel
def k_mma(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, cnt: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    lane = tid % 32; g = lane // 4; t = lane % 4
    # A frag 16x16 bf16 row-major (4 x b32 regs/thread), B frag 16x8 col-major (2 regs)
    def ld2(ten, r, col):  # pack two consecutive bf16 into u32
        lo = ten[r, col].to(Float32); hi = ten[r, col + 1].to(Float32)
        return inline_ptx("cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};", write_only_types=[Int32], read_only_args=[lo, hi])
    a0 = ld2(a, g, 2 * t); a1 = ld2(a, g + 8, 2 * t); a2 = ld2(a, g, 2 * t + 8); a3 = ld2(a, g + 8, 2 * t + 8)
    # b is [N=8][K=16] (B col-major == n-major rows of k)
    b0 = ld2(b, g, 2 * t); b1 = ld2(b, g, 2 * t + 8)
    d0, d1, d2, d3 = inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, {0f00000000, 0f00000000, 0f00000000, 0f00000000};",
        write_only_types=[Float32] * 4, read_only_args=[a0, a1, a2, a3, b0, b1])
    c[g, 2 * t] = d0; c[g, 2 * t + 1] = d1; c[g + 8, 2 * t] = d2; c[g + 8, 2 * t + 1] = d3
    if tid == 0:
        cute.arch.atomic_add(cnt.iterator, Int32(1), sem="relaxed", scope="gpu")

@cute.jit
def run(a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, cnt: cute.Tensor):
    k_mma(a, b, c, cnt).launch(grid=[4, 1, 1], block=[32, 1, 1])

if __name__ == "__main__":
    torch.manual_seed(0)
    a = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    c = torch.zeros(16, 8, device="cuda", dtype=torch.float32)
    cnt = torch.zeros(1, device="cuda", dtype=torch.int32)
    ta, tb, tc, tn = [from_dlpack(x, assumed_align=16) for x in (a, b, c, cnt)]
    t0 = time.time(); fn = cute.compile(run, ta, tb, tc, tn); print(f"compile {time.time()-t0:.2f}s")
    t0 = time.time(); fn(ta, tb, tc, tn); torch.cuda.synchronize(); print(f"first run {time.time()-t0:.3f}s")
    ref = a.float() @ b.float().T
    print("max err", (c - ref).abs().max().item(), "atomic count", cnt.item())
