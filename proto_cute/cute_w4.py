"""W4A16 GEMV (T<=8 rows) in CuTe DSL: a line-by-line port of engine/src/gemv.cu k_w4<NW,KS,U,.,RP=1,NT=1> (repacked layout)."""
import time, torch
import cutlass, cutlass.cute as cute
from cutlass import Int32, Uint32, Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
import ptx
import cuda.bindings.driver as cuda

def cur():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

class W4Gemv:
    def __init__(self, N, K, T, NW=8, KS=4, U=2, grid=164):
        assert N % 16 == 0 and K % 128 == 0 and 1 <= T <= 32 and NW % KS == 0
        self.N, self.K, self.T, self.NW, self.KS, self.U, self.grid = N, K, T, NW, KS, U, grid
        self.NT = (T + 7) // 8
        self.nkg = K // 128; assert self.nkg % KS == 0
        self.per = self.nkg // KS
        self.TPC = NW // KS
        self.groups = (N // 16) // self.TPC; assert (N // 16) % self.TPC == 0

    @cute.jit
    def __call__(self, mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor, stream: cuda.CUstream):
        self.kernel(mW, mS, mX, mO).launch(grid=[self.grid, 1, 1], block=[self.NW * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor):
        N, K, T, NW, KS, U, grid = self.N, self.K, self.T, self.NW, self.KS, self.U, self.grid
        nkg, per, TPC, groups, NT = self.nkg, self.per, self.TPC, self.groups, self.NT
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32; warp = tid // 32; g = lane // 4; t = lane % 4
        tin = warp // KS; ks = warp % KS
        cnt = (groups - bid + grid - 1) // grid
        total = cnt * per
        gridsz = grid * TPC
        tile0 = tin + bid * TPC
        xoffs = []
        for n in cutlass.range_constexpr(NT):                  # lanes whose token row is >= T read the last row (never stored)
            xrow = (n * 8 + g) if (n * 8 + g) < T else T - 1
            xoffs.append(xrow * (K // 2) + (ks * 16 + t * 4) * 4)          # u32 units
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((NW, 32, 4 * NT)))

        wa = cute.make_rmem_tensor((U, 4), Uint32)
        wb = cute.make_rmem_tensor((U, 4), Uint32)
        ss = cute.make_rmem_tensor((U,), Uint32)
        acc = cute.make_rmem_tensor((4 * NT,), Float32)
        for c in cutlass.range_constexpr(4 * NT):
            acc[c] = Float32(0.0)

        # i = flat (tile, group) index of this thread's stream
        for j in cutlass.range_constexpr(U):
            if j < total:
                i = Int32(j)
                tl = i // per; kk = i - tl * per
                tile = tile0 + tl * gridsz
                wo = ((tile * nkg + ks + kk * KS) * 64 + lane) * 4
                a0 = ptx.ldw4(ptx.addr(mW, wo)); a1 = ptx.ldw4(ptx.addr(mW, wo + 128))
                for c in cutlass.range_constexpr(4):
                    wa[j, c] = a0[c]; wb[j, c] = a1[c]
                ss[j] = ptx.ld32(ptx.addr(mS, (tile * nkg + ks + kk * KS) * 8 + g))

        for f in cutlass.range(0, total, U):
            for j in cutlass.range_constexpr(U):
                if f + j < total:
                    i = f + j
                    tl = i // per; kk = i - tl * per
                    ca = (wa[j, 0], wa[j, 1], wa[j, 2], wa[j, 3])
                    cb = (wb[j, 0], wb[j, 1], wb[j, 2], wb[j, 3])
                    sv = ss[j]
                    if f + j + U < total:                    # issue the load U groups ahead into the slot just consumed
                        n = i + U
                        tl2 = n // per; kk2 = n - tl2 * per
                        tile2 = tile0 + tl2 * gridsz
                        wo = ((tile2 * nkg + ks + kk2 * KS) * 64 + lane) * 4
                        a0 = ptx.ldw4(ptx.addr(mW, wo)); a1 = ptx.ldw4(ptx.addr(mW, wo + 128))
                        for c in cutlass.range_constexpr(4):
                            wa[j, c] = a0[c]; wb[j, c] = a1[c]
                        ss[j] = ptx.ld32(ptx.addr(mS, (tile2 * nkg + ks + kk2 * KS) * 8 + g))
                    xv = []
                    for n in cutlass.range_constexpr(NT):
                        for u in cutlass.range_constexpr(4):
                            x4 = ptx.ldx4(ptx.addr(mX, xoffs[n] + kk * (KS * 16 * 4) + u * 4))
                            for c in cutlass.range_constexpr(4):
                                xv.append(x4[c])
                    pp = ptx.group_mma(ca, cb, xv, NT)
                    fsa = ptx.bits2f(sv << 16); fsb = ptx.bits2f(sv & Uint32(0xFFFF0000))
                    for n in cutlass.range_constexpr(NT):
                        acc[4 * n + 0] = fsa * pp[4 * n + 0] + acc[4 * n + 0]; acc[4 * n + 1] = fsa * pp[4 * n + 1] + acc[4 * n + 1]
                        acc[4 * n + 2] = fsb * pp[4 * n + 2] + acc[4 * n + 2]; acc[4 * n + 3] = fsb * pp[4 * n + 3] + acc[4 * n + 3]
                    if kk == per - 1:                         # tile finished
                        row = (tile0 + tl * gridsz) * 16
                        if cutlass.const_expr(KS > 1):
                            if ks > 0:
                                for c in cutlass.range_constexpr(4 * NT):
                                    red[warp, lane, c] = acc[c]
                            cute.arch.sync_threads()
                            if ks == 0:
                                for q in cutlass.range_constexpr(1, KS):
                                    for c in cutlass.range_constexpr(4 * NT):
                                        acc[c] = acc[c] + red[warp + q, lane, c]
                            cute.arch.sync_threads()
                        if ks == 0:
                            r0 = row + g
                            for n in cutlass.range_constexpr(NT):
                                tk0 = 8 * n + 2 * t
                                if tk0 < T:
                                    mO[tk0, r0] = acc[4 * n].to(BFloat16); mO[tk0, r0 + 8] = acc[4 * n + 2].to(BFloat16)
                                if tk0 + 1 < T:
                                    mO[tk0 + 1, r0] = acc[4 * n + 1].to(BFloat16); mO[tk0 + 1, r0 + 8] = acc[4 * n + 3].to(BFloat16)
                        for c in cutlass.range_constexpr(4 * NT):
                            acc[c] = Float32(0.0)

def wrap(w, s, x, out):
    return (from_dlpack(w.view(torch.int32).view(-1), assumed_align=16), from_dlpack(s.view(torch.int32).view(-1), assumed_align=16),
            from_dlpack(x.view(torch.int32).view(-1), assumed_align=16), from_dlpack(out, assumed_align=16))

_cache = {}
def get(N, K, T, w, s, x, out, **cfg):
    key = (N, K, T, tuple(sorted(cfg.items())))
    if key not in _cache:
        k = W4Gemv(N, K, T, **cfg)
        t0 = time.time(); args = wrap(w, s, x, out)
        _cache[key] = cute.compile(k, *args, cur())
        print(f"  [compile N={N} K={K} T={T} {cfg}: {time.time()-t0:.1f}s]", flush=True)
    return _cache[key]
