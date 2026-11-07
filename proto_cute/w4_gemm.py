"""W4A16 prefill GEMM on the engine's repacked weight layout.

gemm(x, w, s) -> out
  x   bf16 [T, K]  contiguous activations (T = 16 .. 2048 rows: prefill chunks)
  w   int32 [N, K/8]   repacked 4-bit weights (one 1 KB unit per (16-row tile, 128-k group): [32 lanes x row g, 16 B chunk t][32 lanes x row g+8])
  s   bf16 [N, K/128]  repacked scales (u32 per (tile, group, g) = bf16(row g) | bf16(row g+8) << 16)
  out bf16 [T, N]

Fused kernel (Marlin style): the weight units are already in mma A-fragment order, so each warp streams its 16-row tiles straight from global
memory into registers (one group ahead), dequantises them in registers to bf16 (bf16(round((nibble - 8) * scale)), the same rounding as the old
dequant + cuBLAS path) and feeds mma.sync.m16n8k16 with the weights as the A operand (rows = output features) and the activations as B (n = tokens).
The activation tile [BT tokens x 128 k] of each k group is staged in shared memory with a cp.async pipeline (bank-conflict-free swizzle).
Block = WM warps along the output rows (32 rows each) x BT = 8*NT tokens; blocks are small (<= 48 KB smem) so that two of them are resident per SM
and never wait for each other at a barrier (measured: this occupancy, not the tile shape, was the biggest lever).
"""
import torch
import cutlass, cutlass.cute as cute
from cutlass import Int32, Uint32, Float32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import ptx

MT = 2           # 16-row weight tiles per warp


class W4Gemm:
    def __init__(self, N, K, WM, NT, STAGES):
        assert N % 16 == 0 and K % 128 == 0 and (4 * NT) % WM == 0
        self.N, self.K, self.WM, self.NT, self.STAGES = N, K, WM, NT, STAGES
        self.nkg = K // 128
        self.BT = 8 * NT                 # tokens per block
        self.threads = 32 * WM
        self.ntiles = N // 16
        self.stage_bytes = self.BT * 256
        self.smem_bytes = STAGES * self.stage_bytes

    @cute.jit
    def __call__(self, mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor, T: Int32, stream: cuda.CUstream):
        gx = (T + (self.BT - 1)) // self.BT                              # token blocks vary fastest: blocks sharing a weight tile run together
        gy = (self.ntiles + (MT * self.WM - 1)) // (MT * self.WM)
        self.kernel(mW, mS, mX, mO, T).launch(grid=[gx, gy, 1], block=[self.threads, 1, 1], stream=stream)

    # ---- helpers (straight-line trace-time code)
    def _load_w(self, mW, mS, wa, wb, ss, kg, tile_base, lane, g):
        nkg = self.nkg
        for m in range(MT):
            tile = cutlass.min(tile_base + m, self.ntiles - 1)           # rows past N read the last tile (never stored)
            wo = ((tile * nkg + kg) * 64 + lane) * 4
            a0 = ptx.ldw4(ptx.addr(mW, wo)); a1 = ptx.ldw4(ptx.addr(mW, wo + 128))
            for c in range(4):
                wa[m, c] = a0[c]; wb[m, c] = a1[c]
            ss[m] = ptx.ld32(ptx.addr(mS, (tile * nkg + kg) * 8 + g))

    def _issue_x(self, mX, sbase, stage, kg, tb, T, tid):
        K, BT, WM, NT = self.K, self.BT, self.WM, self.NT
        c = tid % 16
        r0 = tid // 16
        rows_per_pass = self.threads // 16
        swc = c ^ (((c >> 3) & 1) << 1)
        for j in range(4 * NT // WM):
            tok = r0 + j * rows_per_pass
            xrow = cutlass.min(tb * BT + tok, T - 1)                     # token rows past T read the last row (never stored)
            src = xrow * (K // 2) + kg * 64 + c * 4
            phys = swc ^ (tok & 1)
            dst = sbase + stage * self.stage_bytes + (tok * 16 + phys) * 16
            ptx.cp_async16(dst, ptx.addr(mX, src))

    @cute.kernel
    def kernel(self, mW: cute.Tensor, mS: cute.Tensor, mX: cute.Tensor, mO: cute.Tensor, T: Int32):
        N, K, WM, NT, STAGES = self.N, self.K, self.WM, self.NT, self.STAGES
        BT, nkg = self.BT, self.nkg
        tid, _, _ = cute.arch.thread_idx()
        tb, nb, _ = cute.arch.block_idx()
        lane = tid % 32; warp = tid // 32; g = lane // 4; t = lane % 4
        tile_base = nb * (MT * WM) + warp * MT

        smem = cutlass.utils.SmemAllocator()
        sm = smem.allocate_tensor(Uint32, cute.make_layout(self.smem_bytes // 4), 128)
        sbase = sm.iterator.toint()

        # per-thread x fragment addresses: row (token g) of n-tile n at +n*8*256 B; 16 B chunk (4t+u) with the bank swizzle
        xu_off = []
        for u in cutlass.range_constexpr(4):
            cch = t * 4 + u
            xu_off.append((cch ^ ((t >> 1) << 1) ^ (g & 1)) * 16 + g * 256)

        wa = cute.make_rmem_tensor((MT, 4), Uint32)
        wb = cute.make_rmem_tensor((MT, 4), Uint32)
        ss = cute.make_rmem_tensor((MT,), Uint32)
        nwa = cute.make_rmem_tensor((MT, 4), Uint32)
        nwb = cute.make_rmem_tensor((MT, 4), Uint32)
        nss = cute.make_rmem_tensor((MT,), Uint32)
        acc = cute.make_rmem_tensor((MT * NT * 4,), Float32)
        for c in cutlass.range_constexpr(MT * NT * 4):
            acc[c] = Float32(0.0)

        # prologue: x stages 0..STAGES-2, weights of group 0
        for st in cutlass.range_constexpr(STAGES - 1):
            if st < nkg:
                self._issue_x(mX, sbase, Int32(st), Int32(st), tb, T, tid)
            cute.arch.cp_async_commit_group()
        self._load_w(mW, mS, wa, wb, ss, Int32(0), tile_base, lane, g)

        for it in cutlass.range(nkg):
            cute.arch.cp_async_wait_group(STAGES - 2)
            cute.arch.sync_threads()
            if it + (STAGES - 1) < nkg:
                nit = it + (STAGES - 1)
                self._issue_x(mX, sbase, nit % STAGES, nit, tb, T, tid)
            cute.arch.cp_async_commit_group()
            if it + 1 < nkg:
                self._load_w(mW, mS, nwa, nwb, nss, it + 1, tile_base, lane, g)

            stg = sbase + (it % STAGES) * self.stage_bytes
            sa = []; sb = []
            for m in cutlass.range_constexpr(MT):
                sa.append(ptx.prmt(ss[m], 0x1010)); sb.append(ptx.prmt(ss[m], 0x3232))
            for u in cutlass.range_constexpr(4):
                xv = []
                for n in cutlass.range_constexpr(NT):
                    x4 = ptx.lds4(stg + (xu_off[u] + n * 8 * 256))
                    for c in cutlass.range_constexpr(4):
                        xv.append(x4[c])
                for m in cutlass.range_constexpr(MT):
                    a_in = [acc[(m * NT) * 4 + c] for c in range(4 * NT)]
                    a_out = ptx.wmma_u(wa[m, u], wb[m, u], sa[m], sb[m], xv, a_in, NT)
                    for c in cutlass.range_constexpr(4 * NT):
                        acc[(m * NT) * 4 + c] = a_out[c]
            for m in cutlass.range_constexpr(MT):
                for c in cutlass.range_constexpr(4):
                    wa[m, c] = nwa[m, c]; wb[m, c] = nwb[m, c]
                ss[m] = nss[m]

        # ---- epilogue: C^T fragment (row g / g+8, tokens 2t / 2t+1) -> out[token, row]
        for m in cutlass.range_constexpr(MT):
            tile = tile_base + m
            if tile < self.ntiles:
                r0 = tile * 16 + g
                for n in cutlass.range_constexpr(NT):
                    tk0 = tb * BT + n * 8 + 2 * t
                    i0 = (m * NT + n) * 4
                    if tk0 < T:
                        ptx.st_bf16(ptx.addr(mO, tk0 * N + r0), acc[i0])
                        ptx.st_bf16(ptx.addr(mO, tk0 * N + r0 + 8), acc[i0 + 2])
                    if tk0 + 1 < T:
                        ptx.st_bf16(ptx.addr(mO, (tk0 + 1) * N + r0), acc[i0 + 1])
                        ptx.st_bf16(ptx.addr(mO, (tk0 + 1) * N + r0 + 8), acc[i0 + 3])


def _stream():
    return cuda.CUstream(torch.cuda.current_stream().cuda_stream)

def pick_config(T):
    """(WM, NT, STAGES): 256 rows x 32 tokens per block (measured best for T = 96..512; smaller T just wastes compute on a memory-bound shape)"""
    return 8, 4, 3

_fns = {}
def gemm(x, w, s):
    T, K = x.shape
    N = w.shape[0]
    cfg = pick_config(T)
    out = torch.empty(T, N, dtype=torch.bfloat16, device=x.device)
    args = (from_dlpack(w.view(torch.int32).view(-1), assumed_align=16), from_dlpack(s.view(torch.int32).view(-1), assumed_align=16),
            from_dlpack(x.view(torch.int32).view(-1), assumed_align=16), from_dlpack(out.view(-1), assumed_align=16))
    key = (N, K) + cfg
    if key not in _fns:
        _fns[key] = cute.compile(W4Gemm(N, K, *cfg), *args, Int32(T), _stream())
    _fns[key](*args, Int32(T), _stream())
    return out
