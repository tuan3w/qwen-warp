"""S3/S4: the same stage code run (a) one stage per launch (CUDA graph) or (b) as one persistent cooperative kernel with split arrive/wait barriers.
Stage ops: NORM, GEMV (W4A16, runtime N/K, bit-identical math to cute_w4.py), MIX (elementwise).  Table-driven: a stage is a row of int32."""
import time, torch
import cutlass, cutlass.cute as cute
from cutlass import Int32, Uint32, Float32, BFloat16
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import ptx

OP_NORM, OP_GEMV, OP_MIX = 0, 1, 2
COLS = 12   # op, N, K, w_off, s_off, x_off, o_off, epi, a0, a1, a2, a3
H, QKV, Y6, INTER = 5120, 16384, 6144, 17408

class Mega:
    def __init__(self, T, S, G=164, NW=8, KS=4, U=2, mode="mega", prefetch=0, l2pf=0, timing=False):
        self.T, self.S, self.G, self.NW, self.KS, self.U, self.mode, self.prefetch, self.l2pf, self.timing = T, S, G, NW, KS, U, mode, prefetch, l2pf, timing
        self.TPC = NW // KS

    # ---------------------------------------------------------------- host entry points
    @cute.jit
    def __call__(self, tbl: cute.Tensor, mW: cute.Tensor, mS: cute.Tensor, mA: cute.Tensor, mAi: cute.Tensor, sync: cute.Tensor,
                 stage: Int32, stream: cuda.CUstream):
        self.kernel(tbl, mW, mS, mA, mAi, sync, stage).launch(grid=[self.G, 1, 1], block=[self.NW * 32, 1, 1], stream=stream,
                                                              cooperative=(self.mode == "mega"))

    # ---------------------------------------------------------------- GEMV pieces (plain python: straight-line, inlined at trace time)
    def _gp(self, tbl, row, bid, tid):
        """per-thread GEMV params of the stage in table row `row`"""
        KS, TPC, G = self.KS, self.TPC, self.G
        N = tbl[row + 1]; K = tbl[row + 2]
        nkg = K // 128; per = nkg // KS
        groups = (N // 16) // TPC
        cnt = (groups - bid + G - 1) // G
        lane = tid % 32; warp = tid // 32
        return dict(N=N, K=K, nkg=nkg, per=per, total=cnt * per, tile0=warp // KS + bid * TPC, ks=warp % KS, lane=lane, g=lane // 4, t=lane % 4,
                    woff=tbl[row + 3], soff=tbl[row + 4], xoff=tbl[row + 5], ooff=tbl[row + 6], epi=tbl[row + 7])

    def _issue(self, mW, mS, wa, wb, ss, j, i, p):
        tl = i // p["per"]; kk = i - tl * p["per"]
        tile = p["tile0"] + tl * (self.G * self.TPC)
        gi = tile * p["nkg"] + p["ks"] + kk * self.KS
        wo = p["woff"] + (gi * 64 + p["lane"]) * 4
        a0 = ptx.ldw4(ptx.addr(mW, wo)); a1 = ptx.ldw4(ptx.addr(mW, wo + 128))
        for c in range(4):
            wa[j, c] = a0[c]; wb[j, c] = a1[c]
        ss[j] = ptx.ld32(ptx.addr(mS, p["soff"] + gi * 8 + p["g"]))

    # ---------------------------------------------------------------- the kernel
    @cute.kernel
    def kernel(self, tbl: cute.Tensor, mW: cute.Tensor, mS: cute.Tensor, mA: cute.Tensor, mAi: cute.Tensor, sync: cute.Tensor, stage: Int32):
        T, S, G, NW, KS, U, TPC = self.T, self.S, self.G, self.NW, self.KS, self.U, self.TPC
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        red = smem.allocate_tensor(Float32, cute.make_layout((NW, 32, 4)))
        nred = smem.allocate_tensor(Float32, cute.make_layout((NW,)))
        wa = cute.make_rmem_tensor((U, 4), Uint32)
        wb = cute.make_rmem_tensor((U, 4), Uint32)
        ss = cute.make_rmem_tensor((U,), Uint32)
        acc = cute.make_rmem_tensor((4,), Float32)
        base = Int32(0)
        s_begin = stage; s_end = stage + 1
        if cutlass.const_expr(self.mode == "mega"):
            base = ptx.ld_acq(ptx.addr(sync, 1)) * (S * G)
            s_begin = Int32(0); s_end = Int32(S)

        for s in cutlass.range(s_begin, s_end):
            row = s * COLS
            op = tbl[row + 0]
            if cutlass.const_expr(self.mode == "mega"):
                # ---- wait for stage s-1 (all CTAs arrived); stage 0 has no producer
                if s > 0:
                    if tid == 0:
                        target = base + s * G
                        v = ptx.ld_acq(ptx.addr(sync, 0))
                        while v - target < 0:
                            v = ptx.ld_acq(ptx.addr(sync, 0))
                    cute.arch.sync_threads()
                if cutlass.const_expr(self.timing):
                    if bid == 0:
                        if tid == 0:
                            ptx.st32(ptx.addr(sync, 16 + s), ptx.gtimer32())
                if s == 1:
                    if bid == 0:
                        if tid == 0:
                            ptx.st32(ptx.addr(sync, 1), ptx.ld_acq(ptx.addr(sync, 1)) + 1)   # epoch++ (every CTA has read it before arriving at stage 0)

            # ------------------------------------------------------------ GEMV
            if op == OP_GEMV:
                p = self._gp(tbl, row, bid, tid)
                total = p["total"]
                if cutlass.const_expr(self.prefetch == 0 or self.mode != "mega"):
                    for j in cutlass.range_constexpr(U):
                        if j < total:
                            self._issue(mW, mS, wa, wb, ss, j, Int32(j), p)
                # (prefetch == 1: the previous iteration already issued these loads into wa/wb/ss before waiting)
                for c in cutlass.range_constexpr(4):
                    acc[c] = Float32(0.0)
                xrow = p["g"] if p["g"] < T else T - 1
                xoff = (p["xoff"] // 2) + xrow * (p["K"] // 2) + (p["ks"] * 16 + p["t"] * 4) * 4
                for f in cutlass.range(0, total, U):
                    for j in cutlass.range_constexpr(U):
                        if f + j < total:
                            i = f + j
                            tl = i // p["per"]; kk = i - tl * p["per"]
                            ca = (wa[j, 0], wa[j, 1], wa[j, 2], wa[j, 3])
                            cb = (wb[j, 0], wb[j, 1], wb[j, 2], wb[j, 3])
                            sv = ss[j]
                            if f + j + U < total:
                                self._issue(mW, mS, wa, wb, ss, j, i + U, p)
                            xv = []
                            for u in cutlass.range_constexpr(4):
                                x4 = ptx.ldg4(ptx.addr(mAi, xoff + kk * (KS * 16 * 4) + u * 4))
                                for c in cutlass.range_constexpr(4):
                                    xv.append(x4[c])
                            p0, p1, p2, p3 = ptx.group_mma(ca, cb, xv)
                            fsa = ptx.bits2f(sv << 16); fsb = ptx.bits2f(sv & Uint32(0xFFFF0000))
                            acc[0] = fsa * p0 + acc[0]; acc[1] = fsa * p1 + acc[1]
                            acc[2] = fsb * p2 + acc[2]; acc[3] = fsb * p3 + acc[3]
                            if kk == p["per"] - 1:
                                tile = p["tile0"] + tl * (G * TPC)
                                if cutlass.const_expr(KS > 1):
                                    if p["ks"] > 0:
                                        for c in cutlass.range_constexpr(4):
                                            red[tid // 32, tid % 32, c] = acc[c]
                                    cute.arch.sync_threads()
                                    if p["ks"] == 0:
                                        for q in cutlass.range_constexpr(1, KS):
                                            for c in cutlass.range_constexpr(4):
                                                acc[c] = acc[c] + red[tid // 32 + q, tid % 32, c]
                                    cute.arch.sync_threads()
                                if p["ks"] == 0:
                                    tk0 = 2 * p["t"]
                                    if p["epi"] == 0:
                                        r0 = tile * 16 + p["g"]
                                        N = p["N"]
                                        if tk0 < T:
                                            mA[p["ooff"] + tk0 * N + r0] = acc[0].to(BFloat16); mA[p["ooff"] + tk0 * N + r0 + 8] = acc[2].to(BFloat16)
                                        if tk0 + 1 < T:
                                            mA[p["ooff"] + (tk0 + 1) * N + r0] = acc[1].to(BFloat16); mA[p["ooff"] + (tk0 + 1) * N + r0 + 8] = acc[3].to(BFloat16)
                                    else:
                                        ch = tile * 8 + p["g"]; W2 = p["N"] // 2
                                        if tk0 < T:
                                            mA[p["ooff"] + tk0 * W2 + ch] = silu_mul(acc[0], acc[2])
                                        if tk0 + 1 < T:
                                            mA[p["ooff"] + (tk0 + 1) * W2 + ch] = silu_mul(acc[1], acc[3])
                                for c in cutlass.range_constexpr(4):
                                    acc[c] = Float32(0.0)

            # ------------------------------------------------------------ NORM (CTA t handles token t): h += add; xn = (1+w) * h * rsqrt(mean(h^2)+eps)
            if op == OP_NORM:
                if bid < T:
                    t = bid
                    ho = tbl[row + 8] + t * H; ao = tbl[row + 9] + t * H; xo = tbl[row + 10] + t * H; wo = tbl[row + 11]
                    has_add = tbl[row + 7]
                    sq = Float32(0.0)
                    hv = cute.make_rmem_tensor((20,), Float32)
                    for i in cutlass.range_constexpr(20):
                        k = tid + 256 * i
                        v = mA[ho + k].to(Float32)
                        if has_add != 0:
                            v = v + mA[ao + k].to(Float32)
                        v = v.to(BFloat16).to(Float32)
                        hv[i] = v
                        mA[ho + k] = v.to(BFloat16)
                        sq = sq + v * v
                    sq = cute.arch.warp_reduction_sum(sq)
                    if tid % 32 == 0:
                        nred[tid // 32] = sq
                    cute.arch.sync_threads()
                    tot = Float32(0.0)
                    for w_ in cutlass.range_constexpr(NW):
                        tot = tot + nred[w_]
                    inv = cute.math.rsqrt(tot / H + 1e-6)
                    for i in cutlass.range_constexpr(20):
                        k = tid + 256 * i
                        mA[xo + k] = (hv[i] * inv * (mA[wo + k].to(Float32) + 1.0)).to(BFloat16)

            # ------------------------------------------------------------ MIX: y6 = qkv[:6144]*0.5 + qkv[6144:12288]*0.25  (stand-in for the DeltaNet / attention core)
            if op == OP_MIX:
                nel = T * Y6
                for e in cutlass.range(bid * 256 + tid, nel, G * 256):
                    t = e // Y6; i = e - t * Y6
                    a = mA[tbl[row + 8] + t * QKV + i].to(Float32); b = mA[tbl[row + 8] + t * QKV + Y6 + i].to(Float32)
                    mA[tbl[row + 10] + e] = (a * 0.5 + b * 0.25).to(BFloat16)

            if cutlass.const_expr(self.mode == "mega"):
                # ---- arrive (release) then, for the next GEMV stage, issue its first weight loads before waiting
                cute.arch.sync_threads()
                if tid == 0:
                    ptx.atom_add_rel(ptx.addr(sync, 0), Int32(1))
                if cutlass.const_expr(self.l2pf > 0):
                    if s + 1 < S:
                        nrow = (s + 1) * COLS
                        if tbl[nrow + 0] == OP_GEMV:
                            pn = self._gp(tbl, nrow, bid, tid)
                            for j in cutlass.range_constexpr(self.l2pf):
                                if j < pn["total"]:
                                    tl = Int32(j) // pn["per"]; kk = Int32(j) - tl * pn["per"]
                                    tile = pn["tile0"] + tl * (G * TPC)
                                    gi = tile * pn["nkg"] + pn["ks"] + kk * KS
                                    base_w = pn["woff"] + gi * 256            # u32 units: the 1 KB unit of this (tile, group)
                                    if pn["lane"] < 8:
                                        ptx.prefetch_l2(ptx.addr(mW, base_w + pn["lane"] * 32))
                if cutlass.const_expr(self.prefetch == 1):
                    if s + 1 < S:
                        nrow = (s + 1) * COLS
                        if tbl[nrow + 0] == OP_GEMV:
                            pn = self._gp(tbl, nrow, bid, tid)
                            for j in cutlass.range_constexpr(U):
                                if j < pn["total"]:
                                    self._issue(mW, mS, wa, wb, ss, j, Int32(j), pn)

def silu_mul(g, u):
    gt = g.to(BFloat16).to(Float32); up = u.to(BFloat16).to(Float32)
    sg = (gt / (1.0 + cute.math.exp(-gt))).to(BFloat16).to(Float32)
    return (sg * up).to(BFloat16)
