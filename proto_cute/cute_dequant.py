"""Repacked W4 (engine layout) -> bf16 [rows, K] for the prefill path (dequant a row chunk, then cuBLAS).  One thread = one 16 B chunk = 32 columns of one row."""
import torch, cutlass, cutlass.cute as cute
from cutlass import Int32, Uint32, Float32
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import ptx

class Dequant:
    def __init__(self, K, rows):
        assert K % 128 == 0 and rows % 16 == 0
        self.K, self.rows = K, rows; self.nkg = K // 128; self.units = (rows // 16) * self.nkg

    @cute.jit
    def __call__(self, mW: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor, tile_off: Int32, stream: cuda.CUstream):
        self.kernel(mW, mS, mO, tile_off).launch(grid=[(self.units * 64 + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mW: cute.Tensor, mS: cute.Tensor, mO: cute.Tensor, tile_off: Int32):
        K, nkg = self.K, self.nkg
        tid, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
        idx = bid * 256 + tid
        unit = idx // 64
        if unit < self.units:
            l64 = idx % 64; half = l64 // 32; lane = l64 % 32; g = lane // 4; t = lane % 4
            tile = unit // nkg; kg = unit - tile * nkg                          # tile within the chunk
            gtile = tile_off + tile                                              # tile within the matrix
            wv = ptx.ldw4(ptx.addr(mW, ((gtile * nkg + kg) * 64 + half * 32 + lane) * 4))
            sw = ptx.ld32(ptx.addr(mS, (gtile * nkg + kg) * 8 + g))
            sc = ptx.bits2f((sw >> (16 * half)) << 16)
            row = tile * 16 + half * 8 + g
            base = row * K + kg * 128 + t * 32
            for u in cutlass.range_constexpr(4):
                w = wv[u]
                p = []
                for j in cutlass.range_constexpr(4):
                    lo = ((w >> (4 * j)) & Uint32(15)).to(Int32).to(Float32) - 8.0       # column 2j
                    hi = ((w >> (4 * j + 16)) & Uint32(15)).to(Int32).to(Float32) - 8.0  # column 2j+1 (nibble j+4)
                    p.append(ptx.pack_bf16x2(lo * sc, hi * sc))
                ptx.stg4(ptx.addr(mO, base + u * 8), p[0], p[1], p[2], p[3])

_fns = {}
def chunk_rows(N, K, max_bytes=64 << 20):
    best = 16
    for r in range(16, N + 1, 16):
        if N % r == 0 and r * K * 2 <= max_bytes: best = r
    return best

def dequant_chunk(w, s, scratch, K, rows, r0):
    """dequantise rows [r0, r0 + rows) of the repacked (w, s) into scratch[: rows * K] (bf16)"""
    key = (K, rows)
    args = (from_dlpack(w.view(torch.int32).view(-1), assumed_align=16), from_dlpack(s.view(torch.int32).view(-1), assumed_align=16), from_dlpack(scratch[: rows * K], assumed_align=16))
    if key not in _fns:
        _fns[key] = cute.compile(Dequant(K, rows), *args, Int32(0), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    _fns[key](*args, Int32(r0 // 16), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
