// swift-mk K3 shared device helpers (clean-room; mma/dequant tricks adapted from experiments/2026-10-30-w4-multirow-verify-and-power/kernels.cuh).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>

namespace smk {

__device__ __forceinline__ float bflo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ float bf2f(uint16_t u) { return __uint_as_float((uint32_t)u << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bf16r(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }  // round-to-bf16 value kept in fp32
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// weights: streaming, never allocate in L1 (L1 holds the activation working set)
__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldx(const uint4* p) { return __ldg(p); }  // activations: L1/L2 resident
__device__ __forceinline__ uint32_t ld16(const uint16_t* p) { return (uint32_t)__ldg(p); }
__device__ __forceinline__ uint32_t ld32(const uint32_t* p) { return __ldg(p); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// natural 8 consecutive k (x0x1 x2x3 x4x5 x6x7) -> mma order (x0x4 x1x5 x2x6 x3x7)
__device__ __forceinline__ uint4 perm_b(uint4 v) {
  uint4 r;
  r.x = __byte_perm(v.x, v.z, 0x5410);
  r.y = __byte_perm(v.x, v.z, 0x7632);
  r.z = __byte_perm(v.y, v.w, 0x5410);
  r.w = __byte_perm(v.y, v.w, 0x7632);
  return r;
}

// one 32-bit word of 8 nibbles -> 4 bf16x2 registers (n_j, n_{j+4}) - 8, exact.  (w >> 4j) & 0x000F000F | 0x43004300 is one lop3 with the constants in registers.
__device__ __forceinline__ uint32_t lop_and_or(uint32_t a, uint32_t m, uint32_t c) {
  uint32_t r;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(r) : "r"(a), "r"(m), "r"(c));  // (a & m) | c
  return r;
}
__device__ __forceinline__ void dq_word(uint32_t w, uint32_t (&p)[4]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    uint32_t q = lop_and_or(j ? (w >> (4 * j)) : w, 0x000F000Fu, 0x43004300u);  // bf16x2 (128+n_j, 128+n_{j+4})
    uint32_t r;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(q), "r"(0x3F803F80u), "r"(0xC308C308u));  // q*1.0 - 136.0 -> exact (n-8)
    p[j] = r;
  }
}

#define SMK_CK(x)                                                                                   \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return e_; } \
  } while (0)

}  // namespace smk
