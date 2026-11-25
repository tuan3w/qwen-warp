#pragma once
#include "dense_common.cuh"

// Single-row fused dequant GEMV over a GGUF Q6_K matrix, the lm_head of a Q4_0 file (llama.cpp's own choice for the output
// matrix): logits[N] (fp32) = W[N][K] x[K], K a multiple of 256. The weights are repacked at load time (warpc/cuda/pack.py) into planes
// so that a lane's 8 columns of a 256-column block are read with 4 small aligned loads; a row of KB = K / 256 blocks is
//   ql  KB x 32 lanes x u32: lane l of block j holds the low 4 bits of the codes of columns 256j + 8l + m in the nibble at bit 4m
//   qh  KB x 32 lanes x u16: the top 2 bits of the same codes at bit 2m
//   sc  KB x 16 int8: the scale of columns 16s .. 16s + 15 (lanes 2s and 2s + 1)
//   d   KB f16
// and takes rs = 210 KB bytes rounded up to 16. The weight of a column is bf16((d * sc) * (code - 32)), the GGUF source's
// decoder (the `gguf` package's, d * sc exact in fp32, one fp32 product) rounded to bf16 as the dense path widens it: lane l
// accumulates columns 8k .. 8k + 7, k = l + 32 j, with fmaf in column order and the 32 partials meet in k_gemv's butterfly,
// so the logits are bit-identical to k_gemv (mode 2) over the bf16-widened matrix. One row per warp; U blocks are loaded
// before any is used.
struct Q6GemvP {
  const uint8_t* w;  // N rows of rs bytes
  const bf16* x;  // [K]
  float* out;  // [N]
  int N, K;
  size_t rs;
};

__device__ __forceinline__ uint4 q6_ld16(const void* a) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a));
  return v;
}
__device__ __forceinline__ uint32_t q6_ld32(const void* a) {
  uint32_t v;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(a));
  return v;
}
__device__ __forceinline__ uint32_t q6_ld16u(const void* a) {
  unsigned short v;
  asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(v) : "l"(a));
  return v;
}
__device__ __forceinline__ uint32_t q6_ld8(const void* a) {
  unsigned short v;
  asm("ld.global.nc.L1::no_allocate.u8 %0, [%1];" : "=h"(v) : "l"(a));
  return v;
}

template <int U>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_q6_gemv(const __grid_constant__ Q6GemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * GEMV_WARPS + warp;
  const int KB = p.K / 256;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x);
  const uint8_t* wr = p.w + (size_t)min(row, p.N - 1) * p.rs;
  float acc = 0.f;
  for (int j0 = 0; j0 < KB; j0 += U) {
    uint32_t ql[U], qh[U], sc[U];
    float d[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = j0 + u;
      if (j < KB) {
        ql[u] = q6_ld32(wr + 128 * (size_t)j + 4 * lane);
        qh[u] = q6_ld16u(wr + 128 * (size_t)KB + 64 * (size_t)j + 2 * lane);
        sc[u] = q6_ld8(wr + 192 * (size_t)KB + 16 * (size_t)j + (lane >> 1));
        d[u] = __half2float(__ushort_as_half((unsigned short)q6_ld16u(wr + 208 * (size_t)KB + 2 * (size_t)j)));
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int j = j0 + u;
      if (j < KB) {
        float xf[8];
        unpack8(xw[32 * (size_t)j + lane], xf);
        const float ds = d[u] * (float)(int)(int8_t)sc[u];
#pragma unroll
        for (int m = 0; m < 8; m += 2) {
          float wv[2];
#pragma unroll
          for (int k = 0; k < 2; k++) {
            const uint32_t code = ((ql[u] >> (4 * (m + k))) & 15u) | (((qh[u] >> (2 * (m + k))) & 3u) << 4);
            wv[k] = ds * (__uint_as_float(0x4B000000u | code) - 8388640.f);  // 2^23 + code - (2^23 + 32): code - 32, exact
          }
          const float2 r = __bfloat1622float2(__floats2bfloat162_rn(wv[0], wv[1]));
          acc = fmaf(r.x, xf[m], acc);
          acc = fmaf(r.y, xf[m + 1], acc);
        }
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0 && row < p.N) p.out[row] = acc;
}
