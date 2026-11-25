#pragma once
#include "dense_common.cuh"

// Q4_0 GEMV y[T][N] = x[T][K] W[N][K]^T, W in GGUF Q4_0: the fused dequant variant of k_gemv: the weights are read as Q4_0 (18 bytes per 32 columns: an f16 scale d, then 16 packed
// nibbles, low nibble of byte j = col 32b+j, high nibble = col 32b+16+j) and dequantised to fp32 in registers, w = d * (q - 8),
// while the dot product is accumulated. There is no bf16 weight intermediate, and the dequantised weights stay fp32 for the T
// rows' dot products (a superset of the bf16 path's numerics, the intended one). Everything else (the K-loop, the 32-lane
// butterfly, the fp32 accumulation over K, the epilogues) is k_gemv's, so the reduction order depends on K only. The 18-byte
// blocks are only 2-byte aligned (18b mod 8 cycles with b), so the block is read as nine 16-bit __ldg loads. Modes as in
// k_gemv: 0 bf16 outputs split by row range (up to 3 segments); 1 gate/up pair (w = gate, w2 = up), out = bf16(silu(bf16(gate)))
// * bf16(up); 3 residual add, out[0] is the residual stream h [T][N] (bf16) and h = bf16(h + bf16(y)) in place (each element
// belongs to one row, so no cross-block ordering is needed).
struct Q4GemvP {
  const uint8_t* w;  // Q4_0 [N][K], N rows of K/32 blocks of 18 bytes
  const uint8_t* w2;  // mode 1 only: the up rows, same layout
  const bf16* x;  // [T][K]
  int N, K;
  int end[3];
  void* out[3];
};

// one 18-byte Q4_0 block -> 32 fp32 weights; the f16 scale d and the q - 8 are exact, the one rounding is the final multiply,
// bit-identical to the GGUF source's Q4_0 decoder (and ggml's)
__device__ __forceinline__ void dequant_q4_block(const uint8_t* b, float* qf) {
  const uint16_t s = __ldg(reinterpret_cast<const uint16_t*>(b));
  const float d = __half2float(*reinterpret_cast<const __half*>(&s));
  uint8_t qs[16];
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const uint16_t u = __ldg(reinterpret_cast<const uint16_t*>(b + 2 + 2 * j));
    qs[2 * j] = (uint8_t)u;
    qs[2 * j + 1] = (uint8_t)(u >> 8);
  }
#pragma unroll
  for (int j = 0; j < 16; j++) {
    qf[j] = d * ((float)(qs[j] & 0x0Fu) - 8.f);
    qf[j + 16] = d * ((float)(qs[j] >> 4) - 8.f);
  }
}

template <int T, int RPW, int MODE, int U>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_q4_gemv(const __grid_constant__ Q4GemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * RPW;
  const int KB = p.K / 32;  // Q4_0 blocks per row
  const int K8 = p.K / 8;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x);
  const uint8_t* wr[RPW];
  const uint8_t* wr2[RPW];
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = min(row0 + i, p.N - 1);
    wr[i] = p.w + (size_t)r * KB * 18;
    if constexpr (MODE == 1) wr2[i] = p.w2 + (size_t)r * KB * 18;
  }
  float acc[RPW][T], acc2[RPW][T];
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[i][t] = acc2[i][t] = 0.f;

#pragma unroll U
  for (int b = lane; b < KB; b += 32) {
    // the x values of this 32-column block, in 8-wide chunks (kept short in registers so T = 8 fits)
    float xf[4][T][8];
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
      for (int t = 0; t < T; t++) unpack8(xw[(size_t)t * K8 + 4 * b + c], xf[c][t]);
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      float qw[32];
      dequant_q4_block(wr[i] + 18 * b, qw);
#pragma unroll
      for (int c = 0; c < 4; c++)
#pragma unroll
        for (int t = 0; t < T; t++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc[i][t] = fmaf(qw[8 * c + e], xf[c][t][e], acc[i][t]);
      if (MODE == 1) {
        dequant_q4_block(wr2[i] + 18 * b, qw);
#pragma unroll
        for (int c = 0; c < 4; c++)
#pragma unroll
          for (int t = 0; t < T; t++)
#pragma unroll
            for (int e = 0; e < 8; e++) acc2[i][t] = fmaf(qw[8 * c + e], xf[c][t][e], acc2[i][t]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) {
      acc[i][t] = warp_sum(acc[i][t]);
      if (MODE == 1) acc2[i][t] = warp_sum(acc2[i][t]);
    }
  if (lane >= T) return;
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = row0 + i;
    if (r >= p.N) continue;
    float g = 0.f, u = 0.f;
#pragma unroll
    for (int t = 0; t < T; t++)
      if (lane == t) {
        g = acc[i][t];
        u = acc2[i][t];
      }
    if (MODE == 0) {
      const int s = r < p.end[0] ? 0 : (r < p.end[1] ? 1 : 2);
      const int start = s == 0 ? 0 : p.end[s - 1];
      const int width = p.end[s] - start;
      reinterpret_cast<bf16*>(p.out[s])[(size_t)lane * width + (r - start)] = tb(g);
    } else if (MODE == 3) {
      bf16* hr = reinterpret_cast<bf16*>(p.out[0]) + (size_t)lane * p.N + r;
      *hr = tb(bf(*hr) + rb(g));
    } else {
      const float a = rb(g);
      const float sl = rb(a / (1.f + __expf(-a)));
      reinterpret_cast<bf16*>(p.out[0])[(size_t)lane * p.N + r] = tb(sl * rb(u));
    }
  }
}

// The launch of one instantiation, defined in dense_q4_gemv_inst.cu (so that no translation unit but the instantiation units compiles a k_q4_gemv).
template <int T, int RPW, int MODE>
void warpc_inst_q4_gemv(const Q4GemvP& p, int grid, cudaStream_t st);
