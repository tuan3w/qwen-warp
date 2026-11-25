#pragma once
#include "dense_common.cuh"

// Inkling-2 GEMVs: y[T][N] = x[T][K] W[N][K]^T with k_gemv's arithmetic (dense_gemv.cuh): one warp per RPW rows, 16-byte loads
// (k = lane + 32 j), fp32 fma in the order e = 0..7 per load, the 32 lane partials summed by the fixed butterfly (the order
// depends on K only), so a row's result is bit-identical to warpc_gemv's whatever the epilogue.
//   MODE 0  up to IG_SEGS segments of consecutive rows, each with its own input x[s] [T][K] and bf16 output out[s] [T][width];
//           the segments' ends are multiples of RPW (one warp reads one segment's input)
//   MODE 1  the attention output projection: a = bf16(acc); a short convolution over the rows with the conv state cst
//           bf16 [N][W - 1] (oldest first, updated in place: row n of the state belongs to the one warp that owns channel n),
//           taps cw bf16 [N][W] (tap W - 1 = the current row), c = bf16((conv + a) / div), out f32 [T][N] = f32(c)
//           (+ hres[t][n] bf16 when hres != NULL)
#define IG_SEGS 8
#define IG_TAPS 4  // conv width of MODE 1

struct InkGemvP {
  const bf16* w;
  const bf16* x[IG_SEGS];
  void* out[IG_SEGS];
  int end[IG_SEGS];  // cumulative row ends; unused entries = N
  int nseg, N, K;
  const bf16* cw;
  bf16* cst;
  const bf16* hres;
  float* outf;
  float div;
};

template <int T, int RPW, int MODE, int U>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_ig(const __grid_constant__ InkGemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * RPW;
  if (row0 >= p.N) return;
  const int K8 = p.K / 8;
  int seg = 0;
  if (MODE == 0)
    while (row0 >= p.end[seg]) seg++;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x[seg]);
  const uint4* wr[RPW];
#pragma unroll
  for (int i = 0; i < RPW; i++) wr[i] = reinterpret_cast<const uint4*>(p.w + (size_t)min(row0 + i, p.N - 1) * p.K);
  float acc[RPW][T];
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[i][t] = 0.f;
#pragma unroll U
  for (int k = lane; k < K8; k += 32) {
    uint4 wv[RPW];
#pragma unroll
    for (int i = 0; i < RPW; i++) wv[i] = __ldg(wr[i] + k);
    float xf[T][8];
#pragma unroll
    for (int t = 0; t < T; t++) unpack8(xw[(size_t)t * K8 + k], xf[t]);
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      float wf[8];
      unpack8(wv[i], wf);
#pragma unroll
      for (int t = 0; t < T; t++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[i][t] = fmaf(wf[e], xf[t][e], acc[i][t]);
    }
  }
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[i][t] = warp_sum(acc[i][t]);
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = row0 + i;
    if (r >= p.N) continue;
    if (MODE == 1) {
      // u = [state (3) | a (T)] oldest first; tap j weighs u[t + j]. Every lane holds every acc (butterfly), reads the state
      // before the __syncwarp and lanes 0..2 write it after it
      float u[IG_TAPS - 1 + T];
#pragma unroll
      for (int j = 0; j < IG_TAPS - 1; j++) u[j] = bf(p.cst[(size_t)r * (IG_TAPS - 1) + j]);
#pragma unroll
      for (int t = 0; t < T; t++) u[IG_TAPS - 1 + t] = rb(acc[i][t]);
      float w[IG_TAPS];
#pragma unroll
      for (int j = 0; j < IG_TAPS; j++) w[j] = bf(p.cw[(size_t)r * IG_TAPS + j]);
      __syncwarp();
      if (lane < T) {
#pragma unroll
        for (int t = 0; t < T; t++)
          if (lane == t) {
            float y = __fmul_rn(w[0], u[t]);
#pragma unroll
            for (int j = 1; j < IG_TAPS; j++) y = __fadd_rn(y, __fmul_rn(w[j], u[t + j]));
            float res = rb((y + u[t + IG_TAPS - 1]) / p.div);
            if (p.hres) res += bf(p.hres[(size_t)t * p.N + r]);
            p.outf[(size_t)t * p.N + r] = res;
          }
      }
      if (lane < IG_TAPS - 1) {
#pragma unroll
        for (int j = 0; j < IG_TAPS - 1; j++)
          if (lane == j) p.cst[(size_t)r * (IG_TAPS - 1) + j] = tb(u[T + j]);
      }
    } else if (lane < T) {
      float g = 0.f;
#pragma unroll
      for (int t = 0; t < T; t++)
        if (lane == t) g = acc[i][t];
      const int start = seg == 0 ? 0 : p.end[seg - 1];
      reinterpret_cast<bf16*>(p.out[seg])[(size_t)lane * (p.end[seg] - start) + (r - start)] = tb(g);
    }
  }
}

// The launch of one instantiation, defined in inkling_gemv_inst.cu (no other translation unit compiles a k_ig).
template <int T, int RPW, int MODE>
void warpc_inst_ink_gemv(const InkGemvP& p, int grid, cudaStream_t st);
