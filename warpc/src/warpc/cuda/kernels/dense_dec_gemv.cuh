#pragma once
#include "dense_gemv.cuh"

// Single-row GEMV with every load in flight: k_dg is k_gemv (T = 1) restructured so that all of a lane's weight loads (RPW rows x NIT 16-byte loads, NIT = K / 256) and the activation loads are issued before anything is
// computed: the RMSNorm statistics overlap the DRAM latency of the weights instead of delaying the first weight load. The arithmetic is k_gemv's, operation for operation: the same
// lane -> k mapping (k = lane + 32 j), the same fma order per row, the same warp_sum, the same epilogues, so the results are bit-identical.
// Weights go through ld.global.nc.L1::no_allocate (they are read once; the activation row stays in L1).
__device__ __forceinline__ uint4 dg_ldw(const uint4* a) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(a));
  return v;
}

template <int RPW, int MODE, int NIT, bool NORM>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_dg(const __grid_constant__ GemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * RPW;
  uint4 wv[RPW][NIT];
  uint4 wv2[MODE == 1 ? RPW : 1][MODE == 1 ? NIT : 1];
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = min(row0 + i, p.N - 1);
    const uint4* a = reinterpret_cast<const uint4*>(p.w + (size_t)r * p.K) + lane;
#pragma unroll
    for (int j = 0; j < NIT; j++) wv[i][j] = dg_ldw(a + 32 * j);
    if constexpr (MODE == 1) {
      const uint4* a2 = reinterpret_cast<const uint4*>(p.w2 + (size_t)r * p.K) + lane;
#pragma unroll
      for (int j = 0; j < NIT; j++) wv2[MODE == 1 ? i : 0][MODE == 1 ? j : 0] = dg_ldw(a2 + 32 * j);
    }
  }
  uint4 xv[NIT];
  uint4 nwv[NORM ? NIT : 1];
#pragma unroll
  for (int j = 0; j < NIT; j++) xv[j] = reinterpret_cast<const uint4*>(p.x)[lane + 32 * j];
  if constexpr (NORM) {
#pragma unroll
    for (int j = 0; j < NIT; j++) nwv[NORM ? j : 0] = __ldg(reinterpret_cast<const uint4*>(p.nw) + lane + 32 * j);
  }
  float rs = 0.f;
  if constexpr (NORM) {
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < NIT; j++) {
      float f[8];
      unpack8(xv[j], f);
#pragma unroll
      for (int e = 0; e < 8; e++) ss += f[e] * f[e];
    }
    rs = rsqrtf(warp_sum(ss) / (float)p.K + p.eps);
  }
  float acc[RPW], acc2[MODE == 1 ? RPW : 1];
#pragma unroll
  for (int i = 0; i < RPW; i++) acc[i] = 0.f;
#pragma unroll
  for (int i = 0; i < (MODE == 1 ? RPW : 1); i++) acc2[i] = 0.f;
#pragma unroll
  for (int j = 0; j < NIT; j++) {
    float xf[8];
    unpack8(xv[j], xf);
    if constexpr (NORM) {
      float nwf[8];
      unpack8(nwv[NORM ? j : 0], nwf);
#pragma unroll
      for (int e = 0; e < 8; e++) xf[e] = rb(xf[e] * rs * nwf[e]);
    }
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      float wf[8];
      unpack8(wv[i][j], wf);
#pragma unroll
      for (int e = 0; e < 8; e++) acc[i] = fmaf(wf[e], xf[e], acc[i]);
      if constexpr (MODE == 1) {
        float uf[8];
        unpack8(wv2[MODE == 1 ? i : 0][MODE == 1 ? j : 0], uf);
#pragma unroll
        for (int e = 0; e < 8; e++) acc2[MODE == 1 ? i : 0] = fmaf(uf[e], xf[e], acc2[MODE == 1 ? i : 0]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    acc[i] = warp_sum(acc[i]);
    if constexpr (MODE == 1) acc2[MODE == 1 ? i : 0] = warp_sum(acc2[MODE == 1 ? i : 0]);
  }
  if (lane) return;
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = row0 + i;
    if (r >= p.N) continue;
    const float g = acc[i];
    if constexpr (MODE == 0) {
      const int s = r < p.end[0] ? 0 : (r < p.end[1] ? 1 : 2);
      const int start = s == 0 ? 0 : p.end[s - 1];
      reinterpret_cast<bf16*>(p.out[s])[r - start] = tb(g);
    } else if constexpr (MODE == 3) {
      bf16* hr = reinterpret_cast<bf16*>(p.out[0]) + r;
      *hr = tb(bf(*hr) + rb(g));
    } else if constexpr (MODE == 1) {
      const float u = acc2[MODE == 1 ? i : 0];
      const float a = rb(g);
      const float sl = rb(a / (1.f + __expf(-a)));
      reinterpret_cast<bf16*>(p.out[0])[r] = tb(sl * rb(u));
    } else {
      reinterpret_cast<float*>(p.out[0])[r] = g;
    }
  }
}

// One launcher per mode (dense_dec_gemv_m<mode>.cu): rpw in {1, 2, 4}, NIT = K / 256. They return DEC_NO_KERNEL when no k_dg instantiation exists for the shape (nothing was launched: the caller
// falls back to k_gemv), otherwise the cudaGetLastError() of the launch they made (a launch happened: the caller must return it, never launch a second kernel).
constexpr int DEC_NO_KERNEL = -1;  // not a cudaError_t value (those are >= 0)
int warpc_dec_gemv_m0(const GemvP& p, int rpw, bool norm, cudaStream_t st);
int warpc_dec_gemv_m1(const GemvP& p, int rpw, bool norm, cudaStream_t st);
int warpc_dec_gemv_m2(const GemvP& p, int rpw, bool norm, cudaStream_t st);
int warpc_dec_gemv_m3(const GemvP& p, int rpw, bool norm, cudaStream_t st);
