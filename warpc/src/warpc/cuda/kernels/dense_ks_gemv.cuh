#pragma once
#include "dense_dec_gemv.cuh"  // dg_ldw (includes dense_gemv.cuh)

// K-split GEMV, mode 0 without a norm prologue: one block per output row, KS warps, the K range cut into KS slices of KS_NIT * 32 groups of 8 elements, one slice per warp, every weight load of the
// lane issued up front (as k_dg). For the rows that k_dg cannot hold in registers (K above 4096: Gemma's down projections at K = 6144 / 12288) this halves the serial chain of k_gemv and puts
// KS warps on a row. The reduction order differs from k_gemv's and depends on K and KS only (never on T or the other rows): lane l of slice s sums the 8-element groups k = s * K / (8 KS) + l + 32 j in
// order (fma), the 32 lanes are added with the xor butterfly, then the KS slice sums in slice order starting from 0. Output as mode 0 of k_gemv.
constexpr int KS_NIT = 8;  // 16-byte weight loads per lane: K = 2048 KS

template <int T, int KS>
__global__ void __launch_bounds__(KS * 32) k_ks(const __grid_constant__ GemvP p) {
  const int lane = threadIdx.x & 31, kg = threadIdx.x >> 5, r = blockIdx.x;
  const int K8 = p.K / 8, slice = K8 / KS;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x) + kg * slice + lane;
  uint4 wv[KS_NIT];
  const uint4* a = reinterpret_cast<const uint4*>(p.w + (size_t)r * p.K) + kg * slice + lane;
#pragma unroll
  for (int j = 0; j < KS_NIT; j++) wv[j] = dg_ldw(a + 32 * j);
  float acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = 0.f;
#pragma unroll
  for (int j = 0; j < KS_NIT; j++) {
    float wf[8];
    unpack8(wv[j], wf);
#pragma unroll
    for (int t = 0; t < T; t++) {
      float xf[8];
      unpack8(xw[(size_t)t * K8 + 32 * j], xf);
#pragma unroll
      for (int e = 0; e < 8; e++) acc[t] = fmaf(wf[e], xf[e], acc[t]);
    }
  }
  __shared__ float part[KS * T];
#pragma unroll
  for (int t = 0; t < T; t++) {
    const float s = warp_sum(acc[t]);
    if (lane == 0) part[kg * T + t] = s;
  }
  __syncthreads();
  if (kg || lane >= T) return;
  float g = 0.f;
  for (int s = 0; s < KS; s++) g += part[s * T + lane];
  const int sg = r < p.end[0] ? 0 : (r < p.end[1] ? 1 : 2);
  const int start = sg == 0 ? 0 : p.end[sg - 1];
  const int width = p.end[sg] - start;
  reinterpret_cast<bf16*>(p.out[sg])[(size_t)lane * width + (r - start)] = tb(g);
}

// The launch of one instantiation, defined in dense_ks_gemv_inst.cu.
template <int T, int KS>
void warpc_inst_ks(const GemvP& p, cudaStream_t st);
