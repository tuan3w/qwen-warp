#pragma once
#include "dense_common.cuh"

// GEMV y[T][N] = x[T][K] W[N][K]^T. One warp per RPW rows (a row is read by the 32 lanes with 16-byte loads), fp32 accumulation, the 32 lane partials summed with a fixed butterfly: the
// reduction order depends on K only. mode 0: bf16 outputs split by row range (up to 3 segments); mode 1: gate/up pair, out = bf16(act(bf16(gate))) * bf16(up),
// rounded to bf16 (rows of w = gate, rows of w2 = up; act 0 silu, 1 gelu_tanh); mode 2: fp32 out[0] [T][N], softcapped when cap > 0; mode 3: residual add, out[0] is the residual stream h [T][N] and
// h = bf16(h + bf16(y)) in place (each element belongs to one row, so no cross-block ordering is needed); mode 4: out[0] [T][N] = bf16(act(bf16(y)) * row), row a bf16 [T][N] operand.
// With nw != NULL the input rows are normalised in the prologue exactly as k_rmsnorm does (every warp computes the row statistics itself).
struct GemvP {
  const bf16* w;
  const bf16* w2;
  const bf16* x;
  const bf16* nw;  // != NULL: x is the raw residual stream and is RMS-normalised on the fly with this weight (prologue fusion)
  float eps;
  const bf16* row;  // mode 4: the [T][N] operand of the gated product
  float cap;        // mode 2: > 0 stores cap * tanhf(acc / cap) instead of the accumulator
  int act;          // modes 1, 4: 0 silu, 1 gelu_tanh
  int N, K;
  int end[3];
  void* out[3];
};

// The gate activation of the pair and row epilogues on the fp32 accumulator: bf16(act(bf16(g))). gelu_tanh is torch's gelu(approximate="tanh") on a bf16 input (fp32 inside, tanhf, the same
// expression and fma contraction as ATen's CUDA kernel).
__device__ __forceinline__ float gate_act(float g, int act) {
  const float a = rb(g);
  if (act == 0) return rb(a / (1.f + __expf(-a)));
  const float kBeta = 0.7978845608028654f, kKappa = 0.044715f;  // sqrt(2 / pi)
  return rb(0.5f * a * (1.f + tanhf(kBeta * (a + kKappa * (a * a * a)))));
}
// The fp32 logits of mode 2: the accumulator, softcapped when cap > 0.
__device__ __forceinline__ float logit_out(float g, float cap) { return cap > 0.f ? cap * tanhf(g / cap) : g; }

template <int T, int RPW, int MODE, int U>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_gemv(const __grid_constant__ GemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * RPW;
  const int K8 = p.K / 8;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x);
  const uint4* wr[RPW];
  const uint4* wr2[RPW];
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = min(row0 + i, p.N - 1);
    wr[i] = reinterpret_cast<const uint4*>(p.w + (size_t)r * p.K);
    wr2[i] = MODE == 1 ? reinterpret_cast<const uint4*>(p.w2 + (size_t)r * p.K) : wr[i];
  }
  float acc[RPW][T], acc2[RPW][T], rs[T];
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[i][t] = acc2[i][t] = 0.f;
  if (p.nw) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      float ss = 0.f;
      for (int k = lane; k < K8; k += 32) {
        float f[8];
        unpack8(xw[(size_t)t * K8 + k], f);
#pragma unroll
        for (int e = 0; e < 8; e++) ss += f[e] * f[e];
      }
      rs[t] = rsqrtf(warp_sum(ss) / (float)p.K + p.eps);
    }
  }

#pragma unroll U
  for (int k = lane; k < K8; k += 32) {
    uint4 wv[RPW], wv2[RPW];
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      wv[i] = __ldg(wr[i] + k);
      if (MODE == 1) wv2[i] = __ldg(wr2[i] + k);
    }
    float xf[T][8];
#pragma unroll
    for (int t = 0; t < T; t++) unpack8(xw[(size_t)t * K8 + k], xf[t]);
    if (p.nw) {
      float nwf[8];
      unpack8(reinterpret_cast<const uint4*>(p.nw)[k], nwf);
#pragma unroll
      for (int t = 0; t < T; t++)
#pragma unroll
        for (int e = 0; e < 8; e++) xf[t][e] = rb(xf[t][e] * rs[t] * nwf[e]);
    }
#pragma unroll
    for (int i = 0; i < RPW; i++) {
      float wf[8];
      unpack8(wv[i], wf);
#pragma unroll
      for (int t = 0; t < T; t++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[i][t] = fmaf(wf[e], xf[t][e], acc[i][t]);
      if (MODE == 1) {
        float uf[8];
        unpack8(wv2[i], uf);
#pragma unroll
        for (int t = 0; t < T; t++)
#pragma unroll
          for (int e = 0; e < 8; e++) acc2[i][t] = fmaf(uf[e], xf[t][e], acc2[i][t]);
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
    } else if (MODE == 1) {
      reinterpret_cast<bf16*>(p.out[0])[(size_t)lane * p.N + r] = tb(gate_act(g, p.act) * rb(u));
    } else if (MODE == 4) {
      reinterpret_cast<bf16*>(p.out[0])[(size_t)lane * p.N + r] = tb(gate_act(g, p.act) * bf(p.row[(size_t)lane * p.N + r]));
    } else {
      reinterpret_cast<float*>(p.out[0])[(size_t)lane * p.N + r] = logit_out(g, p.cap);
    }
  }
}

// The launch of one instantiation, defined in dense_gemv_inst.cu (so that no translation unit but the instantiation units compiles a k_gemv).
template <int T, int RPW, int MODE>
void warpc_inst_gemv(const GemvP& p, int grid, cudaStream_t st);
