// attn_res: rows (the block bank + the partial, each [T][H] bf16 or f32) are RMS-normalised, every query q (an [H] fp32 vector)
// softmaxes <q, row_n> H^-0.5 over the rows and mixes them. Query j writes outf[j] (f32 [T][H], optional) and outn[j] (bf16
// [T][H], optional): bf16(rmsnorm(mix) * nw[j]), the norm and cast that follow in the graph.
// ONE block reduction does everything: with c_n = softmax_n / rms_n the mix is sum_n c_n x_n, and both the row statistics and the
// mix's own norm follow from the Gram matrix of the raw rows (|mix|^2 = c^T G c) plus the dots q . x_n, so the block sums
// NR (NR + 1) / 2 + NR NQ values once and everything after is elementwise.
// Grid (T, G): G groups of NQ queries over the same rows, one CTA per (row t, group).
#pragma once
#include "inkling_res.cuh"

// KE column loads of a row (column tid + k * 256), the dtype decided once per row so the loads are all in flight together (a
// branch per element serialised them)
template <int KE>
__device__ __forceinline__ void ar_load_cols(float* x, const void* p, bool f32, size_t base, int tid, int H) {
  if (f32) {
    const float* q = reinterpret_cast<const float*>(p) + base;
#pragma unroll
    for (int k = 0; k < KE; k++) {
      const int i = tid + k * 256;
      x[k] = i < H ? q[i] : 0.f;
    }
  } else {
    const bf16* q = reinterpret_cast<const bf16*>(p) + base;
#pragma unroll
    for (int k = 0; k < KE; k++) {
      const int i = tid + k * 256;
      x[k] = i < H ? bf(q[i]) : 0.f;
    }
  }
}

// All loads of a thread are issued before any use (one memory round trip), then ONE block reduction; the column loop is fully
// unrolled on purpose: a rolled version with two passes over the rows measured 2x slower (6 dependent round trips instead of 1)
// even though its code is 5x smaller.
template <int NR, int NQ>
__global__ void __launch_bounds__(256) k_ink_ar(const __grid_constant__ ArP p) {
  __shared__ float sm[8 * 64];
  const int t = blockIdx.x, g = blockIdx.y, tid = threadIdx.x, H = p.H;
  constexpr int NG = NR * (NR + 1) / 2, NV = NG + NR * NQ;
  float x[NR][AR_KE];
  float v[NV];
#pragma unroll
  for (int n = 0; n < NR; n++) ar_load_cols<AR_KE>(x[n], p.rows[n], (p.f32mask >> n) & 1, (size_t)t * H, tid, H);
  {
    int idx = 0;
#pragma unroll
    for (int a = 0; a < NR; a++)
#pragma unroll
      for (int b = a; b < NR; b++) {
        float s = 0.f;
#pragma unroll
        for (int k = 0; k < AR_KE; k++) s += x[a][k] * x[b][k];
        v[idx++] = s;
      }
  }
  float qv[NQ][AR_KE];
#pragma unroll
  for (int j = 0; j < NQ; j++)
#pragma unroll
    for (int k = 0; k < AR_KE; k++) {
      const int i = tid + k * 256;
      qv[j][k] = i < H ? p.q[g * NQ + j][i] : 0.f;
    }
#pragma unroll
  for (int j = 0; j < NQ; j++)
#pragma unroll
    for (int n = 0; n < NR; n++) {
      float s = 0.f;
#pragma unroll
      for (int k = 0; k < AR_KE; k++) s += qv[j][k] * x[n][k];
      v[NG + j * NR + n] = s;
    }
  block_sum_vec<NV>(v, sm);
  float r[NR], gm[NR][NR];
  {
    int idx = 0;
#pragma unroll
    for (int a = 0; a < NR; a++)
#pragma unroll
      for (int b = a; b < NR; b++) {
        gm[a][b] = gm[b][a] = v[idx++];
      }
  }
#pragma unroll
  for (int n = 0; n < NR; n++) r[n] = rsqrtf(gm[n][n] / (float)H + p.eps);
  const float scale = rsqrtf((float)H);
#pragma unroll
  for (int j = 0; j < NQ; j++) {
    float c[NR];
    float m = -INFINITY;
#pragma unroll
    for (int n = 0; n < NR; n++) {
      c[n] = v[NG + j * NR + n] * r[n] * scale;
      m = fmaxf(m, c[n]);
    }
    float l = 0.f;
#pragma unroll
    for (int n = 0; n < NR; n++) {
      c[n] = expf(c[n] - m);
      l += c[n];
    }
#pragma unroll
    for (int n = 0; n < NR; n++) c[n] = c[n] / l * r[n];  // softmax weight x row scale
    float* of = p.outf[g * NQ + j];
    bf16* on = p.outn[g * NQ + j];
    float rn = 0.f;
    if (on) {
      float q2 = 0.f;
#pragma unroll
      for (int a = 0; a < NR; a++)
#pragma unroll
        for (int b = 0; b < NR; b++) q2 += c[a] * c[b] * gm[a][b];
      rn = rsqrtf(q2 / (float)H + p.eps_n);
    }
#pragma unroll
    for (int k = 0; k < AR_KE; k++) {
      const int i = tid + k * 256;
      float o = 0.f;
#pragma unroll
      for (int n = 0; n < NR; n++) o += c[n] * x[n][k];
      if (i < H) {
        if (of) of[(size_t)t * H + i] = o;
        if (on) on[(size_t)t * H + i] = tb(o * rn * bf(p.nw[g * NQ + j][i]));
      }
    }
  }
}

template <int NR>
cudaError_t ink_ar_run(const ArP& p, int nq, int G, int T, cudaStream_t st) {
  const dim3 grid(T, G);
  switch (nq) {
    case 1: k_ink_ar<NR, 1><<<grid, 256, 0, st>>>(p); break;
    case 2: k_ink_ar<NR, 2><<<grid, 256, 0, st>>>(p); break;
    case 3: k_ink_ar<NR, 3><<<grid, 256, 0, st>>>(p); break;
    case 4: k_ink_ar<NR, 4><<<grid, 256, 0, st>>>(p); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
