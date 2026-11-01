// swift-mk K3: RMSNorm (zero-centred, fused residual add), embed gather, argmax.
#include "common.cuh"
#include "engine.h"

namespace smk {

// one block per row, H/8 threads (one 16 B chunk each; H <= 8192).  Residual: h = bf16(h + y) in place; xn = bf16(h * rsqrt(mean(h^2)+eps) * (1+w)).
__global__ void k_rmsnorm(uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w, uint4* __restrict__ xn, uint4* __restrict__ dbg,
                          int H, float eps) {
  __shared__ float wsum[32];
  const int nch = H >> 3;
  const int c = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * nch;
  float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float ss = 0.f;
  uint4 wv = make_uint4(0, 0, 0, 0);
  if (c < nch) wv = w[c];  // independent of h / y: issued first so its latency hides behind the reduction
  if (c < nch) {
    const uint4 hv = h[base + c];
    v[0] = bflo(hv.x); v[1] = bfhi(hv.x); v[2] = bflo(hv.y); v[3] = bfhi(hv.y); v[4] = bflo(hv.z); v[5] = bfhi(hv.z); v[6] = bflo(hv.w); v[7] = bfhi(hv.w);
    if (y) {
      const uint4 yv = y[base + c];
      v[0] = bf16r(v[0] + bflo(yv.x)); v[1] = bf16r(v[1] + bfhi(yv.x)); v[2] = bf16r(v[2] + bflo(yv.y)); v[3] = bf16r(v[3] + bfhi(yv.y));
      v[4] = bf16r(v[4] + bflo(yv.z)); v[5] = bf16r(v[5] + bfhi(yv.z)); v[6] = bf16r(v[6] + bflo(yv.w)); v[7] = bf16r(v[7] + bfhi(yv.w));
      h[base + c] = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
    }
    if (dbg) dbg[base + c] = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
#pragma unroll
    for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  }
  ss = warp_sum(ss);
  if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < (int)((blockDim.x + 31) >> 5); i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + eps);
  if (c < nch) {
    const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
    xn[base + c] = make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                              pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
  }
}

__global__ void k_embed(const uint4* __restrict__ embed, const int* __restrict__ tok, uint4* __restrict__ out, int H) {
  const int nch = H >> 3;
  const size_t id = (size_t)tok[blockIdx.x];
  for (int c = threadIdx.x; c < nch; c += blockDim.x) out[(size_t)blockIdx.x * nch + c] = embed[id * nch + c];
}

// ---------------------------------------------------------------------------------------------- argmax
constexpr int AM_G = 64;       // blocks per row
constexpr int AM_THREADS = 512;
struct AmPartial { float v; int i; };
__device__ __forceinline__ bool better(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

__global__ void __launch_bounds__(AM_THREADS) k_argmax(const float* __restrict__ logits, int N, int* __restrict__ out_tok, float* __restrict__ out_val, void* wsv) {
  int* cnt = (int*)wsv;                              // [16] counters
  AmPartial* part = (AmPartial*)((char*)wsv + 64);   // [T][AM_G]
  __shared__ AmPartial wred[AM_THREADS / 32];
  __shared__ int is_last;
  const int row = blockIdx.y;
  const float4* lg = (const float4*)(logits + (size_t)row * N);
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = blockIdx.x * AM_THREADS + threadIdx.x; i < (N >> 2); i += AM_G * AM_THREADS) {
    const float4 v = lg[i];
    if (better(v.x, 4 * i, bv, bi)) { bv = v.x; bi = 4 * i; }
    if (better(v.y, 4 * i + 1, bv, bi)) { bv = v.y; bi = 4 * i + 1; }
    if (better(v.z, 4 * i + 2, bv, bi)) { bv = v.z; bi = 4 * i + 2; }
    if (better(v.w, 4 * i + 3, bv, bi)) { bv = v.w; bi = 4 * i + 3; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
  }
  if ((threadIdx.x & 31) == 0) wred[threadIdx.x >> 5] = {bv, bi};
  __syncthreads();
  if (threadIdx.x == 0) {
    AmPartial b = wred[0];
    for (int i = 1; i < AM_THREADS / 32; i++)
      if (better(wred[i].v, wred[i].i, b.v, b.i)) b = wred[i];
    part[row * AM_G + blockIdx.x] = b;
    __threadfence();
    is_last = (atomicAdd(&cnt[row], 1) == AM_G - 1);
  }
  __syncthreads();
  if (is_last) {
    __threadfence();
    if (threadIdx.x < 32) {
      float v = -INFINITY;
      int i = 0x7fffffff;
      for (int k = threadIdx.x; k < AM_G; k += 32) {
        const AmPartial p = {__ldcg(&part[row * AM_G + k].v), __ldcg(&part[row * AM_G + k].i)};
        if (better(p.v, p.i, v, i)) { v = p.v; i = p.i; }
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (better(ov, oi, v, i)) { v = ov; i = oi; }
      }
      if (threadIdx.x == 0) {
        out_tok[row] = i;
        if (out_val) out_val[row] = v;
        cnt[row] = 0;
      }
    }
  }
}

}  // namespace smk

using namespace smk;
extern "C" {

cudaError_t smk_rmsnorm(void* h, const void* y, const void* w, void* xn, void* dbg, int T, int H, float eps, cudaStream_t st) {
  if ((H & 7) || H > 8192 || T < 1) return cudaErrorInvalidValue;
  const int thr = ((H >> 3) + 31) & ~31;
  k_rmsnorm<<<T, thr, 0, st>>>((uint4*)h, (const uint4*)y, (const uint4*)w, (uint4*)xn, (uint4*)dbg, H, eps);
  return cudaGetLastError();
}

cudaError_t smk_embed(const void* embed, const int* tok, void* out, int T, int H, cudaStream_t st) {
  if (H & 7) return cudaErrorInvalidValue;
  k_embed<<<T, 256, 0, st>>>((const uint4*)embed, tok, (uint4*)out, H);
  return cudaGetLastError();
}

size_t smk_argmax_ws_bytes(void) { return 64 + 16 * AM_G * sizeof(AmPartial); }

cudaError_t smk_argmax(const float* logits, int T, int N, int* out_tok, float* out_val, void* ws, cudaStream_t st) {
  if ((N & 3) || T < 1 || T > 16) return cudaErrorInvalidValue;
  k_argmax<<<dim3(AM_G, T), AM_THREADS, 0, st>>>(logits, N, out_tok, out_val, ws);
  return cudaGetLastError();
}

}  // extern "C"
