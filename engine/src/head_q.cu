// swift-mk K9: quantised lm_head GEMV (int8 per-row CUDA-core kernel; int4 g128 = smk_w4_gemv) + top-R select + exact BF16 rescoring.  See include/head_q.h.
// The int8 kernel is the E5 prototype (experiments/2026-10-31-lm-head-quantisation-quality/bench_head.cu, same author): one warp = two rows, lane l reads a uint4
// (16 columns) per 512-column step, int8 -> fp32 by the magic-number trick (xor 0x80 per byte, PRMT into the mantissa of 2^23, subtract 2^23 + 128), x in smem in a
// lane-permuted fp32 layout (conflict-free LDS.128), persistent grid (smem filled once per CTA).
#include "head_q.h"

#include "common.cuh"
#include "engine.h"

using namespace smk;

namespace {
constexpr int K_HID = SMK_HIDDEN;
constexpr int NIT = K_HID / 512;  // 10

// ---------------------------------------------------------------------------------------------- int8 per-row GEMV
__device__ __forceinline__ float i8_to_f(uint32_t w_xor, int e) { return __uint_as_float(__byte_perm(w_xor, 0x4B000000u, 0x7440 | e)) - 8388736.0f; }

template <int T, int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) k_head_i8(const uint4* __restrict__ w, const float* __restrict__ sc, const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int N) {
  extern __shared__ float xs[];  // [T][K] lane-permuted
  for (int c = threadIdx.x; c < K_HID; c += WARPS * 32) {
    const int it = c / 512, r = c % 512, lane = r / 16, j = (r % 16) / 4, e = r % 4;
#pragma unroll
    for (int t = 0; t < T; ++t) xs[t * K_HID + ((it * 4 + j) * 32 + lane) * 4 + e] = __bfloat162float(x[t * K_HID + c]);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pairs = N / 2;
  for (int pr = blockIdx.x * WARPS + warp; pr < pairs; pr += gridDim.x * WARPS) {
    const uint4* w0 = w + (size_t)(2 * pr) * (K_HID / 16);
    const uint4* w1 = w0 + K_HID / 16;
    float acc0[T], acc1[T];
#pragma unroll
    for (int t = 0; t < T; ++t) acc0[t] = acc1[t] = 0.f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      uint4 a[NIT / 2], b[NIT / 2];
#pragma unroll
      for (int i = 0; i < NIT / 2; ++i) {
        a[i] = __ldcs(w0 + (half * (NIT / 2) + i) * 32 + lane);
        b[i] = __ldcs(w1 + (half * (NIT / 2) + i) * 32 + lane);
      }
#pragma unroll
      for (int i = 0; i < NIT / 2; ++i) {
        const int it = half * (NIT / 2) + i;
        const uint32_t wa[4] = {a[i].x ^ 0x80808080u, a[i].y ^ 0x80808080u, a[i].z ^ 0x80808080u, a[i].w ^ 0x80808080u};
        const uint32_t wb[4] = {b[i].x ^ 0x80808080u, b[i].y ^ 0x80808080u, b[i].z ^ 0x80808080u, b[i].w ^ 0x80808080u};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float4 xv[T];
#pragma unroll
          for (int t = 0; t < T; ++t) xv[t] = *reinterpret_cast<const float4*>(xs + t * K_HID + ((it * 4 + j) * 32 + lane) * 4);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float fa = i8_to_f(wa[j], e), fb = i8_to_f(wb[j], e);
#pragma unroll
            for (int t = 0; t < T; ++t) {
              const float xe = e == 0 ? xv[t].x : e == 1 ? xv[t].y : e == 2 ? xv[t].z : xv[t].w;
              acc0[t] = fmaf(fa, xe, acc0[t]);
              acc1[t] = fmaf(fb, xe, acc1[t]);
            }
          }
        }
      }
    }
#pragma unroll
    for (int t = 0; t < T; ++t) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        acc0[t] += __shfl_xor_sync(0xffffffffu, acc0[t], o);
        acc1[t] += __shfl_xor_sync(0xffffffffu, acc1[t], o);
      }
    }
    if (lane < T) {  // lane t writes row t
      float v0 = 0.f, v1 = 0.f;
#pragma unroll
      for (int t = 0; t < T; ++t)
        if (lane == t) { v0 = acc0[t]; v1 = acc1[t]; }
      *reinterpret_cast<float2*>(out + (size_t)lane * N + 2 * pr) = make_float2(v0 * sc[2 * pr], v1 * sc[2 * pr + 1]);
    }
  }
}

template <int T, int WARPS>
cudaError_t launch_i8(const void* w, const float* sc, const void* x, float* out, int N, int nsm, cudaStream_t st) {
  const size_t smem = (size_t)T * K_HID * sizeof(float);
  static bool init = false;
  if (!init) {
    SMK_CK(cudaFuncSetAttribute(k_head_i8<T, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
    init = true;
  }
  k_head_i8<T, WARPS><<<nsm, WARPS * 32, smem, st>>>((const uint4*)w, sc, (const __nv_bfloat16*)x, out, N);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------------------------- select + rescore
constexpr int CHUNK = 2048;      // logits per chunk (256 threads x 8)
constexpr int NCH_MAX = 128;     // chunk-maximum slots per row (N <= 262144)
constexpr int CAP = NCH_MAX * 64;  // survivor list capacity per row: every chunk contributes at most R <= 64 elements (see k_extract), so it never overflows (N <= NCH_MAX * CHUNK)
constexpr int RMAX = SMK_HEADQ_RMAX;
static_assert(RMAX <= 64, "CAP assumes R <= 64");
constexpr int ROWS = 8;  // K11: T <= 8 rows (int4 head; the int8 CUDA-core kernel keeps T <= 4: its smem holds T x 20 KB of activations)

struct WsLayout {  // byte offsets in ws
  static constexpr size_t cnt = 0;                                    // int [16]: [0..7] survivor counters, [8..15] rescore done counters
  static constexpr size_t cm = 64;                                    // uint32 [ROWS][NCH_MAX] ordered chunk maxima
  static constexpr size_t cand = cm + ROWS * NCH_MAX * 4;             // int [4][RMAX]
  static constexpr size_t exact = cand + ROWS * RMAX * 4;             // float [4][RMAX]
  static constexpr size_t list = exact + ROWS * RMAX * 4;             // uint64 [4][CAP]
  static constexpr size_t total = list + (size_t)ROWS * CAP * 8;
};

__device__ __forceinline__ uint32_t ord32(float f) {  // order-preserving float -> uint32 (NaN maps to the minimum)
  if (f != f) return 0u;
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ uint64_t mkkey(uint32_t ord, uint32_t idx) { return ((uint64_t)ord << 32) | (uint64_t)(0xFFFFFFFFu - idx); }
__device__ __forceinline__ uint32_t key_idx(uint64_t k) { return 0xFFFFFFFFu - (uint32_t)k; }

// stage 1: per-chunk maxima.  grid (nch, T), 256 threads.
__global__ void __launch_bounds__(256) k_chunkmax(const float* __restrict__ lg, int N, uint32_t* __restrict__ cm) {
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
  const float* row = lg + (size_t)r * N;
  uint32_t best = 0u;
#pragma unroll
  for (int i = 0; i < CHUNK / 256; i++) {
    const int idx = c * CHUNK + i * 256 + tid;
    if (idx < N) best = max(best, ord32(__ldg(row + idx)));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) best = max(best, __shfl_xor_sync(0xffffffffu, best, o));
  __shared__ uint32_t red[8];
  if ((tid & 31) == 0) red[tid >> 5] = best;
  __syncthreads();
  if (tid == 0) {
    uint32_t g = red[0];
    for (int i = 1; i < 8; i++) g = max(g, red[i]);
    cm[r * NCH_MAX + c] = g;
  }
}

__device__ __forceinline__ uint64_t wmax64(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint64_t w = __shfl_xor_sync(0xffffffffu, v, o);
    v = w > v ? w : v;
  }
  return v;
}

// stage 2: thr = R-th largest chunk maximum (>= R distinct elements are >= thr, so every true top-R element is >= thr).  Every block appends its elements >= thr to the row's
// list, but at most its own top-R (a true top-R element is in the top-R of its chunk): the list holds <= nch * R entries, never overflows, and is the same set on every run
// even when thousands of logits tie / cluster at the threshold (a flat cluster of near-identical logits, e.g. the untrained vocabulary rows, would overflow a plain >= thr list).
__global__ void __launch_bounds__(256) k_extract(const float* __restrict__ lg, int N, int R, const uint32_t* __restrict__ cm_all, int* __restrict__ cnt, uint64_t* __restrict__ list) {
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x, nch = gridDim.x, lane = tid & 31, w = tid >> 5;
  __shared__ uint32_t cm[NCH_MAX], thr_s;
  __shared__ int ns;
  __shared__ uint64_t red[2][8];
  for (int i = tid; i < nch; i += 256) cm[i] = cm_all[r * NCH_MAX + i];
  if (tid == 0) { thr_s = nch <= R ? 0u : 0xFFFFFFFFu; ns = 0; }
  __syncthreads();
  if (nch > R && tid < nch) {
    int g = 0;
    for (int j = 0; j < nch; j++) g += cm[j] > cm[tid];
    if (g < R) atomicMin(&thr_s, cm[tid]);
  }
  __syncthreads();
  const uint32_t thr = thr_s;
  if (cm[c] < thr) return;  // block-uniform
  const float* row = lg + (size_t)r * N;
  uint64_t key[CHUNK / 256];
  int mine = 0;
#pragma unroll
  for (int i = 0; i < CHUNK / 256; i++) {
    const int idx = c * CHUNK + i * 256 + tid;
    key[i] = 0ull;
    if (idx < N) {
      const float v = __ldg(row + idx);
      const uint32_t o = ord32(v);
      if (o >= thr && v == v) { key[i] = mkkey(o, (uint32_t)idx); mine++; }
    }
  }
  if (mine) atomicAdd(&ns, mine);
  __syncthreads();
  if (ns <= R) {  // common case: append every survivor
#pragma unroll
    for (int i = 0; i < CHUNK / 256; i++)
      if (key[i]) list[(size_t)r * CAP + atomicAdd(&cnt[r], 1)] = key[i];
    return;
  }
  for (int it = 0; it < R; it++) {  // flat chunk: its R largest keys by iterated block-argmax
    uint64_t best = key[0];
#pragma unroll
    for (int i = 1; i < CHUNK / 256; i++) best = key[i] > best ? key[i] : best;
    best = wmax64(best);
    if (lane == 0) red[it & 1][w] = best;
    __syncthreads();
    uint64_t g = red[it & 1][0];
#pragma unroll
    for (int i = 1; i < 8; i++) g = red[it & 1][i] > g ? red[it & 1][i] : g;
    if (tid == 0) list[(size_t)r * CAP + atomicAdd(&cnt[r], 1)] = g;
#pragma unroll
    for (int i = 0; i < CHUNK / 256; i++) key[i] = key[i] == g ? 0ull : key[i];
  }
}

// stage 3: rank the survivors (keys are unique), the R best -> cand[r][rank].  One block per row; resets the survivor counter.
__global__ void __launch_bounds__(512) k_rank(int R, int* __restrict__ cnt, const uint64_t* __restrict__ list, int* __restrict__ cand) {
  const int r = blockIdx.x, tid = threadIdx.x;
  extern __shared__ uint64_t keys[];  // CAP entries
  const int n = min(cnt[r], CAP);
  for (int i = tid; i < n; i += 512) keys[i] = list[(size_t)r * CAP + i];
  __syncthreads();
  for (int e = tid; e < n; e += 512) {
    const uint64_t k = keys[e];
    int rank = 0;
    for (int j = 0; j < n; j++) rank += keys[j] > k;
    if (rank < R) cand[r * RMAX + rank] = (int)key_idx(k);
  }
  for (int e = n + tid; e < R; e += 512) cand[r * RMAX + e] = -1;
  if (tid == 0) cnt[r] = 0;
}

// stage 4: exact logits of the candidates (one warp per candidate: fp32 accumulate over exact bf16 products), written over logits[t][cand]; the last CTA of a row
// takes the argmax over its candidates (value, then lowest index).  grid (ceil(R/4), T), 128 threads.
__global__ void __launch_bounds__(128) k_rescore(const uint4* __restrict__ w, const uint4* __restrict__ x, int N, int R, float* __restrict__ logits, const int* __restrict__ cand,
                                                 float* __restrict__ exact, int* __restrict__ done, int* __restrict__ out_tok, float* __restrict__ out_val) {
  const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int r = blockIdx.x * 4 + wp;
  const int id = r < R ? cand[t * RMAX + r] : -1;
  float v = -INFINITY;
  if (id >= 0) {
    const uint4* wr = w + (size_t)id * (K_HID / 8);
    const uint4* xr = x + (size_t)t * (K_HID / 8);
    uint4 wv[K_HID / 256], xv[K_HID / 256];
#pragma unroll
    for (int i = 0; i < K_HID / 256; i++) {
      wv[i] = __ldcs(wr + i * 32 + lane);
      xv[i] = __ldg(xr + i * 32 + lane);
    }
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < K_HID / 256; i++) {
      a = fmaf(bflo(wv[i].x), bflo(xv[i].x), a); a = fmaf(bfhi(wv[i].x), bfhi(xv[i].x), a);
      a = fmaf(bflo(wv[i].y), bflo(xv[i].y), a); a = fmaf(bfhi(wv[i].y), bfhi(xv[i].y), a);
      a = fmaf(bflo(wv[i].z), bflo(xv[i].z), a); a = fmaf(bfhi(wv[i].z), bfhi(xv[i].z), a);
      a = fmaf(bflo(wv[i].w), bflo(xv[i].w), a); a = fmaf(bfhi(wv[i].w), bfhi(xv[i].w), a);
    }
    v = warp_sum(a);
    if (lane == 0) {
      exact[t * RMAX + r] = v;
      logits[(size_t)t * N + id] = v;
    }
  }
  __syncthreads();
  __shared__ int is_last;
  if (tid == 0) {
    __threadfence();
    is_last = (atomicAdd(&done[t], 1) == (int)gridDim.x - 1);
  }
  __syncthreads();
  if (!is_last || wp != 0) return;
  __threadfence();
  float bv = -INFINITY;
  int bi = 0x7fffffff;
  for (int q = lane; q < R; q += 32) {
    const int ci = __ldcg(cand + t * RMAX + q);
    if (ci < 0) continue;
    const float cv = __ldcg(exact + t * RMAX + q);
    if (cv > bv || (cv == bv && ci < bi)) { bv = cv; bi = ci; }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
  }
  if (lane == 0) {
    out_tok[t] = bi == 0x7fffffff ? 0 : bi;
    if (out_val) out_val[t] = bv;
    done[t] = 0;
  }
}
}  // namespace

extern "C" {

size_t smk_headq_ws_bytes(void) { return WsLayout::total; }
const int* smk_headq_ws_cand(const void* ws) { return (const int*)((const char*)ws + WsLayout::cand); }
const float* smk_headq_ws_exact(const void* ws) { return (const float*)((const char*)ws + WsLayout::exact); }

cudaError_t smk_headq_gemv(int kind, const void* wq, const void* sq, const void* x, int N, int K, int T, float* logits, cudaStream_t st) {
  if (T < 1 || T > ROWS) return cudaErrorInvalidValue;
  if (kind == SMK_HEADQ_I8) {
    if (T > 4 || K != K_HID || (N & 1)) return cudaErrorInvalidValue;
    static int nsm = 0;
    if (!nsm) cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    switch (T) {
      case 1: return launch_i8<1, 16>(wq, (const float*)sq, x, logits, N, nsm, st);
      case 2: return launch_i8<2, 16>(wq, (const float*)sq, x, logits, N, nsm, st);
      case 3: return launch_i8<3, 8>(wq, (const float*)sq, x, logits, N, nsm, st);
      default: return launch_i8<4, 8>(wq, (const float*)sq, x, logits, N, nsm, st);
    }
  }
  if (kind == SMK_HEADQ_I4) {
    smk_epi ep = {};
    ep.kind = SMK_EPI_F32;
    ep.out[0] = logits;
    return smk_w4_gemv_cfg(wq, sq, x, N, K, T, &ep, 1, st);  // w8k8u2: best config at every T (E5 Table 5)
  }
  return cudaErrorInvalidValue;
}

cudaError_t smk_headq_select_rescore(float* logits, const void* w_bf16, const void* x, int N, int K, int T, int R, int* out_tok, float* out_val, void* ws, cudaStream_t st) {
  if (T < 1 || T > ROWS || R < 1 || R > RMAX || (K & 255) || K != K_HID || (N & 3) || N > NCH_MAX * CHUNK || N < 128 * R) return cudaErrorInvalidValue;
  char* b = (char*)ws;
  const int nch = (N + CHUNK - 1) / CHUNK;
  uint32_t* cm = (uint32_t*)(b + WsLayout::cm);
  int* cnt = (int*)(b + WsLayout::cnt);
  int* cand = (int*)(b + WsLayout::cand);
  float* exact = (float*)(b + WsLayout::exact);
  uint64_t* list = (uint64_t*)(b + WsLayout::list);
  k_chunkmax<<<dim3(nch, T), 256, 0, st>>>(logits, N, cm);
  k_extract<<<dim3(nch, T), 256, 0, st>>>(logits, N, R, cm, cnt, list);
  static bool rank_init = false;
  if (!rank_init) {
    SMK_CK(cudaFuncSetAttribute(k_rank, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)(CAP * sizeof(uint64_t))));
    rank_init = true;
  }
  k_rank<<<T, 512, CAP * sizeof(uint64_t), st>>>(R, cnt, list, cand);
  k_rescore<<<dim3((R + 3) / 4, T), 128, 0, st>>>((const uint4*)w_bf16, (const uint4*)x, N, R, logits, cand, exact, cnt + 8, out_tok, out_val);
  return cudaGetLastError();
}

}  // extern "C"
