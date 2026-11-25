// The int4 head with exact rescoring (family 7): from the approximate fp32 logits [T][N] of an int4 g128 copy of the lm_head
// (w4g128_gemv.cu with the F32 epilogue, the caller's launch) to the argmax token of the exact bf16 head over the R best
// candidates.
//
// Specification (the order below is the contract; the results do not depend on the schedule):
//   1. candidates of a row = the R largest approximate logits, ties broken towards the lower index (NaN ranks lowest);
//      candidate rank 0 = the largest. Fewer than R survivors leave the remaining slots at -1.
//   2. exact(c) = fp32 sum over the exact bf16 products w[c][k] * x[k]: lane l of a warp owns the 16-byte chunks
//      i * 32 + l (i ascending) of the row, 8 fma per chunk in column order into ONE accumulator, then the butterfly
//      sum over xor 16, 8, 4, 2, 1. exact(c) overwrites logits[t][c]; the other logits stay approximate.
//   3. out_tok[t] = the candidate with the largest exact logit, the lowest index on ties (token 0 if no candidate).
// Shapes are arguments: K % 256 == 0, N % 4 == 0, 128 R <= N <= 128 * CHUNK, R <= 64, T <= STEP_ROWS. Four launches:
// per-chunk maxima, survivor extraction (a chunk contributes at most its own top R among the elements >= the R-th largest
// chunk maximum, a set that never overflows however many logits tie), ranking, rescoring (the last block of a row takes
// the argmax). `ws` is warpc_headq_ws_bytes() of device scratch, zeroed once; the kernels leave it zeroed.
#include "dense_common.cuh"

namespace hq {

constexpr int CHUNK = 2048;  // logits per chunk (256 threads x 8)
constexpr int NCH_MAX = 128;  // chunk-maximum slots per row (N <= NCH_MAX * CHUNK)
constexpr int RMAX = 64;
constexpr int CAP = NCH_MAX * RMAX;  // survivor list capacity per row: every chunk contributes at most R <= 64 elements
constexpr int FLAT_MAX = 512;  // largest survivor set of a chunk ranked by counting
constexpr int ROWS = STEP_ROWS;

struct Ws {  // byte offsets in ws
  static constexpr size_t cnt = 0;  // int [2][ROWS]: survivor counters, rescore done counters
  static constexpr size_t cm = 2 * ROWS * 4;  // uint32 [ROWS][NCH_MAX] ordered chunk maxima
  static constexpr size_t cand = cm + ROWS * NCH_MAX * 4;  // int [ROWS][RMAX]
  static constexpr size_t exact = cand + ROWS * RMAX * 4;  // float [ROWS][RMAX]
  static constexpr size_t list = exact + ROWS * RMAX * 4;  // uint64 [ROWS][CAP]
  static constexpr size_t total = list + (size_t)ROWS * CAP * 8;
};

__device__ __forceinline__ float lo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float hi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }

__device__ __forceinline__ uint32_t ord32(float f) {  // order-preserving float -> uint32 (NaN maps to the minimum)
  if (f != f) return 0u;
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ uint64_t mkkey(uint32_t ord, uint32_t idx) { return ((uint64_t)ord << 32) | (uint64_t)(0xFFFFFFFFu - idx); }
__device__ __forceinline__ uint32_t key_idx(uint64_t k) { return 0xFFFFFFFFu - (uint32_t)k; }

// stage 1: per-chunk maxima. grid (nch, T), 256 threads.
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

// stage 2: thr = the R-th largest chunk maximum (>= R distinct elements are >= thr, so every true top-R element is >= thr).
// Every block appends its elements >= thr to the row's list, but at most its own top R (a true top-R element is in the top R
// of its chunk): the list holds <= nch * R entries.
__global__ void __launch_bounds__(256)
    k_extract(const float* __restrict__ lg, int N, int R, const uint32_t* __restrict__ cm_all, int* __restrict__ cnt, uint64_t* __restrict__ list) {
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x, nch = gridDim.x, lane = tid & 31, w = tid >> 5;
  __shared__ uint32_t cm[NCH_MAX], thr_s;
  __shared__ int ns, np;
  __shared__ uint64_t red[2][8], sk[FLAT_MAX];
  for (int i = tid; i < nch; i += 256) cm[i] = cm_all[r * NCH_MAX + i];
  if (tid == 0) {
    thr_s = nch <= R ? 0u : 0xFFFFFFFFu;
    ns = 0;
    np = 0;
  }
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
      if (o >= thr && v == v) {
        key[i] = mkkey(o, (uint32_t)idx);
        mine++;
      }
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
  if (ns <= FLAT_MAX) {  // moderately flat chunk: compact the survivors in smem; a key is among the R largest iff fewer than R survivors are larger
#pragma unroll
    for (int i = 0; i < CHUNK / 256; i++)
      if (key[i]) sk[atomicAdd(&np, 1)] = key[i];
    __syncthreads();
    for (int e = tid; e < ns; e += 256) {
      const uint64_t k = sk[e];
      int rank = 0;
      for (int j = 0; j < ns; j++) rank += sk[j] > k;
      if (rank < R) list[(size_t)r * CAP + atomicAdd(&cnt[r], 1)] = k;
    }
    return;
  }
  for (int it = 0; it < R; it++) {  // very flat chunk: its R largest keys by iterated block-argmax
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

// stage 3: rank the survivors (keys are unique), the R best -> cand[r][rank]. One block per row; resets the survivor counter.
// The survivors of a row never exceed nch * R, which the launch uses as the dynamic smem size.
__global__ void __launch_bounds__(512) k_rank(int R, int* __restrict__ cnt, const uint64_t* __restrict__ list, int* __restrict__ cand) {
  const int r = blockIdx.x, tid = threadIdx.x;
  extern __shared__ uint64_t keys[];
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

// stage 4: exact logits of the candidates (one warp per candidate), written over logits[t][cand]; the last CTA of a row takes
// the argmax over its candidates (value, then lowest index). The chunks of a row are loaded up front in batches of NB.
// grid (ceil(R / 4), T), 128 threads.
constexpr int NB = 20;  // 16-byte chunks per lane in flight (K = 5120 is one batch)
__global__ void __launch_bounds__(128)
    k_rescore(const uint4* __restrict__ w, const uint4* __restrict__ x, int N, int K, int R, float* __restrict__ logits, int* __restrict__ cand,
              float* __restrict__ exact, int* __restrict__ done, int* __restrict__ out_tok) {
  const int t = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int r = blockIdx.x * 4 + wp;
  const int row = r < R ? cand[t * RMAX + r] : -1;
  float v = -INFINITY;
  if (row >= 0) {
    const uint4* wr = w + (size_t)row * (K >> 3);
    const uint4* xr = x + (size_t)t * (K >> 3);
    const int nit = K >> 8;
    float a = 0.f;
    for (int i0 = 0; i0 < nit; i0 += NB) {
      uint4 wv[NB], xv[NB];
#pragma unroll
      for (int j = 0; j < NB; j++)
        if (i0 + j < nit) {
          wv[j] = __ldcs(wr + (i0 + j) * 32 + lane);
          xv[j] = __ldg(xr + (i0 + j) * 32 + lane);
        }
#pragma unroll
      for (int j = 0; j < NB; j++)
        if (i0 + j < nit) {
          a = fmaf(lo(wv[j].x), lo(xv[j].x), a);
          a = fmaf(hi(wv[j].x), hi(xv[j].x), a);
          a = fmaf(lo(wv[j].y), lo(xv[j].y), a);
          a = fmaf(hi(wv[j].y), hi(xv[j].y), a);
          a = fmaf(lo(wv[j].z), lo(xv[j].z), a);
          a = fmaf(hi(wv[j].z), hi(xv[j].z), a);
          a = fmaf(lo(wv[j].w), lo(xv[j].w), a);
          a = fmaf(hi(wv[j].w), hi(xv[j].w), a);
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    v = a;
    if (lane == 0) {
      exact[t * RMAX + r] = v;
      logits[(size_t)t * N + row] = v;
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
    if (cv > bv || (cv == bv && ci < bi)) {
      bv = cv;
      bi = ci;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > bv || (ov == bv && oi < bi)) {
      bv = ov;
      bi = oi;
    }
  }
  if (lane == 0) {
    out_tok[t] = bi == 0x7fffffff ? 0 : bi;
    done[t] = 0;
  }
}

}  // namespace hq

extern "C" {

size_t warpc_headq_ws_bytes() { return hq::Ws::total; }
const void* warpc_headq_ws_cand(const void* ws) { return (const char*)ws + hq::Ws::cand; }
const void* warpc_headq_ws_exact(const void* ws) { return (const char*)ws + hq::Ws::exact; }

// logits fp32 [T][N] (approximate; exact over the candidates on return), w the bf16 head [N][K], x bf16 [T][K], out_tok int [T].
int warpc_headq_select(float* logits, const void* w, const void* x, int N, int K, int T, int R, int* out_tok, void* ws, cudaStream_t st) {
  using namespace hq;
  if (T < 1 || T > ROWS || R < 1 || R > RMAX || (K & 255) || (N & 3) || N > NCH_MAX * CHUNK || N < 128 * R) return cudaErrorInvalidValue;
  char* b = (char*)ws;
  const int nch = (N + CHUNK - 1) / CHUNK;
  uint32_t* cm = (uint32_t*)(b + Ws::cm);
  int* cnt = (int*)(b + Ws::cnt);
  int* cand = (int*)(b + Ws::cand);
  float* exact = (float*)(b + Ws::exact);
  uint64_t* list = (uint64_t*)(b + Ws::list);
  k_chunkmax<<<dim3(nch, T), 256, 0, st>>>(logits, N, cm);
  k_extract<<<dim3(nch, T), 256, 0, st>>>(logits, N, R, cm, cnt, list);
  const size_t smem = (size_t)nch * R * sizeof(uint64_t);
  if (smem > 48 * 1024) {
    const cudaError_t e = cudaFuncSetAttribute(k_rank, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
  }
  k_rank<<<T, 512, smem, st>>>(R, cnt, list, cand);
  k_rescore<<<dim3((R + 3) / 4, T), 128, 0, st>>>((const uint4*)w, (const uint4*)x, N, K, R, logits, cand, exact, cnt + ROWS, out_tok);
  return cudaGetLastError();
}

}  // extern "C"
