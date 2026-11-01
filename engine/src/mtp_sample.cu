// swift-mk K7: top-k / top-p sampling kernels and exact speculative-sampling acceptance (device side) + test hooks.  See include/mtp.h, src/mtp_sample.h.
#include "mtp_sample.h"

using namespace smk;

namespace {
constexpr uint32_t FULL = 0xffffffffu;

// ---------------------------------------------------------------------------------------------- philox4x32-10
__device__ __forceinline__ uint4 philox(uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3, uint32_t k0, uint32_t k1) {
#pragma unroll
  for (int r = 0; r < 10; r++) {
    const uint32_t hi0 = __umulhi(0xD2511F53u, c0), lo0 = 0xD2511F53u * c0;
    const uint32_t hi1 = __umulhi(0xCD9E8D57u, c2), lo1 = 0xCD9E8D57u * c2;
    c0 = hi1 ^ c1 ^ k0;
    c1 = lo1;
    c2 = hi0 ^ c3 ^ k1;
    c3 = lo0;
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return make_uint4(c0, c1, c2, c3);
}
__device__ __forceinline__ uint4 rng(const SampP* sp, uint32_t ctr, uint32_t stream) {
  return philox(ctr, stream, 0u, 0u, (uint32_t)sp->seed, (uint32_t)(sp->seed >> 32));
}
__device__ __forceinline__ float u01(uint32_t x) { return (float)(x >> 8) * (1.0f / 16777216.0f); }  // [0, 1 - 2^-24]

// ---------------------------------------------------------------------------------------------- top-k stage 1 / 2
// Order-preserving 64-bit key: high word = ordered fp32 bits, low word = ~index (larger logit wins, then the lower index).  Never 0.
__device__ __forceinline__ uint64_t mkkey(float f, uint32_t idx) {
  const uint32_t u = __float_as_uint(f);
  const uint32_t kh = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((uint64_t)kh << 32) | (uint64_t)(0xFFFFFFFFu - idx);
}
__device__ __forceinline__ float key_val(uint64_t k) {
  const uint32_t kh = (uint32_t)(k >> 32);
  return __uint_as_float((kh & 0x80000000u) ? (kh & 0x7FFFFFFFu) : ~kh);
}
__device__ __forceinline__ uint32_t key_idx(uint64_t k) { return 0xFFFFFFFFu - (uint32_t)k; }
__device__ __forceinline__ uint64_t wmax64(uint64_t v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const uint64_t w = __shfl_xor_sync(FULL, v, o);
    v = w > v ? w : v;
  }
  return v;
}

// Stage 0: per-chunk maximum key -> cm[r][c] (uint64 [R][SK_NCH_T] at the end of ws).
__device__ __forceinline__ uint64_t* cm_of(uint64_t* ws) { return ws + (size_t)SK_ROWS * SK_NCH_T * SK_KMAX; }
__global__ void __launch_bounds__(256) k_topk0(const float* __restrict__ logits, size_t stride, int N, uint64_t* __restrict__ ws) {
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const float* row = logits + (size_t)r * stride;
  uint64_t best = 0ull;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int idx = c * SK_CHUNK + i * 256 + tid;
    const uint64_t k = idx < N ? mkkey(__ldg(row + idx), (uint32_t)idx) : 0ull;
    best = k > best ? k : best;
  }
  best = wmax64(best);
  __shared__ uint64_t red[8];
  if (lane == 0) red[w] = best;
  __syncthreads();
  if (tid == 0) {
    uint64_t g = red[0];
    for (int i = 1; i < 8; i++) g = red[i] > g ? red[i] : g;
    cm_of(ws)[r * SK_NCH_T + c] = g;
  }
}

// Stage 1: each block extracts, by iterated block-argmax, the elements of its chunk that can be in the row's top-K: those >= thr = the K-th largest chunk maximum
// (K distinct elements are >= thr, so everything below it is outside the top-K).  Written descending, zero padded to K, to ws.
__global__ void __launch_bounds__(256) k_topk1(const float* __restrict__ logits, size_t stride, int N, const SampP* __restrict__ sp, int sel, uint64_t* __restrict__ ws, int nch) {
  const int K = sel ? sp->d_top_k : sp->top_k;
  const int r = blockIdx.y, c = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  const float* row = logits + (size_t)r * stride;
  uint64_t key[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int idx = c * SK_CHUNK + i * 256 + tid;
    key[i] = idx < N ? mkkey(__ldg(row + idx), (uint32_t)idx) : 0ull;
  }
  __shared__ uint64_t red[2][8], cmv[SK_NCH_T], thr_s;
  const uint64_t* cm = cm_of(ws) + (size_t)r * SK_NCH_T;
  for (int i = tid; i < nch; i += 256) cmv[i] = cm[i];
  if (tid == 0) thr_s = 0ull;
  __syncthreads();
  if (tid < nch && K <= nch) {
    int greater = 0;
    for (int i = 0; i < nch; i++) greater += cmv[i] > cmv[tid];
    if (greater == K - 1) thr_s = cmv[tid];
  }
  __syncthreads();
  const uint64_t thr = thr_s;
  uint64_t* dst = ws + ((size_t)r * SK_NCH_T + c) * SK_KMAX;
  int cnt = 0;
  for (int it = 0; it < K; it++) {
    uint64_t best = key[0];
#pragma unroll
    for (int i = 1; i < 8; i++) best = key[i] > best ? key[i] : best;
    best = wmax64(best);
    if (lane == 0) red[it & 1][w] = best;
    __syncthreads();
    uint64_t g = red[it & 1][0];
#pragma unroll
    for (int i = 1; i < 8; i++) g = red[it & 1][i] > g ? red[it & 1][i] : g;
    if (g < thr || g == 0ull) break;  // block-uniform
    if (tid == 0) dst[it] = g;
#pragma unroll
    for (int i = 0; i < 8; i++) key[i] = key[i] == g ? 0ull : key[i];
    cnt++;
  }
  for (int i = cnt + tid; i < K; i += 256) dst[i] = 0ull;
}

// One block (128 threads) per row: K-way merge of the nch sorted stage-1 lists held in shared memory (thread c owns list c; per iteration one block-argmax over the
// list heads, the owner advances), then temperature / softmax / top-p.  Dynamic smem: nch * K uint64.
constexpr int SK2_THREADS = 128;
__global__ void __launch_bounds__(SK2_THREADS) k_topk2(const SampP* __restrict__ sp, int sel, const uint64_t* __restrict__ ws, int nch, const int* __restrict__ idmap, Dist* __restrict__ out) {
  extern __shared__ uint64_t lists[];
  const int K = sel ? sp->d_top_k : sp->top_k;
  const float inv_t = sel ? sp->d_inv_t : sp->inv_t, top_p = sel ? sp->d_top_p : sp->top_p;
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  for (int m = tid; m < nch * K; m += SK2_THREADS) {
    const int c = m / K, i = m - c * K;
    lists[m] = ws[((size_t)r * SK_NCH_T + c) * SK_KMAX + i];
  }
  __syncthreads();
  __shared__ uint64_t red[2][SK2_THREADS / 32];
  __shared__ uint64_t sk[SK_KMAX];
  int h = 0;
  uint64_t cur = tid < nch ? lists[tid * K] : 0ull;
  for (int it = 0; it < K; it++) {
    const uint64_t best = wmax64(cur);
    if (lane == 0) red[it & 1][w] = best;
    __syncthreads();
    uint64_t g = red[it & 1][0];
#pragma unroll
    for (int i = 1; i < SK2_THREADS / 32; i++) g = red[it & 1][i] > g ? red[it & 1][i] : g;
    if (tid == 0) sk[it] = g;
    if (cur == g) {
      h++;
      cur = h < K ? lists[tid * K + h] : 0ull;
    }
  }
  __syncthreads();
  if (tid == 0) {
    Dist* o = out + r;
    const float v0 = key_val(sk[0]);
    float e[SK_KMAX], sum = 0.f;
    for (int i = 0; i < K; i++) {
      e[i] = expf((key_val(sk[i]) - v0) * inv_t);
      sum += e[i];
    }
    // top-p on the softmax over the top-k survivors (HF: TopP after TopK; keep token i iff the mass before it is < top_p, at least one)
    int n = 1;
    float cum = e[0] / sum;
    while (n < K && (top_p >= 1.0f || cum < top_p)) {  // top_p >= 1: disabled (keep all K)
      cum += e[n] / sum;
      n++;
    }
    float kept = 0.f;
    for (int i = 0; i < n; i++) kept += e[i] / sum;
    o->n = n;
    for (int i = 0; i < n; i++) {
      const int idx = (int)key_idx(sk[i]);
      o->ids[i] = idmap ? idmap[idx] : idx;
      o->p[i] = (e[i] / sum) / kept;
    }
    for (int i = n; i < SK_KMAX; i++) {
      o->ids[i] = -1;
      o->p[i] = 0.f;
    }
  }
}

// ---------------------------------------------------------------------------------------------- warp-level sampling helpers
struct LD {  // this lane's entries (lane, lane + 32) of a Dist
  int id0, id1;
  float p0, p1;
  int n;
};
__device__ __forceinline__ LD ld(const Dist* d, int lane) {
  LD r;
  r.n = d->n;
  r.id0 = lane < r.n ? d->ids[lane] : -1;
  r.p0 = lane < r.n ? d->p[lane] : 0.f;
  r.id1 = lane + 32 < r.n ? d->ids[lane + 32] : -1;
  r.p1 = lane + 32 < r.n ? d->p[lane + 32] : 0.f;
  return r;
}
__device__ __forceinline__ float wsum_all(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
  return v;
}
__device__ __forceinline__ float lookup(const LD& x, int tok) { return wsum_all((x.id0 == tok ? x.p0 : 0.f) + (x.id1 == tok ? x.p1 : 0.f)); }
__device__ __forceinline__ int entry_id(const LD& x, int e) { return __shfl_sync(FULL, e < 32 ? x.id0 : x.id1, e & 31); }

// inverse-CDF draw over the weights (w0 for entry lane, w1 for entry lane + 32) with u in [0,1): returns the entry index e (warp-uniform)
__device__ __forceinline__ int warp_draw(float w0, float w1, float u, int lane) {
  float s0 = w0, s1 = w1;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const float a = __shfl_up_sync(FULL, s0, o), b = __shfl_up_sync(FULL, s1, o);
    if (lane >= o) { s0 += a; s1 += b; }
  }
  const float t0 = __shfl_sync(FULL, s0, 31), tot = t0 + __shfl_sync(FULL, s1, 31);
  const float target = u * tot;
  const uint32_t b0 = __ballot_sync(FULL, w0 > 0.f && s0 > target), b1 = __ballot_sync(FULL, w1 > 0.f && t0 + s1 > target);
  if (b0) return __ffs(b0) - 1;
  if (b1) return 32 + __ffs(b1) - 1;
  const uint32_t p0 = __ballot_sync(FULL, w0 > 0.f), p1 = __ballot_sync(FULL, w1 > 0.f);  // rounding: last positive entry
  return p1 ? 63 - __clz(p1) : (p0 ? 31 - __clz(p0) : 0);
}

// Speculative-sampling acceptance of the k drafts d[1..k] against the verify rows pd[0..k] and proposals qd[0..k-1] (all lanes of one warp).
// Returns a = emitted tokens (accepted drafts + 1), y[0..a-1] written by lane 0.
__device__ int accept_core(const Dist* pd, const Dist* qd, const int* d, int k, const SampP* sp, uint32_t ctr, int lane, int* y) {
  for (int t = 0; t < k; t++) {
    const LD P = ld(pd + t, lane), Q = ld(qd + t, lane);
    const int dt = d[t + 1];
    const float pv = lookup(P, dt), qv = lookup(Q, dt);
    const uint4 rr = rng(sp, ctr, ST_ACCEPT + t);
    if (u01(rr.x) * qv < pv) {  // accept w.p. min(1, p/q)
      y[t] = dt;  // every lane stores the same value: keeps the warp converged for the shuffles / syncwarp that follow
      continue;
    }
    float w0 = P.p0, w1 = P.p1;
    if (!(sp->flags & 1)) {  // residual max(0, p - q) / Z; its support is inside p's
      float q0 = 0.f, q1 = 0.f;
      for (int s = 0; s < Q.n; s++) {
        const int qi = __shfl_sync(FULL, s < 32 ? Q.id0 : Q.id1, s & 31);
        const float qp = __shfl_sync(FULL, s < 32 ? Q.p0 : Q.p1, s & 31);
        q0 = qi == P.id0 ? qp : q0;
        q1 = qi == P.id1 ? qp : q1;
      }
      const float r0 = fmaxf(P.p0 - q0, 0.f), r1 = fmaxf(P.p1 - q1, 0.f);
      if (wsum_all(r0 + r1) > 0.f) { w0 = r0; w1 = r1; }  // Z == 0 only if p == q everywhere (rejection then has probability 0)
    }
    const int e = warp_draw(w0, w1, u01(rr.y), lane);
    const int tok = entry_id(P, e);
    y[t] = tok;
    return t + 1;
  }
  const LD P = ld(pd + k, lane);  // all accepted: bonus token from p of the last row
  const int e = warp_draw(P.p0, P.p1, u01(rng(sp, ctr, ST_ACCEPT + k).y), lane);
  y[k] = entry_id(P, e);
  return k + 1;
}

__device__ __forceinline__ int sample_dist(const Dist* dd, const SampP* sp, uint32_t ctr, uint32_t stream, int lane) {
  const LD x = ld(dd, lane);
  return entry_id(x, warp_draw(x.p0, x.p1, u01(rng(sp, ctr, stream).x), lane));
}

// ---------------------------------------------------------------------------------------------- graph kernels (one warp)
__global__ void k_accept_s(int* ds, int* ss, int* ring, int k, const Dist* pd, const Dist* qd, const SampP* sp) {
  __shared__ int y[8];
  const int lane = threadIdx.x;
  const int a = accept_core(pd, qd, ds + DS_TOK, k, sp, (uint32_t)ss[SS_NCYC], lane, y);
  __syncwarp();
  if (lane < a) ds[DS_OUT + lane] = y[lane];
  __syncwarp();
  if (lane == 0) spec_finish(ds, ss, ring, a - 1);
}

__global__ void k_draft_pick(const Dist* qd, const SampP* sp, const int* ss, int j, int* dst) {
  const int tok = sample_dist(qd, sp, (uint32_t)ss[SS_NCYC], ST_DRAFT + j, threadIdx.x);
  if (threadIdx.x == 0) *dst = tok;
}

__global__ void k_delta_q(const int* toks, Dist* qd, int from, int to) {  // toks[t] = the proposed token of proposal t
  const int t = from + blockIdx.x;
  if (t >= to) return;
  Dist* o = qd + t;
  const int i = threadIdx.x;
  if (i == 0) o->n = 1;
  if (i < SK_KMAX) {
    o->ids[i] = i == 0 ? toks[t] : -1;
    o->p[i] = i == 0 ? 1.f : 0.f;
  }
}

__global__ void k_first_pick(const Dist* pd, const SampP* sp, int* ss, int* ds, int bump) {
  const int lane = threadIdx.x;
  if (!(sp->flags & 2)) {
    const int tok = sample_dist(pd, sp, (uint32_t)ss[SS_NCYC], ST_FIRST, lane);
    if (lane == 0) {
      ds[DS_OUT] = tok;
      ds[DS_TOK] = tok;
    }
  }
  if (lane == 0 && bump) ss[SS_NCYC] += 1;
}

// ---------------------------------------------------------------------------------------------- test kernels
__global__ void k_draw_n(const Dist* pd, const SampP* sp, int n, uint32_t ctr0, int* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float u = u01(rng(sp, ctr0 + (uint32_t)i, ST_DRAW).x);
  float cum = 0.f;
  int tok = pd->ids[pd->n - 1];
  for (int e = 0; e < pd->n; e++) {
    cum += pd->p[e];
    if (cum > u) { tok = pd->ids[e]; break; }
  }
  out[i] = tok;
}

// one speculative cycle per warp on fixed rows: draft pick j = 1..k from qd[j-1], then accept_core
__global__ void k_trials(const Dist* pd, const Dist* qd, const SampP* sp, int k, int ntrials, uint32_t ctr0, int* y, int* a, int nchain, const int* tail, int ystride) {
  const int lane = threadIdx.x & 31;
  const int trial = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (trial >= ntrials) return;
  int d[SK_ROWS + 1];
  d[0] = -1;
  for (int j = 1; j <= k; j++) d[j] = j <= nchain ? sample_dist(qd + (j - 1), sp, ctr0 + (uint32_t)trial, ST_DRAFT + j, lane) : tail[j - nchain - 1];
  int yy[8];
  const int n = accept_core(pd, qd, d, k, sp, ctr0 + (uint32_t)trial, lane, yy);
  if (lane == 0) {
    for (int t = 0; t < ystride; t++) y[trial * ystride + t] = t < n ? yy[t] : -1;
    a[trial] = n;
  }
}

__global__ void k_philox_test(const unsigned* c, const unsigned* kk, unsigned* out) {
  const uint4 r = philox(c[0], c[1], c[2], c[3], kk[0], kk[1]);
  out[0] = r.x; out[1] = r.y; out[2] = r.z; out[3] = r.w;
}

SampP make_sampp(const smk_sampling* c) {
  SampP p = {};
  p.inv_t = 1.0f / c->temperature;
  p.top_k = c->top_k;
  p.top_p = c->top_p;
  const float dt = c->draft_temperature > 0.f ? c->draft_temperature : c->temperature;
  p.d_inv_t = 1.0f / dt;
  p.d_top_k = c->draft_sample ? (c->draft_top_k > 0 ? c->draft_top_k : c->top_k) : 1;
  p.d_top_p = c->draft_top_p > 0.f ? c->draft_top_p : c->top_p;
  p.flags = c->test_flags;
  p.seed = c->seed;
  return p;
}
bool valid(const smk_sampling* c) {
  return c->temperature > 0.f && c->top_k >= 1 && c->top_k <= SK_KMAX && c->top_p > 0.f && c->top_p <= 1.f && c->draft_top_k >= 0 && c->draft_top_k <= SK_KMAX &&
         c->draft_temperature >= 0.f && c->draft_top_p >= 0.f && c->draft_top_p <= 1.f;
}
}  // namespace

// ---------------------------------------------------------------------------------------------- launchers
namespace smk {
cudaError_t sk_topk(const float* logits, int R, int N, size_t stride, const int* idmap, const SampP* sp, int sel, uint64_t* ws, Dist* out, cudaStream_t st) {
  const int nch = (N + SK_CHUNK - 1) / SK_CHUNK;
  if (R < 1 || R > SK_ROWS || nch > (sel ? SK_NCH_D : SK_NCH_T)) return cudaErrorInvalidValue;
  k_topk0<<<dim3(nch, R), 256, 0, st>>>(logits, stride, N, ws);
  SMK_CK(cudaGetLastError());
  k_topk1<<<dim3(nch, R), 256, 0, st>>>(logits, stride, N, sp, sel, ws, nch);
  SMK_CK(cudaGetLastError());
  static bool smem_ok = false;  // opt in to > 48 KB dynamic shared memory (top_k up to 64: nch * 64 * 8 B)
  if (!smem_ok) {
    SMK_CK(cudaFuncSetAttribute(k_topk2, cudaFuncAttributeMaxDynamicSharedMemorySize, SK_NCH_T * SK_KMAX * 8));
    smem_ok = true;
  }
  k_topk2<<<R, SK2_THREADS, (size_t)nch * SK_KMAX * 8, st>>>(sp, sel, ws, nch, idmap, out);
  return cudaGetLastError();
}
cudaError_t sk_first_pick(const Dist* pd, const SampP* sp, const int* ss, int* ds, int bump, cudaStream_t st) {
  k_first_pick<<<1, 32, 0, st>>>(pd, sp, (int*)ss, ds, bump);
  return cudaGetLastError();
}
cudaError_t sk_draft_pick(Dist* qd, const SampP* sp, const int* ss, int j, int* dst, cudaStream_t st) {
  k_draft_pick<<<1, 32, 0, st>>>(qd, sp, ss, j, dst);
  return cudaGetLastError();
}
cudaError_t sk_delta_q(const int* ds, Dist* qd, int from, int to, cudaStream_t st) {
  if (to <= from) return cudaSuccess;
  k_delta_q<<<to - from, SK_KMAX, 0, st>>>(ds + DS_TOK + 1, qd, from, to);
  return cudaGetLastError();
}
cudaError_t sk_accept(int* ds, int* ss, int* ring, int k, const Dist* pd, const Dist* qd, const SampP* sp, cudaStream_t st) {
  k_accept_s<<<1, 32, 0, st>>>(ds, ss, ring, k, pd, qd, sp);
  return cudaGetLastError();
}
SampP smk_make_sampp(const smk_sampling* c) { return make_sampp(c); }
bool smk_sampling_valid(const smk_sampling* c) { return valid(c); }
cudaError_t sk_draw_n(const Dist* pd, const SampP* sp, int n, uint32_t ctr0, int* out, cudaStream_t st) {
  k_draw_n<<<(n + 255) / 256, 256, 0, st>>>(pd, sp, n, ctr0, out);
  return cudaGetLastError();
}
}  // namespace smk

// ---------------------------------------------------------------------------------------------- test hooks that need no spec object
extern "C" {

cudaError_t smk_test_philox(const unsigned* c, const unsigned* kk, unsigned* out) {
  unsigned *dc, *dk, *dout;
  cudaMalloc(&dc, 16);
  cudaMalloc(&dk, 8);
  cudaMalloc(&dout, 16);
  cudaMemcpy(dc, c, 16, cudaMemcpyHostToDevice);
  cudaMemcpy(dk, kk, 8, cudaMemcpyHostToDevice);
  k_philox_test<<<1, 1>>>(dc, dk, dout);
  cudaError_t e = cudaMemcpy(out, dout, 16, cudaMemcpyDeviceToHost);
  cudaFree(dc);
  cudaFree(dk);
  cudaFree(dout);
  return e;
}

// device scratch for the host-fed test hooks
struct TestBufs {
  float *tl = nullptr, *dl = nullptr;
  int *idmap = nullptr, *y = nullptr, *a = nullptr;
  uint64_t* ws = nullptr;
  Dist *pd = nullptr, *qd = nullptr;
  SampP* sp = nullptr;
  ~TestBufs() {
    for (void* p : {(void*)tl, (void*)dl, (void*)idmap, (void*)y, (void*)a, (void*)ws, (void*)pd, (void*)qd, (void*)sp}) if (p) cudaFree(p);
  }
};

cudaError_t smk_test_dist(const float* logits, int R, int N, const int* idmap, const smk_sampling* cfg, int sel, int* out_n, int* out_ids, float* out_p) {
  if (!valid(cfg) || R < 1 || R > SK_ROWS) return cudaErrorInvalidValue;
  TestBufs b;
  const SampP sp = make_sampp(cfg);
  cudaMalloc(&b.tl, (size_t)R * N * 4);
  cudaMalloc(&b.ws, sizeof(uint64_t) * SK_WS_ENTRIES);
  cudaMalloc(&b.pd, sizeof(Dist) * R);
  cudaMalloc(&b.sp, sizeof(SampP));
  if (idmap) { cudaMalloc(&b.idmap, (size_t)N * 4); cudaMemcpy(b.idmap, idmap, (size_t)N * 4, cudaMemcpyHostToDevice); }
  cudaMemcpy(b.tl, logits, (size_t)R * N * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.sp, &sp, sizeof(sp), cudaMemcpyHostToDevice);
  SMK_CK(sk_topk(b.tl, R, N, N, b.idmap, b.sp, sel, b.ws, b.pd, 0));
  std::vector<Dist> h(R);
  SMK_CK(cudaMemcpy(h.data(), b.pd, sizeof(Dist) * R, cudaMemcpyDeviceToHost));
  for (int r = 0; r < R; r++) {
    out_n[r] = h[r].n;
    memcpy(out_ids + r * SK_KMAX, h[r].ids, sizeof(int) * SK_KMAX);
    memcpy(out_p + r * SK_KMAX, h[r].p, sizeof(float) * SK_KMAX);
  }
  return cudaSuccess;
}

cudaError_t smk_test_trials(const float* tl, int NT, const float* dl, int ND, const int* idmap, const smk_sampling* cfg, int k, int ntrials, unsigned ctr0, int* y, int* a) {
  if (!valid(cfg) || k < 1 || k > SMK_SPEC_KMAX || ntrials < 1) return cudaErrorInvalidValue;
  TestBufs b;
  const SampP sp = make_sampp(cfg);
  cudaMalloc(&b.tl, (size_t)(k + 1) * NT * 4);
  cudaMalloc(&b.dl, (size_t)k * ND * 4);
  cudaMalloc(&b.ws, sizeof(uint64_t) * SK_WS_ENTRIES);
  cudaMalloc(&b.pd, sizeof(Dist) * (k + 1));
  cudaMalloc(&b.qd, sizeof(Dist) * k);
  cudaMalloc(&b.sp, sizeof(SampP));
  cudaMalloc(&b.y, (size_t)ntrials * 4 * 4);
  cudaMalloc(&b.a, (size_t)ntrials * 4);
  if (idmap) { cudaMalloc(&b.idmap, (size_t)ND * 4); cudaMemcpy(b.idmap, idmap, (size_t)ND * 4, cudaMemcpyHostToDevice); }
  cudaMemcpy(b.tl, tl, (size_t)(k + 1) * NT * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.dl, dl, (size_t)k * ND * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.sp, &sp, sizeof(sp), cudaMemcpyHostToDevice);
  SMK_CK(sk_topk(b.tl, k + 1, NT, NT, nullptr, b.sp, 0, b.ws, b.pd, 0));
  for (int j = 0; j < k; j++) SMK_CK(sk_topk(b.dl + (size_t)j * ND, 1, ND, ND, b.idmap, b.sp, 1, b.ws, b.qd + j, 0));
  k_trials<<<(ntrials + 3) / 4, 128>>>(b.pd, b.qd, b.sp, k, ntrials, ctr0, b.y, b.a, k, nullptr, 4);
  SMK_CK(cudaGetLastError());
  SMK_CK(cudaMemcpy(y, b.y, (size_t)ntrials * 16, cudaMemcpyDeviceToHost));
  return cudaMemcpy(a, b.a, (size_t)ntrials * 4, cudaMemcpyDeviceToHost);
}


// K11: nchain chain drafts (from the draft logits) + ntail delta tail tokens; y[ntrials][8]
cudaError_t smk_test_trials_pld(const float* tl, int NT, const float* dl, int ND, const int* idmap, const smk_sampling* cfg, int nchain, const int* tail, int ntail, int ntrials, unsigned ctr0, int* y, int* a) {
  const int k = nchain + ntail;
  if (!valid(cfg) || nchain < 1 || nchain > SMK_SPEC_KMAX || ntail < 0 || k > SMK_SPEC_PMAX || ntrials < 1) return cudaErrorInvalidValue;
  TestBufs b;
  int* dtail = nullptr;
  const SampP sp = make_sampp(cfg);
  cudaMalloc(&b.tl, (size_t)(k + 1) * NT * 4);
  cudaMalloc(&b.dl, (size_t)nchain * ND * 4);
  cudaMalloc(&b.ws, sizeof(uint64_t) * SK_WS_ENTRIES);
  cudaMalloc(&b.pd, sizeof(Dist) * (k + 1));
  cudaMalloc(&b.qd, sizeof(Dist) * k);
  cudaMalloc(&b.sp, sizeof(SampP));
  cudaMalloc(&b.y, (size_t)ntrials * 8 * 4);
  cudaMalloc(&b.a, (size_t)ntrials * 4);
  cudaMalloc(&dtail, 8 * sizeof(int));
  if (idmap) { cudaMalloc(&b.idmap, (size_t)ND * 4); cudaMemcpy(b.idmap, idmap, (size_t)ND * 4, cudaMemcpyHostToDevice); }
  cudaMemcpy(b.tl, tl, (size_t)(k + 1) * NT * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.dl, dl, (size_t)nchain * ND * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(b.sp, &sp, sizeof(sp), cudaMemcpyHostToDevice);
  if (ntail > 0) cudaMemcpy(dtail, tail, ntail * sizeof(int), cudaMemcpyHostToDevice);
  SMK_CK(sk_topk(b.tl, k + 1, NT, NT, nullptr, b.sp, 0, b.ws, b.pd, 0));
  for (int j = 0; j < nchain; j++) SMK_CK(sk_topk(b.dl + (size_t)j * ND, 1, ND, ND, b.idmap, b.sp, 1, b.ws, b.qd + j, 0));
  if (ntail > 0) { k_delta_q<<<ntail, SK_KMAX>>>(dtail - nchain, b.qd, nchain, k); SMK_CK(cudaGetLastError()); }
  k_trials<<<(ntrials + 3) / 4, 128>>>(b.pd, b.qd, b.sp, k, ntrials, ctr0, b.y, b.a, nchain, dtail, 8);
  SMK_CK(cudaGetLastError());
  SMK_CK(cudaMemcpy(y, b.y, (size_t)ntrials * 32, cudaMemcpyDeviceToHost));
  const cudaError_t er = cudaMemcpy(a, b.a, (size_t)ntrials * 4, cudaMemcpyDeviceToHost);
  cudaFree(dtail);
  return er;
}

}  // extern "C"
