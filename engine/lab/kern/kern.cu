// T1 kernel harness for the sparsity program (Part C).  ONE translation unit; compiles the repo's own gemv.cu + attn.cu in (the dense baselines are the engine's kernels, not
// re-implementations), adds
//   (i)   k_w4s       : K-block-sparse copy of k_w4<RP=1> (the engine's repacked W4 mma GEMV) driven by a device-side index list (128 / 64 / 32-column blocks),
//   (ii)  k_select    : fused "score + top-k + union + x-zeroing" mask builder over a [M][K] activation (M CTAs + ticket),
//   (iii) attn_split_i4s_kernel (attn_sparse.cuh): block-gather copy of attn_split_i4_kernel over the rotated int4 KV layout (kvrot.cuh), + fp32 reference kernel.
// Timing: cuda events around CUDA graphs of N launches that cycle over DISTINCT weight / KV copies (L2 is 6 MB).  See run.sh / KERN.md.
#include "gemv.cu"
#include "attn.cu"
#include "attn_sparse.cuh"
#include "attn_tok.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <stdarg.h>

#define CK(x)                                                                                             \
  do {                                                                                                    \
    cudaError_t e_ = (x);                                                                                 \
    if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } \
  } while (0)

static const char* WDIR = "/media/fatman/data/sparselab/kern";
static cudaStream_t g_st;
static bool g_quick = false;
static int g_n_launch = 100;  // launches per graph
static int g_reps = 3;

// ------------------------------------------------------------------------------------------------------------------------------ json / table output
struct Json {
  FILE* f = nullptr;
  bool first = true;
  void open(const char* p) { f = fopen(p, "w"); if (f) fputs("[\n", f); first = true; }
  void close() { if (f) { fputs("\n]\n", f); fclose(f); f = nullptr; } }
  void rec(const char* fmt, ...) {
    if (!f) return;
    if (!first) fputs(",\n", f);
    first = false;
    va_list ap; va_start(ap, fmt); vfprintf(f, fmt, ap); va_end(ap);
  }
};
static Json jg, ja, jm;

// ------------------------------------------------------------------------------------------------------------------------------ timing
// Median per-launch ms of a graph of n launches.  launch(i, st) enqueues launch i (distinct buffer copy chosen by the caller from i).
template <class F>
static double time_graph(F&& launch, int n = 0, int reps = 0) {
  if (!n) n = g_n_launch;
  if (!reps) reps = g_reps;
  cudaGraph_t gr; cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(g_st, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < n; i++) launch(i, g_st);
  CK(cudaGetLastError());
  CK(cudaStreamEndCapture(g_st, &gr));
  CK(cudaGraphInstantiate(&ge, gr, 0));
  CK(cudaGraphLaunch(ge, g_st));  // warm
  CK(cudaStreamSynchronize(g_st));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  std::vector<float> ms;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(a, g_st));
    CK(cudaGraphLaunch(ge, g_st));
    CK(cudaEventRecord(b, g_st));
    CK(cudaEventSynchronize(b));
    float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m);
  }
  std::sort(ms.begin(), ms.end());
  CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(gr)); CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms[ms.size() / 2] / n;
}
// plain (non-graph) launch loop: n launches back to back, per-launch ms
template <class F>
static double time_plain(F&& launch, int n = 200, int reps = 3) {
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  for (int i = 0; i < 20; i++) launch(i, g_st);
  CK(cudaStreamSynchronize(g_st));
  std::vector<float> ms;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(a, g_st));
    for (int i = 0; i < n; i++) launch(i, g_st);
    CK(cudaEventRecord(b, g_st));
    CK(cudaEventSynchronize(b));
    float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m);
  }
  std::sort(ms.begin(), ms.end());
  CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
  return ms[ms.size() / 2] / n;
}

// ------------------------------------------------------------------------------------------------------------------------------ small host helpers
static uint16_t f2bf_h(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  u += 0x7fff + ((u >> 16) & 1);
  return (uint16_t)(u >> 16);
}
static float bf2f_h(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static void* dmalloc(size_t b) { void* p; CK(cudaMalloc(&p, b)); return p; }
static std::vector<char> read_file(const std::string& p) {
  FILE* f = fopen(p.c_str(), "rb");
  if (!f) { fprintf(stderr, "cannot open %s (run export_weights.py once)\n", p.c_str()); exit(3); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  std::vector<char> v(n);
  if (fread(v.data(), 1, n, f) != (size_t)n) exit(3);
  fclose(f);
  return v;
}

// ------------------------------------------------------------------------------------------------------------------------------ latency floors and stream bandwidth
__global__ void k_empty() {}
__global__ void k_stream(const uint4* __restrict__ p, size_t n, uint32_t* out) {
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = __ldg(p + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
static void measure_floors() {
  const double lg1 = time_graph([&](int, cudaStream_t s) { k_empty<<<1, 32, 0, s>>>(); }, 200, 3) * 1e3;
  const double lgf = time_graph([&](int, cudaStream_t s) { k_empty<<<656, 256, 0, s>>>(); }, 200, 3) * 1e3;
  const double lp1 = time_plain([&](int, cudaStream_t s) { k_empty<<<1, 32, 0, s>>>(); }, 500) * 1e3;
  const double lpf = time_plain([&](int, cudaStream_t s) { k_empty<<<656, 256, 0, s>>>(); }, 500) * 1e3;
  const size_t bytes = (size_t)512 << 20;
  uint4* buf = (uint4*)dmalloc(bytes);
  CK(cudaMemset(buf, 1, bytes));
  uint32_t* o = (uint32_t*)dmalloc(4);
  const double ms = time_graph([&](int, cudaStream_t s) { k_stream<<<82 * 8, 512, 0, s>>>(buf, bytes / 16, o); }, 10, 3);
  const double gbs = bytes / ms * 1e-6;
  printf("[floors] empty kernel 1x32 : graph %.2f us/launch, plain %.2f us/launch | empty 656x256: graph %.2f us, plain %.2f us | stream-read 512MB: %.1f GB/s (%.2f ms)\n", lg1, lp1, lgf, lpf, gbs, ms);
  jm.rec("{\"what\":\"floors\",\"empty_1x32_graph_us\":%.3f,\"empty_1x32_plain_us\":%.3f,\"empty_656x256_graph_us\":%.3f,\"empty_656x256_plain_us\":%.3f,\"stream_read_GBps\":%.1f}", lg1, lp1, lgf, lpf, gbs);
  cudaFree(buf); cudaFree(o);
}

// ================================================================================================================================ (i) sparse W4 GEMV
constexpr int SP_MAXG = 136;  // K / 128 for K = 17408
// k_w4 (gemv.cu, RP = 1) with the (tile, group) stream replaced by the kept-group list.  lst[0] = number of kept groups nk; lst[1..nk] = group | chunkmask << 16, ascending.
// chunkmask bit t = 32-column chunk t of the group is kept (lane t of every quad owns chunk t, see gemv.cu); CM = false: whole 128-column groups only.
// K-split warp ks takes list entries ks, ks + KS, ...; every K-split warp of a tile gets ceil / floor(nk / KS) groups (imbalance at small nk: choose KS per nk).
template <int NW, int KS, int U, int MINB, bool CM>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_w4s(const uint4* __restrict__ W, const uint32_t* __restrict__ S32, const uint4* __restrict__ X, const smk_epi epi, int N, int K, int T, const uint32_t* __restrict__ lst) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4];
  __shared__ uint32_t sl[SP_MAXG];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7;
  // one dependent global round trip before the first weight load: count and (up to SP_MAXG) entries are fetched together, unconditionally (the list buffer is >= SP_MAXG + 1 words)
  const int nk = min((int)lst[0], nkg);
  for (int i = threadIdx.x; i < SP_MAXG; i += NW * 32) sl[i] = lst[1 + i];
  __syncthreads();
  const int per = nk > ks ? (nk - ks + KS - 1) / KS : 0;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const unsigned gridsz = gridDim.x * TPC;
  const unsigned tile0 = (unsigned)tin + blockIdx.x * TPC;
  if constexpr (KS > 1) {
    if (per == 0) {  // no group for this K-split warp: only take part in the tile-end reductions
      for (int cm = 0; cm < cnt; cm++) {
#pragma unroll
        for (int c = 0; c < 4; c++) red[warp][lane][c] = 0.f;
        __syncthreads();
        __syncthreads();
      }
      return;
    }
  }
  const uint4* xrow = X + (size_t)(g < T ? g : T - 1) * (K >> 3) + t * 4;
  unsigned tile = tile0;
  int lk = 0;
  uint4 wa[U], wb[U];
  uint32_t ss[U], ee[U];
  auto issue = [&](int j) {
    const uint32_t e = sl[ks + KS * lk];
    const size_t tw = (size_t)tile * nkg + (e & 0xFFFFu);
    ee[j] = e;
    if (!CM || ((e >> (16 + t)) & 1u)) {
      wa[j] = ldw(W + tw * 64 + lane); wb[j] = ldw(W + tw * 64 + lane + 32);
    } else {
      wa[j] = wb[j] = make_uint4(0x88888888u, 0x88888888u, 0x88888888u, 0x88888888u);  // nibble 8 = weight 0
    }
    ss[j] = ld32(S32 + tw * 8 + g);
    if (++lk == per) { lk = 0; tile += gridsz; }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        const uint32_t sv = ss[j], ej = ee[j];
        if (f + j + U < total) issue(j);
        uint4 xb[4];
        const uint4* xp = xrow + (size_t)(ej & 0xFFFFu) * 16;
        const bool on = !CM || ((ej >> (16 + t)) & 1u);
#pragma unroll
        for (int u = 0; u < 4; u++) xb[u] = on ? ldx(xp + u) : make_uint4(0, 0, 0, 0);
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t wra[4] = {ca.x, ca.y, ca.z, ca.w}, wrb[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[4], pb[4];
          dq_word(wra[u], pa);
          dq_word(wrb[u], pb);
          mma16816(part, pa[0], pb[0], pa[1], pb[1], xb[u].x, xb[u].y);
          mma16816(part, pa[2], pb[2], pa[3], pb[3], xb[u].z, xb[u].w);
        }
        const float fsa = bflo(sv), fsb = bfhi(sv);
        acc[0] = fmaf(fsa, part[0], acc[0]); acc[1] = fmaf(fsa, part[1], acc[1]);
        acc[2] = fmaf(fsb, part[2], acc[2]); acc[3] = fmaf(fsb, part[3], acc[3]);
        if (++ct == per) {
          ct = 0;
          const int row = (int)(tile0 + (unsigned)cm * gridsz) << 4;
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int c = 0; c < 4; c++) red[warp][lane][c] = acc[c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int q = 1; q < KS; q++)
#pragma unroll
                for (int c = 0; c < 4; c++) acc[c] += red[warp + q][lane][c];
              epilogue(acc, row, g, t, T, N, epi);
            }
            __syncthreads();
          } else {
            epilogue(acc, row, g, t, T, N, epi);
          }
          acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
      }
    }
  }
}

template <int NW, int KS, int U, int MINB, bool CM>
static cudaError_t run_w4s(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, const uint32_t* lst, cudaStream_t st) {
  constexpr int TPC = NW / KS;
  if (((N >> 4) % TPC) || (K & 127) || (N & 15) || T < 1 || T > 8 || (K >> 7) > SP_MAXG) return cudaErrorInvalidValue;
  static int gn = -1, gg = 0;
  const int groups = (N >> 4) / TPC;
  if (gn != N) { gg = grid_for(k_w4s<NW, KS, U, MINB, CM>, NW * 32, groups); gn = N; }
  k_w4s<NW, KS, U, MINB, CM><<<gg, NW * 32, 0, st>>>((const uint4*)w, (const uint32_t*)s, (const uint4*)x, *epi, N, K, T, lst);
  return cudaGetLastError();
}
// the engine's six configs (gemv.cu w4_cfg): 0 w8k4u2, 1 w8k8u2, 2 w8k2u3, 3 w4k4u3, 4 w16k8u2, 5 w8k1u3
// 6..9 are sparse-only additions (deeper prefetch / fewer K-split warps for small kept counts)
constexpr int NCFG = 10;
static const char* cfg_name[NCFG] = {"w8k4u2", "w8k8u2", "w8k2u3", "w4k4u3", "w16k8u2", "w8k1u3", "w8k2u4", "w8k4u3", "w8k1u5", "w4k2u4"};
template <bool CM>
static cudaError_t sp_cfg(int cfg, const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, const uint32_t* lst, cudaStream_t st) {
  switch (cfg) {
    case 0: return run_w4s<8, 4, 2, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 1: return run_w4s<8, 8, 2, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 2: return run_w4s<8, 2, 3, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 3: return run_w4s<4, 4, 3, 4, CM>(w, s, x, N, K, T, epi, lst, st);
    case 4: return run_w4s<16, 8, 2, 1, CM>(w, s, x, N, K, T, epi, lst, st);
    case 5: return run_w4s<8, 1, 3, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 6: return run_w4s<8, 2, 4, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 7: return run_w4s<8, 4, 3, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 8: return run_w4s<8, 1, 5, 2, CM>(w, s, x, N, K, T, epi, lst, st);
    case 9: return run_w4s<4, 2, 4, 4, CM>(w, s, x, N, K, T, epi, lst, st);
  }
  return cudaErrorInvalidValue;
}

// ------------------------------------------------------------------------------------------------------------------------------ (ii) fused score + select
// Grid = M CTAs (one per activation row, 128 threads), then a ticket: the last CTA to finish builds the union list.
//   per row: score(block) = sum of squares over the 128-column block (a half-warp reads one block: coalesced 256 B + 16-lane xor reduction), then (mode 0) warp 0 finds the
//            kkeep-th largest score by a 32-step bit search over the float bits (ballot counts; ties may keep a few extra blocks), writes the row's block bitmap, and zeroes the
//            dropped (row, block) slices of x into xz (mode 0 only: the GEMV reads xz);
//   last CTA: mode 0: list = union over rows of the bitmaps; mode 1: scores are summed over rows, one kkeep-best bitmap shared by all rows (the GEMV reads x itself, no xz);
//            list = ascending block ids (ballot compaction), then resets the ticket (graph-safe, the scratch must be zero once).
// ws: 4 KB-aligned scratch of SEL_WS_BYTES, zero-initialised once.  K % 512 == 0, nb <= 160.
constexpr int SEL_MAXB = SP_MAXG;
constexpr size_t SEL_WS_BYTES = 8192;
__device__ __forceinline__ uint32_t sel_threshold(const uint32_t (&kv)[5], int k) {  // largest T with #(kv >= T) >= k (kv = 0 for padding lanes; candidate keys are never 0)
  uint32_t T = 0;
#pragma unroll
  for (int bit = 31; bit >= 0; bit--) {
    const uint32_t cand = T | (1u << bit);
    int c = 0;
#pragma unroll
    for (int i = 0; i < 5; i++) c += (kv[i] >= cand);
    if (__reduce_add_sync(0xffffffffu, c) >= k) T = cand;
  }
  return T;
}
__global__ void __launch_bounds__(128) k_select(const uint16_t* __restrict__ X, int M, int K, int kkeep, int mode, uint32_t* __restrict__ lst, uint16_t* __restrict__ xz, uint32_t* __restrict__ ws) {
  __shared__ float sc[SEL_MAXB + 24];
  __shared__ uint32_t bm[8];
  __shared__ int is_last;
  uint32_t* ticket = ws;
  uint32_t* gbm = ws + 16;                         // [8][8] row bitmaps
  float* gsc = reinterpret_cast<float*>(ws + 128);  // [8][SEL_MAXB + 24] row scores
  const int r = blockIdx.x, tid = threadIdx.x, w4 = tid >> 5, lane = tid & 31, half = lane >> 4, c16 = lane & 15;
  const int nb = K >> 7;
  const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)r * K);
#pragma unroll 4
  for (int pi = w4; pi < (nb + 1) / 2; pi += 4) {
    const int b = 2 * pi + half;
    float s = 0.f;
    if (b < nb) {
      const uint4 v = __ldg(xr + (size_t)b * 16 + c16);
      const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int q = 0; q < 4; q++) { const float a = bflo(w[q]), c = bfhi(w[q]); s = fmaf(a, a, fmaf(c, c, s)); }
    }
    s += __shfl_xor_sync(0xffffffffu, s, 1); s += __shfl_xor_sync(0xffffffffu, s, 2);
    s += __shfl_xor_sync(0xffffffffu, s, 4); s += __shfl_xor_sync(0xffffffffu, s, 8);
    if (c16 == 0 && b < nb) sc[b] = s;
  }
  __syncthreads();
  if (mode == 0) {
    if (w4 == 0) {
      uint32_t kv[5];
#pragma unroll
      for (int i = 0; i < 5; i++) { const int b = lane + 32 * i; kv[i] = b < nb ? __float_as_uint(sc[b]) : 0u; }
      const uint32_t T = sel_threshold(kv, kkeep);
#pragma unroll
      for (int i = 0; i < 5; i++) {
        const uint32_t bal = __ballot_sync(0xffffffffu, lane + 32 * i < nb && kv[i] >= T);
        if (lane == 0) bm[i] = bal;
      }
    }
    __syncthreads();
    if (tid < 5) gbm[r * 8 + tid] = bm[tid];
    uint4* zr = reinterpret_cast<uint4*>(xz + (size_t)r * K);
#pragma unroll 4
    for (int pi = w4; pi < (nb + 1) / 2; pi += 4) {  // zero the dropped slices of this row (half-warp per block)
      const int b = 2 * pi + half;
      if (b < nb) zr[(size_t)b * 16 + c16] = ((bm[b >> 5] >> (b & 31)) & 1u) ? __ldg(xr + (size_t)b * 16 + c16) : make_uint4(0, 0, 0, 0);
    }
  } else {
    for (int b = tid; b < nb; b += 128) gsc[r * (SEL_MAXB + 24) + b] = sc[b];
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) is_last = (atomicAdd(ticket, 1u) == (uint32_t)(M - 1));
  __syncthreads();
  if (!is_last) return;
  __threadfence();
  if (w4 != 0) return;
  uint32_t ubm[5];
  if (mode == 0) {
#pragma unroll
    for (int i = 0; i < 5; i++) {
      uint32_t o = 0;
      for (int rr = 0; rr < M; rr++) o |= __ldcg(gbm + rr * 8 + i);
      ubm[i] = o;
    }
  } else {
    uint32_t kv[5];
#pragma unroll
    for (int i = 0; i < 5; i++) {
      const int b = lane + 32 * i;
      float a = 0.f;
      if (b < nb) for (int rr = 0; rr < M; rr++) a += __ldcg(gsc + rr * (SEL_MAXB + 24) + b);
      kv[i] = b < nb ? __float_as_uint(a) : 0u;
    }
    const uint32_t T = sel_threshold(kv, kkeep);
#pragma unroll
    for (int i = 0; i < 5; i++) ubm[i] = __ballot_sync(0xffffffffu, lane + 32 * i < nb && kv[i] >= T);
  }
  int base = 0;
#pragma unroll
  for (int i = 0; i < 5; i++) {
    const bool u = (ubm[i] >> lane) & 1u;
    const uint32_t bal = __ballot_sync(0xffffffffu, u);
    if (u) lst[1 + base + __popc(bal & ((1u << lane) - 1u))] = (uint32_t)(lane + 32 * i) | (0xFu << 16);
    base += __popc(bal);
  }
  if (lane == 0) { lst[0] = (uint32_t)base; *ticket = 0; }
}

#include "kern_gemv.cuh"
#include "kern_attn.cuh"
