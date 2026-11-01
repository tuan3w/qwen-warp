// lm_head byte-reduction speed model: CUDA-core int8 per-row GEMV prototype (T = 1..4) vs the shipped bf16 head (smk_bf16_gemv_f32) and the int4 g128
// head (smk_w4_gemv, SMK_EPI_F32), N = 248320, K = 5120, L2-cold.   Clean-room; does not touch src/gemv.cu.
//
// build: nvcc -O3 -arch=sm_86 -std=c++17 -I<engine>/include bench_head.cu -L<engine>/build -lswiftmk -Xlinker -rpath=<engine>/build -o runs/bench_head
// run  : runs/bench_head [reps]       (needs the GPU lease)
//
// int8 kernel: weights int8 [N][K] (codes, signed) + fp32 scale [N]; x bf16 [T][K]; out fp32 [T][N] = scale[n] * sum_k q[n][k] * x[t][k].
// One warp = two rows (x fragments reused from smem); lane l reads uint4 (16 columns) per 512-column step, 10 steps; int8 -> fp32 by the
// magic-number trick (xor 0x80 per byte, PRMT into the mantissa of 2^23, subtract 2^23 + 128): 1 PRMT + 1 FADD per weight, no I2F.
// x lives in smem in a lane-permuted fp32 layout so every LDS.128 is conflict-free.  Persistent grid (smem filled once per CTA).
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include "engine.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int N = SMK_VOCAB, K = SMK_HIDDEN;
constexpr int NIT = K / 512;  // 10

__device__ __forceinline__ float i8_to_f(uint32_t w_xor, int e) {
  uint32_t u = __byte_perm(w_xor, 0x4B000000u, 0x7440 | e);
  return __uint_as_float(u) - 8388736.0f;
}

template <int T, int WARPS>
__global__ void __launch_bounds__(WARPS * 32, 1) head_i8(const uint4* __restrict__ w, const float* __restrict__ sc, const __nv_bfloat16* __restrict__ x, float* __restrict__ out) {
  extern __shared__ float xs[];  // [T][K] lane-permuted
  for (int c = threadIdx.x; c < K; c += WARPS * 32) {
    int it = c / 512, r = c % 512, lane = r / 16, j = (r % 16) / 4, e = r % 4;
#pragma unroll
    for (int t = 0; t < T; ++t) xs[t * K + ((it * 4 + j) * 32 + lane) * 4 + e] = __bfloat162float(x[t * K + c]);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pairs = N / 2;
  for (int pr = blockIdx.x * WARPS + warp; pr < pairs; pr += gridDim.x * WARPS) {
    const uint4* w0 = w + (size_t)(2 * pr) * (K / 16);
    const uint4* w1 = w0 + K / 16;
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
          for (int t = 0; t < T; ++t) xv[t] = *reinterpret_cast<const float4*>(xs + t * K + ((it * 4 + j) * 32 + lane) * 4);
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
      out[(size_t)lane * N + 2 * pr] = v0 * sc[2 * pr];
      out[(size_t)lane * N + 2 * pr + 1] = v1 * sc[2 * pr + 1];
    }
  }
}

template <int T, int WARPS>
static void launch_i8(const void* w, const float* sc, const void* x, float* out, int nsm, cudaStream_t st) {
  size_t smem = (size_t)T * K * sizeof(float);
  static bool init = false;
  if (!init) { CK(cudaFuncSetAttribute(head_i8<T, WARPS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)); init = true; }
  head_i8<T, WARPS><<<nsm, WARPS * 32, smem, st>>>((const uint4*)w, sc, (const __nv_bfloat16*)x, out);
}

static void run_i8(int T, const void* w, const float* sc, const void* x, float* out, int nsm, cudaStream_t st) {
  switch (T) {
    case 1: launch_i8<1, 16>(w, sc, x, out, nsm, st); break;
    case 2: launch_i8<2, 16>(w, sc, x, out, nsm, st); break;
    case 3: launch_i8<3, 8>(w, sc, x, out, nsm, st); break;
    case 4: launch_i8<4, 8>(w, sc, x, out, nsm, st); break;
  }
}

__global__ void naive_i8(const int8_t* w, const float* sc, const __nv_bfloat16* x, float* out, int T) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  for (int t = 0; t < T; ++t) {
    double a = 0;
    for (int k = 0; k < K; ++k) a += (double)w[(size_t)n * K + k] * (double)__bfloat162float(x[t * K + k]);
    out[(size_t)t * N + n] = (float)(a * sc[n]);
  }
}

__global__ void fill_u32(uint32_t* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)i * 2654435761u ^ seed;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  p[i] = h;
}
__global__ void fill_bf16(__nv_bfloat16* p, size_t n, uint32_t seed) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t h = (uint32_t)i * 2654435761u ^ seed;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
  p[i] = __float2bfloat16(((h & 0xffff) / 65536.f - 0.5f) * 2.f);
}
__global__ void fill_f32(float* p, size_t n, float v) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (i < n) p[i] = v * (1.f + 1e-3f * (i % 7));
}


// ---- exact-rescore overhead: top-M candidates of the quantised logits, then exact bf16 dot of each candidate row.
constexpr int SEL_NB = 64, SEL_PER = 16;  // stage 1: SEL_NB blocks per row, 256 threads x 16 elements (>= N / 64 = 3880)
template <int M>
__global__ void __launch_bounds__(256) select1(const float* __restrict__ lg, float* __restrict__ cv, int* __restrict__ ci) {
  const int t = blockIdx.y, b = blockIdx.x, tid = threadIdx.x;
  const int slice = (N + SEL_NB - 1) / SEL_NB, base = b * slice;
  float v[SEL_PER];
#pragma unroll
  for (int i = 0; i < SEL_PER; ++i) { int g = base + i * 256 + tid; v[i] = (i * 256 + tid < slice && g < N) ? lg[(size_t)t * N + g] : -INFINITY; }
  __shared__ float sv[8];
  __shared__ int si[8];
  for (int m = 0; m < M; ++m) {
    float bv = v[0]; int bi = 0;
#pragma unroll
    for (int i = 1; i < SEL_PER; ++i) if (v[i] > bv) { bv = v[i]; bi = i; }
    int gi = base + bi * 256 + tid;
    float wv = bv; int wi = gi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float ov = __shfl_xor_sync(0xffffffffu, wv, o); int oi = __shfl_xor_sync(0xffffffffu, wi, o); if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; } }
    if ((tid & 31) == 0) { sv[tid >> 5] = wv; si[tid >> 5] = wi; }
    __syncthreads();
    float fv = sv[0]; int fi = si[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) if (sv[w] > fv || (sv[w] == fv && si[w] < fi)) { fv = sv[w]; fi = si[w]; }
    if (tid == 0) { cv[((size_t)t * SEL_NB + b) * M + m] = fv; ci[((size_t)t * SEL_NB + b) * M + m] = fi; }
    if (gi == fi) v[bi] = -INFINITY;
    __syncthreads();
  }
}
template <int M>
__global__ void __launch_bounds__(1024) select2(const float* __restrict__ cv, const int* __restrict__ ci, int* __restrict__ out_idx) {
  const int t = blockIdx.x, tid = threadIdx.x;  // SEL_NB * M == 1024 candidates
  float v = cv[(size_t)t * SEL_NB * M + tid];
  const int myi = ci[(size_t)t * SEL_NB * M + tid];
  __shared__ float sv[32];
  __shared__ int si[32];
  for (int m = 0; m < M; ++m) {
    float wv = v; int wi = myi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float ov = __shfl_xor_sync(0xffffffffu, wv, o); int oi = __shfl_xor_sync(0xffffffffu, wi, o); if (ov > wv || (ov == wv && oi < wi)) { wv = ov; wi = oi; } }
    if ((tid & 31) == 0) { sv[tid >> 5] = wv; si[tid >> 5] = wi; }
    __syncthreads();
    float fv = sv[0]; int fi = si[0];
    for (int w = 1; w < 32; ++w) if (sv[w] > fv || (sv[w] == fv && si[w] < fi)) { fv = sv[w]; fi = si[w]; }
    if (tid == 0) out_idx[t * M + m] = fi;
    if (myi == fi) v = -INFINITY;
    __syncthreads();
  }
}
// exact bf16 dot of candidate rows: one CTA (128 threads) per (t, m)
__global__ void __launch_bounds__(128) rescore(const __nv_bfloat16* __restrict__ w, const __nv_bfloat16* __restrict__ x, const int* __restrict__ idx, float* __restrict__ out, int M) {
  const int t = blockIdx.x / M;
  const int row = idx[blockIdx.x];
  const uint4* wr = (const uint4*)(w + (size_t)row * K);
  const uint4* xr = (const uint4*)(x + (size_t)t * K);
  float a = 0.f;
  for (int i = threadIdx.x; i < K / 8; i += 128) {
    uint4 wv = wr[i], xv = xr[i];
    const __nv_bfloat162* w2 = (const __nv_bfloat162*)&wv;
    const __nv_bfloat162* x2 = (const __nv_bfloat162*)&xv;
#pragma unroll
    for (int j = 0; j < 4; ++j) { float2 wf = __bfloat1622float2(w2[j]), xf = __bfloat1622float2(x2[j]); a = fmaf(wf.x, xf.x, fmaf(wf.y, xf.y, a)); }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
  __shared__ float s[4];
  if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = a;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = s[0] + s[1] + s[2] + s[3];
}

static float median_ms(std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <class F>
static std::vector<float> time_it(F f, void* flush, size_t flush_bytes, int reps, cudaStream_t st) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  std::vector<float> ms;
  for (int i = 0; i < reps + 3; ++i) {
    CK(cudaMemsetAsync(flush, i & 0xff, flush_bytes, st));  // evict L2
    CK(cudaEventRecord(e0, st));
    f();
    CK(cudaEventRecord(e1, st));
    CK(cudaEventSynchronize(e1));
    float t;
    CK(cudaEventElapsedTime(&t, e0, e1));
    if (i >= 3) ms.push_back(t);
  }
  return ms;
}

int main(int argc, char** argv) {
  int reps = argc > 1 ? atoi(argv[1]) : 40;
  cudaStream_t st = 0;
  cudaDeviceProp pr;
  CK(cudaGetDeviceProperties(&pr, 0));
  int nsm = pr.multiProcessorCount;
  printf("device %s, %d SMs\n", pr.name, nsm);
  void *wbf, *w8, *w4, *s4, *x, *flush;
  float *sc8, *out, *out_ref;
  size_t nb = (size_t)N * K;
  CK(cudaMalloc(&wbf, nb * 2));
  CK(cudaMalloc(&w8, nb));
  CK(cudaMalloc(&w4, nb / 2));
  CK(cudaMalloc(&s4, (size_t)N * (K / 128) * 2));
  CK(cudaMalloc(&sc8, (size_t)N * 4));
  CK(cudaMalloc(&x, (size_t)4 * K * 2));
  CK(cudaMalloc(&out, (size_t)4 * N * 4));
  CK(cudaMalloc(&out_ref, (size_t)4 * N * 4));
  size_t flush_bytes = 256u << 20;
  CK(cudaMalloc(&flush, flush_bytes));
  fill_bf16<<<(nb + 255) / 256, 256>>>((__nv_bfloat16*)wbf, nb, 1);
  fill_u32<<<(nb / 4 + 255) / 256, 256>>>((uint32_t*)w8, nb / 4, 2);
  fill_u32<<<(nb / 8 + 255) / 256, 256>>>((uint32_t*)w4, nb / 8, 3);
  fill_bf16<<<(N * (K / 128) + 255) / 256, 256>>>((__nv_bfloat16*)s4, (size_t)N * (K / 128), 4);
  fill_f32<<<(N + 255) / 256, 256>>>(sc8, N, 0.01f);
  fill_bf16<<<(4 * K + 255) / 256, 256>>>((__nv_bfloat16*)x, 4 * K, 5);
  CK(cudaDeviceSynchronize());

  // correctness of the int8 kernel vs a naive double-accumulate kernel
  for (int T = 1; T <= 4; ++T) {
    run_i8(T, w8, sc8, x, out, nsm, st);
    naive_i8<<<(N + 127) / 128, 128>>>((const int8_t*)w8, sc8, (const __nv_bfloat16*)x, out_ref, T);
    CK(cudaDeviceSynchronize());
    std::vector<float> a((size_t)T * N), b((size_t)T * N);
    CK(cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b.data(), out_ref, b.size() * 4, cudaMemcpyDeviceToHost));
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (double)(a[i] - b[i]) * (a[i] - b[i]); den += (double)b[i] * b[i]; }
    printf("int8 T=%d  rel-L2 vs naive double: %.3e\n", T, sqrt(num / den));
  }

  printf("\nL2-cold (256 MB memset between calls), median of %d, ms (GB/s of weight+scale bytes)\n", reps);
  printf("%-34s %8s %8s %8s %8s\n", "kernel", "T=1", "T=2", "T=3", "T=4");
  auto row = [&](const char* name, double bytes, auto fn) {
    printf("%-34s", name);
    for (int T = 1; T <= 4; ++T) {
      auto ms = time_it([&] { fn(T); }, flush, flush_bytes, reps, st);
      float m = median_ms(ms);
      if (m < 0) printf(" %8s", "n/a"); else printf(" %5.3f(%3.0f)", m, bytes / (m * 1e-3) / 1e9);
    }
    printf("\n");
    fflush(stdout);
  };
  row("bf16 smk_bf16_gemv_f32 (shipped)", (double)nb * 2, [&](int T) { CK(smk_bf16_gemv_f32(wbf, x, N, K, T, out, st)); });
  row("int8 per-row CUDA-core (this)", (double)nb + N * 4.0, [&](int T) { run_i8(T, w8, sc8, x, out, nsm, st); });
  smk_epi epi = {};
  epi.kind = SMK_EPI_F32;
  epi.out[0] = out;
  row("int4 g128 smk_w4_gemv EPI_F32 (def)", (double)nb / 2 + (double)N * (K / 128) * 2, [&](int T) { CK(smk_w4_gemv(w4, s4, x, N, K, T, &epi, st)); });
  for (int cfg = 0; cfg < 6; ++cfg) {
    char nm[64];
    snprintf(nm, sizeof nm, "int4 g128 smk_w4_gemv_cfg %d", cfg);
    if (smk_w4_gemv_cfg(w4, s4, x, N, K, 1, &epi, cfg, st) != cudaSuccess) { cudaGetLastError(); printf("%-34s invalid for N,K\n", nm); continue; }
    CK(cudaDeviceSynchronize());
    row(nm, (double)nb / 2 + (double)N * (K / 128) * 2, [&](int T) { if (smk_w4_gemv_cfg(w4, s4, x, N, K, T, &epi, cfg, st) != cudaSuccess) { cudaGetLastError(); } });
  }

  // ---- rescore overhead (M = 16 candidates per row): select1 + select2 + exact dots, logits already in `out`
  {
    constexpr int M = 16;
    float *cv, *ro;
    int *ci, *ix;
    CK(cudaMalloc(&cv, 4 * SEL_NB * M * 4));
    CK(cudaMalloc(&ci, 4 * SEL_NB * M * 4));
    CK(cudaMalloc(&ix, 4 * M * 4));
    CK(cudaMalloc(&ro, 4 * M * 4));
    printf("\nexact-rescore overhead, M=%d candidates/row (top-M select over fp32 logits + exact bf16 dots), L2-cold weights, ms\n", M);
    for (int T = 1; T <= 4; ++T) {
      run_i8(T, w8, sc8, x, out, nsm, st);  // logits in `out` (random data: distinct values)
      CK(cudaDeviceSynchronize());
      auto ms = time_it([&] {
        select1<M><<<dim3(SEL_NB, T), 256, 0, st>>>(out, cv, ci);
        select2<M><<<T, 1024, 0, st>>>(cv, ci, ix);
        rescore<<<T * M, 128, 0, st>>>((const __nv_bfloat16*)wbf, (const __nv_bfloat16*)x, ix, ro, M);
      }, flush, flush_bytes, reps, st);
      std::vector<float> ms1 = time_it([&] { select1<M><<<dim3(SEL_NB, T), 256, 0, st>>>(out, cv, ci); select2<M><<<T, 1024, 0, st>>>(cv, ci, ix); }, flush, flush_bytes, reps, st);
      printf("T=%d  select+rescore %.4f ms   (select only %.4f ms)\n", T, median_ms(ms), median_ms(ms1));
      // check candidate set vs host top-M on T row 0
      if (T == 1) {
        std::vector<float> h(N);
        std::vector<int> hi(M);
        CK(cudaMemcpy(h.data(), out, N * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hi.data(), ix, M * 4, cudaMemcpyDeviceToHost));
        std::vector<int> ord(N);
        for (int i = 0; i < N; ++i) ord[i] = i;
        std::partial_sort(ord.begin(), ord.begin() + M, ord.end(), [&](int a, int b) { return h[a] > h[b] || (h[a] == h[b] && a < b); });
        int ok = 0;
        for (int m = 0; m < M; ++m) ok += hi[m] == ord[m];
        printf("      select check: %d/%d match host top-M order\n", ok, M);
      }
    }
  }
  return 0;
}
