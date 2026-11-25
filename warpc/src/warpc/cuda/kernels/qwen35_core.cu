// Stage 2 families 1-3 of the Qwen3.5 hybrid: the zero-centred RMSNorm, the T = 1 bf16 lm_head GEMV and the small a|b GEMV.
// Each kernel's floating-point order is its spec (it is the order the hand engine, libswiftmk (v0.12.0), uses, so that the
// swiftmk and warpc kernels can be swapped task by task with bit-identical results); the launch structure is free.
#include "dense_common.cuh"

namespace {

__device__ __forceinline__ float bflo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
// weights stream through L2 only (L1 keeps the activations); activations come through the read-only path
__device__ __forceinline__ uint4 ldw(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldx(const uint4* p) { return __ldg(p); }

// ---------------------------------------------------------------------------------------------- zero-centred RMSNorm
// One block per row of H = 8 * nch elements, nch threads rounded up to a warp (H <= 8192), one 16-byte chunk per thread:
//   y != NULL: h = bf16(h + y) in place;  ss = fma chain of the thread's 8 squares, butterfly over the warp (xor 16..1),
//   the warp sums added in warp order;  r = 1 / sqrt(ss / H + eps);  xn = bf16((h * r) * (1 + w)).
__global__ void k_rmsnorm_zc(uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w, uint4* __restrict__ xn, int H, float eps) {
  __shared__ float wsum[32];
  const int nch = H >> 3;
  const int c = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * nch;
  float v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  float ss = 0.f;
  uint4 wv = make_uint4(0, 0, 0, 0);
  if (c < nch) wv = w[c];  // independent of h and y: issued first, its latency hides behind the reduction
  if (c < nch) {
    const uint4 hv = h[base + c];
    v[0] = bflo(hv.x); v[1] = bfhi(hv.x); v[2] = bflo(hv.y); v[3] = bfhi(hv.y);
    v[4] = bflo(hv.z); v[5] = bfhi(hv.z); v[6] = bflo(hv.w); v[7] = bfhi(hv.w);
    if (y) {
      const uint4 yv = y[base + c];
      v[0] = rb(v[0] + bflo(yv.x)); v[1] = rb(v[1] + bfhi(yv.x)); v[2] = rb(v[2] + bflo(yv.y)); v[3] = rb(v[3] + bfhi(yv.y));
      v[4] = rb(v[4] + bflo(yv.z)); v[5] = rb(v[5] + bfhi(yv.z)); v[6] = rb(v[6] + bflo(yv.w)); v[7] = rb(v[7] + bfhi(yv.w));
      h[base + c] = make_uint4(pack2(v[0], v[1]), pack2(v[2], v[3]), pack2(v[4], v[5]), pack2(v[6], v[7]));
    }
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
    xn[base + c] = make_uint4(pack2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                              pack2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
  }
}

// ---------------------------------------------------------------------------------------------- bf16 head GEMV, one row
// out[n] = sum_k W[n][k] x[k] (fp32, unrounded). A warp owns R consecutive rows; per 256-column tile a lane owns the 8 columns
// 8 * (32 * tile + lane) .. +7 of every row. Per row and tile: s0 = the acc chain through the even columns (fma, in column
// order), s1 = a fresh chain through the odd columns, acc = s0 + s1; after the last tile the 32 lane accumulators are summed by
// the xor butterfly 16..1. The grid / prefetch structure is free: persistent warps with a flat (task, tile) stream and U tiles
// of weights in flight.
template <int NW, int R, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) k_head(const uint4* __restrict__ W, const uint4* __restrict__ XA, float* __restrict__ out, int N, int K) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * NW + warp, Wn = gridDim.x * NW;
  const int tiles = K >> 8, rowu = K >> 3;
  const int ntasks = N / R;
  const int mytasks = gw < ntasks ? (ntasks - gw + Wn - 1) / Wn : 0;
  const int total = mytasks * tiles;
  uint4 wr[U][R];
  int lm = 0, lt = 0;
  auto issue = [&](int j) {
    const size_t row0 = (size_t)(gw + (size_t)lm * Wn) * R;
#pragma unroll
    for (int r = 0; r < R; r++) wr[j][r] = ldw(W + (row0 + r) * rowu + (size_t)lt * 32 + lane);
    if (++lt == tiles) { lt = 0; ++lm; }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  uint4 an = make_uint4(0, 0, 0, 0);
  if (total > 0) an = ldx(XA + lane);
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        uint4 w[R];
#pragma unroll
        for (int r = 0; r < R; r++) w[r] = wr[j][r];
        if (f + j + U < total) issue(j);
        const uint4 a = an;
        if (f + j + 1 < total) an = ldx(XA + (size_t)(ct + 1 == tiles ? 0 : ct + 1) * 32 + lane);
        const float a0 = bflo(a.x), a1 = bfhi(a.x), a2 = bflo(a.y), a3 = bfhi(a.y), a4 = bflo(a.z), a5 = bfhi(a.z), a6 = bflo(a.w), a7 = bfhi(a.w);
#pragma unroll
        for (int r = 0; r < R; r++) {
          float s0 = acc[r], s1 = 0.f;
          s0 = fmaf(bflo(w[r].x), a0, s0); s1 = fmaf(bfhi(w[r].x), a1, s1);
          s0 = fmaf(bflo(w[r].y), a2, s0); s1 = fmaf(bfhi(w[r].y), a3, s1);
          s0 = fmaf(bflo(w[r].z), a4, s0); s1 = fmaf(bfhi(w[r].z), a5, s1);
          s0 = fmaf(bflo(w[r].w), a6, s0); s1 = fmaf(bfhi(w[r].w), a7, s1);
          acc[r] = s0 + s1;
        }
        if (++ct == tiles) {
          ct = 0;
          const size_t row0 = (size_t)(gw + (size_t)cm * Wn) * R;
#pragma unroll
          for (int r = 0; r < R; r++) {
            const float tot = warp_sum(acc[r]);
            if (lane == r) out[row0 + r] = tot;
            acc[r] = 0.f;
          }
          ++cm;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------- small bf16 GEMV
// One 128-thread block per weight row, T <= 8 rows of x. Thread t of the block owns the 16-byte chunks t, t + 128, ... of the
// row, one fma chain per token through them (each chunk's 8 columns in order); then per token the xor butterfly over the warp
// and the four warp sums added as (w0 + w1) + (w2 + w3). Output bf16: rows < na to out_a [T][na], the others to out_b.
__global__ void __launch_bounds__(128, 8) k_small(const uint4* __restrict__ W, const uint4* __restrict__ X, int N, int na, int K, int T, bf16* __restrict__ out_a, bf16* __restrict__ out_b) {
  __shared__ float red[4][8];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x;
  const int nch = K >> 3;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
  const uint4* wr = W + (size_t)row * nch;
#pragma unroll 5
  for (int c = threadIdx.x; c < nch; c += 128) {
    const uint4 w = ldw(wr + c);
    const float w0 = bflo(w.x), w1 = bfhi(w.x), w2 = bflo(w.y), w3 = bfhi(w.y), w4 = bflo(w.z), w5 = bfhi(w.z), w6 = bflo(w.w), w7 = bfhi(w.w);
#pragma unroll
    for (int tk = 0; tk < 8; tk++) {
      if (tk < T) {
        const uint4 x = ldx(X + (size_t)tk * nch + c);
        float s = acc[tk];
        s = fmaf(w0, bflo(x.x), s); s = fmaf(w1, bfhi(x.x), s); s = fmaf(w2, bflo(x.y), s); s = fmaf(w3, bfhi(x.y), s);
        s = fmaf(w4, bflo(x.z), s); s = fmaf(w5, bfhi(x.z), s); s = fmaf(w6, bflo(x.w), s); s = fmaf(w7, bfhi(x.w), s);
        acc[tk] = s;
      }
    }
  }
#pragma unroll
  for (int tk = 0; tk < 8; tk++) {
    if (tk < T) {
      const float s = warp_sum(acc[tk]);
      if (lane == 0) red[warp][tk] = s;
    }
  }
  __syncthreads();
  if (threadIdx.x < T) {
    const int tk = threadIdx.x;
    const float s = (red[0][tk] + red[1][tk]) + (red[2][tk] + red[3][tk]);
    if (row < na)
      out_a[(size_t)tk * na + row] = tb(s);
    else
      out_b[(size_t)tk * (N - na) + (row - na)] = tb(s);
  }
}

int sm_count() {
  static int n = 0;
  if (!n) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
  }
  return n;
}

// persistent grids: every warp must do the same amount of work, so use the largest divisor of `groups` that fits the resident
// capacity (unless that leaves more than 40% of the capacity idle)
template <class Kn>
int grid_for(Kn kern, int threads, int groups) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0);
  if (occ < 1) occ = 1;
  const int cap = groups < occ * sm_count() ? groups : occ * sm_count();
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

}  // namespace

extern "C" {

int warpc_rmsnorm_zc(void* h, const void* y, const void* w, void* xn, int T, int H, float eps, cudaStream_t st) {
  if ((H & 7) || H > 8192 || T < 1) return cudaErrorInvalidValue;
  k_rmsnorm_zc<<<T, ((H >> 3) + 31) & ~31, 0, st>>>(reinterpret_cast<uint4*>(h), reinterpret_cast<const uint4*>(y), reinterpret_cast<const uint4*>(w),
                                                   reinterpret_cast<uint4*>(xn), H, eps);
  return cudaGetLastError();
}

int warpc_head_gemv(const void* w, const void* x, int N, int K, float* out, cudaStream_t st) {
  if ((K & 255) || (N & 3)) return cudaErrorInvalidValue;
  static int gn = -1, gg = 0;
  if (gn != N) {
    gg = grid_for(k_head<8, 4, 4, 2>, 256, (N / 4 + 7) / 8);
    gn = N;
  }
  k_head<8, 4, 4, 2><<<gg, 256, 0, st>>>(reinterpret_cast<const uint4*>(w), reinterpret_cast<const uint4*>(x), out, N, K);
  return cudaGetLastError();
}

int warpc_gemv_small(const void* w, const void* x, int N, int na, int K, int T, void* out_a, void* out_b, cudaStream_t st) {
  if ((K & 7) || T < 1 || T > 8) return cudaErrorInvalidValue;
  k_small<<<N, 128, 0, st>>>(reinterpret_cast<const uint4*>(w), reinterpret_cast<const uint4*>(x), N, na, K, T, reinterpret_cast<bf16*>(out_a),
                             reinterpret_cast<bf16*>(out_b));
  return cudaGetLastError();
}

}  // extern "C"
