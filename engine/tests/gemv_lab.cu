// K5 GEMV lab: standalone timing harness for W4 GEMV kernel variants (layout / scale / activation / prefetch experiments).
//   nvcc -arch=sm_86 -O3 -std=c++17 -Iinclude tests/gemv_lab.cu src/gemv.cu -o build/gemv_lab ; build/gemv_lab <shape> [T] [reps]
// Variants flagged "hack" are timing-only (wrong numerics).
#include "../src/gemv.cu"
#include <algorithm>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace smk;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

namespace smk {
__device__ __forceinline__ void epi_lab(const float (&acc)[4], int row, int g, int t, int T, int N, uint16_t* o) {
  const int tk0 = 2 * t, tk1 = tk0 + 1, r0 = row + g, r1 = r0 + 8;
  if (tk0 < T) { o[(size_t)tk0 * N + r0] = f2bf(acc[0]); o[(size_t)tk0 * N + r1] = f2bf(acc[2]); }
  if (tk1 < T) { o[(size_t)tk1 * N + r0] = f2bf(acc[1]); o[(size_t)tk1 * N + r1] = f2bf(acc[3]); }
}
__device__ __forceinline__ uint4 ldw_ef(const uint4* p, uint64_t pol) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p), "l"(pol));
  return v;
}
__device__ __forceinline__ uint64_t pol_ef() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(p));
  return p;
}
struct PfDesc { const char* base; size_t stride; int count; int chunk_lines; int rows; size_t row_stride; };  // prefetch `count` chunks of chunk_lines*128 B at base + i*stride

// LAY: 0 row-major, 1 tile-contiguous units, 2 wave-interleaved units (hack).  SL: 0 u16 scales row-major, 1 packed u32 per (tile,kg,g), 2 none (hack).
// XL: 0 normal, 1 no activation loads (hack).  PF: 0 none, 1 prefetch next weights at end of issue stream.  EF: evict-first policy for weight loads.
// read-only reference: every warp streams a contiguous slice of the weight buffer (lane-contiguous 16 B loads, depth UR), xor-reduce -> 1 store
template <int UR>
__global__ void __launch_bounds__(256, 2) k_readonly(const uint4* __restrict__ W, size_t n16, uint32_t* out) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const size_t nw = (size_t)gridDim.x * 8, gw = (size_t)blockIdx.x * 8 + warp;
  const size_t per = n16 / nw / 32;  // 512 B steps per warp
  const uint4* p = W + gw * per * 32 + lane;
  uint32_t a = 0;
  uint4 r[UR];
#pragma unroll
  for (int j = 0; j < UR; j++) r[j] = ldw(p + (size_t)j * 32);
  for (size_t i = 0; i < per; i += UR) {
#pragma unroll
    for (int j = 0; j < UR; j++) {
      const uint4 v = r[j];
      if (i + j + UR < per) r[j] = ldw(p + (i + j + UR) * 32);
      a ^= v.x ^ v.y ^ v.z ^ v.w;
    }
  }
  if (a == 0x12345678u) out[0] = a;
}
__global__ void k_null(uint32_t* out) { if (threadIdx.x == 1000) out[0] = 0; }

template <int NW, int KS, int U, int MINB, int LAY, int SL, int XL, int PF, int EF, int AB = 0>
__global__ void __launch_bounds__(NW * 32, MINB) k_w4x(const uint4* __restrict__ W, const uint16_t* __restrict__ S, const uint4* __restrict__ X, uint16_t* __restrict__ out, int N, int K, int T, PfDesc pf) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS, rowu = K >> 5;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const bool tvalid = g < T;
  const uint4* xrow = X + (size_t)g * (K >> 3);
  const uint64_t pol = EF ? pol_ef() : 0;
  uint4 wa[U], wb[U];
  uint32_t sa[U], sb[U];
  int lm = 0, lk = 0, lf = 0;
  auto issue = [&](int j) {
    const int tile = ((int)blockIdx.x + lm * (int)gridDim.x) * TPC + tin;
    const int row = tile << 4;
    const int kg = ks + KS * lk;
    if (LAY == 0) {
      const size_t o = (size_t)(row + g) * rowu + (size_t)kg * 4 + t;
      if (EF) { wa[j] = ldw_ef(W + o, pol); wb[j] = ldw_ef(W + o + (size_t)8 * rowu, pol); }
      else { wa[j] = ldw(W + o); wb[j] = ldw(W + o + (size_t)8 * rowu); }
    } else {
      const size_t u = LAY == 1 ? ((size_t)tile * nkg + kg) : ((size_t)lf * gridDim.x * NW + blockIdx.x * NW + warp);
      const uint4* p = W + u * 64 + lane;
      if (EF) { wa[j] = ldw_ef(p, pol); wb[j] = ldw_ef(p + 32, pol); }
      else { wa[j] = ldw(p); wb[j] = ldw(p + 32); }
    }
    if (SL == 0) {
      sa[j] = ld16(S + (size_t)(row + g) * nkg + kg); sb[j] = ld16(S + (size_t)(row + g + 8) * nkg + kg);
    } else if (SL == 1) {
      const uint32_t v = __ldg((const uint32_t*)S + ((size_t)tile * nkg + kg) * 8 + g);
      sa[j] = v & 0xffffu; sb[j] = v >> 16;
    } else { sa[j] = 0x3f80u; sb[j] = 0x3f80u; }
    ++lf;
    if (++lk == per) { lk = 0; ++lm; }
  };
  uint4 xn[4];
  auto loadB = [&](int kk) {
    const int kg = ks + KS * kk;
#pragma unroll
    for (int u = 0; u < 4; u++) xn[u] = XL ? make_uint4(0x3f803f80u, 0x3f803f80u, 0x3f803f80u, 0x3f803f80u) : (tvalid ? perm_b(ldx(xrow + kg * 16 + t * 4 + u)) : make_uint4(0, 0, 0, 0));
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  if (total > 0) loadB(0);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int ct = 0, cm = 0;
  bool pfd = false;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        const float fsa = bflo(sa[j]), fsb = bflo(sb[j]);
        if (f + j + U < total) issue(j);
        else if (PF && !pfd) {
          pfd = true;
          const int nl = pf.count * pf.rows * pf.chunk_lines;
          const int gt = (blockIdx.x * NW + warp) * 32 + lane, gs = gridDim.x * NW * 32;
          for (int i = gt; i < nl; i += gs) {
            const int cr = i / pf.chunk_lines, l = i - cr * pf.chunk_lines;
            const int c = cr / pf.rows, r = cr - c * pf.rows;
            const char* a = pf.base + (size_t)c * pf.stride + (size_t)r * pf.row_stride + (size_t)l * 128;
            asm volatile("prefetch.global.L2 [%0];\n" :: "l"(a));
          }
        }
        uint4 xb[4];
#pragma unroll
        for (int u = 0; u < 4; u++) xb[u] = xn[u];
        if (f + j + 1 < total) loadB(ct + 1 == per ? 0 : ct + 1);
        float part[4] = {0.f, 0.f, 0.f, 0.f};
        const uint32_t wra[4] = {ca.x, ca.y, ca.z, ca.w}, wrb[4] = {cb.x, cb.y, cb.z, cb.w};
        uint32_t xr = 0;
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[4], pb[4];
          if (AB == 1) { xr ^= wra[u] ^ wrb[u] ^ xb[u].x ^ xb[u].y ^ xb[u].z ^ xb[u].w; continue; }
          if (AB == 3) { pa[0] = pa[2] = wra[u]; pa[1] = pa[3] = wrb[u]; pb[0] = pb[1] = pb[2] = pb[3] = wra[u] ^ wrb[u]; }
          else { dq_word(wra[u], pa); dq_word(wrb[u], pb); }
          if (AB == 2) { xr ^= pa[0] ^ pa[1] ^ pa[2] ^ pa[3] ^ pb[0] ^ pb[1] ^ pb[2] ^ pb[3] ^ xb[u].x ^ xb[u].y ^ xb[u].z ^ xb[u].w; continue; }
          mma16816(part, pa[0], pb[0], pa[1], pb[1], xb[u].x, xb[u].y);
          mma16816(part, pa[2], pb[2], pa[3], pb[3], xb[u].z, xb[u].w);
        }
        if (AB == 1 || AB == 2) part[0] = __uint_as_float(xr);
        acc[0] = fmaf(fsa, part[0], acc[0]); acc[1] = fmaf(fsa, part[1], acc[1]);
        acc[2] = fmaf(fsb, part[2], acc[2]); acc[3] = fmaf(fsb, part[3], acc[3]);
        if (++ct == per) {
          ct = 0;
          const int row = (((int)blockIdx.x + cm * (int)gridDim.x) * TPC + tin) << 4;
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
              epi_lab(acc, row, g, t, T, N, out);
            }
            __syncthreads();
          } else {
            epi_lab(acc, row, g, t, T, N, out);
          }
          acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
        }
      }
    }
  }
}

// packer for LAY1/SL1
__global__ void k_pack_tile(const uint4* __restrict__ W, const uint16_t* __restrict__ S, uint4* __restrict__ dW, uint32_t* __restrict__ dS, int N, int K) {
  const int nkg = K >> 7, rowu = K >> 5, ntile = N >> 4;
  const size_t total = (size_t)ntile * nkg * 32;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const int lane = i & 31;
    const size_t unit = i >> 5;
    const int kg = unit % nkg, tile = unit / nkg, g = lane >> 2, t = lane & 3;
    const size_t o = (size_t)(tile * 16 + g) * rowu + (size_t)kg * 4 + t;
    dW[unit * 64 + lane] = W[o];
    dW[unit * 64 + 32 + lane] = W[o + (size_t)8 * rowu];
    if (t == 0) dS[unit * 8 + g] = (uint32_t)S[(size_t)(tile * 16 + g) * nkg + kg] | ((uint32_t)S[(size_t)(tile * 16 + g + 8) * nkg + kg] << 16);
  }
}
}  // namespace smk

#include "lab_v2.inc"

template <class Kn>
static int grid_for_lab(Kn kern, int threads, int groups, int nsm_) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0);
  if (occ < 1) occ = 1;
  const int cap = groups < occ * nsm_ ? groups : occ * nsm_;
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

__global__ void k_fill(uint32_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u + seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16; p[i] = h;
  }
}
__global__ void k_fill16(uint16_t* p, size_t n, uint32_t seed) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = (uint32_t)i * 2654435761u + seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; p[i] = 0x3b00 + (h & 0xff);
  }
}
struct Variant {
  const char* name;
  int NW, KS, U;
  int (*grid)(int N, int nsm);
  void (*launch)(const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st);
  int lay, sl, pf;
  int v2;
};

template <int NW, int KS, int U, int MINB, int LAY, int SL, int XL, int PF, int EF, int AB = 0>
static Variant mkv(const char* name) {
  Variant v;
  v.name = name; v.NW = NW; v.KS = KS; v.U = U; v.lay = LAY; v.sl = SL; v.pf = PF;
  v.grid = [](int N, int nsm_) { return grid_for_lab(k_w4x<NW, KS, U, MINB, LAY, SL, XL, PF, EF, AB>, NW * 32, (N >> 4) / (NW / KS), nsm_); };
  v.launch = [](const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st) {
    k_w4x<NW, KS, U, MINB, LAY, SL, XL, PF, EF, AB><<<grid, NW * 32, 0, st>>>((const uint4*)w, (const uint16_t*)s, (const uint4*)x, out, N, K, T, pf);
  };
  return v;
}

template <int NW, int KS, int U, int MINB, int SM>
static Variant mkv2(const char* name) {
  Variant v = {};
  v.name = name; v.NW = NW; v.KS = KS; v.U = U; v.lay = SM ? 0 : 1; v.sl = SM ? 0 : 1; v.pf = 0; v.v2 = SM == 2 ? 3 : (SM ? 2 : 1);
  v.grid = [](int N, int nsm_) { return grid_for_lab(k_w4v2<NW, KS, U, MINB, SM>, NW * 32, (N >> 4) / (NW / KS), nsm_); };
  v.launch = [](const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st) {
    smk_epi e = {};
    e.kind = SMK_EPI_SEGS; e.end[0] = e.end[1] = e.end[2] = N; e.out[0] = out;
    k_w4v2<NW, KS, U, MINB, SM><<<grid, NW * 32, 0, st>>>((const uint4*)w, (const uint32_t*)s, (const uint4*)x, e, N, K, T);
  };
  return v;
}
template <int CFG>
static Variant mk_real(const char* name) {
  Variant v = {};
  v.name = name; v.NW = 8; v.KS = CFG == 0 ? 4 : 8; v.U = 2; v.lay = 0; v.sl = 0; v.pf = 0;
  v.grid = [](int N, int nsm_) { return 0; };
  v.launch = [](const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st) {
    smk_epi e = {};
    e.kind = SMK_EPI_SEGS; e.end[0] = e.end[1] = e.end[2] = N; e.out[0] = out;
    smk_w4_gemv_cfg(w, s, x, N, K, T, &e, CFG, st);
  };
  return v;
}
static Variant mk_readonly() {
  Variant v = {};
  v.name = "readonly-ref(w+s bytes)"; v.NW = 8; v.KS = 1; v.U = 4; v.lay = 0; v.sl = 0; v.pf = 0;
  v.grid = [](int N, int nsm_) { return 2 * nsm_; };
  v.launch = [](const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st) {
    size_t n16 = (size_t)N * K / 2 / 16;
    n16 = n16 / ((size_t)grid * 8 * 32 * 4) * ((size_t)grid * 8 * 32 * 4);
    k_readonly<4><<<grid, 256, 0, st>>>((const uint4*)w, n16, (uint32_t*)out);
  };
  return v;
}
static Variant mk_null() {
  Variant v = {};
  v.name = "null-kernel"; v.NW = 8; v.KS = 1; v.U = 4; v.lay = 0; v.sl = 0; v.pf = 0;
  v.grid = [](int N, int nsm_) { return 2 * nsm_; };
  v.launch = [](const void* w, const void* s, const void* x, uint16_t* out, int N, int K, int T, PfDesc pf, int grid, cudaStream_t st) { k_null<<<grid, 256, 0, st>>>((uint32_t*)out); };
  return v;
}
#include "lab_variants.inc"

int main(int argc, char** argv) {
  const char* shape = argc > 1 ? argv[1] : "gateup";
  const int T = argc > 2 ? atoi(argv[2]) : 1;
  const int reps = argc > 3 ? atoi(argv[3]) : 30;
  const char* only = argc > 4 ? argv[4] : nullptr;
  int N, K;
  if (!strcmp(shape, "gateup")) { N = 34816; K = 5120; }
  else if (!strcmp(shape, "down")) { N = 5120; K = 17408; }
  else if (!strcmp(shape, "qkvz")) { N = 16384; K = 5120; }
  else if (!strcmp(shape, "qkv")) { N = 14336; K = 5120; }
  else if (!strcmp(shape, "o")) { N = 5120; K = 6144; }
  else { printf("bad shape\n"); return 1; }
  int dev = 0, nsm_ = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&nsm_, cudaDevAttrMultiProcessorCount, dev));
  const size_t wbytes = (size_t)N * K / 2, sbytes = (size_t)N * K / 64;
  const double nbytes = (double)(wbytes + sbytes);
  int nb = (int)std::max<size_t>(2, std::min<size_t>(16, (400u << 20) / (size_t)nbytes + 1));
  const bool nopack = getenv("LAB_NOPACK") != nullptr;
  if (getenv("LAB_NB")) nb = atoi(getenv("LAB_NB"));
  std::vector<uint8_t*> w(nb), s(nb), pw(nb), ps(nb), qw(nb), qs(nb), pn(nb);
  for (int i = 0; i < nb; i++) {
    CK(cudaMalloc(&w[i], wbytes)); CK(cudaMalloc(&s[i], sbytes));
    k_fill<<<1024, 256>>>((uint32_t*)w[i], wbytes / 4, 1234u + i);
    k_fill16<<<256, 256>>>((uint16_t*)s[i], sbytes / 2, 77u + i);
    if (!nopack) {
      CK(cudaMalloc(&pw[i], wbytes)); CK(cudaMalloc(&ps[i], sbytes));
      k_pack_tile<<<2048, 256>>>((const uint4*)w[i], (const uint16_t*)s[i], (uint4*)pw[i], (uint32_t*)ps[i], N, K);
      CK(cudaMalloc(&pn[i], wbytes)); k_pack_perm<<<1024, 256>>>((const uint32_t*)w[i], (uint32_t*)pn[i], wbytes / 4);
      CK(cudaMalloc(&qw[i], wbytes)); CK(cudaMalloc(&qs[i], sbytes));
      k_pack_v2<<<2048, 256>>>((const uint4*)w[i], (const uint16_t*)s[i], (uint4*)qw[i], (uint32_t*)qs[i], N, K);
    }
  }
  CK(cudaDeviceSynchronize());
  uint16_t *x, *out;
  CK(cudaMalloc(&x, 4 * K * 2)); CK(cudaMalloc(&out, 4 * (size_t)N * 2));
  { std::vector<uint16_t> hx(4 * K); for (auto& v : hx) v = 0x3e00 + (rand() & 0xff); CK(cudaMemcpy(x, hx.data(), 4 * K * 2, cudaMemcpyHostToDevice)); }
  printf("# shape %s N=%d K=%d T=%d nb=%d bytes=%.1f MB\n", shape, N, K, T, nb, nbytes / 1e6);
  std::vector<Variant> vs = lab_variants();
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  struct GE { const char* name; cudaGraphExec_t ge; int grid; };
  std::vector<GE> gs;
  for (auto& v : vs) {
    if (only && !strstr(v.name, only)) continue;
    if (nopack && (v.lay || v.v2 == 1 || v.v2 == 3 || v.sl == 1)) continue;
    if (((K >> 7) % v.KS) || ((N >> 4) % (v.NW / v.KS))) continue;
    int grid = v.grid(N, nsm_);
    if (getenv("LAB_GRID")) grid = atoi(getenv("LAB_GRID"));
    const int tiles = N >> 4, tpc = v.NW / v.KS;
    const int gridtiles = grid * tpc;
    // prefetch descriptor for the "next" buffer: first-wave tiles, first `chunk` of each tile (tile-contig layout) -- row-major: row blocks
    cudaGraph_t gr; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < nb; i++) {
      const int nx = (i + 1) % nb;
      PfDesc pf = {};
      if (v.pf) {
        const uint8_t* nbase = v.lay ? pw[nx] : w[nx];
        const int chunk_lines = v.pf;  // lines of 128 B per tile
        if (v.lay) { pf.base = (const char*)nbase; pf.stride = (size_t)(K >> 7) * 1024; pf.count = std::min(tiles, gridtiles); pf.chunk_lines = chunk_lines; pf.rows = 1; pf.row_stride = 0; }
        else { pf.base = (const char*)nbase; pf.stride = (size_t)16 * (K / 2); pf.count = std::min(tiles, gridtiles); pf.chunk_lines = chunk_lines; pf.rows = 16; pf.row_stride = K / 2; }
      }
      if (v.v2 == 1) v.launch(qw[i], qs[i], x, out, N, K, T, pf, grid, st);
      else if (v.v2 == 3) v.launch(pn[i], s[i], x, out, N, K, T, pf, grid, st);
      else v.launch(v.lay ? pw[i] : w[i], v.sl == 1 ? ps[i] : s[i], x, out, N, K, T, pf, grid, st);
    }
    CK(cudaStreamEndCapture(st, &gr));
    CK(cudaGraphInstantiate(&ge, gr, 0));
    if (!nopack) {
      // correctness: v2 vs base (variant 0 named "w8k8u2 base") one-shot on buffer 0
      static std::vector<uint16_t> refo;
      static bool have = false;
      std::vector<uint16_t> h((size_t)T * N);
      CK(cudaMemset(out, 0, 4 * (size_t)N * 2));
      PfDesc z = {};
      if (v.v2 == 1) v.launch(qw[0], qs[0], x, out, N, K, T, z, grid, st);
      else if (v.v2 == 3) v.launch(pn[0], s[0], x, out, N, K, T, z, grid, st);
      else v.launch(v.lay ? pw[0] : w[0], v.sl == 1 ? ps[0] : s[0], x, out, N, K, T, z, grid, st);
      CK(cudaStreamSynchronize(st));
      CK(cudaMemcpy(h.data(), out, h.size() * 2, cudaMemcpyDeviceToHost));
      if (!have && v.v2 == 0 && v.lay == 0 && v.sl == 0 && !strstr(v.name, "hack") && !strstr(v.name, "AB")) { refo = h; have = true; }
      if (have && v.v2) {
        double num = 0, den = 0; size_t bad = 0;
        for (size_t i = 0; i < h.size(); i++) {
          auto f = [](uint16_t b) { uint32_t u = (uint32_t)b << 16; float x; memcpy(&x, &u, 4); return (double)x; };
          const double a = f(h[i]), r = f(refo[i]);
          num += (a - r) * (a - r); den += r * r;
          if (a != r) bad++;
        }
        printf("  check %s: rel-L2 vs v1 = %.3e, %zu/%zu elements differ (bf16)\n", v.name, sqrt(num / (den + 1e-30)), bad, h.size());
      }
    }
    gs.push_back({v.name, ge, grid});
  }
  {
    const double warm_ms = getenv("LAB_WARM") ? atof(getenv("LAB_WARM")) : 200.0;
    const double meas_ms = getenv("LAB_MEAS") ? atof(getenv("LAB_MEAS")) : 400.0;
    const int rounds = getenv("LAB_ROUNDS") ? atoi(getenv("LAB_ROUNDS")) : 3;
    std::vector<std::vector<float>> res(gs.size());
    cudaEvent_t w0, w1; cudaEventCreate(&w0); cudaEventCreate(&w1);
    for (int rd = 0; rd < rounds; rd++) {
      for (size_t k = 0; k < gs.size(); k++) {
        cudaEventRecord(w0, st);
        float el = 0;
        while (el < warm_ms) { for (int q = 0; q < 10; q++) CK(cudaGraphLaunch(gs[k].ge, st)); cudaEventRecord(w1, st); cudaEventSynchronize(w1); cudaEventElapsedTime(&el, w0, w1); }
        std::vector<float> ts;
        double tot = 0;
        while (tot < meas_ms) {
          cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
          cudaEventRecord(e0, st); CK(cudaGraphLaunch(gs[k].ge, st)); cudaEventRecord(e1, st); CK(cudaStreamSynchronize(st));
          float ms; cudaEventElapsedTime(&ms, e0, e1); ts.push_back(ms / nb); tot += ms;
          cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        std::sort(ts.begin(), ts.end());
        res[k].push_back(ts[ts.size() / 2]);
      }
    }
    for (size_t k = 0; k < gs.size(); k++) {
      std::vector<float> r = res[k]; std::sort(r.begin(), r.end());
      const float med = r[r.size() / 2];
      printf("%-28s grid %3d  %7.2f us (rounds %.2f..%.2f)  %6.1f GB/s\n", gs[k].name, gs[k].grid, med * 1e3, r.front() * 1e3, r.back() * 1e3, nbytes / med / 1e6);
    }
  }
  return 0;
}
