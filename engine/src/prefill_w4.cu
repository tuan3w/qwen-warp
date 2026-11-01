// swift-mk K10: W4A16 mma GEMM for short prompts (T = 1..64 rows per call).
// Same tile design / numerics as src/gemv.cu k_w4 (weights are the A operand straight from global memory, dequantised in registers, fp32 accumulate per
// 128-k group then the fp32 group scale, one bf16 rounding in the epilogue), generalised to NT n8 activation tiles (T <= 8*NT): the dequantised weight
// fragments of a 16 x 32 k slab are reused by NT x 2 mma.  Every column of the result depends only on its own activation row, so for equal (NW, KS) the output is
// bit-identical to smk_w4_gemv on the same rows (tests/test_w4mm.py).  Replaces the K6 dequant-to-bf16 + cuBLAS path where that path is dominated by its fixed
// 64 x 0.75 GB of bf16 weight traffic (prompts < ~400 tokens).
// Epilogues as in engine.h (SMK_EPI_SEGS / SMK_EPI_SILU / SMK_EPI_F32); activations bf16 [T][K] row-major, any T <= 64.
#include "common.cuh"
#include "engine.h"

namespace {
using namespace smk;

__device__ __forceinline__ float silu_m(float x) { return x / (1.f + expf(-x)); }

// acc = (row g, tok 2t), (row g, tok 2t+1), (row g+8, tok 2t), (row g+8, tok 2t+1) of the 16-row tile at `row`, tokens offset by tb
__device__ __forceinline__ void epi_mm(const float (&acc)[4], int row, int g, int t, int tb, int T, int N, const smk_epi& e) {
  const int tk0 = tb + 2 * t, tk1 = tk0 + 1;
  if (e.kind == SMK_EPI_SILU) {
    uint16_t* o = (uint16_t*)e.out[0];
    const int ch = (row >> 1) + g, W = N >> 1;
    if (tk0 < T) { const float gt = bf16r(acc[0]), up = bf16r(acc[2]); o[(size_t)tk0 * W + ch] = f2bf(bf16r(silu_m(gt)) * up); }
    if (tk1 < T) { const float gt = bf16r(acc[1]), up = bf16r(acc[3]); o[(size_t)tk1 * W + ch] = f2bf(bf16r(silu_m(gt)) * up); }
  } else if (e.kind == SMK_EPI_F32) {
    float* o = (float*)e.out[0];
    const int r0 = row + g, r1 = r0 + 8;
    if (tk0 < T) { o[(size_t)tk0 * N + r0] = acc[0]; o[(size_t)tk0 * N + r1] = acc[2]; }
    if (tk1 < T) { o[(size_t)tk1 * N + r0] = acc[1]; o[(size_t)tk1 * N + r1] = acc[3]; }
  } else {
    const int seg = row < e.end[0] ? 0 : (row < e.end[1] ? 1 : 2);
    const int start = seg == 0 ? 0 : (seg == 1 ? e.end[0] : e.end[1]), width = (seg == 0 ? e.end[0] : (seg == 1 ? e.end[1] : e.end[2])) - start;
    uint16_t* o = (uint16_t*)(seg == 0 ? e.out[0] : (seg == 1 ? e.out[1] : e.out[2]));
    const int r0 = row + g - start, r1 = r0 + 8;
    if (tk0 < T) { o[(size_t)tk0 * width + r0] = f2bf(acc[0]); o[(size_t)tk0 * width + r1] = f2bf(acc[2]); }
    if (tk1 < T) { o[(size_t)tk1 * width + r0] = f2bf(acc[1]); o[(size_t)tk1 * width + r1] = f2bf(acc[3]); }
  }
}

// NW warps/CTA, KS K-split warps per row block (smem reduce at block end), U weight prefetch depth (128-k groups), NT activation n8 tiles, MT 16-row weight tiles per warp:
// the activation fragments loaded for one k step are shared by the MT tiles (the activation traffic from L1 per mma halves for MT = 2).
// RP = 1: repacked weight layout (smk_w4r_gemv in include/engine.h / gemv.cu k_w4).
template <int MT, int NT, int NW, int KS, int U, int MINB, int RP>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_w4mm(const uint4* __restrict__ W, const void* __restrict__ S, const uint4* __restrict__ X, const smk_epi epi, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][KS > 1 ? 4 * NT * MT : 1];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS, rowu = K >> 5;
  const int groups = (N >> 4) / (TPC * MT);
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  uint32_t xoff[NT];  // activation row of this lane's B column in each n8 tile (lanes past T read row T-1: their columns are never stored)
#pragma unroll
  for (int nt = 0; nt < NT; nt++) xoff[nt] = (uint32_t)((g + 8 * nt < T ? g + 8 * nt : T - 1) * (K >> 3));
  const uint4* xbase = X + ks * 16 + t * 4;
  auto tile_of = [&](int m) { return ((int)blockIdx.x + m * (int)gridDim.x) * TPC + tin; };  // block of MT consecutive 16-row tiles

  const uint4* wp;
  const uint16_t* sp;
  const uint32_t* sp32;
  uint4 wa[U][MT], wb[U][MT];
  uint32_t ss[U][MT];
  int lm = 0, lk = 0;
  auto setp = [&](int m) {
    if (RP) {
      const size_t u0 = (size_t)tile_of(m) * (MT * nkg) + ks;
      wp = W + u0 * 64 + lane;
      sp32 = (const uint32_t*)S + u0 * 8 + g;
    } else {
      const size_t row = (size_t)tile_of(m) * (16 * MT) + g;
      wp = W + row * rowu + ks * 4 + t;
      sp = (const uint16_t*)S + row * nkg + ks;
    }
  };
  setp(0);
  auto issue = [&](int j) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      if (RP) {
        wa[j][mt] = ldw(wp + (size_t)mt * nkg * 64); wb[j][mt] = ldw(wp + (size_t)mt * nkg * 64 + 32);
        ss[j][mt] = ld32(sp32 + (size_t)mt * nkg * 8);
      } else {
        wa[j][mt] = ldw(wp + (size_t)(16 * mt) * rowu); wb[j][mt] = ldw(wp + (size_t)(16 * mt + 8) * rowu);
        ss[j][mt] = ld16(sp + (size_t)(16 * mt) * nkg) | (ld16(sp + (size_t)(16 * mt + 8) * nkg) << 16);
      }
    }
    if (++lk == per) {
      lk = 0; ++lm;
      setp(lm);
    } else if (RP) {
      wp += KS * 64;
      sp32 += KS * 8;
    } else {
      wp += KS * 4;
      sp += KS;
    }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[MT][NT][4];
#pragma unroll
  for (int mt = 0; mt < MT; mt++)
#pragma unroll
    for (int nt = 0; nt < NT; nt++) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
  int ct = 0, cm = 0;
  const uint4* xp = xbase;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        uint4 ca[MT], cb[MT];
        uint32_t sv[MT];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) { ca[mt] = wa[j][mt]; cb[mt] = wb[j][mt]; sv[mt] = ss[j][mt]; }
        if (f + j + U < total) issue(j);
        float part[MT][NT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++)
#pragma unroll
          for (int nt = 0; nt < NT; nt++) part[mt][nt][0] = part[mt][nt][1] = part[mt][nt][2] = part[mt][nt][3] = 0.f;
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[MT][4], pb[MT][4];
#pragma unroll
          for (int mt = 0; mt < MT; mt++) {
            const uint32_t wra = u == 0 ? ca[mt].x : u == 1 ? ca[mt].y : u == 2 ? ca[mt].z : ca[mt].w;
            const uint32_t wrb = u == 0 ? cb[mt].x : u == 1 ? cb[mt].y : u == 2 ? cb[mt].z : cb[mt].w;
            dq_word(wra, pa[mt]);
            dq_word(wrb, pb[mt]);
          }
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            const uint4 xb = RP ? ldx(xp + xoff[nt] + u) : perm_b(ldx(xp + xoff[nt] + u));
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
              mma16816(part[mt][nt], pa[mt][0], pb[mt][0], pa[mt][1], pb[mt][1], xb.x, xb.y);
              mma16816(part[mt][nt], pa[mt][2], pb[mt][2], pa[mt][3], pb[mt][3], xb.z, xb.w);
            }
          }
        }
        xp += KS * 16;
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
          const float fsa = bflo(sv[mt]), fsb = bfhi(sv[mt]);
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            acc[mt][nt][0] = fmaf(fsa, part[mt][nt][0], acc[mt][nt][0]); acc[mt][nt][1] = fmaf(fsa, part[mt][nt][1], acc[mt][nt][1]);
            acc[mt][nt][2] = fmaf(fsb, part[mt][nt][2], acc[mt][nt][2]); acc[mt][nt][3] = fmaf(fsb, part[mt][nt][3], acc[mt][nt][3]);
          }
        }
        if (++ct == per) {  // row block finished
          ct = 0;
          xp = xbase;
          const int row0 = tile_of(cm) * (16 * MT);
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                  for (int c = 0; c < 4; c++) red[warp][lane][4 * (mt * NT + nt) + c] = acc[mt][nt][c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int q = 1; q < KS; q++)
#pragma unroll
                for (int mt = 0; mt < MT; mt++)
#pragma unroll
                  for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int c = 0; c < 4; c++) acc[mt][nt][c] += red[warp + q][lane][4 * (mt * NT + nt) + c];
#pragma unroll
              for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++) epi_mm(acc[mt][nt], row0 + 16 * mt, g, t, 8 * nt, T, N, epi);
            }
            __syncthreads();
          } else {
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
              for (int nt = 0; nt < NT; nt++) epi_mm(acc[mt][nt], row0 + 16 * mt, g, t, 8 * nt, T, N, epi);
          }
#pragma unroll
          for (int mt = 0; mt < MT; mt++)
#pragma unroll
            for (int nt = 0; nt < NT; nt++) acc[mt][nt][0] = acc[mt][nt][1] = acc[mt][nt][2] = acc[mt][nt][3] = 0.f;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------- host
int g_nsm = 0;
int nsm() {
  if (!g_nsm) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, d); }
  return g_nsm;
}
// persistent grid: every warp must do the same amount of work -> the largest divisor of `groups` that fits in the resident capacity
template <class Kn>
int grid_for(Kn kern, int threads, int groups) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0);
  if (occ < 1) occ = 1;
  const int cap = groups < occ * nsm() ? groups : occ * nsm();
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

template <int MT, int NT, int NW, int KS, int U, int MINB, int RP>
cudaError_t run_mm(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  constexpr int TPC = NW / KS;
  if constexpr ((KS > 1 && (size_t)NW * 32 * 4 * NT * MT * 4 > 48 * 1024) || (MT == 2 && NT > 7)) {
    return cudaErrorInvalidValue;  // static smem of the K-split reduce does not fit / MT = 2 with 8 activation tiles spills registers
  } else {
  if (((K >> 7) % KS) || ((N >> 4) % (TPC * MT)) || (K & 127) || (N & 15) || T < 1 || T > 8 * NT) return cudaErrorInvalidValue;
  const int groups = (N >> 4) / (TPC * MT);
  static int cache_n[8 * 16];  // grid cache keyed by (instantiation, N): one slot per N seen, linear probe (few distinct N)
  static int cache_g[8 * 16];
  int slot = 0;
  while (slot < 8 * 16 && cache_n[slot] != 0 && cache_n[slot] != N) ++slot;
  int grid = 0;
  if (slot < 8 * 16 && cache_n[slot] == N) grid = cache_g[slot];
  else {
    grid = grid_for(k_w4mm<MT, NT, NW, KS, U, MINB, RP>, NW * 32, groups);
    if (slot < 8 * 16) { cache_n[slot] = N; cache_g[slot] = grid; }
  }
  k_w4mm<MT, NT, NW, KS, U, MINB, RP><<<grid, NW * 32, 0, st>>>((const uint4*)w, s, (const uint4*)x, *epi, N, K, T);
  return cudaGetLastError();
  }
}

template <int NT, int RP>
cudaError_t dispatch(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  switch (cfg) {
    case 0: return run_mm<1, NT, 8, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 1: return run_mm<1, NT, 8, 2, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 2: return run_mm<1, NT, 8, 1, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 3: return run_mm<1, NT, 4, 1, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 4: return run_mm<1, NT, 4, 2, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 5: return run_mm<1, NT, 8, 4, 3, 1, RP>(w, s, x, N, K, T, epi, st);
    case 6: return run_mm<1, NT, 4, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 7: return run_mm<1, NT, 16, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 8: return run_mm<1, NT, 8, 4, 2, 2, RP>(w, s, x, N, K, T, epi, st);
    case 9: return run_mm<1, NT, 4, 1, 3, 2, RP>(w, s, x, N, K, T, epi, st);
    case 10: return run_mm<1, NT, 16, 2, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 11: return run_mm<1, NT, 16, 1, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 12: return run_mm<1, NT, 8, 2, 3, 1, RP>(w, s, x, N, K, T, epi, st);
    case 13: return run_mm<1, NT, 4, 2, 3, 1, RP>(w, s, x, N, K, T, epi, st);
    case 14: return run_mm<1, NT, 16, 4, 3, 1, RP>(w, s, x, N, K, T, epi, st);
    // MT = 2 weight tiles per warp (activation fragments shared): only for NT <= 4 (registers)
    case 100: return run_mm<2, NT, 8, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 101: return run_mm<2, NT, 8, 2, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 102: return run_mm<2, NT, 4, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 103: return run_mm<2, NT, 4, 2, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 104: return run_mm<2, NT, 16, 4, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 105: return run_mm<2, NT, 8, 1, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 106: return run_mm<2, NT, 4, 1, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 107: return run_mm<2, NT, 8, 2, 3, 1, RP>(w, s, x, N, K, T, epi, st);
    default: return cudaErrorInvalidValue;
  }
}
template <int RP>
cudaError_t w4_gemm_cfg(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  if (T < 1 || T > 64) return cudaErrorInvalidValue;
  const int nt = (T + 7) >> 3;
  static const int kBest[9] = {0, 6, 100, 100, 103, 103, 103, 103, 12};
  if (cfg < 0) cfg = kBest[nt];
  switch (nt) {
    case 1: return dispatch<1, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 2: return dispatch<2, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 3: return dispatch<3, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 4: return dispatch<4, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 5: return dispatch<5, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 6: return dispatch<6, RP>(w, scale, x, N, K, T, epi, cfg, st);
    case 7: return dispatch<7, RP>(w, scale, x, N, K, T, epi, cfg, st);
    default: return dispatch<8, RP>(w, scale, x, N, K, T, epi, cfg, st);
  }
}
}  // namespace

extern "C" {
// out = W4(w, scale)[N][K] x X[T][K]^T, T in 1..64, epilogue `epi` (engine.h).  cfg < 0: the config measured fastest for this number of n8 tiles
// (tests/test_w4mm.py --bench, sum of the qkvz+o+gu+down GEMMs at T = 8*NT; cfg >= 100: two weight tiles per warp share the activation fragments).
cudaError_t smkp_w4_gemm_cfg(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  return w4_gemm_cfg<0>(w, scale, x, N, K, T, epi, cfg, st);
}
cudaError_t smkp_w4_gemm(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  return w4_gemm_cfg<0>(w, scale, x, N, K, T, epi, -1, st);
}
// same on the repacked layout (smk_w4r_gemv, include/engine.h)
cudaError_t smkp_w4r_gemm(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  return w4_gemm_cfg<1>(w, scale, x, N, K, T, epi, -1, st);
}
}
