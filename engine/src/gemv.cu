// swift-mk K3: W4A16 mma GEMV (T<=8 rows), BF16 lm_head GEMV (CUDA-core T=1 / mma T>1), small BF16 GEMV, gate|up interleave.
// Clean-room; the mma tile design, dequant trick (lop3 -> bf16x2 128+n, exact fma -136) and persistent flat (tile,group) stream are from
// an earlier W4 multi-row verify prototype (same author).
//
// mma: A operand = weights (16 output rows per tile), B operand = activations (n8 tile = up to 8 tokens), fp32 accumulate per 128-k group,
// fp32 scale after.  One thread (g=lane>>2, t=lane&3) owns rows g and g+8 of the tile and per 128-k group one 16 B chunk of each row:
// word u holds k = 128*kg + 32*t + 8*u + e.  Inside a word the register pairs are (nib e, nib e+4); the activations are permuted to the
// matching order [x0,x4,x1,x5,x2,x6,x3,x7] in registers (4 prmt per 16 B) so callers hand over plain row-major bf16.
#include "common.cuh"
#include "engine.h"
#include <stdlib.h>
#include <string.h>

namespace smk {

// ---------------------------------------------------------------------------------------------- epilogues
__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// acc[0..3] = (row g, tok tb+2t), (row g, tok tb+2t+1), (row g+8, tok tb+2t), (row g+8, tok tb+2t+1) of the 16-row tile starting at `row` (tb: first token of the n8 tile)
__device__ __forceinline__ void epilogue(const float (&acc)[4], int row, int g, int t, int T, int N, const smk_epi& e, int tb = 0) {
  const int tk0 = tb + 2 * t, tk1 = tk0 + 1;
  if (e.kind == SMK_EPI_SILU) {
    uint16_t* o = (uint16_t*)e.out[0];
    const int ch = (row >> 1) + g, W = N >> 1;
    if (tk0 < T) { const float gt = bf16r(acc[0]), up = bf16r(acc[2]); o[(size_t)tk0 * W + ch] = f2bf(bf16r(silu_f(gt)) * up); }
    if (tk1 < T) { const float gt = bf16r(acc[1]), up = bf16r(acc[3]); o[(size_t)tk1 * W + ch] = f2bf(bf16r(silu_f(gt)) * up); }
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

// ---------------------------------------------------------------------------------------------- W4 mma kernel
// NW warps/CTA, KS = K-split warps per 16-row tile (smem reduce at tile end), U = weight/scale register prefetch depth in 128-k groups.
// Persistent: CTA b owns tiles b, b+grid, ...; the (tile, group) stream is flat per warp so weight prefetch crosses tile boundaries.
// Addressing is pointer-increment (no per-group row arithmetic) and the activation fragments are loaded just in time from L1 (K5, GEMV_LOG.md #7).
// Lanes g >= T read activation row T-1 instead of being predicated off: column n of the mma result depends only on B column n and those columns are never stored.
// NT = n8 token tiles per weight fragment (T <= 8 NT): tile n holds tokens 8n..8n+7 and runs the same instructions on its own B fragment and accumulators as the single tile of an
// NT = 1 launch, so a token row gets the same bits whichever tile (and launch row count) it lands in: row-count invariant up to 16 rows.
// RP = 0: checkpoint-natural weight layout (engine.h).  RP = 1: repacked layout (smk_w4r_gemv, pack.py repack_w4): per (16-row tile, 128-k group) one 1 KB unit
// [lane 0..31: row g, chunk t][32 + lane: row g+8, chunk t], units contiguous per tile; nibble j / j+4 of a word = columns 2j / 2j+1 (the activations need no permutation);
// scales as one u32 (bf16 row g | bf16 row g+8 << 16) per (tile, group, g).  Same bytes, same numerics up to the order of k inside an mma.
template <int NW, int KS, int U, int MINB, int RP, int NT = 1>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_w4(const uint4* __restrict__ W, const void* __restrict__ S, const uint4* __restrict__ X, const smk_epi epi, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4 * NT];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS, rowu = K >> 5;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const uint4* xbase[NT];
#pragma unroll
  for (int n = 0; n < NT; n++) xbase[n] = X + (size_t)(n * 8 + g < T ? n * 8 + g : T - 1) * (K >> 3) + ks * 16 + t * 4;
  const unsigned gridsz = gridDim.x * TPC;  // tiles between two consecutive tiles of this CTA's flat stream

  // Flat (tile, group) stream as 32-bit element offsets: `step` inside a tile, `wrap` from the last group of a tile to the first group of the CTA's next tile (branch-free).
  unsigned woff, soff;
  unsigned wstep, wwrap, sstep, swrap;
  const unsigned tile0 = (unsigned)tin + blockIdx.x * TPC;
  if (RP) {
    woff = (tile0 * nkg + ks) * 64u + lane;
    soff = (tile0 * nkg + ks) * 8u + g;
    wstep = KS * 64u; sstep = KS * 8u;
    wwrap = (gridsz * nkg - (per - 1) * KS) * 64u; swrap = (gridsz * nkg - (per - 1) * KS) * 8u;
  } else {
    woff = (tile0 * 16u + g) * rowu + ks * 4 + t;
    soff = (tile0 * 16u + g) * nkg + ks;
    wstep = KS * 4u; sstep = KS;
    wwrap = gridsz * 16u * rowu - (per - 1) * KS * 4u; swrap = gridsz * 16u * nkg - (per - 1) * KS;
  }
  const uint4* __restrict__ Wp = W;
  const uint32_t* S32 = (const uint32_t*)S;
  const uint16_t* S16 = (const uint16_t*)S;
  uint4 wa[U], wb[U];
  uint32_t ss[U];
  int lk = 0;
  auto issue = [&](int j) {
    if (RP) {
      wa[j] = ldw(Wp + woff); wb[j] = ldw(Wp + woff + 32);
      ss[j] = ld32(S32 + soff);
    } else {
      wa[j] = ldw(Wp + woff); wb[j] = ldw(Wp + woff + (size_t)8 * rowu);
      ss[j] = ld16(S16 + soff) | (ld16(S16 + soff + (size_t)8 * nkg) << 16);
    }
    const bool wr = (++lk == per);
    lk = wr ? 0 : lk;
    woff += wr ? wwrap : wstep;
    soff += wr ? swrap : sstep;
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        const uint32_t sv = ss[j];
        if (f + j + U < total) issue(j);
        uint4 xb[NT][4];
#pragma unroll
        for (int n = 0; n < NT; n++)
#pragma unroll
          for (int u = 0; u < 4; u++) xb[n][u] = RP ? ldx(xbase[n] + ct * (KS * 16) + u) : perm_b(ldx(xbase[n] + ct * (KS * 16) + u));
        float part[NT][4];
#pragma unroll
        for (int n = 0; n < NT; n++) part[n][0] = part[n][1] = part[n][2] = part[n][3] = 0.f;
        const uint32_t wra[4] = {ca.x, ca.y, ca.z, ca.w}, wrb[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[4], pb[4];
          dq_word(wra[u], pa);
          dq_word(wrb[u], pb);
#pragma unroll
          for (int n = 0; n < NT; n++) {
            mma16816(part[n], pa[0], pb[0], pa[1], pb[1], xb[n][u].x, xb[n][u].y);
            mma16816(part[n], pa[2], pb[2], pa[3], pb[3], xb[n][u].z, xb[n][u].w);
          }
        }
        const float fsa = bflo(sv), fsb = bfhi(sv);
#pragma unroll
        for (int n = 0; n < NT; n++) {
          acc[n][0] = fmaf(fsa, part[n][0], acc[n][0]); acc[n][1] = fmaf(fsa, part[n][1], acc[n][1]);
          acc[n][2] = fmaf(fsb, part[n][2], acc[n][2]); acc[n][3] = fmaf(fsb, part[n][3], acc[n][3]);
        }
        if (++ct == per) {  // tile finished
          ct = 0;
          const int row = (int)(tile0 + (unsigned)cm * gridsz) << 4;
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int n = 0; n < NT; n++)
#pragma unroll
                for (int c = 0; c < 4; c++) red[warp][lane][n * 4 + c] = acc[n][c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int n = 0; n < NT; n++) {
#pragma unroll
                for (int q = 1; q < KS; q++)
#pragma unroll
                  for (int c = 0; c < 4; c++) acc[n][c] += red[warp + q][lane][n * 4 + c];
                epilogue(acc[n], row, g, t, T, N, epi, n * 8);
              }
            }
            __syncthreads();
          } else {
#pragma unroll
            for (int n = 0; n < NT; n++) epilogue(acc[n], row, g, t, T, N, epi, n * 8);
          }
#pragma unroll
          for (int n = 0; n < NT; n++) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------- BF16 mma (lm_head, T=2..8): weights are the A operand straight from memory
template <int NW, int KS, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_bf16_mma(const uint4* __restrict__ W, const uint4* __restrict__ X, const smk_epi epi, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nit = K >> 5, per = nit / KS, rowu = K >> 3;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const bool tvalid = g < T;
  uint4 wa[U], wb[U];
  int lm = 0, lk = 0;
  auto issue = [&](int j) {
    const int row = (((int)blockIdx.x + lm * (int)gridDim.x) * TPC + tin) << 4;
    const size_t o = (size_t)(row + g) * rowu + (size_t)(ks + KS * lk) * 4 + t;
    wa[j] = ldw(W + o); wb[j] = ldw(W + o + (size_t)8 * rowu);
    if (++lk == per) { lk = 0; ++lm; }
  };
  uint4 xn;
  auto loadB = [&](int kk) { xn = tvalid ? ldx(X + (size_t)g * (K >> 3) + (ks + KS * kk) * 4 + t) : make_uint4(0, 0, 0, 0); };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  if (total > 0) loadB(0);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        if (f + j + U < total) issue(j);
        const uint4 xb = xn;
        if (f + j + 1 < total) loadB(ct + 1 == per ? 0 : ct + 1);
        mma16816(acc, ca.x, cb.x, ca.y, cb.y, xb.x, xb.y);
        mma16816(acc, ca.z, cb.z, ca.w, cb.w, xb.z, xb.w);
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

// ---------------------------------------------------------------------------------------------- CUDA-core BF16 GEMV (T=1 lm_head)
// warp task = R consecutive rows x full K; lane owns 16 B (8 bf16) of each row per 512 B tile; fp32 FMA; flat (task,tile) stream with
// register prefetch U tiles ahead.  Activation read through L1 (natural layout: lane-contiguous).
template <int NW, int R, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) k_bf16_cc(const uint4* __restrict__ W, const uint4* __restrict__ XA, float* __restrict__ out, int N, int K) {
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

// ---------------------------------------------------------------------------------------------- small BF16 GEMV (in_proj_a|b)
// one 128-thread block per weight row (K split across the 4 warps, smem reduce); T <= 8 token rows; bf16 output (HF rounds the Linear output).
__global__ void __launch_bounds__(128, 8) k_bf16_small(const uint4* __restrict__ W, const uint4* __restrict__ X, int N, int na, int K, int T,
                                                    uint16_t* __restrict__ out_a, uint16_t* __restrict__ out_b) {
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
    if (row < na) out_a[(size_t)tk * na + row] = f2bf(s);
    else out_b[(size_t)tk * (N - na) + (row - na)] = f2bf(s);
  }
}

// ---------------------------------------------------------------------------------------------- gate|up interleave (packer)
__global__ void k_interleave(const uint4* __restrict__ gate, const uint4* __restrict__ up, uint4* __restrict__ dst, int N, size_t cpr) {
  const size_t total = (size_t)2 * N * cpr;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
    const size_t drow = i / cpr, c = i % cpr;
    const size_t srow = (drow >> 4) * 8 + (drow & 7);
    dst[i] = ((drow & 8) ? up : gate)[srow * cpr + c];
  }
}

// ---------------------------------------------------------------------------------------------- host launchers
static int g_nsm = 0;
static int nsm() {
  if (!g_nsm) { int d = 0; cudaGetDevice(&d); cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, d); }
  return g_nsm;
}
// persistent grids: every warp must do the same amount of work, so use the largest divisor of `groups` that fits in the resident capacity
template <class Kn>
static int grid_for(Kn kern, int threads, int groups) {
  int occ = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, 0);
  if (occ < 1) occ = 1;
  const int cap = groups < occ * nsm() ? groups : occ * nsm();
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

template <int NW, int KS, int U, int MINB, int RP>
static cudaError_t run_w4(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  constexpr int TPC = NW / KS;
  if (((K >> 7) % KS) || ((N >> 4) % TPC) || (K & 127) || (N & 15) || T < 1 || T > 16) return cudaErrorInvalidValue;
  const int groups = (N >> 4) / TPC;
  if (T <= 8) {
    static int grid_cache_n = -1, grid_cache_g = 0;  // single-config cache keyed by N (K/T do not change the group count)
    if (grid_cache_n != N) { grid_cache_g = grid_for(k_w4<NW, KS, U, MINB, RP, 1>, NW * 32, groups); grid_cache_n = N; }
    k_w4<NW, KS, U, MINB, RP, 1><<<grid_cache_g, NW * 32, 0, st>>>((const uint4*)w, s, (const uint4*)x, *epi, N, K, T);
  } else {  // 9..16 rows: two n8 token tiles per weight fragment (same per-row bits as a T <= 8 launch)
    static int grid_cache_n = -1, grid_cache_g = 0;
    if (grid_cache_n != N) { grid_cache_g = grid_for(k_w4<NW, KS, U, MINB, RP, 2>, NW * 32, groups); grid_cache_n = N; }
    k_w4<NW, KS, U, MINB, RP, 2><<<grid_cache_g, NW * 32, 0, st>>>((const uint4*)w, s, (const uint4*)x, *epi, N, K, T);
  }
  return cudaGetLastError();
}
template <int NW, int KS, int U, int MINB>
static cudaError_t run_bf16_mma(const void* w, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  constexpr int TPC = NW / KS;
  if (((K >> 5) % KS) || ((N >> 4) % TPC) || (K & 31) || T < 1 || T > 8) return cudaErrorInvalidValue;
  const int groups = (N >> 4) / TPC;
  static int grid_cache_n = -1, grid_cache_g = 0;
  if (grid_cache_n != N) { grid_cache_g = grid_for(k_bf16_mma<NW, KS, U, MINB>, NW * 32, groups); grid_cache_n = N; }
  k_bf16_mma<NW, KS, U, MINB><<<grid_cache_g, NW * 32, 0, st>>>((const uint4*)w, (const uint4*)x, *epi, N, K, T);
  return cudaGetLastError();
}

}  // namespace smk

using namespace smk;

template <int RP>
static cudaError_t w4_cfg(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  switch (cfg) {
    case 0: return run_w4<8, 4, 2, 2, RP>(w, s, x, N, K, T, epi, st);
    case 1: return run_w4<8, 8, 2, 2, RP>(w, s, x, N, K, T, epi, st);
    case 2: return run_w4<8, 2, 3, 2, RP>(w, s, x, N, K, T, epi, st);
    case 3: return run_w4<4, 4, 3, 4, RP>(w, s, x, N, K, T, epi, st);
    case 4: return run_w4<16, 8, 2, 1, RP>(w, s, x, N, K, T, epi, st);
    case 5: return run_w4<8, 1, 3, 2, RP>(w, s, x, N, K, T, epi, st);
  }
  return cudaErrorInvalidValue;
}

template <int RP>
static cudaError_t w4_gemv(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) {
  // In-graph sweep of the full step (k_w4 v2, GEMV_LOG.md): w8k4u2 for every shape except the 14336-row attention q|k|v (w8k8u2); +-0.2% between configs.
  // Tuning override: SMK_CFG="<N>:<cfg>,..." (e.g. 34816:4,5120:0) applies to every call with that N (100000+K keys a K-specific override).
  int cfg = (N == 14336) ? 1 : 0;
  static const char* ov = getenv("SMK_CFG");
  if (ov) {
    for (const char* p = ov; *p;) {
      int n = atoi(p), c = 0;
      const char* q = strchr(p, ':');
      if (!q) break;
      c = atoi(q + 1);
      if (n == N || (n == 100000 + K)) cfg = c;  // 100000+K keys a K-specific override (e.g. 100000+6144 for o_proj)
      p = strchr(q, ',');
      if (!p) break;
      ++p;
    }
  }
  cudaError_t e = w4_cfg<RP>(w, s, x, N, K, T, epi, cfg, st);
  if (e == cudaErrorInvalidValue) e = w4_cfg<RP>(w, s, x, N, K, T, epi, cfg == 0 ? 1 : 0, st);
  if (e == cudaErrorInvalidValue) e = w4_cfg<RP>(w, s, x, N, K, T, epi, 5, st);  // always-valid fallback (KS=1)
  return e;
}

extern "C" {

// config table for tuning (bench) -- ids: 0 w8k4u2, 1 w8k8u2, 2 w8k2u3, 3 w4k4u3, 4 w16k8u2, 5 w8k1u3
cudaError_t smk_w4_gemv_cfg(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  return w4_cfg<0>(w, s, x, N, K, T, epi, cfg, st);
}
cudaError_t smk_w4_gemv(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) { return w4_gemv<0>(w, s, x, N, K, T, epi, st); }
cudaError_t smk_w4r_gemv_cfg(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st) {
  return w4_cfg<1>(w, s, x, N, K, T, epi, cfg, st);
}
cudaError_t smk_w4r_gemv(const void* w, const void* s, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st) { return w4_gemv<1>(w, s, x, N, K, T, epi, st); }

cudaError_t smk_bf16_gemv_f32(const void* w, const void* x, int N, int K, int T, float* out, cudaStream_t st) {
  if (K & 255) return cudaErrorInvalidValue;
  if (T == 1) {
    static int gn = -1, gg = 0;
    if (N % 4) return cudaErrorInvalidValue;
    if (gn != N) { gg = grid_for(k_bf16_cc<8, 4, 4, 2>, 256, (N / 4 + 7) / 8); gn = N; }
    k_bf16_cc<8, 4, 4, 2><<<gg, 256, 0, st>>>((const uint4*)w, (const uint4*)x, out, N, K);
    return cudaGetLastError();
  }
  smk_epi e = {};
  e.kind = SMK_EPI_F32; e.out[0] = out;
  return run_bf16_mma<8, 4, 4, 2>(w, x, N, K, T, &e, st);
}

// Same as smk_bf16_gemv_f32 for T in 2..8 with an explicit mma configuration (tuning; the DFlash2 drafter's 1280-row conv projections): 0 w8k4u4 (default), 1 w8k8u2, 2 w8k8u4, 3 w4k4u4, 4 w4k2u4, 5 w16k8u2.
cudaError_t smk_bf16_gemv_f32_cfg(const void* w, const void* x, int N, int K, int T, float* out, int cfg, cudaStream_t st) {
  smk_epi e = {};
  e.kind = SMK_EPI_F32; e.out[0] = out;
  switch (cfg) {
    case 0: return run_bf16_mma<8, 4, 4, 2>(w, x, N, K, T, &e, st);
    case 1: return run_bf16_mma<8, 8, 2, 2>(w, x, N, K, T, &e, st);
    case 2: return run_bf16_mma<8, 8, 4, 2>(w, x, N, K, T, &e, st);
    case 3: return run_bf16_mma<4, 4, 4, 4>(w, x, N, K, T, &e, st);
    case 4: return run_bf16_mma<4, 2, 4, 4>(w, x, N, K, T, &e, st);
    case 5: return run_bf16_mma<16, 8, 2, 1>(w, x, N, K, T, &e, st);
  }
  return cudaErrorInvalidValue;
}

cudaError_t smk_bf16_gemv_small(const void* w, const void* x, int N, int na, int K, int T, void* out_a, void* out_b, cudaStream_t st) {
  if (T < 1 || T > 16 || (K & 7)) return cudaErrorInvalidValue;
  for (int t0 = 0; t0 < T; t0 += 8) {  // rows 8..15 as a second launch: every row's sum is the same whatever the launch's row count
    const int tn = T - t0 < 8 ? T - t0 : 8;
    k_bf16_small<<<N, 128, 0, st>>>((const uint4*)w, (const uint4*)x + (size_t)t0 * (K >> 3), N, na, K, tn, (uint16_t*)out_a + (size_t)t0 * na, (uint16_t*)out_b + (size_t)t0 * (N - na));
  }
  return cudaGetLastError();
}

cudaError_t smk_interleave_gateup(const void* gate, const void* up, void* dst, int N, size_t row_bytes, cudaStream_t st) {
  if ((N & 7) || (row_bytes & 15)) return cudaErrorInvalidValue;
  k_interleave<<<2048, 256, 0, st>>>((const uint4*)gate, (const uint4*)up, (uint4*)dst, N, row_bytes / 16);
  return cudaGetLastError();
}

}  // extern "C"
