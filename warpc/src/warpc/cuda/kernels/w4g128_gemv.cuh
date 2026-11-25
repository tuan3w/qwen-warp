#pragma once
#include "dense_common.cuh"

// W4G128 GEMV family: y[T][N] = x[T][K] W[N][K]^T for int4 weights with one bf16 scale per 128 columns (the checkpoint's
// w4g128 scheme: value = (nibble - 8) * scale), T = 1..STEP_ROWS rows sharing ONE pass over the weights. Shapes are
// arguments: K % 128 == 0, N % 16 == 0 (silu pair: N = 2 * n with n % 8 == 0), nothing is model-specific.
//
// ------------------------------------------------------------------------------------------------------------ layout
// "w4r" (pack kinds w4g / w4g_cat / w4g_gateup of warpc/cuda/w4.py): one buffer [weights: N*K/2 bytes][scales: N*K/64
// bytes]. Rows are cut into 16-row tiles, K into 128-column groups. Per (tile, group) the weights are one contiguous 1 KB
// unit: 32 lanes x 16 B for the tile's rows g = 0..7, then 32 lanes x 16 B for rows g + 8 (lane = 4 g + t; the 16 B of
// lane (g, t) are the 4 words u = 0..3 of columns 128 kg + 32 t + 8 u + 0..7 of row g). Units of a tile are contiguous
// (group-major), tiles follow each other. Inside a word nibble j and j + 4 hold columns 2j and 2j + 1, so one `dq_word`
// yields the 4 bf16x2 registers of the mma A fragment (columns 2j, 2j + 1 for j = 0..3) and the activations need no
// permutation. Scales: one u32 per (tile, group, g) = bf16(row g) | bf16(row g + 8) << 16, tiles / groups contiguous.
// The gate | up pair is interleaved per 8 rows before the repack: rows 16 i .. 16 i + 7 = gate rows 8 i .. 8 i + 7,
// rows 16 i + 8 .. 16 i + 15 = up rows 8 i .. 8 i + 7 (so one 16-row tile holds matching gate and up channels).
// x: bf16 [T][K] row-major.
//
// ------------------------------------------------------------------------------------------------------ the numerics
// The accumulation order IS the kernel's specification (and `refs` in warpc/tests/test_w4.py is a CPU model of it):
//   1. One mma.m16n8k16 (bf16 x bf16 products are exact in fp32, the tensor core adds the 16 products and the accumulator
//      in its own fixed way, modelled by `tc_mma` in the test) computes, for hardware k = 0..15, A[16 rows][16 k] x
//      B[16 k][8 tokens] + C. A column pair of the fragment is a word's columns (2j, 2j + 1), see "layout". Per (tile,
//      128-column group) the group partial `part` (fp32, starts at 0) goes through 8 chained mma, for u = 0..3:
//         mma(part, A = columns 8u + {0,1,2,3} of lane chunks t = 0..3 (the 4 chunks cover hardware k 0..15), x likewise)
//         mma(part, A = columns 8u + {4,5,6,7}, ...)
//      i.e. chunk t (columns 32 t .. 32 t + 31 of the group) feeds hardware k = 2t, 2t + 1, 2t + 8, 2t + 9 of each mma.
//   2. acc = fma(scale_row, part, acc): fp32 fma, scale_row = the bf16 scale as fp32; the groups of one warp are added in
//      ascending group index.
//   3. K split KS: warp ks of a tile owns the groups ks, ks + KS, ks + 2 KS, ... (acc_ks as in 2); the tile's value is the
//      fp32 left-to-right sum acc_0 + acc_1 + ... + acc_{KS-1} (KS is a launch argument and part of the spec: it changes
//      the bits; w4.py picks it from (N, K)).
//   4. One rounding: bf16 round-to-nearest-even of the fp32 sum. Epilogue SEGS writes that bf16 to out[segment][t][row -
//      segment start] (segment s = rows [end[s - 1], end[s]), ends multiples of 16). Epilogue SILU takes g = bf16(gate
//      sum), u = bf16(up sum) and writes bf16(bf16(g / (1 + expf(-g))) * u) to out[0][t][channel], channel = 8 i + g.
//      Epilogue F32 writes the fp32 sum of step 3 unrounded to out[0][t][row] (the gates compare these bits: a bf16
//      output hides all but 1 in 2^16 of the fp32 differences).
//   Row invariance: token t is column t of the n8 tile; a column of the mma result depends on that B column only, so
//   a row's bits do not depend on T or on its neighbours.
//
// ----------------------------------------------------------------------------------------------------------- kernel
// Persistent: NW warps per CTA, a CTA owns tiles b, b + grid, ... (TPC = NW / KS tiles at a time); the (tile, group)
// stream is flat per warp so the register prefetch of the weights (U groups deep) crosses tile boundaries. The weights
// are streamed with L1::no_allocate loads; activation fragments are loaded just in time from L1. Lanes with g >= T read
// row T - 1 (their columns are never stored).

namespace w4g {

enum { EPI_SEGS = 0, EPI_SILU = 1, EPI_F32 = 2 };
struct Epi {
  int kind;
  int end[3];
  void* out[3];
};

__device__ __forceinline__ float bflo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint16_t f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
__device__ __forceinline__ float bfr(float f) { return __bfloat162float(__float2bfloat16_rn(f)); }

// weights stream through once: do not allocate in L1 (it holds the activations)
__device__ __forceinline__ uint4 ld_w(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// one word of 8 nibbles -> 4 bf16x2 registers ((n_j, n_{j+4}) - 8, exact): (w >> 4j) & 0x000F000F | 0x43004300 is
// bf16x2 (128 + n, 128 + n), then one exact fma by 1.0 - 136.0
__device__ __forceinline__ void dq_word(uint32_t w, uint32_t (&p)[4]) {
#pragma unroll
  for (int j = 0; j < 4; j++) {
    uint32_t q;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;\n" : "=r"(q) : "r"(j ? (w >> (4 * j)) : w), "r"(0x000F000Fu), "r"(0x43004300u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(p[j]) : "r"(q), "r"(0x3F803F80u), "r"(0xC308C308u));
  }
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// acc = (row g, token tk0), (row g, tk0 + 1), (row g + 8, tk0), (row g + 8, tk0 + 1) of the 16-row tile starting at `row`
__device__ __forceinline__ void epilogue(const float (&acc)[4], int row, int g, int t, int T, int N, const Epi& e) {
  const int tk0 = 2 * t, tk1 = tk0 + 1;
  if (e.kind == EPI_SILU) {
    uint16_t* o = (uint16_t*)e.out[0];
    const int ch = (row >> 1) + g, W = N >> 1;
    if (tk0 < T) o[(size_t)tk0 * W + ch] = f2bf(bfr(silu_f(bfr(acc[0]))) * bfr(acc[2]));
    if (tk1 < T) o[(size_t)tk1 * W + ch] = f2bf(bfr(silu_f(bfr(acc[1]))) * bfr(acc[3]));
  } else if (e.kind == EPI_F32) {
    float* o = (float*)e.out[0];
    const int r0 = row + g, r1 = r0 + 8;
    if (tk0 < T) {
      o[(size_t)tk0 * N + r0] = acc[0];
      o[(size_t)tk0 * N + r1] = acc[2];
    }
    if (tk1 < T) {
      o[(size_t)tk1 * N + r0] = acc[1];
      o[(size_t)tk1 * N + r1] = acc[3];
    }
  } else {
    const int seg = row < e.end[0] ? 0 : (row < e.end[1] ? 1 : 2);
    const int start = seg == 0 ? 0 : (seg == 1 ? e.end[0] : e.end[1]);
    const int width = (seg == 0 ? e.end[0] : (seg == 1 ? e.end[1] : e.end[2])) - start;
    uint16_t* o = (uint16_t*)(seg == 0 ? e.out[0] : (seg == 1 ? e.out[1] : e.out[2]));
    const int r0 = row + g - start, r1 = r0 + 8;
    if (tk0 < T) {
      o[(size_t)tk0 * width + r0] = f2bf(acc[0]);
      o[(size_t)tk0 * width + r1] = f2bf(acc[2]);
    }
    if (tk1 < T) {
      o[(size_t)tk1 * width + r0] = f2bf(acc[1]);
      o[(size_t)tk1 * width + r1] = f2bf(acc[3]);
    }
  }
}

// NW warps per CTA, KS K-split warps per tile (smem reduce at tile end), U groups of register prefetch, MINB the
// launch-bound CTAs per SM.
template <int NW, int KS, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_w4g(const uint4* __restrict__ W, const uint32_t* __restrict__ S, const uint4* __restrict__ X, const Epi epi, int N, int K,
          int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  const uint4* xbase = X + (size_t)(g < T ? g : T - 1) * (K >> 3) + ks * 16 + t * 4;
  const unsigned gridsz = gridDim.x * TPC;  // tiles between two consecutive tiles of this CTA's flat stream

  // flat (tile, group) stream as 32-bit element offsets: `step` inside a tile, `wrap` from the last group of a tile to
  // the first group of the CTA's next tile (branch-free)
  const unsigned tile0 = (unsigned)tin + blockIdx.x * TPC;
  unsigned woff = (tile0 * nkg + ks) * 64u + lane;
  unsigned soff = (tile0 * nkg + ks) * 8u + g;
  const unsigned wstep = KS * 64u, sstep = KS * 8u;
  const unsigned wwrap = (gridsz * nkg - (per - 1) * KS) * 64u, swrap = (gridsz * nkg - (per - 1) * KS) * 8u;
  uint4 wa[U], wb[U];
  uint32_t ss[U];
  int lk = 0;
  auto issue = [&](int j) {
    wa[j] = ld_w(W + woff);
    wb[j] = ld_w(W + woff + 32);
    ss[j] = __ldg(S + soff);
    const bool wr = (++lk == per);
    lk = wr ? 0 : lk;
    woff += wr ? wwrap : wstep;
    soff += wr ? swrap : sstep;
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
        const uint32_t sv = ss[j];
        if (f + j + U < total) issue(j);
        uint4 xb[4];
#pragma unroll
        for (int u = 0; u < 4; u++) xb[u] = __ldg(xbase + ct * (KS * 16) + u);
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
        acc[0] = fmaf(fsa, part[0], acc[0]);
        acc[1] = fmaf(fsa, part[1], acc[1]);
        acc[2] = fmaf(fsb, part[2], acc[2]);
        acc[3] = fmaf(fsb, part[3], acc[3]);
        if (++ct == per) {  // tile finished
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

}  // namespace w4g
