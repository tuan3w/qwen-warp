// Clean-room W4A16 (g128, symmetric uint4b8, bf16 scale) multi-row (T tokens) tensor-core GEMM-ish kernels + CUDA-core T=1 GEMVs.
// Layout (same as E1 / compressed-tensors natural): W4 uint32 [N][K/8] (nibble e of word j = k 8j+e, value (n-8)*scale[n][k/128]);
// BF16 weights bf16 [N][K].  Activations: T rows (T<=16) bf16, row-major [tok][K]; mma kernels read pre-permuted copies (see bench.cu).
//
// mma kernels: A operand = weights (M=16 output rows), B operand = activations (n8 tile = 8 tokens), fp32 accumulate.
//   T<=8 -> NT=1 n8 tile, T in 9..16 -> NT=2 tiles that reuse the SAME dequantised A registers.
//   One thread (g=lane>>2, t=lane&3) owns rows g and g+8 of the 16-row tile and, per 128-k group, one 16 B chunk (32 k) of each row:
//   word u (0..3) holds k = 128*kg + 32*t + 8*u + e.  Inside a word the register pairs are (nib e, nib e+4): lop3 with 0x000F000F gives
//   (n_e, n_{e+4}) as the bf16x2 (128+n_e, 128+n_{e+4}) after OR-ing 0x4300 (bf16 0x4300 = 128.0, ulp 1 -> exact).  The B operand is stored
//   in the matching order [x0,x4,x1,x5,x2,x6,x3,x7] per 8 k (the k order inside an mma is arbitrary as long as A and B agree).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace e3 {

__device__ __forceinline__ float bflo(uint32_t u) { return __uint_as_float(u << 16); }
__device__ __forceinline__ float bfhi(uint32_t u) { return __uint_as_float(u & 0xffff0000u); }
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
#ifdef E3_LDW_CS
__device__ __forceinline__ uint4 ldw(const uint4* p) { return __ldcs(p); }  // weights: streaming (allocates in L1, evict-first)
#else
__device__ __forceinline__ uint4 ldw(const uint4* p) {  // weights: streaming, never allocate in L1 (L1 holds the activation working set)
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
#endif
__device__ __forceinline__ uint4 ldx(const uint4* p) { return __ldg(p); }   // activations: L1/L2 resident
__device__ __forceinline__ uint32_t ld16(const uint16_t* p) { return (uint32_t)__ldg(p); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// dequant modes for the mma kernels
enum { MM_SUB = 0,   // lop3 magic -> bf16x2 (128+n) -> fma.bf16x2 (x*1 - 136): exact (n-8) weights, fresh accumulator per group, fp32 scale after
       MM_ONES = 1,  // lop3 magic only (128+n); -136*sum_k(x) per group folded into the fp32 epilogue (table XS) -> no per-pair subtract
       MM_ABL = 2,   // ablated: raw packed words fed to the mma as A (results INVALID), same loads / mma / epilogue as MM_ONES
       MM_ABLB = 3 }; // ablated like MM_ABL AND activations (B operand + XS) loaded only once for group 0 and reused forever (INVALID): isolates mma issue from B delivery

template <int MODE>
__device__ __forceinline__ void dq_word(uint32_t w, uint32_t (&p)[4]) {
  if (MODE >= MM_ABL) {
    p[0] = p[1] = p[2] = p[3] = w;
  } else {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t q = ((w >> (4 * j)) & 0x000F000Fu) | 0x43004300u;  // (128+n_j, 128+n_{j+4})
      if (MODE == MM_SUB) {
        uint32_t r;
        asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(q), "r"(0x3F803F80u), "r"(0xC308C308u));  // q*1.0 - 136.0
        q = r;
      }
      p[j] = q;
    }
  }
}

template <int NT, bool BF>
__device__ __forceinline__ void store_tile(float (&acc)[NT][4], void* out, int row, int g, int t) {
  constexpr int TP = 8 * NT;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    uint32_t* o = reinterpret_cast<uint32_t*>(out);
    o[((size_t)(row + g) * TP + nt * 8 + 2 * t) >> 1] = pack_bf16x2(acc[nt][0], acc[nt][1]);
    o[((size_t)(row + g + 8) * TP + nt * 8 + 2 * t) >> 1] = pack_bf16x2(acc[nt][2], acc[nt][3]);
  }
}

// ---------------------------------------------------------------- W4 mma
// NW warps/CTA, KS = K-split warps per 16-row tile (smem reduce at tile end), U = weight/scale register prefetch depth in 128-k groups.
// Persistent: CTA b owns tiles b, b+grid, ...; the (tile, group) stream is FLAT per warp so weight prefetch crosses tile boundaries (E1 lesson:
// a drain at every tile end costs ~10%); activations (B operand) are prefetched one group ahead (they are L1/L2 resident, shared by all tiles).
template <int MODE, int NT, int NW, int KS, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_mma_w4(const uint4* __restrict__ W, const uint16_t* __restrict__ S, const uint4* __restrict__ XP, const float* __restrict__ XS,
             void* out, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][NT][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nkg = K >> 7, per = nkg / KS, rowu = K >> 5;  // warp ks owns groups ks, ks+KS, ... (the KS warps of a tile read adjacent 64 B pieces: 64*KS B contiguous per row)
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;  // tiles for this CTA
  const int total = cnt * per;
  bool tvalid[NT];
  int tk0[NT], tk1[NT];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) { tvalid[nt] = (nt * 8 + g) < T; tk0[nt] = nt * 8 + 2 * t; tk1[nt] = tk0[nt] + 1; }

  uint4 wa[U], wb[U];
  uint32_t sa[U], sb[U];
  int lm = 0, lk = 0;
  auto issue = [&](int j) {
    const int row = (((int)blockIdx.x + lm * (int)gridDim.x) * TPC + tin) << 4;
    const size_t o = (size_t)(row + g) * rowu + (size_t)(ks + KS * lk) * 4 + t;
    wa[j] = ldw(W + o); wb[j] = ldw(W + o + (size_t)8 * rowu);
    sa[j] = ld16(S + (size_t)(row + g) * nkg + ks + KS * lk); sb[j] = ld16(S + (size_t)(row + g + 8) * nkg + ks + KS * lk);
    if (++lk == per) { lk = 0; ++lm; }
  };
  uint4 xn[NT][4];
  float xsn[NT][2];
  auto loadB = [&](int kk) {  // activations of group kg0+kk
    const int kg = ks + KS * kk;
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
#pragma unroll
      for (int u = 0; u < 4; u++)
        xn[nt][u] = tvalid[nt] ? ldx(XP + (size_t)(nt * 8 + g) * (K >> 3) + kg * 16 + t * 4 + u) : make_uint4(0, 0, 0, 0);
      if (MODE != MM_SUB) {
        xsn[nt][0] = tk0[nt] < T ? __ldg(XS + (size_t)tk0[nt] * nkg + kg) : 0.f;
        xsn[nt][1] = tk1[nt] < T ? __ldg(XS + (size_t)tk1[nt] * nkg + kg) : 0.f;
      }
    }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  if (total > 0) loadB(0);
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        const float fsa = bflo(sa[j]), fsb = bflo(sb[j]);
        if (f + j + U < total) issue(j);
        constexpr bool BEARLY = NT == 1;  // NT=2: loading next group's B early needs a second 32-register copy -> spills; refill at the end instead
        uint4 xb[NT][4];
        float xs[NT][2];
        if (BEARLY) {
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int u = 0; u < 4; u++) xb[nt][u] = xn[nt][u];
            xs[nt][0] = MODE != MM_SUB ? xsn[nt][0] : 0.f; xs[nt][1] = MODE != MM_SUB ? xsn[nt][1] : 0.f;
          }
          if (MODE != MM_ABLB && f + j + 1 < total) loadB(ct + 1 == per ? 0 : ct + 1);
        }
        if (!BEARLY) {
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int u = 0; u < 4; u++) xb[nt][u] = xn[nt][u];
            xs[nt][0] = MODE != MM_SUB ? xsn[nt][0] : 0.f; xs[nt][1] = MODE != MM_SUB ? xsn[nt][1] : 0.f;
          }
        }
        float part[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) part[nt][0] = part[nt][1] = part[nt][2] = part[nt][3] = 0.f;
        const uint32_t wra[4] = {ca.x, ca.y, ca.z, ca.w}, wrb[4] = {cb.x, cb.y, cb.z, cb.w};
#pragma unroll
        for (int u = 0; u < 4; u++) {
          uint32_t pa[4], pb[4];
          dq_word<MODE>(wra[u], pa);
          dq_word<MODE>(wrb[u], pb);
#pragma unroll
          for (int nt = 0; nt < NT; nt++) {
            mma16816(part[nt], pa[0], pb[0], pa[1], pb[1], xb[nt][u].x, xb[nt][u].y);
            mma16816(part[nt], pa[2], pb[2], pa[3], pb[3], xb[nt][u].z, xb[nt][u].w);
          }
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          if (MODE == MM_SUB) {
            acc[nt][0] = fmaf(fsa, part[nt][0], acc[nt][0]); acc[nt][1] = fmaf(fsa, part[nt][1], acc[nt][1]);
            acc[nt][2] = fmaf(fsb, part[nt][2], acc[nt][2]); acc[nt][3] = fmaf(fsb, part[nt][3], acc[nt][3]);
          } else {
            acc[nt][0] = fmaf(fsa, fmaf(-136.f, xs[nt][0], part[nt][0]), acc[nt][0]);
            acc[nt][1] = fmaf(fsa, fmaf(-136.f, xs[nt][1], part[nt][1]), acc[nt][1]);
            acc[nt][2] = fmaf(fsb, fmaf(-136.f, xs[nt][0], part[nt][2]), acc[nt][2]);
            acc[nt][3] = fmaf(fsb, fmaf(-136.f, xs[nt][1], part[nt][3]), acc[nt][3]);
          }
        }
        if (!BEARLY && MODE != MM_ABLB && f + j + 1 < total) loadB(ct + 1 == per ? 0 : ct + 1);
        if (++ct == per) {  // tile finished
          ct = 0;
          const int row = (((int)blockIdx.x + cm * (int)gridDim.x) * TPC + tin) << 4;
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) red[warp][nt][lane][c] = acc[nt][c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int q = 1; q < KS; q++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                  for (int c = 0; c < 4; c++) acc[nt][c] += red[warp + q][nt][lane][c];
              store_tile<NT, true>(acc, out, row, g, t);
            }
            __syncthreads();
          } else {
            store_tile<NT, true>(acc, out, row, g, t);
          }
#pragma unroll
          for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- BF16 mma (lm_head): weights are the A operand straight from memory
template <int NT, int NW, int KS, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_mma_bf16(const uint4* __restrict__ W, const uint4* __restrict__ XB, void* out, int N, int K, int T) {
  constexpr int TPC = NW / KS;
  __shared__ float red[KS > 1 ? NW : 1][NT][32][4];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
  const int tin = warp / KS, ks = warp % KS;
  const int nit = K >> 5, per = nit / KS, rowu = K >> 3;
  const int groups = (N >> 4) / TPC;
  const int cnt = (groups - (int)blockIdx.x + (int)gridDim.x - 1) / (int)gridDim.x;
  const int total = cnt * per;
  bool tvalid[NT];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) tvalid[nt] = (nt * 8 + g) < T;
  uint4 wa[U], wb[U];
  int lm = 0, lk = 0;
  auto issue = [&](int j) {
    const int row = (((int)blockIdx.x + lm * (int)gridDim.x) * TPC + tin) << 4;
    const size_t o = (size_t)(row + g) * rowu + (size_t)(ks + KS * lk) * 4 + t;
    wa[j] = ldw(W + o); wb[j] = ldw(W + o + (size_t)8 * rowu);
    if (++lk == per) { lk = 0; ++lm; }
  };
  uint4 xn[NT];
  auto loadB = [&](int kk) {
#pragma unroll
    for (int nt = 0; nt < NT; nt++)
      xn[nt] = tvalid[nt] ? ldx(XB + (size_t)(nt * 8 + g) * (K >> 3) + (ks + KS * kk) * 4 + t) : make_uint4(0, 0, 0, 0);
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  if (total > 0) loadB(0);
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        const uint4 ca = wa[j], cb = wb[j];
        if (f + j + U < total) issue(j);
        uint4 xb[NT];
#pragma unroll
        for (int nt = 0; nt < NT; nt++) xb[nt] = xn[nt];
        if (f + j + 1 < total) loadB(ct + 1 == per ? 0 : ct + 1);
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
          mma16816(acc[nt], ca.x, cb.x, ca.y, cb.y, xb[nt].x, xb[nt].y);
          mma16816(acc[nt], ca.z, cb.z, ca.w, cb.w, xb[nt].z, xb[nt].w);
        }
        if (++ct == per) {
          ct = 0;
          const int row = (((int)blockIdx.x + cm * (int)gridDim.x) * TPC + tin) << 4;
          ++cm;
          if (KS > 1) {
            if (ks > 0) {
#pragma unroll
              for (int nt = 0; nt < NT; nt++)
#pragma unroll
                for (int c = 0; c < 4; c++) red[warp][nt][lane][c] = acc[nt][c];
            }
            __syncthreads();
            if (ks == 0) {
#pragma unroll
              for (int q = 1; q < KS; q++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                  for (int c = 0; c < 4; c++) acc[nt][c] += red[warp + q][nt][lane][c];
              store_tile<NT, true>(acc, out, row, g, t);
            }
            __syncthreads();
          } else {
            store_tile<NT, true>(acc, out, row, g, t);
          }
#pragma unroll
          for (int nt = 0; nt < NT; nt++) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
        }
      }
    }
  }
}

// ---------------------------------------------------------------- CUDA-core T=1 GEMVs
// warp task = R consecutive rows x full K; lane owns 16 B (32 int4 weights / 8 bf16) of each row per 512 B tile; fp32 FMA; flat (task,tile)
// stream with register prefetch U tiles ahead (crosses task boundaries).  Activation: permuted copy (lane-contiguous) read through L1.
enum { CC_E1 = 0,     // E1 dequant: LOP3 (nibble into fp32 mantissa) + FADD (-1 - 8*2^e) + FFMA with power-of-two prescaled bf16 act
       CC_NAIVE = 1,  // (float)(((u >> 4e) & 15) - 8) per weight (I2F) + FFMA with plain act
       CC_ABL_FMA = 2,// no dequant: bitcast(u) used as the weight in the FFMA (INVALID), act loads + FFMAs kept
       CC_ABL_LOAD = 3 }; // loads only: xor of the weight words (INVALID)

__device__ __forceinline__ float act_prescale(int e) {
  const float t[8] = {8388608.f, 524288.f, 32768.f, 2048.f, 128.f, 8388608.f, 524288.f, 32768.f};
  return t[e];
}
template <int P>
__device__ __forceinline__ float nib(uint32_t u) {
  return __uint_as_float(0x3F800000u | (u & (0xFu << P))) - __uint_as_float(0x3F800000u | (1u << (P + 3)));
}

template <int MODE, int R>
__device__ __forceinline__ void cc_compute_w4(const uint4 (&w)[R], const uint32_t (&sc)[R], const uint4 (&a)[4], float (&acc)[R]) {
  if (MODE == CC_ABL_LOAD) {
#pragma unroll
    for (int r = 0; r < R; r++) acc[r] = __uint_as_float(__float_as_uint(acc[r]) ^ w[r].x ^ w[r].y ^ w[r].z ^ w[r].w ^ sc[r]);
    return;
  }
  float af[32];
#pragma unroll
  for (int c = 0; c < 4; c++) {
    af[c * 8 + 0] = bflo(a[c].x); af[c * 8 + 1] = bfhi(a[c].x);
    af[c * 8 + 2] = bflo(a[c].y); af[c * 8 + 3] = bfhi(a[c].y);
    af[c * 8 + 4] = bflo(a[c].z); af[c * 8 + 5] = bfhi(a[c].z);
    af[c * 8 + 6] = bflo(a[c].w); af[c * 8 + 7] = bfhi(a[c].w);
  }
#pragma unroll
  for (int r = 0; r < R; r++) {
    float S0 = 0.f, S1 = 0.f;
    const uint32_t ww[4] = {w[r].x, w[r].y, w[r].z, w[r].w};
#pragma unroll
    for (int c = 0; c < 4; c++) {
      const uint32_t u = ww[c];
      if (MODE == CC_E1) {
        const uint32_t u2 = u >> 20;
        S0 = fmaf(nib<0>(u), af[c * 8 + 0], S0);
        S1 = fmaf(nib<4>(u), af[c * 8 + 1], S1);
        S0 = fmaf(nib<8>(u), af[c * 8 + 2], S0);
        S1 = fmaf(nib<12>(u), af[c * 8 + 3], S1);
        S0 = fmaf(nib<16>(u), af[c * 8 + 4], S0);
        S1 = fmaf(nib<0>(u2), af[c * 8 + 5], S1);
        S0 = fmaf(nib<4>(u2), af[c * 8 + 6], S0);
        S1 = fmaf(nib<8>(u2), af[c * 8 + 7], S1);
      } else if (MODE == CC_NAIVE) {
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          S0 = fmaf(__int2float_rn((int)((u >> (4 * e)) & 15u) - 8), af[c * 8 + e], S0);
          S1 = fmaf(__int2float_rn((int)((u >> (4 * e + 4)) & 15u) - 8), af[c * 8 + e + 1], S1);
        }
      } else {  // CC_ABL_FMA
        const float v = __uint_as_float(u);
#pragma unroll
        for (int e = 0; e < 8; e += 2) { S0 = fmaf(v, af[c * 8 + e], S0); S1 = fmaf(v, af[c * 8 + e + 1], S1); }
      }
    }
    acc[r] = fmaf(bflo(sc[r]), S0 + S1, acc[r]);
  }
}

template <int R>
__device__ __forceinline__ void cc_compute_bf16(const uint4 (&w)[R], const uint4& a, float (&acc)[R]) {
  const float a0 = bflo(a.x), a1 = bfhi(a.x), a2 = bflo(a.y), a3 = bfhi(a.y);
  const float a4 = bflo(a.z), a5 = bfhi(a.z), a6 = bflo(a.w), a7 = bfhi(a.w);
#pragma unroll
  for (int r = 0; r < R; r++) {
    float s0 = acc[r], s1 = 0.f;
    s0 = fmaf(bflo(w[r].x), a0, s0); s1 = fmaf(bfhi(w[r].x), a1, s1);
    s0 = fmaf(bflo(w[r].y), a2, s0); s1 = fmaf(bfhi(w[r].y), a3, s1);
    s0 = fmaf(bflo(w[r].z), a4, s0); s1 = fmaf(bfhi(w[r].z), a5, s1);
    s0 = fmaf(bflo(w[r].w), a6, s0); s1 = fmaf(bfhi(w[r].w), a7, s1);
    acc[r] = s0 + s1;
  }
}

// W4 = true: tile = 1024 k (8 groups), act = 4 uint4/lane (slot t*128 + c*32 + lane);  W4 = false (bf16): tile = 256 k, act = 1 uint4/lane
template <bool W4, int MODE, int NW, int R, int U, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB)
    k_cc(const uint4* __restrict__ W, const uint16_t* __restrict__ S, const uint4* __restrict__ XA, float* __restrict__ out, int N, int K) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gw = blockIdx.x * NW + warp, Wn = gridDim.x * NW;
  const int tiles = W4 ? (K >> 10) : (K >> 8);
  const int rowu = W4 ? (K >> 5) : (K >> 3);
  const int nkg = K >> 7;
  const int ntasks = N / R;
  const int mytasks = gw < ntasks ? (ntasks - gw + Wn - 1) / Wn : 0;
  const int total = mytasks * tiles;

  uint4 wr[U][R];
  uint32_t sr[U][R];
  int lm = 0, lt = 0;  // load cursor (task index m, tile)
  auto issue = [&](int j) {
    const size_t row0 = (size_t)(gw + (size_t)lm * Wn) * R;
#pragma unroll
    for (int r = 0; r < R; r++) {
      wr[j][r] = ldw(W + (row0 + r) * rowu + (size_t)lt * 32 + lane);
      if (W4) sr[j][r] = ld16(S + (row0 + r) * nkg + lt * 8 + (lane >> 2));
    }
    if (++lt == tiles) { lt = 0; ++lm; }
  };
#pragma unroll
  for (int j = 0; j < U; j++)
    if (j < total) issue(j);
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; r++) acc[r] = 0.f;
  uint4 an[4];
  an[0] = an[1] = an[2] = an[3] = make_uint4(0, 0, 0, 0);
  if (total > 0 && !(W4 && MODE == CC_ABL_LOAD)) {
    if (W4) {
#pragma unroll
      for (int c = 0; c < 4; c++) an[c] = ldx(XA + (size_t)c * 32 + lane);
    } else an[0] = ldx(XA + lane);
  }
  int ct = 0, cm = 0;
  for (int f = 0; f < total; f += U) {
#pragma unroll
    for (int j = 0; j < U; j++) {
      if (f + j < total) {
        uint4 w[R];
        uint32_t sc[R];
#pragma unroll
        for (int r = 0; r < R; r++) { w[r] = wr[j][r]; sc[r] = W4 ? sr[j][r] : 0u; }
        if (f + j + U < total) issue(j);
        if (W4) {
          uint4 a[4];
#pragma unroll
          for (int c = 0; c < 4; c++) a[c] = an[c];
          if (f + j + 1 < total && MODE != CC_ABL_LOAD) {
            const int nx = ct + 1 == tiles ? 0 : ct + 1;
#pragma unroll
            for (int c = 0; c < 4; c++) an[c] = ldx(XA + (size_t)nx * 128 + c * 32 + lane);
          }
          cc_compute_w4<MODE, R>(w, sc, a, acc);
        } else {
          const uint4 a = an[0];
          if (f + j + 1 < total) an[0] = ldx(XA + (size_t)(ct + 1 == tiles ? 0 : ct + 1) * 32 + lane);
          cc_compute_bf16<R>(w, a, acc);
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

// ---------------------------------------------------------------- read-only roofline kernel (same bytes, xor-reduce)
__global__ void __launch_bounds__(512) k_read(const uint4* __restrict__ a, size_t na, const uint4* __restrict__ b, size_t nb, uint32_t* sink) {
  uint32_t x = 0;
  const size_t n = na + nb;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    uint4 v[4];
#pragma unroll
    for (int q = 0; q < 4; q++) { size_t k = i + q * stride; v[q] = k < na ? ldw(a + k) : ldw(b + (k - na)); }
#pragma unroll
    for (int q = 0; q < 4; q++) x ^= v[q].x ^ v[q].y ^ v[q].z ^ v[q].w;
  }
  for (; i < n; i += stride) { uint4 v = i < na ? ldw(a + i) : ldw(b + (i - na)); x ^= v.x ^ v.y ^ v.z ^ v.w; }
  if (x == 0x9e3779b9u) sink[0] = x;
}

}  // namespace e3
