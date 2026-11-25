#pragma once
#include <type_traits>

#include "qwen35_attn_common.cuh"

// Tensor-core split-KV flash decode of the gated hd-256 attention with the bf16 cache (swap-AB: the keys are the M axis of m16n8k16, the T * G query columns of one KV head
// the N axis):
//
//   S^T[16 keys][8 cols] = K[16 keys][256 d] . Q^T      A = K rows, B = the query columns, fp32 accumulators
//   O^T[16 d][8 cols]   += V^T[16 d][16 keys] . P^T     A = V^T, B = P^T (the S^T accumulator transposed with movmatrix), 16 accumulator tiles
//
// A block is (split, KV head); its NT * KS warps are (n-tile, key slice): warp (nt, ks) owns the columns 8 nt .. 8 nt + 7 and the key tiles ks, ks + KS, ... of the split, with its own
// online softmax; the key slices are merged through shared memory at the end. K and V go from global memory to registers (no shared memory staging): the reduction axis of a mma
// may be permuted freely if A and B agree, so a lane (quad qd = lane & 3, row group g = lane >> 2) loads the chunks it needs.
//
// The arithmetic is the hand engine's attn_split_kernel (engine/src/attn.cu) operation for operation, so the output is bit for bit its: the same keys per online softmax (split
// geometry `geom` with 128 keys per split, tiles ks, ks + KS, ..., KS as in the hand's table), k step 2 c + u covers d = 32 c + 8 qd + 4 u + 0..3 and accumulates into the chain u
// (two chains, added at the end), scores (sa + sb) * scale * log2(e), P rounded to bf16 for the PV mma, l updated as ((l alpha + p0) + p2), the key slices merged
// o = sum_k exp2(m_k - M) o_k in order, the splits merged in four interleaved groups (k_merge). The V^T rows are free (each output d is its own dot product over the keys): rows g, g + 8 of
// the m-tile (G, mt) are the elements mt, mt + 4 of the 8-element chunk d = 128 a + 16 g + 8 h + e, G = 2 a + h (qwen35_attn_i4.cuh needs this for the 4-bit decode).
namespace g256 {

using u32 = uint32_t;

template <bool F16>
__device__ __forceinline__ void mma16(float (&c)[4], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  if (F16) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}
__device__ __forceinline__ u32 movT(u32 a) {
  u32 d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
template <bool F16>
__device__ __forceinline__ u32 pack2(float lo, float hi) {
  if (F16) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const u32*>(&v);
  }
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const u32*>(&v);
}
// streaming loads: one n-tile reads every row once (L1::no_allocate), several n-tiles of a block read the same rows from L1
template <int NT>
__device__ __forceinline__ uint4 ld16(const void* p) {
  uint4 v;
  if (NT == 1) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int NT>
__device__ __forceinline__ uint2 ld8(const void* p) {
  uint2 v;
  if (NT == 1) asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  else asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}
// nibble k of both 16-bit halves of w (codes of the d offsets k and k + 4 of an 8-code word, d = 8 w + offset, low nibble first): the codebook levels as fp16x2, evaluated as the hand
// engine's reader does ((2 n | 0x6400) is 1024 + 2 n, minus 1039 gives u = 2 n - 15 exactly, u * u is exact, the rest rounds to fp16): a shift and one LOP3 for two codes
__device__ __forceinline__ u32 lvp(u32 w, int k) {
  const u32 t = ((k ? (w >> (4 * k - 1)) : (w << 1)) & 0x001E001Eu) | 0x64006400u;
  const __half2 u = __hsub2(*reinterpret_cast<const __half2*>(&t), __float2half2_rn(1039.f));
  const __half2 f = __hfma2(__hmul2(u, u), __float2half2_rn(r4::CUBE_B), __float2half2_rn(1.f));
  const __half2 r = __hmul2(u, f);
  return *reinterpret_cast<const u32*>(&r);
}
__device__ __forceinline__ u32 comp(const uint4& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : (i == 2 ? v.z : v.w)); }
// f16-accumulate m16n8k16 (the hand engine's int4 scores): d = {(row g: cols 2 qd, 2 qd + 1), (row g + 8: ...)} as f16x2
__device__ __forceinline__ void mma_h(u32 (&d)[2], u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n" : "+r"(d[0]), "+r"(d[1]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float2 h2f(u32 x) { return __half22float2(*reinterpret_cast<const __half2*>(&x)); }

// O^T += V^T P^T over 16 exact rows: V[kq][G] = the 8-element chunk d = 128 a + 16 g + 8 h .. + 7 of the key slots {2 qd, 2 qd + 1, 2 qd + 8, 2 qd + 9}, b0 / b1 the P^T fragments
template <bool F16>
__device__ __forceinline__ void pv_exact(float (&O)[16][4], const uint4 (&V)[4][4], u32 b0, u32 b1) {
#pragma unroll
  for (int gg = 0; gg < 4; gg++)
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      // rows g, g + 8 are e = mt, mt + 4: element mt of a chunk is the (mt & 1) half of register mt / 2
      const int sel = (mt & 1) ? 0x7632 : 0x5410, lo = mt >> 1, hi = lo + 2;
      mma16<F16>(O[gg * 4 + mt], __byte_perm(comp(V[0][gg], lo), comp(V[1][gg], lo), sel), __byte_perm(comp(V[0][gg], hi), comp(V[1][gg], hi), sel),
                 __byte_perm(comp(V[2][gg], lo), comp(V[3][gg], lo), sel), __byte_perm(comp(V[2][gg], hi), comp(V[3][gg], hi), sel), b0, b1);
    }
}

template <int NT, int KS>
__global__ void __launch_bounds__(32 * NT * KS, 1) k_split(SplitArgs a) {
  using EL = bf16;
  constexpr int NW = NT * KS;
  constexpr float NEG = -INFINITY;
  extern __shared__ __align__(16) float smem[];
  const int T = a.T, Hq = a.Hq, Hkv = a.Hkv, G = Hq / Hkv, ctx = a.ctx;
  const int pos0 = *a.pos0, pe = pos0 + T - 1;
  const int ctxq = pe + 1;
  int nsl, chunk;
  geom(ctxq, a.smax, a.chmin, nsl, chunk);
  const int s = blockIdx.x, kvh = blockIdx.y;
  if (s >= nsl) return;
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, g8 = lane >> 2, qd = lane & 3;
  const int nt = wid % NT, ks = wid / NT;
  const int ncols = T * G;
  const int n0 = nt * 8 + 2 * qd;
  const int live0 = pos0 + n0 / G + 1, live1 = pos0 + (n0 + 1) / G + 1;  // keys below are visible to the lane's two columns

  // query fragments (B operand): column n = 8 nt + g8
  u32 qf[16][2];
  {
    const int n = nt * 8 + g8;
    const bool ok = n < ncols;
    const uint4* qr = reinterpret_cast<const uint4*>(a.q + ((size_t)(ok ? n / G : 0) * Hq + kvh * G + (ok ? n % G : 0)) * HD);
    // k step 2 c + u: d = 32 c + 8 qd + 4 u + 0..3 (the hand engine's reduction order of the scores)
#pragma unroll
    for (int c = 0; c < 8; c++) {
      const uint4 u = ok ? qr[4 * c + qd] : make_uint4(0, 0, 0, 0);
      qf[2 * c][0] = u.x;
      qf[2 * c][1] = u.y;
      qf[2 * c + 1][0] = u.z;
      qf[2 * c + 1][1] = u.w;
    }
  }

  float O[16][4];
#pragma unroll
  for (int i = 0; i < 16; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) O[i][j] = 0.f;
  float m0 = NEG, m1 = NEG, l0 = 0.f, l1 = 0.f;

  // online softmax step of one tile: scores S (fp32 from the mma, key g8 in S[0..1], key g8 + 8 in S[2..3], columns 2 qd, 2 qd + 1), keys >= lim of the column are masked.
  // Returns the probabilities p and rescales the accumulators.
  auto softmax = [&](float (&S)[4], int base, int lim0, int lim1, float (&p)[4]) {
    const int ka = base + g8, kb = ka + 8;
    float x[4];
    x[0] = ka < lim0 ? S[0] * a.scale2 : NEG;
    x[1] = ka < lim1 ? S[1] * a.scale2 : NEG;
    x[2] = kb < lim0 ? S[2] * a.scale2 : NEG;
    x[3] = kb < lim1 ? S[3] * a.scale2 : NEG;
    float t0 = fmaxf(x[0], x[2]), t1 = fmaxf(x[1], x[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, o));
      t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, o));
    }
    const float n0m = fmaxf(m0, t0), n1m = fmaxf(m1, t1);
    const float al0 = m0 == NEG ? 0.f : exp2f(m0 - n0m), al1 = m1 == NEG ? 0.f : exp2f(m1 - n1m);
    p[0] = x[0] == NEG ? 0.f : exp2f(x[0] - n0m);
    p[1] = x[1] == NEG ? 0.f : exp2f(x[1] - n1m);
    p[2] = x[2] == NEG ? 0.f : exp2f(x[2] - n0m);
    p[3] = x[3] == NEG ? 0.f : exp2f(x[3] - n1m);
    l0 = l0 * al0 + p[0] + p[2];
    l1 = l1 * al1 + p[1] + p[3];
    m0 = n0m;
    m1 = n1m;
    if (__any_sync(0xffffffffu, al0 != 1.f || al1 != 1.f)) {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        O[i][0] *= al0;
        O[i][1] *= al1;
        O[i][2] *= al0;
        O[i][3] *= al1;
      }
    }
  };

  // a run of exact tiles of the bf16 cache: tile i holds the 16 keys base0 + i step .. + 15 (while base < end), K rows of `kb`, V rows of `vb`; keys >= min(end, the column's live
  // length) are masked. The loads of the next tile are issued chunk by chunk as the registers of the current one die (K chunk c after its mma, V group gg after its mmas), so they
  // fly during the rest of the tile.
  auto exact_seq = [&](const bf16* kb, const bf16* vb, int kmax, int base0, int step, int end) {
    if (base0 >= end) return;
    const int lim0 = min(end, live0), lim1 = min(end, live1);
    struct Rows {
      const bf16 *k0, *k1, *v[4];
    };
    auto rows_of = [&](int base) {
      Rows r;
      r.k0 = kb + (size_t)min(base + g8, kmax) * HD + 8 * qd;
      r.k1 = kb + (size_t)min(base + g8 + 8, kmax) * HD + 8 * qd;
#pragma unroll
      for (int kq = 0; kq < 4; kq++) r.v[kq] = vb + (size_t)min(base + 2 * qd + (kq & 1) + 8 * (kq >> 1), kmax) * HD + 16 * g8;
      return r;
    };
    uint4 K0[8], K1[8], V[4][4];
    {
      const Rows r = rows_of(base0);
#pragma unroll
      for (int c = 0; c < 8; c++) {
        K0[c] = ld16<NT>(r.k0 + 32 * c);
        K1[c] = ld16<NT>(r.k1 + 32 * c);
      }
#pragma unroll
      for (int kq = 0; kq < 4; kq++)
#pragma unroll
        for (int gg = 0; gg < 4; gg++) V[kq][gg] = ld16<NT>(r.v[kq] + 128 * (gg >> 1) + 8 * (gg & 1));
    }
    for (int base = base0; base < end; base += step) {
      const bool more = base + step < end;
      const Rows rn = rows_of(more ? base + step : base);
      float S[4] = {0.f, 0.f, 0.f, 0.f};
      float sb[4] = {0.f, 0.f, 0.f, 0.f};  // the odd k steps' chain (the hand engine's two chains, added at the end)
#pragma unroll
      for (int c = 0; c < 8; c++) {
        mma16<false>(S, K0[c].x, K1[c].x, K0[c].y, K1[c].y, qf[2 * c][0], qf[2 * c][1]);
        mma16<false>(sb, K0[c].z, K1[c].z, K0[c].w, K1[c].w, qf[2 * c + 1][0], qf[2 * c + 1][1]);
        if (more) {
          K0[c] = ld16<NT>(rn.k0 + 32 * c);
          K1[c] = ld16<NT>(rn.k1 + 32 * c);
        }
      }
#pragma unroll
      for (int i = 0; i < 4; i++) S[i] += sb[i];
      float p[4];
      softmax(S, base, lim0, lim1, p);
      const u32 b0 = movT(pack2<false>(p[0], p[1])), b1 = movT(pack2<false>(p[2], p[3]));
#pragma unroll
      for (int gg = 0; gg < 4; gg++) {
#pragma unroll
        for (int mt = 0; mt < 4; mt++) {
          const int sel = (mt & 1) ? 0x7632 : 0x5410, lo = mt >> 1, hi = lo + 2;
          mma16<false>(O[gg * 4 + mt], __byte_perm(comp(V[0][gg], lo), comp(V[1][gg], lo), sel), __byte_perm(comp(V[0][gg], hi), comp(V[1][gg], hi), sel),
                       __byte_perm(comp(V[2][gg], lo), comp(V[3][gg], lo), sel), __byte_perm(comp(V[2][gg], hi), comp(V[3][gg], hi), sel), b0, b1);
        }
        if (more) {
#pragma unroll
          for (int kq = 0; kq < 4; kq++) V[kq][gg] = ld16<NT>(rn.v[kq] + 128 * (gg >> 1) + 8 * (gg & 1));
        }
      }
    }
  };

  const int lo0 = s * chunk, hi = min(ctxq, lo0 + chunk);
  exact_seq(reinterpret_cast<const bf16*>(a.kv) + (size_t)kvh * ctx * HD, reinterpret_cast<const bf16*>(a.kv) + ((size_t)Hkv + kvh) * ctx * HD, pe, lo0 + 16 * ks, 16 * KS, hi);

  // merge of the key slices and the partial of this split: [T][Hq][smax][HD + 4] (o, m in log2 units, l)
  l0 += __shfl_xor_sync(0xffffffffu, l0, 4);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 4);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 8);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 8);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 16);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 16);
  // the dump of O pads one float per 16 d (rows g of a warp are 16 d apart: without it 16 lanes hit one bank)
  constexpr int PC = HD + HD / 16;
  float* ow = smem + wid * 8 * PC;
  float* mlw = smem + NW * 8 * PC + wid * 16;
#pragma unroll
  for (int gg = 0; gg < 4; gg++)
#pragma unroll
    for (int mt = 0; mt < 4; mt++) {
      const int d = 128 * (gg >> 1) + 16 * g8 + 8 * (gg & 1) + mt;
      ow[(2 * qd) * PC + d + (d >> 4)] = O[gg * 4 + mt][0];
      ow[(2 * qd) * PC + (d + 4) + ((d + 4) >> 4)] = O[gg * 4 + mt][2];
      ow[(2 * qd + 1) * PC + d + (d >> 4)] = O[gg * 4 + mt][1];
      ow[(2 * qd + 1) * PC + (d + 4) + ((d + 4) >> 4)] = O[gg * 4 + mt][3];
    }
  if (g8 == 0) {
    mlw[(2 * qd) * 2] = m0;
    mlw[(2 * qd) * 2 + 1] = l0;
    mlw[(2 * qd + 1) * 2] = m1;
    mlw[(2 * qd + 1) * 2 + 1] = l1;
  }
  __syncthreads();
  // the key slices of each column: weights exp2(m_k - M), M, L = sum_k w_k l_k (one thread per column), then o = sum_k w_k o_k per element, k ascending (the hand engine's order)
  const float* mls = smem + NW * 8 * PC;
  float* wts = smem + NW * 8 * PC + NW * 16;  // [NT * 8][KS] weights
  if (threadIdx.x < NT * 8) {
    const int col = threadIdx.x, ntc = col >> 3, cc = col & 7;
    float M = NEG;
#pragma unroll
    for (int k = 0; k < KS; k++) M = fmaxf(M, mls[((k * NT + ntc) * 8 + cc) * 2]);
    float L = 0.f;
#pragma unroll
    for (int k = 0; k < KS; k++) {
      const int w = k * NT + ntc;
      const float mk = mls[(w * 8 + cc) * 2];
      const float wk = M == NEG || mk == NEG ? 0.f : exp2f(mk - M);
      wts[col * KS + k] = wk;
      L += wk * mls[(w * 8 + cc) * 2 + 1];
    }
    if (col < ncols) {
      float* dst = a.part + (((size_t)(col / G) * Hq + kvh * G + col % G) * a.smax + s) * (HD + 4);
      dst[HD] = M;
      dst[HD + 1] = L;
    }
  }
  __syncthreads();
  for (int col = 0; col < ncols; col++) {
    const int ntc = col >> 3, cc = col & 7;
    float* dst = a.part + (((size_t)(col / G) * Hq + kvh * G + col % G) * a.smax + s) * (HD + 4);
    for (int d = threadIdx.x; d < HD; d += NW * 32) {
      float o = 0.f;
#pragma unroll
      for (int k = 0; k < KS; k++) o += wts[col * KS + k] * smem[((k * NT + ntc) * 8 + cc) * PC + d + (d >> 4)];
      dst[d] = o;
    }
  }
}

template <int NT, int KS>
int run_split(const SplitArgs& a, cudaStream_t st) {
  constexpr size_t smem = ((size_t)NT * KS * (8 * (HD + HD / 16) + 16) + (size_t)NT * 8 * KS) * sizeof(float);
  static bool attr_set = false;  // cudaFuncSetAttribute is not stream ordered: legal during graph capture
  if (!attr_set) {
    const cudaError_t e = cudaFuncSetAttribute(k_split<NT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  k_split<NT, KS><<<dim3(a.smax, a.Hkv), 32 * NT * KS, smem, st>>>(a);
  return cudaGetLastError();
}

}  // namespace g256
