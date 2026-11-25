#pragma once
#include "dense_common.cuh"

// Fused dequant GEMV y[T][N] = x[T][K] W[N][K]^T over GGUF Q4_0 / Q4_1 weights, repacked at load time (warpc/cuda/pack.py) so that
// a lane reads the weights of the columns it owns with one aligned 8-byte load. The lane -> column map is k_gemv's: in round
// j (0 .. K/256 - 1) lane l owns the 8 columns 8k .. 8k + 7 of chunk k = 32 j + l, so a warp reads 512 contiguous bytes of x
// per round, and a lane accumulates its chunks in ascending order with fmaf in column order; the 32 lane partials are
// summed by k_gemv's butterfly, so the reduction order is k_gemv's (a function of K only). K is a multiple of 512. A row is
//   qs  K/512 pairs of rounds x 32 lanes x 8 bytes: two 32-bit words, round 2p then 2p + 1, nibble m (bit 4m) = the
//       code of column 8k + m
//   d   K/512 pairs x 8 groups x 2 rounds x f16: the scale of ggml block 8j + g (g = lane / 4, the 4 lanes of a block
//       read the same 4 bytes)                                    [Q4_1: then the same plane of minimums m]
// and takes KB x 18 (Q4_1: 20) bytes, KB = K/32, a multiple of 16 so that every row starts 16-byte aligned. The weight
// of a column is w = fma(d, q, -8d) (Q4_0) or fma(d, q, m) (Q4_1), q the unsigned nibble as an exact float: the exact
// d * (q - 8) / the correctly rounded d * q + m, bit-identical to the GGUF source's decoders (and ggml's). Q4_0 weights stay
// fp32 for the T rows' dot products (a superset of the bf16 path's numerics, the intended one); Q4_1 weights are rounded
// to bf16 as the dense path widened them before this kernel existed (the oracle of tools/check.py reads bf16 weights:
// fp32 Q4_1 weights measured further from it, 0.0338 against 0.0238 relative L2 on Qwen3-1.7B). Modes as in k_gemv: 0 bf16 outputs
// split by row range (up to 3 segments); 1 gate/up pair (w = gate, w2 = up), out = bf16(silu(bf16(gate))) * bf16(up); 3
// residual add, out[0] is the residual stream h [T][N] (bf16) and h = bf16(h + bf16(y)) in place (each element belongs to one
// row, so no cross-block ordering is needed). U pairs of rounds are loaded before any is used.
struct Q4GemvP {
  const uint8_t* w;  // N rows of rs bytes
  const uint8_t* w2;  // mode 1 only: the up rows, same layout
  const bf16* x;  // [T][K]
  int N, K;
  size_t rs;  // row stride in bytes
  int end[3];
  void* out[3];
};

// weights are read once: no L1 allocation (the activation rows stay cached)
__device__ __forceinline__ uint2 q4_ld8(const void* a) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(a));
  return v;
}
__device__ __forceinline__ uint32_t q4_ld4(const void* a) {
  uint32_t v;
  asm("ld.global.nc.L1::no_allocate.u32 %0, [%1];" : "=r"(v) : "l"(a));
  return v;
}

// 8 weights of one word: byte k of (v & 0x0F0F0F0F) / ((v >> 4) & 0x0F0F0F0F) is the nibble of column 2k / 2k + 1; 2^23 + q is
// a float whose mantissa is q, so the subtraction gives q exactly
template <int FMT>
__device__ __forceinline__ void q4_dequant_word(uint32_t v, float d, float md, float* f) {
  const uint32_t lo = v & 0x0F0F0F0Fu, hi = (v >> 4) & 0x0F0F0F0Fu;
#pragma unroll
  for (int k = 0; k < 4; k++) {
    f[2 * k] = fmaf(d, __uint_as_float(__byte_perm(lo, 0x4B000000u, 0x7440 + k)) - 8388608.f, md);
    f[2 * k + 1] = fmaf(d, __uint_as_float(__byte_perm(hi, 0x4B000000u, 0x7440 + k)) - 8388608.f, md);
  }
  if constexpr (FMT == 1) {
#pragma unroll
    for (int k = 0; k < 8; k += 2) {
      const float2 r = __bfloat1622float2(__floats2bfloat162_rn(f[k], f[k + 1]));
      f[k] = r.x;
      f[k + 1] = r.y;
    }
  }
}

__device__ __forceinline__ float q4_half(uint32_t pair, int r) { return __half2float(__ushort_as_half((unsigned short)(pair >> (16 * r)))); }

// One lane's share of one row in a pair of rounds: the two words and the packed scales (and minimums)
template <int FMT>
struct Q4Pair {
  uint2 q;
  uint32_t d, m;
  __device__ __forceinline__ void load(const uint8_t* row, int K, int p, int lane) {
    q = q4_ld8(row + 256 * (size_t)p + 8 * lane);
    d = q4_ld4(row + K / 2 + 32 * (size_t)p + 4 * (lane >> 2));
    if constexpr (FMT == 1) m = q4_ld4(row + K / 2 + K / 16 + 32 * (size_t)p + 4 * (lane >> 2));
  }
  __device__ __forceinline__ uint32_t word(int r) const { return r ? q.y : q.x; }
  __device__ __forceinline__ float scale(int r) const { return q4_half(d, r); }
  __device__ __forceinline__ float minimum(int r) const {
    if constexpr (FMT == 1) return q4_half(m, r);
    else return -8.f * q4_half(d, r);
  }
};

template <int FMT, int T, int RPW, int MODE, int U>
__global__ void __launch_bounds__(GEMV_WARPS * 32) k_q4_gemv(const __grid_constant__ Q4GemvP p) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row0 = (blockIdx.x * GEMV_WARPS + warp) * RPW;
  const int P = p.K / 512;  // pairs of rounds
  const int K8 = p.K / 8;
  const uint4* xw = reinterpret_cast<const uint4*>(p.x);
  const uint8_t* wr[RPW];
  const uint8_t* wr2[RPW];
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = min(row0 + i, p.N - 1);
    wr[i] = p.w + (size_t)r * p.rs;
    if constexpr (MODE == 1) wr2[i] = p.w2 + (size_t)r * p.rs;
  }
  float acc[RPW][T], acc2[RPW][T];
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) acc[i][t] = acc2[i][t] = 0.f;

  for (int p0 = 0; p0 < P; p0 += U) {
    Q4Pair<FMT> g[U][RPW], g2[U][MODE == 1 ? RPW : 1];
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (p0 + u < P) {
#pragma unroll
        for (int i = 0; i < RPW; i++) {
          g[u][i].load(wr[i], p.K, p0 + u, lane);
          if constexpr (MODE == 1) g2[u][i].load(wr2[i], p.K, p0 + u, lane);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; u++) {
      if (p0 + u < P) {
#pragma unroll
        for (int r = 0; r < 2; r++) {
          float xf[T][8];
#pragma unroll
          for (int t = 0; t < T; t++) unpack8(xw[(size_t)t * K8 + 32 * (2 * (p0 + u) + r) + lane], xf[t]);
#pragma unroll
          for (int i = 0; i < RPW; i++) {
            float qw[8];
            q4_dequant_word<FMT>(g[u][i].word(r), g[u][i].scale(r), g[u][i].minimum(r), qw);
#pragma unroll
            for (int t = 0; t < T; t++)
#pragma unroll
              for (int e = 0; e < 8; e++) acc[i][t] = fmaf(qw[e], xf[t][e], acc[i][t]);
            if constexpr (MODE == 1) {
              q4_dequant_word<FMT>(g2[u][i].word(r), g2[u][i].scale(r), g2[u][i].minimum(r), qw);
#pragma unroll
              for (int t = 0; t < T; t++)
#pragma unroll
                for (int e = 0; e < 8; e++) acc2[i][t] = fmaf(qw[e], xf[t][e], acc2[i][t]);
            }
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < RPW; i++)
#pragma unroll
    for (int t = 0; t < T; t++) {
      acc[i][t] = warp_sum(acc[i][t]);
      if (MODE == 1) acc2[i][t] = warp_sum(acc2[i][t]);
    }
  if (lane >= T) return;
#pragma unroll
  for (int i = 0; i < RPW; i++) {
    const int r = row0 + i;
    if (r >= p.N) continue;
    float gv = 0.f, u = 0.f;
#pragma unroll
    for (int t = 0; t < T; t++)
      if (lane == t) {
        gv = acc[i][t];
        u = acc2[i][t];
      }
    if (MODE == 0) {
      const int s = r < p.end[0] ? 0 : (r < p.end[1] ? 1 : 2);
      const int start = s == 0 ? 0 : p.end[s - 1];
      const int width = p.end[s] - start;
      reinterpret_cast<bf16*>(p.out[s])[(size_t)lane * width + (r - start)] = tb(gv);
    } else if (MODE == 3) {
      bf16* hr = reinterpret_cast<bf16*>(p.out[0]) + (size_t)lane * p.N + r;
      *hr = tb(bf(*hr) + rb(gv));
    } else {
      const float a = rb(gv);
      const float sl = rb(a / (1.f + __expf(-a)));
      reinterpret_cast<bf16*>(p.out[0])[(size_t)lane * p.N + r] = tb(sl * rb(u));
    }
  }
}

// The launch of one instantiation, defined in dense_q4_gemv_inst.cu (so that no translation unit but the instantiation units compiles a k_q4_gemv).
template <int FMT, int T, int RPW, int MODE>
void warpc_inst_q4_gemv(const Q4GemvP& p, int grid, cudaStream_t st);
