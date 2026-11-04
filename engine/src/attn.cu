// swift-mk K2: full-attention decode, see include/attn.h for the API and docs/ARCHITECTURE.md for the contract.  Clean-room.
//
// Design notes
//  * Memory bound on the KV cache (2 KB/token/layer fp8, 4 KB bf16), so every SM must stream: split-KV grid (S_MAX splits x 4 kv heads) sized for
//    the maximum context; the live number of splits / keys per split is derived on the device from *pos0 (+T), blocks beyond it return at once.
//  * Per block: NT x KS warps.  NT = number of 8-column n-tiles needed for the 6*T query columns of one kv head (col c = 6*t + head_in_group);
//    KS = key-parallel warps.  Warp (nt, ks) owns n-tile nt and the 16-key tiles ks, ks+KS, ...; independent online softmax per warp,
//    merged once at the end through shared memory.  No block-level barrier inside the main loop.
//  * "Swap-AB" mma m16n8k16:  S^T[16 keys][8 cols] = K[16 keys][d] . Q^T,   O^T[d][8 cols] += V^T[d][16 keys] . P^T[keys][cols].
//    K / V never touch shared memory: every lane loads 16-byte chunks straight from global memory.  The reduction axis of a mma may be
//    permuted freely if A and B agree, so the chunk a lane loads (contiguous d for QK, contiguous d of one key for PV with the d rows of the
//    mma permuted accordingly) is exactly what that lane must feed.  Q fragments are built once in the same permutation.
//    movmatrix.trans turns the S^T accumulator into the P^T B fragment without shared memory.
//  * fp8 KV is dequantised bit-exactly to fp16 with integer ops (value/256, no cvt instruction on sm_86): the factor 256 and the checkpoint scale
//    are folded into the score scale / the final output scale.  fp8 path computes in fp16 (q, p fp16), bf16 path in bf16.
#include "attn.h"
#include "kvrot.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <type_traits>

namespace {

constexpr int HD = 256, NKV = 4, NH = 24, GRP = 6, QG_STRIDE = 12288, TMAX = 8;
constexpr int S_MAX = SWIFTMK_ATTN_MAX_SPLITS;
#ifndef ATTN_CH_MIN
#define ATTN_CH_MIN 128  // minimum keys per split
#endif
#ifndef ATTN_KS1
#define ATTN_KS1 4  // key-parallel warps per block for T=1 (NT=1), T=2 (NT=2), T=3,4 (NT=3); measured best: 4 / 4 / 2
#endif
#ifndef ATTN_KS2
#define ATTN_KS2 4
#endif
#ifndef ATTN_KS3
#define ATTN_KS3 2
#endif
// K11, T = 5..8 (NT = ceil(6T/8) = 4 / 5 / 6 / 6 n-tiles): key-parallel warps per n-tile
#ifndef ATTN_KS5
#define ATTN_KS5 2
#endif
#ifndef ATTN_KS6
#define ATTN_KS6 1
#endif
#ifndef ATTN_KS7
#define ATTN_KS7 1
#endif
#ifndef ATTN_KS8
#define ATTN_KS8 1
#endif
constexpr int CH_MIN = ATTN_CH_MIN;
constexpr float LOG2E = 1.4426950408889634f;

// ---- workspace layout (bytes)
constexpr size_t WS_Q = 0;                                                  // u16 [TMAX][NH][HD]
constexpr int NCOLS_MAX = GRP * TMAX;                                       // 6 q heads x T rows per kv head
constexpr size_t WS_ML = WS_Q + (size_t)TMAX * NH * HD * 2;                // float2 [NKV][S_MAX][6T]
constexpr size_t WS_PART = WS_ML + (size_t)NKV * S_MAX * NCOLS_MAX * sizeof(float2);  // float [NKV][S_MAX][6T][HD]
constexpr size_t WS_TOTAL = WS_PART + (size_t)NKV * S_MAX * NCOLS_MAX * HD * sizeof(float);

// Live split geometry from the context length (identical in split and reduce).
__host__ __device__ __forceinline__ void split_params(int ctx, int& nsplit, int& chunk) {
  ctx = ctx < 1 ? 1 : ctx;
  int n = (ctx + CH_MIN - 1) / CH_MIN;
  n = n < 1 ? 1 : (n > S_MAX ? S_MAX : n);
  chunk = (((ctx + n - 1) / n) + 15) & ~15;
  nsplit = (ctx + chunk - 1) / chunk;
}

// theta^(-l/32), theta = 1e7, l = 0..31 (correctly rounded doubles: rotate_half rope frequencies for the 64 rotary dims)
__device__ __constant__ double c_inv_freq[32] = {
    1.0, 0.6042963902381329, 0.3651741272548377, 0.220673406908459, 0.1333521432163324, 0.08058421877614819, 0.04869675251658631,
    0.029427271762092817, 0.01778279410038923, 0.010746078283213174, 0.006493816315762113, 0.003924189758484536, 0.0023713737056616554,
    0.0014330125702369627, 0.0008659643233600654, 0.0005232991146814947, 0.00031622776601683794, 0.00019109529749704405,
    0.00011547819846894582, 6.978305848598664e-05, 4.216965034285822e-05, 2.5482967479793464e-05, 1.539926526059492e-05,
    9.30572040929699e-06, 5.623413251903491e-06, 3.3982083289425593e-06, 2.053525026457146e-06, 1.2409377607517195e-06,
    7.498942093324558e-07, 4.531583637600818e-07, 2.7384196342643614e-07, 1.6548170999431814e-07};

// =====================================================================================================================
// prep: q/k norm + rope, KV write, q to workspace
// =====================================================================================================================
template <bool FP8, bool I4>
__global__ void __launch_bounds__(128) attn_prep_kernel(int T, const __nv_bfloat16* __restrict__ q_proj, const __nv_bfloat16* __restrict__ kk,
                                                        const __nv_bfloat16* __restrict__ vv, const __nv_bfloat16* __restrict__ q_norm,
                                                        const __nv_bfloat16* __restrict__ k_norm, void* kv, int ctx_max, float k_scale,
                                                        float v_scale, const int* __restrict__ pos0p, uint16_t* __restrict__ qbuf) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int t = gw >> 5, h = gw & 31;
  if (t >= T) return;
  const int pos = *pos0p + t;
  const __nv_bfloat16* src;
  const __nv_bfloat16* nw = nullptr;
  if (h < NH) { src = q_proj + (size_t)t * QG_STRIDE + h * 2 * HD; nw = q_norm; }
  else if (h < NH + NKV) { src = kk + (size_t)t * (NKV * HD) + (h - NH) * HD; nw = k_norm; }
  else { src = vv + (size_t)t * (NKV * HD) + (h - NH - NKV) * HD; }

  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = __bfloat162float(src[lane + 32 * i]);
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[i] * x[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = rsqrtf(ss * (1.f / HD) + 1e-6f);
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = x[i] * inv * (1.f + __bfloat162float(nw[lane + 32 * i]));
  }
  if (h < NH + NKV) {  // rope on dims 0..63: lane l holds d = l (x[0]) and d = l + 32 (x[1]) = a rotate_half pair
    // angle = pos * theta^(-l/32) in fp64 (an fp32 angle carries ~pos*6e-8 rad of rounding error), Cody-Waite reduced mod 2pi, then fp32 sincos.
    const double ang = (double)pos * c_inv_freq[lane];
    const double kq = rint(ang * 0.15915494309189535);
    const double rr = fma(-kq, 2.4492935982947064e-16, fma(-kq, 6.283185307179586, ang));
    float sn, cs;
    sincosf((float)rr, &sn, &cs);
    const float a = x[0], b = x[1];
    x[0] = a * cs - b * sn;
    x[1] = b * cs + a * sn;
  }
  if constexpr (I4) kvrot::wht256(x, lane);  // rotated int4 format (kvrot.cuh): q, k, v rows all live in the rotated basis, scores are unchanged
  if (h < NH) {
    uint16_t* dst = qbuf + ((size_t)t * NH + h) * HD;
#pragma unroll
    for (int i = 0; i < 8; i++) dst[lane + 32 * i] = (FP8 || I4) ? __half_as_ushort(__float2half_rn(x[i])) : __bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
  } else {
    const int isv = h >= NH + NKV;
    const int kvh = isv ? h - NH - NKV : h - NH;
    const size_t row = ((size_t)(isv * NKV + kvh) * ctx_max + pos) * HD;
    if (FP8) {
      const float is = 1.f / (isv ? v_scale : k_scale);
      uint8_t* dst = reinterpret_cast<uint8_t*>(kv) + row;
#pragma unroll
      for (int i = 0; i < 8; i++) dst[lane + 32 * i] = (uint8_t)__nv_cvt_float_to_fp8(x[i] * is, __NV_SATFINITE, __NV_E4M3);
    } else if (I4) {
      // rotated 4-bit format, see kvrot.cuh: codes [2][NKV][ctx][128] then u32 meta (4 x 8-bit log2 group scale) [2][NKV][ctx_max].
      uint32_t* meta = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(kv) + (size_t)2 * NKV * ctx_max * (HD / 2));
      kvrot::quant_store(x, lane, reinterpret_cast<uint8_t*>(kv) + (row >> 1), meta + (size_t)(isv * NKV + kvh) * ctx_max + pos);
      kvrot::sink_store(x, lane, reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(kv) + kvrot::ring_off(ctx_max)) + ((size_t)(isv * NKV + kvh) * kvrot::RING_R + (pos & (kvrot::RING_R - 1))) * HD);
      if (pos < kvrot::NSINK)
        kvrot::sink_store(x, lane, reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(kv) + kvrot::sink_off(ctx_max)) + ((size_t)(isv * NKV + kvh) * kvrot::NSINK + pos) * HD);
    } else {
      uint16_t* dst = reinterpret_cast<uint16_t*>(kv) + row;
#pragma unroll
      for (int i = 0; i < 8; i++) dst[lane + 32 * i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
    }
  }
}

// =====================================================================================================================
// split-KV flash decode
// =====================================================================================================================
template <bool FP8, bool I4>
__device__ __forceinline__ void mma_t(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  if (FP8 || I4) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}
// fp8 e4m3 pair held as bytes [0, b0, 0, b1] -> fp16x2 holding (b0, b1)/256 (exact incl. subnormals: the e4m3 bias is 8 below fp16's, so the value is 2^-8 x the true one).
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2_div256(uint32_t x) { return (x & 0x80008000u) | ((x >> 1) & 0x3f803f80u); }
// ---- int4 KV format (kvrot.cuh), per token and head: the row is Walsh-Hadamard rotated, 128 B of codes (2/byte, low nibble = even d, natural d order) in
// [2][NKV][ctx][128], then one u32 meta word (4 x 8-bit log2 scale, byte G) per row in [2][NKV][ctx].  y = scale_G * L(n), n = 0..15, G(d) = 2 (d>>7) + ((d&15)>>3).
// The mma consumes the codebook levels L(n) (companded odd integers, fp16); q is rotated at prep, so scores are unchanged, and the output is un-rotated in the reduce kernel:
//   K: S = sum_G scale_kG * (q'_G . L(n_k))            V: O' = sum_G sum_k (p_k * scale_kG) L(n_k)_G      (G = group of the d rows of the mma instance)
constexpr uint32_t I4_MAGIC = 0x64006400u;   // fp16 1024.0 in both halves
constexpr uint32_t I4_MAGIC15 = 0x640F640Fu;  // fp16 1039.0 in both halves
constexpr uint32_t I4_B2 = 0x1A251A25u;       // fp16 CUBE_B (kvrot.cuh) in both halves
constexpr uint32_t I4_ONE2 = 0x3C003C00u;     // fp16 1.0 in both halves
// nibble k of both 16-bit halves of w (n in 0..15) -> fp16x2 holding the codebook levels (L(n_lo), L(n_hi)), L(n) = u (1 + B u^2), u = 2n - 15.
// (2n | 0x6400) is 1024 + 2n, minus 1039 gives u exactly; u*u is exact in fp16, the rest rounds to fp16 (kvrot::level agrees to ~2^-11).
__device__ __forceinline__ uint32_t i4_pair(uint32_t w, int k) {
  const uint32_t t = ((k ? (w >> (4 * k - 1)) : (w << 1)) & 0x001E001Eu) | I4_MAGIC;
  uint32_t u, uu, f, r;
  asm("sub.f16x2 %0, %1, %2;\n" : "=r"(u) : "r"(t), "r"(I4_MAGIC15));
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(uu) : "r"(u), "r"(u));
  asm("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(f) : "r"(uu), "r"(I4_B2), "r"(I4_ONE2));
  asm("mul.rn.f16x2 %0, %1, %2;\n" : "=r"(r) : "r"(u), "r"(f));
  return r;
}
__device__ __forceinline__ uint32_t pack16(float lo, float hi, bool f16) {
  if (f16) { __half2 v = __floats2half2_rn(lo, hi); return *reinterpret_cast<uint32_t*>(&v); }
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}
__device__ __forceinline__ uint32_t movmatrix_t(uint32_t a) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(d) : "r"(a));
  return d;
}
template <int NT>
__device__ __forceinline__ uint4 ld_kv(const void* p) {
  uint4 v;
  if (NT == 1) asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  else asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
template <int NT>
__device__ __forceinline__ uint2 ld_kv8(const void* p) {  // int4: 8 bytes = 16 codes = one k16 step
  uint2 v;
  if (NT == 1) asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  else asm volatile("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

template <bool FP8, bool I4, int NT, int KS>
__global__ void __launch_bounds__(32 * NT * KS, 1)
    attn_split_kernel(int T, const void* __restrict__ kv, int ctx_max, float k_scale, float v_scale, const int* __restrict__ pos0p,
                      const uint16_t* __restrict__ qbuf, float2* __restrict__ ml, float* __restrict__ part) {
  constexpr int NW = NT * KS;
  constexpr int E = (FP8 || I4) ? 16 : 8;  // elements per 16-byte chunk
  constexpr int ROWB = I4 ? HD / 2 : (FP8 ? HD : HD * 2);  // bytes per token row: 128 (int4) / 256 (fp8) / 512 (bf16)
  constexpr int ECHB = I4 ? 8 : 16;                        // bytes per E-element block: 8 (int4) / 16 (fp8, bf16)
  constexpr int NKB = HD / (4 * E);        // K chunks per lane per row (64/32-element blocks); int4: 8-element (1 uint2) blocks
  constexpr int NVC = HD / (8 * E);        // V chunks per lane per key slot; int4: 16-element blocks
  constexpr int NSTEP = 16;                // k16 steps over d
  constexpr int NINST = 16;                // PV mma instances (16 d-rows each)
  // smem: Q fragments [NW][NSTEP][32] uint2 during the key loop, then (aliased, after a barrier) the merge buffers
  extern __shared__ __align__(16) float smem[];

  const int pos0 = *pos0p;
  const int ctx = I4 ? max(pos0 + T - kvrot::RING_W, 0) : pos0 + T;  // int4: the last RING_W keys are attended exactly by the reduce kernel (kvrot.cuh)
  int nsplit, chunk;
  split_params(ctx, nsplit, chunk);
  const int s = blockIdx.x, kvh = blockIdx.y;
  if (s >= nsplit) return;
  const int kstart = s * chunk, kend = min(ctx, kstart + chunk);
  const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31, r = lane >> 2, j = lane & 3;
  const int nt = wid % NT, ks = wid / NT;
  const int ncols = GRP * T;

  const char* kbase = reinterpret_cast<const char*>(kv) + (size_t)(0 * NKV + kvh) * ctx_max * ROWB;
  const char* vbase = reinterpret_cast<const char*>(kv) + (size_t)(1 * NKV + kvh) * ctx_max * ROWB;
  const uint16_t* sinkb = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(kv) + (I4 ? kvrot::sink_off(ctx_max) : 0));  // exact fp16 sink rows [2][NKV][NSINK][HD] (int4 only)
  const uint32_t* kmeta = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(kv) + (size_t)2 * NKV * ctx_max * ROWB) + (size_t)(0 * NKV + kvh) * ctx_max;
  const uint32_t* vmeta = reinterpret_cast<const uint32_t*>(reinterpret_cast<const char*>(kv) + (size_t)2 * NKV * ctx_max * ROWB) + (size_t)(1 * NKV + kvh) * ctx_max;
  const int ntiles = (kend - kstart + 15) / 16;
  struct Tile {
    typename std::conditional<I4, uint2, uint4>::type k[2][NKB];  // K rows r and r+8, chunk j (int4: 16 codes = one k16 step)
    typename std::conditional<I4, uint2, uint4>::type v[4][NVC];  // V key slots {2j, 2j+1, 2j+8, 2j+9}, chunk r
    uint32_t km[2], vm[2];                                        // int4 only: meta words (4 x 8-bit group scale code) of keys r and r+8, K and V
  };
  auto issue = [&](Tile& t, int it) {
    const int base = kstart + it * 16;
    const int klast = kend - 1;
    const char* k0 = kbase + (size_t)min(base + r, klast) * ROWB + j * ECHB;
    const char* k1 = kbase + (size_t)min(base + r + 8, klast) * ROWB + j * ECHB;
#pragma unroll
    for (int b = 0; b < NKB; b++) {
      if constexpr (I4) {
        t.k[0][b] = ld_kv8<NT>(k0 + b * 4 * ECHB);
        t.k[1][b] = ld_kv8<NT>(k1 + b * 4 * ECHB);
      } else {
        t.k[0][b] = ld_kv<NT>(k0 + b * 4 * ECHB);
        t.k[1][b] = ld_kv<NT>(k1 + b * 4 * ECHB);
      }
    }
    if constexpr (I4) {
      t.km[0] = __ldg(kmeta + min(base + r, klast)); t.km[1] = __ldg(kmeta + min(base + r + 8, klast));
      t.vm[0] = __ldg(vmeta + min(base + r, klast)); t.vm[1] = __ldg(vmeta + min(base + r + 8, klast));
    }
#pragma unroll
    for (int sl = 0; sl < 4; sl++) {
      const int kk_ = base + 2 * j + (sl & 1) + (sl >> 1) * 8;
      const char* vp = vbase + (size_t)min(kk_, klast) * ROWB + r * ECHB;
      for (int c = 0; c < NVC; c++) {
        if constexpr (I4) t.v[sl][c] = ld_kv8<NT>(vp + c * 8 * ECHB);
        else t.v[sl][c] = ld_kv<NT>(vp + c * 8 * ECHB);
      }
    }
  };

  // first tile's loads go out before anything else (they do not depend on Q)
  Tile ta, tb;
  if (ks < ntiles) issue(ta, ks);

  // ---- Q B-fragments (permuted reduction axis, see header comment): b0/b1 of step (b,u) = q[col][b*4E + j*E + 4u + {0..3}]
  uint2* qs = reinterpret_cast<uint2*>(smem) + wid * NSTEP * 32 + lane;
  {
    const int c = nt * 8 + r;
    const bool valid = c < ncols;
    const int tq = valid ? c / GRP : 0, hq = valid ? c - tq * GRP : 0;
    const uint16_t* qp = qbuf + ((size_t)tq * NH + kvh * GRP + hq) * HD;
#pragma unroll
    for (int sidx = 0; sidx < NSTEP; sidx++) {
      const int b = sidx / (E / 4), u = sidx % (E / 4);
      if constexpr (I4) {  // pairs of d slots (s0, s0+4) / (s0+1, s0+5) of chunk 4b+j: the layout i4_pair() produces from the K nibbles
        const uint16_t* qq = qp + b * 4 * E + j * E + 8 * (u >> 1) + 2 * (u & 1);
        qs[sidx * 32] = valid ? make_uint2((uint32_t)qq[0] | ((uint32_t)qq[4] << 16), (uint32_t)qq[1] | ((uint32_t)qq[5] << 16)) : make_uint2(0u, 0u);
      } else {
        qs[sidx * 32] = valid ? *reinterpret_cast<const uint2*>(qp + b * 4 * E + j * E + 4 * u) : make_uint2(0u, 0u);
      }
    }
  }
  __syncwarp();
  // columns owned by this lane in the accumulators: 8*nt + 2j + e
  int lim[2];
#pragma unroll
  for (int e = 0; e < 2; e++) {
    const int c = nt * 8 + 2 * j + e;
    const int tc = min(c / GRP, T - 1);
    lim[e] = min(pos0 + tc, kend - 1);  // key position allowed iff kp <= lim
  }
  const float scl = (FP8 ? k_scale * 256.f : 1.f) * (1.f / 16.f) * LOG2E;
  float acc[NINST][4];
#pragma unroll
  for (int i = 0; i < NINST; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};

  auto compute = [&](const Tile& t, int it) {
    const int base = kstart + it * 16;
    float ksc[2][4], vsc[2][4];  // int4: per-key group scales (G = 0..3) of keys r (index 0) and r+8 (index 1), K and V
    if constexpr (I4) {
#pragma unroll
      for (int G = 0; G < 4; G++) {
        ksc[0][G] = kvrot::scale_of((t.km[0] >> (8 * G)) & 255u); ksc[1][G] = kvrot::scale_of((t.km[1] >> (8 * G)) & 255u);
        vsc[0][G] = kvrot::scale_of((t.vm[0] >> (8 * G)) & 255u); vsc[1][G] = kvrot::scale_of((t.vm[1] >> (8 * G)) & 255u);
      }
    }
    // ---- S^T = K Q^T
    float sa[4] = {0.f, 0.f, 0.f, 0.f}, sb[4] = {0.f, 0.f, 0.f, 0.f};
    float sg[4][4];  // int4: one accumulator per scale group (steps (b, u) with G = 2 (b>>1) + (u>>1))
    if constexpr (I4) {
#pragma unroll
      for (int G = 0; G < 4; G++) sg[G][0] = sg[G][1] = sg[G][2] = sg[G][3] = 0.f;
    }
#pragma unroll
    for (int sidx = 0; sidx < NSTEP; sidx++) {
      const int b = sidx / (E / 4), u = sidx % (E / 4);
      const uint32_t* k0 = reinterpret_cast<const uint32_t*>(&t.k[0][b]);
      const uint32_t* k1 = reinterpret_cast<const uint32_t*>(&t.k[1][b]);
      uint32_t a0, a1, a2, a3;
      if constexpr (I4) {
        // chunk = uint2 = 16 codes; word (u>>1) feeds steps 2w, 2w+1.  a0 = nibbles (n, n+4), a2 = (n+1, n+5) of row r; a1 / a3 = row r+8; n = 2*(u&1).
        const uint2 K0 = t.k[0][b], K1 = t.k[1][b];
        const uint32_t w0 = (u >> 1) ? K0.y : K0.x, w1 = (u >> 1) ? K1.y : K1.x;
        a0 = i4_pair(w0, 2 * (u & 1)); a2 = i4_pair(w0, 2 * (u & 1) + 1);
        a1 = i4_pair(w1, 2 * (u & 1)); a3 = i4_pair(w1, 2 * (u & 1) + 1);
      } else if constexpr (FP8) {
        a0 = e4m3x2_to_f16x2_div256(__byte_perm(k0[u], 0, 0x1404));
        a2 = e4m3x2_to_f16x2_div256(__byte_perm(k0[u], 0, 0x3424));
        a1 = e4m3x2_to_f16x2_div256(__byte_perm(k1[u], 0, 0x1404));
        a3 = e4m3x2_to_f16x2_div256(__byte_perm(k1[u], 0, 0x3424));
      } else {
        a0 = k0[2 * u]; a2 = k0[2 * u + 1]; a1 = k1[2 * u]; a3 = k1[2 * u + 1];
      }
      const uint2 q = qs[sidx * 32];
      if constexpr (I4) mma_t<FP8, I4>(sg[2 * (b >> 1) + (u >> 1)], a0, a1, a2, a3, q.x, q.y);
      else mma_t<FP8, I4>((sidx & 1) ? sb : sa, a0, a1, a2, a3, q.x, q.y);
    }
    // ---- scores, mask, online softmax (columns e = 0/1 of this lane; rows = keys r, r+8)
    float sc[4];
#pragma unroll
    for (int i = 0; i < 4; i++) sc[i] = (sa[i] + sb[i]) * scl;
    if constexpr (I4) {  // S = sum_G scale_kG * (q'_G . (n_k - 8)); rows r / r+8 are keys, columns e = 0 / 1
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int row = i >> 1;
        sc[i] = ((sg[0][i] * ksc[row][0] + sg[1][i] * ksc[row][1]) + (sg[2][i] * ksc[row][2] + sg[3][i] * ksc[row][3])) * scl;
      }
    }
    const bool sink_tile = I4 && base == 0;  // tile 0 of split 0: keys 0..NSINK-1 use the exact fp16 sink rows (kvrot.cuh) instead of the 4-bit codes
    if constexpr (I4) {
      if (sink_tile) {
        // exact scores: lane (col = lane>>2, key = (lane&3) + 4m) computes q[col] . k_sink[key]; holder lanes (key r, columns 2j, 2j+1) fetch them by shuffle
        float dv[kvrot::NSINK / 4];
#pragma unroll
        for (int m = 0; m < kvrot::NSINK / 4; m++) {
          const int c = nt * 8 + (lane >> 2);
          const int tq = c < ncols ? c / GRP : 0, hq = c < ncols ? c - tq * GRP : 0;
          const uint4* qv = reinterpret_cast<const uint4*>(qbuf + ((size_t)tq * NH + kvh * GRP + hq) * HD);
          const uint4* kvp_ = reinterpret_cast<const uint4*>(sinkb + ((size_t)(0 * NKV + kvh) * kvrot::NSINK + (lane & 3) + 4 * m) * HD);
          float a = 0.f;
#pragma unroll 4
          for (int x8 = 0; x8 < HD / 8; x8++) {
            const uint4 qq = qv[x8], kk8 = kvp_[x8];
            const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w}, kw[4] = {kk8.x, kk8.y, kk8.z, kk8.w};
#pragma unroll
            for (int w = 0; w < 4; w++) {
              const float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&qw[w])), kf = __half22float2(*reinterpret_cast<const __half2*>(&kw[w]));
              a = fmaf(qf.x, kf.x, a);
              a = fmaf(qf.y, kf.y, a);
            }
          }
          dv[m] = a;
        }
        float ex[2];
#pragma unroll
        for (int e = 0; e < 2; e++) {
          ex[e] = 0.f;
#pragma unroll
          for (int m = 0; m < kvrot::NSINK / 4; m++) {
            const float vsh = __shfl_sync(0xffffffffu, dv[m], (2 * j + e) * 4 + (r & 3));
            if ((r >> 2) == m) ex[e] = vsh;
          }
        }
        if (r < kvrot::NSINK) { sc[0] = ex[0] * scl; sc[1] = ex[1] * scl; }
      }
    }
    const int kp_lo = base + r, kp_hi = base + r + 8;
    if (kp_lo > lim[0]) sc[0] = -INFINITY;
    if (kp_lo > lim[1]) sc[1] = -INFINITY;
    if (kp_hi > lim[0]) sc[2] = -INFINITY;
    if (kp_hi > lim[1]) sc[3] = -INFINITY;
    float mx0 = fmaxf(sc[0], sc[2]), mx1 = fmaxf(sc[1], sc[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float off0 = (mn0 == -INFINITY) ? 0.f : mn0, off1 = (mn1 == -INFINITY) ? 0.f : mn1;
    const float al0 = exp2f(m_run[0] - off0), al1 = exp2f(m_run[1] - off1);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const float p0 = exp2f(sc[0] - off0), p1 = exp2f(sc[1] - off1), p2 = exp2f(sc[2] - off0), p3 = exp2f(sc[3] - off1);
    l_run[0] = l_run[0] * al0 + p0 + p2;
    l_run[1] = l_run[1] * al1 + p1 + p3;
    float pk[I4 ? kvrot::NSINK : 1][2];
    if constexpr (I4) {
      if (sink_tile) {  // exact probabilities of the sink keys (held by lanes 4k + j) go to the exact-V update below; the quantised mma sees p = 0 for them
#pragma unroll
        for (int k = 0; k < kvrot::NSINK; k++) {
          pk[k][0] = __shfl_sync(0xffffffffu, p0, 4 * k + j);
          pk[k][1] = __shfl_sync(0xffffffffu, p1, 4 * k + j);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < NINST; i++) {
      acc[i][0] *= al0; acc[i][1] *= al1; acc[i][2] *= al0; acc[i][3] *= al1;
    }
    uint32_t pb0, pb1;
    uint32_t pbg[4][2];
    if constexpr (I4) {  // fold the per-key, per-group V scale into P: (p_k * scale_kG) multiplies the integer codes of the d rows of group G
#pragma unroll
      for (int G = 0; G < 4; G++) {
        const bool zs = sink_tile && r < kvrot::NSINK;
        pbg[G][0] = movmatrix_t(pack16(zs ? 0.f : p0 * vsc[0][G], zs ? 0.f : p1 * vsc[0][G], true));
        pbg[G][1] = movmatrix_t(pack16(p2 * vsc[1][G], p3 * vsc[1][G], true));
      }
      pb0 = pb1 = 0;
    } else {
      pb0 = movmatrix_t(pack16(p0, p1, FP8));
      pb1 = movmatrix_t(pack16(p2, p3, FP8));
    }
    // ---- O^T += V^T P^T : instance (c, g): d rows r -> c*8E + r*E + 2g, rows r+8 -> +1; k = key slots
#pragma unroll
    for (int c = 0; c < NVC; c++) {
      const uint32_t* v0 = reinterpret_cast<const uint32_t*>(&t.v[0][c]);
      const uint32_t* v1 = reinterpret_cast<const uint32_t*>(&t.v[1][c]);
      const uint32_t* v2 = reinterpret_cast<const uint32_t*>(&t.v[2][c]);
      const uint32_t* v3 = reinterpret_cast<const uint32_t*>(&t.v[3][c]);
#pragma unroll
      for (int g = 0; g < E / 2; g++) {
        uint32_t a0, a1, a2, a3;
        if constexpr (I4) {
          // slot sl = t.v[sl][c] = 16 codes (uint2).  mma rows r / r+8 = elems 2g / 2g+1: a0 = (s0.e2g, s1.e2g), a1 = (s0.e2g+1, s1.e2g+1), a2/a3 = slots 2,3.
          // prmt interleaves the low (or high) halfwords of the two key words, so nibble k of both halves is (key A code, key B code) = one fp16x2.
          const uint2 s0 = t.v[0][c], s1 = t.v[1][c], s2 = t.v[2][c], s3 = t.v[3][c];
          constexpr int dummy = 0; (void)dummy;
          const int wsel = g >> 2, n = (2 * g) & 7, nn = n & 3;  // compile-time: g is unrolled
          const uint32_t sel = n < 4 ? 0x5410u : 0x7632u;
          const uint32_t pab = __byte_perm(wsel ? s0.y : s0.x, wsel ? s1.y : s1.x, sel);
          const uint32_t pcd = __byte_perm(wsel ? s2.y : s2.x, wsel ? s3.y : s3.x, sel);
          a0 = i4_pair(pab, nn); a1 = i4_pair(pab, nn + 1); a2 = i4_pair(pcd, nn); a3 = i4_pair(pcd, nn + 1);
        } else if constexpr (FP8) {
          const int w = g >> 1, hi = g & 1;  // register w holds elems 4w..4w+3: g = 2w -> bytes 0,1; g = 2w+1 -> bytes 2,3
          const uint32_t sel = hi ? 0x7362u : 0x5140u;
          const uint32_t g01 = __byte_perm(v0[w], v1[w], sel);  // [s0.bX, s1.bX, s0.bX+1, s1.bX+1]
          const uint32_t g23 = __byte_perm(v2[w], v3[w], sel);
          a0 = e4m3x2_to_f16x2_div256(__byte_perm(g01, 0, 0x1404));
          a1 = e4m3x2_to_f16x2_div256(__byte_perm(g01, 0, 0x3424));
          a2 = e4m3x2_to_f16x2_div256(__byte_perm(g23, 0, 0x1404));
          a3 = e4m3x2_to_f16x2_div256(__byte_perm(g23, 0, 0x3424));
        } else {
          a0 = __byte_perm(v0[g], v1[g], 0x5410);
          a1 = __byte_perm(v0[g], v1[g], 0x7632);
          a2 = __byte_perm(v2[g], v3[g], 0x5410);
          a3 = __byte_perm(v2[g], v3[g], 0x7632);
        }
        if constexpr (I4) mma_t<FP8, I4>(acc[c * (E / 2) + g], a0, a1, a2, a3, pbg[2 * c + (g >> 2)][0], pbg[2 * c + (g >> 2)][1]);
        else mma_t<FP8, I4>(acc[c * (E / 2) + g], a0, a1, a2, a3, pb0, pb1);
      }
    }

    if constexpr (I4) {
      if (sink_tile) {  // acc[(c, g)] holds d0 = 128 c + 16 r + 2 g (elements 0/2: d0, d0+1 of column 2j; 1/3: column 2j+1)
#pragma unroll
        for (int c = 0; c < NVC; c++) {
#pragma unroll
          for (int g = 0; g < E / 2; g++) {
            const int d0 = c * 8 * E + r * E + 2 * g;
            float* a = acc[c * (E / 2) + g];
#pragma unroll
            for (int k = 0; k < kvrot::NSINK; k++) {
              const __half2 vh = *reinterpret_cast<const __half2*>(sinkb + ((size_t)(1 * NKV + kvh) * kvrot::NSINK + k) * HD + d0);
              const float2 vf = __half22float2(vh);
              a[0] = fmaf(pk[k][0], vf.x, a[0]);
              a[1] = fmaf(pk[k][1], vf.x, a[1]);
              a[2] = fmaf(pk[k][0], vf.y, a[2]);
              a[3] = fmaf(pk[k][1], vf.y, a[3]);
            }
          }
        }
      }
    }
  };

  // ---- key loop.  fp8: two tile buffers, the next tile's loads are in flight while the current one is computed.
  //      bf16: a tile is 128 registers per lane, so only one buffer fits: the next tile is requested right after the current one is consumed.
  if (FP8 || I4) {
    int it = ks;
    while (it < ntiles) {
      const int it1 = it + KS;
      if (it1 < ntiles) issue(tb, it1);
      compute(ta, it);
      if (it1 >= ntiles) break;
      const int it2 = it1 + KS;
      if (it2 < ntiles) issue(ta, it2);
      compute(tb, it1);
      it = it2;
    }
  } else {
    for (int it = ks; it < ntiles; it += KS) {
      compute(ta, it);
      if (it + KS < ntiles) issue(ta, it + KS);
    }
  }

  // ---- finish: complete l over the 8 key-row lanes, dump (m, l, acc) to smem, merge the KS warps of each n-tile
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l_run[0] += __shfl_xor_sync(0xffffffffu, l_run[0], o);
    l_run[1] += __shfl_xor_sync(0xffffffffu, l_run[1], o);
  }
  __syncthreads();  // every warp is done with its Q fragments: the merge buffers alias them
  float* sacc = smem;                           // [NW][8][HD]
  float* sm_m = smem + (size_t)NW * 8 * HD;     // [NW][8]
  float* sm_l = sm_m + NW * 8;                  // [NW][8]
  if (r == 0) {
    sm_m[wid * 8 + 2 * j] = m_run[0]; sm_m[wid * 8 + 2 * j + 1] = m_run[1];
    sm_l[wid * 8 + 2 * j] = l_run[0]; sm_l[wid * 8 + 2 * j + 1] = l_run[1];
  }
#pragma unroll
  for (int c = 0; c < NVC; c++) {
#pragma unroll
    for (int g = 0; g < E / 2; g++) {
      const int d0 = c * 8 * E + r * E + 2 * g;
      const float* a = acc[c * (E / 2) + g];
      *reinterpret_cast<float2*>(sacc + (size_t)(wid * 8 + 2 * j) * HD + d0) = make_float2(a[0], a[2]);
      *reinterpret_cast<float2*>(sacc + (size_t)(wid * 8 + 2 * j + 1) * HD + d0) = make_float2(a[1], a[3]);
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < NT * 8 * HD; e += NW * 32) {
    const int d = e & (HD - 1), col = e >> 8;
    if (col >= ncols) continue;
    const int nt_ = col >> 3, cc = col & 7;
    float M = -INFINITY;
#pragma unroll
    for (int k = 0; k < KS; k++) M = fmaxf(M, sm_m[(k * NT + nt_) * 8 + cc]);
    float o = 0.f;
    if (M != -INFINITY) {
#pragma unroll
      for (int k = 0; k < KS; k++) {
        const int w = k * NT + nt_;
        o += exp2f(sm_m[w * 8 + cc] - M) * sacc[(size_t)(w * 8 + cc) * HD + d];
      }
    }
    part[(((size_t)kvh * S_MAX + s) * ncols + col) * HD + d] = o;
  }
  if (threadIdx.x < NT * 8) {
    const int col = threadIdx.x, nt_ = col >> 3, cc = col & 7;
    if (col < ncols) {
      float M = -INFINITY;
#pragma unroll
      for (int k = 0; k < KS; k++) M = fmaxf(M, sm_m[(k * NT + nt_) * 8 + cc]);
      float L = 0.f;
      if (M != -INFINITY) {
#pragma unroll
        for (int k = 0; k < KS; k++) {
          const int w = k * NT + nt_;
          L += exp2f(sm_m[w * 8 + cc] - M) * sm_l[w * 8 + cc];
        }
      }
      ml[((size_t)kvh * S_MAX + s) * ncols + col] = make_float2(M, L);
    }
  }
}

// =====================================================================================================================
// reduce partials, 1/l, v scale, sigmoid gate
// =====================================================================================================================
template <bool I4>
__global__ void __launch_bounds__(256) attn_reduce_kernel(int T, const __nv_bfloat16* __restrict__ q_proj, float vscale_eff,
                                                          const int* __restrict__ pos0p, const float2* __restrict__ ml,
                                                          const float* __restrict__ part, __nv_bfloat16* __restrict__ out,
                                                          const void* __restrict__ kv, int ctx_max, const uint16_t* __restrict__ qbuf) {
  static_assert(S_MAX <= 64, "reduce kernel keeps two split weights per lane");
  static_assert(kvrot::RING_W == 128 || kvrot::RING_W == 256, "ring scores: 256 / RING_W threads per key");
  constexpr int TPK = 256 / kvrot::RING_W;
  __shared__ float wn[64];
  __shared__ float4 red[3][64];
  __shared__ float wh[I4 ? HD : 1];  // int4: the merged output in the rotated basis, un-rotated in place
  __shared__ float rs[I4 ? kvrot::RING_W : 1];  // int4: exact-ring probabilities exp2(s - M_r)
  __shared__ float rred[I4 ? 20 : 1];            // ring block reductions: [0..8) max, [8..16) sum, 16 = M_r, 17 = L_r, 18 = merge weight of the ring
  const int ncols = GRP * T;
  const int c = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x;
  const int pos0 = *pos0p;
  const int ctx = I4 ? max(pos0 + T - kvrot::RING_W, 0) : pos0 + T;  // keys covered by the split kernels; int4: the last RING_W keys are the exact ring
  int nsplit, chunk;
  split_params(ctx, nsplit, chunk);
  const int dl = tid & 63, sg = tid >> 6;  // float4 index over d, split group
  const int t = c / GRP, head = kvh * GRP + (c - t * GRP);
  uint2 vr[I4 ? kvrot::RING_W / 4 : 1];
  if constexpr (I4) {
    // ---- exact ring attention for this (column, kv head): keys kreg .. kreg + RING_W - 1 (masked beyond this column's position), fp16 rotated rows
    const uint16_t* ring = reinterpret_cast<const uint16_t*>(reinterpret_cast<const char*>(kv) + kvrot::ring_off(ctx_max));
    const uint16_t* ringK = ring + (size_t)(0 * NKV + kvh) * kvrot::RING_R * HD;
    const uint16_t* ringV = ring + (size_t)(1 * NKV + kvh) * kvrot::RING_R * HD;
    const int kreg = ctx;
#pragma unroll
    for (int u = 0; u < kvrot::RING_W / 4; u++) vr[u] = *reinterpret_cast<const uint2*>(ringV + (size_t)((kreg + sg + 4 * u) & (kvrot::RING_R - 1)) * HD + dl * 4);
    const int ki = tid / TPK, hf = tid % TPK, kp = kreg + ki;
    const uint4* kr = reinterpret_cast<const uint4*>(ringK + (size_t)(kp & (kvrot::RING_R - 1)) * HD + hf * (HD / TPK));
    const uint4* qr = reinterpret_cast<const uint4*>(qbuf + ((size_t)t * NH + head) * HD + hf * (HD / TPK));
    float a = 0.f;
#pragma unroll 4
    for (int x8 = 0; x8 < HD / (8 * TPK); x8++) {
      const uint4 qq = qr[x8], kk8 = kr[x8];
      const uint32_t qw[4] = {qq.x, qq.y, qq.z, qq.w}, kw[4] = {kk8.x, kk8.y, kk8.z, kk8.w};
#pragma unroll
      for (int w = 0; w < 4; w++) {
        const float2 qf = __half22float2(*reinterpret_cast<const __half2*>(&qw[w])), kf = __half22float2(*reinterpret_cast<const __half2*>(&kw[w]));
        a = fmaf(qf.x, kf.x, a);
        a = fmaf(qf.y, kf.y, a);
      }
    }
    if (TPK == 2) a += __shfl_xor_sync(0xffffffffu, a, 1);
    const float sc = (kp <= pos0 + t) ? a * (LOG2E / 16.f) : -INFINITY;
    float mx = sc;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((tid & 31) == 0) rred[tid >> 5] = mx;
    __syncthreads();
    float M = rred[0];
#pragma unroll
    for (int i = 1; i < 8; i++) M = fmaxf(M, rred[i]);
    const float p = (sc == -INFINITY) ? 0.f : exp2f(sc - M);
    if (hf == 0) rs[ki] = p;
    float ps = hf == 0 ? p : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, o);
    if ((tid & 31) == 0) rred[8 + (tid >> 5)] = ps;
    __syncthreads();
    if (tid == 0) {
      float L = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) L += rred[8 + i];
      rred[16] = M;
      rred[17] = L;
    }
    __syncthreads();
  }
  const float2* mlp = ml + (size_t)kvh * S_MAX * ncols + c;
  if (tid < 32) {  // normalised merge weights w_s = exp2(m_s - M) / sum_s' exp2(m_s' - M) l_s'
    float m[2], l[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const int sp = tid + 32 * i;
      const float2 v = sp < nsplit ? mlp[(size_t)sp * ncols] : make_float2(-INFINITY, 0.f);
      m[i] = v.x;
      l[i] = v.y;
    }
    float M = fmaxf(m[0], m[1]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
    float mr = -INFINITY, lr = 0.f;
    if constexpr (I4) { mr = rred[16]; lr = rred[17]; M = fmaxf(M, mr); }
    float w[2], L = 0.f;
#pragma unroll
    for (int i = 0; i < 2; i++) {
      w[i] = (m[i] == -INFINITY) ? 0.f : exp2f(m[i] - M);
      L += w[i] * l[i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) L += __shfl_xor_sync(0xffffffffu, L, o);
    const float wr = (mr == -INFINITY) ? 0.f : exp2f(mr - M);
    L += wr * lr;
    const float invL = 1.f / L;
    wn[tid] = w[0] * invL;
    wn[tid + 32] = w[1] * invL;
    if constexpr (I4) { if (tid == 0) rred[18] = wr * invL; }
  }
  __syncthreads();
  const float4* pp = reinterpret_cast<const float4*>(part + ((size_t)kvh * S_MAX * ncols + c) * HD) + dl;
  float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
  for (int sp = sg; sp < nsplit; sp += 4) {
    const float4 v = pp[(size_t)sp * ncols * (HD / 4)];
    const float w = wn[sp];
    o.x += w * v.x; o.y += w * v.y; o.z += w * v.z; o.w += w * v.w;
  }
  if constexpr (I4) {  // exact ring part: sum_i p_i V[kreg + i], this thread's key group sg (i = sg + 4u), dims 4 dl .. 4 dl + 3
    const float wr = rred[18];
#pragma unroll
    for (int u = 0; u < kvrot::RING_W / 4; u++) {
      const float pw = wr * rs[sg + 4 * u];
      const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].x)), v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].y));
      o.x = fmaf(pw, v01.x, o.x); o.y = fmaf(pw, v01.y, o.y); o.z = fmaf(pw, v23.x, o.z); o.w = fmaf(pw, v23.y, o.w);
    }
  }
  if (sg > 0) red[sg - 1][dl] = o;
  __syncthreads();
  if (sg == 0) {
#pragma unroll
    for (int k = 0; k < 3; k++) {
      const float4 v = red[k][dl];
      o.x += v.x; o.y += v.y; o.z += v.z; o.w += v.w;
    }
  }
  if constexpr (I4) {
    // the cache holds the Walsh-Hadamard rotated V rows: O = H O' / 16 (H/16 is symmetric orthogonal), 8 butterfly stages in shared memory
    if (sg == 0) *reinterpret_cast<float4*>(wh + dl * 4) = o;
    __syncthreads();
#pragma unroll
    for (int bit = 0; bit < 8; bit++) {
      if (tid < HD / 2) {
        const int lo = ((tid >> bit) << (bit + 1)) | (tid & ((1 << bit) - 1));
        const float a = wh[lo], b = wh[lo + (1 << bit)];
        wh[lo] = a + b;
        wh[lo + (1 << bit)] = a - b;
      }
      __syncthreads();
    }
    const float g = __bfloat162float(q_proj[(size_t)t * QG_STRIDE + head * 2 * HD + HD + tid]);
    out[(size_t)t * (NH * HD) + head * HD + tid] = __float2bfloat16_rn(wh[tid] * 0.0625f * vscale_eff / (1.f + expf(-g)));
  } else if (sg == 0) {
    const uint2 gb = *reinterpret_cast<const uint2*>(q_proj + (size_t)t * QG_STRIDE + head * 2 * HD + HD + dl * 4);
    const float g0 = __uint_as_float(gb.x << 16), g1 = __uint_as_float(gb.x & 0xffff0000u);
    const float g2 = __uint_as_float(gb.y << 16), g3 = __uint_as_float(gb.y & 0xffff0000u);
    const float y0 = o.x * vscale_eff / (1.f + expf(-g0)), y1 = o.y * vscale_eff / (1.f + expf(-g1));
    const float y2 = o.z * vscale_eff / (1.f + expf(-g2)), y3 = o.w * vscale_eff / (1.f + expf(-g3));
    __nv_bfloat162 lo = __floats2bfloat162_rn(y0, y1), hi = __floats2bfloat162_rn(y2, y3);
    uint2 st;
    st.x = *reinterpret_cast<uint32_t*>(&lo);
    st.y = *reinterpret_cast<uint32_t*>(&hi);
    *reinterpret_cast<uint2*>(out + (size_t)t * (NH * HD) + head * HD + dl * 4) = st;
  }
}

template <bool FP8, bool I4, int NT, int KS>
cudaError_t launch_split(int T, const void* kv, int ctx_max, float k_scale, float v_scale, const int* pos0, const uint16_t* qbuf, float2* ml, float* part,
                         cudaStream_t st) {
  constexpr int NW = NT * KS;
  const size_t smem = (size_t)NW * 8 * HD * sizeof(float) + (size_t)NW * 16 * sizeof(float);
  static bool attr_set = false;  // cudaFuncSetAttribute is not stream ordered: legal during graph capture
  if (!attr_set) {
    cudaError_t e = cudaFuncSetAttribute(attn_split_kernel<FP8, I4, NT, KS>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    attr_set = true;
  }
  attn_split_kernel<FP8, I4, NT, KS><<<dim3(S_MAX, NKV), NW * 32, smem, st>>>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part);
  return cudaGetLastError();
}

template <bool FP8, bool I4>
cudaError_t run(int T, const void* q_proj, const void* k, const void* v, const void* q_norm, const void* k_norm, void* kv, int ctx_max,
                float k_scale, float v_scale, const int* pos0, void* out, void* ws, cudaStream_t st) {
  char* w = reinterpret_cast<char*>(ws);
  uint16_t* qbuf = reinterpret_cast<uint16_t*>(w + WS_Q);
  float2* ml = reinterpret_cast<float2*>(w + WS_ML);
  float* part = reinterpret_cast<float*>(w + WS_PART);
  attn_prep_kernel<FP8, I4><<<T * 8, 128, 0, st>>>(T, (const __nv_bfloat16*)q_proj, (const __nv_bfloat16*)k, (const __nv_bfloat16*)v,
                                                   (const __nv_bfloat16*)q_norm, (const __nv_bfloat16*)k_norm, kv, ctx_max, k_scale, v_scale,
                                                   pos0, qbuf);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  switch (T) {
    case 1: e = launch_split<FP8, I4, 1, ATTN_KS1>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    case 2: e = launch_split<FP8, I4, 2, ATTN_KS2>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    case 3:
    case 4: e = launch_split<FP8, I4, 3, ATTN_KS3>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    case 5: e = launch_split<FP8, I4, 4, ATTN_KS5>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    case 6: e = launch_split<FP8, I4, 5, ATTN_KS6>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    case 7: e = launch_split<FP8, I4, 6, ATTN_KS7>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
    default: e = launch_split<FP8, I4, 6, ATTN_KS8>(T, kv, ctx_max, k_scale, v_scale, pos0, qbuf, ml, part, st); break;
  }
  if (e != cudaSuccess) return e;
  attn_reduce_kernel<I4><<<dim3(GRP * T, NKV), HD, 0, st>>>(T, (const __nv_bfloat16*)q_proj, FP8 ? v_scale * 256.f : 1.f, pos0, ml, part,
                                                         (__nv_bfloat16*)out, kv, ctx_max, qbuf);
  return cudaGetLastError();
}

}  // namespace

extern "C" {

size_t swiftmk_attn_ws_bytes(void) { return WS_TOTAL; }

// kv_dtype: 0 = bf16 (2 B/elem), 1 = fp8 (1 B/elem), 2 = int4: per token and head 128 B of codes + a 4 B meta word (4 x 8-bit log2 group scale, rotated rows, kvrot.cuh; 4.125 bit/elem).
// Layout: codes [K|V][NKV][ctx_max][128] then meta u32 [K|V][NKV][ctx_max].
size_t swiftmk_attn_kv_bytes(int ctx_max, int kv_dtype) {
  if (kv_dtype == 2) return kvrot::ring_off(ctx_max) + kvrot::RING_BYTES;  // + fixed fp16 sink rows and exact recent-token ring
  return (size_t)2 * NKV * (size_t)ctx_max * HD * (kv_dtype == 0 ? 2 : 1);
}

cudaError_t swiftmk_attn_step(int T, const void* q_proj, const void* k, const void* v, const void* q_norm, const void* k_norm, void* kv,
                              int ctx_max, int kv_dtype, float k_scale, float v_scale, const int* pos0, void* out, void* ws,
                              cudaStream_t stream) {
  if (T < 1 || T > TMAX) return cudaErrorInvalidValue;
  if (kv_dtype == 1) return run<true, false>(T, q_proj, k, v, q_norm, k_norm, kv, ctx_max, k_scale, v_scale, pos0, out, ws, stream);
  if (kv_dtype == 2) return run<false, true>(T, q_proj, k, v, q_norm, k_norm, kv, ctx_max, k_scale, v_scale, pos0, out, ws, stream);
  return run<false, false>(T, q_proj, k, v, q_norm, k_norm, kv, ctx_max, k_scale, v_scale, pos0, out, ws, stream);
}

}  // extern "C"
