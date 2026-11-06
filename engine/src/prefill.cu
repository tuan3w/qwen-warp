// swift-mk K6: fast prompt prefill helpers.  The GEMMs (cuBLAS via torch), the chunked gated delta rule (fla) and the causal attention
// (FlashAttention-2) run from python (prefill.py); everything else that has to match the decode kernels' semantics lives here:
//   * W4 -> bf16 dequant of a row range (scratch for the cuBLAS GEMM),
//   * DeltaNet: causal conv1d(k=4)+SiLU with the engine's 3-token conv state, gate/beta, gated RMSNorm (same maths as src/deltanet.cu),
//   * attention: per-head q/k RMSNorm + partial rope (fp64 angle, as src/attn.cu), KV-cache write (bf16 or fp8 e4m3 with the checkpoint scales)
//     into the engine layout, token-major bf16 K/V for flash-attn, history gather (cache -> bf16), * sigmoid(gate),
//   * SiLU*mul on the interleaved gate|up GEMM output, MTP input concat.
// Layouts: include/engine.h, include/attn.h, include/deltanet.h.  All launches are async on the given stream.
#include <cuda_fp8.h>
#include <math.h>

#include "attn.h"
#include "engine_impl.h"
#include "kvrot.cuh"
#include "mtp.h"

using namespace smk;

namespace {
constexpr int HD = 256, NKV = 4, NHQ = 24;
constexpr int CONVD = 10240, KEYD = 2048, VALD = 6144, NV = 48, DV = 128;

__device__ __constant__ double c_inv_freq[32] = {
    1.0, 0.6042963902381329, 0.3651741272548377, 0.220673406908459, 0.1333521432163324, 0.08058421877614819, 0.04869675251658631,
    0.029427271762092817, 0.01778279410038923, 0.010746078283213174, 0.006493816315762113, 0.003924189758484536, 0.0023713737056616554,
    0.0014330125702369627, 0.0008659643233600654, 0.0005232991146814947, 0.00031622776601683794, 0.00019109529749704405,
    0.00011547819846894582, 6.978305848598664e-05, 4.216965034285822e-05, 2.5482967479793464e-05, 1.539926526059492e-05,
    9.30572040929699e-06, 5.623413251903491e-06, 3.3982083289425593e-06, 2.053525026457146e-06, 1.2409377607517195e-06,
    7.498942093324558e-07, 4.531583637600818e-07, 2.7384196342643614e-07, 1.6548170999431814e-07};

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// ------------------------------------------------------------------------------------------------ W4 -> bf16
// out[r][k] = (nibble - 8) * scale[row0 + r][k / 128] for r in [0, nrows).  One thread = 4 words = 32 columns.
// RP = 1: repacked layout of smk_w4r_gemv (engine.h): word chunk (tile, kg, half, g, t), nibble j / j+4 = columns 2j / 2j+1, u32 scale pair per (tile, kg, g).
template <int RP>
__global__ void __launch_bounds__(256) k_dequant(const uint4* __restrict__ w, const void* __restrict__ sv, uint4* __restrict__ out, int K, int row0, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int cpr = K >> 5;  // 32-column chunks per row
  const long long r = i / cpr;
  const int c = (int)(i - r * cpr);
  const size_t row = (size_t)(row0 + r);
  uint4 wv;
  float sc;
  if (RP) {
    const size_t unit = (row >> 4) * (size_t)(K >> 7) + (c >> 2);
    const int half = (int)(row >> 3) & 1, g = (int)row & 7;
    wv = w[unit * 64 + half * 32 + g * 4 + (c & 3)];
    sc = bf2f((uint16_t)(((const uint32_t*)sv)[unit * 8 + g] >> (16 * half)));
  } else {
    wv = w[row * cpr + c];
    sc = bf2f(((const uint16_t*)sv)[row * (K >> 7) + (c >> 2)]);
  }
  const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
  uint4* dst = out + (size_t)r * (K >> 3) + (size_t)c * 4;
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const int sh = RP ? 4 * ((e >> 1) + 4 * (e & 1)) : 4 * e;
      v[e] = (float)((int)((ww[j] >> sh) & 15u) - 8) * sc;
    }
    dst[j] = make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]), pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
  }
}

// ------------------------------------------------------------------------------------------------ DeltaNet
// depthwise conv (taps over [3 stored | T new inputs]) + SiLU on the q|k|v channels of qkvz[T][ld] -> q [T][2048], k [T][2048], v [T][6144] (bf16).
// prev = conv state before the segment [10240][3] (read only); the last 3 inputs of the segment are written to cs_out [10240][3].
constexpr int CV_TT = 64;
__global__ void __launch_bounds__(128) k_conv(const uint16_t* __restrict__ x, int ld, const uint16_t* __restrict__ prev, const uint16_t* __restrict__ cw,
                                              uint16_t* __restrict__ q, uint16_t* __restrict__ k, uint16_t* __restrict__ v, uint16_t* __restrict__ cs_out, int T) {
  const int c = blockIdx.x * 128 + threadIdx.x;
  const int t0 = blockIdx.y * CV_TT;
  const int t1 = min(T, t0 + CV_TT);
  const uint2 wr = *reinterpret_cast<const uint2*>(cw + (size_t)c * 4);
  const float w0 = bflo(wr.x), w1 = bfhi(wr.x), w2 = bflo(wr.y), w3 = bfhi(wr.y);
  float a0, a1, a2;  // inputs t-3, t-2, t-1
  auto in = [&](int t) -> float {
    if (t >= 0) return bf2f(x[(size_t)t * ld + c]);
    return bf2f(prev[(size_t)c * 3 + 3 + t]);  // t = -3..-1 -> state slots 0..2
  };
  a0 = in(t0 - 3); a1 = in(t0 - 2); a2 = in(t0 - 1);
  uint16_t* dst; size_t stride;
  if (c < KEYD) { dst = q + c; stride = KEYD; }
  else if (c < 2 * KEYD) { dst = k + (c - KEYD); stride = KEYD; }
  else { dst = v + (c - 2 * KEYD); stride = VALD; }
  for (int t = t0; t < t1; t++) {
    const float a3 = bf2f(x[(size_t)t * ld + c]);
    float acc = w0 * a0;
    acc = fmaf(w1, a1, acc);
    acc = fmaf(w2, a2, acc);
    acc = fmaf(w3, a3, acc);
    dst[(size_t)t * stride] = f2bf(silu_f(acc));
    a0 = a1; a1 = a2; a2 = a3;
  }
  if (t1 == T) {  // last tile of this channel: new conv state = the last 3 inputs (a0,a1,a2 after the final shift hold t-2.. = last three)
    cs_out[(size_t)c * 3 + 0] = f2bf(a0);
    cs_out[(size_t)c * 3 + 1] = f2bf(a1);
    cs_out[(size_t)c * 3 + 2] = f2bf(a2);
  }
}

// g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b), fp32 [T][48]; ab = bf16 [T][96] (a: cols 0..47, b: 48..95)
__global__ void k_gates(const uint16_t* __restrict__ ab, const uint16_t* __restrict__ A_log, const uint16_t* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta, int T) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * NV) return;
  const int t = i / NV, h = i - t * NV;
  const float av = bf2f(ab[(size_t)t * 96 + h]) + bf2f(dt_bias[h]);
  const float sp = av > 20.0f ? av : log1pf(expf(av));
  g[i] = -expf(bf2f(A_log[h])) * sp;
  beta[i] = 1.0f / (1.0f + expf(-bf2f(ab[(size_t)t * 96 + NV + h])));
}

// y[t][h*128+d] = bf16( (w[d] * o*rsqrt(mean(o^2)+1e-6)) * silu(z) ), one warp per (t, head); o bf16 [T][48][128], z = zbase[t*ldz + h*128 + d]
__global__ void __launch_bounds__(128) k_gnorm(const uint16_t* __restrict__ o, const uint16_t* __restrict__ z, int ldz, const uint16_t* __restrict__ nw, uint16_t* __restrict__ y, int T) {
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (gw >= T * NV) return;
  const int t = gw / NV, h = gw - t * NV;
  const uint2 ov = *reinterpret_cast<const uint2*>(o + (size_t)gw * DV + lane * 4);
  const uint2 zv = *reinterpret_cast<const uint2*>(z + (size_t)t * ldz + h * DV + lane * 4);
  const uint2 wv = *reinterpret_cast<const uint2*>(nw + lane * 4);
  const float x[4] = {bflo(ov.x), bfhi(ov.x), bflo(ov.y), bfhi(ov.y)};
  const float zz[4] = {bflo(zv.x), bfhi(zv.x), bflo(zv.y), bfhi(zv.y)};
  const float ww[4] = {bflo(wv.x), bfhi(wv.x), bflo(wv.y), bfhi(wv.y)};
  float ss = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3];
  ss = warp_sum(ss);
  const float rs = rsqrtf(ss * (1.0f / DV) + 1e-6f);
  float r[4];
#pragma unroll
  for (int i = 0; i < 4; i++) r[i] = (ww[i] * (x[i] * rs)) * silu_f(zz[i]);
  *reinterpret_cast<uint2*>(y + (size_t)gw * DV + lane * 4) = make_uint2(pack_bf16x2(r[0], r[1]), pack_bf16x2(r[2], r[3]));
}

// ------------------------------------------------------------------------------------------------ attention
// Warp per (row t, head slot 0..31): slots 0..23 q heads (skipped if q == nullptr), 24..27 k heads, 28..31 v heads.  q rows are the [q|gate] interleaved
// q_proj layout (ldq); k, v rows have strides ldk, ldv.  Row t is absolute position pos0 + t.
//   qout [T][24][256] bf16 (normed + roped q);  k_all/v_all (nullable) bf16 token-major [pos][4][256] at index pos;
//   kv = engine KV cache of this layer ([2][4][ctx_max][256], bf16 or fp8 e4m3 = x / scale) written at pos.
template <bool FP8, bool I4>
__global__ void __launch_bounds__(128) k_attn_prep(int T, int pos0, const uint16_t* __restrict__ q, int ldq, const uint16_t* __restrict__ kk, int ldk,
                                                   const uint16_t* __restrict__ vv, int ldv, const uint16_t* __restrict__ q_norm, const uint16_t* __restrict__ k_norm,
                                                   void* kv, int ctx_max, float k_scale, float v_scale, uint16_t* __restrict__ qout,
                                                   uint16_t* __restrict__ k_all, uint16_t* __restrict__ v_all) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5);
  const int t = gw >> 5, h = gw & 31;
  if (t >= T) return;
  if (h < NHQ && !q) return;
  const int pos = pos0 + t;
  const uint16_t* src;
  const uint16_t* nw = nullptr;
  if (h < NHQ) { src = q + (size_t)t * ldq + h * 2 * HD; nw = q_norm; }
  else if (h < NHQ + NKV) { src = kk + (size_t)t * ldk + (h - NHQ) * HD; nw = k_norm; }
  else { src = vv + (size_t)t * ldv + (h - NHQ - NKV) * HD; }
  float x[8];
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] = bf2f(src[lane + 32 * i]);
  if (nw) {
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) ss += x[i] * x[i];
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss * (1.f / HD) + 1e-6f);
#pragma unroll
    for (int i = 0; i < 8; i++) x[i] = x[i] * inv * (1.f + bf2f(nw[lane + 32 * i]));
  }
  if (h < NHQ + NKV) {  // rope on dims 0..63: x[0] = d lane, x[1] = d lane+32 (rotate_half pair)
    const double ang = (double)pos * c_inv_freq[lane];
    const double kq = rint(ang * 0.15915494309189535);
    const double rr = fma(-kq, 2.4492935982947064e-16, fma(-kq, 6.283185307179586, ang));
    float sn, cs;
    sincosf((float)rr, &sn, &cs);
    const float a = x[0], b = x[1];
    x[0] = a * cs - b * sn;
    x[1] = b * cs + a * sn;
  }
  if (h < NHQ) {
    uint16_t* dst = qout + ((size_t)t * NHQ + h) * HD;
#pragma unroll
    for (int i = 0; i < 8; i++) dst[lane + 32 * i] = f2bf(x[i]);
  } else {
    const int isv = h >= NHQ + NKV;
    const int kvh = isv ? h - NHQ - NKV : h - NHQ;
    uint16_t* all = isv ? v_all : k_all;
    if (all) {
      uint16_t* dst = all + ((size_t)pos * NKV + kvh) * HD;
#pragma unroll
      for (int i = 0; i < 8; i++) dst[lane + 32 * i] = f2bf(x[i]);
    }
    const size_t row = ((size_t)(isv * NKV + kvh) * ctx_max + pos) * HD;
    if (FP8) {
      const float is = 1.f / (isv ? v_scale : k_scale);
      uint8_t* dst = reinterpret_cast<uint8_t*>(kv) + row;
#pragma unroll
      for (int i = 0; i < 8; i++) dst[lane + 32 * i] = (uint8_t)__nv_cvt_float_to_fp8(x[i] * is, __NV_SATFINITE, __NV_E4M3);
    } else if constexpr (I4) {
      // rotated 4-bit format, identical to attn.cu's decode-time writer (kvrot.cuh): the cache rows are Walsh-Hadamard rotated; k_all / v_all above stay unrotated bf16.
      float y[8];
#pragma unroll
      for (int i = 0; i < 8; i++) y[i] = x[i];
      kvrot::wht256(y, lane);
      uint32_t* meta = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(kv) + (size_t)2 * NKV * ctx_max * (HD / 2));
      kvrot::quant_store(y, lane, reinterpret_cast<uint8_t*>(kv) + (row >> 1), meta + (size_t)(isv * NKV + kvh) * ctx_max + pos);
      if (pos + kvrot::RING_R >= pos0 + T)  // exact ring: only the last RING_R rows of this launch (equal slots would race)
        kvrot::sink_store(y, lane, reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(kv) + kvrot::ring_off(ctx_max)) + ((size_t)(isv * NKV + kvh) * kvrot::RING_R + (pos & (kvrot::RING_R - 1))) * HD);
      if (pos < kvrot::NSINK)
        kvrot::sink_store(y, lane, reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(kv) + kvrot::sink_off(ctx_max)) + ((size_t)(isv * NKV + kvh) * kvrot::NSINK + pos) * HD);
    } else {
      uint16_t* dst = reinterpret_cast<uint16_t*>(kv) + row;
#pragma unroll
      for (int i = 0; i < 8; i++) dst[lane + 32 * i] = f2bf(x[i]);
    }
  }
}

// history gather: cache [2][4][ctx_max][256] (bf16 / fp8*scale) positions [0, npos) -> k_all / v_all bf16 [pos][4][256].  One thread = 8 elements.
template <bool FP8>
__global__ void k_gather_kv(const void* __restrict__ kv, int ctx_max, int npos, float k_scale, float v_scale, uint16_t* __restrict__ k_all, uint16_t* __restrict__ v_all) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long n = (long long)npos * NKV * (HD / 8);
  if (i >= n) return;
  const int d8 = (int)(i & 31);
  const long long pk = i >> 5;  // pos * 4 + kvh
  const int kvh = (int)(pk & 3);
  const int pos = (int)(pk >> 2);
#pragma unroll
  for (int isv = 0; isv < 2; isv++) {
    const size_t src = ((size_t)(isv * NKV + kvh) * ctx_max + pos) * HD + d8 * 8;
    uint16_t* dst = (isv ? v_all : k_all) + (size_t)pk * HD + d8 * 8;
    if (FP8) {
      const float sc = isv ? v_scale : k_scale;
      const uint2 raw = *reinterpret_cast<const uint2*>(reinterpret_cast<const uint8_t*>(kv) + src);
      const uint32_t rr[2] = {raw.x, raw.y};
      float f[8];
#pragma unroll
      for (int b = 0; b < 8; b++) {
        const __nv_fp8_storage_t s8 = (__nv_fp8_storage_t)((rr[b >> 2] >> (8 * (b & 3))) & 0xffu);
        f[b] = __half2float(__half(__nv_cvt_fp8_to_halfraw(s8, __NV_E4M3))) * sc;
      }
      *reinterpret_cast<uint4*>(dst) = make_uint4(pack_bf16x2(f[0], f[1]), pack_bf16x2(f[2], f[3]), pack_bf16x2(f[4], f[5]), pack_bf16x2(f[6], f[7]));
    } else {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(kv) + src);
    }
  }
}

// int4 history gather (rotated format, kvrot.cuh): one warp = one (pos, kvh) row pair; dequantise the rotated K and V rows and un-rotate them (H/16 is its own inverse).
__global__ void __launch_bounds__(128) k_gather_kv_i4(const void* __restrict__ kv, int ctx_max, int npos, uint16_t* __restrict__ k_all, uint16_t* __restrict__ v_all) {
  const int lane = threadIdx.x & 31;
  const long long pk = (long long)blockIdx.x * 4 + (threadIdx.x >> 5);  // pos * 4 + kvh
  if (pk >= (long long)npos * NKV) return;
  const int kvh = (int)(pk & 3);
  const int pos = (int)(pk >> 2);
#pragma unroll
  for (int isv = 0; isv < 2; isv++) {
    const size_t rowi = (size_t)(isv * NKV + kvh) * ctx_max + pos;
    const uint8_t* codes = reinterpret_cast<const uint8_t*>(kv) + rowi * (HD / 2);
    const uint32_t m = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(kv) + (size_t)2 * NKV * ctx_max * (HD / 2) + rowi * 4);
    float y[8];
    if (pos < kvrot::NSINK) {  // exact fp16 sink row (rotated basis)
      const uint16_t* sr = reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(kv) + kvrot::sink_off(ctx_max)) + ((size_t)(isv * NKV + kvh) * kvrot::NSINK + pos) * HD;
#pragma unroll
      for (int i = 0; i < 8; i++) y[i] = __half2float(__ushort_as_half(sr[lane + 32 * i]));
    } else {
      kvrot::dequant_row(codes, m, lane, isv ? kvrot::LOG2_VGAIN : kvrot::LOG2_KGAIN, y);
    }
    kvrot::wht256(y, lane);
    uint16_t* dst = (isv ? v_all : k_all) + (size_t)pk * HD;
#pragma unroll
    for (int i = 0; i < 8; i++) dst[lane + 32 * i] = f2bf(y[i]);
  }
}

// y[t][h*256+d] = bf16( ao[t][h][d] * sigmoid(gate[t][h][d]) ), gate = qproj[t*ldq + h*512 + 256 + d].  One thread = 8 elements.
__global__ void k_attn_gate(const uint16_t* __restrict__ ao, const uint16_t* __restrict__ qproj, int ldq, uint16_t* __restrict__ y, int T) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long long)T * NHQ * 32) return;
  const int d8 = (int)(i & 31);
  const long long th = i >> 5;  // t*24 + h
  const int h = (int)(th % NHQ);
  const long long t = th / NHQ;
  const uint4 a = *reinterpret_cast<const uint4*>(ao + (size_t)th * HD + d8 * 8);
  const uint4 g = *reinterpret_cast<const uint4*>(qproj + (size_t)t * ldq + h * 512 + HD + d8 * 8);
  const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, gw[4] = {g.x, g.y, g.z, g.w};
  uint32_t o[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float g0 = bflo(gw[j]), g1 = bfhi(gw[j]);
    o[j] = pack_bf16x2(bflo(aw[j]) * (1.0f / (1.0f + expf(-g0))), bfhi(aw[j]) * (1.0f / (1.0f + expf(-g1))));
  }
  *reinterpret_cast<uint4*>(y + (size_t)t * (NHQ * HD) + h * HD + d8 * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

// ------------------------------------------------------------------------------------------------ MLP / MTP
// gu [R][2I] interleaved per 8 rows (gate 8i..8i+7 at cols 16i.., up at 16i+8..) -> act [R][I] = bf16(bf16(silu(gate)) * up).  One thread = 8 channels.
__global__ void k_silu_mul(const uint4* __restrict__ gu, uint4* __restrict__ act, long long total) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const uint4 g = gu[2 * i], u = gu[2 * i + 1];
  const uint32_t gw[4] = {g.x, g.y, g.z, g.w}, uw[4] = {u.x, u.y, u.z, u.w};
  uint32_t o[4];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    const float s0 = bf16r(silu_f(bflo(gw[j]))), s1 = bf16r(silu_f(bfhi(gw[j])));
    o[j] = pack_bf16x2(s0 * bflo(uw[j]), s1 * bfhi(uw[j]));
  }
  act[i] = make_uint4(o[0], o[1], o[2], o[3]);
}

constexpr int NH8 = H / 8;  // 640
// cat[t] = [ rmsnorm0(embed[tok[t]]; wE) | rmsnorm0(hid[t]; wH) ]   grid (T, 2), NH8 threads  (same as mtp.cu k_mtp_prep)
__global__ void __launch_bounds__(NH8) k_mtp_cat(const uint4* __restrict__ embed, const int* __restrict__ tok, const uint4* __restrict__ hid, const uint4* __restrict__ wE,
                                                 const uint4* __restrict__ wH, uint4* __restrict__ cat) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, half = blockIdx.y, c = threadIdx.x;
  const uint4* src = half == 0 ? embed + (size_t)tok[t] * NH8 : hid + (size_t)t * NH8;
  const uint4* w = half == 0 ? wE : wH;
  const uint4 hv = src[c];
  float v[8] = {bflo(hv.x), bfhi(hv.x), bflo(hv.y), bfhi(hv.y), bflo(hv.z), bfhi(hv.z), bflo(hv.w), bfhi(hv.w)};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH8 / 32; i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  cat[(size_t)t * 2 * NH8 + half * NH8 + c] =
      make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                 pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
}

inline unsigned cdiv(long long a, long long b) { return (unsigned)((a + b - 1) / b); }

// K10: the last prompt row ran inside a batched prefill and the engine head tail (smk_engine_head_tail) produced ds[DS_OUT]: make the device scalars what the
// T=1 head step leaves behind: pos = `pos`, cur advanced by one ring slot (that row's DeltaNet post-state is in slot cur+1), next input token = the first generated token.
__global__ void k_head_finish(int* ds, int pos) {
  ds[DS_POS] = pos;
  ds[DS_CUR] = (ds[DS_CUR] + 1) % ds[DS_SLOTS];
  ds[DS_PEND] = 0;  // the T=1 row wrote a full snapshot
  ds[DS_TOK] = ds[DS_OUT];
}
}  // namespace

extern "C" {

// ---- accessors into the opaque engine (what: 0 device state ints, 1 DeltaNet state fp32 [48 layers][sslots][48][128][128] (smk_engine_sslots), 2 conv state bf16 [48][tmax+1][10240][3],
//      3 KV cache base (16 layers of kv_layer_bytes), 4 = kv_layer_bytes (returned as a pointer-sized integer))
// Exact sink + ring buffers of the int4 cache (kvrot.cuh) live behind the code/meta region of every layer; they are state that a DeltaNet snapshot must carry along (prefix reuse).
// Bytes of all 16 layers' buffers (0 unless the cache is int4).
size_t swiftmk_kv_aux_bytes(smk_engine* e) {
  if (e->kv_dtype != 2) return 0;
  return (size_t)16 * (e->kv_layer_bytes - kvrot::sink_off(e->ctx_max));
}
// dir 0: device -> host (synchronises the stream before returning), dir 1: host (pinned) -> device, stream ordered.
cudaError_t swiftmk_kv_aux_copy(smk_engine* e, void* host, int dir, cudaStream_t st) {
  if (e->kv_dtype != 2) return cudaSuccess;
  const size_t per = e->kv_layer_bytes - kvrot::sink_off(e->ctx_max);
  for (int l = 0; l < 16; l++) {
    char* dev = e->kv + (size_t)l * e->kv_layer_bytes + kvrot::sink_off(e->ctx_max);
    char* h = (char*)host + (size_t)l * per;
    cudaError_t r = dir == 0 ? cudaMemcpyAsync(h, dev, per, cudaMemcpyDeviceToHost, st) : cudaMemcpyAsync(dev, h, per, cudaMemcpyHostToDevice, st);
    if (r != cudaSuccess) return r;
  }
  return dir == 0 ? cudaStreamSynchronize(st) : cudaSuccess;
}

// Host prefix cache (attn.h "chunk" format): KV rows [a, a + m) of all 16 attention layers, layer after layer.  Bytes of the packed host buffer of one m-row chunk.
size_t swiftmk_kv_range_bytes(smk_engine* e, int m) { return (size_t)16 * swiftmk_attn_kv_range_bytes(e->kv_dtype, m); }
// dir 0: device -> host, 1: host (pinned) -> device; copies rows [r0, r1) of the chunk to / from device rows dev_row0 + r0 ..; stream ordered, no synchronisation (the caller syncs).
cudaError_t swiftmk_kv_range_copy(smk_engine* e, void* host, int m, int dev_row0, int r0, int r1, int dir, cudaStream_t st) {
  const size_t per = swiftmk_attn_kv_range_bytes(e->kv_dtype, m);
  for (int l = 0; l < 16; l++) {
    cudaError_t r = swiftmk_attn_kv_range_copy(e->kv + (size_t)l * e->kv_layer_bytes, e->ctx_max, e->kv_dtype, (char*)host + (size_t)l * per, m, dev_row0, r0, r1, dir, st);
    if (r != cudaSuccess) return r;
  }
  return cudaSuccess;
}

void* smkp_engine_ptr(smk_engine* e, int what) {
  switch (what) {
    case 0: return e->ds;
    case 1: return e->dn_state;
    case 2: return e->dn_conv;
    case 3: return e->kv;
    case 4: return (void*)e->kv_layer_bytes;
    case 5: return e->h;   // bf16 [TMAX][5120] residual stream
    case 6: return e->dn;  // bf16 [TMAX][5120] last layer's down_proj output
    case 7: return e->dbg; // bf16 [66][TMAX][5120] debug taps
    default: return nullptr;
  }
}

// K10: one DeltaNet row through the decode kernel (reads ring slot cur, writes slot cur+1) for DeltaNet layer `di`: qkv = [10240] row followed by z [6144]
// (a fused [qkv|z] GEMM output row), ab = [a 48 | b 48] row.  y = bf16 [6144].
cudaError_t smkp_dn_row(smk_engine* e, int di, const void* qkvz_row, const void* ab_row, const void* conv_w, const void* A_log, const void* dt_bias, const void* norm_w, void* y,
                        cudaStream_t st) {
  return swiftmk_deltanet_step_lazy2(1, qkvz_row, (const char*)qkvz_row + CONVD * 2, ab_row, (const char*)ab_row + NV * 2, conv_w, A_log, dt_bias, norm_w,
                                     e->dn_state + (size_t)di * e->sslots * SWIFTMK_DN_STATE_SLOT_FLOATS,
                                     e->dn_conv + (size_t)di * e->slots * SWIFTMK_DN_CONV_SLOT_ELEMS, e->ds + DS_CUR, e->slots, e->sslots, y, e->ws_dn,
                                     e->dn_dbuf + (size_t)di * swiftmk_deltanet_dbuf_floats(), e->ds + DS_PEND, nullptr, st);
}

// Write the pending lazy DeltaNet state of every layer into ring slot cur (the python prefill reads / writes dn_state[:, cur % sslots] directly) and clear the pending count.
cudaError_t smkp_materialize(smk_engine* e, cudaStream_t st) {
  return swiftmk_deltanet_materialize(e->dn_state, e->dn_dbuf, 48, e->ds + DS_CUR, e->ds + DS_PEND, e->sslots, st);
}

cudaError_t smkp_head_finish(smk_engine* e, int pos, cudaStream_t st) {
  k_head_finish<<<1, 1, 0, st>>>(e->ds, pos);
  return cudaGetLastError();
}

// MTP layer KV cache (bf16 [2][4][ctx_max][256]) of a spec object: defined in mtp.cu.
void* smk_spec_kv_ptr(smk_spec* s);

cudaError_t smkp_dequant_w4(const void* w, const void* scale, void* out, int K, int row0, int nrows, cudaStream_t st) {
  if (K % 128 || nrows < 1) return cudaErrorInvalidValue;
  const long long total = (long long)nrows * (K >> 5);
  k_dequant<0><<<cdiv(total, 256), 256, 0, st>>>((const uint4*)w, scale, (uint4*)out, K, row0, total);
  return cudaGetLastError();
}

// same on the repacked layout (smk_w4r_gemv, include/engine.h)
cudaError_t smkp_dequant_w4r(const void* w, const void* scale, void* out, int K, int row0, int nrows, cudaStream_t st) {
  if (K % 128 || nrows < 1) return cudaErrorInvalidValue;
  const long long total = (long long)nrows * (K >> 5);
  k_dequant<1><<<cdiv(total, 256), 256, 0, st>>>((const uint4*)w, scale, (uint4*)out, K, row0, total);
  return cudaGetLastError();
}

cudaError_t smkp_conv(const void* x, int ld, const void* prev, const void* conv_w, void* q, void* k, void* v, void* cs_out, int T, cudaStream_t st) {
  if (T < 1) return cudaErrorInvalidValue;
  k_conv<<<dim3(CONVD / 128, cdiv(T, CV_TT)), 128, 0, st>>>((const uint16_t*)x, ld, (const uint16_t*)prev, (const uint16_t*)conv_w, (uint16_t*)q, (uint16_t*)k, (uint16_t*)v,
                                                          (uint16_t*)cs_out, T);
  return cudaGetLastError();
}

cudaError_t smkp_gates(const void* ab, const void* A_log, const void* dt_bias, void* g, void* beta, int T, cudaStream_t st) {
  k_gates<<<cdiv((long long)T * NV, 256), 256, 0, st>>>((const uint16_t*)ab, (const uint16_t*)A_log, (const uint16_t*)dt_bias, (float*)g, (float*)beta, T);
  return cudaGetLastError();
}

cudaError_t smkp_gnorm(const void* o, const void* z, int ldz, const void* norm_w, void* y, int T, cudaStream_t st) {
  k_gnorm<<<cdiv((long long)T * NV, 4), 128, 0, st>>>((const uint16_t*)o, (const uint16_t*)z, ldz, (const uint16_t*)norm_w, (uint16_t*)y, T);
  return cudaGetLastError();
}

// q may be NULL (MTP K/V only).  k_all / v_all may be NULL.
cudaError_t smkp_attn_prep(int T, int pos0, const void* q, int ldq, const void* k, int ldk, const void* v, int ldv, const void* q_norm, const void* k_norm, void* kv,
                           int ctx_max, int kv_dtype, float k_scale, float v_scale, void* qout, void* k_all, void* v_all, cudaStream_t st) {  // kv_dtype 0 bf16, 1 fp8, 2 int4
  if (T < 1) return cudaErrorInvalidValue;
  const unsigned grid = cdiv((long long)T * 32, 4);
#define SMKP_PREP_ARGS (T, pos0, (const uint16_t*)q, ldq, (const uint16_t*)k, ldk, (const uint16_t*)v, ldv, (const uint16_t*)q_norm, (const uint16_t*)k_norm, kv, \
                        ctx_max, k_scale, v_scale, (uint16_t*)qout, (uint16_t*)k_all, (uint16_t*)v_all)
  if (kv_dtype == 1) k_attn_prep<true, false><<<grid, 128, 0, st>>> SMKP_PREP_ARGS;
  else if (kv_dtype == 2) k_attn_prep<false, true><<<grid, 128, 0, st>>> SMKP_PREP_ARGS;
  else k_attn_prep<false, false><<<grid, 128, 0, st>>> SMKP_PREP_ARGS;
#undef SMKP_PREP_ARGS
  return cudaGetLastError();
}

cudaError_t smkp_gather_kv(const void* kv, int ctx_max, int kv_dtype, int npos, float k_scale, float v_scale, void* k_all, void* v_all, cudaStream_t st) {
  if (npos < 1) return cudaSuccess;
  const unsigned grid = cdiv((long long)npos * NKV * 32, 256);
  if (kv_dtype == 2) {
    k_gather_kv_i4<<<cdiv((long long)npos * NKV, 4), 128, 0, st>>>(kv, ctx_max, npos, (uint16_t*)k_all, (uint16_t*)v_all);
    return cudaGetLastError();
  }
  if (kv_dtype == 1) k_gather_kv<true><<<grid, 256, 0, st>>>(kv, ctx_max, npos, k_scale, v_scale, (uint16_t*)k_all, (uint16_t*)v_all);
  else k_gather_kv<false><<<grid, 256, 0, st>>>(kv, ctx_max, npos, k_scale, v_scale, (uint16_t*)k_all, (uint16_t*)v_all);
  return cudaGetLastError();
}

cudaError_t smkp_attn_gate(const void* ao, const void* qproj, int ldq, void* y, int T, cudaStream_t st) {
  k_attn_gate<<<cdiv((long long)T * NHQ * 32, 256), 256, 0, st>>>((const uint16_t*)ao, (const uint16_t*)qproj, ldq, (uint16_t*)y, T);
  return cudaGetLastError();
}

cudaError_t smkp_silu_mul(const void* gu, void* act, int R, cudaStream_t st) {
  const long long total = (long long)R * (SMK_INTER / 8);
  k_silu_mul<<<cdiv(total, 256), 256, 0, st>>>((const uint4*)gu, (uint4*)act, total);
  return cudaGetLastError();
}

cudaError_t smkp_mtp_cat(const void* embed, const int* tok, const void* hid, const void* pre_e, const void* pre_h, void* cat, int T, cudaStream_t st) {
  k_mtp_cat<<<dim3(T, 2), NH8, 0, st>>>((const uint4*)embed, tok, (const uint4*)hid, (const uint4*)pre_e, (const uint4*)pre_h, (uint4*)cat);
  return cudaGetLastError();
}

}  // extern "C"
