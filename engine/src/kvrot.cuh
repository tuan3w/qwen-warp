// Rotated int4 KV format shared by the decode writer (attn.cu attn_prep_kernel), the prefill writer (prefill.cu k_attn_prep) and the prefill history gather.
//
// Per token and KV head: the 256-vector x (K after rope) is rotated by the orthonormal Walsh-Hadamard transform  y = H x / 16  (H[d][j] = (-1)^popcount(d&j), natural
// order; H/16 is symmetric and its own inverse), then y is quantised with a symmetric 16-level companded code  y ~ s_G * L(n), n = 0..15, L(n) = u (1 + B u^2), u = 2n - 15 (odd integers -15..15), one scale per group G of 64
// rotated coefficients.  Groups (shared by K and V, chosen so a group is constant over one mma step of the reader):
//     G(d) = 2 * (d >> 7) + ((d & 15) >> 3)                       d = rotated coefficient index 0..255
// The 4 scales are 8-bit logarithmic codes e_G packed in the 4-byte meta word (byte G):  s_G = 2^((e_G - 176) / 12).  Codes: 128 B, 2 per byte, natural d order
// (low nibble = even d).  Scores are unchanged by the rotation (q is rotated the same way at prep) and the attention output is un-rotated in the reduce kernel.
//
// Lane layout of the warp writers: lane owns d = lane + 32 i, i = 0..7 (x[i]); G(d) = 2 * (i >> 2) + ((lane >> 3) & 1).
#pragma once
#include <stdint.h>
#include <cuda_fp16.h>

namespace kvrot {

constexpr int E_BIAS = 176;
constexpr float E_RES = 12.f;  // scale codes per octave
constexpr int NSINK = 4;  // positions 0..NSINK-1 keep exact (fp16, rotated basis) K and V rows in a fixed per-layer buffer [2][4 kv heads][NSINK][256] after the meta words (fixed bytes, not per token)
__host__ __device__ __forceinline__ size_t sink_off(int ctx_max) { return (size_t)2 * 4 * (size_t)ctx_max * (128 + 4); }
// Exact recent-token ring: the last RING_W positions (before the current step's queries) are attended from fp16 rotated rows in a fixed per-layer ring [2][4][RING_R][256]
// (slot = pos & (RING_R - 1), RING_R >= 2 RING_W covers the up-to-8 rows of a step), after the sink rows.  The split kernels stop RING_W keys before the end of the context;
// the reduce kernel attends the ring exactly and merges it as one more split.  Fixed bytes per layer, not per token.
constexpr int RING_W = 256;
constexpr int RING_R = 512;
__host__ __device__ __forceinline__ size_t ring_off(int ctx_max) { return sink_off(ctx_max) + (size_t)2 * 4 * NSINK * 256 * 2; }
constexpr size_t RING_BYTES = (size_t)2 * 4 * RING_R * 256 * 2;

// exact sink row store (rotated basis, fp16), warp-wide: sink points at the [NSINK][256] block of (K|V, kv head); only pos < NSINK writes
__device__ __forceinline__ void sink_store(const float (&y)[8], int lane, uint16_t* __restrict__ sink_row) {
#pragma unroll
  for (int i = 0; i < 8; i++) sink_row[lane + 32 * i] = __half_as_ushort(__float2half_rn(y[i]));
}
constexpr float CUBE_B = 0.0030002594f;  // fp16 value of 0.003 (the reader computes L in fp16x2: u*u, fma(B, u*u, 1), * u; 0x1A25 / 0x3C00)

// codebook level of code n (fp32 formula; the reader's fp16 evaluation agrees to ~2^-11 relative)
__host__ __device__ __forceinline__ float level(int n) {
  const float u = (float)(2 * n - 15);
  return u * fmaf(CUBE_B, u * u, 1.f);
}

__device__ __forceinline__ float ex2f(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float scale_of(uint32_t e) { return ex2f((float)((int)e - E_BIAS) * (1.f / E_RES)); }

// in place y = H x / 16 over the 256 values held by a warp (x[i] = element lane + 32 i).  All 32 lanes must call.
__device__ __forceinline__ void wht256(float (&x)[8], int lane) {
#pragma unroll
  for (int h = 1; h < 8; h <<= 1) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if (!(i & h)) {
        const float a = x[i], b = x[i + h];
        x[i] = a + b;
        x[i + h] = a - b;
      }
    }
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const bool up = (lane & o) != 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float y = __shfl_xor_sync(0xffffffffu, x[i], o);
      x[i] = up ? y - x[i] : x[i] + y;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) x[i] *= 0.0625f;
}

// sum over the 16 lanes sharing bit 3 of the lane index (xor 1, 2, 4, 16): identical result on all of them
__device__ __forceinline__ float gsum(float v) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
    if (o != 8) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float gmax(float v) {
#pragma unroll
  for (int o = 1; o < 32; o <<= 1)
    if (o != 8) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

constexpr int NCAND = 8;
__device__ __constant__ int c_cand_off[NCAND] = {1, 0, -1, -2, -3, -4, -6, -8};  // scale-code offsets tried around absmax/L_max (MSE pick = clipping of heavy tails)

// nearest codebook level to x / s (inv = 1 / s): count the decision thresholds (midpoints of adjacent levels, folded at compile time) below it
__device__ __forceinline__ int quant_code(float v, float inv) {
  const float z = v * inv;
  int n = 0;
#pragma unroll
  for (int k = 0; k < 15; k++) n += (z > 0.5f * (level(k) + level(k + 1)));
  return n;
}

// Quantise the rotated row y (warp-wide) and store 128 code bytes at `codes` and the meta word at `meta`.  All 32 lanes must call.
__device__ __forceinline__ void quant_store(const float (&y)[8], int lane, uint8_t* __restrict__ codes, uint32_t* __restrict__ meta) {
  const int h = (lane >> 3) & 1;
  int eb[2];
#pragma unroll
  for (int c = 0; c < 2; c++) {
    float a = fmaxf(fmaxf(fabsf(y[4 * c]), fabsf(y[4 * c + 1])), fmaxf(fabsf(y[4 * c + 2]), fabsf(y[4 * c + 3])));
    a = gmax(a);
    int e0 = __float2int_rn(E_RES * log2f(fmaxf(a, 1e-30f) * (1.f / level(15))) + (float)E_BIAS);
    e0 = min(max(e0, 0), 255);
    float err[NCAND];
    int ec[NCAND];
#pragma unroll
    for (int k = 0; k < NCAND; k++) {
      ec[k] = min(max(e0 + c_cand_off[k], 0), 255);
      const float s = scale_of(ec[k]), inv = 1.f / s;
      float er = 0.f;
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const float r = y[4 * c + q] - s * level(quant_code(y[4 * c + q], inv));
        er = fmaf(r, r, er);
      }
      err[k] = er;
    }
#pragma unroll
    for (int k = 0; k < NCAND; k++) err[k] = gsum(err[k]);
    int best = 0;
    float be = err[0];
#pragma unroll
    for (int k = 1; k < NCAND; k++)
      if (err[k] < be) { be = err[k]; best = k; }
    int e = ec[0];
#pragma unroll
    for (int k = 1; k < NCAND; k++)
      if (best == k) e = ec[k];
    eb[c] = e;
  }
  const float inv0 = 1.f / scale_of(eb[0]), inv1 = 1.f / scale_of(eb[1]);
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const int code = quant_code(y[i], (i >> 2) ? inv1 : inv0);
    const int hi_c = __shfl_xor_sync(0xffffffffu, code, 1);  // all 32 lanes must execute the shuffle
    if (!(lane & 1)) codes[16 * i + (lane >> 1)] = (uint8_t)((hi_c << 4) | code);
  }
  const uint32_t mine = ((uint32_t)eb[0] << (8 * h)) | ((uint32_t)eb[1] << (16 + 8 * h));
  const uint32_t other = __shfl_sync(0xffffffffu, mine, 8);
  if (lane == 0) *meta = mine | other;
}

// dequantised rotated row (warp-wide, element lane + 32 i), from the code byte array and the meta word
__device__ __forceinline__ void dequant_row(const uint8_t* __restrict__ codes, uint32_t meta, int lane, float (&y)[8]) {
  const int h = (lane >> 3) & 1;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint32_t b = codes[16 * i + (lane >> 1)];
    const int n = (lane & 1) ? (int)(b >> 4) : (int)(b & 15u);
    y[i] = scale_of((meta >> (8 * (2 * (i >> 2) + h))) & 255u) * level(n);
  }
}

}  // namespace kvrot
