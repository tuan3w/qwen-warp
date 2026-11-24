#pragma once
#include "dense_attn_common.cuh"

// Quantised KV cache (warpc/kvcache.py holds the reference semantics): K int8 and V int8 or int4, symmetric grid, one fp16 scale per (token, KV head)
// and tensor (amax / 127 or amax / 7), the head dim rotated by an orthonormal Walsh-Hadamard transform before quantising (the query is rotated the same way, the
// attention output un-rotated once in the reduce kernel), optionally the first `sink` positions and the newest `win` positions of each query row read from
// exact fp16 rows (a sink buffer and a ring of at least win + T slots, slot = position % ring). The buffer layout is KVFormat.layout(); 4-bit rows hold 8-element words with
// elements 0-3 in the low nibbles and 4-7 in the high nibbles of the 4 bytes.

// code (0 .. 2^BITS - 1) of x for the symmetric grid: level = code - OFF, scale s = fp16(amax / QMAX); the same operations as the reference quantiser
template <int BITS>
__device__ __forceinline__ int q_encode(float x, float s) {
  constexpr int QMAX = BITS == 8 ? 127 : 7, OFF = BITS == 8 ? 128 : 8;
  const float u = s > 0.f ? __fdiv_rn(x, s) : 0.f;
  return (int)fminf(fmaxf(rintf(u), (float)-QMAX), (float)QMAX) + OFF;
}

// quantise x (thread i of the HD-thread block holds element i) with one scale for the whole head vector and store codes + scale
template <int HD, int BITS>
__device__ __forceinline__ void q_store(float x, uint8_t* crow, __half* srow, float* sm) {
  constexpr int QMAX = BITS == 8 ? 127 : 7;
  const int i = threadIdx.x;
  const float amax = block_max(fabsf(x), sm);
  const float s = __half2float(__float2half_rn(__fdiv_rn(amax, (float)QMAX)));
  const int c = q_encode<BITS>(x, s);
  if (BITS == 8) {
    crow[i] = (uint8_t)c;
  } else {
    const int other = __shfl_down_sync(0xffffffffu, c, 4);
    if ((i & 7) < 4) crow[(i >> 3) * 4 + (i & 7)] = (uint8_t)(c | (other << 4));
  }
  if (i == 0) *srow = __float2half_rn(__fdiv_rn(amax, (float)QMAX));
}

// qk-norm + rope (+ rotation) of q, k and the quantised write of k and v at position pos0 + t; EX: also the exact fp16 sink / ring rows
template <int HD, int VB, int EX>
__global__ void k_attn_prep_q(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* qout, uint8_t* kvb, QKV L, const int* pos0, int Hq, int Hkv,
                              int rd, float theta, float eps, int ctx, int qk_norm) {
  __shared__ float xs[256];
  __shared__ float sm[32];
  const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
  const int p = *pos0 + t;
  const bf16 *src, *nw = nullptr;
  bool rope = true;
  int kvh = 0;
  if (h < Hq) {
    src = q + ((size_t)t * Hq + h) * HD;
    nw = qn;
  } else if (h < Hq + Hkv) {
    kvh = h - Hq;
    src = k + ((size_t)t * Hkv + kvh) * HD;
    nw = kn;
  } else {
    kvh = h - Hq - Hkv;
    src = v + ((size_t)t * Hkv + kvh) * HD;
    rope = false;
  }
  float x = bf(src[i]);
  if (rope) {
    if (qk_norm) {
      const float ss = block_sum(x * x, sm);
      const float r = rsqrtf(ss / (float)HD + eps);
      x = rb(x * r * bf(nw[i]));
    }
    xs[i] = x;
    __syncthreads();
    float o = x;
    if (i < rd) {
      const int half = rd / 2;
      const int j = i < half ? i : i - half;
      const float inv = exp2f(-(2.f * (float)j / (float)rd) * log2f(theta));
      float s, c;
      sincosf((float)p * inv, &s, &c);
      const float partner = i < half ? xs[i + half] : xs[i - half];
      o = i < half ? x * c - partner * s : x * c + partner * s;
    }
    x = rb(o);
    __syncthreads();
  }
  x = wht_blk<HD>(x, xs);
  if (h < Hq) {
    qout[((size_t)t * Hq + h) * HD + i] = tb(x);
    return;
  }
  const bool isk = h < Hq + Hkv;
  const size_t row = (size_t)kvh * ctx + p;
  if (isk) q_store<HD, 8>(x, kvb + L.kc + row * HD, reinterpret_cast<__half*>(kvb + L.ks) + row, sm);
  else q_store<HD, VB>(x, kvb + L.vc + row * (HD * VB / 8), reinterpret_cast<__half*>(kvb + L.vs) + row, sm);
  if (EX) {
    const __half xh = __float2half_rn(x);
    if (p < L.sink) reinterpret_cast<__half*>(kvb + (isk ? L.kx : L.vx))[((size_t)kvh * L.sink + p) * HD + i] = xh;
    if (L.win) reinterpret_cast<__half*>(kvb + (isk ? L.kr : L.vr))[((size_t)kvh * L.ring + p % L.ring) * HD + i] = xh;
  }
}

// float of code byte `E` of word w for the symmetric grid (exact: 2^23 + code minus 2^23 + OFF)
template <int E, int OFF>
__device__ __forceinline__ float q_lv(uint32_t w) {
  return __int_as_float(__byte_perm(w, 0x4B000000u, 0x7540 | E)) - (8388608.f + (float)OFF);
}

// Split-K flash decode over a quantised cache: block (kv head, split, row t), 128 threads, `nc` chunks of 64 keys per block (online softmax across chunks).
//   scores : two threads per key (each half of the rotated head dim): 16-byte code loads, sum_e q_e * level_e in registers, one multiply by the key's scale
//   softmax: one warp per query head
//   P V    : a thread owns 8 consecutive dims of the head and every KS-th key of the chunk (4 B (int4) / 8 B (int8) code load per key), acc += (p * scale) * level;
//            the partial sums of the key subsets are added at the end (shuffle, then the four warps through shared memory)
template <int HD, int VB, int EX, int GM>  // GM: register array size for the query heads per KV head (G <= GM)
__global__ void __launch_bounds__(128) k_attn_split_q(const bf16* q, const uint8_t* kvb, QKV L, const int* pos0, float* part, int Hq, int Hkv, int ctx, int nsplit, int nc,
                                                       float scale) {
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DS = HD / 8, KS = 128 / DS, NK = CH / KS;  // dim slots, key subsets, keys per thread and chunk
  constexpr int VROW = HD * VB / 8;
  float* qs = smem;              // [G][HD]
  float* sc = qs + G * HD;       // [G][CH]
  float* po = sc + G * CH;       // [4][G][HD]
  float* ml = po + 4 * G * HD;   // [3][8]
  float* vsc = ml + 24;          // [CH] V scales of the chunk
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int p = *pos0 + t, live = p + 1;
  const int start0 = split * nc * CH;
  if (start0 >= live) return;
  for (int g = 0; g < G; g++)
    for (int d = tid; d < HD; d += 128) qs[g * HD + d] = bf(q[((size_t)t * Hq + kvh * G + g) * HD + d]);
  if (tid < 8) {
    ml[tid] = -INFINITY;
    ml[8 + tid] = 0.f;
  }
  const int ds = tid % DS, ksub = tid / DS;
  float o[GM][8];
#pragma unroll
  for (int g = 0; g < GM; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[g][e] = 0.f;
  const uint8_t* kc = kvb + L.kc + (size_t)kvh * ctx * HD;
  const uint8_t* vc = kvb + L.vc + (size_t)kvh * ctx * VROW;
  const __half* ksc = reinterpret_cast<const __half*>(kvb + L.ks) + (size_t)kvh * ctx;
  const __half* vss = reinterpret_cast<const __half*>(kvb + L.vs) + (size_t)kvh * ctx;
  // Loads of one chunk, issued before they are needed: this thread's half row of its key (K codes + scale) and its keys' V code words (none for exact keys).
  constexpr int CWN = VB == 8 ? 2 : 1;
  const int kj = tid >> 1, khalf = tid & 1;
  uint4 raw[HD / 32], rawn[HD / 32];
  uint32_t cw[NK][CWN], cwn[NK][CWN];
  float ks = 0.f, ksn = 0.f;
  auto load = [&](int start, int n, uint4 (&r)[HD / 32], float& kscale, uint32_t (&w)[NK][CWN]) {
    const int kp = start + kj;
    if (kj < n && !(EX && (kp < L.sink || p - kp < L.win))) {
      const uint4* kr = reinterpret_cast<const uint4*>(kc + (size_t)kp * HD) + khalf * (HD / 32);
#pragma unroll
      for (int cc = 0; cc < HD / 32; cc++) r[cc] = __ldg(kr + cc);
      kscale = __half2float(ksc[kp]);
    }
#pragma unroll
    for (int jj = 0; jj < NK; jj++) {
      const int j = ksub + jj * KS, kq = start + j;
      w[jj][0] = 0;
      if (VB == 8) w[jj][1] = 0;
      if (j < n && !(EX && (kq < L.sink || p - kq < L.win))) {
        const uint32_t* vp = reinterpret_cast<const uint32_t*>(vc + (size_t)kq * VROW) + ds * CWN;
        w[jj][0] = __ldg(vp);
        if (VB == 8) w[jj][1] = __ldg(vp + 1);
      }
    }
  };
  load(start0, min(CH, live - start0), raw, ks, cw);
  for (int c = 0; c < nc; c++) {
    const int start = start0 + c * CH;
    if (start >= live) break;
    const int n = min(CH, live - start);
    __syncthreads();
    if (tid < CH) vsc[tid] = tid < n ? __half2float(vss[start + tid]) : 0.f;
    {
      const int kp = start + kj;
      float dot[GM];
#pragma unroll
      for (int g = 0; g < GM; g++) dot[g] = 0.f;
      if (kj < n) {
        const bool ex = EX && (kp < L.sink || p - kp < L.win);
        if (!ex) {
#pragma unroll
          for (int cc = 0; cc < HD / 32; cc++) {
            const uint32_t w4[4] = {raw[cc].x, raw[cc].y, raw[cc].z, raw[cc].w};
#pragma unroll
            for (int wi = 0; wi < 4; wi++) {
              const float lv[4] = {q_lv<0, 128>(w4[wi]), q_lv<1, 128>(w4[wi]), q_lv<2, 128>(w4[wi]), q_lv<3, 128>(w4[wi])};
              const int e0 = khalf * (HD / 2) + cc * 16 + wi * 4;
#pragma unroll
              for (int g = 0; g < GM; g++)
                if (g < G) {
#pragma unroll
                  for (int e = 0; e < 4; e++) dot[g] = fmaf(lv[e], qs[g * HD + e0 + e], dot[g]);
                }
            }
          }
#pragma unroll
          for (int g = 0; g < GM; g++) dot[g] *= ks;
        } else {  // exact row (fp16, rotated basis): the sink buffer for the first positions, else the ring
          const __half* row = kp < L.sink ? reinterpret_cast<const __half*>(kvb + L.kx) + ((size_t)kvh * L.sink + kp) * HD
                                          : reinterpret_cast<const __half*>(kvb + L.kr) + ((size_t)kvh * L.ring + kp % L.ring) * HD;
          const uint4* kr = reinterpret_cast<const uint4*>(row + khalf * (HD / 2));
#pragma unroll
          for (int cc = 0; cc < HD / 16; cc++) {
            const uint4 u = kr[cc];
            const __half2* hp = reinterpret_cast<const __half2*>(&u);
            float f[8];
#pragma unroll
            for (int e = 0; e < 4; e++) {
              const float2 tt = __half22float2(hp[e]);
              f[2 * e] = tt.x;
              f[2 * e + 1] = tt.y;
            }
#pragma unroll
            for (int g = 0; g < GM; g++)
              if (g < G) {
#pragma unroll
                for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + khalf * (HD / 2) + cc * 8 + e], dot[g]);
              }
          }
        }
      }
#pragma unroll
      for (int g = 0; g < GM; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
      if (khalf == 0)
        for (int g = 0; g < G; g++) sc[g * CH + kj] = kj < n ? dot[g] * scale : -INFINITY;
    }
    // the next chunk's loads go out now and overlap the softmax and the P V phase of this one
    const int startn = start + CH;
    const bool more = c + 1 < nc && startn < live;
    if (more) load(startn, min(CH, live - startn), rawn, ksn, cwn);
    __syncthreads();
    for (int g = tid >> 5; g < G; g += 4) {
      const int lane = tid & 31;
      const float s0 = sc[g * CH + lane], s1 = sc[g * CH + lane + 32];
      const float mold = ml[g];
      const float m = fmaxf(mold, warp_max(fmaxf(s0, s1)));
      const float e0 = s0 == -INFINITY ? 0.f : __expf(s0 - m), e1 = s1 == -INFINITY ? 0.f : __expf(s1 - m);
      const float l = warp_sum(e0 + e1);
      const float alpha = mold == -INFINITY ? 0.f : __expf(mold - m);
      sc[g * CH + lane] = e0;
      sc[g * CH + lane + 32] = e1;
      __syncwarp();  // the lanes' reads of ml[g] (mold above) precede lane 0's writes below: the shuffles are not a memory barrier (racecheck: WAR on ml[g])
      if (lane == 0) {
        ml[g] = m;
        ml[8 + g] = ml[8 + g] * alpha + l;
        ml[16 + g] = alpha;
      }
    }
    __syncthreads();
    if (c > 0) {
#pragma unroll
      for (int g = 0; g < GM; g++)
        if (g < G) {
          const float al = ml[16 + g];
#pragma unroll
          for (int e = 0; e < 8; e++) o[g][e] *= al;
        }
    }
#pragma unroll
    for (int jj = 0; jj < NK; jj++) {
      const int j = ksub + jj * KS, kp = start + j;
      if (j >= n) continue;
      if (EX && (kp < L.sink || p - kp < L.win)) {
        const __half* row = kp < L.sink ? reinterpret_cast<const __half*>(kvb + L.vx) + ((size_t)kvh * L.sink + kp) * HD
                                        : reinterpret_cast<const __half*>(kvb + L.vr) + ((size_t)kvh * L.ring + kp % L.ring) * HD;
        const uint4 u = *reinterpret_cast<const uint4*>(row + ds * 8);
        const __half2* hp = reinterpret_cast<const __half2*>(&u);
        float f[8];
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const float2 tt = __half22float2(hp[e]);
          f[2 * e] = tt.x;
          f[2 * e + 1] = tt.y;
        }
#pragma unroll
        for (int g = 0; g < GM; g++)
          if (g < G) {
            const float pj = sc[g * CH + j];
#pragma unroll
            for (int e = 0; e < 8; e++) o[g][e] = fmaf(pj, f[e], o[g][e]);
          }
      } else {
        float lv[8];
        if (VB == 8) {
          lv[0] = q_lv<0, 128>(cw[jj][0]); lv[1] = q_lv<1, 128>(cw[jj][0]); lv[2] = q_lv<2, 128>(cw[jj][0]); lv[3] = q_lv<3, 128>(cw[jj][0]);
          lv[4] = q_lv<0, 128>(cw[jj][1]); lv[5] = q_lv<1, 128>(cw[jj][1]); lv[6] = q_lv<2, 128>(cw[jj][1]); lv[7] = q_lv<3, 128>(cw[jj][1]);
        } else {
          const uint32_t lo = cw[jj][0] & 0x0F0F0F0Fu, hi = (cw[jj][0] >> 4) & 0x0F0F0F0Fu;
          lv[0] = q_lv<0, 8>(lo); lv[1] = q_lv<1, 8>(lo); lv[2] = q_lv<2, 8>(lo); lv[3] = q_lv<3, 8>(lo);
          lv[4] = q_lv<0, 8>(hi); lv[5] = q_lv<1, 8>(hi); lv[6] = q_lv<2, 8>(hi); lv[7] = q_lv<3, 8>(hi);
        }
        const float s = vsc[j];
#pragma unroll
        for (int g = 0; g < GM; g++)
          if (g < G) {
            const float w = sc[g * CH + j] * s;
#pragma unroll
            for (int e = 0; e < 8; e++) o[g][e] = fmaf(w, lv[e], o[g][e]);
          }
      }
    }
    if (more) {
#pragma unroll
      for (int cc = 0; cc < HD / 32; cc++) raw[cc] = rawn[cc];
      ks = ksn;
#pragma unroll
      for (int jj = 0; jj < NK; jj++) {
        cw[jj][0] = cwn[jj][0];
        if (VB == 8) cw[jj][1] = cwn[jj][1];
      }
    }
  }
  // key subsets: lanes DS apart share a dim slot (shuffle), the four warps are added through shared memory in a fixed order
#pragma unroll
  for (int g = 0; g < GM; g++)
    if (g < G) {
#pragma unroll
      for (int e = 0; e < 8; e++) {
#pragma unroll
        for (int off = DS; off < 32; off <<= 1) o[g][e] += __shfl_xor_sync(0xffffffffu, o[g][e], off);
      }
    }
  __syncthreads();
  if ((tid & 31) < DS) {
    const int w = tid >> 5;
#pragma unroll
    for (int g = 0; g < GM; g++)
      if (g < G) {
#pragma unroll
        for (int e = 0; e < 8; e++) po[((size_t)w * G + g) * HD + ds * 8 + e] = o[g][e];
      }
  }
  __syncthreads();
  for (int g = 0; g < G; g++) {
    float* pp = part + (((size_t)t * Hq + kvh * G + g) * nsplit + split) * (HD + 4);
    for (int d = tid; d < HD; d += 128) pp[d] = ((po[(size_t)0 * G * HD + g * HD + d] + po[(size_t)1 * G * HD + g * HD + d]) + po[(size_t)2 * G * HD + g * HD + d]) + po[(size_t)3 * G * HD + g * HD + d];
    if (tid == 0) {
      pp[HD] = ml[g];
      pp[HD + 1] = ml[8 + g];
    }
  }
}
