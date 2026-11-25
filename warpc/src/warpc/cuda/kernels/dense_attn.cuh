#pragma once
#include "dense_attn_common.cuh"

// Flash decode attention. Split over the key axis in chunks of CH = 64 keys so that a short context still spreads over many SMs. Block (kv head, split, row t), 128 threads:
//   scores : two threads per key (each half of the head dim), all G = Hq / Hkv query heads that share the kv head scored against the same K reads
//   softmax: one warp per query head, inside the chunk
//   P V    : a thread owns a pair of output dims and a subset of the keys, 4-byte coalesced loads, several keys in flight
// Partials per (row, head, split) = {o[hd], m, l}; k_attn_reduce merges them with the usual rescaling (a last-block-reduces variant with a
// ticket counter measured 1-2 us slower per layer).
template <int HD>
__global__ void __launch_bounds__(128) k_attn_split(const bf16* q, const bf16* kv, const int* pos0, float* part, int Hq, int Hkv, int ctx, int nsplit, float scale) {
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DP = HD / 2;
  constexpr int KG = 128 / DP > 0 ? 128 / DP : 1;
  float* qs = smem;              // [G][HD]
  float* sc = qs + G * HD;       // [G][CH]
  float* po = sc + G * CH;       // [KG][G][HD]
  float* ml = po + KG * G * HD;  // [2][8]
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int live = *pos0 + t + 1;
  const int start = split * CH;
  if (start >= live) return;
  const int n = min(CH, live - start);
  for (int g = 0; g < G; g++)
    for (int d = tid; d < HD; d += 128) qs[g * HD + d] = bf(q[((size_t)t * Hq + kvh * G + g) * HD + d]);
  __syncthreads();
  const bf16* K = kv + ((size_t)kvh * ctx + start) * HD;
  const bf16* V = kv + (((size_t)Hkv + kvh) * ctx + start) * HD;
  {
    const int j = tid >> 1, half = tid & 1;
    float dot[8];
#pragma unroll
    for (int g = 0; g < 8; g++) dot[g] = 0.f;
    if (j < n) {
      const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)j * HD + half * (HD / 2));
      uint4 kvv[HD / 16];
#pragma unroll
      for (int c = 0; c < HD / 16; c++) kvv[c] = kr[c];
#pragma unroll
      for (int c = 0; c < HD / 16; c++) {
        float f[8];
        unpack8(kvv[c], f);
#pragma unroll
        for (int g = 0; g < 8; g++)
          if (g < G) {
#pragma unroll
            for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + half * (HD / 2) + c * 8 + e], dot[g]);
          }
      }
    }
#pragma unroll
    for (int g = 0; g < 8; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
    if (half == 0)
      for (int g = 0; g < G; g++) sc[g * CH + j] = j < n ? dot[g] * scale : -INFINITY;
  }
  __syncthreads();
  for (int g = tid >> 5; g < G; g += 4) {  // one warp per query head (CH = 64: two scores per lane)
    const int lane = tid & 31;
    const float s0 = sc[g * CH + lane], s1 = sc[g * CH + lane + 32];
    const float m = warp_max(fmaxf(s0, s1));
    const float e0 = s0 == -INFINITY ? 0.f : __expf(s0 - m), e1 = s1 == -INFINITY ? 0.f : __expf(s1 - m);
    const float l = warp_sum(e0 + e1);
    sc[g * CH + lane] = e0;
    sc[g * CH + lane + 32] = e1;
    if (lane == 0) {
      ml[g] = m;
      ml[8 + g] = l;
    }
  }
  __syncthreads();
  const int dp = tid % DP, kg = tid / DP;
  if (tid < DP * KG) {
    float2 o[8];
#pragma unroll
    for (int g = 0; g < 8; g++) o[g] = make_float2(0.f, 0.f);
    const __nv_bfloat162* Vp = reinterpret_cast<const __nv_bfloat162*>(V) + dp;
#pragma unroll 8
    for (int j = kg; j < n; j += KG) {
      const float2 vv = __bfloat1622float2(Vp[(size_t)j * DP]);
#pragma unroll
      for (int g = 0; g < 8; g++)
        if (g < G) {
          const float pj = sc[g * CH + j];
          o[g].x = fmaf(pj, vv.x, o[g].x);
          o[g].y = fmaf(pj, vv.y, o[g].y);
        }
    }
    for (int g = 0; g < G; g++) {
      po[((size_t)kg * G + g) * HD + 2 * dp] = o[g].x;
      po[((size_t)kg * G + g) * HD + 2 * dp + 1] = o[g].y;
    }
  }
  __syncthreads();
  for (int g = 0; g < G; g++) {
    float* pp = part + (((size_t)t * Hq + kvh * G + g) * nsplit + split) * (HD + 2);
    for (int d = tid; d < HD; d += 128) {
      float a2 = 0.f;
#pragma unroll
      for (int c = 0; c < KG; c++) a2 += po[((size_t)c * G + g) * HD + d];
      pp[d] = a2;
    }
    if (tid == 0) {
      pp[HD] = ml[g];
      pp[HD + 1] = ml[8 + g];
    }
  }
}

// Merge of the split partials. HD = 128, ns <= 32 (live <= 2048): every warp derives the merge weights itself (split s on lane s: every warp runs the same shuffle trees, so
// they agree bit for bit), no shared memory and no barrier. The partial rows 0..7 and the (m, l) pairs are loaded before the position (and so the number of splits) is known: rows of
// splits that are not live are read and ignored, they are inside the workspace.
template <int HD>
__global__ void __launch_bounds__(128) k_attn_reduce(const float* part, const int* pos0, bf16* out, int Hq, int nsplit) {
  __shared__ float wgt[MAX_SPLITS];
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int pz = *pos0;
  const float* base = part + ((size_t)t * Hq + head) * nsplit * (HD + 2);
  const int lane = tid & 31;
  float ov[8], ms = -INFINITY, ls = 0.f;
  if (HD == 128) {
#pragma unroll
    for (int s = 0; s < 8; s++)
      if (s < nsplit) ov[s] = base[s * (HD + 2) + tid];
    if (lane < nsplit) {
      ms = base[lane * (HD + 2) + HD];
      ls = base[lane * (HD + 2) + HD + 1];
    }
  }
  const int live = pz + t + 1;
  const int ns = (live + CH - 1) / CH;
  if (HD == 128 && ns <= 32) {
    if (lane >= ns) ms = -INFINITY;
    const float m = warp_max(ms);
    float l = 0.f;
    if (lane < ns) l += ls * __expf(ms - m);
    l = warp_sum(l);
    const float w = __expf(ms - m) / l;
    float a = 0.f;
#pragma unroll
    for (int s = 0; s < 8; s++)
      if (s < ns) a += ov[s] * __shfl_sync(0xffffffffu, w, s);
    for (int s0 = 8; s0 < ns; s0 += 8) {  // longer contexts: the next 8 rows in flight together
      float vv[8];
#pragma unroll
      for (int i = 0; i < 8; i++)
        if (s0 + i < ns) vv[i] = base[(s0 + i) * (HD + 2) + tid];
#pragma unroll
      for (int i = 0; i < 8; i++)
        if (s0 + i < ns) a += vv[i] * __shfl_sync(0xffffffffu, w, s0 + i);
    }
    out[((size_t)t * Hq + head) * HD + tid] = tb(a);
    return;
  }
  if (tid < 32) {  // merge weights exp(m_s - M) / L, split s on lane s (+32, +64, +96)
    float m = -INFINITY;
    for (int s = tid; s < ns; s += 32) m = fmaxf(m, base[s * (HD + 2) + HD]);
    m = warp_max(m);
    float l = 0.f;
    for (int s = tid; s < ns; s += 32) l += base[s * (HD + 2) + HD + 1] * __expf(base[s * (HD + 2) + HD] - m);
    l = warp_sum(l);
    for (int s = tid; s < ns; s += 32) wgt[s] = __expf(base[s * (HD + 2) + HD] - m) / l;
  }
  __syncthreads();
  for (int d = tid; d < HD; d += 128) {
    float a = 0.f;
#pragma unroll 8
    for (int s = 0; s < ns; s++) a += base[s * (HD + 2) + d] * wgt[s];
    out[((size_t)t * Hq + head) * HD + d] = tb(a);
  }
}

// Fused prep + split (T = 1, HD = 128). The split kernel with k_attn_prep folded in: each block recomputes (qk-norm, rope, bf16 rounding: the same arithmetic as k_attn_prep, one warp per head, lane i holds elements
// i, i+32, i+64, i+96) the G query heads of its kv head instead of reading them from a prep kernel; the block that owns the new position additionally computes its K / V row,
// writes it to the cache and uses the rounded row from shared memory instead of reading it back. Loads of K and V (the whole chunk, in registers) are issued before the first
// barrier, so the global-memory latencies of prep, Q read and KV read overlap. Two threads that share a pair of output dims (neighbouring lanes, one per key parity) add their partial
// sums with a shuffle (parity 0 + parity 1, k_attn_split's order).
template <int HD, int GM>
__global__ void __launch_bounds__(128) k_attn_fused(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, float* part, int Hq, int Hkv,
                                                   int ctx, int nsplit, float scale, float theta, float eps, int qk_norm) {
  static_assert(HD == 128 && CH == 64, "fused attention: head dim 128, chunk 64");
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DP = HD / 2;
  float* qs = smem;         // [G][HD] rounded q rows
  float* sc = qs + G * HD;  // [G][CH]
  float* ks = sc + G * CH;  // [HD] the new key row (bf16-rounded), owner block only
  float* vs = ks + HD;      // [HD] the new value row
  float* ml = vs + HD;      // [2][8]
  const int kvh = blockIdx.x, split = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  const int p = *pos0;
  const int ntask = G + 2;  // G query heads, then K, then V (the last two only in the owner block)
  float xr[4], wr[4];
  int u = wp;
  auto load_task = [&](int uu) {
    const bf16* src = uu < G ? q + (size_t)(kvh * G + uu) * HD : (uu == G ? k : v) + (size_t)kvh * HD;
    const bf16* nw = uu < G ? qn : kn;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      xr[i] = bf(src[lane + 32 * i]);
      wr[i] = (uu < G + 1 && qk_norm) ? bf(nw[lane + 32 * i]) : 0.f;
    }
  };
  // the raw rows of this warp's first task do not depend on the position: their loads overlap the position load
  if (u < ntask) load_task(u);
  const int start = split * CH;
  // K rows of the chunk: two threads per key, each half of the head dim; V rows: thread (dp = tid >> 1, parity kg = tid & 1) owns dims 2dp, 2dp + 1 of the keys j = kg, kg + 2, ...
  const int jk = tid >> 1, half = tid & 1, kg = tid & 1, dp = tid >> 1;
  uint4 kvv[HD / 16];
  __nv_bfloat162 vpre[CH / 2];
  const bf16* K = kv + ((size_t)kvh * ctx + start) * HD;
  const bf16* V = kv + (((size_t)Hkv + kvh) * ctx + start) * HD;
  const __nv_bfloat162 zero2 = __floats2bfloat162_rn(0.f, 0.f);
  auto load_kv = [&](int jl) {  // rows below jl
    if (jk < jl) {
      const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)jk * HD + half * (HD / 2));
#pragma unroll
      for (int c = 0; c < HD / 16; c++) kvv[c] = kr[c];
    }
    const __nv_bfloat162* Vp = reinterpret_cast<const __nv_bfloat162*>(V) + dp;
#pragma unroll
    for (int i = 0; i < CH / 2; i++) {
      const int j = kg + 2 * i;
      vpre[i] = zero2;
      if (2 * i < jl && j < jl) vpre[i] = Vp[(size_t)j * DP];
    }
  };
  const int live = p + 1;
  if (start >= live) return;
  const int n = min(CH, live - start);
  const bool owner = live - start <= CH;  // this block's chunk holds position p (row n - 1)
  bool has = u < ntask && (u < G || owner);
  load_kv(n);
  while (has) {
    float x0 = xr[0], x1 = xr[1], x2 = xr[2], x3 = xr[3];
    if (u == G + 1) {  // V row: plain copy into the cache
      bf16* dst = kv + (((size_t)Hkv + kvh) * ctx + p) * HD;
      dst[lane] = tb(x0);
      dst[lane + 32] = tb(x1);
      dst[lane + 64] = tb(x2);
      dst[lane + 96] = tb(x3);
      vs[lane] = x0;
      vs[lane + 32] = x1;
      vs[lane + 64] = x2;
      vs[lane + 96] = x3;
    } else {
      if (qk_norm) {
        const float s0 = warp_sum(x0 * x0), s1 = warp_sum(x1 * x1), s2 = warp_sum(x2 * x2), s3 = warp_sum(x3 * x3);
        const float ss = (((0.f + s0) + s1) + s2) + s3;
        const float r = rsqrtf(ss / (float)HD + eps);
        x0 = rb(x0 * r * wr[0]);
        x1 = rb(x1 * r * wr[1]);
        x2 = rb(x2 * r * wr[2]);
        x3 = rb(x3 * r * wr[3]);
      }
      float o[4];
#pragma unroll
      for (int jw = 0; jw < 2; jw++) {
        const int j = lane + 32 * jw;
        const float inv = exp2f(-(2.f * (float)j / (float)HD) * log2f(theta));
        float s, c;
        sincosf((float)p * inv, &s, &c);
        const float lo = jw ? x1 : x0, hi = jw ? x3 : x2;
        o[jw] = __fmaf_rn(lo, c, -__fmul_rn(hi, s));
        o[jw + 2] = __fmaf_rn(hi, c, __fmul_rn(lo, s));
      }
      if (u < G) {
#pragma unroll
        for (int i = 0; i < 4; i++) qs[u * HD + lane + 32 * i] = rb(o[i]);
      } else {
        bf16* dst = kv + ((size_t)kvh * ctx + p) * HD;
#pragma unroll
        for (int i = 0; i < 4; i++) {
          dst[lane + 32 * i] = tb(o[i]);
          ks[lane + 32 * i] = rb(o[i]);
        }
      }
    }
    u += 4;
    has = u < ntask && (u < G || owner);
    if (has) load_task(u);
  }
  __syncthreads();
  {
    float dot[GM];
#pragma unroll
    for (int g = 0; g < GM; g++) dot[g] = 0.f;
    if (jk < n) {
      const bool mine = owner && jk == n - 1;
#pragma unroll
      for (int c = 0; c < HD / 16; c++) {
        float f[8];
        if (mine) {
#pragma unroll
          for (int e = 0; e < 8; e++) f[e] = ks[half * (HD / 2) + c * 8 + e];
        } else {
          unpack8(kvv[c], f);
        }
#pragma unroll
        for (int g = 0; g < GM; g++)
          if (g < G) {
#pragma unroll
            for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + half * (HD / 2) + c * 8 + e], dot[g]);
          }
      }
    }
#pragma unroll
    for (int g = 0; g < GM; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
    if (half == 0) {
#pragma unroll
      for (int g = 0; g < GM; g++)
        if (g < G) sc[g * CH + jk] = jk < n ? dot[g] * scale : -INFINITY;
    }
  }
  __syncthreads();
  for (int g = wp; g < G; g += 4) {  // one warp per query head (CH = 64: two scores per lane)
    const float s0 = sc[g * CH + lane], s1 = sc[g * CH + lane + 32];
    const float m = warp_max(fmaxf(s0, s1));
    const float e0 = s0 == -INFINITY ? 0.f : __expf(s0 - m), e1 = s1 == -INFINITY ? 0.f : __expf(s1 - m);
    const float l = warp_sum(e0 + e1);
    sc[g * CH + lane] = e0;
    sc[g * CH + lane + 32] = e1;
    if (lane == 0) {
      ml[g] = m;
      ml[8 + g] = l;
    }
  }
  const int jlast = owner ? n - 1 : -1;  // the row written by this block: taken from shared memory
  const float2 vlast = owner ? make_float2(vs[2 * dp], vs[2 * dp + 1]) : make_float2(0.f, 0.f);
  __syncthreads();
  float2 o[GM];
#pragma unroll
  for (int g = 0; g < GM; g++) o[g] = make_float2(0.f, 0.f);
  // keys j >= n carry p = 0 and a zero V row: they add +0 and change nothing
#pragma unroll
  for (int i = 0; i < CH / 2; i++) {
    if (2 * i < n) {
      const int j = kg + 2 * i;
      float2 vv = __bfloat1622float2(vpre[i]);
      if (j == jlast) vv = vlast;
#pragma unroll
      for (int g = 0; g < GM; g++)
        if (g < G) {
          const float pj = sc[g * CH + j];
          o[g].x = fmaf(pj, vv.x, o[g].x);
          o[g].y = fmaf(pj, vv.y, o[g].y);
        }
    }
  }
#pragma unroll
  for (int g = 0; g < GM; g++)
    if (g < G) {
      // dims 2dp (parity-0 lane keeps .x) and 2dp + 1 (parity-1 lane keeps .y): parity 0 + parity 1, k_attn_split's order
      const float send = kg ? o[g].x : o[g].y;
      const float recv = __shfl_xor_sync(0xffffffffu, send, 1);
      float* pp = part + ((size_t)(kvh * G + g) * nsplit + split) * (HD + 2);
      pp[tid] = kg ? o[g].y + recv : o[g].x + recv;
      if (tid == 0) {
        pp[HD] = ml[g];
        pp[HD + 1] = ml[8 + g];
      }
    }
}

// Chunked bf16 flash decode: the same split kernel as k_attn_split, with `nc` consecutive 64-key chunks per block and the running (m, l) of the online softmax, so a long context does not
// need more than MAX_SPLITS blocks per head. With nc = 1 it computes exactly what k_attn_split computes, bit for bit.
template <int HD>
__global__ void __launch_bounds__(128) k_attn_split_x(const bf16* q, const bf16* kv, const int* pos0, float* part, int Hq, int Hkv, int ctx, int nsplit, int nc, float scale) {
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DP = HD / 2;
  constexpr int KG = 128 / DP > 0 ? 128 / DP : 1;
  float* qs = smem;              // [G][HD]
  float* sc = qs + G * HD;       // [G][CH]
  float* po = sc + G * CH;       // [KG][G][HD]
  float* ml = po + KG * G * HD;  // [3][8]: running m, running l, rescale factor of the last chunk
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int live = *pos0 + t + 1;
  const int start0 = split * nc * CH;
  if (start0 >= live) return;
  for (int g = 0; g < G; g++)
    for (int d = tid; d < HD; d += 128) qs[g * HD + d] = bf(q[((size_t)t * Hq + kvh * G + g) * HD + d]);
  if (tid < 8) {
    ml[tid] = -INFINITY;
    ml[8 + tid] = 0.f;
  }
  const int dp = tid % DP, kg = tid / DP;
  float2 o[8];
#pragma unroll
  for (int g = 0; g < 8; g++) o[g] = make_float2(0.f, 0.f);
  for (int c = 0; c < nc; c++) {
    const int start = start0 + c * CH;
    if (start >= live) break;
    const int n = min(CH, live - start);
    __syncthreads();
    const bf16* K = kv + ((size_t)kvh * ctx + start) * HD;
    const bf16* V = kv + (((size_t)Hkv + kvh) * ctx + start) * HD;
    {
      const int j = tid >> 1, half = tid & 1;
      float dot[8];
#pragma unroll
      for (int g = 0; g < 8; g++) dot[g] = 0.f;
      if (j < n) {
        const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)j * HD + half * (HD / 2));
        uint4 kvv[HD / 16];
#pragma unroll
        for (int cc = 0; cc < HD / 16; cc++) kvv[cc] = kr[cc];
#pragma unroll
        for (int cc = 0; cc < HD / 16; cc++) {
          float f[8];
          unpack8(kvv[cc], f);
#pragma unroll
          for (int g = 0; g < 8; g++)
            if (g < G) {
#pragma unroll
              for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + half * (HD / 2) + cc * 8 + e], dot[g]);
            }
        }
      }
#pragma unroll
      for (int g = 0; g < 8; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
      if (half == 0)
        for (int g = 0; g < G; g++) sc[g * CH + j] = j < n ? dot[g] * scale : -INFINITY;
    }
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
    if (tid < DP * KG) {
      if (c > 0) {
#pragma unroll
        for (int g = 0; g < 8; g++)
          if (g < G) {
            const float al = ml[16 + g];
            o[g].x *= al;
            o[g].y *= al;
          }
      }
      const __nv_bfloat162* Vp = reinterpret_cast<const __nv_bfloat162*>(V) + dp;
#pragma unroll 8
      for (int j = kg; j < n; j += KG) {
        const float2 vv = __bfloat1622float2(Vp[(size_t)j * DP]);
#pragma unroll
        for (int g = 0; g < 8; g++)
          if (g < G) {
            const float pj = sc[g * CH + j];
            o[g].x = fmaf(pj, vv.x, o[g].x);
            o[g].y = fmaf(pj, vv.y, o[g].y);
          }
      }
    }
  }
  __syncthreads();
  if (tid < DP * KG) {
#pragma unroll
    for (int g = 0; g < 8; g++)
      if (g < G) {
        po[((size_t)kg * G + g) * HD + 2 * dp] = o[g].x;
        po[((size_t)kg * G + g) * HD + 2 * dp + 1] = o[g].y;
      }
  }
  __syncthreads();
  for (int g = 0; g < G; g++) {
    float* pp = part + (((size_t)t * Hq + kvh * G + g) * nsplit + split) * (HD + 4);
    for (int d = tid; d < HD; d += 128) {
      float a2 = 0.f;
#pragma unroll
      for (int c = 0; c < KG; c++) a2 += po[((size_t)c * G + g) * HD + d];
      pp[d] = a2;
    }
    if (tid == 0) {
      pp[HD] = ml[g];
      pp[HD + 1] = ml[8 + g];
    }
  }
}
