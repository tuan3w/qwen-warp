#pragma once
#include "dense_attn_common.cuh"

// Flash decode attention. Split over the key axis in chunks of CH = 64 keys so that a short context still spreads over many SMs. Block (kv head, split, row t), 128 threads:
//   scores : two threads per key (each half of the head dim), all G = Hq / Hkv query heads that share the kv head scored against the same K reads
//   softmax: one warp per query head, inside the chunk
//   P V    : a thread owns a pair of output dims and a subset of the keys, 4-byte coalesced loads, several keys in flight
// Partials per (row, head, split) = {o[hd], m, l}; k_attn_reduce merges them with the usual rescaling (a last-block-reduces variant with a
// ticket counter measured 1-2 us slower per layer).
// A sliding window (window > 0) skips the chunks wholly below the row's oldest live key and masks the keys below it inside the first live chunk; split s of a row is chunk
// lo / CH + s, and the state is a ring (slot = position mod ctx, ctx a multiple of CH). HD 512: each thread owns two pairs of output dims and loads K in two rounds of 16 uint4.
template <int HD>
__global__ void __launch_bounds__(128) k_attn_split(const bf16* q, const bf16* kv, const int* pos0, float* part, int Hq, int Hkv, int ctx, int nsplit, float scale, int window) {
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DP = HD / 2;
  constexpr int KG = 128 / DP > 0 ? 128 / DP : 1;
  constexpr int DT = DP < 128 ? DP : 128;           // threads along the pairs of output dims in P V (HD 512: each owns two pairs)
  constexpr int NP = DP / DT;                       // pairs per thread
  constexpr int KC = HD / 16, KB = KC > 16 ? 16 : KC;  // 16-byte K loads per thread, and how many are in flight (HD 512: two rounds, not 128 registers of K)
  float* qs = smem;              // [G][HD]
  float* sc = qs + G * HD;       // [G][CH]
  float* po = sc + G * CH;       // [KG][G][HD]
  float* ml = po + KG * G * HD;  // [2][8]
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int live = *pos0 + t + 1;
  const int lo = attn_lo(live, window);
  const int start = (lo / CH + split) * CH;
  if (start >= live) return;
  const int n = min(CH, live - start);
  const int jlo = max(0, lo - start);  // keys of the chunk below the window
  for (int g = 0; g < G; g++)
    for (int d = tid; d < HD; d += 128) qs[g * HD + d] = bf(q[((size_t)t * Hq + kvh * G + g) * HD + d]);
  __syncthreads();
  const size_t row0 = window ? start % ctx : start;
  const bf16* K = kv + ((size_t)kvh * ctx + row0) * HD;
  const bf16* V = kv + (((size_t)Hkv + kvh) * ctx + row0) * HD;
  {
    const int j = tid >> 1, half = tid & 1;
    float dot[8];
#pragma unroll
    for (int g = 0; g < 8; g++) dot[g] = 0.f;
    if (j >= jlo && j < n) {
      const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)j * HD + half * (HD / 2));
      for (int c0 = 0; c0 < KC; c0 += KB) {
        uint4 kvv[KB];
#pragma unroll
        for (int c = 0; c < KB; c++) kvv[c] = kr[c0 + c];
#pragma unroll
        for (int c = 0; c < KB; c++) {
          float f[8];
          unpack8(kvv[c], f);
#pragma unroll
          for (int g = 0; g < 8; g++)
            if (g < G) {
#pragma unroll
              for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + half * (HD / 2) + (c0 + c) * 8 + e], dot[g]);
            }
        }
      }
    }
#pragma unroll
    for (int g = 0; g < 8; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
    if (half == 0)
      for (int g = 0; g < G; g++) sc[g * CH + j] = j >= jlo && j < n ? dot[g] * scale : -INFINITY;
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
  const int dp = tid % DT, kg = tid / DT;
  if (tid < DT * KG) {
    float2 o[NP][8];
#pragma unroll
    for (int r = 0; r < NP; r++)
#pragma unroll
      for (int g = 0; g < 8; g++) o[r][g] = make_float2(0.f, 0.f);
    const __nv_bfloat162* Vp = reinterpret_cast<const __nv_bfloat162*>(V) + dp;
#pragma unroll 8
    for (int j = kg; j < n; j += KG) {
      float2 vv[NP];
#pragma unroll
      for (int r = 0; r < NP; r++) vv[r] = __bfloat1622float2(Vp[(size_t)j * DP + r * DT]);
#pragma unroll
      for (int g = 0; g < 8; g++)
        if (g < G) {
          const float pj = sc[g * CH + j];
#pragma unroll
          for (int r = 0; r < NP; r++) {
            o[r][g].x = fmaf(pj, vv[r].x, o[r][g].x);
            o[r][g].y = fmaf(pj, vv[r].y, o[r][g].y);
          }
        }
    }
    for (int g = 0; g < G; g++)
#pragma unroll
      for (int r = 0; r < NP; r++) {
        po[((size_t)kg * G + g) * HD + 2 * (dp + r * DT)] = o[r][g].x;
        po[((size_t)kg * G + g) * HD + 2 * (dp + r * DT) + 1] = o[r][g].y;
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
__global__ void __launch_bounds__(128) k_attn_reduce(const float* part, const int* pos0, bf16* out, int Hq, int nsplit, int window) {
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
  const int ns = attn_blocks(live, window, CH);
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
                                                   int ctx, int nsplit, float scale, const float* __restrict__ inv_freq, float eps, int qk_norm) {
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
  const float inv[2] = {inv_freq[lane], inv_freq[lane + 32]};  // the lane's rotary pairs, issued beside the position load
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
        float s, c;
        sincosf((float)p * inv[jw], &s, &c);
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
__global__ void __launch_bounds__(128) k_attn_split_x(const bf16* q, const bf16* kv, const int* pos0, float* part, int Hq, int Hkv, int ctx, int nsplit, int nc, float scale, int window) {
  extern __shared__ float smem[];
  const int G = Hq / Hkv;
  constexpr int DP = HD / 2;
  constexpr int KG = 128 / DP > 0 ? 128 / DP : 1;
  constexpr int DT = DP < 128 ? DP : 128;  // as k_attn_split: HD 512 gives each thread two pairs of output dims
  constexpr int NP = DP / DT;
  constexpr int KC = HD / 16, KB = KC > 16 ? 16 : KC;
  float* qs = smem;              // [G][HD]
  float* sc = qs + G * HD;       // [G][CH]
  float* po = sc + G * CH;       // [KG][G][HD]
  float* ml = po + KG * G * HD;  // [3][8]: running m, running l, rescale factor of the last chunk
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int live = *pos0 + t + 1;
  const int lo = attn_lo(live, window);
  const int start0 = (lo / CH + split * nc) * CH;
  if (start0 >= live) return;
  for (int g = 0; g < G; g++)
    for (int d = tid; d < HD; d += 128) qs[g * HD + d] = bf(q[((size_t)t * Hq + kvh * G + g) * HD + d]);
  if (tid < 8) {
    ml[tid] = -INFINITY;
    ml[8 + tid] = 0.f;
  }
  const int dp = tid % DT, kg = tid / DT;
  float2 o[NP][8];
#pragma unroll
  for (int r = 0; r < NP; r++)
#pragma unroll
    for (int g = 0; g < 8; g++) o[r][g] = make_float2(0.f, 0.f);
  for (int c = 0; c < nc; c++) {
    const int start = start0 + c * CH;
    if (start >= live) break;
    const int n = min(CH, live - start);
    const int jlo = max(0, lo - start);
    __syncthreads();
    const size_t row0 = window ? start % ctx : start;
    const bf16* K = kv + ((size_t)kvh * ctx + row0) * HD;
    const bf16* V = kv + (((size_t)Hkv + kvh) * ctx + row0) * HD;
    {
      const int j = tid >> 1, half = tid & 1;
      float dot[8];
#pragma unroll
      for (int g = 0; g < 8; g++) dot[g] = 0.f;
      if (j >= jlo && j < n) {
        const uint4* kr = reinterpret_cast<const uint4*>(K + (size_t)j * HD + half * (HD / 2));
        for (int c0 = 0; c0 < KC; c0 += KB) {
          uint4 kvv[KB];
#pragma unroll
          for (int cc = 0; cc < KB; cc++) kvv[cc] = kr[c0 + cc];
#pragma unroll
          for (int cc = 0; cc < KB; cc++) {
            float f[8];
            unpack8(kvv[cc], f);
#pragma unroll
            for (int g = 0; g < 8; g++)
              if (g < G) {
#pragma unroll
                for (int e = 0; e < 8; e++) dot[g] = fmaf(f[e], qs[g * HD + half * (HD / 2) + (c0 + cc) * 8 + e], dot[g]);
              }
          }
        }
      }
#pragma unroll
      for (int g = 0; g < 8; g++) dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
      if (half == 0)
        for (int g = 0; g < G; g++) sc[g * CH + j] = j >= jlo && j < n ? dot[g] * scale : -INFINITY;
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
    if (tid < DT * KG) {
      if (c > 0) {
#pragma unroll
        for (int g = 0; g < 8; g++)
          if (g < G) {
            const float al = ml[16 + g];
#pragma unroll
            for (int r = 0; r < NP; r++) {
              o[r][g].x *= al;
              o[r][g].y *= al;
            }
          }
      }
      const __nv_bfloat162* Vp = reinterpret_cast<const __nv_bfloat162*>(V) + dp;
#pragma unroll 8
      for (int j = kg; j < n; j += KG) {
        float2 vv[NP];
#pragma unroll
        for (int r = 0; r < NP; r++) vv[r] = __bfloat1622float2(Vp[(size_t)j * DP + r * DT]);
#pragma unroll
        for (int g = 0; g < 8; g++)
          if (g < G) {
            const float pj = sc[g * CH + j];
#pragma unroll
            for (int r = 0; r < NP; r++) {
              o[r][g].x = fmaf(pj, vv[r].x, o[r][g].x);
              o[r][g].y = fmaf(pj, vv[r].y, o[r][g].y);
            }
          }
      }
    }
  }
  __syncthreads();
  if (tid < DT * KG) {
#pragma unroll
    for (int g = 0; g < 8; g++)
      if (g < G) {
#pragma unroll
        for (int r = 0; r < NP; r++) {
          po[((size_t)kg * G + g) * HD + 2 * (dp + r * DT)] = o[r][g].x;
          po[((size_t)kg * G + g) * HD + 2 * (dp + r * DT) + 1] = o[r][g].y;
        }
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

// One-row decode, head dim 256 / 512, rotary dims = head dim: the whole attention in one launch (two past MERGE_FUSE chunks). Chunks of DK = 32 keys, block (kv head, chunk) of HD
// threads (many warps, so that the latencies of the phases overlap):
//   loads  : the K and V rows of the chunk go to shared memory with cp.async (rows padded by 16 bytes against bank conflicts) before anything else, so their latency overlaps the prep
//   prep   : one warp per head recomputes the G query heads of its kv head from the projection output (norm, rope, bf16 rounding: k_attn_fused's arithmetic, lane l holds the elements
//            l, l + 32, ...: a rope pair is the same lane); the block that holds position p (the last chunk) also computes the K / V row, writes it to slot p mod ctx and into its
//            shared tiles. READ: the cache is another layer's, no K / V row.
//   scores : tensor cores: S[key][head] = K q^T, m16n8k16 bf16 (K and the rounded q are bf16, fp32 accumulation; the 8 heads are the n = 8), 2 m-tiles x HD / 64 k-groups of 4 steps
//            over the warps, the k-groups added through shared memory; softmax one warp per head; P rounded to bf16 as the reference does, O[head][dim] = P V on the tensor cores
//            (rows 8 .. 15 of the m16 tile are zero), V read with ldmatrix.trans
//   merge  : partials {o[HD], m, l} with stride HD + 4; the block that takes the last ticket of its kv head merges the live chunks, one warp per head (k_attn_reduce2's arithmetic).
//            A launch with more than MERGE_FUSE chunks leaves the merge to k_attn_reduce2: one block would read too many partials.
// The loops are rolled and the code is a few hundred instructions: every block of a launch runs on a cold instruction cache (the first, fully unrolled FMA version was 8000
// instructions and cost 20 us), and the FMA scores were shared-memory-bandwidth bound (5000 cycles).
#define DK 32
#define MERGE_FUSE 32
#define PTS 40  // row stride of the bf16 probabilities (elements): rows 80 bytes apart, no bank conflicts
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void ldsm_x2_t(unsigned (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"((unsigned)__cvta_generic_to_shared(p)));
}
__device__ __forceinline__ void mma16816(float (&c)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int HD>
constexpr size_t decode1_smem() {  // bytes of dynamic shared memory
  constexpr int QS = HD + 8, KG = HD / 64;
  return (size_t)8 * QS * 2 + 2 * (size_t)DK * QS * 2 + (size_t)KG * DK * 8 * 4 + DK * 8 * 4 + 8 * PTS * 2 + 16 * 4;
}
template <int HD, int READ>
__global__ void __launch_bounds__(HD)
    k_attn_decode1(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, float* part, bf16* out, unsigned* cnt, int Hq, int Hkv, int ctx,
                   int nsplit, float scale, const float* __restrict__ inv_freq, float eps, int qk_norm, int v_norm, int window, int merge) {
  static_assert(HD == 256 || HD == 512, "decode1: head dim 256 or 512");
  constexpr int NT = HD, NW = NT / 32, EL = HD / 32, HP = HD / 64, U4 = HD / 8, PS = HD + 4, QS = HD + 8, KG = HD / 64;
  extern __shared__ __align__(16) unsigned char smem_d1[];
  __shared__ int last;
  const int G = Hq / Hkv;
  bf16* qs = reinterpret_cast<bf16*>(smem_d1);             // [8][QS] rounded q rows (rows >= G unused)
  bf16* Ks = qs + 8 * QS;                                  // [DK][QS] key rows
  bf16* Vs = Ks + DK * QS;                                 // [DK][QS] value rows (rows outside the live range zero)
  float* psc = reinterpret_cast<float*>(Vs + DK * QS);     // [KG][DK][8] partial scores of the k-groups
  float* sct = psc + KG * DK * 8;                          // [DK][8] scores
  bf16* pt = reinterpret_cast<bf16*>(sct + DK * 8);        // [8][PTS] probabilities
  float* ml = reinterpret_cast<float*>(pt + 8 * PTS);      // [2][8]
  const int kvh = blockIdx.x, split = blockIdx.y, tid = threadIdx.x, lane = tid & 31, wp = tid >> 5;
  float inv[HP], wq[EL], wk[EL], xr[EL];
#pragma unroll
  for (int i = 0; i < HP; i++) inv[i] = inv_freq[lane + 32 * i];
#pragma unroll
  for (int i = 0; i < EL; i++) {
    wq[i] = qk_norm ? bf(qn[lane + 32 * i]) : 0.f;
    wk[i] = (qk_norm && !READ) ? bf(kn[lane + 32 * i]) : 0.f;
  }
  const int ntask_all = G + (READ ? 0 : 2);
  auto load_task = [&](int u) {  // the raw projection row of task u: G query heads, then K, then V
    const bf16* src = u < G ? q + (size_t)(kvh * G + u) * HD : (u == G ? k : v) + (size_t)kvh * HD;
#pragma unroll
    for (int i = 0; i < EL; i++) xr[i] = bf(src[lane + 32 * i]);
  };
  if (wp < ntask_all) load_task(wp);  // does not depend on the position
  const int p = *pos0, live = p + 1;
  const int lo = attn_lo(live, window);
  const int start = (lo / DK + split) * DK;
  if (start >= live) return;
  const int n = min(DK, live - start), jlo = max(0, lo - start);
  const bool owner = !READ && live - start <= DK;  // this block's chunk holds position p (row n - 1)
  {
    const size_t row0 = window ? start % ctx : start;
    const char* Kg = reinterpret_cast<const char*>(kv + ((size_t)kvh * ctx + row0) * HD);
    const char* Vg = reinterpret_cast<const char*>(kv + (((size_t)Hkv + kvh) * ctx + row0) * HD);
    const int jend = owner ? n - 1 : n;  // the owner's newest row comes from the prep below
    for (int i = tid; i < DK * U4; i += NT) {
      const int jj = i / U4, c = i % U4;
      if (jj >= jlo && jj < jend) {
        cp_async16(reinterpret_cast<char*>(Ks + jj * QS) + c * 16, Kg + (size_t)i * 16);
        cp_async16(reinterpret_cast<char*>(Vs + jj * QS) + c * 16, Vg + (size_t)i * 16);
      } else if (jj < jlo || jj >= n) {  // P is 0 there, so the value row must be finite
        *(reinterpret_cast<uint4*>(Vs + jj * QS) + c) = make_uint4(0, 0, 0, 0);
      }
    }
    asm volatile("cp.async.commit_group;");
  }
  const int ntask = G + (owner ? 2 : 0);
  for (int u = wp; u < ntask; u += NW) {
    float x[EL];
#pragma unroll
    for (int i = 0; i < EL; i++) x[i] = xr[i];
    if (u + NW < ntask) load_task(u + NW);
    if (u == G + 1) {  // V row
      if (v_norm) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < EL; i++) s += x[i] * x[i];
        const float r = rsqrtf(warp_sum(s) / (float)HD + eps);
#pragma unroll
        for (int i = 0; i < EL; i++) x[i] = rb(x[i] * r);
      }
      bf16* dst = kv + (((size_t)Hkv + kvh) * ctx + p % ctx) * HD;
      bf16* tile = Vs + (n - 1) * QS;
#pragma unroll
      for (int i = 0; i < EL; i++) {
        dst[lane + 32 * i] = tb(x[i]);
        tile[lane + 32 * i] = tb(x[i]);
      }
    } else {
      if (qk_norm) {
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < EL; i++) s += x[i] * x[i];
        const float r = rsqrtf(warp_sum(s) / (float)HD + eps);
#pragma unroll
        for (int i = 0; i < EL; i++) x[i] = rb(x[i] * r * (u < G ? wq[i] : wk[i]));
      }
      float o[EL];
#pragma unroll
      for (int jw = 0; jw < HP; jw++) {
        float s, c;
        sincosf((float)p * inv[jw], &s, &c);
        o[jw] = __fmaf_rn(x[jw], c, -__fmul_rn(x[jw + HP], s));
        o[jw + HP] = __fmaf_rn(x[jw + HP], c, __fmul_rn(x[jw], s));
      }
      if (u < G) {
#pragma unroll
        for (int i = 0; i < EL; i++) qs[u * QS + lane + 32 * i] = tb(o[i]);
      } else {
        bf16* dst = kv + ((size_t)kvh * ctx + p % ctx) * HD;
        bf16* tile = Ks + (n - 1) * QS;
#pragma unroll
        for (int i = 0; i < EL; i++) {
          dst[lane + 32 * i] = tb(o[i]);
          tile[lane + 32 * i] = tb(o[i]);
        }
      }
    }
  }
  asm volatile("cp.async.wait_all;");
  __syncthreads();
  {
    const int mt = wp / KG, kg = wp % KG, g = lane >> 2, t = lane & 3;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
      const int k0 = kg * 64 + ks * 16;
      unsigned a[4], b[2];
      ldsm_x4(a, Ks + (mt * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * QS + k0 + (lane >> 4) * 8);
      ldsm_x2(b, qs + (lane & 7) * QS + k0 + ((lane >> 3) & 1) * 8);
      mma16816(c, a, b);
    }
    *reinterpret_cast<float2*>(psc + (kg * DK + mt * 16 + g) * 8 + 2 * t) = make_float2(c[0], c[1]);
    *reinterpret_cast<float2*>(psc + (kg * DK + mt * 16 + g + 8) * 8 + 2 * t) = make_float2(c[2], c[3]);
  }
  __syncthreads();
  if (tid < DK * 8) {
    const int j = tid >> 3;
    float sum = 0.f;
#pragma unroll
    for (int kg = 0; kg < KG; kg++) sum += psc[kg * DK * 8 + tid];
    sct[tid] = (j >= jlo && j < n) ? sum * scale : -INFINITY;
  }
  __syncthreads();
  if (wp < 8) {  // one warp per query head, a key per lane
    const int g = wp;
    const float s = sct[lane * 8 + g];
    const float m = warp_max(s);
    const float e = s == -INFINITY ? 0.f : __expf(s - m);
    const float l = warp_sum(e);
    pt[g * PTS + lane] = tb(e);
    if (lane == 0) {
      ml[g] = m;
      ml[8 + g] = l;
    }
  }
  __syncthreads();
  {
    const int g = lane >> 2, t = lane & 3;
    unsigned a[4];
    const unsigned* prow = reinterpret_cast<const unsigned*>(pt + g * PTS);
    for (int i = 0; i < 4; i++) {
      const int nt = wp + NW * i;  // 8 output dims per tile; HD / 8 = 4 * NW tiles
      float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int ks = 0; ks < 2; ks++) {
        a[0] = prow[ks * 8 + t];
        a[1] = 0;
        a[2] = prow[ks * 8 + 4 + t];
        a[3] = 0;
        unsigned b[2];
        ldsm_x2_t(b, Vs + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * QS + nt * 8);
        mma16816(c, a, b);
      }
      if (g < G) *reinterpret_cast<float2*>(part + ((size_t)(kvh * G + g) * nsplit + split) * PS + nt * 8 + 2 * t) = make_float2(c[0], c[1]);
    }
  }
  if (tid < G) {
    float* pp = part + ((size_t)(kvh * G + tid) * nsplit + split) * PS;
    pp[HD] = ml[tid];
    pp[HD + 1] = ml[8 + tid];
  }
  if (!merge) return;
  const int ns = attn_blocks(live, window, DK);  // the blocks of this kv head that ran
  __syncthreads();
  if (tid == 0) {  // the barrier orders the block's partial rows before this fence, which publishes them (cumulatively) before the ticket
    __threadfence();
    last = atomicAdd(&cnt[kvh], 1u) == (unsigned)(ns - 1);
    if (last) cnt[kvh] = 0;  // every block of the launch has arrived: the next launch starts from zero
    __threadfence();
  }
  __syncthreads();
  if (!last || wp >= G) return;
  // merge, a warp per query head: lane s < ns holds the (m, l) of chunk s; lane i of the HD / 4 float4 columns i, i + 32, ... of the o rows
  const float* base = part + (size_t)(kvh * G + wp) * nsplit * PS;
  float ms = -INFINITY, ls = 0.f;
  if (lane < ns) {
    ms = __ldcg(base + lane * PS + HD);
    ls = __ldcg(base + lane * PS + HD + 1);
  }
  const float M = warp_max(ms);
  const float e = lane < ns ? __expf(ms - M) : 0.f;
  const float wl = e / warp_sum(ls * e);
  float4 a[HD / 128];
#pragma unroll
  for (int i = 0; i < HD / 128; i++) a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
  for (int s = 0; s < ns; s++) {
    const float w = __shfl_sync(0xffffffffu, wl, s);
#pragma unroll
    for (int i = 0; i < HD / 128; i++) {
      const float4 pv = __ldcg(reinterpret_cast<const float4*>(base + (size_t)s * PS) + lane + 32 * i);
      a[i].x += pv.x * w;
      a[i].y += pv.y * w;
      a[i].z += pv.z * w;
      a[i].w += pv.w * w;
    }
  }
#pragma unroll
  for (int i = 0; i < HD / 128; i++) {
    bf16* o = out + (size_t)(kvh * G + wp) * HD + 4 * (lane + 32 * i);
    *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(a[i].x, a[i].y);
    *reinterpret_cast<__nv_bfloat162*>(o + 2) = __floats2bfloat162_rn(a[i].z, a[i].w);
  }
}
