// swift-mk DSpine block drafter on the device (see include/dspine.h).
//
// DSpine = standard Qwen3 GQA attention (causal, window 8192, NO conv) + a rank-512 code bridge
// (predecessor-token feature injection) after every layer + a serial refinement tail.
//
// Layer loop: [in_ln, qkv GEMV, qk-norm+RoPE, causal attention, o_proj, residual,
//              post_ln, MLP, residual, code bridge write -> next in_ln]
//
// The code bridge: rms(raw) -> normed @ read_w.T -> codes (anchor from codebook) ->
// codes @ packed_w.T -> q/s/v -> gate = sigmoid(q+s+gate_b) -> msg = [0, gate*v] ->
// msg @ write_w.T -> delta -> out = raw + rms*delta -> rmsnorm(out)
#include <math.h>

#include "common.cuh"
#include "dspine.h"
#include "engine_impl.h"
#include "head_q.h"
#include "mtp_sample.h"

using namespace smk;

extern "C" cudaError_t smk_bf16_gemv_f32_cfg(const void* w, const void* x, int N, int K, int T, float* out, int cfg, cudaStream_t st);  // gemv.cu

namespace {
constexpr int NH = H / 8;
constexpr int NL = SMK_SP_NL;
constexpr int R8 = SMK_SP_ROWS;
constexpr int NQ = 32, NKV = 8, HD = 128;
constexpr int QD = NQ * HD, KVD = NKV * HD;
constexpr int FC_K = 5 * H;
constexpr int KV_N = NL * 2 * KVD;
constexpr int ATT_SMAX = 8208;  // window 8192 + 16 noise rows
constexpr int ATT_TILE = 256;    // key tile size for the tiled attention
constexpr int SP_R = SMK_SP_R, SP_C = SMK_SP_C, SP_K = SMK_SP_K;
constexpr int DD_NNEW = 0, DD_BASE = 1, DD_N = 4;
constexpr float SQRT_C = 32.f;
}  // namespace

struct smk_dspine {
  smk_engine* e;
  smk_sp_w w;
  int ctxd;
  int* dd;
  int* ss;
  int* ring;
  int* log;
  uint16_t *fcin, *th_raw, *th, *kvraw, *h, *xn, *qb, *kb, *vb, *att, *oo, *act, *mlp, *hid, *kc, *vc, *raw_last, *ref_as;
  uint16_t *br_normed, *br_msg;
  float *br_rms, *br_proj, *br_packed, *br_delta, *ref_base, *ref_tg, *ref_tsv;
  int *cand_id;
  int* tmp_tok;
  int* hist;
  int *pinned_ss, *pinned_ring;
  cudaGraphExec_t g_cycle = nullptr, g_verify = nullptr, g_stage = nullptr, g_ctx = nullptr, g_noise = nullptr, g_head = nullptr, g_start = nullptr;
  unsigned epoch = 0;
  size_t bytes = 0;
};

// ---------------------------------------------------------------------------------------------- kernels

// fcin[t][s * H ..] = tap slot s of row t.  grid (8, 5), NH threads.
__global__ void __launch_bounds__(NH) k_sp_gather_taps(const uint4* __restrict__ dbg, uint4* __restrict__ out) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x;
  const int slot = 2 + 15 * s;  // 2, 17, 32, 47, 62: after target layers 1, 16, 31, 46, 61
  out[(size_t)t * (FC_K / 8) + s * NH + c] = dbg[((size_t)slot * TM + t) * NH + c];
}

// out[t] = bf16(w * bf16(x[t] * rsqrt(mean(x[t]^2) + eps)))  grid T, NH threads.
__global__ void __launch_bounds__(NH) k_sp_rms(const uint4* __restrict__ x, const uint4* __restrict__ w, uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  const uint4 hv = x[(size_t)t * NH + c];
  float v[8] = {bflo(hv.x), bfhi(hv.x), bflo(hv.y), bfhi(hv.y), bflo(hv.z), bfhi(hv.z), bflo(hv.w), bfhi(hv.w)};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = rsqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  out[(size_t)t * NH + c] = make_uint4(pack_bf16x2(w0 * bf16r(v[0] * r), w1 * bf16r(v[1] * r)), pack_bf16x2(w2 * bf16r(v[2] * r), w3 * bf16r(v[3] * r)),
                                       pack_bf16x2(w4 * bf16r(v[4] * r), w5 * bf16r(v[5] * r)), pack_bf16x2(w6 * bf16r(v[6] * r), w7 * bf16r(v[7] * r)));
}

__device__ __forceinline__ void unpack8(const uint4& u, float (&v)[8]) {
  v[0] = bflo(u.x); v[1] = bfhi(u.x); v[2] = bflo(u.y); v[3] = bfhi(u.y); v[4] = bflo(u.z); v[5] = bfhi(u.z); v[6] = bflo(u.w); v[7] = bfhi(u.w);
}

// per-head RMSNorm + RoPE of one 128-element head by 128 threads.
__device__ __forceinline__ float head_norm_rope(float x, float w, int pos, const float* __restrict__ invf, float* sh, float* red, int d) {
  float ss = warp_sum(x * x);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  const float tot = red[0] + red[1] + red[2] + red[3];
  const float r = rsqrtf(tot * (1.f / 128.f) + 1e-6f);
  const float n = bf16r(w * bf16r(x * r));
  sh[d] = n;
  __syncthreads();
  float sn, cs;
  sincosf((float)pos * invf[d & 63], &sn, &cs);
  cs = bf16r(cs);
  sn = bf16r(sn);
  const float rot = d < 64 ? -sh[d + 64] : sh[d - 64];
  const float o = bf16r(bf16r(n * cs) + bf16r(rot * sn));
  __syncthreads();
  return o;
}

struct SpCtxPtrs {
  const uint16_t* knorm[NL];
};

// context K/V from kvraw -> caches.  grid (8, 80), 128 threads.
__global__ void __launch_bounds__(128) k_sp_ctx_kv(const uint16_t* __restrict__ kvraw, SpCtxPtrs cp, const float* __restrict__ invf, const int* __restrict__ dd, uint16_t* __restrict__ kc,
                                                   uint16_t* __restrict__ vc, int ctxd) {
  __shared__ float sh[128], red[4];
  const int t = blockIdx.x, l = blockIdx.y >> 4, j = blockIdx.y & 15, d = threadIdx.x;
  if (t >= dd[DD_NNEW]) return;
  const int pos = dd[DD_BASE] + t;
  const size_t lay = (size_t)l * NKV * ctxd * HD;
  if (j < NKV) {
    const float x = bf2f(kvraw[t * KV_N + l * 2 * KVD + j * HD + d]);
    const float o = head_norm_rope(x, bf2f(cp.knorm[l][d]), pos, invf, sh, red, d);
    if (pos < ctxd) kc[lay + ((size_t)j * ctxd + pos) * HD + d] = f2bf(o);
  } else {
    const int vh = j - NKV;
    if (pos < ctxd) vc[lay + ((size_t)vh * ctxd + pos) * HD + d] = kvraw[t * KV_N + l * 2 * KVD + KVD + vh * HD + d];
  }
}

// Causal attention over the cache with a sliding window of 8192.
// Tiled FlashAttention-style: one block per query head, 8 query rows, keys in ATT_TILE chunks.
// grid 32, 512 threads, dynamic smem (q 4 KB + scores 8 KB = 12 KB).
constexpr int SP_ATT_T = 512;
__global__ void __launch_bounds__(SP_ATT_T) k_sp_attn(const uint16_t* __restrict__ qb, const uint16_t* __restrict__ kb, const uint16_t* __restrict__ vb, const uint16_t* __restrict__ qnw,
                                                      const uint16_t* __restrict__ knw, const float* __restrict__ invf, uint16_t* kc, uint16_t* vc, const int* __restrict__ dd, int ctxd,
                                                      uint16_t* __restrict__ out) {
  extern __shared__ float sm[];
  __shared__ float smm[8];
  __shared__ float psh[4][128], pred[4][4];
  float* sq = sm;
  float* sc = sm + 8 * HD;
  const int h = blockIdx.x, kvh = h >> 2, tid = threadIdx.x;
  const int start = dd[DD_BASE] + dd[DD_NNEW];
  const int lo0 = max(0, start - 8191);
  const int nk = min(start + R8 - lo0, ATT_SMAX);
  {
    // q / k / v prep of the 8 noise rows (was k_sp_qk_prep): q head h -> smem, the 4 query heads of a KV group each write the same K / V rows of their KV head (identical values)
    const int grp = tid >> 7, d = tid & 127;
    const float qw = bf2f(qnw[d]), kw = bf2f(knw[d]);
#pragma unroll
    for (int pass = 0; pass < 2; pass++) {
      const int t = pass * 4 + grp, pos = start + t;
      sq[t * HD + d] = head_norm_rope(bf2f(qb[t * QD + h * HD + d]), qw, pos, invf, psh[grp], pred[grp], d);
      const float ko = head_norm_rope(bf2f(kb[t * KVD + kvh * HD + d]), kw, pos, invf, psh[grp], pred[grp], d);
      if (pos < ctxd) {
        kc[((size_t)kvh * ctxd + pos) * HD + d] = f2bf(ko);
        vc[((size_t)kvh * ctxd + pos) * HD + d] = vb[t * KVD + kvh * HD + d];
      }
    }
  }
  __syncthreads();
  const uint16_t* kbase = kc + (size_t)kvh * ctxd * HD;
  const float4* sq4 = (const float4*)sq;
  const uint32_t* vbase = (const uint32_t*)(vc + (size_t)kvh * ctxd * HD);
  const int g = tid >> 6, dp = tid & 63;
  float o0 = 0.f, o1 = 0.f, m = -INFINITY, l = 0.f;
  const int ntile = (nk + ATT_TILE - 1) / ATT_TILE;
  for (int t = 0; t < ntile; t++) {
    const int j0 = t * ATT_TILE;
    const int jn = min(ATT_TILE, nk - j0);
    for (int j = tid; j < jn; j += SP_ATT_T) {
      const int kpos = lo0 + j0 + j;
      const uint4* kp = (const uint4*)(kbase + (size_t)kpos * HD);
      uint4 ku[16];
#pragma unroll
      for (int c = 0; c < 16; c++) ku[c] = kp[c];
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = 0.f;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        const uint4 u = ku[c];
        const float kv[8] = {bflo(u.x), bfhi(u.x), bflo(u.y), bfhi(u.y), bflo(u.z), bfhi(u.z), bflo(u.w), bfhi(u.w)};
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const float4 qa = sq4[i * 32 + c * 2], qb4 = sq4[i * 32 + c * 2 + 1];
          acc[i] = fmaf(qa.x, kv[0], acc[i]); acc[i] = fmaf(qa.y, kv[1], acc[i]); acc[i] = fmaf(qa.z, kv[2], acc[i]); acc[i] = fmaf(qa.w, kv[3], acc[i]);
          acc[i] = fmaf(qb4.x, kv[4], acc[i]); acc[i] = fmaf(qb4.y, kv[5], acc[i]); acc[i] = fmaf(qb4.z, kv[6], acc[i]); acc[i] = fmaf(qb4.w, kv[7], acc[i]);
        }
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int qpos = start + i;
        sc[i * ATT_TILE + j] = (kpos <= qpos) ? acc[i] * 0.08838834764831845f : -INFINITY;
      }
    }
    __syncthreads();
    float tm = -INFINITY;
    {
      float t0 = -INFINITY, t1 = -INFINITY, t2 = -INFINITY, t3 = -INFINITY;
      int j = 0;
      for (; j + 4 <= jn; j += 4) {
        t0 = fmaxf(t0, sc[g * ATT_TILE + j]);
        t1 = fmaxf(t1, sc[g * ATT_TILE + j + 1]);
        t2 = fmaxf(t2, sc[g * ATT_TILE + j + 2]);
        t3 = fmaxf(t3, sc[g * ATT_TILE + j + 3]);
      }
      for (; j < jn; j++) t0 = fmaxf(t0, sc[g * ATT_TILE + j]);
      tm = fmaxf(fmaxf(t0, t1), fmaxf(t2, t3));
    }
    const float m_new = fmaxf(m, tm);
    const float s_old = (m == -INFINITY) ? 0.f : expf(m - m_new);
    l *= s_old;
    o0 *= s_old;
    o1 *= s_old;
    m = m_new;
    if (dp == 0) smm[g] = m;
    __syncthreads();
    if (tid < jn) {  // p = exp(score - m) once per (row, key) instead of once per (row, key, dim pair)
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const float mi = smm[i];
        sc[i * ATT_TILE + tid] = mi > -INFINITY ? expf(sc[i * ATT_TILE + tid] - mi) : 0.f;
      }
    }
    __syncthreads();
    if (m > -INFINITY) {
      // V loads of 16 keys issued ahead of the (in-order) l / o accumulation: the loop was bound by the global load latency
      int j = 0;
      for (; j + 16 <= jn; j += 16) {
        uint32_t vv[16];
#pragma unroll
        for (int u = 0; u < 16; u++) vv[u] = vbase[(size_t)(lo0 + j0 + j + u) * (HD / 2) + dp];
#pragma unroll
        for (int u = 0; u < 16; u++) {
          const float p = sc[g * ATT_TILE + j + u];
          l += p;
          o0 = fmaf(p, bflo(vv[u]), o0);
          o1 = fmaf(p, bfhi(vv[u]), o1);
        }
      }
      for (; j < jn; j++) {
        const float p = sc[g * ATT_TILE + j];
        l += p;
        const uint32_t v = vbase[(size_t)(lo0 + j0 + j) * (HD / 2) + dp];
        o0 = fmaf(p, bflo(v), o0);
        o1 = fmaf(p, bfhi(v), o1);
      }
    }
    __syncthreads();
  }
  if (l > 0.f) {
    const float inv = 1.f / l;
    out[g * QD + h * HD + 2 * dp] = f2bf(o0 * inv);
    out[g * QD + h * HD + 2 * dp + 1] = f2bf(o1 * inv);
  }
}

// Code bridge: apply delta + rmsnorm.  h[t] = bf16(h[t] + rms[t] * delta[t]) (t=0: no delta), out[t] = rmsnorm(h[t]).
// grid 8, NH threads.
__global__ void __launch_bounds__(NH) k_sp_br_apply(const uint16_t* __restrict__ h, const float* __restrict__ delta, const float* __restrict__ rms, const uint4* __restrict__ w, uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  const uint4 hu = ((const uint4*)h)[(size_t)t * NH + c];
  const float rv = rms[t];
  float hv[8];
  unpack8(hu, hv);
  if (t > 0) {
    const float4* dp = (const float4*)(delta + (size_t)t * H + (size_t)c * 8);
    const float4 d0 = dp[0], d1 = dp[1];
    const float dv[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
#pragma unroll
    for (int i = 0; i < 8; i++) hv[i] = bf16r(hv[i] + rv * bf16r(dv[i]));
  }
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(hv[i], hv[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = rsqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  ((uint4*)h)[(size_t)t * NH + c] = make_uint4(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]), pack_bf16x2(hv[4], hv[5]), pack_bf16x2(hv[6], hv[7]));
  out[(size_t)t * NH + c] = make_uint4(pack_bf16x2(w0 * bf16r(hv[0] * r), w1 * bf16r(hv[1] * r)), pack_bf16x2(w2 * bf16r(hv[2] * r), w3 * bf16r(hv[3] * r)),
                                       pack_bf16x2(w4 * bf16r(hv[4] * r), w5 * bf16r(hv[5] * r)), pack_bf16x2(w6 * bf16r(hv[6] * r), w7 * bf16r(hv[7] * r)));
}

// noise rows (row 0 = embed[anchor], rows 1..7 = embed[mask_id]) + rmsnorm with `w` (layer 0 input norm).  grid 8, NH threads.
__global__ void __launch_bounds__(NH) k_sp_embed_rms(const uint4* __restrict__ embed, const int* __restrict__ ds, int mask_id, const uint4* __restrict__ w, uint4* __restrict__ h, uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  const int tok = t == 0 ? ds[DS_TOK] : mask_id;
  const uint4 hu = embed[(size_t)tok * NH + c];
  h[(size_t)t * NH + c] = hu;
  float v[8];
  unpack8(hu, v);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = rsqrtf(tot / (float)H + 1e-6f);
  float wf[8];
  unpack8(w[c], wf);
  out[(size_t)t * NH + c] = make_uint4(pack_bf16x2(wf[0] * bf16r(v[0] * r), wf[1] * bf16r(v[1] * r)), pack_bf16x2(wf[2] * bf16r(v[2] * r), wf[3] * bf16r(v[3] * r)),
                                       pack_bf16x2(wf[4] * bf16r(v[4] * r), wf[5] * bf16r(v[5] * r)), pack_bf16x2(wf[6] * bf16r(v[6] * r), wf[7] * bf16r(v[7] * r)));
}

// h[t] = bf16(h[t] + y[t]); out[t] = bf16(w * bf16(h[t] * rsqrt(mean(h[t]^2) + eps))).  grid 8, NH threads.
__global__ void __launch_bounds__(NH) k_sp_res_rms(uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w, uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  float hv[8], yv[8];
  unpack8(h[(size_t)t * NH + c], hv);
  unpack8(y[(size_t)t * NH + c], yv);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    hv[i] = bf16r(hv[i] + yv[i]);
    ss = fmaf(hv[i], hv[i], ss);
  }
  h[(size_t)t * NH + c] = make_uint4(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]), pack_bf16x2(hv[4], hv[5]), pack_bf16x2(hv[6], hv[7]));
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = rsqrtf(tot / (float)H + 1e-6f);
  float wf[8];
  unpack8(w[c], wf);
  out[(size_t)t * NH + c] = make_uint4(pack_bf16x2(wf[0] * bf16r(hv[0] * r), wf[1] * bf16r(hv[1] * r)), pack_bf16x2(wf[2] * bf16r(hv[2] * r), wf[3] * bf16r(hv[3] * r)),
                                       pack_bf16x2(wf[4] * bf16r(hv[4] * r), wf[5] * bf16r(hv[5] * r)), pack_bf16x2(wf[6] * bf16r(hv[6] * r), wf[7] * bf16r(hv[7] * r)));
}

// h[t] = bf16(h[t] + y[t]) (the MLP residual); bridge norm of h: normed[t] = bf16(h * r), rms[t] = 1 / r; optionally also copy h -> raw_copy (pre-delta residual of the last layer).  grid 8, NH threads.
__global__ void __launch_bounds__(NH) k_sp_res_brnorm(uint4* __restrict__ h, const uint4* __restrict__ y, uint4* __restrict__ normed, float* __restrict__ rms_out, uint4* __restrict__ raw_copy) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  float hv[8], yv[8];
  unpack8(h[(size_t)t * NH + c], hv);
  unpack8(y[(size_t)t * NH + c], yv);
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    hv[i] = bf16r(hv[i] + yv[i]);
    ss = fmaf(hv[i], hv[i], ss);
  }
  const uint4 ho = make_uint4(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]), pack_bf16x2(hv[4], hv[5]), pack_bf16x2(hv[6], hv[7]));
  h[(size_t)t * NH + c] = ho;
  if (raw_copy) raw_copy[(size_t)t * NH + c] = ho;
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = rsqrtf(tot / (float)H + 1e-6f);
  if (c == 0) rms_out[t] = 1.f / r;
  normed[(size_t)t * NH + c] = make_uint4(pack_bf16x2(bf16r(hv[0] * r), bf16r(hv[1] * r)), pack_bf16x2(bf16r(hv[2] * r), bf16r(hv[3] * r)),
                                          pack_bf16x2(bf16r(hv[4] * r), bf16r(hv[5] * r)), pack_bf16x2(bf16r(hv[6] * r), bf16r(hv[7] * r)));
}

// Code bridge, codes + gate GEMV + message in one kernel (was br_codes, a bf16 GEMV and br_msg).
//   codes[t] = bf16(sqrt_c * proj[t] / (|proj[t]| + 1e-6)), codes[0] = bf16(sqrt_c * codebook[anchor])   (kept in smem as fp32, 8 rows)
//   packed[t][g * SP_R + ch] = codes[t] . gate_w[g * SP_R + ch]   (g: 0 gate q, 1 pred_gate s, 2 prev v; fp32, read by the refine chain)
//   msg[t] = bf16(bf16(sigmoid(q[t] + s[t-1] + b) * v[t-1])), msg[0] = 0
// One warp per channel, BG_CH channels per block.  grid SP_R / BG_CH, 32 * BG_CH threads.
constexpr int BG_CH = 4;
__global__ void __launch_bounds__(32 * BG_CH) k_sp_br_gate(const float* __restrict__ proj, const uint16_t* __restrict__ codebook, const int* __restrict__ ds, const uint16_t* __restrict__ gate_w,
                                                           const float* __restrict__ gate_b, float* __restrict__ packed, uint16_t* __restrict__ msg) {
  __shared__ float4 cs[R8][SP_C / 4];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
#pragma unroll
  for (int rr = 0; rr < R8 / BG_CH; rr++) {
    const int t = w * (R8 / BG_CH) + rr;
    float* cr = (float*)cs[t];
    if (t == 0) {
      const uint16_t* cb = codebook + (size_t)ds[DS_TOK] * SP_C;
      for (int c = lane; c < SP_C; c += 32) cr[c] = bf16r(SQRT_C * bf2f(cb[c]));
    } else {
      float p[SP_C / 32];
      float ss = 0.f;
#pragma unroll
      for (int j = 0; j < SP_C / 32; j++) {
        p[j] = proj[t * SP_C + lane + 32 * j];
        ss = fmaf(p[j], p[j], ss);
      }
      ss = warp_sum(ss);
      const float r = 1.f / (sqrtf(ss) + 1e-6f);
#pragma unroll
      for (int j = 0; j < SP_C / 32; j++) cr[lane + 32 * j] = bf16r(SQRT_C * p[j] * r);
    }
  }
  __syncthreads();
  const int ch = blockIdx.x * BG_CH + w;
  float acc[3][R8];
#pragma unroll
  for (int g = 0; g < 3; g++)
#pragma unroll
    for (int t = 0; t < R8; t++) acc[g][t] = 0.f;
#pragma unroll
  for (int j = 0; j < SP_C / 128; j++) {
    const int c4 = lane + 32 * j;
    uint2 wu[3];
#pragma unroll
    for (int g = 0; g < 3; g++) wu[g] = ((const uint2*)(gate_w + (size_t)(g * SP_R + ch) * SP_C))[c4];
#pragma unroll
    for (int g = 0; g < 3; g++) {
      const float w0 = bflo(wu[g].x), w1 = bfhi(wu[g].x), w2 = bflo(wu[g].y), w3 = bfhi(wu[g].y);
#pragma unroll
      for (int t = 0; t < R8; t++) {
        const float4 x = cs[t][c4];
        acc[g][t] = fmaf(x.x, w0, acc[g][t]);
        acc[g][t] = fmaf(x.y, w1, acc[g][t]);
        acc[g][t] = fmaf(x.z, w2, acc[g][t]);
        acc[g][t] = fmaf(x.w, w3, acc[g][t]);
      }
    }
  }
#pragma unroll
  for (int g = 0; g < 3; g++)
#pragma unroll
    for (int t = 0; t < R8; t++) acc[g][t] = warp_sum(acc[g][t]);
  if (lane == 0) {
    const float b = gate_b[ch];
#pragma unroll
    for (int t = 0; t < R8; t++) {
#pragma unroll
      for (int g = 0; g < 3; g++) packed[(size_t)t * 3 * SP_R + g * SP_R + ch] = acc[g][t];
      if (t == 0) {
        msg[ch] = 0;
      } else {
        const float gt = 1.f / (1.f + expf(-(acc[0][t] + acc[1][t - 1] + b)));
        msg[(size_t)t * SP_R + ch] = f2bf(bf16r(gt * acc[2][t - 1]));
      }
    }
  }
}

// Candidate order: the SP_K ids of noise row i (rank order of the head select) sorted ascending by id (the reference sorts them).  Thread c < SP_K of the caller computes the rank of
// its own id and stores it at sid[rank]; the caller syncs afterwards.
__device__ __forceinline__ void cand_sort(const int* __restrict__ ws_cand, int i, int c, int* sid) {
  const int* row = ws_cand + i * SMK_HEADQ_RMAX;
  const int me = row[c];
  int rank = 0;
#pragma unroll
  for (int j = 0; j < SP_K; j++) {
    const int o = row[j];
    rank += (o < me) || (o == me && j < c);
  }
  sid[rank] = me;
}

// Serial refine, candidate rows j = i * SP_K + k (i = noise row - 1):
//   as[j]   = bf16(lm_head[cand[j]] * norm_w)                       (input of the tg product)
//   base[j] = sum_h (raw[i + 1][h] * norm_w[h]) * lm_head[cand[j]][h]
// grid 7 * SP_K, NH threads.
__global__ void __launch_bounds__(NH) k_sp_ref_prep(const uint16_t* __restrict__ head, const uint16_t* __restrict__ norm_w, const uint16_t* __restrict__ raw, const int* __restrict__ ws_cand,
                                                    int* __restrict__ cand_id, uint16_t* __restrict__ as, float* __restrict__ base) {
  __shared__ float wsum[NH / 32];
  __shared__ int sid[SP_K];
  const int j = blockIdx.x, i = j / SP_K, c = threadIdx.x;
  if (c < SP_K) cand_sort(ws_cand, i, c, sid);
  __syncthreads();
  const int tok = sid[j % SP_K];
  if (c == 0) cand_id[j] = tok;
  float hv[8], wv[8], rv[8];
  unpack8(((const uint4*)head)[(size_t)tok * NH + c], hv);
  unpack8(((const uint4*)norm_w)[c], wv);
  unpack8(((const uint4*)raw)[(size_t)(i + 1) * NH + c], rv);
  float dot = 0.f;
#pragma unroll
  for (int e = 0; e < 8; e++) dot = fmaf(rv[e] * wv[e], hv[e], dot);
  ((uint4*)as)[(size_t)j * NH + c] = make_uint4(pack_bf16x2(hv[0] * wv[0], hv[1] * wv[1]), pack_bf16x2(hv[2] * wv[2], hv[3] * wv[3]), pack_bf16x2(hv[4] * wv[4], hv[5] * wv[5]), pack_bf16x2(hv[6] * wv[6], hv[7] * wv[7]));
  dot = warp_sum(dot);
  if ((c & 31) == 0) wsum[c >> 5] = dot;
  __syncthreads();
  if (c == 0) {
    float t = 0.f;
    for (int w = 0; w < NH / 32; w++) t += wsum[w];
    base[j] = t;
  }
}

// tg partials: tgp[hs][j][r] = sum_{h in slice hs} as[j][h] * write_w_last[h][r]  (write_w_last: bf16 [H][SP_R]; 4 h-slices of 1280 -> deterministic, the consumer sums the 4 and rounds to bf16).
// Block = 64 r x 16 rows; the as tile is staged in smem, 4 h-groups per block.  grid (SP_R / 64, 7 * SP_K / 16, 4), 256 threads.
constexpr int TG_HS = 4, TG_HC = 320;
__global__ void __launch_bounds__(256) k_sp_ref_tg(const uint16_t* __restrict__ as, const uint16_t* __restrict__ w, float* __restrict__ tgp) {
  __shared__ float at[16][TG_HC];
  __shared__ float red[4][16][64];
  const int rl = threadIdx.x & 63, hg = threadIdx.x >> 6;
  const int r = blockIdx.x * 64 + rl;
  const int j0 = blockIdx.y * 16, hs = blockIdx.z;
  float acc[16];
#pragma unroll
  for (int jj = 0; jj < 16; jj++) acc[jj] = 0.f;
  for (int ch = 0; ch < H / TG_HS / TG_HC; ch++) {
    const int hb = hs * (H / TG_HS) + ch * TG_HC;
    __syncthreads();
#pragma unroll 10
    for (int idx = threadIdx.x; idx < 16 * TG_HC; idx += 256) {
      const int jj = idx / TG_HC, hl = idx % TG_HC;
      at[jj][hl] = bf2f(as[(size_t)(j0 + jj) * H + hb + hl]);
    }
    __syncthreads();
#pragma unroll 5
    for (int hl = hg * (TG_HC / 4); hl < (hg + 1) * (TG_HC / 4); hl += 4) {
      float wv[4];
#pragma unroll
      for (int e = 0; e < 4; e++) wv[e] = bf2f(w[(size_t)(hb + hl + e) * SP_R + r]);
#pragma unroll
      for (int jj = 0; jj < 16; jj++) {
        const float4 a = *(const float4*)&at[jj][hl];
        acc[jj] = fmaf(a.x, wv[0], acc[jj]);
        acc[jj] = fmaf(a.y, wv[1], acc[jj]);
        acc[jj] = fmaf(a.z, wv[2], acc[jj]);
        acc[jj] = fmaf(a.w, wv[3], acc[jj]);
      }
    }
  }
#pragma unroll
  for (int jj = 0; jj < 16; jj++) red[hg][jj][rl] = acc[jj];
  __syncthreads();
  for (int idx = threadIdx.x; idx < 16 * 64; idx += 256) {
    const int jj = idx >> 6, r2 = idx & 63;
    tgp[((size_t)hs * (SMK_SP_DRAFTS * SP_K) + j0 + jj) * SP_R + blockIdx.x * 64 + r2] = red[0][jj][r2] + red[1][jj][r2] + red[2][jj][r2] + red[3][jj][r2];
  }
}

// tsv slots: slot 0 = anchor, slot 1 + (i - 1) * SP_K + k = candidate k of noise row i - 1 (the possible predecessors of noise rows 1..6).
// tsv[slot] = bf16(bf16(sqrt_c * codebook[token]) @ pv_w^T)  (pv_w = [pred_gate; prev] bf16 [2 * SP_R][SP_C]: tvg = first SP_R outputs, tvv = last SP_R).
// A block does TSV_PB slots x a quarter (TSV_RQ) of the output rows, one warp per output row (coalesced 16 B loads).  grid (ceil(97 / TSV_PB), TSV_RQ), 512 threads.
constexpr int TSV_SLOTS = 1 + (SMK_SP_DRAFTS - 1) * SP_K, TSV_PB = 4, TSV_RQ = 4;
__global__ void __launch_bounds__(512) k_sp_ref_tsv(const uint16_t* __restrict__ codebook, const uint16_t* __restrict__ pv_w, const int* __restrict__ ws_cand, const int* __restrict__ ds, float* __restrict__ tsv) {
  __shared__ float4 cs[TSV_PB][SP_C / 4];
  __shared__ int stok[TSV_PB][SP_K];
  const int s0 = blockIdx.x * TSV_PB, tid = threadIdx.x;
  if (tid < TSV_PB * SP_K) {
    const int q = tid / SP_K, m = tid % SP_K, slot = s0 + q;
    if (slot >= 1 && slot < TSV_SLOTS) cand_sort(ws_cand, (slot - 1) / SP_K, m, stok[q]);  // sorted ids of row (slot-1)/SP_K: entry (slot-1) % SP_K is the token
    else if (m == 0) stok[q][0] = slot == 0 ? ds[DS_TOK] : 0;
  }
  __syncthreads();
  for (int q = 0; q < TSV_PB; q++) {
    const int slot = s0 + q;
    const int tok = stok[q][slot >= 1 && slot < TSV_SLOTS ? (slot - 1) % SP_K : 0];
    for (int c = tid; c < SP_C / 4; c += 512) {
      const uint2 u = ((const uint2*)(codebook + (size_t)tok * SP_C))[c];
      cs[q][c] = make_float4(bf16r(SQRT_C * bflo(u.x)), bf16r(SQRT_C * bfhi(u.x)), bf16r(SQRT_C * bflo(u.y)), bf16r(SQRT_C * bfhi(u.y)));
    }
  }
  __syncthreads();
  const int w = tid >> 5, lane = tid & 31;
#pragma unroll 4
  for (int o = blockIdx.y * (2 * SP_R / TSV_RQ) + w; o < (blockIdx.y + 1) * (2 * SP_R / TSV_RQ); o += 16) {
    const uint4* wr = (const uint4*)(pv_w + (size_t)o * SP_C);
    float acc[TSV_PB] = {0.f, 0.f, 0.f, 0.f};
    for (int c8 = lane; c8 < SP_C / 8; c8 += 32) {
      float a[8];
      unpack8(wr[c8], a);
#pragma unroll
      for (int q = 0; q < TSV_PB; q++) {
        const float4 x0 = cs[q][c8 * 2], x1 = cs[q][c8 * 2 + 1];
        acc[q] = fmaf(x0.x, a[0], acc[q]); acc[q] = fmaf(x0.y, a[1], acc[q]); acc[q] = fmaf(x0.z, a[2], acc[q]); acc[q] = fmaf(x0.w, a[3], acc[q]);
        acc[q] = fmaf(x1.x, a[4], acc[q]); acc[q] = fmaf(x1.y, a[5], acc[q]); acc[q] = fmaf(x1.z, a[6], acc[q]); acc[q] = fmaf(x1.w, a[7], acc[q]);
      }
    }
#pragma unroll
    for (int q = 0; q < TSV_PB; q++) acc[q] = warp_sum(acc[q]);
    if (lane == 0) {
#pragma unroll
      for (int q = 0; q < TSV_PB; q++)
        if (s0 + q < TSV_SLOTS) tsv[(size_t)(s0 + q) * 2 * SP_R + o] = bf16r(acc[q]);
    }
  }
}

// Serial walk (reference propose.py serial_table greedy): predecessor = anchor; for each noise row i:
//   m = sigmoid(q_i + tvg + gate_b) * tvv,  s[k] = m . tg[i][k],  score[k] = base[i][k] + rms_i * s[k]
//   pick the first maximum k*, token = cand[i][k*] -> ds[TOK + 1 + i], the next predecessor is that candidate (tsv slot 1 + i * SP_K + k*).
// 1 block, SP_R threads (thread r owns channel r).
__global__ void __launch_bounds__(SP_R) k_sp_ref_chain(const float* __restrict__ base, const float* __restrict__ rms, const float* __restrict__ packed, const float* __restrict__ gate_b,
                                                       const float* __restrict__ tgp, const float* __restrict__ tsv, const int* __restrict__ cand_id, int* ds, int n) {
  __shared__ float part[SP_R / 32][SP_K];
  __shared__ int sh_slot;
  const int r = threadIdx.x, lane = r & 31, w = r >> 5;
  int slot = 0;
  // tg of row i (sum of the 4 h-slices, rounded to bf16) for this thread's channel; the next row's is loaded ahead of the serial argmax
  float tgn[SP_K];
#pragma unroll
  for (int k = 0; k < SP_K; k++) {
    float t = 0.f;
#pragma unroll
    for (int hs = 0; hs < TG_HS; hs++) t += tgp[((size_t)hs * (SMK_SP_DRAFTS * SP_K) + k) * SP_R + r];
    tgn[k] = bf16r(t);
  }
  for (int i = 0; i < n; i++) {
    const float tvg = tsv[(size_t)slot * 2 * SP_R + r], tvv = tsv[(size_t)slot * 2 * SP_R + SP_R + r];
    const float q = bf16r(packed[(size_t)(i + 1) * 3 * SP_R + r]);
    float tgc[SP_K];
#pragma unroll
    for (int k = 0; k < SP_K; k++) tgc[k] = tgn[k];
    if (i + 1 < n) {
#pragma unroll
      for (int k = 0; k < SP_K; k++) {
        const size_t j = (size_t)(i + 1) * SP_K + k;
        float t = 0.f;
#pragma unroll
        for (int hs = 0; hs < TG_HS; hs++) t += tgp[((size_t)hs * (SMK_SP_DRAFTS * SP_K) + j) * SP_R + r];
        tgn[k] = bf16r(t);
      }
    }
    const float m = (1.f / (1.f + expf(-(q + tvg + gate_b[r])))) * tvv;
    float s[SP_K];
#pragma unroll
    for (int k = 0; k < SP_K; k++) s[k] = warp_sum(m * tgc[k]);
    if (lane == 0) {
#pragma unroll
      for (int k = 0; k < SP_K; k++) part[w][k] = s[k];
    }
    __syncthreads();
    if (r < 32) {  // warp 0: lane k < SP_K sums candidate k's partials (same order as before), then first-maximum argmax over the lanes
      float sc = -INFINITY;
      int bk = SP_K;
      if (r < SP_K) {
        float tot = 0.f;
        for (int ww = 0; ww < SP_R / 32; ww++) tot += part[ww][r];
        sc = base[i * SP_K + r] + rms[i + 1] * tot;
        bk = r;
      }
#pragma unroll
      for (int off = 8; off >= 1; off >>= 1) {
        const float os = __shfl_down_sync(0xffffffffu, sc, off);
        const int ok = __shfl_down_sync(0xffffffffu, bk, off);
        if (os > sc || (os == sc && ok < bk)) { sc = os; bk = ok; }
      }
      if (r == 0) {
        ds[DS_TOK + 1 + i] = cand_id[i * SP_K + bk];
        sh_slot = 1 + i * SP_K + bk;
      }
    }
    __syncthreads();
    slot = sh_slot;
    __syncthreads();
  }
}

// greedy accept + bookkeeping
__global__ void k_sp_accept(int* ds, int* ss, int* ring, int* dd, int* log, int* hist) {
  int m = 0;
  while (m < SMK_SP_DRAFTS && ds[DS_TOK + m + 1] == ds[DS_OUT + m]) m++;
  const int P = ds[DS_POS], ci = ss[SS_NCYC];
  int* lg = log + (size_t)(ci % SMK_SP_LOG) * SMK_SP_LOGW;
  lg[0] = m + 1;
  for (int t = 0; t < 8; t++) { lg[1 + t] = ds[DS_TOK + t]; lg[9 + t] = ds[DS_OUT + t]; }
  lg[17] = P;
  for (int j = 0; j <= m; j++) hist[P + 1 + j] = ds[DS_OUT + j];
  spec_finish(ds, ss, ring, m, 8);
  dd[DD_NNEW] = m + 1;
  dd[DD_BASE] = P;
}
__global__ void k_sp_start(const int* ds, int* dd, int* hist) {
  dd[DD_NNEW] = 1;
  dd[DD_BASE] = ds[DS_POS] - 1;
  hist[ds[DS_POS]] = ds[DS_TOK];
}
__global__ void k_sp_set(int* dd, int n, int base) {
  dd[DD_NNEW] = n;
  dd[DD_BASE] = base;
}

// ---------------------------------------------------------------------------------------------- wiring
#define SPCK(x)                                                                                   \
  do {                                                                                              \
    cudaError_t e_ = (x);                                                                           \
    if (e_ != cudaSuccess) { fprintf(stderr, "dspine %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return e_; } \
  } while (0)

// out[t][n] (bf16) = W4A16 GEMV over the 8 rows (smk_w4r_gemv, one rounding to bf16).
static cudaError_t gw4(const void* w, const void* sc, const void* x, int N, int K, void* out, cudaStream_t st) {
  smk_epi ep = {};
  ep.kind = SMK_EPI_SEGS; ep.end[0] = ep.end[1] = ep.end[2] = N; ep.out[0] = out;
  return smk_w4r_gemv(w, sc, x, N, K, R8, &ep, st);
}

static cudaError_t sp_ctx(smk_dspine* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_sp_w& W = s->w;
  k_sp_gather_taps<<<dim3(R8, 5), NH, 0, st>>>((const uint4*)e->dbg, (uint4*)s->fcin);
  SPCK(cudaGetLastError());
  SPCK(gw4(W.fc_w, W.fc_s, s->fcin, H, FC_K, s->th_raw, st));
  k_sp_rms<<<R8, NH, 0, st>>>((const uint4*)s->th_raw, (const uint4*)W.hidden_norm, (uint4*)s->th);
  SPCK(gw4(W.kv_w, W.kv_s, s->th, KV_N, H, s->kvraw, st));
  SpCtxPtrs cp;
  for (int l = 0; l < NL; l++) cp.knorm[l] = (const uint16_t*)W.layers[l].k_norm;
  k_sp_ctx_kv<<<dim3(R8, NL * 16), 128, 0, st>>>(s->kvraw, cp, W.inv_freq, s->dd, s->kc, s->vc, s->ctxd);
  return cudaGetLastError();
}

static cudaError_t sp_noise(smk_dspine* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_sp_w& W = s->w;
  k_sp_embed_rms<<<R8, NH, 0, st>>>((const uint4*)e->w.embed, e->ds, W.mask_id, (const uint4*)W.layers[0].in_ln, (uint4*)s->h, (uint4*)s->xn);
  SPCK(cudaGetLastError());
  for (int l = 0; l < NL; l++) {
    const smk_sp_layer_w& L = W.layers[l];
    const size_t lay = (size_t)l * NKV * s->ctxd * HD;
    // (layer 0: embed + in_ln already done above)
    // qkv W4 GEMV (segmented output: q | k | v)
    smk_epi eq = {};
    eq.kind = SMK_EPI_SEGS; eq.end[0] = QD; eq.end[1] = QD + KVD; eq.end[2] = QD + 2 * KVD; eq.out[0] = s->qb; eq.out[1] = s->kb; eq.out[2] = s->vb;
    SPCK(smk_w4r_gemv(L.qkv_w, L.qkv_s, s->xn, QD + 2 * KVD, H, R8, &eq, st));
    // qk norm + RoPE, K/V cache write and causal attention
    k_sp_attn<<<NQ, SP_ATT_T, (R8 * HD + R8 * ATT_TILE) * sizeof(float), st>>>(s->qb, s->kb, s->vb, (const uint16_t*)L.q_norm, (const uint16_t*)L.k_norm, W.inv_freq, s->kc + lay, s->vc + lay, s->dd, s->ctxd, s->att);
    // o_proj
    SPCK(gw4(L.o_w, L.o_s, s->att, H, QD, s->oo, st));
    // residual h += oo, then post_ln
    k_sp_res_rms<<<R8, NH, 0, st>>>((uint4*)s->h, (const uint4*)s->oo, (const uint4*)L.post_ln, (uint4*)s->xn);
    // MLP: gate|up W4 GEMV with the SILU epilogue -> down
    smk_epi eg = {};
    eg.kind = SMK_EPI_SILU; eg.out[0] = s->act;
    SPCK(smk_w4r_gemv(L.gu_w, L.gu_s, s->xn, 2 * I, H, R8, &eg, st));
    SPCK(gw4(L.dn_w, L.dn_s, s->act, H, I, s->mlp, st));
    // residual h += mlp + code bridge norm (the last layer also keeps the pre-delta residual for the serial refine)
    const uint16_t* read_w = (l < NL - 1) ? (const uint16_t*)W.early_read + (size_t)l * SP_C * H : (const uint16_t*)W.read_w_last;
    k_sp_res_brnorm<<<R8, NH, 0, st>>>((uint4*)s->h, (const uint4*)s->mlp, (uint4*)s->br_normed, s->br_rms, l == NL - 1 ? (uint4*)s->raw_last : nullptr);
    SPCK(smk_bf16_gemv_f32(read_w, s->br_normed, SP_C, H, R8, s->br_proj, st));
    k_sp_br_gate<<<SP_R / BG_CH, 32 * BG_CH, 0, st>>>(s->br_proj, (const uint16_t*)W.codebook, e->ds, (const uint16_t*)W.gate_w, W.gate_b, s->br_packed, s->br_msg);
    SPCK(smk_bf16_gemv_f32(L.write_w, s->br_msg, H, SP_R, R8, s->br_delta, st));
    const uint4* nnw = (l + 1 < NL) ? (const uint4*)W.layers[l + 1].in_ln : (const uint4*)W.norm;
    uint4* nnout = (l + 1 < NL) ? (uint4*)s->xn : (uint4*)s->hid;
    // (raw_last written by res_brnorm above)
    k_sp_br_apply<<<R8, NH, 0, st>>>(s->h, s->br_delta, s->br_rms, nnw, nnout);
  }
  return cudaGetLastError();
}

// Head over noise rows 1..7: quantised head + exact rescore of the top-16, candidates sorted by id, serial_table greedy refine (reference: propose.py:191-216) -> ds[TOK + 1 ..].
static cudaError_t sp_head(smk_dspine* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_sp_w& W = s->w;
  const uint16_t* x = s->hid + H;
  const int T = SMK_SP_DRAFTS;
  SPCK(smk_headq_gemv(e->w.head_q_kind, e->w.head_q, e->w.head_qs, x, VOC, H, T, e->logits, st));
  SPCK(smk_headq_select_rescore(e->logits, e->w.lm_head, x, VOC, H, T, SP_K, s->tmp_tok, nullptr, e->ws_hq, st));
  // the tsv table only needs the candidate ids: parallel branch next to the refine prep / tg products
  SPCK(cudaEventRecord(e->ev_fork, st));
  SPCK(cudaStreamWaitEvent(e->side, e->ev_fork, 0));
  k_sp_ref_tsv<<<dim3((TSV_SLOTS + TSV_PB - 1) / TSV_PB, TSV_RQ), 512, 0, e->side>>>((const uint16_t*)W.codebook, (const uint16_t*)W.gate_w + (size_t)SP_R * SP_C, smk_headq_ws_cand(e->ws_hq), e->ds, s->ref_tsv);
  SPCK(cudaEventRecord(e->ev_join, e->side));
  k_sp_ref_prep<<<T * SP_K, NH, 0, st>>>((const uint16_t*)e->w.lm_head, (const uint16_t*)W.norm, s->raw_last, smk_headq_ws_cand(e->ws_hq), s->cand_id, s->ref_as, s->ref_base);
  k_sp_ref_tg<<<dim3(SP_R / 64, T * SP_K / 16, TG_HS), 256, 0, st>>>(s->ref_as, (const uint16_t*)W.layers[NL - 1].write_w, s->ref_tg);
  SPCK(cudaStreamWaitEvent(st, e->ev_join, 0));
  k_sp_ref_chain<<<1, SP_R, 0, st>>>(s->ref_base, s->br_rms, s->br_packed, W.gate_b, s->ref_tg, s->ref_tsv, s->cand_id, e->ds, T);
  return cudaGetLastError();
}

static cudaError_t sp_stage(smk_dspine* s, cudaStream_t st) {
  SPCK(sp_ctx(s, st));
  SPCK(sp_noise(s, st));
  return sp_head(s, st);
}

static cudaError_t sp_verify_accept(smk_dspine* s, cudaStream_t st) {
  smk_engine* e = s->e;
  SPCK(smk_run_step(e, R8, 1, 1, st, 0));
  k_sp_accept<<<1, 1, 0, st>>>(e->ds, s->ss, s->ring, s->dd, s->log, s->hist);
  return cudaGetLastError();
}

static void drop_graphs(smk_dspine* s) {
  for (cudaGraphExec_t* g : {&s->g_cycle, &s->g_verify, &s->g_stage, &s->g_ctx, &s->g_noise, &s->g_head, &s->g_start}) if (*g) { cudaGraphExecDestroy(*g); *g = nullptr; }
}
static void sync_epoch(smk_dspine* s) {
  if (s->epoch != s->e->head_epoch) { drop_graphs(s); s->epoch = s->e->head_epoch; }
}

extern "C" {

smk_dspine* smk_dspine_create(smk_engine* e, const smk_sp_w* w) {
  if (e->tmax < R8) return nullptr;
  smk_dspine* s = new smk_dspine();
  s->e = e;
  s->w = *w;
  s->ctxd = e->ctx_max + R8;
  const size_t b0 = e->bytes;
  auto B = [&](size_t n, bool z = false) { return dalloc(e, n, z); };
  s->dd = (int*)B(DD_N * sizeof(int), true);
  s->ss = (int*)B(SS_N * sizeof(int), true);
  s->ring = (int*)B(RING * sizeof(int), true);
  s->log = (int*)B((size_t)SMK_SP_LOG * SMK_SP_LOGW * sizeof(int), true);
  s->fcin = (uint16_t*)B((size_t)R8 * FC_K * 2);
  s->th_raw = (uint16_t*)B((size_t)R8 * H * 2);
  s->th = (uint16_t*)B((size_t)R8 * H * 2);
  s->kvraw = (uint16_t*)B((size_t)R8 * KV_N * 2);
  s->h = (uint16_t*)B((size_t)R8 * H * 2);
  s->xn = (uint16_t*)B((size_t)R8 * H * 2);
  s->qb = (uint16_t*)B((size_t)R8 * QD * 2);
  s->kb = (uint16_t*)B((size_t)R8 * KVD * 2);
  s->vb = (uint16_t*)B((size_t)R8 * KVD * 2);
  s->att = (uint16_t*)B((size_t)R8 * QD * 2);
  s->oo = (uint16_t*)B((size_t)R8 * H * 2);
  s->act = (uint16_t*)B((size_t)R8 * I * 2);
  s->mlp = (uint16_t*)B((size_t)R8 * H * 2);
  s->hid = (uint16_t*)B((size_t)R8 * H * 2, true);
  s->kc = (uint16_t*)B((size_t)NL * NKV * s->ctxd * HD * 2, true);
  s->vc = (uint16_t*)B((size_t)NL * NKV * s->ctxd * HD * 2, true);
  s->br_normed = (uint16_t*)B((size_t)R8 * H * 2);
  s->br_msg = (uint16_t*)B((size_t)R8 * SP_R * 2);
  s->br_rms = (float*)B(R8 * 4);
  s->br_proj = (float*)B((size_t)R8 * SP_C * 4);
  s->br_packed = (float*)B((size_t)R8 * 3 * SP_R * 4, true);
  s->br_delta = (float*)B((size_t)R8 * H * 4);
  s->cand_id = (int*)B((size_t)R8 * SP_K * 4, true);
  s->ref_base = (float*)B((size_t)R8 * SP_K * 4, true);
  s->tmp_tok = (int*)B(R8 * sizeof(int), true);
  s->hist = (int*)B((size_t)(e->ctx_max + 2 * R8) * sizeof(int), true);
  s->raw_last = (uint16_t*)B((size_t)R8 * H * 2);
  s->ref_as = (uint16_t*)B((size_t)SMK_SP_DRAFTS * SP_K * H * 2);
  s->ref_tg = (float*)B((size_t)TG_HS * SMK_SP_DRAFTS * SP_K * SP_R * 4);
  s->ref_tsv = (float*)B((size_t)TSV_SLOTS * 2 * SP_R * 4);
  cudaMallocHost((void**)&s->pinned_ss, SS_N * sizeof(int));
  cudaMallocHost((void**)&s->pinned_ring, RING * sizeof(int));
  if (cudaFuncSetAttribute(k_sp_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((R8 * HD + R8 * ATT_TILE) * sizeof(float))) != cudaSuccess) e->failed = true;
  s->bytes = e->bytes - b0;
  if (e->failed || cudaGetLastError() != cudaSuccess) { smk_dspine_destroy(s); return nullptr; }
  return s;
}

void smk_dspine_destroy(smk_dspine* s) {
  if (!s) return;
  cudaDeviceSynchronize();
  drop_graphs(s);
  if (s->pinned_ss) cudaFreeHost(s->pinned_ss);
  if (s->pinned_ring) cudaFreeHost(s->pinned_ring);
  delete s;
}

cudaError_t smk_dspine_reset(smk_dspine* s, cudaStream_t st) {
  const size_t kvb = (size_t)NL * NKV * s->ctxd * HD * 2;
  cudaMemsetAsync(s->dd, 0, DD_N * sizeof(int), st);
  cudaMemsetAsync(s->ss, 0, SS_N * sizeof(int), st);
  cudaMemsetAsync(s->ring, 0, RING * sizeof(int), st);
  cudaMemsetAsync(s->log, 0, (size_t)SMK_SP_LOG * SMK_SP_LOGW * sizeof(int), st);
  cudaMemsetAsync(s->hist, 0, (size_t)(s->e->ctx_max + 2 * R8) * sizeof(int), st);
  cudaMemsetAsync(s->kc, 0, kvb, st);
  cudaMemsetAsync(s->vc, 0, kvb, st);
  return cudaGetLastError();
}

cudaError_t smk_dspine_prefill_chunk(smk_dspine* s, int T, int pos0, cudaStream_t st) {
  if (T < 1 || T > R8 || pos0 + T > s->ctxd) return cudaErrorInvalidValue;
  k_sp_set<<<1, 1, 0, st>>>(s->dd, T, pos0);
  SPCK(cudaGetLastError());
  return sp_ctx(s, st);
}

cudaError_t smk_dspine_start(smk_dspine* s, cudaStream_t st) {
  sync_epoch(s);
  return smk_graph_launch(s->e, &s->g_start, [&](cudaStream_t c) {
    k_sp_start<<<1, 1, 0, c>>>(s->e->ds, s->dd, s->hist);
    SPCK(cudaGetLastError());
    return sp_stage(s, c);
  }, st);
}

cudaError_t smk_dspine_launch(smk_dspine* s, int which, int n, cudaStream_t st) {
  sync_epoch(s);
  smk_engine* e = s->e;
  for (int i = 0; i < n; i++) {
    switch (which) {
      case 0: SPCK(smk_graph_launch(e, &s->g_cycle, [&](cudaStream_t c) { SPCK(sp_verify_accept(s, c)); return sp_stage(s, c); }, st)); break;
      case 1: SPCK(smk_graph_launch(e, &s->g_verify, [&](cudaStream_t c) { return sp_verify_accept(s, c); }, st)); break;
      case 2: SPCK(smk_graph_launch(e, &s->g_stage, [&](cudaStream_t c) { return sp_stage(s, c); }, st)); break;
      case 3: SPCK(smk_graph_launch(e, &s->g_ctx, [&](cudaStream_t c) { return sp_ctx(s, c); }, st)); break;
      case 4: SPCK(smk_graph_launch(e, &s->g_noise, [&](cudaStream_t c) { return sp_noise(s, c); }, st)); break;
      case 5: SPCK(smk_graph_launch(e, &s->g_head, [&](cudaStream_t c) { return sp_head(s, c); }, st)); break;
      case 6: SPCK(smk_graph_launch(e, &s->g_noise, [&](cudaStream_t c) { k_sp_start<<<1, 1, 0, c>>>(e->ds, s->dd, s->hist); SPCK(cudaGetLastError()); return sp_noise(s, c); }, st)); break;
      default: return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t smk_dspine_run(smk_dspine* s, int n, cudaStream_t st, int* host_ss, int* host_ring) {
  SPCK(smk_dspine_launch(s, 0, n, st));
  SPCK(cudaMemcpyAsync(s->pinned_ss, s->ss, SS_N * sizeof(int), cudaMemcpyDeviceToHost, st));
  SPCK(cudaMemcpyAsync(s->pinned_ring, s->ring, RING * sizeof(int), cudaMemcpyDeviceToHost, st));
  SPCK(cudaStreamSynchronize(st));
  memcpy(host_ss, s->pinned_ss, SS_N * sizeof(int));
  memcpy(host_ring, s->pinned_ring, RING * sizeof(int));
  return cudaSuccess;
}

cudaError_t smk_dspine_read(smk_dspine* s, int what, void* host, size_t bytes) {
  const void* src = nullptr;
  size_t cap = 0;
  switch (what) {
    case 0: src = s->ss; cap = SS_N * sizeof(int); break;
    case 1: src = s->hid; cap = (size_t)R8 * H * 2; break;
    case 2: src = s->log; cap = (size_t)SMK_SP_LOG * SMK_SP_LOGW * sizeof(int); break;
    case 3: src = s->h; cap = (size_t)R8 * H * 2; break;
    case 4: src = s->kc; cap = (size_t)NL * NKV * s->ctxd * HD * 2; break;
    case 5: src = s->vc; cap = (size_t)NL * NKV * s->ctxd * HD * 2; break;
    case 6: src = s->th; cap = (size_t)R8 * H * 2; break;
    case 7: src = s->att; cap = (size_t)R8 * QD * 2; break;
    case 9: src = s->oo; cap = (size_t)R8 * H * 2; break;
    case 10: src = s->mlp; cap = (size_t)R8 * H * 2; break;
    case 11: src = s->br_normed; cap = (size_t)R8 * H * 2; break;
    case 13: src = s->br_msg; cap = (size_t)R8 * SP_R * 2; break;
    case 14: src = s->br_delta; cap = (size_t)R8 * H * 4; break;
  }
  if (!src || bytes > cap) return cudaErrorInvalidValue;
  return cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost);
}

size_t smk_dspine_bytes(smk_dspine* s) { return s->bytes; }
int smk_dspine_ctxd(smk_dspine* s) { return s->ctxd; }
void* smk_dspine_kv_ptr(smk_dspine* s, int which) { return which == 0 ? (void*)s->kc : which == 1 ? (void*)s->vc : nullptr; }

}  // extern "C"
