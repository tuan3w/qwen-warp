// swift-mk DFlash2 block drafter on the device (see include/dflash.h).
#include <math.h>

#include "dflash.h"
#include "head_q.h"
#include "mtp_sample.h"

using namespace smk;

extern "C" cudaError_t smk_bf16_gemv_f32_cfg(const void* w, const void* x, int N, int K, int T, float* out, int cfg, cudaStream_t st);  // gemv.cu

namespace {
constexpr int NH = H / 8;           // uint4 chunks per hidden row = 640
constexpr int NL = SMK_DF_NL;
constexpr int R8 = SMK_DF_ROWS;     // rows per block
constexpr int NQ = 32, NKV = 8, HD = 128;
constexpr int QD = NQ * HD, KVD = NKV * HD;   // 4096, 1024
constexpr int QKV_N = QD + 2 * KVD;           // 6144
constexpr int KP_N = 1280;                    // conv kernel projection rows: [prepare | finish] x 2 offsets x 320 groups
constexpr int NG = H / 16;                    // conv groups = 320
constexpr int FC_K = 5 * H;                   // 25600
constexpr int KV_N = NL * 2 * KVD;            // 10240
constexpr int ATT_SMAX = 2056;                // window 2048 + 8 noise rows (keys per query <= 2055)
constexpr int SEL_K = 16, SEL_RMAX = SMK_HEADQ_RMAX, SEL_D = 256;
constexpr int DD_NNEW = 0, DD_BASE = 1, DD_N = 4;
}  // namespace

struct smk_dflash {
  smk_engine* e;
  smk_df_w w;
  int ctxd;                      // drafter KV rows per head = ctx_max + 8
  int* dd;                       // [DD_N] nnew, base
  int* ss;                       // spec_finish state
  int* ring;
  int* log;
  uint16_t *fcin, *th_raw, *th, *kvraw, *h, *xn, *xc, *kfin, *qb, *kb, *vb, *qn, *att, *oo, *act, *mlp, *hid, *kc, *vc;
  float *dyn, *hp, *sel_un;
  int *tmp_tok, *sel_id;
  int *pinned_ss, *pinned_ring;
  cudaGraphExec_t g_cycle = nullptr, g_verify = nullptr, g_stage = nullptr, g_ctx = nullptr, g_noise = nullptr, g_head = nullptr, g_start = nullptr;
  int kp_cfg = 1;                // smk_bf16_gemv_f32_cfg configuration of the conv projections (env SMK_DF_KP_CFG)
  unsigned epoch = 0;
  size_t bytes = 0;
};

// ---------------------------------------------------------------------------------------------- kernels
// fcin[t][s * H ..] = tap slot s of row t.  grid (8, 5), NH threads.
__global__ void __launch_bounds__(NH) k_gather_taps(const uint4* __restrict__ dbg, uint4* __restrict__ out) {
  const int t = blockIdx.x, s = blockIdx.y, c = threadIdx.x;
  const int slot = 6 + 14 * s;  // 6, 20, 34, 48, 62: residual stream after target layers 5, 19, 33, 47, 61 (debug tap slot = layer + 1)
  out[(size_t)t * (FC_K / 8) + s * NH + c] = dbg[((size_t)slot * TM + t) * NH + c];
}

// out[t] = bf16(w * bf16(x[t] * rsqrt(mean(x[t]^2) + eps)))  (plain RMSNorm, HF Qwen3 / the reference's `rms`).  grid T, NH threads.
__global__ void __launch_bounds__(NH) k_rms(const uint4* __restrict__ x, const uint4* __restrict__ w, uint4* __restrict__ out) {
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

// noise rows: row 0 = embed[anchor = ds[TOK]], rows 1..7 = embed[mask_id].  grid 8, NH threads.
__global__ void __launch_bounds__(NH) k_noise_embed(const uint4* __restrict__ embed, const int* __restrict__ ds, int mask_id, uint4* __restrict__ h) {
  const int t = blockIdx.x, c = threadIdx.x;
  const int tok = t == 0 ? ds[DS_TOK] : mask_id;
  h[(size_t)t * NH + c] = embed[(size_t)tok * NH + c];
}

// Grouped dynamic causal conv (k = 2, group 16), the prepare half: x [8][5120] (bf16), dyn fp32 [8][1280] (kernel projection of x; rounded to bf16 like the reference's bf16 linear).
// out[t][c] = sum_off  bf16-chained  (base[off][c] * v_off + dyn_prep[off][g] * v_off),  v_0 = x[t], v_1 = x[t-1] (0 for t = 0).  kfin[t][off * 320 + g] = bf16(dyn finish).
// grid (20, 8), 256 threads (one channel each).
__global__ void __launch_bounds__(256) k_conv_prep(const uint16_t* __restrict__ x, const float* __restrict__ dyn, const uint16_t* __restrict__ base, uint16_t* __restrict__ out,
                                                   uint16_t* __restrict__ kfin) {
  const int t = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, g = c >> 4;
  const float v0 = bf2f(x[t * H + c]), v1 = t > 0 ? bf2f(x[(t - 1) * H + c]) : 0.f;
  const float d0 = bf16r(dyn[t * KP_N + g]), d1 = bf16r(dyn[t * KP_N + NG + g]);
  const float k0 = bf2f(base[c]), k1 = bf2f(base[H + c]);
  float o = bf16r(k0 * v0);
  o = bf16r(fmaf(d0, v0, o));
  o = bf16r(o + bf16r(k1 * v1));
  o = bf16r(fmaf(d1, v1, o));
  out[t * H + c] = f2bf(o);
  if (c < 2 * NG) kfin[t * 2 * NG + c] = f2bf(dyn[t * KP_N + 2 * NG + c]);
}

// finish half + residual: h[t][c] = bf16(h[t][c] + conv(y)[t][c]) with the finish base kernel (base + 2 * H) and dynamic kernel kfin.  grid (20, 8).
__global__ void __launch_bounds__(256) k_conv_fin(const uint16_t* __restrict__ y, const uint16_t* __restrict__ kfin, const uint16_t* __restrict__ base, uint16_t* __restrict__ h) {
  const int t = blockIdx.y, c = blockIdx.x * 256 + threadIdx.x, g = c >> 4;
  const float v0 = bf2f(y[t * H + c]), v1 = t > 0 ? bf2f(y[(t - 1) * H + c]) : 0.f;
  const float d0 = bf2f(kfin[t * 2 * NG + g]), d1 = bf2f(kfin[t * 2 * NG + NG + g]);
  const float k0 = bf2f(base[2 * H + c]), k1 = bf2f(base[3 * H + c]);
  float o = bf16r(k0 * v0);
  o = bf16r(fmaf(d0, v0, o));
  o = bf16r(o + bf16r(k1 * v1));
  o = bf16r(fmaf(d1, v1, o));
  h[t * H + c] = f2bf(bf2f(h[t * H + c]) + o);
}

// per-head RMSNorm (plain weight, eps 1e-6) + RoPE (rotate-half, bf16-rounded cos / sin table) of one 128-element head by 128 threads (thread = dim d).
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

// noise rows: q / k / v of the layer.  q -> qout [8][4096] (norm + rope), k -> cache K at position start + t (norm + rope), v -> cache V.  grid (8, 48), 128 threads.
__global__ void __launch_bounds__(128) k_qk_prep(const uint16_t* __restrict__ qb, const uint16_t* __restrict__ kb, const uint16_t* __restrict__ vb, const uint16_t* __restrict__ qn,
                                                 const uint16_t* __restrict__ kn, const float* __restrict__ invf, const int* __restrict__ dd, uint16_t* __restrict__ qout,
                                                 uint16_t* __restrict__ kc, uint16_t* __restrict__ vc, int ctxd) {
  __shared__ float sh[128], red[4];
  const int t = blockIdx.x, hh = blockIdx.y, d = threadIdx.x;
  const int start = dd[DD_BASE] + dd[DD_NNEW];
  const int pos = start + t;
  if (hh < NQ) {
    const float x = bf2f(qb[t * QD + hh * HD + d]);
    qout[t * QD + hh * HD + d] = f2bf(head_norm_rope(x, bf2f(qn[d]), pos, invf, sh, red, d));
  } else if (hh < NQ + NKV) {
    const int kh = hh - NQ;
    const float x = bf2f(kb[t * KVD + kh * HD + d]);
    const float o = head_norm_rope(x, bf2f(kn[d]), pos, invf, sh, red, d);
    if (pos < ctxd) kc[((size_t)kh * ctxd + pos) * HD + d] = f2bf(o);
  } else {
    const int vh = hh - NQ - NKV;
    if (pos < ctxd) vc[((size_t)vh * ctxd + pos) * HD + d] = vb[t * KVD + vh * HD + d];
  }
}

struct CtxPtrs {
  const uint16_t* knorm[NL];
};

// context rows: K / V of all 5 layers from the fused GEMV output kvraw [8][10240] -> the caches at positions base + t (rows t < nnew only).  grid (8, 80), 128 threads.
__global__ void __launch_bounds__(128) k_ctx_kv(const uint16_t* __restrict__ kvraw, CtxPtrs cp, const float* __restrict__ invf, const int* __restrict__ dd, uint16_t* __restrict__ kc,
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

// Attention of the 8 noise rows over the cache rows [max(0, start - 2047), start + 8) (committed context ++ the noise rows' own K/V, written at start..start+7): non-causal,
// |q_pos - k_pos| < 2048.  One block per query head (4 heads share a KV head), 8 query rows; fp32 scores / softmax in smem.
// QK: a key per thread; PV: 8 key groups x 64 dim pairs (every thread keeps 8 rows x 2 dims, independent V loads in flight), reduced through smem.  grid 32, 512 threads, dynamic smem.
constexpr int ATT_T = 512;
__global__ void __launch_bounds__(ATT_T) k_attn(const uint16_t* __restrict__ qn, const uint16_t* __restrict__ kc, const uint16_t* __restrict__ vc, const int* __restrict__ dd, int ctxd,
                                                uint16_t* __restrict__ out) {
  extern __shared__ float sm[];
  float* sq = sm;               // [8][128]
  float* sc = sm + 8 * HD;      // [8][ATT_SMAX]
  __shared__ float rsum[8];
  const int h = blockIdx.x, kvh = h >> 2, tid = threadIdx.x;
  const int start = dd[DD_BASE] + dd[DD_NNEW];
  const int lo0 = max(0, start - 2047), nk = start + R8 - lo0;
  for (int i = tid; i < 8 * HD; i += ATT_T) sq[i] = bf2f(qn[(i >> 7) * QD + h * HD + (i & 127)]);
  __syncthreads();
  const uint16_t* kbase = kc + (size_t)kvh * ctxd * HD;
  const float4* sq4 = (const float4*)sq;
  for (int j = tid; j < nk; j += ATT_T) {
    const int kpos = lo0 + j;
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
    for (int i = 0; i < 8; i++) sc[i * ATT_SMAX + j] = (start + i - kpos) < 2048 ? acc[i] * 0.08838834764831845f : -INFINITY;
  }
  __syncthreads();
  if (tid < 256) {
    const int w = tid >> 5, lane = tid & 31;  // warp = row
    float m = -INFINITY;
    for (int j = lane; j < nk; j += 32) m = fmaxf(m, sc[w * ATT_SMAX + j]);
    m = warp_max(m);
    float s = 0.f;
    for (int j = lane; j < nk; j += 32) {
      const float p = expf(sc[w * ATT_SMAX + j] - m);
      sc[w * ATT_SMAX + j] = p;
      s += p;
    }
    s = warp_sum(s);
    if (lane == 0) rsum[w] = 1.f / s;
  }
  __syncthreads();
  const int g = tid >> 6, dp = tid & 63;
  const uint32_t* vbase = (const uint32_t*)(vc + (size_t)kvh * ctxd * HD) + dp;
  float a[8][2];
#pragma unroll
  for (int i = 0; i < 8; i++) a[i][0] = a[i][1] = 0.f;
#pragma unroll 8
  for (int j = g; j < nk; j += 8) {
    const uint32_t v = vbase[(size_t)(lo0 + j) * (HD / 2)];
    const float v0 = bflo(v), v1 = bfhi(v);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const float p = sc[i * ATT_SMAX + j];
      a[i][0] = fmaf(p, v0, a[i][0]);
      a[i][1] = fmaf(p, v1, a[i][1]);
    }
  }
  __syncthreads();  // sc is dead: reuse it for the group reduction [8 groups][8 rows][128]
#pragma unroll
  for (int i = 0; i < 8; i++) {
    sc[(g * 8 + i) * HD + 2 * dp] = a[i][0];
    sc[(g * 8 + i) * HD + 2 * dp + 1] = a[i][1];
  }
  __syncthreads();
#pragma unroll
  for (int e = 0; e < 2; e++) {
    const int idx = tid + ATT_T * e, i = idx >> 7, d = idx & 127;
    float s = 0.f;
#pragma unroll
    for (int gg = 0; gg < 8; gg++) s += sc[(gg * 8 + i) * HD + d];
    out[i * QD + h * HD + d] = f2bf(s * rsum[i]);
  }
}

// Candidate selector, greedy path (reference: CandidateSelector.select).  Rows i = 0..nrow-1 = noise rows 1..nrow.
//   A: per row the top-16 of the exact bf16-rounded logits of the rescored candidates, in the order torch.topk(sorted=False) returns them (values above the 16th in ascending id, then the
//      ties of the 16th value in ascending id);
//   B: pred = anchor; for i: score_k = bf16(unary_k + bf16(sum_r bf16(pred_cb[pred][r] * hp[i][r]) * succ_cb[cand_k][r])), pick the first maximum, pred = that token -> ds[TOK + 1 + i].
// 1 block, 256 threads.
__global__ void __launch_bounds__(256) k_select(const int* __restrict__ cand, const float* __restrict__ exact, const float* __restrict__ hp, const uint16_t* __restrict__ pred_cb,
                                                const uint16_t* __restrict__ succ_cb, int* ds, int* __restrict__ sel_id_out, float* __restrict__ sel_un_out, int R, int nrow) {
  __shared__ int cid[8][SEL_RMAX];
  __shared__ float cval[8][SEL_RMAX];
  __shared__ int rank_s[8][SEL_RMAX];
  __shared__ float thr_s[8];
  __shared__ int selId[8][SEL_K];
  __shared__ float selUn[8][SEL_K];
  __shared__ float vec[SEL_D], sc[SEL_K];
  __shared__ int pred_s;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  for (int i = tid; i < nrow * SEL_RMAX; i += 256) {
    const int r = i / SEL_RMAX, c = i % SEL_RMAX;
    const int id = c < R ? cand[r * SEL_RMAX + c] : -1;
    cid[r][c] = id;
    cval[r][c] = id >= 0 ? bf16r(exact[r * SEL_RMAX + c]) : -INFINITY;
  }
  __syncthreads();
  if (w < nrow) {
    for (int q = 0; q < 2; q++) {
      const int c = lane + 32 * q;
      int rank = 0;
      if (cid[w][c] >= 0) {
        const float v = cval[w][c];
        const int id = cid[w][c];
        for (int j = 0; j < SEL_RMAX; j++) {
          if (cid[w][j] < 0) continue;
          const float vj = cval[w][j];
          rank += (vj > v) || (vj == v && cid[w][j] < id);
        }
      } else {
        rank = 1 << 20;
      }
      rank_s[w][c] = rank;
      if (rank == SEL_K - 1) thr_s[w] = cval[w][c];
    }
    __syncwarp();
    const float thr = thr_s[w];
    for (int q = 0; q < 2; q++) {
      const int c = lane + 32 * q;
      if (rank_s[w][c] < SEL_K) {
        const int id = cid[w][c];
        const int gi = cval[w][c] > thr ? 0 : 1;
        int pos = 0;
        for (int j = 0; j < SEL_RMAX; j++) {
          if (rank_s[w][j] >= SEL_K) continue;
          const int gj = cval[w][j] > thr ? 0 : 1;
          pos += (gj < gi) || (gj == gi && cid[w][j] < id);
        }
        selId[w][pos] = id;
        selUn[w][pos] = cval[w][c];
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < nrow * SEL_K; i += 256) {
    sel_id_out[i] = selId[i / SEL_K][i % SEL_K];
    sel_un_out[i] = selUn[i / SEL_K][i % SEL_K];
  }
  int pred = ds[DS_TOK];
  for (int i = 0; i < nrow; i++) {
    vec[tid] = bf16r(bf2f(pred_cb[(size_t)pred * SEL_D + tid]) * bf16r(hp[i * SEL_D + tid]));
    __syncthreads();
    for (int k = w; k < SEL_K; k += 8) {
      const uint4 u = *(const uint4*)(succ_cb + (size_t)selId[i][k] * SEL_D + lane * 8);
      const float* v8 = vec + lane * 8;
      float a = v8[0] * bflo(u.x);
      a = fmaf(v8[1], bfhi(u.x), a); a = fmaf(v8[2], bflo(u.y), a); a = fmaf(v8[3], bfhi(u.y), a);
      a = fmaf(v8[4], bflo(u.z), a); a = fmaf(v8[5], bfhi(u.z), a); a = fmaf(v8[6], bflo(u.w), a); a = fmaf(v8[7], bfhi(u.w), a);
      a = warp_sum(a);
      if (lane == 0) sc[k] = bf16r(selUn[i][k] + bf16r(a));
    }
    __syncthreads();
    if (tid == 0) {
      int best = 0;
      for (int k = 1; k < SEL_K; k++) if (sc[k] > sc[best]) best = k;
      pred_s = selId[i][best];
      ds[DS_TOK + 1 + i] = pred_s;
    }
    __syncthreads();
    pred = pred_s;
  }
}

// greedy accept of a T = 8 verify (7 drafts) + bookkeeping + the drafter's next ctx rows (a rows at positions P ..) + the per-cycle log
__global__ void k_df_accept(int* ds, int* ss, int* ring, int* dd, int* log) {
  int m = 0;
  while (m < SMK_DF_DRAFTS && ds[DS_TOK + m + 1] == ds[DS_OUT + m]) m++;
  const int P = ds[DS_POS], ci = ss[SS_NCYC];
  int* lg = log + (size_t)(ci % SMK_DF_LOG) * SMK_DF_LOGW;
  lg[0] = m + 1;
  for (int t = 0; t < 8; t++) { lg[1 + t] = ds[DS_TOK + t]; lg[9 + t] = ds[DS_OUT + t]; }
  lg[17] = P;
  spec_finish(ds, ss, ring, m, 8);
  dd[DD_NNEW] = m + 1;
  dd[DD_BASE] = P;
}
__global__ void k_df_start(const int* ds, int* dd) {
  dd[DD_NNEW] = 1;
  dd[DD_BASE] = ds[DS_POS] - 1;
}
__global__ void k_df_set(int* dd, int n, int base) {
  dd[DD_NNEW] = n;
  dd[DD_BASE] = base;
}

// ---------------------------------------------------------------------------------------------- wiring
#define DCK(x)                                                                                    \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) { fprintf(stderr, "dflash %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return e_; } \
  } while (0)

static cudaError_t gw4(const void* w, const void* s, const void* x, int N, int K, void* out, cudaStream_t st) {
  smk_epi ep = {};
  ep.kind = SMK_EPI_SEGS; ep.end[0] = ep.end[1] = ep.end[2] = N; ep.out[0] = out;
  return smk_w4r_gemv(w, s, x, N, K, R8, &ep, st);
}

// (1) ctx append: taps of the new rows -> fc -> hidden_norm -> K / V of the 5 layers -> cache
static cudaError_t df_ctx(smk_dflash* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_df_w& W = s->w;
  k_gather_taps<<<dim3(R8, 5), NH, 0, st>>>((const uint4*)e->dbg, (uint4*)s->fcin);
  DCK(cudaGetLastError());
  DCK(gw4(W.fc_w, W.fc_s, s->fcin, H, FC_K, s->th_raw, st));
  k_rms<<<R8, NH, 0, st>>>((const uint4*)s->th_raw, (const uint4*)W.hidden_norm, (uint4*)s->th);
  DCK(gw4(W.kv_w, W.kv_s, s->th, KV_N, H, s->kvraw, st));
  CtxPtrs cp;
  for (int l = 0; l < NL; l++) cp.knorm[l] = (const uint16_t*)W.layers[l].k_norm;
  k_ctx_kv<<<dim3(R8, NL * 16), 128, 0, st>>>(s->kvraw, cp, W.inv_freq, s->dd, s->kc, s->vc, s->ctxd);
  return cudaGetLastError();
}

// (2) the noise block: 5 layers + final norm -> hid [8][5120]
static cudaError_t df_noise(smk_dflash* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_df_w& W = s->w;
  k_noise_embed<<<R8, NH, 0, st>>>((const uint4*)e->w.embed, e->ds, W.mask_id, (uint4*)s->h);
  DCK(cudaGetLastError());
  for (int l = 0; l < NL; l++) {
    const smk_df_layer_w& L = W.layers[l];
    const size_t lay = (size_t)l * NKV * s->ctxd * HD;
    k_rms<<<R8, NH, 0, st>>>((const uint4*)s->h, (const uint4*)L.in_ln, (uint4*)s->xn);
    DCK(smk_bf16_gemv_f32_cfg(L.aconv_kp, s->xn, KP_N, H, R8, s->dyn, s->kp_cfg, st));
    k_conv_prep<<<dim3(H / 256, R8), 256, 0, st>>>(s->xn, s->dyn, (const uint16_t*)L.aconv_base, s->xc, s->kfin);
    smk_epi eq = {};
    eq.kind = SMK_EPI_SEGS; eq.end[0] = QD; eq.end[1] = QD + KVD; eq.end[2] = QKV_N; eq.out[0] = s->qb; eq.out[1] = s->kb; eq.out[2] = s->vb;
    DCK(smk_w4r_gemv(L.qkv_w, L.qkv_s, s->xc, QKV_N, H, R8, &eq, st));
    k_qk_prep<<<dim3(R8, NQ + 2 * NKV), 128, 0, st>>>(s->qb, s->kb, s->vb, (const uint16_t*)L.q_norm, (const uint16_t*)L.k_norm, W.inv_freq, s->dd, s->qn, s->kc + lay, s->vc + lay, s->ctxd);
    k_attn<<<NQ, ATT_T, (R8 * HD + R8 * ATT_SMAX) * sizeof(float), st>>>(s->qn, s->kc + lay, s->vc + lay, s->dd, s->ctxd, s->att);
    DCK(gw4(L.o_w, L.o_s, s->att, H, QD, s->oo, st));
    k_conv_fin<<<dim3(H / 256, R8), 256, 0, st>>>(s->oo, s->kfin, (const uint16_t*)L.aconv_base, s->h);
    k_rms<<<R8, NH, 0, st>>>((const uint4*)s->h, (const uint4*)L.post_ln, (uint4*)s->xn);
    DCK(smk_bf16_gemv_f32_cfg(L.mconv_kp, s->xn, KP_N, H, R8, s->dyn, s->kp_cfg, st));
    k_conv_prep<<<dim3(H / 256, R8), 256, 0, st>>>(s->xn, s->dyn, (const uint16_t*)L.mconv_base, s->xc, s->kfin);
    smk_epi eg = {};
    eg.kind = SMK_EPI_SILU; eg.out[0] = s->act;
    DCK(smk_w4r_gemv(L.gu_w, L.gu_s, s->xc, 2 * I, H, R8, &eg, st));
    DCK(gw4(L.dn_w, L.dn_s, s->act, H, I, s->mlp, st));
    k_conv_fin<<<dim3(H / 256, R8), 256, 0, st>>>(s->mlp, s->kfin, (const uint16_t*)L.mconv_base, s->h);
  }
  k_rms<<<R8, NH, 0, st>>>((const uint4*)s->h, (const uint4*)W.norm, (uint4*)s->hid);
  return cudaGetLastError();
}

// (3) head over rows 1..7 + selector -> ds[TOK + 1 ..]
static cudaError_t df_head(smk_dflash* s, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_df_w& W = s->w;
  const uint16_t* x = s->hid + H;
  const int T = SMK_DF_DRAFTS;
  DCK(smk_headq_gemv(e->w.head_q_kind, e->w.head_q, e->w.head_qs, x, VOC, H, T, e->logits, st));
  DCK(smk_headq_select_rescore(e->logits, e->w.lm_head, x, VOC, H, T, W.rescore_r, s->tmp_tok, nullptr, e->ws_hq, st));
  DCK(smk_bf16_gemv_f32(W.sel_hp, x, SEL_D, H, T, s->hp, st));
  k_select<<<1, 256, 0, st>>>(smk_headq_ws_cand(e->ws_hq), smk_headq_ws_exact(e->ws_hq), s->hp, (const uint16_t*)W.pred_cb, (const uint16_t*)W.succ_cb, e->ds, s->sel_id, s->sel_un, W.rescore_r, T);
  return cudaGetLastError();
}

static cudaError_t df_stage(smk_dflash* s, cudaStream_t st) {
  DCK(df_ctx(s, st));
  DCK(df_noise(s, st));
  return df_head(s, st);
}

static cudaError_t df_verify_accept(smk_dflash* s, cudaStream_t st) {
  smk_engine* e = s->e;
  DCK(smk_run_step(e, R8, 1, 1, st, 0));
  k_df_accept<<<1, 1, 0, st>>>(e->ds, s->ss, s->ring, s->dd, s->log);
  return cudaGetLastError();
}

static void drop_graphs(smk_dflash* s) {
  for (cudaGraphExec_t* g : {&s->g_cycle, &s->g_verify, &s->g_stage, &s->g_ctx, &s->g_noise, &s->g_head, &s->g_start}) if (*g) { cudaGraphExecDestroy(*g); *g = nullptr; }
}
static void sync_epoch(smk_dflash* s) {
  if (s->epoch != s->e->head_epoch) { drop_graphs(s); s->epoch = s->e->head_epoch; }
}

extern "C" {

smk_dflash* smk_dflash_create(smk_engine* e, const smk_df_w* w) {
  if (e->tmax < R8 || e->w.head_q_kind != SMK_HEADQ_I4R || w->rescore_r < SEL_K || w->rescore_r > SEL_RMAX) return nullptr;
  smk_dflash* s = new smk_dflash();
  s->e = e;
  s->w = *w;
  s->ctxd = e->ctx_max + R8;
  if (const char* kc_env = getenv("SMK_DF_KP_CFG")) s->kp_cfg = atoi(kc_env);
  const size_t b0 = e->bytes;
  auto B = [&](size_t n, bool z = false) { return dalloc(e, n, z); };
  s->dd = (int*)B(DD_N * sizeof(int), true);
  s->ss = (int*)B(SS_N * sizeof(int), true);
  s->ring = (int*)B(RING * sizeof(int), true);
  s->log = (int*)B((size_t)SMK_DF_LOG * SMK_DF_LOGW * sizeof(int), true);
  s->fcin = (uint16_t*)B((size_t)R8 * FC_K * 2);
  s->th_raw = (uint16_t*)B((size_t)R8 * H * 2);
  s->th = (uint16_t*)B((size_t)R8 * H * 2);
  s->kvraw = (uint16_t*)B((size_t)R8 * KV_N * 2);
  s->h = (uint16_t*)B((size_t)R8 * H * 2);
  s->xn = (uint16_t*)B((size_t)R8 * H * 2);
  s->xc = (uint16_t*)B((size_t)R8 * H * 2);
  s->kfin = (uint16_t*)B((size_t)R8 * 2 * NG * 2);
  s->qb = (uint16_t*)B((size_t)R8 * QD * 2);
  s->kb = (uint16_t*)B((size_t)R8 * KVD * 2);
  s->vb = (uint16_t*)B((size_t)R8 * KVD * 2);
  s->qn = (uint16_t*)B((size_t)R8 * QD * 2);
  s->att = (uint16_t*)B((size_t)R8 * QD * 2);
  s->oo = (uint16_t*)B((size_t)R8 * H * 2);
  s->act = (uint16_t*)B((size_t)R8 * I * 2);
  s->mlp = (uint16_t*)B((size_t)R8 * H * 2);
  s->hid = (uint16_t*)B((size_t)R8 * H * 2, true);
  s->kc = (uint16_t*)B((size_t)NL * NKV * s->ctxd * HD * 2, true);
  s->vc = (uint16_t*)B((size_t)NL * NKV * s->ctxd * HD * 2, true);
  s->dyn = (float*)B((size_t)R8 * KP_N * 4);
  s->hp = (float*)B((size_t)R8 * SEL_D * 4, true);
  s->sel_un = (float*)B((size_t)R8 * SEL_K * 4, true);
  s->sel_id = (int*)B((size_t)R8 * SEL_K * 4, true);
  s->tmp_tok = (int*)B(R8 * sizeof(int), true);
  cudaMallocHost((void**)&s->pinned_ss, SS_N * sizeof(int));
  cudaMallocHost((void**)&s->pinned_ring, RING * sizeof(int));
  if (cudaFuncSetAttribute(k_attn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)((R8 * HD + R8 * ATT_SMAX) * sizeof(float))) != cudaSuccess) e->failed = true;
  s->bytes = e->bytes - b0;
  if (e->failed || cudaGetLastError() != cudaSuccess) { smk_dflash_destroy(s); return nullptr; }
  return s;
}

void smk_dflash_destroy(smk_dflash* s) {
  if (!s) return;
  cudaDeviceSynchronize();
  drop_graphs(s);
  if (s->pinned_ss) cudaFreeHost(s->pinned_ss);
  if (s->pinned_ring) cudaFreeHost(s->pinned_ring);
  delete s;  // device buffers belong to the engine's allocation list
}

cudaError_t smk_dflash_reset(smk_dflash* s, cudaStream_t st) {
  const size_t kvb = (size_t)NL * NKV * s->ctxd * HD * 2;
  cudaMemsetAsync(s->dd, 0, DD_N * sizeof(int), st);
  cudaMemsetAsync(s->ss, 0, SS_N * sizeof(int), st);
  cudaMemsetAsync(s->ring, 0, RING * sizeof(int), st);
  cudaMemsetAsync(s->log, 0, (size_t)SMK_DF_LOG * SMK_DF_LOGW * sizeof(int), st);
  cudaMemsetAsync(s->kc, 0, kvb, st);
  cudaMemsetAsync(s->vc, 0, kvb, st);
  return cudaGetLastError();
}

cudaError_t smk_dflash_prefill_chunk(smk_dflash* s, int T, int pos0, cudaStream_t st) {
  if (T < 1 || T > R8 || pos0 + T > s->ctxd) return cudaErrorInvalidValue;
  k_df_set<<<1, 1, 0, st>>>(s->dd, T, pos0);
  DCK(cudaGetLastError());
  return df_ctx(s, st);
}

cudaError_t smk_dflash_start(smk_dflash* s, cudaStream_t st) {
  sync_epoch(s);
  return smk_graph_launch(s->e, &s->g_start, [&](cudaStream_t c) {
    k_df_start<<<1, 1, 0, c>>>(s->e->ds, s->dd);
    DCK(cudaGetLastError());
    return df_stage(s, c);
  }, st);
}

cudaError_t smk_dflash_launch(smk_dflash* s, int which, int n, cudaStream_t st) {
  sync_epoch(s);
  smk_engine* e = s->e;
  for (int i = 0; i < n; i++) {
    switch (which) {
      case 0: DCK(smk_graph_launch(e, &s->g_cycle, [&](cudaStream_t c) { DCK(df_verify_accept(s, c)); return df_stage(s, c); }, st)); break;
      case 1: DCK(smk_graph_launch(e, &s->g_verify, [&](cudaStream_t c) { return df_verify_accept(s, c); }, st)); break;
      case 2: DCK(smk_graph_launch(e, &s->g_stage, [&](cudaStream_t c) { return df_stage(s, c); }, st)); break;
      case 3: DCK(smk_graph_launch(e, &s->g_ctx, [&](cudaStream_t c) { return df_ctx(s, c); }, st)); break;
      case 4: DCK(smk_graph_launch(e, &s->g_noise, [&](cudaStream_t c) { return df_noise(s, c); }, st)); break;
      case 5: DCK(smk_graph_launch(e, &s->g_head, [&](cudaStream_t c) { return df_head(s, c); }, st)); break;
      default: return cudaErrorInvalidValue;
    }
  }
  return cudaSuccess;
}

cudaError_t smk_dflash_run(smk_dflash* s, int n, cudaStream_t st, int* host_ss, int* host_ring) {
  DCK(smk_dflash_launch(s, 0, n, st));
  DCK(cudaMemcpyAsync(s->pinned_ss, s->ss, SS_N * sizeof(int), cudaMemcpyDeviceToHost, st));
  DCK(cudaMemcpyAsync(s->pinned_ring, s->ring, RING * sizeof(int), cudaMemcpyDeviceToHost, st));
  DCK(cudaStreamSynchronize(st));
  memcpy(host_ss, s->pinned_ss, SS_N * sizeof(int));
  memcpy(host_ring, s->pinned_ring, RING * sizeof(int));
  return cudaSuccess;
}

cudaError_t smk_dflash_read(smk_dflash* s, int what, void* host, size_t bytes) {
  const void* src = nullptr;
  size_t cap = 0;
  switch (what) {
    case 0: src = s->ss; cap = SS_N * sizeof(int); break;
    case 1: src = s->hid; cap = (size_t)R8 * H * 2; break;
    case 2: src = s->log; cap = (size_t)SMK_DF_LOG * SMK_DF_LOGW * sizeof(int); break;
    case 3: src = s->hp; cap = (size_t)R8 * SEL_D * 4; break;
    case 4: src = s->kc; cap = (size_t)NL * NKV * s->ctxd * HD * 2; break;
    case 5: src = s->sel_id; cap = (size_t)R8 * SEL_K * 4; break;
    case 6: src = s->th; cap = (size_t)R8 * H * 2; break;
    case 7: src = s->vc; cap = (size_t)NL * NKV * s->ctxd * HD * 2; break;
    case 8: src = s->sel_un; cap = (size_t)R8 * SEL_K * 4; break;
  }
  if (!src || bytes > cap) return cudaErrorInvalidValue;
  return cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost);
}

size_t smk_dflash_bytes(smk_dflash* s) { return s->bytes; }
int smk_dflash_ctxd(smk_dflash* s) { return s->ctxd; }

}  // extern "C"
