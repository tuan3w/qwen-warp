// Inkling-2 attention block (head dim 128, no rope): prep + split-K flash decode + merge, three launches. Derived from the old tree's wik_prep / winkl_split / winkl_merge
// (inkling.cu, inkling_attn.cu) and the chunked split machinery of dense_attn.cuh. Every entry is extern "C", takes raw device pointers and a stream, allocates nothing and never
// synchronises, so it can be captured in a CUDA graph. The position is a DEVICE scalar (pos0 = tokens already cached): row t sits at p = *pos0 + t.
//
// KV state: bf16 [2 (K, V)][Hkv][R][128]; position p lives in slot p mod R (R = the state's rows: a ring on sliding visits, R >= window + rows - 1, or the whole context).
// Row t sees the positions l in [lo, p], lo = max(0, p - window + 1) (0 without a window), d = p - l:
//   bias[t,h,d]  = bf16( sum_{j<16} r[t,h,j] * proj[j][d] )   for d < E (0 beyond)       (computed per key and head; proj is read transposed, [E][16])
//   logit_l      = (q[t,h] . K[g][l]) * scale + bias,  g = h / (Hq / Hkv)
//   softmax over [logits..., sink[t,h]] (the sink column has a zero value vector), att = sum_l p_l V[g][l]  (fp32, P not rounded to bf16)
//   out[t, h*128+i] = bf16( bf16(att[i]) * sigmoid(gate[t, h*128+i]) )
//
// prep   (Hq + 2 Hkv blocks x 128 threads, every block loops over the T rows in order): q / r l2_clip with the learned scale (+ log scaling on full layers), k / v short conv
//        (static weights [C][4], residual, or dynamic [4][C] + per-token taps) over bf16 states [C][3] updated in place, k l2_clip, v rmsnorm without weight, k / v written to
//        the KV state at slot pos mod R. Arithmetic order follows warpc/reference.py: products rounded one by one, left-fold sums, the clip factor thr * (1 / (norm + 1e-6)).
// split  (Hkv, nsplit, T) blocks: the row's live range is cut into chunks of 64 keys counted from lo (so a ring's window is always at most ceil(window / 64) chunks wide), nc
//        consecutive chunks per block with the running (m, l) of the online softmax; partials {o[128], m, l} per (row, head, split).
// merge  (Hq, T) blocks: the live splits and the sink column, the bf16 rounding of att, the output gate.
#include "dense_attn_common.cuh"

#define IA_HD 128
#define IA_DREL 16
#define IA_W 4
#define IA_PS (IA_HD + 4)  // floats per partial row: o[128], m, l, 2 pad (16-byte aligned)
#define IA_MAXG 8

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- prep
struct IaPrep {
  const bf16 *q, *r, *k, *v, *ktap, *vtap;  // projection outputs [T][Hq*128], [T][Hq*16], [T][C], [T][C], taps [T][Hkv*4] (dynamic convs only)
  const bf16 *qs, *rs, *kw, *vw;            // learned clip scales [Hq][128], [Hq][16]; conv weights [C][4] (static) or [4][C] (dynamic)
  bf16 *kst, *vst;                          // conv states [C][3], oldest first
  bf16 *qo, *ro, *kv;                       // clipped q [T][Hq][128], r [T][Hq][16]; the KV state
  const int* pos0;
  int Hq, Hkv, R, dyn, logs;
  float thr_q, thr_r, thr_k, eps, div, alpha, floor_;
};

// the clip factor min(1, thr / (norm + 1e-6)) as torch evaluates `thr / tensor` (reciprocal, then product)
__device__ __forceinline__ float ia_clip(float ss, float thr) { return fminf(__fmul_rn(thr, __frcp_rn(__fadd_rn(sqrtf(ss), 1e-6f))), 1.f); }

// Sums of N per-thread values over the 4 warps of the block, one barrier, `sm` (4 N floats) used once. Same value as dense_common's block_sum: warp trees, then the warps added
// in index order.
template <int N>
__device__ __forceinline__ void ia_block_sum(float (&v)[N], float* sm) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < N; j++) v[j] = warp_sum(v[j]);
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < N; j++) sm[w * N + j] = v[j];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < N; j++) v[j] = ((sm[j] + sm[N + j]) + sm[2 * N + j]) + sm[3 * N + j];
}

// NT rows, all of a block's rows at once: every row's clip / conv / norm depends on the inputs and the state alone, so the NT reductions share one barrier instead of
// running one after the other.
template <int NT>
__global__ void __launch_bounds__(128) k_ink_prep(IaPrep p) {
  __shared__ float sm[4 * NT];
  const int b = blockIdx.x, i = threadIdx.x;
  const int p0 = *p.pos0;
  if (b < p.Hq) {
    const int h = b;
    const float qs = bf(p.qs[h * IA_HD + i]);
    float x[NT], ss[NT], ls[NT], xr[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
      x[t] = bf(p.q[((size_t)t * p.Hq + h) * IA_HD + i]) * qs;
      ss[t] = x[t] * x[t];
      ls[t] = 1.f;
      if (p.logs) ls[t] = __fadd_rn(1.f, __fmul_rn(p.alpha, logf(fmaxf(((float)(p0 + t) + 1.f) / p.floor_, 1.f))));
    }
    if (i < 32) {  // r: 16 features per head, the first warp
      const float rs = i < IA_DREL ? bf(p.rs[h * IA_DREL + i]) : 0.f;
#pragma unroll
      for (int t = 0; t < NT; t++) xr[t] = i < IA_DREL ? bf(p.r[((size_t)t * p.Hq + h) * IA_DREL + i]) * rs : 0.f;
    }
    ia_block_sum<NT>(ss, sm);
#pragma unroll
    for (int t = 0; t < NT; t++) {
      float y = rb(x[t] * ia_clip(ss[t], p.thr_q));
      if (p.logs) y *= ls[t];
      p.qo[((size_t)t * p.Hq + h) * IA_HD + i] = tb(y);
    }
    if (i < 32) {
      float s[NT];
#pragma unroll
      for (int t = 0; t < NT; t++) s[t] = warp_sum(xr[t] * xr[t]);
#pragma unroll
      for (int t = 0; t < NT; t++) {
        float yr = rb(xr[t] * ia_clip(s[t], p.thr_r));
        if (p.logs) yr *= ls[t];
        if (i < IA_DREL) p.ro[((size_t)t * p.Hq + h) * IA_DREL + i] = tb(yr);
      }
    }
    return;
  }
  const bool isv = b >= p.Hq + p.Hkv;
  const int g = b - p.Hq - (isv ? p.Hkv : 0);
  const int C = p.Hkv * IA_HD, c = g * IA_HD + i;
  const bf16* raw = isv ? p.v : p.k;
  const bf16* tap = isv ? p.vtap : p.ktap;
  const bf16* cw = isv ? p.vw : p.kw;
  bf16* st = isv ? p.vst : p.kst;
  float x[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) x[t] = bf(raw[(size_t)t * C + c]);
  float u0 = bf(st[(size_t)c * (IA_W - 1)]), u1 = bf(st[(size_t)c * (IA_W - 1) + 1]), u2 = bf(st[(size_t)c * (IA_W - 1) + 2]);  // the 3 rows before the window, oldest first
  float w[IA_W], co[NT];
  if (p.dyn) {  // lag l = 0 is the current row: weight[l][c] + tap[t][head][l]
#pragma unroll
    for (int l = 0; l < IA_W; l++) w[l] = bf(cw[(size_t)l * C + c]);
  } else {
#pragma unroll
    for (int j = 0; j < IA_W; j++) w[j] = bf(cw[(size_t)c * IA_W + j]);  // tap 3 = the current row
  }
  float tp[NT][IA_W];
  if (p.dyn) {
#pragma unroll
    for (int t = 0; t < NT; t++)
#pragma unroll
      for (int l = 0; l < IA_W; l++) tp[t][l] = bf(tap[(size_t)t * p.Hkv * IA_W + g * IA_W + l]);
  }
  const float rdiv = 1.f / p.div;  // torch divides a CUDA tensor by a host scalar as x * (1 / scalar)
  int slot0 = p0 % p.R;
  float ss[NT];
#pragma unroll
  for (int t = 0; t < NT; t++) {
    float y;
    if (p.dyn) {  // terms summed in lag order
      const float w0 = __fadd_rn(w[0], tp[t][0]), w1 = __fadd_rn(w[1], tp[t][1]), w2 = __fadd_rn(w[2], tp[t][2]), w3 = __fadd_rn(w[3], tp[t][3]);
      y = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(w0, x[t]), __fmul_rn(w1, u2)), __fmul_rn(w2, u1)), __fmul_rn(w3, u0));
    } else {
      y = __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(w[0], u0), __fmul_rn(w[1], u1)), __fmul_rn(w[2], u2)), __fmul_rn(w[3], x[t]));
    }
    co[t] = rb(__fmul_rn(__fadd_rn(y, x[t]), rdiv));
    ss[t] = co[t] * co[t];
    u0 = u1;
    u1 = u2;
    u2 = x[t];
  }
  st[(size_t)c * (IA_W - 1)] = tb(u0);
  st[(size_t)c * (IA_W - 1) + 1] = tb(u1);
  st[(size_t)c * (IA_W - 1) + 2] = tb(u2);
  ia_block_sum<NT>(ss, sm);
#pragma unroll
  for (int t = 0; t < NT; t++) {
    const float o = isv ? co[t] * rsqrtf(ss[t] / (float)IA_HD + p.eps) : co[t] * ia_clip(ss[t], p.thr_k);
    int slot = slot0 + t;  // R >= T
    if (slot >= p.R) slot -= p.R;
    p.kv[(((size_t)(isv ? p.Hkv : 0) + g) * p.R + slot) * IA_HD + i] = tb(o);
  }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- split
// Block (kv head, split, row t), 128 threads; the G = Hq / Hkv query heads of the kv head share the K / V reads.
//   scores : two threads per key (interleaved 16-byte pieces, a pair reads one contiguous 256-byte row), plus half of the 16 bias features each
//   softmax: one warp per query head inside the chunk, running (m, l) across the block's chunks
//   P V    : thread = (8 output dims, one of 8 key groups); K / V / proj loads of a chunk are issued before anything is staged or computed
template <int G>
__global__ void __launch_bounds__(128) k_ink_split(const bf16* __restrict__ q, const bf16* __restrict__ r, const bf16* __restrict__ projT, const bf16* __restrict__ kv,
                                                   const int* __restrict__ pos0, float* __restrict__ part, int Hq, int Hkv, int R, int E, int window, int nsplit, int nc,
                                                   float scale) {
  __shared__ __align__(16) float qs[G * IA_HD];
  __shared__ float rs[G * IA_DREL];
  __shared__ float sc[G * CH];
  __shared__ __align__(16) float po[4 * G * IA_HD];
  __shared__ float ml[24];  // running m [8], running l [8], rescale factor of the last chunk [8]
  const int kvh = blockIdx.x, split = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
  const int p = *pos0 + t;
  const int lo = window > 0 ? max(0, p - window + 1) : 0;
  const int start0 = lo + split * nc * CH;
  if (start0 > p) return;
  const int j = tid >> 1, half = tid & 1;  // scores: key j of the chunk, half of the features
  const int dg = tid & 15, kg = tid >> 4;  // P V: 8 output dims dg * 8.., key group kg (keys kg + 8 i)
  const bf16* Kp = kv + (size_t)kvh * R * IA_HD;
  const bf16* Vp = kv + ((size_t)Hkv + kvh) * R * IA_HD;
  if (tid < 8) {
    ml[tid] = -INFINITY;
    ml[8 + tid] = 0.f;
  }
  float o[G][8];
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[g][e] = 0.f;
  for (int c = 0; c < nc; c++) {
    const int start = start0 + c * CH;
    if (start > p) break;
    const int n = min(CH, p - start + 1);
    const int s0 = start % R;
    // ---- every global load of the chunk first
    const bool kvalid = j < n;
    uint4 kk[8];
    if (kvalid) {
      int sl = s0 + j;
      if (sl >= R) sl -= R;
      const uint4* kr = reinterpret_cast<const uint4*>(Kp + (size_t)sl * IA_HD);
#pragma unroll
      for (int cc = 0; cc < 8; cc++) kk[cc] = __ldg(kr + cc * 2 + half);
    } else {
#pragma unroll
      for (int cc = 0; cc < 8; cc++) kk[cc] = make_uint4(0, 0, 0, 0);
    }
    const int d = p - (start + j);
    const bool rvalid = kvalid && d < E;
    uint4 pr = make_uint4(0, 0, 0, 0);
    if (rvalid) pr = __ldg(reinterpret_cast<const uint4*>(projT + (size_t)d * IA_DREL) + half);
    uint4 vv[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int jv = kg + 8 * i;
      if (jv < n) {
        int sl = s0 + jv;
        if (sl >= R) sl -= R;
        vv[i] = __ldg(reinterpret_cast<const uint4*>(Vp + (size_t)sl * IA_HD) + dg);
      } else {
        vv[i] = make_uint4(0, 0, 0, 0);
      }
    }
    if (c == 0) {  // stage q and r (contiguous over the G heads of this kv head)
      if (tid < G * 16) {
        float f[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(q + ((size_t)t * Hq + (size_t)kvh * G) * IA_HD) + tid), f);
        float4* dst = reinterpret_cast<float4*>(qs + tid * 8);
        dst[0] = make_float4(f[0], f[1], f[2], f[3]);
        dst[1] = make_float4(f[4], f[5], f[6], f[7]);
      }
      if (tid < G * 2) {
        float f[8];
        unpack8(__ldg(reinterpret_cast<const uint4*>(r + ((size_t)t * Hq + (size_t)kvh * G) * IA_DREL) + tid), f);
#pragma unroll
        for (int e = 0; e < 8; e++) rs[tid * 8 + e] = f[e];
      }
    }
    __syncthreads();  // staging visible; the previous chunk's reads of sc / ml are done
    // ---- scores
    {
      float dot[G], rp[G];
#pragma unroll
      for (int g = 0; g < G; g++) {
        dot[g] = 0.f;
        rp[g] = 0.f;
      }
#pragma unroll
      for (int cc = 0; cc < 8; cc++) {
        float f[8];
        unpack8(kk[cc], f);
#pragma unroll
        for (int g = 0; g < G; g++) {
          const float4* qp = reinterpret_cast<const float4*>(qs + g * IA_HD + (cc * 2 + half) * 8);
          const float4 a = qp[0], bq = qp[1];
          dot[g] = fmaf(f[0], a.x, dot[g]);
          dot[g] = fmaf(f[1], a.y, dot[g]);
          dot[g] = fmaf(f[2], a.z, dot[g]);
          dot[g] = fmaf(f[3], a.w, dot[g]);
          dot[g] = fmaf(f[4], bq.x, dot[g]);
          dot[g] = fmaf(f[5], bq.y, dot[g]);
          dot[g] = fmaf(f[6], bq.z, dot[g]);
          dot[g] = fmaf(f[7], bq.w, dot[g]);
        }
      }
      if (rvalid) {
        float pj[8];
        unpack8(pr, pj);
#pragma unroll
        for (int jj = 0; jj < 8; jj++)
#pragma unroll
          for (int g = 0; g < G; g++) rp[g] = fmaf(rs[g * IA_DREL + 8 * half + jj], pj[jj], rp[g]);
      }
#pragma unroll
      for (int g = 0; g < G; g++) {
        dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], 1);
        rp[g] += __shfl_xor_sync(0xffffffffu, rp[g], 1);  // a + b == b + a: both halves hold the same sum
      }
      if (half == 0) {
#pragma unroll
        for (int g = 0; g < G; g++) sc[g * CH + j] = kvalid ? __fadd_rn(__fmul_rn(dot[g], scale), rb(rp[g])) : -INFINITY;
      }
    }
    __syncthreads();
    // ---- softmax inside the chunk, one warp per query head (64 scores: two per lane)
    for (int g = tid >> 5; g < G; g += 4) {
      const int lane = tid & 31;
      const float s0v = sc[g * CH + lane], s1v = sc[g * CH + lane + 32];
      const float mold = ml[g];
      const float m = fmaxf(mold, warp_max(fmaxf(s0v, s1v)));  // finite: the chunk holds at least one key
      const float e0 = s0v == -INFINITY ? 0.f : __expf(s0v - m), e1 = s1v == -INFINITY ? 0.f : __expf(s1v - m);
      const float l = warp_sum(e0 + e1);
      const float alpha = mold == -INFINITY ? 0.f : __expf(mold - m);
      sc[g * CH + lane] = e0;
      sc[g * CH + lane + 32] = e1;
      __syncwarp();  // the lanes' reads of ml[g] precede lane 0's writes (the shuffles are not a memory barrier)
      if (lane == 0) {
        ml[g] = m;
        ml[8 + g] = ml[8 + g] * alpha + l;
        ml[16 + g] = alpha;
      }
    }
    __syncthreads();
    // ---- P V
    if (c > 0) {
#pragma unroll
      for (int g = 0; g < G; g++) {
        const float al = ml[16 + g];
#pragma unroll
        for (int e = 0; e < 8; e++) o[g][e] *= al;
      }
    }
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float f[8];
      unpack8(vv[i], f);
#pragma unroll
      for (int g = 0; g < G; g++) {
        const float pjk = sc[g * CH + kg + 8 * i];
#pragma unroll
        for (int e = 0; e < 8; e++) o[g][e] = fmaf(pjk, f[e], o[g][e]);
      }
    }
  }
#pragma unroll
  for (int g = 0; g < G; g++)
#pragma unroll
    for (int e = 0; e < 8; e++) o[g][e] += __shfl_xor_sync(0xffffffffu, o[g][e], 16);
  if ((tid & 31) < 16) {
    const int w = tid >> 5;
#pragma unroll
    for (int g = 0; g < G; g++) {
      float4* dst = reinterpret_cast<float4*>(po + ((size_t)w * G + g) * IA_HD + dg * 8);
      dst[0] = make_float4(o[g][0], o[g][1], o[g][2], o[g][3]);
      dst[1] = make_float4(o[g][4], o[g][5], o[g][6], o[g][7]);
    }
  }
  __syncthreads();
#pragma unroll
  for (int g = 0; g < G; g++) {
    float* pp = part + (((size_t)t * Hq + (size_t)kvh * G + g) * nsplit + split) * IA_PS;
    pp[tid] = ((po[(0 * G + g) * IA_HD + tid] + po[(1 * G + g) * IA_HD + tid]) + po[(2 * G + g) * IA_HD + tid]) + po[(3 * G + g) * IA_HD + tid];
    if (tid == 0) {
      pp[IA_HD] = ml[g];
      pp[IA_HD + 1] = ml[8 + g];
    }
  }
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- merge
// Block (head, row t), 128 threads (one per output dim). Up to 32 live splits: every warp derives the merge weights itself (split s on lane s: the same shuffle trees in every
// warp, so they agree bit for bit), no shared memory and no barrier; more: block reductions and a weight table. Either way the output sums the splits in ascending order.
__global__ void __launch_bounds__(128) k_ink_merge(const float* __restrict__ part, const int* __restrict__ pos0, const bf16* __restrict__ sinks, const bf16* __restrict__ gate,
                                                   bf16* __restrict__ out, int Hq, int window, int nsplit, int kpb) {
  __shared__ float wgt[MAX_SPLITS2];
  __shared__ float red[32];
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x, lane = tid & 31;
  const int p = *pos0 + t;
  const int lo = window > 0 ? max(0, p - window + 1) : 0;
  const int ns = (p - lo + kpb) / kpb;  // ceil((p - lo + 1) / kpb)
  const float* base = part + ((size_t)t * Hq + head) * nsplit * IA_PS;
  const float sink = bf(sinks[(size_t)t * Hq + head]);
  const size_t oi = ((size_t)t * Hq + head) * IA_HD + tid;
  const float gt = bf(gate[oi]);
  float a = 0.f;
  if (ns <= 32) {
    float ov[8];
#pragma unroll
    for (int s = 0; s < 8; s++)
      if (s < ns) ov[s] = base[s * IA_PS + tid];
    float ms = -INFINITY, ls = 0.f;
    if (lane < ns) {
      ms = base[lane * IA_PS + IA_HD];
      ls = base[lane * IA_PS + IA_HD + 1];
    }
    const float m = warp_max(fmaxf(ms, sink));
    const float es = __expf(ms - m);  // 0 on the lanes beyond ns
    const float l = warp_sum(ls * es) + __expf(sink - m);
    const float w = es * (1.f / l);
#pragma unroll
    for (int s = 0; s < 8; s++)
      if (s < ns) a = fmaf(ov[s], __shfl_sync(0xffffffffu, w, s), a);
    for (int s0 = 8; s0 < ns; s0 += 8) {
      float vv[8];
#pragma unroll
      for (int i = 0; i < 8; i++)
        if (s0 + i < ns) vv[i] = base[(s0 + i) * IA_PS + tid];
#pragma unroll
      for (int i = 0; i < 8; i++)
        if (s0 + i < ns) a = fmaf(vv[i], __shfl_sync(0xffffffffu, w, s0 + i), a);
    }
  } else {
    float m = sink;
    for (int s = tid; s < ns; s += 128) m = fmaxf(m, base[s * IA_PS + IA_HD]);
    m = block_max(m, red);
    float l = 0.f;
    for (int s = tid; s < ns; s += 128) l += base[s * IA_PS + IA_HD + 1] * __expf(base[s * IA_PS + IA_HD] - m);
    l = block_sum(l, red) + __expf(sink - m);
    const float inv = 1.f / l;
    for (int s = tid; s < ns; s += 128) wgt[s] = __expf(base[s * IA_PS + IA_HD] - m) * inv;
    __syncthreads();
#pragma unroll 8
    for (int s = 0; s < ns; s++) a = fmaf(base[s * IA_PS + tid], wgt[s], a);
  }
  const float sig = 1.f / (1.f + expf(-gt));
  out[oi] = tb(rb(a) * sig);
}

// ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- entry points
extern "C" {

// workspace of the split / merge partials
size_t warpc_ink_attn_ws_bytes(int T, int Hq, int nsplit) { return (size_t)T * Hq * nsplit * IA_PS * sizeof(float); }

int warpc_ink_attn_prep(int T, const void* q, const void* r, const void* k, const void* v, const void* ktap, const void* vtap, const void* qs, const void* rs, const void* kw,
                        const void* vw, void* kst, void* vst, void* qo, void* ro, void* kv, const int* pos0, int Hq, int Hkv, int R, int dyn, int logs, float thr_q,
                        float thr_r, float thr_k, float eps, float div, float alpha, float floor_, cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS || Hq < 1 || Hkv < 1 || R < T || (dyn && (!ktap || !vtap))) return cudaErrorInvalidValue;
  IaPrep p;
  p.q = reinterpret_cast<const bf16*>(q);
  p.r = reinterpret_cast<const bf16*>(r);
  p.k = reinterpret_cast<const bf16*>(k);
  p.v = reinterpret_cast<const bf16*>(v);
  p.ktap = reinterpret_cast<const bf16*>(ktap);
  p.vtap = reinterpret_cast<const bf16*>(vtap);
  p.qs = reinterpret_cast<const bf16*>(qs);
  p.rs = reinterpret_cast<const bf16*>(rs);
  p.kw = reinterpret_cast<const bf16*>(kw);
  p.vw = reinterpret_cast<const bf16*>(vw);
  p.kst = reinterpret_cast<bf16*>(kst);
  p.vst = reinterpret_cast<bf16*>(vst);
  p.qo = reinterpret_cast<bf16*>(qo);
  p.ro = reinterpret_cast<bf16*>(ro);
  p.kv = reinterpret_cast<bf16*>(kv);
  p.pos0 = pos0;
    p.Hq = Hq;
  p.Hkv = Hkv;
  p.R = R;
  p.dyn = dyn;
  p.logs = logs;
  p.thr_q = thr_q;
  p.thr_r = thr_r;
  p.thr_k = thr_k;
  p.eps = eps;
  p.div = div;
  p.alpha = alpha;
  p.floor_ = floor_;
  const int blocks = Hq + 2 * Hkv;
  switch (T) {
#define IA_PREP(N)                                       \
  case N:                                                \
    k_ink_prep<N><<<blocks, 128, 0, st>>>(p);            \
    break;
    IA_PREP(1) IA_PREP(2) IA_PREP(3) IA_PREP(4) IA_PREP(5) IA_PREP(6) IA_PREP(7) IA_PREP(8)
#undef IA_PREP
  }
  return cudaGetLastError();
}

int warpc_ink_attn_split(int T, const void* q, const void* r, const void* projT, const void* kv, const int* pos0, void* ws, int Hq, int Hkv, int R, int E, int window, int nc,
                         int nsplit, float scale, cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS || Hkv < 1 || Hq < Hkv || Hq % Hkv || Hq / Hkv > IA_MAXG || R < 1 || E < 1 || window < 0 || nc < 1 || nsplit < 1 || nsplit > MAX_SPLITS2)
    return cudaErrorInvalidValue;
  const dim3 grid(Hkv, nsplit, T);
  const bf16* qp = reinterpret_cast<const bf16*>(q);
  const bf16* rp = reinterpret_cast<const bf16*>(r);
  const bf16* pp = reinterpret_cast<const bf16*>(projT);
  const bf16* kp = reinterpret_cast<const bf16*>(kv);
  float* part = reinterpret_cast<float*>(ws);
#define IA_CASE(N)                                                                                                         \
  case N:                                                                                                                  \
    k_ink_split<N><<<grid, 128, 0, st>>>(qp, rp, pp, kp, pos0, part, Hq, Hkv, R, E, window, nsplit, nc, scale);          \
    break;
  switch (Hq / Hkv) {
    IA_CASE(1) IA_CASE(2) IA_CASE(3) IA_CASE(4) IA_CASE(5) IA_CASE(6) IA_CASE(7) IA_CASE(8)
    default: return cudaErrorInvalidValue;
  }
#undef IA_CASE
  return cudaGetLastError();
}

int warpc_ink_attn_merge(int T, const void* sinks, const void* gate, const int* pos0, const void* ws, void* out, int Hq, int window, int nc, int nsplit, cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS || Hq < 1 || window < 0 || nc < 1 || nsplit < 1 || nsplit > MAX_SPLITS2) return cudaErrorInvalidValue;
  k_ink_merge<<<dim3(Hq, T), 128, 0, st>>>(reinterpret_cast<const float*>(ws), pos0, reinterpret_cast<const bf16*>(sinks), reinterpret_cast<const bf16*>(gate),
                                           reinterpret_cast<bf16*>(out), Hq, window, nsplit, CH * nc);
  return cudaGetLastError();
}

}  // extern "C"
