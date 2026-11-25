// The Gated DeltaNet step kernel (see qwen35_deltanet.cuh for the contract).  Included only by the translation units that
// instantiate it.
//
// Numerics: the kernel computes the hand engine's fp32 values bit for bit (every sum has the hand's tree), so the two
// kernels' states never diverge: the conv is the same fma chain and the same expf / IEEE-division SiLU, the l2 sums and the
// gated-norm sums are warp reduce-scatters of one element per lane combined over 32-element groups in order, the S^T k and S^T q
// dots are per-lane chains over 4 consecutive rows (one float4 of the hand's [dv][dk] layout), added over the lane's chunks in
// order and butterflied over LPR lanes.  LPR is the hand's V-slice count of the row count (lpr_for).  Test: test_deltanet.py.
//
// Layout of the work: CTA = NW warps = (head h, VR = NW * VW columns of dv); a warp owns VW = 128 / LPR columns x dk rows of S;
// lane = cg * LPR + lk holds, for the 4 columns of column group cg, rows 4 (lk + LPR i) .. +3 of every chunk i < dk / 4 / LPR,
// in registers across the T rows (rows are float4 of the v-contiguous graph layout: a warp's load is whole 32-64 B sectors).
// Nothing in the recurrence touches shared memory or a barrier.
//
// Cross-CTA couplings, both by arrival counters in ws (the last arrival does the work, nobody waits):
//  - the gated RMSNorm needs the whole o of a head: the head's last CTA normalises (as the hand engine does);
//  - the conv state is in place and the q / k channels of a q/k head are read by (Hv / Hk) * NS CTAs: the last of them to
//    finish reading writes the new state.  v channels belong to one CTA.
#pragma once
#include "qwen35_deltanet.cuh"

namespace dn {
using bf16 = __nv_bfloat16;

namespace {
__device__ __forceinline__ float bf(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__host__ __device__ constexpr int ilog2(int n) { return n <= 1 ? 0 : 1 + ilog2(n >> 1); }
__host__ __device__ constexpr int pow2_ge(int n) { return n <= 1 ? 1 : 2 * pow2_ge((n + 1) >> 1); }

// Reduce-scatter warp sum of N (a power of two <= 32) values in N - 1 + (5 - log2 N) shuffles instead of 5 N: afterwards every lane
// holds in v[0] the full-warp sum of value index lane >> (5 - log2 N).
template <int N, int OFF>
__device__ __forceinline__ void tsum_rec(float* v, int lane) {
  if constexpr (N > 1) {
    constexpr int half = N / 2;
    const bool hi = (lane & OFF) != 0;
#pragma unroll
    for (int i = 0; i < half; ++i) {
      const float send = hi ? v[i] : v[i + half];
      const float keep = hi ? v[i + half] : v[i];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
    }
    tsum_rec<half, OFF / 2>(v, lane);
  } else {
#pragma unroll
    for (int off = OFF; off; off >>= 1) v[0] += __shfl_xor_sync(0xffffffffu, v[0], off);
  }
}
template <int N>
__device__ __forceinline__ float tsum(float (&v)[N], int lane) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "N must be a power of two");
  tsum_rec<N, 16>(v, lane);
  return v[0];
}

// xor-butterfly sum over the LPR consecutive lanes of a group
template <int LPR>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = LPR >> 1; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// One channel's depthwise conv over [W - 1 stored inputs | T new inputs], split into a load half and a compute half so that the loads of
// all of a thread's channels are in flight together.
template <int T, int W>
struct Conv {
  float seq[W - 1 + T];
  float w[W];
};
template <int T, int W>
__device__ __forceinline__ void conv_load(const Args& p, int c, Conv<T, W>& cv) {
  const bf16* cs = p.conv + (size_t)c * (W - 1);
#pragma unroll
  for (int j = 0; j < W - 1; ++j) cv.seq[j] = bf(cs[j]);
#pragma unroll
  for (int t = 0; t < T; ++t) cv.seq[W - 1 + t] = bf(p.qkv[(size_t)t * p.C + c]);
  if constexpr (W == 4) {
    const uint2 u = *reinterpret_cast<const uint2*>(p.conv_w + (size_t)c * 4);
    const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&u.x);
    const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&u.y);
    cv.w[0] = __low2float(w01), cv.w[1] = __high2float(w01), cv.w[2] = __low2float(w23), cv.w[3] = __high2float(w23);
  } else {
#pragma unroll
    for (int i = 0; i < W; ++i) cv.w[i] = bf(p.conv_w[(size_t)c * W + i]);
  }
}
template <int T, int W>
__device__ __forceinline__ float conv_out(const Conv<T, W>& cv, int t) {
  float acc = cv.w[0] * cv.seq[t];
#pragma unroll
  for (int i = 1; i < W; ++i) acc = fmaf(cv.w[i], cv.seq[t + i], acc);
  return silu_f(acc);
}
template <int T, int W>
__device__ __forceinline__ void conv_store(const Args& p, int c, const Conv<T, W>& cv) {
#pragma unroll
  for (int j = 0; j < W - 1; ++j) p.conv[(size_t)c * (W - 1) + j] = __float2bfloat16(cv.seq[T + j]);
}
}  // namespace

template <int T, int NW, int DK, int DV, int W>
__global__ void __launch_bounds__(32 * NW) k_step(const __grid_constant__ Args p) {
  constexpr int LPR = lpr_for(T);  // lanes per column: the hand's V-slice count at this row count
  constexpr int NCG = 32 / LPR;    // column groups (float4 of columns) per warp
  constexpr int VW = 4 * NCG;      // columns per warp
  constexpr int NT = 32 * NW;
  constexpr int VR = VW * NW;      // columns of dv per CTA
  constexpr int NS = DV / VR;      // CTAs per head
  constexpr int RC = DK / 4 / LPR; // chunks of 4 rows per lane
  constexpr int NVW = DK / 32;     // 32-element groups of the l2 sums (the hand's warps): element 32 v + lane
  constexpr int NVO = DV / 32;     // the same for the gated norm
  constexpr int CPT = NVW / NW;    // q (and k) channels per thread
  constexpr int EPT = NVO / NW;    // output elements per thread of the normalisation tail
  constexpr int NSS = pow2_ge(2 * T), NSF = pow2_ge(T);
  static_assert(DV % VR == 0 && DK % (4 * LPR) == 0 && DK % 32 == 0 && DV % 32 == 0, "unsupported shape");
  static_assert(NVW % NW == 0 && NVO % NW == 0 && VR <= NT, "unsupported warp count");

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, cg = lane / LPR, lk = lane % LPR;
  const int h = blockIdx.x / NS, s = blockIdx.x % NS, kh = h / p.rep;

  __shared__ __align__(16) float sq[T][DK], sk[T][DK], sv[T][VR];
  __shared__ float sge[T], sbeta[T], red[NVW][NSS], red2[NVO][NSF];
  __shared__ int s_conv_last, s_last;

  // 1. the state loads first: the long-latency part.  st[i][r]: row 4 (lk + LPR i) + r, the 4 columns of group cg
  float* const S = p.state + ((size_t)h * DK + 4 * lk) * DV + s * VR + warp * VW + cg * 4;
  float4 st[RC][4];
#pragma unroll
  for (int i = 0; i < RC; ++i)
#pragma unroll
    for (int r = 0; r < 4; ++r) st[i][r] = __ldcs(reinterpret_cast<const float4*>(S + (size_t)(4 * LPR * i + r) * DV));

  float ga = 0.f, gdt = 0.f, gal = 0.f, gb = 0.f;
  if (tid < T) {
    ga = bf(p.a[tid * p.Hv + h]);
    gdt = bf(p.dt_bias[h]);
    gal = bf(p.A_log[h]);
    gb = bf(p.b[tid * p.Hv + h]);
  }

  // 2. conv + silu: q and k channels of head kh (element e = 32 (warp + NW i) + lane), the v channels of this CTA's columns
  const int kd = p.Hk * DK;
  Conv<T, W> cq[CPT], ck[CPT], cvv;
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    const int e = 32 * (warp + NW * i) + lane;
    conv_load<T, W>(p, kh * DK + e, cq[i]);
    conv_load<T, W>(p, kd + kh * DK + e, ck[i]);
  }
  const int vc = 2 * kd + h * DV + s * VR + tid;
  if (tid < VR) conv_load<T, W>(p, vc, cvv);

  float qv[CPT][T], kv[CPT][T];
#pragma unroll
  for (int i = 0; i < CPT; ++i)
#pragma unroll
    for (int t = 0; t < T; ++t) {
      qv[i][t] = conv_out<T, W>(cq[i], t);
      kv[i][t] = conv_out<T, W>(ck[i], t);
    }
  if (tid < VR) {
#pragma unroll
    for (int t = 0; t < T; ++t) sv[t][tid] = conv_out<T, W>(cvv, t);
    conv_store<T, W>(p, vc, cvv);  // this CTA alone reads and writes these channels
  }
  if (tid < T) {
    const float av = ga + gdt;
    const float sp = av > 20.0f ? av : log1pf(expf(av));
    const float g = -expf(gal) * sp;
    sge[tid] = expf(g);
    sbeta[tid] = 1.0f / (1.0f + expf(-gb));
  }

  // 2b. the q / k conv state: the last CTA of the group of the q/k head to be done reading writes it
  __threadfence();
  __syncthreads();
  if (tid == 0) s_conv_last = atomicAdd(&p.cnt_kh[kh], 1u) == (unsigned)(p.rep * NS - 1);
  __syncthreads();
  if (s_conv_last) {
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      const int e = 32 * (warp + NW * i) + lane;
      conv_store<T, W>(p, kh * DK + e, cq[i]);
      conv_store<T, W>(p, kd + kh * DK + e, ck[i]);
    }
    if (tid == 0) p.cnt_kh[kh] = 0;  // every CTA of the group has arrived
  }

  // 3. l2 norm of q and k per row: a warp sum per 32-element group, the groups added in order
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    float ss[NSS];
#pragma unroll
    for (int t = 0; t < T; ++t) {
      ss[2 * t] = qv[i][t] * qv[i][t];
      ss[2 * t + 1] = kv[i][t] * kv[i][t];
    }
#pragma unroll
    for (int j = 2 * T; j < NSS; ++j) ss[j] = 0.f;
    constexpr int SH = 5 - ilog2(NSS);
    const float tot = tsum<NSS>(ss, lane);
    if ((lane & ((1 << SH) - 1)) == 0 && (lane >> SH) < 2 * T) red[warp + NW * i][lane >> SH] = tot;
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float sqq = red[0][2 * t], skk = red[0][2 * t + 1];
#pragma unroll
    for (int g = 1; g < NVW; ++g) sqq += red[g][2 * t], skk += red[g][2 * t + 1];
    const float rq = rsqrtf(sqq + p.eps) * p.scale, rk = rsqrtf(skk + p.eps);
#pragma unroll
    for (int i = 0; i < CPT; ++i) {
      const int e = 32 * (warp + NW * i) + lane;
      sq[t][e] = qv[i][t] * rq;
      sk[t][e] = kv[i][t] * rk;
    }
  }
  __syncthreads();

  // 4. the recurrence, T rows sequentially on the register state
  float o_acc[T][4];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float ge = sge[t], beta = sbeta[t];
    float4 k4[RC], q4[RC];
#pragma unroll
    for (int i = 0; i < RC; ++i) {
      k4[i] = *reinterpret_cast<const float4*>(&sk[t][4 * (lk + LPR * i)]);
      q4[i] = *reinterpret_cast<const float4*>(&sq[t][4 * (lk + LPR * i)]);
    }
    float m[4];
#pragma unroll
    for (int i = 0; i < RC; ++i) {
      float a[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) st[i][r].x *= ge, st[i][r].y *= ge, st[i][r].z *= ge, st[i][r].w *= ge;
      a[0] = st[i][0].x * k4[i].x, a[1] = st[i][0].y * k4[i].x, a[2] = st[i][0].z * k4[i].x, a[3] = st[i][0].w * k4[i].x;
      a[0] = fmaf(st[i][1].x, k4[i].y, a[0]), a[1] = fmaf(st[i][1].y, k4[i].y, a[1]);
      a[2] = fmaf(st[i][1].z, k4[i].y, a[2]), a[3] = fmaf(st[i][1].w, k4[i].y, a[3]);
      a[0] = fmaf(st[i][2].x, k4[i].z, a[0]), a[1] = fmaf(st[i][2].y, k4[i].z, a[1]);
      a[2] = fmaf(st[i][2].z, k4[i].z, a[2]), a[3] = fmaf(st[i][2].w, k4[i].z, a[3]);
      a[0] = fmaf(st[i][3].x, k4[i].w, a[0]), a[1] = fmaf(st[i][3].y, k4[i].w, a[1]);
      a[2] = fmaf(st[i][3].z, k4[i].w, a[2]), a[3] = fmaf(st[i][3].w, k4[i].w, a[3]);
#pragma unroll
      for (int c = 0; c < 4; ++c) m[c] = i == 0 ? a[c] : m[c] + a[c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) m[c] = group_sum<LPR>(m[c]);
    const float4 v4 = *reinterpret_cast<const float4*>(&sv[t][warp * VW + cg * 4]);
    const float d[4] = {(v4.x - m[0]) * beta, (v4.y - m[1]) * beta, (v4.z - m[2]) * beta, (v4.w - m[3]) * beta};
    float o[4];
#pragma unroll
    for (int i = 0; i < RC; ++i) {
      float a[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float kr = r == 0 ? k4[i].x : r == 1 ? k4[i].y : r == 2 ? k4[i].z : k4[i].w;
        st[i][r].x = fmaf(d[0], kr, st[i][r].x), st[i][r].y = fmaf(d[1], kr, st[i][r].y);
        st[i][r].z = fmaf(d[2], kr, st[i][r].z), st[i][r].w = fmaf(d[3], kr, st[i][r].w);
      }
      a[0] = st[i][0].x * q4[i].x, a[1] = st[i][0].y * q4[i].x, a[2] = st[i][0].z * q4[i].x, a[3] = st[i][0].w * q4[i].x;
      a[0] = fmaf(st[i][1].x, q4[i].y, a[0]), a[1] = fmaf(st[i][1].y, q4[i].y, a[1]);
      a[2] = fmaf(st[i][1].z, q4[i].y, a[2]), a[3] = fmaf(st[i][1].w, q4[i].y, a[3]);
      a[0] = fmaf(st[i][2].x, q4[i].z, a[0]), a[1] = fmaf(st[i][2].y, q4[i].z, a[1]);
      a[2] = fmaf(st[i][2].z, q4[i].z, a[2]), a[3] = fmaf(st[i][2].w, q4[i].z, a[3]);
      a[0] = fmaf(st[i][3].x, q4[i].w, a[0]), a[1] = fmaf(st[i][3].y, q4[i].w, a[1]);
      a[2] = fmaf(st[i][3].z, q4[i].w, a[2]), a[3] = fmaf(st[i][3].w, q4[i].w, a[3]);
#pragma unroll
      for (int c = 0; c < 4; ++c) o[c] = i == 0 ? a[c] : o[c] + a[c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) o_acc[t][c] = group_sum<LPR>(o[c]);
  }
#pragma unroll
  for (int i = 0; i < RC; ++i)
#pragma unroll
    for (int r = 0; r < 4; ++r) __stcs(reinterpret_cast<float4*>(S + (size_t)(4 * LPR * i + r) * DV), st[i][r]);

  // 5. publish o; the head's last CTA does the gated RMSNorm
  if (lk == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t)
      *reinterpret_cast<float4*>(p.o_ws + ((size_t)t * p.Hv + h) * DV + s * VR + warp * VW + cg * 4) =
          make_float4(o_acc[t][0], o_acc[t][1], o_acc[t][2], o_acc[t][3]);
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) s_last = atomicAdd(&p.cnt_head[h], 1u) == (unsigned)(NS - 1);
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) p.cnt_head[h] = 0;

  float ov[EPT][T];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    const int e = 32 * (warp + NW * i) + lane;
    float sum[NSF];
#pragma unroll
    for (int t = 0; t < NSF; ++t) sum[t] = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t) {
      ov[i][t] = __ldcg(&p.o_ws[((size_t)t * p.Hv + h) * DV + e]);
      sum[t] = ov[i][t] * ov[i][t];
    }
    constexpr int SH = 5 - ilog2(NSF);
    const float tot = tsum<NSF>(sum, lane);
    if ((lane & ((1 << SH) - 1)) == 0 && (lane >> SH) < T) red2[warp + NW * i][lane >> SH] = tot;
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float tot = red2[0][t];
#pragma unroll
    for (int g = 1; g < NVO; ++g) tot += red2[g][t];
    const float rs = rsqrtf(tot * (1.0f / DV) + p.norm_eps);
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      const int e = 32 * (warp + NW * i) + lane;
      const float zz = bf(p.z[(size_t)t * p.Hv * DV + h * DV + e]);
      p.y[(size_t)t * p.Hv * DV + h * DV + e] = __float2bfloat16((bf(p.norm_w[e]) * (ov[i][t] * rs)) * silu_f(zz));
    }
  }
}

template <int T, int NW, int DK, int DV, int W>
cudaError_t launch(const Args& a, cudaStream_t s) {
  constexpr int NS = DV / (4 * (32 / lpr_for(T)) * NW);
  k_step<T, NW, DK, DV, W><<<a.Hv * NS, 32 * NW, 0, s>>>(a);
  return cudaGetLastError();
}
}  // namespace dn

// Explicit instantiation of every configuration of the row count DN_T (a translation unit holds a few row counts: compile time)
#define DN_INST_ONE(DK, DV, W) template cudaError_t dn::launch<DN_T, dn::nw_for(DN_T, DK, DV), DK, DV, W>(const dn::Args&, cudaStream_t);
