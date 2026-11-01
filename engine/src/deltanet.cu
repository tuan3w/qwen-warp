// swift-mk K1: fused gated-DeltaNet decode step (conv1d+silu, l2norm, gating, recurrence with per-row ring snapshots, gated RMSNorm).
// Clean-room; algorithm per HF qwen3_5 torch_recurrent_gated_delta_rule (semantics in oracle/NOTES.md §2).
//
// Parallelisation: grid = 48 v-heads x NS V-slices.  Every V row of the state is independent in the recurrence
// (kv[v] = sum_k S[v][k] k[k] ...), so a block owns VR = 128/NS V rows x 128 K of its head, keeps them in registers across
// the T rows, and writes one fp32 snapshot per row into ring slot (cur+1+t)%slots.  The only cross-block coupling is the gated
// RMSNorm over the 128 V outputs of a head: blocks publish o to scratch and the last block of the head (atomic counter) finishes it.
#include "deltanet.h"
#include <cuda_bf16.h>
#include <stdint.h>

// V-slices per head (grid = 48*NS blocks), tuned per T on the 3090 (BENCH.md): T<=2 NS=16 (more bytes in flight), T>=3 NS=8 (fewer shuffles/regs; T=5..8: K11).
#ifndef DN_NS_SMALL_T
#define DN_NS_SMALL_T 16
#endif
#ifndef DN_NS_LARGE_T
#define DN_NS_LARGE_T 8
#endif
#ifndef DN_LZ_NS_MID
#define DN_LZ_NS_MID 8  // lazy mode (engine) V-slices for T = 3..4
#endif
#ifndef DN_LZ_NS_HI
#define DN_LZ_NS_HI 8  // lazy mode, T = 5..8
#endif
#ifndef DN_STORE
#define DN_STORE(p, v) __stcs((p), (v))
#endif
#ifndef DN_LOAD
#define DN_LOAD(p) __ldcs((p))
#endif

namespace {
constexpr int NV = 48, DV = 128, DK = 128, CONVD = 10240, KEYD = 2048, VALD = 6144, TMAX = 8;
constexpr int NT = 128;  // threads per block (4 warps)
constexpr size_t STATE_SLOT = (size_t)NV * DV * DK;
constexpr size_t CONV_SLOT = (size_t)CONVD * 3;
constexpr size_t WS_CNT_BYTES = 256;
constexpr size_t WS_BYTES = WS_CNT_BYTES + (size_t)TMAX * NV * DV * sizeof(float);
// Lazy snapshot mode (engine only; MODE 1/2).  A spec step does not write its T per-row state snapshots; it appends the rank-1 updates of its rows to a delta buffer:
//   S_t = ge_t * S_{t-1} + d_t (x) k_t,  stored per row as ge[48], k[16][128], d[48][128] (fp32, exactly the values the recurrence used), LROWS rows per layer.
// Device scalar *pend = p | base << 8 | half << 12 (delta buffer half holding the pending rows): the state at ring slot `cur` is NOT materialised; it is ring slot `base` (a full state) with the first p rows of the delta buffer
// replayed (same fp32 ops => bitwise the state the full-snapshot kernel would have stored).  p == 0: slot cur is the state.  A MODE 2 step replays the p pending rows
// while its own state load is in flight; if p + T would overflow the buffer it first writes the replayed state to slot cur (new base) and appends its rows at 0, otherwise it
// appends after the pending rows and writes no state at all.  The accept step then does p += a (dn_pend_after in engine_impl.h mirrors this decision).
//   MODE 0: contract of deltanet.h (full snapshots, no pend).  MODE 1 (T == 1): replay pending, snapshot row 0 to slot cur+1 (pend -> 0 is the caller's job).
constexpr int LROWS = SWIFTMK_DN_LAZY_ROWS;
constexpr size_t DB_ROW = NV + 16 * DK + NV * DV;       // floats per row: ge | k | d
constexpr size_t DB_K = NV, DB_D = NV + 16 * DK;
constexpr size_t DB_FLOATS = (size_t)2 * LROWS * DB_ROW;  // per layer: two halves (a flush switches half, so the new rows never overwrite rows other blocks still read)

struct Params {
  const __nv_bfloat16 *qkv, *z, *a, *b, *conv_w, *A_log, *dt_bias, *norm_w;
  float* state;
  __nv_bfloat16* conv_state;
  const int* cur;
  int slots;
  __nv_bfloat16* y;
  unsigned* cnt;
  float* o_ws;
  float* dbuf;       // this layer's delta buffer (MODE > 0)
  const int* pend;   // device scalar p | base << 8 | half << 12 (MODE > 0)
};

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ int ring_next(int cur, int d, int slots) {  // (cur + d) % slots for cur < slots, d < slots
  const int i = cur + d;
  return i >= slots ? i - slots : i;
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}
__device__ __forceinline__ void cp_async4(void* smem, const void* gmem) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "l"(gmem));
}

// depthwise conv (taps over [3 stored inputs | T new inputs]) + silu for one channel; optionally write the T post conv-states.
// Split in a load half and a compute half so that the loads of all channels of a thread are in flight together (the state stores of the compute half order every later load).
template <int T>
struct ConvIn {
  float seq[3 + T];
  uint2 wr;
};
template <int T>
__device__ __forceinline__ void conv_load(const Params& p, int c, int cur, ConvIn<T>& in) {
  const __nv_bfloat16* cs = p.conv_state + (size_t)cur * CONV_SLOT + (size_t)c * 3;
#pragma unroll
  for (int j = 0; j < 3; ++j) in.seq[j] = bf(cs[j]);
#pragma unroll
  for (int t = 0; t < T; ++t) in.seq[3 + t] = bf(p.qkv[(size_t)t * CONVD + c]);
  in.wr = *reinterpret_cast<const uint2*>(p.conv_w + (size_t)c * 4);
}
template <int T>
__device__ __forceinline__ void conv_finish(const Params& p, int c, int cur, bool write_state, const ConvIn<T>& in, float (&out)[T]) {
  const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&in.wr.x);
  const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&in.wr.y);
  const float w0 = __low2float(w01), w1 = __high2float(w01), w2 = __low2float(w23), w3 = __high2float(w23);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float acc = w0 * in.seq[t];
    acc = fmaf(w1, in.seq[t + 1], acc);
    acc = fmaf(w2, in.seq[t + 2], acc);
    acc = fmaf(w3, in.seq[t + 3], acc);
    out[t] = silu_f(acc);
  }
  if (write_state) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      __nv_bfloat16* dst = p.conv_state + (size_t)(ring_next(cur, 1 + t, p.slots)) * CONV_SLOT + (size_t)c * 3;
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j] = __float2bfloat16(in.seq[t + 1 + j]);
    }
  }
}

template <int N>
__device__ __forceinline__ void warp_sum(float (&v)[N]) {
#pragma unroll
  for (int off = 16; off; off >>= 1)
#pragma unroll
    for (int i = 0; i < N; ++i) v[i] += __shfl_xor_sync(0xffffffffu, v[i], off);
}

__host__ __device__ constexpr int ilog2(int n) { return n <= 1 ? 0 : 1 + ilog2(n >> 1); }
__host__ __device__ constexpr int pow2_ge(int n) { return n <= 1 ? 1 : 2 * pow2_ge((n + 1) >> 1); }

// Transposed (reduce-scatter) warp sum of N (power of two <= 32) values: N - 1 + (5 - log2 N) shuffles instead of 5 N.  Every lane returns the full-warp sum of value index
// lane >> (5 - log2 N) (the pairwise adds are the same as in the xor butterfly, so the sums are bitwise those of warp_sum).
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

// xor-butterfly sum over groups of LPR consecutive lanes
template <int LPR>
__device__ __forceinline__ float group_sum(float v) {
#pragma unroll
  for (int off = LPR >> 1; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

template <int T, int NS, int MODE>
__global__ void __launch_bounds__(NT) dn_kernel(Params p) {
  constexpr int VR = DV / NS;     // V rows per block
  constexpr int RPW = VR / 4;     // V rows per warp (4 warps) = float4 registers per lane
  constexpr int LPR = 32 / RPW;   // lanes per V row: lane = rowg * LPR + lk; lane lk holds float4 i * LPR + lk of K (i < RPW) of its row
  static_assert(RPW >= 1 && RPW <= 8 && (RPW & (RPW - 1)) == 0 && VR % 4 == 0, "bad NS");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, rowg = lane / LPR, lk = lane % LPR;
  const int h = blockIdx.x / NS, s = blockIdx.x % NS, kh = h / 3;

  __shared__ __align__(16) float sq[T][DK], sk[T][DK];
  __shared__ float sv[T][VR], sge[T], sbeta[T], red[4][2 * T], sred2[4][T];
  __shared__ int s_last;
  constexpr int PR = MODE > 0 ? LROWS : 1;
  __shared__ __align__(16) float pk[PR][DK];
  __shared__ __align__(16) float pdd[PR][VR];
  __shared__ float pge[PR];

  const int cur = *p.cur;
  int pc = 0, base = cur, row0 = 0, rh = 0, wh = 0;
  bool flush = false;
  if constexpr (MODE > 0) {
    const int pv = *p.pend;
    pc = pv & 255;
    if (pc > 0) base = (pv >> 8) & 15;
    flush = MODE == 2 && pc > 0 && pc + T > LROWS;
    row0 = MODE == 2 && !flush ? pc : 0;
    rh = (pv >> 12) & 1;
    wh = flush ? rh ^ 1 : rh;
  }

  // 1. issue the state loads first (the long-latency part), rows [s*VR + warp*RPW, +RPW) of head h
  const size_t soff = ((size_t)h * DV + s * VR + warp * RPW + rowg) * DK + lk * 4;
  const float* Sin = p.state + (size_t)base * STATE_SLOT + soff;
  float4 st[RPW];

  // 1b. pending rank-1 updates of the previous step -> smem (cp.async: no registers, issued before anything can order behind it)
  if constexpr (MODE > 0) {
    constexpr int CPR = VR / 4;  // 16 B chunks per row of d
    static_assert(LROWS * CPR <= NT && LROWS <= NT, "pending-row staging layout");
    const float* rb = p.dbuf + (size_t)rh * LROWS * DB_ROW;
#pragma unroll
    for (int m = 0; m < LROWS * (DK / 4) / NT; ++m) {
      const int i = tid + m * NT, j = i / (DK / 4), c = i % (DK / 4);
      if (j < pc) cp_async16(&pk[j][c * 4], rb + (size_t)j * DB_ROW + DB_K + kh * DK + c * 4);
    }
    if (tid < LROWS * CPR) {
      const int j = tid / CPR, c = tid % CPR;
      if (j < pc) cp_async16(&pdd[j][c * 4], rb + (size_t)j * DB_ROW + DB_D + h * DV + s * VR + c * 4);
    }
    if (tid < LROWS && tid < pc) cp_async4(&pge[tid], rb + (size_t)tid * DB_ROW + h);
    asm volatile("cp.async.commit_group;");
  }
  float ga = 0.f, gdt = 0.f, gal = 0.f, gb = 0.f;
  if (tid < T) {  // raw gating inputs: loaded before the conv-state stores below (those stores order every later global load)
    ga = bf(p.a[tid * NV + h]);
    gdt = bf(p.dt_bias[h]);
    gal = bf(p.A_log[h]);
    gb = bf(p.b[tid * NV + h]);
  }

  // 2. conv + silu for q,k channels of head kh (tid = element), v channels of this slice
  float qv[T], kvv[T];
  const bool wr_qk = (h % 3 == 0) && (s == 0);
  ConvIn<T> inq, ink, inv;
  conv_load<T>(p, kh * DK + tid, cur, inq);
  conv_load<T>(p, KEYD + kh * DK + tid, cur, ink);
  if (tid < VR) conv_load<T>(p, 2 * KEYD + h * DV + s * VR + tid, cur, inv);
  conv_finish<T>(p, kh * DK + tid, cur, wr_qk, inq, qv);
  conv_finish<T>(p, KEYD + kh * DK + tid, cur, wr_qk, ink, kvv);
  if (tid < VR) {
    float vv[T];
    conv_finish<T>(p, 2 * KEYD + h * DV + s * VR + tid, cur, true, inv, vv);
#pragma unroll
    for (int t = 0; t < T; ++t) sv[t][tid] = vv[t];
  }
  if (tid < T) {
    const float av = ga + gdt;
    const float sp = av > 20.0f ? av : log1pf(expf(av));
    const float g = -expf(gal) * sp;
    sge[tid] = expf(g);
    sbeta[tid] = 1.0f / (1.0f + expf(-gb));
  }
  if constexpr (MODE > 0) asm volatile("cp.async.wait_all;" ::: "memory");

  // 3. l2norm of q and k per row
  constexpr int NSS = pow2_ge(2 * T);
  float ss[NSS];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    ss[2 * t] = qv[t] * qv[t];
    ss[2 * t + 1] = kvv[t] * kvv[t];
  }
#pragma unroll
  for (int i = 2 * T; i < NSS; ++i) ss[i] = 0.0f;
  {
    constexpr int SH = 5 - ilog2(NSS);
    const float tot = tsum<NSS>(ss, lane);
    if ((lane & ((1 << SH) - 1)) == 0 && (lane >> SH) < 2 * T) red[warp][lane >> SH] = tot;
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < RPW; ++r) st[r] = DN_LOAD(reinterpret_cast<const float4*>(Sin + (size_t)r * (LPR * 4)));
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float sqq = red[0][2 * t] + red[1][2 * t] + red[2][2 * t] + red[3][2 * t];
    const float skk = red[0][2 * t + 1] + red[1][2 * t + 1] + red[2][2 * t + 1] + red[3][2 * t + 1];
    sq[t][tid] = qv[t] * (rsqrtf(sqq + 1e-6f) * 0.08838834764831845f);  // 128^-0.5
    sk[t][tid] = kvv[t] * rsqrtf(skk + 1e-6f);
  }
  __syncthreads();
  if constexpr (MODE > 0) {
    if (pc > 0) {
      for (int j = 0; j < pc; ++j) {
        const float ge = pge[j];
        const float d = pdd[j][warp * RPW + rowg];
#pragma unroll
        for (int r = 0; r < RPW; ++r) {
          const float4 k4 = *reinterpret_cast<const float4*>(&pk[j][(r * LPR + lk) * 4]);
          st[r].x *= ge; st[r].y *= ge; st[r].z *= ge; st[r].w *= ge;
          st[r].x = fmaf(d, k4.x, st[r].x);
          st[r].y = fmaf(d, k4.y, st[r].y);
          st[r].z = fmaf(d, k4.z, st[r].z);
          st[r].w = fmaf(d, k4.w, st[r].w);
        }
      }
      if (flush) {
        float* Sc = p.state + (size_t)cur * STATE_SLOT + soff;
#pragma unroll
        for (int r = 0; r < RPW; ++r) DN_STORE(reinterpret_cast<float4*>(Sc + (size_t)r * (LPR * 4)), st[r]);
      }
    }
    if constexpr (MODE == 2) {
      float* dw = p.dbuf + ((size_t)wh * LROWS + row0) * DB_ROW;
      if (wr_qk) {
#pragma unroll
        for (int t = 0; t < T; ++t) dw[t * DB_ROW + DB_K + kh * DK + tid] = sk[t][tid];
      }
      if (s == 0 && tid < T) dw[tid * DB_ROW + h] = sge[tid];
    }
  }

  // 4. recurrence, T rows sequentially, state in registers
  float o_acc[T];
  for (int t = 0; t < T; ++t) o_acc[t] = 0.f;
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float ge = sge[t], beta = sbeta[t];
    float4 k4[RPW], q4[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      k4[r] = *reinterpret_cast<const float4*>(&sk[t][(r * LPR + lk) * 4]);
      q4[r] = *reinterpret_cast<const float4*>(&sq[t][(r * LPR + lk) * 4]);
    }
    float kvp = 0.0f;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      st[r].x *= ge; st[r].y *= ge; st[r].z *= ge; st[r].w *= ge;
      float a = st[r].x * k4[r].x;
      a = fmaf(st[r].y, k4[r].y, a);
      a = fmaf(st[r].z, k4[r].z, a);
      a = fmaf(st[r].w, k4[r].w, a);
      kvp = r == 0 ? a : kvp + a;
    }
    kvp = group_sum<LPR>(kvp);
    const float d = (sv[t][warp * RPW + rowg] - kvp) * beta;
    float* Sout = p.state + (size_t)(ring_next(cur, 1 + t, p.slots)) * STATE_SLOT + soff;
    float op = 0.0f;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      st[r].x = fmaf(d, k4[r].x, st[r].x);
      st[r].y = fmaf(d, k4[r].y, st[r].y);
      st[r].z = fmaf(d, k4[r].z, st[r].z);
      st[r].w = fmaf(d, k4[r].w, st[r].w);
      if constexpr (MODE != 2) DN_STORE(reinterpret_cast<float4*>(Sout + (size_t)r * (LPR * 4)), st[r]);
      float a = st[r].x * q4[r].x;
      a = fmaf(st[r].y, q4[r].y, a);
      a = fmaf(st[r].z, q4[r].z, a);
      a = fmaf(st[r].w, q4[r].w, a);
      op = r == 0 ? a : op + a;
    }
    if constexpr (MODE == 2) {
      if (lk == 0) p.dbuf[((size_t)wh * LROWS + row0 + t) * DB_ROW + DB_D + h * DV + s * VR + warp * RPW + rowg] = d;
    }
    o_acc[t] = group_sum<LPR>(op);
  }

  // 5. publish o (fp32) for the head; the last block of the head does the gated RMSNorm
  if (lk == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t) p.o_ws[((size_t)t * NV + h) * DV + s * VR + warp * RPW + rowg] = o_acc[t];
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned old = atomicAdd(&p.cnt[h], 1u);
    s_last = (old == NS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  if (tid == 0) p.cnt[h] = 0;  // all NS blocks have arrived; reset for the next launch
  const float wn = bf(p.norm_w[tid]);
  float ov[T];
  constexpr int NSF = pow2_ge(T);
  float sum[NSF];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    ov[t] = __ldcg(&p.o_ws[((size_t)t * NV + h) * DV + tid]);
    sum[t] = ov[t] * ov[t];
  }
#pragma unroll
  for (int t = T; t < NSF; ++t) sum[t] = 0.0f;
  {
    constexpr int SH = 5 - ilog2(NSF);
    const float tot = tsum<NSF>(sum, lane);
    if ((lane & ((1 << SH) - 1)) == 0 && (lane >> SH) < T) sred2[warp][lane >> SH] = tot;
  }
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float tot = sred2[0][t] + sred2[1][t] + sred2[2][t] + sred2[3][t];
    const float rs = rsqrtf(tot * (1.0f / DV) + 1e-6f);
    const float zz = bf(p.z[(size_t)t * VALD + h * DV + tid]);
    p.y[(size_t)t * VALD + h * DV + tid] = __float2bfloat16((wn * (ov[t] * rs)) * silu_f(zz));
  }
}

template <int T, int MODE = 0>
cudaError_t launch_t(const Params& p, cudaStream_t s) {
  constexpr int NS = MODE == 2 ? (T <= 2 ? 16 : T <= 4 ? DN_LZ_NS_MID : DN_LZ_NS_HI) : (T <= 2 ? DN_NS_SMALL_T : DN_NS_LARGE_T);
  dn_kernel<T, NS, MODE><<<NV * NS, NT, 0, s>>>(p);
  return cudaGetLastError();
}

// materialise the pending rows of every layer into ring slot cur (state: [layers][slots][48][128][128], dbuf: [layers][DB_FLOATS])
__global__ void dn_materialize_k(float4* __restrict__ state, const float* __restrict__ dbuf, const int* __restrict__ cur_p, const int* __restrict__ pend, int slots) {
  constexpr int SF4 = (int)(STATE_SLOT / 4);
  const int pv = *pend, pc = pv & 255;
  if (pc == 0) return;
  const int li = blockIdx.y, cur = *cur_p, base = (pv >> 8) & 15;
  const float4* in = state + ((size_t)li * slots + base) * SF4;
  float4* out = state + ((size_t)li * slots + cur) * SF4;
  const float* rb = dbuf + (size_t)li * DB_FLOATS + (size_t)((pv >> 12) & 1) * LROWS * DB_ROW;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SF4; i += gridDim.x * blockDim.x) {
    const int h = i / (DV * DK / 4), v = (i / (DK / 4)) % DV, k4i = i % (DK / 4);
    float4 sv = in[i];
    for (int j = 0; j < pc; ++j) {
      const float* r = rb + (size_t)j * DB_ROW;
      const float ge = r[h], d = r[DB_D + h * DV + v];
      const float4 k4 = *reinterpret_cast<const float4*>(r + DB_K + (h / 3) * DK + k4i * 4);
      sv.x *= ge; sv.y *= ge; sv.z *= ge; sv.w *= ge;
      sv.x = fmaf(d, k4.x, sv.x); sv.y = fmaf(d, k4.y, sv.y); sv.z = fmaf(d, k4.z, sv.z); sv.w = fmaf(d, k4.w, sv.w);
    }
    out[i] = sv;
  }
}
__global__ void dn_clear_pend_k(int* pend) { *pend = 0; }
}  // namespace

extern "C" size_t swiftmk_deltanet_ws_bytes(void) { return WS_BYTES; }

extern "C" cudaError_t swiftmk_deltanet_step(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                             const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                             void* conv_state, const int* cur, int slots, void* y, void* ws, cudaStream_t stream) {
  if (slots <= T || slots > 16) return cudaErrorInvalidValue;
  Params p;
  p.qkv = (const __nv_bfloat16*)qkv; p.z = (const __nv_bfloat16*)z; p.a = (const __nv_bfloat16*)a; p.b = (const __nv_bfloat16*)b;
  p.conv_w = (const __nv_bfloat16*)conv_w; p.A_log = (const __nv_bfloat16*)A_log; p.dt_bias = (const __nv_bfloat16*)dt_bias;
  p.norm_w = (const __nv_bfloat16*)norm_w; p.state = state; p.conv_state = (__nv_bfloat16*)conv_state; p.cur = cur; p.slots = slots;
  p.y = (__nv_bfloat16*)y; p.cnt = (unsigned*)ws; p.o_ws = (float*)((char*)ws + WS_CNT_BYTES);
  switch (T) {
    case 1: return launch_t<1>(p, stream);
    case 2: return launch_t<2>(p, stream);
    case 3: return launch_t<3>(p, stream);
    case 4: return launch_t<4>(p, stream);
    case 5: return launch_t<5>(p, stream);
    case 6: return launch_t<6>(p, stream);
    case 7: return launch_t<7>(p, stream);
    case 8: return launch_t<8>(p, stream);
    default: return cudaErrorInvalidValue;
  }
}

extern "C" size_t swiftmk_deltanet_dbuf_floats(void) { return DB_FLOATS; }

// Engine entry (lazy snapshots, see MODE above).  dbuf: this layer's delta buffer (swiftmk_deltanet_dbuf_floats() floats); pend: device scalar p | base << 8 | half << 12.
// T == 1 behaves like swiftmk_deltanet_step after replaying the pending rows (the caller zeroes *pend afterwards); T >= 2 leaves the T row snapshots unwritten:
// the caller's accept kernel must update *pend (dn_pend_after).
extern "C" cudaError_t swiftmk_deltanet_step_lazy(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                                  const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                                  void* conv_state, const int* cur, int slots, void* y, void* ws, float* dbuf, const int* pend,
                                                  cudaStream_t stream) {
  if (slots <= T || slots > 16) return cudaErrorInvalidValue;
  Params p;
  p.qkv = (const __nv_bfloat16*)qkv; p.z = (const __nv_bfloat16*)z; p.a = (const __nv_bfloat16*)a; p.b = (const __nv_bfloat16*)b;
  p.conv_w = (const __nv_bfloat16*)conv_w; p.A_log = (const __nv_bfloat16*)A_log; p.dt_bias = (const __nv_bfloat16*)dt_bias;
  p.norm_w = (const __nv_bfloat16*)norm_w; p.state = state; p.conv_state = (__nv_bfloat16*)conv_state; p.cur = cur; p.slots = slots;
  p.y = (__nv_bfloat16*)y; p.cnt = (unsigned*)ws; p.o_ws = (float*)((char*)ws + WS_CNT_BYTES);
  p.dbuf = dbuf; p.pend = pend;
  switch (T) {
    case 1: return launch_t<1, 1>(p, stream);
    case 2: return launch_t<2, 2>(p, stream);
    case 3: return launch_t<3, 2>(p, stream);
    case 4: return launch_t<4, 2>(p, stream);
    case 5: return launch_t<5, 2>(p, stream);
    case 6: return launch_t<6, 2>(p, stream);
    case 7: return launch_t<7, 2>(p, stream);
    case 8: return launch_t<8, 2>(p, stream);
    default: return cudaErrorInvalidValue;
  }
}

// Write the pending state of every layer into ring slot cur and clear *pend (external readers / writers of the state ring: snapshot, prefill).
extern "C" cudaError_t swiftmk_deltanet_materialize(float* state, const float* dbuf, int nlayers, const int* cur, int* pend, int slots, cudaStream_t stream) {
  dn_materialize_k<<<dim3(32, nlayers), 256, 0, stream>>>((float4*)state, dbuf, cur, pend, slots);
  dn_clear_pend_k<<<1, 1, 0, stream>>>(pend);
  return cudaGetLastError();
}

