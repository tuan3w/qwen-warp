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

struct Params {
  const __nv_bfloat16 *qkv, *z, *a, *b, *conv_w, *A_log, *dt_bias, *norm_w;
  float* state;
  __nv_bfloat16* conv_state;
  const int* cur;
  int slots;
  __nv_bfloat16* y;
  unsigned* cnt;
  float* o_ws;
};

__device__ __forceinline__ float bf(__nv_bfloat16 x) { return __bfloat162float(x); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// depthwise conv (taps over [3 stored inputs | T new inputs]) + silu for one channel; optionally write the T post conv-states.
template <int T>
__device__ __forceinline__ void conv_channel(const Params& p, int c, int cur, bool write_state, float (&out)[T]) {
  float seq[3 + T];
  const __nv_bfloat16* cs = p.conv_state + (size_t)cur * CONV_SLOT + (size_t)c * 3;
#pragma unroll
  for (int j = 0; j < 3; ++j) seq[j] = bf(cs[j]);
#pragma unroll
  for (int t = 0; t < T; ++t) seq[3 + t] = bf(p.qkv[(size_t)t * CONVD + c]);
  const uint2 wr = *reinterpret_cast<const uint2*>(p.conv_w + (size_t)c * 4);
  const __nv_bfloat162 w01 = *reinterpret_cast<const __nv_bfloat162*>(&wr.x);
  const __nv_bfloat162 w23 = *reinterpret_cast<const __nv_bfloat162*>(&wr.y);
  const float w0 = __low2float(w01), w1 = __high2float(w01), w2 = __low2float(w23), w3 = __high2float(w23);
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float acc = w0 * seq[t];
    acc = fmaf(w1, seq[t + 1], acc);
    acc = fmaf(w2, seq[t + 2], acc);
    acc = fmaf(w3, seq[t + 3], acc);
    out[t] = silu_f(acc);
  }
  if (write_state) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      __nv_bfloat16* dst = p.conv_state + (size_t)((cur + 1 + t) % p.slots) * CONV_SLOT + (size_t)c * 3;
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j] = __float2bfloat16(seq[t + 1 + j]);
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

template <int T, int NS>
__global__ void __launch_bounds__(NT) dn_kernel(Params p) {
  constexpr int VR = DV / NS;     // V rows per block
  constexpr int RPW = VR / 4;     // V rows per warp (4 warps)
  static_assert(RPW >= 1 && VR % 4 == 0, "bad NS");
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int h = blockIdx.x / NS, s = blockIdx.x % NS, kh = h / 3;

  __shared__ __align__(16) float sq[T][DK], sk[T][DK];
  __shared__ float sv[T][VR], sge[T], sbeta[T], red[4][2 * T], sred2[4][T];
  __shared__ int s_last;

  const int cur = *p.cur;

  // 1. issue the state loads first (the long-latency part), rows [s*VR + warp*RPW, +RPW) of head h
  const float* Sin = p.state + (size_t)cur * STATE_SLOT + ((size_t)h * DV + s * VR + warp * RPW) * DK + lane * 4;
  float4 st[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) st[r] = DN_LOAD(reinterpret_cast<const float4*>(Sin + (size_t)r * DK));

  // 2. conv + silu for q,k channels of head kh (tid = element), v channels of this slice
  float qv[T], kvv[T];
  const bool wr_qk = (h % 3 == 0) && (s == 0);
  conv_channel<T>(p, kh * DK + tid, cur, wr_qk, qv);
  conv_channel<T>(p, KEYD + kh * DK + tid, cur, wr_qk, kvv);
  if (tid < VR) {
    float vv[T];
    conv_channel<T>(p, 2 * KEYD + h * DV + s * VR + tid, cur, true, vv);
#pragma unroll
    for (int t = 0; t < T; ++t) sv[t][tid] = vv[t];
  }
  if (tid < T) {
    const float av = bf(p.a[tid * NV + h]) + bf(p.dt_bias[h]);
    const float sp = av > 20.0f ? av : log1pf(expf(av));
    const float g = -expf(bf(p.A_log[h])) * sp;
    sge[tid] = expf(g);
    sbeta[tid] = 1.0f / (1.0f + expf(-bf(p.b[tid * NV + h])));
  }

  // 3. l2norm of q and k per row
  float ss[2 * T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    ss[2 * t] = qv[t] * qv[t];
    ss[2 * t + 1] = kvv[t] * kvv[t];
  }
  warp_sum<2 * T>(ss);
  if (lane == 0)
#pragma unroll
    for (int i = 0; i < 2 * T; ++i) red[warp][i] = ss[i];
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float sqq = red[0][2 * t] + red[1][2 * t] + red[2][2 * t] + red[3][2 * t];
    const float skk = red[0][2 * t + 1] + red[1][2 * t + 1] + red[2][2 * t + 1] + red[3][2 * t + 1];
    sq[t][tid] = qv[t] * (rsqrtf(sqq + 1e-6f) * 0.08838834764831845f);  // 128^-0.5
    sk[t][tid] = kvv[t] * rsqrtf(skk + 1e-6f);
  }
  __syncthreads();

  // 4. recurrence, T rows sequentially, state in registers
  float o_acc[T][RPW];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float ge = sge[t], beta = sbeta[t];
    const float4 k4 = *reinterpret_cast<const float4*>(&sk[t][lane * 4]);
    const float4 q4 = *reinterpret_cast<const float4*>(&sq[t][lane * 4]);
    float kvp[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      st[r].x *= ge; st[r].y *= ge; st[r].z *= ge; st[r].w *= ge;
      kvp[r] = st[r].x * k4.x;
      kvp[r] = fmaf(st[r].y, k4.y, kvp[r]);
      kvp[r] = fmaf(st[r].z, k4.z, kvp[r]);
      kvp[r] = fmaf(st[r].w, k4.w, kvp[r]);
    }
    warp_sum<RPW>(kvp);
    float op[RPW];
    float* Sout = p.state + (size_t)((cur + 1 + t) % p.slots) * STATE_SLOT + ((size_t)h * DV + s * VR + warp * RPW) * DK + lane * 4;
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const float d = (sv[t][warp * RPW + r] - kvp[r]) * beta;
      st[r].x = fmaf(d, k4.x, st[r].x);
      st[r].y = fmaf(d, k4.y, st[r].y);
      st[r].z = fmaf(d, k4.z, st[r].z);
      st[r].w = fmaf(d, k4.w, st[r].w);
      DN_STORE(reinterpret_cast<float4*>(Sout + (size_t)r * DK), st[r]);
      op[r] = st[r].x * q4.x;
      op[r] = fmaf(st[r].y, q4.y, op[r]);
      op[r] = fmaf(st[r].z, q4.z, op[r]);
      op[r] = fmaf(st[r].w, q4.w, op[r]);
    }
    warp_sum<RPW>(op);
#pragma unroll
    for (int r = 0; r < RPW; ++r) o_acc[t][r] = op[r];
  }

  // 5. publish o (fp32) for the head; the last block of the head does the gated RMSNorm
  if (lane == 0) {
#pragma unroll
    for (int t = 0; t < T; ++t)
#pragma unroll
      for (int r = 0; r < RPW; ++r) p.o_ws[((size_t)t * NV + h) * DV + s * VR + warp * RPW + r] = o_acc[t][r];
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
  float ov[T], sum[T];
#pragma unroll
  for (int t = 0; t < T; ++t) {
    ov[t] = __ldcg(&p.o_ws[((size_t)t * NV + h) * DV + tid]);
    sum[t] = ov[t] * ov[t];
  }
  warp_sum<T>(sum);
  if (lane == 0)
#pragma unroll
    for (int t = 0; t < T; ++t) sred2[warp][t] = sum[t];
  __syncthreads();
#pragma unroll
  for (int t = 0; t < T; ++t) {
    const float tot = sred2[0][t] + sred2[1][t] + sred2[2][t] + sred2[3][t];
    const float rs = rsqrtf(tot * (1.0f / DV) + 1e-6f);
    const float zz = bf(p.z[(size_t)t * VALD + h * DV + tid]);
    p.y[(size_t)t * VALD + h * DV + tid] = __float2bfloat16((wn * (ov[t] * rs)) * silu_f(zz));
  }
}

template <int T>
cudaError_t launch_t(const Params& p, cudaStream_t s) {
  constexpr int NS = T <= 2 ? DN_NS_SMALL_T : DN_NS_LARGE_T;
  dn_kernel<T, NS><<<NV * NS, NT, 0, s>>>(p);
  return cudaGetLastError();
}
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
