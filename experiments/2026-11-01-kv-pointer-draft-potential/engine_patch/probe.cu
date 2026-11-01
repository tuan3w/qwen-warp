// E7 (experiments/2026-11-01-kv-pointer-draft-potential) PRIVATE probe, compiled into a private copy of the swift-mk engine (HEAD 2d16e21), NOT part of the live engine.
//
// After the K2 attention step of every full-attention layer (non-graph step only) it re-reads the engine's own state: the normed+roped q that the K2 prep kernel left in the
// attention workspace and the bf16 KV cache, recomputes the attention scores s_j = q.k_j / 16 (fp32) for every key j <= pos0+t of every q head and records, per (row t, layer, q head):
//   idx  [4]  int16  the 4 highest-scoring key positions
//   mass [6]  fp16   softmax mass of those 4, of position 0 (attention sink), and of the last 8 keys (local window)
// Optionally (snap) it snapshots q, the gate part of q_proj and K2's own output y6, so that a torch recomputation can verify the probe against the real kernel output.
// Also: hidden-state taps -> a position-indexed store (smk_probe_store), KV cache copy (smk_probe_copy_kv).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

#include "engine_impl.h"

using smk::H;
using smk::TM;

namespace {
constexpr int NH = 24, HD = 256, GRP = 6, NAT = 16;
constexpr int NR = 6;  // mass slots

__global__ void __launch_bounds__(256) probe_scores(const uint16_t* __restrict__ q, const uint16_t* __restrict__ K, int ctx_max, int pos0, float* __restrict__ sc, size_t sc_stride) {
  const int t = blockIdx.z, h = blockIdx.y;
  const int n = pos0 + t + 1;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j0 = blockIdx.x * 64 + warp * 8;
  if (j0 >= n) return;
  const int kvh = h / GRP;
  float qv[8];
  {
    const uint4 raw = *reinterpret_cast<const uint4*>(q + ((size_t)t * NH + h) * HD + lane * 8);
    const uint16_t* r = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; i++) qv[i] = __uint_as_float((uint32_t)r[i] << 16);
  }
  for (int k = 0; k < 8; k++) {
    const int j = j0 + k;
    if (j >= n) break;
    const uint4 raw = *reinterpret_cast<const uint4*>(K + ((size_t)kvh * ctx_max + j) * HD + lane * 8);
    const uint16_t* r = reinterpret_cast<const uint16_t*>(&raw);
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) a += qv[i] * __uint_as_float((uint32_t)r[i] << 16);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
    if (lane == 0) sc[((size_t)t * NH + h) * sc_stride + j] = a * (1.f / 16.f);
  }
}

__device__ __forceinline__ float blk_reduce(float v, bool is_max, float* sm) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : v + u;
  }
  __syncthreads();
  if (lane == 0) sm[w] = v;
  __syncthreads();
  float r = sm[0];
  for (int i = 1; i < nw; i++) r = is_max ? fmaxf(r, sm[i]) : r + sm[i];
  return r;
}

__global__ void __launch_bounds__(512) probe_top(const float* __restrict__ sc, size_t sc_stride, int pos0, int ai, int row_off, int16_t* __restrict__ oidx, __half* __restrict__ omass) {
  __shared__ float sm[32];
  __shared__ float wv[32];
  __shared__ int wi[32];
  const int t = blockIdx.y, h = blockIdx.x;
  const int n = pos0 + t + 1;
  const float* s = sc + ((size_t)t * NH + h) * sc_stride;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) m = fmaxf(m, s[j]);
  m = blk_reduce(m, true, sm);
  float l = 0.f, lw = 0.f;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    const float e = expf(s[j] - m);
    l += e;
    if (j >= n - 8) lw += e;
  }
  l = blk_reduce(l, false, sm);
  lw = blk_reduce(lw, false, sm);
  const size_t ob = (((size_t)(row_off + t) * NAT + ai) * NH + h);
  int chosen[4] = {-1, -1, -1, -1};
  for (int r = 0; r < 4; r++) {
    float bv = -INFINITY;
    int bj = -1;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
      bool skip = false;
      for (int c = 0; c < r; c++) skip |= (j == chosen[c]);
      if (skip) continue;
      const float v = s[j];
      if (v > bv) { bv = v; bj = j; }
    }
    // block argmax (ties -> lowest index)
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float uv = __shfl_xor_sync(0xffffffffu, bv, o);
      const int uj = __shfl_xor_sync(0xffffffffu, bj, o);
      if (uv > bv || (uv == bv && uj >= 0 && (bj < 0 || uj < bj))) { bv = uv; bj = uj; }
    }
    __syncthreads();
    if (lane == 0) { wv[w] = bv; wi[w] = bj; }
    __syncthreads();
    bv = wv[0]; bj = wi[0];
    for (int i = 1; i < nw; i++)
      if (wv[i] > bv || (wv[i] == bv && wi[i] >= 0 && (bj < 0 || wi[i] < bj))) { bv = wv[i]; bj = wi[i]; }
    chosen[r] = bj;
    if (threadIdx.x == 0) {
      oidx[ob * 4 + r] = (int16_t)bj;
      omass[ob * NR + r] = __float2half(bj >= 0 ? expf(bv - m) / l : 0.f);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    omass[ob * NR + 4] = __float2half(expf(s[0] - m) / l);
    omass[ob * NR + 5] = __float2half(lw / l);
  }
}
}  // namespace

cudaError_t smk_probe_attn(smk_engine* e, int l, int T, cudaStream_t st) {
  const int ai = e->at_idx[l];
  const uint16_t* q = reinterpret_cast<const uint16_t*>(e->ws_attn);  // K2 workspace WS_Q = 0: u16 [TMAX][NH][HD] (bf16 path)
  const uint16_t* K = reinterpret_cast<const uint16_t*>(e->kv + (size_t)ai * e->kv_layer_bytes);
  const int ctx = e->probe_pos0 + T;
  dim3 g1((ctx + 63) / 64, NH, T);
  probe_scores<<<g1, 256, 0, st>>>(q, K, e->ctx_max, e->probe_pos0, e->probe_sc, (size_t)e->ctx_max);
  dim3 g2(NH, T);
  probe_top<<<g2, 512, 0, st>>>(e->probe_sc, (size_t)e->ctx_max, e->probe_pos0, ai, e->probe_row_off, e->probe_idx, e->probe_mass);
  if (e->probe_snap) {
    cudaMemcpyAsync(e->probe_q + (size_t)ai * TM * NH * HD, q, (size_t)T * NH * HD * 2, cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(e->probe_out + (size_t)ai * TM * 6144, e->y6, (size_t)T * 6144 * 2, cudaMemcpyDeviceToDevice, st);
    // gate = q_proj[t][h*512 + 256 .. +256]
    cudaMemcpy2DAsync(e->probe_gate + (size_t)ai * TM * NH * HD, HD * 2, e->qb + 256, 512 * 2, HD * 2, (size_t)T * NH, cudaMemcpyDeviceToDevice, st);
  }
  return cudaGetLastError();
}

extern "C" {

// rows: capacity of the idx/mass output rings (rows of T-row steps); returns 0 on success
int smk_probe_alloc(smk_engine* e, int rows) {
  if (e->kv_fp8) return -2;
  e->probe_rows = rows;
  e->probe_sc = (float*)dalloc(e, (size_t)TM * NH * e->ctx_max * 4);
  e->probe_idx = (int16_t*)dalloc(e, (size_t)rows * NAT * NH * 4 * 2, true);
  e->probe_mass = (__half*)dalloc(e, (size_t)rows * NAT * NH * NR * 2, true);
  e->probe_q = (uint16_t*)dalloc(e, (size_t)NAT * TM * NH * HD * 2, true);
  e->probe_gate = (uint16_t*)dalloc(e, (size_t)NAT * TM * NH * HD * 2, true);
  e->probe_out = (uint16_t*)dalloc(e, (size_t)NAT * TM * 6144 * 2, true);
  return e->failed ? -1 : 0;
}

// on: run the attention probe in the next non-graph steps; snap: also snapshot q/gate/out; pos0: absolute position of row 0 of the step; row_off: output ring row of row 0
int smk_probe_cfg(smk_engine* e, int on, int snap, int pos0, int row_off) {
  e->probe_on = on; e->probe_snap = snap; e->probe_pos0 = pos0; e->probe_row_off = row_off;
  return 0;
}

// what: 0 idx, 1 mass, 2 q snap, 3 gate snap, 4 out snap; n bytes from the start (device -> host)
int smk_probe_read(smk_engine* e, int what, void* host, size_t bytes) {
  const void* src = what == 0 ? (void*)e->probe_idx : what == 1 ? (void*)e->probe_mass : what == 2 ? (void*)e->probe_q : what == 3 ? (void*)e->probe_gate : what == 4 ? (void*)e->probe_out : nullptr;
  if (!src) return -1;
  return (int)cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost);
}

// copy rows of the debug taps of the last step into dst bf16 [nsl][cap][H] at position pos0 + t
int smk_probe_store(smk_engine* e, void* dst, int pos0, int T, int nsl, const int* slots, int cap, cudaStream_t st) {
  for (int s = 0; s < nsl; s++)
    for (int t = 0; t < T; t++) {
      cudaError_t err = cudaMemcpyAsync((uint16_t*)dst + ((size_t)s * cap + pos0 + t) * H, e->dbg + ((size_t)slots[s] * TM + t) * H, H * 2, cudaMemcpyDeviceToDevice, st);
      if (err != cudaSuccess) return (int)err;
    }
  return 0;
}

// K cache of attention layer ai ([4 kv heads][ctx_max][256] bf16, the first half of the layer's [2][4][ctx_max][256]) -> device dst
int smk_probe_copy_kv(smk_engine* e, int ai, int which, void* dst, cudaStream_t st) {
  const size_t half = (size_t)4 * e->ctx_max * 256 * 2;
  return (int)cudaMemcpyAsync(dst, e->kv + (size_t)ai * e->kv_layer_bytes + which * half, half, cudaMemcpyDeviceToDevice, st);
}

}  // extern "C"
