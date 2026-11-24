// Parametric decode kernels for dense decoder-only transformers: bf16 activations and weights (or GGUF Q4_0 weights),
// fp32 accumulation, at most STEP_ROWS rows per launch (decode T = 1, prompt chunks up to STEP_ROWS). Written and
// measured for sm_86; warpc/cuda/glue.py builds them for the GPU the runtime runs on. Shapes are arguments, not
// constants. Every entry point is extern "C", takes raw device pointers and a stream, allocates nothing and never
// synchronises, so it can be captured in a CUDA graph. Position is a DEVICE scalar (pos0 = tokens already cached).
//
// KV cache layout (per layer): bf16 [2 (K, V)][Hkv][ctx][hd], or a quantised type (dense_attn_q.cuh, warpc/kvcache.py).
// A bf16 state with a sliding window is a ring: position p lives in row p mod ctx, ctx a multiple of 64 (dense_attn_common.cuh).
//
// Source layout (the Makefile builds every translation unit in parallel into libwarpc_glue.so):
//   dense_common.cuh      types, rounding helpers, warp / block reductions (this file)
//   dense_gemv.cuh        GemvP and the k_gemv kernel template
//   dense_gemv.cu         the run-time dispatch (rows per warp, T, mode) and warpc_gemv
//   dense_gemv_inst.cu    one object per (T set, mode): the k_gemv instantiations (-DGEMV_SET= / -DGEMV_MODE=)
//   dense_dec_gemv.cuh    k_dg: the single-row (T = 1) GEMV with every load issued up front, k_gemv's arithmetic bit
//                         for bit; dense_dec_gemv_body.cuh + dense_dec_gemv_m<mode>.cu: its launchers, one per mode
//   dense_ks_gemv.cuh     k_ks: the K-split GEMV (KS warps share a row, K = 2048 KS; a different reduction order) of the
//                         segmented GEMVs k_dg cannot hold in registers (dense_ks_gemv_inst.cu: instantiations)
//   dense_q4_gemv.cuh     k_q4_gemv: k_gemv's lane map and reduction over GGUF Q4_0 / Q4_1 weights repacked at load time and
//                         dequantised in registers (dense_q4_gemv.cu: entry point, dense_q4_gemv_inst.cu: instantiations)
//   dense_q6_gemv.cuh     k_q6_gemv: the fp32-logits head over a repacked GGUF Q6_K matrix, bit-identical to k_gemv over the
//                         bf16-widened matrix (dense_q6_gemv.cu: entry point and instantiations)
//   dense_norm.cu         embed (optionally scaled), rmsnorm, rmsnorm_post (the norm folded into its residual add),
//                         argmax, the CUDA error string
//   gemma_rows.cu         k_ple: Gemma's per-layer-embedding prelude in one launch
//   dense_attn_common.cuh constants, QKV, the shared WHT and merge kernel (k_attn_reduce2), the attention launchers run_*
//   dense_attn.cuh        bf16 attention kernel templates (split, reduce, chunked split_x, and k_attn_fused = prep +
//                         split of one row, k_attn_decode1 = prep + split + merge of one row at head dim 256 / 512)
//   dense_attn.cu         k_attn_prep and the bf16 attention entry points
//   dense_attn_q.cuh      quantised-KV kernel templates (prep_q, split_q; r4:: the hand engine's int4_rot writer)
//   dense_attn_q.cu       the quantised-KV entry points
//   dense_attn_inst.cu    the attention launchers run_* and their instantiations per head dim and part (-DATTN_HD= /
//                         -DATTN_PARTS=)
//   dense_prefetch.cu     the L2 prefetch kernel
// A kernel template is instantiated in exactly ONE translation unit (the *_inst.cu objects); the dispatching units call
// it through a launcher declared in a header.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#define STEP_ROWS 8  // rows one launch handles at most (warpc/cuda/schedule.py STEP_ROWS)
#define GEMV_WARPS 4  // warps per GEMV block, one row group each (schedule.py GEMV_WARPS)

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 tb(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float rb(float v) { return __bfloat162float(__float2bfloat16(v)); }  // round to bf16 and back

__device__ __forceinline__ void unpack8(const uint4& u, float* f) {
  const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 t = __bfloat1622float2(p[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float* f) {
  uint4 u;
  __nv_bfloat162* p = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; i++) p[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// block-wide reductions (blockDim.x a multiple of 32, at most 1024); `sm` holds 32 floats
static __device__ float block_sum(float v, float* sm) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sm[w] = v;
  __syncthreads();
  float r = 0.f;
  for (int i = 0; i < nw; i++) r += sm[i];
  return r;
}
static __device__ float block_max(float v, float* sm) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) sm[w] = v;
  __syncthreads();
  float r = -INFINITY;
  for (int i = 0; i < nw; i++) r = fmaxf(r, sm[i]);
  return r;
}
