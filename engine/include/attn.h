// swift-mk K2: full-attention decode (T = 1..8 rows) for one full-attention layer (or the MTP layer's attention).  See docs/ARCHITECTURE.md for the contract.
//
// Three launches per layer (all graph-capturable, no host sync, no allocation):
//   1. prep   : per-head q/k RMSNorm (zero-centred, eps 1e-6) + partial rope (first 64 of 256 dims, rotate_half, theta 1e7, angle in fp64),
//               writes K (post-norm, post-rope) and V of rows pos0..pos0+T-1 into the KV cache (bf16, or fp8-e4m3 = x/scale saturating),
//               stores the normed+roped q (bf16 / fp16) in the workspace.
//   2. split  : flash-decode over the KV cache, split over the key axis (grid sized for the maximum context, blocks beyond the live context
//               exit at once) x 4 kv heads.  Tensor cores (mma m16n8k16), K/V streamed global->registers, GQA 6 q heads per kv head, T rows
//               share the K/V reads, causal mask inside the T-row block (row t sees keys 0..pos0+t).
//   3. reduce : merges the partials, applies 1/sum, the fp8 V scale and  * sigmoid(gate)  ->  out [T][6144] (o_proj input).
//
// Layouts (bf16 = __nv_bfloat16 passed as void*, all device pointers):
//   q_proj [T][12288] bf16   q_proj output, per head h (0..23): [q(256) | gate(256)] interleaved.  k, v [T][1024] bf16 (4 kv heads x 256), pre-norm / raw.
//   q_norm, k_norm [256] bf16  (zero-centred: y = x*rsqrt(mean(x^2)+1e-6)*(1+w)).
//   kv     [2 (K,V)][4 kv heads][ctx_max][256] of THIS layer; bf16, or fp8 e4m3 when kv_fp8 != 0 (stored value = x / scale, dequant x * scale).
//          Row r of the step is written to position pos0+r (unaccepted verify rows are simply overwritten next step); pos0 + T <= ctx_max.
//          Use swiftmk_attn_kv_bytes() for the per-layer allocation; a 16-layer cache is just 16 of these back to back.
//   pos0   DEVICE int: absolute position of row 0 (= number of tokens already in the cache).  ctx_len = pos0 + T is derived on the device,
//          so one captured CUDA graph serves every step (T is baked into the graph: capture one graph per T).
//   out    [T][6144] bf16 = softmax(q.K^T/16) V * sigmoid(gate), heads concatenated (head h at columns 256h..256h+255).
//   ws     device scratch of swiftmk_attn_ws_bytes() bytes; may be shared by layers that run sequentially on ONE stream.
//          No initialisation needed (every kernel fully writes what the next one reads).
// k_scale / v_scale: the checkpoint per-layer fp8 KV scales (ignored when kv_fp8 == 0; MTP layer: kv_fp8 = 0).
#pragma once
#include <stddef.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

#ifndef SWIFTMK_ATTN_MAX_SPLITS
#define SWIFTMK_ATTN_MAX_SPLITS 20  // 4 kv heads x 20 = 80 blocks, one per SM (82 SMs); measured best (41/24 give a second partial wave)
#endif

size_t swiftmk_attn_ws_bytes(void);
size_t swiftmk_attn_kv_bytes(int ctx_max, int kv_fp8);  // bytes of one layer's [2][4][ctx_max][256] cache

// Prefix spill / restore (host prefix cache).  A "chunk" is the KV rows [a, a + m) of ONE layer packed as [plane][m][row bytes] (plane = K|V x kv head = 8 planes) and, for the
// int4 cache (kv_dtype 2), followed by [plane][m] u32 meta words.  Row bytes: bf16 512, fp8 256, int4 128 (code bytes).  The int4 sink / ring buffers are not part of a chunk
// (they are fixed-size state, see swiftmk_kv_aux_*).  kv_dtype: 0 bf16, 1 fp8, 2 int4.
size_t swiftmk_attn_kv_range_bytes(int kv_dtype, int m);
// dir 0: device -> host, 1: host -> device.  Copies rows [r0, r1) of the chunk (0 <= r0 <= r1 <= m) between the packed host chunk `host` (m rows) and the device layer cache
// `layer_kv` rows a + r0 .. a + r1 - 1 (a = dev_row0).  Async on `stream`; `host` should be pinned.
cudaError_t swiftmk_attn_kv_range_copy(void* layer_kv, int ctx_max, int kv_dtype, void* host, int m, int dev_row0, int r0, int r1, int dir, cudaStream_t stream);

// Returns cudaSuccess or the first launch error.  T must be 1..8.
cudaError_t swiftmk_attn_step(int T, const void* q_proj, const void* k, const void* v, const void* q_norm, const void* k_norm, void* kv,
                              int ctx_max, int kv_fp8, float k_scale, float v_scale, const int* pos0, void* out, void* ws,
                              cudaStream_t stream);
// Same, with the layer cache read from device memory at launch when kv_pp != NULL: cache = (char*)*kv_pp + kv_off (kv ignored).  Batched decode writes each sequence's cache base
// there before the graph launch, so one captured graph serves any set of sequences.
cudaError_t swiftmk_attn_step_ind(int T, const void* q_proj, const void* k, const void* v, const void* q_norm, const void* k_norm, void* kv, void* const* kv_pp,
                                  size_t kv_off, int ctx_max, int kv_fp8, float k_scale, float v_scale, const int* pos0, void* out, void* ws, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
