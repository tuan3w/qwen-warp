// swift-mk K1: gated DeltaNet decode (T = 1..8 rows) for one linear-attention layer.  See ENGINE.md for the contract.
//
// One launch fuses: causal conv1d(k=4) + SiLU on q|k|v, L2-norm q,k (eps 1e-6 inside rsqrt), q *= 128^-0.5,
// g = -exp(A_log)*softplus(a+dt_bias), beta = sigmoid(b), delta-rule update S = exp(g) S; d = (v - S k) beta; S += d (x) k; o = S q,
// and the gated RMSNorm  y = (norm_w * o * rsqrt(mean(o^2)+1e-6)) * silu(z)   (norm_w NOT zero-centred).
//
// Layouts (all device pointers; bf16 = __nv_bfloat16 passed as void*):
//   qkv   [T][10240] bf16  (q 2048 | k 2048 | v 6144; q,k = 16 heads x 128, v = 48 heads x 128), z [T][6144] bf16,
//   a, b  [T][48] bf16 (in_proj_a / in_proj_b outputs), conv_w [10240][4] bf16 (tap 3 = newest), A_log, dt_bias [48] bf16, norm_w [128] bf16.
//   state      fp32 [slots][48 v-heads][128 V][128 K]  (vLLM layout; slot stride 786432 floats), slots >= T+1 (5 for T <= 4, 9 for T <= 8)
//   conv_state bf16 [slots][10240][3] (last 3 raw conv inputs per channel, oldest first; slot stride 30720 elements)
//   cur   device int in 0..slots-1: the step READS slot cur; row t writes its post-state (state and conv state) to slot (cur+1+t)%slots.
//   y     [T][6144] bf16 (out_proj input).
//   ws    zero-initialised device scratch of swiftmk_deltanet_ws_bytes() bytes; per layer or shared by layers that run
//         sequentially on ONE stream.  The kernel leaves it zeroed again (graph-safe, no memset needed per step).
// Head repeat: v-head h uses q/k head h/3.
#pragma once
#include <stddef.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

#define SWIFTMK_DN_NV 48
#define SWIFTMK_DN_DV 128
#define SWIFTMK_DN_DK 128
#define SWIFTMK_DN_CONVD 10240
#define SWIFTMK_DN_MAX_SLOTS 9  // ring slots at tmax = 8; the engine allocates tmax + 1 (smk_engine_create_ex)
#define SWIFTMK_DN_STATE_SLOT_FLOATS (48 * 128 * 128)
#define SWIFTMK_DN_CONV_SLOT_ELEMS (10240 * 3)

size_t swiftmk_deltanet_ws_bytes(void);

// Returns cudaSuccess or the launch error; T must be 1..8 and < slots.
cudaError_t swiftmk_deltanet_step(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                  const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                  void* conv_state, const int* cur, int slots, void* y, void* ws, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
