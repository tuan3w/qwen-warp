// swift-mk K1: gated DeltaNet decode (T = 1..8 rows) for one linear-attention layer.  See docs/ARCHITECTURE.md for the contract.
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
#define SWIFTMK_DN_LAZY_ROWS 16  // rows of the per-layer delta buffer of the lazy engine entry

size_t swiftmk_deltanet_ws_bytes(void);

// Returns cudaSuccess or the launch error; T must be 1..8 and < slots.
cudaError_t swiftmk_deltanet_step(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                  const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                  void* conv_state, const int* cur, int slots, void* y, void* ws, cudaStream_t stream);

// ---- Engine entry with lazy ring snapshots (K12).  A step with T >= 2 does not write its T per-row state snapshots to the ring; it appends the rank-1 updates of its rows to a
// per-layer delta buffer (swiftmk_deltanet_dbuf_floats() floats, SWIFTMK_DN_LAZY_ROWS rows).  Device scalar *pend = p | base << 8 | half << 12 (p = pending rows, base = ring slot of a full
// state, p == 0: slot cur itself is the state): the state at slot cur is "slot base + the first p delta rows".  Whoever advances cur by `a` accepted rows after a T >= 2
// step does *pend = dn_pend_after(...) (engine_impl.h); after a T == 1 step (full snapshot written to slot cur+1) *pend = 0.  A step replays the pending rows bitwise like the
// recurrence did.  Anything that reads or writes the state ring from outside (snapshot, prefill) calls swiftmk_deltanet_materialize first: it writes the pending state of
// all layers to slot cur and clears *pend.
size_t swiftmk_deltanet_dbuf_floats(void);
cudaError_t swiftmk_deltanet_step_lazy(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                       const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                       void* conv_state, const int* cur, int slots, void* y, void* ws, float* dbuf, const int* pend,
                                       cudaStream_t stream);
// Lazy entry with fewer physical state slots and optional pointer indirection (batched decode).
//   sslots (1..slots): ring slot i keeps its fp32 state in state slot i % sslots; the conv ring keeps `slots` slots.  The lazy entry never needs two states of a ring at once (a
//   T >= 2 step writes no row state, a flush writes the replayed state of slot base to slot cur, a T == 1 step writes slot cur+1 from slot base + pending rows, and every one of these
//   writes is an in-place update by the thread that read the element), so sslots = 1 is exact and saves (slots - 1) x 151 MiB per sequence.  swiftmk_deltanet_step_lazy = sslots == slots.
//   ind (optional, may be NULL): the state / conv / delta-buffer base pointers are *ind.x_pp + ind.x_off (device memory read at launch) instead of state / conv_state / dbuf, so one
//   captured graph serves any sequence whose pointers the caller writes there before the launch.
typedef struct swiftmk_dn_ind {
  float* const* state_pp;
  void* const* conv_pp;  // bf16 elements
  float* const* dbuf_pp;
  size_t state_off, conv_off, dbuf_off;  // in elements of the pointee type
} swiftmk_dn_ind;
cudaError_t swiftmk_deltanet_step_lazy2(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                        const void* A_log, const void* dt_bias, const void* norm_w, float* state,
                                        void* conv_state, const int* cur, int slots, int sslots, void* y, void* ws, float* dbuf, const int* pend,
                                        const swiftmk_dn_ind* ind, cudaStream_t stream);
// Fused multi-sequence lazy step (batched decode): B sequences of T <= 8 rows each (B * T <= 16) in ONE launch, every sequence computing exactly what its own lazy2 step would.
// Sequence i: rows i * T .. i * T + T - 1 of qkv / z / a / b / y, ring scalars cur[i * seq_stride] / pend[i * seq_stride], base pointers ind->x_pp[i * pp_stride] (+ x_off;
// required when B > 1).  ws is shared (each sequence uses its own counters and scratch rows).
cudaError_t swiftmk_deltanet_step_multi(int B, int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w,
                                        const void* A_log, const void* dt_bias, const void* norm_w, float* state, void* conv_state, const int* cur,
                                        int seq_stride, int slots, int sslots, void* y, void* ws, float* dbuf, const int* pend,
                                        const swiftmk_dn_ind* ind, int pp_stride, cudaStream_t stream);
// state = [nlayers][sslots][48][128][128] fp32 (slots = the PHYSICAL state slots, ring slot cur -> cur % slots), dbuf = [nlayers][swiftmk_deltanet_dbuf_floats()]
cudaError_t swiftmk_deltanet_materialize(float* state, const float* dbuf, int nlayers, const int* cur, int* pend, int slots, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
