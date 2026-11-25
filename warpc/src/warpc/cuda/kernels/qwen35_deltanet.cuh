// Gated DeltaNet decode step (family 6): one launch per layer for T = 1..8 rows.
//
// Fuses causal conv1d + SiLU on q|k|v (fp32 through, no bf16 rounding between: the hand engine's numerics), L2 norm of q and k,
// g = -exp(A_log) softplus(a + dt_bias), beta = sigmoid(b), the delta rule S = exp(g) S; d = (v - S^T k) beta; S += k (x) d; o = S^T q on an
// fp32 state, and the gated RMSNorm y = bf16((w * (o * rsqrt(mean(o^2) + eps))) * silu(z)).
//
// Layouts (graph layouts, all in place):
//   qkv [T, C] bf16 (q | k | v), C = 2 Hk dk + Hv dv; z [T, Hv dv]; a, b [T, Hv]; conv_w [C, W]; A_log, dt_bias [Hv]; norm_w [dv] bf16
//   state [Hv, dk, dv] fp32 (the recurrence's S, v contiguous), conv state [C, W-1] bf16 (last W-1 raw inputs, oldest first)
//   y [T, Hv dv] bf16.  v-head h reads q / k head h / (Hv / Hk).
//   ws  zero-initialised persistent scratch of warpc_deltanet_ws_bytes() bytes (shared by layers on one stream; the kernel leaves
//       its counters zero again).
// The state is read once and written once per launch whatever T is: the T rows run sequentially on the state held in registers.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dn {
constexpr int TMAX = 8;

struct Args {
  const __nv_bfloat16 *qkv, *z, *a, *b, *conv_w, *A_log, *dt_bias, *norm_w;
  float* state;
  __nv_bfloat16 *conv, *y;
  unsigned *cnt_head, *cnt_kh;  // arrival counters: v heads, q/k heads
  float* o_ws;                  // [TMAX][Hv][dv] fp32: the unnormalised o of every CTA slice, read by the head's last CTA
  int C, Hk, Hv, rep;
  float eps, scale, norm_eps;
};

// Lanes per column of the S^T k / S^T q sums: the hand engine's V-slices per head (deltanet.cu: 16 for 1-2 rows, 8 for 3 and
// more).  The sums have the hand's trees, which is what makes the two kernels bitwise equal.
__host__ __device__ constexpr int lpr_for(int T) { return T <= 2 ? 16 : 8; }

// Warps per CTA (a warp owns 128 / LPR columns of dv), clamped to what the shape allows.  Measured on the 3090, us per launch over
// 48 DRAM-resident layers, 1 / 2 / 4 warps: 1 row 10.7 / 10.4 / 10.6, 2 rows 12.4 / 11.6 / 11.4, 4 rows 16.4 / 14.5 / 14.3,
// 8 rows 24.4 / 20.0 / 18.0 (fewer CTAs repeat less of the per-head q / k conv prologue).
constexpr int NW_TUNED = 4;
constexpr int nw_for(int T, int dk, int dv) {
  int n = NW_TUNED;
  while (n > 1 && (dv % (128 / lpr_for(T) * n) || (dk / 32) % n || (dv / 32) % n)) n >>= 1;
  return n;
}

// The (dk, dv, W) configurations the library instantiates, and the dispatch to the template of a T.
#define DN_CONFIGS(X) X(128, 128, 4)

template <int T, int NW, int DK, int DV, int W>
cudaError_t launch(const Args& a, cudaStream_t s);
}  // namespace dn
