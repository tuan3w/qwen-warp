// swift-mk DFlash2: block drafter (5 layers, W4A16, grouped dynamic conv, candidate selector) running on the device next to the 64-layer engine.  Contract: docs/ARCHITECTURE.md (DFlash2 section).
//
// One captured CUDA graph per cycle:
//   verify : smk_run_step(T = 8 rows [anchor, d1..d7], head 1 = quantised head + exact rescoring, spec 1)
//   accept : single-thread kernel: m = leading drafts equal to the verify argmax of the previous row, a = m + 1; spec_finish bookkeeping (pos, DeltaNet ring, anchor), output ring,
//            per-cycle log, the drafter's "new context rows" state (rows a at positions P..P+a-1 = the rows of the verify step whose taps are the new context)
//   stage  : (1) ctx append: fc over the 5 tap layers of the new rows (<= 8, padded) + hidden_norm, K/V of all 5 drafter layers (ONE fused GEMV, the ctx path does not depend on the layer input)
//                -> written positionally into the drafter KV cache (rows beyond the committed length are overwritten by the next cycle, like the target KV);
//            (2) noise block: embed([anchor, mask x 7]) (the target's embedding table), 5 layers [norm, dynamic conv, q/k/v, qk-norm, RoPE, attention over cache ++ the 8 noise rows (window 2048,
//                non-causal), o, conv, residual, norm, conv, MLP, conv, residual], final norm; the noise K/V are written at the positions after the committed rows (scratch, overwritten next cycle);
//            (3) quantised head over rows 1..7 + exact bf16 rescoring of the top-R, top-16 candidates per row, the selector's greedy path -> drafts d1..d7 = ds[DS_TOK + 1 ..].
// Needs an engine with tmax = 8 and the I4R quantised head.  Greedy only (sampling is not implemented).
#pragma once
#include "engine.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SMK_DF_NL 5
#define SMK_DF_ROWS 8        // block size: 1 anchor + 7 mask rows
#define SMK_DF_DRAFTS 7
#define SMK_DF_RING 4096
#define SMK_DF_LOG 4096      // per-cycle log entries (ring)
#define SMK_DF_LOGW 18       // per entry: [0] a, [1..8] block (anchor, d1..d7), [9..16] verify argmax y0..y7, [17] position P of the anchor

typedef struct smk_df_layer_w {
  const void *in_ln, *post_ln;            // bf16 [5120]  RMSNorm weights (plain: x * w, NOT zero-centred)
  const void *qkv_w, *qkv_s;              // W4r [6144][5120]  q (4096) | k (1024) | v (1024)
  const void *q_norm, *k_norm;            // bf16 [128]
  const void *o_w, *o_s;                  // W4r [5120][4096]
  const void *gu_w, *gu_s;                // W4r [34816][5120] gate|up interleaved per 8 rows (SILU epilogue layout)
  const void *dn_w, *dn_s;                // W4r [5120][17408]
  const void *aconv_kp, *mconv_kp;        // bf16 [1280][5120] conv kernel projections (attention / MLP path)
  const void *aconv_base, *mconv_base;    // bf16 [2][2][5120]  [prepare | finish][offset][channel]
} smk_df_layer_w;

typedef struct smk_df_w {
  smk_df_layer_w layers[SMK_DF_NL];
  const void *fc_w, *fc_s;                // W4r [5120][25600]  over the concatenated taps of target layers 5, 19, 33, 47, 61
  const void *hidden_norm, *norm;         // bf16 [5120]
  const void *kv_w, *kv_s;                // W4r [10240][5120]: layer l rows [2048 l, 2048 l + 1024) = k_proj, [2048 l + 1024, 2048 (l + 1)) = v_proj
  const void *sel_hp;                     // bf16 [256][5120]  candidate selector hidden projection
  const void *pred_cb, *succ_cb;          // bf16 [248320][256] (device or pinned host memory)
  const float* inv_freq;                  // device float [64]  RoPE inverse frequencies (theta 1e7, head dim 128)
  int mask_id;                            // 248070
  int rescore_r;                          // exact-rescoring set of the quantised head: 16..64
} smk_df_w;

typedef struct smk_dflash smk_dflash;
smk_dflash* smk_dflash_create(smk_engine* e, const smk_df_w* w);
void smk_dflash_destroy(smk_dflash* s);
// Zero the state, the output ring, the log and the drafter KV cache (call together with smk_engine_reset).
cudaError_t smk_dflash_reset(smk_dflash* s, cudaStream_t st);
// Prefill: after smk_engine_step(T <= 8 rows, head 0) of prompt rows at positions pos0 .. pos0 + T - 1: append their taps as drafter context.
cudaError_t smk_dflash_prefill_chunk(smk_dflash* s, int T, int pos0, cudaStream_t st);
// After the final prompt row ran as smk_engine_step(1, head = 1): append its tap and draft d1..d7 (-> ds[DS_TOK + 1 ..]).
// Prompt tokens by position for the device n-gram lookup (pageable host memory is staged by the driver; call after reset, before start).
cudaError_t smk_dflash_set_hist(smk_dflash* s, const int* ids, int n, cudaStream_t st);
cudaError_t smk_dflash_start(smk_dflash* s, cudaStream_t st);
// n full cycles (graph replays), then synchronise and copy the state ints (SMK_DF_SS_N) and the output ring: ss[12] = tokens emitted, ss[13] = cycles, ss[16 + m] = cycles with m accepted drafts.
#define SMK_DF_SS_N 64
cudaError_t smk_dflash_run(smk_dflash* s, int n, cudaStream_t st, int* host_ss, int* host_ring);
// Timing parts (async launches of n replays): which 0 = full cycle, 1 = verify + accept, 2 = drafter stage only (ctx append + noise block + head + selector), 3 = ctx append only,
// 4 = noise block only (layers + final norm), 5 = head + selector only.  State evolves, the cost is state independent.
cudaError_t smk_dflash_launch(smk_dflash* s, int which, int n, cudaStream_t st);
// Debug copies: what 0: ss ints [64]; 1: noise-block output (final norm) bf16 [8][5120]; 2: per-cycle log int [SMK_DF_LOG][SMK_DF_LOGW]; 3: hproj fp32 [8][256]; 4: drafter K cache layer 0..4 bf16
// [5][8][ctxd][128] (V follows); 5: candidates (top-16 ids int [8][16], unary fp32 [8][16]); 6: th (post hidden_norm) bf16 [8][5120].
cudaError_t smk_dflash_read(smk_dflash* s, int what, void* host, size_t bytes);
size_t smk_dflash_bytes(smk_dflash* s);
int smk_dflash_ctxd(smk_dflash* s);

#ifdef __cplusplus
}
#endif
