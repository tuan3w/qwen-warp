// swift-mk DSpine: block drafter (5 layers, BF16, causal attention window 8192, code bridge, serial refine) running on the device next to the 64-layer engine.
//
// One captured CUDA graph per cycle:
//   verify : smk_run_step(T = R rows [anchor, d1..d(R-1)], head 1)
//   accept : single-thread kernel: m = leading drafts equal to the verify argmax, a = m + 1
//   stage  : (1) ctx append: fc over the 5 tap layers of the new rows + hidden_norm, K/V of all 5 drafter layers
//            (2) noise block: embed, 5 layers [norm, qkv, qk-norm+RoPE, causal attention (window 8192), o, residual, norm, MLP, residual, code bridge]
//            (3) head over rows 1..R-1 + serial refine tail -> drafts
// Needs an engine with tmax >= R and the I4R quantised head.  Greedy only.
#pragma once
#include "engine.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SMK_SP_NL 5
#define SMK_SP_ROWS 8
#define SMK_SP_DRAFTS 7
#define SMK_SP_RING 4096
#define SMK_SP_LOG 4096
#define SMK_SP_LOGW 18
#define SMK_SP_R 512
#define SMK_SP_C 1024
#define SMK_SP_K 16

// W4A16 tensors: weight int32 in the smk_w4r_gemv repacked layout + bf16 scales (pack.repack_w4), g128 along K.
typedef struct smk_sp_layer_w {
  const void *in_ln, *post_ln;
  const void *qkv_w, *qkv_s;    // W4 [6144][5120]  (q 4096 | k 1024 | v 1024 rows)
  const void *q_norm, *k_norm;  // bf16 [128]
  const void *o_w, *o_s;        // W4 [5120][4096]
  const void *gu_w, *gu_s;      // W4 [34816][5120]  (gate / up interleaved in groups of 8 rows)
  const void *dn_w, *dn_s;      // W4 [5120][17408]
  const void *write_w;          // bf16 [5120][512]
} smk_sp_layer_w;

typedef struct smk_sp_w {
  smk_sp_layer_w layers[SMK_SP_NL];
  const void *fc_w, *fc_s;      // W4 [5120][25600]
  const void *hidden_norm, *norm;  // bf16 [5120]
  const void *kv_w, *kv_s;      // W4 [10240][5120]  (per layer: k_proj 1024 rows | v_proj 1024 rows)
  // code bridge
  const void *codebook;         // bf16 [248320][1024]
  const void *read_w_last;      // bf16 [1024][5120]
  const void *early_read;       // bf16 [4][1024][5120]
  const void *gate_w;           // bf16 [512][1024]  (actually [1536][1024] = gate|pred_gate|prev packed)
  const float *gate_b;          // fp32 [512]
  const float *inv_freq;        // fp32 [64]
  int mask_id;
} smk_sp_w;

typedef struct smk_dspine smk_dspine;
smk_dspine* smk_dspine_create(smk_engine* e, const smk_sp_w* w);
void smk_dspine_destroy(smk_dspine* s);
cudaError_t smk_dspine_reset(smk_dspine* s, cudaStream_t st);
cudaError_t smk_dspine_prefill_chunk(smk_dspine* s, int T, int pos0, cudaStream_t st);
cudaError_t smk_dspine_start(smk_dspine* s, cudaStream_t st);
#define SMK_SP_SS_N 64
cudaError_t smk_dspine_run(smk_dspine* s, int n, cudaStream_t st, int* host_ss, int* host_ring);
cudaError_t smk_dspine_launch(smk_dspine* s, int which, int n, cudaStream_t st);
cudaError_t smk_dspine_read(smk_dspine* s, int what, void* host, size_t bytes);
size_t smk_dspine_bytes(smk_dspine* s);
int smk_dspine_ctxd(smk_dspine* s);
void* smk_dspine_kv_ptr(smk_dspine* s, int which);

#ifdef __cplusplus
}
#endif
