// swift-mk K9: quantised lm_head + exact BF16 rescoring.  Contract: docs/ENGINE.md, evidence experiments/2026-10-31-lm-head-quantisation-quality/.
//
//   1. approximate logits  logits[t][n] (fp32, all N = 248320 rows, T <= 8 rows; kind 1: T <= 4) from a quantised copy of the lm_head
//        kind 1: int8 per-row   wq int8 [N][K], sq fp32 [N]    (logit = sq[n] * sum_k wq[n][k] * x[k]),  CUDA-core kernel written here
//        kind 2: int4 g128      wq uint32 [N][K/8] + sq bf16 [N][K/128] (engine W4 format, nibble - 8), smk_w4_gemv mma kernel with SMK_EPI_F32
//   2. top-R candidates of every row by approximate logit (R <= 64; exact top-R of the approximate logits, ties -> lower index)
//   3. the R candidates' logits recomputed EXACTLY from the BF16 head (fp32 accumulate of exact bf16 products), written over the approximate values of
//      logits[t][cand] (so the logits buffer = exact BF16-head logits on the candidate set, approximate elsewhere), and argmax over the exact candidate
//      logits (lowest index wins ties) -> out_tok.  The argmax equals the BF16 head's argmax whenever the BF16 top-1 is in the candidate set.
// All calls are stream-ordered and CUDA-graph capturable (no host sync, no allocation); `ws` = smk_headq_ws_bytes() bytes of device scratch, zero-initialised once.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

#define SMK_HEADQ_NONE 0
#define SMK_HEADQ_I8 1
#define SMK_HEADQ_I4 2
#define SMK_HEADQ_I4R 3  // kind 2 in the repacked W4 layout of smk_w4r_gemv (include/engine.h; the engine's own copy, pack.repack_w4)
#define SMK_HEADQ_RMAX 64

size_t smk_headq_ws_bytes(void);
cudaError_t smk_headq_gemv(int kind, const void* wq, const void* sq, const void* x, int N, int K, int T, float* logits, cudaStream_t st);
// Select + rescore (steps 2 and 3).  N = 248320 (any N <= 2^24 with N % 4 == 0 works), K % 256 == 0, 1 <= R <= SMK_HEADQ_RMAX, T <= 8.
// out_tok[t] (device int) = argmax token, out_val[t] (optional) its exact logit.
cudaError_t smk_headq_select_rescore(float* logits, const void* w_bf16, const void* x, int N, int K, int T, int R, int* out_tok, float* out_val, void* ws, cudaStream_t st);
// Candidate ids of the last select_rescore: device int [8][SMK_HEADQ_RMAX] inside ws (row t, rank r; -1 = unused slot), exact logits float [8][SMK_HEADQ_RMAX].
const int* smk_headq_ws_cand(const void* ws);
const float* smk_headq_ws_exact(const void* ws);

#ifdef __cplusplus
}
#endif
