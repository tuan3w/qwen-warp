// swift-mk K3: engine core ops + the 64-layer step.  Pure C API, raw device pointers, every call takes a cudaStream_t and is graph-capturable.
// Contract: docs/ENGINE.md.   bf16 = __nv_bfloat16 passed as void*.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif

// K11: rows per step buffer = 8 (T = 1..8).  The DeltaNet ring has tmax+1 slots, tmax chosen at engine creation (4 = the K4..K10 behaviour, 8 = hybrid MTP + prompt-lookup tail).
#ifndef SMK_TMAX
#define SMK_TMAX 8
#endif
#define SMK_HIDDEN 5120
#define SMK_INTER 17408
#define SMK_VOCAB 248320
#define SMK_NLAYERS 64

// ------------------------------------------------------------------------------------------------ W4 GEMV (mma, T <= 8)
// Weight layout = checkpoint natural: w uint32 [N][K/8] (nibble e of word j = column 8j+e, value (n-8)*scale), scale bf16 [N][K/128].
// N % 16 == 0, K % 128 == 0.  x = bf16 [T][K] natural row-major (the kernel permutes in registers).  fp32 accumulate, one bf16 rounding.
//
// Epilogues (smk_epi.kind):
//   SMK_EPI_SEGS  : up to 3 output segments by row range.  Segment s covers rows [end[s-1], end[s]) and is written bf16 row-major
//                   [T][end[s]-end[s-1]] to out[s].  (qkv|z -> qkv[T][10240], z[T][6144];  q|k|v -> [T][12288],[T][1024],[T][1024].)
//                   Segment boundaries must be multiples of 16.  Unused segments: end[s] = end[s-1] (or N), out[s] = NULL.
//   SMK_EPI_SILU  : rows are interleaved per 16-row tile: rows 16i..16i+7 = gate channels 8i..8i+7, rows 16i+8..16i+15 = up channels 8i..8i+7
//                   (see smk_interleave_gateup).  Writes bf16 [T][N/2] = bf16(bf16(silu(bf16(gate))) * bf16(up))  (HF rounding chain).
//   SMK_EPI_F32   : fp32 [T][N] (logits).
enum { SMK_EPI_SEGS = 0, SMK_EPI_SILU = 1, SMK_EPI_F32 = 2 };
typedef struct smk_epi {
  int kind;
  int end[3];
  void* out[3];
} smk_epi;

cudaError_t smk_w4_gemv(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st);
// Same with an explicit kernel configuration (tuning/bench): 0 w8k4u2, 1 w8k8u2, 2 w8k2u3, 3 w4k4u3, 4 w16k8u2, 5 w8k1u3 (warps/CTA, K-split warps, prefetch depth).
// Returns cudaErrorInvalidValue if (N, K) is not divisible for that config (K/128 % KS, N/16 % (warps/KS)).
cudaError_t smk_w4_gemv_cfg(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st);
// Same kernels on the REPACKED layout the engine / MTP / head / prefill paths use (pack.py repack_w4): per (16-row tile, 128-k group) one contiguous 1 KB unit
// [lane 0..31: row g, 16 B chunk t][lane 32..63: row g+8], units of a tile contiguous; inside a word nibble j / j+4 = columns 2j / 2j+1;
// scale = uint32 per (tile, group, g) = bf16(row g) | bf16(row g+8) << 16.  Same byte counts as the natural layout.
cudaError_t smk_w4r_gemv(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, cudaStream_t st);
cudaError_t smk_w4r_gemv_cfg(const void* w, const void* scale, const void* x, int N, int K, int T, const smk_epi* epi, int cfg, cudaStream_t st);


// BF16 weights [N][K] (K % 256 == 0), fp32 logits out [T][N]: CUDA-core kernel for T == 1, mma for T in 2..8.
cudaError_t smk_bf16_gemv_f32(const void* w, const void* x, int N, int K, int T, float* out, cudaStream_t st);

// Small BF16 GEMV for in_proj_a|b: w bf16 [N][K] (rows 0..na-1 -> out_a [T][na], rows na.. -> out_b [T][N-na]); one warp per row, T <= 8.
cudaError_t smk_bf16_gemv_small(const void* w, const void* x, int N, int na, int K, int T, void* out_a, void* out_b, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ norms / embed / sampling
// Zero-centred RMSNorm over H (multiple of 8), T rows, one block per row:
//   if y != NULL:  h[t] = bf16(h[t] + y[t])   (in place: the residual stream)
//   xn[t] = bf16( h[t] * rsqrt(mean(h[t]^2) + eps) * (1 + w) )          (fp32 math, single rounding)
//   dbg (optional): copy of the new h[t] to dbg[t*H ..]  (caller offsets per layer)
cudaError_t smk_rmsnorm(void* h, const void* y, const void* w, void* xn, void* dbg, int T, int H, float eps, cudaStream_t st);

// out[t] = embed[tok[t]]  (bf16 rows of length H); tok is a DEVICE int array.
cudaError_t smk_embed(const void* embed, const int* tok, void* out, int T, int H, cudaStream_t st);

// argmax over logits fp32 [T][N] -> out_tok[t] (lowest index wins ties) and out_val[t] (optional, may be NULL).  ws: device scratch of smk_argmax_ws_bytes()
// bytes that MUST be zero-initialised once (the kernel re-zeroes it; graph-safe).
size_t smk_argmax_ws_bytes(void);
cudaError_t smk_argmax(const float* logits, int T, int N, int* out_tok, float* out_val, void* ws, cudaStream_t st);

// (top-k: planned, not implemented yet.)

// ------------------------------------------------------------------------------------------------ packer helpers (device side)
// dst rows [2*8*i .. 2*8*i+7] = src_gate rows 8i..8i+7, dst rows [16i+8 .. 16i+15] = src_up rows 8i..8i+7.  `row_bytes` = bytes per row
// (w: K/2, scale: K/64).  Gate and up have N rows each.
cudaError_t smk_interleave_gateup(const void* gate, const void* up, void* dst, int N, size_t row_bytes, cudaStream_t st);

// ------------------------------------------------------------------------------------------------ 64-layer engine (stage 2)
// Weights are owned by the caller (python/torch); the engine only keeps the pointers (they must outlive the engine).  Layouts: pack.py.
typedef struct smk_layer_w {
  int kind;                                  // 0 = DeltaNet (linear attention), 1 = full attention
  const void *ln1, *ln2;                     // bf16 [5120] zero-centred input / post-attention norms
  const void *qkv_w, *qkv_s;                 // W4 fused rows: DeltaNet [qkv 10240 | z 6144] = 16384, attention [q 12288 | k 1024 | v 1024] = 14336; K = 5120
  const void *ab;                            // DeltaNet: bf16 [96][5120] (in_proj_a rows 0..47, in_proj_b rows 48..95)
  const void *conv_w, *A_log, *dt_bias, *norm_w;  // DeltaNet: bf16 [10240][4], [48], [48], [128]
  const void *q_norm, *k_norm;               // attention: bf16 [256] zero-centred
  float k_scale, v_scale;                    // attention: fp8 KV scales (used only if kv_fp8)
  const void *o_w, *o_s;                     // W4 [5120][6144] (out_proj / o_proj)
  const void *gu_w, *gu_s;                   // W4 [34816][5120] gate|up interleaved per 8 rows (smk_interleave_gateup layout)
  const void *dn_w, *dn_s;                   // W4 [5120][17408] down_proj
} smk_layer_w;

typedef struct smk_model_w {
  smk_layer_w layers[SMK_NLAYERS];
  const void* embed;       // bf16 [248320][5120]
  const void* final_norm;  // bf16 [5120] zero-centred
  const void* lm_head;     // bf16 [248320][5120]
  // K9 (include/head_q.h): quantised copy of lm_head used for the approximate logits (the BF16 head above stays resident for the exact rescoring of the top candidates).
  int head_q_kind;         // SMK_HEADQ_NONE (0: BF16 head only), SMK_HEADQ_I8 (1: int8 [N][K] + fp32 scale [N]), SMK_HEADQ_I4 (2: W4 uint32 [N][K/8] + bf16 scale [N][K/128]), SMK_HEADQ_I4R (3: the same in the repacked layout)
  const void *head_q, *head_qs;
} smk_model_w;

typedef struct smk_engine smk_engine;
// Allocates state (DeltaNet state ring of tmax+1 slots per layer, conv state, KV cache for ctx_max positions, activations, graph workspaces).
// smk_engine_create = smk_engine_create_ex with tmax = env SMK_TMAX (default 4: 5 slots, steps of T <= 4 rows); tmax in 4..SMK_TMAX (8: 9 slots, +0.62 GiB, steps of T <= 8).
smk_engine* smk_engine_create(const smk_model_w* w, int ctx_max, int kv_fp8);
smk_engine* smk_engine_create_ex(const smk_model_w* w, int ctx_max, int kv_fp8, int tmax);
int smk_engine_tmax(smk_engine* e);
void smk_engine_destroy(smk_engine* e);
// Zero all sequence state; pos0 = 0, cur = 0.
cudaError_t smk_engine_reset(smk_engine* e, cudaStream_t st);
// Stage token ids (host array, T <= tmax) as the input rows of the next step (stream-ordered H2D copy).
cudaError_t smk_engine_set_tokens(smk_engine* e, const int* tok, int T, cudaStream_t st);
// One step over T rows (T in 1..tmax) as a CUDA graph (captured on first use, then replayed; all per-step scalars live on the device).
//   head = 0: skip final norm / lm_head / argmax (prefill chunks).   head = 1: also computes logits and argmax of every row -> out_tok.
//   head = 2: final norm only (rows' post-norm hidden -> debug tap slot 65; MTP prefill chunks, see mtp.h).
//   Afterwards pos0 += T, cur = (cur+T)%(tmax+1) on the device; if head && T == 1 the sampled token is written back as the next input token
//   (so smk_engine_step(T=1, head=1) can be replayed back to back with no host work).
cudaError_t smk_engine_step(smk_engine* e, int T, int head, cudaStream_t st);
// K9: head = 1 steps use the quantised head + exact top-R rescoring (1, default when smk_model_w.head_q_kind != 0) or the plain BF16 head + argmax (0).  The logits buffer
// (smk_engine_read what = 2) holds exact BF16-head logits on the rescored candidates and approximate logits elsewhere (quantised head).  Drops cached graphs.
cudaError_t smk_engine_set_head(smk_engine* e, int quant);
int smk_engine_head(smk_engine* e);
// Tail of a T=1 head step on a residual stream the caller already placed in the engine: expects e->h / e->dn row 0 = the last row's residual / MLP output (as after the 64th layer
// of smk_engine_step); runs final RMSNorm (-> debug tap NLAYER+1 = the hidden the MTP stage reads) + lm_head (BF16, or the K9 quantised head + exact rescoring; rsamp != 0: R = 64
// candidates, i.e. sampling) + argmax -> device ds[OUT].  No pos / cur advance, no token feedback.  Stream-ordered, capturable.  Used by the fast prefill (prefill.cu).
cudaError_t smk_engine_head_tail(smk_engine* e, int rsamp, cudaStream_t st);
// Same work launched as individual kernels (no graph): for profiling / debugging.
cudaError_t smk_engine_step_nograph(smk_engine* e, int T, int head, cudaStream_t st);
// Greedy decode: n replays of the T=1 head graph, host out_tok[0..n-1] filled (synchronises).
cudaError_t smk_engine_decode(smk_engine* e, int n, int* host_tokens, cudaStream_t st);
// Synchronous copy of an engine debug buffer to host memory.  what = 0: device state int[20] (pos0, cur, tok[8], out_tok[8], ring slots, pad);
// 1: debug taps bf16 [66][SMK_TMAX][5120] (always written; slot 0 = embedding, slot l+1 = hidden after layer l, slot 65 = final-norm output; rows 0..T-1 of the last step);
// 2: logits fp32 [SMK_TMAX][248320] of the last head step.  `bytes` must be <= the buffer size.
cudaError_t smk_engine_read(smk_engine* e, int what, void* host, size_t bytes);
size_t smk_engine_bytes(smk_engine* e);          // device bytes allocated by the engine

#ifdef __cplusplus
}
#endif
