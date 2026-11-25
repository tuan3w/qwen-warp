// The attention launchers and their instantiations. The Makefile compiles this file once per (head dim, part): -DATTN_HD=64|128|256|512 -DATTN_PARTS=<mask> (256: also part 64, 512: parts 1, 2 and 64 only),
// 1 = bf16 split + reduce (warpc_attn_decode), 2 = chunked bf16 (warpc_attn_decode_x), 4 = quantised prep + merge (un-rotate), 8 / 16 = quantised split with 8-bit / 4-bit V,
// 32 = int4_rot prep + split, 64 = the one-row decode of head dims 256 / 512 (prep + split + merge in one launch).
// Every kernel template is instantiated in exactly one of these objects (the instantiations are explicit; the entry points in dense_attn.cu / dense_attn_q.cu only declare the
// launchers). A part includes only the kernel header it needs, so an edit of a bf16 kernel does not recompile the quantised objects (the cache key is the preprocessed text) and vice versa.
#ifndef ATTN_HD
#error "compile with -DATTN_HD=64|128|256 -DATTN_PARTS=mask"
#endif
#ifndef ATTN_PARTS
#define ATTN_PARTS 31
#endif
#include "dense_attn_common.cuh"
#if ATTN_PARTS & 67
#include "dense_attn.cuh"
#endif
#if ATTN_PARTS & 60
#include "dense_attn_q.cuh"
#endif

#if ATTN_PARTS & 1
template <int HD>
void run_decode(dim3 grid, dim3 g2, size_t smem, const bf16* q, const bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, int nsplit, float scale, int window,
                cudaStream_t st) {
  k_attn_split<HD><<<grid, 128, smem, st>>>(q, kv, pos0, ws, Hq, Hkv, ctx, nsplit, scale, window);
  k_attn_reduce<HD><<<g2, 128, 0, st>>>(ws, pos0, o, Hq, nsplit, window);
}
template void run_decode<ATTN_HD>(dim3, dim3, size_t, const bf16*, const bf16*, const int*, bf16*, float*, int, int, int, int, float, int, cudaStream_t);

#if ATTN_HD == 128
template <int HD>
int run_decode1(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, int nsplit,
                float scale, const float* inv_freq, float eps, int qk_norm, cudaStream_t st) {
  const int G = Hq / Hkv;
  const size_t smem = ((size_t)G * (HD + CH) + 2 * HD + 16) * sizeof(float);
  const dim3 grid(Hkv, nsplit);
  if (G <= 2) k_attn_fused<HD, 2><<<grid, 128, smem, st>>>(q, k, v, qn, kn, kv, pos0, ws, Hq, Hkv, ctx, nsplit, scale, inv_freq, eps, qk_norm);
  else if (G <= 4) k_attn_fused<HD, 4><<<grid, 128, smem, st>>>(q, k, v, qn, kn, kv, pos0, ws, Hq, Hkv, ctx, nsplit, scale, inv_freq, eps, qk_norm);
  else k_attn_fused<HD, 8><<<grid, 128, smem, st>>>(q, k, v, qn, kn, kv, pos0, ws, Hq, Hkv, ctx, nsplit, scale, inv_freq, eps, qk_norm);
  k_attn_reduce<HD><<<dim3(Hq, 1), 128, 0, st>>>(ws, pos0, o, Hq, nsplit, 0);
  return cudaGetLastError();
}
template int run_decode1<ATTN_HD>(const bf16*, const bf16*, const bf16*, const bf16*, const bf16*, bf16*, const int*, bf16*, float*, int, int, int, int, float, const float*,
                                  float, int, cudaStream_t);
#endif
#endif

#if ATTN_PARTS & 2
template <int HD>
int run_decode_x(int T, const bf16* q, const bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, float scale, int nc, int window, cudaStream_t st) {
  const int nsplit = attn_nsplit(ctx, window, nc), G = Hq / Hkv;
  if (nsplit > MAX_SPLITS2) return cudaErrorInvalidValue;
  constexpr int DP = HD / 2, KG = 128 / DP > 0 ? 128 / DP : 1;
  const size_t smem = ((size_t)G * (HD + CH) + (size_t)KG * G * HD + 24) * sizeof(float);
  k_attn_split_x<HD><<<dim3(Hkv, nsplit, T), 128, smem, st>>>(q, kv, pos0, ws, Hq, Hkv, ctx, nsplit, nc, scale, window);
  k_attn_reduce2<HD, 0><<<dim3(Hq, T), 128, 0, st>>>(ws, pos0, o, Hq, nsplit, CH * nc, window);
  return cudaGetLastError();
}
template int run_decode_x<ATTN_HD>(int, const bf16*, const bf16*, const int*, bf16*, float*, int, int, int, float, int, int, cudaStream_t);
#endif

#if ATTN_PARTS & 36
template <int HD, int VB, int EX, int R4>
int run_prep_q(int T, const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* qout, uint8_t* kvb, const QKV& L, const int* pos0, int Hq, int Hkv,
                      int rd, const float* inv_freq, float eps, int ctx, int qk_norm, cudaStream_t st) {
  k_attn_prep_q<HD, VB, EX, R4><<<dim3(Hq + 2 * Hkv, T), HD, 0, st>>>(q, k, v, qn, kn, qout, kvb, L, pos0, Hq, Hkv, rd, inv_freq, eps, ctx, qk_norm);
  return cudaGetLastError();
}
#define PQ_INST(VB, EX, R4)                                                                                                                                          \
  template int run_prep_q<ATTN_HD, VB, EX, R4>(int, const bf16*, const bf16*, const bf16*, const bf16*, const bf16*, bf16*, uint8_t*, const QKV&, const int*, int, int, int,       \
                                               const float*, float, int, int, cudaStream_t);
#if ATTN_PARTS & 4
PQ_INST(8, 0, 0) PQ_INST(8, 1, 0) PQ_INST(4, 0, 0) PQ_INST(4, 1, 0)
#endif
#if ATTN_PARTS & 32
PQ_INST(4, 1, 1)
#endif
#undef PQ_INST
#endif

#if ATTN_PARTS & 4
template <int HD, int ROT>
void run_merge_q(int T, const int* pos0, bf16* o, float* ws, int Hq, int nsplit, int kpb, cudaStream_t st) {
  k_attn_reduce2<HD, ROT><<<dim3(Hq, T), 128, 0, st>>>(ws, pos0, o, Hq, nsplit, kpb, 0);
}
template void run_merge_q<ATTN_HD, 1>(int, const int*, bf16*, float*, int, int, int, cudaStream_t);
template void run_merge_q<ATTN_HD, 2>(int, const int*, bf16*, float*, int, int, int, cudaStream_t);
#endif

#if ATTN_PARTS & 56
template <int HD, int VB, int EX, int R4>
int run_split_q(int T, const bf16* q, const uint8_t* kvb, const QKV& L, const int* pos0, float* ws, int Hq, int Hkv, int ctx, float scale, int nc, cudaStream_t st) {
  const int nsplit = (ctx + CH * nc - 1) / (CH * nc), G = Hq / Hkv;
  if (nsplit > MAX_SPLITS2) return cudaErrorInvalidValue;
  const size_t smem = ((size_t)G * HD + (size_t)G * CH + 4 * (size_t)G * HD + 24 + (size_t)CH * (R4 ? HD / 64 : 1)) * sizeof(float);
  if (smem > 48 * 1024) return cudaErrorInvalidValue;
  dim3 grid(Hkv, nsplit, T);
  if (G <= 2) k_attn_split_q<HD, VB, EX, 2, R4><<<grid, 128, smem, st>>>(q, kvb, L, pos0, ws, Hq, Hkv, ctx, nsplit, nc, scale);
  else if (G <= 4) k_attn_split_q<HD, VB, EX, 4, R4><<<grid, 128, smem, st>>>(q, kvb, L, pos0, ws, Hq, Hkv, ctx, nsplit, nc, scale);
  else k_attn_split_q<HD, VB, EX, 8, R4><<<grid, 128, smem, st>>>(q, kvb, L, pos0, ws, Hq, Hkv, ctx, nsplit, nc, scale);
  return cudaSuccess;
}
#define SQ_INST(VB, EX, R4) template int run_split_q<ATTN_HD, VB, EX, R4>(int, const bf16*, const uint8_t*, const QKV&, const int*, float*, int, int, int, float, int, cudaStream_t);
#if ATTN_PARTS & 8
SQ_INST(8, 0, 0) SQ_INST(8, 1, 0)
#endif
#if ATTN_PARTS & 16
SQ_INST(4, 0, 0) SQ_INST(4, 1, 0)
#endif
#if ATTN_PARTS & 32
SQ_INST(4, 1, 1)
#endif
#undef SQ_INST
#endif

#if ATTN_PARTS & 64
template <int HD, int READ>
int run_decode1x(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, bf16* o, float* ws, unsigned* cnt, int Hq, int Hkv, int ctx,
                 float scale, const float* inv_freq, float eps, int qk_norm, int v_norm, int window, cudaStream_t st) {
  const int nsplit = attn_nsplit(ctx, window, 1, DK);
  if (nsplit > MAX_SPLITS2) return cudaErrorInvalidValue;
  const int merge = nsplit <= MERGE_FUSE;
  const size_t smem = decode1_smem<HD>();
  static const cudaError_t attr = cudaFuncSetAttribute(k_attn_decode1<HD, READ>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)decode1_smem<HD>());
  if (attr != cudaSuccess) return attr;
  k_attn_decode1<HD, READ><<<dim3(Hkv, nsplit), HD, smem, st>>>(q, k, v, qn, kn, kv, pos0, ws, o, cnt, Hq, Hkv, ctx, nsplit, scale, inv_freq, eps, qk_norm, v_norm, window, merge);
  if (!merge) k_attn_reduce2<HD, 0><<<dim3(Hq, 1), 128, 0, st>>>(ws, pos0, o, Hq, nsplit, DK, window);
  return cudaGetLastError();
}
template int run_decode1x<ATTN_HD, 0>(const bf16*, const bf16*, const bf16*, const bf16*, const bf16*, bf16*, const int*, bf16*, float*, unsigned*, int, int, int, float, const float*,
                                      float, int, int, int, cudaStream_t);
template int run_decode1x<ATTN_HD, 1>(const bf16*, const bf16*, const bf16*, const bf16*, const bf16*, bf16*, const int*, bf16*, float*, unsigned*, int, int, int, float, const float*,
                                      float, int, int, int, cudaStream_t);
#endif
