#include "dense_q6_gemv.cuh"

extern "C" {

// logits[N] (fp32) = W x for the repacked Q6_K rows w of dense_q6_gemv.cuh; one row of x (T = 1), K a multiple of 256. unroll: 1 or 2 blocks of
// 256 columns loaded per lane before use (the caller's schedule).
int warpc_q6_gemv(const void* w, const void* x, int N, int K, int T, void* out, int unroll, cudaStream_t st) {
  if (K % 256 || T != 1 || (unroll != 1 && unroll != 2)) return cudaErrorInvalidValue;
  Q6GemvP p;
  p.w = reinterpret_cast<const uint8_t*>(w);
  p.x = reinterpret_cast<const bf16*>(x);
  p.out = reinterpret_cast<float*>(out);
  p.N = N;
  p.K = K;
  p.rs = ((size_t)(K / 256) * 210 + 15) & ~(size_t)15;
  const int grid = (N + GEMV_WARPS - 1) / GEMV_WARPS;
  if (unroll == 2) k_q6_gemv<2><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
  else k_q6_gemv<1><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
  return cudaGetLastError();
}

}  // extern "C"
