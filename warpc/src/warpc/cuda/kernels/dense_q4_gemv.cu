#include "dense_q4_gemv.cuh"

// The fused Q4_0 / Q4_1 dequant GEMV has no norm prologue (no nw / eps): a linear behind an rmsnorm runs the norm as its own task.
template <int FMT, int T, int MODE>
static cudaError_t launch_q4_gemv_t(const Q4GemvP& p, int rpw, cudaStream_t st) {
  const int grid = (p.N + GEMV_WARPS * rpw - 1) / (GEMV_WARPS * rpw);
  if (rpw == 2) warpc_inst_q4_gemv<FMT, T, 2, MODE>(p, grid, st);
  else warpc_inst_q4_gemv<FMT, T, 1, MODE>(p, grid, st);
  return cudaGetLastError();
}
template <int FMT, int MODE>
static cudaError_t launch_q4_gemv_mode(const Q4GemvP& p, int T, int rpw, cudaStream_t st) {
  switch (T) {
    case 1: return launch_q4_gemv_t<FMT, 1, MODE>(p, rpw, st);
    case 2: return launch_q4_gemv_t<FMT, 2, MODE>(p, rpw, st);
    case 3: return launch_q4_gemv_t<FMT, 3, MODE>(p, rpw, st);
    case 4: return launch_q4_gemv_t<FMT, 4, MODE>(p, rpw, st);
    case 5: return launch_q4_gemv_t<FMT, 5, MODE>(p, rpw, st);
    case 6: return launch_q4_gemv_t<FMT, 6, MODE>(p, rpw, st);
    case 7: return launch_q4_gemv_t<FMT, 7, MODE>(p, rpw, st);
    case 8: return launch_q4_gemv_t<FMT, 8, MODE>(p, rpw, st);
  }
  return cudaErrorInvalidValue;
}
template <int FMT>
static cudaError_t launch_q4_gemv_fmt(const Q4GemvP& p, int T, int mode, int rpw, cudaStream_t st) {
  if (mode == 0) return launch_q4_gemv_mode<FMT, 0>(p, T, rpw, st);
  if (mode == 1) return launch_q4_gemv_mode<FMT, 1>(p, T, rpw, st);
  if (mode == 3) return launch_q4_gemv_mode<FMT, 3>(p, T, rpw, st);
  return cudaErrorInvalidValue;
}

extern "C" {

// w: the repacked rows of dense_q4_gemv.cuh, fmt 0 Q4_0, 1 Q4_1. mode 0: segments (end[0..2] cumulative row ends, out[0..2]); 1: gate/up pair
// (w = gate, w2 = up, out[0] bf16 [T][N]); 3: residual add (out[0] = h [T][N] bf16, h = bf16(h + bf16(y)) in place). rpw: rows per warp
// (1, 2; the caller's schedule).
int warpc_q4_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, int fmt, const int* end, void* const* out, int rpw, cudaStream_t st) {
  if (K % 512 || T < 1 || T > STEP_ROWS || (rpw != 1 && rpw != 2) || (fmt != 0 && fmt != 1)) return cudaErrorInvalidValue;
  Q4GemvP p;
  p.w = reinterpret_cast<const uint8_t*>(w);
  p.w2 = reinterpret_cast<const uint8_t*>(w2);
  p.x = reinterpret_cast<const bf16*>(x);
  p.N = N;
  p.K = K;
  p.rs = (size_t)(K / 32) * (fmt ? 20 : 18);
  for (int i = 0; i < 3; i++) {
    p.end[i] = end ? end[i] : N;
    p.out[i] = out[i];
  }
  return fmt ? launch_q4_gemv_fmt<1>(p, T, mode, rpw, st) : launch_q4_gemv_fmt<0>(p, T, mode, rpw, st);
}

}  // extern "C"
