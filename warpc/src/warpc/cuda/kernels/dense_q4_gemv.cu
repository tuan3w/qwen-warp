#include "dense_q4_gemv.cuh"

// The fused Q4_0 dequant GEMV has no norm prologue (no nw / eps): a Q4_0 linear behind an rmsnorm runs the norm as its own task.
template <int T, int MODE>
static cudaError_t launch_q4_gemv_t(const Q4GemvP& p, int rpw, cudaStream_t st) {
  const int grid = (p.N + GEMV_WARPS * rpw - 1) / (GEMV_WARPS * rpw);
  if (rpw == 4) warpc_inst_q4_gemv<T, 4, MODE>(p, grid, st);
  else if (rpw == 2) warpc_inst_q4_gemv<T, 2, MODE>(p, grid, st);
  else warpc_inst_q4_gemv<T, 1, MODE>(p, grid, st);
  return cudaGetLastError();
}
template <int MODE>
static cudaError_t launch_q4_gemv_mode(const Q4GemvP& p, int T, int rpw, cudaStream_t st) {
  switch (T) {
    case 1: return launch_q4_gemv_t<1, MODE>(p, rpw, st);
    case 2: return launch_q4_gemv_t<2, MODE>(p, rpw, st);
    case 3: return launch_q4_gemv_t<3, MODE>(p, rpw, st);
    case 4: return launch_q4_gemv_t<4, MODE>(p, rpw, st);
    case 5: return launch_q4_gemv_t<5, MODE>(p, rpw, st);
    case 6: return launch_q4_gemv_t<6, MODE>(p, rpw, st);
    case 7: return launch_q4_gemv_t<7, MODE>(p, rpw, st);
    case 8: return launch_q4_gemv_t<8, MODE>(p, rpw, st);
  }
  return cudaErrorInvalidValue;
}

extern "C" {

// mode 0: segments (end[0..2] cumulative row ends, out[0..2]); 1: gate/up pair (w = gate, w2 = up, out[0] bf16 [T][N]);
// 3: residual add (out[0] = h [T][N] bf16, h = bf16(h + bf16(y)) in place). rpw: rows per warp (1, 2, 4; the caller's schedule).
int warpc_q4_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, const int* end, void* const* out, int rpw, cudaStream_t st) {
  if (K % 32 || T < 1 || T > STEP_ROWS || (rpw != 1 && rpw != 2 && rpw != 4)) return cudaErrorInvalidValue;
  Q4GemvP p;
  p.w = reinterpret_cast<const uint8_t*>(w);
  p.w2 = reinterpret_cast<const uint8_t*>(w2);
  p.x = reinterpret_cast<const bf16*>(x);
  p.N = N;
  p.K = K;
  for (int i = 0; i < 3; i++) {
    p.end[i] = end ? end[i] : N;
    p.out[i] = out[i];
  }
  if (mode == 0) return launch_q4_gemv_mode<0>(p, T, rpw, st);
  if (mode == 1) return launch_q4_gemv_mode<1>(p, T, rpw, st);
  if (mode == 3) return launch_q4_gemv_mode<3>(p, T, rpw, st);
  return cudaErrorInvalidValue;
}

}  // extern "C"
