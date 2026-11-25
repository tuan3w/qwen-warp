#include "inkling_gemv.cuh"

template <int T, int MODE>
static cudaError_t launch_ig_t(const InkGemvP& p, int rpw, cudaStream_t st) {
  const int grid = (p.N + GEMV_WARPS * rpw - 1) / (GEMV_WARPS * rpw);
  if (rpw == 4) warpc_inst_ink_gemv<T, 4, MODE>(p, grid, st);
  else if (rpw == 2) warpc_inst_ink_gemv<T, 2, MODE>(p, grid, st);
  else warpc_inst_ink_gemv<T, 1, MODE>(p, grid, st);
  return cudaGetLastError();
}
template <int MODE>
static cudaError_t launch_ig(const InkGemvP& p, int T, int rpw, cudaStream_t st) {
  switch (T) {
    case 1: return launch_ig_t<1, MODE>(p, rpw, st);
    case 2: return launch_ig_t<2, MODE>(p, rpw, st);
    case 3: return launch_ig_t<3, MODE>(p, rpw, st);
    case 4: return launch_ig_t<4, MODE>(p, rpw, st);
    case 5: return launch_ig_t<5, MODE>(p, rpw, st);
    case 6: return launch_ig_t<6, MODE>(p, rpw, st);
    case 7: return launch_ig_t<7, MODE>(p, rpw, st);
    case 8: return launch_ig_t<8, MODE>(p, rpw, st);
  }
  return cudaErrorInvalidValue;
}

static bool ig_args_ok(int N, int K, int T, int rpw) {
  return N > 0 && K > 0 && K % 256 == 0 && T >= 1 && T <= STEP_ROWS && (rpw == 1 || rpw == 2 || rpw == 4);
}

extern "C" {

// Segments of one weight [N][K]: nseg <= IG_SEGS, end[s] the cumulative row ends (end[nseg - 1] = N, each a multiple of rpw), x[s] the bf16 input [T][K] of
// segment s (pointers may repeat), out[s] its bf16 output [T][end[s] - end[s - 1]]. rpw: rows per warp (1, 2, 4).
int warpc_ink_gemv(const void* w, int N, int K, int T, int nseg, const int* end, const void* const* x, void* const* out, int rpw, cudaStream_t st) {
  if (!ig_args_ok(N, K, T, rpw) || nseg < 1 || nseg > IG_SEGS || end[nseg - 1] != N) return cudaErrorInvalidValue;
  InkGemvP p = {};
  p.w = reinterpret_cast<const bf16*>(w);
  p.N = N;
  p.K = K;
  p.nseg = nseg;
  for (int s = 0; s < IG_SEGS; s++) {
    p.end[s] = s < nseg ? end[s] : N;
    p.x[s] = reinterpret_cast<const bf16*>(x[s < nseg ? s : nseg - 1]);
    p.out[s] = out[s < nseg ? s : nseg - 1];
    if (s < nseg - 1 && end[s] % rpw) return cudaErrorInvalidValue;
  }
  return launch_ig<0>(p, T, rpw, st);
}

// The output projection with the short-convolution epilogue (see inkling_gemv.cuh MODE 1): x bf16 [T][K], cw bf16 [N][4], cst bf16 [N][3] (updated in place),
// hres bf16 [T][N] or NULL, outf f32 [T][N].
int warpc_ink_gemv_convres(const void* w, const void* x, int N, int K, int T, const void* cw, void* cst, const void* hres, void* outf, float div, int rpw,
                           cudaStream_t st) {
  if (!ig_args_ok(N, K, T, rpw) || !cw || !cst || !outf) return cudaErrorInvalidValue;
  InkGemvP p = {};
  p.w = reinterpret_cast<const bf16*>(w);
  p.x[0] = reinterpret_cast<const bf16*>(x);
  p.N = N;
  p.K = K;
  p.nseg = 1;
  p.cw = reinterpret_cast<const bf16*>(cw);
  p.cst = reinterpret_cast<bf16*>(cst);
  p.hres = reinterpret_cast<const bf16*>(hres);
  p.outf = reinterpret_cast<float*>(outf);
  p.div = div;
  return launch_ig<1>(p, T, rpw, st);
}

}  // extern "C"
