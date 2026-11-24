#include "dense_dec_gemv.cuh"  // (includes dense_gemv.cuh)

// Rows per warp of the rmsnorm-prologue GEMVs at T = 1 (k_dg): q|k|v 2, gate|up pair 1; lighter warps balance the 82 SMs better than the k_gemv rule
#define DEC_RPW_QKV 2
#define DEC_RPW_GU 1

template <int T, int MODE>
static cudaError_t launch_gemv_t(const GemvP& p, int rpw, cudaStream_t st) {
  const int grid = (p.N + GEMV_WARPS * rpw - 1) / (GEMV_WARPS * rpw);
  if (rpw == 4) warpc_inst_gemv<T, 4, MODE>(p, grid, st);
  else if (rpw == 2) warpc_inst_gemv<T, 2, MODE>(p, grid, st);
  else warpc_inst_gemv<T, 1, MODE>(p, grid, st);
  return cudaGetLastError();
}
template <int MODE>
static cudaError_t launch_gemv_mode(const GemvP& p, int T, int rpw, cudaStream_t st) {
  switch (T) {
    case 1: return launch_gemv_t<1, MODE>(p, rpw, st);
    case 2: return launch_gemv_t<2, MODE>(p, rpw, st);
    case 3: return launch_gemv_t<3, MODE>(p, rpw, st);
    case 4: return launch_gemv_t<4, MODE>(p, rpw, st);
    case 5: return launch_gemv_t<5, MODE>(p, rpw, st);
    case 6: return launch_gemv_t<6, MODE>(p, rpw, st);
    case 7: return launch_gemv_t<7, MODE>(p, rpw, st);
    case 8: return launch_gemv_t<8, MODE>(p, rpw, st);
  }
  return cudaErrorInvalidValue;
}

extern "C" {

// mode 0: segments (end[0..2] are cumulative row ends, out[0..2]); mode 1: gate/up pair (w = gate, w2 = up, N rows each, out[0] bf16 [T][N]); mode 2: fp32 out[0];
// mode 3: residual add in place (out[0] = h). rpw: rows per warp of k_gemv (1, 2, 4; the caller's schedule).
int warpc_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, const int* end, void* const* out, const void* nw, float eps, int rpw,
               cudaStream_t st) {
  if (K % 256 || T < 1 || T > STEP_ROWS || (rpw != 1 && rpw != 2 && rpw != 4)) return cudaErrorInvalidValue;
  GemvP p;
  p.w = reinterpret_cast<const bf16*>(w);
  p.w2 = reinterpret_cast<const bf16*>(w2);
  p.x = reinterpret_cast<const bf16*>(x);
  p.nw = reinterpret_cast<const bf16*>(nw);
  p.eps = eps;
  p.N = N;
  p.K = K;
  for (int i = 0; i < 3; i++) {
    p.end[i] = end ? end[i] : N;
    p.out[i] = out[i];
  }
  if (T == 1) {  // single-row decode: the kernels with every load in flight (same arithmetic, bit-identical); a shape they do not instantiate falls through to k_gemv
    int r = rpw;  // rows per warp: the caller's, except for the rmsnorm-prologue GEMVs and the residual-add GEMVs (one row per warp)
    if (nw) r = mode == 0 ? DEC_RPW_QKV : DEC_RPW_GU;
    else if (mode == 3) r = 1;
    const bool norm = nw != nullptr;
    const int rc = mode == 0 ? warpc_dec_gemv_m0(p, r, norm, st) : mode == 1 ? warpc_dec_gemv_m1(p, r, norm, st) : mode == 2 ? warpc_dec_gemv_m2(p, r, norm, st) : warpc_dec_gemv_m3(p, r, norm, st);
    if (rc != DEC_NO_KERNEL) return rc;  // a k_dg launched (rc = its launch status, success or not): never run k_gemv after it, mode 3 would add the residual twice
  }
  if (mode == 0) return launch_gemv_mode<0>(p, T, rpw, st);
  if (mode == 1) return launch_gemv_mode<1>(p, T, rpw, st);
  if (mode == 2) return launch_gemv_mode<2>(p, T, rpw, st);
  if (mode == 3) return launch_gemv_mode<3>(p, T, rpw, st);
  return cudaErrorInvalidValue;
}

}  // extern "C"
