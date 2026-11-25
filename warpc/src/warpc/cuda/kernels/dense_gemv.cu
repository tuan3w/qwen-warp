#include "dense_dec_gemv.cuh"  // (includes dense_gemv.cuh)
#include "dense_ks_gemv.cuh"

// Rows per warp of the rmsnorm-prologue GEMVs at T = 1 (k_dg): one for q|k|v and the gate|up pair; lighter warps balance the 82 SMs better than the k_gemv rule (q|k|v with two rows measured
// 1-4% slower on the dense shapes and 11% on Gemma's 2560 x 1536)
#define DEC_RPW_QKV 1
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

template <int T>
static cudaError_t launch_ks_t(const GemvP& p, int ks, cudaStream_t st) {
  switch (ks) {
    case 3: warpc_inst_ks<T, 3>(p, st); break;
    case 4: warpc_inst_ks<T, 4>(p, st); break;
    case 6: warpc_inst_ks<T, 6>(p, st); break;
    case 8: warpc_inst_ks<T, 8>(p, st); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
// the K-split launch: segments (mode 0) without a norm prologue, K = 2048 ks (see dense_ks_gemv.cuh)
static cudaError_t launch_ks(const GemvP& p, int T, int ks, cudaStream_t st) {
  switch (T) {
    case 1: return launch_ks_t<1>(p, ks, st);
    case 2: return launch_ks_t<2>(p, ks, st);
    case 3: return launch_ks_t<3>(p, ks, st);
    case 4: return launch_ks_t<4>(p, ks, st);
    case 5: return launch_ks_t<5>(p, ks, st);
    case 6: return launch_ks_t<6>(p, ks, st);
    case 7: return launch_ks_t<7>(p, ks, st);
    case 8: return launch_ks_t<8>(p, ks, st);
  }
  return cudaErrorInvalidValue;
}

extern "C" {

// mode 0: segments (end[0..2] are cumulative row ends, out[0..2]); mode 1: gate/up pair (w = gate, w2 = up, N rows each, out[0] bf16 [T][N], act 0 silu / 1 gelu_tanh);
// mode 2: fp32 out[0], softcapped with `cap` when cap > 0; mode 3: residual add in place (out[0] = h); mode 4: out[0] = bf16(act(y) * row), row a bf16 [T][N] operand (act as mode 1).
// rpw: rows per warp of k_gemv (1, 2, 4; the caller's schedule). ks > 1: the K-split kernel (mode 0, no norm, K = 2048 ks, ks in 3, 4, 6, 8); its reduction order differs from ks = 1's.
int warpc_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, const int* end, void* const* out, const void* nw, float eps, int rpw, const void* row,
               int act, float cap, int ks, cudaStream_t st) {
  if (K % 256 || T < 1 || T > STEP_ROWS || (rpw != 1 && rpw != 2 && rpw != 4) || mode < 0 || mode > 4 || (mode == 4 && !row)) return cudaErrorInvalidValue;
  if (ks != 1 && (mode != 0 || nw || K != 2048 * ks)) return cudaErrorInvalidValue;
  GemvP p;
  p.w = reinterpret_cast<const bf16*>(w);
  p.w2 = reinterpret_cast<const bf16*>(w2);
  p.x = reinterpret_cast<const bf16*>(x);
  p.nw = reinterpret_cast<const bf16*>(nw);
  p.eps = eps;
  p.row = reinterpret_cast<const bf16*>(row);
  p.act = act;
  p.cap = cap;
  p.N = N;
  p.K = K;
  for (int i = 0; i < 3; i++) {
    p.end[i] = end ? end[i] : N;
    p.out[i] = out[i];
  }
  if (ks > 1) return launch_ks(p, T, ks, st);
  if (T == 1) {  // single-row decode: the kernels with every load in flight (same arithmetic, bit-identical); a shape they do not instantiate falls through to k_gemv
    int r = rpw;  // rows per warp: the caller's, except for the rmsnorm-prologue GEMVs, the residual-add and the row-operand GEMVs (one row per warp)
    if (nw) r = mode == 0 ? DEC_RPW_QKV : DEC_RPW_GU;
    else if (mode == 3 || mode == 4 || K == 256) r = 1;  // K = 256 (Gemma's pproj): one load per row and lane, so the rows are spread over the warps (4 rows per warp left 96 CTAs)
    const bool norm = nw != nullptr;
    const int rc = mode == 0   ? warpc_dec_gemv_m0(p, r, norm, st)
                   : mode == 1 ? warpc_dec_gemv_m1(p, r, norm, st)
                   : mode == 2 ? warpc_dec_gemv_m2(p, r, norm, st)
                   : mode == 3 ? warpc_dec_gemv_m3(p, r, norm, st)
                               : warpc_dec_gemv_m4(p, r, norm, st);
    if (rc != DEC_NO_KERNEL) return rc;  // a k_dg launched (rc = its launch status, success or not): never run k_gemv after it, mode 3 would add the residual twice
  }
  if (mode == 0) return launch_gemv_mode<0>(p, T, rpw, st);
  if (mode == 1) return launch_gemv_mode<1>(p, T, rpw, st);
  if (mode == 2) return launch_gemv_mode<2>(p, T, rpw, st);
  if (mode == 3) return launch_gemv_mode<3>(p, T, rpw, st);
  return launch_gemv_mode<4>(p, T, rpw, st);
}

}  // extern "C"
