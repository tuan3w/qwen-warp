// Entry point of the Gated DeltaNet step: shape arguments -> the instantiated kernel (qwen35_deltanet_t*.cu).
#include "qwen35_deltanet.cuh"

namespace {
using namespace dn;

constexpr size_t CNT_BYTES_ALIGN = 256;

template <int DK, int DV, int W>
cudaError_t run(int T, const Args& a, cudaStream_t s) {
  switch (T) {
#define DN_CASE(t) \
  case t: return launch<t, nw_for(t, DK, DV), DK, DV, W>(a, s);
    DN_CASE(1) DN_CASE(2) DN_CASE(3) DN_CASE(4) DN_CASE(5) DN_CASE(6) DN_CASE(7) DN_CASE(8)
#undef DN_CASE
    default: return cudaErrorInvalidValue;
  }
}
}  // namespace

// Bytes of the zero-initialised scratch (counters, then the o rows of up to TMAX rows)
extern "C" size_t warpc_deltanet_ws_bytes(int Hk, int Hv, int dv) {
  const size_t cnt = ((size_t)(Hv + Hk) * sizeof(unsigned) + CNT_BYTES_ALIGN - 1) / CNT_BYTES_ALIGN * CNT_BYTES_ALIGN;
  return cnt + (size_t)TMAX * Hv * dv * sizeof(float);
}

extern "C" int warpc_deltanet_step(int T, const void* qkv, const void* z, const void* a, const void* b, const void* conv_w, const void* A_log,
                                   const void* dt_bias, const void* norm_w, void* state, void* conv, void* y, void* ws, int C, int W, int Hk,
                                   int Hv, int dk, int dv, float eps, float scale, float norm_eps, cudaStream_t stream) {
  if (T < 1 || T > TMAX || Hk < 1 || Hv < Hk || Hv % Hk || C != 2 * Hk * dk + Hv * dv) return cudaErrorInvalidValue;
  Args p = {};
  p.qkv = (const __nv_bfloat16*)qkv, p.z = (const __nv_bfloat16*)z, p.a = (const __nv_bfloat16*)a, p.b = (const __nv_bfloat16*)b;
  p.conv_w = (const __nv_bfloat16*)conv_w, p.A_log = (const __nv_bfloat16*)A_log, p.dt_bias = (const __nv_bfloat16*)dt_bias;
  p.norm_w = (const __nv_bfloat16*)norm_w;
  p.state = (float*)state, p.conv = (__nv_bfloat16*)conv, p.y = (__nv_bfloat16*)y;
  p.cnt_head = (unsigned*)ws, p.cnt_kh = p.cnt_head + Hv;
  const size_t cnt = ((size_t)(Hv + Hk) * sizeof(unsigned) + CNT_BYTES_ALIGN - 1) / CNT_BYTES_ALIGN * CNT_BYTES_ALIGN;
  p.o_ws = (float*)((char*)ws + cnt);
  p.C = C, p.Hk = Hk, p.Hv = Hv, p.rep = Hv / Hk;
  p.eps = eps, p.scale = scale, p.norm_eps = norm_eps;
#define DN_CFG(DK, DV, W) \
  if (dk == DK && dv == DV && W_ == W) return run<DK, DV, W>(T, p, stream);
  const int W_ = W;
  DN_CONFIGS(DN_CFG)
#undef DN_CFG
  return cudaErrorInvalidValue;
}
