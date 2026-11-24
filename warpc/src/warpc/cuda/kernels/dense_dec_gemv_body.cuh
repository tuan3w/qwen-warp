// Launchers of k_dg for ONE mode (DECM), included by dense_dec_gemv_m<mode>.cu: rows per warp 1, 2, 4 (as far as the register budget of RPW x NIT in-flight 16-byte loads allows), NIT = K / 256.
#include "dense_dec_gemv.cuh"

#ifndef DECM
#error "define DECM"
#endif
#define DEC_CAT2(a, b) a##b
#define DEC_CAT(a, b) DEC_CAT2(a, b)

namespace {
constexpr int kPer = DECM == 1 ? 2 : 1;  // weight rows streamed per output row
constexpr int kMaxLoads = 24;            // in-flight uint4 weight loads per lane (above this the kernel spills)

template <int RPW, int NIT, bool NORM>
int run(const GemvP& p, cudaStream_t st) {
  const int grid = (p.N + GEMV_WARPS * RPW - 1) / (GEMV_WARPS * RPW);
  k_dg<RPW, DECM, NIT, NORM><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
  return cudaGetLastError();
}
template <int NIT, bool NORM>
int by_rpw(const GemvP& p, int rpw, cudaStream_t st) {
  // rows per warp 1 is exempt from the budget (k_dg<1, 3, 16> holds 16 loads, the pair kernel without the norm 32 in 128 registers) except with the norm prologue: the norm weights
  // add NIT more uint4 and k_dg<1, 1, 16, true> spilled (REG 255 STACK 72); its k_gemv fall-back measured 1.7% faster on the Llama-3-8B shape
  if constexpr (!(NORM && NIT * kPer > kMaxLoads)) {
    if (rpw == 1) return run<1, NIT, NORM>(p, st);
  }
  if constexpr (2 * NIT * kPer <= kMaxLoads) {
    if (rpw == 2) return run<2, NIT, NORM>(p, st);
  }
  if constexpr (4 * NIT * kPer <= kMaxLoads) {
    if (rpw == 4) return run<4, NIT, NORM>(p, st);
  }
  return DEC_NO_KERNEL;
}
template <bool NORM>
int by_nit(const GemvP& p, int rpw, cudaStream_t st) {
  switch (p.K / 256) {
    case 4: return by_rpw<4, NORM>(p, rpw, st);
    case 6: return by_rpw<6, NORM>(p, rpw, st);
    case 8: return by_rpw<8, NORM>(p, rpw, st);
    case 10: return by_rpw<10, NORM>(p, rpw, st);
    case 12: return by_rpw<12, NORM>(p, rpw, st);
    case 16: return by_rpw<16, NORM>(p, rpw, st);
  }
  return DEC_NO_KERNEL;
}
}  // namespace

int DEC_CAT(warpc_dec_gemv_m, DECM)(const GemvP& p, int rpw, bool norm, cudaStream_t st) {
  if (norm) {
    if constexpr (DECM < 2) return by_nit<true>(p, rpw, st);
    else return DEC_NO_KERNEL;
  }
  return by_nit<false>(p, rpw, st);
}
