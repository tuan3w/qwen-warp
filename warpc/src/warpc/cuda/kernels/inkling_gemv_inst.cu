// The k_ig instantiations, one object per (T set, mode): the Makefile compiles this file once for each with -DIGEMV_SET=T1|T2|T3_5|T6_8 -DIGEMV_MODE=0..1.
// The launched set is what launch_ig_t (inkling_gemv.cu) reaches: T 1..8, rows per warp 1, 2, 4, K-loop unroll 4.
#include "inkling_gemv.cuh"

template <int T, int RPW, int MODE>
void warpc_inst_ink_gemv(const InkGemvP& p, int grid, cudaStream_t st) {
  k_ig<T, RPW, MODE, 4><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
}

#ifndef IGEMV_SET
#error "compile with -DIGEMV_SET=T1|T2|T3_5|T6_8 -DIGEMV_MODE=0..1"
#endif
#define INST(T, R, M) template void warpc_inst_ink_gemv<T, R, M>(const InkGemvP&, int, cudaStream_t);
#define INST_R(T, M) INST(T, 1, M) INST(T, 2, M) INST(T, 4, M)
#define IGEMV_SET_T1(M) INST_R(1, M)
#define IGEMV_SET_T2(M) INST_R(2, M)
#define IGEMV_SET_T3_5(M) INST_R(3, M) INST_R(4, M) INST_R(5, M)
#define IGEMV_SET_T6_8(M) INST_R(6, M) INST_R(7, M) INST_R(8, M)
#define IGEMV_CAT(a, b) a##b
#define IGEMV_SETM(s) IGEMV_CAT(IGEMV_SET_, s)
IGEMV_SETM(IGEMV_SET)(IGEMV_MODE)
