// The k_gemv instantiations, one object per (T set, mode): the Makefile compiles this file once for each with -DGEMV_SET=T1|T2|T3_5|T6_8 -DGEMV_MODE=0..3.
// The launched set is what launch_gemv_t (dense_gemv.cu) reaches: T 1..8, rows per warp 1, 2, 4, K-loop unroll 4.
#include "dense_gemv.cuh"

template <int T, int RPW, int MODE>
void warpc_inst_gemv(const GemvP& p, int grid, cudaStream_t st) {
  k_gemv<T, RPW, MODE, 4><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
}

#ifndef GEMV_SET
#error "compile with -DGEMV_SET=T1|T2|T3_5|T6_8 -DGEMV_MODE=0..3"
#endif
#define INST(T, R, M) template void warpc_inst_gemv<T, R, M>(const GemvP&, int, cudaStream_t);
#define INST_R(T, M) INST(T, 1, M) INST(T, 2, M) INST(T, 4, M)
#define GEMV_SET_T1(M) INST_R(1, M)
#define GEMV_SET_T2(M) INST_R(2, M)
#define GEMV_SET_T3_5(M) INST_R(3, M) INST_R(4, M) INST_R(5, M)
#define GEMV_SET_T6_8(M) INST_R(6, M) INST_R(7, M) INST_R(8, M)
#define GEMV_CAT(a, b) a##b
#define GEMV_SETM(s) GEMV_CAT(GEMV_SET_, s)
GEMV_SETM(GEMV_SET)(GEMV_MODE)
