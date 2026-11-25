// The k_q4_gemv instantiations, one object per (T set, mode): the Makefile compiles this file once for each with
// -DQ4GEMV_SET=T1|T2|T3_5|T6_8 -DQ4GEMV_MODE=0|1|3. The launched set is what launch_q4_gemv_t (dense_q4_gemv.cu) reaches:
// T 1..8, rows per warp 1, 2, 4, K-loop unroll 4.
#include "dense_q4_gemv.cuh"

template <int T, int RPW, int MODE>
void warpc_inst_q4_gemv(const Q4GemvP& p, int grid, cudaStream_t st) {
  k_q4_gemv<T, RPW, MODE, 4><<<grid, GEMV_WARPS * 32, 0, st>>>(p);
}

#ifndef Q4GEMV_SET
#error "compile with -DQ4GEMV_SET=T1|T2|T3_5|T6_8 -DQ4GEMV_MODE=0|1|3"
#endif
#define Q4INST(T, R, M) template void warpc_inst_q4_gemv<T, R, M>(const Q4GemvP&, int, cudaStream_t);
#define Q4INST_R(T, M) Q4INST(T, 1, M) Q4INST(T, 2, M) Q4INST(T, 4, M)
#define Q4GEMV_SET_T1(M) Q4INST_R(1, M)
#define Q4GEMV_SET_T2(M) Q4INST_R(2, M)
#define Q4GEMV_SET_T3_5(M) Q4INST_R(3, M) Q4INST_R(4, M) Q4INST_R(5, M)
#define Q4GEMV_SET_T6_8(M) Q4INST_R(6, M) Q4INST_R(7, M) Q4INST_R(8, M)
#define Q4GEMV_CAT(a, b) a##b
#define Q4GEMV_SETM(s) Q4GEMV_CAT(Q4GEMV_SET_, s)
Q4GEMV_SETM(Q4GEMV_SET)(Q4GEMV_MODE)
