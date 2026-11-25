// The k_ks instantiations, one object per T set: the Makefile compiles this file with -DGEMV_SET=T1|T2|T3_5|T6_8 (the sets of dense_gemv_inst.cu); KS 3, 4, 6, 8 (K = 6144, 8192, 12288, 16384).
#include "dense_ks_gemv.cuh"

template <int T, int KS>
void warpc_inst_ks(const GemvP& p, cudaStream_t st) {
  k_ks<T, KS><<<p.N, KS * 32, 0, st>>>(p);
}

#ifndef GEMV_SET
#error "compile with -DGEMV_SET=T1|T2|T3_5|T6_8"
#endif
#define KS_INST(T, S) template void warpc_inst_ks<T, S>(const GemvP&, cudaStream_t);
#define KS_INST_S(T) KS_INST(T, 3) KS_INST(T, 4) KS_INST(T, 6) KS_INST(T, 8)
#define KS_SET_T1 KS_INST_S(1)
#define KS_SET_T2 KS_INST_S(2)
#define KS_SET_T3_5 KS_INST_S(3) KS_INST_S(4) KS_INST_S(5)
#define KS_SET_T6_8 KS_INST_S(6) KS_INST_S(7) KS_INST_S(8)
#define KS_CAT(a, b) a##b
#define KS_SETM(s) KS_CAT(KS_SET_, s)
KS_SETM(GEMV_SET)
