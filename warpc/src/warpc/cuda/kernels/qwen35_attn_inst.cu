// The split kernels, one object per configuration: -DQ35_I4=0 -DQ35_NT=<n-tiles> -DQ35_KS=<key slices> (k_split, bf16 cache) or -DQ35_I4=1 -DQ35_KS=<key slices> (k_split_i4).
#include "qwen35_attn_i4.cuh"

#ifndef Q35_I4
#error "compile with -DQ35_I4=0|1 -DQ35_KS=<key slices> [-DQ35_NT=<n-tiles>]"
#endif
#if Q35_I4
template int g256::run_split_i4<Q35_KS>(const SplitArgs&, int, cudaStream_t);
#else
template int g256::run_split<Q35_NT, Q35_KS>(const SplitArgs&, cudaStream_t);
#endif
