// attn_res launcher for 1 rows (one object per row count: the kernel is unrolled over rows and queries).
#include "inkling_res_ar.cuh"
template cudaError_t ink_ar_run<1>(const ArP&, int, int, int, cudaStream_t);
