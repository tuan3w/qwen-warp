#pragma once
#include "dense_attn_q.cuh"

// Shared by the gated hd-256 attention's translation units (qwen35_attn.cu: entries, prep and merge; qwen35_attn_inst.cu: the split kernel).
namespace g256 {

constexpr int HD = 256, RD = 64, NI = HD / 32;

// split geometry (the hand engine's rule: at least chmin keys per split, at most smax splits, chunks of a multiple of 16) from the keys of the split loop: the keys per split and the
// live number of splits, identical in the split and the merge kernel
__host__ __device__ __forceinline__ void geom(int ctxq, int smax, int chmin, int& nsplit, int& chunk) {
  ctxq = ctxq < 1 ? 1 : ctxq;
  int n = (ctxq + chmin - 1) / chmin;
  n = n < 1 ? 1 : (n > smax ? smax : n);
  chunk = (((ctxq + n - 1) / n) + 15) & ~15;
  nsplit = (ctxq + chunk - 1) / chunk;
}

// arguments of k_split / k_split_i4 (qwen35_attn_split.cuh, qwen35_attn_i4.cuh); q: the prepped query rows [T][Hq][HD], bf16 (bf16 cache) or fp16 (int4_rot); part: [T][Hq][smax][HD + 4] floats (o, m in log2
// units, l)
struct SplitArgs {
  const bf16* q;
  const uint8_t* kv;
  QKV L;
  const int* pos0;
  float* part;
  int T, Hq, Hkv, ctx, smax, chmin;
  float scale2;  // scale * log2(e)
};
// the split kernels, instantiated in qwen35_attn_inst.cu: k_split (bf16 cache, one object per n-tiles of 8 query columns and key slices) and k_split_i4 (int4_rot, one object per
// key slices, one n-tile per block: blockIdx.z)
template <int NT, int KS>
int run_split(const SplitArgs& a, cudaStream_t st);
template <int KS>
int run_split_i4(const SplitArgs& a, int nt, cudaStream_t st);

}  // namespace g256
