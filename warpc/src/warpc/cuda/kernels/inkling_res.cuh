// Inkling-2 message and residual kernels (family `res`): attn_res (block-residual read, up to 48 queries per launch, optional
// per-query rmsnorm + bf16 cast), message_gate (rank-64 loop-message controller, up to 4 per launch, optional rmsnorm + cast) and
// the short convolution with residual (+ cast, add, cast). Conventions of dense_common.cuh. Sources:
//   inkling_res.cuh      this file: constants, parameter structs, shared block reductions
//   inkling_res_ar.cuh   the attn_res kernel template and its launcher
//   inkling_res_ar<N>.cu one object per row count N (the explicit instantiations of the launcher)
//   inkling_res.cu       message_gate, the convolution, and the extern "C" entry points
#pragma once
#include "dense_common.cuh"

constexpr int AR_MAXROWS = 6;  // rows an attn_res reads: the block bank (<= 5) + the partial
constexpr int AR_MAXQ = 4;  // queries one CTA mixes over the same rows
constexpr int AR_MAXSEG = 48;  // queries one launch serves (12 groups x 4: a pass's 12 loop layers x 4 messages)
constexpr int AR_KE = 9;  // columns per thread: H <= 256 * AR_KE

// one attn_res launch: `nseg` queries over the same rows; query j writes outf[j] (f32 mix, optional) and outn[j] (bf16
// of rmsnorm(mix) * nw[j], optional). A dead slot of the last group (nseg not a multiple of the group size) has q[j] = q[0]
// and no output.
struct ArP {
  const void* rows[AR_MAXROWS];
  const float* q[AR_MAXSEG];
  float* outf[AR_MAXSEG];
  bf16* outn[AR_MAXSEG];
  const bf16* nw[AR_MAXSEG];
  unsigned f32mask;  // bit n: row n is f32
  int H, T;
  float eps, eps_n;  // eps of the row norms, eps of the norm epilogue
};

// the launcher of the attn_res kernel for NR rows, instantiated in inkling_res_ar<NR>.cu; nq queries per CTA, G groups
template <int NR>
cudaError_t ink_ar_run(const ArP& p, int nq, int G, int T, cudaStream_t st);

constexpr int MG_R = 64;  // controller rank
constexpr int MG_MAXSEG = 4;  // gates one launch serves
constexpr int MG_MAXNB = 64;  // CTAs per (gate, row) of the output kernel: H <= 128 * MG_MAXNB

// one message_gate launch: gate c reads x[c], m[c] (f32 [T][H]); down [64][2H], fw / gw [H][64] bf16. It writes
// outf[c] (f32, optional) and outn[c] (bf16 rmsnorm(out) * nw[c], optional); at least one of the two.
struct MgP {
  const float *x[MG_MAXSEG], *m[MG_MAXSEG];
  const bf16 *down[MG_MAXSEG], *fw[MG_MAXSEG], *gw[MG_MAXSEG];
  float* outf[MG_MAXSEG];
  bf16* outn[MG_MAXSEG];
  const bf16* nw[MG_MAXSEG];
  float* y[MG_MAXSEG];  // where gate c's f32 result lives while its norm runs: outf[c], else a slice of the workspace
  float* hid;  // workspace [nseg][T][64]
  float* part;  // workspace [nseg][T][MG_MAXNB]: per-CTA sums of squares
  unsigned* cnt;  // workspace [nseg][T]: CTAs finished per row; zero between launches
  int nseg, T, H;
  float eps, eps_n;
};

// sum of N per-thread values over a block of NW warps; sm: at least NW * N floats. Fixed order (warps in index order),
// so the result is deterministic.
template <int N, int NW = 8>
__device__ __forceinline__ void block_sum_vec(float* v, float* sm) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < N; j++) v[j] = warp_sum(v[j]);
  __syncthreads();
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < N; j++) sm[w * N + j] = v[j];
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < N; j++) {
    float s = sm[j];
#pragma unroll
    for (int i = 1; i < NW; i++) s += sm[i * N + j];
    v[j] = s;
  }
  __syncthreads();
}
