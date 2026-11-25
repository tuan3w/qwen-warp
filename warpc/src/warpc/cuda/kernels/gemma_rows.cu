#include "dense_common.cuh"

// Gemma's per-layer-embedding prelude as one launch. For row t and layer slice j (D elements, a multiple of 256):
//   c   = bf16(ctx[t][j] * f1)                      ctx: the [T][L*D] projection, f1 = hidden^-0.5
//   n   = bf16(c * rsqrt(mean(c^2) + eps) * w)      the D-wide norm, w [D]
//   e   = bf16(table[tok[t]][j] * bf16(escale))     the per-layer embedding row, [vocab][L*D]
//   out[j][t] = bf16(bf16(n + e) * f2)              f2 = 2^-0.5; out[j] is a [T][D] buffer of its own
// The statistics: each thread sums the squares of its 8 consecutive elements in order (fma), block_sum adds the D / 8
// partials (32-lane butterfly per warp, then the warp sums in order; D = 256 is one butterfly).
#define PLE_MAX_LAYERS 64
struct PleP {
  void* out[PLE_MAX_LAYERS];
};

__global__ void __launch_bounds__(128) k_ple(const bf16* ctx, const int* tok, const bf16* table, const bf16* w, const __grid_constant__ PleP p, int L, int D, float f1, float f2,
                                            float eps, float escale) {
  __shared__ float sm[32];
  const int j = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
  const size_t col = (size_t)j * D + i * 8, width = (size_t)L * D;
  float c[8], wf[8], tf[8];
  unpack8(*reinterpret_cast<const uint4*>(ctx + t * width + col), c);
  unpack8(*reinterpret_cast<const uint4*>(w + i * 8), wf);
  unpack8(*reinterpret_cast<const uint4*>(table + (size_t)tok[t] * width + col), tf);
  float ss = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    c[k] = rb(c[k] * f1);
    ss += c[k] * c[k];
  }
  ss = block_sum(ss, sm);
  const float r = rsqrtf(ss / (float)D + eps), es = rb(escale);
  float o[8];
#pragma unroll
  for (int k = 0; k < 8; k++) o[k] = rb(rb(c[k] * r * wf[k]) + rb(tf[k] * es)) * f2;
  *reinterpret_cast<uint4*>(reinterpret_cast<bf16*>(p.out[j]) + (size_t)t * D + i * 8) = pack8(o);
}

extern "C" {

// ctx [T][L*D], tok [T] ids, table [vocab][L*D], w [D]; outs: L device pointers (read during the call), each a [T][D] buffer
int warpc_ple(const void* ctx, const int* tok, const void* table, const void* w, void* const* outs, int T, int L, int D, float f1, float f2, float eps, float escale,
              cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS || L < 1 || L > PLE_MAX_LAYERS || D % 256 || D > 1024) return cudaErrorInvalidValue;
  PleP p;
  for (int j = 0; j < L; j++) p.out[j] = outs[j];
  k_ple<<<dim3(L, T), D / 8, 0, st>>>(reinterpret_cast<const bf16*>(ctx), tok, reinterpret_cast<const bf16*>(table), reinterpret_cast<const bf16*>(w), p, L, D, f1, f2, eps, escale);
  return cudaGetLastError();
}

}  // extern "C"
