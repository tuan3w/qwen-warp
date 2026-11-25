// Data movement of the MTP draft graph: concat, select_row and take. Each is exact (bytes are copied, an id is an exact
// fp32 integer), allocates nothing and never synchronises, so it can be captured in a CUDA graph. Row indices are DEVICE
// scalars read inside the kernel; there is no bounds check (the graph's contract, warpc/ir/ops.py).
#include "dense_common.cuh"

namespace {

// out[t] = [x[t], y[t]] for t < T: rows of `a` and `b` elements of `esz` bytes; ca = a * esz / 16 and cb = b * esz / 16 chunks.
__global__ void k_concat(const uint4* __restrict__ x, const uint4* __restrict__ y, uint4* __restrict__ out, int ca, int cb) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= ca + cb) return;
  const size_t t = blockIdx.y;
  out[t * (size_t)(ca + cb) + c] = c < ca ? x[t * (size_t)ca + c] : y[t * (size_t)cb + (c - ca)];
}

// out = x[i[0]]: one row of `chunks` 16-byte chunks.
__global__ void k_select_row(const uint4* __restrict__ x, const int* __restrict__ i, uint4* __restrict__ out, int chunks) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chunks) return;
  out[c] = x[(size_t)i[0] * chunks + c];
}

// out[0] = table[i[0]]; the table is an fp32 constant of exact integers (< 2^24).
__global__ void k_take(const int* __restrict__ i, const float* __restrict__ table, int* __restrict__ out) { out[0] = (int)table[i[0]]; }

}  // namespace

extern "C" {

// x [T][a], y [T][b], out [T][a + b], elements of esz bytes (2: bf16, 4: f32); every row a multiple of 16 bytes.
int warpc_concat(const void* x, const void* y, void* out, int T, int a, int b, int esz, cudaStream_t st) {
  if (T < 1 || T > 65535 || (a * esz) % 16 || (b * esz) % 16 || a < 1 || b < 1) return cudaErrorInvalidValue;
  const int ca = a * esz / 16, cb = b * esz / 16;
  k_concat<<<dim3((ca + cb + 255) / 256, T), 256, 0, st>>>(reinterpret_cast<const uint4*>(x), reinterpret_cast<const uint4*>(y),
                                                           reinterpret_cast<uint4*>(out), ca, cb);
  return cudaGetLastError();
}

// out [1][D] = x[i[0]] of x [T][D]; D * esz a multiple of 16.
int warpc_select_row(const void* x, const int* i, void* out, int D, int esz, cudaStream_t st) {
  if ((D * esz) % 16 || D < 1) return cudaErrorInvalidValue;
  const int chunks = D * esz / 16;
  k_select_row<<<(chunks + 255) / 256, 256, 0, st>>>(reinterpret_cast<const uint4*>(x), i, reinterpret_cast<uint4*>(out), chunks);
  return cudaGetLastError();
}

int warpc_take(const int* i, const float* table, int* out, cudaStream_t st) {
  k_take<<<1, 1, 0, st>>>(i, table, out);
  return cudaGetLastError();
}

}  // extern "C"
