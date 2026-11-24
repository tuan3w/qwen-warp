#include "dense_common.cuh"

__global__ void k_embed(const uint4* table, const int* tok, uint4* out, int H8) {
  const int t = blockIdx.x;
  const size_t row = (size_t)tok[t] * H8;
  for (int i = threadIdx.x; i < H8; i += blockDim.x) out[(size_t)t * H8 + i] = table[row + i];
}

// rmsnorm (+ residual add). y != NULL: h[t] = bf16(h[t] + y[t]) in place. xn[t] = bf16(h[t] * rsqrt(mean(h[t]^2) + eps) * w); fp32 math, one rounding.
__global__ void k_rmsnorm(bf16* h, const bf16* y, const bf16* w, bf16* xn, int H, float eps) {
  __shared__ float sm[32];
  const size_t base = (size_t)blockIdx.x * H;
  const int H8 = H / 8;
  uint4* hv = reinterpret_cast<uint4*>(h + base);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H8; i += blockDim.x) {
    float f[8];
    unpack8(hv[i], f);
    if (y) {
      float g[8];
      unpack8(reinterpret_cast<const uint4*>(y + base)[i], g);
#pragma unroll
      for (int k = 0; k < 8; k++) f[k] = rb(f[k] + g[k]);
      hv[i] = pack8(f);
    }
#pragma unroll
    for (int k = 0; k < 8; k++) ss += f[k] * f[k];
  }
  ss = block_sum(ss, sm);
  const float r = rsqrtf(ss / (float)H + eps);
  for (int i = threadIdx.x; i < H8; i += blockDim.x) {
    float f[8], wf[8];
    unpack8(hv[i], f);
    unpack8(reinterpret_cast<const uint4*>(w)[i], wf);
#pragma unroll
    for (int k = 0; k < 8; k++) f[k] = f[k] * r * wf[k];
    reinterpret_cast<uint4*>(xn + base)[i] = pack8(f);
  }
}

// argmax, lowest index wins ties
__global__ void k_argmax_part(const float* logits, int N, float* pv, int* pi) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int t = blockIdx.y, b = blockIdx.x, nb = gridDim.x;
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = b * blockDim.x + threadIdx.x; i < N; i += nb * blockDim.x) {
    const float v = logits[(size_t)t * N + i];
    if (v > best || (v == best && i < bi)) {
      best = v;
      bi = i;
    }
  }
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > best || (ov == best && oi < bi)) {
      best = ov;
      bi = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    sv[threadIdx.x >> 5] = best;
    si[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    best = sv[0];
    bi = si[0];
    for (int w = 1; w < (blockDim.x >> 5); w++)
      if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
        best = sv[w];
        bi = si[w];
      }
    pv[t * nb + b] = best;
    pi[t * nb + b] = bi;
  }
}
__global__ void k_argmax_final(const float* pv, const int* pi, int nb, int* out) {
  __shared__ float sv[32];
  __shared__ int si[32];
  const int t = blockIdx.x;
  float best = threadIdx.x < nb ? pv[t * nb + threadIdx.x] : -INFINITY;
  int bi = threadIdx.x < nb ? pi[t * nb + threadIdx.x] : 0x7fffffff;
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (ov > best || (ov == best && oi < bi)) {
      best = ov;
      bi = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    sv[threadIdx.x >> 5] = best;
    si[threadIdx.x >> 5] = bi;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    best = sv[0];
    bi = si[0];
    for (int w = 1; w < (blockDim.x >> 5); w++)
      if (sv[w] > best || (sv[w] == best && si[w] < bi)) {
        best = sv[w];
        bi = si[w];
      }
    out[t] = bi;
  }
}

// Step bookkeeping on the device, so one captured graph serves every position: pos += T, and with feedback the argmax of the last row becomes the next input token.
__global__ void k_advance(int* pos, int* tok, const int* ids, int T, int feedback) {
  pos[0] += T;
  if (feedback) tok[0] = ids[T - 1];
}

extern "C" {

int warpc_embed(const void* table, const int* tok, void* out, int T, int H, cudaStream_t st) {
  if (H % 8) return cudaErrorInvalidValue;
  k_embed<<<T, 128, 0, st>>>(reinterpret_cast<const uint4*>(table), tok, reinterpret_cast<uint4*>(out), H / 8);
  return cudaGetLastError();
}

int warpc_rmsnorm(void* h, const void* y, const void* w, void* xn, int T, int H, float eps, cudaStream_t st) {
  if (H % 8) return cudaErrorInvalidValue;
  k_rmsnorm<<<T, H >= 4096 ? 512 : 256, 0, st>>>(reinterpret_cast<bf16*>(h), reinterpret_cast<const bf16*>(y), reinterpret_cast<const bf16*>(w),
                                                  reinterpret_cast<bf16*>(xn), H, eps);
  return cudaGetLastError();
}

#define ARGMAX_BLOCKS 256
size_t warpc_argmax_ws_bytes() { return STEP_ROWS * ARGMAX_BLOCKS * (sizeof(float) + sizeof(int)); }
int warpc_argmax(const float* logits, int T, int N, int* out_tok, void* ws, cudaStream_t st) {
  if (T < 1 || T > STEP_ROWS) return cudaErrorInvalidValue;  // the workspace holds STEP_ROWS rows of partials
  float* pv = reinterpret_cast<float*>(ws);
  int* pi = reinterpret_cast<int*>(pv + STEP_ROWS * ARGMAX_BLOCKS);
  k_argmax_part<<<dim3(ARGMAX_BLOCKS, T), 256, 0, st>>>(logits, N, pv, pi);
  k_argmax_final<<<T, ARGMAX_BLOCKS, 0, st>>>(pv, pi, ARGMAX_BLOCKS, out_tok);
  return cudaGetLastError();
}

int warpc_advance(int* pos, int* tok, const int* ids, int T, int feedback, cudaStream_t st) {
  k_advance<<<1, 1, 0, st>>>(pos, tok, ids, T, feedback);
  return cudaGetLastError();
}

const char* warpc_error_string(int error) { return cudaGetErrorString(static_cast<cudaError_t>(error)); }

}  // extern "C"
