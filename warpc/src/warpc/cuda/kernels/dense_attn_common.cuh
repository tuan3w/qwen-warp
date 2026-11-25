#pragma once
#include "dense_common.cuh"

// Attention: shared constants and types.
#define CH 64
#define MAX_SPLITS 128
#define MAX_SPLITS2 1024

struct QKV {
  long long kc, vc, ks, vs, kr, vr, kx, vx;  // byte offsets: K / V codes, K / V scales (fp16), ring rows, sink rows
  int sink, win, ring;
};

template <int HD>
__device__ __forceinline__ float wht_blk(float x, float* xs) {  // orthonormal WHT over the HD threads of the block, butterfly h = 1, 2, 4 ... (a + b, a - b)
  const int i = threadIdx.x;
#pragma unroll
  for (int h = 1; h < 32; h <<= 1) {
    const float p = __shfl_xor_sync(0xffffffffu, x, h);
    x = (i & h) ? p - x : x + p;
  }
#pragma unroll
  for (int h = 32; h < HD; h <<= 1) {
    __syncthreads();
    xs[i] = x;
    __syncthreads();
    const float p = xs[i ^ h];
    x = (i & h) ? p - x : x + p;
  }
  return x * (HD == 64 ? 0.125f : (HD == 128 ? 0.08838834764831845f : 0.0625f));
}

// merge of the split partials (kpb keys per split, partial stride HD + 4 floats so the rows are 16-byte aligned); ROT: the output is in the rotated basis of V and is
// un-rotated here (WHT in fp32, strides 1, 2, 4 ..; ROT 2 (int4_rot): the hand engine's order 32 .. HD / 2, then 1 .. 16) before the bf16 rounding. All 128 threads
// weigh the splits, then each thread merges one float4 of the head dim over every SG-th split (SG = 128 / (HD / 4) groups), the groups are added in a fixed order.
template <int HD, int ROT>
__global__ void __launch_bounds__(128) k_attn_reduce2(const float* part, const int* pos0, bf16* out, int Hq, int nsplit, int kpb) {
  constexpr int PS = HD + 4, NSLOT = HD / 4, SG = 128 / NSLOT;
  __shared__ float wgt[MAX_SPLITS2];
  __shared__ float red[32];
  __shared__ float4 acc4[128];
  __shared__ float ov[HD];
  const int head = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
  const int live = *pos0 + t + 1;
  const int ns = (live + kpb - 1) / kpb;
  const float* base = part + ((size_t)t * Hq + head) * nsplit * PS;
  float m = -INFINITY;
  for (int s = tid; s < ns; s += 128) m = fmaxf(m, base[s * PS + HD]);
  m = block_max(m, red);
  float l = 0.f;
  for (int s = tid; s < ns; s += 128) l += base[s * PS + HD + 1] * __expf(base[s * PS + HD] - m);
  l = block_sum(l, red);
  for (int s = tid; s < ns; s += 128) wgt[s] = __expf(base[s * PS + HD] - m) / l;
  __syncthreads();
  const int slot = tid % NSLOT, sg = tid / NSLOT;
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 8
  for (int s = sg; s < ns; s += SG) {
    const float4 v = reinterpret_cast<const float4*>(base + (size_t)s * PS)[slot];
    const float w = wgt[s];
    a.x += v.x * w;
    a.y += v.y * w;
    a.z += v.z * w;
    a.w += v.w * w;
  }
  acc4[tid] = a;
  __syncthreads();
  if (tid < NSLOT) {
    float4 r = acc4[tid];
#pragma unroll
    for (int g = 1; g < SG; g++) {
      const float4 u = acc4[g * NSLOT + tid];
      r.x += u.x;
      r.y += u.y;
      r.z += u.z;
      r.w += u.w;
    }
    if (ROT) {
      ov[4 * tid] = r.x;
      ov[4 * tid + 1] = r.y;
      ov[4 * tid + 2] = r.z;
      ov[4 * tid + 3] = r.w;
    } else {
      bf16* o = out + ((size_t)t * Hq + head) * HD + 4 * tid;
      o[0] = tb(r.x);
      o[1] = tb(r.y);
      o[2] = tb(r.z);
      o[3] = tb(r.w);
    }
  }
  if (ROT) {
    constexpr int LG = HD == 64 ? 6 : (HD == 128 ? 7 : 8);
    for (int k = 0; k < LG; k++) {
      const int h = ROT == 2 ? (k < LG - 5 ? 32 << k : 1 << (k - (LG - 5))) : 1 << k;
      __syncthreads();
      float v0[2], v1[2];
      int ii[2];
#pragma unroll
      for (int r = 0; r < 2; r++) {
        const int pidx = tid + r * 128;
        ii[r] = pidx < HD / 2 ? ((pidx & ~(h - 1)) << 1) | (pidx & (h - 1)) : -1;
        if (ii[r] >= 0) {
          v0[r] = ov[ii[r]];
          v1[r] = ov[ii[r] + h];
        }
      }
      __syncthreads();
#pragma unroll
      for (int r = 0; r < 2; r++)
        if (ii[r] >= 0) {
          ov[ii[r]] = v0[r] + v1[r];
          ov[ii[r] + h] = v0[r] - v1[r];
        }
    }
    __syncthreads();
    for (int d = tid; d < HD; d += 128) out[((size_t)t * Hq + head) * HD + d] = tb(ov[d] * (HD == 64 ? 0.125f : (HD == 128 ? 0.08838834764831845f : 0.0625f)));
  }
}

// Launchers (definitions and instantiations: dense_attn_inst.cu). The dispatching entry points call them; only the instantiation units compile the kernels behind them.
// run_decode: split + reduce launches of the bf16 cache (warpc_attn_decode); run_decode_x / run_prep_q / run_split_q return an error code (cudaGetLastError after the launches, or
// cudaErrorInvalidValue for a configuration they reject).
template <int HD>
void run_decode(dim3 grid, dim3 g2, size_t smem, const bf16* q, const bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, int nsplit, float scale,
                cudaStream_t st);
template <int HD>
int run_decode_x(int T, const bf16* q, const bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, float scale, int nc, cudaStream_t st);
// R4: KV format int4_rot (VB 4, EX 1).
template <int HD, int VB, int EX, int R4>
int run_prep_q(int T, const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* qout, uint8_t* kvb, const QKV& L, const int* pos0, int Hq, int Hkv,
               int rd, const float* inv_freq, float eps, int ctx, int qk_norm, cudaStream_t st);
// run_split_q: the checks (nsplit, shared memory, G variants) and the split launch of the quantised decode; run_merge_q: the merge launch (un-rotates V, k_attn_reduce2's ROT).
template <int HD, int VB, int EX, int R4>
int run_split_q(int T, const bf16* q, const uint8_t* kvb, const QKV& L, const int* pos0, float* ws, int Hq, int Hkv, int ctx, float scale, int nc, cudaStream_t st);
template <int HD, int ROT>
void run_merge_q(int T, const int* pos0, bf16* o, float* ws, int Hq, int nsplit, int kpb, cudaStream_t st);
