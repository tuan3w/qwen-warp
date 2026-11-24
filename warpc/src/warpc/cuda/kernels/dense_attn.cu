#include "dense_attn_common.cuh"

// run_decode1 (dense_attn_inst.cu): one row, head dim 128: the fused prep + split launch and the merge launch; returns an error code.
template <int HD>
int run_decode1(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, int nsplit,
                float scale, float theta, float eps, int qk_norm, cudaStream_t st);

// Attention prep. Per (row t, head): q heads and k heads get an optional RMSNorm over the head dim (weight qn / kn) rounded to bf16, then rotary embedding of the first rd dims
// (half-split pairing, angle = (pos0 + t) * theta^(-2j/rd)), q goes to qout, k and v are written to the KV cache at position pos0 + t.
__global__ void k_attn_prep(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* qout, bf16* kv, const int* pos0, int Hq, int Hkv,
                            int hd, int rd, float theta, float eps, int ctx, int qk_norm) {
  __shared__ float xs[256];
  __shared__ float sm[32];
  const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
  const int p = *pos0 + t;
  const bf16 *src, *nw = nullptr;
  bf16* dst;
  bool rope = true;
  if (h < Hq) {
    src = q + ((size_t)t * Hq + h) * hd;
    nw = qn;
    dst = qout + ((size_t)t * Hq + h) * hd;
  } else if (h < Hq + Hkv) {
    const int g = h - Hq;
    src = k + ((size_t)t * Hkv + g) * hd;
    nw = kn;
    dst = kv + ((size_t)g * ctx + p) * hd;
  } else {
    const int g = h - Hq - Hkv;
    src = v + ((size_t)t * Hkv + g) * hd;
    dst = kv + (((size_t)Hkv + g) * ctx + p) * hd;
    rope = false;
    nw = nullptr;
  }
  float x = i < hd ? bf(src[i]) : 0.f;
  if (!rope) {
    if (i < hd) dst[i] = tb(x);
    return;
  }
  if (qk_norm) {
    const float ss = block_sum(x * x, sm);
    const float r = rsqrtf(ss / (float)hd + eps);
    const float w = i < hd ? bf(nw[i]) : 0.f;
    x = rb(x * r * w);
  }
  xs[i] = x;
  __syncthreads();
  if (i < hd) {
    float o = x;
    if (i < rd) {
      const int half = rd / 2;
      const int j = i < half ? i : i - half;
      const float inv = exp2f(-(2.f * (float)j / (float)rd) * log2f(theta));
      float s, c;
      sincosf((float)p * inv, &s, &c);
      const float partner = i < half ? xs[i + half] : xs[i - half];
      o = i < half ? x * c - partner * s : x * c + partner * s;
    }
    dst[i] = tb(o);
  }
}

extern "C" {

int warpc_attn_prep(int T, const void* q, const void* k, const void* v, const void* qn, const void* kn, void* qout, void* kv, const int* pos0, int Hq, int Hkv, int hd,
                    int rd, float theta, float eps, int ctx, int qk_norm, cudaStream_t st) {
  if (hd > 256 || hd % 32 || rd % 2 || rd > hd) return cudaErrorInvalidValue;
  dim3 grid(Hq + 2 * Hkv, T);
  k_attn_prep<<<grid, hd, 0, st>>>(reinterpret_cast<const bf16*>(q), reinterpret_cast<const bf16*>(k), reinterpret_cast<const bf16*>(v),
                                   reinterpret_cast<const bf16*>(qn), reinterpret_cast<const bf16*>(kn), reinterpret_cast<bf16*>(qout), reinterpret_cast<bf16*>(kv),
                                   pos0, Hq, Hkv, hd, rd, theta, eps, ctx, qk_norm);
  return cudaGetLastError();
}

int warpc_attn_decode(int T, const void* qprep, const void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd, int ctx, float scale, cudaStream_t st) {
  if (Hq % Hkv || Hq / Hkv > 8) return cudaErrorInvalidValue;
  const int nsplit = (ctx + CH - 1) / CH, G = Hq / Hkv;
  if (nsplit > MAX_SPLITS) return cudaErrorInvalidValue;
  dim3 grid(Hkv, nsplit, T);
  const int dp = hd / 2, kg = 128 / dp > 0 ? 128 / dp : 1;
  const size_t smem = ((size_t)G * (hd + CH) + (size_t)kg * G * hd + 16) * sizeof(float);
  const bf16* q = reinterpret_cast<const bf16*>(qprep);
  const bf16* kvp = reinterpret_cast<const bf16*>(kv);
  bf16* o = reinterpret_cast<bf16*>(out);
  dim3 g2(Hq, T);
  if (hd == 128) run_decode<128>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, st);
  else if (hd == 64) run_decode<64>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, st);
  else if (hd == 256) run_decode<256>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, st);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

// one row (decode), head dim 128, rotary dims = head dim: prep folded into the split kernel (see k_attn_fused), then the merge; same partial layout and bits as prep + decode
int warpc_attn_decode_fused(const void* q, const void* k, const void* v, const void* qn, const void* kn, void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd,
                            int rd, float theta, float eps, int ctx, int qk_norm, float scale, cudaStream_t st) {
  if (hd != 128 || rd != 128 || Hq % Hkv || Hq / Hkv > 8) return cudaErrorInvalidValue;
  const int nsplit = (ctx + CH - 1) / CH;
  if (nsplit > MAX_SPLITS) return cudaErrorInvalidValue;
  return run_decode1<128>(reinterpret_cast<const bf16*>(q), reinterpret_cast<const bf16*>(k), reinterpret_cast<const bf16*>(v), reinterpret_cast<const bf16*>(qn),
                          reinterpret_cast<const bf16*>(kn), reinterpret_cast<bf16*>(kv), pos0, reinterpret_cast<bf16*>(out), ws, Hq, Hkv, ctx, nsplit, scale, theta, eps, qk_norm, st);
}

// workspace of every decode entry point (kpb = keys per split block, 64 * nc; the hd + 4 floats per partial row also hold the hd + 2 of the nc = 1 kernels)
size_t warpc_attn_ws_bytes(int T, int Hq, int hd, int ctx, int kpb) { return (size_t)T * Hq * ((ctx + kpb - 1) / kpb) * (hd + 4) * sizeof(float); }

// chunked bf16 decode (nc chunks of 64 keys per block); same partial / merge layout as warpc_attn_decode

int warpc_attn_decode_x(int T, const void* qprep, const void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd, int ctx, float scale, int nc, cudaStream_t st) {
  if (Hq % Hkv || Hq / Hkv > 8 || nc < 1) return cudaErrorInvalidValue;
  const bf16* q = reinterpret_cast<const bf16*>(qprep);
  const bf16* k = reinterpret_cast<const bf16*>(kv);
  bf16* o = reinterpret_cast<bf16*>(out);
  if (hd == 128) return run_decode_x<128>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, st);
  if (hd == 64) return run_decode_x<64>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, st);
  if (hd == 256) return run_decode_x<256>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, st);
  return cudaErrorInvalidValue;
}

}  // extern "C"
