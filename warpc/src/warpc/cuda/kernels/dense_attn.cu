#include "dense_attn_common.cuh"

// run_decode1x (dense_attn_inst.cu): one row, head dim 256 / 512: the whole decode in one launch; returns an error code.
template <int HD, int READ>
int run_decode1x(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, bf16* o, float* ws, unsigned* cnt, int Hq, int Hkv, int ctx,
                 float scale, const float* inv_freq, float eps, int qk_norm, int v_norm, int window, cudaStream_t st);

// run_decode1 (dense_attn_inst.cu): one row, head dim 128: the fused prep + split launch and the merge launch; returns an error code.
template <int HD>
int run_decode1(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* kv, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, int nsplit,
                float scale, const float* inv_freq, float eps, int qk_norm, cudaStream_t st);

// Attention prep. Per (row t, head): q heads and k heads get an optional RMSNorm over the head dim (weight qn / kn) rounded to bf16, then rotary embedding of the first rd dims
// (half-split pairing, angle = (pos0 + t) * inv_freq[j], inv_freq the rd / 2 fp32 inverse frequencies), q goes to qout, k and v are written to the KV cache at slot (pos0 + t) mod ctx.
// v_norm: v gets the weightless RMSNorm (bf16(x-hat)) before the write. Hkv = 0 is the q-only prep of a layer that reads another layer's cache (k, v, kv unused). One thread per
// element of the head dim, so hd 512 runs 512 threads.
__global__ void k_attn_prep(const bf16* q, const bf16* k, const bf16* v, const bf16* qn, const bf16* kn, bf16* qout, bf16* kv, const int* pos0, int Hq, int Hkv,
                            int hd, int rd, const float* __restrict__ inv_freq, float eps, int ctx, int qk_norm, int v_norm) {
  __shared__ float xs[512];
  __shared__ float sm[32];
  const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
  const int half = rd / 2;
  const float inv = i < rd ? inv_freq[i < half ? i : i - half] : 0.f;  // issued beside the position load
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
    dst = kv + ((size_t)g * ctx + p % ctx) * hd;
  } else {
    const int g = h - Hq - Hkv;
    src = v + ((size_t)t * Hkv + g) * hd;
    dst = kv + (((size_t)Hkv + g) * ctx + p % ctx) * hd;
    rope = false;
    nw = nullptr;
  }
  float x = i < hd ? bf(src[i]) : 0.f;
  if (!rope) {
    if (v_norm) {
      const float ss = block_sum(x * x, sm);
      x = rb(x * rsqrtf(ss / (float)hd + eps));
    }
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
                    int rd, const float* inv_freq, float eps, int ctx, int qk_norm, int v_norm, cudaStream_t st) {
  if (hd > 512 || hd % 32 || rd % 2 || rd > hd) return cudaErrorInvalidValue;
  dim3 grid(Hq + 2 * Hkv, T);
  k_attn_prep<<<grid, hd, 0, st>>>(reinterpret_cast<const bf16*>(q), reinterpret_cast<const bf16*>(k), reinterpret_cast<const bf16*>(v),
                                   reinterpret_cast<const bf16*>(qn), reinterpret_cast<const bf16*>(kn), reinterpret_cast<bf16*>(qout), reinterpret_cast<bf16*>(kv),
                                   pos0, Hq, Hkv, hd, rd, inv_freq, eps, ctx, qk_norm, v_norm);
  return cudaGetLastError();
}

// window 0 = none; with a window the state is a ring of ctx rows (a multiple of CH: a chunk never wraps) and the split blocks cover only the chunks the window touches
int warpc_attn_decode(int T, const void* qprep, const void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd, int ctx, float scale, int window, cudaStream_t st) {
  if (Hq % Hkv || Hq / Hkv > 8 || window < 0 || (window && ctx % CH)) return cudaErrorInvalidValue;
  const int nsplit = attn_nsplit(ctx, window, 1), G = Hq / Hkv;
  if (nsplit > MAX_SPLITS) return cudaErrorInvalidValue;
  dim3 grid(Hkv, nsplit, T);
  const int dp = hd / 2, kg = 128 / dp > 0 ? 128 / dp : 1;
  const size_t smem = ((size_t)G * (hd + CH) + (size_t)kg * G * hd + 16) * sizeof(float);
  if (smem > 48 * 1024) return cudaErrorInvalidValue;
  const bf16* q = reinterpret_cast<const bf16*>(qprep);
  const bf16* kvp = reinterpret_cast<const bf16*>(kv);
  bf16* o = reinterpret_cast<bf16*>(out);
  dim3 g2(Hq, T);
  if (hd == 128) run_decode<128>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, window, st);
  else if (hd == 64) run_decode<64>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, window, st);
  else if (hd == 256) run_decode<256>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, window, st);
  else if (hd == 512) run_decode<512>(grid, g2, smem, q, kvp, pos0, o, ws, Hq, Hkv, ctx, nsplit, scale, window, st);
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}

// split blocks per head of a state of `ctx` rows, `nc` chunks of `ch` keys per block, an optional window (the grid and partial layout of warpc_attn_decode / _x: ch = 64; of
// warpc_attn_decode1: ch = 32)
int warpc_attn_splits(int ctx, int window, int nc, int ch) { return attn_nsplit(ctx, window, nc, ch); }

// one row (decode), head dim 128, rotary dims = head dim: prep folded into the split kernel (see k_attn_fused), then the merge; same partial layout and bits as prep + decode
int warpc_attn_decode_fused(const void* q, const void* k, const void* v, const void* qn, const void* kn, void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd,
                            int rd, const float* inv_freq, float eps, int ctx, int qk_norm, float scale, cudaStream_t st) {
  if (hd != 128 || rd != 128 || Hq % Hkv || Hq / Hkv > 8) return cudaErrorInvalidValue;
  const int nsplit = (ctx + CH - 1) / CH;
  if (nsplit > MAX_SPLITS) return cudaErrorInvalidValue;
  return run_decode1<128>(reinterpret_cast<const bf16*>(q), reinterpret_cast<const bf16*>(k), reinterpret_cast<const bf16*>(v), reinterpret_cast<const bf16*>(qn),
                          reinterpret_cast<const bf16*>(kn), reinterpret_cast<bf16*>(kv), pos0, reinterpret_cast<bf16*>(out), ws, Hq, Hkv, ctx, nsplit, scale, inv_freq,
                          eps, qk_norm, st);
}

// one row, head dim 256 or 512, rotary dims = head dim: prep + split + merge in one launch (k_attn_decode1: chunks of DK keys; the merge is a second launch past MERGE_FUSE chunks). `read`: the q heads
// only, on another layer's cache (k, v, kn unused). cnt: Hkv zeroed counters, left zeroed. window 0 = none (as warpc_attn_decode); partials {o, m, l} with stride hd + 4 for each of warpc_attn_splits(ctx, window, 1, 32) chunks.
int warpc_attn_decode1(const void* q, const void* k, const void* v, const void* qn, const void* kn, void* kv, const int* pos0, void* out, float* ws, unsigned* cnt, int Hq, int Hkv,
                       int hd, int rd, const float* inv_freq, float eps, int ctx, int qk_norm, int v_norm, float scale, int window, int read, cudaStream_t st) {
  if ((hd != 256 && hd != 512) || rd != hd || Hq % Hkv || Hq / Hkv > 8 || window < 0 || (window && ctx % CH)) return cudaErrorInvalidValue;
#define DECODE1(HD, READ)                                                                                                                                                  \
  run_decode1x<HD, READ>(reinterpret_cast<const bf16*>(q), reinterpret_cast<const bf16*>(k), reinterpret_cast<const bf16*>(v), reinterpret_cast<const bf16*>(qn),                \
                         reinterpret_cast<const bf16*>(kn), reinterpret_cast<bf16*>(kv), pos0, reinterpret_cast<bf16*>(out), ws, cnt, Hq, Hkv, ctx, scale, inv_freq, eps,      \
                         qk_norm, v_norm, window, st)
  if (hd == 256) return read ? DECODE1(256, 1) : DECODE1(256, 0);
  return read ? DECODE1(512, 1) : DECODE1(512, 0);
#undef DECODE1
}

// workspace of every decode entry point (kpb = keys per split block, 64 * nc; the hd + 4 floats per partial row also hold the hd + 2 of the nc = 1 kernels)
size_t warpc_attn_ws_bytes(int T, int Hq, int hd, int ctx, int kpb) { return (size_t)T * Hq * ((ctx + kpb - 1) / kpb) * (hd + 4) * sizeof(float); }

// chunked bf16 decode (nc chunks of 64 keys per block); same partial / merge layout as warpc_attn_decode

int warpc_attn_decode_x(int T, const void* qprep, const void* kv, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd, int ctx, float scale, int nc, int window,
                        cudaStream_t st) {
  if (Hq % Hkv || Hq / Hkv > 8 || nc < 1 || window < 0 || (window && ctx % CH)) return cudaErrorInvalidValue;
  const bf16* q = reinterpret_cast<const bf16*>(qprep);
  const bf16* k = reinterpret_cast<const bf16*>(kv);
  bf16* o = reinterpret_cast<bf16*>(out);
  if (hd == 128) return run_decode_x<128>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, window, st);
  if (hd == 64) return run_decode_x<64>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, window, st);
  if (hd == 256) return run_decode_x<256>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, window, st);
  if (hd == 512) return run_decode_x<512>(T, q, k, pos0, o, ws, Hq, Hkv, ctx, scale, nc, window, st);
  return cudaErrorInvalidValue;
}

}  // extern "C"
