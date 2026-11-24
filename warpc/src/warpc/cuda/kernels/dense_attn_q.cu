#include "dense_attn_common.cuh"

// quantised caches. meta[0..7] = byte offsets kc, vc, ks, vs, kr, vr, kx, vx; meta[8..10] = sink, win, ring slots; meta[11] = V bits (8 or 4)
static QKV qkv_of(const long long* m) {
  QKV L;
  L.kc = m[0]; L.vc = m[1]; L.ks = m[2]; L.vs = m[3]; L.kr = m[4]; L.vr = m[5]; L.kx = m[6]; L.vx = m[7];
  L.sink = (int)m[8]; L.win = (int)m[9]; L.ring = (int)m[10];
  return L;
}

template <int HD, int VB, int EX>
static int decode_q(int T, const bf16* q, const uint8_t* kvb, const QKV& L, const int* pos0, bf16* o, float* ws, int Hq, int Hkv, int ctx, float scale, int nc, cudaStream_t st) {
  const int e = run_split_q<HD, VB, EX>(T, q, kvb, L, pos0, ws, Hq, Hkv, ctx, scale, nc, st);
  if (e) return e;
  run_merge_q<HD>(T, pos0, o, ws, Hq, (ctx + CH * nc - 1) / (CH * nc), CH * nc, st);
  return cudaGetLastError();
}
extern "C" {

int warpc_attn_prep_q(int T, const void* q, const void* k, const void* v, const void* qn, const void* kn, void* qout, void* kv, const long long* meta, const int* pos0, int Hq,
                      int Hkv, int hd, int rd, const float* inv_freq, float eps, int ctx, int qk_norm, cudaStream_t st) {
  if (hd > 256 || hd % 32 || rd % 2 || rd > hd || (hd & (hd - 1))) return cudaErrorInvalidValue;
  const QKV L = qkv_of(meta);
  const int vb = (int)meta[11], ex = (L.sink > 0 || L.win > 0);
  if (L.win && L.ring < L.win + T) return cudaErrorInvalidValue;  // a step must not evict a row it reads
#define PQ(HD, VB, EX)                                                                                                                                              \
  return run_prep_q<HD, VB, EX>(T, (const bf16*)q, (const bf16*)k, (const bf16*)v, (const bf16*)qn, (const bf16*)kn, (bf16*)qout, (uint8_t*)kv, L, pos0, Hq, Hkv, rd, inv_freq, \
                                eps, ctx, qk_norm, st)
  if (vb != 8 && vb != 4) return cudaErrorInvalidValue;
  if (hd == 128) { if (vb == 8) { if (ex) PQ(128, 8, 1); else PQ(128, 8, 0); } else { if (ex) PQ(128, 4, 1); else PQ(128, 4, 0); } }
  if (hd == 64) { if (vb == 8) { if (ex) PQ(64, 8, 1); else PQ(64, 8, 0); } else { if (ex) PQ(64, 4, 1); else PQ(64, 4, 0); } }
  if (hd == 256) { if (vb == 8) { if (ex) PQ(256, 8, 1); else PQ(256, 8, 0); } else { if (ex) PQ(256, 4, 1); else PQ(256, 4, 0); } }
#undef PQ
  return cudaErrorInvalidValue;
}

int warpc_attn_decode_q(int T, const void* qprep, const void* kv, const long long* meta, const int* pos0, void* out, float* ws, int Hq, int Hkv, int hd, int ctx, float scale, int nc,
                        cudaStream_t st) {
  if (Hq % Hkv || Hq / Hkv > 8 || nc < 1) return cudaErrorInvalidValue;
  const QKV L = qkv_of(meta);
  const int vb = (int)meta[11], ex = (L.sink > 0 || L.win > 0);
#define DQ(HD, VB, EX) return decode_q<HD, VB, EX>(T, (const bf16*)qprep, (const uint8_t*)kv, L, pos0, (bf16*)out, ws, Hq, Hkv, ctx, scale, nc, st)
  if (vb != 8 && vb != 4) return cudaErrorInvalidValue;
  if (hd == 128) { if (vb == 8) { if (ex) DQ(128, 8, 1); else DQ(128, 8, 0); } else { if (ex) DQ(128, 4, 1); else DQ(128, 4, 0); } }
  if (hd == 64) { if (vb == 8) { if (ex) DQ(64, 8, 1); else DQ(64, 8, 0); } else { if (ex) DQ(64, 4, 1); else DQ(64, 4, 0); } }
  if (hd == 256) { if (vb == 8) { if (ex) DQ(256, 8, 1); else DQ(256, 8, 0); } else { if (ex) DQ(256, 4, 1); else DQ(256, 4, 0); } }
#undef DQ
  return cudaErrorInvalidValue;
}

}  // extern "C"
