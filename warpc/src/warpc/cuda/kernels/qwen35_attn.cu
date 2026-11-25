#include "qwen35_attn.cuh"

// Entry points of the gated hd-256 attention (qwen35_attn.cuh, qwen35_attn_split.cuh). `meta` (13 int64s, schedule.kv_meta) selects the cache: NULL = bf16 [2][Hkv][ctx][256],
// else int4_rot (the hand engine's layout).

static QKV qkv_of(const long long* m) {
  QKV L;
  L.kc = m[0]; L.vc = m[1]; L.ks = m[2]; L.vs = m[3]; L.kr = m[4]; L.vr = m[5]; L.kx = m[6]; L.vx = m[7];
  L.sink = (int)m[8]; L.win = (int)m[9]; L.ring = (int)m[10];
  return L;
}

extern "C" {

int warpc_attn_g_prep(int T, const void* q_proj, const void* k, const void* v, const void* qn, const void* kn, void* qout, void* kv, const long long* meta, const int* pos0, int Hq,
                      int Hkv, int ctx, float eps, float offset, cudaStream_t st) {
  if (T < 1 || Hq < 1 || Hkv < 1) return cudaErrorInvalidValue;
  const bf16* qp = (const bf16*)q_proj;
  if (!meta) {
    const int blocks = (T * (Hq + 2 * Hkv) + 3) / 4;
    g256::k_prep<<<blocks, 128, 0, st>>>(T, qp, (const bf16*)k, (const bf16*)v, (const bf16*)qn, (const bf16*)kn, (bf16*)qout, (bf16*)kv, pos0, Hq, Hkv, ctx, eps, offset);
    return cudaGetLastError();
  }
  const QKV L = qkv_of(meta);
  if (meta[12] != 1 || L.ring < L.win + T) return cudaErrorInvalidValue;
  const int blocks = T * ((Hq + 7) / 8 + 2 * Hkv);
  g256::k_prep_i4<<<blocks, 256, 0, st>>>(T, qp, (const bf16*)k, (const bf16*)v, (const bf16*)qn, (const bf16*)kn, (__half*)qout, (uint8_t*)kv, L, pos0, Hq, Hkv, ctx, eps, offset);
  return cudaGetLastError();
}

// split (at most smax splits per KV head) + gated merge. qprep: bf16 [T][Hq][256] (bf16 cache) or fp16 (int4_rot), from warpc_attn_g_prep; q_proj: the gate source;
// ws: T * Hq * smax * (256 + 4) floats.
int warpc_attn_g_decode(int T, const void* qprep, const void* kv, const long long* meta, const int* pos0, const void* q_proj, void* out, float* ws, int Hq, int Hkv, int ctx,
                        float scale, int smax, int chmin, cudaStream_t st) {
  if (T < 1 || T > 8 || Hkv < 1 || Hq % Hkv || Hq / Hkv > 8 || smax < 1 || smax > 32 || chmin < 16) return cudaErrorInvalidValue;
  const int G = Hq / Hkv, nt = (T * G + 7) / 8;
  g256::SplitArgs a;
  a.q = (const bf16*)qprep;
  a.kv = (const uint8_t*)kv;
  a.L = QKV{};
  if (meta) {
    a.L = qkv_of(meta);
    if (meta[12] != 1) return cudaErrorInvalidValue;
  }
  a.pos0 = pos0;
  a.part = ws;
  a.T = T; a.Hq = Hq; a.Hkv = Hkv; a.ctx = ctx; a.smax = smax; a.chmin = chmin;
  a.scale2 = scale * 1.4426950408889634f;
  int e;
  // key slices per warp column: the hand engine's, for both caches (the output is then bit for bit its)
#define RS(NT, KS) e = g256::run_split<NT, KS>(a, st); break
  if (meta) {
    e = nt <= 3 ? g256::run_split_i4<4>(a, nt, st) : g256::run_split_i4<2>(a, nt, st);
  } else {
    switch (nt) {
      case 1: RS(1, 4);
      case 2: RS(2, 4);
      case 3: RS(3, 2);
      case 4: RS(4, 2);
      case 5: RS(5, 1);
      case 6: RS(6, 1);
      default: return cudaErrorInvalidValue;
    }
  }
#undef RS
  if (e) return e;
  const int win = meta ? a.L.win : 0;
  if (meta) g256::k_merge<2><<<dim3(Hq, T), 64, 0, st>>>(ws, pos0, (const bf16*)q_proj, (bf16*)out, Hq, smax, chmin, win);
  else g256::k_merge<0><<<dim3(Hq, T), 64, 0, st>>>(ws, pos0, (const bf16*)q_proj, (bf16*)out, Hq, smax, chmin, win);
  return cudaGetLastError();
}

}  // extern "C"
