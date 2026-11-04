// swift-mk K4: MTP speculative decoding on the device (draft stage, accept kernel, cycle graphs).  See include/mtp.h.
// K7: sampling graphs (temperature / top-k / top-p, exact speculative sampling) share the structure; kernels in mtp_sample.cu.
#include <atomic>
#include <chrono>
#include <immintrin.h>

#include "mtp_sample.h"
#include "pld.h"

using namespace smk;

namespace {
constexpr int NH = H / 8;  // uint4 chunks per row = 640
}  // namespace

struct smk_spec {
  smk_engine* e;
  smk_mtp_w w;
  int head_mode;
  int* ss;                // device state
  int* ring;              // device output ring
  uint16_t *cat, *hr, *kvbuf;   // cat [8][10240], hr [8][5120] (post-norm MTP hidden of draft j+1), kv of the MTP layer (bf16)
  float* lg;              // draft logits [1][max(nv, VOC)]
  int *pinned_ss, *pinned_ring;
  // K7 sampling state: device config, processed distributions of the verify rows / the draft proposals, top-k workspaces
  SampP* sp = nullptr;
  Dist *pd = nullptr, *qd = nullptr;
  uint64_t *ws_t = nullptr, *ws_d = nullptr;
  int samp = 0;           // 1: launch the sampling graph set
  // cycle graphs, index [samp][k]
  // K11: k = 1..7 (chain k <= 3 + prompt-lookup tail k - 3 tokens), verify rows 1 + k
  cudaGraphExec_t g_cycle[2][SMK_SPEC_PMAX + 1] = {}, g_verify[2][SMK_SPEC_PMAX + 1] = {}, g_stage[2][SMK_SPEC_PMAX + 1] = {}, g_cu[2][SMK_SPEC_PMAX + 1] = {};
  cudaGraphExec_t g_start[2][SMK_SPEC_PMAX + 1] = {}, g_prefill[TM + 1] = {}, g_pld[2][SMK_SPEC_PMAX + 1] = {};
  cudaGraphExec_t g_first = nullptr, g_plain = nullptr, g_smp_t[SMK_SPEC_PMAX + 1] = {}, g_smp_d[SMK_SPEC_KMAX + 1] = {};
  // K11 hybrid draft: host proposer + per-cycle result mailbox (mapped pinned memory written by the last kernel of the cycle graph)
  struct PldRes* res = nullptr;   // host pointer of the mapped mailbox
  struct PldRes* res_dev = nullptr;
  int* pin_tail = nullptr;        // pinned staging of the tail tokens [8]
  smk_pld* pld = nullptr;
  int pld_on = 0, pld_seq = 0,  pld_d[SMK_SPEC_KMAX] = {};
  smk_pld_opts opts = {};
  long long pld_stat[16] = {};    // [0..7] cycles by k (index k), [8] tail tokens proposed, [9] tail cycles, [10] host wait ns, [11] host work ns (lookup + launch)
  // test snapshot (smk_spec_snapshot / restore)
  float* snap_dn = nullptr;
  uint16_t *snap_conv = nullptr, *snap_tap = nullptr;
  int* snap_ds = nullptr;
  int* draw_buf = nullptr;
  int draw_cap = 0;
  size_t bytes = 0;
  unsigned epoch = 0;     // K9: engine head_epoch the graphs above were captured under
};

// ---------------------------------------------------------------------------------------------- kernels
// cat[t] = [ rmsnorm0(embed[tok[t]]; wE) | rmsnorm0(hid[t]; wH) ]   grid (T, 2), NH threads (one uint4 each).
__global__ void __launch_bounds__(NH) k_mtp_prep(const uint4* __restrict__ embed, const int* __restrict__ tok, const uint4* __restrict__ hid,
                                                 const uint4* __restrict__ wE, const uint4* __restrict__ wH, uint4* __restrict__ cat) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, half = blockIdx.y, c = threadIdx.x;
  const uint4* src = half == 0 ? embed + (size_t)tok[t] * NH : hid + (size_t)t * NH;
  const uint4* w = half == 0 ? wE : wH;
  const uint4 hv = src[c];
  float v[8] = {bflo(hv.x), bfhi(hv.x), bflo(hv.y), bfhi(hv.y), bflo(hv.z), bfhi(hv.z), bflo(hv.w), bfhi(hv.w)};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  cat[(size_t)t * 2 * NH + half * NH + c] =
      make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                 pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
}

// out = rmsnorm0(bf16(h[sel] + y[sel]); w) for the single row sel[0] (the other blocks exit).  grid T, NH threads.
__global__ void __launch_bounds__(NH) k_mtp_final(const uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w,
                                                  const int* __restrict__ sel, uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int t = blockIdx.x, c = threadIdx.x;
  if (t != sel[0]) return;
  const uint4 hv = h[(size_t)t * NH + c], yv = y[(size_t)t * NH + c];
  float v[8] = {bf16r(bflo(hv.x) + bflo(yv.x)), bf16r(bfhi(hv.x) + bfhi(yv.x)), bf16r(bflo(hv.y) + bflo(yv.y)), bf16r(bfhi(hv.y) + bfhi(yv.y)),
                bf16r(bflo(hv.z) + bflo(yv.z)), bf16r(bfhi(hv.z) + bfhi(yv.z)), bf16r(bflo(hv.w) + bflo(yv.w)), bf16r(bfhi(hv.w) + bfhi(yv.w))};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH / 32; i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  out[c] = make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                      pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
}

// draft token = head_ids[argmax index]  -> dst
__global__ void k_map_draft(const int* __restrict__ idx, const int* __restrict__ ids, int* __restrict__ dst) { dst[0] = ids[idx[0]]; }

// K11: one-cycle result mailbox in mapped pinned host memory, written by the last kernel of a PLD cycle graph: the host spins on seq, reads the emitted tokens and the new MTP drafts
// (the proposer needs them to pick the width of the next cycle's graph).
struct PldRes {
  volatile int seq;
  int a;
  int y[8];
  int d[SMK_SPEC_KMAX];
  int pad[2];
};
__global__ void k_pld_report(const int* ds, int* ss, PldRes* res) {
  const int l = threadIdx.x;
  const int a = ss[SS_A];
  if (l < 8) res->y[l] = l < a ? ss[SS_MTOK + l] : -1;
  if (l < SMK_SPEC_KMAX) res->d[l] = ds[DS_TOK + 1 + l];
  if (l == 0) res->a = a;
  __threadfence_system();
  __syncwarp();
  if (l == 0) {
    const int sq = ss[SS_PLDSEQ] + 1;
    ss[SS_PLDSEQ] = sq;
    res->seq = sq;
    __threadfence_system();
  }
}

// Greedy acceptance + all device-side bookkeeping.  Rows: token d[t] = ds[TOK+t] (d[0] = anchor), argmax y[t] = ds[OUT+t].
__global__ void k_accept(int* ds, int* ss, int* ring, int k) {
  int m = 0;
  while (m < k && ds[DS_TOK + m + 1] == ds[DS_OUT + m]) m++;
  spec_finish(ds, ss, ring, m, k + 1);
}

// first catch-up after the prefill's last row: pair (hidden of position pos-1, first generated token)
__global__ void k_start(const int* ds, int* ss) {
  ss[SS_A] = 1;
  ss[SS_AM1] = 0;
  ss[SS_DPOS] = ds[DS_POS] - 1;
  ss[SS_DPOS + 1] = ds[DS_POS];
  ss[SS_DPOS + 2] = ds[DS_POS] + 1;
  ss[SS_MTOK] = ds[DS_OUT];
}

// ---------------------------------------------------------------------------------------------- MTP layer passes
#define CKS(x)                                                                                    \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return e_; } \
  } while (0)

// One MTP layer pass over T rows: rows' (token, hidden) -> cat -> fc -> decoder layer (+ final norm of row sel into out unless kv_only).
static cudaError_t mtp_layer(smk_spec* s, int T, const void* hid_src, const int* tok, const int* pos, const int* sel, void* out, bool kv_only, cudaStream_t st) {
  smk_engine* e = s->e;
  const smk_mtp_w& W = s->w;
  const smk_layer_w& L = W.layer;
  k_mtp_prep<<<dim3(T, 2), NH, 0, st>>>((const uint4*)e->w.embed, tok, (const uint4*)hid_src, (const uint4*)W.pre_e, (const uint4*)W.pre_h, (uint4*)s->cat);
  CKS(cudaGetLastError());
  smk_epi ef = {};
  ef.kind = SMK_EPI_SEGS; ef.end[0] = ef.end[1] = ef.end[2] = H; ef.out[0] = e->h;
  CKS(smk_w4r_gemv(W.fc_w, W.fc_s, s->cat, H, 2 * H, T, &ef, st));
  CKS(smk_rmsnorm(e->h, nullptr, L.ln1, e->xn, nullptr, T, H, 1e-6f, st));
  smk_epi eq = {};
  eq.kind = SMK_EPI_SEGS; eq.end[0] = 12288; eq.end[1] = 13312; eq.end[2] = 14336; eq.out[0] = e->qb; eq.out[1] = e->kb; eq.out[2] = e->vb;
  CKS(smk_w4r_gemv(L.qkv_w, L.qkv_s, e->xn, 14336, H, T, &eq, st));
  CKS(swiftmk_attn_step(T, e->qb, e->kb, e->vb, L.q_norm, L.k_norm, s->kvbuf, e->ctx_max, 0, 1.f, 1.f, pos, e->y6, e->ws_attn, st));
  if (kv_only) return cudaSuccess;
  smk_epi eo = {};
  eo.kind = SMK_EPI_SEGS; eo.end[0] = eo.end[1] = eo.end[2] = H; eo.out[0] = e->mix;
  CKS(smk_w4r_gemv(L.o_w, L.o_s, e->y6, H, 6144, T, &eo, st));
  CKS(smk_rmsnorm(e->h, e->mix, L.ln2, e->xn, nullptr, T, H, 1e-6f, st));
  smk_epi eg = {};
  eg.kind = SMK_EPI_SILU; eg.out[0] = e->act;
  CKS(smk_w4r_gemv(L.gu_w, L.gu_s, e->xn, 2 * I, H, T, &eg, st));
  smk_epi ed = {};
  ed.kind = SMK_EPI_SEGS; ed.end[0] = ed.end[1] = ed.end[2] = H; ed.out[0] = e->dn;
  CKS(smk_w4r_gemv(L.dn_w, L.dn_s, e->act, H, I, T, &ed, st));
  k_mtp_final<<<T, NH, 0, st>>>((const uint4*)e->h, (const uint4*)e->dn, (const uint4*)W.norm, sel, (uint4*)out);
  return cudaGetLastError();
}

// draft j (1-based) = argmax of the draft head applied to hr[j-1]  -> ds[TOK + j];  samp: token sampled from q (top-k/top-p of the head logits; q kept in qd[j-1])
static cudaError_t mtp_head(smk_spec* s, int j, bool samp, cudaStream_t st) {
  smk_engine* e = s->e;
  const void* x = s->hr + (size_t)(j - 1) * H;
  int* dst = e->ds + DS_TOK + j;
  if (samp) {
    if (s->head_mode == 1) return cudaErrorNotSupported;  // diagnostic full-vocab head: greedy only
    smk_epi ep = {};
    ep.kind = SMK_EPI_F32; ep.out[0] = s->lg;
    CKS(smk_w4r_gemv(s->w.head_w, s->w.head_s, x, s->w.head_nv, H, 1, &ep, st));
    CKS(sk_topk(s->lg, 1, s->w.head_nv, s->w.head_nv, s->w.head_ids, s->sp, 1, s->ws_d, s->qd + (j - 1), st));
    return sk_draft_pick(s->qd + (j - 1), s->sp, s->ss, j, dst, st);
  }
  if (s->head_mode == 1) {
    CKS(smk_bf16_gemv_f32(e->w.lm_head, x, VOC, H, 1, s->lg, st));
    return smk_argmax(s->lg, 1, VOC, dst, nullptr, e->ws_am, st);
  }
  smk_epi ep = {};
  ep.kind = SMK_EPI_F32; ep.out[0] = s->lg;
  CKS(smk_w4r_gemv(s->w.head_w, s->w.head_s, x, s->w.head_nv, H, 1, &ep, st));
  CKS(smk_argmax(s->lg, 1, s->w.head_nv, s->ss + SS_IDX, nullptr, e->ws_am, st));
  k_map_draft<<<1, 1, 0, st>>>(s->ss + SS_IDX, s->w.head_ids, dst);
  return cudaGetLastError();
}

// Catch-up of Tcu rows (hidden from the engine's final-norm tap, tokens ss[MTOK], a = ss[A] valid) + ndraft drafts.
static cudaError_t mtp_stage(smk_spec* s, int Tcu, int ndraft, bool kv_only, bool samp, cudaStream_t st) {
  smk_engine* e = s->e;
  const void* hid = e->dbg + (size_t)(NLAYER + 1) * TM * H;
  CKS(mtp_layer(s, Tcu, hid, s->ss + SS_MTOK, s->ss + SS_DPOS, s->ss + SS_AM1, s->hr, kv_only, st));
  for (int j = 1; j <= ndraft; j++) {
    CKS(mtp_head(s, j, samp, st));
    if (j < ndraft) CKS(mtp_layer(s, 1, s->hr + (size_t)(j - 1) * H, e->ds + DS_TOK + j, s->ss + SS_DPOS + j, s->ss + SS_ZERO, s->hr + (size_t)j * H, false, st));
  }
  return cudaSuccess;
}

// sampling kernels of the verify step: top-k / top-p of the 1+k logits rows -> pd, then the speculative-sampling accept kernel (bookkeeping included)
static cudaError_t sample_accept(smk_spec* s, int k, cudaStream_t st) {
  smk_engine* e = s->e;
  CKS(sk_topk(e->logits, 1 + k, VOC, VOC, nullptr, s->sp, 0, s->ws_t, s->pd, st));
  if (k > SMK_SPEC_KMAX) CKS(sk_delta_q(e->ds, s->qd, SMK_SPEC_KMAX, k, st));  // K11: prompt-lookup tail tokens are delta proposals
  return sk_accept(e->ds, s->ss, s->ring, k, s->pd, s->qd, s->sp, st);
}

// the cached graphs contain the engine's head step: recapture after smk_engine_set_head (K9)
static void drop_graphs(smk_spec* s) {
  for (int m = 0; m < 2; m++)
    for (auto* arr : {s->g_cycle[m], s->g_verify[m], s->g_stage[m], s->g_cu[m], s->g_start[m], s->g_pld[m]})
      for (int i = 0; i <= SMK_SPEC_PMAX; i++) if (arr[i]) { cudaGraphExecDestroy(arr[i]); arr[i] = nullptr; }
  for (int i = 0; i <= SMK_SPEC_PMAX; i++) if (s->g_smp_t[i]) { cudaGraphExecDestroy(s->g_smp_t[i]); s->g_smp_t[i] = nullptr; }
  for (int i = 0; i <= SMK_SPEC_KMAX; i++) if (s->g_smp_d[i]) { cudaGraphExecDestroy(s->g_smp_d[i]); s->g_smp_d[i] = nullptr; }  // g_smp_d has KMAX + 1 entries (the old shared PMAX loop ran past it into res / res_dev / pin_tail / pld)
  for (int i = 0; i <= TM; i++) if (s->g_prefill[i]) { cudaGraphExecDestroy(s->g_prefill[i]); s->g_prefill[i] = nullptr; }
  if (s->g_first) { cudaGraphExecDestroy(s->g_first); s->g_first = nullptr; }
  if (s->g_plain) { cudaGraphExecDestroy(s->g_plain); s->g_plain = nullptr; }
}
static void sync_epoch(smk_spec* s) {
  if (s->epoch != s->e->head_epoch) {
    drop_graphs(s);
    s->epoch = s->e->head_epoch;
  }
}

static cudaError_t verify_accept(smk_spec* s, int k, bool samp, cudaStream_t st) {
  smk_engine* e = s->e;
  CKS(smk_run_step(e, 1 + k, 1, 1, st, samp));
  if (samp) return sample_accept(s, k, st);
  k_accept<<<1, 1, 0, st>>>(e->ds, s->ss, s->ring, k);
  return cudaGetLastError();
}

extern "C" {

smk_spec* smk_spec_create(smk_engine* e, const smk_mtp_w* w, int head_mode) {
  smk_spec* s = new smk_spec();
  s->e = e;
  s->w = *w;
  s->head_mode = head_mode;
  const size_t b0 = e->bytes;
  s->ss = (int*)dalloc(e, SS_N * sizeof(int), true);
  s->ring = (int*)dalloc(e, RING * sizeof(int), true);
  s->cat = (uint16_t*)dalloc(e, (size_t)TM * 2 * H * 2);
  s->hr = (uint16_t*)dalloc(e, (size_t)TM * H * 2, true);
  s->kvbuf = (uint16_t*)dalloc(e, swiftmk_attn_kv_bytes(e->ctx_max, 0), true);
  s->lg = (float*)dalloc(e, (size_t)(head_mode == 1 ? VOC : w->head_nv) * 4);
  s->sp = (SampP*)dalloc(e, sizeof(SampP), true);
  s->pd = (Dist*)dalloc(e, sizeof(Dist) * SK_ROWS, true);
  s->qd = (Dist*)dalloc(e, sizeof(Dist) * SMK_SPEC_PMAX, true);
  s->ws_t = (uint64_t*)dalloc(e, sizeof(uint64_t) * SK_WS_ENTRIES, true);
  s->ws_d = (uint64_t*)dalloc(e, sizeof(uint64_t) * SK_WS_ENTRIES, true);
  if (cudaHostAlloc((void**)&s->res, sizeof(PldRes), cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer((void**)&s->res_dev, s->res, 0) != cudaSuccess) { smk_spec_destroy(s); return nullptr; }
  memset((void*)s->res, 0, sizeof(PldRes));
  cudaMallocHost((void**)&s->pin_tail, 8 * sizeof(int));
  cudaMallocHost((void**)&s->pinned_ss, SS_N * sizeof(int));
  cudaMallocHost((void**)&s->pinned_ring, RING * sizeof(int));
  s->bytes = e->bytes - b0;
  if (e->failed || cudaGetLastError() != cudaSuccess) { smk_spec_destroy(s); return nullptr; }
  smk_sampling dflt = {};
  dflt.temperature = 1.0f; dflt.top_k = 20; dflt.top_p = 1.0f; dflt.draft_sample = 1;
  SampP p0 = smk_make_sampp(&dflt);
  p0.pr = e->pb.dev;
  if (cudaMemcpy(s->sp, &p0, sizeof(p0), cudaMemcpyHostToDevice) != cudaSuccess) { smk_spec_destroy(s); return nullptr; }
  return s;
}

void smk_spec_destroy(smk_spec* s) {
  if (!s) return;
  cudaDeviceSynchronize();
  drop_graphs(s);
  if (s->draw_buf) cudaFree(s->draw_buf);
  if (s->pinned_ss) cudaFreeHost(s->pinned_ss);
  if (s->res) cudaFreeHost((void*)s->res);
  if (s->pin_tail) cudaFreeHost(s->pin_tail);
  if (s->pld) smk_pld_destroy(s->pld);
  if (s->pinned_ring) cudaFreeHost(s->pinned_ring);
  delete s;  // device buffers belong to the engine's allocation list
}

cudaError_t smk_spec_reset(smk_spec* s, cudaStream_t st) {
  smk_engine* e = s->e;
  cudaMemsetAsync(s->ss, 0, SS_N * sizeof(int), st);
  cudaMemsetAsync(s->ring, 0, RING * sizeof(int), st);
  cudaMemsetAsync(s->kvbuf, 0, swiftmk_attn_kv_bytes(e->ctx_max, 0), st);
  return cudaGetLastError();
}

cudaError_t smk_spec_reset_counters(smk_spec* s, cudaStream_t st) {
  cudaMemsetAsync(s->ss, 0, SS_N * sizeof(int), st);
  cudaMemsetAsync(s->ring, 0, RING * sizeof(int), st);
  return cudaGetLastError();
}

cudaError_t smk_spec_prefill_chunk(smk_spec* s, int T, const int* next_tok, int pos0, cudaStream_t st) {
  if (T < 1 || T > TM) return cudaErrorInvalidValue;
  sync_epoch(s);
  CKS(cudaMemcpyAsync(s->ss + SS_MTOK, next_tok, T * sizeof(int), cudaMemcpyHostToDevice, st));
  CKS(cudaMemcpyAsync(s->ss + SS_DPOS, &pos0, sizeof(int), cudaMemcpyHostToDevice, st));
  return smk_graph_launch(s->e, &s->g_prefill[T], [&](cudaStream_t c) { return mtp_stage(s, T, 0, true, false, c); }, st);
}

cudaError_t smk_spec_start(smk_spec* s, int k, cudaStream_t st) {
  if (k < 1 || k > SMK_SPEC_KMAX) return cudaErrorInvalidValue;
  sync_epoch(s);
  const bool samp = s->samp;
  return smk_graph_launch(s->e, &s->g_start[samp][k], [&](cudaStream_t c) {
    k_start<<<1, 1, 0, c>>>(s->e->ds, s->ss);
    CKS(cudaGetLastError());
    return mtp_stage(s, 1, k, false, samp, c);
  }, st);
}

cudaError_t smk_spec_set_force(smk_spec* s, const int* force, int n, cudaStream_t st) {
  if (n < 0 || n > 8) return cudaErrorInvalidValue;
  CKS(cudaMemcpyAsync(s->ss + SS_FORCE_N, &n, sizeof(int), cudaMemcpyHostToDevice, st));
  if (n) CKS(cudaMemcpyAsync(s->ss + SS_FORCE, force, n * sizeof(int), cudaMemcpyHostToDevice, st));
  return cudaSuccess;
}

static inline int nchain(int k) { return k < SMK_SPEC_KMAX ? k : SMK_SPEC_KMAX; }  // MTP chain drafts of a cycle with k drafts (k - 3 more are prompt-lookup tail tokens)

static cudaError_t launch_part(smk_spec* s, int k, int which, cudaStream_t st) {
  if (k < 1 || k > SMK_SPEC_PMAX || k + 1 > s->e->tmax) return cudaErrorInvalidValue;
  sync_epoch(s);
  const bool samp = s->samp;
  switch (which) {
    case 0:
      return smk_graph_launch(s->e, &s->g_cycle[samp][k], [&](cudaStream_t c) {
        CKS(verify_accept(s, k, samp, c));
        return mtp_stage(s, k + 1, nchain(k), false, samp, c);
      }, st);
    case 1: return smk_graph_launch(s->e, &s->g_verify[samp][k], [&](cudaStream_t c) { return verify_accept(s, k, samp, c); }, st);
    case 2: return smk_graph_launch(s->e, &s->g_stage[samp][k], [&](cudaStream_t c) { return mtp_stage(s, k + 1, nchain(k), false, samp, c); }, st);
    case 3: return smk_graph_launch(s->e, &s->g_cu[samp][k], [&](cudaStream_t c) { return mtp_stage(s, k + 1, 0, false, samp, c); }, st);
    case 4:  // sampling kernels of the verify rows (top-k / top-p of 1+k rows + accept), on the logits of the last verify
      return smk_graph_launch(s->e, &s->g_smp_t[k], [&](cudaStream_t c) { return sample_accept(s, k, c); }, st);
    case 5:  // sampling kernels of the k draft heads (top-k / top-p of the 64k head logits + draw), on the logits of the last draft
      if (k > SMK_SPEC_KMAX) return cudaErrorInvalidValue;
      return smk_graph_launch(s->e, &s->g_smp_d[k], [&](cudaStream_t c) {
        for (int j = 1; j <= k; j++) {
          CKS(sk_topk(s->lg, 1, s->w.head_nv, s->w.head_nv, s->w.head_ids, s->sp, 1, s->ws_d, s->qd + (j - 1), c));
          CKS(sk_draft_pick(s->qd + (j - 1), s->sp, s->ss, j, s->e->ds + DS_TOK + j, c));
        }
        return cudaSuccess;
      }, st);
  }
  return cudaErrorInvalidValue;
}

cudaError_t smk_spec_launch(smk_spec* s, int k, int which, int n, cudaStream_t st) {
  for (int i = 0; i < n; i++) CKS(launch_part(s, k, which, st));
  return cudaSuccess;
}

cudaError_t smk_spec_run(smk_spec* s, int k, int n, cudaStream_t st, int* host_ss, int* host_ring) {
  CKS(smk_spec_launch(s, k, 0, n, st));
  CKS(cudaMemcpyAsync(s->pinned_ss, s->ss, SS_N * sizeof(int), cudaMemcpyDeviceToHost, st));
  CKS(cudaMemcpyAsync(s->pinned_ring, s->ring, RING * sizeof(int), cudaMemcpyDeviceToHost, st));
  CKS(cudaStreamSynchronize(st));
  memcpy(host_ss, s->pinned_ss, SS_N * sizeof(int));
  memcpy(host_ring, s->pinned_ring, RING * sizeof(int));
  return cudaSuccess;
}

// ---------------------------------------------------------------------------------------------- K11 hybrid draft
cudaError_t smk_spec_pld_config(smk_spec* s, const smk_pld_opts* opts) {
  if (s->pld) { smk_pld_destroy(s->pld); s->pld = nullptr; }
  s->pld_on = 0;
  memset(s->pld_stat, 0, sizeof(s->pld_stat));
  if (!opts) return cudaSuccess;
  if (s->e->tmax < 4 + 1 && opts->cfg.max_tail > 0) return cudaErrorInvalidValue;
  s->opts = *opts;
  if (s->opts.cfg.max_tail > s->e->tmax - 4) s->opts.cfg.max_tail = s->e->tmax - 4;
  s->pld = smk_pld_create(&s->opts.cfg);
  if (!s->pld) return cudaErrorMemoryAllocation;
  s->pld_on = 1;
  return cudaSuccess;
}

cudaError_t smk_spec_pld_begin(smk_spec* s, const int* ctx, int n, cudaStream_t st) {
  if (!s->pld_on || n < 1) return cudaErrorInvalidValue;
  CKS(cudaStreamSynchronize(st));
  int d[SMK_SPEC_KMAX];
  CKS(cudaMemcpy(d, s->e->ds + DS_TOK + 1, sizeof(d), cudaMemcpyDeviceToHost));
  for (int i = 0; i < SMK_SPEC_KMAX; i++) s->pld_d[i] = d[i];
  smk_pld_reset(s->pld, ctx, n);
  const int zero = 0;
  CKS(cudaMemcpy(s->ss + SS_PLDSEQ, &zero, sizeof(int), cudaMemcpyHostToDevice));
  s->res->seq = 0;
  s->pld_seq = 0;
  return cudaSuccess;
}

static inline long long now_ns() { return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// the hybrid cycle graph of k drafts: verify T = k + 1 rows + accept + MTP catch-up (k + 1 rows) + 3 chain drafts + the result mailbox write
static cudaError_t pld_cycle(smk_spec* s, int k, bool samp, cudaStream_t c) {
  CKS(verify_accept(s, k, samp, c));
  CKS(mtp_stage(s, k + 1, SMK_SPEC_KMAX, false, samp, c));
  k_pld_report<<<1, 32, 0, c>>>(s->e->ds, s->ss, s->res_dev);
  return cudaGetLastError();
}

cudaError_t smk_spec_pld_prewarm(smk_spec* s) {
  if (!s->pld_on) return cudaErrorInvalidValue;
  sync_epoch(s);
  for (int samp = 0; samp < 2; samp++) {
    if (samp && (s->head_mode == 1 || s->w.head_nv > SK_NCH_D * SK_CHUNK)) continue;  // sampling graphs are not available (smk_spec_set_sampling refuses them too)
    for (int k = SMK_SPEC_KMAX; k <= SMK_SPEC_KMAX + s->opts.cfg.max_tail; k++)
      CKS(smk_graph_capture(s->e, &s->g_pld[samp][k], [&](cudaStream_t c) { return pld_cycle(s, k, samp != 0, c); }));
  }
  return cudaSuccess;
}

cudaError_t smk_spec_run_pld(smk_spec* s, int n, cudaStream_t st, int* host_ss, int* host_ring) {
  if (!s->pld_on) return cudaErrorInvalidValue;
  smk_engine* e = s->e;
  sync_epoch(s);
  const bool samp = s->samp;
  const smk_pld_opts& o = s->opts;
  for (int i = 0; i < n; i++) {
    const long long t0 = now_ns();
    int tail[8] = {}, l = 0;
    int ln = smk_pld_propose(s->pld, s->pld_d, SMK_SPEC_KMAX, tail, &l);
    if (o.force_n > 0) { ln = o.force_n < 4 ? o.force_n : 4; for (int j = 0; j < ln; j++) tail[j] = o.force_tok[j]; }
    const int k = SMK_SPEC_KMAX + ln;
    if (ln > 0) {
      memcpy(s->pin_tail, tail, ln * sizeof(int));
      CKS(cudaMemcpyAsync(e->ds + DS_TOK + 1 + SMK_SPEC_KMAX, s->pin_tail, ln * sizeof(int), cudaMemcpyHostToDevice, st));
      s->pld_stat[8] += ln;
      s->pld_stat[9]++;
    }
    s->pld_stat[k]++;
    CKS(smk_graph_launch(e, &s->g_pld[samp][k], [&](cudaStream_t c) { return pld_cycle(s, k, samp, c); }, st));
    const long long t1 = now_ns();
    const int want = ++s->pld_seq;
    uint64_t spins = 0;
    while (s->res->seq != want) {
      _mm_pause();
      if ((++spins & 0xfffff) == 0) {
        const cudaError_t q = cudaStreamQuery(st);
        if (q != cudaSuccess && q != cudaErrorNotReady) return q;
        if (now_ns() - t1 > 30000000000LL) return cudaErrorLaunchTimeout;
      }
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    const long long t2 = now_ns();
    const int a = s->res->a;
    int y[8];
    for (int j = 0; j < 8; j++) y[j] = s->res->y[j];
    for (int j = 0; j < SMK_SPEC_KMAX; j++) s->pld_d[j] = s->res->d[j];
    smk_pld_append(s->pld, y, a);
    s->pld_stat[10] += t2 - t1;
    s->pld_stat[11] += (t1 - t0) + (now_ns() - t2);
  }
  CKS(cudaMemcpyAsync(s->pinned_ss, s->ss, SS_N * sizeof(int), cudaMemcpyDeviceToHost, st));
  CKS(cudaMemcpyAsync(s->pinned_ring, s->ring, RING * sizeof(int), cudaMemcpyDeviceToHost, st));
  CKS(cudaStreamSynchronize(st));
  memcpy(host_ss, s->pinned_ss, SS_N * sizeof(int));
  memcpy(host_ring, s->pinned_ring, RING * sizeof(int));
  return cudaSuccess;
}

cudaError_t smk_spec_pld_stats(smk_spec* s, long long* out) {
  memcpy(out, s->pld_stat, sizeof(s->pld_stat));
  return cudaSuccess;
}

cudaError_t smk_spec_read(smk_spec* s, int what, void* host, size_t bytes) {
  const void* src = what == 0 ? (const void*)s->ss : what == 1 ? (const void*)s->hr : what == 2 ? (const void*)s->pd : what == 3 ? (const void*)s->qd : nullptr;
  const size_t cap = what == 0 ? SS_N * sizeof(int) : what == 2 ? sizeof(Dist) * SK_ROWS : what == 3 ? sizeof(Dist) * SMK_SPEC_PMAX : (size_t)TM * H * 2;
  if (!src || bytes > cap) return cudaErrorInvalidValue;
  return cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost);
}

size_t smk_spec_bytes(smk_spec* s) { return s->bytes; }
// MTP layer KV cache (bf16 [2][4][ctx_max][256]); used by the fast prefill (prefill.cu) to commit prompt pairs.
void* smk_spec_kv_ptr(smk_spec* s) { return s->kvbuf; }

// ---------------------------------------------------------------------------------------------- K7 sampling
cudaError_t smk_spec_set_sampling(smk_spec* s, const smk_sampling* cfg, cudaStream_t st) {
  if (cfg->temperature == 0.f) {  // greedy graphs; the sampling config is left as it was
    s->samp = 0;
    s->e->hsamp = 0;
    return cudaSuccess;
  }
  if (!smk_sampling_valid(cfg)) return cudaErrorInvalidValue;
  if (s->head_mode == 1 || s->w.head_nv > SK_NCH_D * SK_CHUNK) return cudaErrorNotSupported;
  SampP p = smk_make_sampp(cfg);
  p.pr = s->e->pb.dev;
  CKS(cudaMemcpyAsync(s->sp, &p, sizeof(p), cudaMemcpyHostToDevice, st));
  s->samp = 1;
  s->e->hsamp = 1;  // K9: the engine's head steps rescore R = 64 candidates so the device top-k (<= 64) sees exact logits
  return cudaSuccess;
}

cudaError_t smk_spec_sample_first(smk_spec* s, cudaStream_t st) {
  if (!s->samp) return cudaErrorInvalidValue;
  sync_epoch(s);
  return smk_graph_launch(s->e, &s->g_first, [&](cudaStream_t c) {
    CKS(sk_topk(s->e->logits, 1, VOC, VOC, nullptr, s->sp, 0, s->ws_t, s->pd, c));
    return sk_first_pick(s->pd, s->sp, s->ss, s->e->ds, 0, c);
  }, st);
}

cudaError_t smk_spec_plain_run(smk_spec* s, int n, int* host_tokens, cudaStream_t st) {
  if (!s->samp) return cudaErrorInvalidValue;
  sync_epoch(s);
  smk_engine* e = s->e;
  for (int base = 0; base < n; base += 65536) {
    const int m = n - base < 65536 ? n - base : 65536;
    for (int i = 0; i < m; i++) {
      CKS(smk_graph_launch(e, &s->g_plain, [&](cudaStream_t c) {
        CKS(smk_run_step(e, 1, 1, 0, c, 1));  // advances pos/cur and feeds back the argmax; the pick below overwrites the token
        CKS(sk_topk(e->logits, 1, VOC, VOC, nullptr, s->sp, 0, s->ws_t, s->pd, c));
        return sk_first_pick(s->pd, s->sp, s->ss, e->ds, 1, c);
      }, st));
      CKS(cudaMemcpyAsync(e->pinned + i, e->ds + DS_OUT, sizeof(int), cudaMemcpyDeviceToHost, st));
    }
    CKS(cudaStreamSynchronize(st));
    memcpy(host_tokens + base, e->pinned, m * sizeof(int));
  }
  return cudaSuccess;
}

// copy slot ds_src[DS_CUR] of the DeltaNet state / conv ring of every layer to (dir 0) / from (dir 1) the snapshot buffers
__global__ void k_snap(float4* __restrict__ state, uint32_t* __restrict__ conv, const int* __restrict__ ds_src, float4* __restrict__ snap_s, uint32_t* __restrict__ snap_c, int dir) {
  constexpr int SF4 = SWIFTMK_DN_STATE_SLOT_FLOATS / 4, CW = SWIFTMK_DN_CONV_SLOT_ELEMS / 2;
  const int li = blockIdx.y, cur = ds_src[DS_CUR];
  const int slots = ds_src[DS_SLOTS];
  float4* a = state + ((size_t)li * slots + cur) * SF4;
  float4* b = snap_s + (size_t)li * SF4;
  uint32_t* ca = conv + ((size_t)li * slots + cur) * CW;
  uint32_t* cb = snap_c + (size_t)li * CW;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < SF4; i += gridDim.x * blockDim.x) {
    if (dir == 0) b[i] = a[i]; else a[i] = b[i];
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < CW; i += gridDim.x * blockDim.x) {
    if (dir == 0) cb[i] = ca[i]; else ca[i] = cb[i];
  }
}

static cudaError_t snap_copy(smk_spec* s, int dir, cudaStream_t st) {
  smk_engine* e = s->e;
  const int nd = 48;
  const size_t tap = (size_t)(NLAYER + 1) * TM * H;  // final-norm tap, row 0
  if (!s->snap_dn) {
    s->snap_dn = (float*)dalloc(e, (size_t)nd * SWIFTMK_DN_STATE_SLOT_FLOATS * 4);
    s->snap_conv = (uint16_t*)dalloc(e, (size_t)nd * SWIFTMK_DN_CONV_SLOT_ELEMS * 2);
    s->snap_tap = (uint16_t*)dalloc(e, (size_t)H * 2);
    s->snap_ds = (int*)dalloc(e, DS_N * sizeof(int));
    if (e->failed) return cudaErrorMemoryAllocation;
  }
  if (dir == 0) {
    CKS(swiftmk_deltanet_materialize(e->dn_state, e->dn_dbuf, nd, e->ds + DS_CUR, e->ds + DS_PEND, e->slots, st));
    CKS(cudaMemcpyAsync(s->snap_ds, e->ds, DS_N * sizeof(int), cudaMemcpyDeviceToDevice, st));
    CKS(cudaMemcpyAsync(s->snap_tap, e->dbg + tap, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
  }
  k_snap<<<dim3(32, nd), 256, 0, st>>>((float4*)e->dn_state, (uint32_t*)e->dn_conv, dir == 0 ? e->ds : s->snap_ds, (float4*)s->snap_dn, (uint32_t*)s->snap_conv, dir);
  CKS(cudaGetLastError());
  if (dir == 1) {
    CKS(cudaMemcpyAsync(e->ds, s->snap_ds, DS_N * sizeof(int), cudaMemcpyDeviceToDevice, st));
    CKS(cudaMemcpyAsync(e->dbg + tap, s->snap_tap, (size_t)H * 2, cudaMemcpyDeviceToDevice, st));
  }
  return cudaSuccess;
}
cudaError_t smk_spec_snapshot(smk_spec* s, cudaStream_t st) { return snap_copy(s, 0, st); }
cudaError_t smk_spec_restore(smk_spec* s, cudaStream_t st) { return s->snap_dn ? snap_copy(s, 1, st) : cudaErrorInvalidValue; }

cudaError_t smk_test_draw(smk_spec* s, int n, unsigned ctr0, int row, int* out) {
  if (!s->samp || n < 1 || row < 0 || row >= TM) return cudaErrorInvalidValue;
  smk_engine* e = s->e;
  CKS(cudaDeviceSynchronize());
  if (s->draw_cap < n) {
    if (s->draw_buf) cudaFree(s->draw_buf);
    CKS(cudaMalloc(&s->draw_buf, (size_t)n * 4));
    s->draw_cap = n;
  }
  CKS(sk_topk(e->logits + (size_t)row * VOC, 1, VOC, VOC, nullptr, s->sp, 0, s->ws_t, s->pd, 0));
  CKS(sk_draw_n(s->pd, s->sp, n, ctr0, s->draw_buf, 0));
  return cudaMemcpy(out, s->draw_buf, (size_t)n * 4, cudaMemcpyDeviceToHost);
}

}  // extern "C"
