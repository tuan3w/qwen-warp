// swift-mk: batched speculative decode of several sequences in one weight pass per cycle (include/batch.h).
#include "batch.h"
#include "spec_impl.h"

using namespace smk;

namespace {
constexpr int BMAX = SMK_BATCH_MAX, KMAX = SMK_SPEC_KMAX, PMAX = SMK_SPEC_PMAX;
static_assert(BMAX <= TM, "one row per sequence at least");

// Device table entry of a batch position: where the sequence's big state lives (read by the DeltaNet / attention kernels through swiftmk_*_ind).
struct SeqPtrs {
  float* dn_state;
  void* dn_conv;
  float* dn_dbuf;
  void* kv;
  void* mtp_kv;
};
// What the gather / scatter kernels copy between a sequence's own buffers and its batch position.
struct SlotDesc {
  int* ds;
  int* ss;
  SampP* sp;
  ProcP* pb;
  Dist* qd;
  SeqPtrs ptrs;
  int greedy;
  int pad;
};

template <class T>
__device__ __forceinline__ void copy_words(T* dst, const T* src, int tid, int nt) {
  static_assert(sizeof(T) % 4 == 0, "word copy");
  const int n = (int)(sizeof(T) / 4);
  for (int i = tid; i < n; i += nt) reinterpret_cast<int*>(dst)[i] = reinterpret_cast<const int*>(src)[i];
}

// dir 0: sequence -> batch position blockIdx.x (ds, ss, sampling config with the position's processor struct and the greedy flag, processor struct, draft proposals, state pointers);
// dir 1: batch position -> sequence (everything a cycle changes: ds, ss, the processor counters, the draft proposals).
__global__ void k_gather(const SlotDesc* __restrict__ d, int* ds_b, int* ss_b, SampP* sp_b, ProcP* pb_b, Dist* qd_b, SeqPtrs* ptr_b, int dir) {
  const int bi = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  const SlotDesc D = d[bi];
  if (dir == 0) {
    for (int i = tid; i < DS_N; i += nt) ds_b[bi * DS_N + i] = D.ds[i];
    for (int i = tid; i < SS_N; i += nt) ss_b[bi * SS_N + i] = D.ss[i];
    copy_words(pb_b + bi, D.pb, tid, nt);
    for (int j = 0; j < PMAX; j++) copy_words(qd_b + bi * PMAX + j, D.qd + j, tid, nt);
    copy_words(sp_b + bi, D.sp, tid, nt);
    __syncthreads();
    if (tid == 0) {
      sp_b[bi].pr = pb_b + bi;
      sp_b[bi].greedy = D.greedy;
      ptr_b[bi] = D.ptrs;
    }
  } else {
    for (int i = tid; i < DS_N; i += nt) D.ds[i] = ds_b[bi * DS_N + i];
    for (int i = tid; i < SS_N; i += nt) D.ss[i] = ss_b[bi * SS_N + i];
    copy_words(D.pb, pb_b + bi, tid, nt);
    for (int j = 0; j < PMAX; j++) copy_words(D.qd + j, qd_b + bi * PMAX + j, tid, nt);
  }
}

// argmax of the R = B x T verify rows -> ds[DS_OUT + t] of each sequence
__global__ void k_out_b(const int* __restrict__ bout, int* ds_b, int T) {
  const int r = threadIdx.x;
  ds_b[(r / T) * DS_N + DS_OUT + r % T] = bout[r];
}

// greedy accept of every greedy sequence (thread bi): leading drafts equal to the previous row's argmax, then the common bookkeeping
__global__ void k_accept_b(int* ds_b, int* ss_b, int* ring_b, const SampP* sp_b, int k) {
  const int bi = threadIdx.x;
  if (!sp_b[bi].greedy) return;
  int* ds = ds_b + bi * DS_N;
  int m = 0;
  while (m < k && ds[DS_TOK + m + 1] == ds[DS_OUT + m]) m++;
  spec_finish(ds, ss_b + bi * SS_N, ring_b + (size_t)bi * RING, m, k + 1);
}

// greedy draft j of every greedy sequence: head id of the restricted-vocabulary argmax
__global__ void k_map_b(const int* __restrict__ idx, const int* __restrict__ ids, int* ds_b, const SampP* sp_b, int j) {
  const int bi = threadIdx.x;
  if (sp_b[bi].greedy) ds_b[bi * DS_N + DS_TOK + j] = ids[idx[bi]];
}

// MTP catch-up input: row r = (sequence r / T, row r % T): [ rmsnorm0(embed[ss[MTOK + t]]) | rmsnorm0(hid[r]) ]   grid (R, 2)
__global__ void __launch_bounds__(NH8) k_prep_cu(const uint4* __restrict__ embed, const int* __restrict__ ss_b, int T, const uint4* __restrict__ hid, const uint4* __restrict__ wE,
                                                 const uint4* __restrict__ wH, uint4* __restrict__ cat) {
  __shared__ float wsum[32];
  const int r = blockIdx.x, half = blockIdx.y;
  const int tok = ss_b[(r / T) * SS_N + SS_MTOK + r % T];
  mtp_norm_row(half == 0 ? embed + (size_t)tok * NH8 : hid + (size_t)r * NH8, half == 0 ? wE : wH, cat + (size_t)r * 2 * NH8 + half * NH8, wsum);
}

// MTP draft-chain input of draft j + 1: row bi = [ rmsnorm0(embed[ds[TOK + j]]) | rmsnorm0(hid[bi]) ]   grid (B, 2)
__global__ void __launch_bounds__(NH8) k_prep_ch(const uint4* __restrict__ embed, const int* __restrict__ ds_b, int j, const uint4* __restrict__ hid, const uint4* __restrict__ wE,
                                                 const uint4* __restrict__ wH, uint4* __restrict__ cat) {
  __shared__ float wsum[32];
  const int bi = blockIdx.x, half = blockIdx.y;
  const int tok = ds_b[bi * DS_N + DS_TOK + j];
  mtp_norm_row(half == 0 ? embed + (size_t)tok * NH8 : hid + (size_t)bi * NH8, half == 0 ? wE : wH, cat + (size_t)bi * 2 * NH8 + half * NH8, wsum);
}

// MTP output: sequence bi's row sel = bi * T + (T > 1 ? ss[AM1] : 0) -> out[bi]   grid B
__global__ void __launch_bounds__(NH8) k_final_b(const uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w, const int* __restrict__ ss_b, int T,
                                                 uint4* __restrict__ out) {
  __shared__ float wsum[32];
  const int bi = blockIdx.x;
  const int row = bi * T + (T > 1 ? ss_b[bi * SS_N + SS_AM1] : 0);
  mtp_final_row(h + (size_t)row * NH8, y + (size_t)row * NH8, w, out + (size_t)bi * NH8, wsum);
}

#define CKB(x)                                                                                    \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return e_; } \
  } while (0)
}  // namespace

struct smk_batch {
  smk_spec* s = nullptr;   // owner: weights, activations (its engine), MTP weights, sampling workspaces
  int* ds_b = nullptr;     // [BMAX][DS_N]
  int* ss_b = nullptr;     // [BMAX][SS_N]
  int* ring_b = nullptr;   // [BMAX][RING]
  SampP* sp_b = nullptr;
  ProcP* pb_b = nullptr;
  Dist* qd_b = nullptr;    // [BMAX][PMAX]
  Dist* pd_b = nullptr;    // [BMAX][TM]
  SeqPtrs* ptr_b = nullptr;
  uint16_t* hr_b = nullptr;  // [KMAX][BMAX][H]: post-norm MTP hidden that drafts j + 1 (plane j)
  uint16_t* cat = nullptr;   // [TR][2H]
  uint16_t* fin = nullptr;   // [TR][H]: final-norm output of a cycle of more than TM rows (the engine's tap slots hold TM rows)
  float* lg_b = nullptr;     // [BMAX][head_nv]
  int* idx_b = nullptr;      // [BMAX]
  int* bout = nullptr;       // [TR]
  SlotDesc* desc_d = nullptr;
  SlotDesc* desc_h = nullptr;  // pinned
  int *pin_ss = nullptr, *pin_ring = nullptr;
  size_t bytes = 0;
  unsigned epoch = 0;
  cudaGraphExec_t g[BMAX + 1][KMAX + 1][KMAX + 1][2] = {};
};

namespace {
void drop_graphs(smk_batch* b) {
  for (auto& x : b->g) for (auto& y : x) for (auto& z : y) for (auto& g : z) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
}

// One cycle over the B bound positions: verify B x (kv + 1) rows, accept, MTP catch-up + kd drafts.  rs: some sequence samples (quantised head rescoring R = 64 + processors on all rows).
cudaError_t batch_cycle(smk_batch* b, int B, int kv, int kd, int rs, cudaStream_t st) {
  smk_spec* s = b->s;
  smk_engine* e = s->e;
  const int T = kv + 1, R = B * T;
  auto dbgp = [&](int slot) { return (void*)(e->dbg + (size_t)slot * TM * H); };
  // up to TM rows: the engine's debug taps and its final-norm tap slot, as a single-sequence step; more rows: no taps, the final norm into the batch's own buffer (same bits)
  const bool wide = R > TM;
  auto tap = [&](int slot) { return wide ? nullptr : dbgp(slot); };
  void* fin = wide ? (void*)b->fin : dbgp(NLAYER + 1);
  // ---- verify: the 64-layer step over all rows
  for (int bi = 0; bi < B; bi++) CKB(smk_embed(e->w.embed, b->ds_b + bi * DS_N + DS_TOK, e->h + (size_t)bi * T * H, T, H, st));
  for (int l = 0; l < NLAYER; l++) {
    const smk_layer_w& L = e->w.layers[l];
    CKB(smk_rmsnorm(e->h, l == 0 ? nullptr : e->dn, L.ln1, e->xn, tap(l), R, H, 1e-6f, st));
    if (L.kind == 0) {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = 10240; ep.end[1] = 16384; ep.end[2] = 16384; ep.out[0] = e->qkv; ep.out[1] = e->z;
      CKB(cudaEventRecord(e->ev_fork, st));
      CKB(cudaStreamWaitEvent(e->side, e->ev_fork, 0));
      CKB(smk_bf16_gemv_small(L.ab, e->xn, 96, 48, H, R, e->ba, e->bb, e->side));
      CKB(cudaEventRecord(e->ev_join, e->side));
      CKB(smk_w4r_gemv(L.qkv_w, L.qkv_s, e->xn, 16384, H, R, &ep, st));
      CKB(cudaStreamWaitEvent(st, e->ev_join, 0));
      const int di = e->dn_idx[l];
      for (int bi = 0; bi < B; bi++) {
        swiftmk_dn_ind ind = {};
        ind.state_pp = &b->ptr_b[bi].dn_state;
        ind.conv_pp = &b->ptr_b[bi].dn_conv;
        ind.dbuf_pp = &b->ptr_b[bi].dn_dbuf;
        ind.state_off = (size_t)di * e->sslots * SWIFTMK_DN_STATE_SLOT_FLOATS;
        ind.conv_off = (size_t)di * e->slots * SWIFTMK_DN_CONV_SLOT_ELEMS;
        ind.dbuf_off = (size_t)di * swiftmk_deltanet_dbuf_floats();
        const int* ds = b->ds_b + bi * DS_N;
        const size_t r0 = (size_t)bi * T;
        CKB(swiftmk_deltanet_step_lazy2(T, e->qkv + r0 * 10240, e->z + r0 * 6144, e->ba + r0 * 48, e->bb + r0 * 48, L.conv_w, L.A_log, L.dt_bias, L.norm_w, nullptr, nullptr,
                                        ds + DS_CUR, e->slots, e->sslots, e->y6 + r0 * 6144, e->ws_dn, nullptr, ds + DS_PEND, &ind, st));
      }
    } else {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = 12288; ep.end[1] = 13312; ep.end[2] = 14336; ep.out[0] = e->qb; ep.out[1] = e->kb; ep.out[2] = e->vb;
      CKB(smk_w4r_gemv(L.qkv_w, L.qkv_s, e->xn, 14336, H, R, &ep, st));
      const size_t koff = (size_t)e->at_idx[l] * e->kv_layer_bytes;
      for (int bi = 0; bi < B; bi++) {
        const size_t r0 = (size_t)bi * T;
        CKB(swiftmk_attn_step_ind(T, e->qb + r0 * 12288, e->kb + r0 * 1024, e->vb + r0 * 1024, L.q_norm, L.k_norm, nullptr, &b->ptr_b[bi].kv, koff, e->ctx_max, e->kv_dtype,
                                  L.k_scale, L.v_scale, b->ds_b + bi * DS_N + DS_POS, b->ds_b + bi * DS_N + DS_RDELTA, e->y6 + r0 * 6144, e->ws_attn, st));
      }
    }
    {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = ep.end[1] = ep.end[2] = H; ep.out[0] = e->mix;
      CKB(smk_w4r_gemv(L.o_w, L.o_s, e->y6, H, 6144, R, &ep, st));
    }
    CKB(smk_rmsnorm(e->h, e->mix, L.ln2, e->xn, nullptr, R, H, 1e-6f, st));
    {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SILU; ep.out[0] = e->act;
      CKB(smk_w4r_gemv(L.gu_w, L.gu_s, e->xn, 2 * I, H, R, &ep, st));
      smk_epi ed = {};
      ed.kind = SMK_EPI_SEGS; ed.end[0] = ed.end[1] = ed.end[2] = H; ed.out[0] = e->dn;
      CKB(smk_w4r_gemv(L.dn_w, L.dn_s, e->act, H, I, R, &ed, st));
    }
  }
  // ---- head: final norm, quantised lm_head + exact rescoring over all rows (the sequences' processors on their own rows, with their drafts as history)
  CKB(smk_rmsnorm(e->h, e->dn, e->w.final_norm, fin, tap(NLAYER), R, H, 1e-6f, st));
  CKB(smk_headq_gemv(e->w.head_q_kind, e->w.head_q, e->w.head_qs, fin, VOC, H, R, e->logits, st));
  if (rs)
    for (int bi = 0; bi < B; bi++) CKB(proc_logits(b->pb_b + bi, e->logits + (size_t)bi * T * VOC, b->ds_b + bi * DS_N + DS_TOK + 1, T, st));
  CKB(smk_headq_select_rescore(e->logits, e->w.lm_head, fin, VOC, H, R, rs ? HEAD_R_SAMPLING : HEAD_R_GREEDY, b->bout, nullptr, e->ws_hq, st));
  if (rs)
    for (int bi = 0; bi < B; bi++)
      CKB(proc_cand(b->pb_b + bi, smk_headq_ws_cand(e->ws_hq) + (size_t)bi * T * SMK_HEADQ_RMAX, (float*)smk_headq_ws_exact(e->ws_hq) + (size_t)bi * T * SMK_HEADQ_RMAX, SMK_HEADQ_RMAX,
                    e->logits + (size_t)bi * T * VOC, b->ds_b + bi * DS_N + DS_TOK + 1, T, st));
  k_out_b<<<1, R, 0, st>>>(b->bout, b->ds_b, T);
  CKB(cudaGetLastError());
  // ---- accept
  k_accept_b<<<1, B, 0, st>>>(b->ds_b, b->ss_b, b->ring_b, b->sp_b, kv);
  CKB(cudaGetLastError());
  if (rs)
    for (int bi = 0; bi < B; bi++) {
      CKB(sk_topk(e->logits + (size_t)bi * T * VOC, T, VOC, VOC, nullptr, b->sp_b + bi, 0, s->ws_t, b->pd_b + bi * TM, st));
      CKB(sk_accept(b->ds_b + bi * DS_N, b->ss_b + bi * SS_N, b->ring_b + (size_t)bi * RING, kv, b->pd_b + bi * TM, b->qd_b + bi * PMAX, b->sp_b + bi, st));
    }
  // ---- MTP: catch-up of every sequence's T rows (pairs (hidden of row t, token y[t])), then kd chained drafts
  const smk_mtp_w& W = s->w;
  const smk_layer_w& M = W.layer;
  k_prep_cu<<<dim3(R, 2), NH8, 0, st>>>((const uint4*)e->w.embed, b->ss_b, T, (const uint4*)fin, (const uint4*)W.pre_e, (const uint4*)W.pre_h, (uint4*)b->cat);
  CKB(cudaGetLastError());
  for (int j = 0; j <= kd; j++) {
    const int rows = j == 0 ? R : B, Tq = j == 0 ? T : 1;
    if (j > 0) {  // draft j from the hidden of plane j - 1
      const void* x = b->hr_b + (size_t)(j - 1) * BMAX * H;
      smk_epi ep = {};
      ep.kind = SMK_EPI_F32; ep.out[0] = b->lg_b;
      CKB(smk_w4r_gemv(W.head_w, W.head_s, x, W.head_nv, H, B, &ep, st));
      CKB(smk_argmax(b->lg_b, B, W.head_nv, b->idx_b, nullptr, e->ws_am, st));
      k_map_b<<<1, B, 0, st>>>(b->idx_b, W.head_ids, b->ds_b, b->sp_b, j);
      CKB(cudaGetLastError());
      if (rs)
        for (int bi = 0; bi < B; bi++) {
          CKB(sk_topk(b->lg_b + (size_t)bi * W.head_nv, 1, W.head_nv, W.head_nv, W.head_ids, b->sp_b + bi, 1, s->ws_d, b->qd_b + bi * PMAX + (j - 1), st));
          CKB(sk_draft_pick(b->qd_b + bi * PMAX + (j - 1), b->sp_b + bi, b->ss_b + bi * SS_N, j, b->ds_b + bi * DS_N + DS_TOK + j, st));
        }
      if (j == kd) break;
      k_prep_ch<<<dim3(B, 2), NH8, 0, st>>>((const uint4*)e->w.embed, b->ds_b, j, (const uint4*)x, (const uint4*)W.pre_e, (const uint4*)W.pre_h, (uint4*)b->cat);
      CKB(cudaGetLastError());
    }
    smk_epi ef = {};
    ef.kind = SMK_EPI_SEGS; ef.end[0] = ef.end[1] = ef.end[2] = H; ef.out[0] = e->h;
    CKB(smk_w4r_gemv(W.fc_w, W.fc_s, b->cat, H, 2 * H, rows, &ef, st));
    CKB(smk_rmsnorm(e->h, nullptr, M.ln1, e->xn, nullptr, rows, H, 1e-6f, st));
    smk_epi eq = {};
    eq.kind = SMK_EPI_SEGS; eq.end[0] = 12288; eq.end[1] = 13312; eq.end[2] = 14336; eq.out[0] = e->qb; eq.out[1] = e->kb; eq.out[2] = e->vb;
    CKB(smk_w4r_gemv(M.qkv_w, M.qkv_s, e->xn, 14336, H, rows, &eq, st));
    for (int bi = 0; bi < B; bi++) {
      const size_t r0 = (size_t)bi * Tq;
      CKB(swiftmk_attn_step_ind(Tq, e->qb + r0 * 12288, e->kb + r0 * 1024, e->vb + r0 * 1024, M.q_norm, M.k_norm, nullptr, &b->ptr_b[bi].mtp_kv, 0, e->ctx_max, 0, 1.f, 1.f,
                                b->ss_b + bi * SS_N + SS_DPOS + j, b->ds_b + bi * DS_N + DS_RDELTA, e->y6 + r0 * 6144, e->ws_attn, st));
    }
    if (kd == 0) break;  // catch-up only: the MTP KV of the accepted rows is what later drafting needs
    smk_epi eo = {};
    eo.kind = SMK_EPI_SEGS; eo.end[0] = eo.end[1] = eo.end[2] = H; eo.out[0] = e->mix;
    CKB(smk_w4r_gemv(M.o_w, M.o_s, e->y6, H, 6144, rows, &eo, st));
    CKB(smk_rmsnorm(e->h, e->mix, M.ln2, e->xn, nullptr, rows, H, 1e-6f, st));
    smk_epi eg = {};
    eg.kind = SMK_EPI_SILU; eg.out[0] = e->act;
    CKB(smk_w4r_gemv(M.gu_w, M.gu_s, e->xn, 2 * I, H, rows, &eg, st));
    smk_epi ed = {};
    ed.kind = SMK_EPI_SEGS; ed.end[0] = ed.end[1] = ed.end[2] = H; ed.out[0] = e->dn;
    CKB(smk_w4r_gemv(M.dn_w, M.dn_s, e->act, H, I, rows, &ed, st));
    k_final_b<<<B, NH8, 0, st>>>((const uint4*)e->h, (const uint4*)e->dn, (const uint4*)W.norm, b->ss_b, Tq, (uint4*)(b->hr_b + (size_t)j * BMAX * H));
    CKB(cudaGetLastError());
  }
  return cudaSuccess;
}

cudaError_t launch(smk_batch* b, int B, int kv, int kd, int rs, cudaStream_t st, bool capture_only) {
  smk_engine* e = b->s->e;
  if (b->epoch != e->head_epoch) {
    drop_graphs(b);
    b->epoch = e->head_epoch;
  }
  cudaGraphExec_t* gx = &b->g[B][kv][kd][rs];
  auto f = [&](cudaStream_t c) { return batch_cycle(b, B, kv, kd, rs, c); };
  return capture_only ? smk_graph_capture(e, gx, f) : smk_graph_launch(e, gx, f, st);
}

bool shape_ok(smk_batch* b, int B, int kv, int kd) {
  return B >= 1 && B <= BMAX && kv >= 0 && kv <= KMAX && kd >= 0 && kd <= KMAX && B * (kv + 1) <= TR;
}
}  // namespace

extern "C" {

smk_batch* smk_batch_create(smk_spec* owner) {
  smk_engine* e = owner->e;
  if (!e->hq_on || e->share || owner->head_mode != 0) { fprintf(stderr, "smk_batch: the owner needs the quantised head, its own activations and the restricted draft head\n"); return nullptr; }
  smk_batch* b = new smk_batch();
  b->s = owner;
  const size_t b0 = e->bytes;
  b->ds_b = (int*)dalloc(e, (size_t)BMAX * DS_N * 4, true);
  b->ss_b = (int*)dalloc(e, (size_t)BMAX * SS_N * 4, true);
  b->ring_b = (int*)dalloc(e, (size_t)BMAX * RING * 4, true);
  b->sp_b = (SampP*)dalloc(e, sizeof(SampP) * BMAX, true);
  b->pb_b = (ProcP*)dalloc(e, sizeof(ProcP) * BMAX, true);
  b->qd_b = (Dist*)dalloc(e, sizeof(Dist) * BMAX * PMAX, true);
  b->pd_b = (Dist*)dalloc(e, sizeof(Dist) * BMAX * TM, true);
  b->ptr_b = (SeqPtrs*)dalloc(e, sizeof(SeqPtrs) * BMAX, true);
  b->hr_b = (uint16_t*)dalloc(e, (size_t)KMAX * BMAX * H * 2, true);
  b->cat = (uint16_t*)dalloc(e, (size_t)TR * 2 * H * 2, true);
  b->fin = (uint16_t*)dalloc(e, (size_t)TR * H * 2, true);
  b->lg_b = (float*)dalloc(e, (size_t)BMAX * owner->w.head_nv * 4, true);
  b->idx_b = (int*)dalloc(e, BMAX * 4, true);
  b->bout = (int*)dalloc(e, TR * 4, true);
  b->desc_d = (SlotDesc*)dalloc(e, sizeof(SlotDesc) * BMAX, true);
  b->bytes = e->bytes - b0;
  cudaMallocHost((void**)&b->desc_h, sizeof(SlotDesc) * BMAX);
  cudaMallocHost((void**)&b->pin_ss, (size_t)BMAX * SS_N * 4);
  cudaMallocHost((void**)&b->pin_ring, (size_t)BMAX * RING * 4);
  b->epoch = e->head_epoch;
  if (e->failed || !b->desc_h || !b->pin_ss || !b->pin_ring || cudaGetLastError() != cudaSuccess) { smk_batch_destroy(b); return nullptr; }
  return b;
}

void smk_batch_destroy(smk_batch* b) {
  if (!b) return;
  cudaDeviceSynchronize();
  drop_graphs(b);
  if (b->desc_h) cudaFreeHost(b->desc_h);
  if (b->pin_ss) cudaFreeHost(b->pin_ss);
  if (b->pin_ring) cudaFreeHost(b->pin_ring);
  delete b;  // device buffers belong to the owner engine's allocation list
}

size_t smk_batch_bytes(smk_batch* b) { return b->bytes; }

cudaError_t smk_batch_prewarm(smk_batch* b, int B, int kv, int kd, int samp) {
  if (!shape_ok(b, B, kv, kd)) return cudaErrorInvalidValue;
  return launch(b, B, kv, kd, samp ? 1 : 0, nullptr, true);
}

cudaError_t smk_batch_cycles(smk_batch* b, smk_spec* const* seqs, int B, int kv0, int kd, int n, cudaStream_t st, int* host_ss, int* host_ring) {
  if (n < 1 || !shape_ok(b, B, kv0, kd) || !shape_ok(b, B, kd, kd)) return cudaErrorInvalidValue;
  smk_engine* oe = b->s->e;
  int rs = 0;
  for (int i = 0; i < B; i++) {
    smk_spec* q = seqs[i];
    smk_engine* qe = q->e;
    if ((qe != oe && qe->share != oe) || qe->sslots != oe->sslots || q->head_mode != 0) return cudaErrorInvalidValue;
    for (int j = 0; j < i; j++) if (seqs[j] == q) return cudaErrorInvalidValue;
    SlotDesc& d = b->desc_h[i];
    d.ds = qe->ds; d.ss = q->ss; d.sp = q->sp; d.pb = qe->pb.dev; d.qd = q->qd;
    d.ptrs = {qe->dn_state, qe->dn_conv, qe->dn_dbuf, qe->kv, q->kvbuf};
    d.greedy = q->samp ? 0 : 1;
    rs |= q->samp;
  }
  CKB(cudaMemcpyAsync(b->desc_d, b->desc_h, sizeof(SlotDesc) * B, cudaMemcpyHostToDevice, st));
  k_gather<<<B, 256, 0, st>>>(b->desc_d, b->ds_b, b->ss_b, b->sp_b, b->pb_b, b->qd_b, b->ptr_b, 0);
  CKB(cudaGetLastError());
  for (int i = 0; i < n; i++) CKB(launch(b, B, i == 0 ? kv0 : kd, kd, rs, st, false));
  k_gather<<<B, 256, 0, st>>>(b->desc_d, b->ds_b, b->ss_b, b->sp_b, b->pb_b, b->qd_b, b->ptr_b, 1);
  CKB(cudaGetLastError());
  CKB(cudaMemcpyAsync(b->pin_ss, b->ss_b, (size_t)B * SS_N * 4, cudaMemcpyDeviceToHost, st));
  CKB(cudaMemcpyAsync(b->pin_ring, b->ring_b, (size_t)B * RING * 4, cudaMemcpyDeviceToHost, st));
  CKB(cudaStreamSynchronize(st));
  memcpy(host_ss, b->pin_ss, (size_t)B * SS_N * 4);
  memcpy(host_ring, b->pin_ring, (size_t)B * RING * 4);
  return cudaSuccess;
}

}  // extern "C"
