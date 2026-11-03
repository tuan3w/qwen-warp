// swift-mk K3: the 64-layer decode step wired from the K1 (DeltaNet) / K2 (attention) / K3 (GEMV, norm, lm_head) kernels + CUDA-graph capture.
#include "engine_impl.h"

using namespace smk;

// ---------------------------------------------------------------------------------------------- tiny kernels
__global__ void k_advance(int* ds, int acc, int feedback) {
  ds[DS_POS] += acc;
  dn_pend_after(ds, acc, acc);
  ds[DS_CUR] = (ds[DS_CUR] + acc) % ds[DS_SLOTS];
  if (feedback) ds[DS_TOK] = ds[DS_OUT + acc - 1];
}

// ---------------------------------------------------------------------------------------------- the step
#define CK(x)                                                                                         \
  do {                                                                                                \
    cudaError_t e_ = (x);                                                                             \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d (layer %d) %s\n", __FILE__, __LINE__, l, cudaGetErrorString(e_)); return e_; } \
  } while (0)

// Final norm of the residual stream (e->h + e->dn -> debug taps NLAYER / NLAYER+1) and, for head == 1, the lm_head + argmax: BF16 head (plain) or the K9 quantised head
// + exact BF16 rescoring of the top-R candidates (R = HEAD_R_SAMPLING if rsamp else HEAD_R_GREEDY; the argmax is written to ds[DS_OUT..]).  head == 2: final norm only.
static cudaError_t head_tail(smk_engine* e, int T, int head, int rsamp, cudaStream_t st) {
  int l = NLAYER;
  auto dbgp = [&](int slot) { return (void*)(e->dbg + (size_t)slot * TM * H); };
  CK(smk_rmsnorm(e->h, e->dn, e->w.final_norm, dbgp(NLAYER + 1), dbgp(NLAYER), T, H, 1e-6f, st));
  if (head == 1 && !e->hq_on) {
    CK(smk_bf16_gemv_f32(e->w.lm_head, dbgp(NLAYER + 1), VOC, H, T, e->logits, st));
    CK(smk_argmax(e->logits, T, VOC, (int*)(e->ds + DS_OUT), nullptr, e->ws_am, st));
  } else if (head == 1) {
    CK(smk_headq_gemv(e->w.head_q_kind, e->w.head_q, e->w.head_qs, dbgp(NLAYER + 1), VOC, H, T, e->logits, st));
    CK(smk_headq_select_rescore(e->logits, e->w.lm_head, dbgp(NLAYER + 1), VOC, H, T, rsamp ? HEAD_R_SAMPLING : HEAD_R_GREEDY, (int*)(e->ds + DS_OUT), nullptr, e->ws_hq, st));
  }
  return cudaSuccess;
}

cudaError_t smk_run_step(smk_engine* e, int T, int head, int spec, cudaStream_t st, int rsamp) {
  int l = -1;
  if (T < 1 || T > e->tmax) return cudaErrorInvalidValue;
  const int* ds = e->ds;
  auto dbgp = [&](int slot) { return (void*)(e->dbg + (size_t)slot * TM * H); };
  CK(smk_embed(e->w.embed, ds + DS_TOK, e->h, T, H, st));
  for (l = 0; l < NLAYER; l++) {
    const smk_layer_w& L = e->w.layers[l];
    CK(smk_rmsnorm(e->h, l == 0 ? nullptr : e->dn, L.ln1, e->xn, dbgp(l), T, H, 1e-6f, st));
    if (L.kind == 0) {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = 10240; ep.end[1] = 16384; ep.end[2] = 16384; ep.out[0] = e->qkv; ep.out[1] = e->z;
      CK(cudaEventRecord(e->ev_fork, st));
      CK(cudaStreamWaitEvent(e->side, e->ev_fork, 0));
      CK(smk_bf16_gemv_small(L.ab, e->xn, 96, 48, H, T, e->ba, e->bb, e->side));  // in_proj_a|b: parallel branch, overlaps the qkv|z GEMV
      CK(cudaEventRecord(e->ev_join, e->side));
      CK(smk_w4r_gemv(L.qkv_w, L.qkv_s, e->xn, 16384, H, T, &ep, st));
      CK(cudaStreamWaitEvent(st, e->ev_join, 0));
      const int di = e->dn_idx[l];
      CK(swiftmk_deltanet_step_lazy(T, e->qkv, e->z, e->ba, e->bb, L.conv_w, L.A_log, L.dt_bias, L.norm_w,
                                    e->dn_state + (size_t)di * e->slots * SWIFTMK_DN_STATE_SLOT_FLOATS,
                                    e->dn_conv + (size_t)di * e->slots * SWIFTMK_DN_CONV_SLOT_ELEMS, ds + DS_CUR, e->slots, e->y6, e->ws_dn,
                                    e->dn_dbuf + (size_t)di * swiftmk_deltanet_dbuf_floats(), ds + DS_PEND, st));
    } else {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = 12288; ep.end[1] = 13312; ep.end[2] = 14336; ep.out[0] = e->qb; ep.out[1] = e->kb; ep.out[2] = e->vb;
      CK(smk_w4r_gemv(L.qkv_w, L.qkv_s, e->xn, 14336, H, T, &ep, st));
      CK(swiftmk_attn_step(T, e->qb, e->kb, e->vb, L.q_norm, L.k_norm, e->kv + (size_t)e->at_idx[l] * e->kv_layer_bytes, e->ctx_max, e->kv_dtype,
                           L.k_scale, L.v_scale, ds + DS_POS, e->y6, e->ws_attn, st));
    }
    {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SEGS; ep.end[0] = ep.end[1] = ep.end[2] = H; ep.out[0] = e->mix;
      CK(smk_w4r_gemv(L.o_w, L.o_s, e->y6, H, 6144, T, &ep, st));
    }
    CK(smk_rmsnorm(e->h, e->mix, L.ln2, e->xn, nullptr, T, H, 1e-6f, st));
    {
      smk_epi ep = {};
      ep.kind = SMK_EPI_SILU; ep.out[0] = e->act;
      CK(smk_w4r_gemv(L.gu_w, L.gu_s, e->xn, 2 * I, H, T, &ep, st));
      smk_epi ed = {};
      ed.kind = SMK_EPI_SEGS; ed.end[0] = ed.end[1] = ed.end[2] = H; ed.out[0] = e->dn;
      CK(smk_w4r_gemv(L.dn_w, L.dn_s, e->act, H, I, T, &ed, st));
    }
  }
  l = NLAYER;
  if (head) CK(head_tail(e, T, head, rsamp, st));
  if (!spec) k_advance<<<1, 1, 0, st>>>(e->ds, T, head == 1 && T == 1);
  CK(cudaGetLastError());
  return cudaSuccess;
}

static void smk_engine_drop_graphs(smk_engine* e) {
  for (auto& r : e->gexec) for (auto& v : r) for (auto& g : v) if (g) { cudaGraphExecDestroy(g); g = nullptr; }
}

extern "C" {

smk_engine* smk_engine_create(const smk_model_w* w, int ctx_max, int kv_fp8) {
  const char* env = getenv("SMK_TMAX");
  return smk_engine_create_ex(w, ctx_max, kv_fp8, env ? atoi(env) : 4);
}

int smk_engine_tmax(smk_engine* e) { return e->tmax; }

smk_engine* smk_engine_create_ex(const smk_model_w* w, int ctx_max, int kv_fp8, int tmax) {
  if (tmax < 4 || tmax > TM) { fprintf(stderr, "smk_engine: tmax %d outside 4..%d\n", tmax, TM); return nullptr; }
  smk_engine* e = new smk_engine();
  e->tmax = tmax;
  e->slots = tmax + 1;
  e->w = *w;
  e->ctx_max = ctx_max;
  e->kv_dtype = kv_fp8;  // 0=bf16 1=fp8 2=int4
  int nd = 0, na = 0;
  for (int l = 0; l < NLAYER; l++) { e->dn_idx[l] = w->layers[l].kind == 0 ? nd++ : -1; e->at_idx[l] = w->layers[l].kind == 1 ? na++ : -1; }
  auto B = [&](size_t n) { return dalloc(e, n); };
  e->ds = (int*)dalloc(e, DS_N * sizeof(int), true);
  { const int sl = e->slots; cudaMemcpy(e->ds + DS_SLOTS, &sl, sizeof(int), cudaMemcpyHostToDevice); }
  e->h = (uint16_t*)B((size_t)TM * H * 2);
  e->xn = (uint16_t*)B((size_t)TM * H * 2);
  e->qkv = (uint16_t*)B((size_t)TM * 10240 * 2);
  e->z = (uint16_t*)B((size_t)TM * 6144 * 2);
  e->qb = (uint16_t*)B((size_t)TM * 12288 * 2);
  e->kb = (uint16_t*)B((size_t)TM * 1024 * 2);
  e->vb = (uint16_t*)B((size_t)TM * 1024 * 2);
  e->ba = (uint16_t*)B((size_t)TM * 48 * 2);
  e->bb = (uint16_t*)B((size_t)TM * 48 * 2);
  e->y6 = (uint16_t*)B((size_t)TM * 6144 * 2);
  e->mix = (uint16_t*)B((size_t)TM * H * 2);
  e->act = (uint16_t*)B((size_t)TM * I * 2);
  e->dn = (uint16_t*)B((size_t)TM * H * 2);
  e->dbg = (uint16_t*)dalloc(e, (size_t)DBG_SLOTS * TM * H * 2, true);
  e->logits = (float*)B((size_t)TM * VOC * 4);
  e->ws_dn = dalloc(e, swiftmk_deltanet_ws_bytes(), true);
  e->ws_attn = dalloc(e, swiftmk_attn_ws_bytes(), true);
  e->ws_am = dalloc(e, smk_argmax_ws_bytes(), true);
  e->ws_hq = dalloc(e, smk_headq_ws_bytes(), true);
  e->hq_on = w->head_q_kind != SMK_HEADQ_NONE;
  e->dn_state = (float*)dalloc(e, (size_t)nd * e->slots * SWIFTMK_DN_STATE_SLOT_FLOATS * 4, true);
  e->dn_conv = (uint16_t*)dalloc(e, (size_t)nd * e->slots * SWIFTMK_DN_CONV_SLOT_ELEMS * 2, true);
  e->dn_dbuf = (float*)dalloc(e, (size_t)nd * swiftmk_deltanet_dbuf_floats() * 4, true);
  e->kv_layer_bytes = swiftmk_attn_kv_bytes(ctx_max, kv_fp8);
  e->kv = (char*)dalloc(e, (size_t)na * e->kv_layer_bytes, true);
  cudaStreamCreateWithFlags(&e->cap, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&e->side, cudaStreamNonBlocking);
  cudaEventCreateWithFlags(&e->ev_fork, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&e->ev_join, cudaEventDisableTiming);
  cudaMallocHost((void**)&e->pinned, 65536 * sizeof(int));
  if (e->failed || cudaGetLastError() != cudaSuccess) { smk_engine_destroy(e); return nullptr; }
  return e;
}

void smk_engine_destroy(smk_engine* e) {
  if (!e) return;
  cudaDeviceSynchronize();
  smk_engine_drop_graphs(e);
  for (void* p : e->allocs) if (p) cudaFree(p);
  if (e->cap) cudaStreamDestroy(e->cap);
  if (e->side) cudaStreamDestroy(e->side);
  if (e->ev_fork) cudaEventDestroy(e->ev_fork);
  if (e->ev_join) cudaEventDestroy(e->ev_join);
  if (e->pinned) cudaFreeHost(e->pinned);
  delete e;
}

cudaError_t smk_engine_reset(smk_engine* e, cudaStream_t st) {
  const int nd = 48, na = 16;
  cudaMemsetAsync(e->ds, 0, DS_SLOTS * sizeof(int), st);  // ds[DS_SLOTS] (ring size) is constant
  cudaMemsetAsync(e->ds + DS_PEND, 0, sizeof(int), st);
  cudaMemsetAsync(e->dn_state, 0, (size_t)nd * e->slots * SWIFTMK_DN_STATE_SLOT_FLOATS * 4, st);
  cudaMemsetAsync(e->dn_conv, 0, (size_t)nd * e->slots * SWIFTMK_DN_CONV_SLOT_ELEMS * 2, st);
  cudaMemsetAsync(e->kv, 0, (size_t)na * e->kv_layer_bytes, st);
  return cudaGetLastError();
}

cudaError_t smk_engine_set_tokens(smk_engine* e, const int* tok, int T, cudaStream_t st) {
  if (T < 1 || T > e->tmax) return cudaErrorInvalidValue;
  return cudaMemcpyAsync(e->ds + DS_TOK, tok, T * sizeof(int), cudaMemcpyHostToDevice, st);
}

cudaError_t smk_engine_step_nograph(smk_engine* e, int T, int head, cudaStream_t st) { return smk_run_step(e, T, head, 0, st); }

cudaError_t smk_engine_set_head(smk_engine* e, int quant) {
  if (quant && e->w.head_q_kind == SMK_HEADQ_NONE) return cudaErrorInvalidValue;
  cudaDeviceSynchronize();
  e->hq_on = quant ? 1 : 0;
  smk_engine_drop_graphs(e);
  e->head_epoch++;
  return cudaSuccess;
}
int smk_engine_head(smk_engine* e) { return e->hq_on; }
cudaError_t smk_engine_head_tail(smk_engine* e, int rsamp, cudaStream_t st) { return head_tail(e, 1, 1, rsamp, st); }

cudaError_t smk_engine_step(smk_engine* e, int T, int head, cudaStream_t st) {
  if (T < 1 || T > e->tmax) return cudaErrorInvalidValue;
  if (head < 0 || head >= NHEAD_MODES) return cudaErrorInvalidValue;
  const int var = (head == 1 && e->hq_on) ? 1 + e->hsamp : 0;
  cudaGraphExec_t& gx = e->gexec[T][head][var];
  if (!gx) {
    cudaGraph_t g = nullptr;
    cudaError_t err = cudaStreamBeginCapture(e->cap, cudaStreamCaptureModeThreadLocal);
    if (err != cudaSuccess) return err;
    cudaError_t err2 = smk_run_step(e, T, head, 0, e->cap, e->hsamp);
    err = cudaStreamEndCapture(e->cap, &g);
    if (err2 != cudaSuccess) return err2;
    if (err != cudaSuccess) return err;
    err = cudaGraphInstantiate(&gx, g, 0);
    cudaGraphDestroy(g);
    if (err != cudaSuccess) return err;
  }
  return cudaGraphLaunch(gx, st);
}

cudaError_t smk_engine_decode(smk_engine* e, int n, int* host_tokens, cudaStream_t st) {
  for (int base = 0; base < n; base += 65536) {
    const int m = n - base < 65536 ? n - base : 65536;
    for (int i = 0; i < m; i++) {
      cudaError_t err = smk_engine_step(e, 1, 1, st);
      if (err != cudaSuccess) return err;
      err = cudaMemcpyAsync(e->pinned + i, e->ds + DS_OUT, sizeof(int), cudaMemcpyDeviceToHost, st);
      if (err != cudaSuccess) return err;
    }
    cudaError_t err = cudaStreamSynchronize(st);
    if (err != cudaSuccess) return err;
    memcpy(host_tokens + base, e->pinned, m * sizeof(int));
  }
  return cudaSuccess;
}

cudaError_t smk_engine_read(smk_engine* e, int what, void* host, size_t bytes) {
  const void* src = what == 0 ? (const void*)e->ds : what == 1 ? (const void*)e->dbg : what == 2 ? (const void*)e->logits : nullptr;
  const size_t cap = what == 0 ? DS_N * sizeof(int) : what == 1 ? (size_t)DBG_SLOTS * TM * H * 2 : (size_t)TM * VOC * 4;
  if (!src || bytes > cap) return cudaErrorInvalidValue;
  return cudaMemcpy(host, src, bytes, cudaMemcpyDeviceToHost);
}
size_t smk_engine_bytes(smk_engine* e) { return e->bytes; }

}  // extern "C"
