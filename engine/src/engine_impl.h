// swift-mk: private engine state shared by engine.cu (plain decode) and mtp.cu (speculative decode).  Not a public header.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "attn.h"
#include "common.cuh"
#include "deltanet.h"
#include "engine.h"
#include "head_q.h"
#include "proc.h"

namespace smk {
constexpr int H = SMK_HIDDEN, I = SMK_INTER, VOC = SMK_VOCAB, NLAYER = SMK_NLAYERS, TM = SMK_TMAX;
constexpr int DS_POS = 0, DS_CUR = 1, DS_TOK = 2, DS_OUT = 10, DS_SLOTS = 18, DS_PEND = 19, DS_N = 20;  // DS_SLOTS: DeltaNet ring slots (tmax + 1); DS_PEND: pending lazy DeltaNet rows | delta-buffer parity << 8 (deltanet.h)
constexpr int DBG_SLOTS = NLAYER + 2;  // 0 embed, 1..64 layer outputs, 65 final norm out
constexpr int NHEAD_MODES = 3;         // head 0: none, 1: final norm + lm_head + argmax, 2: final norm only (MTP prefill chunks)
constexpr int NHEAD_VARIANTS = 3;      // head 1 graphs: 0 BF16 head, 1 quantised head + rescore R=16 (greedy), 2 quantised head + rescore R=64 (sampling: top_k <= 64)
constexpr int HEAD_R_GREEDY = 16, HEAD_R_SAMPLING = 64;

// DeltaNet lazy snapshots (deltanet.h): call BEFORE ds[DS_CUR] advances by a (a rows accepted of a step of T rows).  Mirrors the flush decision of the kernel:
// a T >= 2 step that finds pending rows and no room for its T rows writes the pending state to slot cur (the new base) and restarts the delta buffer.
__device__ __forceinline__ void dn_pend_after(int* ds, int a, int T) {
  if (T < 2) { ds[DS_PEND] = 0; return; }
  const int pv = ds[DS_PEND];
  int pc = pv & 255, base = (pv >> 8) & 15, half = (pv >> 12) & 1;
  if (pc == 0) base = ds[DS_CUR];
  else if (pc + T > SWIFTMK_DN_LAZY_ROWS) { pc = 0; base = ds[DS_CUR]; half ^= 1; }
  ds[DS_PEND] = (pc + a) | (base << 8) | (half << 12);
}
}  // namespace smk

struct smk_engine {
  smk_model_w w;
  int ctx_max = 0, kv_dtype = 0;  // 0=bf16 1=fp8 2=int4
  int tmax = 4, slots = 5;  // K11: largest step (rows) and DeltaNet ring slots = tmax + 1
  size_t bytes = 0;
  bool failed = false;
  std::vector<void*> allocs;
  int* ds = nullptr;
  uint16_t *h, *xn, *qkv, *z, *qb, *kb, *vb, *ba, *bb, *y6, *mix, *act, *dn, *dbg;
  float* logits;
  void *ws_dn, *ws_attn, *ws_am;
  float* dn_state;
  float* dn_dbuf;  // [48][swiftmk_deltanet_dbuf_floats()] rank-1 updates of the last two lazy steps
  uint16_t* dn_conv;
  char* kv;
  size_t kv_layer_bytes;
  cudaStream_t cap = nullptr, side = nullptr;
  cudaEvent_t ev_fork = nullptr, ev_join = nullptr;
  cudaGraphExec_t gexec[smk::TM + 1][smk::NHEAD_MODES][smk::NHEAD_VARIANTS] = {};
  cudaGraphExec_t gverify[smk::TM + 1][2] = {};  // smk_engine_verify graphs [T][BF16 head | quantised head] (greedy, spec = 1)
  // K9 quantised head: hq_on = use it (else the BF16 head); hsamp = sampling configured (rescoring set R = 64 so the device top-k <= 64 sees exact logits)
  int hq_on = 0, hsamp = 0;
  unsigned head_epoch = 0;  // bumped by smk_engine_set_head: graphs of the speculative driver are recaptured
  void* ws_hq = nullptr;
  smk::ProcBufs pb = {};  // request-level logit processors (proc.h)
  int* pinned = nullptr;
  int dn_idx[smk::NLAYER], at_idx[smk::NLAYER];
};

static inline void* dalloc(smk_engine* e, size_t n, bool zero = false) {
  void* p = nullptr;
  if (cudaMalloc(&p, n) != cudaSuccess) { fprintf(stderr, "smk_engine: cudaMalloc(%zu) failed\n", n); e->failed = true; return nullptr; }
  if (zero) cudaMemset(p, 0, n);
  e->allocs.push_back(p);
  e->bytes += n;
  return p;
}

// The 64-layer step (engine.cu).  head: see NHEAD_MODES.  spec != 0: do not advance pos/cur (the speculative accept kernel does).  rsamp: rescoring set of the
// quantised head (HEAD_R_SAMPLING if the caller samples from the logits, else HEAD_R_GREEDY; ignored by the BF16 head).
cudaError_t smk_run_step(smk_engine* e, int T, int head, int spec, cudaStream_t st, int rsamp = 0);

// Capture `f(cudaStream_t cap_stream)` (returns cudaError_t) into a CUDA graph the first time (*gx == nullptr), then launch it on `st`.
// smk_graph_capture: capture only (K11: the prompt-lookup widths are captured up front, not by the first request that needs them, when the allocator may have no memory left for the graph).
template <class F>
static inline cudaError_t smk_graph_capture(smk_engine* e, cudaGraphExec_t* gx, F f) {
  if (*gx) return cudaSuccess;
  cudaGraph_t g = nullptr;
  cudaError_t err = cudaStreamBeginCapture(e->cap, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess) return err;
  cudaError_t err2 = f(e->cap);
  err = cudaStreamEndCapture(e->cap, &g);
  if (err2 != cudaSuccess) return err2;
  if (err != cudaSuccess) return err;
  err = cudaGraphInstantiate(gx, g, 0);
  cudaGraphDestroy(g);
  return err;
}
template <class F>
static inline cudaError_t smk_graph_launch(smk_engine* e, cudaGraphExec_t* gx, F f, cudaStream_t st) {
  cudaError_t err = smk_graph_capture(e, gx, f);
  if (err != cudaSuccess) return err;
  return cudaGraphLaunch(*gx, st);
}
