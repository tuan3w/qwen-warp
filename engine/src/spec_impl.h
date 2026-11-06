// swift-mk: private state of the speculative driver (mtp.cu) shared with the batched decoder (batch.cu).  Not a public header.
#pragma once
#include "mtp_sample.h"
#include "pld.h"

namespace smk {
constexpr int NH8 = H / 8;  // uint4 chunks per hidden row = 640

// MTP input row: dst = rmsnorm0(src; w) of one bf16 [5120] row (one block of NH8 threads, thread c owns chunk c; wsum = __shared__ float[32]).  Used for both halves of
// cat = [ rmsnorm0(embed[tok]; pre_e) | rmsnorm0(hidden; pre_h) ].
__device__ __forceinline__ void mtp_norm_row(const uint4* __restrict__ src, const uint4* __restrict__ w, uint4* __restrict__ dst, float* wsum) {
  const int c = threadIdx.x;
  const uint4 hv = src[c];
  float v[8] = {bflo(hv.x), bfhi(hv.x), bflo(hv.y), bfhi(hv.y), bflo(hv.z), bfhi(hv.z), bflo(hv.w), bfhi(hv.w)};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH8 / 32; i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  dst[c] = make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                      pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
}

// MTP output row: out = rmsnorm0(bf16(h + y); w) (one block of NH8 threads).
__device__ __forceinline__ void mtp_final_row(const uint4* __restrict__ h, const uint4* __restrict__ y, const uint4* __restrict__ w, uint4* __restrict__ out, float* wsum) {
  const int c = threadIdx.x;
  const uint4 hv = h[c], yv = y[c];
  float v[8] = {bf16r(bflo(hv.x) + bflo(yv.x)), bf16r(bfhi(hv.x) + bfhi(yv.x)), bf16r(bflo(hv.y) + bflo(yv.y)), bf16r(bfhi(hv.y) + bfhi(yv.y)),
                bf16r(bflo(hv.z) + bflo(yv.z)), bf16r(bfhi(hv.z) + bfhi(yv.z)), bf16r(bflo(hv.w) + bflo(yv.w)), bf16r(bfhi(hv.w) + bfhi(yv.w))};
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) ss = fmaf(v[i], v[i], ss);
  ss = warp_sum(ss);
  if ((c & 31) == 0) wsum[c >> 5] = ss;
  __syncthreads();
  float tot = 0.f;
  for (int i = 0; i < NH8 / 32; i++) tot += wsum[i];
  const float r = 1.0f / sqrtf(tot / (float)H + 1e-6f);
  const uint4 wv = w[c];
  const float w0 = bflo(wv.x), w1 = bfhi(wv.x), w2 = bflo(wv.y), w3 = bfhi(wv.y), w4 = bflo(wv.z), w5 = bfhi(wv.z), w6 = bflo(wv.w), w7 = bfhi(wv.w);
  out[c] = make_uint4(pack_bf16x2(v[0] * r * (1.f + w0), v[1] * r * (1.f + w1)), pack_bf16x2(v[2] * r * (1.f + w2), v[3] * r * (1.f + w3)),
                      pack_bf16x2(v[4] * r * (1.f + w4), v[5] * r * (1.f + w5)), pack_bf16x2(v[6] * r * (1.f + w6), v[7] * r * (1.f + w7)));
}
}  // namespace smk

// K11: one-cycle result mailbox in mapped pinned host memory, written by the last kernel of a PLD cycle graph: the host spins on seq, reads the emitted tokens and the new MTP drafts
// (the proposer needs them to pick the width of the next cycle's graph).
struct PldRes {
  volatile int seq;
  int a;
  int y[8];
  int d[SMK_SPEC_KMAX];
  int pad[2];
};

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
  smk::SampP* sp = nullptr;
  smk::Dist *pd = nullptr, *qd = nullptr;
  uint64_t *ws_t = nullptr, *ws_d = nullptr;
  int samp = 0;           // 1: launch the sampling graph set
  // cycle graphs, index [samp][k]
  // K11: k = 1..7 (chain k <= 3 + prompt-lookup tail k - 3 tokens), verify rows 1 + k
  cudaGraphExec_t g_cycle[2][SMK_SPEC_PMAX + 1] = {}, g_verify[2][SMK_SPEC_PMAX + 1] = {}, g_stage[2][SMK_SPEC_PMAX + 1] = {}, g_cu[2][SMK_SPEC_PMAX + 1] = {};
  cudaGraphExec_t g_start[2][SMK_SPEC_PMAX + 1] = {}, g_prefill[smk::TM + 1] = {}, g_pld[2][SMK_SPEC_PMAX + 1] = {};
  cudaGraphExec_t g_first = nullptr, g_plain = nullptr, g_smp_t[SMK_SPEC_PMAX + 1] = {}, g_smp_d[SMK_SPEC_KMAX + 1] = {};
  // K11 hybrid draft: host proposer + per-cycle result mailbox (mapped pinned memory written by the last kernel of the cycle graph)
  PldRes* res = nullptr;   // host pointer of the mapped mailbox
  PldRes* res_dev = nullptr;
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
