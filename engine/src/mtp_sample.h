// swift-mk K7: sampling (temperature / top-k / top-p) and exact speculative-sampling acceptance, device side.  Internal header shared by mtp.cu (graph
// wiring, greedy accept) and mtp_sample.cu (kernels, test hooks).  Contract: include/mtp.h (smk_sampling) + ENGINE.md.
#pragma once
#include "engine_impl.h"
#include "mtp.h"

namespace smk {
// ---- device state ints (smk_spec::ss)
// K11: SS_MTOK moved 8 -> 48 (8 catch-up tokens; 8..11 collided with SS_OUTN once rows > 4), SS_HIST holds m = 0..7 accepted drafts, SS_FORCE up to 8 values.
constexpr int SS_A = 0, SS_AM1 = 1, SS_ZERO = 2, SS_DPOS = 3, SS_OUTN = 12, SS_NCYC = 13, SS_FORCE_N = 14, SS_HIST = 16, SS_FORCE = 24,
              SS_IDX = 32, SS_MTOK = 48, SS_PLDSEQ = 56, SS_N = SMK_SPEC_SS_N;
constexpr int RING = SMK_SPEC_RING;

// ---- sampling
constexpr int SK_KMAX = 64;                                        // max top_k
constexpr int SK_CHUNK = 2048;                                     // logits per stage-1 block (256 threads x 8)
constexpr int SK_NCH_T = (SMK_VOCAB + SK_CHUNK - 1) / SK_CHUNK;    // 122 stage-1 chunks per target row
constexpr int SK_NCH_D = 40;                                       // draft rows: up to 40 chunks (81920 logits)
constexpr int SK_ROWS = SMK_TMAX;                                  // verify rows 1 + k, k <= SMK_SPEC_PMAX = 7
constexpr size_t SK_WS_ENTRIES = (size_t)SK_ROWS * SK_NCH_T * (SK_KMAX + 1);  // uint64 workspace of sk_topk: stage-1 lists + per-chunk maxima

// A processed distribution: <= top_k tokens sorted by probability (descending), probabilities after temperature / top-k / top-p and renormalised.
struct Dist {
  int n;
  int ids[SK_KMAX];
  float p[SK_KMAX];
};

// Device-resident sampling configuration (filled by smk_spec_set_sampling; target = verify rows, d_* = the draft proposal q).
struct SampP {
  float inv_t, top_p;
  int top_k;
  float d_inv_t, d_top_p;
  int d_top_k;
  int flags;  // bit0: TEST ONLY (broken variant): on rejection resample from p instead of the residual.  bit1: keep the engine's argmax as the first token.
  float min_p;  // min_p > 0: target rows keep the tokens with p >= min_p * p_max (before top-p)
  unsigned long long seed;
  ProcP* pr;  // request logit processors (proc.h); null in the host-fed test hooks
};

// rng streams (philox counter word 1): the counter word 0 is the cycle counter ss[SS_NCYC]
constexpr uint32_t ST_ACCEPT = 0x1000, ST_DRAFT = 0x2000, ST_FIRST = 0x3000, ST_DRAW = 0x4000;

// Two-stage top-k + temperature/top-k/top-p processing of R rows of logits [R][stride] over N entries -> out[R]; sel 0: target params, 1: draft params.
// idmap (optional, int32 [N]) maps a column to the full-vocab token id.  ws: SK_WS_ENTRIES uint64.
cudaError_t sk_topk(const float* logits, int R, int N, size_t stride, const int* idmap, const SampP* sp, int sel, uint64_t* ws, Dist* out, cudaStream_t st);
// after the prefill's last head step: first token = sample of pd[0] (unless flags bit1) -> ds[DS_OUT], ds[DS_TOK]
cudaError_t sk_first_pick(const Dist* pd, const SampP* sp, const int* ss, int* ds, int bump, cudaStream_t st);
// draft j (1-based): token sampled from q = qd -> dst
cudaError_t sk_draft_pick(Dist* qd, const SampP* sp, const int* ss, int j, int* dst, cudaStream_t st);
// K11: proposals qd[t] = delta at ds[DS_TOK + 1 + t] for t = from..to-1 (prompt-lookup tail tokens: q = delta, accepted w.p. p(d), residual = p without d)
cudaError_t sk_delta_q(const int* ds, Dist* qd, int from, int to, cudaStream_t st);
// verify rows pd[0..k], proposals qd[0..k-1], drafts ds[DS_TOK+1..k] -> y into ds[DS_OUT..], bookkeeping (spec_finish)
cudaError_t sk_accept(int* ds, int* ss, int* ring, int k, const Dist* pd, const Dist* qd, const SampP* sp, cudaStream_t st);
// config -> device struct (mtp_sample.cu); validation of a smk_sampling
SampP smk_make_sampp(const smk_sampling* c);
bool smk_sampling_valid(const smk_sampling* c);
// n draws from pd[0] (counter ctr0 + i, stream ST_DRAW) -> device out[n]
cudaError_t sk_draw_n(const Dist* pd, const SampP* sp, int n, uint32_t ctr0, int* out, cudaStream_t st);

// ---- shared bookkeeping of one cycle after the emitted tokens y[0..m] (= ds[DS_OUT..]) are known; m = accepted drafts (a = m+1 tokens).
__device__ __forceinline__ void spec_finish(int* ds, int* ss, int* ring, int m, int T) {
  const int P = ds[DS_POS];
  int a = m + 1;
  const int fn = ss[SS_FORCE_N], ci = ss[SS_NCYC];
  if (fn > 0) {
    const int f = ss[SS_FORCE + (ci % fn)];
    if (f >= 1 && f < a) a = f;
  }
  const int outn = ss[SS_OUTN];
  for (int t = 0; t < a; t++) {
    const int y = ds[DS_OUT + t];
    ring[(outn + t) % RING] = y;
    ss[SS_MTOK + t] = y;
  }
  ss[SS_OUTN] = outn + a;
  ss[SS_NCYC] = ci + 1;
  ss[SS_HIST + m] += 1;
  ds[DS_TOK] = ds[DS_OUT + a - 1];
  ss[SS_A] = a;
  ss[SS_AM1] = a - 1;
  ss[SS_DPOS] = P;
  ss[SS_DPOS + 1] = P + a;
  ss[SS_DPOS + 2] = P + a + 1;
  ds[DS_POS] = P + a;
  dn_pend_after(ds, a, T);
  ds[DS_CUR] = (ds[DS_CUR] + a) % ds[DS_SLOTS];
}
}  // namespace smk
