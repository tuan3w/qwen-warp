// swift-mk: request-level logit processors on the device (OpenAI / vLLM semantics: allowed_token_ids, min_tokens, logit_bias, repetition / frequency / presence penalties), internal header.
// Contract: include/engine.h (smk_procs) + docs/ARCHITECTURE.md.  Kernels and the per-request state upload: proc.cu; the callers: engine.cu (head tail), mtp_sample.cu (token notes).
//
// Order (vLLM v1 sampler.py): allowed mask -> min_tokens mask -> logit_bias -> repetition -> frequency -> presence, all elementwise in the token's own logit given its state:
// count = how often it was generated (device counts + the draft tokens of the verify rows before it), flags = in the prompt / allowed / stop token, bias.
#pragma once
#include <stdint.h>
#include <cuda_runtime.h>

#include "engine.h"

namespace smk {
constexpr uint8_t PF_PROMPT = 1, PF_ALLOW = 2, PF_STOP = 4;

// Device-resident processor state of the one running request.  Every processor kernel returns at once when on == 0 (the default).
struct ProcP {
  int on;            // any elementwise processor active
  int has_allowed;   // allowed_token_ids given: tokens without PF_ALLOW are masked
  int min_tokens;    // stop tokens (PF_STOP) are masked while fewer tokens have been generated
  int ngen;          // tokens generated so far (device-updated by proc_note)
  float rep, pres, freq, pad;
  int* counts;       // int [VOC]: generated-token counts (device-updated by proc_note)
  uint8_t* flags;    // [VOC]: PF_* bits
  float* bias;       // [VOC]: logit_bias (0 where not set)
};

struct ProcBufs {  // device buffers behind a ProcP
  ProcP* dev;
  int* counts;
  uint8_t* flags;
  float* bias;
  int* ids;          // upload scratch [PROC_CHUNK]
  float* vals;       // upload scratch [PROC_CHUNK]
};
constexpr int PROC_CHUNK = 65536;

// The processed logit of one token for a verify row that has already seen `ngen_t` generated tokens.  Explicit IEEE intrinsics (no fma contraction): bit-identical to
// the torch reference ops in the same order.
__device__ __forceinline__ float proc_logit(const ProcP& P, float x, int cnt, uint32_t fl, float bias, int ngen_t) {
  if (P.has_allowed && !(fl & PF_ALLOW)) x = -INFINITY;
  if (P.min_tokens > ngen_t && (fl & PF_STOP)) x = -INFINITY;
  x = __fadd_rn(x, bias);
  if (P.rep != 1.0f && ((fl & PF_PROMPT) || cnt > 0)) x = x > 0.0f ? __fdiv_rn(x, P.rep) : __fmul_rn(x, P.rep);
  if (cnt > 0) {
    x = __fsub_rn(x, __fmul_rn(P.freq, (float)cnt));
    x = __fsub_rn(x, P.pres);
  }
  return x;
}

// Record the emitted tokens y[0..a-1] as generated (single thread).
__device__ __forceinline__ void proc_note(ProcP* P, const int* y, int a) {
  if (!P || !P->on) return;
  for (int t = 0; t < a; t++) P->counts[y[t]] += 1;
  P->ngen += a;
}

// ---- launchers (proc.cu).  d = the draft tokens of the verify rows (device int [T - 1]: row t's history ends with d[0..t-1]).
// Approximate logits [T][VOC] in place (all rows, before the candidate selection of the quantised head).
cudaError_t proc_logits(const ProcP* pr, float* logits, const int* d, int T, cudaStream_t st);
// Exact logits of the candidates: exact[t * rstride + r] for candidate id cand[t * rstride + r] (-1 = unused), also written to logits[t][id] when logits != null.
cudaError_t proc_cand(const ProcP* pr, const int* cand, float* exact, int rstride, float* logits, const int* d, int T, cudaStream_t st);

// Scratch processor state for the host-fed test hooks (no engine): init = allocate, start the request p and note the generated tokens gen[0..n_gen-1] through the device path.
struct ProcScratch {
  ProcBufs b = {};
  int* gen = nullptr;
  ~ProcScratch();
  cudaError_t init(const smk_procs* p, const int* host_gen, int n_gen);
};
}  // namespace smk
