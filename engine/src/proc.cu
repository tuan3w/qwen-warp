// swift-mk: request-level logit processors (allowed_token_ids, min_tokens, logit_bias, repetition / frequency / presence penalty).  See proc.h and include/engine.h (smk_procs).
#include "engine_impl.h"

using namespace smk;

namespace {
// Approximate logits [T][VOC] in place: 4 consecutive tokens per thread, the token state is read once and applied to every row.  Row t sees the draft tokens d[0..t-1] as generated.
__global__ void __launch_bounds__(256) k_proc_logits(const ProcP* __restrict__ pr, float* __restrict__ logits, const int* __restrict__ d, int T) {
  const ProcP P = *pr;
  if (!P.on) return;
  const int i4 = blockIdx.x * 256 + threadIdx.x;
  if (i4 * 4 >= VOC) return;
  const int4 cnt = reinterpret_cast<const int4*>(P.counts)[i4];
  const uint32_t fl4 = reinterpret_cast<const uint32_t*>(P.flags)[i4];
  const float4 b4 = reinterpret_cast<const float4*>(P.bias)[i4];
  int c[4] = {cnt.x, cnt.y, cnt.z, cnt.w};
  const float b[4] = {b4.x, b4.y, b4.z, b4.w};
  const int v0 = i4 * 4;
  for (int t = 0; t < T; t++) {
    float4* row = reinterpret_cast<float4*>(logits + (size_t)t * VOC) + i4;
    float x[4] = {row->x, row->y, row->z, row->w};
#pragma unroll
    for (int e = 0; e < 4; e++) {
      if (t > 0) c[e] += (d[t - 1] == v0 + e);
      x[e] = proc_logit(P, x[e], c[e], (fl4 >> (8 * e)) & 255u, b[e], P.ngen + t);
    }
    *row = make_float4(x[0], x[1], x[2], x[3]);
  }
}

// Candidate logits: one thread per (row, candidate).
__global__ void k_proc_cand(const ProcP* __restrict__ pr, const int* __restrict__ cand, float* __restrict__ exact, int rstride, float* __restrict__ logits, const int* __restrict__ d, int T) {
  const ProcP P = *pr;
  if (!P.on) return;
  const int t = blockIdx.x, r = threadIdx.x;
  const int id = cand[t * rstride + r];
  if (id < 0) return;
  int c = P.counts[id];
  for (int i = 0; i < t; i++) c += (d[i] == id);
  const float x = proc_logit(P, exact[t * rstride + r], c, P.flags[id], P.bias[id], P.ngen + t);
  exact[t * rstride + r] = x;
  if (logits) logits[(size_t)t * VOC + id] = x;
}

__global__ void k_mark(uint8_t* __restrict__ flags, const int* __restrict__ ids, int n, uint8_t bit) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[ids[i]] |= bit;  // same-bit writers of one byte store the same value
}
__global__ void k_set_bias(float* __restrict__ bias, const int* __restrict__ ids, const float* __restrict__ vals, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) bias[ids[i]] = vals[i];
}
__global__ void k_note_ids(ProcP* pr, const int* ids, int n) { proc_note(pr, ids, n); }

bool ids_ok(const int* ids, int n) {
  if (n < 0 || (n && !ids)) return false;
  for (int i = 0; i < n; i++) if (ids[i] < 0 || ids[i] >= VOC) return false;
  return true;
}

cudaError_t mark_ids(const ProcBufs& b, const int* ids, int n, uint8_t bit, cudaStream_t st) {
  for (int o = 0; o < n; o += PROC_CHUNK) {
    const int m = n - o < PROC_CHUNK ? n - o : PROC_CHUNK;
    SMK_CK(cudaMemcpyAsync(b.ids, ids + o, m * sizeof(int), cudaMemcpyHostToDevice, st));
    k_mark<<<(m + 255) / 256, 256, 0, st>>>(b.flags, b.ids, m, bit);
    SMK_CK(cudaGetLastError());
  }
  return cudaSuccess;
}

// Start a request: zero the state, upload the prompt / allowed / stop / bias tables, set the scalars.  No active processor: only `on` = 0.
cudaError_t proc_begin(const ProcBufs& b, const smk_procs* p, cudaStream_t st) {
  ProcP h = {};
  h.counts = b.counts;
  h.flags = b.flags;
  h.bias = b.bias;
  h.rep = 1.0f;
  const bool any = p && (p->presence != 0.f || p->frequency != 0.f || p->repetition != 1.f || p->min_tokens > 0 || p->n_bias > 0 || p->n_allowed > 0);
  if (any) {
    if (!ids_ok(p->prompt, p->n_prompt) || !ids_ok(p->bias_ids, p->n_bias) || !ids_ok(p->allowed, p->n_allowed) || !ids_ok(p->stop, p->n_stop) || (p->n_bias && !p->bias_vals) ||
        !(p->repetition > 0.f) || p->min_tokens < 0 || p->n_bias > PROC_CHUNK)
      return cudaErrorInvalidValue;
    h.on = 1;
    h.has_allowed = p->n_allowed > 0;
    h.min_tokens = p->min_tokens;
    h.rep = p->repetition;
    h.pres = p->presence;
    h.freq = p->frequency;
    SMK_CK(cudaMemsetAsync(b.counts, 0, (size_t)VOC * sizeof(int), st));
    SMK_CK(cudaMemsetAsync(b.flags, 0, VOC, st));
    SMK_CK(cudaMemsetAsync(b.bias, 0, (size_t)VOC * sizeof(float), st));
    SMK_CK(mark_ids(b, p->prompt, p->n_prompt, PF_PROMPT, st));
    SMK_CK(mark_ids(b, p->allowed, p->n_allowed, PF_ALLOW, st));
    if (p->min_tokens > 0) SMK_CK(mark_ids(b, p->stop, p->n_stop, PF_STOP, st));
    if (p->n_bias) {
      SMK_CK(cudaMemcpyAsync(b.ids, p->bias_ids, p->n_bias * sizeof(int), cudaMemcpyHostToDevice, st));
      SMK_CK(cudaMemcpyAsync(b.vals, p->bias_vals, p->n_bias * sizeof(float), cudaMemcpyHostToDevice, st));
      k_set_bias<<<(p->n_bias + 255) / 256, 256, 0, st>>>(b.bias, b.ids, b.vals, p->n_bias);
      SMK_CK(cudaGetLastError());
    }
  }
  return cudaMemcpyAsync(b.dev, &h, sizeof(h), cudaMemcpyHostToDevice, st);
}
}  // namespace

namespace smk {
cudaError_t proc_logits(const ProcP* pr, float* logits, const int* d, int T, cudaStream_t st) {
  k_proc_logits<<<(VOC / 4 + 255) / 256, 256, 0, st>>>(pr, logits, d, T);
  return cudaGetLastError();
}
cudaError_t proc_cand(const ProcP* pr, const int* cand, float* exact, int rstride, float* logits, const int* d, int T, cudaStream_t st) {
  k_proc_cand<<<T, SMK_HEADQ_RMAX, 0, st>>>(pr, cand, exact, rstride, logits, d, T);
  return cudaGetLastError();
}
}  // namespace smk

// ---------------------------------------------------------------------------------------------- test scratch state (host-fed test hooks)
namespace smk {
ProcScratch::~ProcScratch() {
  for (void* q : {(void*)b.dev, (void*)b.counts, (void*)b.flags, (void*)b.bias, (void*)b.ids, (void*)b.vals, (void*)gen}) if (q) cudaFree(q);
}
cudaError_t ProcScratch::init(const smk_procs* p, const int* host_gen, int n_gen) {
  SMK_CK(cudaMalloc(&b.dev, sizeof(ProcP)));
  SMK_CK(cudaMalloc(&b.counts, (size_t)VOC * 4));
  SMK_CK(cudaMalloc(&b.flags, VOC));
  SMK_CK(cudaMalloc(&b.bias, (size_t)VOC * 4));
  SMK_CK(cudaMalloc(&b.ids, PROC_CHUNK * 4));
  SMK_CK(cudaMalloc(&b.vals, PROC_CHUNK * 4));
  SMK_CK(cudaMalloc(&gen, (n_gen + 1) * 4));
  SMK_CK(proc_begin(b, p, 0));
  if (n_gen) {
    SMK_CK(cudaMemcpy(gen, host_gen, n_gen * 4, cudaMemcpyHostToDevice));
    k_note_ids<<<1, 1>>>(b.dev, gen, n_gen);  // the device note path the speculative cycle uses
    SMK_CK(cudaGetLastError());
  }
  return cudaSuccess;
}
}  // namespace smk

extern "C" {
cudaError_t smk_engine_set_procs(smk_engine* e, const smk_procs* p, cudaStream_t st) { return proc_begin(e->pb, p, st); }

cudaError_t smk_test_proc_logits(const smk_procs* p, int T, const int* gen, int n_gen, const int* drafts, const float* logits, float* out) {
  if (T < 1 || T > SMK_TMAX) return cudaErrorInvalidValue;
  ProcScratch s;
  float* lg = nullptr;
  int* d = nullptr;
  SMK_CK(s.init(p, gen, n_gen));
  SMK_CK(cudaMalloc(&lg, (size_t)T * VOC * 4));
  SMK_CK(cudaMalloc(&d, 8 * 4));
  SMK_CK(cudaMemcpy(lg, logits, (size_t)T * VOC * 4, cudaMemcpyHostToDevice));
  SMK_CK(cudaMemcpy(d, drafts, (T > 1 ? T - 1 : 0) * 4, cudaMemcpyHostToDevice));
  cudaError_t e = proc_logits(s.b.dev, lg, d, T, 0);
  if (e == cudaSuccess) e = cudaMemcpy(out, lg, (size_t)T * VOC * 4, cudaMemcpyDeviceToHost);
  cudaFree(lg);
  cudaFree(d);
  return e;
}

cudaError_t smk_test_proc_cand(const smk_procs* p, int T, const int* gen, int n_gen, const int* drafts, const int* cand, const float* exact, float* out) {
  if (T < 1 || T > SMK_TMAX) return cudaErrorInvalidValue;
  ProcScratch s;
  int *c = nullptr, *d = nullptr;
  float* ex = nullptr;
  const size_t n = (size_t)T * SMK_HEADQ_RMAX;
  SMK_CK(s.init(p, gen, n_gen));
  SMK_CK(cudaMalloc(&c, n * 4));
  SMK_CK(cudaMalloc(&ex, n * 4));
  SMK_CK(cudaMalloc(&d, 8 * 4));
  SMK_CK(cudaMemcpy(c, cand, n * 4, cudaMemcpyHostToDevice));
  SMK_CK(cudaMemcpy(ex, exact, n * 4, cudaMemcpyHostToDevice));
  SMK_CK(cudaMemcpy(d, drafts, (T > 1 ? T - 1 : 0) * 4, cudaMemcpyHostToDevice));
  cudaError_t e = proc_cand(s.b.dev, c, ex, SMK_HEADQ_RMAX, nullptr, d, T, 0);
  if (e == cudaSuccess) e = cudaMemcpy(out, ex, n * 4, cudaMemcpyDeviceToHost);
  cudaFree(c);
  cudaFree(ex);
  cudaFree(d);
  return e;
}
}  // extern "C"
