// swift-mk K4: on-device MTP speculative decoding (greedy) for the 64-layer engine.  Contract: docs/ENGINE.md + oracle/NOTES.md §14.
//
// One captured CUDA graph per cycle (k = 1..3 drafts):
//   verify : T = 1+k rows [anchor, d1..dk] through the main model (engine step, final norm + lm_head + argmax for every row: the full BF16 head, or by default the K9 quantised head + exact rescoring of the top-R candidates, include/head_q.h)
//   accept : single-thread kernel: m = leading rows whose argmax equals the next draft, a = m+1 (optionally forced lower for rollback tests);
//            appends y[0..a-1] to the device output ring, a -> next anchor y[a-1], pos += a, cur = (cur+a)%slots (DeltaNet ring slot, slots = tmax+1), MTP positions
//   mtp    : catch-up of the a accepted (hidden, next-token) pairs through the MTP layer (T = k+1 rows, padded; KV written positionally at
//            P..P+a-1), draft 1 from the last accepted row, drafts 2..k = single-row chains on the MTP's own post-norm hidden (uncommitted KV
//            at P+a.., overwritten next cycle).  Draft head = restricted-vocab int4 head (head_mode 0) or the full BF16 target head (1, diagnostic).
// All per-cycle scalars (pos, cur, anchor, drafts, a, MTP positions, output counter) live on the device: the host launches the graph n times
// and reads the output ring.  Everything uses the engine's streams/buffers; the engine and the spec object must outlive each other's use.
#pragma once
#include "engine.h"
#include "pld.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SMK_SPEC_KMAX 3     // MTP chain drafts
#define SMK_SPEC_PMAX 7     // K11: chain + prompt-lookup tail: verify rows 1 + k <= 8 (needs an engine with tmax = 8 for k > 3)
#define SMK_SPEC_RING 4096   // device output ring (tokens); the host must fetch at least every RING/2 tokens
#define SMK_SPEC_SS_N 64     // host-visible device state ints (see smk_spec_run)

// MTP weights in engine layouts (python packer: pack_mtp.py).  W4 = uint32 [N][K/8] + bf16 scale [N][K/128].
typedef struct smk_mtp_w {
  smk_layer_w layer;           // kind = 1: ln1, ln2, qkv (W4 [14336][5120] = q|k|v), q_norm, k_norm, o (W4 [5120][6144]), gu (W4 [34816][5120] interleaved), dn
  const void *fc_w, *fc_s;     // W4 [5120][10240]; input row = [rmsnorm(embed(tok); pre_e) | rmsnorm(hidden; pre_h)]
  const void *pre_e, *pre_h, *norm;  // bf16 [5120] zero-centred
  const void *head_w, *head_s; // W4 [head_nv][5120]: restricted draft head (rows = lm_head rows of head_ids)
  const int* head_ids;         // device int32 [head_nv], ascending full-vocab token ids
  int head_nv;                 // multiple of 16
} smk_mtp_w;

typedef struct smk_spec smk_spec;
// head_mode 0: restricted W4 head, 1: full BF16 target lm_head (diagnostic).
smk_spec* smk_spec_create(smk_engine* e, const smk_mtp_w* w, int head_mode);
void smk_spec_destroy(smk_spec* s);
// Zero the MTP KV cache, the output ring and all counters (call together with smk_engine_reset).
cudaError_t smk_spec_reset(smk_spec* s, cudaStream_t st);
// Zero only the device state ints (cycle counter = philox counter, output count, histogram, force) and the output ring; the MTP KV cache is kept (prefix reuse across turns).
cudaError_t smk_spec_reset_counters(smk_spec* s, cudaStream_t st);
// Prefill: after smk_engine_step(T, head=2) of prompt rows at positions pos0..pos0+T-1, commit their MTP pairs (hidden_p, token_{p+1}):
// next_tok[0..T-1] = prompt tokens pos0+1..pos0+T (host array).  Only the MTP attention K/V are computed.
cudaError_t smk_spec_prefill_chunk(smk_spec* s, int T, const int* next_tok, int pos0, cudaStream_t st);
// After the final prompt row ran as smk_engine_step(1, head=1) (sets the first anchor): commit its MTP pair and produce drafts 1..k (-> device).
cudaError_t smk_spec_start(smk_spec* s, int k, cudaStream_t st);
// Forced partial acceptance (rollback test): cycle i accepts min(natural, force[i % n]) rows (force values >= 1; n = 0 disables).
cudaError_t smk_spec_set_force(smk_spec* s, const int* force, int n, cudaStream_t st);
// Launch n full cycles (graph replays) then synchronise and copy the device state ints (SMK_SPEC_SS_N) and the output ring to the host.
//   ss[12] = total tokens emitted since reset (ring index = i % SMK_SPEC_RING), ss[13] = cycles, ss[16+m] = cycles with m accepted drafts (m=0..3),
//   ss[0] = a of the last cycle.  Tokens emitted by the cycles are the greedy continuation after the prefill's first token.
cudaError_t smk_spec_run(smk_spec* s, int k, int n, cudaStream_t st, int* host_ss, int* host_ring);
// Timing parts (async launches of n replays): which 0 = full cycle, 1 = verify + accept only, 2 = MTP stage only (catch-up k+1 rows + k drafts),
// 3 = MTP catch-up only (k+1 rows, full layer, no drafts); sampling mode only: 4 = the sampling kernels of the verify rows (top-k/top-p of 1+k rows + accept),
// 5 = the sampling kernels of the k draft heads (top-k/top-p of the head logits + draw).  State evolves (or is replayed on stale state), the cost is state independent.
cudaError_t smk_spec_launch(smk_spec* s, int k, int which, int n, cudaStream_t st);
// Debug copies: what 0: ss ints [SMK_SPEC_SS_N]; 1: MTP post-norm hidden of the drafts bf16 [4][5120] (row j = draft j+1); 2: processed verify-row distributions
// (struct {int n; int ids[64]; float p[64]} x 4); 3: processed draft proposals q (same struct x 3).
cudaError_t smk_spec_read(smk_spec* s, int what, void* host, size_t bytes);
size_t smk_spec_bytes(smk_spec* s);

// ------------------------------------------------------------------------------------------------ K11: hybrid draft = MTP chain + prompt-lookup (n-gram) tail
// Needs an engine created with tmax = 8 (smk_engine_create_ex / env SMK_TMAX=8: 9 DeltaNet ring slots) for tails longer than 1 token.  Each cycle verifies T = 4 + Ln rows
// [anchor, d1, d2, d3, tail_1..tail_Ln] (Ln = 0..4; T = 4 is the plain k=3 cycle): the host proposer (pld.h) looks the n-gram that ends the MTP chain up in prompt + generated text
// and appends the continuation of the best earlier occurrence; the matching cycle graph (one per T) is launched.  The verify rows / accept / MTP catch-up (T rows) / drafts of the
// next cycle all stay on the device; the per-cycle host work is: spin on a mapped mailbox written by the last kernel of the graph (emitted tokens + the new drafts d1..d3), update the
// proposer, look up, copy <= 4 tail tokens to the device, launch.  Greedy: tail token i is accepted iff it equals the verify argmax of its row (leading match).  Sampling (K7): tail
// tokens are delta proposals, q = delta: accepted w.p. p(d), on rejection the residual max(0, p - q)/Z = p without d, renormalised; the emitted stream stays exactly p.
typedef struct smk_pld_opts {
  smk_pld_cfg cfg;      // proposer (smk_pld_default_cfg: n = 2 longest suffix, agreement >= 3, tail <= 4 = E6's E3 cell)
  int force_n;          // TEST ONLY: > 0: every cycle carries exactly these force_n tail tokens (force_tok) instead of the proposer's
  int force_tok[4];
  int reserved[6];
} smk_pld_opts;
// opts == NULL: disable (smk_spec_run_pld then behaves like smk_spec_run with k = 3 but host-driven).  Resets the statistics.
cudaError_t smk_spec_pld_config(smk_spec* s, const smk_pld_opts* opts);
// Capture the cycle graphs of every width (greedy and sampling) now instead of lazily at the first cycle that needs one: the server calls this at startup (a lazy capture under memory pressure
// fails with cudaErrorMemoryAllocation once the allocator holds the free memory).
cudaError_t smk_spec_pld_prewarm(smk_spec* s);
// After smk_spec_start (greedy or sampling): ctx = prompt tokens + the first generated token (the anchor = last element); reads the first drafts from the device (synchronises).
cudaError_t smk_spec_pld_begin(smk_spec* s, const int* ctx, int n, cudaStream_t st);
// n cycles, one host round trip each (width chosen per cycle); then copies the device state ints and the output ring to the host exactly like smk_spec_run.
cudaError_t smk_spec_run_pld(smk_spec* s, int n, cudaStream_t st, int* host_ss, int* host_ring);
// out[16] int64: [1..7] cycles launched with k drafts (T = k + 1 rows), [8] tail tokens proposed, [9] cycles with a tail, [10] host wait for the result (ns), [11] host proposer + launch work (ns),
// [12] unused.  ss[16 + m] (smk_spec_run's histogram) counts cycles by accepted drafts m = 0..7.
cudaError_t smk_spec_pld_stats(smk_spec* s, long long* out);
// ------------------------------------------------------------------------------------------------ K7: sampling with exact speculative verification
// temperature > 0 switches the spec cycle (and smk_spec_start / smk_spec_sample_first / smk_spec_plain_run) to the sampling graphs; temperature == 0 = the greedy graphs above.
// Target distribution p (per verify row) = HF/vLLM logits-processor order: logits / temperature -> top-k (<= 64) -> top-p (keep token i iff the mass of the tokens before it < top_p,
// at least one) -> softmax over the survivors.  Drafts are sampled from q = the same processing of the draft head's logits over the restricted vocabulary (draft_sample = 1; the
// draft head temperature / top-k / top-p default to the target's), or q = delta at the draft argmax (draft_sample = 0).  Acceptance: d accepted w.p. min(1, p(d)/q(d)); on rejection
// sample from max(0, p - q)/Z (p may put mass outside the draft vocabulary: handled by the residual); all accepted: bonus token from p of the last row.  The emitted tokens are then
// distributed exactly as autoregressive sampling from p.  RNG: philox4x32-10, key = seed, counter = (cycle counter ss[13], stream, 0, 0); the cycle counter is a device scalar
// advanced per cycle, so a run is reproducible from (seed, smk_spec_reset); no host sync.
typedef struct smk_sampling {
  float temperature;         // 0 = greedy
  int top_k;                 // 1..64
  float top_p;               // (0, 1]
  int draft_sample;          // 1: sample drafts from q; 0: greedy draft (q = delta at the draft head argmax)
  float draft_temperature;   // 0 = temperature
  int draft_top_k;           // 0 = top_k
  float draft_top_p;         // 0 = top_p
  int test_flags;            // TEST ONLY.  bit0: broken variant (on rejection resample from p, not the residual); bit1: keep the engine's argmax as the first token
  unsigned long long seed;
} smk_sampling;
cudaError_t smk_spec_set_sampling(smk_spec* s, const smk_sampling* cfg, cudaStream_t st);
// After the prefill's last head step (logits row 0 in the engine): sample the first generated token from it (-> device anchor / next input).  Call before smk_spec_start.
cudaError_t smk_spec_sample_first(smk_spec* s, cudaStream_t st);
// Plain (non-speculative) sampled decode: n replays of [engine T=1 step + top-k/top-p/sample], host tokens[0..n-1] (synchronises).  Continues from the staged next input token.
cudaError_t smk_spec_plain_run(smk_spec* s, int n, int* host_tokens, cudaStream_t st);
// Test hooks.  snapshot: copy the engine state a cycle may modify (device scalars ds, the current DeltaNet state / conv slot of every layer, the final-norm tap row 0);
// restore: put it back (the cycle counter / output ring are NOT restored, so repeated runs draw fresh random numbers).
cudaError_t smk_spec_snapshot(smk_spec* s, cudaStream_t st);
cudaError_t smk_spec_restore(smk_spec* s, cudaStream_t st);
// n independent draws from the processed distribution of engine logits row `row` (counter ctr0 + i, stream 0x4000) -> host out[n] (synchronises).
cudaError_t smk_test_draw(smk_spec* s, int n, unsigned ctr0, int row, int* out);
// Processed distributions (the same kernels as the graphs) of host logits [R][N] (R <= 4, N <= 248320; idmap optional): out_n[R], out_ids[R][64], out_p[R][64]; sel 0 = target, 1 = draft params.
cudaError_t smk_test_dist(const float* logits, int R, int N, const int* idmap, const smk_sampling* cfg, int sel, int* out_n, int* out_ids, float* out_p);
// ntrials independent speculative-sampling cycles on fixed rows: target logits tl [k+1][NT], draft logits dl [k][ND] (+ idmap [ND] -> full ids; NULL = identity), the SAME device
// functions as the graph (draft pick, accept / residual / bonus).  y[ntrials][4] emitted tokens, a[ntrials] = emitted count.
cudaError_t smk_test_trials(const float* tl, int NT, const float* dl, int ND, const int* idmap, const smk_sampling* cfg, int k, int ntrials, unsigned ctr0, int* y, int* a);
// K11 test hook: ntrials speculative-sampling cycles on fixed rows with nchain drafts from the draft logits dl [nchain][ND] and ntail delta tail tokens tail[0..ntail-1] (rows tl [nchain+ntail+1][NT]).
// y[ntrials][8] emitted tokens (-1 padded), a[ntrials] = emitted count.  The same device functions as the graph.
cudaError_t smk_test_trials_pld(const float* tl, int NT, const float* dl, int ND, const int* idmap, const smk_sampling* cfg, int nchain, const int* tail, int ntail, int ntrials, unsigned ctr0, int* y, int* a);
// philox4x32-10 reference output for counter c[4], key kk[2] (checked against the Random123 known-answer vectors).
cudaError_t smk_test_philox(const unsigned* c, const unsigned* kk, unsigned* out);

#ifdef __cplusplus
}
#endif
