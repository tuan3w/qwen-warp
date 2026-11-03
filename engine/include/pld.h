// swift-mk E6: host-side prompt-lookup (PLD) tail proposer for the hybrid draft policy "MTP chain + PLD tail".
// The MTP head drafts d1..d3; this pure-CPU component appends up to max_tail (<= 7) tokens copied from earlier in the context.
// Reference algorithm: n-gram prompt lookup, longest suffix match (n = 2), agreement >= 3, tail <= 4.
// Incremental hash index (n-gram -> ascending end positions), n-grams of the history are indexed only up to the current anchor; drafts are never indexed.
// No CUDA calls: plain C++17 (src/pld.cu is compiled by the Makefile glob src/pld*.cu only because it lives next to the other sources).
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

typedef struct smk_pld_cfg {
  int n;         // key n-gram length (1..4), default 2
  int min_len;   // gate m: backward agreement l >= m, default 3
  int max_tail;  // Lt: proposed tail tokens, default 4, <= 7
  int cap;       // candidates examined (most recent first), default 64
  int back;      // backward extension limit, default 64
  int longest;   // 1: longest backward agreement, ties -> most recent; 0: most recent candidate only
} smk_pld_cfg;

typedef struct smk_pld smk_pld;

void smk_pld_default_cfg(smk_pld_cfg* c);
smk_pld* smk_pld_create(const smk_pld_cfg* c);  // NULL cfg = defaults; out-of-range fields are clamped
void smk_pld_destroy(smk_pld* p);
// Context tokens S[0..n-1]; the LAST one is the anchor (already emitted, not yet verified).
void smk_pld_reset(smk_pld* p, const int* ctx, int n);
// Committed tokens appended after the anchor (the previous anchor stays in the history); the last appended token is the new anchor.
void smk_pld_append(smk_pld* p, const int* toks, int n);
int smk_pld_len(const smk_pld* p);
// K = S[0..len-1] ++ drafts[0..nd-1].  Query = the last n tokens of K; candidates = earlier occurrences of that n-gram ending at j <= len-1 (known history only).
// l = n extended backwards while l < back and j-l >= 0 and K[j-l] == K[end-l]; best = largest l (ties: most recent; at most `cap` most recent candidates);
// proposal = K[j+1 .. min(j+1+max_tail, end+1)-1] (may run into the draft tokens), end = len-1+nd.
// Returns the tail length Ln (0 when there is no candidate OR best l < min_len), writes tokens to out[0..Ln-1] (room for max_tail ints)
// and the agreement length to *l_out (0 if no candidate; also set when gated off; l_out may be NULL).
int smk_pld_propose(smk_pld* p, const int* drafts, int nd, int* out, int* l_out);

#ifdef __cplusplus
}
#endif
