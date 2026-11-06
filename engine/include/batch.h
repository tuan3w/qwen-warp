// swift-mk: batched speculative decode of several sequences in ONE weight pass per cycle.  Contract: docs/ARCHITECTURE.md (Batched decode).
//
// A sequence lives in its own engine + spec pair ("slot": smk_engine_create_slot(..., sslots = 1, share = owner engine) + smk_spec_create), which keeps everything that is
// per sequence: the device scalars (pos, cur, tokens, drafts, pending DeltaNet rows), the DeltaNet state + conv ring + delta buffer, the KV cache of the 16 attention layers and of
// the MTP layer, the request's logit processors and sampling configuration, the speculative state ints and the output ring.  Prefill and single-sequence decode use the slot's own
// engine API unchanged.  smk_batch_cycles runs n cycles over B such sequences:
//   verify : the B x (kv+1) rows [anchor, d1..dkv] of all sequences go through the 64 layers together: embeddings, norms, every W4 GEMV and the quantised lm_head + exact rescoring
//            read the weights once for all rows; the DeltaNet and attention kernels run per sequence on its own rows, state and KV cache (their base pointers are read from a
//            device table, so one graph per (B, kv, kd, sampling) serves any set of sequences);
//   accept : per sequence, greedy (leading argmax matches) or exact speculative sampling with the sequence's own parameters and processors (both kernels are in the graph, the
//            sequence's flag selects which one acts);
//   mtp    : catch-up of the accepted rows of every sequence (B x (kv+1) rows) and kd chained drafts, each draft step one B-row pass through the MTP layer and draft head.
// Rows per cycle: B x (kv + 1) <= the engine's tmax (8), kv, kd <= SMK_SPEC_KMAX.  kd = 0: plain batched decode (the catch-up still writes the MTP KV, so drafting can resume).
// The first cycle of a call verifies kv0 drafts (<= what the previous cycle of each sequence drafted), the others kd.
#pragma once
#include "mtp.h"
#ifdef __cplusplus
extern "C" {
#endif
#define SMK_BATCH_MAX 8
typedef struct smk_batch smk_batch;
// owner: the spec object whose engine owns the activations every slot engine shares (share = owner's engine); its MTP weights serve all slots.  The engine needs the quantised head.
smk_batch* smk_batch_create(smk_spec* owner);
void smk_batch_destroy(smk_batch* b);
// n cycles over the B sequences seqs[0..B-1] (spec objects of slot engines of the owner's configuration, all on stream st), then synchronise and copy to the host:
//   host_ss [B][SMK_SPEC_SS_N]: each sequence's state ints (as smk_spec_run: [12] tokens emitted since its reset, [13] cycles, ...); host_ring [B][SMK_SPEC_RING]: its output ring.
// The sequences' device state is read before the first cycle and written back after the last one, so their own API (smk_spec_run, snapshots, prefill) continues from there.
cudaError_t smk_batch_cycles(smk_batch* b, smk_spec* const* seqs, int B, int kv0, int kd, int n, cudaStream_t st, int* host_ss, int* host_ring);
// Capture the graph of (B, kv, kd, sampling) now (startup), instead of at its first use.
cudaError_t smk_batch_prewarm(smk_batch* b, int B, int kv, int kd, int samp);
size_t smk_batch_bytes(smk_batch* b);
#ifdef __cplusplus
}
#endif
