"""swift-mk batched decode: S sequence slots on one set of weights, B of them decoded per cycle in one weight pass (include/batch.h).

  from batch import BatchDecoder
  bd = BatchDecoder(ctx_max=8192, slots=4)            # slot 0 = a SpecEngine (tmax 8, one in-place DeltaNet state); slots 1..3 share its weights and activations
  for s, ids in zip(bd.slots, prompts):                # prefill per slot with the slot's own API (SpecEngine.prefill_spec, FastPrefill(s).prefill_spec, ...)
      first = s.prefill_spec(ids, k=1)
  new = bd.cycles(bd.slots, kv0=1, kd=1, n=8)          # -> list of new tokens per slot, in the order given

A slot is a SpecEngine view (copy.copy of slot 0 with its own C engine + spec object): everything that is per sequence (device scalars, DeltaNet state / conv ring / delta
buffer, KV caches, processors, sampling config, output ring) is its own; prefill, snapshots, sampling setup, single-sequence cycles and the prompt-lookup tail work on it unchanged.
Rows per cycle: B x (kv + 1) <= ROWS (16: the W4 GEMVs take two n8 token tiles per weight fragment, row-count invariant).  The first cycle of a call verifies kv0 drafts, which must not exceed what every sequence's previous cycle drafted (`drafted`).
Needs an RTX 3090 (sm_86; no other GPU has been tested).
"""
import copy
import ctypes as C

import torch

from spec import RING, SS_N, SpecEngine, smk

BMAX = 8
KMAX = 3
ROWS = 16  # rows of one batched weight pass (SMK_RMAX, include/engine.h)


def setup_batch_api(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_batch_create.argtypes = [vp]
    lib.smk_batch_create.restype = vp
    lib.smk_batch_destroy.argtypes = [vp]
    lib.smk_batch_cycles.argtypes = [vp, C.POINTER(vp), ci, ci, ci, ci, vp, C.POINTER(ci), C.POINTER(ci)]
    lib.smk_batch_cycles.restype = ci
    lib.smk_batch_prewarm.argtypes = [vp, ci, ci, ci, ci]
    lib.smk_batch_prewarm.restype = ci
    lib.smk_batch_bytes.argtypes = [vp]
    lib.smk_batch_bytes.restype = C.c_size_t
    lib.smk_engine_bytes.argtypes = [vp]
    lib.smk_engine_bytes.restype = C.c_size_t
    lib.smk_engine_canary_check.argtypes = [vp]
    lib.smk_engine_canary_check.restype = ci


def new_slot(main):
    """Another sequence slot of `main` (a SpecEngine created with state_slots=1): its own C engine (sharing main's weights, activations, logits and workspaces) and spec object."""
    lib = main.lib
    h = lib.smk_engine_create_slot(C.byref(main.m), main.ctx_max, main.kv_fp8, main.tmax, lib.smk_engine_sslots(main.h), main.h)
    if not h:
        raise RuntimeError("smk_engine_create_slot failed (out of device memory?)")
    sp = lib.smk_spec_create(h, C.byref(main.mw), main.head_mode)
    if not sp:
        raise RuntimeError("smk_spec_create failed (out of device memory?)")
    s = copy.copy(main)
    s.h, s.sp = h, sp
    s._ss = (C.c_int * SS_N)()
    s._ring = (C.c_int * RING)()
    s._emitted = 0
    s.sampling = False
    s.pld = False
    s.pos = 0
    s._fp_holder = main  # FastPrefill scratch: one for all slots (prefills run one at a time)
    s.__dict__.pop("_fp_alloc", None)
    torch.cuda.synchronize()
    return s


class BatchDecoder:
    def __init__(self, slots, ctx_max=4096, kv_fp8=0, verbose=False, **kw):
        """slots: number of sequence slots S (1..8).  kw: SpecEngine arguments (host_tables, head_quant, ...); the engine is created with tmax = 8 and one DeltaNet state per slot."""
        assert 1 <= slots <= BMAX
        main = SpecEngine(ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=verbose, tmax=8, state_slots=1, **kw)
        setup_batch_api(main.lib)
        self.lib = main.lib
        self.b = self.lib.smk_batch_create(main.sp)
        if not self.b:
            raise RuntimeError("smk_batch_create failed")
        self.slots = [main] + [new_slot(main) for _ in range(slots - 1)]
        for s in self.slots:
            s.drafted = 0  # drafts the sequence's last cycle / start produced (what the next cycle may verify)
        self._ss = (C.c_int * (BMAX * SS_N))()
        self._ring = (C.c_int * (BMAX * RING))()
        self.st = main.st
        self.rows = ROWS  # the scheduler sizes the drafts per sequence from this (server/smk_batch.py)

    def slot_bytes(self, i):
        """Device bytes the C objects of slot i allocated (slot 0 includes the shared activations and the batch buffers)."""
        s = self.slots[i]
        return int(self.lib.smk_engine_bytes(s.h))

    def prewarm(self, B, kv, kd, samp=False):
        smk.check(self.lib.smk_batch_prewarm(self.b, B, kv, kd, int(bool(samp))), "batch_prewarm")

    def cycles(self, seqs, kv0, kd, n):
        """n batched cycles over the slots `seqs`; returns the new tokens of each (list of lists, same order) and their state ints."""
        B = len(seqs)
        arr = (C.c_void_p * B)(*[s.sp for s in seqs])
        smk.check(self.lib.smk_batch_cycles(self.b, arr, B, kv0, kd, n, self.st, self._ss, self._ring), "batch_cycles")
        out, sss = [], []
        for i, s in enumerate(seqs):
            ss = list(self._ss[i * SS_N : (i + 1) * SS_N])
            outn = ss[12]
            assert outn - s._emitted <= RING, "output ring overrun"
            out.append([self._ring[i * RING + (j % RING)] for j in range(s._emitted, outn)])
            s._emitted = outn
            s.drafted = kd
            sss.append(ss)
        return out, sss
