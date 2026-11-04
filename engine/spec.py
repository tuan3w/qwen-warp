"""swift-mk speculative driver: Engine + packed int4 MTP head + on-device draft / verify / accept cycle graph (include/mtp.h).

  from spec import SpecEngine;  e = SpecEngine(ctx_max=4096);  toks = e.generate(ids, 128, k=3)
Needs an RTX 3090 (sm_86; no other GPU has been tested).
"""
import ctypes as C
import time

import torch

from engine import Engine, smk, pack  # noqa: F401
import pack_mtp

RING = 4096
SS_N = 64
PMAX = 7  # K11: chain k <= 3 + prompt-lookup tail <= 4 (verify rows <= 8; needs Engine(tmax=8))


class MtpW(C.Structure):
    _fields_ = [("layer", smk.LayerW)] + [(n, C.c_void_p) for n in ("fc_w", "fc_s", "pre_e", "pre_h", "norm", "head_w", "head_s", "head_ids")] + [("head_nv", C.c_int)]


class PldCfg(C.Structure):
    _fields_ = [(n, C.c_int) for n in ("n", "min_len", "max_tail", "cap", "back", "longest")]


class PldOpts(C.Structure):
    _fields_ = [("cfg", PldCfg), ("force_n", C.c_int), ("force_tok", C.c_int * 4), ("reserved", C.c_int * 6)]


def make_pld(n=2, min_len=3, max_tail=4, cap=64, back=64, longest=1, force_tail=()):
    """E6's best cell (n=2 longest suffix match of the chain's last two drafts, agreement >= 3, tail <= 4)."""
    ft = list(force_tail)[:4]  # TEST ONLY: every cycle carries exactly these tail tokens
    return PldOpts(PldCfg(n, min_len, max_tail, cap, back, longest), len(ft), (C.c_int * 4)(*(ft + [0] * (4 - len(ft)))))


class Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_k", C.c_int), ("top_p", C.c_float), ("draft_sample", C.c_int), ("draft_temperature", C.c_float), ("draft_top_k", C.c_int),
                ("draft_top_p", C.c_float), ("test_flags", C.c_int), ("seed", C.c_ulonglong), ("min_p", C.c_float)]


def make_sampling(temperature=0.0, top_k=20, top_p=1.0, draft_sample=1, draft_temperature=0.0, draft_top_k=0, draft_top_p=0.0, seed=0, test_flags=0, min_p=0.0):
    return Sampling(temperature, top_k, top_p, draft_sample, draft_temperature, draft_top_k, draft_top_p, test_flags, seed, min_p)


class Procs(C.Structure):
    """include/engine.h smk_procs: request-level logit processors."""
    _fields_ = [("presence", C.c_float), ("frequency", C.c_float), ("repetition", C.c_float), ("min_tokens", C.c_int), ("prompt", C.POINTER(C.c_int)), ("n_prompt", C.c_int),
                ("bias_ids", C.POINTER(C.c_int)), ("bias_vals", C.POINTER(C.c_float)), ("n_bias", C.c_int), ("allowed", C.POINTER(C.c_int)), ("n_allowed", C.c_int),
                ("stop", C.POINTER(C.c_int)), ("n_stop", C.c_int)]


def make_procs(prompt=(), presence=0.0, frequency=0.0, repetition=1.0, min_tokens=0, bias=None, allowed=None, stop=()):
    """Returns (Procs, keepalive): bias = {token id: value}, allowed = token ids (None / empty: unrestricted), stop = ids masked by min_tokens."""
    def ints(x):
        t = torch.as_tensor(list(x), dtype=torch.int32).contiguous()
        return t, C.cast(C.c_void_p(t.data_ptr()), C.POINTER(C.c_int)), len(t)

    keep = []
    pr, bi, al, st = ints(prompt), ints((bias or {}).keys()), ints(allowed or ()), ints(stop)
    bv = torch.as_tensor(list((bias or {}).values()), dtype=torch.float32).contiguous()
    keep += [pr[0], bi[0], al[0], st[0], bv]
    p = Procs(presence, frequency, repetition, min_tokens, pr[1], pr[2], bi[1], C.cast(C.c_void_p(bv.data_ptr()), C.POINTER(C.c_float)), bi[2], al[1], al[2], st[1], st[2])
    return p, keep


def setup_spec_api(lib):
    vp, ci, cu = C.c_void_p, C.c_int, C.c_uint
    lib.smk_spec_create.argtypes = [vp, C.POINTER(MtpW), ci]
    lib.smk_spec_create.restype = vp
    lib.smk_spec_destroy.argtypes = [vp]
    lib.smk_spec_bytes.argtypes = [vp]
    lib.smk_spec_bytes.restype = C.c_size_t
    sig = {
        "smk_spec_reset": [vp, vp],
        "smk_spec_reset_counters": [vp, vp],
        "smk_spec_prefill_chunk": [vp, ci, C.POINTER(ci), ci, vp],
        "smk_spec_start": [vp, ci, vp],
        "smk_spec_set_force": [vp, C.POINTER(ci), ci, vp],
        "smk_spec_run": [vp, ci, ci, vp, C.POINTER(ci), C.POINTER(ci)],
        "smk_spec_pld_config": [vp, C.POINTER(PldOpts)],
        "smk_spec_pld_begin": [vp, C.POINTER(ci), ci, vp],
        "smk_spec_pld_prewarm": [vp],
        "smk_spec_run_pld": [vp, ci, vp, C.POINTER(ci), C.POINTER(ci)],
        "smk_spec_pld_stats": [vp, C.POINTER(C.c_longlong)],
        "smk_test_trials_pld": [vp, ci, vp, ci, vp, C.POINTER(Sampling), ci, C.POINTER(ci), ci, ci, cu, C.POINTER(ci), C.POINTER(ci)],
        "smk_spec_launch": [vp, ci, ci, ci, vp],
        "smk_spec_read": [vp, ci, vp, C.c_size_t],
        "smk_spec_set_sampling": [vp, C.POINTER(Sampling), vp],
        "smk_spec_sample_first": [vp, vp],
        "smk_spec_plain_run": [vp, ci, C.POINTER(ci), vp],
        "smk_spec_snapshot": [vp, vp],
        "smk_spec_restore": [vp, vp],
        "smk_test_draw": [vp, ci, cu, ci, C.POINTER(ci)],
        "smk_test_dist": [vp, ci, ci, vp, C.POINTER(Sampling), ci, C.POINTER(ci), C.POINTER(ci), C.POINTER(C.c_float)],
        "smk_test_trials": [vp, ci, vp, ci, vp, C.POINTER(Sampling), ci, ci, cu, C.POINTER(ci), C.POINTER(ci)],
        "smk_test_philox": [C.POINTER(cu), C.POINTER(cu), C.POINTER(cu)],
        "smk_engine_set_procs": [vp, C.POINTER(Procs), vp],
        "smk_test_trials_procs": [vp, ci, vp, ci, C.POINTER(ci), ci, C.POINTER(Sampling), C.POINTER(Procs), C.POINTER(ci), ci, ci, ci, cu, C.POINTER(ci), C.POINTER(ci)],
        "smk_test_proc_logits": [C.POINTER(Procs), ci, C.POINTER(ci), ci, C.POINTER(ci), vp, vp],
        "smk_test_proc_cand": [C.POINTER(Procs), ci, C.POINTER(ci), ci, C.POINTER(ci), vp, vp, vp],
    }
    for n, a in sig.items():
        f = getattr(lib, n)
        f.argtypes, f.restype = a, ci


class SpecEngine(Engine):
    def __init__(self, ctx_max=4096, kv_fp8=0, quant="int4clip", head_quant="int4", head_mode=0, nv=None, verbose=True, **kw):
        if head_mode == 1:  # the MTP draft head is the full BF16 lm_head: it must stay on the GPU
            kw["host_tables"] = False
        super().__init__(ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=verbose, **kw)
        setup_spec_api(self.lib)
        t0 = time.time()
        self.M = pack_mtp.load_mtp(self.G, quant=quant, head_quant=head_quant, nv=nv, verbose=verbose)
        M, p = self.M, smk.ptr
        for w4 in (M.fc, M.qkv, M.o, M.gu, M.down):  # engine layout (smk_w4r_gemv, pack.repack_w4), in place
            pack.repack_w4(w4.w, w4.s)
        M.head_r = pack_mtp.W4t(M.head.w.clone(), M.head.s.clone())  # the draft head is small: M.head stays natural (tests / references), the engine reads this copy
        pack.repack_w4(M.head_r.w, M.head_r.s)
        w = MtpW()
        d = w.layer
        d.kind = 1
        d.ln1, d.ln2, d.q_norm, d.k_norm = p(M.ln1), p(M.ln2), p(M.q_norm), p(M.k_norm)
        d.qkv_w, d.qkv_s, d.o_w, d.o_s = p(M.qkv.w), p(M.qkv.s), p(M.o.w), p(M.o.s)
        d.gu_w, d.gu_s, d.dn_w, d.dn_s = p(M.gu.w), p(M.gu.s), p(M.down.w), p(M.down.s)
        w.fc_w, w.fc_s, w.pre_e, w.pre_h, w.norm = p(M.fc.w), p(M.fc.s), p(M.pre_e), p(M.pre_h), p(M.norm)
        w.head_w, w.head_s, w.head_ids, w.head_nv = p(M.head_r.w), p(M.head_r.s), p(M.ids_t), M.nv
        self.mw = w
        self.sp = self.lib.smk_spec_create(self.h, C.byref(w), head_mode)
        if not self.sp:
            raise RuntimeError("smk_spec_create failed")
        self.pack_s = time.time() - t0
        self.head_mode = head_mode
        self._ss = (C.c_int * SS_N)()
        self._ring = (C.c_int * RING)()
        self._emitted = 0
        self.sampling = False
        self.pld = False  # K11 hybrid draft (enable_pld)
        torch.cuda.synchronize()

    # ---- state / prefill
    def reset(self):
        super().reset()
        smk.check(self.lib.smk_spec_reset(self.sp, self.st), "spec_reset")
        smk.check(self.lib.smk_spec_set_force(self.sp, None, 0, self.st), "set_force")
        self._emitted = 0

    def reset_counters(self):
        """Zero the device counters (philox cycle counter, emitted count, histogram) and the output ring but keep the MTP KV (prefix reuse); the engine state is not touched."""
        smk.check(self.lib.smk_spec_reset_counters(self.sp, self.st), "spec_reset_counters")
        self._emitted = 0

    def set_force(self, force):
        """Forced partial acceptance: cycle i accepts min(natural, force[i % len]) rows (None / [] disables)."""
        force = list(force or [])
        arr = (C.c_int * max(1, len(force)))(*force)
        smk.check(self.lib.smk_spec_set_force(self.sp, arr, len(force), self.st), "set_force")

    def prefill_spec(self, ids, k=3, chunk=4, start=True):
        """Prefill the main model + the MTP KV (pairs (h_p, ids[p+1])), then draft 1..k.  Returns the first generated token (the anchor).
        start=False stops before the drafting stage (tests: snapshot the state, then smk_spec_start)."""
        ids = list(ids)
        n = len(ids) - 1
        i = 0
        while i < n:
            T = min(chunk, n - i)
            self._step(ids[i : i + T], 2)
            nt = (C.c_int * T)(*ids[i + 1 : i + 1 + T])
            smk.check(self.lib.smk_spec_prefill_chunk(self.sp, T, nt, i, self.st), "spec_prefill_chunk")
            i += T
        self._step(ids[-1:], 1)
        if self.sampling:  # first generated token = a sample of the prefill's last row (instead of the argmax the engine step wrote)
            smk.check(self.lib.smk_spec_sample_first(self.sp, self.st), "spec_sample_first")
        if start:
            smk.check(self.lib.smk_spec_start(self.sp, k, self.st), "spec_start")
        self._emitted = 0
        return self.read_out_tok()

    # ---- sampling (K7): temperature > 0 switches cycles / start / plain decode to the sampling graphs; 0 = greedy
    def set_sampling(self, temperature=0.0, top_k=20, top_p=1.0, draft_sample=1, draft_temperature=0.0, draft_top_k=0, draft_top_p=0.0, seed=0, test_flags=0, min_p=0.0):
        cfg = make_sampling(temperature, top_k, top_p, draft_sample, draft_temperature, draft_top_k, draft_top_p, seed, test_flags, min_p)
        smk.check(self.lib.smk_spec_set_sampling(self.sp, C.byref(cfg), self.st), "set_sampling")
        self.sampling = temperature > 0

    def set_procs(self, **kw):
        """Request-level logit processors (make_procs arguments; none = off).  Call after set_sampling and before the prefill of every request; they act on sampling steps only
        (temperature > 0; a greedy request with processors is run as top_k = 1 sampling by the caller)."""
        p, keep = make_procs(**kw)
        smk.check(self.lib.smk_engine_set_procs(self.h, C.byref(p), self.st), "set_procs")  # pageable host arrays: copied before the call returns
        assert keep

    def proc_state(self):
        """(generated-token counts int32 [V], tokens generated so far) kept on the device by the request processors (only maintained while a processor is active)."""
        counts = self._read(3, 248320, torch.int32)
        head = self._read(4, 4, torch.int32)  # ProcP: on, has_allowed, min_tokens, ngen
        return counts, int(head[3])

    def plain_sample(self, n):
        """n plain (non-speculative) sampled decode steps from the staged next input token; returns the sampled tokens."""
        out = (C.c_int * n)()
        smk.check(self.lib.smk_spec_plain_run(self.sp, n, out, self.st), "plain_run")
        self.pos += n
        return list(out)

    def snapshot(self):
        smk.check(self.lib.smk_spec_snapshot(self.sp, self.st), "snapshot")

    def restore(self):
        smk.check(self.lib.smk_spec_restore(self.sp, self.st), "restore")

    def test_draw(self, n, ctr0=0, row=0):
        """n draws from the processed distribution of the engine's last logits row `row` (sampling config must be set)."""
        out = (C.c_int * n)()
        smk.check(self.lib.smk_test_draw(self.sp, n, ctr0, row, out), "test_draw")
        return torch.frombuffer(out, dtype=torch.int32).clone()

    def cycles(self, k, n):
        """n verify+draft cycles, returns (new tokens since the previous call, ss state ints)."""
        smk.check(self.lib.smk_spec_run(self.sp, k, n, self.st, self._ss, self._ring), "spec_run")
        outn = self._ss[12]
        assert outn - self._emitted <= RING, "output ring overrun"
        new = [self._ring[i % RING] for i in range(self._emitted, outn)]
        self._emitted = outn
        return new, list(self._ss)

    # ---- K11: MTP chain + prompt-lookup tail (needs SpecEngine(tmax=8))
    def enable_pld(self, opts=None):
        """opts: make_pld(...) (None = E6's best cell).  False / disable_pld() restores the plain k-draft cycles."""
        opts = opts or make_pld()
        assert self.tmax >= 5, "the prompt-lookup tail needs an engine with tmax = 8 (Engine(tmax=8) / env SMK_TMAX=8)"
        smk.check(self.lib.smk_spec_pld_config(self.sp, C.byref(opts)), "pld_config")
        smk.check(self.lib.smk_spec_pld_prewarm(self.sp), "pld_prewarm")  # capture the T = 4..8 cycle graphs now (no-op when already captured)
        self.pld = True

    def disable_pld(self):
        smk.check(self.lib.smk_spec_pld_config(self.sp, None), "pld_config")
        self.pld = False

    def pld_begin(self, ctx_ids):
        """After prefill_spec: ctx_ids = prompt ids + the first generated token (the anchor)."""
        arr = (C.c_int * len(ctx_ids))(*ctx_ids)
        smk.check(self.lib.smk_spec_pld_begin(self.sp, arr, len(ctx_ids), self.st), "pld_begin")

    def cycles_pld(self, n):
        """n hybrid cycles (one host round trip each); same return as cycles()."""
        smk.check(self.lib.smk_spec_run_pld(self.sp, n, self.st, self._ss, self._ring), "spec_run_pld")
        outn = self._ss[12]
        assert outn - self._emitted <= RING, "output ring overrun"
        new = [self._ring[i % RING] for i in range(self._emitted, outn)]
        self._emitted = outn
        return new, list(self._ss)

    def pld_stats(self):
        """dict: cycles by verify rows T (k = T - 1), tail tokens proposed, cycles with a tail, host wait / work ns."""
        out = (C.c_longlong * 16)()
        smk.check(self.lib.smk_spec_pld_stats(self.sp, out), "pld_stats")
        o = list(out)
        return dict(cycles_by_k={k: o[k] for k in range(1, 8) if o[k]}, tail_tokens=o[8], tail_cycles=o[9], wait_ns=o[10], work_ns=o[11])

    def generate(self, ids, n_new, k=3, group=8, eos=None, force=None):
        """Greedy speculative generation: returns (tokens incl. the first one: n_new tokens, or fewer ending with the first EOS token), stats dict).
        `group` cycles are launched per host sync (4..16 hides the latency)."""
        assert len(ids) + n_new + group * (8 if self.pld else k + 1) + 8 <= self.ctx_max, "ctx_max too small"
        self.reset()
        self.set_force(force)
        first = self.prefill_spec(ids, k)
        toks = [first]
        ss = [0] * SS_N
        if self.pld:
            assert k == 3
            self.pld_begin(list(ids) + [first])
        while len(toks) < n_new:
            new, ss = self.cycles_pld(group) if self.pld else self.cycles(k, group)
            toks += new
            if eos is not None and any(t in eos for t in new):
                break
        hist = ss[16:24] if self.pld else ss[16:20]
        cyc = ss[13]
        toks = toks[:n_new]
        if eos is not None:  # EOS stop is host-side: the cycles already launched may have produced extra tokens, discard them
            cut = next((i for i, t in enumerate(toks) if t in eos), None)
            if cut is not None:
                toks = toks[: cut + 1]
        stats = dict(cycles=cyc, hist=hist, tokens_per_cycle=ss[12] / max(cyc, 1), emitted=ss[12])
        if self.pld:
            stats["pld"] = self.pld_stats()
        return toks, stats

    def draft_state(self):
        """device ints: (pos, cur, anchor, d1, d2, d3)."""
        s = self._read(0, 18, torch.int32)
        return int(s[0]), int(s[1]), [int(x) for x in s[2:6]]

    def mtp_hidden(self):
        """bf16 [4][5120] post-norm MTP hidden of draft 1..3 (row j = draft j+1) of the last stage, fp32."""
        out = torch.empty(4 * 5120, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        smk.check(self.lib.smk_spec_read(self.sp, 1, C.c_void_p(out.data_ptr()), out.numel() * 2), "spec_read")
        return out.view(4, 5120).float()

    def launch_part(self, k, which, n):
        smk.check(self.lib.smk_spec_launch(self.sp, k, which, n, self.st), "spec_launch")
