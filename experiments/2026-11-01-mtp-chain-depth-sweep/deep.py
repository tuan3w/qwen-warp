"""Python side of the deep-chain experiment: ctypes wrapper for the private library's `smk_spec_*deep*` entry points (lib/deep.patch) on top of the committed spec.py.

  DeepEngine(SpecEngine): deep_begin / cycles_deep / deep_log / deep_parts.   Needs SMK_BUILD=<abs path of runs/lib/build> (lib/build.sh) and an engine with tmax = 8.
"""
import ctypes as C
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parents[1] / "engine"
os.environ.setdefault("SMK_BUILD", str(HERE / "runs" / "lib" / "build"))
sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(ENGINE / "tests"))

import torch  # noqa: E402
from spec import SpecEngine, make_pld, smk, SS_N, RING  # noqa: E402,F401
from prefill import FastPrefill  # noqa: E402,F401

KC = 7


class DeepOpts(C.Structure):
    _fields_ = [("kc", C.c_int), ("kc_lo", C.c_int), ("adapt_thr", C.c_int), ("gate", C.c_int), ("use_q", C.c_int), ("tail", C.c_int), ("tail_nch", C.c_int), ("fixed_kv", C.c_int),
                ("tau", C.c_float * 8), ("adapt_rho", C.c_float), ("adapt_mode", C.c_int), ("reserved", C.c_int * 6)]


def make_opts(kc=3, kc_lo=None, adapt_thr=0, gate=0, use_q=0, tail=0, tail_nch=3, fixed_kv=0, tau=(), adapt_rho=0.0, adapt_mode=0):
    t = list(tau) + [0.0] * (8 - len(tau))
    return DeepOpts(kc, kc if kc_lo is None else kc_lo, adapt_thr, gate, use_q, tail, tail_nch, fixed_kv, (C.c_float * 8)(*t[:8]), adapt_rho, adapt_mode, (C.c_int * 6)())


class DeepEngine(SpecEngine):
    def __init__(self, *a, **kw):
        kw.setdefault("tmax", 8)
        super().__init__(*a, **kw)
        lib, vp, ci = self.lib, C.c_void_p, C.c_int
        sig = {
            "smk_spec_deep_launch": ([vp, ci, ci, ci, ci, vp], ci),
            "smk_spec_deep_prewarm": ([vp, ci, ci, ci], ci),
            "smk_spec_deep_begin": ([vp, C.POINTER(ci), ci, ci, vp], ci),
            "smk_spec_run_deep": ([vp, C.POINTER(DeepOpts), ci, vp, C.POINTER(ci), C.POINTER(ci)], ci),
            "smk_spec_deep_log": ([vp, C.POINTER(ci), C.POINTER(C.c_float), ci, ci], ci),
            "smk_spec_deep_stats": ([vp, C.POINTER(C.c_longlong)], ci),
        }
        for n, (a_, r) in sig.items():
            f = getattr(lib, n)
            f.argtypes, f.restype = a_, r
        self._tail_cfg = None

    # ---- proposer (tail): cfg = make_pld(...) or None (no tail at all)
    def set_tail(self, cfg):
        if cfg is None:
            smk.check(self.lib.smk_spec_pld_config(self.sp, None), "pld_config")
        else:
            smk.check(self.lib.smk_spec_pld_config(self.sp, C.byref(cfg)), "pld_config")

    def prewarm(self, kc_lo=3, kc_hi=7, kv_max=7):
        smk.check(self.lib.smk_spec_deep_prewarm(self.sp, kc_lo, kc_hi, kv_max), "deep_prewarm")

    def deep_begin(self, ctx_ids, kc):
        """After FastPrefill.prefill_spec(ids, k, start=False): ctx_ids = prompt + the first generated token."""
        arr = (C.c_int * len(ctx_ids))(*ctx_ids)
        smk.check(self.lib.smk_spec_deep_begin(self.sp, arr, len(ctx_ids), kc, self.st), "deep_begin")
        self._emitted = 0

    def cycles_deep(self, n, opts):
        smk.check(self.lib.smk_spec_run_deep(self.sp, C.byref(opts), n, self.st, self._ss, self._ring), "run_deep")
        outn = self._ss[12]
        assert outn - self._emitted <= RING, "output ring overrun"
        new = [self._ring[i % RING] for i in range(self._emitted, outn)]
        self._emitted = outn
        return new, list(self._ss)

    def deep_log(self, cap=65536, clear=True):
        """-> (ints [n][8] = a, nch, ln, nd, kc_next, tail_prio, 0, 0; floats [n][16] = conf of the drafts the cycle used: raw max-prob 0..7, processed-q max-prob 8..15)"""
        oi = (C.c_int * (cap * 8))()
        of = (C.c_float * (cap * 16))()
        n = self.lib.smk_spec_deep_log(self.sp, oi, of, cap, 1 if clear else 0)
        import numpy as np

        return np.frombuffer(oi, dtype=np.int32)[: n * 8].reshape(n, 8).copy(), np.frombuffer(of, dtype=np.float32)[: n * 16].reshape(n, 16).copy()

    def deep_stats(self):
        o = (C.c_longlong * 2)()
        smk.check(self.lib.smk_spec_deep_stats(self.sp, o), "deep_stats")
        return dict(wait_ns=o[0], work_ns=o[1])

    def deep_launch(self, kc, kv, which, n):
        smk.check(self.lib.smk_spec_deep_launch(self.sp, kc, kv, which, n, self.st), "deep_launch")

    def hist(self):
        return list(self._ss[16:24])
