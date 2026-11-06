"""DFlash2 block drafter for the qwen-warp engine: W4A16 checkpoint -> engine layouts, ctypes wrapper and driver (include/dflash.h, src/dflash.cu).

  from dflash import DflashEngine;  e = DflashEngine(ctx_max=2048, tmax=8, host_tables=True);  toks, st = e.generate_df(ids, 256)
The drafter checkpoint is incoai/Qwen3.8-27B-DFlash2 in compressed-tensors W4A16 (pack-quantized, group 128, fp16 scales -> bf16 here) or its bf16 original (quantised here with the MTP packer).
Greedy only.  Needs Engine(tmax=8) and the int4 head.
"""
import ctypes as C
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from engine import smk, pack  # noqa: F401
from spec import RING, SS_N, SpecEngine

DF_PATH = os.environ.get("DFLASH_W4", os.path.expanduser("~/llm-lab/models/dflash2-w4a16"))
LAYERS = [5, 19, 33, 47, 61]
NL, H, I, VOC = 5, 5120, 17408, 248320
DF_LOG, DF_LOGW = 4096, 18


class DfLayerW(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("in_ln", "post_ln", "qkv_w", "qkv_s", "q_norm", "k_norm", "o_w", "o_s", "gu_w", "gu_s", "dn_w", "dn_s", "aconv_kp", "mconv_kp", "aconv_base", "mconv_base")]


class DfW(C.Structure):
    _fields_ = [("layers", DfLayerW * NL)] + [(n, C.c_void_p) for n in ("fc_w", "fc_s", "hidden_norm", "norm", "kv_w", "kv_s", "sel_hp", "pred_cb", "succ_cb", "inv_freq")] + \
               [("mask_id", C.c_int), ("rescore_r", C.c_int)]


def setup_df_api(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_dflash_create.argtypes = [vp, C.POINTER(DfW)]
    lib.smk_dflash_create.restype = vp
    lib.smk_dflash_destroy.argtypes = [vp]
    for n in ("reset", "start"):
        getattr(lib, f"smk_dflash_{n}").argtypes = [vp, vp]
        getattr(lib, f"smk_dflash_{n}").restype = ci
    lib.smk_dflash_set_hist.argtypes = [vp, C.POINTER(ci), ci, vp]
    lib.smk_dflash_set_hist.restype = ci
    lib.smk_dflash_prefill_chunk.argtypes = [vp, ci, ci, vp]
    lib.smk_dflash_prefill_chunk.restype = ci
    lib.smk_dflash_run.argtypes = [vp, ci, vp, C.POINTER(ci), C.POINTER(ci)]
    lib.smk_dflash_run.restype = ci
    lib.smk_dflash_launch.argtypes = [vp, ci, ci, vp]
    lib.smk_dflash_launch.restype = ci
    lib.smk_dflash_read.argtypes = [vp, ci, vp, C.c_size_t]
    lib.smk_dflash_read.restype = ci
    lib.smk_dflash_bytes.argtypes = [vp]
    lib.smk_dflash_bytes.restype = C.c_size_t
    lib.smk_dflash_ctxd.argtypes = [vp]
    lib.smk_dflash_ctxd.restype = ci
    lib.smk_dflash_kv_ptr.argtypes = [vp, ci]
    lib.smk_dflash_kv_ptr.restype = vp


class DfPack:
    """Device tensors of the packed drafter (kept alive for the C engine)."""


def _w4(f, base, dev):
    """compressed-tensors pack-quantized Linear -> (int32 [N][K/8] natural layout, bf16 scale [N][K/128]) on the GPU.  fp16 scales are rounded to bf16 (the engine kernels take bf16)."""
    p = f.get_tensor(base + ".weight_packed").to(dev)
    s = f.get_tensor(base + ".weight_scale").to(dev).to(torch.bfloat16)
    return p, s


def _cat4(parts):
    return torch.cat([p for p, _ in parts]), torch.cat([s for _, s in parts])


def load_dflash(path=DF_PATH, dev="cuda", codebooks_host=False, verbose=True):
    """-> DfPack with engine-layout tensors (smk_w4r_gemv: pack.repack_w4)."""
    t0 = time.time()
    f = safe_open(str(Path(path) / "model.safetensors"), framework="pt", device="cpu")
    P = DfPack()
    P.keep = []
    bf = lambda n: f.get_tensor(n).to(dev)
    P.layers = []
    kv_parts = []
    for l in range(NL):
        a = f"layers.{l}."
        L = DfPack()
        L.in_ln, L.post_ln = bf(a + "input_layernorm.weight"), bf(a + "post_attention_layernorm.weight")
        L.q_norm, L.k_norm = bf(a + "self_attn.q_norm.weight"), bf(a + "self_attn.k_norm.weight")
        q, k, v = (_w4(f, a + f"self_attn.{n}_proj", dev) for n in "qkv")
        L.qkv = _cat4([q, k, v])
        kv_parts += [k, v]
        L.o = _w4(f, a + "self_attn.o_proj", dev)
        g, u = _w4(f, a + "mlp.gate_proj", dev), _w4(f, a + "mlp.up_proj", dev)
        n = g[0].shape[0]
        gw = torch.empty(2 * n, g[0].shape[1], dtype=torch.int32, device=dev)
        gs = torch.empty(2 * n, g[1].shape[1], dtype=torch.bfloat16, device=dev)
        for dst, x, y in ((gw, g[0], u[0]), (gs, g[1], u[1])):
            d = dst.view(n // 8, 2, 8, dst.shape[1])
            d[:, 0].copy_(x.view(n // 8, 8, -1))
            d[:, 1].copy_(y.view(n // 8, 8, -1))
        L.gu = (gw, gs)
        L.dn = _w4(f, a + "mlp.down_proj", dev)
        L.aconv_kp, L.mconv_kp = bf(a + "attention_conv.kernel_projection.weight"), bf(a + "mlp_conv.kernel_projection.weight")
        L.aconv_base, L.mconv_base = bf(a + "attention_conv.base_kernel").contiguous(), bf(a + "mlp_conv.base_kernel").contiguous()
        P.layers.append(L)
    P.kv = _cat4(kv_parts)  # per layer: k_proj (1024 rows) | v_proj (1024 rows)
    P.fc = _w4(f, "fc", dev)
    P.hidden_norm, P.norm = bf("hidden_norm.weight"), bf("norm.weight")
    P.sel_hp = bf("candidate_selector.hidden_projection.weight")
    cb = lambda n: (lambda t: (torch.empty(t.shape, dtype=t.dtype, pin_memory=True).copy_(t) if codebooks_host else t.to(dev)))(f.get_tensor(n))
    P.pred_cb, P.succ_cb = cb("candidate_selector.predecessor_codebook"), cb("candidate_selector.successor_codebook")
    P.inv_freq = (1.0 / (1e7 ** (torch.arange(0, 128, 2, dtype=torch.int64).float().to(dev) / 128))).contiguous()
    for L in P.layers:
        for w4 in (L.qkv, L.o, L.gu, L.dn):
            pack.repack_w4(*w4)
    pack.repack_w4(*P.kv)
    pack.repack_w4(*P.fc)
    torch.cuda.synchronize()
    nb = sum(t.numel() * t.element_size() for L in P.layers for w4 in (L.qkv, L.o, L.gu, L.dn) for t in w4) + sum(t.numel() * t.element_size() for t in (*P.kv, *P.fc))
    if verbose:
        print(f"DFlash2 packed (W4A16 -> engine layout): W4 {nb / 2**20:.0f} MiB, codebooks {'host' if codebooks_host else 'device'}, {time.time() - t0:.1f}s", flush=True)
    return P


def make_dfw(P, mask_id=248070, rescore_r=32):
    p = smk.ptr
    w = DfW()
    for i, L in enumerate(P.layers):
        d = w.layers[i]
        d.in_ln, d.post_ln, d.q_norm, d.k_norm = p(L.in_ln), p(L.post_ln), p(L.q_norm), p(L.k_norm)
        d.qkv_w, d.qkv_s, d.o_w, d.o_s = p(L.qkv[0]), p(L.qkv[1]), p(L.o[0]), p(L.o[1])
        d.gu_w, d.gu_s, d.dn_w, d.dn_s = p(L.gu[0]), p(L.gu[1]), p(L.dn[0]), p(L.dn[1])
        d.aconv_kp, d.mconv_kp, d.aconv_base, d.mconv_base = p(L.aconv_kp), p(L.mconv_kp), p(L.aconv_base), p(L.mconv_base)
    w.fc_w, w.fc_s, w.hidden_norm, w.norm = p(P.fc[0]), p(P.fc[1]), p(P.hidden_norm), p(P.norm)
    w.kv_w, w.kv_s, w.sel_hp, w.pred_cb, w.succ_cb, w.inv_freq = p(P.kv[0]), p(P.kv[1]), p(P.sel_hp), p(P.pred_cb), p(P.succ_cb), p(P.inv_freq)
    w.mask_id, w.rescore_r = mask_id, rescore_r
    return w


class DflashEngine(SpecEngine):
    """SpecEngine (MTP k=3 path intact) + the DFlash2 drafter.  tmax must be 8."""

    def __init__(self, dflash_path=DF_PATH, rescore_r=32, codebooks_host=False, mask_id=248070, **kw):
        retry = False
        try:
            self._build(dflash_path, rescore_r, codebooks_host, mask_id, **kw)
        except RuntimeError:
            # Engine keeps the bf16 lm_head table on the device when the free VRAM covers the engine's estimated needs; if the estimate was short (out of memory somewhere in the
            # construction), rebuild with the table in pinned host memory.
            if getattr(self, "lm_head_on_host", True) or not kw.get("host_tables", True):
                raise
            retry = True
        if retry:  # outside the handler: the traceback of the failed attempt would keep its tensors alive
            lib = getattr(self, "lib", None)
            for attr, fn in (("df", "smk_dflash_destroy"), ("sp", "smk_spec_destroy"), ("h", "smk_engine_destroy")):  # the C objects of the failed attempt (device memory)
                if lib is not None and getattr(self, attr, None):
                    getattr(lib, fn)(getattr(self, attr))
            for name in list(vars(self)):
                delattr(self, name)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self._build(dflash_path, rescore_r, codebooks_host, mask_id, lm_head_host=True, **kw)

    def _build(self, dflash_path, rescore_r, codebooks_host, mask_id, **kw):
        kw.setdefault("tmax", 8)
        kw.setdefault("lm_head_host", False)  # keep the bf16 lm_head table on the device when it fits (rescoring over PCIe costs ~0.2 ms per cycle)
        super().__init__(**kw)
        assert self.tmax == 8, "DFlash2 verifies 8 rows per cycle (Engine(tmax=8))"
        setup_df_api(self.lib)
        torch.cuda.empty_cache()
        free0 = torch.cuda.mem_get_info()[0]
        t0 = time.time()
        self.P = load_dflash(dflash_path, codebooks_host=codebooks_host, verbose=kw.get("verbose", True))
        self.dfw = make_dfw(self.P, mask_id, rescore_r)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # the weight repack leaves idle blocks in torch's pool: smk_dflash_create's cudaMallocs (K/V cache, 20 KiB per position) must see that memory
        self.df = self.lib.smk_dflash_create(self.h, C.byref(self.dfw))
        if not self.df:
            raise RuntimeError("smk_dflash_create failed (out of device memory, or the engine has no int4 head / tmax != 8)")
        self.df_load_s = time.time() - t0
        torch.cuda.empty_cache()
        self.df_mib = (free0 - torch.cuda.mem_get_info()[0]) / 2**20  # device memory of the drafter: weights, K/V cache, ring, scratch (not the graphs of the first cycles)
        self.ctxd = self.lib.smk_dflash_ctxd(self.df)
        self._dss = (C.c_int * SS_N)()
        self._dring = (C.c_int * RING)()
        self._demitted = 0
        torch.cuda.synchronize()

    def reset(self):
        super().reset()
        if getattr(self, "df", None):
            smk.check(self.lib.smk_dflash_reset(self.df, self.st), "dflash_reset")
        self._demitted = 0

    def df_prefill(self, ids):
        """Prefill the target in 8-row chunks, appending the taps as drafter context, then draft d1..d7 (device).  Returns the first generated token (the anchor)."""
        ids = list(ids)
        assert len(ids) + 64 < self.ctx_max
        self.reset()
        hbuf = (C.c_int * len(ids))(*ids)
        smk.check(self.lib.smk_dflash_set_hist(self.df, hbuf, len(ids), self.st), "dflash_set_hist")
        n, i = len(ids) - 1, 0
        while i < n:
            T = min(8, n - i)
            self._step(ids[i : i + T], 0)
            smk.check(self.lib.smk_dflash_prefill_chunk(self.df, T, i, self.st), "dflash_prefill_chunk")
            i += T
        self._step(ids[-1:], 1)
        smk.check(self.lib.smk_dflash_start(self.df, self.st), "dflash_start")
        self._demitted = 0
        return self.read_out_tok()

    def df_cycles(self, n):
        """n full DFlash2 cycles (verify T=8 + accept + drafter stage), one sync; returns (new tokens, ss ints)."""
        smk.check(self.lib.smk_dflash_run(self.df, n, self.st, self._dss, self._dring), "dflash_run")
        outn = self._dss[12]
        assert outn - self._demitted <= RING, "output ring overrun"
        new = [self._dring[i % RING] for i in range(self._demitted, outn)]
        self._demitted = outn
        return new, list(self._dss)

    def generate_df(self, ids, n_new, group=8, eos=None):
        """Greedy DFlash2 generation: (tokens, stats)."""
        assert len(ids) + n_new + group * 8 + 16 <= self.ctx_max, "ctx_max too small"
        first = self.df_prefill(ids)
        toks, ss = [first], [0] * SS_N
        while len(toks) < n_new:
            new, ss = self.df_cycles(group)
            toks += new
            if eos is not None and any(t in eos for t in new):
                break
        toks = toks[:n_new]
        if eos is not None:
            cut = next((i for i, t in enumerate(toks) if t in eos), None)
            if cut is not None:
                toks = toks[: cut + 1]
        cyc = ss[13]
        return toks, dict(cycles=cyc, hist=ss[16:24], tokens_per_cycle=ss[12] / max(cyc, 1), emitted=ss[12])

    def df_launch(self, which, n):
        smk.check(self.lib.smk_dflash_launch(self.df, which, n, self.st), "dflash_launch")

    def df_log(self, ncyc):
        """per-cycle log of the first ncyc cycles (<= 4096): int32 [ncyc][18] = a, block[8] (anchor, d1..d7), y[8] (verify argmax), anchor position."""
        out = torch.empty(DF_LOG * DF_LOGW, dtype=torch.int32)
        torch.cuda.synchronize()
        smk.check(self.lib.smk_dflash_read(self.df, 2, C.c_void_p(out.data_ptr()), out.numel() * 4), "dflash_read")
        return out.view(DF_LOG, DF_LOGW)[:ncyc]

    def df_read(self, what, shape, dtype):
        out = torch.empty(shape, dtype=dtype)
        torch.cuda.synchronize()
        smk.check(self.lib.smk_dflash_read(self.df, what, C.c_void_p(out.data_ptr()), out.numel() * out.element_size()), "dflash_read")
        return out

    def df_start(self):
        """Append the tap of the last prompt row (dbg row 0 of the head=1 step / folded head) as drafter context and draft d1..d7: the DFlash2 counterpart of smk_spec_start."""
        smk.check(self.lib.smk_dflash_start(self.df, self.st), "dflash_start")

    def df_set_hist(self, ids):
        """Token history by position for the device n-gram lookup (SMK_DF_LOOKUP=1): the prompt ids, before df_start."""
        buf = (C.c_int * len(ids))(*ids)
        smk.check(self.lib.smk_dflash_set_hist(self.df, buf, len(ids), self.st), "dflash_set_hist")

    def df_append(self, T, pos0):
        """Append dbg rows 0..T-1 (T <= 8: taps of prompt rows pos0 .. pos0 + T - 1) as drafter context (no drafting)."""
        smk.check(self.lib.smk_dflash_prefill_chunk(self.df, T, pos0, self.st), "dflash_prefill_chunk")

    def df_kv(self, which):
        """The drafter's device K (which 0) / V (1) cache as a bf16 tensor [5 layers][8 kv heads][ctxd positions][128]."""
        from prefill import _view

        return _view(self.lib.smk_dflash_kv_ptr(self.df, which), 5 * 8 * self.ctxd * 128 * 2, torch.bfloat16).view(5, 8, self.ctxd, 128)

    def df_window_save(self, lo, hi):
        """Drafter K and V rows [lo, hi) -> pinned host tensors [5][8][hi - lo][128] (stream ordered: synchronise before reading them)."""
        out = []
        for w in (0, 1):
            t = torch.empty(5, 8, hi - lo, 128, dtype=torch.bfloat16, pin_memory=True)
            t.copy_(self.df_kv(w)[:, :, lo:hi], non_blocking=True)
            out.append(t)
        return tuple(out)

    def df_window_load(self, lo, kv):
        """The inverse of df_window_save: write the rows back at positions lo.. (stream ordered)."""
        for w, t in zip((0, 1), kv):
            self.df_kv(w)[:, :, lo : lo + t.shape[2]].copy_(t, non_blocking=True)

    def df_state(self):
        """The drafter's device state ints (ss[12] = tokens emitted, ss[13] = cycles, ss[16 + m] = cycles with m accepted drafts); counters run on across requests that resume a cached lineage."""
        return [int(x) for x in self.df_read(0, SS_N, torch.int32)]


TAP_SLOTS = tuple(6 + 14 * s for s in range(5))  # engine dbg slots (= target layer index + 1) of the drafter's taps: the residual stream after layers 5, 19, 33, 47, 61


class TapCapture:
    """Drafter context of prompt rows that the fast prefill computes: stages the residual rows entering target layers 6/20/34/48/62 of the window [lo, hi) while FastPrefill runs
    (FastPrefill.tap = this object, it only READS the residual), then `flush` appends them to the drafter's K/V cache in 8-row chunks through the engine's debug taps
    (smk_dflash_prefill_chunk, the same kernel path as DflashEngine.df_prefill).  Layer-major prefill produces all rows of one layer before the next layer, so the rows of a group wait
    here (5 x rows x 10 KiB) until its last layer is done.  `last` holds the last prompt row of a folded head segment (it has no head step that fills dbg row 0)."""

    def __init__(self, eng, rows):
        from prefill import _view, setup_prefill_api

        setup_prefill_api(eng.lib)  # smkp_engine_ptr: argtypes / restype (FastPrefill sets them too, but not before this constructor)

        self.e, self.rows = eng, rows
        self.buf = torch.empty(5, rows, H, dtype=torch.bfloat16, device="cuda")
        self.last = torch.empty(5, H, dtype=torch.bfloat16, device="cuda")
        self.dbg = _view(eng.lib.smkp_engine_ptr(eng.h, 7), 66 * 8 * H * 2, torch.bfloat16).view(66, 8, H)[TAP_SLOTS[0] : TAP_SLOTS[-1] + 1 : 14]  # [5][8][H] strided view of the 5 slots
        self.lo = self.hi = 0
        self.nbytes = (self.buf.numel() + self.last.numel()) * 2

    def begin(self, lo, hi):
        """Stage prompt rows [lo, hi) (absolute positions, hi - lo <= rows)."""
        assert 0 <= hi - lo <= self.rows
        self.lo, self.hi = lo, hi

    def take(self, k, h, pos0):
        """h = residual rows [T][H] of a segment at positions pos0.., entering the k-th tap layer."""
        a, b = max(self.lo, pos0), min(self.hi, pos0 + h.shape[0])
        if a < b:
            self.buf[k, a - self.lo : b - self.lo].copy_(h[a - pos0 : b - pos0])

    def take_last(self, k, row):
        self.last[k].copy_(row)

    def flush(self, a, b):
        """Append the staged rows [a, b) as drafter context (8 rows per chunk)."""
        for c in range(a, b, 8):
            T = min(8, b - c)
            self.dbg[:, :T].copy_(self.buf[:, c - self.lo : c - self.lo + T])
            self.e.df_append(T, c)

    def put_last(self):
        """The staged last prompt row -> dbg row 0 of the 5 taps (what the T = 1 head step would have written), ready for df_start / df_append(1, n - 1)."""
        self.dbg[:, 0].copy_(self.last)
