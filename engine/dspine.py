"""DSpine block drafter for the qwen-warp engine: bf16 checkpoint -> W4A16 engine layouts, ctypes wrapper and driver (include/dspine.h, src/dspine.cu)."""

import ctypes as C
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

import pack_mtp
from engine import smk, pack  # noqa: F401
from spec import RING, SS_N, SpecEngine

SP_PATH = os.environ.get(
    "DSPINE_PATH", os.path.expanduser("~/llm-lab/models/dspine-b16")
)
LAYERS = [1, 16, 31, 46, 61]
NL, H, I, VOC = 5, 5120, 17408, 248320
SP_LOG, SP_LOGW = 4096, 18
SP_R, SP_C, SP_K = 512, 1024, 16


class SpLayerW(C.Structure):
    _fields_ = [
        (n, C.c_void_p)
        for n in (
            "in_ln",
            "post_ln",
            "qkv_w",
            "qkv_s",
            "q_norm",
            "k_norm",
            "o_w",
            "o_s",
            "gu_w",
            "gu_s",
            "dn_w",
            "dn_s",
            "write_w",
        )
    ]


class SpW(C.Structure):
    _fields_ = (
        [("layers", SpLayerW * NL)]
        + [
            (n, C.c_void_p)
            for n in (
                "fc_w",
                "fc_s",
                "hidden_norm",
                "norm",
                "kv_w",
                "kv_s",
                "codebook",
                "read_w_last",
                "early_read",
                "gate_w",
                "gate_b",
                "inv_freq",
            )
        ]
        + [("mask_id", C.c_int)]
    )


def setup_sp_api(lib):
    vp, ci = C.c_void_p, C.c_int
    lib.smk_dspine_create.argtypes = [vp, C.POINTER(SpW)]
    lib.smk_dspine_create.restype = vp
    lib.smk_dspine_destroy.argtypes = [vp]
    for n in ("reset", "start"):
        getattr(lib, f"smk_dspine_{n}").argtypes = [vp, vp]
        getattr(lib, f"smk_dspine_{n}").restype = ci
    lib.smk_dspine_prefill_chunk.argtypes = [vp, ci, ci, vp]
    lib.smk_dspine_prefill_chunk.restype = ci
    lib.smk_dspine_run.argtypes = [vp, ci, vp, C.POINTER(ci), C.POINTER(ci)]
    lib.smk_dspine_run.restype = ci
    lib.smk_dspine_launch.argtypes = [vp, ci, ci, vp]
    lib.smk_dspine_launch.restype = ci
    lib.smk_dspine_read.argtypes = [vp, ci, vp, C.c_size_t]
    lib.smk_dspine_read.restype = ci
    lib.smk_dspine_bytes.argtypes = [vp]
    lib.smk_dspine_bytes.restype = C.c_size_t
    lib.smk_dspine_ctxd.argtypes = [vp]
    lib.smk_dspine_ctxd.restype = ci
    lib.smk_dspine_kv_ptr.argtypes = [vp, ci]
    lib.smk_dspine_kv_ptr.restype = vp


class SpPack:
    """Device tensors of the packed drafter (kept alive for the C engine)."""


def _q4(w, clip=True):
    """bf16 [N, K] (CPU or GPU) -> (int32 [N, K/8], bf16 scale [N, K/128]) on the GPU: symmetric g128, per-group MSE-optimal clip (pack_mtp.quant_w4)."""
    return pack_mtp.quant_w4(w, clip)


def load_dspine(path=SP_PATH, dev="cuda", verbose=True, clip=True):
    """-> SpPack: the big projections (qkv, o, gate|up, down, fc, kv) quantised to W4A16 in the smk_w4r_gemv layout, the small bridge tensors in bf16."""
    t0 = time.time()
    f = safe_open(str(Path(path) / "model.safetensors"), framework="pt", device="cpu")
    P = SpPack()
    P.keep = []
    cpu = lambda n: f.get_tensor(n).to(torch.bfloat16)
    bf = lambda n: cpu(n).to(dev).contiguous()
    wfw = f.get_tensor("selected_prefix_bridge.feature_write_weights").to(
        torch.bfloat16
    )
    P.layers = []
    kv_parts = []
    for l in range(NL):
        a = f"layers.{l}."
        L = SpPack()
        L.in_ln = bf(a + "input_layernorm.weight")
        L.post_ln = bf(a + "post_attention_layernorm.weight")
        L.q_norm = bf(a + "self_attn.q_norm.weight")
        L.k_norm = bf(a + "self_attn.k_norm.weight")
        q, k, v = (cpu(a + f"self_attn.{n}_proj.weight") for n in "qkv")
        L.qkv = _q4(torch.cat([q, k, v], 0), clip)
        kv_parts += [k, v]
        L.o = _q4(cpu(a + "self_attn.o_proj.weight"), clip)
        g, u = (
            _q4(cpu(a + "mlp.gate_proj.weight"), clip),
            _q4(cpu(a + "mlp.up_proj.weight"), clip),
        )
        n = g[0].shape[0]
        gw = torch.empty(2 * n, g[0].shape[1], dtype=torch.int32, device=dev)
        gs = torch.empty(2 * n, g[1].shape[1], dtype=torch.bfloat16, device=dev)
        for dst, x, y in (
            (gw, g[0], u[0]),
            (gs, g[1], u[1]),
        ):  # gate / up interleaved in groups of 8 rows (SILU epilogue)
            d = dst.view(n // 8, 2, 8, dst.shape[1])
            d[:, 0].copy_(x.view(n // 8, 8, -1))
            d[:, 1].copy_(y.view(n // 8, 8, -1))
        L.gu = (gw, gs)
        del g, u
        L.dn = _q4(cpu(a + "mlp.down_proj.weight"), clip)
        L.write_w = wfw[l].to(dev).contiguous()
        for w4 in (L.qkv, L.o, L.gu, L.dn):
            pack.repack_w4(*w4)
            P.keep += list(w4)
        P.layers.append(L)
        P.keep += [L.in_ln, L.post_ln, L.q_norm, L.k_norm, L.write_w]
    P.kv = _q4(
        torch.cat(kv_parts, 0), clip
    )  # per layer: k_proj (1024 rows) | v_proj (1024 rows)
    P.fc = _q4(cpu("fc.weight"), clip)
    pack.repack_w4(*P.kv)
    pack.repack_w4(*P.fc)
    P.hidden_norm = bf("hidden_norm.weight")
    P.norm = bf("norm.weight")
    P.codebook = bf("selected_prefix_bridge.codebook")
    P.read_w_last = bf("selected_prefix_bridge.read_weight")
    P.early_read = bf("selected_prefix_bridge.early_read_weights")
    P.gate_w = torch.cat(
        [
            bf("selected_prefix_bridge.gate.weight"),
            bf("selected_prefix_bridge.predecessor_gate.weight"),
            bf("selected_prefix_bridge.prev.weight"),
        ],
        0,
    ).contiguous()
    P.gate_b = bf("selected_prefix_bridge.gate.bias").float()
    P.inv_freq = (
        1.0
        / (1e7 ** (torch.arange(0, 128, 2, dtype=torch.int64).float().to(dev) / 128))
    ).contiguous()
    P.keep += [
        *P.kv,
        *P.fc,
        P.hidden_norm,
        P.norm,
        P.codebook,
        P.read_w_last,
        P.early_read,
        P.gate_w,
        P.gate_b,
        P.inv_freq,
    ]
    torch.cuda.synchronize()
    nb = sum(t.numel() * t.element_size() for t in P.keep)
    if verbose:
        print(
            f"DSpine packed (W4A16): {nb / 2**20:.0f} MiB, {time.time() - t0:.1f}s",
            flush=True,
        )
    return P


def make_spw(P, mask_id=248070):
    p = smk.ptr
    w = SpW()
    for i, L in enumerate(P.layers):
        d = w.layers[i]
        d.in_ln, d.post_ln, d.q_norm, d.k_norm = (
            p(L.in_ln),
            p(L.post_ln),
            p(L.q_norm),
            p(L.k_norm),
        )
        d.qkv_w, d.qkv_s, d.o_w, d.o_s = p(L.qkv[0]), p(L.qkv[1]), p(L.o[0]), p(L.o[1])
        d.gu_w, d.gu_s, d.dn_w, d.dn_s = p(L.gu[0]), p(L.gu[1]), p(L.dn[0]), p(L.dn[1])
        d.write_w = p(L.write_w)
    w.fc_w, w.fc_s, w.hidden_norm, w.norm = (
        p(P.fc[0]),
        p(P.fc[1]),
        p(P.hidden_norm),
        p(P.norm),
    )
    w.kv_w, w.kv_s, w.codebook, w.read_w_last, w.early_read = (
        p(P.kv[0]),
        p(P.kv[1]),
        p(P.codebook),
        p(P.read_w_last),
        p(P.early_read),
    )
    w.gate_w, w.gate_b = p(P.gate_w), p(P.gate_b)
    w.inv_freq = p(P.inv_freq)
    w.mask_id = mask_id
    return w


class DspineEngine(SpecEngine):
    """SpecEngine + the DSpine drafter.  tmax must be 8."""

    def __init__(self, dspine_path=SP_PATH, **kw):
        retry = False
        try:
            self._build(dspine_path, **kw)
        except Exception:
            if not retry:
                retry = True
                torch.cuda.empty_cache()
                self._build(dspine_path, **kw)
            else:
                raise

    def _build(self, dspine_path, **kw):
        kw.setdefault("tmax", 8)
        P = load_dspine(dspine_path, verbose=True)
        self.sp_pack = P
        w = make_spw(P)
        super().__init__(host_tables=True, **kw)
        self._dss = (C.c_int * SS_N)()
        self._dring = (C.c_int * RING)()
        self._demitted = 0
        setup_sp_api(self.lib)
        self.sp_w = w
        self.dsp = self.lib.smk_dspine_create(self.h, C.byref(w))
        if not self.dsp:
            raise RuntimeError("smk_dspine_create failed")
        self.lib.smk_dspine_reset(self.dsp, self.st)
        self.sp_bytes = self.lib.smk_dspine_bytes(self.dsp)
        self.sp_ctxd = self.lib.smk_dspine_ctxd(self.dsp)
        print(
            f"DSpine: {self.sp_bytes / 2**20:.0f} MiB, ctxd={self.sp_ctxd}", flush=True
        )

    def sp_read(self, what, n, dtype):
        t = torch.empty(n, dtype=dtype)
        smk.check(
            self.lib.smk_dspine_read(
                self.dsp, what, t.data_ptr(), t.numel() * t.element_size()
            ),
            f"sp_read{what}",
        )
        return t

    def reset(self):
        super().reset()
        if getattr(self, "dsp", None):
            smk.check(self.lib.smk_dspine_reset(self.dsp, self.st), "dspine_reset")
        self._demitted = 0

    def sp_prefill_chunk(self, T, pos0):
        smk.check(
            self.lib.smk_dspine_prefill_chunk(self.dsp, T, pos0, self.st),
            "sp_prefill_chunk",
        )

    def sp_launch(self, which, n):
        smk.check(self.lib.smk_dspine_launch(self.dsp, which, n, self.st), "sp_launch")

    def sp_prefill(self, ids):
        """Prefill the target in 8-row chunks, appending the taps as drafter context, then draft d1..d7 (device).  Returns the first generated token (the anchor)."""
        ids = list(ids)
        assert len(ids) + 64 < self.ctx_max
        self.reset()
        n, i = len(ids) - 1, 0
        while i < n:
            T = min(8, n - i)
            self._step(ids[i : i + T], 0)
            self.sp_prefill_chunk(T, i)
            i += T
        self._step(ids[-1:], 1)
        smk.check(self.lib.smk_dspine_start(self.dsp, self.st), "dspine_start")
        self._demitted = 0
        return self.read_out_tok()

    def sp_cycles(self, n):
        """n full cycles (verify T=8 + accept + drafter stage), one sync; returns (new tokens, ss ints)."""
        smk.check(
            self.lib.smk_dspine_run(self.dsp, n, self.st, self._dss, self._dring),
            "dspine_run",
        )
        outn = self._dss[12]
        assert outn - self._demitted <= RING, "output ring overrun"
        new = [self._dring[i % RING] for i in range(self._demitted, outn)]
        self._demitted = outn
        return new, list(self._dss)

    def generate_sp(self, ids, n_new, group=8, eos=None):
        """Greedy DSpine generation: (tokens, stats)."""
        assert len(ids) + n_new + group * 8 + 16 <= self.ctx_max, "ctx_max too small"
        first = self.sp_prefill(ids)
        toks, ss = [first], [0] * SS_N
        while len(toks) < n_new:
            new, ss = self.sp_cycles(group)
            toks += new
            if eos is not None and any(t in eos for t in new):
                break
        toks = toks[:n_new]
        if eos is not None:
            cut = next((i for i, t in enumerate(toks) if t in eos), None)
            if cut is not None:
                toks = toks[: cut + 1]
        cyc = ss[13]
        return toks, dict(
            cycles=cyc,
            hist=ss[16:24],
            tokens_per_cycle=ss[12] / max(cyc, 1),
            emitted=ss[12],
        )

    def sp_hid(self):
        return self.sp_read(1, 8 * H, torch.bfloat16).view(8, H)

    def __del__(self):
        if getattr(self, "dsp", None):
            self.lib.smk_dspine_destroy(self.dsp)
            self.dsp = None


TAP_SLOTS = tuple(l + 1 for l in LAYERS)  # engine dbg slots of the drafter's taps
