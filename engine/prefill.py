"""swift-mk fast prompt prefill (K6): layer-major, whole-segment prefill on top of the engine's weights and state buffers.

  from prefill import FastPrefill
  fp = FastPrefill(engine)                  # Engine or SpecEngine (SpecEngine: the MTP KV cache is filled as well)
  first = fp.prefill(ids)                   # == Engine.prefill(ids): returns the first generated token, engine state ready for decode()
  first = fp.prefill_spec(ids, k=3)         # == SpecEngine.prefill_spec(ids, k): MTP KV + drafts, ready for cycles()

Design (numbers: docs/PERFORMANCE.md):
  * Per layer, W4 weights are dequantised ((nibble-8)*bf16 scale -> bf16, src/prefill.cu) into a 0.5 GB scratch and multiplied with the activations of a
    whole segment (<= `seg` tokens) by cuBLAS (torch.mm, fp32 accumulate, bf16 out).  The MLP runs in row blocks of `rows` tokens (dequant once per layer).
  * DeltaNet layers: conv1d+SiLU kernel (src/prefill.cu, same maths as the decode kernel, 3-token conv state in/out of ring slot `cur`), then fla's chunked
    gated delta rule (`chunk_gated_delta_rule`, GVA 16 q/k heads -> 48 v heads, state_v_first = the engine's [V][K] layout) carries the fp32 state in/out of slot `cur`.
  * Attention layers: q/k RMSNorm + rope + KV-cache write (bf16 or fp8 e4m3 with the checkpoint scales) in the engine layout (src/prefill.cu), then
    FlashAttention-2 (vllm_flash_attn) over token-major bf16 K/V (segments after the first read the history back from the cache), * sigmoid(gate).
  * Segments (> `seg` tokens): state carried exactly as across decode steps (DeltaNet state + conv state in slot cur, KV cache positions).
  * MTP (SpecEngine): the final-norm hidden of every prompt position + the next prompt token go through the int4 MTP fc / K,V projections; K,V (k_norm + rope) are
    written positionally into the MTP KV cache, exactly what SpecEngine.prefill_spec's smk_spec_prefill_chunk loop does.
  * The last prompt token is run as a normal decode step (head=1): logits, first token, device scalars, and smk_spec_start work unchanged.
Needs an RTX 3090 (sm_86; no other GPU has been tested).
"""
import ctypes as C
import time

import torch

from engine import Engine, smk, pack  # noqa: F401
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

H, I = pack.H, pack.I
EPS = 1e-6
TAP_LI = {6: 0, 20: 1, 34: 2, 48: 3, 62: 4}  # target layers whose INPUT residual is a DFlash2 drafter tap (the residual after layers 5, 19, 33, 47, 61) -> index in the tap staging buffer


class _Buf:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False), "version": 3}


def _view(ptr, nbytes, dtype):
    return torch.as_tensor(_Buf(ptr, nbytes), device="cuda").view(dtype)


def _ptr(t):
    return None if t is None else t.data_ptr()


class MMRows:
    """Image inputs of ONE prompt for the fast prefill (FastPrefill.mm):
      pos3  int32 [n][3] device tensor, the MRoPE (T, H, W) positions of every prompt row (engine/vision.py mrope_positions, transposed);
      spans [(start, end, emb)]: placeholder rows start..end-1 of an image take the rows of emb ([end - start][5120] bf16, contiguous, the ViT output) instead of the token embedding;
            emb may be None for an image that lies entirely before the resume position (its rows are never prefilled)."""

    def __init__(self, pos3, spans):
        self.pos3, self.spans = pos3, spans
        self.end = max((t_end for _, t_end, _ in spans), default=0)  # rows before this may be image rows or text between images (their rope is not KV position + the final delta)
        self._ar = None

    def arange(self, n):
        """Device pointer of int32 [0, 1, .., n - 1] (row indices into an emb block for smkp_mtp_cat)."""
        if self._ar is None or self._ar.numel() < n:
            self._ar = torch.arange(max(n, 4096), dtype=torch.int32, device="cuda")
        return self._ar.data_ptr()


def setup_prefill_api(lib):
    vp, ci, cf = C.c_void_p, C.c_int, C.c_float
    sig = {
        "smkp_dequant_w4r": [vp, vp, vp, ci, ci, ci, vp],
        "smkp_conv": [vp, ci, vp, vp, vp, vp, vp, vp, ci, vp],
        "smkp_gates": [vp, vp, vp, vp, vp, ci, vp],
        "smkp_gnorm": [vp, vp, ci, vp, vp, ci, vp],
        "smkp_attn_prep": [ci, ci, vp, ci, vp, ci, vp, ci, vp, vp, vp, ci, ci, cf, cf, vp, vp, vp, vp, vp],
        "smkp_gather_kv": [vp, ci, ci, ci, cf, cf, vp, vp, vp],
        "smkp_attn_gate": [vp, vp, ci, vp, ci, vp],
        "smkp_silu_mul": [vp, vp, ci, vp],
        "smkp_mtp_cat": [vp, vp, vp, vp, vp, vp, ci, vp],
        "smkp_w4r_gemm": [vp, vp, vp, ci, ci, ci, C.POINTER(smk.Epi), vp],
        "smkp_dn_row": [vp, ci, vp, vp, vp, vp, vp, vp, vp, vp],
        "smkp_head_finish": [vp, ci, vp],
        "smkp_materialize": [vp, vp],
        "smkp_ds_n": [],
        "smk_engine_head_tail": [vp, ci, vp],
    }
    for n, a in sig.items():
        f = getattr(lib, n)
        f.argtypes, f.restype = a, ci
    lib.smkp_engine_ptr.argtypes = [vp, ci]
    lib.smkp_engine_ptr.restype = vp
    lib.smk_spec_kv_ptr.argtypes = [vp]
    lib.smk_spec_kv_ptr.restype = vp


class FastPrefill:
    DS_RDELTA = 20  # engine_impl.h: the sequence's MRoPE rope offset (decode ropes at KV position + ds[DS_RDELTA]); the Runner sets it per request

    def __init__(self, eng, seg=4096, rows=2048, min_fast=2, w4_max=240, fold=True):
        self.e, self.lib = eng, eng.lib
        setup_prefill_api(self.lib)
        self.seg, self.rows = seg, rows
        self.min_fast = min_fast  # prompts with fewer prefill tokens than this use the engine's own prefill-by-decode
        self.fold = fold  # K10: the last prompt row rides in the final short batch (see _run)
        self.w4_max = w4_max  # segments of <= w4_max tokens run their GEMMs through the W4 mma kernel (src/prefill_w4.cu, 64-row chunks) instead of dequant + cuBLAS
        self._small = False
        self._fold = False
        self.taps = False  # test hook: write the last row's per-layer residual into the engine debug taps (a folded head has no step that does)
        self.tap = None  # DFlash2 drafter context capture (dflash.TapCapture): reads the residual entering the tap layers of every segment / group; never writes the prefill's own buffers
        self.layer_hook = None  # called with the segment's (group's) row count after every layer of a fast segment: the batched server decodes the other sequences there (a long prefill does not freeze them)
        self.mm = None  # image inputs of the prompt being prefilled (MMRows): MRoPE positions + ViT rows of the placeholder tokens; None = text (the v0.7.0 path, unchanged)
        self.ctx_max, self.fp8 = eng.ctx_max, eng.kv_fp8
        e, lib = eng.h, self.lib
        n_ds = lib.smkp_ds_n()  # engine_impl.h DS_N (a library older than the sources lacks the symbol: AttributeError); DS_PEND = 19 pending lazy DeltaNet rows
        assert n_ds == self.DS_RDELTA + 1, n_ds
        self.ds = _view(lib.smkp_engine_ptr(e, 0), n_ds * 4, torch.int32)
        nsl = lib.smk_engine_sslots(e)  # physical DeltaNet state slots (tmax + 1, or 1 for a batched-decode sequence slot: ring slot cur lives in state slot cur % nsl)
        self.nss = nsl
        self.dn_state = _view(lib.smkp_engine_ptr(e, 1), 48 * nsl * 48 * 128 * 128 * 4, torch.float32).view(48, nsl, 48, 128, 128)
        ncv = eng.tmax + 1  # conv ring slots (K11: tmax + 1)
        self.dn_conv = _view(lib.smkp_engine_ptr(e, 2), 48 * ncv * 10240 * 3 * 2, torch.bfloat16).view(48, ncv, 10240 * 3)
        self.kv_layer_bytes = lib.smkp_engine_ptr(e, 4)
        self.kv_base = lib.smkp_engine_ptr(e, 3)
        self.mtp_kv = None
        if hasattr(eng, "sp"):
            self.mtp_kv = lib.smk_spec_kv_ptr(eng.sp)
        self.dn_idx, self.at_idx = {}, {}
        for i, L in enumerate(eng.Ls):
            (self.dn_idx if L.kind == "linear" else self.at_idx)[i] = len(self.dn_idx if L.kind == "linear" else self.at_idx)
        self.prof = None  # dict name -> ms when profiling is on

    # ------------------------------------------------------------------------------------------------ scratch
    def _scratch(self, S):
        box = getattr(self.e, "_fp_holder", self.e)  # shared by every FastPrefill of this engine (and of the batched decoder's sequence slots, engine/batch.py)
        old = getattr(box, "_fp_alloc", None)
        if old and old["S"] >= S and old["R"] == self.rows:
            return old
        box._fp_alloc = old = None
        torch.cuda.empty_cache()
        d, bf = "cuda", torch.bfloat16
        R = self.rows
        a = dict(S=S, R=R)
        a["h"] = torch.empty(S, H, dtype=bf, device=d)
        a["xn"] = torch.empty(S, H, dtype=bf, device=d)
        a["dn"] = torch.empty(S, H, dtype=bf, device=d)
        a["mix"] = torch.empty(S, H, dtype=bf, device=d)
        a["y6"] = torch.empty(S, 6144, dtype=bf, device=d)
        a["qkvz"] = torch.empty(S * 16384, dtype=bf, device=d)
        a["tmpA"] = torch.empty(S * 12288, dtype=bf, device=d)
        a["ab"] = torch.empty(S, 96, dtype=bf, device=d)
        a["g"] = torch.empty(S, 48, dtype=torch.float32, device=d)
        a["beta"] = torch.empty(S, 48, dtype=torch.float32, device=d)
        a["gu"] = torch.empty(R, 2 * I, dtype=bf, device=d)
        a["act"] = torch.empty(R, I, dtype=bf, device=d)
        a["W1"] = torch.empty(2 * I * H, dtype=bf, device=d)  # gate|up
        a["W2"] = torch.empty(H * I, dtype=bf, device=d)  # everything else (<= 5120 x 17408)
        a["kall"] = torch.empty(self.ctx_max, 4, 256, dtype=bf, device=d)
        a["vall"] = torch.empty(self.ctx_max, 4, 256, dtype=bf, device=d)
        a["conv_prev"] = torch.empty(10240 * 3, dtype=bf, device=d)
        a["tok"] = torch.empty(S + 1, dtype=torch.int32, device=d)
        box._fp_alloc = a
        return a

    # ------------------------------------------------------------------------------------------------ helpers
    class _T:
        def __init__(s, fp, name):
            s.fp, s.name = fp, name

        def __enter__(s):
            if s.fp.prof is not None:
                s.a = torch.cuda.Event(enable_timing=True)
                s.b = torch.cuda.Event(enable_timing=True)
                s.a.record()

        def __exit__(s, *x):
            if s.fp.prof is not None:
                s.b.record()
                s.fp._ev.append((s.name, s.a, s.b))

    def _t(self, name):
        return FastPrefill._T(self, name)

    def _norm(self, h, y, w, xn, T):
        with self._t("norm"):
            smk.check(self.lib.smk_rmsnorm(h.data_ptr(), y.data_ptr() if y is not None else None, w.data_ptr(), xn.data_ptr(), None, T, H, EPS, self.e.st), "rmsnorm")

    def _deq(self, w4, wbuf, row0=0, nrows=None):
        N, K = w4.w.shape[0], w4.w.shape[1] * 8
        nrows = N - row0 if nrows is None else nrows
        with self._t("dequant"):
            smk.check(self.lib.smkp_dequant_w4r(w4.w.data_ptr(), w4.s.data_ptr(), wbuf.data_ptr(), K, row0, nrows, self.e.st), "dequant")
        return wbuf[: nrows * K].view(nrows, K)

    def _mm(self, x, w, out):
        with self._t("gemm"):
            torch.mm(x, w.t(), out=out)

    # ms of one weight pass (qkvz + o + gu + down, 64 layers) of the W4 kernel at 8 * NT rows (default configs): the cheapest way to cover T rows
    # with passes of <= 8 NT rows (a pass costs the weight stream whatever its row count: 14 GB / ~0.9 TB/s = 16 ms floor)
    W4_PASS_MS = (16.2, 19.5, 23.2, 29.1, 33.9, 40.5, 48.3, 66.0)
    _plans = {}

    @classmethod
    def _w4_plan(cls, T):
        """Row counts of the kernel calls covering T rows at minimum total time (dynamic programme over W4_PASS_MS), e.g. 40 -> [40], 64 -> [40, 24], 100 -> [56, 44]."""
        plans = cls._plans
        if T not in plans:
            best = [0.0] * (T + 1)
            choice = [0] * (T + 1)
            for t in range(1, T + 1):
                best[t], choice[t] = min((best[max(0, t - 8 * nt)] + cls.W4_PASS_MS[nt - 1], nt) for nt in range(1, 9))
            parts, t = [], T
            while t > 0:
                parts.append(min(t, 8 * choice[t]))
                t -= 8 * choice[t]
            plans[T] = parts
        return plans[T]

    def _w4g(self, x, w4, out, row0=0, nrows=None, silu=False):
        """out[T][N] = x[T][K] @ W4[row0 : row0 + N]^T through the W4 mma kernel (chunks of <= 64 rows, see _w4_plan; every chunk streams the weights once).  silu: gate|up
        interleaved rows -> out[T][N/2] = silu(gate) * up (decode epilogue, bf16 rounding chain of the decode step)."""
        Nt, K = w4.w.shape[0], w4.w.shape[1] * 8
        N = Nt - row0 if nrows is None else nrows
        wp, sp = w4.w.data_ptr() + row0 * (K // 2), w4.s.data_ptr() + row0 * (K // 64)
        T = x.shape[0]
        with self._t("w4mm"):
            r = 0
            for n in self._w4_plan(T):
                ep = smk.epi_simple(smk.SMK_EPI_SILU, out[r : r + n]) if silu else smk.epi_segs(N, [out[r : r + n]])
                smk.check(self.lib.smkp_w4r_gemm(wp, sp, x[r : r + n].data_ptr(), N, K, n, C.byref(ep), self.e.st), "w4gemm")
                r += n

    def _w4mm(self, x, w4, wbuf, out):
        if self._small:
            self._w4g(x, w4, out)
        else:
            self._mm(x, self._deq(w4, wbuf), out)

    # ------------------------------------------------------------------------------------------------ layers
    def _deltanet(self, li, L, a, T, cur):
        lib, st = self.lib, self.e.st
        di = self.dn_idx[li]
        xn = a["xn"][:T]
        Tn = T - 1 if self._fold else T  # rows through fla (in place in slot cur); a folded last row goes through the decode kernel (slot cur -> cur+1)
        qkvz = a["qkvz"][: T * 16384].view(T, 16384)
        self._w4mm(xn, L.qkvz, a["W2"], qkvz)
        self._mm(xn, L.ab, a["ab"][:T])
        with self._t("deltanet"):
            smk.check(lib.smkp_gates(a["ab"].data_ptr(), L.A_log.data_ptr(), L.dt_bias.data_ptr(), a["g"].data_ptr(), a["beta"].data_ptr(), Tn, st), "gates")
            cs = self.dn_conv[di, cur]
            a["conv_prev"].copy_(cs)
            tmp = a["tmpA"]
            qc, kc, vc = tmp[: Tn * 2048], tmp[Tn * 2048 : Tn * 4096], tmp[Tn * 4096 : Tn * 10240]
            smk.check(lib.smkp_conv(qkvz.data_ptr(), 16384, a["conv_prev"].data_ptr(), L.conv_w.data_ptr(), qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), cs.data_ptr(), Tn, st), "conv")
            S0 = self.dn_state[di, cur % self.nss]
            o, sf = chunk_gated_delta_rule(
                qc.view(1, Tn, 16, 128), kc.view(1, Tn, 16, 128), vc.view(1, Tn, 48, 128), a["g"][:Tn].view(1, Tn, 48), a["beta"][:Tn].view(1, Tn, 48),
                scale=128**-0.5, initial_state=S0.view(1, 48, 128, 128), output_final_state=True, use_qk_l2norm_in_kernel=True, state_v_first=True)
            S0.copy_(sf.view(48, 128, 128))
            smk.check(lib.smkp_gnorm(o.data_ptr(), qkvz.data_ptr() + 10240 * 2, 16384, L.norm_w.data_ptr(), a["y6"].data_ptr(), Tn, st), "gnorm")
            if self._fold:
                smk.check(lib.smkp_dn_row(self.e.h, di, qkvz.data_ptr() + Tn * 16384 * 2, a["ab"].data_ptr() + Tn * 96 * 2, L.conv_w.data_ptr(), L.A_log.data_ptr(),
                                          L.dt_bias.data_ptr(), L.norm_w.data_ptr(), a["y6"].data_ptr() + Tn * 6144 * 2, st), "dn_row")
        self._w4mm(a["y6"][:T], L.o, a["W2"], a["mix"][:T])

    def _attention(self, li, L, a, T, pos0):
        lib, st = self.lib, self.e.st
        ai = self.at_idx[li]
        kvp = self.kv_base + ai * self.kv_layer_bytes
        xn = a["xn"][:T]
        qkv = a["qkvz"][: T * 14336].view(T, 14336)
        self._w4mm(xn, L.qkvz, a["W2"], qkv)
        Ltot = pos0 + T
        kall, vall = a["kall"], a["vall"]
        with self._t("attn"):
            if pos0:
                smk.check(lib.smkp_gather_kv(kvp, self.ctx_max, self.fp8, pos0, L.k_scale, L.v_scale, kall.data_ptr(), vall.data_ptr(), st), "gather")
            q0 = qkv.data_ptr()
            qatt = a["tmpA"][: T * 6144]
            ao = a["tmpA"][T * 6144 : T * 12288]
            smk.check(lib.smkp_attn_prep(T, pos0, q0, 14336, q0 + 12288 * 2, 14336, q0 + 13312 * 2, 14336, L.q_norm.data_ptr(), L.k_norm.data_ptr(), kvp, self.ctx_max,
                                         self.fp8, L.k_scale, L.v_scale, qatt.data_ptr(), kall.data_ptr(), vall.data_ptr(), _ptr(a["pos3"]), st), "attn_prep")
            flash_attn_varlen_func(qatt.view(T, 24, 256), kall[:Ltot], vall[:Ltot], max_seqlen_q=T, cu_seqlens_q=a["cuq"], max_seqlen_k=Ltot, cu_seqlens_k=a["cuk"],
                                   causal=True, softmax_scale=1.0 / 16, fa_version=2, out=ao.view(T, 24, 256))
            smk.check(lib.smkp_attn_gate(ao.data_ptr(), q0, 14336, a["y6"].data_ptr(), T, st), "attn_gate")
        self._w4mm(a["y6"][:T], L.o, a["W2"], a["mix"][:T])

    def _mlp(self, L, a, T):
        lib, st = self.lib, self.e.st
        xn, dn, gu, act = a["xn"], a["dn"], a["gu"], a["act"]
        if self._small:
            for r0 in range(0, T, self.rows):
                r1 = min(T, r0 + self.rows)
                self._w4g(xn[r0:r1], L.gu, act[: r1 - r0], silu=True)
                self._w4g(act[: r1 - r0], L.down, dn[r0:r1])
            return
        Wg = self._deq(L.gu, a["W1"])
        Wd = self._deq(L.down, a["W2"])
        for r0 in range(0, T, self.rows):
            r1 = min(T, r0 + self.rows)
            n = r1 - r0
            self._mm(xn[r0:r1], Wg, gu[:n])
            with self._t("silu"):
                smk.check(lib.smkp_silu_mul(gu.data_ptr(), act.data_ptr(), n, st), "silu_mul")
            self._mm(act[:n], Wd, dn[r0:r1])

    def _tap(self, slot, row):
        """Test hook (taps=True): store the last row's residual into the engine's debug tap `slot` row 0 (what a T=1 head step would have written)."""
        _view(self.lib.smkp_engine_ptr(self.e.h, 7) + slot * 8 * H * 2, H * 2, torch.bfloat16).copy_(row)

    def _mm_rows(self, a, pos0, T):
        """Image inputs of the rows pos0 .. pos0 + T - 1 (after their token embedding): the segment's MRoPE table a['pos3'] (None for a text prompt) and the ViT rows of placeholder tokens."""
        mm = self.mm
        if mm is None:
            a["pos3"] = None
            return
        a["pos3"] = mm.pos3[pos0 : pos0 + T]
        for s, t_end, emb in mm.spans:
            r0, r1 = max(s, pos0), min(t_end, pos0 + T)
            if r0 < r1:
                a["h"][r0 - pos0 : r1 - pos0].copy_(emb[r0 - s : r1 - s])

    # ------------------------------------------------------------------------------------------------ segment / prompt
    def _segment(self, ids_seg, pos0, cur, nxt=None, head=False):
        """Run tokens ids_seg (list) at positions pos0.. through the 64 layers (state carried through the engine buffers).
        nxt: list of the following token of each row (MTP pairs) or None.  Leaves the final-norm hidden in a['xn'][:T].
        head: the LAST row of ids_seg is the last prompt token (T >= 2, small mode): it runs inside the batch, its DeltaNet row through the decode kernel (post-state in ring slot
        cur+1, slot cur keeps the state after the other rows), then the engine's own head tail (final norm + lm_head + argmax -> logits / out token) and the device scalars of a T=1
        head step (pos = pos0 + T, cur + 1, next input token).  nxt then has T-1 entries (the last row's MTP pair is committed by smk_spec_start)."""
        a = self._scratch(max(self.seg, len(ids_seg)))
        T = len(ids_seg)
        assert not head or T >= 2
        self._small = T <= self.w4_max
        self._fold = head
        e, st = self.e, self.e.st
        a["tok"][:T].copy_(torch.tensor(ids_seg, dtype=torch.int32), non_blocking=False)
        a["cuq"] = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        a["cuk"] = torch.tensor([0, pos0 + T], dtype=torch.int32, device="cuda")
        with self._t("embed"):
            smk.check(self.lib.smk_embed(e.G.embed.data_ptr(), a["tok"].data_ptr(), a["h"].data_ptr(), T, H, st), "embed")
        self._mm_rows(a, pos0, T)
        if head and self.taps:
            self._tap(0, a["h"][T - 1])
        for li, L in enumerate(e.Ls):
            self._norm(a["h"], None if li == 0 else a["dn"], L.ln1, a["xn"], T)
            if self.tap is not None and li in TAP_LI:
                self.tap.take(TAP_LI[li], a["h"][:T], pos0)
                if head:
                    self.tap.take_last(TAP_LI[li], a["h"][T - 1])
            if head and self.taps and li:
                self._tap(li, a["h"][T - 1])
            if L.kind == "linear":
                self._deltanet(li, L, a, T, cur)
            else:
                self._attention(li, L, a, T, pos0)
            self._norm(a["h"], a["mix"], L.ln2, a["xn"], T)
            self._mlp(L, a, T)
            if self.layer_hook is not None:
                self.layer_hook(T)
        if head:  # the engine's head tail takes the pre-norm (residual, last down_proj output) of the last row from its own buffers
            _view(self.lib.smkp_engine_ptr(e.h, 5), H * 2, torch.bfloat16).copy_(a["h"][T - 1])
            _view(self.lib.smkp_engine_ptr(e.h, 6), H * 2, torch.bfloat16).copy_(a["dn"][T - 1])
        self._norm(a["h"], a["dn"], e.G.final_norm, a["xn"], T)
        if nxt is not None and len(nxt):
            self._mtp(a, len(nxt), pos0, nxt)
        if head:
            with self._t("head"):
                smk.check(self.lib.smk_engine_head_tail(e.h, int(bool(getattr(e, "sampling", False))), st), "head_tail")
                smk.check(self.lib.smkp_head_finish(e.h, pos0 + T, st), "head_finish")
            self._fold = False
            e.pos = pos0 + T
        return a

    # ------------------------------------------------------------------------------------------------ layer-major group of segments
    def _group(self, segs, cur):
        """Run several consecutive segments LAYER BY LAYER: every layer's weights are dequantized once for the whole group instead of once per segment (a segment of <= w4_max rows
        needs no dequant, the W4 kernel reads the int4 weights).  segs: list of (ids_seg, pos0, nxt) with pos0 of segment k + len = pos0 of segment k + 1; no head / fold segment.
        Bit-identical to calling _segment on each segment in turn: every segment runs the same kernels on the same row counts, and within a layer the segments run in order (a
        segment's DeltaNet state and KV rows of this layer are written before the next segment reads them).  The group's rows share the scratch of one segment (sum of rows <= self.seg
        or the largest segment); the per-layer scratch (qkvz, tmpA, W1, W2, ...) is reused segment after segment, only h / xn / dn / mix / y6 are per-segment slices."""
        e, st = self.e, self.e.st
        Ts = [len(x[0]) for x in segs]
        total = sum(Ts)
        a = self._scratch(max(self.seg, total))
        views, r0 = [], 0
        for (ids_seg, pos0, nxt), T in zip(segs, Ts):
            av = dict(a)
            for name in ("h", "xn", "dn", "mix", "y6", "tok"):
                av[name] = a[name][r0 : r0 + T]
            av["tok"].copy_(torch.tensor(ids_seg, dtype=torch.int32), non_blocking=False)
            av["cuq"] = torch.tensor([0, T], dtype=torch.int32, device="cuda")
            av["cuk"] = torch.tensor([0, pos0 + T], dtype=torch.int32, device="cuda")
            views.append(av)
            r0 += T
        poss = [x[1] for x in segs]
        big = any(T > self.w4_max for T in Ts)
        self._fold = False
        for av, T, pos0 in zip(views, Ts, poss):
            with self._t("embed"):
                smk.check(self.lib.smk_embed(e.G.embed.data_ptr(), av["tok"].data_ptr(), av["h"].data_ptr(), T, H, st), "embed")
            self._mm_rows(av, pos0, T)
        for li, L in enumerate(e.Ls):
            for av, T in zip(views, Ts):
                self._norm(av["h"], None if li == 0 else av["dn"], L.ln1, av["xn"], T)
            if self.tap is not None and li in TAP_LI:
                for av, T, pos0 in zip(views, Ts, poss):
                    self.tap.take(TAP_LI[li], av["h"], pos0)
            W = self._deq(L.qkvz, a["W2"]) if big else None  # one dequant of the qkv(z) weights for every segment of the group
            for av, T, pos0 in zip(views, Ts, poss):
                if L.kind == "linear":
                    self._dn_front(li, L, av, T, cur, W)
                else:
                    self._attn_front(li, L, av, T, pos0, W)
            W = self._deq(L.o, a["W2"]) if big else None
            for av, T in zip(views, Ts):
                self._proj(av["y6"][:T], L.o, W, av["mix"][:T])
            for av, T in zip(views, Ts):
                self._norm(av["h"], av["mix"], L.ln2, av["xn"], T)
            Wg = self._deq(L.gu, a["W1"]) if big else None
            Wd = self._deq(L.down, a["W2"]) if big else None
            for av, T in zip(views, Ts):
                self._mlp_seg(L, av, T, Wg, Wd)
            if self.layer_hook is not None:
                self.layer_hook(total)
        for av, (ids_seg, pos0, nxt), T in zip(views, segs, Ts):
            self._norm(av["h"], av["dn"], e.G.final_norm, av["xn"], T)
            if nxt is not None and len(nxt):
                self._small = T <= self.w4_max
                self._mtp(av, len(nxt), pos0, nxt)
        return a

    def _proj(self, x, w4, W, out):
        """out = x @ W4^T for one segment of a group: the W4 kernel for a segment of <= w4_max rows, else cuBLAS on the group's dequantized weights W."""
        if x.shape[0] <= self.w4_max:
            self._w4g(x, w4, out)
        else:
            self._mm(x, W, out)

    def _dn_front(self, li, L, a, T, cur, W):
        """_deltanet up to (excluding) the o projection, for one segment of a group; a = the segment's scratch view; W = dequantized qkvz weights (None for a small segment)."""
        lib, st = self.lib, self.e.st
        di = self.dn_idx[li]
        xn = a["xn"][:T]
        qkvz = a["qkvz"][: T * 16384].view(T, 16384)
        self._proj(xn, L.qkvz, W, qkvz)
        self._mm(xn, L.ab, a["ab"][:T])
        with self._t("deltanet"):
            smk.check(lib.smkp_gates(a["ab"].data_ptr(), L.A_log.data_ptr(), L.dt_bias.data_ptr(), a["g"].data_ptr(), a["beta"].data_ptr(), T, st), "gates")
            cs = self.dn_conv[di, cur]
            a["conv_prev"].copy_(cs)
            tmp = a["tmpA"]
            qc, kc, vc = tmp[: T * 2048], tmp[T * 2048 : T * 4096], tmp[T * 4096 : T * 10240]
            smk.check(lib.smkp_conv(qkvz.data_ptr(), 16384, a["conv_prev"].data_ptr(), L.conv_w.data_ptr(), qc.data_ptr(), kc.data_ptr(), vc.data_ptr(), cs.data_ptr(), T, st), "conv")
            S0 = self.dn_state[di, cur % self.nss]
            o, sf = chunk_gated_delta_rule(
                qc.view(1, T, 16, 128), kc.view(1, T, 16, 128), vc.view(1, T, 48, 128), a["g"][:T].view(1, T, 48), a["beta"][:T].view(1, T, 48),
                scale=128**-0.5, initial_state=S0.view(1, 48, 128, 128), output_final_state=True, use_qk_l2norm_in_kernel=True, state_v_first=True)
            S0.copy_(sf.view(48, 128, 128))
            smk.check(lib.smkp_gnorm(o.data_ptr(), qkvz.data_ptr() + 10240 * 2, 16384, L.norm_w.data_ptr(), a["y6"].data_ptr(), T, st), "gnorm")

    def _attn_front(self, li, L, a, T, pos0, W):
        """_attention up to (excluding) the o projection, for one segment of a group."""
        lib, st = self.lib, self.e.st
        ai = self.at_idx[li]
        kvp = self.kv_base + ai * self.kv_layer_bytes
        xn = a["xn"][:T]
        qkv = a["qkvz"][: T * 14336].view(T, 14336)
        self._proj(xn, L.qkvz, W, qkv)
        Ltot = pos0 + T
        kall, vall = a["kall"], a["vall"]
        with self._t("attn"):
            if pos0:
                smk.check(lib.smkp_gather_kv(kvp, self.ctx_max, self.fp8, pos0, L.k_scale, L.v_scale, kall.data_ptr(), vall.data_ptr(), st), "gather")
            q0 = qkv.data_ptr()
            qatt = a["tmpA"][: T * 6144]
            ao = a["tmpA"][T * 6144 : T * 12288]
            smk.check(lib.smkp_attn_prep(T, pos0, q0, 14336, q0 + 12288 * 2, 14336, q0 + 13312 * 2, 14336, L.q_norm.data_ptr(), L.k_norm.data_ptr(), kvp, self.ctx_max,
                                         self.fp8, L.k_scale, L.v_scale, qatt.data_ptr(), kall.data_ptr(), vall.data_ptr(), _ptr(a["pos3"]), st), "attn_prep")
            flash_attn_varlen_func(qatt.view(T, 24, 256), kall[:Ltot], vall[:Ltot], max_seqlen_q=T, cu_seqlens_q=a["cuq"], max_seqlen_k=Ltot, cu_seqlens_k=a["cuk"],
                                   causal=True, softmax_scale=1.0 / 16, fa_version=2, out=ao.view(T, 24, 256))
            smk.check(lib.smkp_attn_gate(ao.data_ptr(), q0, 14336, a["y6"].data_ptr(), T, st), "attn_gate")

    def _mlp_seg(self, L, a, T, Wg, Wd):
        """_mlp for one segment of a group: Wg / Wd = the group's dequantized gate|up / down weights (None when the segment is small: the W4 kernel path)."""
        lib, st = self.lib, self.e.st
        xn, dn, gu, act = a["xn"], a["dn"], a["gu"], a["act"]
        if T <= self.w4_max:
            for r0 in range(0, T, self.rows):
                r1 = min(T, r0 + self.rows)
                self._w4g(xn[r0:r1], L.gu, act[: r1 - r0], silu=True)
                self._w4g(act[: r1 - r0], L.down, dn[r0:r1])
            return
        for r0 in range(0, T, self.rows):
            r1 = min(T, r0 + self.rows)
            n = r1 - r0
            self._mm(xn[r0:r1], Wg, gu[:n])
            with self._t("silu"):
                smk.check(lib.smkp_silu_mul(gu.data_ptr(), act.data_ptr(), n, st), "silu_mul")
            self._mm(act[:n], Wd, dn[r0:r1])

    def _mtp(self, a, T, pos0, nxt):
        """MTP KV for pairs (final-norm hidden of position p (a['xn'][t]), token p+1): cat -> fc -> ln1 -> K,V (k_norm + rope) -> MTP KV cache."""
        e, lib, st = self.e, self.lib, self.e.st
        M = e.M
        a["tok"][:T].copy_(torch.tensor(nxt, dtype=torch.int32), non_blocking=False)
        cat = a["tmpA"][: T * 2 * H].view(T, 2 * H)
        with self._t("mtp"):
            smk.check(lib.smkp_mtp_cat(e.G.embed.data_ptr(), a["tok"].data_ptr(), a["xn"].data_ptr(), M.pre_e.data_ptr(), M.pre_h.data_ptr(), cat.data_ptr(), T, st), "mtp_cat")
            if self.mm is not None:  # pairs whose next token is an image placeholder take its ViT row as the MTP input embedding (vLLM's MTP proposer feeds the merged mm embeddings)
                for s, t_end, emb in self.mm.spans:
                    t0, t1 = max(s - 1 - pos0, 0), min(t_end - 1 - pos0, T)
                    if t0 < t1:
                        j0 = pos0 + t0 + 1 - s
                        smk.check(lib.smkp_mtp_cat(emb.data_ptr() + j0 * H * 2, self.mm.arange(t1 - t0), a["xn"].data_ptr() + t0 * H * 2, M.pre_e.data_ptr(), M.pre_h.data_ptr(),
                                                   cat.data_ptr() + t0 * 2 * H * 2, t1 - t0, st), "mtp_cat_mm")
        fcout = a["mix"][:T]
        self._w4mm(cat, M.fc, a["W1"], fcout)
        self._norm(fcout, None, M.ln1, a["xn"], T)
        kv = a["qkvz"][: T * 2048].view(T, 2048)
        if self._small:
            self._w4g(a["xn"][:T], M.qkv, kv, row0=12288, nrows=2048)
        else:
            self._mm(a["xn"][:T], self._deq(M.qkv, a["W2"], 12288, 2048), kv)
        with self._t("mtp"):
            smk.check(lib.smkp_attn_prep(T, pos0, None, 0, kv.data_ptr(), 2048, kv.data_ptr() + 1024 * 2, 2048, None, M.k_norm.data_ptr(), self.mtp_kv, self.ctx_max, 0, 1.0, 1.0,
                                         None, None, None, _ptr(a["pos3"]), st), "mtp_kv")

    def _run(self, ids, spec):
        """Fast prefill of ids[:-1] from the current engine position (rows at positions pos0..), then ids[-1] as the head=1 decode step.  Returns the first token.
        Short prompts (2 <= remaining rows <= min(seg, w4_max)): the last row rides in the final batch (_segment head=True) instead of a separate 20 ms T=1 step."""
        e = self.e
        ids = list(ids)
        n = len(ids) - 1
        pos0 = e.pos
        assert pos0 + len(ids) <= self.ctx_max, "ctx_max too small"
        if spec:
            assert pos0 == 0 and self.mtp_kv, "MTP prefill needs a fresh SpecEngine"
        smk.check(self.lib.smkp_materialize(e.h, e.st), "materialize")  # lazy DeltaNet snapshots: slot cur must hold the state before dn_state is read / written
        torch.cuda.synchronize()
        cur = int(self.ds[1])
        self._ev = []
        i = 0
        folded = False
        while i < n:
            if self.fold and 2 <= n + 1 - i <= min(self.seg, self.w4_max):
                self._segment(ids[i:], pos0 + i, cur, ids[i + 1 :] if spec else None, head=True)
                folded = True
                break
            T = min(self.seg, n - i)
            self._segment(ids[i : i + T], pos0 + i, cur, ids[i + 1 : i + 1 + T] if spec else None)
            i += T
        if not folded:
            if n:
                self.ds[0] = pos0 + n  # engine position scalar (DS_POS); cur is unchanged (state lives in slot cur)
                e.pos = pos0 + n
            e._step(ids[-1:], 1)
        if self.prof is not None:
            torch.cuda.synchronize()
            for name, s, t in self._ev:
                self.prof[name] = self.prof.get(name, 0.0) + s.elapsed_time(t)

    def prefill(self, ids):
        """Engine.prefill replacement: returns the first generated token (staged on the device as the next input)."""
        if len(ids) - 1 < self.min_fast:
            return self.e.prefill(ids)
        self._run(ids, False)
        return self.e.read_out_tok()

    def prefill_spec(self, ids, k=3, start=True):
        """SpecEngine.prefill_spec replacement (the engine must be freshly reset)."""
        e = self.e
        if len(ids) - 1 < self.min_fast:
            return e.prefill_spec(ids, k, start=start)
        self._run(ids, True)
        if getattr(e, "sampling", False):  # K7: first token = a sample of the last prompt row's logits
            smk.check(e.lib.smk_spec_sample_first(e.sp, e.st), "spec_sample_first")
        if start:
            smk.check(e.lib.smk_spec_start(e.sp, k, e.st), "spec_start")
        e._emitted = 0
        return e.read_out_tok()
