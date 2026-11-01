"""swift-mk engine driver: loads the packed weights to the GPU, owns the C engine, prefill-by-decode + greedy decode.

  from engine import Engine;  e = Engine(ctx_max=4096);  e.reset();  first = e.prefill(prompt_ids);  toks = e.decode(32)
Run with the lab venv (~/llm-lab/venv/bin/python); needs the GPU lease.
"""
import ctypes as C
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DEFAULT_HEAD = "int4"  # K9: lm_head path: "int8" | "int4" (quantised head + exact BF16 rescoring) | "bf16" (the plain BF16 head); env SMK_HEAD overrides
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
import smk  # noqa: E402
import pack  # noqa: E402
from weights import Checkpoint  # noqa: E402


class Engine:
    def __init__(self, ctx_max=4096, kv_fp8=0, layers=pack.NL, lib="libswiftmk.so", verbose=True, head=None, tmax=None):
        assert layers == pack.NL
        self.lib = smk.load(lib)
        smk.setup_engine_api(self.lib)
        self.ctx_max, self.kv_fp8 = ctx_max, kv_fp8
        head = head or os.environ.get("SMK_HEAD", DEFAULT_HEAD)
        self.head_kind = head
        t0 = time.time()
        G, Ls = pack.load_all(verbose=verbose, head=head)
        self.G, self.Ls = G, Ls  # keep the tensors alive: the C engine only holds pointers
        self.load_s = time.time() - t0
        m = smk.ModelW()
        p = smk.ptr
        for i, L in enumerate(Ls):
            d = m.layers[i]
            d.kind = 0 if L.kind == "linear" else 1
            d.ln1, d.ln2 = p(L.ln1), p(L.ln2)
            d.qkv_w, d.qkv_s = p(L.qkvz.w), p(L.qkvz.s)
            if L.kind == "linear":
                d.ab, d.conv_w, d.A_log, d.dt_bias, d.norm_w = p(L.ab), p(L.conv_w), p(L.A_log), p(L.dt_bias), p(L.norm_w)
            else:
                d.q_norm, d.k_norm, d.k_scale, d.v_scale = p(L.q_norm), p(L.k_norm), L.k_scale, L.v_scale
            d.o_w, d.o_s = p(L.o.w), p(L.o.s)
            d.gu_w, d.gu_s = p(L.gu.w), p(L.gu.s)
            d.dn_w, d.dn_s = p(L.down.w), p(L.down.s)
        m.embed, m.final_norm, m.lm_head = p(G.embed), p(G.final_norm), p(G.lm_head)
        if G.hq is not None:  # K9: quantised head + exact BF16 rescoring (the BF16 head above stays resident for the rescoring)
            m.head_q_kind, m.head_q, m.head_qs = G.hq.kind, p(G.hq.w), p(G.hq.s)
        self.m = m
        self.tmax = int(tmax or os.environ.get("SMK_TMAX", 4))  # K11: largest step in rows; the DeltaNet ring has tmax + 1 slots (8: +0.62 GiB, needed for the prompt-lookup tail)
        self.h = self.lib.smk_engine_create_ex(C.byref(m), ctx_max, kv_fp8, self.tmax)
        if not self.h:
            raise RuntimeError("smk_engine_create failed (out of device memory?)")
        self.st = smk.stream()
        self.pos = 0
        torch.cuda.synchronize()

    def set_head(self, head):
        """'bf16' = plain BF16 head + argmax (the pre-K9 path), 'q' = quantised head + exact top-R rescoring (needs Engine(head='int8'|'int4'))."""
        smk.check(self.lib.smk_engine_set_head(self.h, 0 if head == "bf16" else 1), "set_head")

    # ---- state
    def reset(self):
        smk.check(self.lib.smk_engine_reset(self.h, self.st), "reset")
        self.pos = 0

    def _step(self, toks, head, graph=True):
        T = len(toks)
        arr = (C.c_int * T)(*toks)
        smk.check(self.lib.smk_engine_set_tokens(self.h, arr, T, self.st), "set_tokens")
        f = self.lib.smk_engine_step if graph else self.lib.smk_engine_step_nograph
        smk.check(f(self.h, T, head, self.st), "step")
        self.pos += T

    def prefill(self, ids, chunk=4, hook=None):
        """Feed all prompt tokens (T=`chunk` graph chunks, no lm_head), the last one as a head step.  Returns the first generated token
        (also staged on the device as the next input).  `hook(pos)` is called after every step (debug)."""
        ids = list(ids)
        n = len(ids) - 1
        i = 0
        while i < n:
            T = min(chunk, n - i)
            self._step(ids[i : i + T], 0)
            i += T
        self._step(ids[-1:], 1)
        return self.read_out_tok()

    def _read(self, what, n, dtype):
        """Copy n elements of the engine's debug buffer `what` (0 device state int[20], 1 dbg taps bf16 [66][8][5120], 2 logits fp32 [8][V]) to a host tensor."""
        out = torch.empty(n, dtype=dtype)
        torch.cuda.synchronize()
        smk.check(self.lib.smk_engine_read(self.h, what, C.c_void_p(out.data_ptr()), out.numel() * out.element_size()), "read")
        return out

    def read_out_tok(self):
        return int(self._read(0, 18, torch.int32)[10])

    def decode(self, n):
        out = (C.c_int * n)()
        smk.check(self.lib.smk_engine_decode(self.h, n, out, self.st), "decode")
        self.pos += n
        return list(out)

    def dbg_hidden(self, T=1):
        """bf16 -> fp32 [66][T][5120] debug taps of the last step: slot 0 embedding, l+1 = after layer l, 65 = final norm out."""
        x = self._read(1, 66 * 8 * 5120, torch.bfloat16).view(66, 8, 5120)
        return x[:, :T].float()

    def logits(self, T=1):
        return self._read(2, 8 * pack.V, torch.float32).view(8, pack.V)[:T]

