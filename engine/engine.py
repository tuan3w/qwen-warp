"""qwen-warp engine driver: loads the packed weights to the GPU, owns the C engine, prefill-by-decode + greedy decode.

  from engine import Engine;  e = Engine(ctx_max=4096);  e.reset();  first = e.prefill(prompt_ids);  toks = e.decode(32)
Checkpoint: $SWIFT_MODEL (default ~/models/swift-1.5-int4).  Needs an RTX 3090 (sm_86; no other GPU has been tested) and the built libswiftmk.so (make -C engine).
"""
import ctypes as C
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
HEADQ_I4R = 3  # include/head_q.h SMK_HEADQ_I4R: int4 head in the repacked layout
DEFAULT_HEAD = "int4"  # K9: lm_head path: "int8" | "int4" (quantised head + exact BF16 rescoring) | "bf16" (the plain BF16 head); env SMK_HEAD overrides
sys.path.insert(0, str(HERE))
import smk  # noqa: E402
import pack  # noqa: E402
from weights import Checkpoint  # noqa: E402


class Engine:
    def __init__(self, ctx_max=4096, kv_fp8=0, layers=pack.NL, lib="libswiftmk.so", verbose=True, head=None, tmax=None, host_tables=None):
        assert layers == pack.NL
        self.lib = smk.load(lib)
        smk.setup_engine_api(self.lib)
        self.ctx_max, self.kv_fp8 = ctx_max, kv_fp8
        head = head or os.environ.get("SMK_HEAD", DEFAULT_HEAD)
        self.head_kind = head
        t0 = time.time()
        G, Ls = pack.load_all(verbose=verbose, head=head)
        self.G, self.Ls = G, Ls  # keep the tensors alive: the C engine only holds pointers
        for L in Ls:  # the engine kernels read the repacked W4 layout (smk_w4r_gemv, pack.repack_w4); converted in place, the tensors are not used in the natural layout afterwards
            for w4 in (L.qkvz, L.o, L.gu, L.down):
                w4.repack()
        if G.hq is not None and G.hq.kind == pack.HEAD_KINDS["int4"]:
            pack.repack_w4(G.hq.w, G.hq.s)
            self.hq_kind = HEADQ_I4R
        elif G.hq is not None:
            self.hq_kind = G.hq.kind
        # Long context: the bf16 embedding and lm_head tables (2 x 2.37 GiB) are only gathered by row when the quantised head is on (embedding of the step's tokens, exact rescoring
        # of the top-R candidates), so they can live in pinned host memory: decode is unchanged (tests/test_host_tables.py) and ~4.7 GiB of VRAM becomes KV cache.  Done before the C engine
        # allocates its KV cache.  The plain BF16 head (set_head('bf16'), MTP head_mode 1) scans a whole table per step and would stream it over PCIe, so it is refused.
        self.tables_on_host = False
        if (os.environ.get("SMK_HOST_TABLES", "0") == "1") if host_tables is None else bool(host_tables):
            assert G.hq is not None, "host tables need the quantised head (Engine(head='int4'|'int8'))"
            emb = torch.empty(G.embed.shape, dtype=G.embed.dtype, pin_memory=True)
            emb.copy_(G.embed)
            lmh = torch.empty(G.lm_head.shape, dtype=G.lm_head.dtype, pin_memory=True)
            lmh.copy_(G.lm_head)
            G.embed, G.lm_head = emb, lmh  # the C engine holds raw pointers into these: they live as long as G
            self.tables_on_host = True
        torch.cuda.empty_cache()  # weight packing leaves ~1.5 GiB of idle blocks in torch's pool (and the GPU tables just freed): hand them to the driver before the engine's cudaMallocs
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
            m.head_q_kind, m.head_q, m.head_qs = self.hq_kind, p(G.hq.w), p(G.hq.s)
        self.m = m
        self.tmax = int(tmax or os.environ.get("SMK_TMAX", 4))  # K11: largest step in rows; the DeltaNet ring has tmax + 1 slots (8: +0.56 GiB, needed for the prompt-lookup tail)
        self.h = self.lib.smk_engine_create_ex(C.byref(m), ctx_max, kv_fp8, self.tmax)
        if not self.h:
            raise RuntimeError("smk_engine_create failed (out of device memory?)")
        self.st = smk.stream()
        self.pos = 0
        torch.cuda.synchronize()

    def set_head(self, head):
        """'bf16' = plain BF16 head + argmax (the pre-K9 path), 'q' = quantised head + exact top-R rescoring (needs Engine(head='int8'|'int4'))."""
        assert not (head == "bf16" and self.tables_on_host), "the plain BF16 head needs the lm_head table on the GPU (Engine(host_tables=False))"
        smk.check(self.lib.smk_engine_set_head(self.h, 0 if head == "bf16" else 1), "set_head")

    # ---- state
    def reset(self):
        smk.check(self.lib.smk_engine_reset(self.h, self.st), "reset")
        self.pos = 0

    # ---- exact sink + ring buffers of the int4 KV cache (src/kvrot.cuh): state outside the per-token cache that a prefix snapshot must carry
    def kv_aux_new(self):
        """Pinned host uint8 tensor big enough for all layers' sink + ring buffers, one per snapshot slot; None for the bf16 / fp8 caches (no such buffers)."""
        n = int(self.lib.swiftmk_kv_aux_bytes(self.h))
        return torch.empty(n, dtype=torch.uint8, pin_memory=True) if n else None

    def kv_aux_save(self, t):
        """Copy the device sink + ring buffers into t (synchronises the engine stream)."""
        assert t.numel() == self.lib.swiftmk_kv_aux_bytes(self.h) and t.is_pinned()
        smk.check(self.lib.swiftmk_kv_aux_copy(self.h, C.c_void_p(t.data_ptr()), 0, self.st), "kv_aux_save")

    def kv_aux_restore(self, t):
        """Copy t back into the device sink + ring buffers, ordered on the engine stream."""
        assert t.numel() == self.lib.swiftmk_kv_aux_bytes(self.h) and t.is_pinned()
        smk.check(self.lib.swiftmk_kv_aux_copy(self.h, C.c_void_p(t.data_ptr()), 1, self.st), "kv_aux_restore")

    # ---- host prefix cache: KV rows [a, a + m) of all 16 attention layers as a packed pinned chunk (include/attn.h)
    def kv_chunk_new(self, m):
        """Pinned host uint8 tensor for a chunk of m KV rows."""
        return torch.empty(int(self.lib.swiftmk_kv_range_bytes(self.h, m)), dtype=torch.uint8, pin_memory=True)

    def kv_chunk_copy(self, t, m, dev_row0, r0, r1, to_host):
        """Rows [r0, r1) of the m-row chunk t <-> device rows dev_row0 + r0 ..; stream ordered, the caller synchronises (torch.cuda.synchronize) before reading t."""
        assert t.numel() == self.lib.swiftmk_kv_range_bytes(self.h, m) and t.is_pinned()
        smk.check(self.lib.swiftmk_kv_range_copy(self.h, C.c_void_p(t.data_ptr()), m, dev_row0, r0, r1, 0 if to_host else 1, self.st), "kv_chunk_copy")

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

    # ---- external verify (feat/dflash2): T rows over host-chosen tokens, host-side greedy accept, commit a rows (include/engine.h smk_engine_verify)
    def verify(self, toks):
        """Run the T = len(toks) rows [anchor, drafts...] at positions pos.. through the target (graph, head 1, no advance); returns the greedy argmax of every row."""
        T = len(toks)
        arr = (C.c_int * T)(*toks)
        smk.check(self.lib.smk_engine_set_tokens(self.h, arr, T, self.st), "set_tokens")
        smk.check(self.lib.smk_engine_verify(self.h, T, self.st), "verify")
        return [int(x) for x in self._read(0, 18, torch.int32)[10 : 10 + T]]

    def commit(self, a, T):
        """Keep the first `a` rows of the last verify of T rows (pos += a, DeltaNet ring slot += a); rows a..T-1 are rolled back."""
        smk.check(self.lib.smk_engine_commit(self.h, a, T, self.st), "commit")
        self.pos += a

    def taps(self, layer_ids, T, out=None):
        """bf16 cuda tensor [len(layer_ids)][T][5120]: hidden after target layer l (the residual stream = HF hidden_states[l + 1]) of rows 0..T-1 of the last step, stream ordered."""
        slots = (C.c_int * len(layer_ids))(*[l + 1 for l in layer_ids])
        if out is None:
            out = torch.empty(len(layer_ids), T, 5120, dtype=torch.bfloat16, device="cuda")
        smk.check(self.lib.smk_engine_tap_gather(self.h, slots, len(layer_ids), T, C.c_void_p(out.data_ptr()), self.st), "tap_gather")
        return out

    def dbg_hidden(self, T=1):
        """bf16 -> fp32 [66][T][5120] debug taps of the last step: slot 0 embedding, l+1 = after layer l, 65 = final norm out."""
        x = self._read(1, 66 * 8 * 5120, torch.bfloat16).view(66, 8, 5120)
        return x[:, :T].float()

    def logits(self, T=1):
        return self._read(2, 8 * pack.V, torch.float32).view(8, pack.V)[:T]

