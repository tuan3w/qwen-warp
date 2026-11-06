"""Real-model end-to-end quality of a sparsity rule on the true decode window (GPU, ~0.5 s per config after a one-time ~30 s load + prefill).

  SWIFT_MODEL=... flock /tmp/gpu-engine.lock python kl_eval.py --doc tr32_notes --preset calib_act [--out results/kl/x.jsonl]
  python kl_eval.py --doc D --cfg '[{"name":"down50","act":{"kinds":["down"],"rule":"abs_topk","f":0.5,"b":1}}]'

How: engine + FastPrefill (engine/prefill.py hooks `xhook` / `attn_hook`, both None by default), prefill of the doc's ids[:-1] once, DeltaNet state of the window start saved; every config restores the
state and re-runs the SAME 128-token teacher-forced decode window (the model's own greedy continuation, traces/dump.py) through the 64 layers with the rule applied, then compares the dense logits:
  kl   = mean over window rows of KL(P_dense || P_cfg) in nats  (x100 = the '%' of the campaign: shipped int4 KV = 0.069% = 6.9e-4 nats)
  top1 = fraction of rows whose argmax equals dense;  dnll = mean change of the NLL of the true next window token.
Config grammar (dict): name, act: {kinds:[qkv,o,gu,down] | f_by_kind:{kind:f}, layers:[..]|None, rule: abs_topk|max_topk, f, b (block cols, 1 = per channel), mode: row|union|max (union / max-over-rows of the
8-row groups = the T=8 verify)}, attn: {layers:[..]|None, scorer: dense|tok_oracle|oracle|maxlogit|quest|quest_rot|mean|samp4, b, budget (total tokens read per row/head incl. sink 4 + ring 256), pool: head|kv_p|row8_p}.
The 'attn' override re-implements the attention of the window rows in torch over the dequantised cache (the same K/V the flash path reads); scorer dense must give kl ~ 0 (self check)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp-traces/engine")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "t0"))
from engine import Engine  # noqa: E402
from prefill import FastPrefill  # noqa: E402
import smk  # noqa: E402

import b2_attn  # noqa: E402
import sp  # noqa: E402

DOCS = Path("/media/fatman/data/sparselab/traces/docs")
TRACES = Path("/media/fatman/data/sparselab/traces")
RES = Path("/media/fatman/data/workspace/research/experiments/2026-11-06-sparsity-probes/results/kl")


def logits_chunked(xn, lm_head, chunk=24832):
    x = xn.float()
    return torch.cat([x @ lm_head[c0 : c0 + chunk].to(x.device).float().t() for c0 in range(0, lm_head.shape[0], chunk)], 1)


class Hooks:
    def __init__(self, fp):
        self.fp = fp
        self.kind = {}
        for L in fp.e.Ls:
            self.kind[id(L.qkvz)] = "qkv"
            self.kind[id(L.o)] = "o"
            self.kind[id(L.gu)] = "gu"
            self.kind[id(L.down)] = "down"
        self.act = None
        self.attn = None
        self.gen = torch.Generator(device="cuda").manual_seed(0)
        self.xl = None
        self.stats = {}
        self.seqmask = {}
        self.lab = None

    def set(self, cfg):
        self.act, self.attn = cfg.get("act"), cfg.get("attn")
        self.seqmask = {} if cfg.get("name") is None or True else self.seqmask
        self.fp.xhook = self.x_hook if self.act else None
        self.fp.attn_hook = self.a_hook if self.attn else None
        self.fp.mlp_hook = self.m_hook if cfg.get("mlp") else None
        self.mlp = cfg.get("mlp")
        self.stats = {}

    # ---- activation sparsity
    def x_hook(self, w4, x):
        a = self.act
        kind = self.kind.get(id(w4))
        li = self.fp._li
        if kind is None or (a.get("layers") is not None and li not in a["layers"]):
            return x
        f = a["f_by_kind"].get(kind) if "f_by_kind" in a else (a["f"] if kind in a.get("kinds", sp.KINDS) else None)
        if f is None or f >= 1.0:
            return x
        b, rule, mode = a.get("b", 1), a.get("rule", "abs_topk"), a.get("mode", "row")
        xf = x.float()
        if rule.startswith("seq"):  # GRIFFIN-style sequence-level mask: ONE frozen keep-set per (layer, kind) from the statistics of the last N PROMPT rows, applied to every row (T=1 and T=8 alike)
            key = (li, kind, rule, a.get("N", 1024), f, b)
            m = self.seqmask.get(key)
            if m is None:
                st = self.lab.seq_stat(li, kind, rule, a.get("N", 1024))
                bs = sp.block_score(st.unsqueeze(0), b, "l2")[0] if b > 1 else st
                keep = max(1, int(round(f * st.numel() / b)))
                mb = torch.zeros(bs.numel(), dtype=torch.bool, device=st.device)
                mb[bs.topk(keep).indices] = True
                m = mb.repeat_interleave(b) if b > 1 else mb
                self.seqmask[key] = m
            by = sp.bytes_frac(m.unsqueeze(0), b).mean().item()
            s = self.stats.setdefault(kind, [0.0, 0])
            s[0] += by
            s[1] += 1
            if kind == "down":  # a pruned MLP channel also drops its gate and up rows (tiles of 8 channels = 16 interleaved rows): bytes of gu follow the 8-channel tile occupancy
                s2 = self.stats.setdefault("gu_rows", [0.0, 0])
                s2[0] += sp.bytes_frac(m.unsqueeze(0), 8).mean().item() if b < 8 else by
                s2[1] += 1
            r1 = a.get("rows")  # apply only to window rows < r1 (G0: sparsify the first 128 rows, then go dense)
            if r1 is None:
                return (xf * m).to(x.dtype)
            xo = xf.clone()
            xo[:r1] *= m
            return xo.to(x.dtype)
        if a.get("rows") is not None:
            r1 = a["rows"]
            xs = xf[:r1]
            m = sp.make_mask(xs, rule, f, b) if mode == "row" else sp.union_mask(xs, rule, f, b, None, 8, mode)
            xo = xf.clone()
            xo[:r1] = xs * m
            s = self.stats.setdefault(kind, [0.0, 0])
            s[0] += sp.bytes_frac(m if mode == "row" else m[::8], b).mean().item()
            s[1] += 1
            return xo.to(x.dtype)
        if mode == "row":
            m = sp.make_mask(xf, rule, f, b)
        else:
            T = xf.shape[0]
            G = T // 8
            m = sp.union_mask(xf[: G * 8], rule, f, b, None, 8, mode)
            if G * 8 < T:
                m = torch.cat([m, sp.make_mask(xf[G * 8 :], rule, f, b)])
        by = sp.bytes_frac(m if mode == "row" else m[::8], b).mean().item()
        s = self.stats.setdefault(kind, [0.0, 0])
        s[0] += by
        s[1] += 1
        return (xf * m).to(x.dtype)

    # ---- MLP channel sparsity with a gate-predicted (or oracle) score: skipping channel c skips the up row, the gate-independent down column (CATS / TEAL-on-the-intermediate)
    def m_hook(self, L, x, act_out, fp):
        c = self.mlp
        li = fp._li
        n = x.shape[0]
        if c.get("layers") is not None and li not in c["layers"]:
            fp._w4g(x, L.gu, act_out, silu=True)
            return
        gu = torch.empty(n, 34816, dtype=torch.bfloat16, device=x.device)
        fp._w4g(x, L.gu, gu)  # plain epilogue: [gate 8 | up 8] interleaved per 8 channels
        v = gu.view(n, 2176, 2, 8)
        g, u = v[:, :, 0].reshape(n, 17408), v[:, :, 1].reshape(n, 17408)
        sg = (g.float() * torch.sigmoid(g.float())).to(torch.bfloat16)
        a = (sg.float() * u.float()).to(torch.bfloat16)
        sc = {"gate": sg.float().abs(), "act": a.float().abs(), "up": u.float().abs()}[c.get("score", "gate")]
        f, b, mode = c["f"], c.get("b", 8), c.get("mode", "row")
        if mode == "row":
            m = sp.make_mask(sc, "abs_topk", f, b)
        else:
            G = n // 8
            m = sp.union_mask(sc[: G * 8], "abs_topk", f, b, None, 8, mode)
            if G * 8 < n:
                m = torch.cat([m, sp.make_mask(sc[G * 8 :], "abs_topk", f, b)])
        s_ = self.stats.setdefault("mlp", [0.0, 0])
        s_[0] += sp.bytes_frac(m if mode == "row" else m[::8], b).mean().item()
        s_[1] += 1
        act_out.copy_(a * m)

    # ---- attention sparsity
    def a_hook(self, li, q, kall, vall, Ltot, pos0, T, out):
        a = self.attn
        if a.get("layers") is not None and li not in a["layers"]:
            return self._dense(q, kall, vall, Ltot, T, out)
        K = kall[:Ltot].permute(1, 0, 2).float()
        V = vall[:Ltot].permute(1, 0, 2).float()
        Ly = b2_attn.Layer(None, None, q.float(), K, V)
        sc, b, bud, pool = a.get("scorer", "dense"), a.get("b", 64), a.get("budget", 1024), a.get("pool", "head")
        bv = b2_attn.pad_blocks(Ly.region.float(), b).amax(-1) > 0
        res = torch.empty(T, 24, 256, device=q.device)
        for g in range(4):
            qg = Ly.q[:, 6 * g : 6 * g + 6]
            lg = (torch.einsum("jhd,td->jht", qg, K[g]) / 16).masked_fill(~Ly.valid.unsqueeze(1), float("-inf"))
            P = torch.softmax(lg, -1)
            if sc == "dense":
                allowed = Ly.valid.unsqueeze(1).expand_as(lg)
            elif sc == "tok_oracle":
                allowed = (b2_attn.tok_select(P, Ly.region, bud - 260, pool) | Ly.always.unsqueeze(1)) & Ly.valid.unsqueeze(1)
            else:
                s_ = b2_attn.block_scores(Ly, g, sc, b, lg, P, self.gen)
                nsel = max(1, (bud - 260) // b)
                M = b2_attn.select(s_, bv, pool[:-2] if (sc == "oracle" and pool.endswith("_p")) else pool, nsel)
                allowed = (M.repeat_interleave(b, -1)[..., : Ly.L] | Ly.always.unsqueeze(1)) & Ly.valid.unsqueeze(1)
            Q = torch.softmax(lg.masked_fill(~allowed, float("-inf")), -1)
            res[:, 6 * g : 6 * g + 6] = torch.einsum("jht,td->jhd", Q, V[g])
        out.copy_(res.to(out.dtype))

    def _dense(self, q, kall, vall, Ltot, T, out):
        K = kall[:Ltot].permute(1, 0, 2).float()
        V = vall[:Ltot].permute(1, 0, 2).float()
        L = K.shape[1]
        valid = torch.arange(L, device=q.device).view(1, -1) <= (L - T + torch.arange(T, device=q.device)).view(-1, 1)
        for g in range(4):
            lg = (torch.einsum("jhd,td->jht", q[:, 6 * g : 6 * g + 6].float(), K[g]) / 16).masked_fill(~valid.unsqueeze(1), float("-inf"))
            out[:, 6 * g : 6 * g + 6] = torch.einsum("jht,td->jhd", torch.softmax(lg, -1), V[g]).to(out.dtype)


def presets():
    P = {}
    calib = []
    for kind in sp.KINDS:
        for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            calib.append(dict(name=f"act_{kind}_f{f}_b1", act=dict(kinds=[kind], rule="abs_topk", f=f, b=1)))
    for f in (0.9, 0.8, 0.7, 0.6, 0.5):
        calib.append(dict(name=f"act_all_f{f}_b1", act=dict(kinds=list(sp.KINDS), rule="abs_topk", f=f, b=1)))
    P["calib_act"] = calib
    at = [dict(name="attn_dense_torch", attn=dict(scorer="dense"))]
    for bud in (512, 1024, 2048, 4096, 8192):
        at.append(dict(name=f"attn_tok_oracle_{bud}", attn=dict(scorer="tok_oracle", budget=bud)))
        for sc in ("oracle", "quest", "quest_rot"):
            at.append(dict(name=f"attn_{sc}_b64_{bud}", attn=dict(scorer=sc, b=64, budget=bud, pool="head")))
        at.append(dict(name=f"attn_quest_b64_{bud}_kv", attn=dict(scorer="quest", b=64, budget=bud, pool="kv_p")))
        at.append(dict(name=f"attn_quest_b64_{bud}_row8", attn=dict(scorer="quest", b=64, budget=bud, pool="row8_p")))
    P["calib_attn"] = at
    ml = [dict(name="mlp_dense_check", mlp=dict(score="gate", f=1.0, b=8))]
    for sc in ("gate", "act"):
        for f in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            for b in (8, 32):
                ml.append(dict(name=f"mlp_{sc}_f{f}_b{b}", mlp=dict(score=sc, f=f, b=b)))
    for f in (0.8, 0.6):
        for mode in ("union", "max"):
            ml.append(dict(name=f"mlp_gate_f{f}_b8_{mode}8", mlp=dict(score="gate", f=f, b=8, mode=mode)))
    P["calib_mlp"] = ml
    blk = []
    for kind in sp.KINDS:
        for b in (8, 32, 128):
            for f in (0.95, 0.9, 0.8, 0.7, 0.5):
                blk.append(dict(name=f"actb_{kind}_f{f}_b{b}", act=dict(kinds=[kind], rule="abs_topk", f=f, b=b)))
    for kind in sp.KINDS:
        for mode in ("union", "max"):
            for f in (0.9, 0.7, 0.5):
                for b in (1, 8):
                    blk.append(dict(name=f"actu_{kind}_f{f}_b{b}_{mode}8", act=dict(kinds=[kind], rule="abs_topk", f=f, b=b, mode=mode)))
    P["calib_act_blocks"] = blk
    sa = [dict(name=f"sensact_L{l}_all_f0.8", act=dict(kinds=list(sp.KINDS), rule="abs_topk", f=0.8, b=1, layers=[l])) for l in range(64)]
    P["sens_act"] = sa
    sn = []
    for l in range(3, 64, 4):
        sn.append(dict(name=f"sensattn_L{l}_tok1024", attn=dict(scorer="tok_oracle", budget=1024, layers=[l])))
        sn.append(dict(name=f"sensattn_L{l}_oracle64_2048", attn=dict(scorer="oracle", b=64, budget=2048, layers=[l])))
        sn.append(dict(name=f"sensattn_L{l}_quest64_4096", attn=dict(scorer="quest", b=64, budget=4096, layers=[l])))
    P["sens_attn"] = sn
    a128 = [dict(name="attn_dense_torch", attn=dict(scorer="dense"))]
    for bud in (1024, 2048, 4096, 8192, 16384):
        a128.append(dict(name=f"attn_tok_oracle_{bud}", attn=dict(scorer="tok_oracle", budget=bud)))
        a128.append(dict(name=f"attn_oracle_b64_{bud}", attn=dict(scorer="oracle", b=64, budget=bud)))
        a128.append(dict(name=f"attn_quest_b64_{bud}", attn=dict(scorer="quest", b=64, budget=bud)))
        a128.append(dict(name=f"attn_quest_b64_{bud}_kv", attn=dict(scorer="quest", b=64, budget=bud, pool="kv_p")))
    P["calib_attn128"] = a128
    gr = []
    for bud in (1024, 2048, 4096, 8192):
        for pool in ("kv", "row8"):
            gr.append(dict(name=f"attnT_tok_oracle_{pool}_{bud}", attn=dict(scorer="tok_oracle", budget=bud, pool=pool)))
        for bb in (8, 16, 32):
            gr.append(dict(name=f"attnT_oracle_b{bb}_{bud}", attn=dict(scorer="oracle", b=bb, budget=bud)))
    P["calib_attn_gran"] = gr
    sq = []
    for kind in sp.KINDS:
        for N in (256, 1024, 4096):
            for f in (0.8, 0.7, 0.6, 0.5):
                for b in (1, 8, 64, 128):
                    sq.append(dict(name=f"seq_{kind}_N{N}_f{f}_b{b}", act=dict(kinds=[kind], rule="seq_rms", N=N, f=f, b=b)))
    for f in (0.8, 0.7, 0.6, 0.5):
        for b in (8, 128):
            sq.append(dict(name=f"seqn_down_N1024_f{f}_b{b}", act=dict(kinds=["down"], rule="seq_nrm", N=1024, f=f, b=b)))
    for f in (0.9, 0.8):
        for b in (1, 128):
            sq.append(dict(name=f"seq_all_N1024_f{f}_b{b}", act=dict(kinds=list(sp.KINDS), rule="seq_rms", N=1024, f=f, b=b)))
    P["seq"] = sq
    DN = [l for l in range(64) if l % 4 != 3]
    AT = [l for l in range(64) if l % 4 == 3]
    g0 = [dict(name="g0_dense_control", g0=True, act=dict(kinds=[], f=1.0, rows=128))]
    for tag, ls in (("all", None), ("DN", DN), ("ATT", AT)):
        for f in (0.9, 0.8, 0.7):
            g0.append(dict(name=f"g0_{tag}_allkinds_f{f}_b1", g0=True, act=dict(kinds=list(sp.KINDS), rule="abs_topk", f=f, b=1, rows=128, layers=ls)))
    for kind in sp.KINDS:
        g0.append(dict(name=f"g0_DN_{kind}_f0.7_b1", g0=True, act=dict(kinds=[kind], rule="abs_topk", f=0.7, b=1, rows=128, layers=DN)))
    for f in (0.8, 0.6):
        g0.append(dict(name=f"g0_DN_allkinds_f{f}_b8", g0=True, act=dict(kinds=list(sp.KINDS), rule="abs_topk", f=f, b=8, rows=128, layers=DN)))
        g0.append(dict(name=f"g0_DN_allkinds_seq1024_f{f}_b64", g0=True, act=dict(kinds=list(sp.KINDS), rule="seq_rms", N=1024, f=f, b=64, rows=128, layers=DN)))
    P["g0"] = g0
    P["smoke"] = [dict(name="dense_rerun"), dict(name="attn_dense_torch", attn=dict(scorer="dense")), dict(name="act_down_f0.7", act=dict(kinds=["down"], rule="abs_topk", f=0.7, b=1))]
    return P


class Lab:
    """Resident evaluator: engine + doc state.  `lab.load_doc(name, shift=0, nwin=128)`, `lab.run(cfg) -> dict`, `lab.run2(cfg)` (G0 two-phase run)."""

    def __init__(self, ctx=32768 + 512, host=False):
        t0 = time.time()
        self.e = Engine(ctx_max=ctx, kv_fp8=2, verbose=False, host_tables=ctx > 40000 or host)  # long contexts: embedding + lm_head tables in pinned host memory (4.7 GB of VRAM for the attention override)
        self.fp = FastPrefill(self.e)
        self.fp.w4_max = 300  # windows of up to 256 rows through the W4 mma path (xhook / mlp_hook live there)
        self.hk = Hooks(self.fp)
        self.hk.lab = self
        from smktrace import Tracer

        self.fp.trace = Tracer()  # statistics sink only (no rows stored)
        self.load_s = time.time() - t0
        self.stat = {}

    # ---- sequence-level statistics of the last N prompt rows (collected during the prefill by the Tracer sink)
    def _sink(self, li, kind, t, row0, pos0, T):
        if not self._collect:
            return
        n = t.shape[0]
        gp = pos0 + row0 + torch.arange(n, device=t.device)
        x = t.float()
        sq = x * x
        nrm = sq / sq.sum(1, keepdim=True).clamp_min(1e-20)
        for N in (256, 1024, 4096):
            sel = gp >= self.n_prompt - N
            c = int(sel.sum())
            if c == 0:
                continue
            acc = self.acc.setdefault((li, kind, N), [torch.zeros(t.shape[1], device=t.device), torch.zeros(t.shape[1], device=t.device), 0])
            acc[0] += sq[sel].sum(0)
            acc[1] += nrm[sel].sum(0)
            acc[2] += c

    def seq_stat(self, li, kind, rule, N):
        a = self.acc[(li, kind, N)]
        return (a[0] / a[2]).sqrt() if rule == "seq_rms" else (a[1] / a[2]).sqrt()

    def load_doc(self, doc, shift=0, nwin=128):
        """shift > 0: the window is generated tokens shift .. shift + nwin - 1 of the model's greedy continuation (window 0 = the first 128 = what the traces hold)."""
        e, fp = self.e, self.fp
        t0 = time.time()
        meta = json.loads((TRACES / doc / "meta.json").read_text())
        ids = np.load(DOCS / f"{doc}.npy").astype(np.int32).tolist()
        n = len(ids) - 1
        self.n_prompt = n
        self.win, self.n, self.doc = meta["win_tokens"], n, doc + (f"@{shift}" if shift else "")
        self.acc, self._collect = {}, True
        fp.trace.sink = self._sink
        e.reset()
        smk.check(fp.lib.smkp_materialize(e.h, e.st), "materialize")
        torch.cuda.synchronize()
        self.cur = int(fp.ds[1])
        # segments: the LAST one is exactly 4096 rows (the statistics windows 256 / 1024 / 4096 are suffixes of it)
        cuts = [0]
        while n - cuts[-1] > 4096 + 4096:
            cuts.append(cuts[-1] + 4096)
        if n - cuts[-1] > 4096:
            cuts.append(n - 4096)
        cuts.append(n)
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            fp._segment(ids[a_:b_], a_, self.cur)
        self._collect = False
        fp.trace.sink = None
        fp.ds[0] = n
        e.pos = n
        torch.cuda.synchronize()
        if shift or nwin > 128:  # the model's own continuation by the engine decode graph, then the state is advanced to position n + shift by a teacher-forced segment
            ds_s, st_s, cv_s = fp.ds.clone(), fp.dn_state[:, self.cur].clone(), fp.dn_conv[:, self.cur].clone()
            e._step(ids[-1:], 1)
            seq = [e.read_out_tok()] + e.decode(shift + nwin + 2)
            fp.ds.copy_(ds_s)
            fp.dn_state[:, self.cur].copy_(st_s)
            fp.dn_conv[:, self.cur].copy_(cv_s)
            e.pos = n
            c = [ids[-1]] + seq
            assert not any(t in (248046, 248044) for t in c[: shift + nwin]), "EOS inside the window"
            if shift:
                fp._segment(c[:shift], n, self.cur)
                torch.cuda.synchronize()
            self.win, self.n = c[shift : shift + nwin], n + shift
            fp.ds[0] = self.n
            e.pos = self.n
        self.ds0 = fp.ds.clone()
        self.st0 = fp.dn_state[:, self.cur].clone()
        self.cv0 = fp.dn_conv[:, self.cur].clone()
        self.hk.set({})
        self.ref = None
        self.ref = self._logits()
        self.ref2 = None
        if nwin >= 256:
            self.ref2 = self._logits2(None, None)
            self._ref_states = self._st2
        self.prefill_s = time.time() - t0
        return self.prefill_s

    def _restore(self):
        fp, e = self.fp, self.e
        fp.ds.copy_(self.ds0)
        fp.dn_state[:, self.cur].copy_(self.st0)
        fp.dn_conv[:, self.cur].copy_(self.cv0)
        e.pos = self.n

    def _logits(self):
        self._restore()
        self.fp._segment(self.win, self.n, self.cur)
        xn = self.e._fp_alloc["xn"][: len(self.win)]
        return logits_chunked(xn, self.e.G.lm_head)

    def _logits2(self, cfg1, cfg2):
        """two consecutive 128-row segments (rows 0-127 under cfg1, rows 128-255 under cfg2); -> (logits [256][V], dn_state after seg 1, after seg 2) with the DeltaNet states of the 48 layers"""
        fp, e = self.fp, self.e
        self._restore()
        outs, sts = [], []
        for k, cfg in enumerate((cfg1, cfg2)):
            self.hk.set(cfg or {})
            fp._segment(self.win[128 * k : 128 * (k + 1)], self.n + 128 * k, self.cur)
            if k == 0:
                self._stats2 = dict(self.hk.stats)
            xn = e._fp_alloc["xn"][:128]
            outs.append(logits_chunked(xn, e.G.lm_head))
            sts.append(fp.dn_state[:, self.cur].clone())
        self.hk.set({})
        self._st2 = sts
        return torch.cat(outs)

    def run(self, cfg):
        t0 = time.time()
        self.hk.set(cfg)
        lg = self._logits()
        stats = dict(self.hk.stats)
        self.hk.set({})
        torch.cuda.synchronize()
        dt = time.time() - t0
        return self._score(cfg, lg, self.ref, dt, stats)

    def _score(self, cfg, lg, ref, dt, stats):
        W = lg.shape[0]
        kls, nll = [], []
        tgt = torch.tensor(self.win[1:W] + [0], device=lg.device)
        for r0 in range(0, W, 32):  # chunked: a 256-row fp32 logit block is 254 MB and the card is full
            lp_ref, lp = torch.log_softmax(ref[r0 : r0 + 32], -1), torch.log_softmax(lg[r0 : r0 + 32], -1)
            kls.append((lp_ref.exp() * (lp_ref - lp)).sum(-1))
            ar = torch.arange(lp.shape[0], device=lg.device)
            nll.append(-lp[ar, tgt[r0 : r0 + 32]] + lp_ref[ar, tgt[r0 : r0 + 32]])
        kl = torch.cat(kls)
        dnll = torch.cat(nll)[: W - 1].mean().item()
        top1 = (lg.argmax(-1) == ref.argmax(-1)).float().mean().item()
        r = dict(doc=self.doc, name=cfg.get("name"), kl=kl.mean().item(), kl_p90=kl.quantile(0.9).item(), kl_max=kl.max().item(), top1=top1, dnll=dnll, secs=round(dt, 2))
        if stats:
            r["bytes_frac"] = {k: round(v[0] / v[1], 4) for k, v in stats.items()}
        self._kl_rows = kl
        return r

    def run2(self, cfg):
        """G0: cfg applies to rows 0-127 only (hooks restricted by act.rows = 128), rows 128-255 run dense; reports KL on both halves and the relative error of the 48 DeltaNet states after each segment."""
        t0 = time.time()
        cfg1 = json.loads(json.dumps(cfg))
        lg = self._logits2(cfg1, None)
        stats = self._stats2
        torch.cuda.synchronize()
        dt = time.time() - t0
        sts = self._st2
        ref_sts = self._ref_states
        r = self._score(cfg, lg, self.ref2, dt, stats)
        kl = self._kl_rows.cpu().numpy()
        r["kl_first128"], r["kl_second128"] = float(kl[:128].mean()), float(kl[128:].mean())
        r["kl_rows_8"] = [float(kl[i : i + 16].mean()) for i in range(0, 256, 16)]
        for k in (0, 1):
            d = (sts[k] - ref_sts[k]).flatten(1).norm(dim=1) / ref_sts[k].flatten(1).norm(dim=1).clamp_min(1e-9) if False else (sts[k] - ref_sts[k]).reshape(48, -1).norm(dim=1) / ref_sts[k].reshape(48, -1).norm(dim=1)
            r[f"state_err_after{128 * (k + 1)}_mean"] = d.mean().item()
            r[f"state_err_after{128 * (k + 1)}_max"] = d.max().item()
        r["state_err_per_layer_after128"] = [round(x, 4) for x in ((sts[0] - ref_sts[0]).reshape(48, -1).norm(dim=1) / ref_sts[0].reshape(48, -1).norm(dim=1)).tolist()]
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="tr32_notes")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--shift", type=int, default=0)
    ap.add_argument("--nwin", type=int, default=128)
    ap.add_argument("--host-tables", action="store_true")
    ap.add_argument("--ctx", type=int, default=32768 + 512)
    a = ap.parse_args()
    cfgs = json.loads(a.cfg) if a.cfg else presets()[a.preset]
    lab = Lab(a.ctx, a.host_tables or a.nwin > 128)
    print(f"engine {lab.load_s:.1f}s", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    for doc in a.doc.split(","):
        print(f"doc {doc}: prefill + dense window {lab.load_doc(doc, a.shift, a.nwin):.1f}s", flush=True)
        od = RES / f"shift{a.shift}" if a.shift else RES
        od.mkdir(parents=True, exist_ok=True)
        out = open(a.out or od / f"{doc}_{a.preset or 'cfg'}.jsonl", "w")
        for c in cfgs:
            r = lab.run2(c) if c.get("g0") else lab.run(c)
            print(json.dumps(r), flush=True)
            out.write(json.dumps(r) + "\n")
        out.close()


if __name__ == "__main__":
    main()
