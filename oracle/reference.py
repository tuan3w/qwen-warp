"""Pure-torch CPU reference ("oracle") for Swift-1.5-Qwen3.8-27B-INT4 (Qwen3_5 hybrid, text path only) + its MTP head.

Semantics are transcribed from HF transformers 5.18 `models/qwen3_5/modeling_qwen3_5.py` (file:line pointers in
NOTES.md) and, for the MTP head, vLLM 0.30 `models/qwen3_5_mtp.py`. `check_vs_hf.py` compares layer outputs against the
actual HF modules. Weights are dequantised one layer at a time (`weights.py`), never all at once.

Two numeric modes (`Oracle(bf16=...)`):
  fp32  : every activation is fp32. State/accumulators are always fp32.
  bf16  : activations are rounded to bf16 (round-to-nearest-even, values kept in fp32 containers) at the points an HF-bf16
          model / our GPU engine rounds them: embedding, after each RMSNorm, after every linear (qkv/z/a/b/out/q/k/v/o/gate/up/down/fc),
          after SiLU, after each elementwise product, after each residual add, after conv1d+SiLU, the DeltaNet/attention
          core output, gated norm output. NOT rounded (fp32 inside, like fused GPU kernels): attention scores/softmax, DeltaNet
          recurrence + state, g (decay) computation, l2norm of q/k, matmul accumulation, final logits.

State per sequence (`SeqState`):
  linear layers : S fp32 [48(v heads), 128(K), 128(V)] (HF layout `[H, K, V]`; vLLM stores [H, V, K]),
                  conv fp32 [10240, 3]  = last 3 pre-conv qkv inputs per channel, oldest first (zeros initially)
  full-attn     : k, v fp32 [4, T, 256] post-rope (k) / raw (v), bf16-valued in bf16 mode
  MTP           : k, v fp32 [4, T, 256] of the single MTP attention layer
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from weights import MODEL_DIR, Checkpoint, DenseLinear, QuantizedLinear

TEXT = json.load(open(MODEL_DIR / "config.json"))["text_config"]
H = TEXT["hidden_size"]  # 5120
NL = TEXT["num_hidden_layers"]  # 64
EPS = TEXT["rms_norm_eps"]
LAYER_TYPES = TEXT["layer_types"]
# full attention
NH, NKV, HD = TEXT["num_attention_heads"], TEXT["num_key_value_heads"], TEXT["head_dim"]  # 24, 4, 256
ROT = int(HD * TEXT["rope_parameters"]["partial_rotary_factor"])  # 64
THETA = float(TEXT["rope_parameters"]["rope_theta"])
# deltanet
NK, NV = TEXT["linear_num_key_heads"], TEXT["linear_num_value_heads"]  # 16, 48
DK, DV = TEXT["linear_key_head_dim"], TEXT["linear_value_head_dim"]  # 128, 128
KEYD, VALD = NK * DK, NV * DV  # 2048, 6144
CONVD = 2 * KEYD + VALD  # 10240
CK = TEXT["linear_conv_kernel_dim"]  # 4


# ---------------------------------------------------------------------------------------------- numerics helpers
class Num:
    """Rounding policy. `rnd` rounds to bf16 in bf16 mode, identity in fp32 mode."""

    def __init__(self, bf16: bool):
        self.bf16 = bf16

    def rnd(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).to(torch.float32) if self.bf16 else x


def rmsnorm0(x, w, eps=EPS):
    """Zero-centred RMSNorm (HF Qwen3_5RMSNorm :840-857, vLLM GemmaRMSNorm): y = x*rsqrt(mean(x^2)+eps)*(1+w), fp32 math."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def l2norm(x, eps=1e-6):
    """HF l2norm :293-296 (rsqrt(sum(x^2)+eps) * x)."""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_cos_sin(positions: torch.Tensor, num: Num):
    """[L, ROT] cos/sin. Text-only mrope == plain rope (all 3 position rows equal); inv_freq over rotary dim 64, theta 1e7.
    HF Qwen3_5TextRotaryEmbedding :142-212: cat(freqs, freqs) layout (rotate_half style, pair (i, i+32))."""
    inv = 1.0 / (THETA ** (torch.arange(0, ROT, 2, dtype=torch.float32) / ROT))
    fr = positions.float()[:, None] * inv[None, :]
    emb = torch.cat((fr, fr), dim=-1)
    return num.rnd(emb.cos()), num.rnd(emb.sin())


def apply_rope(x, cos, sin, num: Num):
    """x [L, heads, HD]; rotate first ROT dims (HF apply_rotary_pos_emb :673-708). bf16 mode rounds each product like HF."""
    c, s = cos[:, None, :], sin[:, None, :]
    xr, xp = x[..., :ROT], x[..., ROT:]
    out = num.rnd(num.rnd(xr * c) + num.rnd(rotate_half(xr) * s))
    return torch.cat((out, xp), dim=-1)


# ---------------------------------------------------------------------------------------------- delta rule
def delta_recurrent(q, k, v, g, beta, S):
    """Token-by-token gated delta rule (HF torch_recurrent_gated_delta_rule :437-496), fp32.
    q,k [L,Hv,DK] already l2-normalised & head-repeated (q NOT yet scaled); v [L,Hv,DV]; g,beta [L,Hv]; S [Hv,DK,DV] (updated in place).
    Returns out [L,Hv,DV]."""
    q = q * (DK ** -0.5)
    out = torch.empty(q.shape[0], NV, DV)
    for t in range(q.shape[0]):
        S.mul_(g[t].exp()[:, None, None])
        kv_mem = (S * k[t][:, :, None]).sum(1)  # [Hv, DV]  = S^T k
        delta = (v[t] - kv_mem) * beta[t][:, None]
        S.add_(k[t][:, :, None] * delta[:, None, :])
        out[t] = (S * q[t][:, :, None]).sum(1)
    return out


def delta_chunk(q, k, v, g, beta, S, chunk=64):
    """Chunked gated delta rule (HF torch_chunk_gated_delta_rule :300-433), fp32, same arguments/return as delta_recurrent.
    S [Hv,DK,DV] is read as initial state and updated in place with the final state."""
    L = q.shape[0]
    q = q.transpose(0, 1) * (DK ** -0.5)  # [Hv, L, DK]
    k, v = k.transpose(0, 1), v.transpose(0, 1)
    beta, g = beta.transpose(0, 1), g.transpose(0, 1)
    pad = (chunk - L % chunk) % chunk
    q, k, v = (F.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
    beta, g = (F.pad(x, (0, pad)) for x in (beta, g))
    n = (L + pad) // chunk
    v_beta, k_beta = v * beta[..., None], k * beta[..., None]
    q, k, k_beta, v_beta = (x.reshape(NV, n, chunk, -1) for x in (q, k, k_beta, v_beta))
    g = g.reshape(NV, n, chunk)
    upper = torch.ones(chunk, chunk, dtype=torch.bool).triu(1)
    cum = g.cumsum(-1)
    pair = (cum.unsqueeze(-1) - cum.unsqueeze(-2)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ k.transpose(-1, -2)) * pair
    intra = (q @ k.transpose(-1, -2)) * pair
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, k_beta * cum.exp().unsqueeze(-1), upper=False, unitriangular=True)
    out = torch.zeros_like(new_values)
    q = q * cum.exp().unsqueeze(-1)
    kd = k * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cdec = cum[..., -1].exp()[..., None, None]
    for i in range(n):
        v_new = new_values[:, i] - k_cumdecay[:, i] @ S
        out[:, i] = q[:, i] @ S + intra[:, i] @ v_new
        S.copy_(S * cdec[:, i] + kd[:, i].transpose(-1, -2) @ v_new)
    return out.reshape(NV, -1, DV)[:, :L].transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------------------------- state
@dataclass
class SeqState:
    pos: int = 0  # number of tokens already processed
    lin: dict = field(default_factory=dict)  # layer -> {"S": [48,128,128], "conv": [10240,3]}
    kv: dict = field(default_factory=dict)  # layer -> [k, v] each [4, T, 256]
    mtp_kv: list = field(default_factory=lambda: [None, None])  # [k, v] of the MTP attention layer, committed positions only

    def clone_lin(self):
        return {i: {"S": d["S"].clone(), "conv": d["conv"].clone()} for i, d in self.lin.items()}


class LayerW:
    """Everything for one decoder layer, loaded from the mmap'd shards (packed int4 stays packed)."""

    def __init__(self, ck: Checkpoint, i: int):
        p = f"model.language_model.layers.{i}."
        g = lambda n: ck.get(p + n).float()
        self.kind = LAYER_TYPES[i]
        self.in_ln, self.post_ln = g("input_layernorm.weight"), g("post_attention_layernorm.weight")
        q = lambda n: QuantizedLinear.load(ck, p + n)
        self.gate, self.up, self.down = q("mlp.gate_proj"), q("mlp.up_proj"), q("mlp.down_proj")
        if self.kind == "linear_attention":
            a = "linear_attn."
            self.qkv, self.z, self.out = q(a + "in_proj_qkv"), q(a + "in_proj_z"), q(a + "out_proj")
            self.a, self.b = DenseLinear(ck.get(p + a + "in_proj_a.weight")), DenseLinear(ck.get(p + a + "in_proj_b.weight"))
            self.conv_w = g(a + "conv1d.weight").squeeze(1)  # [10240, 4]
            self.A_log, self.dt_bias, self.norm_w = g(a + "A_log"), g(a + "dt_bias"), g(a + "norm.weight")
        else:
            a = "self_attn."
            self.q, self.k, self.v, self.o = q(a + "q_proj"), q(a + "k_proj"), q(a + "v_proj"), q(a + "o_proj")
            self.q_norm, self.k_norm = g(a + "q_norm.weight"), g(a + "k_norm.weight")


class MtpW:
    """MTP head weights (all BF16 in the checkpoint; kept bf16, multiplied in fp32)."""

    def __init__(self, ck: Checkpoint):
        d = lambda n: DenseLinear(ck.get("mtp." + n))
        f = lambda n: ck.get("mtp." + n).float()
        self.fc = d("fc.weight")  # [5120, 10240]
        self.pre_emb, self.pre_hid, self.norm = f("pre_fc_norm_embedding.weight"), f("pre_fc_norm_hidden.weight"), f("norm.weight")
        l = "layers.0."
        self.in_ln, self.post_ln = f(l + "input_layernorm.weight"), f(l + "post_attention_layernorm.weight")
        self.q, self.k, self.v, self.o = (d(l + f"self_attn.{n}_proj.weight") for n in "qkvo")
        self.q_norm, self.k_norm = f(l + "self_attn.q_norm.weight"), f(l + "self_attn.k_norm.weight")
        self.gate, self.up, self.down = (d(l + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))


# ---------------------------------------------------------------------------------------------- the oracle
class Oracle:
    def __init__(self, bf16: bool = False, ck: Checkpoint | None = None, mtp: bool = True):
        self.ck = ck or Checkpoint()
        self.num = Num(bf16)
        self.embed_slice = self.ck._h(self.ck.index["model.language_model.embed_tokens.weight"]).get_slice(
            "model.language_model.embed_tokens.weight"
        )
        self.final_norm = self.ck.get("model.language_model.norm.weight").float()
        self.lm_head = DenseLinear(self.ck.get("lm_head.weight"))  # bf16 [248320, 5120], shared with the MTP head
        self.mtp = MtpW(self.ck) if mtp else None

    # -- helpers
    def embed(self, toks) -> torch.Tensor:
        return torch.stack([self.embed_slice[int(t) : int(t) + 1][0] for t in toks]).float()

    def _mlp(self, W, x, tap=None, pre=""):
        r = self.num.rnd
        g, u = r(W.gate.matmul(x)), r(W.up.matmul(x))
        a = r(r(F.silu(g)) * u)
        d = r(W.down.matmul(a))
        if tap is not None:
            tap.update({pre + "mlp_gate": g, pre + "mlp_up": u, pre + "mlp_act": a, pre + "mlp_down": d})
        return d

    # -- DeltaNet mixer for one sequence (projections are done by caller on the concatenated batch)
    def _deltanet(self, W: LayerW, st: dict, qkv, z, a, b, tap=None):
        """qkv [L,10240], z [L,6144], a,b [L,48] (all already rounded). Updates st in place, returns core out [L,6144]
        (after gated RMSNorm, before out_proj)."""
        r = self.num.rnd
        L = qkv.shape[0]
        X = torch.cat([st["conv"], qkv.T], dim=1)  # [10240, 3+L], oldest first
        conv = F.conv1d(X[None], W.conv_w[:, None, :], groups=CONVD)[0]  # [10240, L]; tap j=3 <-> newest token
        st["conv"] = X[:, -(CK - 1) :].clone()
        conv = r(F.silu(conv)).T  # [L, 10240]
        q, k, v = torch.split(conv, [KEYD, KEYD, VALD], dim=-1)
        q, k = l2norm(q.reshape(L, NK, DK)), l2norm(k.reshape(L, NK, DK))
        v = v.reshape(L, NV, DV)
        rep = NV // NK
        qr, kr = q.repeat_interleave(rep, dim=1), k.repeat_interleave(rep, dim=1)  # head h of 48 uses key head h//3
        beta = r(torch.sigmoid(b))
        g = -W.A_log.exp() * F.softplus(a + W.dt_bias)  # fp32; a is bf16-valued
        core = (delta_chunk if L > 1 else delta_recurrent)(qr, kr, v, g, beta, st["S"])
        core = r(core)  # [L,48,128]
        zz = z.reshape(L, NV, DV)
        n = r(core * torch.rsqrt(core.pow(2).mean(-1, keepdim=True) + EPS))  # HF RMSNormGated :217-233 (weight NOT zero-centred)
        n = r(W.norm_w * n)
        out = r(n * F.silu(zz)).reshape(L, VALD)
        if tap is not None:
            tap.update(
                {"conv_out": conv, "q_l2": q, "k_l2": k, "v": v, "beta": beta, "g": g, "core_out": core,
                 "gated_norm_out": out, "S_out": st["S"].clone(), "conv_state_out": st["conv"].clone()}
            )
        return out

    # -- full attention for one sequence (q/k/v projections done by caller)
    def _attn(self, W, kvc: list, qg, k, v, pos0: int, q_norm, k_norm, tap=None):
        """qg [L,12288] = per head [q(256)|gate(256)]; k,v [L,1024]. kvc=[K,V] [4,T,256] or [None,None] (updated in place).
        Returns attn output * sigmoid(gate)  [L, 6144]."""
        r = self.num.rnd
        L = qg.shape[0]
        qg = qg.reshape(L, NH, 2 * HD)
        q, gate = qg[..., :HD], qg[..., HD:]  # HF :786-789 chunk(2, dim=-1) on the per-head view: [q|gate] per head
        q = r(rmsnorm0(q, q_norm))
        k = r(rmsnorm0(k.reshape(L, NKV, HD), k_norm))
        v = v.reshape(L, NKV, HD)
        cos, sin = rope_cos_sin(torch.arange(pos0, pos0 + L), self.num)
        q, k = apply_rope(q, cos, sin, self.num), apply_rope(k, cos, sin, self.num)
        kn, vn = k.transpose(0, 1), v.transpose(0, 1)  # [4, L, 256]
        if kvc[0] is None:
            kvc[0], kvc[1] = kn.clone(), vn.clone()
        else:
            kvc[0], kvc[1] = torch.cat([kvc[0], kn], 1), torch.cat([kvc[1], vn], 1)
        K, V = kvc
        T = K.shape[1]
        grp = NH // NKV
        qh = q.transpose(0, 1).reshape(NKV, grp, L, HD)  # head h -> kv head h // 6 (HF repeat_kv :711-720)
        s = torch.einsum("kgld,ktd->kglt", qh, K) * (HD ** -0.5)
        qpos = torch.arange(pos0, pos0 + L)[:, None]
        s = s.masked_fill(torch.arange(T)[None, :] > qpos, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("kglt,ktd->kgld", p, V).reshape(NH, L, HD).transpose(0, 1)  # [L,24,256]
        o = r(o)
        gated = r(o * r(torch.sigmoid(gate)))  # HF :816-817
        if tap is not None:
            tap.update({"q_rope": q, "k_rope": k, "v": v, "attn_out": o, "gated": gated, "gate": gate})
        return gated.reshape(L, NH * HD)

    # -- one forward over new tokens of several sequences (layer-major so each layer is dequantised once)
    def forward(self, seqs: list[SeqState], toks: list[list[int]], tap_layers=(), want_hidden_all=False):
        """Process `toks[i]` for sequence i. Returns dict:
          hid_in/hid_out : [n_seq][64] last-token hidden entering/leaving each layer (fp32 [5120])
          final          : per seq post-final-norm hidden of ALL new tokens [L,5120]  (input of lm_head and of the MTP head)
          logits         : per seq logits of the last token [248320]
          taps           : {(seq, layer): {name: tensor}} for layers in tap_layers (last-token or all-token values, see names)
        """
        r = self.num.rnd
        n = len(seqs)
        lens = [len(t) for t in toks]
        hs = [r(self.embed(t)) for t in toks]
        hid_in = [[None] * NL for _ in range(n)]
        hid_out = [[None] * NL for _ in range(n)]
        taps: dict = {}
        for li in range(NL):
            W = LayerW(self.ck, li)
            Hc = torch.cat(hs)
            h_in_all = Hc
            for i in range(n):
                hid_in[i][li] = hs[i][-1].clone()
            tp = {i: ({} if li in tap_layers else None) for i in range(n)}
            xn = r(rmsnorm0(Hc, W.in_ln))
            if W.kind == "linear_attention":
                qkv, z = r(W.qkv.matmul(xn)), r(W.z.matmul(xn))
                a, b = r(W.a.matmul(xn)), r(W.b.matmul(xn))
                outs, o = [], 0
                for i, L in enumerate(lens):
                    st = seqs[i].lin.setdefault(li, {"S": torch.zeros(NV, DK, DV), "conv": torch.zeros(CONVD, CK - 1)})
                    sl = slice(o, o + L)
                    if tp[i] is not None:
                        tp[i].update({"xnorm": xn[sl], "qkv_proj": qkv[sl], "z": z[sl], "a": a[sl], "b": b[sl]})
                        tp[i]["S_in"], tp[i]["conv_state_in"] = st["S"].clone(), st["conv"].clone()
                    outs.append(self._deltanet(W, st, qkv[sl], z[sl], a[sl], b[sl], tp[i]))
                    o += L
                mix = r(W.out.matmul(torch.cat(outs)))
            else:
                qg, k, v = r(W.q.matmul(xn)), r(W.k.matmul(xn)), r(W.v.matmul(xn))
                outs, o = [], 0
                for i, L in enumerate(lens):
                    kvc = seqs[i].kv.setdefault(li, [None, None])
                    sl = slice(o, o + L)
                    if tp[i] is not None:
                        tp[i].update({"xnorm": xn[sl], "q_proj": qg[sl], "k_proj": k[sl], "v_proj": v[sl]})
                    outs.append(self._attn(W, kvc, qg[sl], k[sl], v[sl], seqs[i].pos, W.q_norm, W.k_norm, tp[i]))
                    o += L
                mix = r(W.o.matmul(torch.cat(outs)))
            Hc = r(Hc + mix)
            x2 = r(rmsnorm0(Hc, W.post_ln))
            mlp_tap = {}
            d = self._mlp(W, x2, mlp_tap if li in tap_layers else None)
            Hn = r(Hc + d)
            o = 0
            for i, L in enumerate(lens):
                sl = slice(o, o + L)
                if tp[i] is not None:
                    tp[i].update({"h_in": h_in_all[sl], "mixer_out": mix[sl], "h_mid": Hc[sl], "xnorm2": x2[sl], "h_out": Hn[sl]})
                    tp[i].update({k: v[sl] for k, v in mlp_tap.items()})
                    taps[(i, li)] = tp[i]
                hid_out[i][li] = Hn[sl][-1].clone()
                o += L
            hs = list(torch.split(Hn, lens))
            del W
        finals = [r(rmsnorm0(h, self.final_norm)) for h in hs]
        logits = [self.lm_head.matmul(f[-1:])[0] for f in finals]
        for i in range(n):
            seqs[i].pos += lens[i]
        return {"hid_in": hid_in, "hid_out": hid_out, "final": finals, "logits": logits, "taps": taps}

    # -- MTP head (vLLM qwen3_5_mtp.py:144-192; one full-attention decoder layer, shared lm_head)
    def mtp_forward(self, seq_kv: list, hidden: torch.Tensor, toks, pos0: int, tap=None):
        """hidden [M,5120]: post-final-norm target hidden (or previous MTP output, post mtp.norm) at consecutive positions
        pos0..pos0+M-1; toks: the M *next* tokens (token following each position). seq_kv=[K,V] committed MTP cache.
        The new K/V are appended to `seq_kv` in place (pass a shallow copy `list(seq_kv)` to keep the original untouched:
        tensors are never mutated, only the list entries are replaced). Returns the MTP output hidden [M,5120] (post mtp.norm)."""
        M, r = self.mtp, self.num.rnd
        e = r(rmsnorm0(r(self.embed(toks)), M.pre_emb))
        h = r(rmsnorm0(hidden, M.pre_hid))
        cat = torch.cat([e, h], dim=-1)  # order [embedding, hidden]  (vLLM :164)
        x = r(M.fc.matmul(cat))
        xn = r(rmsnorm0(x, M.in_ln))
        qg, k, v = r(M.q.matmul(xn)), r(M.k.matmul(xn)), r(M.v.matmul(xn))
        gated = self._attn(M, seq_kv, qg, k, v, pos0, M.q_norm, M.k_norm, tap)
        mix = r(M.o.matmul(gated))
        hm = r(x + mix)  # layer is called with residual=None: residual = x (pre-norm input)
        x2 = r(rmsnorm0(hm, M.post_ln))
        d = self._mlp(M, x2)
        out = r(hm + d)
        res = r(rmsnorm0(out, M.norm))  # mtp.norm applied to the layer output (fused add of the residual stream)
        if tap is not None:
            tap.update({"fc_out": x, "xnorm": xn, "h_mid": hm, "layer_out": out, "norm_out": res, "q_proj": qg, "mixer_out": mix})
        return res

    def mtp_logits(self, hid_last):
        return self.lm_head.matmul(hid_last[None])[0]

    def mtp_draft(self, seq: SeqState, hidden_last, next_tok: int, pos: int, k: int = 3, tap=None):
        """Commit the (hidden, next_tok) pair at `pos` into the MTP cache and chain k greedy drafts (drafts 2..k use
        uncommitted cache entries at pos+1.. built from the MTP's own previous output, exactly like vLLM's drafter loop).
        Returns (draft ids list, mtp hidden list [k][5120], logits of draft 1)."""
        drafts, hids, first_logits = [], [], None
        h, t = hidden_last[None], [next_tok]
        kv = seq.mtp_kv  # j == 0 writes the committed cache; later drafts extend a private copy of the list
        for j in range(k):
            out = self.mtp_forward(kv, h, t, pos + j, tap=tap if j == 0 else None)
            if j == 0:
                kv = list(seq.mtp_kv)
            lg = self.mtp_logits(out[0])
            if j == 0:
                first_logits = lg
            d = int(lg.argmax())
            drafts.append(d)
            hids.append(out[0])
            h, t = out, [d]
        return drafts, hids, first_logits


# ---------------------------------------------------------------------------------------------- prompts / CLI
def build_prompts(tok, texts: dict[str, str]) -> dict[str, list[int]]:
    """Chat template from the checkpoint, thinking off (`enable_thinking=False`), generation prompt appended."""
    out = {}
    for name, t in texts.items():
        s = tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        out[name] = tok(s, add_special_tokens=False)["input_ids"]
    return out


E0_SHORT = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}  # verbatim from experiments/2026-10-30-vllm-swift-int4-bs1-baseline/run.py:26-30


STATE_KEYS = {"S_in", "conv_state_in", "S_out", "conv_state_out"}
PREFILL_KEYS = STATE_KEYS | {"h_in", "xnorm", "qkv_proj", "z", "a", "b", "mixer_out", "h_mid", "h_out", "q_proj", "k_proj", "v_proj", "k_rope", "v"}


def generate(o: Oracle, seqs, prompts, n_new, mtp_k=3, tap_layers=(), log=print):
    """Greedy: prefill each prompt then n_new decode steps for all sequences together. Returns per-seq results."""
    t0 = time.time()
    res = o.forward(seqs, prompts, tap_layers=tap_layers)
    log(f"prefill {[len(p) for p in prompts]} tokens: {time.time() - t0:.1f}s")
    n = len(seqs)
    events = [dict(hid_in=[], hid_out=[], final=[], top=[], taps=[], gen=[], drafts=[], mtp_h=[], mtp_taps=[], prefill_taps=None) for _ in range(n)]
    gen = [[] for _ in range(n)]
    prefill_final = [f.clone() for f in res["final"]]
    # MTP cache for prompt positions 0..L-2 : (h_p, prompt[p+1])
    for i in range(n):
        L = len(prompts[i])
        if o.mtp is not None and L > 1:
            o.mtp_forward(seqs[i].mtp_kv, prefill_final[i][:-1], prompts[i][1:], 0)
    for step in range(n_new + 1):
        for i in range(n):
            ev = events[i]
            logit = res["logits"][i]
            top = torch.topk(logit, 20)
            nxt = int(top.indices[0])
            ev["hid_in"].append(torch.stack(res["hid_in"][i]))
            ev["hid_out"].append(torch.stack(res["hid_out"][i]))
            ev["final"].append(res["final"][i][-1].clone())
            ev["top"].append((top.values.clone(), top.indices.clone()))
            ev["taps"].append({li: {k: (v.clone() if k in STATE_KEYS else v[-1:].clone()) for k, v in res["taps"][(i, li)].items()} for li in tap_layers})
            if step == 0:
                ev["prefill_taps"] = {li: {k: v.clone() for k, v in res["taps"][(i, li)].items() if k in PREFILL_KEYS} for li in tap_layers}
            gen[i].append(nxt)
            ev["gen"].append(nxt)
            if o.mtp is not None:
                tapm = {}
                pos = seqs[i].pos - 1  # position of the hidden state just produced
                d, hs, _ = o.mtp_draft(seqs[i], res["final"][i][-1], nxt, pos, k=mtp_k, tap=tapm)
                ev["drafts"].append(d)
                ev["mtp_h"].append(torch.stack(hs))
                ev["mtp_taps"].append(tapm)
        if step == n_new:
            break
        t0 = time.time()
        res = o.forward(seqs, [[g[-1]] for g in gen], tap_layers=tap_layers)
        log(f"decode step {step + 1}/{n_new} ({n} seqs): {time.time() - t0:.1f}s  tokens={[g[-1] for g in gen]}")
    return events, gen, prefill_final


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prompt", default=None, help="user text (default: the 3 E0 prompts)")
    ap.add_argument("--n", type=int, default=8, help="decode steps")
    ap.add_argument("--bf16-rounding", action="store_true")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ORACLE_THREADS", 18)))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = {"user": a.prompt} if a.prompt else E0_SHORT
    pr = build_prompts(tok, texts)
    o = Oracle(bf16=a.bf16_rounding)
    seqs = [SeqState() for _ in pr]
    ev, gen, _ = generate(o, seqs, list(pr.values()), a.n)
    for (name, ids), g, e in zip(pr.items(), gen, ev):
        print(f"--- {name} ({len(ids)} prompt tokens)\n{tok.decode(g)!r}\ndrafts(k=1..3) per event: {e['drafts']}")
