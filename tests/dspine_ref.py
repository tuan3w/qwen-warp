"""Independent torch reference of the DSpine block drafter (Tencent Qwen3.8-27B-DSpine-b16): the oracle the CUDA drafter (engine/src/dspine.cu) is tested against.

Written from the checkpoint's tensors and the published algorithm (causal GQA over the target's hidden taps, a rank-512 code bridge after every layer, serial greedy refine over the
top-16 head candidates), not from the CUDA code, so a shared mistake needs two independent misreadings.  Two weight modes:

  RefDSpine(quantize=False)   the checkpoint's bf16 weights: what the model means.
  RefDSpine(quantize=True)    every big projection replaced by dequant(quant_w4(w)) with the engine's own quantiser: the weights the CUDA drafter really multiplies with.

The CUDA drafter is compared against the quantised reference (tight, NMSE: a kernel bug shows up here) and the quantised reference against the bf16 one (loose: that is quantisation quality).
Everything runs on the CPU in bf16 / fp32 (about 2 minutes for one draft block); only the quantiser may use the GPU.
"""

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ENG = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENG))

DSPINE = os.environ.get(
    "DSPINE_PATH", os.path.expanduser("~/llm-lab/models/dspine-b16")
)
TARGET = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
MASK_ID = 248070
TAP_LAYERS = (1, 16, 31, 46, 61)  # 0-based target layers whose output the drafter reads
H, R, C, K = 5120, 512, 1024, 16


def nmse(a, b):
    """normalized mean squared error = mse(a, b) / mse(a, 0) (llama.cpp tests/test-backend-ops.cpp): a is the reference."""
    a, b = a.double().flatten(), b.double().flatten()
    return float(((a - b) ** 2).sum() / (a**2).sum().clamp_min(1e-30))


def target_tensor(name):
    """one bf16 tensor of the target checkpoint (embedding / lm_head), read without loading the model."""
    wm = json.loads((Path(TARGET) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    return (
        safe_open(str(Path(TARGET) / wm[name]), framework="pt", device="cpu")
        .get_tensor(name)
        .to(torch.bfloat16)
    )


class RefDSpine:
    def __init__(self, path=DSPINE, quantize=False, quant_device="cuda"):
        import pack_mtp

        f = safe_open(
            str(Path(path) / "model.safetensors"), framework="pt", device="cpu"
        )
        cpu = lambda n: f.get_tensor(n).to(torch.bfloat16)

        def proj(n):  # [out, in] -> [in, out] so that x @ W
            w = cpu(n)
            if quantize:
                q, s = pack_mtp.quant_w4(w, True, device=quant_device)
                w = pack_mtp.dequant_w4(q, s).to("cpu").to(torch.bfloat16)
            return w.T.contiguous()

        self.nl, self.nh, self.nkv, self.hd = 5, 32, 8, 128
        self.scale = 1.0 / (128**0.5)
        self.L = []
        for l in range(self.nl):
            a = f"layers.{l}."
            self.L.append(
                dict(
                    in_ln=cpu(a + "input_layernorm.weight"),
                    post_ln=cpu(a + "post_attention_layernorm.weight"),
                    q=proj(a + "self_attn.q_proj.weight"),
                    k=proj(a + "self_attn.k_proj.weight"),
                    v=proj(a + "self_attn.v_proj.weight"),
                    o=proj(a + "self_attn.o_proj.weight"),
                    qn=cpu(a + "self_attn.q_norm.weight"),
                    kn=cpu(a + "self_attn.k_norm.weight"),
                    g=proj(a + "mlp.gate_proj.weight"),
                    u=proj(a + "mlp.up_proj.weight"),
                    d=proj(a + "mlp.down_proj.weight"),
                )
            )
        self.fc = proj("fc.weight")  # [25600, 5120]
        self.hidden_norm, self.norm = cpu("hidden_norm.weight"), cpu("norm.weight")
        br = "selected_prefix_bridge."
        self.codebook, self.read_w, self.early_read = (
            cpu(br + "codebook"),
            cpu(br + "read_weight"),
            cpu(br + "early_read_weights"),
        )
        self.write_w = cpu(br + "feature_write_weights")
        self.gate_w, self.pred_gate_w, self.prev_w = (
            cpu(br + "gate.weight"),
            cpu(br + "predecessor_gate.weight"),
            cpu(br + "prev.weight"),
        )
        self.gate_b = cpu(br + "gate.bias").float()
        self.packed = torch.cat(
            [self.gate_w, self.pred_gate_w, self.prev_w], 0
        ).contiguous()
        inv = 1.0 / (1e7 ** (torch.arange(0, 128, 2, dtype=torch.float32) / 128))
        ang = torch.outer(torch.arange(32768, dtype=torch.float32), inv)
        self.cos, self.sin = ang.cos().to(torch.bfloat16), ang.sin().to(torch.bfloat16)
        self.embed = None

    # ------------------------------------------------------------------ pieces
    def rms(self, x, w):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (y.to(torch.bfloat16).float() * w.float()).to(torch.bfloat16)

    def rope(self, x, pos0, T):
        c, s = (
            self.cos[pos0 : pos0 + T].unsqueeze(1),
            self.sin[pos0 : pos0 + T].unsqueeze(1),
        )
        x1, x2 = x[..., :64], x[..., 64:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    def ctx_rows(self, taps):
        """taps [t, 5, 5120] bf16 (target outputs after layers 1/16/31/46/61) -> drafter context rows [t, 5120]."""
        return self.rms(taps.reshape(taps.shape[0], -1) @ self.fc, self.hidden_norm)

    def attn(self, l, x, ctx, pos0, T):
        L = self.L[l]
        B = x.shape[0]
        q = self.rms(
            (x @ L["q"]).view(B, T, self.nh, self.hd).reshape(-1, self.hd), L["qn"]
        ).view(B, T, self.nh, self.hd)
        q = self.rope(q, pos0, T).transpose(1, 2)
        k = self.rms(
            (x @ L["k"]).view(B, T, self.nkv, self.hd).reshape(-1, self.hd), L["kn"]
        ).view(B, T, self.nkv, self.hd)
        k = self.rope(k, pos0, T).transpose(1, 2)
        v = (x @ L["v"]).view(B, T, self.nkv, self.hd).transpose(1, 2)
        S = ctx.shape[1]
        kc = (
            self.rms((ctx @ L["k"]).view(-1, self.hd), L["kn"])
            .view(1, S, self.nkv, self.hd)
            .transpose(1, 2)
        )
        c, s = (
            self.cos[pos0 - S : pos0].view(1, 1, S, 64),
            self.sin[pos0 - S : pos0].view(1, 1, S, 64),
        )
        kc = torch.cat(
            [kc[..., :64] * c - kc[..., 64:] * s, kc[..., 64:] * c + kc[..., :64] * s],
            -1,
        )
        vc = (ctx @ L["v"]).view(1, S, self.nkv, self.hd).transpose(1, 2)
        G = self.nh // self.nkv
        keys, vals = (
            torch.cat([kc, k], 2).repeat_interleave(G, 1),
            torch.cat([vc, v], 2).repeat_interleave(G, 1),
        )
        idx = torch.arange(S + T)
        ok = (idx[None, :] < S) & ((S - 1 - idx[None, :]) < 8192) | (idx[None, :] >= S)
        mask = ((idx[None, :] <= idx[:, None]) & ok)[S:].view(1, 1, T, S + T)
        out = F.scaled_dot_product_attention(
            q, keys, vals, attn_mask=mask, scale=self.scale
        )
        return out.transpose(1, 2).reshape(B, T, self.nh * self.hd) @ L["o"]

    def mlp(self, l, x):
        L = self.L[l]
        return (F.silu(x @ L["g"]) * (x @ L["u"])).to(torch.bfloat16) @ L["d"]

    def bridge(self, layer, out, ids, next_norm_w):
        """code bridge after layer `layer`: -> (next layer input, residual stream, stash of the raw / rms / packed values the refine needs)."""
        B, T = ids.shape
        raw = out.to(torch.bfloat16)
        f = raw.float()
        rms = torch.sqrt(torch.clamp(f.pow(2).mean(-1), min=1.1754943508222875e-38))
        normed = (f / (rms[:, :, None] + 1e-6)).to(torch.bfloat16)
        read = self.read_w if layer == self.nl - 1 else self.early_read[layer]
        proj = normed @ read.T
        codes = (
            32.0 * proj / (torch.sqrt(proj.pow(2).sum(-1, keepdim=True)) + 1e-6)
        ).to(torch.bfloat16)
        anchor = (32.0 * self.codebook[ids[:, 0]].float()).to(torch.bfloat16)
        codes = torch.cat([anchor[:, None, :], codes[:, 1:, :]], 1)
        packed = codes @ self.packed.T
        q, s, v = packed.float().chunk(3, -1)
        gate = torch.sigmoid(q[:, 1:, :] + s[:, :-1, :] + self.gate_b)
        msg = torch.cat([torch.zeros_like(q[:, :1, :]), gate * v[:, :-1, :]], 1).to(
            torch.bfloat16
        )
        delta = msg @ self.write_w[layer].T
        keep = (torch.arange(T) != 0).float()[None, :, None]
        o = (f + (rms[:, :, None] * delta.float()) * keep).to(torch.bfloat16)
        x = o.float()
        y = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
        stash = dict(
            raw=raw.reshape(-1, H),
            rms=rms.reshape(-1),
            packed=packed.reshape(-1, 3 * R),
        )
        return (y.float() * next_norm_w.float()).to(torch.bfloat16), o, stash

    # ------------------------------------------------------------------ block forward
    def draft(self, block_ids, ctx, pos0, return_layers=False):
        """block_ids [1, 8] (anchor, 7 mask tokens), ctx [1, S, 5120] drafter context rows at positions pos0 - S .. pos0 - 1 -> (hid [8, 5120], stash, per-layer residuals)."""
        if self.embed is None:
            self.embed = target_tensor("model.language_model.embed_tokens.weight")
        B, T = block_ids.shape
        res = F.embedding(block_ids.long(), self.embed)
        xin = self.rms(res, self.L[0]["in_ln"])
        layers, stash = [], None
        for i in range(self.nl):
            h = res + self.attn(i, xin, ctx, pos0, T)
            out = h + self.mlp(i, self.rms(h, self.L[i]["post_ln"]))
            xin, res, st = self.bridge(
                i,
                out,
                block_ids,
                self.L[i + 1]["in_ln"] if i + 1 < self.nl else self.norm,
            )
            layers.append(res[0].clone())
            stash = st
        return xin[0], stash, (layers if return_layers else None)

    # ------------------------------------------------------------------ head + refine
    def candidates(self, hid, lm_head):
        """top-K ids per noise row 1..7 by exact bf16 logit, sorted ascending by id (the reference's `torch.sort(order)`)."""
        rows = hid[1:].float()
        logit = torch.cat(
            [
                rows @ lm_head[s : s + 16384].float().T
                for s in range(0, lm_head.shape[0], 16384)
            ],
            1,
        )
        return torch.sort(
            torch.argsort(-logit, dim=-1, stable=True)[:, :K], dim=-1
        ).values, logit

    def refine(self, cand, stash, anchor, lm_head):
        """serial_table greedy refine over `cand` [7, K] -> 7 drafts (list of ids)."""
        n = cand.shape[0]
        raw, rms, q = (
            stash["raw"][1:].float(),
            stash["rms"][1:].float(),
            stash["packed"][1:, :R].float(),
        )
        rows = lm_head[cand].float()  # [n, K, H]
        base = ((raw[:, None, :] * self.norm.float()) * rows).sum(-1)
        wlast = self.write_w[-1].float()  # [H, R]
        tg = (
            (
                (rows * self.norm.float()).to(torch.bfloat16).float().reshape(-1, H)
                @ wlast
            )
            .to(torch.bfloat16)
            .float()
            .reshape(n, K, R)
        )
        pv = torch.cat([self.pred_gate_w, self.prev_w], 0).float()  # [2R, C]
        pred, out = anchor, []
        for i in range(n):
            tv = (
                ((32.0 * self.codebook[pred].float()).to(torch.bfloat16).float() @ pv.T)
                .to(torch.bfloat16)
                .float()
            )
            m = torch.sigmoid(q[i] + tv[:R] + self.gate_b) * tv[R:]
            score = base[i] + rms[i] * (tg[i] @ m)
            pred = int(cand[i, int(torch.argmax(score))])
            out.append(pred)
        return out


def quant_signature(path=DSPINE):
    """sha1 over the W4 words of layer 0's q_proj: changes when the checkpoint or the quantiser changes (a golden file made with another signature is stale)."""
    import hashlib

    import pack_mtp

    w = safe_open(str(Path(path) / "model.safetensors"), framework="pt", device="cpu").get_tensor("layers.0.self_attn.q_proj.weight").to(torch.bfloat16)
    q, s = pack_mtp.quant_w4(w, True, device="cuda")
    return hashlib.sha1(q.cpu().numpy().tobytes() + s.float().cpu().numpy().tobytes()).hexdigest()[:16]
