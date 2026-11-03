"""qwen-warp MTP draft-path packer: BF16 MTP head -> int4 g128 engine W4 format + restricted-vocab int4 draft head.

  W4 quantiser: symmetric g128 along K, q in [-8, 7], scale = amax / 7.5 (clip: per-group MSE-optimal ratio in {1.0, 0.95, .. 0.5}); the scale is rounded to
  bf16 *before* quantising so the engine dequant (nibble - 8) * bf16_scale reproduces exactly what was optimised.  Output layout = the checkpoint's: uint32
  [N][K/8], nibble e of word j = column 8j+e, bf16 scale [N][K/128].
  Draft vocab: the shipped ascending token-id lists draft_vocab_ext_68864.npy (default) and draft_vocab_64k.npy (65536 rows); the lowest full id wins argmax
  ties like the full head.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pack  # noqa: E402
from pack import H, I  # noqa: E402

MODEL = Path(os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4")))
# (the draft-vocab lists are built offline and shipped as .npy files next to this module)
VOCAB_NPY = HERE / "draft_vocab_64k.npy"                  # legacy English/code list (65536 rows), nv=65536
VOCAB_EXT_NPY = HERE / "draft_vocab_ext_68864.npy"        # shipped default: legacy list + 3314 extension tokens + 14 pad, 68864 rows
GROUP = 128


def quant_w4(w, clip=True, rows_per_chunk=None, device="cuda"):
    """w: bf16/fp32 [N, K] (any device) -> (uint32-as-int32 [N, K/8], bf16 scale [N, K/128]) on the GPU."""
    N, K = w.shape
    assert K % GROUP == 0 and K % 8 == 0
    out_w = torch.empty(N, K // 8, dtype=torch.int32, device=device)
    out_s = torch.empty(N, K // GROUP, dtype=torch.bfloat16, device=device)
    rows = rows_per_chunk or max(16, 4_000_000 // K)
    shifts = (4 * torch.arange(8, device=device, dtype=torch.int64)).view(1, 1, 8)
    ratios = [1.0 - 0.05 * i for i in range(11)] if clip else [1.0]
    for r0 in range(0, N, rows):
        g = w[r0 : r0 + rows].to(device).float().view(-1, K // GROUP, GROUP)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best_err, best_q, best_s = None, None, None
        for ratio in ratios:
            s = (amax * ratio / 7.5).to(torch.bfloat16).float()
            q = torch.round(g / s).clamp(-8, 7)
            err = (q * s - g).pow(2).sum(-1, keepdim=True)
            if best_err is None:
                best_err, best_q, best_s = err, q, s
            else:
                m = err < best_err
                best_err = torch.where(m, err, best_err)
                best_q = torch.where(m, q, best_q)
                best_s = torch.where(m, s, best_s)
        n = g.shape[0]
        nib = (best_q.to(torch.int64) + 8).view(n, K // 8, 8)
        word = (nib << shifts).sum(-1)  # < 2**32
        word = torch.where(word >= 2**31, word - 2**32, word)
        out_w[r0 : r0 + n] = word.to(torch.int32)
        out_s[r0 : r0 + n] = best_s.squeeze(-1).to(torch.bfloat16)
    return out_w, out_s


def dequant_w4(w, s):
    """-> fp32 [N, K] (reference for tests)."""
    N, K8 = w.shape
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    nib = ((w.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> sh) & 0xF
    q = (nib - 8).float().view(N, K8 * 8)
    return (q.view(N, -1, GROUP) * s.float().unsqueeze(-1)).view(N, -1)


class W4t:
    def __init__(self, w, s):
        self.w, self.s = w, s


def draft_vocab(nv=None):
    """Ascending int64 numpy array of the draft-vocab token ids.  nv=None (default): VOCAB_EXT_NPY (68864 rows: the 65536-row list + 3314 extension tokens + 14 pad).
    nv=65536: the legacy English/code list (VOCAB_NPY).  Other sizes need a frequency list that is not part of this repository."""
    if nv is None:
        return np.load(VOCAB_EXT_NPY)
    if nv == 65536:
        return np.load(VOCAB_NPY)
    raise ValueError(f"draft vocab size {nv}: only the shipped lists (None = 68864, 65536) are available")


class MtpPack:
    """Device tensors of the packed MTP head (kept alive for the C engine)."""


def load_mtp(G, ck=None, quant="int4clip", head_quant="int4", nv=None, verbose=True):
    """G: pack.Globals (embed / lm_head on the GPU).  quant: 'int4clip' | 'int4' (MTP layer + fc); head_quant: 'int4clip' | 'int4' for the draft head rows."""
    from weights import Checkpoint

    ck = ck or Checkpoint()
    clip, hclip = quant == "int4clip", head_quant == "int4clip"
    M = MtpPack()
    g = lambda n: ck.get("mtp." + n)
    L = "layers.0."
    d = "cuda"
    M.pre_e, M.pre_h, M.norm = (g(n).to(d) for n in ("pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight"))
    M.ln1, M.ln2 = g(L + "input_layernorm.weight").to(d), g(L + "post_attention_layernorm.weight").to(d)
    M.q_norm, M.k_norm = g(L + "self_attn.q_norm.weight").to(d), g(L + "self_attn.k_norm.weight").to(d)
    M.fc = W4t(*quant_w4(g("fc.weight"), clip))
    qkv = torch.cat([g(L + f"self_attn.{n}_proj.weight") for n in ("q", "k", "v")])
    M.qkv = W4t(*quant_w4(qkv, clip))
    M.o = W4t(*quant_w4(g(L + "self_attn.o_proj.weight"), clip))
    gate, up = quant_w4(g(L + "mlp.gate_proj.weight"), clip), quant_w4(g(L + "mlp.up_proj.weight"), clip)
    gu_w = torch.empty(2 * I, H // 8, dtype=torch.int32, device=d)
    gu_s = torch.empty(2 * I, H // GROUP, dtype=torch.bfloat16, device=d)
    for dst, a, b in ((gu_w, gate[0], up[0]), (gu_s, gate[1], up[1])):
        v = dst.view(I // 8, 2, 8, dst.shape[1])
        v[:, 0].copy_(a.view(I // 8, 8, -1))
        v[:, 1].copy_(b.view(I // 8, 8, -1))
    M.gu = W4t(gu_w, gu_s)
    del gate, up
    M.down = W4t(*quant_w4(g(L + "mlp.down_proj.weight"), clip))
    ids = draft_vocab(nv)
    M.ids_t = torch.from_numpy(ids.astype(np.int32)).to(d)
    M.head = W4t(*quant_w4(G.lm_head[torch.from_numpy(ids).to(G.lm_head.device)].to(d), hclip))  # lm_head may be a pinned-host table (Engine host_tables)
    M.nv = len(ids)
    torch.cuda.synchronize()
    if verbose:
        mb = sum(t.numel() * t.element_size() for t in (M.fc.w, M.fc.s, M.qkv.w, M.qkv.s, M.o.w, M.o.s, M.gu.w, M.gu.s, M.down.w, M.down.s)) / 2**20
        hb = (M.head.w.numel() * 4 + M.head.s.numel() * 2) / 2**20
        print(f"MTP packed ({quant}, head {head_quant}, nv={M.nv}): layer+fc {mb:.0f} MiB, draft head {hb:.0f} MiB", flush=True)
    return M
