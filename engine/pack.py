"""swift-mk weight packer: checkpoint (safetensors mmap) -> engine device tensors, no dequant.

Layouts (see include/engine.h):
  * every W4 tensor stays in checkpoint-natural layout (int32 [N][K/8], bf16 scale [N][K/128]); the mma kernel permutes activations instead of weights.
  * fused rows (plain concatenation along N):   DeltaNet  qkvz = [in_proj_qkv(10240) | in_proj_z(6144)]   -> 16384 rows
                                                 attention  qkv  = [q_proj(12288) | k_proj(1024) | v_proj(1024)] -> 14336 rows
  * gate|up interleaved per 8 rows: row 16i+j = gate row 8i+j, row 16i+8+j = up row 8i+j  (SiLU*mul epilogue)
  * DeltaNet in_proj_a|b: one bf16 [96][5120] tensor (a rows 0..47, b rows 48..95)
Everything is loaded layer by layer straight from the mmap'd shards into preallocated device tensors (loading 17.7 GB incl. embed + lm_head).
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "oracle"))
from weights import Checkpoint  # noqa: E402

LM = "model.language_model."
H, I, V = 5120, 17408, 248320
NL = 64
LAYER_TYPES = ["attention" if i % 4 == 3 else "linear" for i in range(NL)]


def repack_w4(w, s, rows=2048):
    """In place: checkpoint-natural W4 (w int32 [N, K/8], s bf16 [N, K/128]) -> the repacked layout of smk_w4r_gemv (include/engine.h): per (16-row tile, 128-k group) one
    contiguous 1 KB unit [row g chunk t (32 x 16 B)][row g+8 chunk t], units of a tile contiguous; nibble j / j+4 of a word = columns 2j / 2j+1; scales one u32
    (bf16 row g | bf16 row g+8 << 16) per (tile, group, g).  Same tensors, same sizes.  Returns (w, s)."""
    N, K8 = w.shape
    nkg = K8 // 16
    assert N % 16 == 0 and K8 % 16 == 0 and rows % 16 == 0
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    order = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=w.device)
    for r0 in range(0, N, rows):
        n = min(rows, N - r0)
        nt = n // 16
        wc = w[r0 : r0 + n].view(nt, 2, 8, nkg, 4, 4)  # [tile, half, g, kg, t, u]
        nib = ((wc.to(torch.int64).unsqueeze(-1) >> sh) & 0xF)[..., order]  # [..., 8] new nibble order
        word = (nib << sh).sum(-1)  # < 2**32
        word = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        w[r0 : r0 + n].copy_(word.permute(0, 3, 1, 2, 4, 5).reshape(n, K8))  # [tile, kg, half, g, t, u]
        sc = s[r0 : r0 + n].view(nt, 2, 8, nkg).permute(0, 3, 2, 1).contiguous()  # [tile, kg, g, half]
        s[r0 : r0 + n].copy_(sc.view(n, nkg))
    return w, s


class W4:
    """Packed int4 weight on the device: w int32 [N, K/8], s bf16 [N, K/128]; repack() converts to the engine layout (engine.py / spec.py do it after loading)."""

    def repack(self):
        repack_w4(self.w, self.s)
        return self

    def __init__(self, N, K, dev):
        self.N, self.K = N, K
        self.w = torch.empty(N, K // 8, dtype=torch.int32, device=dev)
        self.s = torch.empty(N, K // 128, dtype=torch.bfloat16, device=dev)

    def fill(self, ck, base, r0=0, rows=None):
        p, s = ck.get(base + ".weight_packed"), ck.get(base + ".weight_scale")
        n = p.shape[0]
        assert p.shape[1] == self.K // 8 and not ck.has(base + ".weight_g_idx"), base
        self.w[r0 : r0 + n].copy_(p, non_blocking=False)
        self.s[r0 : r0 + n].copy_(s, non_blocking=False)
        return n

    def fill_gateup(self, ck, gate, up):
        g, u = ck.get(gate + ".weight_packed"), ck.get(up + ".weight_packed")
        gs, us = ck.get(gate + ".weight_scale"), ck.get(up + ".weight_scale")
        n = g.shape[0]
        for dst, a, b in ((self.w, g, u), (self.s, gs, us)):
            d = dst.view(n // 8, 2, 8, dst.shape[1])
            d[:, 0].copy_(a.view(n // 8, 8, -1))
            d[:, 1].copy_(b.view(n // 8, 8, -1))


def dev_bf16(ck, name, dev):
    return ck.get(name).to(dev)


class Layer:
    pass


def load_layer(ck: Checkpoint, i: int, dev="cuda") -> Layer:
    p = f"{LM}layers.{i}."
    L = Layer()
    L.kind = LAYER_TYPES[i]
    L.ln1 = dev_bf16(ck, p + "input_layernorm.weight", dev)
    L.ln2 = dev_bf16(ck, p + "post_attention_layernorm.weight", dev)
    if L.kind == "linear":
        a = p + "linear_attn."
        L.qkvz = W4(16384, H, dev)
        n = L.qkvz.fill(ck, a + "in_proj_qkv")
        L.qkvz.fill(ck, a + "in_proj_z", r0=n)
        L.ab = torch.cat([ck.get(a + "in_proj_a.weight"), ck.get(a + "in_proj_b.weight")]).to(dev)  # [96, 5120]
        L.conv_w = ck.get(a + "conv1d.weight").squeeze(1).contiguous().to(dev)  # [10240, 4]
        L.A_log, L.dt_bias, L.norm_w = (ck.get(a + n_).to(dev) for n_ in ("A_log", "dt_bias", "norm.weight"))
        L.o = W4(H, 6144, dev)
        L.o.fill(ck, a + "out_proj")
    else:
        a = p + "self_attn."
        L.qkvz = W4(12288 + 1024 + 1024, H, dev)  # q | k | v
        r = 0
        for n_ in ("q_proj", "k_proj", "v_proj"):
            r += L.qkvz.fill(ck, a + n_, r0=r)
        L.q_norm, L.k_norm = ck.get(a + "q_norm.weight").to(dev), ck.get(a + "k_norm.weight").to(dev)
        L.k_scale, L.v_scale = float(ck.get(a + "k_scale").float()), float(ck.get(a + "v_scale").float())
        L.o = W4(H, 6144, dev)
        L.o.fill(ck, a + "o_proj")
    L.gu = W4(2 * I, H, dev)
    L.gu.fill_gateup(ck, p + "mlp.gate_proj", p + "mlp.up_proj")
    L.down = W4(H, I, dev)
    L.down.fill(ck, p + "mlp.down_proj")
    return L


class Globals:
    pass


HEAD_KINDS = {"bf16": 0, "int8": 1, "int4": 2}  # SMK_HEADQ_*


def quant_head_i8(w, rows=8192):
    """K9 head, int8 per row (E5 'int8_rtn'): scale = amax/127 (fp32), q = round(w/scale) in [-127, 127].  w bf16 [N, K] (GPU) -> (int8 [N, K], fp32 scale [N])."""
    N, K = w.shape
    q = torch.empty(N, K, dtype=torch.int8, device=w.device)
    s = torch.empty(N, dtype=torch.float32, device=w.device)
    for r0 in range(0, N, rows):
        f = w[r0 : r0 + rows].float()
        sc = f.abs().amax(1).clamp_min(1e-12) / 127
        q[r0 : r0 + rows] = torch.round(f / sc[:, None]).clamp_(-127, 127).to(torch.int8)
        s[r0 : r0 + rows] = sc
    return q, s


def quant_head_i4(w, rows=4096):
    """K9 head, int4 g128 RTN (E5 'int4_g128_rtn'): scale = bf16(amax/7) per group of 128 columns, q = round(w/scale) in [-8, 7], engine W4 layout (uint32 [N][K/8],
    nibble e of word j = column 8j+e, value (nibble-8)*scale; bf16 scale [N][K/128]).  w bf16 [N, K] (GPU)."""
    N, K = w.shape
    ow = torch.empty(N, K // 8, dtype=torch.int32, device=w.device)
    os_ = torch.empty(N, K // 128, dtype=torch.bfloat16, device=w.device)
    sh = (4 * torch.arange(8, device=w.device, dtype=torch.int64)).view(1, 1, 8)
    for r0 in range(0, N, rows):
        g = w[r0 : r0 + rows].float().view(-1, K // 128, 128)
        n = g.shape[0]
        sb = (g.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 7).to(torch.bfloat16).float()
        q = torch.round(g / sb).clamp_(-8, 7)
        word = ((q.to(torch.int64) + 8).view(n, K // 8, 8) << sh).sum(-1)  # < 2**32
        ow[r0 : r0 + n] = torch.where(word >= 2**31, word - 2**32, word).to(torch.int32)
        os_[r0 : r0 + n] = sb.squeeze(-1).to(torch.bfloat16)
    return ow, os_


class QHead:
    """Quantised lm_head copy (device tensors kept alive for the C engine): kind in HEAD_KINDS (1 int8, 2 int4), w, s."""

    def __init__(self, kind, w, s):
        self.kind, self.w, self.s = kind, w, s


def load_globals(ck: Checkpoint, dev="cuda", embed=True, lm_head=True, head="int8") -> Globals:
    """head: 'bf16' (BF16 lm_head only) | 'int8' | 'int4': also build the K9 quantised head copy G.hq (quantised on the GPU from the BF16 head at load, ~1 s)."""
    G = Globals()
    G.final_norm = ck.get(LM + "norm.weight").to(dev)
    G.embed = G.lm_head = G.hq = None
    if embed:
        G.embed = torch.empty(V, H, dtype=torch.bfloat16, device=dev)
        G.embed.copy_(ck.get(LM + "embed_tokens.weight"))
    if lm_head:
        G.lm_head = torch.empty(V, H, dtype=torch.bfloat16, device=dev)
        G.lm_head.copy_(ck.get("lm_head.weight"))
        if head != "bf16":
            G.hq = QHead(HEAD_KINDS[head], *(quant_head_i8 if head == "int8" else quant_head_i4)(G.lm_head))
    return G


def load_all(dev="cuda", layers=range(NL), verbose=True, head="int8"):
    ck = Checkpoint()
    t0 = time.time()
    G = load_globals(ck, dev, head=head)
    Ls = []
    for i in layers:
        Ls.append(load_layer(ck, i, dev))
        if verbose and i % 8 == 7:
            print(f"  packed layer {i} ({time.time() - t0:.1f}s)", flush=True)
    torch.cuda.synchronize()
    if verbose:
        print(f"weights loaded in {time.time() - t0:.1f}s; cuda mem {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return G, Ls
