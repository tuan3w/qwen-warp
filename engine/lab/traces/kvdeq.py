"""Torch dequantiser of the engine's rotated int4 KV cache (engine/src/kvrot.cuh) + trace loader.

  from kvdeq import dequant_rows;  K = dequant_rows(codes[0, h], meta[0, h], sink[0, h], gain)  # -> [n][256] float32, UNROTATED basis (what k_gather_kv_i4 returns)
"""
import numpy as np
import torch

B = 0.0030002594
LOG2_GAIN = 0.010779
E_BIAS, E_RES = 176, 12.0


def _wht256(x):  # x [..., 256] ; y = H x / 16 (own inverse)
    sh = x.shape
    y = x.reshape(-1, 256)
    h = 1
    while h < 256:
        y = y.view(-1, 256 // (2 * h), 2, h)
        a, b = y[:, :, 0], y[:, :, 1]
        y = torch.stack([a + b, a - b], 2)
        h *= 2
    return (y.reshape(-1, 256) * 0.0625).reshape(sh)


_LEVEL = None


def level_table(dev):
    n = torch.arange(16, dtype=torch.float32, device=dev)
    u = 2 * n - 15
    return u * (1 + B * u * u)


def dequant_rows(codes, meta, sink=None, rotated=False, dev="cuda", gain=LOG2_GAIN):
    """codes uint8 [n][128], meta int32 [n], sink int16(fp16 bits) [4][256] or None (rows 0..3 exact).  -> float32 [n][256] (unrotated unless rotated=True)."""
    c = torch.as_tensor(np.ascontiguousarray(codes), device=dev)
    m = torch.as_tensor(np.ascontiguousarray(meta), device=dev).view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    n = c.shape[0]
    lv = level_table(dev)
    lo, hi = (c & 15).long(), (c >> 4).long()
    nib = torch.stack([lv[lo], lv[hi]], 2).reshape(n, 256)  # natural d order (low nibble = even d)
    d = torch.arange(256, device=dev)
    g = 2 * (d >> 7) + ((d & 15) >> 3)  # group of d
    e = ((m.unsqueeze(1) >> (8 * g).unsqueeze(0)) & 255).float()
    sc = torch.exp2(e / E_RES + gain - E_BIAS / E_RES)
    y = nib * sc
    if sink is not None:
        sk = torch.as_tensor(np.ascontiguousarray(sink), device=dev).view(torch.float16).float()
        y[: sk.shape[0]] = sk
    return y if rotated else _wht256(y)


def hadamard(x):
    return _wht256(x.float())
