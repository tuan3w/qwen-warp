"""Baseline scheme = a torch port of the shipped 4-bit KV format (engine/src/kvrot.cuh): 256-point Walsh-Hadamard rotation per (token, kv head), 4 scale groups of 64
rotated coefficients (8-bit log2 scale codes, MSE-optimal among 8 candidates around absmax), 16-level companded symmetric grid, decoder gain 1.0075,
first 4 rows and the last 496 rows (RING_W) exact.  Payload per token and layer: K and V x 4 heads x (128 code bytes + 4 scale bytes) = 1056 B."""

import math

import torch

HD = 256
B = 0.0030002594
E_BIAS, E_RES = 176, 12.0
LOG2_GAIN = 0.010779
CAND = torch.tensor([1, 0, -1, -2, -3, -4, -6, -8])


def _level(n):
    u = 2.0 * n - 15.0
    return u * (1.0 + B * u * u)


LEVELS = torch.tensor([_level(n) for n in range(16)], dtype=torch.float32)
MID = (LEVELS[:-1] + LEVELS[1:]) / 2


def _wht(x):  # orthonormal, own inverse
    n = x.shape[-1]
    y = x.float()
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([a + b, a - b], -2).reshape(*x.shape)
        h *= 2
    return y / math.sqrt(n)


# group of coefficient d: G(d) = 2 (d >> 7) + ((d & 15) >> 3)
_d = torch.arange(HD)
GROUP = 2 * (_d >> 7) + ((_d & 15) >> 3)


class Scheme:
    window = 496
    sink = 4

    def __init__(self):
        self.dev = {}

    def _t(self, dev):
        if dev not in self.dev:
            self.dev[dev] = (LEVELS.to(dev), MID.to(dev), CAND.to(dev), GROUP.to(dev))
        return self.dev[dev]

    def _enc(self, x):  # x [n, 4, 256] -> codes uint8 [n,4,128], meta uint8 [n,4,4]
        lv, mid, cand, grp = self._t(x.device)
        y = _wht(x)
        n, h, _ = y.shape
        yg = y.new_empty(n, h, 4, 64)
        for g in range(4):
            yg[:, :, g] = y[..., grp == g]
        amax = yg.abs().amax(-1).clamp_min(1e-30)
        e0 = torch.round(E_RES * torch.log2(amax / lv[15]) + E_BIAS).clamp(0, 255)
        ec = (e0[..., None] + cand).clamp(0, 255)  # [n,h,4,8]
        s = torch.exp2((ec - E_BIAS) / E_RES)
        z = yg[..., None, :] / s[..., None]  # [n,h,4,8,64]
        q = torch.bucketize(z, mid)
        err = ((yg[..., None, :] - s[..., None] * lv[q]) ** 2).sum(-1)
        best = err.argmin(-1, keepdim=True)
        e = ec.gather(-1, best).squeeze(-1)  # [n,h,4]
        s = torch.exp2((e - E_BIAS) / E_RES)
        q = torch.bucketize(yg / s[..., None], mid).to(torch.uint8)  # [n,h,4,64]
        codes = torch.empty(n, h, HD, dtype=torch.uint8, device=x.device)
        for g in range(4):
            codes[..., grp == g] = q[:, :, g]
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        return packed, e.to(torch.uint8)

    def _dec(self, packed, e):
        lv, mid, cand, grp = self._t(packed.device)
        codes = (
            torch.stack([packed & 15, packed >> 4], -1)
            .reshape(*packed.shape[:-1], HD)
            .long()
        )
        s = torch.exp2((e.float() - E_BIAS) / E_RES + LOG2_GAIN)  # [n,h,4]
        y = lv[codes] * s[..., grp]
        return _wht(y).to(torch.bfloat16)

    def encode(self, layer, k, v, pos0):
        kc, km = self._enc(k)
        vc, vm = self._enc(v)
        return {"kc": kc, "km": km, "vc": vc, "vm": vm}

    def decode(self, layer, p):
        return self._dec(p["kc"], p["km"]), self._dec(p["vc"], p["vm"])


def make():
    return Scheme()
