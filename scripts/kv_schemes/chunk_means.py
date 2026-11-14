"""Research scheme (not implemented in the engine): the shipped rotated format plus static means and per-chunk means.  Found by an autonomous search against scripts/kv_emu.py;
see docs/KV_CACHE.md "Scheme lab".  Per layer and kv head: a calibrated static mean of K (stored before rope, re-rotated per token; pos0 rides in the payload) and of V is subtracted
before the Walsh-Hadamard rotation and added back at decode; the 32-token chunk mean of the 60 highest-variance K channels is sent as int8 + fp16 scale; the scale codes shrink
from 8 to 6 bits (8 per octave) to pay for them.  Exact window 480 rows.  Emulator result against shipped.py at window 496: dev 1.029x, held-out 1.139x (score = 1 / dKL), and
1.05x to 1.12x on two 64k-token texts; one calibration text of 4k tokens; no kernel exists, so no speed claim."""

import math

import torch

HD = 256
B = 0.0030002594
E_RES = 8.0  # scale codes: 8 per octave, 6 bits, window of 64 codes around a calibrated base
CENTER = 36
NMEAN = 60  # channels per kv head whose 32-token chunk mean is sent explicitly (K only)
LOG2_GAIN = 0.010779
CAND = torch.tensor([1, 0, -1, -2, -3, -4, -5, -6])


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


_FREQ = torch.tensor([1e7 ** (-i / 32.0) for i in range(32)], dtype=torch.float64)


def _rope(x, pos, sign=1.0):  # partial rope (rotate_half, first 64 dims) of x [n,4,256] at integer positions pos [n]; sign=-1 un-rotates
    ang = pos.to(torch.float64)[:, None] * _FREQ.to(pos.device)[None, :]  # [n,32]
    c = torch.cos(ang).float()[:, None, :]
    sn = (sign * torch.sin(ang)).float()[:, None, :]
    x1, x2 = x[..., :32], x[..., 32:64]
    return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn, x[..., 64:]], -1)


class Scheme:
    window = 480  # exact rows: (480+4+32) rows x 4 KiB + tables stay under the 2.2 MB fixed budget
    sink = 4

    def __init__(self):
        self.dev = {}
        self.muK = None  # [16,4,256] static means, f32
        self.muV = None
        self.base = None  # [16,2,4] int16 calibrated scale-code bases (K,V x group)
        self.idx = None  # [16,4,NMEAN] int16 channels with explicit chunk means

    def _t(self, dev):
        if dev not in self.dev:
            self.dev[dev] = (LEVELS.to(dev), MID.to(dev), CAND.to(dev), GROUP.to(dev))
        return self.dev[dev]

    @staticmethod
    def _group_amax(y):
        n, h, _ = y.shape
        yg = y.new_empty(n, h, 4, 64)
        for g in range(4):
            yg[:, :, g] = y[..., GROUP.to(y.device) == g]
        return yg

    def calibrate(self, layers):
        nl = len(layers)
        dev = layers[0]["K"].device
        muK = torch.zeros(nl, 4, HD)
        muV = torch.zeros(nl, 4, HD)
        base = torch.zeros(nl, 2, 4, dtype=torch.int16)
        idx = torch.zeros(nl, 4, NMEAN, dtype=torch.int16)
        lv15 = float(LEVELS[15])
        for l, d in enumerate(layers):
            for j, name in enumerate(("K", "V")):
                x = d[name].float()
                if j == 0:
                    pos = torch.arange(x.shape[0], device=x.device)
                    mu = _rope(x, pos, -1.0).mean(0)
                    xc = x - _rope(mu[None].expand(x.shape[0], -1, -1), pos)
                else:
                    mu = x.mean(0)
                    xc = x - mu
                (muK if j == 0 else muV)[l] = mu.cpu()
                if j == 0:
                    nc = xc.shape[0] // 32
                    cm = xc[: nc * 32].reshape(nc, 32, 4, HD).mean(1)  # [nc,4,256]
                    idx[l] = cm.pow(2).mean(0).topk(NMEAN, dim=-1).indices.cpu().to(torch.int16)
                    # remove the explicit chunk means (unquantised) before measuring the scales
                    xr = xc[: nc * 32].reshape(nc, 32, 4, HD)
                    m = torch.zeros(4, HD, device=x.device)
                    mask = torch.zeros(4, HD, dtype=torch.bool, device=x.device)
                    mask.scatter_(1, idx[l].long().to(x.device), True)
                    xc = (xr - xr.mean(1, keepdim=True) * mask).reshape(-1, 4, HD)
                y = _wht(xc)
                yg = self._group_amax(y)
                amax = yg.abs().amax(-1).clamp_min(1e-30)
                e = torch.round(E_RES * torch.log2(amax / lv15))  # [n,4,4]
                base[l, j] = e.reshape(-1, 4, 4).float().median(0).values.mean(0).round().cpu().to(torch.int16) if False else \
                    e.permute(2, 0, 1).reshape(4, -1).float().median(1).values.round().cpu().to(torch.int16)
        self.muK, self.muV, self.base, self.idx = muK, muV, base, idx

    # ---- scale codes: 6 bits, 4 codes -> 3 bytes
    @staticmethod
    def _pack6(c):  # c uint8 [..., m], m % 4 == 0
        c = c.long().reshape(*c.shape[:-1], -1, 4)
        v = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
        return torch.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], -1).to(torch.uint8).reshape(*c.shape[:-2], -1)

    @staticmethod
    def _unpack6(p):
        p = p.long().reshape(*p.shape[:-1], -1, 3)
        v = p[..., 0] | (p[..., 1] << 8) | (p[..., 2] << 16)
        return torch.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], -1).reshape(*p.shape[:-2], -1)

    def _enc(self, x, base):  # x [n,4,256] f32 (mean removed) -> codes uint8 [n,4,128], scale codes uint8 [n,4,4]
        lv, mid, cand, grp = self._t(x.device)
        y = _wht(x)
        n, h, _ = y.shape
        yg = self._group_amax(y)
        amax = yg.abs().amax(-1).clamp_min(1e-30)
        lo = (base - CENTER).to(x.device).float()  # [4]
        e0 = torch.round(E_RES * torch.log2(amax / lv[15]))
        ec = (e0[..., None] + cand).clamp(min=lo[:, None], max=lo[:, None] + 63)  # [n,h,4,8]
        s = torch.exp2(ec / E_RES)
        z = yg[..., None, :] / s[..., None]
        q = torch.bucketize(z, mid)
        err = ((yg[..., None, :] - s[..., None] * lv[q]) ** 2).sum(-1)
        best = err.argmin(-1, keepdim=True)
        e = ec.gather(-1, best).squeeze(-1)
        s = torch.exp2(e / E_RES)
        q = torch.bucketize(yg / s[..., None], mid).to(torch.uint8)
        codes = torch.empty(n, h, HD, dtype=torch.uint8, device=x.device)
        for g in range(4):
            codes[..., grp == g] = q[:, :, g]
        packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
        return packed, (e - lo[:, None]).to(torch.uint8)

    def _dec(self, packed, ecode, base):
        lv, mid, cand, grp = self._t(packed.device)
        codes = torch.stack([packed & 15, packed >> 4], -1).reshape(*packed.shape[:-1], HD).long()
        lo = (base - CENTER).to(packed.device).float()
        s = torch.exp2((ecode.float() + lo[:, None]) / E_RES + LOG2_GAIN)
        y = lv[codes] * s[..., grp]
        return _wht(y)

    def encode(self, layer, k, v, pos0):
        dev = k.device
        muK, muV = self.muK[layer].to(dev), self.muV[layer].to(dev)
        base = self.base[layer]
        idx = self.idx[layer].to(dev).long()  # [4,NMEAN]
        pos = pos0 + torch.arange(k.shape[0], device=dev)
        xk = k.float() - _rope(muK[None].expand(k.shape[0], -1, -1), pos)
        cm = xk.gather(2, idx[None].expand(xk.shape[0], -1, -1)).mean(0)  # [4,NMEAN]
        sm = (cm.abs().amax(-1) / 127.0).clamp_min(1e-12).half()  # [4]
        mq = torch.round(cm / sm.float()[:, None]).clamp(-127, 127)
        xk = xk.scatter_add(2, idx[None].expand(xk.shape[0], -1, -1), (-mq * sm.float()[:, None])[None].expand(xk.shape[0], -1, -1))
        kc, ke = self._enc(xk, base[0])
        vc, ve = self._enc(v.float() - muV, base[1])
        return {"kc": kc, "vc": vc, "ke": self._pack6(ke.reshape(-1)), "ve": self._pack6(ve.reshape(-1)),
                "mq": mq.to(torch.int8), "ms": sm, "pos": torch.tensor([pos0], dtype=torch.int32, device=dev)}

    def decode(self, layer, p):
        dev = p["kc"].device
        base = self.base[layer]
        idx = self.idx[layer].to(dev).long()
        n = p["kc"].shape[0]
        ke = self._unpack6(p["ke"]).reshape(n, 4, 4)
        ve = self._unpack6(p["ve"]).reshape(n, 4, 4)
        k = self._dec(p["kc"], ke, base[0])
        mean = torch.zeros(4, HD, device=dev).scatter(1, idx, p["mq"].float() * p["ms"].float()[:, None])
        pos = int(p["pos"][0]) + torch.arange(n, device=dev)
        k = k + _rope(self.muK[layer].to(dev)[None].expand(n, -1, -1), pos) + mean
        v = self._dec(p["vc"], ve, base[1]) + self.muV[layer].to(dev)
        return k.to(torch.bfloat16), v.to(torch.bfloat16)


def make():
    return Scheme()
