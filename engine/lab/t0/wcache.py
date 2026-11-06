"""CPU-only: per traced layer and linear type, cache (a) column L2 norms of the dequantised W (all rows, fp32 [K]) and (b) NR random output rows dequantised to fp16 (relative output errors
over a random row subset are unbiased estimates of the full-matrix ones: ||dy||^2 / ||y||^2 summed over rows).  -> /media/fatman/data/sparselab/t0/w_L{li}.npz

Linear types (K = input width):  qkv (5120; DeltaNet in_proj_qkv|z = 16384 rows, attention q|k|v = 14336 rows)   o (6144)   gu (5120; gate|up, rows sampled half from each)   down (17408).
Dequantisation: w[n, k] = (nibble - 8) * scale[n, k // 128]  (engine/weights.py)."""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/media/fatman/data/workspace/qwen-warp-traces/engine")
from weights import Checkpoint  # noqa: E402

LM = "model.language_model.layers."
OUT = Path("/media/fatman/data/sparselab/t0")
NR = 384
LAYERS = [2, 7, 14, 23, 30, 39, 46, 63]
SH = (4 * torch.arange(8)).view(1, 1, 8)


def deq_rows(ck, base, rows):
    """-> float32 [len(rows)][K] dequantised rows of Linear `base`."""
    p = ck.get(base + ".weight_packed")[rows]
    s = ck.get(base + ".weight_scale")[rows].float()
    nib = ((p.unsqueeze(-1) >> SH) & 0xF).reshape(len(rows), -1).float() - 8.0
    return nib * s.repeat_interleave(128, dim=1)


def colnorm2(ck, base, chunk=1024):
    N = ck.shape(base + ".weight_packed")[0]
    acc = None
    for r0 in range(0, N, chunk):
        rows = torch.arange(r0, min(N, r0 + chunk))
        w = deq_rows(ck, base, rows)
        a = (w * w).sum(0)
        acc = a if acc is None else acc + a
    return acc, N


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint()
    rng = np.random.default_rng(0)
    for li in LAYERS:
        t0 = time.time()
        p = f"{LM}{li}."
        attn = li % 4 == 3
        mats = {
            "qkv": [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"] if attn else [p + "linear_attn.in_proj_qkv", p + "linear_attn.in_proj_z"],
            "o": [p + ("self_attn.o_proj" if attn else "linear_attn.out_proj")],
            "gu": [p + "mlp.gate_proj", p + "mlp.up_proj"],
            "down": [p + "mlp.down_proj"],
        }
        out = {}
        for kind, bases in mats.items():
            ns = [ck.shape(b + ".weight_packed")[0] for b in bases]
            tot = sum(ns)
            cn2, sub, ridx = None, [], []
            for b, n in zip(bases, ns):
                c2, _ = colnorm2(ck, b)
                cn2 = c2 if cn2 is None else cn2 + c2
                k = max(1, round(NR * n / tot)) if kind != "gu" else NR // 2
                rows = torch.tensor(np.sort(rng.choice(n, size=min(k, n), replace=False)))
                sub.append(deq_rows(ck, b, rows).half())
                ridx.append(rows)
            out[f"{kind}_cn"] = cn2.sqrt().numpy()
            out[f"{kind}_w"] = torch.cat(sub).numpy()
            out[f"{kind}_nrows"] = np.array(tot)
        np.savez(OUT / f"w_L{li}.npz", **out)
        print(f"L{li} done {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
