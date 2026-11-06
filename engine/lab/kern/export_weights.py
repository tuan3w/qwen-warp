#!/usr/bin/env python3
"""One-time export of real captured layer weights in the engine's REPACKED W4 layout (smk_w4r_gemv) to raw .bin files.

  out dir: /media/fatman/data/sparselab/kern/   (cap 600 MB; this writes ~215 MB)
  files  : <name>.w.bin (int32 [N][K/8], repacked)  <name>.s.bin (bf16 [N][K/128] -> u32 per (tile,group,g), repacked)  <name>.meta (N K)
  layer 31 (full attention): qkv 14336x5120, o 5120x6144, gu 34816x5120 (gate|up interleaved per 8 rows), dn 5120x17408
  layer 20 (DeltaNet):       qkvz 16384x5120

Run:  flock /tmp/gpu-engine.lock /media/fatman/data/workspace/qwen-warp/.venv/bin/python3 export_weights.py
"""
import os
import sys

ENGINE = os.environ.get("SMK_ENGINE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ENGINE)
os.environ.setdefault("SWIFT_MODEL", "/media/fatman/data/models/Swift-1.5-Qwen3.8-27b-INT4")
import torch  # noqa: E402
import pack  # noqa: E402
from weights import Checkpoint  # noqa: E402

OUT = "/media/fatman/data/sparselab/kern"
os.makedirs(OUT, exist_ok=True)


def dump(name, w4):
    w4.repack()
    N, K = w4.N, w4.K
    w4.w.cpu().numpy().tofile(f"{OUT}/{name}.w.bin")
    w4.s.view(torch.int16).cpu().numpy().tofile(f"{OUT}/{name}.s.bin")
    open(f"{OUT}/{name}.meta", "w").write(f"{N} {K}\n")
    print(name, N, K, w4.w.numel() * 4 + w4.s.numel() * 2, "bytes")


ck = Checkpoint()
L = pack.load_layer(ck, 31)
for name, w4 in (("l31_qkv", L.qkvz), ("l31_o", L.o), ("l31_gu", L.gu), ("l31_dn", L.down)):
    dump(name, w4)
del L
torch.cuda.empty_cache()
L = pack.load_layer(ck, 20)
dump("l20_qkvz", L.qkvz)
