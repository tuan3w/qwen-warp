"""CPU-only cross-check of the engine's int4 MTP head against the oracle run with the SAME int4 weights (fake-quant, fp32 dequant, bf16 rounding of activations).

  1. ~/llm-lab/venv/bin/python tests/test_mtp.py --only drafts --save /tmp/k4_eng_drafts.pt     (GPU lease)
  2. CUDA_VISIBLE_DEVICES= ORACLE_THREADS=12 ~/llm-lab/venv/bin/python tests/check_mtp_fq.py /tmp/k4_eng_drafts.pt     (CPU, ~10 min)
Separates "engine MTP kernels differ from the MTP math" (would show as drafts/hidden differing from this fake-quant oracle) from "int4 quantisation changes the
draft" (fake-quant oracle differs from the BF16 dump but the engine agrees with the fake-quant oracle).
"""
import os
import sys
from pathlib import Path

import torch

ENG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENG))
sys.path.insert(0, str(ENG.parent / "oracle"))
import pack_mtp  # noqa: E402
import reference as R  # noqa: E402
from weights import Checkpoint, DenseLinear  # noqa: E402

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", 12)))
REF = os.path.expanduser("~/llm-lab/swift-ref")
saved = torch.load(sys.argv[1])
quant = os.environ.get("MTP_QUANT", "int4clip")

ck = Checkpoint()
o = R.Oracle(bf16=True, ck=ck)
M = o.mtp


def fq(w):
    p, s = pack_mtp.quant_w4(w, clip=quant == "int4clip", device="cpu")
    return DenseLinear(pack_mtp.dequant_w4(p, s))


for name in ("fc", "q", "k", "v", "o", "gate", "up", "down"):
    setattr(M, name, fq(getattr(M, name).w))
    print("fake-quantised", name, flush=True)


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


for name, s in saved.items():
    zb = torch.load(f"{REF}/bf16/{name}.pt")
    ids, gen = zb["prompt_ids"].tolist(), zb["gen_ids"].tolist()
    L = len(ids)
    seq = R.SeqState()
    o.mtp_forward(seq.mtp_kv, zb["prefill_final"][:-1], ids[1:], 0)
    fqd, fqh = [], []
    for ev in range(9):
        d, hs, _ = o.mtp_draft(seq, zb["final_norm_out"][ev], gen[ev], L - 1 + ev, k=3)
        fqd.append(d)
        fqh.append(torch.stack(hs))
    eng_d, eng_h = s["drafts"].tolist(), s["hidden"]  # [9][3], [9,3,5120]
    bf = zb["drafts"].tolist()
    m_eng = sum(eng_d[e] == fqd[e] for e in range(9))
    m_fq_bf = sum(fqd[e] == bf[e] for e in range(9))
    m_eng_bf = sum(eng_d[e] == bf[e] for e in range(9))
    hr = torch.tensor([[rel(eng_h[e, j], fqh[e][j]) for j in range(3)] for e in range(9)]).mean(0)
    hb = torch.tensor([[rel(fqh[e][j], zb["mtp_hidden"][e, j]) for j in range(3)] for e in range(9)]).mean(0)
    print(f"{name}: engine drafts == fake-quant oracle drafts {m_eng}/9 (all 3 ids); engine hidden vs fake-quant oracle rel-L2 d1/d2/d3 "
          f"{hr[0]:.4f}/{hr[1]:.4f}/{hr[2]:.4f}; fake-quant oracle vs BF16 dump: drafts {m_fq_bf}/9, hidden rel {hb[0]:.4f}/{hb[1]:.4f}/{hb[2]:.4f}; "
          f"engine vs BF16 dump drafts {m_eng_bf}/9", flush=True)
