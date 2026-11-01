"""Cross-check the oracle against the actual HF transformers Qwen3_5 modules (fp32, dequantised weights), on real checkpoint
weights, for layers 0-3 (3 DeltaNet + 1 full attention) and for the MTP decoder layer, plus incremental-vs-full-sequence
consistency (recurrent decode path == chunked prefill path) and chunk-vs-recurrent delta rule.

Run: CUDA_VISIBLE_DEVICES= ~/llm-lab/venv/bin/python check_vs_hf.py      (~3 min on CPU, <8 GB RAM)
"""
import json
import os
import sys

sys.modules["fla"] = None  # force HF's pure-torch delta-rule fallback (the fla Triton kernels need a GPU)
import torch

import reference as R
from reference import Oracle, SeqState
from weights import MODEL_DIR, Checkpoint, load_state_dict

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", 18)))
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def hf_layer(cfg, ck, prefix, li, layer_idx_for_type):
    layer = Qwen3_5DecoderLayer(cfg, layer_idx_for_type).float().eval()
    sd = load_state_dict(ck, prefix)
    sd.pop("self_attn.k_scale", None), sd.pop("self_attn.v_scale", None)
    missing = layer.load_state_dict(sd, strict=True)
    return layer


def main():
    ck = Checkpoint()
    tcfg = json.load(open(MODEL_DIR / "config.json"))["text_config"]
    cfg = Qwen3_5TextConfig(**tcfg)
    cfg._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ids = R.build_prompts(tok, {"p": R.E0_SHORT["code"]})["p"]
    ids = ids[:24]
    L = len(ids)
    print(f"tokens L={L}")
    R.NL = 4  # only layers 0..3
    o = Oracle(bf16=False, mtp=True)
    taps_layers = (0, 1, 2, 3)

    # ---- A: oracle full prefill of L tokens
    sA = SeqState()
    rA = o.forward([sA], [ids], tap_layers=taps_layers)
    # ---- B: oracle prefill of L-1 then one decode token -> must equal A's last token (incremental consistency)
    sB = SeqState()
    o.forward([sB], [ids[:-1]])
    rB = o.forward([sB], [ids[-1:]], tap_layers=taps_layers)
    for li in taps_layers:
        a = rA["taps"][(0, li)]["h_out"][-1]
        b = rB["taps"][(0, li)]["h_out"][-1]
        print(f"incremental vs full  layer {li}: rel err {rel(b, a):.2e}")
        assert rel(b, a) < 1e-4
    for li in (0, 1, 2):
        e = rel(sB.lin[li]["S"], sA.lin[li]["S"])
        print(f"  final DeltaNet state layer {li}: rel err {e:.2e}")
        assert e < 1e-4
    print(f"  KV layer 3: rel err {rel(sB.kv[3][0], sA.kv[3][0]):.2e}")

    # ---- C: HF modules, layer by layer, on the same inputs
    rot = Qwen3_5TextRotaryEmbedding(cfg)
    pos = torch.arange(L)[None]
    cos, sin = rot(torch.zeros(1, L, R.H), pos[None].expand(3, 1, L))
    mask = torch.full((L, L), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        h = o.embed(ids)[None]
        for li in taps_layers:
            layer = hf_layer(cfg, ck, f"model.language_model.layers.{li}.", li, li)
            h_in_oracle = rA["taps"][(0, li)]["h_in"]
            print(f"  layer {li} ({R.LAYER_TYPES[li]}): input matches oracle: rel {rel(h[0], h_in_oracle):.2e}", end="")
            out = layer(h, position_embeddings=(cos, sin), attention_mask=(mask if R.LAYER_TYPES[li] == 'full_attention' else None), position_ids=pos)
            e = rel(out[0], rA["taps"][(0, li)]["h_out"])
            print(f"   HF-vs-oracle output rel err {e:.2e}")
            assert e < 1e-3, (li, e)
            h = rA["taps"][(0, li)]["h_out"][None]  # feed oracle's own output so errors do not compound
            del layer

        # ---- D: MTP decoder layer vs HF full-attention layer with the mtp.layers.0 weights
        o.mtp  # loaded
        sM = SeqState()
        hid = rA["final"][0]  # [L,5120] (final-norm of layer-3 output here; arbitrary but realistic scale)
        tap = {}
        o.mtp_forward(sM.mtp_kv, hid[:-1], ids[1:], 0, tap=tap)
        mcfg = Qwen3_5TextConfig(**tcfg)
        mcfg._attn_implementation = "eager"
        ml = Qwen3_5DecoderLayer(mcfg, 3).float().eval()
        sd = {k[len("layers.0."):]: v.float() for k, v in ((n[len("mtp."):], ck.get(n)) for n in ck.names("mtp.layers.0."))}
        ml.load_state_dict(sd, strict=True)
        Lm = L - 1
        cos, sin = rot(torch.zeros(1, Lm, R.H), torch.arange(Lm)[None][None].expand(3, 1, Lm))
        mask = torch.full((Lm, Lm), float("-inf")).triu(1)[None, None]
        out = ml(tap["fc_out"][None], position_embeddings=(cos, sin), attention_mask=mask, position_ids=torch.arange(Lm)[None])
        e = rel(out[0], tap["layer_out"])
        print(f"  MTP decoder layer HF-vs-oracle rel err {e:.2e}")
        assert e < 1e-3

    # ---- E: chunk == recurrent delta rule on random data
    g = torch.Generator().manual_seed(0)
    Lr = 150
    q = R.l2norm(torch.randn(Lr, R.NV, R.DK, generator=g)); k = R.l2norm(torch.randn(Lr, R.NV, R.DK, generator=g))
    v = torch.randn(Lr, R.NV, R.DV, generator=g); gg = -torch.rand(Lr, R.NV, generator=g) * 0.5; be = torch.rand(Lr, R.NV, generator=g)
    S1, S2 = torch.zeros(R.NV, R.DK, R.DV), torch.zeros(R.NV, R.DK, R.DV)
    o1, o2 = R.delta_recurrent(q, k, v, gg, be, S1), R.delta_chunk(q, k, v, gg, be, S2)
    print(f"  delta rule chunk-vs-recurrent: out rel {rel(o2, o1):.2e}  state rel {rel(S2, S1):.2e}")
    assert rel(o2, o1) < 1e-4 and rel(S2, S1) < 1e-4
    print("check_vs_hf OK")


if __name__ == "__main__":
    main()
