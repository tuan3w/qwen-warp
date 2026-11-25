"""The MTP head's re-quantisation (sources/requantize.py) against the hand engine's pack_mtp.

uv run --package warpc python warpc/tests/test_requantize.py                     # CPU tests
WARPC_LEGACY_DIR=<v0.12.0> uv run --package warpc python warpc/tests/test_requantize.py
    # the CPU tests also run the hand engine's pack_mtp.quant_w4 and compare the draft vocabulary with its file
WARPC_LEGACY_DIR=<v0.12.0> WARPC_TEST_SWIFT27=<27B checkpoint directory> \\
    uv run --package warpc python warpc/tests/test_requantize.py
    # adds the real-data q|k|v check and G1 (design 4.1): every MTP pack of warpc's swiftmk Packer, built from
    # Requantized.mtp, is byte-equal to the hand engine's (needs a CUDA device)
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch

from harness import Skip, legacy, run
from warpc import sources
from warpc.compiler import Pack
from warpc.errors import UnsupportedModel
from warpc.ir import WeightDecl
from warpc.sources import ModelConfig, Requantized, draft_vocab, quantize_w4
from warpc.sources.requantize import DRAFT_HEAD, MTP_MATRICES
from warpc.swiftmk.pack import Packer, w4_scale_offset

VOCAB_SHA256 = "e05393340fbd26043a9641f3a84036f8db37f165580df2bffc76a0384cded2ec"
GROUP = 128


def _hand_quant_w4(w, clip=True, rows_per_chunk=None, device="cuda"):
    """engine/pack_mtp.py:quant_w4 of v0.12.0 verbatim (used when $WARPC_LEGACY_DIR is not set)."""
    N, K = w.shape
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
        word = (nib << shifts).sum(-1)
        word = torch.where(word >= 2**31, word - 2**32, word)
        out_w[r0 : r0 + n] = word.to(torch.int32)
        out_s[r0 : r0 + n] = best_s.squeeze(-1).to(torch.bfloat16)
    return out_w, out_s


def _hand_modules():
    """(pack_mtp, pack) of the hand engine (the v0.12.0 checkout's engine/), or None when $WARPC_LEGACY_DIR is unset."""
    try:
        engine = legacy().engine_dir()
    except Skip:
        return None
    sys.path.insert(0, str(engine))
    try:
        import pack
        import pack_mtp
    finally:
        sys.path.remove(str(engine))
    return pack_mtp, pack


def _hand_quant():
    mods = _hand_modules()
    return mods[0].quant_w4 if mods else _hand_quant_w4


def _checkpoint() -> Path:
    p = os.environ.get("WARPC_TEST_SWIFT27")
    if not p:
        raise Skip("WARPC_TEST_SWIFT27 (the 27B checkpoint directory) not set")
    return Path(p)


def _random_matrix(n: int, k: int, seed: int) -> torch.Tensor:
    """bf16 weights with outlier channels (a few columns 30x larger), the shape of real LLM matrices."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=gen) * 0.02
    w[:, torch.randint(0, k, (6,), generator=gen)] *= 30
    w[torch.randint(0, n, (3,), generator=gen)] *= 5
    return w.to(torch.bfloat16)


def _same(a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]) -> bool:
    return torch.equal(a[0].cpu(), b[0].cpu()) and torch.equal(a[1].cpu(), b[1].cpu())


def test_draft_vocab():
    ids = draft_vocab()
    assert ids.dtype == np.int64 and ids.shape == (68864,) and bool(np.all(np.diff(ids) > 0))
    path = Path(sources.requantize.__file__).parent / "data" / "draft_vocab_ext_68864.npy"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == VOCAB_SHA256
    try:
        hand = legacy().engine_dir() / "draft_vocab_ext_68864.npy"
    except Skip:
        return
    assert hand.read_bytes() == path.read_bytes()


def test_quantize_matches_hand_on_cpu():
    """Same code on the CPU as the hand's quant_w4(device="cpu"): codes and scales equal, with and without the clip
    search, across several chunks (K = 8192: 488 rows per chunk) and a matrix with 16 rows."""
    hand = _hand_quant()
    for n, k, seed in ((1100, 8192, 0), (16, 256, 1), (40, 1024, 2)):
        w = _random_matrix(n, k, seed)
        for clip in (True, False):
            got = quantize_w4(w, clip=clip, device="cpu")
            assert got[0].dtype == torch.int32 and got[1].dtype == torch.bfloat16 and got[0].device.type == "cpu"
            assert got[0].shape == (n, k // 8) and got[1].shape == (n, k // GROUP)
            assert _same(got, hand(w, clip, device="cpu")), (n, k, clip)
    for bad in (torch.zeros(4, 100), torch.zeros(128)):
        try:
            quantize_w4(bad, clip=True, device="cpu")
        except ValueError:
            pass
        else:
            raise AssertionError("a matrix whose K is not a multiple of 128 must be refused")


def test_quantize_matches_hand_on_gpu():
    """The GPU path is the hand's CUDA path bit for bit; the CPU result differs only where two clip ratios' squared
    errors tie to the last float bit (error sums add in another order)."""
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    hand = _hand_quant()
    w = _random_matrix(1100, 8192, 3)
    for clip in (True, False):
        gpu = quantize_w4(w, clip=clip, device="cuda")
        assert _same(gpu, hand(w, clip, device="cuda")), clip
        cpu = quantize_w4(w, clip=clip, device="cpu")
        differing = int((gpu[1] != cpu[1]).sum())
        print(f"  clip={clip}: {differing} of {gpu[1].numel()} group scales differ between the CPU and the GPU")
        assert differing <= gpu[1].numel() // 1000


def test_quantize_semantics():
    """The format itself: nibble e of word j is column 8j+e, dequant (nibble - 8) * scale, int4 range, and the clip
    search never increases a group's squared error."""
    w = _random_matrix(32, 512, 4)
    errs = {}
    for clip in (False, True):
        packed, scale = quantize_w4(w, clip=clip, device="cpu")
        sh = 4 * torch.arange(8, dtype=torch.int64)
        nib = ((packed.to(torch.int64) & 0xFFFFFFFF).unsqueeze(-1) >> sh) & 0xF
        deq = ((nib - 8).float().view(32, -1, GROUP) * scale.float().unsqueeze(-1)).view(32, 512)
        errs[clip] = (deq - w.float()).view(32, -1, GROUP).pow(2).sum(-1)
        if not clip:
            amax = w.float().view(32, -1, GROUP).abs().amax(-1)
            assert torch.equal(scale, (amax / 7.5).to(torch.bfloat16))
    assert bool((errs[True] <= errs[False]).all()) and bool((errs[True] < errs[False]).any())


class _MemorySource:
    """A Source of bf16 tensors held in memory that counts get() calls."""

    def __init__(self, tensors: dict[str, torch.Tensor], vocab: int):
        text = {
            "model_type": "qwen3_5_text",
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 64,
            "linear_value_head_dim": 64,
            "linear_conv_kernel_dim": 4,
            "num_hidden_layers": 2,
            "attn_output_gate": True,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25},
            "vocab_size": vocab,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rms_norm_eps": 1e-6,
            "mtp_num_hidden_layers": 1,
        }
        self.config = ModelConfig.from_hf({"model_type": "qwen3_5", "text_config": text})
        self.tensors, self.gets = tensors, 0

    def __contains__(self, name):
        return name in self.tensors

    def get(self, name, dtype=torch.bfloat16):
        self.gets += 1
        return self.tensors[name].to(dtype)

    def quant(self, name):
        if name not in self.tensors:
            raise KeyError(name)
        return None

    def raw(self, name):
        raise ValueError(f"{name}: a float tensor has no on-disk codes")


def test_requantized_source():
    vocab = 300
    tensors = {
        "lm_head.weight": _random_matrix(vocab, 256, 5),
        "embed.weight": _random_matrix(vocab, 256, 6),
        "norm.weight": _random_matrix(1, 256, 7)[0],
        "m.a": _random_matrix(32, 256, 8),
        "m.b": _random_matrix(16, 128, 9),
    }
    base = _MemorySource(tensors, vocab)
    ids = np.sort(np.random.default_rng(0).choice(vocab, 70, replace=False)).astype(np.int64)
    rq = Requantized(base, matrices=["m.a"], draft_ids=ids, clip=True)
    assert base.gets == 0, "construction must not quantise"
    assert rq.config is base.config and rq.draft_ids is ids
    # forwarded names
    for name in ("m.b", "embed.weight", "lm_head.weight", "norm.weight"):
        assert name in rq and rq.quant(name) is None
        assert torch.equal(rq.get(name), tensors[name])
    assert torch.equal(rq.get("norm.weight", torch.float32), tensors["norm.weight"].float())
    assert "nope" not in rq
    try:
        rq.quant("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown name is the base's KeyError")
    try:
        rq.raw("m.b")
    except ValueError:
        pass
    else:
        raise AssertionError("raw of a forwarded float tensor is the base's ValueError")
    # the matrix
    assert "m.a" in rq and rq.quant("m.a") == "W4G128"
    raw = rq.raw("m.a")
    assert set(raw) == {"packed", "scale"} and _same(
        (raw["packed"], raw["scale"]), quantize_w4(tensors["m.a"], clip=True)
    )
    # the draft head: gathered rows, no clip search
    assert DRAFT_HEAD in rq and rq.quant(DRAFT_HEAD) == "W4G128"
    head = rq.raw(DRAFT_HEAD)
    rows = tensors["lm_head.weight"][torch.from_numpy(ids)]
    assert head["packed"].shape == (70, 32) and _same((head["packed"], head["scale"]), quantize_w4(rows, clip=False))
    for name in ("m.a", DRAFT_HEAD):
        try:
            rq.get(name)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name}: get() of a W4G128 tensor must raise ValueError")
    # clip=False applies to the matrices
    plain = Requantized(base, matrices=["m.a"], draft_ids=ids, clip=False).raw("m.a")
    assert _same((plain["packed"], plain["scale"]), quantize_w4(tensors["m.a"], clip=False))
    # refusals
    for kwargs, exc in (
        ({"matrices": ["missing"], "draft_ids": ids}, UnsupportedModel),
        ({"matrices": ["m.a"], "draft_ids": ids[::-1].copy()}, ValueError),
        ({"matrices": ["m.a"], "draft_ids": ids.astype(np.int32)}, ValueError),
        ({"matrices": ["m.a"], "draft_ids": np.array([0, vocab], dtype=np.int64)}, ValueError),
    ):
        try:
            Requantized(base, **kwargs)
        except exc:
            pass
        else:
            raise AssertionError(f"{kwargs['matrices']}: accepted")
    try:
        Requantized.mtp(base)
    except UnsupportedModel as e:
        assert "MTP" in str(e), str(e)
    else:
        raise AssertionError("a checkpoint without mtp.* tensors must raise UnsupportedModel")


def test_qkv_per_matrix_equals_concatenated():
    """q, k, v quantised one by one are byte-identical to the hand's quantisation of their concatenation."""
    q, k, v = (_random_matrix(n, 1024, 10 + i) for i, n in enumerate((384, 64, 64)))
    for clip in (True, False):
        whole = quantize_w4(torch.cat([q, k, v]), clip=clip, device="cpu")
        parts = [quantize_w4(m, clip=clip, device="cpu") for m in (q, k, v)]
        assert _same(whole, (torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])))


def test_mtp_layers_config():
    base = _MemorySource({}, 300)
    assert base.config.family == "qwen3_5" and base.config.mtp_layers == 1
    dense = ModelConfig.from_hf(
        {
            "model_type": "qwen3",
            "vocab_size": 300,
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 1e6,
            "mtp_num_hidden_layers": 1,
        }
    )
    assert dense.mtp_layers == 0, "only the qwen3_5 family reads mtp_num_hidden_layers"


def test_27b_mtp_layers_and_qkv_on_real_data():
    src = sources.open(_checkpoint())
    assert src.config.mtp_layers == 1
    rq = Requantized.mtp(src)
    assert rq.draft_ids.shape == (68864,) and int(rq.draft_ids[-1]) < src.config.vocab_size
    base = "mtp.layers.0.self_attn."
    mats = [src.get(f"{base}{p}_proj.weight") for p in "qkv"]
    whole = quantize_w4(torch.cat(mats), clip=True)
    parts = [rq.raw(f"{base}{p}_proj.weight") for p in "qkv"]
    assert _same(whole, (torch.cat([p["packed"] for p in parts]), torch.cat([p["scale"] for p in parts])))


def test_g1_mtp_packs_equal_the_hand_engine():
    """G1 (design 4.1): each MTP pack built by warpc's Packer from Requantized.mtp is byte-equal to what the hand
    engine's SpecEngine.__init__ holds after pack.repack_w4 (pack_mtp.load_mtp's quantisation, the gate / up
    interleave per 8 rows, the head clone), and the draft ids are the hand's ids_t."""
    path = _checkpoint()
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")
    mods = _hand_modules()
    if mods is None:
        raise Skip("WARPC_LEGACY_DIR unset")
    hand_mtp, hand_pack = mods
    src = sources.open(path)
    H, INTER = 5120, 17408
    L = "mtp.layers.0."

    def g(name):
        return src.get("mtp." + name)

    # the hand's load_mtp + SpecEngine.__init__, on the checkpoint's MTP tensors and the host-gathered lm_head rows
    quant = hand_mtp.quant_w4
    qkv_w = torch.cat([g(f"layers.0.self_attn.{p}_proj.weight") for p in "qkv"])
    gate, up = quant(g("layers.0.mlp.gate_proj.weight"), True), quant(g("layers.0.mlp.up_proj.weight"), True)
    gu_w = torch.empty(2 * INTER, H // 8, dtype=torch.int32, device="cuda")
    gu_s = torch.empty(2 * INTER, H // GROUP, dtype=torch.bfloat16, device="cuda")
    for dst, a, b in ((gu_w, gate[0], up[0]), (gu_s, gate[1], up[1])):
        v = dst.view(INTER // 8, 2, 8, dst.shape[1])
        v[:, 0].copy_(a.view(INTER // 8, 8, -1))
        v[:, 1].copy_(b.view(INTER // 8, 8, -1))
    del gate, up
    ids = hand_mtp.draft_vocab()
    head_rows = src.get("lm_head.weight")[torch.from_numpy(ids)]
    hand = {
        "fc": quant(g("fc.weight"), True),
        "qkv": quant(qkv_w, True),
        "o": quant(g("layers.0.self_attn.o_proj.weight"), True),
        "gu": (gu_w, gu_s),
        "down": quant(g("layers.0.mlp.down_proj.weight"), True),
        "head": quant(head_rows, False),
    }
    del qkv_w, head_rows
    for w, s in hand.values():
        hand_pack.repack_w4(w, s)

    rq = Requantized.mtp(src)
    assert np.array_equal(rq.draft_ids, ids), "the draft vocabulary differs from the hand's"
    assert torch.equal(torch.from_numpy(rq.draft_ids.astype(np.int32)), torch.from_numpy(ids.astype(np.int32)))
    shapes = {
        "mtp.fc.weight": (H, 2 * H),
        f"{L}self_attn.q_proj.weight": (12288, H),
        f"{L}self_attn.k_proj.weight": (1024, H),
        f"{L}self_attn.v_proj.weight": (1024, H),
        f"{L}self_attn.o_proj.weight": (H, 6144),
        f"{L}mlp.gate_proj.weight": (INTER, H),
        f"{L}mlp.up_proj.weight": (INTER, H),
        f"{L}mlp.down_proj.weight": (H, INTER),
        DRAFT_HEAD: (68864, H),
    }
    assert set(shapes) == {*MTP_MATRICES, DRAFT_HEAD}
    weights = {n: WeightDecl(n, s, "w4g128", n) for n, s in shapes.items()}
    packer = Packer(weights, rq)
    qkv = [f"{L}self_attn.{p}_proj.weight" for p in "qkv"]
    gu = [f"{L}mlp.gate_proj.weight", f"{L}mlp.up_proj.weight"]
    packs = {
        "fc": Pack("fc", "w4r", ["mtp.fc.weight"]),
        "qkv": Pack("qkv", "w4r_cat", qkv),
        "o": Pack("o", "w4r", [f"{L}self_attn.o_proj.weight"]),
        "gu": Pack("gu", "w4r_gateup", gu),
        "down": Pack("down", "w4r", [f"{L}mlp.down_proj.weight"]),
        "head": Pack("head", "w4r", [DRAFT_HEAD]),
    }
    for name, pack in packs.items():
        w, s = hand[name]
        built = packer.build(pack)
        off = w4_scale_offset(w.shape[0], w.shape[1] * 8)
        assert built.numel() == off + s.numel() * 2, (name, built.numel())
        ok_w = torch.equal(built[:off], w.view(torch.uint8).reshape(-1))
        ok_s = torch.equal(built[off:], s.view(torch.uint8).reshape(-1))
        assert ok_w and ok_s, f"{name}: words equal {ok_w}, scales equal {ok_s}"
        print(f"  {name}: {tuple(w.shape)} words + {tuple(s.shape)} scales, {built.numel() / 2**20:.1f} MiB byte-equal")
        del built


if __name__ == "__main__":
    run(globals())
