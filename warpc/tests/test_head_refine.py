"""Family 7, the int4 head with exact rescoring (warpc/cuda/head_refine.py, kernels/qwen35_head_refine.cu), against the
hand engine's smk_headq_gemv(I4R) + smk_headq_select_rescore.

  pack:    head_i4g is byte for byte the hand's head_i4r (CPU)
  match:   the capability claims the 27B head like s_head_refine, and only in table("int4_refine") (CPU)
  select:  on synthetic logits (ties, flat clusters, -inf, sorted) the candidate ids, the exact logits, the logits
           buffer and the token ids equal the hand's bit for bit, T = 1..8
  pair:    the approximate GEMV + select on a random 248320 x 5120 head: logits buffer, candidates, exact logits, ids
           bitwise, T = 1..8
  speed:   graph time of the pair and of the select alone, against the hand's

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> WARPC_TEST_SWIFT27=<27B checkpoint dir> \\
      uv run --package warpc python warpc/tests/test_head_refine.py

The GPU tests need a CUDA device and $WARPC_SWIFTMK_LIB; they skip without. The speed test needs $WARPC_HEAD_SPEED.
"""

import ctypes as C
import os
from pathlib import Path

import torch

from ab_util import ab_us, stream
from harness import Skip, run
from test_compiler import Qwen35Shapes
from test_w4 import _device, _hand
from warpc import frontend, sources
from warpc.compiler import CapsTable, Pack
from warpc.compiler.fuse import fuse
from warpc.cuda import head_refine, w4
from warpc.cuda.pack import Packer
from warpc.cuda.table import RTX3090, TABLE, table
from warpc.ir import WeightDecl

N, K, R = 248320, 5120, head_refine.R
HAND_CAND, HAND_EXACT, HAND_ROWS = 8320, 12416, 16  # head_q.cu WsLayout: byte offsets of cand / exact (rows 16)


class _Dense:
    def __init__(self, w: torch.Tensor):
        self.w = w

    def get(self, name: str) -> torch.Tensor:
        return self.w


def _packs(w: torch.Tensor, device: str):
    """(mine, hand) int4 head packs of the bf16 head `w`."""
    from warpc.swiftmk.pack import Packer as HandPacker

    decls = {"head": WeightDecl("head", tuple(w.shape), "dense", "head")}
    mine = Packer(decls, _Dense(w), device).build(Pack("head.i4g", "head_i4g", ["head"]))
    hand = HandPacker(decls, _Dense(w), device).build(Pack("head.i4r", "head_i4r", ["head"]))
    return mine, hand


def test_pack_is_the_hand_head_i4r():
    gen = torch.Generator().manual_seed(1)
    w = (torch.randn(384, 512, generator=gen) * 0.05).to(torch.bfloat16)
    mine, hand = _packs(w, "cpu")
    assert torch.equal(mine, hand)


def _regions(caps, g):
    return fuse(g, CapsTable(caps, RTX3090, TABLE.unfused_ctas), 8).regions


def test_cap_claims_the_head_like_the_hand():
    from warpc.swiftmk import table as hand

    path = os.environ.get("WARPC_TEST_SWIFT27")
    g = frontend.build(sources.open(Path(path)) if path else Qwen35Shapes(), ctx_max=4608, kv="bf16")
    assert "g_head_refine" not in {r.kernel for r in fuse(g, table("bf16"), 8).regions}
    (mine,) = _regions(head_refine.CAPS, g)
    (theirs,) = _regions([c for c in hand.CAPS if c.kernel == "s_head_refine"], g)
    assert sorted(mine.covers) == sorted(theirs.covers) and mine.ins == theirs.ins and mine.outs == theirs.outs
    assert mine.attrs == theirs.attrs and mine.tolerance == theirs.tolerance, (mine.attrs, mine.tolerance)
    assert [p.kind for p in mine.packs] == ["dense", "head_i4g"], mine.packs
    assert mine.weights["w"] == theirs.weights["w"]


def _ws_arrays(ws: torch.Tensor, cand: int, exact: int, rows: int, T: int):
    i32 = ws.view(torch.int32)
    return (
        i32[cand // 4 : cand // 4 + rows * 64].view(rows, 64)[:T, :R].clone(),
        i32[exact // 4 : exact // 4 + rows * 64].view(rows, 64)[:T, :R].clone(),
    )


def _synthetic(T: int, gen) -> torch.Tensor:
    """Logit rows [T, N] that stress the select: ties, a flat cluster, -inf, sorted, a constant row."""
    rows = []
    for t in range(T):
        kind = t % 8
        v = torch.randn(N, generator=gen)
        if kind == 1:
            v = torch.round(v * 20) / 20  # ties at the threshold
        elif kind == 2:
            v = torch.full((N,), 3.25)  # flat: the lowest indices win
        elif kind == 3:
            v[5000:8000] = 9.0  # a flat cluster of 3000 maxima in two chunks
        elif kind == 4:
            v = torch.randn(N, generator=gen) * 0.01
            v[torch.arange(40) * 6000 + 17] = 7.0  # the maxima tie across 40 chunks
        elif kind == 5:
            v[torch.randperm(N, generator=gen)[:200000]] = float("-inf")
        elif kind == 6:
            v = torch.zeros(N)
            v[torch.randperm(N, generator=gen)[:5]] = torch.tensor(
                [5.0, 4.0, 3.0, 2.0, 1.0]
            )  # 5 spikes over a field of ties
        elif kind == 7:
            v = torch.arange(N, dtype=torch.float32)  # sorted, the top in the last chunk
        rows.append(v)
    return torch.stack(rows)


def _head_data(gen):
    w = (torch.randn(N, K, generator=gen, device="cuda") * 0.03).to(torch.bfloat16)
    x = torch.randn(8, K, generator=gen, device="cuda").to(torch.bfloat16)
    return w, x


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().view(torch.int32)


def _sel_mine(lib, logits, w, x, T, tok, ws) -> int:
    args = (logits.data_ptr(), w.data_ptr(), x.data_ptr(), N, K, T, R, tok.data_ptr(), ws.data_ptr(), stream())
    return lib.warpc_headq_select(*args)


def _sel_hand(hand, logits, w, x, T, tok, ws) -> int:
    args = (logits.data_ptr(), w.data_ptr(), x.data_ptr(), N, K, T, R, tok.data_ptr(), None, ws.data_ptr(), stream())
    return hand.smk_headq_select_rescore(*args)


def _gemv_mine(lib, pack, x, T, logits) -> int:
    outs = (C.c_void_p * 3)()
    outs[0] = logits.data_ptr()
    p = pack.data_ptr()
    args = (p, p + w4.scale_offset(N, K), x.data_ptr(), N, K, T, w4.EPI_F32, None, outs, head_refine.KS, stream())
    return lib.warpc_w4g128_gemv(*args)


def _gemv_hand(hand, pack, x, T, logits) -> int:
    p = pack.data_ptr()
    return hand.smk_headq_gemv(3, p, p + w4.scale_offset(N, K), x.data_ptr(), N, K, T, logits.data_ptr(), stream())


class _Ws:
    """The scratch of both libraries and where their candidate / exact arrays sit in it."""

    def __init__(self, lib, hand):
        self.m = torch.zeros(int(lib.warpc_headq_ws_bytes()), dtype=torch.uint8, device="cuda")
        self.h = torch.zeros(int(hand.smk_headq_ws_bytes()), dtype=torch.uint8, device="cuda")
        base = self.m.data_ptr()
        self.cand_m = C.cast(lib.warpc_headq_ws_cand(base), C.c_void_p).value - base
        self.exact_m = C.cast(lib.warpc_headq_ws_exact(base), C.c_void_p).value - base

    def compare(self, T, lm, lh, tok_m, tok_h) -> dict:
        cm, em = _ws_arrays(self.m, self.cand_m, self.exact_m, 8, T)
        ch, eh = _ws_arrays(self.h, HAND_CAND, HAND_EXACT, HAND_ROWS, T)
        return {
            "candidates": int((cm != ch).sum()),
            "exact": int((_bits(em) != _bits(eh)).sum()),
            "logits": int((_bits(lm) != _bits(lh)).sum()),
            "ids": int((tok_m != tok_h).sum()),
        }


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.cpu().view(torch.int32)


def test_select_equals_the_hand():
    lib, hand = _device(), _hand()
    gen = torch.Generator(device="cuda").manual_seed(5)
    w, x = _head_data(gen)
    cgen = torch.Generator().manual_seed(6)
    ws = _Ws(lib, hand)
    for T in (1, 2, 3, 8):
        base = _synthetic(8, cgen)[:T].cuda()
        lm, lh = base.clone(), base.clone()
        tok_m = torch.full((T,), -1, dtype=torch.int32, device="cuda")
        tok_h = tok_m.clone()
        assert _sel_mine(lib, lm, w, x, T, tok_m, ws.m) == 0
        hand.check(_sel_hand(hand, lh, w, x, T, tok_h, ws.h), "hand select")
        torch.cuda.synchronize()
        bad = ws.compare(T, lm, lh, tok_m, tok_h)
        print(f"select T={T}: differing {bad}")
        assert not any(bad.values()), bad


def test_pair_equals_the_hand():
    lib, hand = _device(), _hand()
    gen = torch.Generator(device="cuda").manual_seed(7)
    w, x = _head_data(gen)
    mine, theirs = _packs(w, "cuda")
    ws = _Ws(lib, hand)
    for T in (1, 2, 4, 8):
        lm = torch.empty(T, N, dtype=torch.float32, device="cuda")
        lh = torch.empty_like(lm)
        tok_m = torch.empty(T, dtype=torch.int32, device="cuda")
        tok_h = torch.empty_like(tok_m)
        assert _gemv_mine(lib, mine, x, T, lm) == 0
        assert _sel_mine(lib, lm, w, x, T, tok_m, ws.m) == 0
        hand.check(_gemv_hand(hand, theirs, x, T, lh), "hand gemv")
        hand.check(_sel_hand(hand, lh, w, x, T, tok_h, ws.h), "hand select")
        torch.cuda.synchronize()
        bad = ws.compare(T, lm, lh, tok_m, tok_h)
        print(f"pair T={T}: differing {bad} (of {T * N} logits)")
        assert not any(bad.values()), bad


def test_speed_against_the_hand():
    if not os.environ.get("WARPC_HEAD_SPEED"):
        raise Skip("WARPC_HEAD_SPEED not set")
    lib, hand = _device(), _hand()
    gen = torch.Generator(device="cuda").manual_seed(8)
    w, x = _head_data(gen)
    mine, theirs = _packs(w, "cuda")
    ws = _Ws(lib, hand)
    logits = torch.randn(8, N, device="cuda")  # the select only rewrites the 16 candidates of a row
    tok = torch.empty(8, dtype=torch.int32, device="cuda")
    print("   T  pair warpc us  hand us  ratio | select warpc us  hand us  ratio")
    for T in (1, 2, 4, 8):

        def sel_m(T=T):
            _sel_mine(lib, logits, w, x, T, tok, ws.m)

        def sel_h(T=T):
            _sel_hand(hand, logits, w, x, T, tok, ws.h)

        def pair_m(T=T):
            _gemv_mine(lib, mine, x, T, logits)
            sel_m(T)

        def pair_h(T=T):
            _gemv_hand(hand, theirs, x, T, logits)
            sel_h(T)

        pm, ph = ab_us(pair_m, pair_h)
        sm, sh = ab_us(sel_m, sel_h)
        print(f"{T:4d} {pm:12.1f} {ph:9.1f} {ph / pm:6.3f} | {sm:14.1f} {sh:9.1f} {sh / sm:6.3f}")


if __name__ == "__main__":
    run(globals())
