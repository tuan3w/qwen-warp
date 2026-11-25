"""MTP speculative decoding: the device bookkeeping of a cycle, the exactness and resume properties on the 27B.

  WARPC_SWIFTMK_LIB=<engine/build/libswiftmk.so> WARPC_TEST_SWIFT27=<27B checkpoint> \
      uv run --package warpc python warpc/tests/test_speculate.py

The accept program and the ring cursors need a CUDA device; the 27B tests also the checkpoint and the hand engine's
library, and skip without them. The bitwise comparison with the hand engine's MTP cycles (drafts, verify logits, every
cycle's tokens) is tools/gate27.py --mtp.
"""

import gc
import itertools
import os
import random
import sys
from pathlib import Path

import torch

import warpc
from harness import Skip, run
from warpc import speculate
from warpc.composite import CompositeBackend
from warpc.cuda import CudaBackend
from warpc.speculate import RING, Speculator
from warpc.swiftmk import SwiftmkBackend
from warpc.swiftmk import lib as smk
from warpc.swiftmk.executor import LAZY_ROWS, ring_commit

GLUE = ["g_select_row", "g_take", "g_concat"]
SOURCES = sorted((Path(__file__).resolve().parents[2] / "engine" / "src").glob("*.cu"))


def _cuda() -> None:
    if not torch.cuda.is_available():
        raise Skip("no CUDA device")


# ------------------------------------------------------------------------------------------------ device bookkeeping


def dn_pend_after(pend: int, cur: int, a: int, T: int, slots: int) -> tuple[int, int]:
    """engine_impl.h dn_pend_after + the cursor advance of spec_finish / k_commit, as plain Python."""
    if T < 2:
        pend = 0
    else:
        pc, base, half = pend & 255, (pend >> 8) & 31, (pend >> 13) & 1
        if pc == 0:
            base = cur
        elif pc + T > LAZY_ROWS:
            pc, base, half = 0, cur, half ^ 1
        pend = (pc + a) | (base << 8) | (half << 13)
    return pend, (cur + a) % slots


def test_ring_commit_matches_the_hand_bookkeeping() -> None:
    """Random steps of T rows of which a are kept (a < T included, and a = 0), across the 16-row delta buffer's
    flushes and wraps: the cursors equal the hand engine's bookkeeping after every step."""
    _cuda()
    rng, slots = random.Random(0), 9
    cursors = torch.zeros(2, dtype=torch.int32, device="cuda")
    cur, pend = cursors[0:1], cursors[1:2]
    want_pend, want_cur = 0, 0
    flushes = 0
    for _ in range(4000):
        T = rng.randint(1, 8)
        a = rng.randint(0, T) if rng.random() < 0.7 else T
        before = want_pend
        want_pend, want_cur = dn_pend_after(want_pend, want_cur, a, T, slots)
        flushes += T >= 2 and (before & 255) > 0 and (before & 255) + T > LAZY_ROWS
        ring_commit(cur, pend, torch.full((1,), a, dtype=torch.int32, device="cuda"), T, slots)
        assert (int(cur), int(pend)) == (want_cur, want_pend), (T, a)
    assert flushes > 50, flushes  # the flush corner was exercised


class _Stub:
    """What `Speculator._accept` reads of a Runtime: tensors and `commit`."""

    def __init__(self, rows: int, hidden: int, seed: int):
        gen = torch.Generator().manual_seed(seed)
        self.inputs = {
            "tokens": torch.zeros(rows, dtype=torch.int32, device="cuda"),
            "hidden": torch.zeros(rows, hidden, dtype=torch.bfloat16, device="cuda"),
            "sel": torch.zeros(1, dtype=torch.int32, device="cuda"),
        }
        self.outputs = {
            "ids": torch.zeros(rows, dtype=torch.int32, device="cuda"),
            "hidden": torch.randn(rows, hidden, generator=gen).to("cuda", torch.bfloat16),
        }
        self.pos = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.committed: list[int] = []

    def commit(self, T: int, accepted: torch.Tensor) -> None:
        self.committed.append(int(accepted))
        self.pos.add_(accepted)


def _bare_speculator(t: _Stub, d: _Stub, k_max: int) -> Speculator:
    """A Speculator with the device state `_accept` uses and nothing else (no runtimes, no graphs)."""
    s = object.__new__(Speculator)
    dev = {"device": "cuda", "dtype": torch.int32}
    s._target, s._draft, s.k_max = t, d, k_max
    s._st, s._ring = torch.zeros(speculate.ST_N, **dev), torch.zeros(RING, **dev)
    s._log, s._cap = torch.zeros(speculate.LOG, 2, **dev), torch.zeros(speculate.CAP_N, **dev)
    s._stop = torch.full((speculate.STOP_N,), -1, **dev)
    s._arange = torch.arange(8, **dev)
    s._last_hidden = torch.zeros(1, t.outputs["hidden"].shape[-1], device="cuda", dtype=torch.bfloat16)
    return s


def test_accept_program_matches_a_python_model() -> None:
    """Every draft / argmax pattern over a two-letter alphabet for k = 1, 2, 3, with forced-acceptance patterns, token
    budgets, stop ids, ring positions at the wrap and cycle counters: the device program's emitted tokens, anchor,
    positions, counters, histogram, log and commit equal a Python transcription of spec_finish and the clamp."""
    _cuda()
    hidden = 64
    checked = 0
    for k in (1, 2, 3):
        T = k + 1
        for drafts, argmax in itertools.product(
            itertools.product((0, 1), repeat=k), itertools.product((0, 1), repeat=T)
        ):
            for cap, budget, stop, outn, ncyc in itertools.product(
                (None, [1], [2, 1, 3][: k + 1], [T]), (0, 1, 2, 3, 100), ((), (1,)), (0, RING - 2), (0, 3)
            ):
                t, d = _Stub(8, hidden, checked), _Stub(8, hidden, 99)
                s = _bare_speculator(t, d, 3)
                pat = [T] if cap is None else [min(v, T) for v in cap]
                s._cap[: len(pat)] = torch.tensor(pat, dtype=torch.int32)
                s._st[speculate.S_CAPN] = len(pat)
                s._stop[: len(stop)] = torch.tensor(list(stop), dtype=torch.int32) if stop else s._stop[:0]
                s._st[speculate.S_BUDGET], s._st[speculate.S_OUTN], s._st[speculate.S_NCYC] = budget, outn, ncyc
                P = 40 + checked % 7
                t.pos.fill_(P)
                t.inputs["tokens"][:T] = torch.tensor([7, *drafts], dtype=torch.int32)
                t.outputs["ids"][:T] = torch.tensor(argmax, dtype=torch.int32)
                s._accept(k)
                # the model
                m = next((i for i in range(k) if drafts[i] != argmax[i]), k)
                a = min(m + 1, pat[ncyc % len(pat)], budget)
                stopped = next((i for i in range(a) if argmax[i] in stop), None)
                if stopped is not None:
                    a = stopped + 1
                live = budget > 0
                st = s._st.tolist()
                assert t.committed == [a], (k, drafts, argmax, cap, budget, stop, t.committed, a)
                assert st[speculate.S_OUTN] == outn + a and st[speculate.S_A] == a
                assert st[speculate.S_NCYC] == ncyc + live
                assert st[speculate.S_BUDGET] == (0 if stopped is not None else budget - a)
                hist = st[speculate.S_HIST : speculate.S_HIST + speculate.HIST_N]
                assert hist == [int(live and i == m) for i in range(speculate.HIST_N)], (hist, m, live)
                ring = s._ring.tolist()
                assert [ring[(outn + i) % RING] for i in range(a)] == list(argmax[:a])
                assert sum(1 for v in ring if v) <= a  # nothing else was written (ids 0 / 1: 1s only)
                assert int(t.inputs["tokens"][0]) == (argmax[a - 1] if a else 7)
                assert d.inputs["tokens"][:T].tolist() == list(argmax)
                assert int(d.inputs["sel"]) == max(a - 1, 0) and int(d.pos) == P
                want_row = t.outputs["hidden"][max(a - 1, 0)] if a else torch.zeros(hidden, dtype=torch.bfloat16)
                assert torch.equal(s._last_hidden[0].cpu(), want_row.cpu())
                if live:
                    assert s._log[ncyc % speculate.LOG].tolist() == [P, a]
                checked += 1
    assert checked > 10000, checked


# ------------------------------------------------------------------------------------------------ the 27B


def _checkpoint() -> Path:
    _cuda()
    if not os.environ.get(smk.ENV):
        raise Skip(f"${smk.ENV} not set")
    path = os.environ.get("WARPC_TEST_SWIFT27")
    if not path:
        raise Skip("WARPC_TEST_SWIFT27 not set")
    return Path(path)


_LIVE: dict = {}


def _model(kv: str, ctx_max: int = 4608, *, k: int = 3, host_tables: bool = False):
    """One compiled 27B at a time (each holds 18 GB)."""
    key = (kv, ctx_max, k, host_tables)
    if _LIVE.get("key") != key:
        _LIVE.clear()
        gc.collect()
        torch.cuda.empty_cache()
        backend = CompositeBackend(CudaBackend("int4_refine"), SwiftmkBackend("int4_refine"), kernels=GLUE)
        model = warpc.compile(
            _checkpoint(), ctx_max=ctx_max, kv=kv, backend=backend, speculate=k, host_tables=host_tables
        )
        _LIVE.update(key=key, model=model)
    return _LIVE["model"]


def _ids(n: int, slice_: int = 0) -> list[int]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import prompts

    return prompts.prompt_ids(_checkpoint(), SOURCES, n, slice_ * prompts.SLICE_STRIDE)


def _greedy(m, ids: list[int], n: int, k: int, cap=None) -> list[int]:
    m.reset()
    m.decoder.k, m.decoder.cap = k, cap
    out = m.generate(ids, n)
    assert m.position == len(ids) + n - 1
    return out


def test_k1_is_exact_and_so_is_a_clamp_to_one_row() -> None:
    """G-exact-k1 and G-clamp-k1: MTP with one draft (T = 2 verify rows) emits plain greedy's tokens, bitwise, up to the
    first cycle whose attention geometry differs between the T = 2 step and the T = 1 step (near_tie.exact_prefix; the
    hand engine diverges from its own plain decode at the same token, and gate27.py --mtp 1 is bitwise against it); so
    does k = 1 with every cycle clamped to one row. bf16 and int4_rot, three prompts at 385 and one at 4,096."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from near_tie import exact_prefix

    for kv in ("bf16", "int4_rot"):
        m = None  # the previous format's model (18 GB) must be freed before the next compile
        m = _model(kv)
        cells = [(385, s) for s in range(3)] + [(4096, 0)]
        for length, s in cells:
            ids = _ids(length, s)
            plain = _greedy(m, ids, 256, 0)
            for cap in (None, [1]):
                got = _greedy(m, ids, 256, 1, cap=cap)
                exact = exact_prefix(kv, m.decoder.cycle_log())
                assert got[:exact] == plain[:exact], (kv, length, s, cap, exact)
                first = next((i for i, (x, y) in enumerate(zip(got, plain, strict=True)) if x != y), None)
                print(f"  {kv} {length}/{s} cap {cap}: exact over {exact} tokens, first difference {first}")


def test_decode_leaves_the_cache_holding_exactly_the_emitted_tokens() -> None:
    """decode(n) returns n tokens for every n and leaves position == start + n; a decode ending at a stop id returns
    through it and stops there; the clamp equals forced acceptance: replaying the accepted counts of a decode as a cap
    pattern emits the same tokens."""
    m = _model("bf16")
    spec, ids = m.decoder, _ids(385)
    m.reset()
    first = m.append(ids)
    out, pos = [first], m.position
    for n in (1, 2, 3, 4, 5, 8, 17, 1, 63):
        got = m.decode(n)
        assert len(got) == n and m.position == pos + n, (n, len(got), m.position, pos)
        out += got
        pos += n
    # a stop id in the middle of a cycle
    full = _greedy(m, ids, 120, 3)
    for at in (5, 40, 41, 119):
        m.reset()
        spec.k, spec.cap = 3, None
        first = m.append(ids)
        stop = full[at]
        cut = full.index(stop)
        got = [first] if first == stop else [first, *m.decode(119, stop=[stop])]
        assert got == full[: cut + 1], (at, cut, got[-3:], full[cut - 2 : cut + 1])
        assert m.position == len(ids) + len(got) - 1, (m.position, len(got))
    # the clamp is forced acceptance
    m.reset()
    spec.k, spec.cap = 3, None
    first = m.append(ids)
    spec.reset_counters()
    got = m.decode(20)
    log = spec.cycle_log()
    assert sum(a for _, a in log) == 20 and len(log) <= speculate.CAP_N, log
    m.reset()
    first2 = m.append(ids)
    spec.cap = [a for _, a in log]
    again = spec.cycles(len(log))
    assert first2 == first and again == got, (got, again)
    assert [p for p, _ in spec.cycle_log()] == [p for p, _ in log]


def test_snapshot_restore_resumes_bitwise() -> None:
    """G-resume on both KV formats: append, decode, snapshot, decode, restore, decode equals the first decode after the
    snapshot; append(A + B) after a restore at len(A) equals a cold append(A + B); a snapshot without its KV aux
    (int4_rot) or its draft states is refused with ValueError."""
    for kv in ("bf16", "int4_rot"):
        m = None
        m = _model(kv)
        ids = _ids(448)
        m.reset()
        m.append(ids[:385])
        m.decode(64)
        snap = m.snapshot()
        a = m.decode(64)
        m.restore(snap)
        assert m.position == snap.position
        assert m.decode(64) == a, kv
        # resume == cold, on the same chunking (len(A) a multiple of 8)
        m.reset()
        cold_first = m.append(ids)
        cold = m.decode(64)
        m.reset()
        m.append(ids[:384], head=False)
        snap = m.snapshot()
        m.append(ids[384:], head=False)  # dirty the cache past the snapshot
        m.restore(snap)
        first = m.append(ids[384:])
        assert first == cold_first and m.decode(64) == cold, kv
        # a snapshot missing its aux state (the sink and ring of int4_rot) must not restore
        if kv == "int4_rot":
            assert any(key.endswith(".aux") for key in snap.states), sorted(snap.states)
            for key in [k for k in snap.states if k.endswith(".aux")][:1]:
                broken = type(snap)(snap.position, snap.anchor, snap.hidden, dict(snap.states), snap.device)
                del broken.states[key]
                try:
                    m.restore(broken)
                except ValueError as e:
                    assert key in str(e), e
                else:
                    raise AssertionError("restore accepted a snapshot without its aux state")
        print(f"  {kv}: resume equals continuing, cold and resumed append agree")


def test_kv_export_import_rebuilds_a_cache() -> None:
    """A cache emptied and rebuilt from kv_export rows + a snapshot continues bitwise (int4_rot: rows + sink + ring)."""
    for kv in ("bf16", "int4_rot"):
        m = None
        m = _model(kv)
        ids = _ids(385)
        m.reset()
        m.append(ids)
        snap = m.snapshot()
        chunk = m.kv_export(0, snap.position)
        want = m.decode(64)
        assert any(key.startswith("mtp.kv/") for key in chunk.planes), sorted(chunk.planes)
        m.reset()
        m.kv_import(chunk, 0)
        m.restore(snap)
        assert m.decode(64) == want, kv
        # a chunk lands at any offset: rows [64, 128) written back to the same place change nothing
        mid = m.kv_export(64, 128)
        m.kv_import(mid, 64)
        print(f"  {kv}: {chunk.nbytes / 2**20:.1f} MiB exported and rebuilt")


def test_room_rule_and_untouched_rows() -> None:
    """decode(n) with position + n + 2k - 1 above the KV cache raises and changes nothing; with equality it runs; and
    a decode whose bound is below the cache leaves the KV rows above position + n + 2k - 1 (target and head) at their
    canary bytes."""
    n, k, spare = 64, 3, 64
    ids = _ids(385)
    limit = len(ids) + n + 2 * k - 1  # the first row a decode(n) cannot touch
    ctx = limit + spare
    m = _model("bf16", ctx)
    spec = m.decoder
    m.reset()
    spec.k, spec.cap = k, None
    m.append(ids)
    planes = []
    for rt in (spec._target, spec._draft):
        for d, sections in rt._kv_states():
            for s in sections.values():
                plane = rt._kv_plane(d, s, limit, ctx)
                plane.fill_(0x5A)
                planes.append(plane)
    before = m.position
    try:
        m.decode(ctx - len(ids) - 2 * k + 2)
    except ValueError as e:
        assert "exceed the KV cache" in str(e)
    else:
        raise AssertionError("the room rule let a decode run past the cache")
    assert m.position == before
    assert len(m.decode(n)) == n
    assert all(bool((p == 0x5A).all()) for p in planes)
    assert len(m.decode(ctx - m.position - 2 * k + 1)) == ctx - before - n - 2 * k + 1  # equality runs to the end


def test_compile_refuses_what_cannot_fit() -> None:
    """The pre-flight names the numbers and raises before anything is loaded."""
    ckpt = _checkpoint()
    backend = CompositeBackend(CudaBackend("int4_refine"), SwiftmkBackend("int4_refine"), kernels=GLUE)
    _LIVE.clear()
    gc.collect()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_allocated()
    for host in (False, True):
        try:
            warpc.compile(ckpt, ctx_max=4_000_000, kv="int4_rot", backend=backend, speculate=3, host_tables=host)
        except ValueError as e:
            assert "ctx_max 4000000 needs" in str(e) and "GiB free" in str(e), e
            assert ("host_tables" in str(e)) == (not host), e
        else:
            raise AssertionError("compile accepted a context that cannot fit")
    assert torch.cuda.memory_allocated() == before


if __name__ == "__main__":
    run(globals())
