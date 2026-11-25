"""PrefixCache on the real 27B Runtime: a resumed turn equals a cold prefill of the same prompt bit for bit (KV rows,
every non-positional state, the last row's logits, the greedy tokens after it), for the window plan and the 8-row plan,
from the device lineage and from the host store; a snapshot / kv_export / kv_import / restore round trip is byte-equal.

  WARPC_SWIFTMK_LIB=<libswiftmk.so> WARPC_TEST_SWIFT27=<27B checkpoint directory> \
      uv run --package warpc python warpc/tests/test_longctx.py

`test_speculator_resume_equals_cold` is the MTP case (the draft KV rows travel with the cache, the pairs of every
window row are committed, the tokens of the speculative cycles after a resume equal those after a cold prefill) and
`test_dropped_models_release_memory` builds and drops models, speculators and caches: the file runs in one process, a
17 GiB model at a time.
Needs a CUDA device, the library, the checkpoint and `warpc[prefill]`; the tests skip without them."""

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from harness import Skip, run
from warpc import pinned
from warpc.prefix import PrefixCache, class_d_plan
from warpc.windows import IM_START, plan_windows

IM_END, NL = 248046, 198
WINDOW_ROWS = 256  # small windows so that a short prompt has several; the plan cuts at turn headers after 16 rows


def plan(ids):
    return plan_windows(ids, rows=WINDOW_ROWS, w4_max=16)


_models: dict[int, object] = {}


def build(speculate: int = 0):
    path, lib = os.environ.get("WARPC_TEST_SWIFT27"), os.environ.get("WARPC_SWIFTMK_LIB")
    if not (path and lib and torch.cuda.is_available()):
        raise Skip("needs a CUDA device, WARPC_SWIFTMK_LIB and WARPC_TEST_SWIFT27")
    import warpc
    from warpc.composite import CompositeBackend
    from warpc.cuda import CudaBackend
    from warpc.swiftmk import SwiftmkBackend

    if speculate:  # the MTP head's three data-movement kernels come from warpc's own table
        backend = CompositeBackend(
            CudaBackend("int4_refine"),
            SwiftmkBackend("int4_refine", window=True),
            kernels=["g_select_row", "g_take", "g_concat"],
        )
    else:
        backend = SwiftmkBackend("bf16", window=True)
    return warpc.compile(
        Path(path),
        ctx_max=4096,
        kv="int4_rot",
        backend=backend,
        window_rows=WINDOW_ROWS,
        speculate=speculate,
        host_tables=bool(speculate),
    )


def model(speculate: int = 0):
    """The model of the tests, built on first use; one 17 GiB model at a time (a dropped one is freed by refcount)."""
    if speculate not in _models:
        _models.clear()
        _models[speculate] = build(speculate)
    return _models[speculate]


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def fingerprint_states(m) -> dict:
    """KV rows below the position and the non-positional states."""
    rt = m.decoder
    out = {k: digest(v) for k, v in rt.snapshot().states.items()}
    return out | {f"kv:{k}": digest(v) for k, v in rt.kv_export(0, rt.position).planes.items()}


def fingerprint(m) -> dict:
    """The states and the logits of the last head step."""
    return fingerprint_states(m) | {"logits": digest(m.logits)}


def conversation(seed: int, prefix: int, turns: int) -> list[list[int]]:
    g = np.random.default_rng(seed)

    def toks(n):
        return g.integers(1000, 9000, size=n).tolist()

    head = [IM_START, 8678, NL, *toks(prefix), IM_END, NL, IM_START, 846, NL, *toks(40), IM_END, NL]
    msgs = [(toks(int(g.integers(20, 60))), toks(int(g.integers(120, 400)))) for _ in range(turns)]
    prompts = []
    for k in range(turns):
        ids = list(head)
        for a, t in msgs[:k]:
            ids += [IM_START, 74455, NL, *a, IM_END, NL, IM_START, 846, NL, *t, IM_END, NL]
        prompts.append([*ids, IM_START, 74455, NL, 248068, NL])
    return prompts


def judge(m, planner, prompts, *, host_bytes=0, decode=12):
    """Run `prompts` through one cache and each from scratch; every turn must be identical. Returns the sources."""
    cold = []
    for ids in prompts:
        m.reset()
        c = PrefixCache(m.decoder, host_bytes=0, plan=planner, min_resume=64)
        tok = c.run(ids)
        cold.append((tok, fingerprint(m), c.decode(decode)))
    m.reset()
    cache = PrefixCache(m.decoder, host_bytes=host_bytes, plan=planner, min_resume=64)
    sources = []
    for ids, (tok, fp, out) in zip(prompts, cold, strict=True):
        assert cache.run(ids) == tok
        got = fingerprint(m)
        bad = [k for k in fp if fp[k] != got[k]]
        assert not bad, f"{cache.stats.last.source} resume differs from cold in {bad[:6]}"
        assert cache.decode(decode) == out
        sources.append((cache.stats.last.source, cache.stats.last.cached))
    return sources


def test_window_resume_equals_cold():
    sources = judge(model(), plan, conversation(0, 700, 4))
    assert sources[0][0] == "cold" and all(s == "device" and c > 600 for s, c in sources[1:]), sources


def test_window_host_tier_equals_cold():
    a, b = conversation(1, 500, 3), conversation(2, 600, 3)
    sources = judge(model(), plan, [a[0], b[0], a[1], b[1], a[2], b[2]], host_bytes=1 << 30)
    assert [s for s, _ in sources] == ["cold", "cold", "host", "host", "host", "host"], sources


def test_eight_row_plan_resume_equals_cold():
    sources = judge(model(), class_d_plan, conversation(3, 300, 3))
    assert all(s == "device" and c > 300 and c % 8 == 0 for s, c in sources[1:]), sources


def test_snapshot_export_import_round_trip():
    m = model()
    ids = conversation(4, 400, 1)[0]
    m.reset()
    m.append(ids[:-1], head=False, ends=plan([*ids[:-1], 0]))
    snap = m.snapshot(device=True)
    chunk = m.kv_export(0, m.position)
    want = fingerprint_states(m)
    m.reset()
    m.kv_import(chunk, 0)
    m.restore(snap)
    assert fingerprint_states(m) == want


def test_ends_are_validated():
    m = model()
    m.reset()
    for bad in ([5, 3, 10], [10, 11], [0, 10]):
        try:
            m.append(list(range(1000, 1011)), ends=bad)
        except ValueError:
            continue
        raise AssertionError(f"ends {bad} must be refused")


def test_dropped_models_release_memory():
    """A service that builds and drops models, speculators and caches in one process gets its memory back when the last
    reference goes: the device memory (the driver's view, not only torch's) to within 50 MiB of what the first drop of
    the same kind left, the pinned host memory (host tables, snapshots, KV chunks) to zero. The cyclic collector is off:
    a drop must free by refcount, a collection that runs inside another model's CUDA graph capture invalidates it."""
    _models.clear()
    gc.collect()
    pool = lambda: torch.cuda.host_memory_stats().get("allocated_bytes.current", 0)  # noqa: E731  torch's own pinned pool
    pool0 = pool()
    gc.disable()
    try:
        left: dict[int, int] = {}
        for speculate in (0, 0, 3, 3):
            m = build(speculate)
            prompts = conversation(7, 500, 3)
            cache = PrefixCache(m.decoder, host_bytes=1 << 30, plan=plan, min_resume=64)
            for ids in prompts:
                cache.run(ids)
                cache.decode(8)
            cache.run(conversation(8, 400, 1)[0])  # another conversation: the first is spilled to the host store
            assert cache.stats.host_bytes > 0 and pinned.live_bytes() >= cache.stats.host_bytes
            m.snapshot()
            m.kv_export(0, 64)
            del m, cache
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free, total = torch.cuda.mem_get_info()
            left.setdefault(speculate, total - free)
            drift = (total - free - left[speculate]) >> 20
            assert drift <= 50, f"speculate={speculate}: {drift} MiB of device memory left after the drop"
            assert torch.cuda.memory_allocated() < 50 << 20, torch.cuda.memory_allocated()
            assert pinned.live_bytes() == 0, f"{pinned.live_bytes()} pinned bytes left after the drop"
            assert pool() == pool0, "warpc went through torch's pinned pool"
    finally:
        gc.enable()


def test_speculator_resume_equals_cold():
    m = model(3)
    sources = judge(m, plan, conversation(5, 500, 3), decode=40)
    assert all(s == "device" and c > 400 for s, c in sources[1:]), sources


if __name__ == "__main__":
    run(globals())
