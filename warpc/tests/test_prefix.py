"""PrefixCache against a fake decoder (CPU): resume equals cold bit for bit, the host tier and its budget, plan
consistency, the append mode.

  uv run --package warpc python warpc/tests/test_prefix.py

The fake decoder's state is a hash chain over its chunks (tokens, positions, the chunk partition) and its KV rows
depend on the state before their chunk, so a resume that cuts the prompt differently from a cold prefill, or imports
the rows of another lineage, changes bits exactly as a window prefill would."""

import numpy as np
import torch

from harness import run
from warpc.kvcache import KVChunk
from warpc.prefix import PrefixCache, class_d_plan
from warpc.runtime import Snapshot
from warpc.windows import IM_START, plan_windows

IM_END, NL = 248046, 198
STATE_BYTES = 4096


def mix(*xs) -> int:
    h = 1469598103934665603
    for x in xs:
        h = ((h ^ (hash(x) & 0xFFFFFFFFFFFF)) * 1099511628211) & 0x7FFFFFFFFFFFFFFF
    return h


class Fake:
    """A Decoder whose bits depend on the chunking."""

    def __init__(self, ctx: int = 8192):
        self.ctx = ctx
        self.reset()

    def reset(self):
        self.position, self.h, self.next = 0, 1, None
        self.kv = np.full(self.ctx, -1, dtype=np.int64)
        self.log = []  # (position, rows, kind)

    def _chunk(self, toks, kind):
        h0, p0 = self.h, self.position
        for i, t in enumerate(toks):
            self.kv[p0 + i] = mix(h0, t, len(toks), i)
        self.h = mix(h0, tuple(toks), p0)
        self.position += len(toks)
        self.log.append((p0, len(toks), kind))

    def append(self, ids, *, head=True, ends=None):
        n = len(ids) - 1 if head else len(ids)
        p0, a = self.position, 0
        for e in ends if ends is not None else range(p0 + 8, p0 + n + 8, 8):
            e = min(e, p0 + n)
            assert e > p0 + a, f"ends {ends} do not ascend within the bulk rows"
            self._chunk(ids[a : e - p0], "bulk")
            a = e - p0
        assert a == n, f"ends {ends} leave rows {a}..{n} of the bulk rows"
        if head:
            self._chunk(ids[-1:], "head")
            self.next = mix(self.h) % 1000
            return self.next
        return -1

    def decode(self, n, *, stop=()):
        out = []
        for _ in range(n):
            self._chunk([self.next], "decode")
            self.next = mix(self.h) % 1000
            out.append(self.next)
        return out

    def snapshot(self, into=None, *, device=False):
        state = torch.full((STATE_BYTES // 8,), self.h, dtype=torch.int64)
        if into is None:
            return Snapshot(self.position, None, None, {"dn": state}, device)
        into.states["dn"].copy_(state)
        into.position = self.position
        return into

    def restore(self, snap):
        self.h, self.position, self.next = int(snap.states["dn"][0]), snap.position, None
        self.kv[self.position :] = -1  # rows past the position are dead

    def kv_export(self, start, stop, into=None):
        assert 0 <= start <= stop <= self.position
        return KVChunk(start, stop, {"kv/k": torch.from_numpy(self.kv[start:stop].copy())})

    def kv_import(self, chunk, start):
        self.kv[start : start + chunk.stop - chunk.start] = chunk.planes["kv/k"].numpy()


def conversation(seed: int, prefix_len: int, turns: int):
    """Prompts of a chat loop whose history is re-rendered without the think tail of the last turn."""
    g = np.random.default_rng(seed)

    def toks(n):
        return g.integers(1000, 9000, size=n).tolist()

    prefix = [IM_START, 8678, NL, *toks(prefix_len), IM_END, NL, IM_START, 846, NL, *toks(30), IM_END, NL]
    msgs = [(toks(int(g.integers(20, 60))), toks(int(g.integers(120, 600)))) for _ in range(turns)]
    prompts = []
    for k in range(turns):
        ids = list(prefix)
        for a, t in msgs[:k]:
            ids += [IM_START, 74455, NL, *a, IM_END, NL, IM_START, 846, NL, *t, IM_END, NL]
        prompts.append([*ids, IM_START, 74455, NL, 248068, NL])
    return prompts


def window_plan(ids):
    return plan_windows(ids, rows=256, w4_max=16)


def cold(ids, plan):
    """The reference: a fresh decoder through a cache with nothing in it."""
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=plan, min_resume=64)
    token = cache.run(ids)
    return d, token


def same(a: Fake, b: Fake, n: int):
    assert a.position == b.position == n, (a.position, b.position, n)
    assert a.h == b.h, "states differ"
    assert np.array_equal(a.kv[:n], b.kv[:n]), "KV rows differ"


def test_plans():
    ids = list(range(1000))
    assert class_d_plan(ids) == [*range(8, 999, 8), 999]
    assert class_d_plan([5]) == [] and class_d_plan([5, 6]) == [1]
    assert plan_windows(ids) == [999]
    assert plan_windows(list(range(10000))) == [4096, 8192, 9999]
    ids = [*range(300), IM_START, 1, 2, *range(1000, 1300), IM_START, 5, 6, *range(100)]
    # cuts at p + 2, each more than 240 rows after the previous end
    assert plan_windows(ids) == [302, 605, len(ids) - 1]
    assert plan_windows(ids, turn_token=None) == [len(ids) - 1]


def chat_loop(plan, device_snapshots=2, host_bytes=0, **kw):
    prompts = conversation(1, 700, 6)
    d = Fake()
    cache = PrefixCache(d, host_bytes=host_bytes, plan=plan, min_resume=64, device_snapshots=device_snapshots, **kw)
    sources = []
    for ids in prompts:
        token = cache.run(ids)
        ref, ref_token = cold(ids, plan)
        same(d, ref, len(ids))
        assert token == ref_token
        sources.append((cache.stats.last.source, cache.stats.last.cached))
        cache.decode(20)
    return sources, cache


def test_resume_equals_cold_windows():
    sources, _ = chat_loop(window_plan)
    assert sources[0] == ("cold", 0)
    assert all(s == "device" and c > 600 for s, c in sources[1:]), sources


def test_resume_equals_cold_class_d():
    sources, _ = chat_loop(class_d_plan)
    assert all(s == "device" and c > 600 and c % 8 == 0 for s, c in sources[1:]), sources


def test_snapshot_positions_are_plan_ends_and_bounded():
    prompts = conversation(2, 700, 3)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64, device_snapshots=2)
    cache.run(prompts[0])
    plan = window_plan(prompts[0])
    taken = cache.stats.last.snapshots
    assert len(taken) <= 2 and all(p in plan for p in taken), (taken, plan)
    assert cache.stats.device_snapshot_bytes == len(cache._dev) * STATE_BYTES


def test_resume_never_past_the_common_prefix():
    prompts = conversation(3, 700, 3)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64)
    cache.run(prompts[0])
    cache.run(prompts[0])  # the same prompt again: the last token always runs
    s = cache.stats.last
    assert s.cached <= len(prompts[0]) - 1 and s.source == "device"
    ref, _ = cold(prompts[0], window_plan)
    same(d, ref, len(prompts[0]))


def test_host_tier_round_trip():
    a, b = conversation(4, 700, 4), conversation(5, 900, 4)
    d = Fake()
    cache = PrefixCache(d, host_bytes=1 << 30, plan=window_plan, min_resume=64)
    seen = []
    for k in range(4):  # two conversations interleaved: each turn finds the other one's lineage on the device
        for conv in (a, b):
            token = cache.run(conv[k])
            ref, ref_token = cold(conv[k], window_plan)
            same(d, ref, len(conv[k]))
            assert token == ref_token
            seen.append(cache.stats.last.source)
            cache.decode(10)
    assert seen[:2] == ["cold", "cold"] and set(seen[2:]) == {"host"}, seen
    assert cache.stats.host_entries == 2 and cache.stats.host_bytes > 0


def test_budget_evicts_lru_and_refuses_what_cannot_fit():
    a, b, c = conversation(6, 700, 2), conversation(7, 700, 2), conversation(8, 700, 2)
    d = Fake()
    one = Fake()
    probe = PrefixCache(one, host_bytes=1 << 30, plan=window_plan, min_resume=64)
    probe.run(a[0])
    probe.spill()
    entry_bytes = probe.stats.host_bytes or sum(e.nbytes for e in probe._entries)
    assert entry_bytes > 0
    cache = PrefixCache(d, host_bytes=int(entry_bytes * 1.2), plan=window_plan, min_resume=64)
    for conv in (a, b, c):
        cache.run(conv[0])
    cache.spill()
    assert len(cache._entries) == 1 and cache.stats.evicted >= 2, (len(cache._entries), cache.stats.evicted)
    # only the most recent conversation is left: the first one is cold again, the last one is host
    cache.run(a[1])
    assert cache.stats.last.source == "cold"
    tiny = PrefixCache(Fake(), host_bytes=entry_bytes // 4, plan=window_plan, min_resume=64)
    tiny.run(a[0])
    tiny.run(b[0])
    assert tiny.stats.refused >= 1 and not tiny._entries
    tiny.run(a[1])
    assert tiny.stats.last.source == "cold"


def test_inconsistent_plan_is_not_resumed():
    """Same tokens, a plan that cuts them differently (the step depends on the parity of the length): the snapshots of
    the other cut are not usable, the cache goes cold, and a prompt with the same cut resumes."""

    def odd_plan(ids):
        step = 48 if len(ids) % 2 else 64
        n = len(ids) - 1
        return [*range(step, n, step), n]

    base = conversation(9, 700, 1)[0]
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=odd_plan, min_resume=64)
    for ids in (base, [*base, 5], [*base, 5, 6, 7]):
        cache.run(ids)
        ref, _ = cold(ids, odd_plan)
        same(d, ref, len(ids))
    sources = [(s.source, s.cached > 0) for s in cache.stats.calls]
    assert sources == [("cold", False), ("cold", False), ("device", True)], sources


def test_decoded_tokens_in_the_lineage_do_not_break_exactness():
    """The next prompt extends the previous one plus its answer (a linear loop): exact mode resumes at a snapshot."""
    g = np.random.default_rng(10)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64)
    ids = [IM_START, 8678, NL, *g.integers(1000, 9000, 800).tolist(), IM_END, NL, IM_START, 74455, NL]
    for _ in range(4):
        first = cache.run(ids)
        out = cache.decode(30)
        _, ref_token = cold(ids, window_plan)
        assert first == ref_token
        ids = [
            *ids,
            first,
            *out[:-1],
            out[-1],
            IM_END,
            NL,
            IM_START,
            846,
            NL,
            *g.integers(1000, 9000, 200).tolist(),
            IM_END,
            NL,
            IM_START,
            74455,
            NL,
        ]
    assert cache.stats.last.source == "device" and cache.stats.last.cached > 700


def test_append_mode_continues_the_lineage():
    g = np.random.default_rng(11)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64, exact=False)
    ids = [IM_START, 8678, NL, *g.integers(1000, 9000, 800).tolist(), IM_END, NL, IM_START, 74455, NL]
    first = cache.run(ids)
    out = cache.decode(30)
    pos = d.position
    nxt = [*ids, first, *out, IM_END, NL, *g.integers(1000, 9000, 200).tolist()]
    cache.run(nxt)
    s = cache.stats.last
    assert s.source == "append" and s.cached == pos == len(ids) + 30, (s, pos)
    assert d.position == len(nxt)
    # not equal to a cold prefill (the decoded rows were class-D rows), and it says so
    ref, _ = cold(nxt, window_plan)
    assert ref.h != d.h


def test_append_mode_keeps_the_device_snapshots_bounded():
    """Snapshots of an append loop are taken at every turn; the ones a prompt does not want go back to the pool."""
    g = np.random.default_rng(14)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64, exact=False, device_snapshots=2)
    ids = [IM_START, 8678, NL, *g.integers(1000, 9000, 800).tolist(), IM_END, NL, IM_START, 74455, NL]
    for _ in range(8):
        first = cache.run(ids)
        out = cache.decode(20)
        ids = [*ids, first, *out, IM_END, NL, IM_START, 846, NL, *g.integers(1000, 9000, 300).tolist(), IM_END, NL]
        ids += [IM_START, 74455, NL]
        assert len(cache._dev) <= 2 and len(cache._dev) + len(cache._free) <= 3, (len(cache._dev), len(cache._free))
    assert cache.stats.last.source == "append"


def test_min_resume_and_clear():
    prompts = conversation(12, 100, 3)
    d = Fake()
    cache = PrefixCache(d, host_bytes=1 << 20, plan=window_plan, min_resume=100000)
    cache.run(prompts[0])
    cache.run(prompts[1])
    assert cache.stats.last.source == "cold" and not cache.stats.last.snapshots
    cache.clear()
    assert d.position == 0 and not cache._entries


def test_decoder_reset_behind_the_cache():
    prompts = conversation(13, 700, 2)
    d = Fake()
    cache = PrefixCache(d, host_bytes=0, plan=window_plan, min_resume=64)
    cache.run(prompts[0])
    d.reset()
    cache.run(prompts[1])
    assert cache.stats.last.source == "cold"
    ref, _ = cold(prompts[1], window_plan)
    same(d, ref, len(prompts[1]))


if __name__ == "__main__":
    run(globals())
