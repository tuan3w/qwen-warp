#!/usr/bin/env python
"""Host prefix store bookkeeping (server/prefix_store.py): what a request may resume from, how a growing conversation extends its entry, LRU + budget.  No GPU, no checkpoint.

  python tests/test_prefix_store.py
The properties that decide correctness of a restore (the engine copies are tested end to end in tests/test_lineage_cache.py):
  * resume only AT a snapshot s <= min(L - 1, len(prompt) - 1) that is a segment end of the new prompt's plan (L = common prefix with the stored prompt: the MTP KV row s - 1 needs
    token s to be common), the largest one wins, across entries;
  * an entry is extended (not duplicated) by a longer lineage of the same conversation, the new chunk starts exactly where the entry ends, same-position snapshots are replaced;
  * the byte budget evicts least recently used entries, never one that is protected, and returns every snapshot buffer it drops (the runner pools them).
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
from prefix_store import Chunk, PrefixStore, common_prefix  # noqa: E402


def snap(pos):
    return SimpleNamespace(pos=pos)


def chunk(a, m, nbytes=100):
    return Chunk(a, m, None, None, nbytes)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg)


def main():
    # common_prefix
    check(common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2 and common_prefix([], [1]) == 0 and common_prefix([1, 2], [1, 2]) == 2, "common_prefix")

    ids = list(range(1000, 1000 + 5000))
    st = PrefixStore(budget_bytes=10_000)
    s_a, s_b = snap(2000), snap(4000)
    e, rel = st.add(ids, 4000, chunk(0, 4000, 4000), [s_a, s_b])
    check(rel == [] and e.n_kv == 4000 and len(e.ids) == 4001, "add keeps n_kv + 1 prompt tokens")

    # a continuation resumes at the largest snapshot allowed by the plan and the common prefix
    longer = ids[:4500] + [7, 8, 9]
    got = st.best(longer, ends={2000, 4000, 4500})
    check(got is not None and got[1] is s_b, "continuation resumes at the last snapshot")
    check(st.best(longer, ends={2000})[1] is s_a, "a snapshot that is not a segment end of the new plan is skipped")
    check(st.best(longer, ends={4000, 4400}) is not None and st.best(longer, ends={999}) is None, "no snapshot in the plan: nothing to resume")
    div = ids[:3000] + [5] + ids[3001:4600]  # diverges at 3000: snapshot 4000 is past it
    check(st.best(div, ends={2000, 4000})[1] is s_a, "divergence before a snapshot: resume at the earlier one")
    exact = ids[:4001]  # L = len(prompt): s <= len - 1 = 4000 is allowed only because token 4000 matches too
    check(st.best(exact, ends={4000})[1] is s_b, "s = len(prompt) - 1 is allowed when token s is common")
    check(st.best(ids[:4000], ends={4000}) is None, "s = len(prompt) is not (the last token must be computed)")
    short = ids[:2000] + [1]  # L = 2000: s <= 1999
    check(st.best(short, ends={2000}) is None, "L - 1 rule: the snapshot at L itself is not usable (MTP row s - 1 needs token s)")

    # largest across entries
    other_ids = [9] * 3000 + ids[:2500]
    e2, _ = st.add(other_ids, 5000, chunk(0, 5000, 100), [snap(5000)])
    check(st.best(longer, ends={2000, 4000, 4500})[0] is e, "best() ignores entries that do not match")

    # extension
    st = PrefixStore(budget_bytes=10_000)
    e, _ = st.add(ids, 2000, chunk(0, 2000, 2000), [snap(1000), snap(2000)])
    lin = ids[:4001]
    check(st.extendable(lin, 4000) is e, "extendable: the lineage continues the entry")
    check(st.extendable([1] + lin[1:], 4000) is None and st.extendable(ids[:1500], 1400) is None, "extendable: not for a different or a shorter lineage")
    ns = snap(2000)
    rel = st.extend(e, lin, 4000, chunk(2000, 2000, 2000), [ns, snap(4000)])
    check(e.n_kv == 4000 and [c.a for c in e.chunks] == [0, 2000] and len(e.ids) == 4001, "extend appends a chunk that starts where the entry ended")
    check(len(rel) == 1 and rel[0].pos == 2000 and ns in e.snaps, "same-position snapshot replaced, old buffer released")
    try:
        st.extend(e, lin, 4000, chunk(1, 10), [])
        check(False, "extend rejects a chunk that does not start at n_kv")
    except AssertionError:
        check(True, "extend rejects a chunk that does not start at n_kv")

    # snapshot cap per entry drops the lowest positions
    st = PrefixStore(budget_bytes=10_000, max_snaps_per_entry=3)
    e, rel = st.add(ids, 4000, chunk(0, 4000), [snap(1000), snap(2000), snap(3000), snap(4000)])
    check([s.pos for s in rel] == [1000] and sorted(s.pos for s in e.snaps) == [2000, 3000, 4000], "snapshot cap releases the lowest position")

    # LRU + budget
    st = PrefixStore(budget_bytes=1000)
    ents = [st.add([i] * 10, 5, chunk(0, 5, 400), [snap(5)])[0] for i in range(2)]  # 800 bytes
    st.touch(ents[0])  # entry 1 is now the LRU
    rel = st.fit(400, keep=None)
    check(len(rel) == 1 and st.entries == [ents[0]], "fit evicts the least recently used entry and returns its snapshots")
    ents.append(st.add([7] * 10, 5, chunk(0, 5, 400), [snap(5)])[0])
    st.touch(ents[0])
    rel = st.fit(400, keep=(ents[0], ents[2]))
    check(st.entries == [ents[0], ents[2]] and rel == [], "protected entries are never evicted (budget may be exceeded rather than drop them)")
    check(st.evict_lru(keep=tuple(ents)) is None, "evict_lru with everything protected returns None")
    rel = st.clear()
    check(len(rel) == 2 and st.entries == [] and st.bytes() == 0, "clear returns every snapshot buffer")
    print("ALL PASS")


if __name__ == "__main__":
    main()
