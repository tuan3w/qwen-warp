"""Host-memory prefix store: the second tier under the engine's single device lineage (see smk_runner.py).

The engine's device caches hold ONE conversation (`Runner.lineage`).  Agent traffic interleaves several (a long main conversation, short helper requests, parallel subagents), so
every request that does not continue the device lineage used to wipe it.  Instead the runner now SPILLS the device lineage to this store before overwriting it and RESTORES a stored
one when a later request continues it.

An `Entry` is one spilled conversation prefix:
  * `ids`      the prompt tokens [0, n_kv] (n_kv + 1 tokens: the MTP KV row of the last position needs the NEXT token to be common, see Runner._choose);
  * `chunks`   the KV rows [a, a + m) of the main cache (16 layers) and of the MTP layer, packed by the engine (include/attn.h); chunks are appended when a longer turn of the same
               conversation is spilled, so a growing conversation never re-copies rows it already stored;
  * `snaps`    DeltaNet + conv state (+ int4 sink / ring) at plan cut points (<= n_kv).  A request can only resume AT a snapshot: KV alone is not enough for a recurrent layer.
Matching mirrors the device rule: resume at the largest snapshot s <= min(L - 1, len(prompt) - 1) that is a segment end of the new prompt's own prefill plan (L = common prefix).

Nothing here touches the GPU: the runner passes the copies, this file is bookkeeping (ownership, LRU, a byte budget), so it is unit-tested on the CPU (tests/test_prefix_store.py).
Design notes and the SGLang / vLLM ideas it borrows (deepest state-bearing node wins, bounded states, LRU, host tier under the device tier): research repo
experiments/2026-11-05-qwen-warp-lineage-cache/README.md.  Not borrowed: int8-compressed states (lossy; a resumed request here equals a cold one).
"""
from dataclasses import dataclass, field

import numpy as np


def common_prefix(a, b):
    """Length of the common prefix of two int arrays."""
    n = min(len(a), len(b))
    if n == 0:
        return 0
    ne = np.flatnonzero(np.asarray(a[:n]) != np.asarray(b[:n]))
    return int(ne[0]) if len(ne) else n


@dataclass
class Chunk:
    a: int  # first KV row of the chunk
    m: int  # rows
    main: object  # packed pinned host buffer of the main cache rows
    mtp: object  # packed pinned host buffer of the MTP layer rows
    nbytes: int = 0


@dataclass(eq=False)  # identity comparison: the fields hold arrays / tensors, whose == is not a bool
class Entry:
    ids: np.ndarray
    n_kv: int
    chunks: list = field(default_factory=list)
    snaps: list = field(default_factory=list)
    tick: int = 0

    @property
    def kv_bytes(self):
        return sum(c.nbytes for c in self.chunks)


class PrefixStore:
    def __init__(self, budget_bytes, max_snaps_per_entry=5):  # the system checkpoint + the last two cut points of two successive turns
        self.budget = int(budget_bytes)
        self.max_snaps_per_entry = max_snaps_per_entry
        self.entries = []
        self._tick = 0

    # ---------------------------------------------------------------- queries
    def bytes(self):
        return sum(e.kv_bytes for e in self.entries)

    def touch(self, e):
        self._tick += 1
        e.tick = self._tick

    def best(self, ids, ends):
        """(entry, snap) with the largest resume position over all entries, None when nothing is resumable.  `ends` = the segment ends of the new prompt's plan."""
        best = None
        for e in self.entries:
            lim = min(common_prefix(ids, e.ids) - 1, len(ids) - 1)
            ok = [s for s in e.snaps if s.pos <= lim and s.pos in ends]
            if ok:
                s = max(ok, key=lambda x: x.pos)
                if best is None or s.pos > best[1].pos:
                    best = (e, s)
        return best

    def extendable(self, ids, n_kv):
        """An entry that is a proper prefix of the lineage `ids[: n_kv + 1]` (same first n_kv + 1 tokens up to ITS length): spilling the lineage can append to it."""
        for e in self.entries:
            if 0 < e.n_kv <= n_kv and len(ids) > e.n_kv and common_prefix(ids, e.ids) == e.n_kv + 1:
                return e
        return None

    # ---------------------------------------------------------------- updates; every method returns the snapshot buffers it released (the caller pools them)
    def add(self, ids, n_kv, chunk, snaps):
        """New entry; returns released snaps (none)."""
        e = Entry(ids=np.asarray(ids[: n_kv + 1], dtype=np.int32).copy(), n_kv=n_kv, chunks=[chunk] if chunk is not None else [], snaps=list(snaps))
        self.entries.append(e)
        self.touch(e)
        return e, self._cap_snaps(e)

    def extend(self, e, ids, n_kv, chunk, snaps):
        """Append the rows of a longer lineage of the same conversation; its snapshots replace same-position ones.  Returns released snaps."""
        assert chunk is None or chunk.a == e.n_kv, "chunk must start where the entry ends"
        released = []
        new_pos = {s.pos for s in snaps}
        keep = [s for s in e.snaps if s.pos not in new_pos]
        released += [s for s in e.snaps if s.pos in new_pos]
        e.snaps = keep + list(snaps)
        if chunk is not None:
            e.chunks.append(chunk)
        e.ids = np.asarray(ids[: n_kv + 1], dtype=np.int32).copy()
        e.n_kv = n_kv
        self.touch(e)
        return released + self._cap_snaps(e)

    def _cap_snaps(self, e):
        released = []
        while len(e.snaps) > self.max_snaps_per_entry:
            low = min(e.snaps, key=lambda s: (getattr(s, 'sticky', False), s.pos))  # prompts grow: the lowest boundaries are the least useful (same rule as the device), but not the system-prompt checkpoint
            e.snaps = [s for s in e.snaps if s is not low]
            released.append(low)
        return released

    def evict_lru(self, keep=None):
        """Drop the least recently used entry that is not in `keep` (an Entry, a tuple of them or None); returns its snaps (None when there is nothing to drop)."""
        ks = keep if isinstance(keep, tuple) else (keep,)
        cand = [e for e in self.entries if all(e is not k for k in ks)]
        if not cand:
            return None
        v = min(cand, key=lambda e: e.tick)
        self.entries.remove(v)
        return v.snaps

    def fit(self, extra, keep=None):
        """Evict LRU entries until bytes() + extra <= budget (an entry that can never fit is the caller's problem: check `extra <= budget`).  Returns released snaps."""
        released = []
        while self.bytes() + extra > self.budget:
            s = self.evict_lru(keep)
            if s is None:
                break
            released += s
        return released

    def clear(self):
        out = [s for e in self.entries for s in e.snaps]
        self.entries = []
        return out
