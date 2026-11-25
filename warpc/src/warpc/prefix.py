"""Prefix reuse for agent loops: resume a conversation from a snapshot instead of prefilling it again.

A hybrid model's recurrent state cannot be rewound, so a cache keeps copies of it (`Snapshot`) at chosen positions and
resumes a new prompt at the largest one that its tokens still share with the cached conversation. KV rows are
positional: the rows below the resume position are either still on the device (the lineage) or imported from the host.

  cache = PrefixCache(model.decoder, host_bytes=8 << 30, plan=plan_windows)
  first = cache.run(prompt_ids)        # restores the longest valid prefix, prefills the rest, returns the next token
  more = cache.decode(255)             # decode through the cache so that a linear loop can append to the lineage

One conversation (the lineage) lives on the device, with a few device snapshots; other conversations live in a host
store (pinned memory, a byte budget, LRU), into which the lineage is spilled before another prompt overwrites it.

Exactness. A prefill's bits depend on how it is cut into chunks (class D: 8-row steps from position 0; class W:
`plan_windows`). A snapshot is taken only where a plan chunk ends and records the plan's ends up to there; a resume is
legal only if the new prompt's plan has the same ends up to the resume position. Then a resumed prefill equals a cold
prefill of the same prompt bit for bit, whatever the cache held before. `exact=False` additionally continues a
lineage that the new prompt extends (a linear loop that feeds back its own output) with no restore: that prefills
the decoded rows as class-D rows, so it is not bit-equal to a cold prefill (`Resume.source == "append"`).

Nothing here launches a kernel: the decoder does (`append`, `snapshot`, `restore`, `kv_export`, `kv_import`)."""

from __future__ import annotations

import bisect
import dataclasses
import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import pinned
from .windows import IM_START

if TYPE_CHECKING:
    from .kvcache import KVChunk
    from .runtime import Decoder, Snapshot

SYSTEM_MIN = 1024  # the system-prompt checkpoint is the first turn boundary at or after this many tokens
MAX_SNAPS_PER_ENTRY = 5  # per host entry: the system checkpoint, two turns' boundaries and their last ends


def class_d_plan(ids: Sequence[int], rows: int = 8) -> list[int]:
    """The chunking of the 8-row decode path: ends at the multiples of `rows` from position 0, then the last row
    before the head token."""
    n = len(ids) - 1
    if n <= 0:
        return []
    return [*range(rows, n, rows), n]


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    ne = np.flatnonzero(a[:n] != b[:n])
    return int(ne[0]) if len(ne) else n


def snapshot_nbytes(s: Snapshot) -> int:
    n = sum(t.numel() * t.element_size() for t in s.states.values())
    return n + (s.hidden.numel() * s.hidden.element_size() if s.hidden is not None else 0)


def _to_host(s: Snapshot) -> Snapshot:
    """A pinned-host copy of a device snapshot (blocking)."""
    hidden = None if s.hidden is None else pinned.like(s.hidden)
    return dataclasses.replace(s, states={k: pinned.like(v) for k, v in s.states.items()}, hidden=hidden, device=False)


@dataclass(eq=False)  # identity: the fields hold tensors
class _Snap:
    position: int
    ends: tuple[int, ...]  # the plan chunk ends that led here (the last one is `position`)
    snap: Snapshot
    sticky: bool  # the system checkpoint: kept when the budget caps an entry
    exact: bool  # reached by plan chunks only: a resume from it equals a cold prefill

    @property
    def nbytes(self) -> int:
        return snapshot_nbytes(self.snap)


@dataclass(eq=False)
class _Entry:
    """A spilled conversation prefix: the tokens [0, n_kv] (token n_kv commits the MTP pair of row n_kv - 1), the KV
    rows [0, n_kv) in chunks that end at snapshot positions (a resume imports whole chunks), the snapshots."""

    ids: np.ndarray
    n_kv: int
    ends: tuple[int, ...]  # the plan ends of the lineage the rows and the snapshots came from, up to n_kv
    chunks: list[KVChunk] = field(default_factory=list)
    snaps: list[_Snap] = field(default_factory=list)
    tick: int = 0

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.chunks) + sum(s.nbytes for s in self.snaps)


@dataclass(frozen=True)
class Resume:
    position: int  # the cache holds this many tokens of the prompt after `begin`
    source: str  # "cold", "device" (snapshot of the lineage), "host" (a stored conversation), "append" (lineage as is)
    restore_s: float = 0.0
    spill_s: float = 0.0


@dataclass
class CallStats:
    prompt: int
    cached: int  # tokens not prefilled
    source: str
    restore_s: float = 0.0
    spill_s: float = 0.0
    snapshot_s: float = 0.0  # inside prefill_s
    prefill_s: float = 0.0  # the appends, the snapshots included, up to the first token
    snapshots: list[int] = field(default_factory=list)  # positions taken by this call


@dataclass
class CacheStats:
    calls: list[CallStats] = field(default_factory=list)
    host_bytes: int = 0  # pinned bytes held by the host store
    host_entries: int = 0
    device_snapshot_bytes: int = 0
    evicted: int = 0  # host entries dropped by the budget
    refused: int = 0  # spills the budget could never hold

    @property
    def last(self) -> CallStats | None:
        return self.calls[-1] if self.calls else None


class PrefixCache:
    """Typed on the `Decoder` protocol, so a `Speculator` works the same as a `Runtime`.

    `plan(ids)` -> ascending end positions of the chunks of ids[:-1] (the last token is the head step); a pure function
    of the tokens. `turn_token` marks chat-turn boundaries (p + 2 for ids[p] == turn_token): a conversation diverges
    from its re-rendered successor right after such a header, so snapshots go to the last boundary and to the first one
    at or after SYSTEM_MIN tokens (the system prompt that parallel conversations share). `device_snapshots` bounds the
    snapshots the device lineage keeps (the system checkpoint, then the last boundary, then the last ends);
    `min_resume`: a resume below this many tokens is not worth a restore. `host_bytes`: the budget of the host store
    (KV rows and snapshots); an entry that cannot fit is not kept."""

    def __init__(
        self,
        decoder: Decoder,
        *,
        host_bytes: int,
        device_snapshots: int = 2,
        min_resume: int = 512,
        plan: Callable[[Sequence[int]], list[int]] = class_d_plan,
        turn_token: int | None = IM_START,
        exact: bool = True,
    ):
        if device_snapshots < 1:
            raise ValueError("device_snapshots must be at least 1")
        self.decoder, self.host_budget, self.exact = decoder, int(host_bytes), exact
        self.device_snapshots, self.min_resume, self.turn_token = device_snapshots, min_resume, turn_token
        self._plan = plan
        self.stats = CacheStats()
        self._entries: list[_Entry] = []
        self._tick = 0
        self._row_bytes = 0  # host bytes per KV row, learned from the first spill
        self._lin = np.empty(0, dtype=np.int32)  # the tokens of the device KV rows [0, len)
        self._ends: list[int] = []  # the plan ends the device lineage was prefilled with
        self._exact = True  # every row of the lineage came from plan chunks
        self._dev: list[_Snap] = []  # device snapshots of the lineage, ascending
        self._free: list[Snapshot] = []  # device snapshot buffers to reuse
        self._anchor: int | None = None  # the token the decoder returned last (not in the cache)
        self._system: int | None = None  # the system checkpoint position of the current prompt
        self._wanted: list[int] = []  # the snapshot positions of the current prompt
        self._cur_plan: list[int] = []  # its plan

    # ---- plan

    def plan(self, ids: Sequence[int]) -> list[int]:
        """The chunk ends of `ids`: ascending, the last one `len(ids) - 1`."""
        ends = self._plan(ids)
        bad = any(a >= b for a, b in itertools.pairwise(ends))
        if len(ids) > 1 and (not ends or ends[-1] != len(ids) - 1 or bad):
            raise ValueError("a plan must ascend and end at len(ids) - 1")
        return ends

    def _points(self, ids: np.ndarray, ends: list[int]) -> tuple[list[int], int | None]:
        """Snapshot positions of a prompt, at most `device_snapshots`, and the system checkpoint among them. By
        priority: the system checkpoint, the last turn boundary, then the last ends. A mark m = p + 2 snaps to the
        largest end <= m."""
        n = len(ids) - 1
        out: list[int] = []
        system = None

        def add(p: int | None) -> None:
            if p is not None and p >= self.min_resume and p not in out:
                out.append(p)

        def at(m: int) -> int | None:
            k = bisect.bisect_right(ends, m)
            return ends[k - 1] if k else None

        if self.turn_token is not None and n > 1:
            marks = np.flatnonzero(ids[: n - 1] == self.turn_token) + 2
            big = marks[marks >= SYSTEM_MIN]
            if len(big):
                system = at(int(big[0]))
                add(system)
            if len(marks):
                add(at(int(marks[-1])))
        for e in reversed(ends):
            add(e)
        out = out[: self.device_snapshots]
        return sorted(out), system if system in out else None

    # ---- the device lineage

    def _sync_lineage(self) -> None:
        """The decoder may have been reset or rewound behind our back: keep only what it still holds."""
        pos = self.decoder.position
        if pos < len(self._lin):
            self._lin = self._lin[:pos]
            self._ends = [e for e in self._ends if e <= pos]
            self._release([s for s in self._dev if s.position > pos])

    def _release(self, drop: list[_Snap]) -> None:
        for s in drop:
            self._dev.remove(s)
            self._free.append(s.snap)

    def _usable(self, s: _Snap, lim: int, plan: list[int]) -> bool:
        if s.position < self.min_resume or s.position > lim:
            return False
        if not self.exact:
            return True
        return s.exact and tuple(plan[: bisect.bisect_right(plan, s.position)]) == s.ends

    def _pick(self, ids: np.ndarray, plan: list[int]) -> tuple[_Snap, _Entry | None] | None:
        """The largest valid resume point: (snapshot, None) on the device lineage or (snapshot, entry) in the host
        store; a tie goes to the device (no copy). A resume is at most the common prefix - 1 (the MTP pair of the last
        cached row needs the next token to be common) and at most len - 1 (the last token is always run)."""
        n = len(ids)
        best: tuple[_Snap, _Entry | None] | None = None
        if self._dev:
            lim = min(common_prefix(ids, self._lin) - 1, n - 1)
            ok = [s for s in self._dev if self._usable(s, lim, plan)]
            if ok:
                best = (max(ok, key=lambda s: s.position), None)
        for e in self._entries:
            lim = min(common_prefix(ids, e.ids) - 1, n - 1)
            ok = [s for s in e.snaps if self._usable(s, lim, plan)]
            if ok:
                s = max(ok, key=lambda s: s.position)
                if best is None or s.position > best[0].position:
                    best = (s, e)
        return best

    def begin(self, ids: Sequence[int]) -> Resume:
        """Put the decoder at the longest valid resume point of `ids` and report it: `position` tokens of `ids` are
        cached and the caller appends the rest (`run` does). The device lineage is spilled to the host first if the
        resume leaves it."""
        if not len(ids):
            raise ValueError("a prompt needs at least one token")
        arr = np.asarray(ids, dtype=np.int32)
        plan = self._cur_plan = self.plan(ids)
        self._sync_lineage()
        dec = self.decoder
        self._wanted, self._system = self._points(arr, plan)
        if (
            not self.exact
            and 0 < len(self._lin) == dec.position < len(arr)
            and np.array_equal(arr[: len(self._lin)], self._lin)
        ):
            self._exact = False
            self._release([s for s in self._dev if s.position not in self._wanted])
            return Resume(dec.position, "append")
        pick = self._pick(arr, plan)
        spill_s = restore_s = 0.0
        if pick is None:
            spill_s = self._spill(None)
            dec.reset()
            self._lin, self._ends, self._exact = np.empty(0, dtype=np.int32), [], True
            source, pos = "cold", 0
        elif pick[1] is None:
            snap = pick[0]
            self._release([s for s in self._dev if s.position > snap.position])
            t0 = time.perf_counter()
            dec.restore(snap.snap)
            restore_s = time.perf_counter() - t0
            self._lin, self._ends, self._exact = self._lin[: snap.position], list(snap.ends), snap.exact
            source, pos = "device", snap.position
        else:
            snap, ent = pick
            spill_s = self._spill(ent)
            t0 = time.perf_counter()
            for c in ent.chunks:
                if c.start >= snap.position:
                    break
                if c.stop > snap.position:
                    raise RuntimeError(f"snapshot at {snap.position} is inside the KV chunk [{c.start}, {c.stop})")
                dec.kv_import(c, c.start)
            dec.restore(snap.snap)
            restore_s = time.perf_counter() - t0
            self._tick += 1
            ent.tick = self._tick
            self._lin, self._ends, self._exact = arr[: snap.position].copy(), list(snap.ends), snap.exact
            source, pos = "host", snap.position
        # snapshots of the lineage that this prompt does not want make room for the ones it does
        self._release([s for s in self._dev if s.position not in self._wanted])
        return Resume(pos, source, restore_s, spill_s)

    def run(self, ids: Sequence[int]) -> int:
        """`begin`, then prefill the rest of `ids` in plan chunks with a snapshot at every wanted position; returns the
        next token (argmax of the last row)."""
        r = self.begin(ids)
        arr, plan = np.asarray(ids, dtype=np.int32), self._cur_plan
        st = CallStats(len(arr), r.position, r.source, r.restore_s, r.spill_s)
        t0 = time.perf_counter()
        pos = r.position
        for p in (p for p in self._wanted if p > pos and all(s.position != p for s in self._dev)):
            self._extend(arr, pos, p, plan, head=False)
            st.snapshot_s += self._take(p, plan)
            st.snapshots.append(p)
            pos = p
        token = self._extend(arr, pos, len(arr), plan, head=True)
        self._ends = [e for e in self._ends if e <= r.position] + [e for e in plan if e > r.position]
        self._anchor = token
        st.prefill_s = time.perf_counter() - t0
        self.stats.calls.append(st)
        self.stats.host_bytes = sum(e.nbytes for e in self._entries)
        self.stats.host_entries = len(self._entries)
        self.stats.device_snapshot_bytes = sum(s.nbytes for s in self._dev)
        return token

    def _extend(self, ids: np.ndarray, a: int, b: int, plan: list[int], *, head: bool) -> int:
        """Append ids[a:b] through the decoder in the plan's chunks (those that end within the bulk rows)."""
        lo, hi = bisect.bisect_right(plan, a), bisect.bisect_right(plan, len(ids) - 1 if head else b)
        out = self.decoder.append(ids[a:b].tolist(), head=head, ends=plan[lo:hi])
        self._lin = ids[:b].copy()
        return out

    def _take(self, p: int, plan: list[int]) -> float:
        """Snapshot the device state at position p into a reusable device buffer."""
        t0 = time.perf_counter()
        snap = self.decoder.snapshot(self._free.pop() if self._free else None, device=True)
        ends = tuple(plan[: bisect.bisect_right(plan, p)])
        self._dev.append(_Snap(p, ends, snap, p == self._system, self._exact))
        self._dev.sort(key=lambda s: s.position)
        return time.perf_counter() - t0

    # ---- decode bookkeeping

    def decode(self, n: int, *, stop: Sequence[int] = ()) -> list[int]:
        """`decoder.decode`, recording the tokens it puts in the cache (the anchor and all but the last token returned)
        so that a later prompt that extends them can append to the lineage."""
        pos = self.decoder.position
        out = self.decoder.decode(n, stop=stop)
        if self._anchor is not None and len(self._lin) == pos and out:
            self._lin = np.concatenate([self._lin, np.asarray([self._anchor, *out[:-1]], dtype=np.int32)])
        if out:
            self._anchor = out[-1]
        return out

    # ---- the host store

    def _fit(self, extra: int, keep: tuple[_Entry | None, ...]) -> None:
        """Evict least recently used entries (not `keep`) until the store plus `extra` bytes fits the budget."""
        while sum(e.nbytes for e in self._entries) + extra > self.host_budget:
            cand = [e for e in self._entries if all(e is not k for k in keep)]
            if not cand:
                return
            self._entries.remove(min(cand, key=lambda e: e.tick))
            self.stats.evicted += 1

    @staticmethod
    def _shared(e: _Entry, ids: np.ndarray, ends: tuple[int, ...], keep: _Entry | None) -> int:
        """The largest chunk boundary b of `e` whose rows a lineage can share: tokens [0, b] equal and the same plan
        ends up to b. The entry a resume is about to read (`keep`) is never cut."""
        limit = min(common_prefix(ids, e.ids) - 1, len(ids) - 1)
        best = 0
        for c in e.chunks:
            b = c.stop
            if (
                b <= limit
                and (e is not keep or b == e.n_kv)
                and b in ends
                and tuple(x for x in ends if x <= b) == tuple(x for x in e.ends if x <= b)
            ):
                best = b
        return best

    def spill(self) -> None:
        """Move the device lineage to the host store (its KV rows up to its last snapshot, its snapshots)."""
        self._spill(None)

    def _spill(self, keep: _Entry | None) -> float:
        """Store the device lineage before it is overwritten; `keep`: the entry the caller is about to resume from.
        Returns the seconds spent."""
        t0 = time.perf_counter()
        self._sync_lineage()
        dev, lin = list(self._dev), self._lin
        s_max = max((s.position for s in dev), default=0)
        if not dev or s_max < self.min_resume or len(lin) <= s_max or self.host_budget <= 0:
            self._release(dev)
            return time.perf_counter() - t0
        ids = lin[: s_max + 1]
        ends = tuple(e for e in self._ends if e <= s_max)
        # the entry that shares the most rows (same tokens, same plan ends, up to one of its chunk boundaries) takes the
        # new rows instead of storing the shared ones twice; its rows and snapshots past the boundary are superseded
        ent, start = max(
            ((e, self._shared(e, ids, ends, keep)) for e in self._entries), key=lambda x: x[1], default=(None, 0)
        )
        if start == 0:
            ent = None
        elif start < ent.n_kv:
            ent.chunks = [c for c in ent.chunks if c.stop <= start]
            ent.snaps = [s for s in ent.snaps if s.position <= start]
        old = ent.nbytes if ent is not None else 0
        if self._row_bytes:
            need = self._row_bytes * (s_max - start) + sum(s.nbytes for s in dev)
            if old + need > self.host_budget:
                self.stats.refused += 1
                self._release(dev)
                return time.perf_counter() - t0
            self._fit(need, (keep, ent))
        cuts = sorted({s.position for s in dev if s.position > start} | ({s_max} if s_max > start else set()))
        chunks, a = [], start
        for c in cuts:
            chunks.append(self.decoder.kv_export(a, c))
            a = c
        if chunks:
            self._row_bytes = sum(c.nbytes for c in chunks) // (s_max - start)
        carried = [dataclasses.replace(s, snap=_to_host(s.snap)) for s in dev if s.position > start]
        if ent is None:
            ent = _Entry(ids.copy(), s_max, ends)
            self._entries.append(ent)
        else:
            ent.ids, ent.n_kv, ent.ends = ids.copy(), s_max, ends
        ent.chunks += chunks
        ent.snaps += carried
        while len(ent.snaps) > MAX_SNAPS_PER_ENTRY:  # prompts grow: the lowest boundaries are the least useful
            ent.snaps.remove(min(ent.snaps, key=lambda s: (s.sticky, s.position)))
        self._tick += 1
        ent.tick = self._tick
        if ent.nbytes > self.host_budget:
            self._entries.remove(ent)
            self.stats.refused += 1
        else:
            self._fit(0, (keep, ent))
        self._release(dev)
        return time.perf_counter() - t0

    def clear(self) -> None:
        """Forget everything: the host store and the device lineage; the decoder is reset."""
        self._entries.clear()
        self._dev.clear()
        self._free.clear()
        self._lin, self._ends, self._exact, self._anchor = np.empty(0, dtype=np.int32), [], True, None
        self.decoder.reset()
        self.stats.host_bytes = self.stats.host_entries = self.stats.device_snapshot_bytes = 0
