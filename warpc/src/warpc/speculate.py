"""Greedy speculative decoding with the checkpoint's MTP head: the head drafts k <= k_max tokens, the target verifies
k + 1 rows in one step, the accepted prefix stays, the rest is rolled back by not advancing.

One CUDA graph per cycle (verify, accept, draft), built by composing the target's and the draft's `Runtime.enqueue`
inside one capture: the host launches a few cycles at a time and reads the output ring once per group, and no decision
needs the host (the accepted count, the positions, the anchor and the drafts are device scalars). A cycle:

  verify   target rows [anchor, d1 .. dk] at positions P .. P + k, final norm + head of every row   -> ids[0..k]
  accept   m = leading rows whose draft equals the previous row's argmax; a = min(m + 1, cap, budget, first stop + 1)
           ring append of ids[0 .. a - 1], anchor = ids[a - 1], P += a, the backend's cursors by `a`
  draft    catch-up of the pairs (hidden of row t, ids[t]) at MTP positions P .. P + k (rows >= a are junk: the next
           cycle rewrites them), d1 from row a - 1, then k - 1 single-row chain passes on the head's own hidden

Rollback costs nothing: KV is positional and the DeltaNet state is lazy (a rejected row sits beyond the pending count
and is overwritten). The clamp `a <= natural` keeps the cache holding exactly the emitted tokens (no row past
`max_new_tokens` or a stop id): after `decode(n)` the cache holds the anchor's predecessors and `position` grew by n.

Greedy MTP is not bitwise equal to plain greedy decode: a verify row is computed inside a (k + 1)-row step and the hand
kernels' rounding depends on the step's row count, so a near-tie can flip (measured: a T = 3 or 4 verify row differs
from the T = 1 step in nearly every logit by up to 0.15). k = 1 equals plain decode bitwise as long as a T = 2 step
attends the same keys with the same split geometry as a T = 1 step: on bf16 until the geometry of the kv length
changes (every 80 - 320 positions), on int4_rot only below position 495 (a T = 2 row reads one more key as int4 than
the T = 1 step, which reads it from the exact ring); beyond that it differs by rounding like k >= 2 (the hand engine
does the same). k = 0 is plain decode.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import torch

from . import pinned
from .kvcache import KVChunk
from .runtime import Runtime, Snapshot

RING = 4096  # device output ring (tokens); fetched after every group of cycles, far below RING / 2
LOG = 4096  # device cycle log (P, a) entries, a ring
CAP_N = 16  # entries of the forced-acceptance pattern
STOP_N = 16  # stop ids of one decode
GROUP = 8  # cycles launched per host round trip
HIST_N = 8  # histogram bins of accepted drafts m (the last bin counts m >= 7)
# the device state vector (int32)
S_OUTN, S_NCYC, S_BUDGET, S_A, S_CAPN, S_HIST = 0, 1, 2, 3, 4, 8
ST_N = S_HIST + HIST_N


@dataclass(frozen=True)
class SpecStats:
    """Since the last `reset_counters`: cycles run, tokens emitted, and cycles by accepted drafts m = 0 .. k_max
    (m is the natural count: a cycle clamped by `cap`, the budget or a stop id counts under its natural m)."""

    cycles: int
    emitted: int
    accepted: tuple[int, ...]


class Speculator:
    """The speculative `Decoder`. `target` verifies, `draft` (the MTP head graph, `frontend.build_draft`) drafts; both
    are runtimes of one arena. `k_max` <= target.rows_max - 1 bounds `k` (drafts per cycle, 0 = plain steps through
    the target; 1 is nearest to plain decode, see above)."""

    def __init__(self, target: Runtime, draft: Runtime, *, k_max: int = 3):
        if not 1 <= k_max <= min(3, target._rows_max - 1):
            raise ValueError(
                f"k_max {k_max}: the head drafts 1 .. 3 tokens within the target's {target._rows_max} rows"
            )
        if target._ctx_max != draft._ctx_max:
            raise ValueError("the target and the draft head need the same number of KV positions")
        self._target, self._draft, self.k_max = target, draft, k_max
        self._k = k_max
        dev = "cuda"
        hidden = target.outputs["hidden"].shape[-1]
        zeros = lambda *s, dt=torch.int32: torch.zeros(s, dtype=dt, device=dev)  # noqa: E731
        self._st = zeros(ST_N)
        self._ring = zeros(RING)
        self._log = zeros(LOG, 2)
        self._cap = zeros(CAP_N)
        self._stop = zeros(STOP_N)
        self._arange = torch.arange(target._rows_max, dtype=torch.int32, device=dev)
        self._last_hidden = zeros(1, hidden, dt=torch.bfloat16)
        self._hr = zeros(k_max + 1, hidden, dt=torch.bfloat16)  # the head's post-norm hidden of drafts 1 .. k
        self._host_st = pinned.empty(ST_N, torch.int32)
        self._host_ring = pinned.empty(RING, torch.int32)
        self._draft_states = set(draft._graph.states)
        self._cycle_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._start_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.reset()  # the cap pattern's length is a divisor: set before the first cycle runs
        self._warm()
        self.reset()

    # ---- the device program

    def _one(self, i: int) -> torch.Tensor:
        return self._st[i : i + 1]

    def _accept(self, k: int) -> None:
        """Greedy acceptance and every device-side bookkeeping of one cycle, then the target's commit. The torch
        sequence of stage 1 (60 launches, 65 us per replay); a fused kernel replaces it in a later stage."""
        t, d, st = self._target, self._draft, self._st
        T = k + 1
        tok, ids, hidden, ar = t.inputs["tokens"], t.outputs["ids"], t.outputs["hidden"], self._arange[:T]
        m = (tok[1:T] == ids[:k]).to(torch.int32).cumprod(0).sum(0, keepdim=True, dtype=torch.int32)
        budget, ncyc, outn = self._one(S_BUDGET), self._one(S_NCYC), self._one(S_OUTN)
        live = budget > 0
        cap = torch.index_select(self._cap, 0, ncyc % self._one(S_CAPN))
        a = torch.minimum(torch.minimum(m + 1, cap), budget)
        hit = (ids[:T, None] == self._stop[None, :]).any(1) & (ar < a)
        first = torch.where(hit, ar, T).amin(0, keepdim=True)
        stopped = first < T
        a = torch.where(stopped, first + 1, a)
        ring_at = ((outn + ar) % RING).long()
        self._ring.index_put_((ring_at,), torch.where(ar < a, ids[:T], self._ring[ring_at]))
        am1 = torch.clamp(a - 1, min=0)
        some = a > 0
        tok[0:1].copy_(torch.where(some, torch.index_select(ids, 0, am1), tok[0:1]))
        self._last_hidden.copy_(torch.where(some, torch.index_select(hidden, 0, am1), self._last_hidden))
        d.inputs["tokens"][:T].copy_(ids[:T])
        d.inputs["sel"].copy_(am1)
        d.pos.copy_(t.pos)
        at = (ncyc % LOG).long()
        entry = torch.cat([t.pos, a])[None]
        self._log.index_put_((at,), torch.where(live, entry, self._log[at]))
        bin_ = (S_HIST + torch.clamp(m, max=HIST_N - 1)).long()
        st.index_put_((bin_,), st[bin_] + live.to(torch.int32))
        self._one(S_BUDGET).copy_(torch.where(stopped, 0, budget - a))
        self._one(S_A).copy_(a)
        outn.add_(a)
        ncyc.add_(live.to(torch.int32))
        t.commit(T, a)

    def _draft_passes(self, k: int, rows: int) -> None:
        """The head's catch-up over `rows` rows (positions and inputs set by the caller), then its k - 1 chain passes
        at P + a + j - 1 (the target's position is P + a by now); d1 .. dk land in the target's token rows 1 .. k."""
        t, d = self._target, self._draft
        d.enqueue(rows, ("hidden", "ids"), advance=False)
        self._hr[0:1].copy_(d.outputs["hidden"])
        t.inputs["tokens"][1:2].copy_(d.outputs["ids"])
        d.inputs["sel"].zero_()
        for j in range(1, k):
            d.inputs["tokens"][:1].copy_(t.inputs["tokens"][j : j + 1])
            d.inputs["hidden"][:1].copy_(self._hr[j - 1 : j])
            d.pos.copy_(t.pos).add_(j - 1)
            d.enqueue(1, ("hidden", "ids"), advance=False)
            self._hr[j : j + 1].copy_(d.outputs["hidden"])
            t.inputs["tokens"][j + 1 : j + 2].copy_(d.outputs["ids"])

    def _cycle(self, k: int) -> None:
        t, d = self._target, self._draft
        t.enqueue(k + 1, ("ids", "hidden"), advance=False)
        self._accept(k)
        d.inputs["hidden"][: k + 1].copy_(t.outputs["hidden"][: k + 1])
        self._draft_passes(k, k + 1)

    def _start(self, k: int) -> None:
        """The pair of the last cached row (its hidden, the anchor) at MTP position P - 1, and drafts d1 .. dk. One
        catch-up row, as the hand's smk_spec_start: the head's kernels are not bitwise row-count invariant."""
        t, d = self._target, self._draft
        d.inputs["tokens"][:1].copy_(t.inputs["tokens"][:1])
        d.inputs["hidden"][:1].copy_(self._last_hidden)
        d.inputs["sel"].zero_()
        d.pos.copy_(t.pos).sub_(1)
        self._draft_passes(k, 1)

    def _warm(self) -> None:
        """Eager runs of every program before the arena freezes: kernel modules load, the scratch reaches its size."""
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for k in range(1, self.k_max + 1):
                self._target.pos.fill_(1)
                self._start(k)
                self._cycle(k)
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()

    def finalize(self) -> None:
        """Capture the cycle and start graphs (after the arena froze its scratch)."""
        for k in range(1, self.k_max + 1):
            self._target.pos.fill_(1)
            self._cycle_graphs[k] = self._target.capture(lambda k=k: self._cycle(k))
            self._start_graphs[k] = self._target.capture(lambda k=k: self._start(k))
        self.reset()

    # ---- state

    @property
    def k(self) -> int:
        return self._k

    @k.setter
    def k(self, k: int) -> None:
        if not 0 <= k <= self.k_max:
            raise ValueError(f"k {k} is not within 0 .. {self.k_max}")
        self._k = k

    @property
    def position(self) -> int:
        return self._target.position

    @position.setter
    def position(self, p: int) -> None:
        self._target.position = p

    @property
    def anchor(self) -> int | None:
        """The token to feed next (the last one `append` or `decode` returned, not in the cache); None before an
        append with a head step. Reads the device row: it is never recomputed on the host."""
        return int(self._target.inputs["tokens"][0]) if self._anchor_ok else None

    @property
    def cap(self) -> torch.Tensor:
        """int32 [n] on the device: cycle i accepts at most cap[i % n] rows (default [k_max + 1]: no clamp)."""
        return self._cap[: int(self._st[S_CAPN])]

    @cap.setter
    def cap(self, pattern: Sequence[int] | torch.Tensor | None) -> None:
        """The hand engine's `set_force` (the exactness test hook): None restores the default."""
        pat = [self.k_max + 1] if pattern is None else [int(v) for v in pattern]
        if not 1 <= len(pat) <= CAP_N or not all(1 <= v <= self.k_max + 1 for v in pat):
            raise ValueError(f"cap: 1 .. {CAP_N} values within 1 .. {self.k_max + 1}, got {pat}")
        self._cap[: len(pat)].copy_(torch.tensor(pat, dtype=torch.int32))
        self._one(S_CAPN).fill_(len(pat))

    @property
    def stats(self) -> SpecStats:
        st = self._st.tolist()
        return SpecStats(st[S_NCYC], st[S_OUTN], tuple(st[S_HIST : S_HIST + self.k_max + 1]))

    @property
    def drafts(self) -> list[int]:
        """d1 .. dk of the cycle that runs next."""
        return self._target.inputs["tokens"][1 : self._k + 1].tolist()

    @property
    def draft_hidden(self) -> torch.Tensor:
        """bf16 [k_max + 1, H]: the head's post-norm hidden of d1 .. dk (row j = draft j + 1), last draft stage."""
        return self._hr

    def cycle_log(self) -> list[tuple[int, int]]:
        """(P, a) of every cycle since `reset_counters` (the last LOG of them): the position it started at and the
        rows it kept."""
        n = int(self._st[S_NCYC])
        log = self._log.cpu().tolist()
        return [tuple(log[i % LOG]) for i in range(max(0, n - LOG), n)]

    def reset_counters(self) -> None:
        """Zero the cycle counter, the emitted count and the histogram; the cache and the drafts stay."""
        self._st[S_OUTN : S_OUTN + 1].zero_()
        self._st[S_NCYC : S_NCYC + 1].zero_()
        self._st[S_HIST : S_HIST + HIST_N].zero_()
        self._outn = 0

    def reset(self) -> None:
        self._target.reset()
        self._draft.reset()
        self._st.zero_()
        self._ring.zero_()
        self._stop.fill_(-1)
        self._anchor_ok = False
        self._drafts_k = 0
        self._outn = 0
        self._last_hidden_pos = 0
        self.cap = None

    def _have_hidden(self) -> bool:
        """last_hidden is the final-norm row of the last cached token."""
        return self.position > 0 and self._last_hidden_pos == self.position

    # ---- Decoder

    def _commit_pair(self, tokens: Sequence[int], hidden: torch.Tensor, pos0: int) -> None:
        """MTP KV rows pos0 .. pos0 + T - 1 from the pairs (hidden[t], tokens[t]), in passes of at most the draft
        runtime's rows (a window chunk has up to 4096 rows); a missing token leaves its row junk until the pair is
        committed again. `hidden` is consumed here: a window's view is overwritten by the next window."""
        d, T, step = self._draft, hidden.shape[0], self._draft._rows_max
        toks = [int(t) for t in tokens]
        for j in range(0, T, step):
            n = min(step, T - j)
            d.inputs["hidden"][:n].copy_(hidden[j : j + n])
            part = toks[j : j + n]
            d.inputs["tokens"][: len(part)].copy_(torch.tensor(part, dtype=torch.int32))
            d.pos.fill_(pos0 + j)
            d.graph(n, ()).replay()

    def append(self, ids: Sequence[int], *, head: bool = True, ends: Sequence[int] | None = None) -> int:
        """The target's append plus, for every chunk, the MTP pairs of its rows; then (with `head`) the drafts of
        `k` (the hand's `prefill_spec`). Continuing a cache, `ids` starts with the token that has no pair yet (the
        anchor after a decode): the pair of the last cached row is committed first."""
        ids = [int(i) for i in ids]
        t = self._target
        self._anchor_ok = False
        self._drafts_k = 0
        if self._have_hidden():
            self._commit_pair(ids[:1], self._last_hidden, t.position - 1)

        def on_hidden(hidden: torch.Tensor, pos0: int, next_ids: Sequence[int]) -> None:
            self._commit_pair(next_ids, hidden, pos0)
            self._last_hidden.copy_(hidden[-1:])
            self._last_hidden_pos = pos0 + hidden.shape[0]

        first = t.append(ids, head=head, on_hidden=on_hidden, ends=ends)
        if not head:
            return first
        self._last_hidden.copy_(t.outputs["hidden"][:1])
        self._last_hidden_pos = t.position
        self._anchor_ok = True
        if self._k:
            self.start(self._k)
        return first

    def start(self, k: int) -> None:
        """Commit the pair of the last cached row and draft d1 .. dk (the hand's `smk_spec_start`): needs the anchor
        and the hidden row of the last cached token."""
        if not 1 <= k <= self.k_max:
            raise ValueError(f"k {k} is not within 1 .. {self.k_max}")
        if not (self._anchor_ok and self._have_hidden()):
            raise ValueError("start needs an append with a head step (or a speculative snapshot) first")
        self._start_graphs[k].replay()
        self._drafts_k = k

    def decode(self, n: int, *, stop: Collection[int] = ()) -> list[int]:
        """Exactly n tokens after the anchor, or up to and including the first stop id, by speculative cycles at the
        current `k`. The cache then holds every token before the last one returned (the new anchor): `position` grew
        by the number returned. A cycle writes target and MTP KV rows up to P + 2k - 1 before the clamp decides how
        many it keeps, so `position + n + 2k - 1` must not exceed the KV cache. Greedy MTP can differ from plain
        greedy at near-ties (module docstring)."""
        t, k = self._target, self._k
        if not self._anchor_ok:
            raise ValueError("decode continues an append with a head step: there is no anchor")
        if n < 0:
            raise ValueError(f"decode of {n} tokens")
        if k == 0:
            self._drafts_k = 0
            return t.decode(n, stop=stop)
        if n == 0:
            return []
        if t.position + n + 2 * k - 1 > t._ctx_max:
            raise ValueError(
                f"{t.position} cached + {n} new tokens + {2 * k - 1} rows of speculation exceed the KV cache of"
                f" {t._ctx_max} positions"
            )
        stop = sorted({int(s) for s in stop})
        if len(stop) > STOP_N:
            raise ValueError(f"at most {STOP_N} stop ids, got {len(stop)}")
        if not self._have_hidden():
            raise ValueError("speculation needs the hidden row of the last cached token: append with a head step first")
        if self._drafts_k != k:
            self.start(k)
        self._stop.copy_(torch.tensor(stop + [-1] * (STOP_N - len(stop)), dtype=torch.int32))
        self._one(S_BUDGET).fill_(n)
        out = []
        while True:
            for _ in range(max(1, min(GROUP, (n - len(out)) // (k + 1)))):
                self._cycle_graphs[k].replay()
            out += self._fetch()
            if int(self._host_st[S_BUDGET]) == 0:
                break
        self._stop.fill_(-1)
        t.position += len(out)
        self._last_hidden_pos = t.position
        return out

    def _fetch(self) -> list[int]:
        """The tokens emitted since the last fetch, and the device state in `_host_st`: one synchronisation."""
        self._host_st.copy_(self._st, non_blocking=True)
        self._host_ring.copy_(self._ring, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        outn = int(self._host_st[S_OUTN])
        new = [int(self._host_ring[i % RING]) for i in range(self._outn, outn)]
        self._outn = outn
        return new

    def cycles(self, n: int) -> list[int]:
        """n cycles at the current `k` with no token budget and no stop (the hand engine's `cycles`): the tokens
        they emitted, as many as they accepted; `position` grows by their number. For comparing with the hand engine
        and timing a cycle: `decode` is the exact-count API."""
        t, k = self._target, self._k
        if not (self._anchor_ok and self._have_hidden()) or k == 0:
            raise ValueError("cycles needs k >= 1 and an append with a head step")
        if t.position + n * (k + 1) + 2 * k - 1 > t._ctx_max:
            raise ValueError(f"{n} cycles from position {t.position} exceed the KV cache of {t._ctx_max} positions")
        if self._drafts_k != k:
            self.start(k)
        self._stop.fill_(-1)
        self._one(S_BUDGET).fill_(2**30)
        for _ in range(n):
            self._cycle_graphs[k].replay()
        out = self._fetch()
        t.position += len(out)
        self._last_hidden_pos = t.position
        return out

    def snapshot(self, into: Snapshot | None = None, *, device: bool = False) -> Snapshot:
        """The target's snapshot plus the draft's non-positional states, the anchor (the device token row) and the
        final-norm row of the last cached token: `restore` can recompute the MTP pair and draft again from them."""
        parts = self._split(into.states) if into is not None else (None, None)
        snaps = []
        for rt, states in zip((self._target, self._draft), parts, strict=True):
            into_part = None if states is None else Snapshot(0, None, None, states, into.device)
            snaps.append(rt.snapshot(into_part, device=device if into is None else into.device))
        states = {**snaps[0].states, **snaps[1].states}
        anchor = int(self._target.inputs["tokens"][0]) if self._anchor_ok else None
        hidden = None
        if self._have_hidden():
            on_device = device if into is None else into.device
            hidden = self._last_hidden[0].clone() if on_device else pinned.like(self._last_hidden[0])
        snap = Snapshot(self.position, anchor, hidden, states, device if into is None else into.device)
        if into is not None:
            into.position, into.anchor, into.hidden = snap.position, snap.anchor, snap.hidden
            return into
        return snap

    def _split(self, states: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        mine = {k: v for k, v in states.items() if k.removesuffix(".aux") not in self._draft_states}
        return mine, {k: v for k, v in states.items() if k not in mine}

    def restore(self, snap: Snapshot) -> None:
        """Back to `snap`; the drafts are invalid until `start` (the next `decode` runs it from the snapshot's anchor
        and hidden row). A snapshot without a draft state (`kv_aux` of the MTP head) raises ValueError."""
        target_states, draft_states = self._split(snap.states)
        for rt, states in ((self._target, target_states), (self._draft, draft_states)):
            rt.restore(Snapshot(snap.position, None, None, states, snap.device))
        self._anchor_ok, self._drafts_k = snap.anchor is not None, 0
        if snap.anchor is not None:
            self._target.inputs["tokens"][0:1].fill_(snap.anchor)
        self._last_hidden_pos = -1
        if snap.hidden is not None:
            self._last_hidden.copy_(snap.hidden[None])
            self._last_hidden_pos = snap.position

    def kv_export(self, start: int, stop: int, into: KVChunk | None = None) -> KVChunk:
        """Rows [start, stop) of the target's KV states and of the head's ("mtp.kv/...")."""
        t = self._target.kv_export(start, stop, into)
        self._draft.position = self.position  # the head's cache holds the same positions; only the target tracks them
        d = self._draft.kv_export(start, stop, None if into is None else into)
        return KVChunk(start, stop, {**t.planes, **d.planes})

    def kv_import(self, chunk: KVChunk, start: int) -> None:
        self._target.kv_import(chunk, start)
        self._draft.kv_import(chunk, start)
