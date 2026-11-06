"""swift-mk continuous batching (--max-num-seqs S > 1): S sequence slots on one set of weights, the running requests decoded together in one weight pass per cycle.

No vLLM imports here (the vLLM glue is smk_core_client.py).  Needs an RTX 3090 (sm_86; no other GPU has been tested).

  r = BatchRunner(ctx_max=8192, max_num_seqs=4, k=3, group=2, pld=True)
  r.start(Task(rid, ids, GenParams(...), emit, done, cancelled))   # admission + prefill of one request (the running ones keep decoding inside it)
  r.decode_step()                                                  # one sync group of cycles for every running request

Scheduling (one thread owns the GPU; smk_core_client._batch_worker drives it):
  * admission: a free slot whose device lineage holds the longest resumable prefix of the prompt (else the host prefix cache, else a cold start on the least recently used free
    slot, whose lineage is spilled to the host store first: Runner._begin).  The prefill, prefix reuse, snapshots and the exactness of a resumed prefill are the Runner's (smk_runner.py).
  * prefills run one at a time (they share the fast prefill's scratch); after every PREFILL_STALL_ROWS x 64 row-layers of a prefill the running requests decode one sync group (the
    FastPrefill layer hook), so a long prompt delays the running streams by the time of a ~512-token prefill (about 0.3 s), not by its whole prefill.
  * decode: B = 1 running request uses the single-sequence path of its slot (MTP k = 3 + the prompt-lookup tail: the speed of --max-num-seqs 1); B >= 2 run batched cycles
    (engine/batch.py) with K_BY_B[B] drafts per sequence (B x (k + 1) <= 8 rows per weight pass).  Every request keeps its own sampling parameters, seed, processors and stops.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from smk_runner import GenParams, Runner, Seq, Stats

# drafts per sequence by the number of running requests: the most that fits the 8 rows of a weight pass (B x (k + 1) <= 8); measured best for every B (docs/PERFORMANCE.md, B x k table)
K_BY_B = {1: 3, 2: 3, 3: 1, 4: 1, 5: 0, 6: 0, 7: 0, 8: 0}
PREFILL_STALL_ROWS = 512  # a prefill pauses for one decode group of the running requests every 512 x 64 row-layers (the work of a 512-token prefill)
# Exact mode (tests): SMK_BATCH_K=k decodes EVERY request with batched cycles of k drafts, whatever the batch size (needs max_num_seqs x (k + 1) <= 8).  A request's tokens are then a
# function of the request alone (every cycle runs the same per-sequence kernels, and the row-wise kernels are row-count invariant: tests/test_batch_engine.py part A), so a request
# decoded next to others equals the same request decoded alone, token for token, greedy and seeded sampling.  Default (unset): the fastest schedule (K_BY_B, the native path at B = 1).
EXACT_K = int(os.environ["SMK_BATCH_K"]) if os.environ.get("SMK_BATCH_K", "") != "" else None


@dataclass(eq=False)
class Task:
    rid: object
    ids: list
    p: GenParams
    emit: Callable  # emit(task, tokens, finish_reason | None, stop_reason)
    done: Callable  # done(task): the request left its slot (finished, aborted or failed)
    cancelled: Callable = lambda: False
    seq: Seq = None
    st: Stats = None
    n_new: int = 0
    produced: int = 0
    gen: list = field(default_factory=list)  # generated tokens (the prompt-lookup proposer's context after a batched phase)
    pld_live: bool = False  # the slot's prompt-lookup proposer follows this request's tokens
    calls: list = field(default_factory=list)  # the decode calls that produced its tokens, in order (tests replay them alone: an exact oracle, tests/test_batch_engine.py)
    t1: float = 0.0
    finished: bool = False


class BatchRunner(Runner):
    def __init__(self, *a, max_num_seqs, **kw):
        assert max_num_seqs > 1
        self.tasks: list[Task] = []  # decoding requests, in admission order
        self.prefilling = None
        self._rows = 0
        self.batched_cycles = self.single_cycles = 0
        self.exact_k = EXACT_K
        assert self.exact_k is None or (0 <= self.exact_k <= 3 and max_num_seqs * (self.exact_k + 1) <= 8), f"SMK_BATCH_K={self.exact_k}: needs max_num_seqs x (k + 1) <= 8"
        super().__init__(*a, max_num_seqs=max_num_seqs, **kw)

    def step(self, waiting, refuse):
        """One scheduler iteration: admit at most one Task of the FIFO `waiting` (a deque) into a free slot (its prefill runs now, the running requests decode inside it), then one
        sync group of cycles for every running request.  refuse(task, exception): a request whose admission failed (ValueError: its prompt does not fit; anything else is re-raised)."""
        while waiting and self.free():
            t = waiting.popleft()
            if t.cancelled():
                t.done(t)
                continue
            try:
                self.start(t)
            except Exception as ex:  # noqa: BLE001
                refuse(t, ex)
                if not isinstance(ex, ValueError):
                    raise
            break  # one admission per iteration: the running requests get their next group first
        self.decode_step()

    def kb(self, B):
        """Drafts per sequence of a batched cycle of B sequences."""
        return min(K_BY_B[B] if self.exact_k is None else self.exact_k, self.k)

    # ------------------------------------------------------------------------------------------------ admission
    def free(self):
        return [q for q in self.seqs if not q.busy]

    def _slot_for(self, ids):
        """The free slot a prompt goes to: the one whose device lineage resumes it furthest, else the least recently used (an empty one first)."""
        ends = set(self._plan(ids)[0])
        best, bpos = None, -1
        cur = self.seq
        for q in self.free():
            self.seq = q
            s = self._choose(ids, ends)
            if s is not None and s.pos > bpos:
                best, bpos = q, s.pos
        self.seq = cur
        if best is None:
            best = min(self.free(), key=lambda q: (q.lineage is not None, q.used))
        return best

    def start(self, t: Task):
        """Admit t into a free slot and prefill it (the running requests decode inside the prefill); afterwards t decodes with the others.  Raises ValueError for a prompt that does not fit."""
        q = self._slot_for([int(x) for x in t.ids])
        q.busy = True
        t.seq = q
        self.seq = q
        try:
            b = self._begin(t.ids, t.p)
            t.ids, t.st, t.n_new = b.ids, b.st, b.n_new
            q.fp.layer_hook = self._layer_hook
            self.prefilling, self._rows = t, 0
            try:
                ok = self._prefill(t.ids, b.q, b.ends, b.folded, t.cancelled, False)
            finally:
                q.fp.layer_hook = None
                self.prefilling = None
                self.seq = q
        except BaseException:
            self._release(t)
            raise
        if not ok or t.cancelled():  # cancelled during the prefill: nothing is delivered (the slot keeps the prefilled lineage)
            self._release(t)
            return
        e = q.e
        e._emitted = self._ss()[12]
        e.drafted = self.k  # smk_spec_start drafted k tokens
        first = e.read_out_tok()
        t.st.prefill_s = time.time() - b.t0
        t.st.snaps = [s.pos for s in q.snaps]
        t.t1 = time.time()
        self.tasks.append(t)
        self._deliver(t, [first])

    def _layer_hook(self, rows):
        self._rows += rows
        if self._rows >= 64 * PREFILL_STALL_ROWS and self.tasks:
            self._rows = 0
            self.decode_step()

    # ------------------------------------------------------------------------------------------------ decode
    def decode_step(self):
        """One sync group (`group` cycles) for every running request; tokens go out through each task's emit."""
        for t in [t for t in self.tasks if t.cancelled()]:
            t.st.gen_s = time.time() - t.t1
            self._release(t)
        ts = list(self.tasks)
        if not ts:
            return
        g = self.group
        if len(ts) == 1 and self.k and self.exact_k is None:
            t = ts[0]
            e = t.seq.e
            if e.drafted < self.k:  # coming out of a batched phase with fewer drafts: one cycle that verifies what is there and drafts the full chain
                t.calls.append(("batch", e.drafted, self.k, 1))
                (new,), _ = self.bd.cycles([e], e.drafted, self.k, 1)
                t.pld_live = False
                t.st.cycles += 1
                self.batched_cycles += 1
            elif self.pld:
                t.calls.append(("pld", g, not t.pld_live))
                if not t.pld_live:  # the proposer's context: the prompt and everything generated so far (a batched phase did not feed it)
                    if not e.pld:  # first use on this slot (the warm-up enables it on every slot)
                        e.enable_pld(self.pld)
                    e.pld_begin(t.ids + t.gen)
                    t.pld_live = True
                new, _ = e.cycles_pld(g)
                t.st.cycles += g
                self.single_cycles += g
            else:
                t.calls.append(("mtp", self.k, g))
                new, _ = e.cycles(self.k, g)
                t.st.cycles += g
                self.single_cycles += g
            news = [new]
        else:
            k = self.kb(len(ts))
            kv0 = min([k] + [t.seq.e.drafted for t in ts])
            news, _ = self.bd.cycles([t.seq.e for t in ts], kv0, k, g)
            for t in ts:
                t.calls.append(("batch", kv0, k, g))
                t.pld_live = False
                t.st.cycles += g
            self.batched_cycles += g
        for t, new in zip(ts, news):
            self._deliver(t, new)

    def _deliver(self, t, new):
        out, fin, sr = self._cut(new, t.produced, t.p, len(t.ids), t.n_new)
        t.produced += len(out)
        t.gen += out
        t.st.tokens = t.produced
        if fin is not None:
            t.st.gen_s = time.time() - t.t1
        if out or fin is not None:
            t.emit(t, out, fin, sr)
        if fin is not None:
            self._release(t)

    def _release(self, t):
        if t.finished:
            return
        t.finished = True
        if t in self.tasks:
            self.tasks.remove(t)
        if t.seq is not None:
            t.seq.busy = False
            t.seq.used = time.time()
        t.done(t)

    def fail_all(self):
        """After an engine error: every running request leaves its slot (the caller reports the error to the clients)."""
        for t in list(self.tasks):
            self._release(t)
        for q in self.seqs:
            q.busy = False

    # ------------------------------------------------------------------------------------------------ startup
    def _warmup(self):
        """The Runner's warm-up on slot 0 (prefill paths, scratch size), a short request on every other slot (its own prefill / start / cycle graphs, the prompt-lookup graphs), and the
        batched cycle graphs of every (B, verified drafts, drafts) the scheduler can launch, greedy and sampling (captured now: a lazy capture under memory pressure can fail)."""
        super()._warmup()
        rng = np.random.default_rng(1)
        for q in self.seqs[1:]:
            self.seq = q
            for n in (40, 130):
                ids = rng.integers(1000, 100000, size=n).tolist()
                for _ in self.run(ids, GenParams(max_tokens=2 * (self.group * (8 if self.pld else self.k + 1)) + 2, temperature=0.0)):
                    pass
        self.seq = self.seqs[0]
        for B in range(1, self.max_num_seqs + 1):
            k = self.kb(B) if B > 1 or self.exact_k is not None or not self.k else self.k  # B = 1 (native path): the transition cycle out of a batched phase drafts the full chain
            for kv0 in range(0, k + 1):
                for samp in (0, 1):
                    self.bd.prewarm(B, kv0, k, samp)
        self.drop_cache()
        for q in self.seqs:
            q.used = 0.0
