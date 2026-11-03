"""swift-mk runner: one bs=1 sequence at a time on the C engine, with exact stops, streaming-sized syncs and multi-turn prefix reuse.

No vLLM imports here (the vLLM glue is smk_core_client.py).  Needs an RTX 3090 (sm_86; no other GPU has been tested).

  r = Runner(ctx_max=8192, kv_fp8=0, k=3, group=2)
  for toks, finish, stop_reason in r.run(prompt_ids, GenParams(max_tokens=64, eos=248046), cancelled=lambda: False): ...

Prefix reuse (DeltaNet state cannot be rewound, so the state is checkpointed instead):
  * the engine's device caches always hold ONE lineage: the prompt ids of the last request (`lineage`; main KV positions [0, len), MTP KV pairs p <= len-2);
  * during the prefill of a request, DeltaNet + conv snapshots (host pinned memory, 151 MB each) are taken after the tokens [0, q) for
      - q = len(prompt)-1  (before the last prompt token; the next request that extends / repeats this prompt resumes here: agent loops, completions continuation),
      - q = p+2 where p = last <|im_start|> in the prompt  (chat: the next turn re-renders the previous assistant message WITHOUT the generation-prompt tail
        "<think>\\n", so the first divergence is right after "<|im_start|>assistant\\n"; resuming at p+2 re-prefills only the previous answer + the new user turn);
  * a new request with ids R: L = common prefix of R and the lineage; resume from the snapshot with the largest q <= L-1 (MTP pair q-1 needs token q to be common),
    restore it into the current DeltaNet ring slot, continue the prefill at position q (main KV / MTP KV positions >= q are simply overwritten).  No snapshot -> reset + full prefill.
"""
import ctypes as C
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))
from spec import SpecEngine, make_pld, smk  # noqa: E402
from prefill import FastPrefill  # noqa: E402

IM_START = 248045  # <|im_start|>
STOP, LENGTH = "stop", "length"


@dataclass
class GenParams:
    max_tokens: int
    eos: int | None = None  # primary EOS (finish_reason stop, stop_reason None); None with ignore_eos
    stop_ids: frozenset = frozenset()  # other stop token ids (finish_reason stop, stop_reason = the id)
    temperature: float = 0.0
    top_k: int = 0  # <= 0: disabled (the engine's sampler truncates at 64)
    top_p: float = 1.0
    seed: int | None = None


@dataclass
class Snap:
    pos: int
    dn: torch.Tensor
    conv: torch.Tensor


@dataclass
class Stats:
    prompt: int = 0
    cached: int = 0  # prompt tokens whose prefill was skipped by a snapshot restore
    prefill_s: float = 0.0
    tokens: int = 0
    cycles: int = 0
    gen_s: float = 0.0
    snaps: list = field(default_factory=list)


class Runner:
    def __init__(self, ctx_max, kv_fp8=0, k=3, group=2, max_snaps=4, seg=4096, min_fast=None, warmup=True, log=print, pld=False, host_tables=False):
        """pld: K11 hybrid draft (MTP chain + prompt-lookup tail, verify T <= 8; needs k = 3; +0.62 GiB for the 9-slot DeltaNet ring): False / True / a spec.make_pld(...) options object."""
        assert 0 <= k <= 3 and group >= 1
        assert not pld or k == 3, "the prompt-lookup tail extends the full k = 3 MTP chain"
        self.k, self.group, self.max_snaps, self.log = k, group, max_snaps, log
        self.pld = (make_pld() if pld is True else pld) or None
        t0 = time.time()
        try:
            self.e = SpecEngine(ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=False, tmax=8 if self.pld else 4, host_tables=host_tables)
        except AttributeError as ex:  # ctypes: symbol missing in an old libswiftmk.so
            raise RuntimeError(f"libswiftmk.so is older than the sources ({ex}): run `make -C engine` (or point SMK_BUILD at a current build dir)") from ex
        self.fp = FastPrefill(self.e, seg=seg, **({} if min_fast is None else {"min_fast": min_fast}))
        self.ctx_max = ctx_max
        self.load_s = time.time() - t0
        self.lineage = None  # np.int32 array: tokens whose main KV is valid in the device caches (see module docstring)
        self.snaps: list[Snap] = []
        # all snapshot buffers are allocated up front (cudaHostAlloc of 151 MB costs ~20 ms): len(snaps) + len(pool) == max_snaps
        self.pool: list[Snap] = [Snap(0, torch.empty(48, 48, 128, 128, dtype=torch.float32).pin_memory(), torch.empty(48, 10240 * 3, dtype=torch.bfloat16).pin_memory()) for _ in range(max_snaps)]
        self.last = Stats()
        self._side = torch.cuda.Stream()  # K10: the snapshot D2H copy of a short (folded) prompt runs here, overlapped with the first-token chunk (see _snapshot)
        self._snap_ev = None
        self._snap_pending = None
        if warmup:
            t0 = time.time()
            self._warmup()
            self.warmup_s = time.time() - t0

    # ------------------------------------------------------------------------------------------------ helpers
    @staticmethod
    def slack(group, k, pld=False):
        """Positions above max-model-len the engine needs: a sync launches `group` cycles of up to k+1 tokens (8 with the prompt-lookup tail) past the stop, plus the verify rows."""
        return group * (8 if pld else k + 1) + 16

    def _cur(self):
        torch.cuda.synchronize()
        return int(self.fp.ds[1])

    def _ss(self):
        buf = (C.c_int * 40)()
        torch.cuda.current_stream().synchronize()  # not the whole device: an asynchronous snapshot copy (side stream) may still be running
        smk.check(self.e.lib.smk_spec_read(self.e.sp, 0, buf, 160), "spec_read")
        return list(buf)

    def _free_snaps(self, keep):
        """Keep only the snapshots in `keep`, the rest go back to the pool."""
        self.pool += [s for s in self.snaps if s not in keep]
        self.snaps = list(keep)

    def _snap_sync(self):
        """Wait for an asynchronous snapshot copy (must complete before anything overwrites ring slot `cur` or recycles the buffer)."""
        if self._snap_ev is not None:
            self._snap_ev.synchronize()
            self._snap_ev = None

    def _snap_flush(self):
        """Issue the snapshot a folded prefill left pending (asynchronous copy of ring slot `cur`)."""
        if self._snap_pending is not None:
            args, self._snap_pending = self._snap_pending, None
            self._snapshot(*args, wait=False)

    def _snapshot(self, i, cur=None, wait=True):
        """DeltaNet + conv state of ring slot `cur` (default: the engine's current slot) -> a host snapshot buffer for position i.  wait=False: issue the copies on a side
        stream and return; the caller guarantees nothing writes slot `cur` (the next speculative cycle may) before _snap_sync()."""
        self._snap_sync()
        self._free_snaps([x for x in self.snaps if x.pos != i])  # same position: reuse the buffer
        if not self.pool:  # evict the oldest (lowest position: conversation prompts grow, early boundaries are the least useful)
            old = min(self.snaps, key=lambda x: x.pos)
            self._free_snaps([x for x in self.snaps if x is not old])
        s = self.pool.pop()
        if cur is None:
            cur = self._cur()
        s.pos = i
        if wait:
            s.dn.copy_(self.fp.dn_state[:, cur])
            s.conv.copy_(self.fp.dn_conv[:, cur])
        else:
            ev = torch.cuda.Event()
            ev.record()
            with torch.cuda.stream(self._side):
                self._side.wait_event(ev)
                s.dn.copy_(self.fp.dn_state[:, cur], non_blocking=True)
                s.conv.copy_(self.fp.dn_conv[:, cur], non_blocking=True)
                self._snap_ev = torch.cuda.Event()
                self._snap_ev.record(self._side)
        self.snaps.append(s)

    def _restore(self, s):
        fp, e = self.fp, self.e
        cur = self._cur()
        fp.dn_state[:, cur].copy_(s.dn)
        fp.dn_conv[:, cur].copy_(s.conv)
        fp.ds[19] = 0  # DS_PEND (engine_impl.h): the restored state replaces what the lazy DeltaNet ring still held pending from the previous request (the prefill's materialize would replay it on top)
        fp.ds[0] = s.pos
        e.pos = s.pos
        torch.cuda.synchronize()

    def drop_cache(self):
        self._snap_sync()
        self.lineage = None
        self._free_snaps([])

    def _choose(self, ids):
        """Snapshot to resume from for prompt `ids` (None = cold start)."""
        if self.lineage is None or not self.snaps:
            return None
        n = min(len(ids), len(self.lineage))
        a, b = np.asarray(ids[:n], dtype=np.int32), self.lineage[:n]
        ne = np.flatnonzero(a != b)
        L = int(ne[0]) if len(ne) else n
        ok = [s for s in self.snaps if s.pos <= min(L - 1, len(ids) - 1)]
        return max(ok, key=lambda s: s.pos) if ok else None

    # ------------------------------------------------------------------------------------------------ prefill
    def _prefill(self, ids, q, snap_pos, cancelled):
        """Process ids[q:] (state valid for [0, q)); the last token as the head step.  Snapshots after the tokens [0, s) for s in snap_pos.
        Returns False when cancelled (self.lineage then holds the valid prefix)."""
        e, fp = self.e, self.fp
        n = len(ids) - 1
        bounds = sorted(s for s in set(snap_pos) if q < s <= n)
        i = q
        folded = False
        while i < n:
            # K10 short tail: the last row rides in the final batch of <= min(seg, w4_max) rows (no 20 ms T=1 head step); the snapshot "before the last token" is then the
            # state in the ring slot the batch started from (the folded last row's DeltaNet row goes to the next slot), copied asynchronously.
            if fp.fold and not any(i < b < n for b in bounds) and 2 <= n + 1 - i <= min(fp.seg, fp.w4_max):
                if cancelled():
                    self.lineage = np.asarray(ids[:i], dtype=np.int32)
                    return False
                cur = self._cur()
                fp._ev = []
                fp._segment(ids[i:], i, cur, ids[i + 1 :], head=True)
                if n in bounds:
                    self._snap_pending = (n, cur)  # issued after the first token has been read and yielded (see run): the 151 MB D2H copy occupies the single DMA engine
                folded = True
                break
            if cancelled():
                self.lineage = np.asarray(ids[:i], dtype=np.int32)
                return False
            j = min(n, i + fp.seg)
            nb = next((b for b in bounds if b > i), None)
            if nb is not None and nb < j:
                j = nb
            if j - i >= fp.min_fast:
                cur = self._cur()
                fp._ev = []
                fp._segment(ids[i:j], i, cur, ids[i + 1 : j + 1])
                fp.ds[0] = j
                e.pos = j
                torch.cuda.synchronize()
            else:
                for a in range(i, j, 4):
                    T = min(4, j - a)
                    e._step(ids[a : a + T], 2)
                    nt = (C.c_int * T)(*ids[a + 1 : a + 1 + T])
                    smk.check(e.lib.smk_spec_prefill_chunk(e.sp, T, nt, a, e.st), "spec_prefill_chunk")
            i = j
            if i in bounds:
                self._snapshot(i)
            self.lineage = np.asarray(ids[:i], dtype=np.int32)
        if not folded:
            e._step(ids[-1:], 1)
        if e.sampling:
            smk.check(e.lib.smk_spec_sample_first(e.sp, e.st), "spec_sample_first")
        if self.k:
            smk.check(e.lib.smk_spec_start(e.sp, self.k, e.st), "spec_start")
        self.lineage = np.asarray(ids, dtype=np.int32)
        return True

    def _snap_positions(self, ids):
        if not self.max_snaps:
            return []
        n = len(ids) - 1
        pos = [n]
        for p in range(n, -1, -1):  # last <|im_start|>
            if ids[p] == IM_START:
                if p + 2 > self.fp.w4_max:  # a snapshot at p+2 <= w4_max would split a short prompt into two weight passes (~+55 ms) to save < that much re-prefill later
                    pos.append(p + 2)
                break
        return [s for s in pos if s > 0]

    # ------------------------------------------------------------------------------------------------ generation
    def run(self, ids, p: GenParams, cancelled=lambda: False):
        """Generator of (new_tokens, finish_reason | None, stop_reason); tokens are cut exactly at the stop / length.  Nothing is yielded after a cancel."""
        e, k = self.e, self.k
        ids = [int(t) for t in ids]
        st = self.last = Stats(prompt=len(ids))
        n_new = min(p.max_tokens, self.ctx_max - self.slack(self.group, k, bool(self.pld)) - len(ids))
        if n_new < 1 or len(ids) < 1:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the engine context {self.ctx_max}")
        t0 = time.time()
        self._snap_sync()
        self._snap_pending = None
        if p.temperature > 0:
            seed = p.seed if p.seed is not None else random.getrandbits(62)
            top_k = p.top_k if 0 < p.top_k <= 64 else 64
            e.set_sampling(temperature=p.temperature, top_k=top_k, top_p=p.top_p, seed=seed)
        else:
            e.set_sampling(0.0)
        snap = self._choose(ids)
        if snap is None:
            self._free_snaps([])
            e.reset()
            q = 0
        else:
            self._free_snaps([s for s in self.snaps if s.pos <= snap.pos])
            e.reset_counters()
            self._restore(snap)
            q = snap.pos
        st.cached = q
        if not self._prefill(ids, q, self._snap_positions(ids), cancelled):
            return
        e._emitted = self._ss()[12]
        first = e.read_out_tok()
        use_pld = bool(self.pld) and k == 3
        if use_pld:
            e.enable_pld(self.pld)
            e.pld_begin(ids + [first])
        st.prefill_s = time.time() - t0
        st.snaps = [s.pos for s in self.snaps] + ([self._snap_pending[0]] if self._snap_pending else [])
        t1 = time.time()
        produced = 0
        new = [first]
        while True:
            out, fin, sr = self._cut(new, produced, p, len(ids), n_new)
            produced += len(out)
            st.tokens = produced
            if fin is not None:
                st.gen_s = time.time() - t1
                yield out, fin, sr
                self._snap_flush()
                return
            yield out, None, None
            self._snap_flush()  # K10: the snapshot of a folded short prompt is copied now, after the first token left (a copy in flight delays every later D2H read by ~13 ms)
            self._snap_sync()
            if cancelled():
                st.gen_s = time.time() - t1
                return
            if use_pld:
                new, ss = e.cycles_pld(self.group)
                st.cycles = ss[13]
            elif k:
                new, ss = e.cycles(k, self.group)
                st.cycles = ss[13]
            elif e.sampling:
                new = e.plain_sample(self.group)
            else:
                new = e.decode(self.group)

    @staticmethod
    def _cut(new, produced, p, n_prompt, n_new):
        out = []
        for t in new:
            out.append(t)
            produced += 1
            if p.eos is not None and t == p.eos:
                return out, STOP, None
            if t in p.stop_ids:
                return out, STOP, t
            if produced >= n_new:
                return out, LENGTH, None
        return out, None, None

    def _warmup(self):
        """Compile / autotune the prefill paths and size the scratch (a segment-sized prompt) so the first request is not slow and the VRAM peak is paid up front."""
        e = self.e
        rng = np.random.default_rng(0)
        for n in (40, min(self.fp.seg + 64, self.ctx_max - 64), 130):
            ids = rng.integers(1000, 100000, size=n).tolist()
            for _ in self.run(ids, GenParams(max_tokens=2 * (self.group * (8 if self.pld else self.k + 1)) + 2, temperature=0.0)):
                pass
        self.drop_cache()
        torch.cuda.empty_cache()
