"""swift-mk runner: one bs=1 sequence at a time on the C engine, with exact stops, streaming-sized syncs and multi-turn prefix reuse.

No vLLM imports here (the vLLM glue is smk_core_client.py).  Needs an RTX 3090 (sm_86; no other GPU has been tested).

  r = Runner(ctx_max=8192, kv_fp8=0, k=3, group=2)
  for toks, finish, stop_reason in r.run(prompt_ids, GenParams(max_tokens=64, eos=248046), cancelled=lambda: False): ...

Prefix reuse (DeltaNet state cannot be rewound, so the state is checkpointed instead):
  * the engine's device caches always hold ONE lineage: the prompt ids of the last request (`lineage`; main KV positions [0, len), MTP KV pairs p <= len-2);
  * the prefill of a prompt is cut into segments by `_plan`, a function of the prompt tokens ALONE (the fast prefill's numerics depend on where the segments start and end: the
    fla chunking of the DeltaNet scan, the GEMM kernel of a segment <= w4_max rows, the folded last row).  A request resumed from a snapshot continues that same plan, so its state
    is bit-identical to a cold prefill of the same prompt and the output of a request depends on (prompt, sampling parameters, seed) only, never on the cache;
  * during the prefill DeltaNet + conv snapshots (host pinned memory, 151 MB each) are taken at the last two segment ends of the plan: segments end at <= seg tokens, at a chat
    turn boundary q = p+2 (p = a <|im_start|> position; the next turn re-renders the previous assistant message WITHOUT the generation-prompt tail "<think>\\n", so the first
    divergence is right after "<|im_start|>assistant\\n") that is more than w4_max tokens after the previous end, and at len(prompt)-1 when the last row is not folded (a repeated prompt);
  * a new request with ids R: L = common prefix of R and the lineage; resume from the snapshot with the largest q <= L-1 (MTP pair q-1 needs token q to be common) that is a segment
    end of R's plan, restore it into the current DeltaNet ring slot, continue the prefill at position q (main KV / MTP KV positions >= q are simply overwritten).  No snapshot -> reset + full prefill.
  * host tier (prefix_store.py): before the device lineage is overwritten by a request that does not continue it, its KV rows up to its last snapshot and its snapshots are SPILLED to a
    pinned host store (budget `prefix_cache_gb`); a later request that continues a stored conversation RESTORES it (KV rows, then the snapshot) and prefills only the new tokens.  Same exactness.
"""
import ctypes as C
import os
import random
import sys
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

ENGINE = Path(__file__).resolve().parent.parent / "engine"
sys.path.insert(0, str(ENGINE))
from spec import SpecEngine, make_pld, smk  # noqa: E402
from prefill import FastPrefill  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_store import Chunk, DfWindow, PrefixStore  # noqa: E402

IM_START = 248045  # <|im_start|>
MAX_TOP_K = 64  # the engine's sampler keeps at most the 64 highest logits (top_k > 64 is rejected by the server; top_k <= 0 = no top-k filter = these 64)
STOP, LENGTH = "stop", "length"


@dataclass
class GenParams:
    max_tokens: int
    eos: int | None = None  # primary EOS (finish_reason stop, stop_reason None); None with ignore_eos
    stop_ids: frozenset = frozenset()  # other stop token ids (finish_reason stop, stop_reason = the id)
    temperature: float = 0.0
    top_k: int = 0  # 1..MAX_TOP_K; <= 0: no top-k filter (the sampler then ranks the MAX_TOP_K best candidates)
    top_p: float = 1.0
    seed: int | None = None
    # request-level logit processors (vLLM / OpenAI semantics, engine/src/proc.h); the defaults are off
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    min_tokens: int = 0
    min_tokens_ids: frozenset = frozenset()  # tokens masked until min_tokens are generated (eos + stop token ids)
    logit_bias: dict | None = None
    allowed_token_ids: tuple | None = None

    def procs(self, prompt_ids):
        """spec.make_procs arguments of the elementwise processors, None when none is active (min_p is a sampler parameter)."""
        if not (self.presence_penalty or self.frequency_penalty or self.repetition_penalty != 1.0 or self.min_tokens or self.logit_bias or self.allowed_token_ids):
            return None
        return dict(prompt=prompt_ids, presence=self.presence_penalty, frequency=self.frequency_penalty, repetition=self.repetition_penalty, min_tokens=self.min_tokens,
                    bias=self.logit_bias, allowed=self.allowed_token_ids, stop=sorted(self.min_tokens_ids))


@dataclass(eq=False)  # identity comparison: the fields are tensors (== is not a bool), and list.remove / `in` must find THIS buffer
class Snap:
    pos: int
    dn: torch.Tensor
    conv: torch.Tensor
    aux: object = None  # engine-specific extra cache state saved with the snapshot (e.g. an exact recent-token ring); None for engines that have none
    sticky: bool = False  # the system-prompt checkpoint: kept when a conversation outgrows its snapshot budget (the lowest position would otherwise go first)


@dataclass
class Stats:
    prompt: int = 0
    cached: int = 0  # prompt tokens whose prefill was skipped by a snapshot restore
    prefill_s: float = 0.0
    tokens: int = 0
    cycles: int = 0
    gen_s: float = 0.0
    snaps: list = field(default_factory=list)
    source: str = "cold"  # where the resume point came from: cold | device | host
    spill_s: float = 0.0  # time spent moving the previous device lineage to the host store
    restore_s: float = 0.0  # time spent restoring a stored lineage (KV rows + snapshot)
    route: str = "mtp"  # the draft path that decoded the request: mtp (MTP chain + prompt-lookup tail) | dflash2 (block drafter)
    hist: list = field(default_factory=list)  # dflash2: cycles by number of accepted drafts (0..7)


class Seq:
    """One sequence slot of the Runner: its engine (a SpecEngine, or a batched-decode slot view of one, engine/batch.py), its fast prefill, the lineage its device caches hold and the
    host snapshots of that lineage.  A Runner with max_num_seqs = 1 has one (the engine and caches of every earlier release)."""

    def __init__(self, idx, e, fp):
        self.idx, self.e, self.fp = idx, e, fp
        self.lineage = None  # np.int32 array: tokens whose main KV is valid in the device caches (see module docstring)
        self.snaps: list[Snap] = []
        self.procs_on = False  # the engine's request processors are on (switched off once by the next request without any)
        self.used = 0.0  # time.time() of the last release (LRU choice of the slot a new conversation overwrites)
        self.busy = False  # a request is prefilling or decoding on this slot


def _seq_attr(name):
    return property(lambda self: getattr(self.seq, name), lambda self, v: setattr(self.seq, name, v))


class Runner:
    # the attributes of the CURRENT sequence slot (self.seq); one slot unless max_num_seqs > 1 (server/smk_batch.py selects the slot of the request it works on)
    e = property(lambda self: self.seq.e)
    fp = property(lambda self: self.seq.fp)
    lineage = _seq_attr("lineage")
    snaps = _seq_attr("snaps")
    _procs_on = _seq_attr("procs_on")

    def __init__(self, ctx_max, kv_fp8=0, k=3, group=2, max_snaps=4, seg=4096, min_fast=None, warmup=True, log=print, pld=False, host_tables=False, prefix_cache_gb=0.0, min_spill=1024, system_checkpoint=True, group_prefill=True,
                 drafter="mtp", dflash_path=None, dflash_codebooks_host=False, dflash_lookup=False, max_num_seqs=1):
        """pld: K11 hybrid draft (MTP chain + prompt-lookup tail, verify T <= 8; needs k = 3; +0.56 GiB for the 9-slot DeltaNet ring): False / True / a spec.make_pld(...) options object.
        drafter: "mtp" (default) or "dflash2": greedy requests without logit processors are decoded by the in-engine DFlash2 block drafter (engine/dflash.py: 7 drafts per cycle, verify T = 8) when
        its context window is on the device (see `_df_covered`), everything else by the MTP path.  dflash2 needs the engine's 9-slot ring (tmax = 8, as the prompt-lookup tail), so the tail is free there.
        max_num_seqs > 1: that many sequence slots on one set of weights (engine/batch.py: one DeltaNet state per slot, tmax 8); server/smk_batch.py decodes them together (needs drafter mtp)."""
        assert 0 <= k <= 3 and group >= 1
        assert not pld or k == 3, "the prompt-lookup tail extends the full k = 3 MTP chain"
        assert drafter in ("mtp", "dflash2"), drafter
        assert max_num_seqs == 1 or drafter == "mtp", "--drafter dflash2 serves one sequence at a time (--max-num-seqs 1)"
        self.max_num_seqs = max_num_seqs
        self.bd = None  # engine/batch.py BatchDecoder (max_num_seqs > 1)
        self.k, self.group, self.max_snaps, self.log = k, group, max_snaps, log
        self.pld = (make_pld() if pld is True else pld) or None
        self.drafter = drafter
        self.df = drafter == "dflash2"
        self.df_lo = self.df_hi = 0  # drafter K/V rows [df_lo, df_hi) of the device lineage's prompt are valid (see _df_note); empty when equal
        self.tap = None
        self.df_lookup = bool(dflash_lookup)  # the drafter's device n-gram lookup over the token history (SMK_DF_LOOKUP): the prompt is uploaded to its history at every dflash2 request
        self.df_mib = 0.0  # device memory of the drafter (weights, caches, ring, graphs of the first request not included), measured at construction
        t0 = time.time()
        try:
            if self.df:
                from dflash import DF_PATH, DflashEngine, TapCapture

                if self.df_lookup:
                    os.environ["SMK_DF_LOOKUP"] = "1"  # read once by smk_dflash_create
                e = DflashEngine(dflash_path=dflash_path or DF_PATH, codebooks_host=dflash_codebooks_host, ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=False, host_tables=host_tables, lm_head_host=True)
                self.tap = TapCapture(e, self.DF_WINDOW)
                self.df_mib = e.df_mib + self.tap.nbytes / 2**20  # drafter weights + K/V cache + scratch + the tap staging buffers
                engines = [e]
            elif max_num_seqs > 1:
                from batch import BatchDecoder

                self.bd = BatchDecoder(slots=max_num_seqs, ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=False, host_tables=host_tables)
                engines = self.bd.slots
            else:
                engines = [SpecEngine(ctx_max=ctx_max, kv_fp8=kv_fp8, verbose=False, tmax=8 if self.pld else 4, host_tables=host_tables)]
        except AttributeError as ex:  # ctypes: symbol missing in an old libswiftmk.so
            raise RuntimeError(f"libswiftmk.so is older than the sources ({ex}): run `make -C engine` (or point SMK_BUILD at a current build dir)") from ex
        self.seqs = [Seq(i, x, FastPrefill(x, seg=seg, **({} if min_fast is None else {"min_fast": min_fast}))) for i, x in enumerate(engines)]
        self.seq = self.seqs[0]
        self.ctx_max = ctx_max
        self.load_s = time.time() - t0
        self.pool: list[Snap] = []
        self._made = 0  # snapshot buffers allocated so far (device snapshots + pool + stored entries'), at most max_snaps
        for _ in range(min(max_snaps, 4)):  # the first ones up front (cudaHostAlloc of 151 MB costs ~20 ms); the rest are made on demand
            self.pool.append(self._new_snap())
        self.store = PrefixStore(int(prefix_cache_gb * (1 << 30))) if prefix_cache_gb > 0 and max_snaps > 0 else None  # host tier under the device lineage
        self.system_checkpoint = system_checkpoint and max_snaps >= 3  # one more snapshot per long prompt: the first chat-turn boundary after SYSTEM_MIN tokens (see _system_end)
        self.group_prefill = group_prefill  # run consecutive prefill segments layer by layer (one weight dequant per layer for the group); False = one segment at a time
        self.min_spill = min_spill  # lineages whose last snapshot is below this are not worth a spill (a cold prefill of them is shorter than a restore)
        self.last = Stats()
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
        torch.cuda.current_stream().synchronize()
        smk.check(self.e.lib.smk_spec_read(self.e.sp, 0, buf, 160), "spec_read")
        return list(buf)

    def _free_snaps(self, keep):
        """Keep only the snapshots in `keep`, the rest go back to the pool."""
        self.pool += [s for s in self.snaps if s not in keep]
        self.snaps = list(keep)

    def _snapshot(self, i, sticky=False):
        """DeltaNet + conv state of the engine's current ring slot -> a host snapshot buffer for position i."""
        self._free_snaps([x for x in self.snaps if x.pos != i])  # same position: reuse the buffer
        if not self.pool:
            self._grow_pool()
        s = self.pool.pop()
        cur = self._cur()
        s.pos = i
        s.sticky = sticky
        s.dn.copy_(self.fp.dn_state[:, cur % self.fp.nss])
        s.conv.copy_(self.fp.dn_conv[:, cur])
        if s.aux is not None:
            self.e.kv_aux_save(s.aux)
        self.snaps.append(s)

    def _restore(self, s):
        fp, e = self.fp, self.e
        cur = self._cur()
        fp.dn_state[:, cur % fp.nss].copy_(s.dn)
        fp.dn_conv[:, cur].copy_(s.conv)
        if s.aux is not None:
            e.kv_aux_restore(s.aux)
        fp.ds[19] = 0  # DS_PEND (engine_impl.h): the restored state replaces what the lazy DeltaNet ring still held pending from the previous request (the prefill's materialize would replay it on top)
        fp.ds[0] = s.pos
        e.pos = s.pos
        torch.cuda.synchronize()

    def drop_cache(self):
        cur = self.seq
        for q in self.seqs:
            self.seq = q
            self.lineage = None
            self._free_snaps([])
        self.seq = cur
        self.df_lo = self.df_hi = 0
        if self.store is not None:
            self.pool += self.store.clear()

    # ------------------------------------------------------------------------------------------------ DFlash2 drafter context
    # The drafter attends the last 2047 committed rows of its own K/V cache (positions [n - 2047, n) at the first cycle of a prompt of n tokens); row p is a function of the target's residual taps at
    # row p alone (engine/dflash.py), so the rows of the window are rebuilt from the taps of the fast prefill (TapCapture) instead of re-running the target.  The Runner tracks which rows of the
    # device lineage's prompt are valid: [df_lo, df_hi).  Every prefill of a dflash2 Runner captures its window whatever route decodes the request, so the lineage stays resumable by a
    # later greedy turn.  A missing row never makes the output wrong (greedy accept = argmax equality of the T = 8 verify), it only lowers the tokens per cycle: the router sends a request to
    # the MTP path when its window is not on the device.
    DF_WINDOW = 2047

    def _df_window(self, n_prompt, q):
        """Prompt rows [c0, c1) the prefill of a prompt of n_prompt tokens resumed at q must capture: the window [n_prompt - W, n_prompt - 1) minus what a resume skips; row n_prompt - 1 comes from the head step."""
        return max(q, n_prompt - self.DF_WINDOW), n_prompt - 1

    def _df_note(self, a, b):
        """Rows [a, b) were just (re)written: extend the valid interval when they continue it, else they start a new one."""
        if self.df_hi > self.df_lo and a == self.df_hi:
            self.df_hi = b
        else:
            self.df_lo, self.df_hi = a, b

    def _df_cut(self, q):
        """A resume at q keeps the drafter rows below q (same tokens) and drops the rest."""
        self.df_hi = min(self.df_hi, q)
        if self.df_hi <= self.df_lo:
            self.df_lo = self.df_hi = 0

    def _df_covered(self, n_prompt, q):
        """Is every row of the window of a prompt of n_prompt tokens resumed at q on the device after the prefill (captured by it, or valid from the lineage below q)?"""
        lo_need = max(0, n_prompt - self.DF_WINDOW)
        return lo_need >= q or (self.df_hi > self.df_lo and self.df_lo <= lo_need and self.df_hi >= q)

    def _new_snap(self):
        self._made += 1
        return Snap(0, torch.empty(48, 48, 128, 128, dtype=torch.float32).pin_memory(), torch.empty(48, 10240 * 3, dtype=torch.bfloat16).pin_memory(), self.e.kv_aux_new() if hasattr(self.e, "kv_aux_new") else None)

    def _grow_pool(self):
        """Get a snapshot buffer into the pool: allocate one while under max_snaps, else take the buffers of the least recently used stored entry, else evict the oldest device snapshot
        (conversation prompts grow, so the lowest boundary is the least useful)."""
        if self._made < self.max_snaps:
            self.pool.append(self._new_snap())
            return
        while self.store is not None and not self.pool:
            rel = self.store.evict_lru()
            if rel is None:
                break
            self.pool += rel
        if not self.pool:  # the current slot's lowest snapshot, else (several sequence slots) the lowest one of the least recently used other slot that has any
            owner = self.seq if self.snaps else min((q for q in self.seqs if q.snaps), key=lambda q: q.used)
            old = min(owner.snaps, key=lambda x: (x.sticky, x.pos))
            self.pool.append(old)
            owner.snaps = [x for x in owner.snaps if x is not old]

    def _spill_device(self, keep=None):
        """Move the device lineage (KV rows up to its last snapshot + its snapshots) into the host store before the device caches are overwritten.  Without a store, or for a lineage
        shorter than min_spill, the snapshots are just freed.  Returns the seconds spent."""
        t0 = time.time()
        snaps, lin, store = self.snaps, self.lineage, self.store
        s_max = max((x.pos for x in snaps), default=0)
        if store is None or lin is None or not snaps or s_max < self.min_spill or len(lin) <= s_max:
            self._free_snaps([])
            return 0.0
        ids = lin[: s_max + 1]
        ent = store.extendable(ids, s_max)
        start = ent.n_kv if ent is not None else 0
        m = s_max - start
        e = self.e
        chunk = dfw = None
        if m > 0:
            main, mtp = e.kv_chunk_new(m), e.mtp_chunk_new(m)
            nb = main.numel() + mtp.numel()
            dnb = 0
            if self.df and self.df_hi > self.df_lo and min(self.df_hi, s_max) > self.df_lo:  # the drafter rows of this lineage (a resume at <= s_max needs the rows below it)
                dfw = DfWindow(self.df_lo, min(self.df_hi, s_max), None)
                dnb = 5 * 8 * (dfw.hi - dfw.lo) * 128 * 2 * 2
            if nb + dnb > store.budget:  # can never fit: do not keep a lineage the budget cannot hold
                self._free_snaps([])
                return time.time() - t0
            self.pool += store.fit(nb + dnb, keep=(keep, ent))
            e.kv_chunk_copy(main, m, start, 0, m, True)
            e.mtp_chunk_copy(mtp, m, start, 0, m, True)
            if dfw is not None:
                dfw.kv = e.df_window_save(dfw.lo, dfw.hi)
            torch.cuda.synchronize()
            chunk = Chunk(start, m, main, mtp, nb)
        moved, self.snaps = self.snaps, []  # the snapshot buffers now belong to the entry
        if ent is None:
            _, rel = store.add(ids, s_max, chunk, moved, dfw)
        else:
            rel = store.extend(ent, ids, s_max, chunk, moved, dfw)
        self.pool += rel
        return time.time() - t0

    def _load_entry(self, ent, snap, ids):
        """Restore the stored lineage `ent` up to snap.pos into the device caches: the KV rows of the main cache and of the MTP layer, then the snapshot (DeltaNet + conv + int4 aux)."""
        e, s = self.e, snap.pos
        for ch in ent.chunks:
            if ch.a >= s:
                break
            r1 = min(ch.m, s - ch.a)
            e.kv_chunk_copy(ch.main, ch.m, ch.a, 0, r1, False)
            e.mtp_chunk_copy(ch.mtp, ch.m, ch.a, 0, r1, False)
        torch.cuda.synchronize()
        self.store.touch(ent)
        self.lineage = np.asarray(ids[:s], dtype=np.int32)  # the device KV now holds rows [0, s) of THIS prompt
        e.reset_counters()
        self._restore(snap)
        self.pool += self.store.fit(0, keep=ent)  # the spill before the restore had to keep `ent` alive: trim back to the budget now (LRU, so the older conversation goes)

    def _pick(self, ids, ends):
        """Where to resume `ids`: ("device", snap), ("host", entry, snap) or None (cold).  The largest resume position wins, ties go to the device (no copy)."""
        dev = self._choose(ids, ends)
        host = self.store.best(ids, ends) if self.store is not None else None
        if host is not None and (dev is None or host[1].pos > dev.pos):
            return ("host", host[0], host[1])
        return ("device", dev) if dev is not None else None

    def _choose(self, ids, ends):
        """Snapshot to resume from for prompt `ids` whose plan has the segment ends `ends` (None = cold start)."""
        if self.lineage is None or not self.snaps:
            return None
        n = min(len(ids), len(self.lineage))
        a, b = np.asarray(ids[:n], dtype=np.int32), self.lineage[:n]
        ne = np.flatnonzero(a != b)
        L = int(ne[0]) if len(ne) else n
        ok = [s for s in self.snaps if s.pos <= min(L - 1, len(ids) - 1) and s.pos in ends]
        return max(ok, key=lambda s: s.pos) if ok else None

    # ------------------------------------------------------------------------------------------------ prefill
    def _plan(self, ids):
        """The segmentation of the prefill of `ids` (every request, cold or resumed, follows it): (ends, folded).  ends = the end position of each segment run before the last
        prompt token; folded: the last row rides in a final batch of <= min(seg, w4_max) rows (K10, no 20 ms T=1 head step) instead of a head step after the last segment.
        A segment ends after seg rows or at a chat turn boundary p+2 (p = a <|im_start|>) more than w4_max rows after the previous end (a cut that does not save a re-prefill of
        more than a weight pass would cost ~+55 ms); the decision at a position only looks at the tokens before it, so a prompt extending another one keeps its cuts."""
        fp = self.fp
        n = len(ids) - 1
        marks = np.flatnonzero(np.asarray(ids[: max(n - 1, 0)], dtype=np.int32) == IM_START) + 2  # turn boundaries <= n
        ends, i = [], 0
        while i < n:
            if fp.fold and 2 <= n + 1 - i <= min(fp.seg, fp.w4_max):
                return ends, True
            j = min(n, i + fp.seg)
            k = int(np.searchsorted(marks, i + fp.w4_max, side="right"))
            if k < len(marks) and marks[k] < j:
                j = int(marks[k])
            ends.append(j)
            i = j
        return ends, False

    SYSTEM_MIN = 1024

    def _system_end(self, ids, ends):
        """The system-prompt checkpoint: the first segment end that is a chat-turn boundary (token ends[j] - 2 is <|im_start|>) at >= SYSTEM_MIN tokens, else None.  Parallel agents of one type
        share a long system prompt and differ after it, so the state exactly at the end of that prompt is the resume point they have in common (SGLang's branching checkpoint)."""
        if not self.system_checkpoint:
            return None
        for j in ends:
            if j >= self.SYSTEM_MIN and ids[j - 2] == IM_START:
                return j
        return None

    @staticmethod
    def plan_groups(ends, q, seg, min_fast, snap_at, enabled=True):
        """Group the plan segments [prev end, end) after position q into runs executed layer by layer (FastPrefill._group): consecutive segments while their total rows stay <= seg (the
        scratch of one segment), a group ends at every snapshot position (the DeltaNet state must be exact there) and a segment of fewer than min_fast rows (the decode-step path) is
        alone.  enabled=False: one segment per group (the previous behaviour).  Returns a list of lists of (start, end)."""
        groups, cur, rows, i = [], [], 0, q
        for j in (x for x in ends if x > q):
            n = j - i
            fast = n >= min_fast
            if cur and (not enabled or not fast or rows + n > seg):
                groups.append(cur)
                cur, rows = [], 0
            cur.append((i, j))
            rows += n
            i = j
            if not enabled or not fast or j in snap_at:
                groups.append(cur)
                cur, rows = [], 0
        if cur:
            groups.append(cur)
        return groups

    def _prefill(self, ids, q, ends, folded, cancelled, use_df=False):
        """Process ids[q:] (state valid for [0, q), q = 0 or a segment end of the plan) following the plan; the last token as the head step.  Snapshots after the last (up to)
        two segment ends.  Returns False when cancelled (self.lineage then holds the valid prefix).  dflash2 Runner: also builds the drafter's context window (TapCapture); use_df: the
        request is decoded by the block drafter, so the prefill ends with smk_dflash_start instead of the MTP draft (smk_spec_start)."""
        tap = self.tap
        if tap is not None:
            tap.begin(*self._df_window(len(ids), q))
            self.fp.tap = tap
        try:
            return self._prefill_rows(ids, q, ends, folded, cancelled, use_df, tap)
        finally:
            self.fp.tap = None

    def _df_flush(self, tap, a, b):
        """Append the staged rows of [a, b) that are inside the capture window to the drafter's context."""
        a, b = max(a, tap.lo), min(b, tap.hi)
        if a < b:
            tap.flush(a, b)
            self._df_note(a, b)

    def _prefill_rows(self, ids, q, ends, folded, cancelled, use_df, tap):
        e, fp = self.e, self.fp
        snap_at = set(ends[len(ends) - min(2, self.max_snaps) :]) if self.max_snaps else set()
        sys_at = self._system_end(ids, ends)
        if sys_at is not None and sys_at > q:
            snap_at.add(sys_at)
        i = q
        for g in self.plan_groups(ends, q, fp.seg, fp.min_fast, snap_at, self.group_prefill):
            if cancelled():
                self.lineage = np.asarray(ids[:i], dtype=np.int32)
                return False
            j = g[-1][1]
            if g[0][1] - g[0][0] >= fp.min_fast:
                cur = self._cur()
                fp._ev = []
                if len(g) == 1:
                    fp._segment(ids[i:j], i, cur, ids[i + 1 : j + 1])
                else:  # several segments layer by layer: one dequant of each layer's weights for all of them, bit-identical to running them one after the other
                    fp._group([(ids[x:y], x, ids[x + 1 : y + 1]) for x, y in g], cur)
                fp.ds[0] = j
                e.pos = j
                torch.cuda.synchronize()
                if tap is not None:
                    self._df_flush(tap, i, j)
            else:
                for a in range(i, j, 4):
                    T = min(4, j - a)
                    e._step(ids[a : a + T], 2)
                    nt = (C.c_int * T)(*ids[a + 1 : a + 1 + T])
                    smk.check(e.lib.smk_spec_prefill_chunk(e.sp, T, nt, a, e.st), "spec_prefill_chunk")
                    if tap is not None and a + T > tap.lo:  # the step left the taps of these rows in dbg rows 0..T-1
                        e.df_append(T, a)
                        self._df_note(a, a + T)
            i = j
            if i in snap_at:
                self._snapshot(i, sticky=(i == sys_at))
            self.lineage = np.asarray(ids[:i], dtype=np.int32)
        if folded:
            if cancelled():
                self.lineage = np.asarray(ids[:i], dtype=np.int32)
                return False
            cur = self._cur()
            fp._ev = []
            fp._segment(ids[i:], i, cur, ids[i + 1 :], head=True)
            if tap is not None:
                torch.cuda.synchronize()
                self._df_flush(tap, i, len(ids) - 1)
                tap.put_last()  # the folded head has no step that fills dbg row 0 with the last row's taps
        else:
            e._step(ids[-1:], 1)
        n = len(ids) - 1  # position of the last prompt row (its taps are in dbg row 0 now)
        if tap is not None and not use_df:  # the lineage stays resumable by a later greedy turn: its last row joins the window (smk_dflash_start does it for the dflash2 route)
            e.df_append(1, n)
            self._df_note(n, n + 1)
        if e.sampling:
            smk.check(e.lib.smk_spec_sample_first(e.sp, e.st), "spec_sample_first")
        if use_df:
            e.df_start()
            self._df_note(n, n + 1)
        elif self.k:
            smk.check(e.lib.smk_spec_start(e.sp, self.k, e.st), "spec_start")
        self.lineage = np.asarray(ids, dtype=np.int32)
        return True

    # ------------------------------------------------------------------------------------------------ generation
    def _begin(self, ids, p: GenParams, route=None):
        """The part of a request before its prefill, on the current sequence slot: the length budget, the prefill plan, the sampling configuration and processors of the slot's
        engine, and the resume point (device snapshot, host prefix cache entry, or a cold start after spilling the slot's lineage).  Returns a namespace for _prefill / the decode loop."""
        e, k = self.e, self.k
        ids = [int(t) for t in ids]
        st = self.last = Stats(prompt=len(ids))
        n_new = min(p.max_tokens, self.ctx_max - self.slack(self.group, k, bool(self.pld) or self.df) - len(ids))
        if n_new < 1 or len(ids) < 1:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the engine context {self.ctx_max}")
        t0 = time.time()
        ends, folded = self._plan(ids)
        procs = p.procs(ids)
        if p.temperature > 0:
            seed = p.seed if p.seed is not None else random.getrandbits(62)
            e.set_sampling(temperature=p.temperature, top_k=p.top_k if p.top_k > 0 else MAX_TOP_K, top_p=p.top_p, seed=seed, min_p=p.min_p)
        elif procs:  # greedy over the PROCESSED logits: the sampling graphs with top_k = 1 (the greedy graphs have no processors)
            e.set_sampling(temperature=1.0, top_k=1, top_p=1.0, seed=0)
        else:
            e.set_sampling(0.0)
        if procs or self._procs_on:  # requests without processors touch nothing (a previous request's processors are switched off once)
            e.set_procs(**(procs or {}))
            self._procs_on = bool(procs)
        pick = self._pick(ids, set(ends))
        if pick is None:
            st.spill_s = self._spill_device()
            e.reset()  # (a dflash2 engine also zeroes the drafter's K/V cache)
            self.df_lo = self.df_hi = 0
            q = 0
        elif pick[0] == "device":
            snap = pick[1]
            self._free_snaps([s for s in self.snaps if s.pos <= snap.pos])
            e.reset_counters()
            self._restore(snap)
            self._df_cut(snap.pos)  # the drafter rows below the resume position are the same tokens' rows
            q = snap.pos
            st.source = "device"
        else:
            _, ent, snap = pick
            tr = time.time()
            st.spill_s = self._spill_device(keep=ent)
            self._load_entry(ent, snap, ids)
            st.restore_s = time.time() - tr - st.spill_s
            self.df_lo = self.df_hi = 0  # the drafter cache holds another lineage's rows ...
            if self.df and ent.dfw is not None:  # ... unless the entry carries the window of the conversation it holds
                e.df_window_load(ent.dfw.lo, ent.dfw.kv)
                self.df_lo, self.df_hi = ent.dfw.lo, ent.dfw.hi
                self._df_cut(snap.pos)
            q = snap.pos
            st.source = "host"
        st.cached = q
        use_df = self.df and route != "mtp" and p.temperature <= 0 and not procs and self._df_covered(len(ids), q)
        st.route = "dflash2" if use_df else "mtp"
        if use_df and self.df_lookup:
            e.df_set_hist(ids)
        return SimpleNamespace(ids=ids, st=st, n_new=n_new, t0=t0, ends=ends, folded=folded, q=q, use_df=use_df)

    def run(self, ids, p: GenParams, cancelled=lambda: False, route=None):
        """Generator of (new_tokens, finish_reason | None, stop_reason); tokens are cut exactly at the stop / length.  Nothing is yielded after a cancel.
        route: None = the router's choice; "mtp" forces the MTP path (a dflash2 Runner otherwise decodes greedy requests without logit processors with the block drafter when its window is covered)."""
        b = self._begin(ids, p, route)
        e, k, ids, st, n_new, t0, use_df = self.e, self.k, b.ids, b.st, b.n_new, b.t0, b.use_df
        if not self._prefill(ids, b.q, b.ends, b.folded, cancelled, use_df):
            return
        if use_df:
            ss0 = e.df_state()  # the counters run on across requests that resume a cached lineage: this request's share is the difference
            e._demitted = ss0[12]
        else:
            e._emitted = self._ss()[12]
        first = e.read_out_tok()
        use_pld = bool(self.pld) and k == 3 and not use_df
        if use_pld:
            e.enable_pld(self.pld)
            e.pld_begin(ids + [first])
        st.prefill_s = time.time() - t0
        st.snaps = [s.pos for s in self.snaps]
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
                return
            yield out, None, None
            if cancelled():
                st.gen_s = time.time() - t1
                return
            if use_df:
                new, ss = e.df_cycles(self.group)
                st.cycles = ss[13] - ss0[13]
                st.hist = [a - b for a, b in zip(ss[16:24], ss0[16:24])]
            elif use_pld:
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
        routes = (None, "mtp") if self.df else (None,)  # a dflash2 Runner warms both draft paths: the graphs of the MTP route (T = 4..8 cycles) are captured now, not under memory pressure at the first sampled request
        for n, route in [(40, r) for r in routes] + [(min(self.fp.seg + 64, self.ctx_max - 64), None), (130, None)]:
            ids = rng.integers(1000, 100000, size=n).tolist()
            for _ in self.run(ids, GenParams(max_tokens=2 * (self.group * (8 if self.pld or self.df else self.k + 1)) + 2, temperature=0.0), route=route):
                pass
        self.drop_cache()
        torch.cuda.empty_cache()
