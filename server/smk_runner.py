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

    def _snapshot(self, i):
        """DeltaNet + conv state of the engine's current ring slot -> a host snapshot buffer for position i."""
        self._free_snaps([x for x in self.snaps if x.pos != i])  # same position: reuse the buffer
        if not self.pool:  # evict the oldest (lowest position: conversation prompts grow, early boundaries are the least useful)
            old = min(self.snaps, key=lambda x: x.pos)
            self._free_snaps([x for x in self.snaps if x is not old])
        s = self.pool.pop()
        cur = self._cur()
        s.pos = i
        s.dn.copy_(self.fp.dn_state[:, cur])
        s.conv.copy_(self.fp.dn_conv[:, cur])
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
        self.lineage = None
        self._free_snaps([])

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

    def _prefill(self, ids, q, ends, folded, cancelled):
        """Process ids[q:] (state valid for [0, q), q = 0 or a segment end of the plan) following the plan; the last token as the head step.  Snapshots after the last (up to)
        two segment ends.  Returns False when cancelled (self.lineage then holds the valid prefix)."""
        e, fp = self.e, self.fp
        snap_at = set(ends[len(ends) - min(2, self.max_snaps) :]) if self.max_snaps else set()
        i = q
        for j in (x for x in ends if x > q):
            if cancelled():
                self.lineage = np.asarray(ids[:i], dtype=np.int32)
                return False
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
            if i in snap_at:
                self._snapshot(i)
            self.lineage = np.asarray(ids[:i], dtype=np.int32)
        if folded:
            if cancelled():
                self.lineage = np.asarray(ids[:i], dtype=np.int32)
                return False
            cur = self._cur()
            fp._ev = []
            fp._segment(ids[i:], i, cur, ids[i + 1 :], head=True)
        else:
            e._step(ids[-1:], 1)
        if e.sampling:
            smk.check(e.lib.smk_spec_sample_first(e.sp, e.st), "spec_sample_first")
        if self.k:
            smk.check(e.lib.smk_spec_start(e.sp, self.k, e.st), "spec_start")
        self.lineage = np.asarray(ids, dtype=np.int32)
        return True

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
        ends, folded = self._plan(ids)
        if p.temperature > 0:
            seed = p.seed if p.seed is not None else random.getrandbits(62)
            e.set_sampling(temperature=p.temperature, top_k=p.top_k if p.top_k > 0 else MAX_TOP_K, top_p=p.top_p, seed=seed)
        else:
            e.set_sampling(0.0)
        snap = self._choose(ids, set(ends))
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
        if not self._prefill(ids, q, ends, folded, cancelled):
            return
        e._emitted = self._ss()[12]
        first = e.read_out_tok()
        use_pld = bool(self.pld) and k == 3
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
