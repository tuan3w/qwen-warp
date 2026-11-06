"""Continuous-batching scheduler (server/smk_batch.py BatchRunner) against a deterministic fake engine: random arrivals, multi-turn prompts (prefix-cache hits and misses, host-store
spills / restores), lengths, max_tokens, stop tokens, aborts and slot reuse; no GPU.

Tier: 0

The fake engine's next token is a function of the hash of everything its sequence state has consumed (token = F(H(prompt + generated))), its "DeltaNet state" is that hash (copied
by the real snapshot / restore code), its KV and MTP-KV caches hold the token ids (spilled to / restored from the host store by the real code).  So a wrong resume point, a wrong
KV or snapshot restore, a token delivered to the wrong request or a decode call on the wrong slot changes the stream.  Asserted for every request: the delivered tokens are exactly
the expected stream cut at its own stop / max_tokens (a prefix of it when aborted, and nothing after the abort), each request is admitted in arrival order and finishes, every
slot is released, the decode calls respect the engine's limits (rows <= 16, verified drafts <= drafted) and the decode cap (at most DECODE_CAP sequences per group outside exact mode,
no running request skipped by two groups in a row), the running requests decode while a long prompt prefills (stall bound), the prompt-lookup proposer gets the true context, and the
snapshot buffers and host-store bytes add up (no leak).
  python tests/test_batch_sched.py [--seeds N]
"""
import argparse
import ctypes as C
import random
import sys
import types
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "engine"))
# the fake needs no GPU kernels: stub the prefill's GPU libraries (imported at module level) so the import is fast
for name in ("fla", "fla.ops", "fla.ops.gated_delta_rule", "vllm", "vllm.vllm_flash_attn", "vllm.vllm_flash_attn.flash_attn_interface"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["fla.ops.gated_delta_rule"].chunk_gated_delta_rule = None
sys.modules["vllm.vllm_flash_attn.flash_attn_interface"].flash_attn_varlen_func = None

import torch  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None  # the runner synchronises the (absent) GPU around host copies

import smk_batch as SB  # noqa: E402
import smk_runner as R  # noqa: E402
from prefix_store import PrefixStore  # noqa: E402

EOS, STOP_ID, VOC0, NVOC = 1007, 1013, 1000, 256  # small fake vocabulary: stops happen
MOD = (1 << 61) - 1


def H(h, t):
    return (h * 1000003 + t + 1) % MOD


def Hs(ids, h=0):
    for t in ids:
        h = H(h, t)
    return h


def F(h):
    return VOC0 + (h * 2654435761 >> 7) % NVOC


class FakeLib:
    def smk_spec_prefill_chunk(self, *a):
        return 0

    def smk_spec_start(self, *a):
        return 0

    def smk_spec_sample_first(self, *a):
        return 0


class FakeFP:
    def __init__(self, e):
        self.e = e
        self.seg, self.w4_max, self.fold, self.min_fast, self.nss = 64, 16, True, 2, 1
        self.ds = torch.zeros(21, dtype=torch.int32)  # engine_impl.h DS_N (DS_RDELTA = 20)
        self.mm = None
        self.dn_state = torch.zeros(48, 1, 2, dtype=torch.int64)  # [0, 0] = (hash, tokens) of the sequence state
        self.dn_conv = torch.zeros(48, 9, 2, dtype=torch.int64)
        self.layer_hook = None
        self.tap = None

    def _rows(self, ids_seg, pos0):
        e = self.e
        assert int(self.dn_state[0, 0, 1]) == pos0 == e.pos, f"slot {e.idx}: segment at {pos0}, state at {int(self.dn_state[0, 0, 1])}, engine at {e.pos}"
        e.check_prefix(pos0)
        e.advance(ids_seg)

    def _layers(self, rows):
        for _ in range(64):
            if self.layer_hook is not None:
                self.e.world.layer(rows)
                self.layer_hook(rows)

    def _segment(self, ids_seg, pos0, cur, nxt=None, head=False):
        self._layers(len(ids_seg))
        self._rows(ids_seg, pos0)
        if head:  # the folded last row: first generated token, device scalars of a head step
            self.e.head()
            self.ds[1] = (int(self.ds[1]) + 1) % 9

    def _group(self, segs, cur):
        self._layers(sum(len(s[0]) for s in segs))
        for ids_seg, pos0, _ in segs:
            self._rows(ids_seg, pos0)


class FakeEngine:
    """One sequence slot.  KV / MTP-KV hold token ids, the DeltaNet state is (hash, length) in fp.dn_state."""

    def __init__(self, idx, world, ctx_max):
        self.idx, self.world, self.ctx_max = idx, world, ctx_max
        self.lib, self.sp, self.st, self.h = FakeLib(), idx, None, idx
        self.fp = FakeFP(self)
        self.kv = [-1] * ctx_max
        self.mtp = [-1] * ctx_max
        self.sampling = self.pld = False
        self.drafted = 0
        self._emitted = 0
        self.anchor = None
        self.pld_ctx = None
        self.prompt = None

    pos = property(lambda s: int(s.fp.ds[0]), lambda s, v: s.fp.ds.__setitem__(0, v))

    def check_prefix(self, n):
        assert self.kv[:n] == self.prompt[:n], f"slot {self.idx}: KV rows below {n} are not the prompt's"
        assert self.mtp[:n] == self.prompt[:n], f"slot {self.idx}: MTP KV rows below {n} are not the prompt's"
        assert int(self.fp.dn_state[0, 0, 0]) == Hs(self.prompt[:n]), f"slot {self.idx}: DeltaNet state at {n} is not the prompt's"

    def advance(self, toks):
        h, n = int(self.fp.dn_state[0, 0, 0]), int(self.fp.dn_state[0, 0, 1])
        for t in toks:
            self.kv[n] = self.mtp[n] = t
            h, n = H(h, t), n + 1
        self.fp.dn_state[0, 0, 0], self.fp.dn_state[0, 0, 1] = h, n
        self.pos = n

    def head(self):
        self.anchor = F(int(self.fp.dn_state[0, 0, 0]))

    # ---- Engine / SpecEngine API used by the Runner
    def set_sampling(self, temperature=0.0, **kw):
        self.sampling = temperature > 0

    def set_procs(self, **kw):
        pass

    def reset(self):
        self.kv = [-1] * self.ctx_max
        self.mtp = [-1] * self.ctx_max
        self.fp.dn_state.zero_()
        self.fp.ds.zero_()

    def reset_counters(self):
        pass

    def _step(self, toks, head):
        self.check_prefix(self.pos)
        self.advance(list(toks))
        if head == 1:
            self.head()

    def read_out_tok(self):
        return self.anchor

    def kv_aux_new(self):
        return None

    def kv_chunk_new(self, m):
        return torch.full((m,), -1, dtype=torch.int64)

    mtp_chunk_new = kv_chunk_new

    def _chunk(self, buf, t, m, dev_row0, r0, r1, to_host):
        assert 0 <= r0 <= r1 <= m == t.numel() and dev_row0 + r1 <= self.ctx_max
        if to_host:
            t[r0:r1] = torch.tensor(buf[dev_row0 + r0 : dev_row0 + r1], dtype=torch.int64)
        else:
            buf[dev_row0 + r0 : dev_row0 + r1] = t[r0:r1].tolist()

    def kv_chunk_copy(self, t, m, dev_row0, r0, r1, to_host):
        self._chunk(self.kv, t, m, dev_row0, r0, r1, to_host)

    def mtp_chunk_copy(self, t, m, dev_row0, r0, r1, to_host):
        self._chunk(self.mtp, t, m, dev_row0, r0, r1, to_host)

    def enable_pld(self, opts=None):
        self.pld = True

    # ---- decode
    def emit(self, a):
        """a accepted tokens: commit the anchor, the next token becomes the output and the new anchor (KV = prompt + generated[:-1], as on the engine)."""
        out = []
        for _ in range(a):
            self.advance([self.anchor])
            self.anchor = F(int(self.fp.dn_state[0, 0, 0]))
            out.append(self.anchor)
        return out

    def cycle_tokens(self, kv, n):
        return sum((self.emit(self.world.rng.randint(1, kv + 1)) for _ in range(n)), [])

    def cycles(self, k, n):
        assert self.drafted >= k, f"slot {self.idx}: native k={k} cycle after only {self.drafted} drafts"
        self.world.decoded(self, n)
        return self.cycle_tokens(k, n), None

    def pld_begin(self, ctx):
        assert ctx == self.kv[: self.pos] + [self.anchor], f"slot {self.idx}: prompt-lookup context is not the sequence"
        self.pld_ctx = True

    def cycles_pld(self, n):
        assert self.pld and self.pld_ctx and self.drafted >= 3
        self.world.decoded(self, n)
        return self.cycle_tokens(3 + self.world.rng.randint(0, 4), n), None


class FakeBD:
    def __init__(self, world):
        self.world = world

    def cycles(self, seqs, kv0, kd, n):
        B = len(seqs)
        assert len(set(id(s) for s in seqs)) == B and 1 <= B <= 8
        assert B <= SB.DECODE_CAP or self.world.runner.exact_k is not None, f"{B} sequences decode in one group (cap {SB.DECODE_CAP})"
        assert B * (kv0 + 1) <= SB.ROWS and B * (kd + 1) <= SB.ROWS and 0 <= kv0 <= 3 and 0 <= kd <= 3
        for s in seqs:
            assert kv0 <= s.drafted, f"slot {s.idx}: batched cycle verifies {kv0} drafts, it drafted {s.drafted}"
            s.pld_ctx = None  # the proposer no longer follows the sequence
        self.world.decoded(None, n)
        news = []
        for s in seqs:
            toks = s.cycle_tokens(kv0, 1) + (s.cycle_tokens(kd, n - 1) if n > 1 else [])
            s.drafted = kd
            news.append(toks)
        return news, [[0] * 64 for _ in seqs]


class World:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.rows_since_decode = 0
        self.max_stall = 0
        self.decoding = 0

    def layer(self, rows):
        if not self.runner.tasks:  # nobody is waiting for a decode group
            self.rows_since_decode = 0
            return
        self.rows_since_decode += rows
        self.max_stall = max(self.max_stall, self.rows_since_decode)

    def decoded(self, e, n):
        self.rows_since_decode = 0


class FakeRunner(SB.BatchRunner):
    def __init__(self, S, world, budget, max_snaps, ctx_max=4096, exact_k=None):
        self.tasks, self.prefilling, self._rows, self._tick, self.batched_cycles, self.single_cycles = [], None, 0, 0, 0, 0
        self.k, self.group, self.max_snaps, self.log = 3, 2, max_snaps, print
        self.pld, self.drafter, self.df, self.df_lo, self.df_hi, self.tap, self.df_lookup, self.df_mib = True, "mtp", False, 0, 0, None, False, 0.0
        self.max_num_seqs, self.bd, self.exact_k, self.vit = S, FakeBD(world), exact_k, None
        self.seqs = [R.Seq(i, e, e.fp) for i, e in enumerate(FakeEngine(i, world, ctx_max) for i in range(S))]
        self.seq = self.seqs[0]
        self.ctx_max, self.load_s, self.pool, self._made = ctx_max, 0.0, [], 0
        self.store = PrefixStore(budget) if budget else None
        self.system_checkpoint, self.group_prefill, self.min_spill, self.last = True, True, 32, R.Stats()
        self.SYSTEM_MIN = 96

    def _new_snap(self):
        self._made += 1
        return R.Snap(0, torch.zeros(48, 2, dtype=torch.int64), torch.zeros(48, 2, dtype=torch.int64), None)

    def _cur(self):
        return int(self.fp.ds[1])

    def _ss(self):
        return [0] * 40

    def _begin(self, ids, p, route=None, images=None):
        self.e.prompt = [int(t) for t in ids]
        return super()._begin(ids, p, route, images)


def expected(prompt, p, n_new):
    h, out = Hs(prompt), []
    while True:
        y = F(h)
        out.append(y)
        if y == p.eos or y in p.stop_ids or len(out) >= n_new:
            return out
        h = H(h, y)


def run(seed, S, n_req, budget, max_snaps, verbose=False, exact_k=None, burst=False):
    """burst: 2..4 arrivals every iteration and long answers, so that more requests run than DECODE_CAP lets decode in one group."""
    world = World(seed)
    rng = world.rng
    r = FakeRunner(S, world, budget, max_snaps, exact_k=exact_k)
    world.runner = r
    for _ in range(min(max_snaps, 4)):
        r.pool.append(r._new_snap())
    convs = []  # finished (prompt, output) pairs that a later request may continue
    reqs, waiting, admitted = [], deque(), []
    it = 0

    def new_request():
        if convs and rng.random() < 0.6:  # a follow-up turn: the previous prompt + its answer + a new turn (resumes from a snapshot or the host store)
            pr, out = rng.choice(convs)
            prompt = pr + out[: rng.randint(0, len(out))] + [R.IM_START] + [rng.randrange(2000, 3000) for _ in range(rng.randint(3, 60))]
        else:
            sysp = rng.choice([None, [R.IM_START] + [5000 + i for i in range(120)]])  # a shared system prompt sometimes
            prompt = (sysp or []) + [R.IM_START] + [rng.randrange(2000, 3000) for _ in range(rng.randint(1, 300))]
        p = R.GenParams(max_tokens=rng.randint(60, 240) if burst else rng.randint(1, 120), eos=EOS, stop_ids=frozenset([STOP_ID]) if rng.random() < 0.5 else frozenset(),
                        temperature=rng.choice([0.0, 0.7]), seed=rng.randrange(1 << 30))
        rec = dict(prompt=prompt, p=p, toks=[], fin=None, done=0, abort_at=None, aborted=False, arrive=it, start=None)
        if rng.random() < 0.15:
            rec["abort_at"] = it + rng.randint(0, 12)

        def emit(t, toks, fin, sr):
            assert not rec["aborted"], "tokens after the abort"
            assert rec["fin"] is None, "tokens after the finish"
            rec["toks"] += toks
            rec["fin"] = fin

        def done(t):
            rec["done"] += 1

        t = SB.Task(len(reqs), prompt, p, emit, done, cancelled=lambda: rec["aborted"])
        rec["task"] = t
        reqs.append(rec)
        waiting.append(t)

    def refuse(t, ex):
        raise ex

    while it < 4000:
        if len(reqs) < n_req and (burst or rng.random() < 0.5):
            for _ in range(rng.randint(2, 4) if burst else rng.randint(1, 3)):
                if len(reqs) < n_req:
                    new_request()
        for rec in reqs:
            if rec["abort_at"] is not None and it >= rec["abort_at"] and rec["fin"] is None:
                rec["aborted"] = True
        COUNT["capped"] += len(r.tasks) > SB.DECODE_CAP
        r.step(waiting, refuse)
        late = [t.rid for t in r.tasks if r._tick - t.dtick > 1]
        assert not late, f"seed {seed}: requests {late} were left out of two sync groups in a row (DECODE_CAP round robin)"
        for rec in reqs:
            t = rec["task"]
            if rec["start"] is None and (t.seq is not None or t.finished):
                rec["start"] = it
                admitted.append(t.rid)
            if t.finished and rec["fin"] is not None and not rec["aborted"] and not rec.get("conv"):
                rec["conv"] = True
                convs.append((rec["prompt"], rec["toks"]))
        it += 1
        if len(reqs) >= n_req and not waiting and not r.tasks:
            break
    assert not waiting and not r.tasks, f"seed {seed}: {len(waiting)} waiting, {len(r.tasks)} running after {it} iterations (starvation or a lost request)"
    # ---- per request
    for rec in reqs:
        t = rec["task"]
        assert rec["done"] == 1, f"request {t.rid}: done called {rec['done']} times"
        n_new = min(rec["p"].max_tokens, r.ctx_max - r.slack(r.group, r.k, True) - len(rec["prompt"]))
        exp = expected(rec["prompt"], rec["p"], n_new)
        if rec["aborted"] and rec["fin"] is None:
            assert rec["toks"] == exp[: len(rec["toks"])], f"request {t.rid} (aborted): stream is not its own"
        else:
            assert rec["toks"] == exp, f"seed {seed} request {t.rid}: got {rec['toks']} expected {exp} (prompt {len(rec['prompt'])}, cached {t.st.cached if t.st else None} from {t.st.source if t.st else None})"
            want = R.STOP if exp[-1] in (EOS, STOP_ID) and (exp[-1] == EOS or exp[-1] in rec["p"].stop_ids) else R.LENGTH
            assert rec["fin"] == want, f"request {t.rid}: finish {rec['fin']} expected {want}"
    # FIFO admission (no request overtakes an earlier one), every slot free
    assert admitted == sorted(admitted), f"admission order {admitted}"
    assert all(not q.busy for q in r.seqs), "a slot stayed busy"
    # snapshot buffers: every one is in the pool, on a slot or in a stored entry; the store stays in its budget
    held = len(r.pool) + sum(len(q.snaps) for q in r.seqs) + (sum(len(e.snaps) for e in r.store.entries) if r.store else 0)
    assert held == r._made <= max(max_snaps, 4), f"snapshot buffers: {held} held, {r._made} made"
    ids = [id(s) for s in r.pool] + [id(s) for q in r.seqs for s in q.snaps] + ([id(s) for e in r.store.entries for s in e.snaps] if r.store else [])
    assert len(ids) == len(set(ids)), "a snapshot buffer is in two places"
    if r.store:
        assert r.store.bytes() <= r.store.budget, f"host store {r.store.bytes()} > budget {r.store.budget}"
        for ent in r.store.entries:  # the accounted bytes are the buffers the chunks hold, and the chunks tile the stored rows [0, n_kv)
            assert all(c.nbytes == c.main.numel() + c.mtp.numel() for c in ent.chunks)
            assert [c.a for c in ent.chunks] == [sum(c.m for c in ent.chunks[:i]) for i in range(len(ent.chunks))] and sum(c.m for c in ent.chunks) == ent.n_kv
    bound = 64 * SB.PREFILL_STALL_ROWS + 64 * 64 * 4  # one decode group per PREFILL_STALL_ROWS row-layers, at layer granularity (a group of <= 4 segments of 64 rows)
    assert world.max_stall <= bound, f"running requests waited {world.max_stall} row-layers of prefill (bound {bound})"
    src = [rec["task"].st.source for rec in reqs if rec["task"].st is not None]
    if verbose:
        print(f"seed {seed} S={S}: {len(reqs)} requests in {it} iterations, sources {dict((s, src.count(s)) for s in set(src))}, batched cycles {r.batched_cycles}, single {r.single_cycles}, "
              f"aborted {sum(rec['aborted'] for rec in reqs)}, max stall {world.max_stall} row-layers")
    COUNT["batched"] += r.batched_cycles
    COUNT["single"] += r.single_cycles
    COUNT["stall_max"] = max(COUNT["stall_max"], world.max_stall)
    return src


COUNT = dict(batched=0, single=0, stall_max=0, capped=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=24)
    a = ap.parse_args()
    SB.PREFILL_STALL_ROWS = 32  # interleave often (the fake prompts are short)
    srcs = []
    for seed in range(a.seeds):
        S = [2, 3, 4, 8][seed % 4]
        exact_k = [None, None, 1, 0][seed % 4] if seed % 3 == 2 else None  # the exact mode (SMK_BATCH_K) on some seeds: batched cycles at every batch size
        srcs += run(seed, S, n_req=40, budget=[0, 3000, 20000][seed % 3], max_snaps=[4, 8, 16][seed % 3], verbose=seed < 4, exact_k=exact_k)
    for seed in range(a.seeds, a.seeds + 4):  # bursts on 6 and 8 slots: more running requests than DECODE_CAP (round robin), once in exact mode (no cap)
        srcs += run(seed, [6, 8][seed % 2], n_req=40, budget=20000, max_snaps=16, exact_k=1 if seed == a.seeds + 3 else None, burst=True)
    for s in ("cold", "device", "host"):
        assert s in srcs, f"no request resumed from {s}: the fuzz does not cover it"
    assert COUNT["batched"] > 0 and COUNT["single"] > 0 and COUNT["capped"] > 0, f"coverage: {COUNT}"
    print(f"PASS: {a.seeds} + 4 burst seeds, {len(srcs)} requests, sources {dict((s, srcs.count(s)) for s in set(srcs))}, cycles and capped groups {COUNT}")


if __name__ == "__main__":
    main()
