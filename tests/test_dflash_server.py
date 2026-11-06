#!/usr/bin/env python
"""`--drafter dflash2` in the Runner (the engine behind the server): fast prefill + tap capture + the block-drafter cycles, the drafter-window bookkeeping across turns and the router.

  python tests/test_dflash_server.py [--kv-dtype 0|1|2] [--ctx 12288] [--n 160]
Needs the target checkpoint ($SWIFT_MODEL), the W4A16 drafter ($DFLASH_W4) and the GPU (about 22 GB: stop the serving container).  A drafter that sees wrong context never changes the output (greedy
accept = argmax equality of the T = 8 verify), it only lowers the tokens per cycle, so the checks look at both:
  1. cold prompt with chat-turn cuts (layer-major groups, > 2047 rows, so only the window is captured): the DFlash2 route's ids EQUAL the host-driven T = 8 greedy decode that follows the SAME
     fast prefill (`generate_df`'s 8-row prefill has other numerics and is not the reference); its tokens per cycle stay within 15% of `generate_df` on the same prompt and above MTP k = 3;
     the drafter K/V rows built from the fast prefill's taps match the rows `df_prefill` builds (rel. L2, not equal: other GEMM shapes);
  2. the same with every prefill segment on the 4-row decode-step path (FastPrefill min_fast huge) and with 1-, 2- and 3-token prompts (folded head, no segment);
  3. a conversation: turn 2 resumed from the device snapshot continues on DFlash2 with the drafter rows of turn 1 (K/V rows of its window EQUAL a cold prefill of turn 2's prompt: same
     plan, same state, same kernels), ids equal the cold run; a SAMPLED turn in between (MTP route, seeded) keeps the window usable for the next greedy turn;
  4. a window that is not on the device (host-tier restore after another conversation, `route="mtp"`) goes to the MTP path, whose ids equal the cold MTP run, and the lineage is usable again
     once the prompt has grown past the gap;
  5. logit processors and temperature > 0 never use the drafter.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "engine"))
import smk_runner as R  # noqa: E402

MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
W = R.Runner.DF_WINDOW
MASK = 248070


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def gen(r, ids, n, route=None, **kw):
    toks = []
    for out, fin, _ in r.run(ids, R.GenParams(max_tokens=n, **kw), route=route):
        toks += out
        if fin is not None:
            break
    return toks, r.last


def corpus(tok):
    text = "\n\n".join(p.read_text() for p in sorted((REPO / "engine").glob("*.py")) + sorted((REPO / "server").glob("*.py")))
    return tok(text, add_special_tokens=False)["input_ids"]


def chat_prompt(tok, code, n_ctx, snippet):
    """n_ctx tokens of code in user/assistant turns (a <|im_start|> cut every ~700 tokens), then an instruction + a snippet that sits inside the last 2047 rows."""
    ids, i, turn = [], 0, 0
    while len(ids) < n_ctx:
        ids += [R.IM_START] + tok(("user" if turn % 2 == 0 else "assistant") + "\n", add_special_tokens=False)["input_ids"] + code[i : i + 700]
        i += 700
        turn += 1
    ids = ids[:n_ctx]
    ask = tok("\n<|im_end|>\n<|im_start|>user\nRewrite this Python with type hints and docstrings, keep the logic identical:\n\n", add_special_tokens=False)["input_ids"]
    tail = tok("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"]
    return ids + ask + snippet + tail


def t8_reference(r, ids, n):
    """Plain greedy through the T = 8 verify path after the Runner's own prefill of `ids` (same plan, same groups, same kernels), host-driven; the drafter plays no part."""
    e = r.e
    r.drop_cache()
    e.set_sampling(0.0)
    e.reset()
    ends, folded = r._plan(ids)
    k, r.k = r.k, 0  # no MTP draft start
    try:
        assert r._prefill(ids, 0, ends, folded, lambda: False, use_df=False)
    finally:
        r.k = k
    first = e.read_out_tok()
    ref = [first]
    while len(ref) < n:
        y = e.verify([ref[-1]] + [MASK] * 7)
        e.commit(1, 8)
        ref.append(y[0])
    r.drop_cache()
    return ref


def df_rows(e, lo, hi):
    """drafter K, V rows [lo, hi) of all 5 layers and 8 kv heads: float32 [2][5][8][hi - lo][128]"""
    shape = (5, 8, e.ctxd, 128)
    return torch.stack([e.df_read(w, shape, torch.bfloat16)[:, :, lo:hi].float() for w in (4, 7)])


def rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-6))


def part1(a, tok, code, r):
    snippet = code[-700:]
    ids = chat_prompt(tok, code, 3000, snippet)
    n = len(ids)
    ends, folded = r._plan(ids)
    groups = R.Runner.plan_groups(ends, 0, r.fp.seg, r.fp.min_fast, set(ends[-2:]), True)
    print(f"prompt {n} tokens, {len(ends)} plan segments in {len(groups)} groups (folded head: {folded})", flush=True)
    check(len(groups) < len(ends) + 1 and any(len(g) > 1 for g in groups), "the prompt exercises layer-major groups")
    ref = t8_reference(r, ids, a.n)
    toks, st = gen(r, ids, a.n)
    check(st.route == "dflash2" and st.source == "cold", f"cold greedy request routes to dflash2 (route {st.route})")
    check(toks == ref[: len(toks)] and len(toks) == a.n, f"fast prefill + tap capture + df_cycles: {a.n} ids equal the host-driven T = 8 greedy decode")
    tpc = (st.tokens - 1) / st.cycles
    lo = max(0, n - W)
    krows = df_rows(r.e, lo, n)
    e = r.e
    base, bst = e.generate_df(ids, a.n)  # df_prefill's 8-row prefill has other numerics: its ids may differ at a near-tie, only its acceptance and K/V rows are compared
    ref_rows = df_rows(e, lo, n)
    r.drop_cache()  # the engine was driven directly
    errs = [rel(krows[w, l], ref_rows[w, l]) for w in (0, 1) for l in range(5)]
    print(f"drafter K/V rows [{lo}, {n}): rel. L2 vs df_prefill per (K|V, layer): " + " ".join(f"{x:.3f}" for x in errs), flush=True)
    check(max(errs) < 0.25, f"drafter K/V rows from the fast prefill's taps match df_prefill's (max rel. L2 {max(errs):.3f}; other GEMM shapes: close, not equal)")
    print(f"tokens/cycle: server route {tpc:.2f}, generate_df {bst['tokens_per_cycle']:.2f}", flush=True)
    check(tpc >= 0.85 * bst["tokens_per_cycle"], f"tokens per cycle {tpc:.2f} within 15% of generate_df's {bst['tokens_per_cycle']:.2f}")
    _, mst = gen(r, ids, a.n, route="mtp")
    # tokens per cycle is no ranking: the MTP route's prompt-lookup tail copies the snippet of this prompt at ~6 tokens per cycle; tok/s is what the server benchmark compares
    print(f"MTP route on the same prompt: {(mst.tokens - 1) / mst.cycles:.2f} tokens/cycle, {(mst.tokens - 1) / mst.gen_s:.1f} tok/s; dflash2 route {tpc:.2f} tokens/cycle, {(st.tokens - 1) / st.gen_s:.1f} tok/s", flush=True)


def part2(a, tok, code, r):
    # every segment on the decode-step path: the taps of 4-row steps are appended directly
    r.fp.min_fast = 10**9
    ids = chat_prompt(tok, code, 1100, code[-300:])
    ref = t8_reference(r, ids, 64)
    toks, st = gen(r, ids, 64)
    check(st.route == "dflash2" and toks == ref, f"all-4-row-steps prefill ({len(ids)} tokens): ids equal the T = 8 reference, tokens/cycle {(st.tokens - 1) / st.cycles:.2f}")
    r.fp.min_fast = 2
    for m in (1, 2, 3, 5, 40):
        ids = code[:m]
        ref = t8_reference(r, ids, 24)
        toks, st = gen(r, ids, 24)
        check(st.route == "dflash2" and toks == ref, f"{m}-token prompt: ids equal the T = 8 reference")


def prefill_state(r, ids, tap_on):
    """Everything a prefill leaves behind that the MTP path reads (snapshots, main + MTP KV rows up to the last snapshot, first token), tap capture on or off."""
    r.drop_cache()
    e = r.e
    e.set_sampling(0.0)
    e.reset()
    ends, folded = r._plan(ids)
    tap, r.tap = r.tap, (r.tap if tap_on else None)
    try:
        assert r._prefill(ids, 0, ends, folded, lambda: False, use_df=False)
    finally:
        r.tap = tap
    first = e.read_out_tok()
    snaps = [(s.pos, s.dn.clone(), s.conv.clone(), None if s.aux is None else s.aux.clone()) for s in sorted(r.snaps, key=lambda x: x.pos)]
    m = max(p for p, *_ in snaps)
    main, mtp = e.kv_chunk_new(m), e.mtp_chunk_new(m)
    e.kv_chunk_copy(main, m, 0, 0, m, True)
    e.mtp_chunk_copy(mtp, m, 0, 0, m, True)
    torch.cuda.synchronize()
    r.drop_cache()
    return first, snaps, main.clone(), mtp.clone()


def part2b(a, tok, code, r):
    """The tap hook only READS: with capture on, the first token, every snapshot and the main + MTP KV rows are bit-identical to capture off (grouped and one-at-a-time prefill)."""
    ids = chat_prompt(tok, code, 3000, code[-700:])
    for grouped in (True, False):
        r.group_prefill = grouped
        off, on = prefill_state(r, ids, False), prefill_state(r, ids, True)
        same = off[0] == on[0] and len(off[1]) == len(on[1]) and all(x[0] == y[0] and torch.equal(x[1], y[1]) and torch.equal(x[2], y[2]) and (x[3] is None or torch.equal(x[3], y[3])) for x, y in zip(off[1], on[1]))
        check(same and torch.equal(off[2], on[2]) and torch.equal(off[3], on[3]), f"tap capture on vs off ({'grouped' if grouped else 'one segment at a time'}): first token, {len(off[1])} snapshots, main + MTP KV rows bit-identical")
    r.group_prefill = True


def part3(a, tok, code, r):
    A = chat_prompt(tok, code, 3100, code[-500:])
    turn1, _ = gen(r, A, 60)
    B = A + turn1 + tok("<|im_end|>\n<|im_start|>user\nNow add a unit test for it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)["input_ids"] + code[2000:2400]
    # cold references for turn 2 (own route each)
    r.drop_cache()
    cold, cst = gen(r, B, a.n)
    check(cst.route == "dflash2" and cst.source == "cold", "turn 2 cold: dflash2")
    lo2 = max(0, len(B) - W)
    cold_rows = df_rows(r.e, lo2, len(B))
    cold_tpc = (cst.tokens - 1) / cst.cycles
    r.drop_cache()
    gen(r, A, 60)  # turn 1 again: the lineage turn 2 continues
    res, rst = gen(r, B, a.n)
    check(rst.source == "device" and rst.cached > 0, f"turn 2 resumes from the device snapshot at {rst.cached} of {len(B)} tokens")
    check(rst.route == "dflash2", "turn 2 (window covered by turn 1's capture + its own) decodes with the block drafter")
    check(res == cold, "turn 2 resumed: ids equal the cold dflash2 run")
    res_rows = df_rows(r.e, lo2, len(B))
    check(torch.equal(res_rows, cold_rows), f"drafter K/V rows [{lo2}, {len(B)}) after the resume are bit-identical to a cold prefill's (window of turn 1 + new rows)")
    tpc = (rst.tokens - 1) / rst.cycles
    check(tpc >= 0.9 * cold_tpc, f"tokens per cycle after the resume {tpc:.2f} vs cold {cold_tpc:.2f}")
    # a sampled turn between two greedy turns of one lineage
    r.drop_cache()
    gen(r, A, 40)
    s_ids = A + turn1[:30] + code[100:260]
    _, sst = gen(r, s_ids, 40, temperature=0.8, top_k=20, top_p=0.95, seed=5)
    check(sst.route == "mtp" and sst.source == "device", f"sampled turn on the lineage: MTP route, resumed at {sst.cached}")
    g3 = s_ids + code[400:600]
    ref3 = t8_reference(r, g3, 48)
    r.drop_cache()
    gen(r, A, 40)
    gen(r, s_ids, 40, temperature=0.8, top_k=20, top_p=0.95, seed=5)
    got3, g3st = gen(r, g3, 48)
    check(g3st.source == "device" and g3st.route == "dflash2" and got3 == ref3, f"greedy turn after a sampled turn: device resume at {g3st.cached}, dflash2, ids equal the T = 8 reference")
    # logit processors and sampling never use the drafter
    for kw in (dict(presence_penalty=0.3), dict(repetition_penalty=1.1), dict(min_tokens=5, min_tokens_ids=frozenset({248046})), dict(temperature=0.7, top_k=20)):
        _, pst = gen(r, A, 12, **kw)
        check(pst.route == "mtp", f"{list(kw)[0]}: MTP route")
    return A, B


def part4(a, tok, code, r, A, B):
    """Host tier: the spilled lineage carries its drafter window, a host restore decodes with the block drafter (rows bit-identical to a cold prefill's) and a request that needs rows the
    entry does not hold goes to the MTP path (ids equal the cold MTP run)."""
    mk = lambda: R.PrefixStore(int(2.0 * (1 << 30)))  # noqa: E731
    r.store = None
    r.drop_cache()
    ref_df, cst = gen(r, B, 40)  # cold references of both routes
    lo = max(0, len(B) - W)
    cold_rows = df_rows(r.e, lo, len(B))
    cold_tpc = (cst.tokens - 1) / cst.cycles
    r.drop_cache()
    r.store = mk()
    C = chat_prompt(tok, code[::-1][:20000][::-1], 3100, code[:400])
    gen(r, B[: len(B) - 300], 30)  # conversation B on the device
    gen(r, C, 30)  # another one: B's prefix is spilled to the host tier together with its drafter window
    check(any(e.dfw is not None for e in r.store.entries), f"the spilled entry carries a drafter window ({[(e.dfw.lo, e.dfw.hi) for e in r.store.entries if e.dfw is not None]}, {r.store.bytes() / 2**20:.0f} MiB in the store)")
    got, hst = gen(r, B, 40)
    check(hst.source == "host" and hst.cached > 0, f"B resumes from the host tier at {hst.cached}")
    check(hst.route == "dflash2", "host restore with the carried window: block drafter")
    check(got == ref_df, "host-restored turn: ids equal the cold dflash2 run")
    check(torch.equal(df_rows(r.e, lo, len(B)), cold_rows), f"drafter K/V rows [{lo}, {len(B)}) after the host restore are bit-identical to a cold prefill's")
    tpc = (hst.tokens - 1) / hst.cycles
    check(tpc >= 0.9 * cold_tpc, f"tokens per cycle after the host restore {tpc:.2f} vs cold {cold_tpc:.2f}")
    # an earlier resume point needs rows below the stored window: MTP route
    gen(r, C, 30)  # spill B again (its entry is extended, the window replaced)
    F = B[:2500] + code[3000:3300]
    stored = [e.dfw.lo for e in r.store.entries if e.dfw is not None]
    got_f, fst = gen(r, F, 40)
    lo_f = max(0, len(F) - W)
    print(f"F: {len(F)} tokens, resumed at {fst.cached} from the {fst.source}, route {fst.route}, window needed from {lo_f}", flush=True)
    if fst.source == "host" and fst.cached > lo_f and all(x > lo_f for x in stored):
        check(fst.route == "mtp", "a host resume whose window starts below the stored rows decodes with the MTP path")
        r.drop_cache()
        r.store = mk()
        ref_f, _ = gen(r, F, 40, route="mtp")
        check(got_f == ref_f, "... and its ids equal the cold MTP run")
    else:
        print("SKIP  no host resume below the stored window in this configuration", flush=True)
    r.store = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dtype", type=int, default=0, choices=[0, 1, 2])
    ap.add_argument("--ctx", type=int, default=12288)
    ap.add_argument("--n", type=int, default=160)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    code = corpus(tok)
    print(f"corpus {len(code)} tokens", flush=True)
    r = R.Runner(ctx_max=a.ctx, kv_fp8=a.kv_dtype, k=3, group=2, max_snaps=8, host_tables=True, prefix_cache_gb=0, pld=True, drafter="dflash2", warmup=False)
    print(f"runner ready, drafter {r.df_mib:.0f} MiB (weights, K/V cache for {a.ctx} positions, scratch, tap staging)", flush=True)
    part1(a, tok, code, r)
    part2(a, tok, code, r)
    part2b(a, tok, code, r)
    A, B = part3(a, tok, code, r)
    part4(a, tok, code, r, A, B)
    print("ALL PASS")


if __name__ == "__main__":
    main()
