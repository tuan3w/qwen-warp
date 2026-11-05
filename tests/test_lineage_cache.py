#!/usr/bin/env python
"""Host prefix cache end to end (needs the GPU and the checkpoint): interleaved conversations resume from host memory and every output equals the cold output.

  python tests/test_lineage_cache.py [--kv-dtype 0|1|2] [--ctx 16384] [--max-tokens 40]
Four prompts share one Runner: A (a 6k-token conversation), B (a different 6k one), A2 = A + a new turn, B2 = B + a new turn, plus a short helper prompt H.  For each prompt a COLD
reference is generated first (store off, cache dropped).  Then the interleaved sequence A, B, A2, H, B2, A2' (a different extension of A) runs with the host store on and must show:
  * A2 / B2 resume from the HOST (source == "host", cached > 0) although the device held another conversation (the state of the art before this change: cached == 0);
  * every token of every request equals its cold reference, greedy AND seeded sampling (a restored KV row or state that differs in one bit changes tokens quickly);
  * a store budget too small for two conversations still gives exact outputs (entries evicted, requests cold);
  * H (shorter than min_spill) is not spilled and does not evict the cached conversations.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import smk_runner as R  # noqa: E402


def conv(seed, n, turn=900):
    """n tokens of a pseudo-conversation: random ids with <|im_start|> user/assistant markers every ~turn tokens (so the prefill plan has turn-boundary cuts)."""
    rng = np.random.RandomState(seed)
    out = []
    while len(out) < n:
        out += [R.IM_START, 872 if len(out) % 2 == 0 else 77091] + [int(x) for x in rng.randint(1000, 30000, size=turn)]
    return out[:n]


def gen(r, ids, max_tokens, temperature=0.0, seed=None):
    toks = []
    p = R.GenParams(max_tokens=max_tokens, temperature=temperature, top_k=20 if temperature else 0, top_p=0.95 if temperature else 1.0, seed=seed)
    for out, fin, _ in r.run(ids, p):
        toks += out
        if fin is not None:
            break
    return toks, r.last


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-dtype", type=int, default=2, choices=[0, 1, 2], help="0 bf16, 1 fp8, 2 int4 KV")
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--host-tables", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    r = R.Runner(ctx_max=a.ctx, kv_fp8=a.kv_dtype, k=3, group=2, max_snaps=16, host_tables=a.host_tables, prefix_cache_gb=2.0, pld=False)
    print(f"runner ready in {time.time() - t0:.0f} s (kv dtype {a.kv_dtype})", flush=True)
    A, B = conv(1, 6000), conv(2, 6000)
    A2, B2 = A + conv(3, 700), B + conv(4, 700)
    A2b = A + conv(5, 900)
    H = conv(6, 450)
    prompts = dict(A=A, B=B, A2=A2, B2=B2, A2b=A2b, H=H)

    # cold references (store off, cache dropped before each): greedy and seeded sampling
    store, r.store = r.store, None
    ref, ref_s = {}, {}
    for k, ids in prompts.items():
        r.drop_cache()
        ref[k], _ = gen(r, ids, a.max_tokens)
        r.drop_cache()
        ref_s[k], _ = gen(r, ids, a.max_tokens, temperature=0.8, seed=1234)
    r.drop_cache()
    r.store = store
    print("cold references done", flush=True)

    def run_seq(seq, temperature=0.0):
        res = []
        for k in seq:
            toks, st = gen(r, prompts[k], a.max_tokens, temperature=temperature, seed=1234 if temperature else None)
            exp = (ref_s if temperature else ref)[k]
            check(toks == exp, f"{k}: tokens equal the cold reference ({'sampling' if temperature else 'greedy'}, resumed at {st.cached} from {st.source}, restore {st.restore_s * 1e3:.0f} ms, spill {st.spill_s * 1e3:.0f} ms)")
            res.append((k, st))
        return res

    r.drop_cache()
    res = dict(run_seq(["A", "B", "A2", "H", "B2", "A2b"]))
    check(res["A"].source == "cold" and res["B"].source == "cold", "first requests are cold")
    check(res["A2"].source == "host" and res["A2"].cached >= 4000, f"A2 resumes conversation A from the host store (cached {res['A2'].cached})")
    check(res["B2"].source == "host" and res["B2"].cached >= 4000, f"B2 resumes conversation B from the host store (cached {res['B2'].cached})")
    check(res["H"].source == "cold" and len(r.store.entries) >= 2, "a short helper request neither evicts the stored conversations nor is spilled")
    check(res["A2b"].source == "host" and res["A2b"].cached >= 4000, f"a different extension of A also resumes from the host (cached {res['A2b'].cached})")

    r.drop_cache()
    run_seq(["A", "B", "A2", "B2"], temperature=0.8)
    print("sampling sequence matches cold references", flush=True)

    # budget too small for two conversations: exact outputs, entries evicted
    r.drop_cache()
    one_conv = r.store.bytes()  # 0 after drop_cache; measure with a real spill
    gen(r, A, 8)
    gen(r, B, 8)
    one_conv = r.store.bytes()
    check(one_conv > 0, f"spilled conversation occupies {one_conv / 1e6:.0f} MB of host memory")
    r.store.budget = int(one_conv * 1.3)
    r.drop_cache()
    run_seq(["A", "B", "A2", "B2", "A2b"])
    check(len(r.store.entries) <= 1 and r.store.bytes() <= r.store.budget * 1.0 + 1, "a one-conversation budget keeps at most one entry")

    # system-prompt checkpoint: two prompts share a 3k-token system prompt and differ after it (parallel agents of one type)
    SYS = conv(11, 3000)
    S1, S2 = SYS + conv(12, 1800), SYS + conv(13, 1800)
    r.store.budget = 2 << 30
    r.drop_cache()
    r.system_checkpoint = False
    cold_s2 = gen(r, S2, a.max_tokens)[0]
    r.drop_cache()
    gen(r, S1, 8)
    toks, st_off = gen(r, S2, a.max_tokens)
    check(toks == cold_s2 and st_off.cached == 0, f"without the system checkpoint the second agent resumes nothing (cached {st_off.cached})")
    r.system_checkpoint = True
    r.drop_cache()
    gen(r, S1, 8)
    toks, st_on = gen(r, S2, a.max_tokens)
    check(toks == cold_s2 and 1500 <= st_on.cached <= 3000 and st_on.source == "device", f"with it the second agent resumes at the end of the shared system prompt from the device (cached {st_on.cached}), same tokens as cold")
    r.drop_cache()
    gen(r, S1, 8)
    gen(r, H, 8)  # an unrelated request overwrites the device lineage; S1 is spilled with its system checkpoint
    gen(r, B, 8)
    toks, st_host = gen(r, S2, a.max_tokens)
    check(toks == cold_s2 and st_host.source == "host" and 1500 <= st_host.cached <= 3000, f"... and from the host store after other conversations ran in between (cached {st_host.cached})")

    # snapshot buffer accounting: every buffer is in exactly one place
    total = len(r.snaps) + len(r.pool) + sum(len(e.snaps) for e in r.store.entries)
    check(total == r._made <= r.max_snaps, f"snapshot buffers conserved ({total} = made {r._made} <= {r.max_snaps})")
    print("ALL PASS")


if __name__ == "__main__":
    main()
