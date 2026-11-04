"""Request-level logit processors (presence / frequency / repetition penalty, logit_bias, min_tokens, allowed_token_ids, min_p) against a torch reference of the vLLM semantics.  Needs a GPU (sm_86) and the
built engine, NOT the checkpoint: the engine's processor kernels are called through the test hooks of include/engine.h.

  python tests/test_logit_procs.py

A. processed logits of the verify rows (the kernel the quantised-head candidate selection sees), BIT-identical to the reference: every parameter alone and combined, the vLLM / sglang / llama.cpp
   unit cases, draft tokens of earlier rows counted as generated, min_tokens boundary; negative controls (a reference that ignores the draft tokens / the order / an off-by-one MUST differ)
B. the same function on exact candidate logits (the kernel that runs after the exact rescoring)
C. min_p in the sampler: processed distribution vs the reference (llama.cpp min_p cases, random logits): max abs error and total variation
The statistical checks of the speculative cycle live in tests/test_spec_procs.py.  Reference: vllm/v1/sample/{sampler.py, ops/penalties.py, logits_processor/builtin.py}.
"""
import ctypes as C
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
import smk  # noqa: E402
from spec import make_procs, make_sampling, setup_spec_api  # noqa: E402
from vllm.model_executor.layers.utils import apply_penalties  # noqa: E402

V = 248320
INF = float("inf")
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------------------------------------ reference (vLLM v1 sampler order, torch fp32)
def ref_rows(logits, prompt=(), gen=(), drafts=(), pres=0.0, freq=0.0, rep=1.0, min_tokens=0, stop=(), bias=None, allowed=None, *, use_drafts=True, order_swap=False, min_tokens_off=0):
    """Processed logits of the rows: row t has generated gen + drafts[:t] (use_drafts=False / order_swap / min_tokens_off are the negative controls)."""
    out = logits.clone()
    V = out.shape[1]
    for t in range(out.shape[0]):
        hist = list(gen) + (list(drafts[:t]) if use_drafts else [])
        x = out[t : t + 1]
        if allowed:  # sampler.py: allowed_token_ids mask first
            m = torch.ones(V, dtype=torch.bool)
            m[list(allowed)] = False
            x.masked_fill_(m.unsqueeze(0), -INF)
        if min_tokens and len(hist) + min_tokens_off < min_tokens and stop:  # MinTokensLogitsProcessor: stop tokens -> -inf while len(output) < min_tokens
            x[0, list(stop)] = -INF
        if bias and not order_swap:  # LogitBiasLogitsProcessor: logits[req, tok] += bias
            for k, v in bias.items():
                x[0, k] += v
        if pres != 0.0 or freq != 0.0 or rep != 1.0:  # ops/penalties.apply_all_penalties: repetition, then frequency, then presence
            if rep != 1.0:  # the CUDA op csrc/sampler.cu apply_repetition_penalties_kernel (logit > 0 ? logit / penalty : logit * penalty); the torch fallback multiplies by 1 / penalty (1 ulp off)
                seen = torch.zeros(V, dtype=torch.bool)
                seen[list(prompt) + hist] = True
                x[0] = torch.where(seen, torch.where(x[0] > 0, x[0] / rep, x[0] * rep), x[0])
            otk = torch.tensor([hist if hist else [V]], dtype=torch.int64)
            ptk = torch.tensor([list(prompt) if len(prompt) else [V]], dtype=torch.int64)
            apply_penalties(x, ptk, otk, torch.tensor([pres]), torch.tensor([freq]), torch.tensor([1.0]))
        if bias and order_swap:
            for k, v in bias.items():
                x[0, k] += v
    return out


# ------------------------------------------------------------------------------------------------ engine hooks
LIB = smk.load()
setup_spec_api(LIB)


def eng_rows(logits, prompt=(), gen=(), drafts=(), pres=0.0, freq=0.0, rep=1.0, min_tokens=0, stop=(), bias=None, allowed=None):
    T = logits.shape[0]
    p, keep = make_procs(prompt=prompt, presence=pres, frequency=freq, repetition=rep, min_tokens=min_tokens, bias=bias, allowed=allowed, stop=stop)
    g = (C.c_int * max(1, len(gen)))(*gen)
    d = (C.c_int * 8)(*(list(drafts) + [0] * (8 - len(drafts))))
    lg = logits.contiguous()
    out = torch.empty_like(lg)
    smk.check(LIB.smk_test_proc_logits(C.byref(p), T, g, len(gen), d, C.c_void_p(lg.data_ptr()), C.c_void_p(out.data_ptr())), "test_proc_logits")
    assert keep
    return out


def eng_cand(cand, exact, prompt=(), gen=(), drafts=(), pres=0.0, freq=0.0, rep=1.0, min_tokens=0, stop=(), bias=None, allowed=None):
    T = cand.shape[0]
    p, keep = make_procs(prompt=prompt, presence=pres, frequency=freq, repetition=rep, min_tokens=min_tokens, bias=bias, allowed=allowed, stop=stop)
    g = (C.c_int * max(1, len(gen)))(*gen)
    d = (C.c_int * 8)(*(list(drafts) + [0] * (8 - len(drafts))))
    c, ex = cand.contiguous(), exact.contiguous()
    out = torch.empty_like(ex)
    smk.check(LIB.smk_test_proc_cand(C.byref(p), T, g, len(gen), d, C.c_void_p(c.data_ptr()), C.c_void_p(ex.data_ptr()), C.c_void_p(out.data_ptr())), "test_proc_cand")
    assert keep
    return out


def same(a, b):
    return bool(torch.equal(a, b))


# ------------------------------------------------------------------------------------------------ A / B
def random_case(rng, T=4):
    g = torch.Generator().manual_seed(int(rng))
    logits = torch.randn(T, V, generator=g) * 3.0
    logits[:, :50] = 0.0  # exact zeros: the sign branch of the repetition penalty
    prompt = torch.randint(0, V, (300,), generator=g).tolist()
    pool = torch.randint(0, V, (12,), generator=g).tolist()
    gen = [pool[int(i)] for i in torch.randint(0, 12, (40,), generator=g)]  # counts > 1
    drafts = [pool[0], prompt[5], 7][: T - 1]  # a repeated generated token, a prompt token, a fresh token
    return g, logits, prompt, gen, drafts, pool


def part_ab():
    g, logits, prompt, gen, drafts, pool = random_case(1)
    stop = [248046, 248044]
    bias = {pool[1]: 5.5, 11: -100.0, pool[2]: 100.0, 123456: -3.25}
    allowed_small = [pool[0], 4242]
    allowed_big = torch.randperm(V, generator=g)[:10000].tolist() + pool
    base = dict(prompt=prompt, gen=gen, drafts=drafts)
    cases = {
        "presence 1.5": dict(pres=1.5),
        "presence -2": dict(pres=-2.0),
        "frequency 0.7 (counts > 1)": dict(freq=0.7),
        "frequency -1.3": dict(freq=-1.3),
        "repetition 1.0 (off)": dict(rep=1.0),
        "repetition 1.3": dict(rep=1.3),
        "repetition 0.6": dict(rep=0.6),
        "repetition 2 on zero/negative logits": dict(rep=2.0),
        "logit_bias": dict(bias=bias),
        "allowed 2 tokens": dict(allowed=allowed_small),
        "allowed 10,000 tokens": dict(allowed=allowed_big),
        "min_tokens covers rows 0,1 (len(gen) + 2)": dict(min_tokens=len(gen) + 2, stop=stop),
        "min_tokens already reached": dict(min_tokens=len(gen), stop=stop),
        "min_tokens 1 beyond (all four rows masked)": dict(min_tokens=len(gen) + 4, stop=stop),
        "all penalties": dict(pres=0.5, freq=0.4, rep=1.2),
        "everything": dict(pres=0.5, freq=0.4, rep=1.2, bias=bias, allowed=allowed_big, min_tokens=len(gen) + 2, stop=stop + [pool[3]]),
        "everything with a 2-token allowed set": dict(pres=1.0, freq=1.0, rep=1.5, bias=bias, allowed=allowed_small, min_tokens=len(gen) + 1, stop=[allowed_small[1]]),
    }
    for name, kw in cases.items():
        kw = {**base, **kw}
        ref, got = ref_rows(logits, **kw), eng_rows(logits, **kw)
        check(f"A processed logits bit-identical: {name}", same(ref, got), f"{int((ref != got).sum())} of {ref.numel()} differ" if not same(ref, got) else "")
    x = logits[:1].clone()  # vLLM's torch fallback of the repetition penalty (x * (1 / penalty)) is within 1 ulp of the CUDA op the engine mirrors
    apply_penalties(x, torch.tensor([prompt]), torch.tensor([gen]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([1.3]))
    got = eng_rows(logits[:1], prompt=prompt, gen=gen, rep=1.3)
    check("A repetition 1.3 within 1 ulp of vLLM's torch fallback", bool(torch.allclose(x, got, rtol=2.4e-7, atol=0.0)), f"max rel diff {float(((x - got).abs() / x.abs().clamp_min(1e-30)).max()):.2e}")
    # one row, no history, no drafts (the first token of a request)
    kw = dict(prompt=prompt, pres=1.0, freq=1.0, rep=1.5, bias=bias, min_tokens=3, stop=stop)
    check("A first-token row (empty history, T = 1) bit-identical", same(ref_rows(logits[:1], **kw), eng_rows(logits[:1], **kw)))
    # verify rows T = 8 (prompt-lookup tail): drafts d1..d7 all matter
    l8 = torch.randn(8, V, generator=g) * 3.0
    d8 = [pool[0], pool[0], pool[1], 9, pool[0], prompt[1], pool[1]]
    kw = dict(prompt=prompt, gen=gen, drafts=d8, pres=0.8, freq=0.6, rep=1.1, min_tokens=len(gen) + 5, stop=stop)
    check("A T = 8 rows with 7 draft tokens bit-identical", same(ref_rows(l8, **kw), eng_rows(l8, **kw)))
    # negative controls: each wrong reference must be caught
    kw = dict(base, pres=1.5, freq=0.7)
    check("A negative control: ignoring the draft tokens is detected", not same(ref_rows(logits, **kw, use_drafts=False), eng_rows(logits, **kw)))
    kw = dict(base, min_tokens=len(gen) + 2, stop=stop)
    check("A negative control: min_tokens off by one is detected", not same(ref_rows(logits, **kw, min_tokens_off=1), eng_rows(logits, **kw)))
    kw = dict(base, bias={pool[0]: 8.0}, rep=1.7)
    check("A negative control: bias after the penalties (wrong order) is detected", not same(ref_rows(logits, **kw, order_swap=True), eng_rows(logits, **kw)))

    # B: candidates = exact logits at chosen ids: same values as the full-row kernel at those ids
    cand = torch.full((4, 64), -1, dtype=torch.int32)
    for t in range(4):
        ids = torch.randperm(V, generator=g)[:40].tolist() + drafts + pool + prompt[:8] + [11, 123456, 248046, 248044, allowed_small[1]]
        cand[t, : len(ids[:56])] = torch.tensor(ids[:56], dtype=torch.int32)  # the last 8 slots stay -1
    cm = cand.long().clamp_min(0)
    for name, kw in {"everything": cases["everything"], "all penalties": cases["all penalties"], "bias + allowed 2": dict(bias=bias, allowed=allowed_small),
                     "min_tokens": cases["min_tokens covers rows 0,1 (len(gen) + 2)"]}.items():
        kw = {**base, **kw}
        full = logits.clone()
        exact = torch.gather(full, 1, cm)
        ref = torch.gather(ref_rows(full, **kw), 1, cm)
        got = eng_cand(cand, exact, **kw)
        valid = cand >= 0
        ok = same(torch.where(valid, ref, torch.zeros(())), torch.where(valid, got, torch.zeros(()))) and bool(torch.all(got[~valid] == exact[~valid]))
        check(f"B candidate kernel bit-identical to the reference (unused slots untouched): {name}", ok)

    # ported unit cases ------------------------------------------------------------------------------
    c = torch.full((1, V), 1e-2)  # vLLM tests/v1/sample/test_sampler.py: constant logits, the penalised token is the argmin
    out_tok = [3, 4, 5, 5, 4, 3]
    for pres in (-2.0, 2.0):
        o = eng_rows(c, gen=out_tok, pres=pres)[0]
        pen, non = int(o.argmin()), int(o.argmax())
        check(f"vLLM case: presence_penalty {pres:+.0f} " + ("penalises tokens already generated" if pres > 0 else "prefers tokens already generated"),
              (pen in out_tok and non not in out_tok) if pres > 0 else (non in out_tok and pen not in out_tok))
    for fp in (-2.0, 2.0):
        o = eng_rows(c, gen=[3, 4, 4, 5, 5, 5], freq=fp)[0]
        pen, non = int(o.argmin()), int(o.argmax())
        check(f"vLLM case: frequency_penalty {fp:+.0f} " + ("penalises the most frequent token most" if fp > 0 else "prefers the most frequent token"), (pen == 5 and non not in (3, 4, 5)) if fp > 0 else (non == 5 and pen not in (3, 4, 5)))
    for rp in (0.1, 1.9):
        o = eng_rows(c, prompt=[20, 21], gen=[30], rep=rp)[0]
        pen, non = int(o.argmin()), int(o.argmax())
        seen = {20, 21, 30}
        check(f"vLLM case: repetition_penalty {rp} on prompt + generated tokens", (pen in seen and non not in seen) if rp > 1 else (non in seen and pen not in seen))
    o = eng_rows(c, allowed=[3, 4])[0]
    check("vLLM case: allowed_token_ids masks every other token with -inf", bool(torch.all(torch.isinf(o[5:]))) and bool(torch.all(o[:3] == -INF)) and bool(torch.all(o[3:5] == c[0, 3:5])))
    o = eng_rows(torch.zeros(1, V), gen=[5], freq=2.0)[0]  # sglang test_penaltylib.py: frequency 2.0 on one generated token -> -2.0, others untouched
    check("sglang case: frequency 2.0, token 5 generated once -> -2.0, others 0", float(o[5]) == -2.0 and float(o[0]) == 0.0)
    o = eng_rows(torch.zeros(1, V), gen=[3, 3], freq=1.0)[0]
    check("sglang case: frequency doubles with the count", float(o[3]) == -2.0)
    o = eng_rows(torch.zeros(1, V), gen=[7, 7], pres=1.0)[0]
    check("sglang case: presence is flat (does not grow with the count)", float(o[7]) == -1.0)
    o = eng_rows(torch.zeros(2, V), min_tokens=3, stop=[2, 5, 10], gen=[])
    check("sglang case: stop tokens blocked before min_tokens (rows with 0 and 1 generated)", bool(torch.all(o[:, [2, 5, 10]] == -INF)) and float(o[0, 0]) == 0.0)
    o = eng_rows(torch.zeros(1, V), min_tokens=2, stop=[2], gen=[10, 11])[0]
    check("sglang case: stop tokens allowed once min_tokens are generated", float(o[2]) == 0.0)
    # llama.cpp tests/test-sampling.cpp test_penalties: 5 equiprobable tokens (the engine rows are softmax inputs, so logits = log p)
    lp = torch.full((1, V), -INF)
    lp[0, :5] = torch.log(torch.tensor(0.2))

    def probs(o):
        return torch.softmax(o[0], 0)[:5].tolist()

    def close(a, b, tol=2e-6):
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    check("llama.cpp case: repeat 50 on one token", close(probs(eng_rows(lp, gen=[0], rep=50.0)), [0, .25, .25, .25, .25]))
    check("llama.cpp case: repeat 50 on three tokens", close(probs(eng_rows(lp, gen=[0, 1, 2], rep=50.0)), [0, 0, 0, .5, .5]))
    check("llama.cpp case: repeat 50, repeated occurrences count once", close(probs(eng_rows(lp, gen=[0, 1, 2, 0, 0], rep=50.0)), [0, 0, 0, .5, .5]))
    check("llama.cpp case: frequency 5 + presence 5, one token", close(probs(eng_rows(lp, gen=[0], freq=5.0, pres=5.0)), [0.000011, .249997, .249997, .249997, .249997], 2e-6))
    check("llama.cpp case: frequency 5 + presence 5, three tokens", close(probs(eng_rows(lp, gen=[0, 1, 2], freq=5.0, pres=5.0)), [0.000023, 0.000023, 0.000023, .499966, .499966], 2e-6))
    check("llama.cpp case: frequency 5 + presence 5, counts (3, 1, 1)", close(probs(eng_rows(lp, gen=[0, 1, 2, 0, 0], freq=5.0, pres=5.0)), [0.0, 0.000023, 0.000023, .499977, .499977], 2e-6))


# ------------------------------------------------------------------------------------------------ C: min_p inside the sampler
def eng_dist(logits, **cfgkw):
    R, N = logits.shape
    cfg = make_sampling(**cfgkw)
    lg = logits.contiguous().float()
    n = (C.c_int * R)()
    ids = (C.c_int * (R * 64))()
    pr = (C.c_float * (R * 64))()
    smk.check(LIB.smk_test_dist(C.c_void_p(lg.data_ptr()), R, N, None, C.byref(cfg), 0, n, ids, pr), "test_dist")
    res = []
    for r in range(R):
        res.append({int(ids[r * 64 + i]): float(pr[r * 64 + i]) for i in range(n[r])})
    return res


def ref_dist(logits, temperature, top_k, top_p, min_p):
    """vLLM order: temperature -> min_p (softmax, drop p < min_p * p_max) -> top-k -> top-p (keep token i iff the mass of the larger tokens < top_p) -> softmax."""
    x = logits.double() / temperature
    p = torch.softmax(x, 0)
    if min_p > 0:
        x = x.masked_fill(p < min_p * p.max(), -INF)
    if top_k > 0:
        kth = torch.topk(x, min(top_k, x.numel())).values[-1]
        x = x.masked_fill(x < kth, -INF)
    p = torch.softmax(x, 0)
    sp, si = torch.sort(p, descending=True)
    before = torch.cumsum(sp, 0) - sp
    keep = (before < top_p) | (torch.arange(len(sp)) == 0)
    keep &= sp > 0
    q = torch.zeros_like(p)
    q[si[keep]] = sp[keep]
    return q / q.sum()


def dist_err(d, q):
    full = torch.zeros_like(q)
    for i, v in d.items():
        full[i] = v
    return float((full - q).abs().max()), 0.5 * float((full - q).abs().sum()), set(d) == set(torch.nonzero(q).flatten().tolist())


def part_c():
    # llama.cpp tests/test-sampling.cpp test_min_p: probabilities {0.1, 0.2, 0.3, 0.4}, min_p -> the survivors renormalised
    probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    for mp, want in [(0.0, [.1, .2, .3, .4]), (0.24, [.1, .2, .3, .4]), (0.26, [.2 / .9, .3 / .9, .4 / .9]), (0.49, [.2 / .9, .3 / .9, .4 / .9]), (0.51, [.3 / .7, .4 / .7]),
                     (0.74, [.3 / .7, .4 / .7]), (0.76, [1.0]), (1.0, [1.0])]:
        d = eng_dist(torch.log(probs).unsqueeze(0), temperature=1.0, top_k=4, top_p=1.0, min_p=mp)[0]
        got = sorted(d.values())
        ok = len(got) == len(want) and all(abs(a - b) < 1e-6 for a, b in zip(got, sorted(want)))
        check(f"C llama.cpp case: min_p {mp}", ok, str(got))
    # random logits: temperature / min_p / top-k / top-p combinations
    g = torch.Generator().manual_seed(7)
    worst = (0.0, 0.0)
    ok = True
    n = 0
    for T in (0.7, 1.0, 1.5):
        for mp in (0.0, 0.02, 0.1, 0.5, 0.99):
            for k, tp in ((64, 1.0), (20, 0.95), (5, 0.8)):
                lg = torch.randn(3, 30000, generator=g) * 2.5
                for r, d in enumerate(eng_dist(lg, temperature=T, top_k=k, top_p=tp, min_p=mp)):
                    q = ref_dist(lg[r], T, k, tp, mp)
                    e, tv, sup = dist_err(d, q)
                    worst = (max(worst[0], e), max(worst[1], tv))
                    ok &= sup and e < 1e-5
                    n += 1
    check(f"C min_p / temperature / top-k / top-p vs reference ({n} rows)", ok, f"max abs err {worst[0]:.2e}, max TV {worst[1]:.2e}")
    # negative control: the reference without min_p differs when min_p is active
    lg = torch.randn(1, 30000, generator=g) * 2.5
    d = eng_dist(lg, temperature=1.0, top_k=64, top_p=1.0, min_p=0.5)[0]
    e, tv, sup = dist_err(d, ref_dist(lg[0], 1.0, 64, 1.0, 0.0))
    check("C negative control: a reference without min_p is detected", e > 1e-3 or not sup, f"max abs err {e:.2e}")


def main():
    part_ab()
    part_c()
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
