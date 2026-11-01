#!/usr/bin/env python
"""Engine-level checks of smk_runner.Runner without HTTP (needs the GPU lease, ~1 min):
  * greedy ids identical for k = 3, 2, 1, 0 (speculation must not change the answer; k=0 = plain decode graph)
  * stop-token / max_tokens cut exactly at the stop (extra speculated tokens of the last cycle group are discarded)
  * prefix reuse: repeating a prompt resumes at the last prompt token and gives the identical answer; a growing chat resumes at the snapshot and its first-token logits agree with a cold run to bf16 noise
  * sampling: same seed -> same tokens (cold and warm), different seed -> different tokens
  ~/llm-lab/venv/bin/python server/runner_check.py [--fp8] [--pld]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoTokenizer  # noqa: E402

from smk_runner import GenParams, Runner  # noqa: E402

MODEL = Path.home() / "models" / "swift-1.5-int4"
EOS = 248046
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail if not cond else ''}", flush=True)


def gen(r, ids, **kw):
    toks, fin = [], None
    for t, f, sr in r.run(ids, GenParams(**kw)):
        toks += t
        fin = f or fin
    return toks, fin


def first_logits(r, ids):
    """Run a request up to its first token and return (fp32 logits of the last prompt row, 48 greedy tokens of a second run from the final snapshot, cached tokens of the first) -- the engine's logits buffer is intact while the generator is suspended."""
    g = r.run(ids, GenParams(max_tokens=1, eos=EOS))
    next(g)
    lg, cached = r.e.logits(1)[0].clone(), r.last.cached
    g.close()
    toks, _ = gen(r, ids, max_tokens=48, eos=EOS)
    return lg, toks, cached


def main():
    fp8 = "--fp8" in sys.argv
    tok = AutoTokenizer.from_pretrained(str(MODEL))
    chat = lambda msgs: tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False), add_special_tokens=False)["input_ids"]  # noqa: E731
    r = Runner(ctx_max=8192, kv_fp8=int(fp8), k=3, group=2, pld="--pld" in sys.argv)  # --pld: K11 hybrid draft for k = 3 (k = 2, 1, 0 run the plain graphs on the 9-slot ring)
    p1 = chat([{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity."}])
    ref, fin = gen(r, p1, max_tokens=96, eos=EOS)
    check("greedy k=3 produces 96 tokens", len(ref) == 96 and fin == "length", f"{len(ref)} {fin}")
    for k in (2, 1, 0):
        r.k = k
        got, _ = gen(r, p1, max_tokens=96, eos=EOS)
        check(f"greedy k={k} == k=3", got == ref, f"first diff at {next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), None)}")
    r.k = 3
    for n in (1, 2, 3, 4, 5, 7, 8, 9, 13):  # every position inside a 2-cycle group
        got, fin = gen(r, p1, max_tokens=n, eos=EOS)
        check(f"max_tokens={n} cuts exactly", got == ref[:n] and fin == "length", f"{len(got)} {fin}")
    stop_tok = ref[20]
    got, fin = gen(r, p1, max_tokens=96, eos=None, stop_ids=frozenset({stop_tok}))
    cut = ref.index(stop_tok)
    check("stop token id: cut at its first occurrence, kept, finish=stop", got == ref[: cut + 1] and fin == "stop", f"{len(got)} vs {cut + 1} {fin}")
    got, fin = gen(r, p1, max_tokens=96, eos=ref[30])
    check("eos id: cut at its first occurrence", got == ref[: ref.index(ref[30]) + 1] and fin == "stop")

    # prefix reuse
    st = r.last
    again, _ = gen(r, p1, max_tokens=96, eos=EOS)
    check("repeat prompt resumes at the last prompt token (cached = len-1)", r.last.cached == len(p1) - 1 and again == ref, f"cached {r.last.cached} of {len(p1)}")
    doc = "\n\n".join(p.read_text(errors="ignore") for p in sorted(Path("topics").glob("*.md")))
    doc = tok.decode(tok(doc, add_special_tokens=False)["input_ids"][:1200])
    for think in (False, True):  # thinking off: the next turn's history keeps the generation prompt -> resume at the last prompt token; on: it drops "<think>\n" -> resume at the <|im_start|>assistant\n snapshot
        chat_t = lambda msgs: tok(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=think), add_special_tokens=False)["input_ids"]  # noqa: E731
        convo = [{"role": "user", "content": doc + "\n\nQuestion 1: list the three most important topics above."}]
        outs = []
        for q in (None, "Question 2: which of them is the least certain? One sentence.", "Question 3: summarise your last answer in one sentence."):
            if q:
                convo += [{"role": "assistant", "content": tok.decode(outs[-1][:-1]).split("</think>")[-1].strip()}, {"role": "user", "content": q}]
            ids = chat_t(convo)
            lg_w, warm, c_warm = first_logits(r, ids)
            outs.append(warm)
            gen(r, chat([{"role": "user", "content": "17*23?"}]), max_tokens=4, eos=EOS)  # replace the lineage
            lg_c, cold, _ = first_logits(r, ids)
            top = lg_c.topk(64)
            rel = float((lg_w[top.indices] - top.values).norm() / top.values.norm())
            margin = float(top.values[0] - top.values[1])
            label = f"chat (thinking {'on' if think else 'off'}) turn {len(outs)}: resumed at {c_warm}/{len(ids)}"
            # the resumed prefill has another fast-prefill segmentation than the cold one: same state up to bf16 accumulation order (K6 noise floor: hidden rel-L2 1.4-2.8e-2)
            check(label + f": first-token logits warm vs cold rel-L2 {rel:.4f} (top-64), top1-top2 margin {margin:.2f}", rel < 0.05 and (int(lg_w.argmax()) == int(lg_c.argmax()) or margin < 0.2), f"argmax {int(lg_w.argmax())} vs {int(lg_c.argmax())}")
            if len(outs) > 1:
                check(label + ": snapshot used", c_warm > 0)
    # sampling
    kw = dict(max_tokens=40, eos=EOS, temperature=0.8, top_p=0.95, top_k=20)
    a, _ = gen(r, p1, seed=5, **kw)
    b, _ = gen(r, p1, seed=5, **kw)  # warm (snapshot restore + counter reset)
    gen(r, chat([{"role": "user", "content": "17*23?"}]), max_tokens=4, eos=EOS)
    c, _ = gen(r, p1, seed=5, **kw)  # cold
    d, _ = gen(r, p1, seed=6, **kw)
    check("sampling: seed reproducible warm and cold", a == b == c, f"{a[:6]} {b[:6]} {c[:6]}")
    check("sampling: different seed differs", a != d)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
