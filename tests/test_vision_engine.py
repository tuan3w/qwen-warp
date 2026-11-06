#!/usr/bin/env python
"""Engine-level (Runner) exactness of the image path and of text requests next to it.  Needs an RTX 3090 and the checkpoint ($SWIFT_MODEL).

  python tests/test_vision_engine.py                     # compare with tests/data/text_ids_v070.json
  python tests/test_vision_engine.py --write-ref DIR     # (once) write that file from the Runner of a 0.7.0 checkout DIR with its engine built

Tier: 1

Exact oracles, no tolerances:
  text-only greedy ids (3 chat prompts), a seeded sampled request and a two-turn conversation (turn 2 resumed from the cache) == 0.7.0's Runner on the same configuration,
  before AND after the image requests (an image request leaves no state behind: the rope offset, the lineage key);
  an image request twice -> the same ids (the second resumed from the prefix cache past the image); turn 2 of a conversation about an image, resumed vs cold -> the same ids;
  the same text with another image resumes nowhere past the image.  Sanity: the model reads the word in the image.
Engine construction goes through tests/_engine.py (make_engine).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "data" / "text_ids_v070.json"
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))
PROMPTS = {
    "code": "Write a Python function that returns the n-th Fibonacci number using memoisation, then explain its complexity.",
    "prose": "Write a short story about a lighthouse keeper who discovers that the light has been guiding something other than ships.",
    "math": "Prove that the sum of the first n odd numbers equals n squared, then give two different proofs.",
}
EOS, IM_END = 248046, 248044
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def render(tok, messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)


def gen(r, GenParams, ids, images=None, n=64, **kw):
    out = []
    p = GenParams(max_tokens=n, eos=EOS, stop_ids=frozenset({IM_END}), **kw)
    for toks, _, _ in r.run(ids, p, images=images) if images else r.run(ids, p):
        out += toks
    return out, r.last.cached


def text_cases(r, GenParams, tok):
    """The text-only requests whose ids must equal 0.7.0's."""
    res = {}
    for name, p in PROMPTS.items():
        ids = tok(render(tok, [{"role": "user", "content": p}]), add_special_tokens=False)["input_ids"]
        res[name] = gen(r, GenParams, ids)[0]
    ids = tok(render(tok, [{"role": "user", "content": PROMPTS["prose"]}]), add_special_tokens=False)["input_ids"]
    res["sampled"] = gen(r, GenParams, ids, n=48, temperature=0.8, top_p=0.95, top_k=20, seed=7)[0]
    doc = tok.decode(tok((HERE / "long_doc.txt").read_text(), add_special_tokens=False)["input_ids"][:600])
    m = [{"role": "user", "content": doc + "\n\nWhat does this code do? Two sentences."}]
    t1 = gen(r, GenParams, tok(render(tok, m), add_special_tokens=False)["input_ids"], n=48)[0]
    m += [{"role": "assistant", "content": tok.decode(t1, skip_special_tokens=True)}, {"role": "user", "content": "Name one function it defines."}]
    t2, c2 = gen(r, GenParams, tok(render(tok, m), add_special_tokens=False)["input_ids"], n=32)
    res["turn1"], res["turn2"], res["turn2_cached"] = t1, t2, c2
    return res


def write_ref(tree):
    sys.path.insert(0, str(HERE))
    from _engine import DEFAULTS

    sys.path[:0] = [str(Path(tree) / "server"), str(Path(tree) / "engine")]
    import smk_runner
    from smk_runner import GenParams, Runner
    from transformers import AutoTokenizer

    assert Path(smk_runner.__file__).resolve().is_relative_to(Path(tree).resolve()), smk_runner.__file__
    tok = AutoTokenizer.from_pretrained(MODEL)
    res = text_cases(Runner(**DEFAULTS), GenParams, tok)
    REF.parent.mkdir(exist_ok=True)
    json.dump(dict(runner=smk_runner.__file__, defaults={k: v for k, v in DEFAULTS.items() if k != "log"}, ids=res), open(REF, "w"))
    print(f"wrote {REF} with {smk_runner.__file__}: {', '.join(f'{k} {len(v) if isinstance(v, list) else v}' for k, v in res.items())}")


def image_prompt(proc, text, im, key):
    from vision import IMAGE_TOKEN, ImageInput

    b = proc(text=[render(proc.tokenizer, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}])], images=[im], return_tensors="pt")
    ids = b["input_ids"][0].tolist()
    return ids, [ImageInput(ids.index(IMAGE_TOKEN), tuple(b["image_grid_thw"][0].tolist()), key, b["pixel_values"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-ref", metavar="DIR")
    a = ap.parse_args()
    if a.write_ref:
        return write_ref(a.write_ref)
    sys.path.insert(0, str(HERE))
    from _engine import make_engine
    from smk_runner import GenParams
    from test_vision_server import solid, words
    from transformers import AutoProcessor
    from vision import IMAGE_TOKEN

    ref = json.load(open(REF))["ids"]
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    t0 = time.time()
    r = make_engine(vision="device")
    print(f"engine ready in {time.time() - t0:.1f} s", flush=True)

    def same_as_ref(res, when):
        for k in ref:
            check(f"text {k} == 0.7.0 ({when})", res[k] == ref[k], f"{res[k] if k == 'turn2_cached' else len(res[k])} vs {ref[k] if k == 'turn2_cached' else len(ref[k])}")

    same_as_ref(text_cases(r, GenParams, tok), "before the image requests")
    q = "What word is written in this image? Reply with the word only."
    ids, ims = image_prompt(proc, q, words("TIGER", 1024, 768), "tiger")
    end = ims[0].offset + ims[0].n
    r.drop_cache()
    A, ca = gen(r, GenParams, ids, ims, n=16)
    B, cb = gen(r, GenParams, ids, ims, n=16)
    check("image request: reads the word", "tiger" in tok.decode(A).lower(), repr(tok.decode(A)))
    check("image request twice: same ids, the second resumed past the image", A == B and ca == 0 and cb >= end, f"cached {ca} / {cb}, image rows end at {end}")
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}, {"role": "assistant", "content": tok.decode(A, skip_special_tokens=True)},
         {"role": "user", "content": "How many letters does that word have? Answer with a number."}]
    b2 = proc(text=[render(tok, m)], images=[words("TIGER", 1024, 768)], return_tensors="pt")
    ids2 = b2["input_ids"][0].tolist()
    ims2 = [type(ims[0])(ids2.index(IMAGE_TOKEN), ims[0].grid, "tiger", b2["pixel_values"])]
    C, cc = gen(r, GenParams, ids2, ims2, n=16)
    r.drop_cache()
    D, cd = gen(r, GenParams, ids2, ims2, n=16)
    check("image conversation turn 2: resumed == cold, same ids", C == D and cc >= end and cd == 0, f"cached {cc} / {cd}, {tok.decode(C)!r}")
    ids3, ims3 = image_prompt(proc, q, solid((20, 40, 220), 1024, 768), "blue")
    gen(r, GenParams, ids, ims, n=4)  # the tiger prompt is the device lineage again
    E, ce = gen(r, GenParams, ids3, ims3, n=16)
    check("same text, another image: resumes nowhere past the image, another answer", ce <= ims3[0].offset and E != A, f"cached {ce}, image at {ims3[0].offset}, {tok.decode(E)!r}")
    same_as_ref(text_cases(r, GenParams, tok), "after the image requests")
    print(f"{sum(RES)}/{len(RES)} passed in {time.time() - t0:.1f} s", flush=True)
    sys.exit(0 if all(RES) else 1)


if __name__ == "__main__":
    main()
