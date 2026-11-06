"""Batched decode of image and text sequences together, against exact solo references, on the GPU (engine level: BatchRunner(vision="device") without the vLLM server).

Tier: 1

Same oracle as tests/test_batch_engine.py (no tolerance: a request decoded in a batch == the same request decoded alone from a cold prefill with the same decode calls), here
with sequences whose decode ropes at KV position + a nonzero MRoPE delta (ds[DS_RDELTA], per slot, gathered into the batch) next to text sequences (delta 0):
  B. every batch size B = 1..4 x every draft depth with B (k + 1) <= 16 (up to 16 rows per cycle), each batch mixing image prompts (one to two images, 70..300 placeholder rows) and text prompts,
     greedy / seeded sampling / min_p / logit_bias + penalties
  H. a batch of an image follow-up turn resumed from the HOST prefix store past its image (its slot was overwritten by a text request in between), an image follow-up turn
     resumed from the slot's device snapshot, and a cold text prompt; batched cycles of every verified-draft count
  C. continuous-batching fuzz through BatchRunner.step: image and text requests, image follow-up turns (device / host resumes past the image), aborts, max_tokens
  T. text-only ids of an S = 4 BatchRunner == 0.7.0's Runner (tests/data/text_ids_v070.json, code / prose / math greedy, seeded sample, two-turn resumed) for a request alone in the
     default mode (the native single-sequence path), and in the exact mode (every cycle batched with k = 0 / 1 / 3 drafts) for every greedy case at every k and the seeded sample
     at k = 3 (a seeded sample's draws depend on the draft depth, so 0.7.0's sample is no reference at k < 3; a greedy case would differ only at a bf16 near-tie, none in these 64 tokens)
  D. SMK_DEBUG=1 canaries of every slot intact
Solo references are cached under ~/.cache/qwen-warp/batch_solo (tests/test_batch_engine.py).  SMK_RESIDENT=1: resident weights (tests/_engine.py).
  python tests/test_batch_vision_engine.py [--fuzz 36] [--slots 4]
"""
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_batch_engine as TB  # noqa: E402  (sets SMK_DEBUG=1 before the engine library loads)
from _engine import make_engine  # noqa: E402

import smk_batch as SB  # noqa: E402
from smk_runner import GenParams  # noqa: E402

EOS, IM_END = TB.EOS, 248044
FOLLOW = "<|im_end|>\n<|im_start|>user\nTell me one more detail about it, briefly.<|im_end|>\n<|im_start|>assistant\n"


def image_prompt(proc, text, pics, keys, system=None):
    """(ids, [ImageInput]) of a one-turn chat with the images before the text."""
    from test_vision_engine import render
    from vision import IMAGE_TOKEN, ImageInput

    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": [{"type": "image"} for _ in pics] + [{"type": "text", "text": text}]}]
    b = proc(text=[render(proc.tokenizer, msgs)], images=pics, return_tensors="pt")
    ids = b["input_ids"][0].tolist()
    ims, at, row = [], 0, 0
    for g, key in zip(b["image_grid_thw"].tolist(), keys):
        n = g[0] * g[1] * g[2]
        im = ImageInput(ids.index(IMAGE_TOKEN, at), tuple(g), key, b["pixel_values"][row : row + n])
        ims.append(im)
        at, row = im.offset + im.n, row + n
    return ids, ims


def img_end(ims):
    return max(im.offset + im.n for im in ims)


class ViaScheduler:
    """The Runner.run interface of test_vision_engine.text_cases, served by BatchRunner.step (one request at a time: admission, slot choice, the scheduler's decode path)."""

    def __init__(self, r):
        self.r, self.last = r, None

    def run(self, ids, p, images=None):
        out, w = [], deque()
        t = SB.Task(0, ids, p, lambda t, toks, fin, sr: out.extend(toks), lambda t: None, images=images)
        w.append(t)
        while w or self.r.tasks:
            self.r.step(w, lambda t, ex: (_ for _ in ()).throw(ex))
        self.last = t.st
        yield out, "stop", None


# ---------------------------------------------------------------------------------------------------------------------------------------------- H
def part_h(r, tk, conv_a, conv_b, text):
    """conv_a / conv_b: (ids, images) of image prompts longer than w4_max rows (the prefill snapshots the end of the prompt); text: a text prompt."""
    r.drop_cache()
    q0, q1, q2 = r.seqs[:3]
    greedy = GenParams(max_tokens=24, eos=EOS)
    turn1 = {}
    for q, (ids, ims) in ((q0, conv_a), (q1, conv_b)):
        r.seq = q
        out = []
        for toks, _, _ in r.run(ids, greedy, images=ims):
            out += toks
        turn1[id(q)] = out
    r.seq = q0
    for _ in r.run(text[0], GenParams(max_tokens=4, eos=EOS)):  # a cold text request on slot 0: its image lineage is spilled to the host store
        pass
    jobs = [(q0, conv_a[0] + turn1[id(q0)] + tk.encode(FOLLOW), conv_a[1], GenParams(max_tokens=64, eos=EOS, temperature=0.8, top_k=20, top_p=0.95, seed=31), "host"),
            (q1, conv_b[0] + turn1[id(q1)] + tk.encode(FOLLOW), conv_b[1], GenParams(max_tokens=64, eos=EOS), "device"),
            (q2, text[0], None, GenParams(max_tokens=64, eos=EOS, temperature=0.6, seed=32, min_p=0.05), "cold")]
    rec = []
    for q, ids, ims, p, src in jobs:
        r.seq = q
        b = r._begin(ids, p, images=ims)
        if ims:
            assert b.st.source == src and b.q >= img_end(ims), f"H: expected a {src} resume past the image (rows < {img_end(ims)}), got {b.st.source} at {b.q}"
        try:
            assert r._prefill(b.ids, b.q, b.ends, b.folded, lambda: False, False, b.key)
        finally:
            q.fp.mm = None
        q.e._emitted = r._ss()[12]
        q.e.drafted = r.k
        rec.append(dict(ids=b.ids, ims=ims, p=p, src=b.st.source, cached=b.q, toks=[q.e.read_out_tok()], calls=[]))
    es = [q.e for q, *_ in jobs]
    for call in [(1, 1, 2), (1, 1, 3), (1, 0, 2), (0, 1, 2), (1, 1, 2), (1, 1, 3), (1, 1, 2), (1, 0, 1), (0, 0, 2), (0, 1, 2), (1, 1, 3)]:
        news, _ = r.bd.cycles(es, *call)
        for x, new in zip(rec, news):
            x["toks"] += new
            x["calls"].append(("batch",) + call)
    for x in rec:
        ref = TB.solo(r, x["ids"], x["p"], x["calls"], x["ims"])
        assert x["toks"] == ref[: len(x["toks"])], f"H: {x['src']} (cached {x['cached']}, images {bool(x['ims'])}): batched != cold solo at {TB.first_diff(x['toks'], ref)}"
    print(f"H: B = 3 batch of an image turn 2 resumed from the host store (cached {rec[0]['cached']}), one from the device (cached {rec[1]['cached']}), a cold text prompt: "
          f"{[len(x['toks']) for x in rec]} tokens identical to cold solo replays", flush=True)


# ---------------------------------------------------------------------------------------------------------------------------------------------- T
def part_t(r, tk):
    from test_vision_engine import REF, text_cases

    ref = json.load(open(REF))["ids"]
    via = ViaScheduler(r)
    r.drop_cache()
    res = text_cases(via, GenParams, tk)
    bad = {k: TB.first_diff(res[k], ref[k]) if isinstance(ref[k], list) else (res[k], ref[k]) for k in ref if res[k] != ref[k]}
    assert not bad, f"T: text ids of a request alone at S = {len(r.seqs)} (default mode) != 0.7.0: {bad}"
    exact, bad = {}, []
    for k in (0, 1, 3):
        r.exact_k = k
        r.drop_cache()
        try:
            got = text_cases(via, GenParams, tk)
        finally:
            r.exact_k = SB.EXACT_K
        exact[k] = {n: ("==" if got[n] == ref[n] else (f"diff@{TB.first_diff(got[n], ref[n])}" if isinstance(ref[n], list) else f"{got[n]} vs {ref[n]}")) for n in ref}
        # where 0.7.0's ids apply: every greedy case at every depth (no near-tie in these 64 tokens), the seeded sample at k = 3 only (its draws depend on the draft depth)
        bad += [(k, n) for n in ref if exact[k][n] != "==" and (n != "sampled" or k == 3)]
    print(f"T: text ids alone at S = {len(r.seqs)}, default mode == 0.7.0 ({', '.join(ref)}); exact mode SMK_BATCH_K = k vs 0.7.0: {exact}", flush=True)
    assert not bad, f"T: exact-mode text ids != 0.7.0 at (k, case) {bad}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=36)
    ap.add_argument("--slots", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    from test_vision_engine import MODEL
    from test_vision_server import solid, words
    from transformers import AutoProcessor

    r = make_engine(max_num_seqs=a.slots, vision="device", host_tables=True, prefix_cache_gb=0.5, max_snaps=12, min_spill=128)  # ctx 8224: the configuration of text_ids_v070.json
    print(f"engine ({a.slots} slots, ViT on the device) in {time.time() - t0:.1f} s", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    tk = proc.tokenizer
    txt = [(TB.chat_ids(tk, [{"role": "user", "content": m}]), None) for m in TB.MSGS]
    pics = {"tiger": words("TIGER", 448, 448), "red": solid((220, 20, 20), 320, 224), "blue": solid((20, 40, 220), 224, 448), "moon": words("MOON", 640, 480),
            "lamp": words("LAMP", 640, 480), "green": solid((20, 180, 40), 448, 320)}
    img = [image_prompt(proc, "What word is written in this image?", [pics["tiger"]], ["tiger"]),
           image_prompt(proc, "What colour is this image? One word.", [pics["red"]], ["red"]),
           image_prompt(proc, "Describe this image in one sentence.", [pics["moon"]], ["moon"]),
           image_prompt(proc, "Which of the two images is bluer, and why?", [pics["red"], pics["blue"]], ["red", "blue"]),
           image_prompt(proc, "Describe the colour of this picture.", [pics["green"]], ["green"]),
           image_prompt(proc, "Read the text in the image and use it in a sentence.", [pics["lamp"]], ["lamp"])]
    print(f"image prompts: {[(len(i), [im.n for im in ims]) for i, ims in img]} (tokens, placeholder rows)", flush=True)
    mixed = [x for pair in zip(img, txt) for x in pair]  # image, text, image, text, ...: every batch of consecutive entries mixes both
    TB.part_b(r, mixed)
    part_h(r, tk, img[2], img[5], txt[3])
    sysmsg = " ".join(f"Rule {i}: answer the user's question accurately, keep it short, and never invent facts about item {i}." for i in range(30))
    long_img = [image_prompt(proc, q, [pics[k]], [k], system=sysmsg) for q, k in (("What word is written here?", "tiger"), ("Name the colour.", "red"), ("What is shown?", "moon"))]
    long_txt = [(TB.chat_ids(tk, [{"role": "system", "content": sysmsg}, {"role": "user", "content": m}]), None) for m in TB.MSGS[:3]]
    r.drop_cache()
    reqs = TB.part_c(r, tk, mixed[:6], long_img + long_txt, a.fuzz, seed=12)  # seed 12: 26 image requests, image follow-ups resumed from the device and from the host store
    resumed = [x["task"].st.source for x in reqs if x["task"].images and x["task"].st is not None and x["task"].st.source != "cold" and x["task"].st.cached >= img_end(x["task"].images)]
    assert "device" in resumed and "host" in resumed, f"C: image requests resumed past the image from {resumed}: the fuzz must cover device and host resumes of image prompts"
    print(f"C: image requests resumed past the image: {resumed.count('device')} from the device, {resumed.count('host')} from the host store", flush=True)
    part_t(r, tk)
    bad = [int(r.e.lib.smk_engine_canary_check(q.e.h)) for q in r.seqs]
    assert all(b == 0 for b in bad), f"D: canaries overwritten per slot {bad} (-1 = SMK_DEBUG off)"
    print(f"D: canaries intact after every device buffer of the {len(r.seqs)} slots and the batch buffers", flush=True)
    print(f"PASS in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
