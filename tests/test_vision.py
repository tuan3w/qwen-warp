#!/usr/bin/env python
"""Image input pieces against their references.  Part 1 needs neither GPU nor checkpoint weights; part 2 needs an RTX 3090 and the checkpoint ($SWIFT_MODEL).

  python tests/test_vision.py [--no-gpu] [--time]

1. MRoPE positions (engine/vision.py mrope_positions) == vLLM's Qwen3VLForConditionalGeneration._get_mrope_input_positions (the function vLLM's Qwen3.5 model uses) for
   prompts with 0..3 images of several grids (square, wide, tall, tiny, at the start / end of the prompt), positions AND delta; and the kernel's interleaved section map
   (mrope_axis in engine/src/prefill.cu) == vLLM's apply_interleaved_rope for mrope_section [11, 11, 10].
2. ViT (engine/vision.py Vision.encode) against transformers' Qwen3_5VisionModel on the processor's pixel values of synthetic images of several sizes (one image per call, and three in
   one call): (a) the implementation: our fp32 forward == transformers' fp32 forward, per-token cosine >= 0.99999 (max abs diff reported); (b) the served bf16 path against the
   fp32 forward: mean cosine >= 0.9995 and the worst token no worse than transformers' own bf16 model (- 0.01): 27 bf16 blocks leave a few tokens at a cosine of 0.97-0.99 in
   both implementations, so a per-token 0.999 bar in bf16 is one that the reference itself misses.  --time: encode ms for 448x448, 1024x768, 1920x1080 (median of 10 after warmup).
"""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "engine"))
MODEL = os.environ.get("SWIFT_MODEL", os.path.expanduser("~/models/swift-1.5-int4"))


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    return bool(ok)


# ------------------------------------------------------------------------------------------------ 1. MRoPE
def mm_feature(off, grid):
    import torch

    return SimpleNamespace(mm_position=SimpleNamespace(offset=off), modality="image", data={"image_grid_thw": SimpleNamespace(data=torch.tensor(grid))})


def test_mrope():
    from transformers import AutoConfig
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vision import IMAGE_TOKEN, VISION_END, VISION_START, mrope_positions, n_tokens

    cfg = AutoConfig.from_pretrained(MODEL)
    ok = True
    cases = [
        [], [(1, 28, 28)], [(1, 4, 4)], [(1, 64, 48)], [(1, 34, 60)], [(1, 60, 34)], [(1, 2, 2)],
        [(1, 28, 28), (1, 16, 40)], [(1, 8, 8), (1, 64, 64), (1, 12, 6)],
    ]
    rng = np.random.default_rng(0)
    for grids in cases:
        for lead, trail in ((5, 9), (0, 7), (12, 0)):
            ids, imgs = list(rng.integers(100, 1000, size=lead)), []
            for g in grids:
                ids.append(VISION_START)
                imgs.append((len(ids), g))
                ids += [IMAGE_TOKEN] * n_tokens(g)
                ids += [VISION_END] + list(rng.integers(100, 1000, size=3))
            ids += list(rng.integers(100, 1000, size=trail))
            if imgs and not trail:  # a prompt that ends on the image placeholder itself
                ids = ids[: imgs[-1][0] + n_tokens(imgs[-1][1])]
            ref, rd = Qwen3VLForConditionalGeneration._get_mrope_input_positions(ids, [mm_feature(o, g) for o, g in imgs], cfg)
            pos, d = mrope_positions(len(ids), imgs)
            ok &= check(f"mrope {grids} lead {lead} trail {trail}", np.array_equal(ref.numpy(), pos) and rd == d, f"n {len(ids)} delta {d} (vLLM {rd})")
    # interleaved section map: vLLM's apply_interleaved_rope picks axis H for freq 1:33:3, W for 2:30:3, T otherwise
    import torch
    from vllm.model_executor.layers.rotary_embedding.mrope import apply_interleaved_rope

    sec = cfg.text_config.rope_parameters["mrope_section"]
    x = torch.stack([torch.full((1, 32), float(a)) for a in range(3)])
    ref = apply_interleaved_rope(x, sec)[0].long().tolist()
    mine = [mrope_axis(l) for l in range(32)]
    ok &= check("mrope interleaved section map == vLLM apply_interleaved_rope", ref == mine and sec == [11, 11, 10] and cfg.text_config.rope_parameters.get("mrope_interleaved"), str(mine))
    return ok


def mrope_axis(l):
    """Python copy of the kernel's mrope_axis (engine/src/prefill.cu): rotary frequency l -> 0 (T), 1 (H), 2 (W) for mrope_section [11, 11, 10], interleaved."""
    return 1 if l % 3 == 1 and l < 33 else 2 if l % 3 == 2 and l < 30 else 0


# ------------------------------------------------------------------------------------------------ 2. ViT
def synth(w, h, kind):
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (w, h), (30, 90, 200) if kind == 0 else (240, 240, 240))
    d = ImageDraw.Draw(im)
    d.ellipse([w * 0.1, h * 0.2, w * 0.5, h * 0.7], fill=(220, 30, 30))
    d.rectangle([w * 0.55, h * 0.1, w * 0.9, h * 0.5], fill=(20, 180, 40))
    d.text((w * 0.1, h * 0.8), "HELLO 42", fill=(0, 0, 0))
    return im


def test_vit(timing):
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from vision import Vision

    proc = AutoProcessor.from_pretrained(MODEL)
    vit = Vision()
    vit32 = Vision(dtype=torch.float32, attn="sdpa")
    print(f"ViT weights on the GPU: {vit.mib:.0f} MiB", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False  # the fp32 references are fp32
    vc = AutoConfig.from_pretrained(MODEL).vision_config
    vc._attn_implementation = "sdpa"
    sd = {k: (v.view(vit.hid, 3, vit.tp, vit.patch, vit.patch) if k == "patch_embed.proj.weight" else v) for k, v in vit32.W.items()}
    refs = {}
    ok = True
    for dt in (torch.float32, torch.bfloat16):
        with torch.device("cuda"):
            refs[dt] = Qwen3_5VisionModel._from_config(vc, dtype=dt).eval()
        missing, unexpected = refs[dt].load_state_dict({k: v.to(dt) for k, v in sd.items()}, strict=False)
        ok &= check(f"transformers reference ({dt}) loads every ViT tensor", not unexpected and not missing, f"missing {missing} unexpected {unexpected}")
    sizes = [(448, 448), (640, 480), (300, 500), (500, 300), (1024, 768), (1920, 1080), (64, 64)]
    cosf = torch.nn.functional.cosine_similarity

    def run(ims, tag):
        b = proc.image_processor(images=ims, return_tensors="pt")
        pv, g = b["pixel_values"], b["image_grid_thw"]
        with torch.inference_mode():
            r32 = refs[torch.float32](pv.cuda().float(), grid_thw=g.cuda()).pooler_output.float()
            r16 = refs[torch.bfloat16](pv.cuda().to(torch.bfloat16), grid_thw=g.cuda()).pooler_output.float()
            m32 = vit32.encode(pv, g).float()
            m16 = vit.encode(pv, g).float()
        c32, c16, ch = cosf(m32, r32, dim=-1), cosf(m16, r32, dim=-1), cosf(r16, r32, dim=-1)
        name = f"ViT {tag} grid {g.tolist()} -> {tuple(m16.shape)}"
        # (a) the implementation: fp32 forward == transformers' fp32 forward, every token
        r = check(f"{name}: fp32 == transformers fp32, per-token cosine >= 0.99999", c32.min() >= 0.99999, f"min {c32.min():.7f}, max abs diff {(m32 - r32).abs().max():.2e} (max |ref| {r32.abs().max():.0f})")
        # (b) the served bf16 path against the fp32 forward, next to transformers' own bf16 model (the bf16 noise floor of a 27-block ViT)
        r &= check(f"{name}: served bf16 vs fp32: mean cosine >= 0.9995, worst token no worse than transformers bf16 - 0.01", c16.mean() >= 0.9995 and c16.min() >= ch.min() - 0.01,
                   f"served bf16: mean {c16.mean():.6f} min {c16.min():.5f} max abs diff {(m16 - r32).abs().max():.3f} | transformers bf16: mean {ch.mean():.6f} min {ch.min():.5f}"
                   f" | served vs transformers bf16: min {cosf(m16, r16, dim=-1).min():.5f}")
        return r

    for i, (w, h) in enumerate(sizes):
        ok &= run([synth(w, h, i % 2)], f"{w}x{h}")
    ok &= run([synth(448, 448, 0), synth(640, 480, 1), synth(300, 500, 0)], "3 images in one call")
    del refs, vit32
    if timing:
        torch.cuda.empty_cache()
        for w, h in [(448, 448), (1024, 768), (1920, 1080)]:
            b = proc.image_processor(images=[synth(w, h, 0)], return_tensors="pt")
            pv, g = b["pixel_values"].pin_memory(), b["image_grid_thw"]
            for _ in range(3):
                vit.encode(pv, g)
            ts = []
            for _ in range(10):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                vit.encode(pv, g)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            vit.encode(pv, g)
            peak = (torch.cuda.max_memory_allocated() - base) / 2**20
            print(f"TIME  ViT encode {w}x{h} (grid {g[0].tolist()}, {pv.shape[0]} patches -> {pv.shape[0] // 4} tokens): median {np.median(ts):.1f} ms (min {min(ts):.1f}, n=10), activation peak {peak:.0f} MiB", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gpu", action="store_true", help="part 1 only")
    ap.add_argument("--time", action="store_true", help="also time the encoder")
    a = ap.parse_args()
    ok = test_mrope()
    if not a.no_gpu:
        ok &= test_vit(a.time)
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
