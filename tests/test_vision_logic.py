#!/usr/bin/env python
"""Image-input logic that needs no GPU: the prefix-cache key with images, the server's parsing / rejection of multimodal features, and the start-up memory decisions of --vision.

  python tests/test_vision_logic.py

Tier: 0

1. vision.cache_key: text rows keep their ids; the placeholder rows of an image carry (negative) values from its hash, so two prompts that differ only in the image diverge exactly at its
   first placeholder row, the same image at the same place gives the same key, and a lineage / host-store match (prefix_store.common_prefix) stops there.
2. SwiftCoreClient._images on fake vLLM features: an image becomes ImageInput(offset, grid, mm hash, pixels); video, --vision off, an image above --max-image-pixels, a placeholder that does
   not match its grid and a feature without data are refused (400 / error); the receiver cache is updated before any refusal.
3. engine_cfg: --vision auto never changes what a text request gets (the tail decision and the length check are those of --vision off), puts the ViT weights on the GPU when 256 MiB stay
   free, else in host memory, else turns image input off; --vision device / host that do not fit are refused with a message naming image input.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
sys.path.insert(0, str(HERE.parent / "engine"))
from vision import IMAGE_TOKEN, ImageInput, cache_key  # noqa: E402

RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def test_key():
    from prefix_store import common_prefix

    ids = [1, 2, 3] + [IMAGE_TOKEN] * 16 + [4, 5, 6]
    a, b = ImageInput(3, (1, 8, 8), "hash-a"), ImageInput(3, (1, 8, 8), "hash-b")
    ka, kb, ka2 = cache_key(ids, [a]), cache_key(ids, [b]), cache_key(ids, [ImageInput(3, (1, 8, 8), "hash-a")])
    check("cache_key: text rows unchanged, placeholder rows negative", list(ka[:3]) == [1, 2, 3] and list(ka[19:]) == [4, 5, 6] and (ka[3:19] < 0).all())
    check("cache_key: two images, same ids -> keys diverge exactly at the first placeholder row", common_prefix(ka, kb) == 3)
    check("cache_key: the same image -> the same key (a follow-up turn resumes past it)", np.array_equal(ka, ka2))
    check("cache_key: a text-only prompt with the same ids shares only the text before the image", common_prefix(ka, np.asarray(ids, dtype=np.int32)) == 3)
    two = [1] + [IMAGE_TOKEN] * 4 + [2] + [IMAGE_TOKEN] * 4 + [3]
    k1 = cache_key(two, [ImageInput(1, (1, 4, 4), "x"), ImageInput(6, (1, 4, 4), "y")])
    k2 = cache_key(two, [ImageInput(1, (1, 4, 4), "x"), ImageInput(6, (1, 4, 4), "z")])
    check("cache_key: two images, the second differs -> diverge at the second image", common_prefix(k1, k2) == 6)


def feature(modality="image", off=5, grid=(1, 8, 8), data=True, length=None, ident="h1"):
    d = {"image_grid_thw": SimpleNamespace(data=torch.tensor(grid)), "pixel_values": SimpleNamespace(data=torch.zeros(grid[0] * grid[1] * grid[2], 1536))} if data else None
    return SimpleNamespace(modality=modality, data=d, identifier=ident, mm_position=SimpleNamespace(offset=off, length=grid[0] * grid[1] * grid[2] // 4 if length is None else length, is_embed=None))


def test_parse():
    from smk_core_client import SwiftCoreClient
    from vllm.exceptions import VLLMValidationError

    seen = []

    class Cache:
        def get_and_update_features(self, feats):
            seen.append(len(feats))
            return feats

    c = object.__new__(SwiftCoreClient)
    c.runner, c.mm_cache, c.max_patches = SimpleNamespace(vit=object()), Cache(), 1024

    def outcome(feats, vit=True):
        c.runner.vit = object() if vit else None
        try:
            return c._images(SimpleNamespace(mm_features=feats))
        except VLLMValidationError as e:
            return ("400", str(e))
        except RuntimeError as e:
            return ("error", str(e))

    ims = outcome([feature(off=7, grid=(1, 8, 12), ident="abc")])
    check("an image feature -> ImageInput(offset, grid, mm hash, pixels)", len(ims) == 1 and ims[0].offset == 7 and ims[0].grid == (1, 8, 12) and ims[0].key == "abc" and ims[0].n == 24
          and tuple(ims[0].pixels.shape) == (96, 1536))
    check("a text prompt -> no images", outcome([]) == [])
    r = outcome([feature("video")])
    check("video -> 400 naming video", r[0] == "400" and "video" in r[1], r[1])
    r = outcome([feature()], vit=False)
    check("image with image input off -> 400", r[0] == "400" and "off" in r[1], r[1])
    r = outcome([feature(grid=(1, 32, 34))])
    check("image above --max-image-pixels -> 400", r[0] == "400" and "max-image-pixels" in r[1], r[1])
    r = outcome([feature(length=10)])
    check("placeholder length != grid tokens -> 400", r[0] == "400" and "placeholder" in r[1], r[1])
    r = outcome([feature(data=False)])
    check("feature without data (processor cache out of step) -> error", r[0] == "error", r[1])
    n = len(seen)
    outcome([feature("video")])
    outcome([feature()], vit=False)
    check("the receiver cache sees every request, refused ones included", len(seen) == n + 2)


def cfg(free, L=20480, mode="auto", pl="auto", kv="auto", ht=False, mp=2097152):
    import smk_core_client as sc

    sc.OPTIONS.update(vision=mode, max_image_pixels=mp, prompt_lookup=pl, host_tables=ht, drafter="mtp")
    vc = SimpleNamespace(model_config=SimpleNamespace(max_model_len=L), speculative_config=SimpleNamespace(num_speculative_tokens=3), cache_config=SimpleNamespace(cache_dtype=kv))
    try:
        return sc.engine_cfg(vc, free)
    except RuntimeError as e:
        return str(e)


def test_memory():
    # the free memory of a 24 GB card next to a 0.6 GiB desktop (23,228 MiB), more, and less
    for free in (24500, 23228, 22900, 22300, 21500):
        off, auto = cfg(free, mode="off"), cfg(free)
        same = isinstance(off, str) and isinstance(auto, str) or (not isinstance(off, str) and not isinstance(auto, str) and off.pld == auto.pld and off.ctx_max == auto.ctx_max)
        check(f"--vision auto at {free} MiB free: the text configuration of --vision off (tail {getattr(off, 'pld', '-')}), image input -> {getattr(auto, 'vision', '-')}", same)
    check("--vision auto: device with room, host with less, off with none", (cfg(24500).vision, cfg(23228).vision, cfg(22300).vision) == ("device", "host", None),
          f"{cfg(24500).vision} {cfg(23228).vision} {cfg(22300).vision}")
    check("--vision host at the default length fits", not isinstance(cfg(23228, mode="host"), str))
    r = cfg(23228, mode="device")
    check("--vision device at the default length: the auto tail makes room for the demanded ViT (tail off, weights on the GPU)", not isinstance(r, str) and not r.pld and r.vision == "device")
    r = cfg(23228, mode="device", pl="1")
    check("--vision device + --prompt-lookup 1 that do not fit -> refused, the message names image input", isinstance(r, str) and "image input" in r, r[:160] if isinstance(r, str) else r)
    check("--vision off == 0.7.0's decision at the default length (tail on at 23,228 MiB)", cfg(23228, mode="off").pld and cfg(23228, mode="off").vision is None)


test_key()
test_parse()
test_memory()
print(f"{sum(RES)}/{len(RES)} passed", flush=True)
sys.exit(0 if all(RES) else 1)
