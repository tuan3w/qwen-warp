#!/usr/bin/env python
"""Image-input logic that needs no GPU: the prefix-cache key with images, the server's parsing / rejection of multimodal features, and the start-up memory decisions of --vision.

  python tests/test_vision_logic.py

Tier: 0

1. vision.cache_key: text rows keep their ids; the placeholder rows of an image carry (negative) values from its hash, so two prompts that differ only in the image diverge exactly at its
   first placeholder row, the same image at the same place gives the same key, and a lineage / host-store match (prefix_store.common_prefix) stops there.
2. SwiftCoreClient._images on fake vLLM features: an image becomes ImageInput(offset, grid, mm hash, pixels); video, --vision off, an image above --max-image-pixels, a placeholder that does
   not match its grid and a feature without data are refused (400 / error); install() makes the frontend's multimodal cache the processor-only one (no engine receiver cache).
3. engine_cfg: --vision auto never changes what a text request gets (the tail decision and the length check are those of --vision off), puts the ViT weights on the GPU when 256 MiB stay
   free, else in host memory, else turns image input off; --vision device / host that do not fit are refused with a message naming image input.
4. image input turned off by --vision auto (one sequence and --max-num-seqs > 1): the reason names settings that fit, and each named setting does turn image input on (the largest such
   --max-model-len exactly); the frontend's image-count check refuses an image with that reason (a 400 before a streaming response starts), video / text / image-on are untouched;
   without --max-model-len, --max-num-seqs > 1 picks a length that keeps image input on (the previous default turned it off at a large --max-image-pixels).
5. serve.py profiles: --profile multiuser = --max-num-seqs 5 --host-tables --vision device at a length the start-up check accepts (explicit flags win); no flags and --profile long keep
   one sequence and --vision auto.
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

    c = object.__new__(SwiftCoreClient)
    c.runner, c.max_patches = SimpleNamespace(vit=object()), 1024

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
    check("feature without data (a frontend cache that is not processor-only) -> error", r[0] == "error", r[1])
    # install(): the frontend's multimodal cache is the processor-only one (an "lru" sender / receiver pair drifts when the frontend refuses a processed image request)
    import smk_core_client as sc
    from vllm.multimodal import MULTIMODAL_REGISTRY

    sc.install()
    vc = SimpleNamespace(model_config=SimpleNamespace(get_multimodal_config=lambda: SimpleNamespace(mm_processor_cache_gb=4, mm_processor_cache_type="lru")),
                         parallel_config=SimpleNamespace(_api_process_count=1, data_parallel_size=1, data_parallel_external_lb=False))
    real = MULTIMODAL_REGISTRY.supports_multimodal_inputs
    MULTIMODAL_REGISTRY.supports_multimodal_inputs = lambda mc: True
    try:
        kinds = MULTIMODAL_REGISTRY._get_cache_type(vc), MULTIMODAL_REGISTRY.engine_receiver_cache_from_config(vc)
    finally:
        MULTIMODAL_REGISTRY.supports_multimodal_inputs = real
    check("install(): an lru processor cache becomes processor-only, no engine receiver cache", kinds == ("processor_only", None), str(kinds))


def cfg(free, L=20480, mode="auto", pl="auto", kv="auto", ht=False, mp=2097152, S=1):
    import smk_core_client as sc

    sc.OPTIONS.update(vision=mode, max_image_pixels=mp, prompt_lookup=pl, host_tables=ht, drafter="mtp")
    vc = SimpleNamespace(model_config=SimpleNamespace(max_model_len=L), speculative_config=SimpleNamespace(num_speculative_tokens=3), cache_config=SimpleNamespace(cache_dtype=kv),
                         scheduler_config=SimpleNamespace(max_num_seqs=S))
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


def test_vision_off():
    import re

    import smk_core_client as sc
    from vllm.exceptions import VLLMValidationError

    # --max-num-seqs 8, --profile long (int4 KV + host tables), --max-model-len 32768, and 100 MiB more free than the text needs: the ViT (156 MiB with host weights) does not fit
    S, L, dt = 8, 32768, 2
    free = int(sc.vram_need_batch_mib(L, dt, True, S) + sc.MIN_HEADROOM_MIB + 100)
    c = cfg(free, L, kv="int4_per_token_head", ht=True, S=S)
    msg = getattr(c, "vision_off", None) or ""
    m = re.search(r"--max-model-len ([\d,]+) or less at --max-num-seqs 8", msg)
    lf = int(m.group(1).replace(",", "")) if m else 0
    on_lf, on_next = (getattr(cfg(free, x, kv="int4_per_token_head", ht=True, S=S), "vision", None) for x in (lf, lf + 1024))
    ms = re.search(r"--max-num-seqs (\d+) at --max-model-len", msg)
    on_s = ms and getattr(cfg(free, L, kv="int4_per_token_head", ht=True, S=int(ms.group(1))), "vision", None)
    check("--max-num-seqs 8 at 32,768, ViT does not fit: image input off, the reason names the largest length that keeps it on and a slot count that does", c.vision is None and lf > 0
          and on_lf is not None and on_next is None and bool(on_s), msg)
    c1 = cfg(22300)
    m = re.search(r"--max-model-len ([\d,]+) or less", c1.vision_off or "")
    l1 = int(m.group(1).replace(",", "")) if m else 0
    check("one sequence, ViT does not fit at 20,480: the named length turns image input on (with the tail decision at that length)", c1.vision is None and l1 > 0
          and cfg(22300, l1).vision is not None and cfg(22300, l1 + 1024).vision is None, c1.vision_off)
    check("--vision off: the reason is --vision off; image input on: no reason", cfg(23228, mode="off").vision_off == "image input is off on this server (--vision off)"
          and cfg(24500).vision_off is None)
    # the frontend's image-count check (vLLM's chat parser calls processing_info.validate_num_items for every image item while it builds the prompt, before any response starts)
    sc.install()
    from vllm.multimodal.processing.context import BaseProcessingInfo

    class Info(BaseProcessingInfo):
        supported_mm_limits = {"image": None, "video": None}
        allowed_mm_limits = {"image": 4, "video": 0}

    info = object.__new__(Info)

    def count(modality, n):
        try:
            info.validate_num_items(modality, n)
            return "ok"
        except VLLMValidationError as e:
            return str(e)

    sc.IMAGE_OFF = msg
    off_image, off_video = count("image", 1), count("video", 1)
    sc.IMAGE_OFF = None
    on_image, on_five = count("image", 1), count("image", 5)
    check("image input off: an image item is refused with the reason (400 before streaming); video keeps vLLM's own refusal", off_image.startswith(msg) and "video" in off_video, off_image[:90])
    check("image input on: vLLM's own count check (1 image ok, 5 refused)", on_image == "ok" and "At most 4" in on_five, on_five)
    # without --max-model-len, --max-num-seqs > 1: the length leaves room for the ViT (8 megapixel images: 352 MiB with host weights, more than the 256 MiB the old default left)
    import serve

    mp = 8388608
    for free in (23700, 23228):
        real = torch.cuda.mem_get_info
        torch.cuda.mem_get_info = lambda device=None, free=free: (free << 20, 24 << 30)
        try:
            a = serve.parse_args(["--max-num-seqs", "8", "--profile", "long", "--max-image-pixels", str(mp)])
        finally:
            torch.cuda.mem_get_info = real
        old = sc.batch_max_len(free, dt, True, S, sc.AUTO_TAIL_HEADROOM_MIB)
        new_on = cfg(free, a.max_model_len, kv="int4_per_token_head", ht=True, S=S, mp=mp).vision
        old_on = cfg(free, old, kv="int4_per_token_head", ht=True, S=S, mp=mp).vision
        check(f"default length at {free} MiB free, 8 slots, --max-image-pixels {mp:,}: {a.max_model_len:,} keeps image input on ({new_on}); the old default {old:,} turned it off",
              new_on is not None and old_on is None and a.max_model_len >= old - 3072)


def test_profiles():
    """serve.parse_args: --profile multiuser = --max-num-seqs 5 --host-tables --vision device with a length that the start-up check accepts; explicit flags win; no flags and
    --profile long are what they were (one sequence, --vision auto)."""
    import serve
    import smk_core_client as sc

    free = 23228  # a 24 GB card next to a 0.6 GiB desktop
    real = torch.cuda.mem_get_info
    torch.cuda.mem_get_info = lambda device=None: (free << 20, 24 << 30)
    try:
        mu = serve.parse_args(["--profile", "multiuser"])
        mu2 = serve.parse_args(["--profile", "multiuser", "--max-num-seqs", "3", "--vision", "host", "--max-model-len", "8192"])
        d, lg = serve.parse_args([]), serve.parse_args(["--profile", "long"])
    finally:
        torch.cuda.mem_get_info = real
    c = cfg(free, mu.max_model_len, mode=mu.vision, ht=mu.host_tables, S=mu.max_num_seqs)
    check(f"--profile multiuser: 5 slots, host tables, ViT on the device, bf16 KV, {mu.max_model_len:,} tokens per slot at {free:,} MiB free, accepted by the start-up check",
          (mu.max_num_seqs, mu.host_tables, mu.vision, mu.kv_cache_dtype in (None, "auto")) == (5, True, "device", True) and mu.max_model_len >= 8192
          and not isinstance(c, str) and c.max_num_seqs == 5 and c.vision == "device", c if isinstance(c, str) else "")
    check("--profile multiuser: explicit --max-num-seqs / --vision / --max-model-len win", (mu2.max_num_seqs, mu2.vision, mu2.max_model_len, mu2.host_tables) == (3, "host", 8192, True))
    check("no flags: one sequence, --vision auto, device tables, the default length (as before)", (d.max_num_seqs, d.vision, d.host_tables, d.max_model_len) == (1, "auto", False, serve.DEFAULT_MAX_MODEL_LEN_PLD))
    check("--profile long: one sequence, --vision auto, int4 KV + host tables, 196,608 tokens (as before)",
          (lg.max_num_seqs, lg.vision, lg.host_tables, lg.kv_cache_dtype, lg.max_model_len) == (1, "auto", True, "int4_per_token_head", serve.LONG_MAX_MODEL_LEN))


test_key()
test_parse()
test_memory()
test_vision_off()
test_profiles()
print(f"{sum(RES)}/{len(RES)} passed", flush=True)
sys.exit(0 if all(RES) else 1)
