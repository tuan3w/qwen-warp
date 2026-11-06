"""swift-mk as a vLLM EngineCore: an in-process `EngineCoreClient` that vLLM's stock `AsyncLLM` (= the `EngineClient` behind the OpenAI frontend) drives.

AsyncLLM owns everything user-visible (input processing / chat template / tokenisation, detokenisation, stop strings, streaming deltas, usage, reasoning + tool parsers
in the serving layer); it only needs an engine core that accepts `EngineCoreRequest`s and returns `EngineCoreOutputs`.  This file is that core: a queue in front of
one worker thread that owns the swift-mk engine: with --max-num-seqs 1 a FIFO over smk_runner.Runner (one request at a time), with --max-num-seqs S > 1 the continuous-batching
scheduler of smk_batch.BatchRunner (S requests decoded together).  vLLM's `EngineCoreClient.make_async_mp_client` is replaced by `install()`.
"""
import asyncio
import collections
import math
import os
import queue
import threading
import time
from types import SimpleNamespace

import torch

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.metrics.stats import PrefillStats

import tool_schema
from smk_runner import GenParams, MAX_TOP_K, Runner, STOP
from vision import ImageInput, n_tokens

logger = init_logger("vllm.swift_mk")


LOGIT_BIAS_MAX = 1024  # logit_bias entries (the device table is dense; the request only uploads the sparse list)
VOCAB = 248320


def check_supported(sp):
    """Reject what the engine cannot do, with a 400 that names the parameter (instead of silently ignoring it); validate the limits of what it can."""
    if sp.top_k > MAX_TOP_K:
        raise VLLMValidationError(f"top_k must be <= {MAX_TOP_K} (the swift-mk sampler keeps at most the {MAX_TOP_K} highest logits; 0 or -1 = no top-k filter), got {sp.top_k}",
                                  parameter="top_k", value=sp.top_k)
    bad = []
    if sp.logprobs is not None:
        bad.append(("logprobs", sp.logprobs))
    if sp.prompt_logprobs is not None:
        bad.append(("prompt_logprobs", sp.prompt_logprobs))
    if sp.bad_words:
        bad.append(("bad_words", sp.bad_words))
    if sp.structured_outputs is not None:
        bad.append(("structured_outputs", "set"))
    if sp.thinking_token_budget is not None:
        bad.append(("thinking_token_budget", sp.thinking_token_budget))
    if sp.logprob_token_ids:
        bad.append(("logprob_token_ids", sp.logprob_token_ids))
    if bad:
        name, val = bad[0]
        raise VLLMValidationError(
            f"swift-mk engine does not support {', '.join(n for n, _ in bad)} (supported sampling: temperature, top_p, top_k <= 64, min_p, presence_penalty, frequency_penalty, repetition_penalty, "
            f"logit_bias, min_tokens, allowed_token_ids, seed, max_tokens, stop, stop_token_ids, ignore_eos, n)",
            parameter=name, value=val)
    lb = sp.logit_bias
    if lb:
        if len(lb) > LOGIT_BIAS_MAX:
            raise VLLMValidationError(f"logit_bias has {len(lb)} entries, at most {LOGIT_BIAS_MAX} are supported", parameter="logit_bias", value=len(lb))
        if any(not 0 <= t < VOCAB for t in lb) or any(not math.isfinite(v) for v in lb.values()):
            raise VLLMValidationError(f"logit_bias: token ids must be in [0, {VOCAB}) and values finite", parameter="logit_bias", value=len(lb))
    al = sp.allowed_token_ids
    if al:
        if len(al) > VOCAB or any(not 0 <= t < VOCAB for t in al):
            raise VLLMValidationError(f"allowed_token_ids: at most {VOCAB} ids, each in [0, {VOCAB})", parameter="allowed_token_ids", value=len(al))
        if sp.min_tokens and set(al) <= set(sp.all_stop_token_ids):
            raise VLLMValidationError("min_tokens masks the stop tokens, but every allowed_token_ids entry is a stop token: nothing could be generated", parameter="min_tokens", value=sp.min_tokens)


def to_gen_params(sp) -> GenParams:
    greedy = sp.temperature < 1e-5
    # InputProcessor has already folded generation_config's EOS ids into stop_token_ids and left eos_token_id None under ignore_eos (SamplingParams.update_from_generation_config)
    return GenParams(max_tokens=sp.max_tokens, eos=sp.eos_token_id, stop_ids=frozenset(sp.stop_token_ids or ()),
                     temperature=0.0 if greedy else sp.temperature, top_k=sp.top_k, top_p=sp.top_p, seed=sp.seed, min_p=sp.min_p,
                     presence_penalty=sp.presence_penalty, frequency_penalty=sp.frequency_penalty, repetition_penalty=sp.repetition_penalty, min_tokens=sp.min_tokens,
                     min_tokens_ids=frozenset(sp.all_stop_token_ids), logit_bias=dict(sp.logit_bias) if sp.logit_bias else None,
                     allowed_token_ids=tuple(dict.fromkeys(sp.allowed_token_ids)) if sp.allowed_token_ids else None)


OPTIONS = dict(stream_cycles=2, snapshots=16, prefix_cache_gb=6.0, system_checkpoint=True, prompt_lookup="auto", host_tables=False, sort_tool_schema=True, drafter="mtp", dflash_path=None, dflash_codebooks="host",
               vision="auto", max_image_pixels=2097152)
KV_DTYPES = {"auto": 0, "bfloat16": 0, "fp8": 1, "fp8_e4m3": 1, "int4_per_token_head": 2}  # engine kv dtype: 0 bf16, 1 fp8 (checkpoint scales), 2 int4 (per token+head, asymmetric, fp16 scale+min)
KV_NAMES = ("bf16", "fp8", "int4")

# GPU memory the engine needs after a request at --max-model-len (peak, MiB): VRAM_FIXED_MIB[host tables, prompt-lookup tail] + VRAM_KV_FIXED_MIB[kv dtype] + VRAM_KIB_PER_TOKEN[kv dtype] * --max-model-len.
# per token = the KV cache (64 / 32 / 16.5 KiB for bf16 / fp8 / int4; the int4 layout's fixed sink + ring buffers are VRAM_KV_FIXED_MIB) + 8 KiB that also grow with the length (the MTP layer's bf16 KV, the fast-prefill K/V scratch).
# Fitted to the 21 real server starts that bracket, for every cache layout, the largest length which starts AND serves a prompt of that length (RTX 3090, a desktop holding 0.6 GiB,
# 23228 MiB free at start): within +-40 MiB (about 1 K tokens at bf16) of each of them.  The fixed part is the weights, the activations, the 4-slot DeltaNet ring and the engine
# buffers; --host-tables moves the 4.7 GiB bf16 embedding + lm_head to pinned host memory; the tail adds the 9-slot ring and its cycle graphs.
# The server compares it with the memory that is free at start instead of trusting lengths measured next to some other desktop.
VRAM_FIXED_MIB = {(False, False): 20740, (False, True): 21312, (True, False): 15800, (True, True): 16488}
VRAM_KIB_PER_TOKEN = (72.0, 40.0, 24.5)
VRAM_KV_FIXED_MIB = (0, 0, 34)  # per kv dtype: the int4 cache's exact fp16 sink rows and recent-token ring (engine/src/kvrot.cuh) are a fixed device allocation, not per token (torch.cuda.mem_get_info: +34 MiB at 8 K and 64 K tokens)
MIN_HEADROOM_MIB = 64  # refuse to start with less than this predicted to be left after a request at --max-model-len (the prediction is good to +-40 MiB)
AUTO_TAIL_HEADROOM_MIB = 256  # `--prompt-lookup auto` only turns the tail on if at least this much stays free
MODEL_MAX_LEN = 262144

# --drafter dflash2 (engine/dflash.py): the engine always has the 9-slot ring (tmax = 8: the (host tables, tail) fixed term), plus the drafter: VRAM_DF_FIXED_MIB[codebooks on the host] = the W4A16 weights
# (1.0 GiB), the two 121 MiB candidate-selector codebooks (device only), the output ring + log, the tap staging buffers of the fast prefill (5 x 2047 rows x 10 KiB), the captured cycle graphs of both draft routes;
# VRAM_DF_KIB_PER_TOKEN = the drafter's own K/V cache over the whole context (absolute positions: 5 layers x 8 kv heads x 128 x (K, V) x bf16 = 20 KiB) + its token history (4 B).  MEASURED as the difference to
# the --drafter mtp Runner with the tail at the same length after warmup + a request at that length (RTX 3090, experiments/2026-11-06-dflash2-server/memfit.py): +1872 / +2352 MiB at 24,576 / 49,152 tokens
# (20.0 KiB per token), +3310 MiB at 98,304 with int4 KV, +2670 MiB at 65,536 with fp8 KV, +1628 MiB at 24,576 with the codebooks on the host.
VRAM_DF_FIXED_MIB = {False: 1400, True: 1150}
VRAM_DF_KIB_PER_TOKEN = 20.004

# Image input (engine/vision.py): the ViT's weights on the GPU (--vision device: all 879 MiB; host: 153 MiB, the 27 blocks' GEMM weights are streamed from pinned memory per encode) + what
# its activations add (one ViT call per image, so the number of images does not matter; the peak at 8,160 patches is 333 MiB, mostly reused from the fast prefill's transient pool).
# MEASURED as the difference of the free GPU memory after warmup + a largest-image request + a request at the full length between Runner(vision=...) and Runner(vision=None)
# (RTX 3090, 8,192 tokens, prompt-lookup tail on, experiments/2026-11-06-qwen-warp-concurrent-vision/memfit_vision.py): device +792 / +876 / +888 MiB at 1,024 / 4,096 / 8,100 patches,
# host +84 / +154 MiB at 2,024 / 8,100 patches.  The model below is at or above every measurement.
VRAM_VISION_FIXED_MIB = {"device": 880.0, "host": 90.0}
VRAM_VISION_MIB_PER_KPATCH = {"device": 1.0, "host": 8.0}
VISION_MODES = ("auto", "device", "host", "off")


def vision_mib(weights, max_image_pixels):
    """GPU memory of the image input with the ViT weights on `weights` ("device" | "host"; None = image input off) at --max-image-pixels."""
    if weights is None:
        return 0.0
    return VRAM_VISION_FIXED_MIB[weights] + VRAM_VISION_MIB_PER_KPATCH[weights] * max_patches(max_image_pixels) / 1000


def max_patches(max_image_pixels):
    """Patches (16 x 16 pixels) of the largest image the processor yields under --max-image-pixels (its resize keeps h * w <= max pixels)."""
    return max_image_pixels // 256


def vram_need_mib(ctx, dt, ht, pld, df=False, cb_host=False, vis=0.0):
    """vis: MiB of the image input (vision_mib)."""
    if df:  # the drafter's cycle verifies 8 rows: the ring of the tail is paid whether or not the MTP route uses the tail
        return vis + VRAM_FIXED_MIB[ht, True] + VRAM_DF_FIXED_MIB[bool(cb_host)] + VRAM_KV_FIXED_MIB[dt] + ctx * (VRAM_KIB_PER_TOKEN[dt] + VRAM_DF_KIB_PER_TOKEN) / 1024
    return vis + VRAM_FIXED_MIB[ht, bool(pld)] + VRAM_KV_FIXED_MIB[dt] + ctx * VRAM_KIB_PER_TOKEN[dt] / 1024


def vram_max_len(free_mib, dt, ht, pld, headroom=AUTO_TAIL_HEADROOM_MIB, df=False, cb_host=False, vis=0.0):
    """Largest --max-model-len (a multiple of 1024) whose predicted need leaves `headroom` MiB of `free_mib`; 0 if not even a short context fits."""
    per_token = VRAM_KIB_PER_TOKEN[dt] + (VRAM_DF_KIB_PER_TOKEN if df else 0.0)
    n = (free_mib - headroom - vram_need_mib(0, dt, ht, pld, df, cb_host, vis)) * 1024 / per_token
    return min(MODEL_MAX_LEN, max(0, int(n) // 1024 * 1024))


# --max-num-seqs S > 1 (engine/batch.py, server/smk_batch.py): S sequence slots on one set of weights.  Every slot has ONE in-place DeltaNet state (151 MiB, instead of the 5- or 9-slot ring of the
# single-sequence engine), a 9-slot conv ring, the lazy delta buffer, processor and spec state and its captured graphs (VRAM_BATCH_SLOT_MIB), and its own KV cache + MTP-layer KV at --max-model-len
# (VRAM_BATCH_KIB_PER_SLOT_TOKEN); the fast-prefill K/V scratch (4 KiB per token) and everything else (weights, tables, activations, prefill scratch, the batched cycle graphs) is shared
# (VRAM_BATCH_FIXED_MIB[host tables]).  The prompt-lookup tail costs no ring here (the conv ring has 9 slots anyway).  MEASURED with experiments/2026-11-06-qwen-warp-concurrent-vision memfit
# (least free GPU memory after the warm-up, a request at --max-model-len on one slot and short ones on the others, RTX 3090): S x len x KV = 2 x 8k x bf16 16,926 MiB with host tables / 21,762
# without, 4 x 8k bf16 18,750, 4 x 16k bf16 20,958, 4 x 32k int4 19,422 (host tables): the model below is within 8 MiB of every point (fp8: the same layout at 32 KiB, not measured).
VRAM_BATCH_FIXED_MIB = {False: 19920, True: 15080}
VRAM_BATCH_SLOT_MIB = 368
VRAM_BATCH_KIB_PER_SLOT_TOKEN = (68.0, 36.0, 20.5)
MAX_NUM_SEQS = 8  # sequence slots of a batched cycle (SMK_BATCH_MAX, include/batch.h)


def vram_need_batch_mib(ctx, dt, ht, S, vis=0.0):
    return vis + VRAM_BATCH_FIXED_MIB[ht] + S * (VRAM_BATCH_SLOT_MIB + VRAM_KV_FIXED_MIB[dt]) + ctx * (S * VRAM_BATCH_KIB_PER_SLOT_TOKEN[dt] + 4.0) / 1024


def batch_max_len(free_mib, dt, ht, S, headroom=MIN_HEADROOM_MIB, vis=0.0):
    """Largest --max-model-len (a multiple of 1024) per sequence for S slots that leaves `headroom` MiB of `free_mib`; 0 if none."""
    n = (free_mib - headroom - vram_need_batch_mib(0, dt, ht, S, vis)) * 1024 / (S * VRAM_BATCH_KIB_PER_SLOT_TOKEN[dt] + 4.0)
    return min(MODEL_MAX_LEN, max(0, int(n) // 1024 * 1024))


def batch_limits_text(free_mib, dt, ht, S, L, vis=0.0):
    """The largest --max-num-seqs at this --max-model-len and the largest --max-model-len at this --max-num-seqs (and at 2, 4, 8) for the memory free now."""
    room = free_mib - MIN_HEADROOM_MIB
    s_fit = max((s for s in range(1, MAX_NUM_SEQS + 1) if vram_need_batch_mib(L, dt, ht, s, vis) <= room), default=0)
    rows = [f"  at --max-model-len {L:,}: --max-num-seqs {s_fit}" + (" (1 = the single-sequence engine, which has its own memory model)" if s_fit <= 1 else "")]
    rows += [f"  at --max-num-seqs {s}: --max-model-len {batch_max_len(free_mib, dt, ht, s, vis=vis):,}" for s in sorted({S, 2, 4, 8}) if s > 1]
    return "\n".join(rows)


def vram_limits_text(free_mib, k, df=False, cb_host=False, vis=0.0):
    """One line per cache layout: the largest --max-model-len for the memory that is free now (with the tail, without it; --drafter dflash2: with the block drafter); vis: image input MiB."""
    rows = []
    for dt, ht, flags in ((0, False, "(default)"), (1, False, "--kv-cache-dtype fp8"), (2, False, "--kv-cache-dtype int4_per_token_head"), (2, True, "--profile long")):
        if df:
            rows.append(f"  {KV_NAMES[dt]} KV{' + host tables' if ht else ''} {flags}: {vram_max_len(free_mib, dt, ht, True, df=True, cb_host=cb_host, vis=vis):,} with --drafter dflash2")
            continue
        with_tail = f"{vram_max_len(free_mib, dt, ht, True, vis=vis):,} with the prompt-lookup tail, " if k == 3 else ""
        rows.append(f"  {KV_NAMES[dt]} KV{' + host tables' if ht else ''} {flags}: {with_tail}{vram_max_len(free_mib, dt, ht, False, vis=vis):,} without (--prompt-lookup 0)")
    if vis:
        rows.append(f"  (image input takes {vis:.0f} MiB of that at --max-image-pixels {OPTIONS['max_image_pixels']:,}; --vision off gives it back, a lower --max-image-pixels some of it)")
    return "\n".join(rows)


def free_gpu_mib():
    """GPU memory free now (MiB), capped by $SMK_FREE_MIB if set (tests: a start-up decision at a chosen free memory; the engine still allocates from the real GPU)."""
    free = torch.cuda.mem_get_info()[0] >> 20
    cap = os.environ.get("SMK_FREE_MIB", "")
    return min(free, int(cap)) if cap else free


def tail_on(L, dt, ht, k, pl, df, free_mib, vis=0.0):
    """The single-sequence engine's prompt-lookup tail decision (--prompt-lookup auto: on iff it leaves AUTO_TAIL_HEADROOM_MIB free next to the length and `vis`)."""
    if df:  # the 9-slot ring is paid by the drafter's verify anyway: the MTP route's tail is free
        return k == 3 and pl is not False
    return k == 3 and (vram_need_mib(L, dt, ht, True, vis=vis) <= free_mib - AUTO_TAIL_HEADROOM_MIB if pl == "auto" else bool(pl))


def auto_vision(L, dt, ht, S, pld, df, cbh, mp, free_mib):
    """--vision auto: "device" iff the ViT weights on the GPU leave AUTO_TAIL_HEADROOM_MIB free at this configuration, else "host" iff that fits (MIN_HEADROOM_MIB), else None (off)."""
    for w, hr in (("device", AUTO_TAIL_HEADROOM_MIB), ("host", MIN_HEADROOM_MIB)):
        vis = vision_mib(w, mp)
        if (vram_need_batch_mib(L, dt, ht, S, vis) if S > 1 else vram_need_mib(L, dt, ht, pld or df, df, cbh, vis)) <= free_mib - hr:
            return w
    return None


def vision_off_text(free_mib, L, dt, ht, S, k, pl, df, cbh, mp):
    """Why --vision auto left image input off and what would turn it on (the 400 of an image request and the start-up warning say the same)."""
    def on(L_, S_, pl_=pl):
        pld_ = (k == 3 and pl_ is not False) if S_ > 1 else tail_on(L_, dt, ht, k, pl_, df, free_mib)
        return auto_vision(L_, dt, ht, S_, pld_, df, cbh, mp, free_mib)

    lens = range(min(L, MODEL_MAX_LEN) // 1024 * 1024, 0, -1024)
    l_any = next((x for x in lens if on(x, S)), 0)
    l_dev = next((x for x in lens if on(x, S) == "device"), 0)
    opts = []
    if l_any:
        opts.append(f"--max-model-len {l_any:,} or less" + (f" at --max-num-seqs {S}" if S > 1 else "") + (f" ({l_dev:,} or less puts the ViT weights on the GPU, the faster encode)" if l_dev and l_dev < l_any else ""))
    s_fit = next((s for s in range(S - 1, 1, -1) if on(L, s)), 0) if S > 1 else 0
    if s_fit:
        opts.append(f"--max-num-seqs {s_fit} at --max-model-len {L:,}")
    if S == 1 and pl == "auto" and not df and on(L, 1, False):
        opts.append("--prompt-lookup 0 at this length")
    kv = f"{KV_NAMES[dt]} KV{' + host tables' if ht else ''}"
    what = f"{S} sequence slots at --max-model-len {L:,} with {kv} leave" if S > 1 else f"--max-model-len {L:,} with {kv} leaves"
    return (f"image input is off on this server: --vision auto found no GPU memory for the vision tower ({what} less than the {vision_mib('host', mp):.0f} MiB it needs with its "
            f"weights streamed from host memory; {free_mib:,} MiB were free at start).  Image input fits with " + "; or ".join(opts + [f"a lower --max-image-pixels (now {mp:,})"]))


IMAGE_OFF = None  # the reason image input is off on this server (None: on); set by SwiftCoreClient, read by the frontend's image-count check (install) and by _images


def engine_cfg(vllm_config, free_mib):
    """swift-mk engine settings from the stock vLLM config: --max-model-len, --kv-cache-dtype, --speculative-config (num_speculative_tokens), --max-num-seqs.  free_mib: GPU memory free at start."""
    mc, sc = vllm_config.model_config, vllm_config.speculative_config
    k = sc.num_speculative_tokens if sc is not None else 0
    if not 0 <= k <= 3:
        raise ValueError(f"swift-mk supports 0..3 speculative tokens (the MTP head drafts at most 3), got {k}")
    kv = vllm_config.cache_config.cache_dtype
    if kv not in KV_DTYPES:
        raise ValueError(f"--kv-cache-dtype {kv}: swift-mk supports auto/bfloat16 (bf16 KV), fp8 (e4m3 with the checkpoint k/v scales) and int4_per_token_head (4-bit codes + fp16 scale/min per token and head)")
    g = OPTIONS["stream_cycles"]
    pl = OPTIONS["prompt_lookup"]  # K11 hybrid draft (MTP chain + prompt-lookup tail, 9-slot ring): True / False / "auto" = on iff k == 3 and --max-model-len fits next to the ring
    dt, ht = KV_DTYPES[kv], bool(OPTIONS["host_tables"])
    df, cbh = OPTIONS["drafter"] == "dflash2", OPTIONS["dflash_codebooks"] == "host"
    L = mc.max_model_len
    mode, mp = OPTIONS["vision"], OPTIONS["max_image_pixels"]
    assert mode in VISION_MODES, mode
    # --vision auto never costs a text request anything: the tail decision ignores it, then the ViT weights go on the GPU iff that still leaves AUTO_TAIL_HEADROOM_MIB free at this length,
    # else in pinned host memory iff that fits (MIN_HEADROOM_MIB), else image input is off.  device / host are demands: the start is refused when they do not fit.
    wts = mode if mode in ("device", "host") else None
    vis = vision_mib(wts, mp)
    off = "image input is off on this server (--vision off)" if mode == "off" else None
    S = int(getattr(getattr(vllm_config, "scheduler_config", None), "max_num_seqs", 1) or 1)
    if S > 1:  # S sequence slots (smk_batch.BatchRunner): its own memory model; image input as for one sequence (auto: device weights iff 256 MiB stay free, else host, else off)
        if df:
            raise ValueError("swift-mk: --drafter dflash2 serves one sequence at a time: use --max-num-seqs 1 with it (or --drafter mtp, the batched decoder's drafter)")
        if S > MAX_NUM_SEQS:
            raise ValueError(f"swift-mk: --max-num-seqs {S}: at most {MAX_NUM_SEQS} sequences (the batched cycle's sequence slots)")
        pld = k == 3 and pl is not False
        if mode == "auto":
            wts = auto_vision(L, dt, ht, S, pld, df, cbh, mp, free_mib)
            vis = vision_mib(wts, mp)
            if wts is None:
                off = vision_off_text(free_mib, L, dt, ht, S, k, pl, df, cbh, mp)
                logger.warning("swift-mk: %s", off)
        vtxt = f" and image input ({vis:.0f} MiB with the ViT weights on the {wts})" if vis else ""
        need = vram_need_batch_mib(L, dt, ht, S, vis)
        if need > free_mib - MIN_HEADROOM_MIB:
            raise RuntimeError(f"swift-mk: --max-num-seqs {S} with --max-model-len {L:,} per sequence and {KV_NAMES[dt]} KV{' + host tables' if ht else ''}{vtxt} needs about {need:.0f} MiB of GPU memory "
                               f"but only {free_mib} MiB are free on the GPU (each sequence slot takes {VRAM_BATCH_SLOT_MIB} MiB + {VRAM_BATCH_KIB_PER_SLOT_TOKEN[dt]:.1f} KiB per token of --max-model-len).  "
                               f"What fits the free memory with {KV_NAMES[dt]} KV{' + host tables' if ht else ''}{vtxt} (--profile long = int4 KV + host tables needs the least):\n{batch_limits_text(free_mib, dt, ht, S, L, vis)}")
        return SimpleNamespace(ctx_max=L + Runner.slack(g, k, pld), kv_fp8=dt, host_tables=ht, k=k, group=g, snapshots=OPTIONS["snapshots"], prefix_cache_gb=OPTIONS["prefix_cache_gb"],
                               system_checkpoint=OPTIONS["system_checkpoint"], pld=pld, drafter="mtp", dflash_path=None, dflash_codebooks_host=cbh, max_num_seqs=S, vision=wts, vis_mib=vis,
                               vision_off=off, max_patches=max_patches(mp))
    pld = tail_on(L, dt, ht, k, pl, df, free_mib, vis)
    if mode == "auto":
        wts = auto_vision(L, dt, ht, 1, pld, df, cbh, mp, free_mib)
        vis = vision_mib(wts, mp)
        if wts is None:
            off = vision_off_text(free_mib, L, dt, ht, 1, k, pl, df, cbh, mp)
            logger.warning("swift-mk: %s", off)
    vtxt = f" and image input ({vis:.0f} MiB with the ViT weights on the {wts} at --max-image-pixels {mp:,})" if vis else ""
    if df:
        need = vram_need_mib(L, dt, ht, True, True, cbh, vis)
        if need > free_mib - MIN_HEADROOM_MIB:
            raise RuntimeError(f"swift-mk: --drafter dflash2 with --max-model-len {L:,} and {KV_NAMES[dt]} KV{' + host tables' if ht else ''}{vtxt} needs about {need:.0f} MiB of GPU memory "
                               f"but only {free_mib} MiB are free on the GPU (a desktop and other programs hold the rest; the drafter takes {VRAM_DF_FIXED_MIB[cbh]} MiB + {VRAM_DF_KIB_PER_TOKEN:.0f} KiB per context token).  "
                               f"Largest --max-model-len with --drafter dflash2 that fits the free memory ({'--host-tables is required for any useful length' if not ht else 'this layout'}):\n{vram_limits_text(free_mib, k, True, cbh, vis)}")
    else:
        if pl == "auto" and k == 3 and not pld:
            logger.info("swift-mk: prompt-lookup tail off: at --max-model-len %d with %s KV%s%s it needs %.0f MiB of the %d MiB free (it keeps %d MiB free at most); pass --prompt-lookup 1 to force it",
                        L, KV_NAMES[dt], " + host tables" if ht else "", vtxt, vram_need_mib(L, dt, ht, True, vis=vis), free_mib, AUTO_TAIL_HEADROOM_MIB)
        need = vram_need_mib(L, dt, ht, pld, vis=vis)
        if need > free_mib - MIN_HEADROOM_MIB:
            raise RuntimeError(f"swift-mk: --max-model-len {L:,} with {KV_NAMES[dt]} KV{' + host tables' if ht else ''}{' and the prompt-lookup tail' if pld else ''}{vtxt} needs about {need:.0f} MiB of GPU memory "
                               f"but only {free_mib} MiB are free on the GPU (a desktop and other programs hold the rest).  "
                               f"Largest --max-model-len that fits the free memory:\n{vram_limits_text(free_mib, k, vis=vis)}")
    return SimpleNamespace(ctx_max=L + Runner.slack(g, k, pld or df), kv_fp8=dt, host_tables=ht, k=k, group=g, snapshots=OPTIONS["snapshots"], prefix_cache_gb=OPTIONS["prefix_cache_gb"], system_checkpoint=OPTIONS["system_checkpoint"], pld=pld,
                           drafter="dflash2" if df else "mtp", dflash_path=OPTIONS["dflash_path"], dflash_codebooks_host=cbh, vision=wts, vis_mib=vis, vision_off=off, max_patches=max_patches(mp), max_num_seqs=1)


def warmup_grid(patches):
    """A (1, h, w) patch grid (h, w even) of at most `patches` patches and close to it: the Runner's warmup image (the ViT's activation peak grows with the patches)."""
    h = max(2, int(patches**0.5) // 2 * 2)
    return (1, h, max(2, patches // h // 2 * 2))


class SwiftCoreClient(EngineCoreClient):
    engine_ranks_managed = [0]

    def __init__(self, vllm_config, executor_class, log_stats, client_addresses=None, client_count=1, client_index=0, renderer=None):
        global IMAGE_OFF
        free_mib = free_gpu_mib()
        cfg = engine_cfg(vllm_config, free_mib)
        IMAGE_OFF = cfg.vision_off
        self.vllm_config = vllm_config
        self.loop = asyncio.get_running_loop()
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.req_q: queue.Queue = queue.Queue()
        self.live: set[str] = set()
        self.aborted: set[str] = set()
        self.resources = SimpleNamespace(engine_dead=False)
        t0 = time.time()
        self.max_num_seqs = S = cfg.max_num_seqs
        try:
            kw = dict(ctx_max=cfg.ctx_max, kv_fp8=cfg.kv_fp8, k=cfg.k, group=cfg.group, max_snaps=cfg.snapshots, log=logger.info, pld=cfg.pld, host_tables=cfg.host_tables, prefix_cache_gb=cfg.prefix_cache_gb,
                      system_checkpoint=cfg.system_checkpoint, drafter=cfg.drafter, dflash_path=cfg.dflash_path, dflash_codebooks_host=cfg.dflash_codebooks_host,
                      vision=cfg.vision, vision_warmup_grid=warmup_grid(cfg.max_patches))
            if S > 1:
                from smk_batch import BatchRunner

                self.runner = BatchRunner(max_num_seqs=S, **kw)
            else:
                self.runner = Runner(**kw)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "older than the sources" in str(e):
                raise
            fits = batch_limits_text(free_mib, cfg.kv_fp8, cfg.host_tables, S, vllm_config.model_config.max_model_len) if S > 1 else vram_limits_text(free_mib, cfg.k, cfg.drafter == 'dflash2', cfg.dflash_codebooks_host)
            raise RuntimeError(f"swift-mk engine ran out of GPU memory at --max-model-len {vllm_config.model_config.max_model_len:,}{f' x --max-num-seqs {S}' if S > 1 else ''} with {KV_NAMES[cfg.kv_fp8]} KV{' + host tables' if cfg.host_tables else ''}"
                               f"{' and the prompt-lookup tail' if cfg.pld else ''}{' and --drafter dflash2' if cfg.drafter == 'dflash2' else ''} ({free_mib} MiB were free at start, another program may have taken memory since).  "
                               f"What fits that memory:\n{batch_limits_text(free_mib, cfg.kv_fp8, cfg.host_tables, S, vllm_config.model_config.max_model_len, cfg.vis_mib) if S > 1 else vram_limits_text(free_mib, cfg.k, cfg.drafter == 'dflash2', cfg.dflash_codebooks_host, cfg.vis_mib)}\n{e}") from e
        logger.info("swift-mk engine ready in %.1f s (weights+MTP %.1f s, warmup %.1f s): ctx_max %d, k=%d, %d cycles per sync, kv %s%s, %d snapshots, %.1f GiB host prefix cache, prompt-lookup tail %s%s%s",
                    time.time() - t0, self.runner.load_s, self.runner.warmup_s, cfg.ctx_max, cfg.k, cfg.group, KV_NAMES[cfg.kv_fp8], " + host tables" if cfg.host_tables else "", cfg.snapshots, cfg.prefix_cache_gb, "on" if cfg.pld else "off",
                    f", drafter dflash2 (greedy requests without logit processors; {self.runner.df_mib:.0f} MiB device, codebooks on the {'host' if cfg.dflash_codebooks_host else 'device'})" if cfg.drafter == "dflash2" else "",
                    f", {S} sequence slots (batched decode)" if S > 1 else "")
        logger.info("swift-mk image input %s", f"on: ViT weights on the {cfg.vision} ({self.runner.vit.mib:.0f} MiB on the GPU), images up to {cfg.max_patches * 256:,} pixels "
                    f"({cfg.max_patches // 4} placeholder tokens), {cfg.vis_mib:.0f} MiB counted" if cfg.vision else f"off: {cfg.vision_off}")
        self.max_patches = cfg.max_patches
        self.thread = threading.Thread(target=self._batch_worker if S > 1 else self._worker, name="swift-mk-engine", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------------------------------------ engine thread
    def _emit(self, outs):
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, EngineCoreOutputs(outputs=outs))

    def _worker(self):
        while True:
            item = self.req_q.get()
            if item is None:
                return
            req, images = item
            rid = req.request_id
            try:
                if rid in self.aborted:
                    continue
                self._serve(req, images)
            except Exception:  # noqa: BLE001
                logger.exception("request %s failed", rid)
                self._emit([EngineCoreOutput(request_id=rid, new_token_ids=[], finish_reason=FinishReason.ERROR)])
                self.runner.drop_cache()  # the engine state is unknown after an exception
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001  sticky CUDA error: the engine is gone, /health must say so
                    logger.error("CUDA context is broken: marking the engine dead")
                    self.resources.engine_dead = True
            finally:
                self.live.discard(rid)
                self.aborted.discard(rid)

    def _serve(self, req, images):
        rid = req.request_id
        sp = req.sampling_params
        first = True
        n = 0
        for toks, fin, sr in self.runner.run(req.prompt_token_ids, to_gen_params(sp), cancelled=lambda: rid in self.aborted, images=images):
            n += len(toks)
            out = EngineCoreOutput(request_id=rid, new_token_ids=toks,
                                   finish_reason=None if fin is None else (FinishReason.STOP if fin == STOP else FinishReason.LENGTH), stop_reason=sr)
            if first:
                st = self.runner.last
                out.prefill_stats = PrefillStats(num_prompt_tokens=st.prompt, num_computed_tokens=st.prompt - st.cached, num_cached_tokens=st.cached, num_local_cached_tokens=st.cached)
                first = False
            self._emit([out])
        self._log_done(rid, self.runner.last)

    def _log_done(self, rid, st, slot=None):
        logger.info("%s: prompt %d (resumed at %d from %s) prefill %.3f s (spill %.0f ms, restore %.0f ms), %d tokens in %.2f s = %.1f tok/s, %.2f tok/cycle, snapshots %s, host cache %d lineages %.2f GiB%s%s%s", rid, st.prompt, st.cached, st.source, st.prefill_s,
                    st.spill_s * 1e3, st.restore_s * 1e3, st.tokens, st.gen_s, (st.tokens - 1) / st.gen_s if st.gen_s > 0 else 0.0, (st.tokens - 1) / st.cycles if st.cycles else 0.0, st.snaps,
                    len(self.runner.store.entries) if self.runner.store else 0, self.runner.store.bytes() / (1 << 30) if self.runner.store else 0.0,
                    f", route {st.route}" + (f" (accepted drafts per cycle {st.hist})" if st.route == "dflash2" else "") if self.runner.df else "",
                    f", slot {slot}" if slot is not None else "", f", {st.images} images ({st.vit_images} encoded in {st.vit_s * 1e3:.0f} ms)" if st.images else "")

    # --max-num-seqs > 1: continuous batching (smk_batch.BatchRunner.step: at most one admission, then one sync group of cycles for every running request).
    def _batch_worker(self):
        R = self.runner
        waiting = collections.deque()

        def refuse(t, ex):
            if isinstance(ex, ValueError):  # the prompt does not fit: this request fails, the others go on
                logger.warning("request %s refused: %s", t.rid, ex)
            self._emit([EngineCoreOutput(request_id=t.rid, new_token_ids=[], finish_reason=FinishReason.ERROR)])

        while True:
            try:
                block = not R.tasks and not waiting
                while True:
                    item = self.req_q.get(block=block)
                    if item is None:
                        return
                    waiting.append(self._task(*item))
                    block = False
            except queue.Empty:
                pass
            try:
                R.step(waiting, refuse)
            except Exception:  # noqa: BLE001
                logger.exception("batched decode failed: %d running requests fail", len(R.tasks))
                dead = [t.rid for t in R.tasks]
                R.fail_all()
                self._emit([EngineCoreOutput(request_id=rid, new_token_ids=[], finish_reason=FinishReason.ERROR) for rid in dict.fromkeys(dead)])
                for rid in dead:
                    self._forget(rid)
                R.drop_cache()  # the engine state is unknown after an exception
                try:
                    torch.cuda.synchronize()
                except Exception:  # noqa: BLE001  sticky CUDA error: the engine is gone, /health must say so
                    logger.error("CUDA context is broken: marking the engine dead")
                    self.resources.engine_dead = True
                    return

    def _forget(self, rid):
        self.live.discard(rid)
        self.aborted.discard(rid)

    def _task(self, req, images=None):
        from smk_batch import Task

        rid = req.request_id
        state = {"first": True}

        def emit(t, toks, fin, sr):
            out = EngineCoreOutput(request_id=rid, new_token_ids=toks, finish_reason=None if fin is None else (FinishReason.STOP if fin == STOP else FinishReason.LENGTH), stop_reason=sr)
            if state["first"]:
                st = t.st
                out.prefill_stats = PrefillStats(num_prompt_tokens=st.prompt, num_computed_tokens=st.prompt - st.cached, num_cached_tokens=st.cached, num_local_cached_tokens=st.cached)
                state["first"] = False
            self._emit([out])

        def done(t):
            if t.st is not None and not state["first"]:
                self._log_done(rid, t.st, t.seq.idx if t.seq is not None else None)
            self._forget(rid)

        return Task(rid, list(req.prompt_token_ids), to_gen_params(req.sampling_params), emit, done, cancelled=lambda: rid in self.aborted, images=images or None)

    def _images(self, request):
        """vision.ImageInput list of a request's multimodal features (validated); [] for a text prompt.  Every feature carries its data: the frontend's processor-only cache (install)
        never sends an image as already sent."""
        if not request.mm_features:
            return []
        feats = request.mm_features
        if self.runner.vit is None:  # (the frontend refuses such a request before it streams: install(); this is the engine's own check)
            raise VLLMValidationError(IMAGE_OFF or "image input is off on this server", parameter="messages")
        bad = sorted({f.modality for f in feats} - {"image"})
        if bad:
            raise VLLMValidationError(f"swift-mk serves image inputs only, not {', '.join(bad)}", parameter=bad[0])
        images = []
        for f in feats:
            if f.data is None:
                raise RuntimeError(f"multimodal item {f.identifier} arrived without its data (the frontend's processor cache is not the processor-only one: install())")
            grid = tuple(int(x) for x in f.data["image_grid_thw"].data.tolist())
            if grid[1] * grid[2] > self.max_patches:
                raise VLLMValidationError(f"image of {grid[1] * 16} x {grid[2] * 16} pixels after resizing is larger than --max-image-pixels {self.max_patches * 256:,}", parameter="mm_processor_kwargs")
            if f.mm_position.length != n_tokens(grid) or f.mm_position.is_embed is not None and not bool(f.mm_position.is_embed.all()):
                raise VLLMValidationError(f"image placeholder of {f.mm_position.length} tokens does not match its grid {grid}", parameter="messages")
            images.append(ImageInput(f.mm_position.offset, grid, f.identifier, f.data["pixel_values"].data))
        return images

    # ------------------------------------------------------------------------------------------------ EngineCoreClient
    async def add_request_async(self, request) -> None:
        if self.resources.engine_dead:
            raise RuntimeError("swift-mk engine thread is dead")
        if request.prompt_token_ids is None or request.prompt_embeds is not None:
            raise VLLMValidationError("swift-mk engine serves token prompts (text and images), not prompt_embeds")
        images = self._images(request)
        check_supported(request.sampling_params)
        self.live.add(request.request_id)
        self.req_q.put((request, images))

    async def abort_requests_async(self, request_ids) -> None:
        for rid in request_ids:
            if rid in self.live:
                self.aborted.add(rid)

    async def get_output_async(self) -> EngineCoreOutputs:
        return await self.out_q.get()

    async def get_supported_tasks_async(self):
        return ("generate",)

    async def reset_mm_cache_async(self) -> None:
        return None  # the frontend owns the only multimodal cache (processor-only, install()) and resets it itself

    async def reset_encoder_cache_async(self) -> None:
        return None

    async def reset_prefix_cache_async(self, reset_running_requests: bool = False, reset_connector: bool = False) -> bool:
        if self.live and not reset_running_requests:
            return False
        self.runner.drop_cache()
        return True

    async def is_sleeping_async(self) -> bool:
        return False

    async def is_scheduler_paused_async(self) -> bool:
        return False

    async def profile_async(self, is_start: bool = True, profile_prefix=None) -> None:
        raise NotImplementedError("swift-mk: no profiler")

    def dp_engines_running(self) -> bool:
        return False

    def shutdown(self, timeout=None) -> None:
        self.req_q.put(None)


def install(**options):
    """Make vLLM's AsyncLLM build SwiftCoreClient instead of spawning an EngineCore process."""
    OPTIONS.update(options)
    if OPTIONS["sort_tool_schema"]:
        tool_schema.install()

    def make(vllm_config, executor_class, log_stats, client_addresses=None, client_count=1, client_index=0, renderer=None):
        return SwiftCoreClient(vllm_config, executor_class, log_stats, client_addresses, client_count, client_index, renderer)

    EngineCoreClient.make_async_mp_client = staticmethod(make)
    # An image request to a server whose image input is off gets a 400 naming the reason and what would fit BEFORE its response starts (a streaming request would otherwise get a 200
    # and an error event): the chat parser counts every image item through the model's processing info while it builds the prompt, so the refusal goes there.
    from vllm.multimodal.processing.context import BaseProcessingInfo

    count = BaseProcessingInfo.validate_num_items
    if not getattr(count, "swift_mk", False):
        def validate_num_items(self, modality, num_items):
            if modality == "image" and IMAGE_OFF:
                raise VLLMValidationError(IMAGE_OFF, parameter="image")
            return count(self, modality, num_items)

        validate_num_items.swift_mk = True
        BaseProcessingInfo.validate_num_items = validate_num_items
    # The engine runs in the frontend's process, so the processed images need no second cache: vLLM's "lru" pair (the frontend keeps metadata and sends data=None for an item it already
    # sent; the engine keeps the data) drifts when the frontend refuses a request AFTER processing its image (a prompt over --max-model-len): the image counts as sent, never reaches the
    # engine, and every later request with it failed (HTTP 500).  The processor-only cache keeps the processed items in the frontend and always sends the data.
    from vllm.multimodal.registry import MultiModalRegistry

    cache_type = MultiModalRegistry._get_cache_type
    if not getattr(cache_type, "swift_mk", False):
        def _get_cache_type(self, vllm_config):
            t = cache_type(self, vllm_config)
            return "processor_only" if t == "lru" else t

        _get_cache_type.swift_mk = True
        MultiModalRegistry._get_cache_type = _get_cache_type
    # vLLM refuses min_p and logit_bias whenever a speculative config is set (SamplingParams._validate_spec_decode): a limit of ITS rejection sampler, which applies neither to the verify rows
    # (v1/sample/rejection_sampler.py apply_logits_processors).  This engine's cycle applies every supported processor to every verify row, exactly (engine/src/proc.h), so the check goes.
    SamplingParams._validate_spec_decode = lambda self, speculative_config: None
