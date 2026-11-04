"""swift-mk as a vLLM EngineCore: an in-process `EngineCoreClient` that vLLM's stock `AsyncLLM` (= the `EngineClient` behind the OpenAI frontend) drives.

AsyncLLM owns everything user-visible (input processing / chat template / tokenisation, detokenisation, stop strings, streaming deltas, usage, reasoning + tool parsers
in the serving layer); it only needs an engine core that accepts `EngineCoreRequest`s and returns `EngineCoreOutputs`.  This file is that core: a FIFO queue in front of
one worker thread that owns the swift-mk engine (smk_runner.Runner), vLLM's `EngineCoreClient.make_async_mp_client` is replaced by `install()`.
"""
import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import torch

from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.metrics.stats import PrefillStats

from smk_runner import GenParams, Runner, STOP

logger = init_logger("vllm.swift_mk")


def check_supported(sp):
    """Reject what the engine cannot do, with a 400 that names the parameter (instead of silently ignoring it)."""
    bad = []
    if sp.logprobs is not None:
        bad.append(("logprobs", sp.logprobs))
    if sp.prompt_logprobs is not None:
        bad.append(("prompt_logprobs", sp.prompt_logprobs))
    if sp.presence_penalty:
        bad.append(("presence_penalty", sp.presence_penalty))
    if sp.frequency_penalty:
        bad.append(("frequency_penalty", sp.frequency_penalty))
    if sp.repetition_penalty != 1.0:
        bad.append(("repetition_penalty", sp.repetition_penalty))
    if sp.min_p:
        bad.append(("min_p", sp.min_p))
    if sp.min_tokens:
        bad.append(("min_tokens", sp.min_tokens))
    if sp.logit_bias:
        bad.append(("logit_bias", sp.logit_bias))
    if sp.allowed_token_ids:
        bad.append(("allowed_token_ids", sp.allowed_token_ids))
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
            f"swift-mk engine does not support {', '.join(n for n, _ in bad)} (supported sampling: temperature, top_p, top_k <= 64, seed, max_tokens, stop, stop_token_ids, ignore_eos, n)",
            parameter=name, value=val)


def to_gen_params(sp) -> GenParams:
    greedy = sp.temperature < 1e-5
    # InputProcessor has already folded generation_config's EOS ids into stop_token_ids and left eos_token_id None under ignore_eos (SamplingParams.update_from_generation_config)
    return GenParams(max_tokens=sp.max_tokens, eos=sp.eos_token_id, stop_ids=frozenset(sp.stop_token_ids or ()),
                     temperature=0.0 if greedy else sp.temperature, top_k=sp.top_k, top_p=sp.top_p, seed=sp.seed)


OPTIONS = dict(stream_cycles=2, snapshots=4, prompt_lookup="auto", host_tables=False)
KV_DTYPES = {"auto": 0, "bfloat16": 0, "fp8": 1, "fp8_e4m3": 1, "int4_per_token_head": 2}  # engine kv dtype: 0 bf16, 1 fp8 (checkpoint scales), 2 int4 (per token+head, asymmetric, fp16 scale+min)
KV_NAMES = ("bf16", "fp8", "int4")

# GPU memory the engine needs after a request at --max-model-len (peak, MiB): VRAM_FIXED_MIB[host tables, prompt-lookup tail] + VRAM_KIB_PER_TOKEN[kv dtype] * --max-model-len.
# per token = the KV cache (64 / 32 / 16.5 KiB for bf16 / fp8 / int4) + 8 KiB that also grow with the length (the MTP layer's bf16 KV, the fast-prefill K/V scratch).
# Fitted to the 21 real server starts that bracket, for every cache layout, the largest length which starts AND serves a prompt of that length (RTX 3090, a desktop holding 0.6 GiB,
# 23228 MiB free at start): within +-40 MiB (about 1 K tokens at bf16) of each of them.  The fixed part is the weights, the activations, the 4-slot DeltaNet ring and the engine
# buffers; --host-tables moves the 4.7 GiB bf16 embedding + lm_head to pinned host memory; the tail adds the 9-slot ring and its cycle graphs.
# The server compares it with the memory that is free at start instead of trusting lengths measured next to some other desktop.
VRAM_FIXED_MIB = {(False, False): 20740, (False, True): 21312, (True, False): 15800, (True, True): 16488}
VRAM_KIB_PER_TOKEN = (72.0, 40.0, 24.5)
MIN_HEADROOM_MIB = 64  # refuse to start with less than this predicted to be left after a request at --max-model-len (the prediction is good to +-40 MiB)
AUTO_TAIL_HEADROOM_MIB = 256  # `--prompt-lookup auto` only turns the tail on if at least this much stays free
MODEL_MAX_LEN = 262144


def vram_need_mib(ctx, dt, ht, pld):
    return VRAM_FIXED_MIB[ht, bool(pld)] + ctx * VRAM_KIB_PER_TOKEN[dt] / 1024


def vram_max_len(free_mib, dt, ht, pld, headroom=AUTO_TAIL_HEADROOM_MIB):
    """Largest --max-model-len (a multiple of 1024) whose predicted need leaves `headroom` MiB of `free_mib`; 0 if not even a short context fits."""
    n = (free_mib - headroom - vram_need_mib(0, dt, ht, pld)) * 1024 / VRAM_KIB_PER_TOKEN[dt]
    return min(MODEL_MAX_LEN, max(0, int(n) // 1024 * 1024))


def vram_limits_text(free_mib, k):
    """One line per cache layout: the largest --max-model-len for the memory that is free now (with the tail, without it)."""
    rows = []
    for dt, ht, flags in ((0, False, "(default)"), (1, False, "--kv-cache-dtype fp8"), (2, False, "--kv-cache-dtype int4_per_token_head"), (2, True, "--profile long")):
        with_tail = f"{vram_max_len(free_mib, dt, ht, True):,} with the prompt-lookup tail, " if k == 3 else ""
        rows.append(f"  {KV_NAMES[dt]} KV{' + host tables' if ht else ''} {flags}: {with_tail}{vram_max_len(free_mib, dt, ht, False):,} without (--prompt-lookup 0)")
    return "\n".join(rows)


def engine_cfg(vllm_config, free_mib):
    """swift-mk engine settings from the stock vLLM config: --max-model-len, --kv-cache-dtype, --speculative-config (num_speculative_tokens).  free_mib: GPU memory free at start."""
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
    L = mc.max_model_len
    pld = k == 3 and (vram_need_mib(L, dt, ht, True) <= free_mib - AUTO_TAIL_HEADROOM_MIB if pl == "auto" else bool(pl))
    if pl == "auto" and k == 3 and not pld:
        logger.info("swift-mk: prompt-lookup tail off: at --max-model-len %d with %s KV%s it needs %.0f MiB of the %d MiB free (it keeps %d MiB free at most); pass --prompt-lookup 1 to force it",
                    L, KV_NAMES[dt], " + host tables" if ht else "", vram_need_mib(L, dt, ht, True), free_mib, AUTO_TAIL_HEADROOM_MIB)
    need = vram_need_mib(L, dt, ht, pld)
    if need > free_mib - MIN_HEADROOM_MIB:
        raise RuntimeError(f"swift-mk: --max-model-len {L:,} with {KV_NAMES[dt]} KV{' + host tables' if ht else ''}{' and the prompt-lookup tail' if pld else ''} needs about {need:.0f} MiB of GPU memory "
                           f"but only {free_mib} MiB are free on the GPU (a desktop and other programs hold the rest).  "
                           f"Largest --max-model-len that fits the free memory:\n{vram_limits_text(free_mib, k)}")
    return SimpleNamespace(ctx_max=L + Runner.slack(g, k, pld), kv_fp8=dt, host_tables=ht, k=k, group=g, snapshots=OPTIONS["snapshots"], pld=pld)


class SwiftCoreClient(EngineCoreClient):
    engine_ranks_managed = [0]

    def __init__(self, vllm_config, executor_class, log_stats, client_addresses=None, client_count=1, client_index=0, renderer=None):
        free_mib = torch.cuda.mem_get_info()[0] >> 20
        cfg = engine_cfg(vllm_config, free_mib)
        self.vllm_config = vllm_config
        self.loop = asyncio.get_running_loop()
        self.out_q: asyncio.Queue = asyncio.Queue()
        self.req_q: queue.Queue = queue.Queue()
        self.live: set[str] = set()
        self.aborted: set[str] = set()
        self.resources = SimpleNamespace(engine_dead=False)
        t0 = time.time()
        try:
            self.runner = Runner(ctx_max=cfg.ctx_max, kv_fp8=cfg.kv_fp8, k=cfg.k, group=cfg.group, max_snaps=cfg.snapshots, log=logger.info, pld=cfg.pld, host_tables=cfg.host_tables)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "older than the sources" in str(e):
                raise
            raise RuntimeError(f"swift-mk engine ran out of GPU memory at --max-model-len {vllm_config.model_config.max_model_len:,} with {KV_NAMES[cfg.kv_fp8]} KV{' + host tables' if cfg.host_tables else ''}"
                               f"{' and the prompt-lookup tail' if cfg.pld else ''} ({free_mib} MiB were free at start, another program may have taken memory since).  "
                               f"Largest --max-model-len that fits that memory:\n{vram_limits_text(free_mib, cfg.k)}\n{e}") from e
        logger.info("swift-mk engine ready in %.1f s (weights+MTP %.1f s, warmup %.1f s): ctx_max %d, k=%d, %d cycles per sync, kv %s%s, %d snapshots, prompt-lookup tail %s",
                    time.time() - t0, self.runner.load_s, self.runner.warmup_s, cfg.ctx_max, cfg.k, cfg.group, KV_NAMES[cfg.kv_fp8], " + host tables" if cfg.host_tables else "", cfg.snapshots, "on" if cfg.pld else "off")
        self.thread = threading.Thread(target=self._worker, name="swift-mk-engine", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------------------------------------ engine thread
    def _emit(self, outs):
        self.loop.call_soon_threadsafe(self.out_q.put_nowait, EngineCoreOutputs(outputs=outs))

    def _worker(self):
        while True:
            req = self.req_q.get()
            if req is None:
                return
            rid = req.request_id
            try:
                if rid in self.aborted:
                    continue
                self._serve(req)
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

    def _serve(self, req):
        rid = req.request_id
        sp = req.sampling_params
        t0 = time.time()
        first = True
        n = 0
        for toks, fin, sr in self.runner.run(req.prompt_token_ids, to_gen_params(sp), cancelled=lambda: rid in self.aborted):
            n += len(toks)
            out = EngineCoreOutput(request_id=rid, new_token_ids=toks,
                                   finish_reason=None if fin is None else (FinishReason.STOP if fin == STOP else FinishReason.LENGTH), stop_reason=sr)
            if first:
                st = self.runner.last
                out.prefill_stats = PrefillStats(num_prompt_tokens=st.prompt, num_computed_tokens=st.prompt - st.cached, num_cached_tokens=st.cached, num_local_cached_tokens=st.cached)
                first = False
            self._emit([out])
        st = self.runner.last
        logger.info("%s: prompt %d (resumed at %d) prefill %.3f s, %d tokens in %.2f s = %.1f tok/s, %.2f tok/cycle, snapshots %s", rid, st.prompt, st.cached, st.prefill_s,
                    st.tokens, st.gen_s, (st.tokens - 1) / st.gen_s if st.gen_s > 0 else 0.0, (st.tokens - 1) / st.cycles if st.cycles else 0.0, st.snaps)

    # ------------------------------------------------------------------------------------------------ EngineCoreClient
    async def add_request_async(self, request) -> None:
        if self.resources.engine_dead:
            raise RuntimeError("swift-mk engine thread is dead")
        if request.prompt_token_ids is None or request.mm_features or request.prompt_embeds is not None:
            raise VLLMValidationError("swift-mk engine serves text token prompts only (no prompt_embeds / multimodal inputs)")
        check_supported(request.sampling_params)
        self.live.add(request.request_id)
        self.req_q.put(request)

    async def abort_requests_async(self, request_ids) -> None:
        for rid in request_ids:
            if rid in self.live:
                self.aborted.add(rid)

    async def get_output_async(self) -> EngineCoreOutputs:
        return await self.out_q.get()

    async def get_supported_tasks_async(self):
        return ("generate",)

    async def reset_mm_cache_async(self) -> None:
        return None

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

    def make(vllm_config, executor_class, log_stats, client_addresses=None, client_count=1, client_index=0, renderer=None):
        return SwiftCoreClient(vllm_config, executor_class, log_stats, client_addresses, client_count, client_index, renderer)

    EngineCoreClient.make_async_mp_client = staticmethod(make)
