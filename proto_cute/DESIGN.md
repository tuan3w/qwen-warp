# Design: CuTe DSL kernels under vLLM, models added by config (branch `cute-vllm-plugin`)

Goal: keep this project's speed on one RTX 3090, stop owning the model and the engine loop, and make a new model mostly a matter of *op coverage*, not new engine code.
Basis: [SURVEY.md](SURVEY.md) (what vLLM/SGLang do), [FINDINGS.md](FINDINGS.md) (what the 3090 rewards), `vllm_mp_kernel.py` + `s6_vllm_hook.py` (the first adapter, run against vLLM 0.30.0).

## Principle

**vLLM owns the model graph, scheduler, KV manager, sampling, API. We own kernels and the few format-specific backends, registered through vLLM's public extension points.**
A new model then costs: "does vLLM already have this architecture?" (yes for most) + "do our ops cover its layer types?". No per-model 64-layer wiring (today: `engine.cu` hardcodes `NLAYER`, hidden 5120, the 3:1 DeltaNet/attention pattern, `engine.h` shapes).

## Layers

```
L0  cutekit/      op library, no vLLM imports
                  each op = reference twin (torch) + CuTe DSL kernel (constexpr-specialised, lazily compiled, cached by shape key)
                            + can_implement(shape, dtype) + bit-exact/tolerance test against the twin
L1  adapters/     one thin file per vLLM extension point, vLLM imports only here
L2  (later, optional)  decode executor for pure-decode FULL-graph steps, behind a flag, falls back to stock forward
```

### L0 ops (status)

| Op | Needed for | Status |
|---|---|---|
| `w4_gemv` (any N, K%512, T<=8/16) + epilogues | every dense W4A16 linear, any model | **done**: bit-exact vs `smk_w4r_gemv`, 98-100 % speed (S2); SILU-mul epilogue in `mega.py` |
| `w4_dequant` -> bf16 scratch + cuBLAS | prefill / T>16 on the same weight layout | torch spike only; production = layer-major scratch (what `FastPrefill` does) |
| `rmsnorm(+residual)`, fused into GEMV x-load | all | not done; naive norm costs 10 us/stage (FINDINGS) |
| `gdn_decode` | Qwen3.5/3.8, Qwen-Next, Kimi-linear style | not done; vLLM 0.30.0 already ships `fused_gdn_decode_post_conv_mtp` (CUDA op, `qwen_gdn_linear_attn.py`): measure before porting |
| `attn_decode_int4kv` | rotated 4-bit KV | not done; vLLM has `int4_per_token_head` KV dtype, ours is a different (rotated, sink+ring) format |
| `lm_head` quantised + exact rescoring | decode head | not done |

### L1 adapters (vLLM 0.30.0 API unless noted)

| vLLM extension point | We plug | Covers | Verified |
|---|---|---|---|
| `register_linear_kernel(cls, PlatformEnum.CUDA, "mp")` / `MPLinearKernel` | `CuteW4A16LinearKernel`: repack in `process_weights_after_loading`, CuTe GEMV <=16 rows, dequant+GEMM above | **every** compressed-tensors/GPTQ symmetric W4A16 g128 linear of every model | **yes** (`s6`): stock picks Marlin, plugin picks ours; outputs within bf16 rounding of fp32 ref for T=1..64; 3-D input; CUDA-graph capture + replay equal |
| `CustomOp.register_oot` (GDN `ChunkGatedDeltaRule`, rotary, norms) | CuTe ops as `forward_cuda` | layer-level swaps | not yet |
| `register_backend(AttentionBackendEnum.CUSTOM, ...)` / `MambaAttentionBackendEnum` | int4-KV attention decode, GDN state backend | KV format | not yet |
| `ModelRegistry.register_model` | only for architectures vLLM lacks | new/odd models | not needed so far |
| `vllm.general_plugins` entry point | one `register()` that does all of the above, re-entrant | install | `server/plugin/` already uses it (for the engine swap) |
| spec decode: vLLM MTP / DFlash proposers (`v1/spec_decode/dflash.py` exists in 0.30.0) | none initially | draft + verify | measure first |

Selection details found: OOT kernels are *appended* to `_POSSIBLE_KERNELS` (lowest priority, Marlin wins). The adapter inserts at index 0 (private dict) or users set `VLLM_DISABLED_KERNELS=MarlinLinearKernel,...`. Selection is per layer config, not per batch, so one class must serve decode **and** prefill on one weight layout (24 GB leaves no room for two layouts) - hence dequant+GEMM inside the same kernel class.

## Adding a model (the checklist this design buys)

1. It runs on stock vLLM (Marlin) -> that is the correctness reference and the speed baseline.
2. Enable the plugin. A startup report lists, per layer type, which kernel was chosen and why others were refused (`can_implement` reasons), as SGLang's admission check does. W4A16 linears are covered with no model code.
3. New mixer or norm type? Add one op module: torch twin, CuTe kernel, shape-sweep test against the twin (shapes taken from the HF config). Register it through the matching extension point. Nothing else changes.
4. Different quant format (e.g. W4A8, MXFP4, FP8)? New `MPLinearKernel`/`MMLinearKernel` subclass in `adapters/`; the CuTe GEMV core is reused (dequant block is the only change).
5. Bisect by swapping ops back to their twins (`VLLM_DISABLED_KERNELS` for linears, an env list for ours).

## What this gives up, and what upstream already has

| Today (own engine) | Under vLLM |
|---|---|
| MTP/DFlash cycle fully inside one CUDA graph, no host work between cycles | vLLM's proposers + model runner; per-step host overhead is vLLM's. **Unmeasured.** |
| Rotated 4-bit KV (sink + fp16 ring, 262k in 24 GB) | vLLM `int4_per_token_head` is a different format; ours would be an attention backend + KV spec on the hybrid KV manager. Largest piece of work. |
| Host-pinned embedding/lm_head tables, host prefix store, DeltaNet snapshots | vLLM prefix caching for hybrid models exists (docs/design/hybrid_kv_cache_manager.md); host tiers: unknown |
| 16-row shared weight pass across sequences, exact spec sampling + processors | vLLM batching; min_p/logit_bias with spec decode refused by stock (ARCHITECTURE.md:85) |

[INFERENCE] vLLM 0.30.0 already ships int4 KV, a DFlash proposer and a fused GDN MTP decode op, so part of what this repo built has upstream counterparts. None of them has been run on this checkpoint on this card by me.

## The decisive missing measurement

**Stock vLLM 0.30.0 on this checkpoint on the 3090** (Marlin, `--kv-cache-dtype int4_per_token_head`, MTP k=3, FULL cudagraph) at 1 and 4 clients, versus this engine's 181 tok/s (README) and 425 tok/s at 4 users.
Then the same with `CuteW4A16LinearKernel` swapped in. The gap tells us where the work is: kernels (we are good at that), vLLM host overhead (we cannot fix it from a plugin), or KV/spec format. It needs the GPU: `serve.py` holds 22.6 of 24 GB.

## Plan

| Phase | Work | Exit |
|---|---|---|
| P0 (done) | CuTe GEMV parity; stage executor study (megakernel: no); vLLM linear-kernel hook | `FINDINGS.md`, `s6` passes |
| P1 | Baseline and swapped-kernel runs on stock vLLM, real checkpoint, GPU free | tok/s table + nsys of one decode step; go/no-go on the plugin route |
| P2 | Package `cutekit/` + `adapters/` as `swift_cute_vllm` (entry point), lazy compile cache on disk, report at startup, tests in the repo's style | `vllm serve` with the plugin passes the existing server tests that apply |
| P3 | Fused norm into GEMV, SILU epilogue, then `gdn_decode` and int4-KV attention backend if P1 says they matter | per-op bit-exact tests; step time vs P1 |
| P4 | Second model (e.g. Qwen3-4B-W4A16 or a Qwen3-Next-style) with zero engine edits as the acceptance test of the "easy to add" claim | works from the checklist above |

## Risks

- Private vLLM dicts (`_POSSIBLE_KERNELS`) and plugin timing across worker processes: pin vLLM, test on each bump (the repo already pins 0.30.0; 0.33.0 exists).
- Host overhead: the adapter's Python path costs 41 us per linear call eagerly (dlpack wrapping); irrelevant inside a captured graph, material in eager steps. Fix = cached wrappers / CuTe TVM-FFI calling.
- CuTe DSL is NVIDIA-versioned (4.7.1 here); inline PTX pins us to sm_80/86 behaviour. Other arches would need their own dequant/mma blocks.
- Quality parity (bit-for-bit with today's engine) is not a goal under vLLM; tolerance tests against the twin replace it.
