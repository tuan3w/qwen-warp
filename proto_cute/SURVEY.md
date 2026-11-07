# Prior art survey (clones in ../../refs, shallow, 2026-11-07)

Question: how do vLLM / SGLang / the megakernel projects structure "specialised whole-step decode", and what do we reuse?

## What exists

| Project | Where | Shape of the design | Arch | Covers our model? |
|---|---|---|---|---|
| **vLLM** | `vllm/models/<model>/{nvidia,amd}/` | Per-model package, opt-in via `--additional-config '{"decode_backend": "latency_specialized"}'`. A *decode executor* takes over the decoder layers only for pure-decode FULL-CUDA-graph steps (<=384 rows); every other step runs the stock layers untouched. Startup admission check raises with every unmet requirement. Kernels: CuTe DSL (FlashInfer GEMMs, own all-gather, sparse-MLA), Triton, TRTLLM. | GB300 (sm103) | DeepSeek-V4.1 only. `models/qwen3_5/nvidia` is stock layers + fused norm/quant (Triton). |
| **vLLM AMD "mono"** | `vllm/models/deepseek_v41/amd/mono/` (FlyDSL, a Python DSL like CuTe DSL) | **Real persistent megakernel.** 256 CTAs, one per CU, all resident. A *stage* = task table; CTA b runs tasks b, b+256, ... Stages in order; consumer only waits on earlier stages (no deadlock while grid resident). Hand-off = (value, tag) pairs or data+flags at device scope; tag = launch *epoch* (device word advanced by CTA 0), so no clearing between steps, no host hook, graph-safe fixed pointers. GEMV tasks issue weight loads *before* polling activations. | gfx950 | DeepSeek-V4.1 only |
| **SGLang** | `kernels/plugin_kernels/nvidia/spec.py`, `srt/kernel_plugins/nvidia/binding.py`, lib `sgl-nvidia-mega` | Same idea formalised: `MegakernelSpec` per HF architecture (registered by entry point), `Requirements` (isa, tp, resident_blocks, dtypes, hf_config, server_args), declared `Numerics` (BITWISE or TOLERANCE), `Step(mode, rows)`. Binding replaces model.forward only for covered steps. **Slot plan**: each slot (hop, moe, gdn_verify, ...) has a *stock implementation that reproduces SGLang bit for bit* and a specialised kernel; `SGLANG_NVIDIA_MEGAKERNEL_SLOTS` selects slots for bisecting. | sm103 only | Qwen3.5-397B MoE, MTP verify, TP8 |
| sgl-nvidia-mega internals | `common/sync.py`, `kimi_k3/cutedsl/common/mailbox.py`, `common/jit.py` | **Not one persistent kernel.** Many kernels fused at boundaries through Lamport mailboxes (-0.0 sentinel, poll + re-arm) and PDL (`griddepcontrol`); mix of CuTe DSL, NVRTC `.cu`, Gluon. Inline PTX via `dsl_user_op` + `llvm.inline_asm` (`ld.global.v4`, `ld.volatile`...). NVRTC builds cached on disk by digest of source+toolchain. | sm103 | Qwen3.5 MoE only |
| **Mirage MPK** | `mirage/include/mirage/persistent_kernel/`, `tasks/ampere/` | Compiler: model graph -> task graph with events; persistent kernel with workers + schedulers; task library per arch **incl. Ampere** (`linear`, `rmsnorm`, `norm_linear`, `single_batch_decoding`, `silu_mul_linear`, paged attention). Qwen3-8B demo (dense, bf16). | sm80+ | Dense Qwen3 yes; no W4A16 GEMV, no gated DeltaNet |
| **HazyResearch Megakernels** | `Megakernels/` | Python scheduler emits an instruction tensor per SM (DAG -> per-SM list, priority/zig-zag/wave assignment); CUDA interpreter with `dispatch_op` over opcodes (ThunderKittens, loader/consumer/storer warps); global counters for dependencies. Llama-1B low-latency. | H100/B200 | no |

## What transfers to a 3090 (sm_86, 82 SMs, no TMA/PDL/tcgen05)

1. **Slot plan with a stock twin per slot** (SGLang). Gives modularity + a correctness oracle for free: each op has a pure-torch/vLLM reference path and a specialised kernel; swap per slot, bisect with an env list, declare BITWISE/TOLERANCE.
2. **Cover only pure-decode FULL-graph steps, fall back otherwise** (vLLM, SGLang). Prefill, mixed batches, unsupported sampling params never touch the specialised path.
3. **Admission check up front, listing every reason** (both). Replaces today's scattered HTTP-400s.
4. **Persistent grid + stage task tables + epoch-tagged flags** (vLLM mono, same as MPK/Hazy in essence). Epoch tags remove per-step barrier resets and keep the graph fixed. Works on sm_86 because 82 SMs x k resident CTAs is a launch-time guarantee we can check.
5. **Weight prefetch before the dependency wait** (mono). Weights do not depend on the previous op; on a memory-bound decode this is the one place a megakernel can beat a graph of kernels (kills the fill/drain bubble per op).
6. **CuTe DSL + inline PTX is established practice** (sgl-nvidia-mega `common/sync.py`, vLLM `kernels/linear/cute_dsl/`). No evidence in these repos of W4A16 mma GEMV or gated-DeltaNet decode on Ampere in CuTe DSL; that part is ours.

## What does NOT transfer

- PDL / `griddepcontrol` (sm90+): not on sm_86. The in-graph alternative is the persistent grid.
- Their numbers (GB300, TP4-8, MoE). Nothing here measures a 3090 W4 dense/hybrid model.
- vLLM's `decode_backend` hook is main-branch; this repo pins 0.30.0 (`server/smk_core_client.py` uses internals of it).

## Consequence for the prototype

The existing engine's own design note says a persistent kernel was only within 2.5 % for the GEMV chain. The prior art says the gain, where it exists, comes from (a) cross-op weight prefetch and (b) fused boundaries, not from fewer launches per se. So S3/S4 must measure *bubbles between ops*, with and without prefetch, not just launch count.
