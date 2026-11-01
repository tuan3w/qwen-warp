---
title: pdl-megakernel-reconstruction
type: source
tags: []
status: studied
created: 2026-10-30
author: agent
repo: https://github.com/tie-pilot-qxw/pdl-megakernel-reconstruction
path: vendor/pdl-megakernel-reconstruction
commit: 8664720
branch: main
commit_date: 2026-08-06
---

# pdl-megakernel-reconstruction

One-liner: "81 Kernels, 97.4% of a Megakernel" = take Hazy's Llama-3.2-1B ThunderKittens megakernel (H100), keep every opcode body, delete the persistent interpreter, launch the 81 ops as a CUDA graph of ordinary kernels whose edges are rewritten to Programmatic Dependent Launch (PDL), and measure how much of the persistence win comes back. Answer: all but 2.7% (complete decode step), 82.8-85.7% of the *boundary loss*. MIT licence (inherits Hazy).

Pointer legend: `B:n` = `docs/reconstructing-a-megakernel-with-pdl.md:n`; `E:n` = `experiments/pdl-reconstruction/results/<file>:n`; bare = repo-root path at `8664720`. `[INF]` = my inference, not stated by authors. Read only (no GPU job run; H100 + CUDA 12.8 required, README:30-35); every number is the authors'.

## Why I cloned it

- What exactly is "reconstructing persistence with PDL", which mechanism carries the gain, on which GPUs, and what failed? Sibling of `related-work/megakernels-vs-cuda-graphs.md` (same Hazy baseline, different authors). Decides how much of our 3090 megakernel budget should go to persistence vs a graph+PDL chain.

## Orientation

- `docs/reconstructing-a-megakernel-with-pdl.md` (508 lines): the whole argument, tables B:40-45, 127-152, 220-224, 270-274, 346-349.
- `experiments/pdl-reconstruction/results/` curated summaries (`end_to_end_reconstruction.csv`, `direct_tk_pdl_summary.md`, `trigger-timing/summary.md`, `all_micro_summary.md`, `vtp_signal_followup_summary.md`); raw samples are in a GitHub release, not vendored (README:136-138).
- Mechanism code: `demos/low-latency-llama/matvec_pipeline.cuh` (trigger :200-229, wait :446-460), `matvec_adds.cu`, `attention_partial.cu`, `mlp_simt_module.cu` (graph edge rewrite :296-395, launch attr :86-101), `demos/low-latency-llama/llama.cu:100-160` (config structs).
- Build/run: README:38-92 (`make -C demos/low-latency-llama GPU=H100`, `NVCC=cuda-12.8 PERSISTENT_VM=1`; CUDA 13.1 ptxas rejects `setmaxnreg`, README:32-36); `experiments/pdl-reconstruction/run_body.sh`, `run_e2e.sh`.

## Mechanisms

### 1. What "reconstruction" means
**What:** same BF16 TK opcode bodies (GEMV, TMA weight streaming, FP32 accum) but no resident interpreter / on-device smem page allocator / opcode dispatch / cross-op counters (B:74-79); each op becomes one kernel node: 16 layers x 5 active opcodes (1 RMS+QKV+RoPE+KV, 2 attention, 4 O-proj, 5 RMS+up/gate+SiLU, 6 down; opcode 3 inactive) + LM head = **81 nodes, 80 edges** (79 body edges + opcode6->LM head) (B:53-66; `experiments/pdl-reconstruction/results/direct_tk_pdl_summary.md:85-87`). One CUDA graph **per decode position** (127 graphs for P32/D128) because `pos_id` is baked in (B:426-431; `experiments/pdl-reconstruction/results/all_micro_summary.md:20-25`). **Verified:** read.

### 2. The PDL protocol (two lines per edge)
**What:** (a) producer calls `cudaTriggerProgrammaticLaunchCompletion()` once its loader has *issued* (not completed) its last weight TMA: `demos/low-latency-llama/matvec_pipeline.cuh:200-229` (the "arrival" control at :208-222 waits for `weights_arrived` first; issue variant skips it); also `demos/low-latency-llama/matvec_adds.cu:199-200`, `demos/low-latency-llama/attention_partial.cu:415,730`. (b) consumer's loader issues its own weight/KV loads, then the activation-reading thread calls `cudaGridDependencySynchronize()` only at the first read of the producer's output: `demos/low-latency-llama/matvec_pipeline.cuh:446-460`, `demos/low-latency-llama/attention_partial.cu:301,434`, `demos/low-latency-llama/matvec_adds.cu:132`. Config: `demos/low-latency-llama/llama.cu:122-137` (`DIRECT_PDL_TRIGGER_AFTER_LOAD=true`, `..._WAIT_FOR_LOAD_ARRIVAL=false`, `DIRECT_SKIP_GLOBAL_COUNTERS=true`).
**Why:** "release early" != "release at entry": trigger-after-issue vs trigger-after-arrival saves **34.032 us** of an 836 us body (4.07%, sd 0.066, 3 fresh processes, matched 13-page/3-stage kernels) (`experiments/pdl-reconstruction/results/trigger-timing/summary.md:15-26`, B:118-125). **Verified:** read.

### 3. Graph edges rewritten to programmatic edges
**What:** capture the kernel chain, then `cudaGraphGetEdges` -> `RemoveDependencies` -> `AddDependencies` with `cudaGraphEdgeData{type=cudaGraphDependencyTypeProgrammatic, from_port=cudaGraphKernelNodePortProgrammatic|LaunchCompletion}` for each kernel->kernel edge: `demos/low-latency-llama/mlp_simt_module.cu:296-395` (replacement built :383-391; port counters :276-283); CUDA>=13 vs `_v2` API shim :21-42. Also supports plain stream launches with `cudaLaunchAttributeProgrammaticStreamSerialization` (:86-101, 165-168). Harness flag `--graph-edge-mode capture|programmatic|launch-completion` (`experiments/pdl-reconstruction/harness/bench_full_standalone_graph_direct_tk_e2e.py:141-147`). The default-dependency baseline is the *same binary*, only edge semantics differ (B:444-447). **Verified:** read.

### 4. Results (H100 SXM5 80 GB, CUDA 12.8, torch 2.7.0+cu128, 700 W, no clock lock; B:392-399)
| Scope | Persistent VM | PDL chain | Default-dep chain | Source |
|---|---:|---:|---:|---|
| Real-weight complete step (embed..argmax), P32/D128 | 983.127 us | 1009.711 (+2.704%) | 1169.071 (+18.9%) | README:96-100; `experiments/pdl-reconstruction/results/end_to_end_reconstruction.csv:2-4` |
| Synthetic-weight 16-layer body | 776.942 | 809.626 | 967.162 | README:109-113; B:130-137 |
- Recovery of kernel-boundary loss: 85.7% at step scope (README:102-104), 82.8% at body scope (B:135-137). Gap left: 26.584 us/step (B:357).
- Per-edge savings in isolation (non-additive): QKV->attn 20.5, attn->O 32.7, O->upgate 22.4, upgate->down 40.6, down->next-QKV 42.2; all 79 edges 154.0 us (B:141-148).
- Only **H100** was measured by this repo. Hazy-class numbers on H200/GB300 are in the sibling repo (`related-work/megakernels-vs-cuda-graphs.md`: H100 +0.69%, H200 +4.59% megakernel advantage). Nothing on Ampere/consumer parts. SGLang 1210 us row has a different timing scope (CSV `scope_mismatch`, B:47-49).
- Honest caveats: default-dependency row is a single un-replicated process (B:429-431); free-running token hashes are not a bit-exact oracle because both paths use atomic reductions (B:454-458); `experiments/pdl-reconstruction/results/all_micro_summary.md:26-37` records that a reduced-resource "micro" chain with early release was **incorrect** once adjacent grids co-resided (pairwise tests passed, 16-layer chain failed) -> the ~925 us plateau is invalid.

### 5. Why 640-thread/208 KB CTAs still overlap: "rolling tail wave"
**What:** CTAs use 13x16 KiB pages = 208 KiB smem (`include/config.cuh:42-44`; 216,504 B dynamic, `experiments/pdl-reconstruction/results/trigger-timing/summary.md:9-12`), ~90 regs/thread, 640 threads -> one CTA/SM, so no true co-residency (B:154-159). `%globaltimer`/`%smid` traces over 240 boundaries: every boundary admitted a successor CTA before the predecessor grid's last CTA exited; none entered an SM before that SM's predecessor CTA exited; shortest same-SM retire->admit 0.256 us; 63-75 of 80 boundaries issued a successor weight/KV load before the predecessor grid ended; no wait returned early (B:162-191). So PDL = successor grid fills SMs vacated by *uneven* predecessor retirement and starts its weight loads while its consumers wait. **Verified:** read (trace code not re-run).

### 6. Three failed "obvious" improvements (B:193-353)
1. **Co-residency** diagnostic (deliberately incorrect, 4-of-16 K partitions, 65 KiB vs 128 KiB smem padding, same SASS): PDL value 51.0 us (+7.19%) if two grids could share an SM; default edges 0.328 us (B:220-224).
2. **Shrink to fit** (opcode 4: regs 88->37, bit-exact, 1,714/1,792 same-SM successor entries) **loses**: 78.608 us co-resident vs 75.152 blocked vs 71.024 us original; shrink costs +3.42% standalone (B:270-281).
3. **Virtual tensor parallelism** (4 shard counters on the up/gate->down edge instead of whole-grid wait): +4.9 us *slower* (814.945 vs 810.059) (B:346-353; `experiments/pdl-reconstruction/results/vtp_signal_followup_summary.md:103-106`).

### 7. Remaining gap + missing primitive
Resident state is lost per grid (regs/smem/barriers, CTA admission, scheduler context, B:360-374); proposed "phase-scoped resource lifetime" (free smem pages/registers early while the CTA drains) is not exposed by PDL (B:376-390). Not proven to be the whole 26.6 us (B:386-388).

## Worth stealing
- The protocol: trigger after *issuing* independent loads, `GridDependencySynchronize` at the first dependent read, nothing earlier. 34 us of 836 us from trigger placement alone.
- Edge rewrite code (`demos/low-latency-llama/mlp_simt_module.cu:296-395`) is a ready recipe for a graph of kernels with programmatic edges; MIT-licensed, copyable.
- Instrument boundaries with `%globaltimer`+`%smid` at entry/issue/trigger/wait/exit to prove overlap is real (B:162-172); gate any "faster" claim on graph-vs-eager tensor checks at P32/P95/P158 (`experiments/pdl-reconstruction/results/trigger-timing/summary.md:22-25`).
- Evaluate on the full chain, not standalone kernels (B:189-191); keep an admission-only control (smem padding that forces 1 CTA/SM, identical SASS) to isolate co-residency.

## Gotchas
- 1B Llama / 16 layers / BF16 / H100 132 SMs / TMA+`setmaxnreg`: none of this is our regime (27B W4, sm_86, no TMA/bulk async groups beyond cp.async, 82 SMs, 100 KB smem). The 97.4% is a ceiling-type result on *large-CTA* kernels; per-boundary loss there is (967-777)/80 edges = ~2.4 us [INF arithmetic from B:130-137]; our kernels (small CTAs) can co-reside and may behave differently.
- **Hopper-only on the device side. Verified by us (tiny compile, nothing run):** `nvcc -arch=sm_86 -ptx` (CUDA 13.0 at /usr/local/cuda) emits `griddepcontrol.launch_dependents`/`.wait` for `cudaTriggerProgrammaticLaunchCompletion()`/`cudaGridDependencySynchronize()`, but ptxas then aborts: "Instruction 'griddepcontrol' requires .target sm_90 or higher"; sm_90 compiles clean. The sibling repo likewise hard-rejects pre-Hopper (`vendor/megakernels-vs-cuda-graphs/llama/kernels/candidate.py:62-63`). So the whole trigger/wait protocol of mechanism 2 **cannot be written for our RTX 3090**. [INFERENCE: the host-side launch attribute alone cannot give safe early-admission overlap without the device-side wait.]
- Timing is mean-of-medians per position with no clock lock; 3 processes; sd ~0.09 us on the headline.

## Transferable to our 3090/W4/MTP megakernel
- **PDL is not available on sm_86 (see Gotchas), so the graph+PDL chain is not a baseline we can build.** What remains is plain CUDA graph (boundary cost = their default-dependency path: 190 us / 80 edges = ~2.4 us per edge on H100 [INF arithmetic from B:130-137]; at our ~340-880 boundaries per token [INF; bonsai graph ~14 kernels/layer, open-jet note 338 barriers] that is ~0.8-2 ms on a ~17 ms step) vs a persistent kernel with software grid barrier (~2-4 us x 340 [open-jet note, unmeasured]). Both cost about the same order; what a persistent kernel uniquely allows on sm_86 is *explicit* cross-op prefetch (next GEMV's weights into L2/smem while the barrier is pending), which is the thing PDL bought on H100.
- Reusable measurement kit: trigger-placement A/B, admission-blocking smem-padding control, per-edge masks (`--edge-slot-mask`), per-position graphs.
- Design rule for our own persistent kernel: the part of the megakernel win that is real is *cross-op weight prefetch overlap*; implement it explicitly (open-jet-style idle-block L2 prefetch / bonsai `prefetch_l2`) rather than assuming it.

## Dead ends / lessons
- Shrinking resources to get co-residency, finer-grained (shard-level) readiness, and early-release "micro" chains all lost or were incorrect. Hazards of early release: a successor CTA that co-resides can run before the predecessor finishes writing; only a full-grid wait is safe.
- Standalone microbenchmarks mis-price PDL; admission/tail effects only appear in a chain.

## Verdict
- PDL reconstructs ~86% of the megakernel boundary win on H100 for big-CTA BF16 kernels (+2.7% residual): the win is the schedule (early admission + early loads + late dependency wait), not "one kernel".
- Evidence is single-GPU-type (H100 SXM), one tiny model. PDL cannot even be compiled for sm_86 (ptxas error above), so this result is not directly a baseline for the 3090.
- For us: the megakernel's real job is to re-create PDL's overlap in software (prefetch next-op weights across the barrier); graph-only must be measured separately as the no-overlap baseline.

---

Pinned at `8664720` (main, 2026-08-06) · 99 files · 1,9M
Update with: `git -C vendor/pdl-megakernel-reconstruction pull` then refresh `commit:` above.
