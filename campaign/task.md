**Goal: make the swift-mk decode engine faster.** The project in `{w}` is a hand-written CUDA engine (CUDA graphs of fused kernels, no vLLM) that decodes
Swift-1.5-Qwen3.8-27B-INT4 (64 layers: 48 gated-DeltaNet + 16 full-attention, W4A16 GEMVs, int4 lm_head copy + exact BF16 rescoring of the top-R candidates) on one RTX 3090 (sm_86, CUDA 13), single stream, with on-device
MTP speculative decoding as shipped by the server (K11): the MTP chain drafts k=3 tokens on the device, a host-side prompt-lookup (n-gram) proposer appends up to 4 more tokens copied from the context
(code edits, logs, tool output copy their input), and one verify step runs T = 4..8 rows (`Engine(tmax=8)`: 9-slot DeltaNet snapshot ring, one CUDA graph per width, `enable_pld()`, `cycles_pld`).
Plain decode is ~18.4 ms/step (~54 tok/s), a verify step V(T) is ~19.6 ms at T=4 and ~21.6 ms at T=8; greedy decode runs at ~170 / 110 / 165 / 135 tok/s on code / prose / math / 8K-context prompts (the tail rarely fires there)
and at 230-330 tok/s on copy-heavy tasks. Python (`engine.py`, `spec.py`, `pack*.py`, `prefill.py`) drives the library built from `src/*.cu`.

**Read first (all inside `{w}`):** `BENCH.md` (every measurement so far, incl. negative results; K11 = the hybrid draft), `GEMV_LOG.md` (the W4 GEMV attempts that failed: do not repeat them), `include/*.h` (the API
and the device-state contract), `tests/` (unit tests / benchmarks; read-only for you, see below). Weights live in `~/models/swift-1.5-int4` (read-only).

**What is measured** (`try` runs it; the gate runs it against pristine controls): the geometric mean of decode tokens/s of greedy k=3 + prompt-lookup-tail speculative decoding on six fixed prompts
(a code prompt, a prose prompt, a math prompt, an 8K-token-context summarisation prompt, two copy-heavy tasks: a whole-file code edit and a log-to-table extraction), prefill excluded, median of reps, after building your tree
with `make BUILD=build`. Anything that makes the *engine* faster for decoding counts: kernels, the verify / catch-up / draft structure of one cycle (T up to 8 rows), launch structure inside the CUDA graph, memory traffic,
the host loop (`spec.py`, the proposer). Changing which tokens come out (a different model, quantisation that changes outputs beyond rounding, fewer layers, a cheaper draft that changes the
acceptance distribution of the *verified* output) is not an optimisation: outputs are verified (below). Changing the acceptance rate by making drafts better is fine if the verified output stays exact.

**Build:** `cd {w} && nice -n 10 make -j8 BUILD=build` (CPU only; do it yourself to see compile errors, `try` then finds the build up to date). The python side loads `build/libswiftmk.so`.
You may add `.cu` files (the Makefile globs `src/engine*.cu`, `src/attn*.cu`, `src/deltanet*.cu`, `src/mtp*.cu`, `src/prefill*.cu`; add a line for other names).

**Correctness is judged by the harness, not by you** (`try` prints the verdict as `check:`): the engine is teacher-forced on the pristine model's continuation of several prompts at T=8 (widest hybrid verify step), T=4 (chain verify) and T=1 (plain decode), also on the tmax=4 library-default engine (5-slot ring),
and must reproduce its top-1 tokens and top-20 logits within rounding-level limits (near-ties are excused, a different accumulation order is fine, a wrong kernel is not);
greedy speculative decoding (k=2,3, with and without the prompt-lookup tail, forced partial acceptance of up to 8 rows, which exercises the DeltaNet / conv ring rollback and wrap-around) must equal the pristine continuation; the device sampler and the exact
speculative-sampling accept kernels (chain, and chain + tail) must follow their analytic distributions. A hidden second workload (sampled decoding, other prompts and lengths, k=2 and 3 + tail, prompts where the tail helps and where it never fires, and plain)
must not get slower, and plain greedy k=0 decode must not regress. Do not edit `tests/test_*.py` (the harness notes edits and ignores them).

**Machine access:** only through `try` (one at a time; it can wait on a lock). `try {lineage} --only code,math` measures a subset; `try {lineage} --profile` prints an nsys kernel
table of 32 cycles (ms/cycle per kernel) on one prompt (prompt names: `code`, `prose`, `math`, `long8k`, `edit`, `table`) instead of benchmarking. Do not run `nvidia-smi`, `nsys`, `ncu` or python engine scripts yourself.

**Facts about this hardware / engine that earlier work established** (details in BENCH.md): the W4 GEMVs are ~85% of a step at 740-825 GB/s against ~880 GB/s read-only; the verify step
is ~93% of a cycle (the K9 head cut ~2 ms, K11 verify widths 5..8 add 0.4 ms per extra row), the MTP catch-up + drafts ~1.6 ms; DeltaNet snapshot writes make the T=4 DeltaNet step 2.1-2.3x the T=1 one and T=8 3.9x (2.0 ms); the host proposer
costs < 10 us per cycle (mailbox spin); fusing RMSNorm into the GEMV prologue
was slower; L2 prefetch of the next layer's weights was slower; the GPU is power-capped (clocks 1.5-1.8 GHz), so measure ratios, not absolutes. Results are often within +-0.3%: a claim
needs a measurement repeated with and without the change (`git stash` is not allowed: copy the file, or use `git checkout kept -- <file>`).
