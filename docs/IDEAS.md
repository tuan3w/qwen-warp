# Ideas to explore

Open questions we found and did not finish. **This is a list of ideas, not a roadmap**: nothing here is promised, scheduled or in progress. If one interests you, take it; if you do,
open an issue so others know. Back to the [README](../README.md).

| Idea | Where it stands | Effort to settle it |
|---|---|---|
| [A rotated 4-bit KV cache](#a-rotated-4-bit-kv-cache) | measured as bf16 fake-quant: about 2x lower perplexity cost on this 27B; no kernel, no speed number | a decode kernel and a speed measurement |
| [Why the 4-bit cache missed a needle at 100k](#why-the-4-bit-cache-missed-a-needle-at-100k) | one miss in two experiments, cause unknown | more needle runs, a per-layer look |
| [Multi-turn agentic task success per KV dtype](#multi-turn-agentic-task-success-per-kv-dtype) | not measured at all | an agent benchmark run three times |
| [A non-uniform weight codebook](#a-non-uniform-weight-codebook) | helps accuracy on a 4B model; the extra bytes alone bound the 27B step at about +17% | a kernel; likely a dead end at this size |
| [A better quantized model](#a-better-quantized-model-if-one-appears) | nothing to measure yet | kernel work if the format changes |
| [Faster long prefill](#faster-long-prefill) | about 960 tok/s at 262k | profiling first |
| [Repeating the HyperQwen reference measurement](#repeating-the-hyperqwen-reference-measurement) | one pinned build, single runs | a few hours of runs |

Out of scope: other models, other GPUs, batching. (If you try another GPU, the build takes `ARCH=-arch=sm_XX`, but nothing outside sm_86 has been tried.)

## A rotated 4-bit KV cache

**Idea.** Rotate each K and V head vector with a 256-point Walsh-Hadamard transform (foldable into the query and the output, so it costs almost nothing at decode), use a log2 group
scale per 64 coefficients and a fixed Gaussian 16-level codebook, at the same 4.125 bits per element.

| Setting | Current 4-bit cache | Rotated variant | fp8 |
|---|---|---|---|
| Qwen3.5-2B proxy, 262k fresh tokens (perplexity vs bf16) | +2.015% | +0.194% | not measured |
| This 27B, 32k tokens | +0.27% | +0.10% (+-0.04) | 0.00% |
| This 27B, 98k tokens | +0.26% | +0.13% (+-0.03) | +0.03% |

**What we learned.**

- The proxy's 10x did not carry over to the 27B: about 2x. The current cache's loss on the 27B is already small, and what remains looks like the cost of 16 levels per coefficient. [INFERENCE]
- A per-channel mean table, which was most of the gain on the proxy, added nothing on the 27B (no mean gives the same perplexity).
- A 5-bit variant (group 32, asymmetric, with the rotation) measured +0.05% at 32k in an earlier run, at one more bit per element.

**Caveats.** Measured as quantise-then-dequantise in bf16 storage (the format's numbers, not a kernel's): no decode kernel, no speed measurement, code perplexity only, one run per arm.
The CUDA fake-quant was checked against a torch reference on 65,536 real K/V rows (identical scale codes in 99.9996% of groups).

**To settle it.** Write the decode kernel (lookup-table dequant, rotation folded into q and the output), measure speed against the current cache, then perplexity at 32k and 98k.

## Why the 4-bit cache missed a needle at 100k

The 4-bit arm missed a planted needle at 100k tokens in the speed sweep and found it at 32k, 64k, 130k and 190k; in a separate test it got 7 of 8 needles at about 100k (fp8: 8 of 8). That could be noise
at this sample size or a real failure mode of one position range. **To settle it:** many more needle prompts across positions, with fp8 as the control, and a look at which layers' keys carry the miss.

## Multi-turn agentic task success per KV dtype

The cache is aimed at long agent sessions, but quality was only measured with single-shot coding and retrieval proxies. **To settle it:** run an edit-and-test agent benchmark with bf16, fp8 and
4-bit caches, several runs each, and compare task success and the number of tool-call errors.

## A non-uniform weight codebook

On Qwen3-4B, replacing the uniform 4-bit grid by a per-group 16-level codebook gave +2.8 points of next-token top-1 fidelity (symmetric version; +5.0 points with a free grid). On this 27B it is
probably not worth it: a codebook adds about 0.25 bytes per weight, which is +26% weight bytes (11.68 to 14.72 GiB for the quantised linears), so even a kernel with no extra compute would
lengthen the 18.4 ms decode step by about 3 ms (+17%), and the extra 1.85 GiB would cut the longest context. [INFERENCE: byte counts from the checkpoint headers, step time from the measured memory-bound GEMVs,
not a built kernel.] An earlier fused prototype ran about 2x slower than int4.

## A better quantized model, if one appears

The W4 kernels assume uniform symmetric int4 with groups of 128 (`uint32 [N][K/8]` plus bf16 scales `[N][K/128]`, see [Architecture](ARCHITECTURE.md#conventions-the-kernels-share)).
A different weight format means kernel work, and a different model means re-checking the MTP head and the DeltaNet layout.

## Faster long prefill

Cold prefill is 1,830 tok/s at 16k and about 960 tok/s at 262k, so a 190k prompt takes about 3 minutes; later turns reuse the cached prefix. It is within about 10% of the reference build in the
[comparison](PERFORMANCE.md#4-hyperqwen-as-a-reference-point), so a gain would come from the attention and DeltaNet prefill path (`engine/prefill.py`). No profile has been taken.

## Repeating the HyperQwen reference measurement

The reference numbers describe one HyperQwen build and single runs. A repeat with repeats per cell, a newer build, its DFlash2 preset if it starts, and a second power limit would be a better reference.
This is a measurement exercise, not a contest.

## How to help

| You can | How |
|---|---|
| Report what happened on your machine | open an issue with `nvidia-smi`, your command and the log tail |
| Take one of the ideas above | say so in an issue, then send a pull request with the command, hardware and a single-run or repeated measurement (see [CONTRIBUTING](../CONTRIBUTING.md)) |
| Check a number | re-run `make test` or the measurement and tell us where it differs |
