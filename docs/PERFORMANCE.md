# Performance and quality: what was measured, and how far to trust it

Contents: [1. Memory](#1-memory-how-much-context-fits) | [2. Quality of the 4-bit KV cache](#2-quality-of-the-4-bit-kv-cache) | [3. Speed](#3-speed) |
[4. Comparison with HyperQwen](#4-comparison-with-hyperqwen) | [5. Reproducing](#5-reproducing) | [Back to README](../README.md)

Hardware for everything below: one RTX 3090 (24 GB, sm_86) at a 350 W power limit that also drives a desktop (about 0.6 GiB in use before
the engine starts), checkpoint `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, batch size 1. Every number is a **single run** unless a spread is given.
Decode speed is end-to-end tokens/s through the OpenAI server path with MTP k=3 (plus the prompt-lookup tail where stated), greedy.

## 1. Memory: how much context fits

The KV cache is the only thing that grows with context. Per token, for the 16 full-attention layers (4 KV heads x 256 dims):

| KV dtype | bytes / token | 262,144 tokens |
|---|---|---|
| bf16 | 64 KiB | 16 GiB |
| fp8 (e4m3, checkpoint scales) | 32 KiB | 8 GiB |
| int4 (per token+head, asymmetric) | 16.5 KiB | 4.1 GiB |

Largest `--max-model-len` that was **measured to start and serve** with the full stack (MTP + fast prefill) on this card:

- bf16 KV: about 74,000 (the server test used 74,000; 100k does not fit)
- fp8 KV: 65k with the embedding/head tables on the GPU; 131k with `--host-tables`
- int4 KV: 98k with the tables on the GPU; **262,144 with `--host-tables`** (22.4 GiB used, 22.44 ms/token decode, 964 tok/s prefill at that length, engine-level measurement)
- Server-level check of the 262k configuration (`--profile long --max-model-len 262144 --prompt-lookup 0`, one request): a 250,055-token prompt was prefilled in 268 s
  and a code word planted at its very start was returned correctly. One request with one needle shows the length works end to end; it is not a retrieval benchmark.

`--host-tables` moves the bf16 embedding and the rescoring lm_head (4.7 GiB) to pinned host memory. It is bit-identical (greedy ids and logits
equal, `tests/test_host_tables.py`) and cost 0 to 2% of decode speed (two runs: 17.25 vs 17.27 ms/token, and 18.89 vs 19.23 ms/token). The plain bf16 head and the MTP `head_mode 1` are
refused in this mode because they would stream a whole table over PCIe every step.

## 2. Quality of the 4-bit KV cache

Perplexity on code text, teacher-forced, same engine and head, bf16 KV as reference:

- 32k tokens: int4 +0.27%
- 98k tokens: int4 **+0.26% (+-0.05)**, no growth with position. fp8: +0.03%.

Task accuracy (greedy, pass@1, 283 coding tasks = HumanEval+ (163, with HumanEval/32 excluded for a dataset bug) + MBPP+ (120)):

- bf16 244, fp8 243, **int4 241** (-1.1 points vs bf16, p = 0.45: not distinguishable from noise at this sample size, and not proof of equality)
- HumanEval+ alone: int4 142 vs bf16 146 (p = 0.22)
- Token-identical to bf16 on HumanEval+: fp8 88%, int4 67%. Greedy outputs under a lossy cache drift from bf16's; whether a drifted answer is still correct is what the pass@1 rows measure.
- Repo question-answering over a real code base: every arm 82/86 at <= 64k context. At 100k: fp8 36/36, int4 34/36.
- Needle-in-a-haystack at ~100k, 8 prompts: fp8 8/8, int4 7/8. The miss is a real int4 failure mode to know about: in the separate speed sweep this engine's int4 arm also missed the needle at 100k (and found it at 32k, 64k, 130k and 190k), while fp8 and both HyperQwen presets found it at 100k.

**Not measured:** multi-turn agentic task success (tool calls, edit-and-test loops), which is the use case this cache is aimed at. The numbers above
are single-shot coding and retrieval proxies. Treat int4 as "no measurable loss on these proxies", not as "validated for agents".
`tests/test_kv_dtypes.py` is a regression guard for the cache (a broken cache shows up as tens of percent), not a replacement for these measurements.

## 3. Speed

Decode tokens/s at the given prompt length (single run, 350 W):

| context | 32k | 64k | 100k | 130k | 190k |
|---|---|---|---|---|---|
| int4 KV + prompt lookup | 122 | 117 | 107 | 102 | 94 |
| int4 KV | 120 | 128 | 103 | 101 | 91 |
| fp8 KV | 117 | 117 | 108 | 98 | not served in this run |

Short prompts, int4 KV, greedy: 166 tok/s on code, 105 on prose, 156 on math (speculative decoding gain depends on the text; the cause of the prose gap was not isolated).
Cold prefill: 1,830 tok/s at 16k, falling to 964 tok/s at 262k (178 s for a 190k prompt). Attention kernel time with int4 is 0.55 to 0.9 of fp8 at every length measured.

Decode slows from 120 to 91 tok/s between 32k and 190k. [INFERENCE] This is the attention scan over a longer cache, since the weight reads are the same at every length; no profile was taken to separate it.

## 4. HyperQwen as a reference point

This is a reference point, not a competition: [HyperQwen](https://github.com/syv-ai/hyperqwen) (syv-ai) is a general project for several models and GPUs, and it changes quickly.
The numbers below describe **one build only**: its public image `ghcr.io/syv-ai/hyperqwen@sha256:00713ba1e04b68db1be9eba2b15b579250f9fb0c29ed61add40c750228fc0cf8`
(OCI revision label `bd6c5e2f11628483109031d99d1cc4a7d22b1a26`, built 2026-10-04, tag `latest` at the time), run unmodified on the same card, power limit,
checkpoint, client and prompts. A newer build may be faster or slower. Decode tok/s at 32k / 64k / 100k / 130k / 190k:

```
Decode, tokens/s by prompt length           32k   64k  100k  130k  190k
ours int4 KV, --profile long (+ lookup)     122   117   107   102    94
ours int4 KV, --prompt-lookup 0             120   128   103   101    91
ours fp8 KV                                 117   117   108    98   n/s
HyperQwen mtp-long (fp8/int8 KV)             94    88    80    68   n/s
HyperQwen mtp-huge (4/2-bit KVarN)           58    53    39    31    27
Ratio, --profile long / mtp-long         1.30x 1.33x 1.34x 1.50x   n/s
Ratio, --profile long / mtp-huge         2.10x 2.21x 2.74x 3.29x 3.48x

Short prompts (385 tokens), tokens/s       code prose  math
ours int4 KV, --prompt-lookup 0             166   105   156
ours int4 KV + prompt lookup                170   105   173
ours fp8 KV                                 164   101   163
HyperQwen mtp-long                          118    79   121
HyperQwen mtp-huge                          122    82   128
Ratio, int4 / mtp-long                  1.41x 1.33x 1.29x
Ratio, int4 / mtp-huge                  1.36x 1.28x 1.22x

Cold prefill, seconds to first token        32k   64k  100k  130k  190k
ours int4 KV                                 18    41    73   104   178
ours fp8 KV                                  18    41    73   104   n/s
HyperQwen mtp-long                           21    48    79   113   n/s
HyperQwen mtp-huge                           20    45    73    99   168

Coding pass@1, 283 tasks                    passed  vs bf16
bf16 reference                                 244        -
HyperQwen mtp-long                             244        0
ours fp8 KV                                    243       -1
HyperQwen mtp-huge                             242       -2
ours int4 KV, --prompt-lookup 0                241       -3   (-1.1 pts, p = 0.45, not significant)
```

`n/s` = not served in that run. Ratios are our decode tok/s divided by HyperQwen's in the same column. **What the ratios cover:** single-stream decode speed on this card.
Against `mtp-long` the ratio is 1.3x to 1.5x at 32k to 130k (that preset was not served at 190k); against `mtp-huge`, the only HyperQwen preset served at 190k, it is 2.1x to 3.5x across
32k to 190k (91 to 94 tok/s against 27). On short prompts it is 1.2x to 1.4x. **What they do not cover:** prefill (within about 10%) and quality (within noise across all arms).

Caveats, all of which matter:

- HyperQwen's default memory setting (`GPU_UTIL=0.93`) runs out of memory on this desktop card; it was run with `GPU_UTIL=0.88 KV_MEM=` (its own documented workaround).
  `mtp-huge` was run at its default memory settings.
- HyperQwen's `dflash2-long` preset failed to start here and is **not compared**. The cause was not captured, so this is not evidence about that preset.
- HyperQwen is a different design; this engine is hard-wired to one model on sm_86. The comparison is for this one
  workload, a single stream on a 3090. It says nothing about batching or other GPUs.

## 5. Reproducing

`uv run tests/test_kv_dtypes.py`, `tests/test_host_tables.py` and `tests/test_server_smoke.py` check the cache, the host tables and the server end-to-end with only the
checkpoint. The long-context sweeps above were done with throwaway harnesses against this engine and the HyperQwen container; they are not part of this
repository, so the tables in sections 2 to 4 are reported results, not something `make test` regenerates.
