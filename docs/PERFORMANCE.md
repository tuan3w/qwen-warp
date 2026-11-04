# Performance and quality: what was measured, and how far to trust it

Contents: [Summary](#summary) | [1. Memory](#1-memory-how-much-context-fits) | [2. Quality of the 4-bit KV cache](#2-quality-of-the-4-bit-kv-cache) | [3. Speed](#3-speed) |
[4. HyperQwen as a reference point](#4-hyperqwen-as-a-reference-point) | [5. Reproducing](#5-reproducing) | [Back to README](../README.md)

Hardware for everything below: one RTX 3090 (24 GB, sm_86) at a 350 W power limit that also drives a desktop (about 0.6 GiB in use before
the engine starts), checkpoint `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, batch size 1. Every number is a **single run** unless a spread is given.
Decode speed is end-to-end tokens/s through the OpenAI server path with MTP k=3 (plus the prompt-lookup tail where stated), greedy.

## Summary

| Question | Answer | Kind of evidence | Section |
|---|---|---|---|
| How much context fits? | 262,144 tokens on the 4-bit cache with host-pinned tables: 23,227 MiB in use on the card after a 262,080-token prompt with `--prompt-lookup 0` (22.7 GiB, of which 0.57 GiB is the desktop), 23,899 MiB with the default tail | server-level: real starts, literal near-limit prompts | [1](#1-memory-how-much-context-fits) |
| What does the 4-bit cache cost in quality? | perplexity +0.26% at 98k tokens; coding pass@1 244 -> 241 of 283 (p = 0.45, not significant) | proxies (perplexity, single-shot coding) | [2](#2-quality-of-the-4-bit-kv-cache) |
| What does it cost for tool calls? | a minimal one-parameter schema in the usual key order: 39% well-formed on the 4-bit cache against 91% on bf16; 87% with the default sorted rendering | one schema, in-process runs of 150 requests per cell | [2](#tool-calls) |
| How fast is decode? | 94 to 133 tok/s from 32k to 262k tokens (4-bit cache); 105 to 181 tok/s on short prompts | single runs, re-measured 2026-11-04 | [3](#3-speed) |
| How fast is prefill? | 1,888 tok/s at 16k, 1,094 at 190k (173.5 s), 963 at 262k (272 s) | single runs, re-measured 2026-11-04 | [3](#3-speed) |
| How does it compare with HyperQwen? | long context: decode is 1.2x to 1.3x faster than its `mtp-long` at 32k to 130k and 1.9x to 3.6x faster than its `mtp-huge` at 32k to 190k; short prompts: 1.2x to 1.3x faster than its MTP presets but slower than its DFlash2 preset (about 1.5x on code, earlier session); DFlash2 was not measured at long context; prefill within about 10% | same day, same card, prompts and client for the MTP presets; one pinned build; single runs | [4](#4-hyperqwen-as-a-reference-point) |
| What was not measured? | multi-turn agentic task success, coding-agent tool sets, other GPUs, other power limits | not measured | |

## 1. Memory: how much context fits

The KV cache is the only thing that grows with context. Per token, for the 16 full-attention layers (4 KV heads x 256 dims):

| KV dtype | bytes / token | 262,144 tokens |
|---|---|---|
| bf16 | 64 KiB | 16 GiB |
| fp8 (e4m3, checkpoint scales) | 32 KiB | 8 GiB |
| int4 (per token+head, asymmetric) | 16.5 KiB | 4.1 GiB |

Largest `--max-model-len` that was **measured to start and serve** with the full stack (MTP + fast prefill), tail = the prompt-lookup tail (`--prompt-lookup`, on by default where it fits). "Serve" = a prompt of
(`--max-model-len` - 64) tokens plus 64 generated tokens answered with HTTP 200 and the server alive afterwards. The card also drives a desktop that held 589 MiB before the server started; torch reported 23,228 MiB free at start.

| KV cache | tail on, tables on the GPU | tail off, tables on the GPU | tail on, `--host-tables` | tail off, `--host-tables` |
|---|---|---|---|---|
| bf16 | 26,624 | 34,816 | 94,208 | 105,472 |
| fp8 | 48,128 | 62,464 | 172,032 | 184,320 |
| int4 | 79,872 | 104,448 | 262,144 (the model maximum) | 262,144 (the model maximum) |

The first tested length above each number failed: 1,024 tokens higher, except bf16 + `--host-tables` + tail (97,280 failed) and fp8 + `--host-tables` + no tail (190,464 failed), whose edges lie inside a 3k and a 6k bracket, and
the 4-bit + `--host-tables` cells, which reach the model maximum (their real edge was not measured). The failure is a torch out-of-memory error in the first warm-up prefill. These numbers move with the free memory
(a headless card has about 0.6 GiB more, a card with a GPU-accelerated browser has less). The server therefore no longer uses a table of lengths: it measures the free memory at start, compares it with a model fitted to 21 real starts
(`needed MiB = fixed + KiB per token x length`; 72, 40 and 24.5 KiB per token for bf16, fp8 and int4, which is the cache plus 8 KiB for the MTP layer's KV and the prefill scratch; fixed 20,740 MiB, or 15,800 with host tables; the tail adds 572 MiB, or 688 with host tables;
within +-40 MiB of every real edge) and refuses a length that leaves less than 64 MiB. What it accepts, the defaults and the refusal message: [Usage](USAGE.md#memory-and-context).

- The documented default before this fix (28,672 with the tail, bf16) did not start on the card above: the real edge is 26,624. [INFERENCE] The old number was set at the edge on a card with 144 to 290 MiB more free memory (an earlier desktop held less), with no margin. The other earlier lengths did not match either:
  bf16 "about 74,000" (neither 34,816 nor 26,624), fp8 "65k" (does not serve; the edge is 62,464), and "131k" (fp8 with host tables) and "98k" (4-bit) were understatements. The table above replaces them.
- Server-level checks of the 4-bit cache with `--host-tables` at the model maximum, after a literal near-limit prompt: `--profile long` (196,608 with the tail): 22,331 MiB in use after a 196,544-token prompt, prefilled in 178.9 s;
  `--max-model-len 262144 --prompt-lookup 0`: 23,227 MiB in use after a 262,080-token prompt (prefilled in 275 s; the answer ended by EOS after 7 tokens, deterministically); `--max-model-len 262144` with the tail (auto, 468 MiB stay free):
  23,899 MiB in use after a 262,080-token prompt plus 64 generated tokens, 277 s. `nvidia-smi` stays near 23.9 GB because PyTorch's caching allocator keeps the request scratch it allocated; it is not a headroom meter here.
- Earlier engine-level measurement at 262,144 with `--host-tables`: 22.44 ms/token decode, 964 tok/s prefill at that length. An earlier server-level request with a 250,055-token prompt was prefilled in 268 s, and a code word planted at its very start was
  returned correctly; one request with one needle shows the length works end to end, it is not a retrieval benchmark.
- Time to `/health`: 19 to 20 s with the tables on the GPU, 24 to 35 s with `--host-tables`, 26 to 35 s for `--profile long`.
- Not measured: other cards, a headless card, request-time memory taken by another program, `--num-speculative-tokens` below 3.

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

**Not measured:** multi-turn agentic task success (edit-and-test loops, coding-agent tool sets), which is the use case this cache is aimed at. The numbers above
are single-shot coding and retrieval proxies. Treat int4 as "no measurable loss on these proxies", not as "validated for agents": the tool-call measurements below show a case where it is not free.
`tests/test_kv_dtypes.py` is a regression guard for the cache (a broken cache shows up as tens of percent), not a replacement for these measurements.

### Tool calls

A well-formed tool call is the fraction of requests whose answer carries a call in the `qwen3_coder` format with a known name and the required parameters. Protocol: thinking on, the model's default sampling (T 1, top_k 20, top_p 0.95),
`max_tokens` 800, the user message "What is the weather in Hanoi?", one tool `get_weather(city: string)`. In-process engine runs (the same engine path as the server), 150 requests per cell, 95% intervals in brackets:

| KV cache | schema in the key order clients send (`type`, `properties`, `required`) | keys sorted (`properties`, `required`, `type`) |
|---|---|---|
| bf16 | 91% (137/150) [86, 95] | 93% (139/150) [87, 96] |
| fp8 | 88% (132/150) [82, 92] | not measured |
| int4 (`--profile long`) | **39% (58/150)** [31, 47] | **87% (130/150)** [80, 91] |
| int4, greedy | 99% (149/150) | |

- The chat template prints each tool with `tojson` in the key order the client sent; `--sort-tool-schema on` (the default) sorts the top-level keys of every `function.parameters` before rendering (`server/tool_schema.py`). The change alters the prompt, not the client's JSON, the parser or the arguments.
- Server level (`tests/test_tool_call_rate.py`, 40 requests): before the sorting, `--profile long` gave 18/40 (45%); with it, 37/40 on the default profile and 40/40 on `--profile long`.
- Other schemas were not fragile: a 3-parameter tool with an enum and a required subset, and a nested-object tool, gave 100% in every cell (n = 100 on bf16, n = 60 on int4, both key orders). A fully recursively sorted one-parameter schema on bf16 gave 95/100. So only the minimal one-parameter schema (the one in these docs and tests) was found to be sensitive.
- The sampler is not the cause: at a failing position (the engine drew `<|im_start|>` after `<tool_call>`) its draws match the exact bf16 reference (P = 0.168 in 20,000 draws against 0.170, total variation 0.0023); over 219 teacher-forced real positions the
  draws pass chi-square with no draw outside the reference support (mean total variation 0.0003 to 0.0007, maximum 0.007), and a reference at a different temperature is rejected (`tests/test_sampler_dist.py`).
- A reference vLLM service behind a gateway (its rendered prompt had only `function.parameters` key-sorted) gave 97/100, 100/100, 98/100 and 98/100 at its default temperature, 0.3, 0.6 and 1.0 (98.3% pooled, [96.4, 99.3]). The bf16 engine is about 5 points below it (93% against 98%). On identical contexts the exact
  log-probability gap between `<` and `<|im_start|>` after `<tool_call>` is 13.7 nats on the service (minimum 8.6, n = 70) and 5.7 on the bf16 engine (minimum -1.6; fp8 5.8), i.e. the engine's hidden state is less sharp at that position.
  The cause was not isolated; untested candidates are the W4A16 weights against the service's unknown weights, and KV or activation numerics.
- **Not measured:** a six-tool coding-agent set, second turns, fp8 with the sorted order, fp8 with host tables, a rotated 4-bit cache, top_p / min_p as a mitigation. One schema, thinking on: do not read these rates as an agent success rate.

## 3. Speed

Re-measured on 2026-11-04 on the released code (merged tree of this repository), with the frozen prompts and the streaming client of the HyperQwen comparison in [section 4](#4-hyperqwen-as-a-reference-point)
(real code from several repositories with a planted passphrase at 40% of the prompt; greedy; 256 tokens after the prompt; decode tok/s = tokens after the first chunk over the time from the first to the last chunk). One run per cell, 350 W.

Decode tokens/s at the given prompt length:

| context | 16k | 32k | 64k | 100k | 130k | 190k | 262k |
|---|---|---|---|---|---|---|---|
| int4 KV + prompt lookup (`--profile long`) | - | 124 | 119 | 115 | 108 | 102 | - |
| int4 KV, `--prompt-lookup 0` | 154 | 124 | 133 | 109 | 105 | 96 | 94 |
| fp8 KV, `--host-tables` | - | 123 | 123 | 112 | 108 | - | - |
| bf16 KV (defaults) | 159 | - | - | - | - | - | - |

`-` = not run (the configuration does not serve that length, or it was not part of the run). The planted passphrase was found in every cell except 100k with the 4-bit cache (both arms); fp8 found it at every length it served, and the 4-bit cache found it at 16k, 32k, 64k, 130k, 190k and 262k.

Cold prefill, seconds to the first token (tokens per second in brackets), same runs:

| context | 16k | 32k | 64k | 100k | 130k | 190k | 262k |
|---|---|---|---|---|---|---|---|
| int4 KV, `--prompt-lookup 0` | 8.4 (1,888) | 18.0 (1,776) | 40.5 (1,576) | 71.5 (1,398) | 101.7 (1,277) | 173.5 (1,094) | 271.8 (963) |
| int4 KV + prompt lookup | - | 17.8 (1,788) | 40.4 (1,583) | 68.8 (1,453) | 97.5 (1,332) | 167.7 (1,133) | - |
| fp8 KV | - | 18.0 (1,775) | 40.6 (1,575) | 71.6 (1,395) | 99.4 (1,307) | - | - |

Short prompts (385 tokens, greedy, median of 3 runs, tokens/s):

| configuration | code | prose | math |
|---|---|---|---|
| bf16 KV (defaults) | 172 | 110 | 172 |
| int4 KV, `--prompt-lookup 0` | 170 | 108 | 162 |
| int4 KV + prompt lookup | 173 | 106 | 175 |
| fp8 KV | 172 | 105 | 171 |
| `--profile long --max-model-len 262144 --prompt-lookup 0` | 181 | 115 | 172 |

The speculative-decoding gain depends on the text (prose is the slowest; the cause was not isolated). The last row differs from the 4-bit row above only in the context length, yet is up to 6% faster: treat differences of that size between runs of different server starts as noise.
Another code prompt (the one used by the smoke tests) decodes at about 192 to 194 tok/s on the default profile, so short-prompt speed depends on the prompt as much as on the configuration.
Time to ready was 21 to 30 s for every configuration. Attention kernel time with int4 is 0.55 to 0.9 of fp8 at every length measured (earlier engine-level measurement).

Decode slows from 124 to 96 tok/s between 32k and 190k with the 4-bit cache and the tail off. [INFERENCE] This is the attention scan over a longer cache, since the weight reads are the same at every length; no profile was taken to separate it.

## 4. HyperQwen as a reference point

[HyperQwen](https://github.com/syv-ai/hyperqwen) (syv-ai) is a general project for several models and GPUs, and it changes quickly; this section compares one thing, single-stream decode of this model on this card.
The numbers below describe **one build only**: its public image `ghcr.io/syv-ai/hyperqwen@sha256:00713ba1e04b68db1be9eba2b15b579250f9fb0c29ed61add40c750228fc0cf8`
(OCI revision label `bd6c5e2f11628483109031d99d1cc4a7d22b1a26`, built 2026-10-04, tag `latest` at the time), run unmodified on the same card, power limit,
checkpoint, client and prompts. A newer build may be faster or slower. Both projects were measured **on the same day, 2026-11-04**, one run per cell (ours: median of 3 for the short prompts), 350 W.

```
Decode, tokens/s by prompt length           32k   64k  100k  130k  190k
ours int4 KV, --profile long (+ lookup)     124   119   115   108   102
ours int4 KV, --prompt-lookup 0             124   133   109   105    96
ours fp8 KV                                 123   123   112   108   n/s
HyperQwen mtp-long (fp8/int8 KV)            106    97    86    83   n/s
HyperQwen mtp-huge (4/2-bit KVarN)           64    57    41    33    28
Ratio, --profile long / mtp-long         1.17x 1.23x 1.34x 1.29x   n/s
Ratio, --profile long / mtp-huge         1.94x 2.10x 2.82x 3.31x 3.58x

Short prompts (385 tokens), tokens/s       code prose  math
ours int4 KV, --prompt-lookup 0             170   108   162
ours int4 KV + prompt lookup                173   106   175
ours fp8 KV                                 172   105   171
HyperQwen mtp-long                          135    89   134
HyperQwen mtp-huge                          132    89   138
Ratio, int4 / mtp-long                  1.27x 1.22x 1.21x
Ratio, int4 / mtp-huge                  1.29x 1.22x 1.18x

Cold prefill, seconds to first token        32k   64k  100k  130k  190k
ours int4 KV                                 18    41    72   102   174
ours fp8 KV                                  18    41    72    99   n/s
HyperQwen mtp-long                           19    45    75   104   n/s
HyperQwen mtp-huge                           19    43    70    95   161
```

`n/s` = not served in that run. Ratios are our decode tok/s divided by HyperQwen's in the same column. **What the ratios cover:** single-stream decode speed on this card.
Against `mtp-long` the ratio is 1.2x to 1.3x at 32k to 130k (that preset was not served at 190k); against `mtp-huge`, the only HyperQwen preset served at 190k, it is 1.9x to 3.6x across
32k to 190k (102 tok/s against 28 at 190k). On short prompts it is 1.2x to 1.3x against its MTP presets, but not against its DFlash2 preset (next paragraph). Prefill is within about 10% in both directions: we are 2% to 11% faster than `mtp-long` and up to 8% slower than `mtp-huge` at 130k and 190k.
Quality is not part of this re-run (see below).

**HyperQwen's DFlash2 preset is faster than this engine on short prompts.** In an earlier session (2026-11-01, same card, 350 W, the short prompts above, greedy, 385 tokens, two rounds, spread of 1% or less) DFlash2 decoded 254 / 103 / 225 tok/s on code / prose / math against 166 / 107 / 166 for this engine then (170 / 108 / 162 on 2026-11-04): 1.5x faster on code, 1.36x on math, equal on prose. Pooled over 32 code, log-extraction, tool-call, prose and math tasks it was 219 against 185 tok/s, and over 48 English tasks 168 against 136.
The difference is tokens per step: DFlash2 accepted 5.8 (code) and 5.2 (math) tokens per step against 3.5 for this engine, at 23.0 against 21.1 ms per step. We did not run DFlash2 at long context (its long preset `dflash2-long` failed to start on this card), so the long-context speedups above are against HyperQwen's MTP presets only, and this engine is not faster than HyperQwen in general.
What HyperQwen's own report measures (its `docs/benchmarks.md` and `bench/run_benchmarks.sh`): 127 tok/s single-stream on an RTX 3090 at 250 W, from 8 short chat prompts (English, Danish, code), 1,024-token answers, default sampling, thinking off, decode rate = concurrency x 1000 / mean time per output token, second run after a restart; 381 tok/s when the answer quotes its own prompt (DFlash2 with a 16-token lookup verify block, 25k context); about 1,035 tok/s aggregate at 64 concurrent requests in batch mode.
None of those is the workload of the tables above (long prompts, summary-style answers, one stream), and this engine serves one request at a time, so it does not compete on the last one.

**The same protocol on 2026-11-03** (an earlier session on the same card; kept to show the run-to-run spread): ours 122 / 117 / 107 / 102 / 94 (`--profile long`) and 120 / 128 / 103 / 101 / 91 (tail off) tok/s; HyperQwen `mtp-long` 94 / 88 / 80 / 68 and `mtp-huge` 58 / 53 / 39 / 31 / 27;
short prompts ours 166 / 105 / 156 against `mtp-long` 118 / 79 / 121. Between the two days our decode cells moved by +1% to +10% and HyperQwen's by +8% to +22% (`mtp-long`) and +5% to +10% (`mtp-huge`), so the ratios against `mtp-long` fell
from 1.3x to 1.5x to 1.2x to 1.3x. The cause of the difference between days was not isolated. Treat ratios within about 10% of each other as equal.

Coding pass@1, 283 tasks, from the 2026-11-03 session (not re-run):

```
                                            passed  vs bf16
bf16 reference                                 244        -
HyperQwen mtp-long                             244        0
ours fp8 KV                                    243       -1
HyperQwen mtp-huge                             242       -2
ours int4 KV, --prompt-lookup 0                241       -3   (-1.1 pts, p = 0.45, not significant)
```

Caveats, all of which matter:

- HyperQwen's default memory setting (`GPU_UTIL=0.93`) runs out of memory on this desktop card; it was run with `GPU_UTIL=0.88 KV_MEM=` (its own documented workaround).
  `mtp-huge` was run at its default memory settings.
- HyperQwen's `dflash2-long` preset failed to start here and is **not compared**. The cause was not captured, so this is not evidence about that preset.
- HyperQwen is a different design; this engine is hard-wired to one model on sm_86. The comparison is for this one
  workload, a single stream on a 3090. It says nothing about batching or other GPUs.

## 5. Reproducing

| What | How | Takes |
|---|---|---|
| Every test in the Makefile list: KV-cache perplexity guard, host tables, logit processors, tool-call and sampler checks, the server end to end, start-up configurations | `make test` (needs the checkpoint and the GPU) | long: it starts the server several times; no total is given here |
| The part that needs no GPU | `scripts/check_repo.sh`, `uv run tests/test_tool_schema.py`, `uv run tests/test_prefill_plan.py` | seconds |
| The tables in sections 2 to 4 | not regenerated by anything here: the long-context sweeps were done with throwaway harnesses against this engine and the HyperQwen container, which are not part of this repository | n/a |

The tests check the cache, the host tables, the sampling parameters, the tool-call path, the start-up configurations and the server with only the checkpoint; they are regression guards, not a replacement for the measurements above. `tests/test_startup_configs.py` re-checks that the documented configurations start and serve; `tests/test_tool_call_rate.py` gates a tool-call rate of 70% over 40 requests.
