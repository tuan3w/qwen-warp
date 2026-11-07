# Performance and quality: what was measured, and how far to trust it

Contents: [Summary](#summary) | [1. Memory](#1-memory-how-much-context-fits) | [2. Quality of the 4-bit KV cache](#2-quality-of-the-4-bit-kv-cache) | [3. Speed](#3-speed) |
[4. HyperQwen as a reference point](#4-hyperqwen-as-a-reference-point) | [5. DFlash2 block drafter](#5-dflash2-block-drafter-experimental) | [6. Image input](#6-image-input) | [7. Concurrent requests](#7-concurrent-requests---max-num-seqs) | [8. Reproducing](#8-reproducing) | [Back to README](../README.md)

Hardware for everything below: one RTX 3090 (24 GB, sm_86) at a 350 W power limit that also drives a desktop (about 0.6 GiB in use before
the engine starts), checkpoint `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, batch size 1. Every number is a **single run** unless a spread is given.
Decode speed is end-to-end tokens/s through the OpenAI server path with MTP k=3 (plus the prompt-lookup tail where stated), greedy.

## Summary

| Question | Answer | Kind of evidence | Section |
|---|---|---|---|
| How much context fits? | 262,144 tokens on the 4-bit cache with host-pinned tables: 23,227 MiB in use on the card after a 262,080-token prompt with `--prompt-lookup 0` (22.7 GiB, of which 0.57 GiB is the desktop), 23,899 MiB with the default tail | server-level: real starts, literal near-limit prompts | [1](#1-memory-how-much-context-fits) |
| What does the 4-bit cache cost in quality? | rotated format (this version): KL divergence from the bf16 cache 0.069% on a 32k code text (previous format 0.51%), perplexity +0.025% (+-0.025) at 32k and **+0.056% (+-0.017) at 98k** (previous format +0.26%); the coding pass@1 below is for the previous format and was not re-measured | fidelity through the real kernels on two 32k texts; perplexity at 98k; proxies | [2](#2-quality-of-the-4-bit-kv-cache), [KV cache](KV_CACHE.md) |
| What does it cost for tool calls? | a minimal one-parameter schema in the usual key order: 92% well-formed on the 4-bit cache through the server (n = 200; the previous 4-bit format: 46%, n = 100; in-process earlier: 39%), bf16 91%; 88% with the default sorted rendering (n = 100) | one schema, 100 to 200 requests per cell | [2](#tool-calls) |
| How fast is decode? | 94 to 133 tok/s from 32k to 262k tokens (4-bit cache); 105 to 181 tok/s on short prompts | single runs, re-measured 2026-11-04 | [3](#3-speed) |
| How fast is prefill? | 1,888 tok/s at 16k, 1,094 at 190k (173.5 s), 963 at 262k (272 s) | single runs, re-measured 2026-11-04 | [3](#3-speed) |
| How does it compare with HyperQwen? | long context: decode is 1.2x to 1.3x faster than its `mtp-long` at 32k to 130k and 1.9x to 3.6x faster than its `mtp-huge` at 32k to 190k; short prompts: 1.2x to 1.3x faster than its MTP presets with MTP; with `--drafter dflash2`, level on code and prose and 13% ahead on math against its DFlash2 preset (same session, server-discounted engine numbers; 1.09x / 1.10x / 1.12x through our server against its numbers of another session of the same day); DFlash2 was not measured at long context; prefill within about 10% | same day, same card, prompts and client for the MTP presets; one pinned build; single runs | [4](#4-hyperqwen-as-a-reference-point), [5](#5-dflash2-block-drafter-experimental) |
| How fast is the experimental DFlash2 drafter? | through the server (`--drafter dflash2`, bf16 KV, 2 interleaved visits per arm, stop at the end token): 276.7 / 113.4 / 254.0 tok/s on code / prose / math against 183.9 / 118.0 / 183.9 for MTP k=3 + tail (+51% / -4% / +38%), +34% on 8 chat prompts, +7% on 12 pooled chat tasks, **-31% on copy-heavy agent turns** (230 against 334 tok/s); cold prefill +1.3%, `cached_tokens` identical; engine-direct 276 / 112 / 261, level with HyperQwen's DFlash2 in an earlier session (254 / 103 / 226); greedy only | interleaved visits, same session | [5](#5-dflash2-block-drafter-experimental) |
| How many users at once? | `--profile multiuser` (5 slots): 425 / 441 tok/s aggregate at 4 / 8 clients (text; 0.7.0 FIFO 155 / 152), 310 / 318 with every other request about an image; single user unchanged within 0.2% (ids identical) | single runs per cell, one session | [7](#7-concurrent-requests---max-num-seqs) |
| What was not measured? | multi-turn agentic task success, coding-agent tool sets, other GPUs, other power limits | not measured | |

## 1. Memory: how much context fits

The KV cache is the only thing that grows with context. Per token, for the 16 full-attention layers (4 KV heads x 256 dims):

| KV dtype | bytes / token | 262,144 tokens |
|---|---|---|
| bf16 | 64 KiB | 16 GiB |
| fp8 (e4m3, checkpoint scales) | 32 KiB | 8 GiB |
| int4 (per token+head, rotated; exact fp16 sink 4 + ring 256: +34 MiB of device memory and +32 MB of pinned host RAM per snapshot) | 16.5 KiB | 4.1 GiB |

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

**Rotated format (this version; format and method: [KV cache](KV_CACHE.md)).** Mean per-token KL divergence from the bf16-cache distribution to the int4 one (top-64 tokens, teacher-forced through the real kernels, 32k code text): **0.0691% (+-0.0031)**, previous format 0.512%; perplexity +0.025% (+-0.025) against bf16 on the same text. On a held-out 32k text in other languages the previous 0.577% control becomes about 0.085% (derived from the 5.67x score ratio). Perplexity at 98k on the corpus of the earlier measurement below (code from several repositories, 98,304 tokens, paired per token): **+0.056% (+-0.017)** over bf16 against +0.26% for the previous format and +0.03% for fp8, with no growth by position (bins 0-4k to 64-98k: +0.00, -0.06, -0.01, +0.10, +0.09, +0.04%). `tests/test_kv_dtypes.py` measured +0.05% to +0.10% on its text. Needle probe at 64k in chat format (8 depths, passphrase negative log-likelihood): 0.079 nats against 0.086 for fp8 and 0.124 for the previous format; needle at 100k through the server: found 4 of 4 runs, the previous format missed 4 of 4. Not re-measured on this format: coding pass@1, repository question-answering. The pass@1 and question-answering numbers below are for the **previous format** (asymmetric, no rotation, no exact window); the perplexity bullets are too.

Perplexity on code text, teacher-forced, same engine and head, bf16 KV as reference (previous int4 format):

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
| int4, previous format (`--profile long`) | **39% (58/150)** [31, 47] | **87% (130/150)** [80, 91] |
| int4, greedy (previous format) | 99% (149/150) | |
| int4, previous format, through the server (n = 100) | 46% (46/100) [37, 56] | 94% (94/100) [88, 97] |
| int4, rotated format (this version), through the server (n = 200 first build / n = 100 final build) | **92% (183/200) [87, 95]** / **96% (96/100) [90, 98]** | 88% (88/100) [80, 93] |

- The chat template prints each tool with `tojson` in the key order the client sent; `--sort-tool-schema on` (the default) sorts the top-level keys of every `function.parameters` before rendering (`server/tool_schema.py`). The change alters the prompt, not the client's JSON, the parser or the arguments.
- Server level (`tests/test_tool_call_rate.py`, 40 requests): before the sorting, `--profile long` gave 18/40 (45%); with it, 37/40 on the default profile and 40/40 on `--profile long`.
- Other schemas were not fragile: a 3-parameter tool with an enum and a required subset, and a nested-object tool, gave 100% in every cell (n = 100 on bf16, n = 60 on int4, both key orders). A fully recursively sorted one-parameter schema on bf16 gave 95/100. So only the minimal one-parameter schema (the one in these docs and tests) was found to be sensitive.
- The sampler is not the cause: at a failing position (the engine drew `<|im_start|>` after `<tool_call>`) its draws match the exact bf16 reference (P = 0.168 in 20,000 draws against 0.170, total variation 0.0023); over 219 teacher-forced real positions the
  draws pass chi-square with no draw outside the reference support (mean total variation 0.0003 to 0.0007, maximum 0.007), and a reference at a different temperature is rejected (`tests/test_sampler_dist.py`).
- A reference vLLM service behind a gateway (its rendered prompt had only `function.parameters` key-sorted) gave 97/100, 100/100, 98/100 and 98/100 at its default temperature, 0.3, 0.6 and 1.0 (98.3% pooled, [96.4, 99.3]). The bf16 engine is about 5 points below it (93% against 98%). On identical contexts the exact
  log-probability gap between `<` and `<|im_start|>` after `<tool_call>` is 13.7 nats on the service (minimum 8.6, n = 70) and 5.7 on the bf16 engine (minimum -1.6; fp8 5.8), i.e. the engine's hidden state is less sharp at that position.
  The cause was not isolated; untested candidates are the W4A16 weights against the service's unknown weights, and KV or activation numerics.
- With the rotated format the key order no longer moves the rate beyond noise for this schema (the intervals overlap); `--sort-tool-schema` stays on by default. The last two rows used the server with `--profile long`, the same prompt and protocol, one session of old and new builds.
- **Not measured:** a six-tool coding-agent set, second turns, fp8 with the sorted order, fp8 with host tables, top_p / min_p as a mitigation. One schema, thinking on: do not read these rates as an agent success rate.

## 3. Speed

**4-bit cache format change (this version, 2026-11-05).** The rotated format with the exact sink and ring ([KV cache](KV_CACHE.md)) was measured against the previous format in one session through the server with the frozen prompts below, two runs per cell, runs interleaved (previous, port without the ring fix, this version); the GPU drifts about 5% inside a session and the previous format measured against itself at the start and end of the session differed by up to 6%, so differences under about 5% are not resolved. Decode tok/s, mean of two runs:

| Arm | Format | 32k | 100k | 190k |
|---|---|---|---|---|
| `--profile long`, prompt-lookup tail on | previous | 130.5 | 115.0 | 98.6 |
| | rotated (this version) | 126.7 | 123.3 | 102.3 |
| tail off (`--prompt-lookup 0`) | previous | 129.0 | 111.6 | 97.2 |
| | rotated (this version) | 128.5 | 124.6 | 96.9 |

Short prompts (code / prose / math): tail on, previous 180 / 110 / 183, rotated 179 / 111 / 179; tail off, previous 176 / 112 / 167, rotated 178 / 114 / 172. Prefill is unchanged (1,785 to 1,860 tok/s at 32k, 1,085 to 1,128 at 190k). The needle at 100k was found in every rotated run and missed in every previous-format run. The tables below were measured on the previous format on 2026-11-04 and were not repeated; this session's previous-format runs gave 98.6 tok/s at 190k with the tail against 102 in that table, which is the drift between sessions. The rotated format is level with the previous one on short prompts and at 32k without the tail, ahead at 100k (+7% to +12%) and at 190k with the tail (+4%), and about 3% behind at 32k with the tail.

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

## 5. DFlash2 block drafter (experimental)

`engine/dflash.py` runs the 7-token block drafter of `incoai/Qwen3.8-27B-DFlash2` (W4A16, 1.19 GiB) inside the engine ([Architecture](ARCHITECTURE.md#dflash2-block-drafter-experimental-greedy-only)), and since this release the server can serve with it (`--drafter dflash2`, [In the server](#in-the-server---drafter-dflash2) below). Greedy, `Engine(tmax=8)`. The engine-direct measurements and the HyperQwen comparison below were made on 2026-11-06 on this card at 350 W with no desktop load beyond the usual 0.6 GiB, 385 new tokens per prompt (256 for the sets of 8 and 48), decode tokens/s after the first token. The server and HyperQwen arms are HTTP; the engine arms call the engine directly (a host sync every 8 cycles), so they carry no HTTP, tokenisation or scheduling cost.

**Same-session comparison** (the tree before the kernel work below; arms round-robin on an idle card; mean of rounds, round-to-round spread at most 1.3%; HyperQwen image `ghcr.io/syv-ai/hyperqwen@sha256:00713ba1`, revision `bd6c5e2f`, 2026-10-04, default `GPU_UTIL`; the n of rounds is 4 for HyperQwen and the engine arms, 3 for the server):

| Arm | code | prose | math | 8 chat prompts (median) | 48 English tasks (median) |
|---|---|---|---|---|---|
| HyperQwen DFlash2 (HTTP) | 254.0 | 103.0 | 225.7 | 175.1 | 167.5 |
| qwen-warp server, MTP k=3 + prompt-lookup tail (HTTP) | 180.9 | 112.3 | 180.9 | 157.6 | 150.1 |
| engine-direct MTP k=3 + prompt-lookup tail | 185.3 | 119.0 | 184.2 | 157.5 | 153.6 |
| **engine-direct DFlash2** | **276.4** | **112.0** | **260.9** | 175.3 | 174.2 |
| engine-direct DFlash2, scaled by (server / engine-direct MTP) | about 270 | about 106 | about 256 | 175.4 | about 170 |

The scaled row assumes the server overhead is the same fraction for DFlash2 as for MTP (measured only on MTP: 0.98 / 0.94 / 0.98). On that basis it is level with HyperQwen's DFlash2 on code and prose (+6%, +3%) and 13% ahead on math, and level on both pooled sets. Both engines run the same drafter file and get the same tokens per step on code (5.98 for HyperQwen from its vLLM counters, 5.8 to 6.0 here); the difference is the step: 21.1 ms here, 23.5 ms for HyperQwen (which also ran about 5% higher SM clocks at the same 323 W). On the README prose prompt DFlash2 accepts only 2.4 tokens per cycle and is no faster than MTP (112 against 116 tok/s in the same engine). The engine-direct MTP rows use `engine/spec.py` with the prompt-lookup tail, not the server loop.

**Tokens per cycle** (eager torch drafter, greedy, 11 prompts of 256 tokens, same INT4 target): 4.68 against 3.01 for MTP k=3 (code 5.87 against 3.49, math 5.50 against 3.33, prose 3.81 against 2.66); the in-engine W4 drafter reaches 4.48 against 2.98. Per draft position the acceptance falls from 0.87 to 0.26 over the 7 drafts.

**Cycle** (nsys, code prompt): 21.0 ms = verify T=8 and accept 18.6 ms + drafter 2.50 ms (context append 0.12, noise block 1.40, head, rescoring and selector 0.99); launch idle 0.04 ms. The W4 GEMVs run at 820 to 855 GB/s, 92% to 96% of the 891 GB/s read peak of this card.

**Kernel work in this release** (autonomous campaign, 3 agent lineages, a gate that compares each candidate with a pristine control run back to back, held-out prompts and a 31k-token document the agents never saw; output tokens identical to the previous tree on every prompt; stock tree against release tree, interleaved, same session, 2 rounds, decoding stops at the end token):

| Set | previous tree | this release | ratio |
|---|---|---|---|
| benchmark set: score = sqrt(short geomean x 31k tok/s) | 173.5 | 176.8 | 1.019x |
| short prompts (geomean of 11) | 209.3 | 211.6 | 1.011x |
| 31k-token prompt | 143.9 | 147.7 | 1.026x |
| held-out set: score | 162.3 | 165.2 | 1.018x |
| held-out short prompts (geomean of 8) / 31k-token document | 223.3 / 118.0 | 225.9 / 120.7 | 1.012x / 1.023x |

Read this as +1% to +2.5%, not more. A first version of the benchmark kept decoding to 256 tokens after the end token and reported 1.049x on the benchmark set: one prompt (`translate`, which ends at token 55) degenerated into a repeating loop that an n-gram lookup copied for 2.0x on that prompt. With the loop excluded the lookup gains nothing, so it is off by default (`SMK_DF_LOOKUP=1`); the lookup rows of the campaign are not evidence of speed. What did not work: a draft-vocabulary head (68,864 rows: prose loses 5% to 12% of its tokens per cycle), L2 prefetch of the next GEMV across the cycle (1.4% slower), fusing the residual add and norm into the GEMV tail (11% slower), and a persistent kernel (1.5% to 4%, [earlier measurement](IDEAS.md)).

### In the server (`--drafter dflash2`)

Same session, `serve.py --host-tables --max-model-len 16384` with `--drafter mtp` (MTP k=3 + prompt-lookup tail, the default) and `--drafter dflash2` (everything else identical), the HTTP client of the 2026-11-01 comparison (SSE on `/v1/completions` with token ids, greedy, `return_token_ids`) with ONE change: **the stream stops at the end token** (`ignore_eos` off), so a prompt that ends early is measured over the tokens it produced. Decode tok/s = tokens after the first chunk / (last - first chunk time). Visits in the order mtp, dflash2, dflash2, mtp (a fresh server each); per visit the median of 3 requests (README prompts, 385 tokens) or 2 (the 8 chat prompts of `bench_common`, 256 tokens); the two visits of an arm differ by at most 0.1%. Same RTX 3090, about 0.4 GiB in use before the server started; the card's clock and power were not pinned (SM clock 1,725 MHz at the start of every visit).

| Prompt (bf16 KV) | MTP + tail | DFlash2 | DFlash2 / MTP | tokens |
|---|---|---|---|---|
| README code | 183.9 | **276.7** | 1.51x | 385 |
| README prose | 118.0 | 113.4 | **0.96x** | 385 |
| README math | 183.9 | 254.0 | 1.38x | 385 |
| chat: qa-explain | 161.8 | 184.5 | 1.14x | 256 |
| chat: reasoning | 194.2 | 252.8 | 1.30x | 256 |
| chat: code-from-spec | 189.5 | 319.1 | 1.68x | 256 |
| chat: writing | 162.4 | 195.2 | 1.20x | 107 (EOS) |
| chat: advice | 133.2 | 141.5 | 1.06x | 256 |
| chat: summary-short | 130.6 | 152.2 | 1.17x | 256 |
| chat: sql | 192.9 | 273.2 | 1.42x | 127 (EOS) |
| chat: translate | 89.5 | 165.3 | 1.85x | 56 (EOS) |
| **mean of the 8 chat prompts** | 156.8 | **210.5** | **1.34x** | |
| 12 chat tasks of the 48-task set (median of per-task tok/s) | 138.7 | 148.5 | 1.07x | |

With 4-bit KV (`--kv-cache-dtype int4_per_token_head --host-tables`, same procedure, two visits per arm): code 180.9 -> 271.9 (1.50x), prose 112.2 -> 123.6 (1.10x), math 180.9 -> 257.1 (1.42x), mean of the 8 chat prompts 155.2 -> 209.4 (1.35x), 12 tasks 136.8 -> 151.8 (1.11x). DFlash2 runs on the fp8 and 4-bit caches unchanged (the verify is the same T = 8 step the prompt-lookup tail uses); its prose speed depends on the cache format (113 against 124 tok/s) for a reason that was not investigated. Codebooks in pinned host memory (`--dflash-codebooks host`, the default) against on the GPU: 276.4 / 113.3 / 254.1 against 276.7 / 113.4 / 254.0 tok/s, identical ids, 244 MiB less VRAM.

**Where it is not faster.** README prose (-4% with bf16 KV: 2.4 tokens per cycle against 2.3, and a DFlash2 cycle (21 ms) is longer than an MTP cycle), and answers that **copy a long stretch of the prompt**: the MTP route's prompt-lookup tail accepts 6.6 tokens per cycle on a "rewrite this snippet" prompt where DFlash2 accepts 4.2 (327 against 193 tok/s, test prompt of 3,733 tokens, runner level), and on the synthetic agent trace of the benchmark (a system prompt, then tool-result turns; 28 generated tokens per turn, mostly a tool call that quotes the previous result) DFlash2 decodes 230 tok/s against 334 for the tail (4.95 against 6.75 tokens per cycle; bf16; 4-bit KV 228 against 330). Coding-agent traffic that quotes files is the case where `--drafter dflash2` is **slower** than the default; chat, code generation and math from a short prompt are where it wins. The drafter's own n-gram lookup (`SMK_DF_LOOKUP=1`, the prompt uploaded to its history; runner level, one process per arm) recovers part of it on one prompt (193 -> 262 tok/s against 327), is slower on another (193 -> 175 against 265) and moves 2 of 6 agent turns by about +17%: it stays off. An adaptive route (decide per request or per conversation) is not built.

**Prefill and cached turns** (`--host-tables`, a 12k-token chat prompt, `/reset_prefix_cache` before each, TTFT = request sent to first token chunk, 3 requests per visit, 2 visits per arm): cold 6.85 s (MTP) against 6.94 s with the drafter's window captured during the prefill (+1.3%; 4-bit KV 6.88 s against 6.97 s, +1.3%); a follow-up turn that adds 150 tokens to the cached 12k: 0.181 s against 0.184 s (+3 ms; 4-bit KV 0.184 against 0.187 s). **`cached_tokens` is identical in the two arms** on every request: the 12k follow-up turns (12,136 / 12,138 / 12,131 of 12,305 / 12,307 / 12,300 prompt tokens) and a six-turn agent-shaped chat (system prompt, tool call and tool result per turn: prompt / cached tokens 945/0, 1,690/920, 2,735/1,685, 4,079/2,730, 5,724/4,074, 7,669/5,719). Every turn after the first resumed from the device snapshot and decoded with the block drafter (window covered).

**Exactness.** The two routes are exact for their own verify (the DFlash2 cycle verifies 8 rows, the MTP cycle 4 to 8) and deterministic (two visits give identical ids), but a near-tie in bf16 can resolve differently between the two batch shapes: over the 11 prompts above the ids are identical for 8 (bf16) or 7 (4-bit KV) and diverge between token 58 and 241 for the others (prose, one chat prompt, ...). `tests/test_dflash_server.py` compares each route with its own host-driven reference.

**Not measured**: sampling (greedy only: sampled requests and requests with logit processors use the MTP path), DFlash2 beyond the lengths in [Usage](USAGE.md#the-dflash2-drafter---drafter-dflash2-experimental) (the drafter KV cache is indexed by absolute position, 20 KiB per token; `--profile long` does not fit and the ring buffer that would cap it is not built), the device lm_head table of `DflashEngine` (the server keeps it on the host), the long-context presets of HyperQwen, and any quoting prompt other than the two above (where HyperQwen's lookup drafter reports 381 tok/s).

## 6. Image input

RTX 3090, 2026-11-06. ViT encode (`tests/test_vision.py --time`, median of 10 after 3 warmup calls, bf16, FlashAttention-2, fp32 residual):

| image (patches -> tokens) | 448x448 (784 -> 196) | 1024x768 (3,072 -> 768) | 1920x1080 (8,160 -> 2,040) |
|---|---|---|---|
| weights on the GPU (`--vision device`) | 16.4 ms | 65.1 ms | 220.5 ms |
| weights streamed from pinned host memory (`--vision host`) | 67.9 ms | 74.2 ms | 222.8 ms |
| activation peak | 32 MiB | 125 MiB | 333 MiB |

Through the server (`tests/test_vision_server.py --bench 3`, `--max-model-len 16384`, `--vision auto`, streamed chat, thinking off, a fresh image per run so nothing is cached; median of 3):
time to first token 218 ms / 617 ms / 1,468 ms at 448x448 / 1024x768 / 1920x1080 (the prefill of 223 / 795 / 2,067 prompt tokens included).
Decode after a 1024x768 image: 108.6 tok/s against 123.4 for a text prompt asking for the same kind of story (385 tokens, n = 3). The step is not slower: 19.27 against 19.20 ms per
cycle (server log: 2.09 against 2.37 tokens per cycle, 184 against 162 cycles), so the gap is the draft acceptance of the two different texts.
Text requests are unchanged: README short prompts (385 tokens, greedy, ignore_eos, median of 3, same session) 184.3 / 118.2 / 184.1 tok/s against 0.7.0's 184.9 / 118.4 / 184.5 (ratio
0.997 to 0.998, within noise), with greedy ids identical to 0.7.0.
GPU memory (`memfit_vision.py` in the research notes, difference of the free memory after warmup + a largest-image request + a full-length request, 8,192 tokens, tail on): +888 MiB
with the ViT weights on the GPU, +154 MiB streamed, at 8,100 patches. Not measured: quality of descriptions on real photographs, more than two images per prompt at speed, other sizes.

## 7. Concurrent requests (`--max-num-seqs`)

Measured 2026-11-07 on one RTX 3090 (23.7 GB free), one session, through the server with `experiments/2026-11-06-qwen-warp-concurrent-vision/bench_conc.py` (research repository):
C closed-loop clients, each sending 3 greedy chat requests back to back (thinking off, `max_tokens` 512, stop at the end token; the 8 short code / prose / math prompts of that script,
cycled), streaming. Mixed: every other request asks about a 1024x768 synthetic image of its own (background colour and three shapes; "describe this image"). Aggregate = all output
tokens / wall time; per request = (tokens - 1) / (last chunk - first chunk); TTFT = request sent to first content chunk. One run per cell. v0.7.0 has no image input, so the mixed
rows compare against this version's one-at-a-time server (`--max-num-seqs 1`, the 0.7.0 schedule).

| Server | C = 1 | C = 2 | C = 4 | C = 8 |
|---|---|---|---|---|
| **text only** | | | | |
| v0.7.0 (FIFO), default profile: aggregate tok/s | 155.5 | 154.0 | 155.1 | 151.6 |
| ... TTFT p50 / p95 (s) | 0.05 / 0.05 | 2.79 / 4.74 | 9.98 / 11.7 | 22.3 / 26.9 |
| this version, `--max-num-seqs 1`: aggregate tok/s | 154.5 | 153.0 | 154.3 | 151.4 |
| `--profile multiuser` (5 slots, 15,360 tokens each here): aggregate tok/s | 154.4 | 278.1 | **424.9 (2.74x)** | **440.5 (2.91x)** |
| ... per request tok/s | 162.8 | 149.7 | 114.8 | 97.8 |
| ... TTFT p50 / p95 (s) | 0.05 / 0.05 | 0.08 / 0.13 | 0.09 / 0.21 | 3.62 / 5.13 (3 queue) |
| **mixed text + image** | | | | |
| this version, `--max-num-seqs 1`: aggregate tok/s | 160.8 | 131.3 | 141.6 | 135.2 |
| ... TTFT p50 / p95 (s) | 0.05 / 0.66 | 2.07 / 4.77 | 7.28 / 9.00 | 15.4 / 21.9 |
| `--profile multiuser`: aggregate tok/s | 160.9 | 229.8 (1.75x) | **310.0 (2.19x)** | **318.3 (2.35x)** |
| ... per request tok/s | 172.8 | 139.5 | 98.7 | 83.0 |
| ... TTFT p50 / p95 (s) | 0.05 / 0.63 | 0.08 / 0.66 | 0.63 / 1.22 | 2.94 / 5.21 |

Before the 16-row pass (2026-11-06, same benchmark, text only): `--max-num-seqs 4` 154.9 / 272.2 / 334.1 / 331.8 tok/s at C = 1 / 2 / 4 / 8, `--max-num-seqs 8 --profile long`
150.6 / 260.6 / 317.3 / 314.0.

Single user, `--max-num-seqs 1` against v0.7.0, runs interleaved (v0.7.0, this version, v0.7.0, this version; 2 visits per prompt per run, 512 tokens): code 181.4 / 181.1, prose
120.4 / 120.2, math 188.9 / 188.5 tok/s (v0.7.0 / this version, means of 4; spread +-0.15), 0.2% slower, and the token ids of every run are identical to v0.7.0's. (The tree before
the 16-row pass: 181.5 / 181.4, 120.5 / 120.3, 188.9 / 188.6.)

Rows per weight pass are the limit: the W4 GEMV read the weights once for at most 8 rows (before the 16-row pass below), so the scheduler gives every sequence the most drafts that fit (engine level, greedy, 8 chat
prompts x 256 tokens, host tables, 2 prompt sets per cell, aggregate tok/s):

| sequences B \ drafts k | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| 1 | 57.8 | 105.8 | 138.3 | 159.0 (native k = 3 + tail: 164.9) |
| 2 | 109.6 | 195.3 | 254.3 | 291.1 |
| 3 | 156.8 | 278.0 | | |
| 4 | 199.5 | 348.6 | | |
| 5 / 6 / 7 / 8 | 239.2 / 274.9 / 308.1 / 338.1 | | | |

The largest k that fits is the best for every B. Four sequences with one draft (8 rows) beat eight without drafts (8 rows), which is why 8 concurrent clients get no more than 4.
A cycle of 8 rows costs 23.7 ms against 17.3 ms for 1 row: besides the GEMVs, every extra sequence adds its own DeltaNet and attention launches (~0.8 ms).

**16 rows per weight pass** (2026-11-07, `k_w4` with two n8 token tiles, [Architecture](ARCHITECTURE.md#batched-decode---max-num-seqs)): engine level, greedy, 8 long-answer prompts
(code, prose, arithmetic), 2 prompt sets per cell, 48 cycles each after one warm-up call, aggregate tok/s / ms per cycle:

| sequences B | 8 rows: best k | 16 rows: best k | rows |
|---|---|---|---|
| 3 | k = 1: 277.4 / 20.2 ms | k = 3: **366.8** / 24.8 ms | 12 |
| 4 | k = 1: 346.5 / 21.5 ms | k = 3: **421.5** / 27.6 ms | 16 |
| 5 | k = 0: 239.3 / 20.9 ms | k = 2: **465.5** / 27.0 ms | 15 |
| 6 | k = 0: 274.9 / 21.8 ms | k = 1: **426.7** / 26.2 ms | 12 |
| 7 | k = 0: 308.0 / 22.7 ms | k = 1: **468.1** / 28.0 ms | 14 |
| 8 | k = 0: 337.6 / 23.7 ms | k = 1: **503.6** / 29.8 ms | 16 |

Again the largest k that fits is the best for every B (`K_BY_B` in `server/smk_batch.py`). The W4 GEMVs at 16 rows against 8 (us, isolated launches): DeltaNet q|k|v|z 69 / 54, attention q|k|v 75 / 49,
o 27 / 24, gate|up 131 / 113, down 64 / 56, quantised head 1,038 / 778: about 3 ms of the 6 ms that 16 rows add to a cycle; the rest are the sequences' own DeltaNet, attention and
MTP rows. Batched ids equal the solo replays (45 sequences in 9 shapes of 9 to 16 rows, greedy, seeded sampling, min_p, penalties) and every row-wise kernel is bit-identical for T = 1..16.

**Server, mixed text + image workload** (2026-11-07, autonomous optimisation campaign, `--profile multiuser` = `--max-num-seqs 5 --host-tables --vision device`): 6 closed-loop clients,
84 requests (36 with an image, code / prose / arithmetic / thinking / a ~5k-token document), aggregate output tok/s against the same tree before this work at its benchmark
configuration (`--profile long --max-model-len 16384 --max-num-seqs 8`, ~159 tok/s), 3 runs interleaved with it: **1.358x (+-0.017), ~216 tok/s**. Held-out workload (8 clients,
other image sizes and prompts, a 12k-token document, 2 seeded sampled requests): 1.20x. Answer accuracy 1.0 on the arithmetic items; 97.6% of the greedy answers are identical to the
reference tree's, the rest differ at bf16 near-ties (other draft depths, see Outputs below). The steps, each measured on the same workload: 16 rows per weight pass with drafts up
to 16 rows (1.31x), one launch per layer for the DeltaNet / attention of all sequences and a prefill pause every 2,048 rows (+1 to 4%), a decode cap of 5 per sync group (6 slots:
1.336x at TTFT p95 3.27 s against 1.272x / 3.17 s without it), asynchronous snapshot copies (13 ms -> 0.6 ms of every prompt over 240 tokens on the request path).
Outputs: with the default schedule a request's draft depth changes with the number of running requests, and in these runs 4 of 9 (`--max-num-seqs 4`) and 6 of 9 (`--profile long`, 8 slots)
requests at C = 8 diverged from the same prompt at C = 1 (at token 83 or 155 of 512); with the depth fixed (`SMK_BATCH_K`, the tests) every request equals its solo run exactly.
These are bf16 near-ties, measured at engine level (`margins.py` in the same experiment directory; 20 prompts x 512 greedy tokens, the native path against batched cycles of 0, 1 and 3
drafts, the prose prompt diverges at token 83 as in the server runs): 20 of 60 answers diverged before their end, and at every first divergent token the reference's two best
exact logits were 0.011 to 0.14 apart (median 0.044; 0.08th to 1.9th percentile of the gap over all 16,432 answer positions, whose median is 6.6), and the batched run's token was the
reference's second best every time. The two runs agree bit for bit on the embedding and on layer 0, and differ from a DeltaNet layer on (layer 1 at 0 / 1 drafts; layer 4 at 3 drafts,
where only the prompt-lookup-tail cycles differ) by 0.0002 to 0.004 (rounding), which grows to at most 0.13 to 0.41 at the final norm output (|x| up to 12 to 40); the top-2 gap
moves between the two runs by 0.028 (median) and 0.17 (p99) at positions that do not flip.

## 8. Reproducing

| What | How | Takes |
|---|---|---|
| Every test in the Makefile list: KV-cache perplexity guard, host tables, logit processors, tool-call and sampler checks, the server end to end, start-up configurations | `make test` (needs the checkpoint and the GPU) | long: it starts the server several times; no total is given here |
| DFlash2 acceptance and exactness | `python tests/test_dflash.py` (needs the W4A16 drafter, `$DFLASH_W4`) | minutes |
| DFlash2 in the server: tap capture, router, drafter window across turns and the host tier | `python tests/test_dflash_server.py` (drafter + GPU, about 3 minutes) | minutes |
| The part that needs no GPU | `scripts/check_repo.sh`, `uv run tests/test_tool_schema.py`, `uv run tests/test_prefill_plan.py` | seconds |
| The tables in sections 2 to 4 | not regenerated by anything here: the long-context sweeps were done with throwaway harnesses against this engine and the HyperQwen container, which are not part of this repository | n/a |

The tests check the cache, the host tables, the sampling parameters, the tool-call path, the start-up configurations and the server with only the checkpoint; they are regression guards, not a replacement for the measurements above. `tests/test_startup_configs.py` re-checks that the documented configurations start and serve; `tests/test_tool_call_rate.py` gates a tool-call rate of 70% over 40 requests.
