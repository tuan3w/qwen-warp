# Usage

How to call the server, which flags matter, and how memory and context fit together. New here? Start with [Getting started](GETTING_STARTED.md). Back to the [README](../README.md).

Contents: [Endpoints](#endpoints) | [Calling it](#calling-it) | [Thinking and reasoning output](#thinking-and-reasoning-output) | [Sampling parameters](#sampling-parameters) |
[Seeds and reproducibility](#seeds-and-reproducibility) | [Tool calls](#tool-calls) | [Multi-turn prefix reuse](#multi-turn-prefix-reuse) | [Server flags](#server-flags) | [Memory and context](#memory-and-context) |
[Using it from a coding agent](#using-it-from-a-coding-agent)

## Endpoints

The server is vLLM's own OpenAI frontend running on top of this engine, so the usual routes exist. The served model name is `swift`.

| Route | Use |
|---|---|
| `GET /health` | HTTP 200 when ready (what the Docker health check calls) |
| `GET /v1/models` | lists `swift` |
| `POST /v1/chat/completions` | chat, streaming, tool calls, reasoning output |
| `POST /v1/completions` | raw text or token-id prompts |

One request is decoded at a time; further requests wait in order, and a client that disconnects aborts its request. Add `--api-key KEY` to require a key.

## Calling it

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "swift",
  "messages": [{"role": "user", "content": "Explain the CAP theorem in three sentences."}],
  "max_tokens": 2000}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")      # the key is only checked if you started the server with --api-key

stream = client.chat.completions.create(model="swift", stream=True, max_tokens=2000,
                                        messages=[{"role": "user", "content": "Write a quicksort in Rust."}])
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Streaming sends a chunk about every 41 ms in our test (`--stream-cycles` sets how many speculative cycles run between two sends).

## Thinking and reasoning output

The model reasons before it answers by default, and the server splits that thinking from the answer (the `qwen3` reasoning parser). Two consequences:

| What you see | Why | What to do |
|---|---|---|
| An empty or truncated `content` | the thinking tokens count against `max_tokens` (our short test spent 76 of 120 tokens thinking) | raise `max_tokens` |
| You do not want the thinking | it costs latency and tokens | send `"chat_template_kwargs": {"enable_thinking": false}` with the request |

```python
client.chat.completions.create(model="swift", messages=[...], max_tokens=400,
                               extra_body={"chat_template_kwargs": {"enable_thinking": False}})
```

The thinking text comes back in the field **`reasoning`** of the message (vLLM 0.30.0; read it from the raw JSON), not in `content`.

`reasoning_effort` is accepted by the chat template of the checkpoint. The lengths below were measured on a stock vLLM server of the same model; the behaviour of the values was also checked on this server on 2026-11-04 (`none` gave 0
reasoning tokens, `low`, `medium` and `xhigh` were accepted and reasoned, `high` returned HTTP 400), in both the native server and the Docker container:

| `reasoning_effort` | Result |
|---|---|
| `xhigh` (the default), `medium`, `low` | accepted. It is only a hint in the prompt: the median reasoning length over 18 requests per setting (6 prompts x 3, T 0.6) was 290 (`xhigh`), 269 (`medium`), 279 (`low`) and 256 (unset) tokens, so no measurable shortening |
| `none` | thinking off (vLLM maps it to `enable_thinking=false`, the same as the `chat_template_kwargs` above); median 0 reasoning tokens |
| `high`, `max`, `minimal` | HTTP 400 ("Unexpected reasoning effort ... Supported types are xhigh (default), medium, and low") |

A hard cap on the thinking length (`thinking_token_budget`) is not supported (HTTP 400); it is listed as an [idea](IDEAS.md#a-thinking-token-budget).

## Sampling parameters

The engine implements sampling itself (exact speculative sampling), so only some parameters exist. Anything unsupported returns **HTTP 400** that names the parameter, rather than being ignored.
Without sampling parameters the model's own defaults apply (T = 1, top_p 0.95, top_k 20). Field names, ranges and semantics are vLLM's; vLLM validates the ranges first.

| Supported | What it does | Limits |
|---|---|---|
| `temperature` | 0 = greedy | at 0, `top_k`, `top_p` and `min_p` are ignored (as in vLLM) |
| `top_p` | nucleus filter | |
| `top_k` | keep the k most likely tokens | **1 to 64**; `0` and `-1` ("no filter") behave as 64: the sampler keeps its 64 best candidates and ignores the probability outside them (not quantified); **above 64 is HTTP 400** (`param: top_k`). At temperature 0 it is ignored, so any value is accepted |
| `min_p` | drop tokens with probability below `min_p` x the best one | 0 to 1 |
| `presence_penalty` | subtract p from the logit of every token generated at least once | -2 to 2; counts generated tokens only, not the prompt |
| `frequency_penalty` | subtract f x (times the token was generated) | -2 to 2; generated tokens only |
| `repetition_penalty` | divide a positive logit (multiply a negative one) by r for every token in the prompt or generated, once per token | greater than 0 |
| `logit_bias` | add a value to the logit of the listed tokens | at most 1,024 entries; ids in the vocabulary (0 to 248,319); values clamped to [-100, 100] |
| `min_tokens` | mask the stop tokens (EOS and `stop_token_ids`) until that many tokens were generated | at most `max_tokens`; stop **strings** are not affected (as in vLLM); with `allowed_token_ids` made only of stop tokens it is HTTP 400 |
| `allowed_token_ids` | every other token is masked | at most 248,320 ids, each in [0, 248,320) |
| `seed` | reproducible sampling, see [below](#seeds-and-reproducibility) | |
| `max_tokens`, `stop`, `stop_token_ids`, `ignore_eos` | `stop` strings cut the text before the match | |
| `n` | several choices, each its own request | choice i uses `seed + i` |
| `chat_template_kwargs.enable_thinking`, `reasoning_effort` | thinking on/off, a prompt hint | [above](#thinking-and-reasoning-output) |

**Order of application**, per generated token, on the logits of the output head: `allowed_token_ids`, `min_tokens`, `logit_bias`, `repetition_penalty`, `frequency_penalty` and `presence_penalty`,
then temperature, `min_p`, `top_k`, `top_p`, then the draw. Greedy decoding with any of the processors before the temperature is the argmax of the processed logits. This is the order and the semantics of vLLM 0.30's v1 sampler,
checked against a torch reference (`tests/test_logit_procs.py`, `tests/test_proc_model.py`).

| Not supported (HTTP 400, names the parameter) | Why it is hard here |
|---|---|
| `logprobs` / `top_logprobs`, `prompt_logprobs`, `logprob_token_ids` | the quantised head returns exact logits only for its top-64 candidates; a full log-softmax needs an extra bf16 pass over the 248k vocabulary per row |
| `bad_words` | multi-token sequences need a matcher over the generated history inside the verify rows, draft tokens included |
| `structured_outputs` (and the `response_format` JSON modes, which become it) | a grammar state machine per token and per verify row on the device; a host round trip per token would defeat the CUDA-graph cycle |
| `thinking_token_budget` | not implemented; a sketch is in [Ideas](IDEAS.md#a-thinking-token-budget) |

**Speculative decoding and the processors.** Every processor is applied to every verify row of a speculative cycle, with the draft tokens before the row counted as generated, so the output
distribution is exact (`tests/test_spec_procs.py`: chi-square on the emitted stream, with a broken variant that must fail). Stock vLLM refuses `min_p` and `logit_bias` together with speculative decoding; here they work.
The draft proposals themselves ignore the processors, which keeps the acceptance exact but can lower the number of accepted tokens.

**What the processors cost** (RTX 3090, default profile; single in-process measurements):

| Setting | Cost |
|---|---|
| No processor in the request | none measured: 19.52 ms per speculative cycle (k = 3, greedy) before the processors existed, 19.49 ms now; server decode 193.5 and 193.8 tok/s |
| One processor (a penalty, `logit_bias` with 8 entries, `min_tokens`) or all of them together | below the measurement noise (about 0.1 ms per cycle, 0.5%): 19.50 to 19.65 ms against 19.67 ms without |
| `allowed_token_ids` with 5,000 to 10,000 ids | **+2.2 to 2.3 ms per cycle (+11%)**; 100 ids cost nothing |
| Fewer accepted draft tokens, T 0.7, top_k 20: `presence_penalty` + `frequency_penalty` + `repetition_penalty` + `min_p` + `logit_bias` together | 3.16 -> 2.75 tokens per cycle, **160 -> 136 tok/s (-15%)**; `repetition_penalty` 1.1 alone 2.90 tokens per cycle (150 tok/s); `min_p` 0.05 alone 3.11, `presence_penalty` 0.5 alone 3.09 |
| Greedy with penalties | tokens per cycle 3.02 -> 2.89 to 3.04 |
| `allowed_token_ids` with 10,000 ids | 3.02 -> 4.39 tokens per cycle (225 tok/s): a restricted vocabulary is easy to draft |

The acceptance numbers come from the server log over 8 prompts x 200 tokens; the script is not in the repository. Running the draft head's logits through the processors would recover part of the loss; it is not done.

## Seeds and reproducibility

| Case | Behaviour |
|---|---|
| Same prompt, same sampling parameters, same `seed`, same server configuration | the same tokens, whatever the cache state: a cold prefill, a request resumed from an earlier turn's snapshot, a repeat, `--snapshots 0`, after a restart, with `--stream-cycles 1` (`tests/test_server_seed.py`: 24 checks, 9 repetitions per scenario) |
| Different `--num-speculative-tokens` or `--prompt-lookup` | **a different realisation**: speculative sampling is exact in distribution, not in the drawn tokens (measured: seeded k = 3 and k = 0 differ from token 1 to 11; tail on and off from token 4 to 25) |
| Different build or GPU | not promised: the realisation is tied to the bf16 kernels |
| `n` > 1 | choice i is sampled with `seed + i`; choice i equals the single request with `seed + i`, and the choices differ |
| No `seed` | a random seed per request; not reproducible |
| Greedy (`temperature` 0) | deterministic for a fixed configuration and independent of the cache. Across `--prompt-lookup` and `--num-speculative-tokens` the ids were identical on the prompts we compared, but that is not guaranteed on long prompts: the prompt-lookup verify widths and the plain k = 3 cycles are not bit-identical, and on a long text a greedy id can flip at a bf16 near-tie (2 of 3 long texts in one test) |

## Tool calls

Tool calling works out of the box (`qwen3_coder` parser with automatic tool choice), including the round trip of sending the tool result back (`tests/test_server_smoke.py`). **It is not always well-formed**: the
model sometimes writes something other than a tool call where one belongs, and the client then gets no tool call. Check `tool_calls` and retry:

```python
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
for _ in range(3):                                  # a well-formed call is likely but not certain: see the table below
    r = client.chat.completions.create(model="swift", messages=[{"role": "user", "content": "What is the weather in Hanoi?"}],
                                       tools=tools, max_tokens=800)
    if r.choices[0].message.tool_calls:
        break
call = r.choices[0].message.tool_calls[0]          # name "get_weather", arguments '{"city": "Hanoi"}'
```

What was measured: tool `get_weather(city: string)`, user "What is the weather in Hanoi?", thinking on, the model's default sampling, `max_tokens` 800; success = a well-formed call with a known name and
the required parameter. Rows 1 to 4: in-process engine runs, n = 150 per cell (the 95% intervals reach 4 to 8 points to either side). The last two rows: through the server on `--profile long`, n = 200 and 100 (Wilson 95% intervals in brackets):

| KV cache | schema as most clients send it (`type`, `properties`, `required`) | keys sorted (`properties`, `required`, `type`; what the server renders by default) |
|---|---|---|
| bf16 | 91% (137/150) | 93% (139/150) |
| fp8 | 88% (132/150) | not measured |
| 4-bit, previous format (0.3.0, `--profile long`) | **39% (58/150)** | **87% (130/150)** |
| 4-bit, previous format, through the server | 46% (46/100) [37, 56] | 94% (94/100) [88, 97] |
| 4-bit, rotated format with exact sink and ring (this version), through the server | **92% (183/200) [87, 95]** | 88% (88/100) [80, 93] |

- The chat template prints the tool schema in the key order the client sent, and the model's choice at the start of a tool call depended on that order, much more so on the previous 4-bit format. `--sort-tool-schema on` (the default)
  sorts the top-level keys of every tool's `function.parameters` before the template renders them; nothing else changes (not your JSON, not the parser, not the arguments, nested objects untouched).
  `--sort-tool-schema off` renders the schema exactly as sent.
- Through the server, 40 identical requests: 18/40 before the sorting on `--profile long`; with it 37/40 on the default profile and 40/40 on `--profile long` (`tests/test_tool_call_rate.py`, which fails below 70%).
- Two other schemas (a 3-parameter tool with an enum and a required subset, and a nested-object tool) gave 100% in every cell, in both key orders (n = 100 on bf16, n = 60 on 4-bit): the minimal one-parameter
  schema is the fragile one.
- The sampler is not the cause: on 219 real positions its draws match the exact bf16 reference distribution (`tests/test_sampler_dist.py`). Greedy decoding on the 4-bit cache gave 99% (149/150, schema as sent).
- Even on bf16 the rate (93%) was below the 98% of a reference vLLM service we compared against (its prompt had the parameter keys sorted); the cause was not isolated ([Ideas](IDEAS.md#why-bf16-kv-is-below-a-reference-service-on-tool-calls)).
- **Not measured:** coding-agent tool sets (six tools or more), second turns of a tool conversation, fp8 with the sorted order, the fp8 profile with host tables, top_p / min_p as a mitigation. With the rotated cache the schema order no longer changes the rate beyond noise for this schema (92% against 88%, intervals overlap); sorting stays on by default. How
  well real agent loops succeed on any cache is unknown.

Constrained decoding (`structured_outputs`) is not available; the parser reads the tool call from the model's text.

## Multi-turn prefix reuse

Follow-up turns of a chat do not re-prefill the whole history. The engine snapshots its recurrent state while prefilling and resumes from the longest cached prefix:

| Detail | Value |
|---|---|
| How you can see it | `usage.prompt_tokens_details.cached_tokens` in the response (0 on a cold prompt) |
| Snapshots kept | 16 by default, in total over the GPU and the host cache below (`--snapshots N`, 151 MB of pinned host RAM each, 183 MB with the 4-bit cache, 0 = off) |
| Where snapshots are taken | at the last two cut points of the prefill plan, plus (`--system-checkpoint on`, default) the first chat-turn boundary after 1,024 tokens, i.e. the end of the system prompt, so parallel agents that share a system prompt and differ after it resume there. Cuts fall every 4,096 tokens, at a chat-turn boundary more than 240 tokens after the previous cut, and just before the last token when more than 240 tokens follow the last cut. A long system prompt or document is therefore reused across different user questions |
| Measured | 3-turn chat over a 1,529-token document, per request including 60 generated tokens: 1.28 s, 0.54 s, 0.41 s (turns 2 and 3 resumed at 1,524 cached tokens; the old cut points gave 1.31 s, 0.60 s, 0.39 s) |
| Several conversations | the GPU holds ONE conversation. Before a request that does not continue it overwrites it, the engine copies its KV rows (up to its last snapshot) and its snapshots to pinned host memory; a later request that continues a stored conversation copies them back and prefills only the new tokens. Budget `--prefix-cache-gb` (default 6 GiB, least recently used conversation goes first, 0 = one conversation only); about 21 KB of host RAM per token with the 4-bit KV cache, 37 KB with fp8, 70 KB with bf16 (measured: 126, 221 and 418 MB for a 6,000-token conversation; the MTP layer's KV is included; the pinned allocator rounds buffers up, so plan for up to ~1.6x). Lineages whose last snapshot is below 1,024 tokens (helper and title requests) are not stored. Measured with the 4-bit cache: spill 31 ms and restore 46 ms at 15,000 tokens, 123 ms and 135 ms at 60,000; a request resumed from the host prefills 0.17 s longer than one that continues the GPU conversation at 15k tokens. A resumed request is bit-identical to a cold one (`tests/test_lineage_cache.py`, all three KV dtypes) |
| Not reused | a repeated prompt with 240 tokens or fewer after its last cut: it is prefilled again (35 to 39 ms, the same time as its first run) |
| To benefit | keep the earlier messages byte-identical and append only; changing an earlier message breaks the prefix |

The output of a request never depends on whether the prefix came from the cache: a resumed prefill is bit-identical to a cold one ([seeds](#seeds-and-reproducibility)).

## Server flags

Every stock `vllm serve` frontend flag works (`--host`, `--port`, `--api-key`, `--served-model-name`, `--chat-template`, `--max-model-len`, ...). The engine adds these:

| Flag | Default | What it does |
|---|---|---|
| `--profile {default,long}` | `default` | `long` = 4-bit KV cache + `--host-tables` + a default `--max-model-len` of 196,608. Explicit flags win over the profile |
| `--kv-cache-dtype` | `auto` (bf16) | `fp8`, or `int4_per_token_head` for the 4-bit cache |
| `--max-model-len` | 20,480; 28,672 with `--prompt-lookup 0`; 196,608 with `--profile long` | context length the engine is sized for. The server refuses a length that does not fit the GPU memory free at start ([below](#memory-and-context)) |
| `--host-tables` | off | keep the bf16 embedding and rescoring head (4.7 GiB) in pinned host memory: bit-identical decode, about 0 to 2% slower, about 4.8 GiB more VRAM for the cache |
| `--prompt-lookup {0,1,auto}` | `auto` | an n-gram draft tail that needs about 0.56 GiB more VRAM (0.67 GiB with `--host-tables`); `auto` turns it on only if the length leaves 256 MiB of the free memory next to it (bf16 KV up to about 23k, fp8 up to 42k, 4-bit up to 68k, any length on 4-bit with `--host-tables`), and the log says what it chose. `1` forces it (refused like any other length that does not fit) |
| `--num-speculative-tokens {0..3}` | 3 | MTP drafts per step; 0 = plain decode (the prompt-lookup tail needs 3) |
| `--drafter {mtp,dflash2}` | `mtp` | `dflash2` (experimental): greedy requests without logit processors are decoded by the in-engine DFlash2 block drafter, everything else by the MTP path ([below](#the-dflash2-drafter---drafter-dflash2-experimental)); needs the drafter checkpoint and practically `--host-tables` |
| `--dflash-path DIR` | `$DFLASH_W4`, else `~/llm-lab/models/dflash2-w4a16` | the W4A16 compressed-tensors DFlash2 checkpoint (`model.safetensors`) for `--drafter dflash2` |
| `--dflash-codebooks {host,device}` | `host` | the drafter's two 121 MiB selector codebooks in pinned host memory (same speed and ids, 244 MiB less VRAM) or on the GPU |
| `--sort-tool-schema {on,off}` | `on` | render the keys of every tool's `function.parameters` sorted, which gives well-formed tool calls more often ([tool calls](#tool-calls)) |
| `--snapshots N` | 16 | multi-turn prefix snapshots (GPU conversation + host cache) |
| `--prefix-cache-gb G` | 6 | pinned host RAM for the KV rows of other conversations (0 = off) |
| `--system-checkpoint on\|off` | on | one extra snapshot at the end of each long system prompt |
| `--stream-cycles N` | 2 | speculative cycles per host sync, i.e. streaming granularity (one cycle is about 24 ms and emits 1 to 4 tokens) |

Set `SWIFT_MODEL` (or pass `--model`) when the weights are not in `~/models/swift-1.5-int4`. With Docker these become `PROFILE`, `EXTRA_ARGS`, `MODEL_DIR` and `PORT` ([table](GETTING_STARTED.md#3a-docker)).

**vLLM version.** The environment pins `vllm==0.30.0` (`pyproject.toml`, `uv.lock`). The engine-core client (`server/smk_core_client.py`) and the tool-schema hook (`server/tool_schema.py`, which wraps
`vllm.renderers.hf.safe_apply_chat_template`) use vLLM internals of that version; other versions are untested (a newer one, 0.33.0, exists). Install from the lock file rather than upgrading vLLM by hand.

## Concurrent requests (`--max-num-seqs`)

`--max-num-seqs 1` (the default) decodes one request at a time and queues the rest, exactly as before. `--max-num-seqs S` with S = 2..8 gives the engine S sequence slots on one set of
weights and decodes the running requests together: every cycle reads the weights once for the rows of all of them, up to 16 rows ([Architecture](ARCHITECTURE.md#batched-decode---max-num-seqs)).

For several users at once (chat clients, agents running in parallel, image and text mixed), use the measured configuration:

```bash
python server/serve.py --profile multiuser --port 8000     # = --max-num-seqs 5 --host-tables --vision device, bf16 KV
```

It picks the longest `--max-model-len` per sequence that the free memory holds (about 15k tokens with 23.2 GB free, at most 28,672); pass `--max-model-len` (or `--max-num-seqs`,
`--vision`) to override. Measured on 6 closed-loop clients sending text and image requests: 1.36x the aggregate tok/s of the one-at-a-time server ([Performance](PERFORMANCE.md#7-concurrent-requests---max-num-seqs)).

| Running requests | How a cycle decodes them |
|---|---|
| 1 | the single-sequence path: MTP k = 3 + the prompt-lookup tail (the speed of `--max-num-seqs 1`) |
| 2 | batched, 3 drafts each (8 rows) |
| 3, 4 | batched, 3 drafts each (12 / 16 rows) |
| 5 | batched, 2 drafts each (15 rows) |
| 6 to 8 | 5 of them per sync group, round robin (least recently decoded first), 2 drafts each: the extra slots let a new request prefill and stream its first token without queueing |

- More requests than slots queue FIFO. A new request is prefilled while the others keep streaming: every ~2,048 tokens of prefill work the running requests get one sync group (`--stream-cycles`), so a long prompt pauses them for about 1 s at a time instead of for its whole prefill.
- Every request keeps its own sampling parameters, seed, logit processors, stops, `max_tokens` and abort; prefix reuse (`cached_tokens`, the host prefix cache) works per slot: a request goes to the free slot whose conversation it continues, else the least recently used free slot, whose conversation is spilled to the host cache first.
- Outputs: in the default mode a greedy request's ids can differ between batch sizes at a bf16 near-tie. A request's tokens are the same whether other requests run next to it as long as its draft depth stays the same, but the scheduler changes the depth with the number of running requests (table above), and the DeltaNet kernel sums in another order at another verify width, so a token whose two best logits are almost equal can flip (as with `--num-speculative-tokens`). Measured on 20 prompts x 512 tokens, the native path against batched cycles of 0, 1 and 3 drafts: 20 of 60 answers diverged, always where the top-2 gap of the exact logits was 0.011 to 0.14 (the median gap over all positions is 6.6; the reduction-order noise of that gap between the two runs has a median of 0.028 and a p99 of 0.17); seeded sampling draws other tokens at another depth. The exact mode `SMK_BATCH_K=k` (tests) fixes the depth at every batch size and removes this: a request's ids are then exactly those it gets alone.
- `--drafter dflash2` needs `--max-num-seqs 1`.
- Memory: every slot has its own KV cache at `--max-model-len` (368 MiB + 68 / 36 / 20.5 KiB per token with bf16 / fp8 / int4 KV, on top of 19.9 GiB, or 15.1 GiB with `--host-tables`). Without `--max-model-len` the server picks the longest length per sequence that fits next to image input (the ViT with its weights in host memory; with `--vision off` the lengths are 1,024 to 4,096 tokens longer); a configuration that does not fit is refused with the largest `--max-num-seqs` at that length and the largest length at 2, 4 and 8 slots. With 23.7 GB free (a desktop holding 0.3 GB), from the memory model:

| Cache | 2 slots | 4 slots | 8 slots |
|---|---|---|---|
| bf16 (default profile) | 18,432 | 6,144 | 1,024 (no room for image input) |
| fp8 | 34,816 | 12,288 | 1,024 |
| bf16 + `--host-tables` | 54,272 | 24,576 | 9,216 |
| int4 + `--host-tables` (`--profile long`) | 167,936 | 77,824 | 29,696 |

Speed: [Performance](PERFORMANCE.md#7-concurrent-requests---max-num-seqs).

## Memory and context

The KV cache is the only thing that grows with context. The server sizes the engine for `--max-model-len` up front and **measures the free GPU memory at start** (`torch.cuda.mem_get_info`): it compares
that with a model of what the engine needs and refuses to start if the length does not fit, instead of dying with an out-of-memory error after loading the weights.

| KV cache | Cache per token (16 full-attention layers, 4 KV heads x 256 dims) | 262,144 tokens | Largest `--max-model-len` measured to start and serve: tail on / tail off (`--prompt-lookup 0`) | The same with `--host-tables` |
|---|---|---|---|---|
| bf16 | 64 KiB | 16 GiB | 26,624 / 34,816 | 94,208 / 105,472 |
| fp8 | 32 KiB | 8 GiB | 48,128 / 62,464 | 172,032 / 184,320 |
| 4-bit | 16.5 KiB | 4.1 GiB | 79,872 / 104,448 | **262,144** (the model maximum), with or without the tail |

"Serve" means a prompt of (`--max-model-len` - 64) tokens plus 64 generated tokens answered with HTTP 200 and the server alive afterwards (greedy). Each number is the largest tested length that did; the first tested length
above it failed, 1,024 tokens higher, except for four cells: bf16 + `--host-tables` + tail (94,208; 97,280 failed, a 3k-wide bracket), fp8 + `--host-tables` + no tail (184,320; 190,464 failed, 6k wide)
and the two 4-bit + `--host-tables` cells, which reach the model maximum (their real edge was not measured).

What the server accepts, and where `auto` switches the tail on, on a card with 23,228 MiB free (the desktop and the CUDA context hold the rest):

| KV cache | `auto` turns the tail on up to | Accepted with the tail | Accepted without it |
|---|---|---|---|
| bf16 | 23,552 | 25,600 | 33,792 |
| fp8 | 41,984 | 47,104 | 61,440 |
| 4-bit | 68,608 | 76,800 | 100,352 |
| bf16 + `--host-tables` | 92,160 | 94,208 | 104,448 |
| fp8 + `--host-tables` | 165,888 | 169,984 | 188,416 |
| 4-bit + `--host-tables` | 262,144 | 262,144 | 262,144 |

These numbers move with the free memory: a headless card has about 0.6 GiB more (predicted by the model, never run), a browser with GPU acceleration has less. The accepted lengths sit a little below the measured edges
because the server keeps at least 64 MiB free at the full length. The model is `needed MiB = fixed + KiB per token x length`, with the KiB per token 72 (bf16), 40 (fp8), 24.5 (4-bit: the cache plus 8 KiB for the MTP layer's KV and
the prefill scratch) and the fixed part 20,740 MiB (21,312 with the tail; 15,800 and 16,488 with `--host-tables`); it reproduces every measured edge within +-40 MiB (`vram_need_mib` in `server/smk_core_client.py`).

**Defaults.** Measured on the same card, with the memory in use after a request near the limit (`nvidia-smi`, desktop included; PyTorch's allocator keeps the scratch it used, so this is not a headroom meter):

| Start | `--max-model-len` | Tail | In use after a near-limit request |
|---|---|---|---|
| no flags | 20,480 | on | 23,901 MiB |
| `--prompt-lookup 0` | 28,672 | off | 23,807 MiB |
| `--kv-cache-dtype fp8` | 20,480 | on | 23,263 MiB |
| `--kv-cache-dtype fp8 --max-model-len 53248` | 53,248 | off (does not fit with it) | 23,871 MiB |
| `--profile long` | 196,608 | on | 22,331 MiB (a 196,544-token prompt) |
| `--profile long --max-model-len 262144 --prompt-lookup 0` | 262,144 | off | 23,227 MiB (a 262,080-token prompt) |
| `--profile long --max-model-len 262144` | 262,144 | on (468 MiB stay free) | 23,899 MiB (a 262,080-token prompt plus 64 generated tokens, 277 s) |

The defaults sit 18% to 23% below the measured edges of their configuration (28,672 against 34,816 without the tail, 20,480 against 26,624 with it). Time to ready: 19 to 20 s with the tables on the GPU, 24 to 35 s with
`--host-tables` (pinning 4.7 GiB), 26 to 35 s for `--profile long`. Prefilling a near-limit prompt takes 179 s at 196,544 tokens and 275 s at 262,080 tokens.

**When the length does not fit**, no weights are loaded and the server exits after about 10 s with an error like this (free memory and list are from the card above):

```
RuntimeError: swift-mk: --max-model-len 36,864 with bf16 KV needs about 23332 MiB of GPU memory but only 23228 MiB are free on the GPU (a desktop and other programs hold the rest).  Largest --max-model-len that fits the free memory:
  bf16 KV (default): 23,552 with the prompt-lookup tail, 31,744 without (--prompt-lookup 0)
  fp8 KV --kv-cache-dtype fp8: 41,984 with the prompt-lookup tail, 56,320 without (--prompt-lookup 0)
  int4 KV --kv-cache-dtype int4_per_token_head: 68,608 with the prompt-lookup tail, 93,184 without (--prompt-lookup 0)
  int4 KV + host tables --profile long: 262,144 with the prompt-lookup tail, 262,144 without (--prompt-lookup 0)
```

If memory is taken between the check and the engine's allocation, the engine's own out-of-memory error is wrapped with the same table. A card busy enough that the default length does not leave 256 MiB with the tail
starts with the tail off (an INFO log line says so). **Not caught:** another program taking GPU memory after the start; a request can then fail with an out-of-memory error.

Rules of thumb:

- Want the full 262,144 tokens? `--profile long --max-model-len 262144 --prompt-lookup 0` (or without `--prompt-lookup 0`: the tail then also fits at that length with 468 MiB free, and costs about 0.7 GiB).
- Want the best short-context speed and do not need length? The defaults.
- A 24 GB desktop card also drives a monitor (about 0.6 GiB); the numbers above were measured that way (589 MiB in use before the server started).
- Context length trades against VRAM only: it does not change output quality except through the 4-bit cache's own loss ([Quality](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache)) and the tool-call numbers above.
- Not measured: cards other than this 3090 with a 0.6 GiB desktop, a headless card, `--num-speculative-tokens` below 3.

## The DFlash2 drafter (`--drafter dflash2`, experimental)

`--drafter dflash2` adds the 5-layer block drafter of `incoai/Qwen3.8-27B-DFlash2` (W4A16) to the engine. `--drafter mtp` (the default) is byte-identical to the previous release. What it does, measured through the server ([Performance, section 5](PERFORMANCE.md#5-dflash2-block-drafter-experimental)):

| Request | Route | Decode tok/s against `--drafter mtp` (bf16 KV) |
|---|---|---|
| greedy, no logit processor, drafter window on the GPU | DFlash2 | code +51%, math +38%, 8 chat prompts +34%, 12 pooled chat tasks +7%, prose -4%, **copy-heavy agent turns -31%** |
| `temperature` > 0, `presence`/`frequency`/`repetition_penalty`, `logit_bias`, `min_tokens`, `allowed_token_ids` | MTP path, as before | unchanged |
| greedy, window not on the GPU (a host-cache restore whose entry does not hold the rows the prompt needs, e.g. a resume at an earlier snapshot or a prompt shorter than the stored lineage) | MTP path | unchanged |

Ids: both routes are exact for their own verify and deterministic, but may resolve a bf16 near-tie differently (8 of 11 benchmark prompts identical; the rest diverge after token 58 to 241). Prefix reuse is unchanged: `cached_tokens` is identical to `--drafter mtp` on the 12k-token and agent-shaped traces, a cold 12k prefill takes 1.3% longer, a cached follow-up turn 3 ms. The host prefix cache carries the drafter's window (40 MiB of pinned memory per entry, counted in `--prefix-cache-gb`).

**Memory.** The drafter takes 1.15 GiB (1.4 GiB with `--dflash-codebooks device`) plus **20 KiB per context token** (absolute positions; no ring buffer yet) on top of the MTP engine with the prompt-lookup ring (the tail is free with it). Measured as the difference to `--drafter mtp` after a request at the full length: +1,872 / +2,352 MiB at 24,576 / 49,152 tokens (bf16, device codebooks), +2,670 MiB at 65,536 with fp8 KV, +3,310 MiB at 98,304 with 4-bit KV. Without `--host-tables` it fits only a few thousand tokens, so the server refuses it with the usual table of lengths (`vram_need_mib`, `server/smk_core_client.py`). Largest `--max-model-len` the model accepts on a card with 23,228 MiB free (all with `--host-tables`, codebooks on the host), each of which started and served a 32,768-token prompt plus 64 tokens on the measuring card (23.7 GiB free, 0.1 GiB desktop):

| KV cache | accepted | in use after the request (measuring card) | one step above (measuring card) |
|---|---|---|---|
| bf16 | 61,440 | 23,669 MiB | 66,560 starts, a request fails with HTTP 500 (out of memory) |
| fp8 | 94,208 | 23,669 MiB | 102,400 still serves (24,045 MiB in use); the Runner measurement ran out of memory at 102,400 and 100,352 was the last that served there |
| 4-bit | 125,952 | 23,655 MiB | 138,240 serves (24,085 MiB in use); 143,360 ran out of memory |

`--profile long` (196,608 tokens) does not fit with the drafter and is refused with the largest length that does (the ring buffer is not built); the start-up check for `--drafter dflash2` without `--host-tables` or at an absurd length refuses within seconds (`tests/test_startup_configs.py`). Time to ready 29 to 47 s.

**Not done:** a drafter ring buffer (long profile), sampling, an adaptive route for copy-heavy answers, carrying the window to a host-cache resume at an earlier snapshot ([Ideas](IDEAS.md#a-block-drafter-dflash2-style)).

## Images

Send images the OpenAI way: a user message whose `content` is a list with `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}` (or an http URL) and a text part.

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "swift", "max_tokens": 64,
  "chat_template_kwargs": {"enable_thinking": false},
  "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 photo.png)"'"}},
                                            {"type": "text", "text": "Describe this image."}]}]}'
```

| Flag | Effect |
|---|---|
| `--vision auto` (default) | the ViT's 879 MiB of weights on the GPU if 256 MiB stay free at `--max-model-len`, else its block weights stay in pinned host memory and are streamed per image (153 MiB on the GPU), else image input is off. It never changes what a text request gets: the prompt-lookup tail and the length are decided as with `--vision off`. When it turns image input off, the start log says so as a WARNING with the reason and what would fit (the largest `--max-model-len` that keeps image input on, with `--max-num-seqs` > 1 also the largest slot count at this length), and an image request gets an HTTP 400 with the same text before any response starts (streamed or not). Without `--max-model-len`, `--max-num-seqs` > 1 picks a length that leaves room for the ViT (with host weights: 156 MiB at the default `--max-image-pixels`, under 1,000 tokens per slot), so image input stays on |
| `--vision device` / `host` | demand a placement (the start is refused when it does not fit; `device` may turn the auto tail off to make room) |
| `--vision off` | text only: the memory and behaviour of 0.7.0, an image request gets a 400 |
| `--max-image-pixels N` | images are scaled down (aspect kept) to at most N pixels; default 2,097,152: 1920x1080 keeps its size, 2,040 placeholder tokens. One token per 32x32 pixels |
| `--limit-mm-per-prompt '{"image": 8}'` | images per prompt (default 4); video is always refused with a 400 |

Encode time of the ViT (median of 10): 16 / 65 / 220 ms at 448x448 / 1024x768 / 1920x1080 with the weights on the GPU; host-streamed weights add 51 ms at 448x448, 9 ms at 1024x768 and nothing at 1920x1080. Time to first token of an image request through the server: 218 ms / 617 ms / 1.47 s at the same sizes ([Performance](PERFORMANCE.md#6-image-input)).
A follow-up turn about the same image resumes past it (`cached_tokens` counts the image rows); the prefix cache keys images by content, so the same text with another image never reuses the first image's state.
Image requests are decoded by the MTP path even with `--drafter dflash2` (the block drafter's rope and its training are text-only).

## Using it from a coding agent

Any tool that speaks the OpenAI API works: set the base URL to `http://<host>:8000/v1`, the model to `swift`, any API key, and the context window to what you started the server with
(196,608 for `--profile long`). Tips:

| Tip | Why |
|---|---|
| Give generous `max_tokens`, or turn thinking off for quick edits | thinking tokens count against it |
| Keep the conversation prefix stable | follow-up turns resume from the cached prefix instead of re-prefilling |
| Expect a slow first turn on a huge prompt | about 3 minutes at 190k tokens, then fast |
| Let the agent retry a turn that came back without a tool call | a minimal tool schema gave a well-formed call 87% to 93% of the time ([tool calls](#tool-calls)); agent-sized tool sets were not measured |
| Do not rely on `logprobs` or JSON-schema constrained output | they return HTTP 400 |

**Not measured:** how well agentic loops (six-tool sets, second turns, edit-and-test cycles) succeed on any cache. The quality numbers in [Performance](PERFORMANCE.md) are single-shot coding and retrieval proxies, and the
tool-call numbers above cover one small schema. If you try it, we would like to hear what happened.
