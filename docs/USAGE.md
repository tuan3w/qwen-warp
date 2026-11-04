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
  "max_tokens": 800}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")      # the key is only checked if you started the server with --api-key

stream = client.chat.completions.create(model="swift", stream=True, max_tokens=800,
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

`reasoning_effort` is accepted by the chat template of the checkpoint. What was measured on a stock vLLM server of the same model (not yet on this server: it goes through the same vLLM frontend and
template, so the same behaviour is expected here, but that was not tested):

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
the required parameter. In-process engine runs, n = 150 per cell (the 95% intervals reach 4 to 8 points to either side):

| KV cache | schema as most clients send it (`type`, `properties`, `required`) | keys sorted (`properties`, `required`, `type`; what the server renders by default) |
|---|---|---|
| bf16 | 91% (137/150) | 93% (139/150) |
| fp8 | 88% (132/150) | not measured |
| 4-bit (`--profile long`) | **39% (58/150)** | **87% (130/150)** |

- The chat template prints the tool schema in the key order the client sent, and the model's choice at the start of a tool call depends on that order, much more so on the 4-bit cache. `--sort-tool-schema on` (the default)
  sorts the top-level keys of every tool's `function.parameters` before the template renders them; nothing else changes (not your JSON, not the parser, not the arguments, nested objects untouched).
  `--sort-tool-schema off` renders the schema exactly as sent.
- Through the server, 40 identical requests: 18/40 before the sorting on `--profile long`; with it 37/40 on the default profile and 40/40 on `--profile long` (`tests/test_tool_call_rate.py`, which fails below 70%).
- Two other schemas (a 3-parameter tool with an enum and a required subset, and a nested-object tool) gave 100% in every cell, in both key orders (n = 100 on bf16, n = 60 on 4-bit): the minimal one-parameter
  schema is the fragile one.
- The sampler is not the cause: on 219 real positions its draws match the exact bf16 reference distribution (`tests/test_sampler_dist.py`). Greedy decoding on the 4-bit cache gave 99% (149/150, schema as sent).
- Even on bf16 the rate (93%) was below the 98% of a reference vLLM service we compared against (its prompt had the parameter keys sorted); the cause was not isolated ([Ideas](IDEAS.md#why-bf16-kv-is-below-a-reference-service-on-tool-calls)).
- **Not measured:** coding-agent tool sets (six tools or more), second turns of a tool conversation, fp8 with the sorted order, the fp8 profile with host tables, a rotated 4-bit cache, top_p / min_p as a mitigation. How
  well real agent loops succeed on any cache is unknown.

Constrained decoding (`structured_outputs`) is not available; the parser reads the tool call from the model's text.

## Multi-turn prefix reuse

Follow-up turns of a chat do not re-prefill the whole history. The engine snapshots its recurrent state while prefilling and resumes from the longest cached prefix:

| Detail | Value |
|---|---|
| How you can see it | `usage.prompt_tokens_details.cached_tokens` in the response (0 on a cold prompt) |
| Snapshots kept | 4 by default (`--snapshots N`, 151 MB of pinned host RAM each, 0 = off) |
| Where snapshots are taken | at the last two cut points of the prefill plan. Cuts fall every 4,096 tokens, at a chat-turn boundary more than 240 tokens after the previous cut, and just before the last token when more than 240 tokens follow the last cut. A long system prompt or document is therefore reused across different user questions |
| Measured | 3-turn chat over a 1,529-token document, per request including 60 generated tokens: 1.28 s, 0.54 s, 0.41 s (turns 2 and 3 resumed at 1,524 cached tokens; the old cut points gave 1.31 s, 0.60 s, 0.39 s) |
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
| `--sort-tool-schema {on,off}` | `on` | render the keys of every tool's `function.parameters` sorted, which gives well-formed tool calls more often ([tool calls](#tool-calls)) |
| `--snapshots N` | 4 | multi-turn prefix snapshots |
| `--stream-cycles N` | 2 | speculative cycles per host sync, i.e. streaming granularity (one cycle is about 24 ms and emits 1 to 4 tokens) |

Set `SWIFT_MODEL` (or pass `--model`) when the weights are not in `~/models/swift-1.5-int4`. With Docker these become `PROFILE`, `EXTRA_ARGS`, `MODEL_DIR` and `PORT` ([table](GETTING_STARTED.md#3a-docker)).

**vLLM version.** The environment pins `vllm==0.30.0` (`pyproject.toml`, `uv.lock`). The engine-core client (`server/smk_core_client.py`) and the tool-schema hook (`server/tool_schema.py`, which wraps
`vllm.renderers.hf.safe_apply_chat_template`) use vLLM internals of that version; other versions are untested (a newer one, 0.33.0, exists). Install from the lock file rather than upgrading vLLM by hand.

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
