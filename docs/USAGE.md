# Usage

How to call the server, which flags matter, and how memory and context fit together. New here? Start with [Getting started](GETTING_STARTED.md). Back to the [README](../README.md).

Contents: [Endpoints](#endpoints) | [Calling it](#calling-it) | [Thinking and reasoning output](#thinking-and-reasoning-output) | [Sampling parameters](#sampling-parameters) |
[Tool calls](#tool-calls) | [Multi-turn prefix reuse](#multi-turn-prefix-reuse) | [Server flags](#server-flags) | [Memory and context](#memory-and-context) | [Using it from a coding agent](#using-it-from-a-coding-agent)

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

The thinking text comes back in a separate field of the message, not in `content`; the field name follows your vLLM version, so check the raw response once.

## Sampling parameters

The engine implements sampling itself (exact speculative sampling), so only some parameters exist. Anything unsupported returns **HTTP 400** that names the parameter, rather than being ignored.

| Supported | Notes |
|---|---|
| `temperature`, `top_p` | without sampling parameters the model's own defaults apply (T = 1, top_p 0.95, top_k 20) |
| `top_k` | up to 64 |
| `seed` | same seed, same output |
| `max_tokens`, `stop`, `stop_token_ids`, `ignore_eos` | `stop` strings cut the text before the match |
| `n` | |

| Not supported (HTTP 400) |
|---|
| `logprobs`, `prompt_logprobs`, `logprob_token_ids` |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` |
| `min_p`, `min_tokens`, `logit_bias`, `allowed_token_ids`, `bad_words` |
| `structured_outputs`, `thinking_token_budget` |

## Tool calls

Tool calling works out of the box (`qwen3_coder` parser with automatic tool choice), including the round trip of sending the tool result back:

```python
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = client.chat.completions.create(model="swift", messages=[{"role": "user", "content": "What is the weather in Hanoi?"}],
                                   tools=tools, max_tokens=800)
call = r.choices[0].message.tool_calls[0]          # name "get_weather", arguments '{"city": "Hanoi"}'
```

Constrained decoding (`structured_outputs`) is not available; the parser reads the tool call from the model's text.

## Multi-turn prefix reuse

Follow-up turns of a chat do not re-prefill the whole history. The engine snapshots its recurrent state at turn boundaries and resumes from the longest cached prefix:

| Detail | Value |
|---|---|
| How you can see it | `usage.prompt_tokens_details.cached_tokens` in the response (0 on a cold prompt) |
| Snapshots kept | 4 by default (`--snapshots N`, 151 MB of pinned host RAM each, 0 = off) |
| Measured | turn 1: 1,529 prompt tokens, 1.35 s; turn 2: 1,528 of 1,622 cached, 0.54 s; turn 3: 1,621 of 1,692 cached, 0.32 s |
| To benefit | keep the earlier messages byte-identical and append only; changing an earlier message breaks the prefix |

## Server flags

Every stock `vllm serve` frontend flag works (`--host`, `--port`, `--api-key`, `--served-model-name`, `--chat-template`, `--max-model-len`, ...). The engine adds these:

| Flag | Default | What it does |
|---|---|---|
| `--profile {default,long}` | `default` | `long` = 4-bit KV cache + `--host-tables` + a default `--max-model-len` of 196,608. Explicit flags win over the profile |
| `--kv-cache-dtype` | `auto` (bf16) | `fp8`, or `int4_per_token_head` for the 4-bit cache |
| `--max-model-len` | 28,672 (32,768 with `--prompt-lookup 0`); 196,608 with `--profile long` | context length the engine is sized for |
| `--host-tables` | off | keep the bf16 embedding and rescoring head (4.7 GiB) in pinned host memory: bit-identical decode, about 0 to 2% slower, 4.7 GiB more VRAM for the cache |
| `--prompt-lookup {0,1,auto}` | `auto` | an n-gram draft tail that needs 0.62 GiB more VRAM; `auto` enables it only where that combination was measured to fit ([table](#memory-and-context)) |
| `--num-speculative-tokens {0..3}` | 3 | MTP drafts per step; 0 = plain decode |
| `--snapshots N` | 4 | multi-turn prefix snapshots |
| `--stream-cycles N` | 2 | speculative cycles per host sync, i.e. streaming granularity (one cycle is about 24 ms and emits 1 to 4 tokens) |

Set `SWIFT_MODEL` (or pass `--model`) when the weights are not in `~/models/swift-1.5-int4`. With Docker these become `PROFILE`, `EXTRA_ARGS`, `MODEL_DIR` and `PORT` ([table](GETTING_STARTED.md#3a-docker)).

## Memory and context

The KV cache is the only thing that grows with context. For the 16 full-attention layers (4 KV heads x 256 dims):

| KV cache | Bytes per token | 262,144 tokens | Largest `--max-model-len` measured to start and serve | `--prompt-lookup auto` turns the tail on up to |
|---|---|---|---|---|
| bf16 | 64 KiB | 16 GiB | about 74,000 (100k does not fit) | 28,672 |
| fp8 | 32 KiB | 8 GiB | 65k; 131k with `--host-tables` | 53,248 |
| 4-bit | 16.5 KiB | 4.1 GiB | 98k; **262,144 with `--host-tables`** (22.4 GiB used) | 196,608 (with `--host-tables`) |

Rules of thumb:

- Want the full 262,144 tokens? `--profile long --max-model-len 262144 --prompt-lookup 0`. The tail would not fit next to a cache that large.
- Want the best short-context speed and do not need length? The defaults.
- A 24 GB desktop card also drives a monitor (about 0.6 GiB); the numbers above were measured that way.
- Context length trades against VRAM only: it does not change output quality except through the 4-bit cache's own loss ([Quality](PERFORMANCE.md#2-quality-of-the-4-bit-kv-cache)).

## Using it from a coding agent

Any tool that speaks the OpenAI API works: set the base URL to `http://<host>:8000/v1`, the model to `swift`, any API key, and the context window to what you started the server with
(196,608 for `--profile long`). Tips:

| Tip | Why |
|---|---|
| Give generous `max_tokens`, or turn thinking off for quick edits | thinking tokens count against it |
| Keep the conversation prefix stable | follow-up turns resume from the cached prefix instead of re-prefilling |
| Expect a slow first turn on a huge prompt | about 3 minutes at 190k tokens, then fast |
| Do not rely on penalties, `logprobs` or JSON-schema constrained output | they return HTTP 400 |

**Not measured:** how well agentic loops (tool calls, edit-and-test cycles) succeed with the 4-bit cache. The quality numbers in [Performance](PERFORMANCE.md) are single-shot coding and retrieval proxies. If you try it, we would like to hear what happened.
