# Changelog

## 0.2.0 - 2026-11-04

Serving-only rework (commit 6d0e932 and the docs commits after it).

- 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`): per token and KV head, asymmetric int4 with fp16 scale and minimum, 4.125 bits per element, 16.5 KiB per token. Perplexity +0.26% at 98k tokens; coding pass@1 244 -> 241 of 283 tasks (not significant).
- `--profile long`: 4-bit KV plus host-pinned embedding and head tables (bit-identical decode); the full 262,144-token context fits in 22.4 GiB on one RTX 3090 (`--prompt-lookup 0`).
- MTP speculative decoding (k <= 3) with an optional prompt-lookup tail; exact speculative sampling.
- OpenAI-compatible server on vLLM's frontend with an in-process engine client; multi-turn prefix reuse via DeltaNet state snapshots.
- Torch stack locked to the +cu130 builds (`uv.lock`).
- Performance tables against HyperQwen (`docs/PERFORMANCE.md`): single runs, caveats listed there.

Known limits: RTX 3090 only, one model, batch size 1, text only; the 4-bit cache drifts from bf16 output earlier than fp8 does.
