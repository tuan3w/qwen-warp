# Changelog

## Unreleased

- Docker: `Dockerfile`, `docker-compose.yml`, `.env.example` and an entrypoint (`PROFILE`, `EXTRA_ARGS`, `MODEL_DIR`, `PORT`); the checkpoint is mounted, not baked in. Checked once: the image built, the container became healthy and answered a chat request with `--profile long`.
- `server/serve.py` honours `$SWIFT_MODEL` as the default model path (the README already documented it).
- Docs reworked around the reader: a README that leads with why and for whom, tables for scanning, and two quick-start paths (Docker, native); new `docs/GETTING_STARTED.md` (prerequisites, weights, troubleshooting), `docs/USAGE.md` (API, flags, memory and context, thinking mode) and `docs/IDEAS.md`; a file map and a "where to start reading" table in `docs/ARCHITECTURE.md`; a summary table in `docs/PERFORMANCE.md`.
- Docs corrected: the list of unsupported sampling parameters was incomplete (the server rejects 13 names, the README listed 3); disk and RAM needs now include the environment and the Docker image; the repository layout (now a table in `docs/ARCHITECTURE.md`) includes `smk.py`, `docker/` and the scripts, which the old README tree missed; the model's default thinking mode and its effect on `max_tokens` are explained; the non-uniform weight codebook entry no longer claims a measured kernel slowdown and states the byte-count bound instead.
- The HyperQwen measurements are a reference point, not a competition, and name the exact build they describe (image digest and source revision). The roadmap became "Ideas to explore", with the 27B result for a rotated 4-bit KV cache (about 2x lower perplexity cost, not the 10x seen on a 2B proxy).

## 0.2.0 - 2026-11-04

Serving-only rework (commit 6d0e932 and the docs commits after it).

- 4-bit KV cache (`--kv-cache-dtype int4_per_token_head`): per token and KV head, asymmetric int4 with fp16 scale and minimum, 4.125 bits per element, 16.5 KiB per token. Perplexity +0.26% at 98k tokens; coding pass@1 244 -> 241 of 283 tasks (not significant).
- `--profile long`: 4-bit KV plus host-pinned embedding and head tables (bit-identical decode); the full 262,144-token context fits in 22.4 GiB on one RTX 3090 (`--prompt-lookup 0`).
- MTP speculative decoding (k <= 3) with an optional prompt-lookup tail; exact speculative sampling.
- OpenAI-compatible server on vLLM's frontend with an in-process engine client; multi-turn prefix reuse via DeltaNet state snapshots.
- Torch stack locked to the +cu130 builds (`uv.lock`).
- Reference measurements next to HyperQwen (`docs/PERFORMANCE.md`): single runs, one pinned HyperQwen build, caveats listed there.

Known limits: RTX 3090 only, one model, batch size 1, text only; the 4-bit cache drifts from bf16 output earlier than fp8 does.
