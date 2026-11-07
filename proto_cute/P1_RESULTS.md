# P1: stock vLLM on the 3090 vs this engine (2026-11-07)

Checkpoint `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, RTX 3090 alone on the GPU (qwen-warp container stopped for the runs), greedy, 300 tokens, `ignore_eos` for completions.
Scripts: `p1_bench.py` (end-to-end, 1-8 clients), `p1_stream.py` (decode rate from SSE timestamps, TTFT excluded), `nsys_decode.py` (kernel breakdown).
vLLM runs: text-only (`--limit-mm-per-prompt image=0,video=0`, else the MTP head's second bf16 lm_head OOMs at load), MTP k=3, `--max-num-seqs 8`, Marlin W4A16, FlashInfer attention, bf16 KV (FlashInfer logs an fp8 KV dtype in its resolved query dtypes; not investigated), max len 8192 (3072 for the CuTe run: KV memory). Engine = the repo's container, `--profile multiuser`, vision on, int4 KV, max len 49152.

## Throughput

| | vLLM 0.30.0 Marlin | vLLM 0.33.0 Marlin | vLLM 0.30.0 + CuTe GEMV (spike) | this engine |
|---|---|---|---|---|
| 1 client, code, decode rate (stream) | 97.5 | 101.7 | **99.2** | **212.9** |
| 1 client, prose, decode rate (stream) | 64.3 | 62.7 | 69.0 | 121.0 |
| 1 client, chat code, e2e | 96.5 | 98.7 | 58.3 (prefill) | 181.2 |
| 1 client, chat explain, e2e | 79.6 | 88.9 | 71.1 (prefill) | 155.8 |
| 2 clients aggregate, e2e | 163.9 | 159.0 | 65.8 (T>32 path) | 281.0 |
| 4 clients aggregate, e2e | 235.4 (first run: 142.5) | 134.7 (first run) | 75.3 (T>32 path) | 325.2 |
| 8 clients aggregate, e2e | 259.8 (first run: 145.0) | 160.5 (first run) | 78.0 (T>32 path) | 347.1 |
| TTFT, code prompt (250 tokens) | 208 ms | 229 ms | 2124 ms (dequant prefill) | 38 ms (prefix hit) |
| tokens / cycle (code) | 3.63-3.66 | 3.63-3.66 | 3.63-3.66 | n/a |

- Single user: engine is **2.2x (code) / 1.9x (prose)** faster than stock vLLM. Four clients: 1.4x, eight: 1.3x (vLLM scales better than a naive reading of one client; the first vLLM multi-client run was 40 % lower, so treat vLLM multi-client numbers as +-40 % until repeated).
- The CuTe rows marked "prefill" / "T>32 path" lose only because the spike's fallback for more than 32 rows is a Python whole-matrix dequant; it is irrelevant for decode (see the stream row). Acceptance is identical to Marlin (3.63-3.66 tokens/cycle): the kernel is numerically equivalent in use.
- vLLM 0.30.0 cannot load this checkpoint with MTP on 24 GB with the vision tower; text-only leaves 1.5 GiB for KV (about 11k tokens at bf16). The engine serves 262k.

- **vLLM 0.33.0 changes nothing for this setup** (same Marlin, same FlashInfer, same 11.7k-token KV at 8192): single-client decode +4 % on code, -2 % on prose. Its multi-client numbers were taken straight after server start like the first 0.30.0 run (about 40 % below the warmed second run), so only compare them with that first run.
- **Public context:** HyperQwen (patch series on vLLM, same model, one 3090 at a 250 W cap) reports 127 tok/s single stream, using requantized heads, a calibrated 40k draft vocabulary and int8 Marlin - the same levers the profile below points at. So the gap is about 2x against *unpatched* vLLM and about 1.3-1.4x against a patched one (this README's own comparison: 1.2-1.3x at long context; power caps differ: 250 W vs 350 W).

## Where a vLLM decode cycle goes (nsys, node-level CUDA graph trace, 16 cycles of code decode)

| ms per cycle | Marlin | CuTe GEMV |
|---|---|---|
| cycle (span) | 36.8 | 36.0 |
| GPU busy | 94 % (host gaps 2.1 ms) | 94 % |
| W4A16 linears (256 calls) | 16.27 | **15.62** (-4 %) |
| **lm_head** (bf16 GEMV x2.9 for MTP drafts + bf16 GEMM for verify) | **11.5** (8.22 + 3.3) | 11.5 |
| other bf16 GEMMs (59 calls, GDN `in_proj_ba` etc.) | 3.8 | 3.8 |
| GDN decode (48), conv1d, flashinfer, KV write, triton fusions, misc | ~3.1 | ~3.0 |

Conclusions that follow from the profile:

1. **vLLM is not host-bound here** (GPU 94 % busy); piecewise graphs cost 2 ms per cycle, not the 2x.
2. **The CuTe GEMV replaces Marlin in situ at parity (-4 % kernel time, +2 to +7 % decode tok/s).** On the linears there is nothing to win beyond that: both sit at ~770-800 GB/s.
3. **The lm_head is 31 % of a vLLM cycle** (bf16 head: 2.5 GB read per forward, and MTP runs it again for each of 3 drafts). This engine's int4 head with exact rescoring over a restricted draft vocabulary is the single biggest piece of the 2x. [INFERENCE] Removing about 9 of the 11.5 ms would put vLLM at ~27.7 ms/cycle, ~130 tok/s, 60-70 % of the engine.
4. The rest of the gap (about 6 ms) is hundreds of small kernels (59 bf16 GEMMs, 274 triton fusions, 117 elementwise) that the engine fuses.

## Integration findings (vLLM 0.30.0)

- OOT kernel registration works, but selection order is append-last (Marlin wins) -> the adapter inserts at index 0 of a private dict.
- **The compile cache key ignores the linear-kernel choice**: a cached AOT graph from the Marlin run was loaded into the CuTe run and crashed with `KeyError: 'weight_zero_point'`. Use a separate `VLLM_CACHE_ROOT` per kernel config (or fix the hash).
- The kernel must be a registered torch custom op (done: `torch.ops.vllm.cute_w4a16_linear` with fake impl) and all CuTe compiles and host-to-device copies must happen at load time, not inside CUDA graph capture (two crashes).
- A single `MPLinearKernel` class has to serve decode and prefill (one weight layout fits in memory), so a real prefill path (dequant kernel + cuBLAS) is mandatory; the spike's torch dequant makes TTFT 10x worse.

## Go / no-go

| Route | Single-user speed (code) | Adds a model by | Verdict |
|---|---|---|---|
| Stock vLLM 0.30 / 0.33 | ~98-102 tok/s (0.5x engine) | nothing (works) | baseline; cannot load with vision + MTP on 24 GB, 11k-token KV |
| vLLM + CuTe GEMV plugin (this branch) | ~99-100 (parity with Marlin) | config | GO as the compat tier, but it is not where the speed is |
| vLLM + plugin + int4 `lm_head` / restricted draft vocab + fused small ops | [INFERENCE] ~130 (60-65 % of engine; HyperQwen's 127 is the public data point) | config + 2-3 op modules | GO, next experiment (P2); this is the test of "can vLLM get close" |
| This engine | 213 (181 chat) | new engine wiring per model | keep as the fast tier until P2 says otherwise |

Decision rule for P2: build the `lm_head` adapter (int4 head + exact rescoring, 64k draft vocabulary for MTP drafts, as an `lm_head` quant method / logits processor) and re-run `p1_stream.py` and the nsys breakdown. If vLLM + plugin reaches >= 85 % of the engine at one user, retire the custom engine loop and keep only the ops library. If it stays near 60-70 %, keep two tiers sharing one CuTe ops library: engine = fast path for the one tuned model, plugin = everything else.

# P2: three CuTe adapters on stock vLLM 0.30.0 (single user, MTP k=3, text-only, bf16/fp8 KV as noted)

Plugin env switches: `VLLM_CUTE_W4` (W4 linears), `VLLM_CUTE_HEAD` (`cute_head.py`: int4 lm_head + exact rescoring of the top-32, 68,864-token draft vocabulary for MTP drafts), `VLLM_CUTE_MISC` (`cute_misc.py`: MTP layer quantised to int4 at load, CuTe GEMV for small bf16 projections such as GDN `in_proj_ba`). Each run needs its own `VLLM_CACHE_ROOT`.

| config | code tok/s | prose tok/s | code tokens/cycle | cycle (ms) |
|---|---|---|---|---|
| stock Marlin (FlashInfer, piecewise graphs) | 97.5 | 64.3 | 3.63-3.66 | 36.8 (nsys) |
| + CuTe W4 linears | 99.2 | 69.0 | 3.66 | 36.0 (nsys) |
| + int4 head (lm_head 11.5 -> 1.9 ms) | 150.8 | 85.0 | 4.00 | 27.5 (nsys) |
| + MTP layer int4 + small GEMV (shape branch moved inside the op) | 153.6 | 89.7 | 3.96 | 26.6 (nsys), GPU 83 % busy |
| + `--attention-backend TRITON_ATTN --kv-cache-dtype bfloat16` (full CUDA graphs with spec decode) | **159.8** | **101.2** | 3.58 | **22.4** (derived) |
| this engine | 212.9 | 121.0 | n/a | ~19-21 (docs) |

- vLLM + plugin reaches **75 % (code) and 84 % (prose)** of the engine at one user; per-cycle time is within 1-3 ms of the engine's. The code-prompt gap that is left is tokens per cycle (3.58 vs the engine's prompt-lookup tail), not kernels.
- Default vLLM on this card picked an **fp8 KV cache** (FlashInfer's resolved dtype); Triton attention refuses fp8 on sm_86, so the last row runs bf16 KV (3.6k tokens of pool in this tiny-memory setup).
- A Python branch on the row count inside a patched layer method is frozen at trace time by vLLM's AOT-compiled graph (no guards): all shape dispatch has to live inside the registered custom op (this hid the small-GEMV patch once).
- Greedy outputs differ from stock after 45-790 characters on 4 of 6 chat prompts, at bf16 near-ties (`'` vs the typographic apostrophe, "Assumed Schema" vs "Approach"); both coherent. Not isolated between head and linears.
- Open: first-token time is 2.1 s (rows > 32 use a Python dequant fallback); heads for > 8 rows fall back to the bf16 path; multi-client numbers are not valid yet; sampling above temperature 0 is truncated to 32 candidates.
