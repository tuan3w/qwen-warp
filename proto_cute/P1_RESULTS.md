# P1: stock vLLM on the 3090 vs this engine (2026-11-07)

Checkpoint `ukisai/Swift-1.5-Qwen3.8-27b-INT4`, RTX 3090 alone on the GPU (qwen-warp container stopped for the runs), greedy, 300 tokens, `ignore_eos` for completions.
Scripts: `p1_bench.py` (end-to-end, 1-8 clients), `p1_stream.py` (decode rate from SSE timestamps, TTFT excluded), `nsys_decode.py` (kernel breakdown).
vLLM runs: text-only (`--limit-mm-per-prompt image=0,video=0`, else the MTP head's second bf16 lm_head OOMs at load), MTP k=3, `--max-num-seqs 8`, Marlin W4A16, FlashInfer attention, bf16 KV (FlashInfer logs an fp8 KV dtype in its resolved query dtypes; not investigated), max len 8192 (3072 for the CuTe run: KV memory). Engine = the repo's container, `--profile multiuser`, vision on, int4 KV, max len 49152.

## Throughput

| | vLLM 0.30.0 Marlin | vLLM 0.30.0 + CuTe GEMV (spike) | this engine |
|---|---|---|---|
| 1 client, code, decode rate (stream) | 97.5 | **99.2** | **212.9** |
| 1 client, prose, decode rate (stream) | 64.3 | 69.0 | 121.0 |
| 1 client, chat code, e2e | 96.5 | 58.3 (prefill) | 181.2 |
| 1 client, chat explain, e2e | 79.6 | 71.1 (prefill) | 155.8 |
| 2 clients aggregate, e2e | 163.9 | 65.8 (T>32 path) | 281.0 |
| 4 clients aggregate, e2e | 235.4 (first run: 142.5) | 75.3 (T>32 path) | 325.2 |
| 8 clients aggregate, e2e | 259.8 (first run: 145.0) | 78.0 (T>32 path) | 347.1 |
| TTFT, code prompt (250 tokens) | 208 ms | 2124 ms (dequant prefill) | 38 ms (prefix hit) |
| tokens / cycle (code) | 3.63-3.66 | 3.63-3.66 | n/a |

- Single user: engine is **2.2x (code) / 1.9x (prose)** faster than stock vLLM. Four clients: 1.4x, eight: 1.3x (vLLM scales better than a naive reading of one client; the first vLLM multi-client run was 40 % lower, so treat vLLM multi-client numbers as +-40 % until repeated).
- The CuTe rows marked "prefill" / "T>32 path" lose only because the spike's fallback for more than 32 rows is a Python whole-matrix dequant; it is irrelevant for decode (see the stream row). Acceptance is identical to Marlin (3.63-3.66 tokens/cycle): the kernel is numerically equivalent in use.
- vLLM 0.30.0 cannot load this checkpoint with MTP on 24 GB with the vision tower; text-only leaves 1.5 GiB for KV (about 11k tokens at bf16). The engine serves 262k.

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
