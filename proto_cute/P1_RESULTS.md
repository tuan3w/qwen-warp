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

# P3: real prefill path, 16-row single-pass kernels (same stack as P2's last row: TRITON_ATTN, bf16 KV, MTP k=3, text-only, 1536 max length, 4 sequences)

New in the branch: `cute_dequant.py` (repacked W4 -> bf16 row chunk, bit-exact vs the natural dequant, ~600 GB/s write), prefill = dequant a row chunk + cuBLAS; W4 GEMV with two n8 token tiles (up to 16 rows in one weight pass; bit-exact, row-count invariant, `s11_nt_test.py`); head and small bf16 GEMV for up to 16 rows; compile only the decode row counts at load (1,2,3,4,8,12,16) because precompiling every count pushed model load past 24 GB.

Decode rate from SSE timestamps (`p1_stream.py`, `p1_stream_mc.py`; chat prompts for the multi-stream script):

| | stock 0.30.0 | plugin | engine | plugin / engine |
|---|---|---|---|---|
| 1 stream, code | 97.5 | 163.5 | 212.9 | 77 % |
| 1 stream, prose | 64.3 | 105.9 | 121.0 | 88 % |
| 1 stream, chat prompt | n/a | 156.0 | 186.0 | 84 % |
| 2 streams, per stream | n/a | 142.4 | 159.9 | 89 % |
| 4 streams, per stream | n/a | 132.7 | 120.2 | 110 % |
| first token, 250-token prompt | 208 ms | 334 ms | 38-167 ms (prefix cache / cold) | |
| 2 clients, aggregate over the whole span (prefill included) | n/a | 263 | 293 | 90 % |
| 4 clients, aggregate over the whole span (prefill included) | n/a | 220 | 369 | 60 % |
| e2e script `p1_bench.py`, 4 clients | 235 (warm) | 187 | 325 | 57 % |

- **Decode is at or near engine level** (77-110 % depending on prompt and concurrency); the 4-stream per-stream rate is above the engine's. What is left of the e2e gap at 4 clients is **prefill**: the plugin prefills four 250-token prompts one after another at ~330 ms each.
- Prefill profile (nsys, one 250-token request, 317 ms, GPU 96 % busy): cuBLAS GEMM 181 ms (~70 TFLOPs, the card's dense bf16 rate: it is compute bound), CuTe dequant 83 ms, torch copies 11 ms, everything else ~40 ms. Marlin does the same GEMMs without the 94 ms of dequant and copies (TTFT 208 ms). The overhead is a fixed ~95 ms per forward chunk, so it matters for short prompts and is ~5 % at 2k tokens. Removing it needs a fused dequant-GEMM (tensor-core, Marlin-style) kernel.
- vLLM's hybrid prefix cache works on 1600-token blocks, so a 250-token system prompt never hits it, where the engine restores a prefix in 38 ms.
- Decision rule (>= 85 % of the engine at one user, holding at four clients): decode part **met** (84-88 % on prose/chat, 77 % on the repeated-code prompt where the engine's lookup tail helps); the aggregate with prefill is **not** met (60 %).

# P4: host-resident tables and the DFlash2 block drafter on stock vLLM 0.30.0

New: `cute_host.py` (pinned host tables + a CuTe gather kernel reading rows over PCIe by UVA pointer, graph-capturable, bit-equal to indexing) used for the embedding and for the lm_head's exact-rescoring rows (`VLLM_CUTE_HOST=1`); the rescoring candidate count is 16 (the engine's greedy setting); GEMV for up to 32 rows in one weight pass (NT up to 4); `CuteDFlash2` (registry override for `DFlash2DraftModel`: vLLM builds the drafter's fused K/V weight stack inside `load_weights`, before a W4A16 layer has a bf16 `weight`, so the build is deferred until our kernel has produced a dequantised qkv copy). Drafter: the engine's W4A16 DFlash2 checkpoint (`~/llm-lab/models/dflash2-w4a16`, 1.28 GB), `{"method":"dflash","num_speculative_tokens":7}`, V2 model runner, Triton attention, bf16 KV, 4 sequences, 4096 max length, text-only.

| | stock vLLM | plugin, MTP k=3 | plugin + host tables, MTP k=3 | **plugin + host tables + DFlash2 k=7** | engine (MTP default) |
|---|---|---|---|---|---|
| model weights in VRAM | 17.4 GiB | 17.4 + 1.2 | **12.8 GiB** | 14.3 GiB (incl. drafter) | |
| KV pool | 1.5 GiB (3.6k tokens) | 1.5 GiB | **8.3 GiB (31k)** | 6.8 GiB (15k) | 262k tokens (4-bit KV) |
| 1 stream, code, tok/s | 97.5 | 163.5 | 159.0 | **321.5** | 212.9 |
| 1 stream, prose | 64.3 | 105.9 | 107.2 | 116.3 | 121.0 |
| 1 stream, chat prompt | n/a | 156.0 | 156.8 | **269.3** | 186.0 |
| 2 streams, per stream (aggregate over the span) | n/a | 142.4 (263) | 141.2 (258) | **180.0 (291)** | 159.9 (293) |
| 4 streams, per stream (aggregate over the span) | n/a | 132.7 (220) | 113.4 (364) | 93.8 (321) | 120.2 (369) |
| first token, 250-token prompt | 208 ms | 334 ms | 331 ms | 335 ms | 38-167 ms |

- Moving the 4.7 GiB of bf16 tables to host memory costs nothing in decode speed and gives 5.6x the KV pool, which is what makes a drafter fit at all; the 4-stream aggregate over the whole span went from 220 to 364 tok/s (the larger pool lets four requests actually run together).
- **DFlash2 on stock vLLM, with this plugin, beats the engine's MTP default at one stream by 1.5x (chat) / 1.5x (code)** and matches it at 2-4 streams; the engine's own DFlash2 mode is reported at 277 tok/s on code. DFlash2 gains nothing on prose (116 vs 107), as in the engine's measurements.
- 4 streams x 8 verify rows = 32 rows per step: the 4-token-tile GEMV runs at 440-520 GB/s (`s13_nt4_test.py`), so the plugin is weight-bandwidth-inefficient exactly where the verify block is widest; this is the target of the `gv-nt2` campaign lineage.
- Without the 32-row path the same run collapsed to 25 tok/s per stream at 4 clients (rows above 16 fell to the dequantise-and-GEMM path every step).

- int4 target KV (`--kv-cache-dtype int4_per_token_head`, drafter KV forced to bf16 with `"kv_cache_dtype":"bfloat16"` in the speculative config, because the int4 path asserts causal attention and the DFlash2 drafter is non-causal): works, 1 stream code 306.8 / chat 253.0 tok/s (-5 % vs bf16 KV), but the pool does not grow (12,220 tokens at 12,000 max length, 1 request): for this hybrid model the per-request recurrent-state pages (k+1 = 8 slots x 151 MB per sequence with 7 drafts) and the 3,184-token attention block size set the capacity, not the attention KV bytes. A bigger KV format only pays at context lengths where attention KV, not state, dominates.

# Like-for-like comparison (same card, same scripts: `p1_stream.py` decode rate, `p1_stream_mc.py` chat prompt, one stream)

Engine runs: MTP = the user's container (`--profile multiuser`, vision on, int4 KV); DFlash2 = `qwen-warp:local` with `--drafter dflash2 --profile long --max-model-len 122880` (the engine requires `--max-num-seqs 1` for it). Plugin runs: stock vLLM 0.30.0, text-only, TRITON_ATTN, bf16 KV, 4 sequences, host tables, merged campaign kernels.

| | MTP: engine | MTP: vLLM + plugin | engine / plugin | DFlash2: engine | DFlash2: vLLM + plugin | engine / plugin |
|---|---|---|---|---|---|---|
| 1 stream, code | 212.9 | 159.0 | 1.34x | 301.3 | 271.0 (321.5 in the previous build; the repeated-code prompt is noisy) | 1.11x (0.94x) |
| 1 stream, prose | 121.0 | 107.2 | 1.13x | 129.2 | 115.0 | 1.12x |
| 1 stream, chat prompt | 186.0 | 156.8 | 1.19x | 290.4 | 274.5 | 1.06x |
| first token, 250-token prompt | 38-177 ms | 331 ms | | 38-177 ms | 137-192 ms (fused prefill GEMM) | |

- **Like for like, the engine is still faster for one user: by 13-34 % with MTP and by 6-12 % with DFlash2** (the code prompt, where the repeated text makes acceptance noisy, ranges from the plugin ahead to 11 % behind across builds). The earlier "plugin beats the engine" statement compared the plugin's DFlash2 with the engine's MTP and is withdrawn.
- DFlash2 vs MTP inside each stack: engine +40 % (code) / +56 % (chat); plugin +70 % (code) / +75 % (chat). The plugin gains more from the drafter because its MTP path is the weaker one.
- At 2 clients the plugin (DFlash2, 4 sequences) aggregates 287 tok/s; the engine's DFlash2 mode serves one sequence at a time (2 clients: 215), its MTP multiuser mode 293.
