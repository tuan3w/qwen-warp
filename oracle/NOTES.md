# Swift-1.5-Qwen3.8-27B-INT4 — checkpoint layout and semantic gotchas

Everything below was read from the checkpoint headers / source and exercised by `weights.py`, `check_vs_hf.py`, `dump_ref.py`
(CPU, fp32). Pointers are `file:line` in the lab venv (`HF` = transformers 5.18.0 `models/qwen3_5/modeling_qwen3_5.py`,
`VLLM` = vllm 0.30 `model_executor/…`, `CT` = compressed-tensors 0.17.0 `compressors/pack_quantized/helpers.py`).
Not measured: equality with vLLM's output (we compare against HF semantics only).

## 1. Checkpoint (`~/models/swift-1.5-int4`)

* 5 shards `model-0000{1..5}-of-00005.safetensors` + `model_mtp.safetensors`; `model.safetensors.index.json` lists all 2031
  tensors including the 15 `mtp.*` ones (→ `model_mtp.safetensors`). 333 of them are vision (`model.visual.*`, 921 MB, ignored here).
* Quantisation (`config.json` quantization_config, `QUANTIZATION_MANIFEST.json`): compressed-tensors `pack-quantized`, 4 bit,
  symmetric, `group` strategy, group_size 128, `actorder: static`, KV-cache scheme fp8 static per-tensor.
  **No `weight_g_idx` and no `weight_zero_point` tensor exists anywhere in the index** (asserted in `weights.selftest`), so despite
  `actorder: static` the groups are the contiguous blocks `k // 128` — no column permutation to undo.
* Per quantised Linear `X`: `X.weight_packed` int32 `[N, K/8]`, `X.weight_scale` bf16 `[N, K/128]`, `X.weight_shape` int64 `[2]` = `[N, K]`.
  **Dequant: `w[n,k] = (nibble - 8) * scale[n, k//128]`**, nibble `i` (LSB first) of word `j` = column `8j+i`, i.e. byte `b` of the row holds
  columns `(2b, 2b+1)` = (low nibble, high nibble). Verified bit-exact against `unpack_from_int32` (CT :91-162, offset `-8` at :159-160)
  and `pack_to_int32` (CT :62-83) in `weights.selftest()`; nibble histogram spans 0..15, mean 7.998.
* Unquantised (BF16) tensors: `embed_tokens`, `lm_head` (**untied**, separate tensor), all norms, `in_proj_a`, `in_proj_b`, `conv1d`, `A_log`,
  `dt_bias`, `linear_attn.norm`, and every `mtp.*` tensor (MTP head is entirely BF16, incl. its MLP/attn linears).
* `self_attn.k_scale` / `v_scale`: bf16 `[1]` per full-attention layer (fp8 KV scales). k_scale 0.028–0.051, v_scale 0.021–0.252 (grows with depth, layer 63: 0.2520).
  The oracle keeps K/V in high precision and ignores them; an fp8-KV engine must use them (value_fp8 = clamp(x / scale)).

### Tensor names / shapes (prefix `model.language_model.layers.{i}.`)

H = 5120, I = 17408. `*` = quantised (packed `[N,K/8]` i32 + scale `[N,K/128]` bf16).

| Layer type | tensor | shape (logical `[N,K]` or vector) | dtype | packed MB |
| --- | --- | --- | --- | --- |
| all 64 | `input_layernorm.weight` | [5120] | bf16 | |
| all 64 | `post_attention_layernorm.weight` | [5120] | bf16 | |
| all 64 | `mlp.gate_proj` `mlp.up_proj` * | [17408, 5120] | w4 | 45.96 each |
| all 64 | `mlp.down_proj` * | [5120, 17408] | w4 | 45.96 |
| linear_attention (48: i%4 != 3) | `linear_attn.in_proj_qkv` * | [10240, 5120] rows = q(2048)\|k(2048)\|v(6144) | w4 | 27.03 |
| | `linear_attn.in_proj_z` * | [6144, 5120] | w4 | 16.22 |
| | `linear_attn.in_proj_a`, `in_proj_b` | [48, 5120] each (**bf16, not quantised**) | bf16 | 0.49 each |
| | `linear_attn.out_proj` * | [5120, 6144] | w4 | 16.22 |
| | `linear_attn.conv1d.weight` | [10240, 1, 4] (no bias) | bf16 | 0.08 |
| | `linear_attn.A_log`, `dt_bias` | [48] each | bf16 | |
| | `linear_attn.norm.weight` | [128] (gated RMSNorm, per v-head dim) | bf16 | |
| full_attention (16: i%4 == 3) | `self_attn.q_proj` * | [12288, 5120] = 24 heads × (q 256 \| gate 256) interleaved per head | w4 | 32.44 |
| | `self_attn.k_proj`, `v_proj` * | [1024, 5120] each (4 kv heads × 256) | w4 | 2.70 each |
| | `self_attn.o_proj` * | [5120, 6144] | w4 | 16.22 |
| | `self_attn.q_norm.weight`, `k_norm.weight` | [256] each | bf16 | |
| | `self_attn.k_scale`, `v_scale` | [1] each | bf16 | |

Globals: `model.language_model.embed_tokens.weight` bf16 [248320, 5120]; `model.language_model.norm.weight` bf16 [5120]; `lm_head.weight` bf16 [248320, 5120].

MTP (`mtp.*`, all bf16): `fc.weight` [5120, 10240]; `pre_fc_norm_embedding.weight`, `pre_fc_norm_hidden.weight`, `norm.weight` [5120];
`layers.0.` = a full-attention decoder layer: `input_layernorm`, `post_attention_layernorm` [5120]; `self_attn.q_proj` [12288,5120],
`k_proj` [1024,5120], `v_proj` [1024,5120], `o_proj` [5120,6144], `q_norm`/`k_norm` [256]; `mlp.gate_proj`/`up_proj` [17408,5120], `down_proj` [5120,17408].
No separate MTP embedding/lm_head: both are shared with the target (`mtp_use_dedicated_embeddings: false`).

Bytes read per decoded token (weights only, from the headers): 48 linear layers × 198.4 MB + 16 full layers × 192.0 MB = **12.60 GB**,
+ `lm_head` 2.54 GB = **15.14 GB/token** (embed lookup is one row). At 936 GB/s that is a 16.2 ms/token floor. MTP head: 0.85 GB per draft step + lm_head 2.54 GB.

Model dims (config `text_config`): 64 layers, layer `i` is full attention iff `i % 4 == 3`; 24 q heads / 4 kv heads / head_dim 256;
rotary: partial 0.25 → 64 dims, theta 1e7; DeltaNet: 16 k heads × 128, 48 v heads × 128, conv kernel 4; rms eps 1e-6; vocab 248320; `attn_output_gate: true`.

## 2. Semantic gotchas (each one was needed to match HF; cross-checked by `check_vs_hf.py`)

1. **Zero-centred RMSNorm everywhere except the DeltaNet gated norm.** `y = x·rsqrt(mean(x²)+eps)·(1 + w)`, fp32 math, single rounding at the end
   (HF :840-857, `1.0 + weight` at :853; vLLM `GemmaRMSNorm` imported as `Qwen3_5RMSNorm`, VLLM/models/qwen3_5.py:41). Stored weights are the `w`
   (layer-0 `input_layernorm` mean 0.14, `q_norm` mean 0.23). Applies to input/post-attn norms, final norm, q_norm/k_norm, all MTP norms.
   `linear_attn.norm.weight` is **not** zero-centred: values ≈ 0.87, plain multiply (HF :217-233, `self.weight * x` at :230).
2. **Gated RMSNorm (DeltaNet output)**: per v-head (128): `n = x·rsqrt(mean(x²)+eps)`; `out = (w·n) · silu(z)` with z from `in_proj_z` reshaped `[L,48,128]`
   (HF :224-233; norm *before* gate; in bf16 mode `n` is rounded to bf16 before `w·n`). config `output_gate_type: swish` = silu (VLLM/layers/mamba/gdn/qwen_gdn_linear_attn.py:480-482).
3. **`in_proj_qkv` row layout is `[q | k | v]`** = 2048 + 2048 + 6144 (HF :602-610); q/k are `[16 heads, 128]`, v is `[48 heads, 128]`.
   The same order is the conv channel order (`conv_dim = key_dim*2 + value_dim`, HF :520).
4. **Head repeat is `repeat_interleave`**: v-head `h` uses k-head `h // 3` (HF :619-621). (Tiling instead would silently permute heads.)
5. **Conv1d**: depthwise, kernel 4, no bias, then SiLU; weight `[10240,1,4]`, **tap 3 multiplies the newest token** (cross-correlation, HF :249-265 / `F.conv1d`).
   State = last 3 pre-conv inputs `[10240, 3]`, oldest first, zero-initialised (left zero padding). HF's cache stores width 4 (cache_utils.py:1045-1046,
   `conv_kernel_size`), vLLM stores `kernel-1 = 3` (VLLM/layers/mamba/mamba_utils.py:284-288) — we use 3.
6. **Gate/decay math** (fp32, HF :616-618): `beta = sigmoid(b)`; `g = -exp(A_log) · softplus(a + dt_bias)` (≤ 0; `dt_bias` ranges ~ -3.5 … +13, bf16-stored; torch softplus
   threshold=20 → identity above). Recurrence multiplies state by `exp(g)`.
7. **q/k L2-normalised then q scaled by 1/√128** inside the delta rule: `x·rsqrt(sum(x²)+1e-6)` (eps *inside* the rsqrt, HF :293-296, applied at :342-347 / :462-467).
8. **Delta rule per token** (HF :479-490): `S ← exp(g)·S; kv = Sᵀk; Δ = (v − kv)·β; S ← S + k⊗Δ; out = Sᵀq` with `S: [48, K=128, V=128]` fp32.
   HF layout is `[H, K, V]`; **vLLM stores `[H, V, K]`** (mamba_utils.py:290-294) — transpose when comparing. Prefill uses the chunked form (chunk 64, HF :300-433);
   we verified chunked == recurrent (rel 4e-7) and incremental == full-sequence (rel 3e-7, `check_vs_hf.py`).
9. **Full attention q_proj is `[q | gate]` per head**: reshape `[L, 24, 512]`, `chunk(2, -1)` → q `[L,24,256]`, gate `[L,24,256]` (HF :786-789; vLLM qwen3_next.py:426-433) —
   *not* `[all q | all gate]`. Output `= o_proj( softmax(..)·V · sigmoid(gate) )` (HF :816-819, sigmoid, vLLM qwen3_next.py:457).
10. **q_norm/k_norm are per-head (256) RMSNorms applied before RoPE** (HF :791-792).
11. **RoPE**: partial, only the first 64 of 256 dims, `rotate_half` half-split style (pairs `(i, i+32)`), `inv_freq = 1/1e7^(2i/64)`, cos/sin via `cat(freqs, freqs)`
    (HF :142-212, :665-708). mrope (`mrope_interleaved`, sections 11/11/10) only matters for image/video positions; for text all three position rows are equal ⇒ plain RoPE
    at the absolute token index. K stored in the cache is post-RoPE.
12. **GQA grouping**: q head `h` uses kv head `h // 6` (HF `repeat_kv` :711-720). Attention scale `256^-0.5` (:757).
13. **MLP** `down(silu(gate(x)) · up(x))` (HF :823-836); decoder layer is pre-norm with two residual adds (HF :860-913). No embedding scaling; `lm_head` is a plain untied Linear (:1685).
14. **MTP head** (VLLM qwen3_5_mtp.py): inputs are the target's **post-final-norm hidden** `h_p` at position `p` and the embedding of the *next* token `x_{p+1}`
    (shared `embed_tokens`): `x = fc( cat[ pre_fc_norm_embedding(embed(x_{p+1})), pre_fc_norm_hidden(h_p) ] )` — **order is [embedding, hidden]** (:162-165);
    then one full-attention decoder layer with `residual = x` (:177-181, layer called with residual=None), then `mtp.norm` on the layer output (:188), then the shared `lm_head`
    (:246-256, :310-320). The drafter's position equals `p` (positions are not shifted, only input ids: VLLM v1/spec_decode/llm_base_proposer.py:839); MTP has its own KV cache
    which for the prompt is filled with the pairs `(h_p, token_{p+1})`, `p = 0..L-2`. Chained draft `k+1` feeds the MTP's own `mtp.norm` output as `hidden` and draft `k` as the token,
    at position `p+k` (uncommitted cache entries, dropped afterwards; llm_base_proposer.py:724-766). The MTP attention layer has the same q|gate structure as target full-attention layers.
15. **K/V fp8 scales exist but are ignored** by the oracle (see §1).
16. **HF fla/triton path needs a GPU**: with `fla` importable, HF dispatches `torch_chunk_gated_delta_rule` to Triton kernels (it fails with `0 active drivers` on this CPU-only run);
    `check_vs_hf.py` sets `sys.modules['fla']=None` to force HF's pure-torch reference.

## 3. Numerics modes

`Oracle(bf16=False)` fp32 everywhere (weights dequantised to fp32, exact: `(nibble-8)·bf16_scale` is exact in fp32). `Oracle(bf16=True)` / `--bf16-rounding`:
bf16 round-to-nearest-even of every activation HF-bf16 would round (list in `reference.py` docstring); attention scores/softmax, delta-rule state and recurrence, g, l2norm,
and matmul accumulation stay fp32; logits are fp32.

## 4. Performance of the oracle itself (CPU, i7-12700K, 18 threads, other jobs running)

Dequant of a `[17408,5120]` weight ≈ 80 ms; chunked `QuantizedLinear.matmul` never materialises a full fp32 layer. Measured: prefill of 3 prompts (32+34+33 tokens) 62 s (cold page
cache), decode step for the 3 sequences together ≈ 22–28 s (≈ 10 s for one sequence alone); full dump (prefill + 8 decode + MTP k=3) 370 s wall (RAM use not measured; one layer fp32-dequantised chunk at a time, packed layer ≈ 200 MB).
