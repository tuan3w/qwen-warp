# swift-mk — clean-room CUDA megakernel work for Swift-1.5-Qwen3.8-27B-INT4 on one RTX 3090

Program: [docs/PROGRAM.md](PROGRAM.md).
Everything in this directory is meant to be unit-tested against a trusted numerical reference first.

## oracle/ — CPU reference of the model (text path + MTP head)

Pure-torch, CPU only (never touches the GPU), loads and dequantises **one layer at a time** from the safetensors shards
(`~/models/swift-1.5-int4`), so RAM stays small. Semantics are transcribed from HF transformers 5.18 `qwen3_5` (and vLLM 0.30 for the MTP head), and
`check_vs_hf.py` checks layers 0–3 and the MTP layer against the real HF modules (rel err ≤ 2e-7 in fp32). Not compared against vLLM output.

| file | what |
| --- | --- |
| `oracle/weights.py` | `Checkpoint` (by-name access through the index json), W4A16 pack-quantized dequant (`dequant_w4a16`, `QuantizedLinear`, `load_state_dict`), `selftest()` against compressed-tensors pack/unpack + requantise round trip |
| `oracle/reference.py` | `Oracle` (prefill + token-by-token decode with explicit state, batched over sequences, per-layer taps, MTP head `mtp_forward`/`mtp_draft`), fp32 or `--bf16-rounding` |
| `oracle/dump_ref.py` | writes the reference dumps (3 prompts × prefill + 8 greedy decode steps + MTP drafts k=1..3) |
| `oracle/check_sanity.py` | prints the greedy continuations, MTP agreement, dump integrity; `--long N` regenerates N decode steps |
| `oracle/check_vs_hf.py` | HF-module cross-check (layers 0–3, MTP layer, chunk-vs-recurrent, incremental-vs-full) |
| `oracle/NOTES.md` | tensor/shape table, bytes/token, every semantic gotcha with source pointers |

### How to run

All commands CPU-only (`CUDA_VISIBLE_DEVICES=` empty) with the lab venv; `ORACLE_THREADS` (default 18) sets torch threads. Run from `oracle/`.

```bash
export CUDA_VISIBLE_DEVICES=""; PY=~/llm-lab/venv/bin/python
$PY weights.py                                  # dequant self-test (~4 s)
$PY check_vs_hf.py                              # oracle vs HF modules, layers 0-3 + MTP (~20 s)
$PY reference.py --prompt "What is 2+2?" --n 8  # one-off greedy generation (~10 s/token); --bf16-rounding for the bf16 mode
$PY dump_ref.py                                 # fp32 dumps -> ~/llm-lab/swift-ref/fp32/   (370 s)
$PY dump_ref.py --bf16-rounding                 # bf16-rounding dumps -> ~/llm-lab/swift-ref/bf16/   (436 s)
$PY check_sanity.py [--mode bf16] [--long 32]   # coherent text + MTP agreement (+ regenerate longer text, ~15 min for 32)
```

Use from a test:

```python
import os, sys; sys.path.insert(0, "oracle")
import torch
from weights import Checkpoint, dequant_w4a16, load_state_dict
w = Checkpoint().quantized_linear("model.language_model.layers.3.mlp.down_proj").dequant()   # fp32 [5120, 17408], exact
z = torch.load(os.path.expanduser("~/llm-lab/swift-ref/fp32/code.pt"))
```

### Dump files (`~/llm-lab/swift-ref/{fp32,bf16}/{code,prose,math}.pt`, 470 MB total, outside git)

Each ~82 MB, `torch.save` dict (fp32 tensors; kernel tests can cast). Prompts are the E0 `SHORT` prompts through the checkpoint chat template, thinking off
(32 / 34 / 33 tokens for code / prose / math — shorter than the ~64 requested because the E0 prompts are short). Event `e = 0..8`: `e = 0` is the last prompt token in the prefill pass,
`e = 1..8` the 8 decode steps whose input token is `gen_ids[e-1]`; `gen_ids[e]` is the greedy token produced by event `e` (`gen_ids[0]` comes from the prefill logits).

| key | shape | meaning |
| --- | --- | --- |
| `prompt_ids`, `gen_ids`, `event_tokens`, `event_pos` | [L], [9], [9], [9] | tokens; `event_pos[e] = L-1+e` absolute position of the event's input token |
| `hid_in`, `hid_out` | [9, 64, 5120] | last-token hidden entering / leaving each decoder layer (`hid_in[e,0]` = embedding) |
| `final_norm_out` | [9, 5120] | post `model.norm` hidden (lm_head input; MTP `hidden` input) |
| `top20_val`, `top20_idx` | [9, 20] | top-20 logits per event |
| `drafts` | [9, 3] | chained greedy MTP drafts k=1..3 after event `e`; draft k predicts `gen_ids[e+k]` |
| `mtp_hidden` | [9, 3, 5120] | `mtp.norm` output of each draft step |
| `mtp_taps[e]` | dict | draft-1 internals: `fc_out`, `xnorm`, `q_proj`, `q_rope`, `k_rope`, `v`, `attn_out`, `gated`, `mixer_out`, `h_mid`, `layer_out`, `norm_out` (leading dim 1) |
| `taps[e][layer][name]` | layers 0,1,2 (DeltaNet) and 3 (full attn) | last-token sub-op tensors (leading dim 1): `h_in xnorm mixer_out h_mid xnorm2 mlp_gate mlp_up mlp_act mlp_down h_out`; DeltaNet adds `qkv_proj z a b conv_out q_l2 k_l2 v beta g core_out gated_norm_out` (`q_l2/k_l2` are the 16-head l2-normed q/k before repeat; `core_out` before the gated norm); attention adds `q_proj k_proj v_proj q_rope k_rope v attn_out gate gated` |
| `taps[e][layer]["S_out","conv_state_out"]` | S `[48,128(K),128(V)]`, conv `[10240,3]` | state *after* event `e`; kept for e=0 (layers 0,1,2 → the state after prefill) and e=1,2 (layer 0) |
| `prefill_taps[layer]` | layers 0, 3; all L prompt tokens | `h_in xnorm qkv_proj z a b mixer_out h_mid h_out` (+ `S_in S_out conv_state_in conv_state_out` for DeltaNet, `q_proj k_proj v_proj k_rope v` for attention) |
| `prefill_final` | [L, 5120] | post-final-norm hidden of every prompt token |
| `kv_layer3_after_prefill` | `k`,`v` [4, L, 256] | post-RoPE K and V of layer 3 |

Conventions (details in NOTES.md): DeltaNet state is HF layout `[H, K, V]` (vLLM stores `[H, V, K]`); conv state holds the last 3 pre-conv inputs oldest first; all RMSNorm weights except
`linear_attn.norm` are zero-centred (`1+w`); fp8 KV scales are ignored (KV kept in fp32/bf16-valued).

### Measured facts (this machine, fp32 dump)

* Greedy text is coherent: code → ```` ```python\ndef fibonacci(n: int, memo: dict = None) -> int:\n    """\n    Return the n-th Fibonacci number using memoization. ````;
  prose → `The fog in the North Atlantic didn’t just roll in; it breathed. It pressed against the glass of the lantern room like a living thing, damp and cold,`;
  math → `The statement to be proven is:\n$$ 1 + 3 + 5 + \dots + (2n-1) = n^2 $$` (32 decode steps, `fp32/long32.json`).
* MTP draft agreement with the next greedy tokens, k=1/2/3: 8-decode-step dump (24/21/18 comparisons) 75% / 62% / 50% (code 100/71/67, prose 25/29/0, math 100/86/83);
  32-decode-step run (96/93/90 comparisons) 90% / 76% / 66% (code 100/90/90, prose 72/48/30, math 97/90/77).
* bf16-rounding mode vs fp32: identical greedy ids and drafts on all 3 prompts; relative error of `hid_out` 0.3% at layer 0, 0.8–1.3% at layer 31, 1.0–1.6% at layer 63.
* Runtime: ~22–28 s per decode step for the 3 sequences together (≈ 10 s per token for a single sequence), 62–160 s prefill (page-cache dependent).
