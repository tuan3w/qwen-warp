---
title: mtp-draft-head-ablation
type: experiment
tags: []
status: done
created: 2026-10-31
author: agent
model: claude-sonnet-5-5
question: "Does the cheap draft path assumed by the E3 model (int4-quantised MTP layer + draft argmax restricted to a 32k-token vocabulary, optionally plus all prompt tokens, and a fake-quantised int8/int4 draft lm_head) keep Swift-1.5's MTP draft acceptance (mean accepted length and per-position acceptance at k=3, greedy and T=1 top_p0.95/top_k20, 384 new tokens, >=24 held-out prompts: code/math/prose/agent/long-context) within 3% of the unmodified BF16-MTP + full-248k-head arm, with the verification path (full-vocab BF16 target) unchanged?"
prediction: "Arms are relative to arm 0 (unmodified; two runs give the noise floor), mean accepted length per step at k=3, verification untouched. (1) Restricted draft vocab, frequency-chosen from vendor/+topics/ text: V=64k loses <=1%, V=32k loses 1-4%, V=16k 3-8%, V=8k loses >15% on code (bet 25%) because code identifiers and non-Latin tokens fall out. (2) V=32k + all prompt-context tokens: loses <=3% on code and math, <=6% on prose (bet: code 1.5%, math 2%, prose 4%, agent 2%, long 2%). (3) MTP layer weights fake-quantised: int8 per-channel loses <=2%; int4 g128 RTN loses 3-8% (bet 5%, larger on code/math where acceptance is high because later positions compound: pos-3 acceptance falls most). (4) Draft lm_head fake-quantised: int8 loses <=1%, int4 g128 loses <=3% (bet 1.5%). (5) Combined V=32k+ctx, MTP int4, head int8: loses <=10% (bet 7%); with head int4 bet 8%. Verdict rule: an arm is 'free' if mean accepted length is within 3% of arm 0 and within the arm-0 run-to-run noise; I expect free = {V=64k, head int8, head int4, MTP int8, probably V=32k+ctx on code/math}, not free = {V=8k, V=16k on code, MTP int4, combined}. Consequence if the combined arm loses ~7%: the E3-modelled 1.56x k=3 ratio over stock vLLM falls to ~1.45x."
answer: "On a broader held-out set (25 prompts: 6 code/6 math/6 prose/4 agent/3 long, greedy + 2x T=1, 384 tokens, k=3, vLLM 0.30 in-process, verification untouched), the cheap draft path is mostly free but the 32k restricted vocabulary is not: int4-g128 RTN MTP layer 0.990/0.998 (greedy/T=1) of baseline mean accepted length, int8 MTP 1.001/1.006, draft lm_head int8 1.002/1.002 and int4 1.000/0.997 are all FREE (>=0.97); restricted vocab V=64k 0.994/0.990 FREE, V=48k+prompt+generated tokens 0.986/0.982 FREE, but V=32k 0.938/0.944, V=32k+prompt 0.954/0.961, V=32k+prompt+generated 0.967/0.977, V=16k 0.869/0.879, V=8k 0.816/0.833 are NOT free (target tokens outside V: 7.4% at 32k, 15% on prose, because the vendor/+topics/ corpus is code/technical). The E3 combined design (V=32k+prompt, int4 MTP, int8 head) keeps 0.944/0.951, which the E3 step-time model with per-arm draft bytes turns into 1.51x/1.52x over stock vLLM k=3 (E3 assumed 1.56x at no loss); a 48k+prompt+generated variant is predicted (independence, not measured) at 0.976/0.982 and 1.54x/1.55x."
confidence: medium
answers: [docs/PROGRAM.md]
builds_on: []
reproduce: LAB_MIN_FREE_GB=2 projects/llm-lab/bin/lab run --detach experiments/2026-10-31-mtp-draft-head-ablation/run.py
---

# mtp-draft-head-ablation

## Question

Does the cheap draft path assumed by the E3 model (`experiments/2026-10-30-w4-multirow-verify-and-power/results/e2e_tables.md` Q3d (b): int4 MTP layer 0.22 GB + 32k-token draft head 0.33 GB = ~0.75 ms/draft step instead of vLLM's BF16 layer + full 248k BF16 head = ~3.9 ms) keep Swift-1.5's MTP draft acceptance within 3% of the unmodified arm, on held-out prompts, with verification (full-vocab BF16 target) untouched? Which of the pieces (restricted draft vocab, vocab + prompt/generated tokens, int8/int4 MTP layer, int8/int4 draft lm_head) are free and which are not?

## Prediction

Frozen in the frontmatter (committed in `a2de66b` before any run). Summary of the bets: V=64k <=1% loss, V=32k 1-4%, V=16k 3-8%, V=8k >15% on code; V=32k+prompt tokens <=3% code/math, <=6% prose; MTP int8 <=2%, MTP int4 3-8% (bet 5%); head int8 <=1%, head int4 <=3%; combined (V=32k+prompt, MTP int4, head int8) <=10% (bet 7%), implying ~1.45x instead of 1.56x. Verdict rule: FREE iff mean accepted length within 3% of arm 0 and within noise.

## Method

**Engine.** vLLM 0.30.0 (lab venv), `~/models/swift-1.5-int4` untouched, `language_model_only`, `{"method":"mtp","num_speculative_tokens":3}`, `kv_cache_dtype=auto`, `max_num_seqs=4`, prefix caching off, V2 model runner, `VLLM_ENABLE_V1_MULTIPROCESSING=0` so the engine runs in the harness process (`run.py`). Deviation from E0 (`run.py` of the baseline): `max_model_len=9728` / `gpu_memory_utilization=0.86` instead of 16384 / 0.93, to leave ~2 GB for the quantised draft head on the 24 GB card (affects KV capacity only). **Attention backends chosen** (`results/arms.json` `info.attention_backends`, log line `Using FLASHINFER attention backend out of potential backends: ['FLASHINFER', 'TRITON_ATTN']`): FLASHINFER for the 17 full-attention layers (16 target + the MTP layer), GDN_ATTN for the 48 Gated-DeltaNet layers (GDN decode kernel `cuda`, FLA/Triton prefill). With FlashInfer + spec decode vLLM logs `CUDAGraphMode.FULL_AND_PIECEWISE is not supported ... setting cudagraph_mode=PIECEWISE`, so the draft `compute_logits` runs eagerly outside CUDA graphs.

**Patch point** (one engine, loaded once, arms switched by python state). Every draft step - step 0 after the verify forward and steps 1..k-1 of the fused decode loop - calls `self.model.compute_logits(h)` in `vllm/v1/worker/gpu/spec_decode/speculator.py:400` (`sample_draft`; with the default `draft_sample_method="greedy"` the token is `argmax` of those logits and is treated as one-hot by the rejection sampler). `Qwen3_5MTP.compute_logits` (`vllm/model_executor/models/qwen3_5_mtp.py:303`) is replaced at class level by `linear(h, head)[:, :V] + BIAS`: `BIAS` is a bf16 0/-inf allowed-token vector (restricted vocab), `head` is the target lm_head (the MTP shares the target's embed and lm_head, `v1/worker/gpu/spec_decode/eagle/utils.py:load_eagle_model`) or an int8-stored fake-quantised copy dequantised chunk-wise (1.27 GB; a bf16 copy does not fit next to the 17.7 GB model + KV on 24 GB). The MTP layer weights (`fc`, `qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`; all bf16 in the checkpoint, `info.mtp_linears`) are fake-quantised in place (`copy_`) from a CPU copy of the originals and restored between arms. The target model, its lm_head and the rejection sampler are never touched, so output is lossless up to bf16 batch-shape numerics: greedy token streams are byte-identical to arm 0 for 19-25 of 25 prompts in every arm (`results/summary.json` `output_identical_to_base`); T=1 streams diverge once any draft differs (the T=1 rows are therefore unpaired draws, see noise below).

**Arms** (`run.py` `ARMS`): `base` (also `base2`, `base3` = same config again), `vNk` = draft argmax restricted to the top-N tokens by frequency (N = 8192, 16384, 32768, 49152, 65536; always includes all 33 added/special tokens, the 256 byte-level base tokens and every token of the rendered chat templates + `chat_template.jinja`: 598 forced ids), `+ctx` = also all tokens of the chat-formatted prompt (static per prompt), `+gen` = also every token the MTP layer sees during decode steps (target-accepted/bonus tokens and drafts; prompt via ctx; hooked in `AutoRegressiveSpeculator._run_model`), `mtp-int8` (per-output-channel symmetric RTN), `mtp-int4` (symmetric g128 along the input dim, compressed-tensors style scale = amax/7.5, q in [-8,7], RTN), `mtp-int4clip` (same with a per-group MSE-optimal clip ratio in {1.0, ..., 0.5}), `head-int8` / `head-int4` (same quantisers on the draft lm_head rows only), `comb-head8` / `comb-head4` = V=32k+ctx, `mtp-int4`, head int8 / int4. Fake-quant = quantise then dequantise to bf16 (no int kernels; the point is the acceptance effect).

**Vocabulary list** (`corpus.py`): token frequencies of 29.97 M tokens of vendor/ source (py 40 MB, md 25 MB, c/h/cpp/cu/cuh 8+8+8+6+4 MB, rs/ts 10 MB each; per-file cap 100 KB, `results/vocab_corpus.json`) + `topics/*.md`, tokenised with the model tokenizer. It never reads papers/, sources/, experiments/ or any eval prompt. Only 68,340 distinct tokens occur; the top 8k/16k/32k/64k cover 90.9/96.2/99.2/99.99% of the corpus (V=64k therefore ~ "every token that occurs in the corpus"). Held out: all 25 prompts are hand-written in `prompts.py`; the 3 long-context prompts (4,036 / 6,045 / 8,040 tokens) are built from papers/ and sources/ notes.

**Prompts** (25): 6 code (Python LRU+pytest, C linked list, SQL window functions, bug-fix, explain-decorator, argparse log script), 6 math/reasoning, 6 prose/chat (story, email, explain-to-child, itinerary, essay, dialogue), 4 agent (JSON tool calls, shell commands, JSON extraction, plan with tool calls), 3 long-context summarise/QA. Chat template with `enable_thinking=False`. One request at a time (bs=1), `max_tokens=384`, natural EOS allowed, greedy (seed 0) and T=1/top_p 0.95/top_k 20 with 2 seeds (0, 1). 75 requests per arm.

**Metric.** Per request, deltas of `vllm:spec_decode_num_drafts`, `..._accepted_tokens`, `..._accepted_tokens_per_pos` (as E0). Mean accepted length E = 1 + accepted/drafts pooled over a set of requests; per-position acceptance = accepted_at_pos/drafts. Relative = arm E / base E on the same prompts. 95% CI: paired bootstrap over prompts (stratified by category; the two T=1 seeds of a prompt move together), 4000 resamples (`report.py`). Verdict rule as pre-registered: FREE iff pooled relative E >= 0.97 for greedy and T=1; "(CI)" if the lower CI bound also >= 0.97; per-category cells below 0.97 are flagged.

**Noise.** `base2` and `base3` reproduce `base` exactly (9342 drafts, E 2.954: same engine, same seeds => deterministic), so duplicated arm 0 measures determinism, not noise. The realistic floor is how far the numerically near-no-op arms move: `mtp-int8` and `head-int8` deviate by at most 0.55% pooled (CI half-width <= 0.8%), and by up to ~1.4% in single categories (4-6 prompts each). Treat pooled differences < ~1% and category cells within ~+-2% as noise.

```bash
LAB_MIN_FREE_GB=2 projects/llm-lab/bin/lab run --detach experiments/2026-10-31-mtp-draft-head-ablation/run.py   # all arms (skips finished ones)
nice -n 10 ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/corpus.py     # vocab frequencies (before run.py)
nice -n 10 ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/oov.py        # needs outputs/base3.json written by run.py
nice -n 10 ~/llm-lab/venv/bin/python experiments/2026-10-31-mtp-draft-head-ablation/report.py     # results/summary.json, tables.md
```

## Results

2026-10-31, RTX 3090 (lab lease), torch 2.13.0, vLLM 0.30.0, transformers 5.18.0 (repo HEAD 29f0e28 at the last results write, `manifest` in `results/arms.json`; the arms were produced by 3 launches of the same `run.py`, base..comb-head4 first, then base3/v32k+gen/v16k+gen, then v48k/v48k+gen). Raw: `results/arms.json` (per prompt x sampler drafts/accepted/per-position/token-stream hash), `results/summary.json`, `results/tables.md` (all tables incl. absolute E and per-position), `results/oov.json`, `results/vocab_corpus.json`. Generated token ids: `~/llm-lab/runs/2026-10-31-mtp-draft-head-ablation/outputs/` (not in git). Base (arm 0) mean accepted length at k=3 over the 25 prompts: greedy 3.077 (code 3.40, math 3.49, prose 2.45, agent 3.47, long 2.91; per-position 0.86/0.68/0.54), T=1 2.896 (0.80/0.61/0.48). E0's single prompts: math 3.49 (same), code 3.59 (here 3.40), prose 2.33 (here 2.45).

**Accepted length relative to base, "greedy / T=1"** (cells: pooled over the category's prompts; 'all' with 95% paired-bootstrap CI over prompts; last column = greedy per-position acceptance, base 0.86/0.68/0.54):

| arm | code | math | prose | agent | long | all [95% CI] | pos 1/2/3 | verdict |
|---|---|---|---|---|---|---|---|---|
| base2 / base3 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | exact replay of base | 0.86/0.68/0.54 | determinism only |
| v64k | 0.999 / 0.994 | 0.996 / 0.993 | 0.985 / 0.977 | 0.995 / 1.003 | 1.000 / 1.003 | 0.994 [0.990,0.997] / 0.990 [0.984,0.997] | 0.86/0.67/0.53 | FREE (CI) |
| v48k | 0.996 / 0.988 | 0.973 / 0.972 | 0.957 / 0.956 | 0.981 / 0.988 | 0.983 / 0.990 | 0.975 [0.966,0.983] / 0.974 [0.966,0.982] | 0.84/0.65/0.51 | FREE (CI straddles 0.97); prose cell <0.97 |
| v48k+gen | 0.997 / 0.996 | 0.990 / 0.992 | 0.966 / 0.954 | 0.995 / 0.997 | 1.000 / 0.999 | 0.986 [0.979,0.991] / 0.982 [0.974,0.989] | 0.84/0.67/0.52 | FREE (CI); prose cell <0.97 |
| v32k | 0.978 / 0.961 | 0.922 / 0.937 | 0.910 / 0.926 | 0.962 / 0.953 | 0.944 / 0.970 | 0.938 [0.926,0.950] / 0.944 [0.930,0.956] | 0.81/0.61/0.47 | NOT free |
| v32k+ctx | 0.986 / 0.975 | 0.946 / 0.955 | 0.921 / 0.936 | 0.978 / 0.959 | 0.981 / 1.028 | 0.954 [0.945,0.965] / 0.961 [0.947,0.973] | 0.82/0.63/0.49 | NOT free |
| v32k+gen | 0.989 / 0.981 | 0.969 / 0.983 | 0.939 / 0.953 | 0.988 / 0.982 | 0.981 / 1.026 | 0.967 [0.958,0.977] / 0.977 [0.966,0.987] | 0.83/0.64/0.50 | NOT free (greedy 0.967) |
| v16k | 0.907 / 0.892 | 0.839 / 0.852 | 0.849 / 0.877 | 0.919 / 0.913 | 0.857 / 0.869 | 0.869 [0.851,0.887] / 0.879 [0.859,0.896] | 0.76/0.53/0.39 | NOT free |
| v16k+gen | 0.940 / 0.962 | 0.921 / 0.931 | 0.902 / 0.906 | 0.962 / 0.957 | 0.972 / 0.984 | 0.929 [0.916,0.941] / 0.939 [0.921,0.956] | 0.80/0.60/0.46 | NOT free |
| v8k | 0.857 / 0.865 | 0.786 / 0.814 | 0.801 / 0.823 | 0.868 / 0.851 | 0.782 / 0.824 | 0.816 [0.795,0.838] / 0.833 [0.812,0.853] | 0.71/0.47/0.33 | NOT free |
| mtp-int8 | 0.993 / 1.004 | 1.008 / 1.010 | 1.001 / 1.008 | 1.010 / 1.001 | 0.995 / 1.006 | 1.001 [0.997,1.006] / 1.006 [0.998,1.014] | 0.86/0.69/0.54 | FREE (CI) |
| mtp-int4 (RTN) | 0.992 / 0.987 | 1.002 / 1.010 | 0.979 / 0.989 | 0.990 / 1.024 | 0.990 / 1.014 | 0.990 [0.984,0.995] / 0.998 [0.988,1.009] | 0.85/0.67/0.52 | FREE (CI) |
| mtp-int4clip | 0.987 / 1.004 | 1.014 / 1.008 | 0.997 / 0.999 | 1.004 / 0.996 | 0.978 / 1.015 | 0.997 [0.991,1.003] / 1.002 [0.990,1.014] | 0.86/0.68/0.53 | FREE (CI) |
| head-int8 | 1.001 / 0.998 | 1.000 / 1.002 | 1.003 / 1.001 | 1.000 / 1.003 | 1.003 / 1.012 | 1.002 [1.000,1.003] / 1.002 [0.995,1.008] | 0.86/0.68/0.54 | FREE (CI) |
| head-int4 | 0.993 / 1.002 | 1.005 / 1.003 | 1.002 / 1.001 | 1.000 / 0.963 | 0.994 / 1.010 | 1.000 [0.995,1.004] / 0.997 [0.986,1.010] | 0.86/0.68/0.54 | FREE (CI); agent T=1 cell 0.963 |
| comb-head8 (v32k+ctx, mtp-int4, head-int8) | 0.973 / 0.953 | 0.944 / 0.957 | 0.906 / 0.924 | 0.966 / 0.969 | 0.972 / 0.995 | 0.944 [0.934,0.954] / 0.951 [0.939,0.963] | 0.82/0.62/0.47 | NOT free |
| comb-head4 (.., head-int4) | 0.972 / 0.952 | 0.946 / 0.967 | 0.903 / 0.924 | 0.966 / 0.981 | 0.969 / 0.980 | 0.943 [0.933,0.952] / 0.952 [0.940,0.965] | 0.82/0.61/0.47 | NOT free |

Per-position acceptance at the full set (greedy; T=1 in `tables.md`): base 0.86/0.68/0.54; the vocabulary arms lose at every position (v32k 0.81/0.61/0.47), the quantisation arms lose at most 0.01 per position (mtp-int4 0.85/0.67/0.52). The three effects multiply: rel(v32k+ctx) x rel(mtp-int4) x rel(head-int8) = 0.946 (greedy) / 0.961 (T=1) vs measured comb-head8 0.944 / 0.951.

**Why the restricted vocabulary loses** (`results/oov.json`, tokens the target generated in the arm-0 run `base3`, greedy + T=1): share outside V = 21.1% (V=8k), 14.5% (16k), 7.4% (32k), 3.5% (48k), 1.4% (64k); at 32k by category code 2.8%, math 6.9%, prose 15.4%, agent 3.4%, long 6.0%. A draft token outside V can never match, so position-1 acceptance is bounded by (1 - OOV): measured relative position-1 acceptance 0.829 / 0.880 / 0.942 / 0.977 / 0.995 vs bounds 0.789 / 0.855 / 0.926 / 0.965 / 0.986 - consistent (draft would often have missed anyway). Adding the prompt tokens cuts OOV at 32k to 6.0% (prose 14.4%, long 2.1%); generated tokens help the rest. The corpus is code/technical text, so story/essay vocabulary (names, rare words) is exactly what is missing; the loss is a corpus-domain effect, not a property of "32k".

**Modelled consequence for E3** (`report.py`; E3 step times, `e2e_tables.md` Q3c, 2K context: verify T=2/3/4 = 21.03/21.28/21.60 ms; draft step = 0.1 ms + MTP-layer bytes/BW + head bytes/BW with each arm's own bytes: layer 0.85/0.43/0.22 GB bf16/int8/int4, head rows x 5120 x {2,1,0.53} B at 883 GB/s, int4 layer at 813 GB/s - this reproduces E3's 3.94 ms (a) for base). Stock vLLM reference = base acceptance at E0's measured k=3 step of 37.2 ms. Modelled fused-engine tok/s at k=3 (all prompts pooled, greedy / T=1) and ratio to stock vLLM k=3:

| arm | draft ms | greedy tok/s, ratio | T=1 tok/s, ratio |
|---|---|---|---|
| base (E3 option (a): vLLM-like draft in the fused engine) | 3.94 | 92, 1.11x | 87, 1.11x |
| v32k + bf16 layer/head (E3 (b) head only) | 1.44 | 111, 1.35x | 105, 1.35x |
| head-int4 only | 1.83 | 114, 1.37x | 107, 1.37x |
| mtp-int4 only | 3.25 | 97, 1.17x | 92, 1.18x |
| comb-head8 (measured acceptance) | 0.56 | 125, 1.51x | 118, 1.52x |
| comb-head4 (measured) | 0.47 | 126, 1.52x | 120, 1.54x |
| V=49152+ctx+gen, MTP int4, head int8 (predicted from the independence of the three effects, NOT measured) | 0.66 | 127, 1.54x | 121, 1.55x |

E3's 1.56x assumed unchanged acceptance and the 0.75 ms (bf16 32k head) draft; with measured acceptance the E3 design point is 1.51-1.52x because an int8/int4 head shrinks the draft step (0.56/0.47 ms) more than the vocabulary loss (-5%) costs. Per category the comb-head8 acceptance factor (greedy / T=1) is 0.97/0.95 (code), 0.94/0.96 (math), 0.91/0.92 (prose), 0.97/0.97 (agent), 0.97/1.0 (long); multiply the E3 per-text ratios (1.56-1.58x) by these.

**Scoring the pre-registered bets** (greedy / T=1 pooled loss unless noted):

| bet | measured | hit? |
|---|---|---|
| V=64k loses <=1% | 0.6% / 1.0% | yes (borderline T=1) |
| V=32k loses 1-4% | 6.2% / 5.6% | no, worse |
| V=16k loses 3-8% | 13.1% / 12.1% | no, worse |
| V=8k loses >15% on code (bet 25%) | code 14.3% / 13.5% (all prompts 18.4% / 16.7%) | no on code (narrowly), yes pooled |
| V=32k+ctx <=3% code & math | code 1.4% / 2.5%, math 5.4% / 4.5% | code yes, math no |
| V=32k+ctx <=6% prose | 7.9% / 6.4% | no |
| MTP int8 <=2% | -0.1% / -0.6% (gain, noise) | yes |
| MTP int4 loses 3-8% (bet 5%) | 1.0% / 0.2% | no, far better than predicted |
| head int8 <=1%, head int4 <=3% | 0.0% / 0.3% (int4) | yes |
| combined <=10% (bet 7%; head int4 bet 8%) | 5.6% / 4.9% (int8 head), 5.7% / 4.8% (int4 head) | yes, better than bet |

## Conclusion

**The quantisation half of the E3 draft design is free; the 32k vocabulary half is not (with a repo-derived list).** int4-g128 RTN MTP layer (0.990/0.998), int8 MTP (1.001/1.006), int8 draft head (1.002/1.002), int4 draft head (1.000/0.997) are all within 3% and mostly within the ~1% noise, on every category (single cells down to 0.963 on agent T=1 for head-int4, 0.979 prose greedy for mtp-int4, i.e. inside the +-2% cell noise). So the int4 MTP layer (0.22 GB) and an int4/int8 draft head are safe: the prediction that int4 MTP would lose 3-8% was wrong (1.0%/0.2%); MSE-clip search is unnecessary. The restricted vocabulary is a quality/bytes trade: with the frequency list built from vendor/ + topics/, V=32k loses 6% (7.4% of the target's tokens are outside V, 15% on prose), prompt tokens recover ~1.6 points and generated tokens another ~1.5 (V=32k+prompt+generated 0.967/0.977, still just under the 3% rule on greedy), V=16k/8k lose 12-18%. V=48k+prompt+generated (0.986/0.982) and V=64k (0.994/0.990) are free; relative to 32k a bf16 head of 48k/64k rows is 0.17/0.34 GB larger (+0.19/+0.38 ms per draft step), half that at int8. The pre-registered V=32k-free bet fails; the corpus-domain explanation is measured (OOV by category) but a prose-rich or chat corpus list was not tried.

**What this does to the E3 model.** Combined E3 design as measured (V=32k+prompt, int4 MTP, int8/int4 head): acceptance 0.944/0.951 of baseline, modelled 125/118 tok/s greedy/T=1 at k=3 = 1.51x/1.52x over stock vLLM k=3, versus E3's 1.56x at unchanged acceptance: the 5% acceptance loss is mostly paid back by the cheaper head. Recommended draft path for the engine: int4 MTP layer + int8 (or int4) head restricted to ~48k rows + prompt/generated tokens; predicted (independence of the three effects, which held to 0.002 greedy / 0.010 T=1 in the one combination measured) 0.976-0.982 of baseline, 1.54-1.55x. That arm was not run. Verdicts stand for the draft path only: nothing here measures engine speed, only acceptance; draft step times are the E3 model with per-arm bytes, not measurements.

Caveats: n=25 prompts (3-6 per category; category cells +-2%), one MTP head quantisation grid (RTN/int8 symmetric fake-quant, not GPTQ with calibration data; clip search tried only for int4 MTP), T=1 rows are unpaired draws (token streams diverge once drafts differ, 0-36 of 50 identical to base), `kv_cache_dtype=auto` (FP8 vs BF16 not confirmed from the log), max_model_len 9728, k=3 only (k=1,2 acceptance is the prefix of the per-position numbers), 2K-context step times from E3 used for all contexts, frequency list from repo code/tech text.

## Dead ends

- **Private bf16 copy of the lm_head for the draft (2.37 GiB)** - the first design (fixed address for CUDA graphs) failed four times on a 24 GB card: (1) the MTP's own throw-away embed + lm_head (4.7 GiB) stay allocated until a gc cycle collection after `load_eagle_model` (fix: `gc.collect()`); (2) allocating the copy before the target compile makes Inductor's combo-kernel autotune OOM (it needs 2.37 GiB scratch); (3) allocating after the KV profile leaves no room for a 4k-token prefill; (4) `kv_cache_memory_bytes` to skip profiling OOMs the same way. Final design: the draft logits use the shared head directly, and the head arms keep an int8-stored copy (1.27 GiB) dequantised in 4096-row chunks (`HEAD_CHUNK`), which is slow (head arms took ~20 min instead of ~6.5) but fits.
- **Assumed the draft ran inside CUDA graphs** (fixed pointers required) - with FlashInfer + spec decode vLLM falls back to PIECEWISE graphs, so `compute_logits` is eager and plain python state works; no graph rebuild was needed.
- **Hooking `Qwen3_5MTP.forward` for the generated-token arms** broke torch.compile's dynamic-shape guards (`ConstraintViolationError: L['input_ids'].size()[0]`); moved the hook to `AutoRegressiveSpeculator._run_model` outside the compiled region.
- **`base2` as a noise estimate** - it is a bit-exact replay (same engine/seeds), so it only proves determinism. Noise had to be estimated from near-no-op arms and bootstrap CIs.
- **GPU lease contention**: the first full launch lost the lease to another agent and exited immediately; later the run was split (v8k+gen, comb-gen-head8, comb48-gen-head8 were defined but not run after a request to stop using the GPU; they were removed from `ARMS`; the dynamic-generated-token variant of the combined arm and the 48k combined arm exist only as an independence-based prediction).

## Follow-ups

- Measure the recommended combined draft path: V=49152 + prompt + generated tokens, int4 MTP, int8 head (`ARMS` entry was removed; re-add `{"vocab": 49152, "ctx": True, "gen": True, "mtp": "int4", "head": "int8"}`), and V=32k+gen with the same quantisers (~25 min each on the GPU).
- Rebuild the vocabulary list from chat/prose + multilingual text (not downloadable here) or from the target's own generations on a training prompt set, and re-measure V=32k; expect prose OOV (15%) to drop the most.
- Build the per-sequence dynamic allowed-token bitmap + restricted-row gather in the fused engine (the `+gen` arm approximates it with a bias vector).
- Adaptive k per text (prose acceptance 0.45-0.50 at pos 2-3) remains open from E0.
