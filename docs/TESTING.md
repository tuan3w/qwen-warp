# Testing

The engine is hand-written CUDA, so a wrong rounding or a missed write does not crash: it gives a plausible token that is slightly worse, and nobody notices until the quality
has drifted. These tests exist to make that class of bug loud. They are built from what llama.cpp, vLLM and SGLang do (read from their sources, not from memory); the table says where
each idea comes from.

## What runs where

| Test | Needs | Time | Catches |
|---|---|---|---|
| `tests/test_dspine_quant.py` | CPU | 3 s | quantiser error, layout, NaN / inf scales on extreme inputs |
| `tests/test_toolcall_validator.py` | CPU | 1 s | the tool-call validator itself: 16 good / broken calls (a validator that accepts everything checks nothing) |
| `tests/test_dspine_vocab.py` | CPU + tokenizer files | 10 s | draft-vocabulary coverage per language, structural facts of the list |
| `tests/test_dspine_ops.py` | GPU | 20 s | the W4 GEMV at the drafter's shapes: NMSE vs fp64, NaN / inf, determinism, out-of-bounds writes, rounding contract |
| `tests/test_dspine.py` | GPU + both checkpoints + golden | ~6 min | 13 prompts (code, prose in 5 languages, 3 tool calls, 2.0k and 8.5k-token documents): lossless output, tool-call validity, determinism, acceptance floor, drafter numerics vs the torch reference |
| `tests/make_dspine_golden.py` | GPU + 8 GB RAM | 5 min | (regenerates the golden file; run on purpose, review, commit) |

`make test` runs them all. GitHub CI has no GPU and no checkpoint, so it cannot; run `make test` before a release and after every change to `engine/src/dspine*.cu`, `engine/dspine.py`
or the quantiser. Stop the serving container first (the engine needs the card).

## What the data covers

| Kind | Prompts | Why |
|---|---|---|
| code | Python, Java, SQL | the main workload; highest acceptance |
| prose | English, Vietnamese x2, Chinese, French | different tokenisation, different draft-vocabulary coverage (see Known gaps) |
| tool calls | one string argument; typed arguments (integer, boolean, optional); a Vietnamese request with an array and a boolean | structured output is the first thing to break when numerics are slightly off |
| long context | ~2.0k and ~8.5k tokens of this repository's docs | 8.5k is past the drafter's 8,192-token attention window, the only way to run the window cut-off |

Tool calls are judged structurally against the tool's own JSON schema (`tests/toolcall_check.py`): the right function, every required argument, no unknown argument, every value of the declared
type, then the values the request implies. They run with thinking off so the call comes inside the token budget. The speculative output equals the reference ids by construction, so a tool-call
failure here is the target path (engine numerics, KV cache, chat template) giving a worse answer, which is exactly the regression class `docs/USAGE.md` records (39% -> 92% from a key order).

## The principles

**1. Compare against an independent reference, with the same weights.** `tests/dspine_ref.py` is a torch implementation written from the checkpoint and the published algorithm, not
from the CUDA code. It runs in two modes: with the dequantised W4 weights the kernels really multiply with (a difference to the CUDA drafter is a kernel bug) and with the bf16
checkpoint (a difference to the W4 one is quantisation quality). Mixing the two hides kernel bugs inside the quantisation error: a cosine of 0.991 to bf16 tells you nothing about
whether a kernel is right. *(llama.cpp `tests/test-backend-ops.cpp`: every op runs on the backend under test and on the CPU reference, error = NMSE `mse(a, b) / mse(a, 0)`, default
tolerance `1e-7`, per-op overrides with a comment saying why.)*

**2. Tolerances come from measurements, with headroom, and are written down.** The W4 GEMV matches the fp64 reference to NMSE 8.6e-14, so the bound is 1e-10: three orders above fp32
accumulation noise, five below what rounding the weight or the activation to bf16 would cost (1e-6 and up). A bound of 1e-6 would have passed that bug. The quantiser bounds are the
measured values times 1.25. *(llama.cpp `tests/test-quantize-fns.cpp`: `MAX_QUANTIZATION_TOTAL_ERROR = 0.002`, 2 / 3 bit variants measured and set separately.)*

**3. NaN and inf are not "large errors".** Any NaN fails; an inf must be an inf of the same sign in the reference. *(llama.cpp `test-backend-ops.cpp`, the compare callback: NaN check,
`isinf_or_max` sign match, then NMSE.)*

**4. Poison the output, fence it.** The GEMV output is filled with NaN and surrounded by guard words: an element the kernel never writes (a tile of the grid missed) stays NaN and an
out-of-bounds write overwrites a guard. Without this a partial-coverage bug passes on stale memory that happens to be right. *(llama.cpp `test-backend-ops.cpp`: `sentinel_size = 1024`
after every tensor, `memcmp` after the run.)* The first CUDA drafter had four real bugs (a block-wide norm computed per thread, a message read from the wrong row, an fp32 buffer read as bf16, and a Python attribute that shadowed the
engine's MTP handle so `reset()` zeroed the drafter's weights) and all of them surfaced as NaN or a cosine of 0.74 with no pointer to the cause. It took a per-stage comparison to find
them. The golden file stores the reference's residual stream after every layer; the test compares the last one (the only stage the engine exposes), a per-layer comparison needs a debug
hook in `smk_dspine_read` and would localise a bug to its layer in one run.

**5. Data must look like production, and be seeded.** The target's residual stream reaches 2.6e4 and has outlier channels; N(0, 1) inputs never exercise the overflow and rounding paths.
The tests use heavy tails, one huge channel, tiny scale, zeros. Every sample is seeded, including the ones that ignore a local generator (`torch.distributions.*.sample` uses the global
RNG: an unseeded Student-t made the quantiser bound change by 10% between runs until it was found).

**6. Lossless means equal until a near-tie.** The speculative loop must produce the ids of a plain greedy decode through the same T = 8 verify kernel (`verify` + `commit(1)`). It is
not always bit-identical: partial accepts are committed through a different chunking of the DeltaNet state, so a rounding differs and a tie can flip (measured: prompt `code_sql` flips at
token 164, the target's top-2 are `),` and `)` at logits 28.98 / 28.84; a bf16 step at that magnitude is 0.125). The test therefore requires identical ids up to the first difference and
demands that the difference is a near-tie of the target's own top-2 logits (margin <= 0.3, exactly the two tokens in question; the seven flips measured over 13 prompts had margins 0.005-0.136); a real bug diverges far from a tie. vLLM handles the same
effect by requiring only 60-90% exact matches against a T = 1 reference (`tests/v1/e2e/spec_decode/utils.py`, `assert_request_outputs_match`) and with `check_logprobs_close(near_tie_tol=...)`
in `tests/models/utils.py` (a token missing from one side's top-k is accepted when it is within the tolerance of the other side's lowest logprob).

**7. Quality is its own check, because a bug in the drafter leaves the output correct.** A wrong rounding in the bridge costs tokens per cycle, not ids. Per prompt, tokens per cycle must
stay at or above 85% of `tests/data/dspine_reference.json` and the mean at or above 95%. *(vLLM `tests/v1/e2e/spec_decode/acceptance_rates/utils.py`: mean acceptance length >= 95% of the
reference figure per dataset; its DFlash2 test pins 6.14 on Qwen3.8-27B.)* The prompts span code, English, Vietnamese, Chinese and French so a data regression in one language is visible.

**8. Same request twice, same bytes.** *(llama.cpp `test-backend-ops.cpp` runs the graph four more times and compares bytes.)*

**9. A golden file has a fingerprint.** `tests/data/dspine_golden.npz` stores the signature of the quantised checkpoint; if the checkpoint, tokenizer or quantiser changes the test says
"regenerate" instead of failing in numerics with no explanation.

## Known gaps (measured, not hidden)

* **Draft vocabulary.** The head scores 68,864 of 248,320 tokens (`engine/draft_vocab_ext_68864.npy`, built for the MTP head). Coverage of real text, `tests/test_dspine_vocab.py`:
  code 99.9%, English 99.9%, Vietnamese 98.7%, Chinese 81%, French 61%. Output stays exact (the target decides); acceptance on languages with tokens outside the list is lower. The test pins
  these numbers so a change cannot silently lose coverage.
* **Prefill and the MTP path** have their own older tests (`tests/test_group_prefill.py`, `test_spec_procs.py`) that do not yet use NMSE / poison / golden data.
* **No perplexity / KL run against a held-out text.** llama.cpp's CI runs `llama-perplexity` and bounds the result (`ci/run.sh`, `check_ppl`), and `tools/perplexity` can compare a model against
  saved base logits (KLD, mean / RMS delta-p, same-top-p). SGLang's `KLDivergenceMixin` bounds the KL between the decode-time and the prefill-time logprobs of the same tokens (thresholds
  0.002 decode, 0.01 prefill). The engine has the pieces (teacher-forced logits, `tests/test_kv_dtypes.py`); a KL-vs-reference-logits check on a fixed text is the next test to add.
* **GitHub CI cannot run any of this** (no GPU, gated checkpoint). `tests/test_dspine_quant.py` needs only torch on the CPU and is the first candidate for a CI job.

## Adding a test

1. Seed everything; write the seed and the date next to any measured number.
2. Measure the real value, set the bound with headroom, say in a comment what a bug of the class you fear would score.
3. Include extreme inputs (outliers, zeros, tiny, huge) not only random normal data.
4. Compare against an independent implementation with the same weights before comparing against the unquantised model.
5. If the output can be wrong without a crash, add a check for the thing that would degrade (acceptance, KL, coverage), not only for the ids.
