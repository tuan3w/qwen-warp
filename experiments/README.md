# Experiments

Pre-registered experiments behind the engine (prediction written and committed before the measurement; `README.md` of each folder holds
question / prediction / answer / confidence / dead ends; `results/` holds the raw outputs that the numbers point to).

The numbering E0..E11 in the texts is the original program numbering. Two experiments of that program are not part of this release
(E8 and E10: a prompt-set study and a draft-list coverage study); where a table or text mentions E8 it means the English chat task set
of E8, whose prompt file is not included. Numbers that remain are unchanged from the original tables.

Pointers such as `topics/`, `papers/`, `vendor/`, `runs/`, `~/llm-lab/...` refer to the private research workspace or to large local artefacts that are not published.
Withheld for licence hygiene: `2026-10-31-lm-head-quantisation-quality/results/docs.json` and `2026-10-31-ngram-draft-potential/results/trace.npz`
(token-id lists of third-party documents); the summaries computed from them are kept, the scripts that regenerate them need the original corpora.
Engine state referenced here: research commit `b8b8feb`.
