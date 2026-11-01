# swift-mk autonomous optimisation campaign (agent-campaign harness)

Agents edit private copies of `engine/` and call `campaign try`; a harness they cannot touch decides keep / revert with paired pristine
control runs, a hidden held-out workload, a plain-decode guard and a teacher-forced correctness check. Package: `~/.omp/plugins/node_modules/agent-campaign`
(skill `autonomous-campaign`; its `references/harness-design.md` failure table is what every choice below answers).

**Engine state this harness is calibrated for: K11 (MTP chain + prompt-lookup tail, verify T <= 8)** = engine commit `a2f79e4` (HEAD `5f82489` at the resync: the engine directory is identical, only the index / dashboard changed after it).
It replaces the f306273 calibration (K9 head + K10 prefill, T <= 4, k = 3, no prompt-lookup); every number below is from the K11 engine.

| path | what |
|---|---|
| `campaign.json` | config (metrics, gate, guards, lock, `forbidden` base list) |
| `task.md`, `templates/brief.md`, `focus.tsv` | agent text (task, brief with the contract lines) and the `init <lineage> <focus>` foci (post-K11: gemv / verify / attn / smallk / mtp / lmhead / width / profile) |
| `adapter/metric.py` | `dev` / `try` / `held` / `plain` metrics (+ internal `tmax4`, `mkrefs`): build, measure, record the correctness data |
| `adapter/check.py`, `adapter/compare.py`, `adapter/limits.json` | correctness check (no GPU), greedy-agreement hook, calibrated limits |
| `adapter/prepare.py`, `adapter/held_prompts.py`, `frozen/e6_prompts.json` | builds the root: exports the committed engine, freezes prompts; the held-out texts are never copied into the root. `frozen/` = the 6 E6 tasks used (token ids; E6's `runs/` dir is gitignored) |
| `adapter/hlib.py`, `adapter/profile_run.py` | shared helpers; nsys target of `try --profile` |
| `resync.sh` | `resync.sh [rev] [root] [src]`: rebuild the root from an engine commit (CPU only; then `campaign baseline` on the GPU) |
| `tests/` | `calibrate.py`, `noise.py` (`chain` mode), `mechanics.py` (harness tests with broken lineages), `validate.sh` (stages of a validation under one lease, `SKIP_BASELINE=1`, `PHASES`, `CASES`), `smoke.sh`, `try_modes.sh`, `with_lease.sh` |

The campaign root `~/llm-lab/swift-campaign` lives outside the repo (disk is ~3 GB free). A lineage is ~7 MB (1.3 MB of committed sources + its own `build/`, gitignored and deletable);
`share` is empty because the library is rebuilt per tree. `base/` (the pristine control) is the committed engine at the commit passed to `prepare.py` (`<src>/.engine_commit`).

## Launch (exact commands)

Preconditions: the engine commit you want to optimise is committed and no other job uses the GPU: `prepare.py` exports `git archive <rev>` of
`engine` only, it prints the uncommitted engine changes it did NOT export. No vLLM / `serve.py` / other GPU process may run for the whole campaign
(`metric.py` waits 5 min for a foreign compute process holding >= 400 MiB of GPU memory, then fails the leg; the 3090 is used exclusively). Hold the lab lease for the campaign.

```bash
cd <repo> && export RSRCH_MODEL=claude-sonnet-5-5 RSRCH_AGENT=omp
C=~/.omp/plugins/node_modules/agent-campaign/skills/autonomous-campaign/scripts/campaign; R=~/llm-lab/swift-campaign
campaign/resync.sh HEAD            # CPU, ~1 min: export HEAD engine -> $R/base, harness, frozen prompts, build (no lease needed). Re-run after any commit to the engine / the harness
projects/llm-lab/bin/lab gpu acquire --holder swift-campaign          # exclusive GPU for the whole campaign; release at the end
$C --root $R baseline          # ~14.5 min GPU: 3 dev + 2 held + 3 plain launches on base/; also writes private/{tf_tokens,ref_tpc}.json (pristine references)
# lineages (one per focus; see focus.tsv):
while IFS=$'\t' read -r name focus; do $C --root $R init "$name" "$focus"; done < campaign/focus.tsv   # or init only the 3 you start with
$C --root $R brief gemv 90     # writes <R>/gemv/TASK.md; spawn one omp `task` per lineage with ONLY: the lineage dir, the contract lines, TASK.md
$C --root $R gate gemv         # after its session returns (detached: nohup setsid ... &); then `plan`, `expand`, `merge` as in the skill
```

Foci: `campaign/focus.tsv` = (a) `gemv`, (b) `verify` (DeltaNet T <= 8 snapshot ring traffic / lazy replay), (c) `attn` (prep+reduce fusion, T > 4 tiles), (d) `smallk` (small kernels + launch structure),
(e) `mtp` (cycle overheads: catch-up rows, draft chain, accept kernel), (f) `lmhead` (int4 head + rescoring select kernel), (g) `width` (verify-width policy: proposer / tail length / gate), (h) `profile` (whatever `try --profile` shows).
Start 3 at a time (one shared GPU lock); agents' `try` runs and gates serialise on `$R/resource.lock`.

## What the harness measures

All metrics are engine-direct (no HTTP) and run the engine **as the server ships it with `--prompt-lookup auto`**: `SpecEngine(tmax=8)` (9-slot DeltaNet ring, verify T <= 8) + `enable_pld()` (MTP chain k = 3 + host prompt-lookup tail <= 4,
E6's best cell), `cycles_pld(2)` (the server's 2 cycles per host sync), greedy exact verification / exact speculative sampling; k < 3 and plain decode run the plain graphs on the same tmax=8 engine. `metric.py` builds the tree
(`nice -n 10 make -j8 BUILD=build all core`, build time not scored), waits for an idle GPU, loads `SpecEngine` + `FastPrefill` (bf16 KV, `ctx_max` = longest prompt + 1084), warms the graphs (every k, tail widths, a copy-heavy prompt)
and the clocks for 4 s, then times with `time.perf_counter` around `e.cycles_pld(2)` / `e.decode` (prefill and the proposer's `pld_begin` excluded; rate = emitted tokens / decode wall time = `tests/bench_pld.py` logic).
The emitted-token count is the length of the host-side token list, not a device counter the candidate could misreport.

| metric | workload | score |
|---|---|---|
| `main` dev (gate) | greedy, k=3 + tail, 384 new tokens, 6 prompts frozen as token ids at `prepare` time: **code** (32 tok) / **prose** (34) / **math** (33) / **long8k** (8030, built once from `topics/*.md`) = the E0 prompts (tail fires in 4-20% of their cycles) + **edit** (E6 a7: add validation + docstrings to two methods of a 3.6K-token file; E = 5.0 tokens/cycle, tail in 41% of cycles) and **table** (E6 b2: nvcc/ptxas log -> table, 4.3K tokens; E = 4.9, tail 54%); 2 reps, median | geometric mean over the 6 prompts of (cycles/s x pristine tokens/cycle), tok/s units (~164.7 at `5f82489`; raw tok/s 169 / 108 / 168 / 132 / 225 / 218) |
| `main` try | same, 1 rep; `--only a,b` subset; `--profile [--only p]` = nsys kernel table (ms/cycle) of 32 shipped cycles instead | same |
| `main` held (hidden) | T=1 top_p .95 top_k 20 sampling (exact speculative-sampling graphs, seeded; 2 reps with seeds 101/102), 256 tokens: **4 original prompts** (JSON tool call 206 tok, SQL 154, 3K-token document QA from `papers/*.md`, reasoning puzzle 107) at k=2 (plain chain graphs on the 9-slot-ring engine) **and** k=3 + tail; **PLD-friendly** E6 c3 (agent turn: edit_file with quoted anchors, 3.5K tokens) and a8 (refactor into a dataclass, 5.6K) and **PLD-hostile** E6 d1 (long-document opinion piece, 3.2K) and e4 (recurrence derivation, 82) at k=3 + tail; + plain k=0 sampled on the JSON prompt = 13 cells | geomean over the cells of cycles/s x pristine tokens/cycle, x a collapse factor on the mean tokens/cycle (charges a drop of more than 8% only) |
| `plain` (guard) | plain greedy k=0 decode on the tmax=8 engine, 256 tokens, code (32) and long8k (8030) contexts, 2 reps | geomean tok/s, ratio must be >= 0.99 |

Why this set (the user's concern: the dev score must not reward the prompt-lookup tail alone): the tail is part of both the pristine control and the candidate, so only a change in its *cost or yield* moves the ratio. The dev set mixes
4 prompts where the tail rarely matters (E0: code / prose / math / 8K summary, 4-20% tail cycles, the 8K prompt being the one place the tail is a small deterministic loss, -2%) with 2 typical (not extreme: the whole-file-copy tasks
E6 a1-a3 with E = 7.7 were left out, they pin every cycle at T = 8) copy-heavy tasks, so a T = 8-only speed-up is worth 2/6 of the exponent in the geomean, a tail-free speed-up 4/6; the held workload then asks the
other questions: friendly prompts under *sampling* (delta-tail speculative sampling), hostile prompts (tail rarely fires: a wider-verify change that slows T = 4 shows there), k = 2 (no tail) and k = 0.
`campaign.json`'s `forbidden` list carries markers and token-id windows of every dev / held / oracle prompt, `try` mode runs the *dev* command so an agent never executes the held code path.

## Correctness: `check` (teacher-forced, calibrated) and `compare`

Free-run text equality is not used (near-ties flip: BENCH.md K7 follow-up). `metric.py dev` records, inside the same engine process, data that `check.py` (CPU only) compares with the pristine control
(the gate's `g-ctrlA` for gate runs, `baseline/main-dev1` for `try`):

1. **Teacher-forced logits.** The 3 oracle prompts (32-34 tokens), the 8K prompt and the copy-heavy `edit` prompt are fed with the *pristine* greedy continuation (96 / 33 / 65 generated tokens, `private/tf_tokens.json`, made once on `base/` by plain T=1
   greedy decode) at **T=8** (the widest hybrid verify step: DeltaNet 9-slot ring, attention T=8 tiles, 8-row head select), **T=4** (chain verify) and **T=1** (plain decode); per position the top-32 logits are saved. Check: top-1 agreement >= 0.96 with 0 *unexcused* flips
   (a flip is excused only if the candidate's top-1 is the control's top-2 and the control's margin < 0.4 = `NEAR_TIE` of `tests/test_mtp.py`), top-20 |dlogit| median <= 0.08, p99 <= 0.75, max <= 3.0, per mode.
2. **Greedy speculative decoding == pristine continuation** (64 tokens; 8K prompt 33, edit prompt 65; a divergence must be a near-tie of the control's T=1 / T=4 / T=8 data at that index, same rule): chain only (k=2, k=3, k=3 with forced partial acceptance
   [1], [2,1,3], [1,4,2,3,1]) on the 3 oracle prompts; **chain + tail** (k=3 + prompt-lookup) natural and with forced acceptance of up to **8 rows** ([8], [1], [5,2,7,1,8,3], [3,4,1,6,2,8,5,7]: `a <= f` rollback of the DeltaNet 9-slot ring and its conv slots,
   KV overwrite, 8-row MTP catch-up, ring wrap-around ~25x per 200 tokens) on the oracle prompts, on the 8K prompt (natural, [5,2,7,1,8,3]) and on the `edit` prompt (copy-heavy: the tail fires in ~40% of the cycles, natural + 4 patterns); 39 generations.
3. **The tmax=4 library default** (`Engine()`, 5-slot ring; the server's engine with `--prompt-lookup 0`, or auto when `--max-model-len` > 28672): decided **yes, `check` covers it**, as a cheap subprocess of the gate's checked legs (35 s): a fresh process loads
   `SpecEngine(tmax=4)` and records T=4 / T=1 teacher-forced logits of the 3 oracle prompts + the 15 chain-only greedy generations with forced rollback ([1], [2,1,3], [1,4,2,3,1] wrap a 5-slot ring) that also run on the tmax=8 engine. Reason: the two engines
   share kernels but not their ring-size constants (`DS_SLOTS` is a runtime value read by every kernel that advances `cur`), the default engine is a shipped configuration, and mutation `ring5` (off-by-one only when `slots == 5`) is invisible to every tmax=8 datum.
   An engine holds ~19 GB, so this runs in its own process before the shipped-config engine is loaded. `try` skips it (the gate has it).
4. **Sampler:** at 7 fixed positions (5 at T=4 rows 0-3, 2 at rows 6 and 7 of a T=8 step) the engine's device sampler (T=1, top_k 20, top_p .95, seeded; `smk_test_draw`) draws 20000 samples: TV vs the analytic distribution of its own logits <= 0.012 and
   vs the control's analytic distribution <= 0.12, no token outside the support.
5. **Exact speculative sampling:** `smk_test_trials` (the graph's own draft-pick / accept / residual / bonus code) on 300k synthetic k=3 cycles: TV of the first emitted token and of the emitted-length histogram vs the analytic
   law <= 0.01, mean length within 0.02 (pristine 0.0016); and the K11 **chain 3 + delta tail 4** version (`smk_test_trials_pld`, 300k cycles, 7 verify rows): same statistics (pristine 0.0016 / 0.0017) + the accepted tail token must equal the
   proposed one (0 strays), `d_t` is never emitted at a tail rejection (0), the tail-rejection and bonus distributions (n >= 3000) within TV 0.03 of the analytic ones (pristine 0.0045 / 0.0094; `test_sampling.py trials-pld` has the 2e6-cycle chi-square version).
6. `tests/test_ops.py` (21 op checks, 2 s) of the candidate build, only if the test file is byte-identical to the pristine one (otherwise skipped and flagged in the facts: an agent must not weaken its own test).
   The `test_mtp.py` / `test_pld_engine.py` greedy / rollback / ring sections are re-implemented natively in (2) against the control's margins; their draft / head sections only concern acceptance (visible in the score).

`compare` = mean over prompts of (longest common prefix of the first 128 greedy k=3 + tail tokens, held: 64) / length, needs >= min(0.7, control-vs-control - 0.03): a cheap smoke test for gross output changes.

**Correctness-data tiers (gate cost).** The package checks only `g-dev1` against `g-ctrlA`; `g-dev2` / `g-ctrlB` only repeat the timing, so `metric.py dev` records no correctness data in those two legs (`LIGHT_LEGS`, 59 s instead of 177 s each).
Consequence: those legs read ~0.25% lower (measured: full legs 164.50-164.58, light legs 164.10-164.20 on the same pristine tree: the correctness work before the timing leaves the clocks hot), which cancels in the gate ratio because
each side of (c1 + c2) / (A + B) has one full and one light leg; the `control A vs B` fact of the old design is gone (the two controls differ by the tier offset, `drift_max` 1% still flags a disturbed card).

### Calibration (`tests/calibrate.py` on the test root, engine K11 `a2f79e4`; report `~/llm-lab/swift-campaign-test/calib/report.json`)

Method unchanged: the same source built with 6 other W4 GEMV accumulation orders (env `SMK_CFG`, e.g. `34816:1` = gate|up with K-split 8; these also change the T=5..8 mma path) = "legit candidates the check MUST accept",
through the identical dev command, vs a pristine run. The engine is **bit-reproducible run to run** (two pristine dev runs: every logit at T = 8 / 4 / 1 and on the tmax=4 engine, token, sampler draw and trials statistic identical; 3 baseline launches have identical
tokens/cycle), so pristine-vs-pristine is exactly 0 and the limits come from the envelope of legit rounding changes and from the engine's own width noise (pristine, same teacher-forced tokens: T=1 vs T=4 top-1 1.000, median |dlogit| 0.029, p99 0.20, max 0.77;
**T=1 vs T=8: 1.000, 0.032, 0.32, 0.59**).

| candidate vs pristine | top-1 agree T8 / T4 / T1 | median abs dlogit T8 / T4 / T1 | p99 (max of modes) | max | sampler TV vs control | compare | raw tok/s ratio | scored ratio | cycle-rate ratio (6 prompts) |
|---|---|---|---|---|---|---|---|---|---|
| pristine rerun | 1.000 / 1.000 / 1.000 | 0 / 0 / 0 | 0 | 0 | 0.004 | 1.000 | 1.0006 | 1.0006 | 1.000-1.001 |
| gate\|up K-split 8 | 1.000 / 1.000 / 0.990 | 0.042 / 0.037 / 0.034 | 0.34 | 0.65 | 0.019 | 1.000 | 0.9986 | 0.9945 | 0.986-0.998 |
| down w8k2u3 | 0.990 / 1.000 / 1.000 | 0.034 / 0.035 / 0.037 | 0.28 | 1.17 | 0.014 | 0.941 | 0.9593 | 0.9567 | 0.954-0.958 |
| o_proj K-split 8 | 0.990 / 1.000 / 1.000 | 0.037 / 0.034 / 0.036 | 0.38 | 0.94 | 0.017 | 0.941 | 1.0026 | 0.9976 | 0.996-0.998 |
| qkv\|z K-split 8 | 0.990 / 0.990 / 1.000 | 0.034 / 0.032 / 0.034 | 0.24 | 0.62 | 0.054 | 1.000 | 0.9977 | 0.9931 | 0.987-0.996 |
| attn q\|k\|v w8k4u2 | 1.000 / 0.990 / 0.979 | 0.031 / 0.033 / 0.034 | 0.31 | 1.16 | 0.042 | 0.793 | **1.0077** | 0.9980 | 0.997-1.000 |
| all five | 1.000 / 0.990 / 1.000 | 0.043 / 0.036 / 0.036 | 0.32 | 0.75 | 0.017 | 0.941 | 0.9509 | 0.9472 | 0.941-0.950 |

Every flip of every variant is an excused near-tie (0 unexcused), 54/54 generations equal for 5 of 6 (the 6th, `all five`: two `prose` chain + tail generations diverge at index 48 at the documented near-tie of the prose prompt, excused), trials TV 0.0016 / 0.0017 (kernel-level, identical for every
variant), `tmax4` data of the variants within the same ranges (median <= 0.037, max <= 0.77). The T=8 rows are as close to the pristine as T=4 / T=1 (median <= 0.043, max <= 1.16): `t8` gets the same limits as `t4` / `t1`.
Limits (`adapter/limits.json`) are ~2x the largest legit value of each statistic: median 0.08 (vs 0.043), p99 0.75 (vs 0.38), max 3.0 (vs 1.17), sampler-vs-control TV 0.12 (vs 0.054), `compare_min` 0.7 (vs 0.793), top-1 agreement 0.96 (vs 0.979).
**Acceptance bands (`accept_band`, the score credits / charges the candidate's own tokens/cycle only beyond them; see below):** code 0.01, prose 0.055, math 0.01 (K10 values), **long8k 0.02, edit 0.02, table 0.03** (new: with the tail a whole-cycle difference
is 0.8% of a 384-token run; seen: prose +3.7% (the same rounding flip as f306273's +4.2%), long8k +1.3% in all variants (one near-tie), table +1.5-1.8%, code / math / edit <= 0.3%).

**The score is cycles/s x pristine tokens/cycle** (not raw tok/s): the `attn q|k|v` rounding change scores **+0.77%** raw although the engine got no faster (cycles/s 0.997-1.000) because it flipped a near-tie in the generated prose (tokens/cycle +3.7%);
the scored ratio is 0.998. The candidate's own tokens/cycle counts only beyond the per-prompt band, so a real acceptance collapse (e.g. garbage drafts or a broken tail) is still charged and a real acceptance gain beyond the band credited; held = cycle rate x the
control's tokens/cycle per cell, times a factor that only charges a collapse of the mean tokens/cycle by more than 8% (sampled trajectories change with any rounding).

## How each row of `harness-design.md`'s failure table is handled

| failure | here |
|---|---|
| Stored cool-GPU baseline | gate = ctrlA, cand x2, ctrlB on the same card back to back; speedup = cand / mean(controls); `drift_max` 1% between the controls |
| Free-run text agreement as guard | teacher-forced logits (T = 8 / 4 / 1 + tmax=4) + near-tie-excused flips; free-run only as the loose `compare` (>= 0.7, legit rounding variants measured 0.79-1.0) |
| Guard calibrated too tightly | limits = ~2x the *measured* envelope of legit rounding-level changes (6 different GEMV accumulation orders) and of T=1-vs-T=4 / T=8 width noise, see table |
| Win tuned to one operating point | hidden held workload: sampled instead of greedy, k=2 and k=3 + tail, PLD-friendly / PLD-hostile / 3K-document prompts, plus k=0 plain; plus the plain-decode guard (a spec-only win that slows k=0 fails at 0.99); dev mixes tail-silent and copy-heavy prompts |
| Agent sees held-out info | held texts/ids exist only in `<root>/private`; `try` never runs held; contract forbids listing the root; markers + token-id windows of every dev/held/oracle prompt, `CAMPAIGN`, `/proc/`, `cmdline`, `private` ... are `forbidden` (strings already present in the pristine tree, e.g. `bench_spec`, are dropped from the list because the gate scans context lines too) |
| Gate misses committed work | package does `git reset --soft kept` before diffing; the contract forbids committing anyway |
| State overwrite race | package re-reads state; gates and `try` are serialised by `lock: gpu` |
| Merge blindness / stacks rounding changes | `plan` / `merge` from the skill; the merged lineage goes through the same check, so stacked rounding changes are judged on the teacher-forced limits, not summed per part |

Other holes closed on purpose: a candidate cannot edit its own tests (test_ops hash), cannot misreport the token count (host-side list), SMK_* tuning env vars are stripped from measurements (including `SMK_TMAX`: the harness passes `tmax` explicitly), the GPU must be
exclusive (foreign process => leg fails => `revert-crash` shows the log, re-gate), `cleanup` kills a stray `metric.py` after every lock holder.

## Baseline, noise floor, gate numbers, cost (engine K11 `a2f79e4` = HEAD `5f82489`; lease held; card power-capped at 322-325 W, SM 1530-1605 MHz)

`campaign baseline` on `~/llm-lab/swift-campaign` (14.65 min): **main dev 164.67** (launches 164.59 / 164.73 / 164.70, SD 0.04%), **held 140.54** (140.49 / 140.60), **plain 54.34** (54.37 / 54.32 / 54.31, SD 0.04%).
Dev cells (tok/s | tokens/cycle | cycles with a tail): code 168.9 | 3.59 | 20%, prose 108.0 | 2.26 | 4%, math 167.8 | 3.57 | 19%, long8k 132.4 | 2.90 | 12%, edit 225.4 | 4.99 | 41%, table 217.6 | 4.89 | 54%. Held cells (tok/s): JSON k2 / k3+tail 136.4 / 157.6,
SQL 121.8 / 163.5, docqa 123.5 / 155.2, reasoning 132.8 / 167.9, agent-edit (friendly) 225.1, refactor (friendly) 203.1, doc-opinion (hostile) 109.7, recurrence (hostile) 165.0, plain sampled k=0 54.9.

| metric | pristine runs | SD |
|---|---|---|
| dev (score) | final baseline 164.59 / 164.73 / 164.70; test root: baseline 164.16 / 164.62 / 164.50, 4 noise runs 164.68 / 164.50 / 164.50 / 164.48, chain legs 164.50-164.58 (full tier) | 0.04-0.12%; all of the run-to-run spread is slow clock drift (SM 1515-1605 MHz at 320-325 W); **bit-identical tokens / cycles per run**, inside one run the 2 reps agree to 0.02% |
| held | 140.49 / 140.60 (final), 140.47 / 140.48 (test baseline), 140.38 / 140.36 / 140.35 (noise) | 0.04% / 0.008% |
| plain | 54.37 / 54.32 / 54.31 (final), 54.415 / 54.412 / 54.414 (noise) | 0.04% / 0.003% |

Gate statistic ratio = mean(c1, c2) / mean(ctrlA, ctrlB). `noise.py chain 12` (back-to-back pristine legs in the gate's own tier pattern full, full, light, light): ratios of the four gate-pattern windows (3 in the chain + the `noop` gate): 1.0003 / 1.0002 / 1.0001 / 0.9995;
the other windows of the chain mix the tiers (0.9973-1.0027: the light legs read 0.25% lower, which is what a non-gate-pattern window would see and what cancels in the real pattern). With the all-full legs of the earlier measurements (SD 0.04-0.14%) the
conservative figure is SD ~0.1%. `keep = 1.005` is then 5 SD: a null candidate is kept with negligible probability, the **minimal detectable speedup (80% power) is ~0.6%** (0.5% + 0.84 x 0.1%); +1% is detected, +0.3% never (below the line).
`held_min 0.99` (held SD <= 0.04%, the legit rounding variants measured 0.997-1.001 on cycle rate), `guard_min 0.99` (plain SD <= 0.04%), `drift_max 0.01` (ctrlA vs ctrlB differ by 0.24-0.35% = the tier offset + drift; 1% flags a disturbance, `revert-noisy`, re-gate).

**Cost** (wall, GPU exclusive, measured): full dev leg 177 s (engine load ~10 s + 4 s warm-up, correctness data 86 s = teacher forcing 30 + 39 generations 56 + trials 0.1, tmax=4 subprocess ~30 s, 6 prompts x 2 reps ~25 s), light dev leg 59 s, held leg 102 s, plain leg 39 s,
`baseline` (3 dev + 2 held + 3 plain + references) **14.7 min**. **One gate = ctrlA, c1 (full) + c2, ctrlB (light) + hctrl, held + 2 plain legs = 12.5-12.7 min of GPU** (4 measured gates: 12.46 / 12.65 / 12.68 / 12.69; the first leg of a candidate includes its build; a
rejected-for-crash candidate costs less) -- 1.4x the f306273 harness (9.2 min) for 3 more tail widths, 2 more dev prompts, 9 more held cells and the tmax=4 engine. One agent `try` = ~2.4 min plus lock wait [estimate: a full leg minus the tmax=4 subprocess; `--only` does not skip the correctness data;
`--profile` + one `--only math` try together took 2.5 min, `--profile` has no correctness data]. Budget for a campaign: 3 agents x ~10 `try` per 90 min session = ~72 GPU-min of `try` + 12.6 of gate per session round, i.e. at 3 lineages the GPU is ~80% busy, `try` waits will appear: start 2-3 lineages, not 6.

## Harness mechanics tests (`tests/mechanics.py`, run by `tests/validate.sh` on the test root `~/llm-lab/swift-campaign-test` at engine `5f82489`)

Each case = `campaign init`, a source mutation in the lineage tree, [`try`], `campaign gate` with the real controls / held / guard / check / compare hooks. New K11 cases plus the three accounting ones (6 cases, ~80 GPU-min; the older cases `slower`, `sign`, `ret`, `skipmlp`, `longctx`, `fp8kv`, `forbidden`, `keep` did not change and were NOT rerun at K11):

| case (mutation) | decision | scored speedup | check / guard / held |
|---|---|---|---|
| `noop` (comment in a .cu: identical binary; also `try`) | **revert**, "no gain" | 0.9995 | check PASS 54/54 generations equal, `try` printed `check: PASS`, ctrl drift 0.35% (tier offset), held 0.999, plain 0.999 |
| `rounding` (gate\|up K-split 8: legit reassociation; T=8 mma path included) | **revert**, "no gain" | 0.9934 | check PASS (t8 median 0.042, p99 0.24; tmax4 median 0.037; sampler TV vs control 0.019), held 0.997, plain 0.997: the kernel is slightly *slower* at T=5..8 (cycle-rate 0.986-0.998), not rewarded |
| `luck` (q\|k\|v w8k4u2 instead of w8k8u2: raw tok/s +0.8% in calibration only because a rounding change moved prose tokens/cycle +3.7%) | **revert** | 0.9992 | check PASS, compare 0.793 (>= 0.7), held 1.001, plain 0.998: the score does not reward it |
| `pldtail` (greedy accept kernel accepts every tail token once the 3 chain drafts matched, without verifying) | **revert-output** although **1.238x faster** | 1.2377 | teacher-forced logits identical (the mutation does not touch them), speculative sampling untouched (held 1.000); caught by the chain + tail generations: 8 of 54 wrong (`code`, `long8k`, `edit` natural and forced [8] / [5,2,7,1,8,3] / [3,4,1,6,2,8,5,7], first wrong token at index 13-34) and by `compare` 0.46 / 0.46 / 0.54 |
| `ringwrap` (DeltaNet 9-slot ring: `cur` update off by one when it wraps, slots > 5) | **revert-output** | 0.9355 | every speculative generation wrong from token 4-23 (chain and chain + tail, prose from index 4), compare 0.14; the teacher-forced logits are fine (single steps never wrap): the forced-acceptance generations are the only thing that sees ring wrap |
| `ring5` (same, only when the ring has 5 slots = the tmax=4 library default) | **revert-output** | 0.9990 | invisible to every tmax=8 datum (all 39 + t8 / t4 / t1 pass, speed 0.999, held 1.000); caught only by the tmax=4 subprocess: 15/15 `m4:` generations wrong from index 2-17 -> this is why `check` covers the default engine |

The reasons of the three `revert-output` cases are in `<root>/campaign/ledger.jsonl` (`facts`). With the expectations as written all 6 cases passed at the first full run.
NOT tested for lack of a real correct speed-up in the tree: keep of a genuine gain (the keep path itself is covered by the lowered-threshold `keep` case of the f306273 run).

## Harness bugs / design defects found while adapting to K11

1. **Profile phase**: `profile_run.py` sized `ctx_max` for the profiled prompt only while the new warm-up runs 1500 tokens of the copy-heavy prompt (`ctx_max too small`); the old `validate.sh` printed `rc=0` because the failure was inside a `grep` pipeline. Fixed; the nsys table now shows the shipped cycles (`k_w4` 255 + 20 calls, `dn_kernel<4>` 33 + `<8>` 15 calls per cycle at 20% tail).
2. **Score still luck-prone for the tail prompts**: the first band values (long8k 0.01, table 0.02) were exceeded by every legit variant (long8k +1.25%: one near-tie, tokens/cycle of a 384-token run is quantised at 0.8% per cycle) -> bands 0.02 / 0.03 (see Calibration).
3. **The correctness cost made the gate 1.3x as long** (16.5 GPU-min with 4 full legs vs 12.6): only `g-dev1` and `g-ctrlA` are ever checked -> `LIGHT_LEGS` (0.25% systematic offset, cancels in the ratio; see tiers) and `try` skips the tmax=4 subprocess.
4. A dev prompt with an extreme copy share (E6 a2: E = 8.0, every cycle verifies T = 8) was the first choice for the copy-heavy slots: it made the dev score a T=8-only metric. Replaced by a7 / b2 (E = 5.0 / 4.9, tail in 41% / 54% of the cycles).
5. `calibrate.py` hard-coded 5 sampler probes and 3 teacher-forced names; it now follows `hlib.TF_NAMES` / the probe list and also reports per-prompt tokens/cycle ratios (the data the bands come from).

## Not covered / honest limits

* The correctness check is teacher-forced on 3 short prompts + the 8K prompt + one copy-heavy prompt, and greedy generations of 33-65 tokens; a bug that only shows for other lengths, `fp8` KV, k=1, or sampled spec-decode *inside the full graph* (the trials tests cover the accept kernels in isolation, the probes cover the sampler on engine logits, including T=8 rows) is not caught;
  the held workload only measures speed. The proposer (`src/pld.cu`, host code) is not tested for equality with E6's simulation by the harness (`tests/test_pld.py` does that; `test_ops.py` is the only engine test re-run); a proposer change that stays exact is just a speed / acceptance change.
* Rounding-level changes cannot be told apart from small real approximations by construction (limits are 2x the envelope of 6 legit K-split variants); `fp8kv` was rejected at f306273 only by 1 flip.
* A candidate can still make the score better by trading acceptance within the noise band (prose 5.5%, long8k / edit 2%, table 3%, code / math 1%) against cycle time: bounded loss <= ~2.4% of the geomean ((5.5 + 2 + 2 + 3 + 1 + 1) / 6).
* The tail is optional in the engine (`enable_pld`); a candidate that disables it in `spec.py` slows the dev copy-heavy prompts (2 of 6) and the held friendly cells: not rewarded.
* The 3 `try` modes and the gate run the same python (`metric.py`) from the root; an agent that edits `spec.py` / `engine.py` / `prefill.py` / `pack*.py` controls what those do (that is legitimate engine work); token counts and timing come from the harness side.
* The tmax=4 coverage is T=4 / T=1 teacher forcing + chain-only generations on the 3 short prompts (no 8K / sampling); `try` does not run it.
* Measured on one card with the power cap at 322-325 W; a different thermal state changes absolute numbers, not the paired ratios. The light dev legs read 0.25% lower than the full ones (clock state after the correctness work).
