# qwen-warp infographic and experiment history

`infographic.svg` is the poster (1200 px wide, self-contained, system fonts only, no scripts or external references: GitHub renders it as an
`<img>`, on light and dark themes, because it carries its own background panel). `infographic.png` is the same poster for social previews.
`history.html` is the single-file interactive experiment history (open it in a browser: filters by outcome, expandable cards with the question,
the prediction frozen before the data, the result and the dead ends, a timeline, the prediction scoreboard and code tabs).

The story it tells: *an experiment to test autonomous research: can a human directing AI agents build a faster Qwen serving stack than vLLM?*

## Files

| file | what |
|---|---|
| `data.json` | **every number shown, each with a `src` id; `sources` maps ids to pointers** (file + section, or a command). The one file to edit. |
| `gen.py`, `poster.py`, `svgkit.py`, `history.py` | generator, standard library only |
| `campaign_snapshot.json` | the autonomous campaign (ledger rounds, gate durations, tree, per-lineage agent notes), copied from the harness state that lives outside the repo |
| `lease_snapshot.json` | GPU lease blocks (start, end, coarse kind, program item) from the lab lease log |
| `human_prompts.json` | time, length and class of the human's messages (no text; unique prompts only, so counts are lower bounds) |
| `exp_texts.json` | question / prediction / answer / dead ends copied from each experiment README |
| `snippets_cache/` | code from a campaign lineage repo (outside this repo) used in one snippet |

## Regenerate

Run it in the research repository this was built from: `data.json` refers to its commit hashes (for timeline times and snippet text), so the generator cannot rebuild the poster from a clone of the public repo. The committed `infographic.svg`, `infographic.png` and `history.html` are the outputs.

```bash
python3 gen.py            # writes infographic.svg and history.html here (needs git history; reads committed state, not the working tree); --out DIR elsewhere, --stats prints the derived numbers
```

Derived at generation time from git (`git show HEAD:...`): commits of the program, wall-clock span, CUDA source lines, test files, experiment outcome tallies,
prediction tallies, GPU hours (from the lease snapshot), timeline timestamps, and the snippet text (pinned to the commit named under each snippet, with the
`file:line` pointer printed, so the pointer stays true after the engine changes). The footer prints the commit the page was generated from.

PNG: open `infographic.svg` in Chromium at 1200 px width, take a full-page screenshot at device scale 1, optionally quantise with Pillow
(`Image.quantize(256)`) to stay below 1.5 MB.

## How to update numbers

The campaign-winner (m3) numbers are **campaign-harness numbers**: engine port and server-level verification pending. When they exist:

1. Edit `data.json` -> `campaign`: set `gate_verdict`, `held_out`, `plain_guard`, `dev_tokps_control`, `dev_tokps_m3` to the new values and
   put the verified ratios in `server_verified` (e.g. `{"code": 1.09, "prose": 1.06}`), and change `label` and the `port` entry in `sources`.
   Everything derived from them (the gauge on the poster, the tiles, the waterfall step, the progression bars, the footer wording) follows.
2. Edit `series.steps[*].vals` if BENCH.md has new rows (each step carries the `src` pointer to the BENCH section).
3. `python3 gen.py`, then re-render the PNG.

To refresh the snapshots (maintainers with the lab machine), run the private refresh script; it is not part of the public export.

## Provenance rules

Every number has a pointer in `data.json["sources"]` (`gen.py` fails if a datum references an unknown source). Numbers that are derived are marked as
derived on the poster (for example the campaign bars: 167.9 x 1.107, and the plain-decode step 18.105 / 1.052). Nothing in the poster is estimated by hand.
