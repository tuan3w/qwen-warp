#!/usr/bin/env bash
# resync.sh [rev] [root] [src]: rebuild the campaign root from a (new) engine commit.  CPU only; no GPU lease needed.
# `campaign setup` never overwrites an existing base/, so a new commit needs base/ and every derived state removed: lineages, campaign/ (state, ledger, tree), baseline/, private/tf_tokens.json +
# ref_tpc.json (references of the OLD pristine engine; prepare.py deletes them).  The prompts, harness files and campaign.json are regenerated.  Afterwards: `campaign baseline` on the GPU.
set -euo pipefail
REV="${1:-HEAD}"; R="${2:-$HOME/llm-lab/swift-campaign}"; SRC="${3:-$HOME/llm-lab/swift-campaign-src}"
H="$(cd "$(dirname "$0")" && pwd)"; REPO="$(cd "$H/.." && pwd)"
CAMP=$HOME/.omp/plugins/node_modules/agent-campaign/skills/autonomous-campaign/scripts/campaign
cd "$REPO"
find "$R" -mindepth 1 -maxdepth 1 -type d ! -name private ! -name adapter ! -name oracle ! -name templates -exec rm -rf {} + 2>/dev/null || true   # base, campaign, baseline, lineages, tmp
rm -f "$R/resource.lock"
$HOME/llm-lab/venv/bin/python "$H/adapter/prepare.py" --root "$R" --src "$SRC" --rev "$REV" --force
$CAMP --root "$R" setup "$SRC"
(cd "$R/base" && nice -n 10 make -j8 BUILD=build all core | tail -1)
echo "root $R ready at engine $(cat "$SRC/.engine_commit"); next (GPU lease held, no other job): $CAMP --root $R baseline"
