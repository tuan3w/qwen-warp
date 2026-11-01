#!/usr/bin/env bash
# try_modes.sh <root>: `try --profile` and `try --only` through the real CLI (GPU lease must be held).
R="${1:-$HOME/llm-lab/swift-campaign-test}"
CAMP=$HOME/.omp/plugins/node_modules/agent-campaign/skills/autonomous-campaign/scripts/campaign
$CAMP --root "$R" init t_prof "profile test" && $CAMP --root "$R" brief t_prof 5 >/dev/null
echo "== try --profile --only code"; $CAMP --root "$R" try t_prof --profile --only code 2>&1 | grep -v -i warn
echo "== try --only math"; $CAMP --root "$R" try t_prof --only math 2>&1 | grep -v -i warn
