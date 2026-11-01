#!/usr/bin/env bash
# smoke.sh <root>: one dev / held / plain measurement of the pristine base (finds harness bugs; creates private/tf_tokens.json on the first run). GPU lease must be held.
set -eu
R="${1:-$HOME/llm-lab/swift-campaign-test}"
export CAMPAIGN_ROOT="$R"
PY=$HOME/llm-lab/venv/bin/python
mkdir -p "$R/smoke"
for k in dev held plain; do
  /usr/bin/time -f "$k: %e s wall" $PY "$R/adapter/metric.py" $k "$R/base" "$R/smoke/$k.json" || { echo "$k FAILED"; exit 1; }
  $PY -c "import json,sys; r=json.load(open('$R/smoke/$k.json')); print('$k score', r['score'], '|', r.get('summary','')[:300])"
done
