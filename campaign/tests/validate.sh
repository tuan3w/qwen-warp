#!/usr/bin/env bash
# validate.sh <root>: fresh state on a prepared test root -> baseline -> try --profile / --only -> calibration -> noise -> mechanics, under one GPU lease (run through with_lease.sh).
# A failed baseline keeps the lease up to 12 min for a fix (`touch <root>/validate/retry` reruns it).
R="${1:-$HOME/llm-lab/swift-campaign-test}"
H="$(cd "$(dirname "$0")" && pwd)"
A="$H/../adapter"
PY=$HOME/llm-lab/venv/bin/python
CAMP=$HOME/.omp/plugins/node_modules/agent-campaign/skills/autonomous-campaign/scripts/campaign
# SKIP_BASELINE=1: keep the existing baseline / references / campaign state (only the lineages t_* and the finished phases' logs are dropped): later stages of one validation (e.g. after a lease hand-over)
if [ -n "${SKIP_BASELINE:-}" ]; then
  rm -rf "$R"/t_* "$R/validate"
  $PY - "$R/campaign/state.json" <<'P'
import json, sys
s = json.load(open(sys.argv[1])); s["lineages"] = {k: v for k, v in s["lineages"].items() if not k.startswith("t_")}
json.dump(s, open(sys.argv[1], "w"))
P
else
  rm -rf "$R"/t_* "$R/campaign" "$R/baseline" "$R/calib" "$R/validate" "$R/smoke"
  rm -f "$R/private/ref_tpc.json"
fi
mkdir -p "$R/validate" "$R/private"
for f in "$A"/*.py "$A/limits.json"; do cp "$f" "$R/adapter/$(basename "$f").new" && mv "$R/adapter/$(basename "$f").new" "$R/adapter/$(basename "$f")"; done
$PY - "$A/../campaign.json" "$R/campaign.json" <<'P'
import json, sys
new, old = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
new["forbidden"] = old["forbidden"]   # the generated strings (prompt token windows, held markers) come from prepare.py
json.dump(new, open(sys.argv[2] + ".new", "w"), indent=1)
P
mv "$R/campaign.json.new" "$R/campaign.json"
cp "$A/../templates/brief.md" "$R/templates/brief.md"; cp "$A/../task.md" "$R/task.md"
[ -n "${SKIP_BASELINE:-}" ] || $CAMP --root "$R" setup "$HOME/llm-lab/swift-campaign-test-src" >/dev/null
phase() { name=$1; shift; echo "== $name $(date +%T)"; "$@" > "$R/validate/$name.log" 2>&1; rc=$?; echo "== $name rc=$rc $(date +%T)"; tail -4 "$R/validate/$name.log" | cut -c1-400; return $rc; }
ok=0
[ -n "${SKIP_BASELINE:-}" ] && ok=1
for attempt in 1 2 3; do
  [ $ok = 1 ] && break
  rm -f "$R/validate/retry"
  if phase baseline "$CAMP" --root "$R" baseline && [ -s "$R/private/ref_tpc.json" ]; then ok=1; break; fi
  for i in $(seq 1 72); do [ -e "$R/validate/retry" ] && break; sleep 10; done
  [ -e "$R/validate/retry" ] || break
done
[ $ok = 1 ] || exit 1
# PHASES (default: all) and CASES (mechanics cases, default: all) select what to run, e.g. PHASES="calibrate mechanics" CASES="noop longctx"
PHASES="${PHASES:-profile calibrate noise mechanics}"
for ph in $PHASES; do
  case $ph in
    profile) phase profile bash "$H/try_modes.sh" "$R";;
    calibrate) phase calibrate "$PY" "$H/calibrate.py" "$R" ${CALIB_VARIANTS:-};;
    noise) phase noise "$PY" "$H/noise.py" "$R" "${NOISE_ROUNDS:-3}";;
    noisechain) phase noisechain "$PY" "$H/noise.py" "$R" chain "${NOISE_CHAIN:-12}";;
    mechanics) phase mechanics "$PY" "$H/mechanics.py" "$R" ${CASES:-};;
  esac
done
