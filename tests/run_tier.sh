#!/usr/bin/env bash
# Run every test of one tier:  tests/run_tier.sh 0|1|2      (PY=<python> to pick the interpreter, GPU_WRAP=<cmd> to prefix GPU runs, e.g. GPU_WRAP=/tmp/qw/gpu.sh)
# A test file declares the tiers it runs in with lines "Tier: <n> [arguments]" at the start of a line of its docstring, one line per tier (the arguments of that tier).
#   Tier 0: no GPU, pure logic, < 15 s in total.   Tier 1: GPU, engine level, < 2 min.   Tier 2: server level (one server start per configuration), < 4 min.
# Prints PASS/FAIL and the wall time of each test and of the tier; exits non-zero if any test failed.
set -uo pipefail
export LC_NUMERIC=C  # printf %f and bc agree on the decimal point whatever the user's locale
cd "$(dirname "$0")/.."
tier="${1:?usage: tests/run_tier.sh 0|1|2}"
PY="${PY:-python3}"
wrap=()
[ "$tier" != 0 ] && [ -n "${GPU_WRAP:-}" ] && wrap=("$GPU_WRAP")
mapfile -t lines < <(grep -HE "^Tier: ${tier}( |$)" tests/test_*.py)
[ "${#lines[@]}" -gt 0 ] || { echo "no test declares tier $tier"; exit 1; }
fail=0
t_all=$(date +%s.%N)
for l in "${lines[@]}"; do
  f="${l%%:*}"
  args="${l#*Tier: ${tier}}"
  t0=$(date +%s.%N)
  # shellcheck disable=SC2086  # the declared arguments are word-split on purpose
  if "${wrap[@]}" "$PY" "$f" $args > "/tmp/run_tier_${tier}_$(basename "$f" .py).log" 2>&1; then r=PASS; else r=FAIL; fail=1; fi
  printf '%s  %s%s  (%.1f s, log /tmp/run_tier_%s_%s.log)\n' "$r" "$f" "$args" "$(echo "$(date +%s.%N) - $t0" | bc)" "$tier" "$(basename "$f" .py)"
done
printf 'tier %s: %s in %.1f s\n' "$tier" "$([ $fail = 0 ] && echo 'all passed' || echo 'FAILURES')" "$(echo "$(date +%s.%N) - $t_all" | bc)"
exit $fail
