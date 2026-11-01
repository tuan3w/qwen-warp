#!/bin/bash
# rounds 1..3 of the interleaved full suite (ours, mtp-fast, dflash2, dflash15); the caller holds the GPU lease.
cd "$(dirname "$0")/../.."
PY=~/llm-lab/venv/bin/python
for r in 1 2 3; do
  for s in ours mtp-fast dflash2 dflash15; do
    echo "=== round $r $s $(date +%T)"
    $PY experiments/2026-11-01-hyperqwen-vs-swift-mk/run.py visit $s --suite full --round $r 2>&1 | grep -v -i "warn"
  done
done
echo "=== session1 done $(date +%T)"
