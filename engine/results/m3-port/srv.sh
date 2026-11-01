#!/bin/bash
cd <repo>
PY=~/llm-lab/venv/bin/python
O=~/llm-lab/port-m3/srv
mkdir -p $O
for r in 1 2; do
  for s in ours vllm; do
    echo "=== r$r $s $(date +%T)"
    $PY server/ab_visit.py $s --round $r --out $O 2>&1 | grep -v -i warn
  done
done
echo "=== done $(date +%T)"
