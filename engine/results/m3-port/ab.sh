#!/bin/bash
# interleaved A/B: base (HEAD engine, engine_base) vs new (m3 port, engine); caller holds the lease
cd <repo>
PY=~/llm-lab/venv/bin/python
O=~/llm-lab/port-m3/ab
for r in 1 2 3; do
  for b in base new; do
    d=engine; [ $b = base ] && d=projects/swift-mk/engine_base
    echo "=== r$r $b $(date +%T)"
    $PY $d/tests/bench_spec.py --ks 3 --reps 3 --fast-prefill --no-breakdown --json $O/spec_${b}_r$r.json > $O/spec_${b}_r$r.log 2>&1
    $PY $d/tests/bench_e2e.py --ctxs 64,2048 > $O/e2e_${b}_r$r.log 2>&1
    SMK_TMAX=8 $PY $d/tests/bench_pld.py --suite e6 --reps 1 --modes base,pld --json $O/pld_${b}_r$r.json > $O/pld_${b}_r$r.log 2>&1
  done
done
echo "=== done $(date +%T)"
