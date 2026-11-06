#!/usr/bin/env bash
# T1 harness: compile (single TU, only when a source changed) + run under the shared GPU lock + print the table.
#   bash run.sh                 QUICK subset (2 GEMV shapes, 3 kept fractions, attention at 32k/128k): ~25 s incl. a 16 s recompile, ~10 s when cached
#   bash run.sh --full          everything (5 GEMV shapes, all kept fractions / modes / 10 kernel configs, attention f = 100/50/25/10/0 %): a few minutes
#   bash run.sh --part=gemv|attn   one half only;  --launches=N  launches per timing graph (default 100);  --out=DIR  json directory
# Needs the exported weights in /media/fatman/data/sparselab/kern (python3 export_weights.py, one time).  Debug: KERN_DBG=1 (attention selections vs reference), KERN_CFGDBG=1 (every GEMV config).
set -euo pipefail
export LC_ALL=C
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SMK_ENGINE:-$HERE/../..}"
BIN=/media/fatman/data/sparselab/kern/kern
SRCS=("$HERE"/kern.cu "$HERE"/kern_gemv.cuh "$HERE"/kern_attn.cuh "$HERE"/attn_sparse.cuh "$ENGINE"/src/gemv.cu "$ENGINE"/src/attn.cu "$ENGINE"/src/kvrot.cuh "$ENGINE"/src/common.cuh)
args=(); mode=--quick
for a in "$@"; do if [ "$a" = "--full" ]; then mode=; else args+=("$a"); fi; done
t0=$(date +%s.%N)
if [ ! -x "$BIN" ] || [ -n "$(find "${SRCS[@]}" -newer "$BIN" 2>/dev/null)" ]; then
  nvcc -arch=sm_86 -O3 -std=c++17 -lineinfo -diag-suppress 177 -I"$ENGINE/include" -I"$ENGINE/src" -I"$HERE" "$HERE/kern.cu" -o "$BIN"
fi
t1=$(date +%s.%N)
flock /tmp/gpu-engine.lock bash -c 't=$(date +%s.%N); "$0" "$@"; echo "[gpu] run $(echo "$(date +%s.%N) - $t" | bc) s (lock wait excluded)"' "$BIN" $mode "${args[@]}"
t2=$(date +%s.%N)
printf '\n[wall] compile %.1fs, lock wait + run %.1fs, total %.1fs\n' "$(echo "$t1 - $t0" | bc)" "$(echo "$t2 - $t1" | bc)" "$(echo "$t2 - $t0" | bc)"
