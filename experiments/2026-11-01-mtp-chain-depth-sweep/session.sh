#!/bin/sh
# usage: sh session.sh <log> '<shell commands>'   - waits for the GPU lease (polls every 2 s), runs the commands from the experiment dir, releases the lease.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$1; CMD=$2
until "$ROOT/projects/llm-lab/bin/lab" gpu acquire --holder RunE9Depth >/dev/null 2>&1; do sleep 2; done
echo "lease acquired $(date)" >> "$LOG"
cd "$ROOT/experiments/2026-11-01-mtp-chain-depth-sweep"
sh -c "$CMD" >> "$LOG" 2>&1
echo "exit $? $(date)" >> "$LOG"
"$ROOT/projects/llm-lab/bin/lab" gpu release --holder RunE9Depth >> "$LOG" 2>&1
