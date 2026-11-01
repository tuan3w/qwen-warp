#!/usr/bin/env bash
# with_lease.sh <holder> <command...>: wait for the lab GPU lease (repo root = cwd), run the command, release the lease (only if we acquired it).
set -u
holder="$1"; shift
until projects/llm-lab/bin/lab gpu acquire --holder "$holder" >/dev/null 2>&1; do sleep "${POLL:-15}"; done
echo "[with_lease] acquired by $holder at $(date +%T)" >&2
"$@"; rc=$?
projects/llm-lab/bin/lab gpu release >/dev/null 2>&1
echo "[with_lease] released at $(date +%T), rc=$rc" >&2
exit $rc
