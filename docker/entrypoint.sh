#!/usr/bin/env bash
# PROFILE (default|long) and EXTRA_ARGS (any serve.py flags, e.g. "--max-model-len 262144 --prompt-lookup 0") come from docker-compose.yml / .env.
# Arguments given to `docker compose run qwen-warp ...` are appended and win over the defaults above them.
set -euo pipefail
cd /app
# shellcheck disable=SC2086
exec python server/serve.py --model "${SWIFT_MODEL}" --host 0.0.0.0 --port 8000 --profile "${PROFILE:-long}" ${EXTRA_ARGS:-} "$@"
