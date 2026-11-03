#!/usr/bin/env bash
# Download the INT4 checkpoint (about 19 GB).  The repo is gated: accept its licence on Hugging Face and run `uv run hf auth login` first.
# Usage: scripts/fetch_model.sh [target-dir]      (default ~/models/swift-1.5-int4, which is where the engine and server look; override with $SWIFT_MODEL)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-$HOME/models/swift-1.5-int4}"
uv run --group dev hf download ukisai/Swift-1.5-Qwen3.8-27b-INT4 --local-dir "$DEST"
echo "model in $DEST"
