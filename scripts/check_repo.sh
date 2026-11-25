#!/usr/bin/env bash
# Checks that need neither a GPU nor the checkpoint (run by CI, runnable locally): warpc compiles and lints, uv.lock matches pyproject.toml,
# relative links in the markdown files resolve, no personal paths or key-like strings are committed.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q warpc/src warpc/tests warpc/tools && echo "warpc compile ok"
uv run --package warpc --only-group dev ruff check warpc && uv run --package warpc --only-group dev ruff format --check warpc && echo "warpc ruff ok"
uv lock --check && echo "uv.lock ok"
bad=0
for f in README.md AGENTS.md CONTRIBUTING.md SECURITY.md warpc/README.md; do
  dir=$(dirname "$f")
  while read -r link; do
    p="${link%%#*}"; [ -z "$p" ] && continue
    [ -e "$dir/$p" ] || { echo "broken link in $f: $link"; bad=1; }
  done < <(grep -oE '\]\((\.\./|\./)?[A-Za-z0-9_./-]+(#[A-Za-z0-9_-]*)?\)' "$f" | sed -E 's/^\]\(//; s/\)$//' | grep -vE '^https?:')
done
git grep -nE '/home/[a-z]+|/media/[a-z]+|/dev/shm/|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}' -- . ':!uv.lock' ':!scripts/check_repo.sh' && { echo "personal path or key-like string found"; bad=1; } || true
[ "$bad" = 0 ] && echo "docs and secrets ok"
exit "$bad"
