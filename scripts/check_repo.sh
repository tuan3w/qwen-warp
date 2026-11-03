#!/usr/bin/env bash
# Checks that need neither a GPU nor the checkpoint (run by CI, runnable locally): sources compile, uv.lock matches pyproject.toml,
# relative links in the markdown files resolve, no personal paths or key-like strings are committed.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m py_compile engine/*.py server/*.py tests/*.py && echo "py_compile ok"
uv lock --check && echo "uv.lock ok"
bad=0
for f in README.md docs/*.md; do
  dir=$(dirname "$f")
  while read -r link; do
    p="${link%%#*}"; [ -z "$p" ] && continue
    [ -e "$dir/$p" ] || { echo "broken link in $f: $link"; bad=1; }
  done < <(grep -oE '\]\((\.\./|\./)?[A-Za-z0-9_./-]+(#[A-Za-z0-9_-]*)?\)' "$f" | sed -E 's/^\]\(//; s/\)$//' | grep -vE '^https?:')
done
git grep -nE '/home/[a-z]+|/media/[a-z]+|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}' -- . ':!uv.lock' ':!scripts/check_repo.sh' && { echo "personal path or key-like string found"; bad=1; } || true
[ "$bad" = 0 ] && echo "docs and secrets ok"
exit "$bad"
