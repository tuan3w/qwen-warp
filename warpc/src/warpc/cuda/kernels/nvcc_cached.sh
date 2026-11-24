#!/bin/sh
# nvcc_cached.sh CACHE_DIR NVCC OUT.o SRC NVCC_FLAGS...
# Compile SRC to OUT.o through a content-addressed object cache. key = sha256(nvcc version, flags, the preprocessed
# translation unit): identical code gives the identical object (nvcc is deterministic), so a comment edit or a touched
# header is a cache hit and only translation units whose code changed are compiled. Preprocessing without line markers
# (-P) makes the key independent of paths and line numbers. Safe for concurrent callers (atomic renames).
set -e
cache=$1 nvcc=$2 out=$3 src=$4
shift 4
mkdir -p "$cache" "$(dirname "$out")"
key=$({ "$nvcc" --version | tail -n 1; echo "$*"; "$nvcc" -E -Xcompiler -P "$@" "$src"; } | sha256sum | cut -c1-24)
if [ -f "$cache/$key.o" ]; then
  cp "$cache/$key.o" "$out"
  exit 0
fi
tmp="$out.tmp.$$"
"$nvcc" "$@" -c "$src" -o "$tmp"
cp "$tmp" "$cache/$key.o.tmp.$$" && mv -f "$cache/$key.o.tmp.$$" "$cache/$key.o"
mv -f "$tmp" "$out"
