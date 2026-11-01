#!/bin/sh
# Build the private deep-chain library: committed engine sources (pinned commit) + lib/deep.patch -> runs/lib/build/libswiftmk.so
# usage: sh lib/build.sh   (from anywhere; ~45 s).  The committed engine is not touched.
set -e
HERE=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
PIN=a2f79e4   # last commit that touched engine/{src,include,Makefile}
rm -rf "$HERE/runs/lib" "$HERE/runs/_x"; mkdir -p "$HERE/runs/_x"
git -C "$ROOT" archive $PIN engine/src engine/include engine/Makefile | tar -x -C "$HERE/runs/_x"
mv "$HERE/runs/_x/engine" "$HERE/runs/lib" && rm -rf "$HERE/runs/_x"
(cd "$HERE/runs/lib" && patch -s -p1 < "$HERE/lib/deep.patch")
make -C "$HERE/runs/lib" -j8 BUILD=build
