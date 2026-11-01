#!/usr/bin/env bash
# K2 standalone build (plain -O3, no fast-math). Output: build/libswiftmk_attn.so. Extra nvcc flags may be passed (e.g. -Xptxas -v).
set -e
cd "$(dirname "$0")/.."
mkdir -p build
nice -n 10 nvcc -arch=sm_86 -O3 -std=c++17 -lineinfo -shared -Xcompiler -fPIC -Iinclude "$@" src/attn.cu -o "${OUT:-build/libswiftmk_attn.so}"
