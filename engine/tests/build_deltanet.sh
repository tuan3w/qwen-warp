#!/usr/bin/env bash
# K1 standalone build (until K3's Makefile exists). Extra nvcc flags may be passed, e.g. -DDN_NS=16. Output name via OUT=.
set -e
cd "$(dirname "$0")/.."
nice -n 10 nvcc -arch=sm_86 -O3 -std=c++17 -shared -Xcompiler -fPIC -Iinclude "$@" src/deltanet.cu -o "${OUT:-build/libswiftmk_deltanet.so}"
