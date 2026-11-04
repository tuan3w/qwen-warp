# swift-mk engine build.   make            -> build/libswiftmk.so        (core + K1 DeltaNet + K2 attention + engine)
#                          make core       -> build/libswiftmk_core.so   (K3 ops only: GEMV, norms, argmax; stage-1 tests)
# Plain -O3 (no --use_fast_math: numerics-sensitive).  One source list line per file; owners append only their own line.
NVCC    ?= nvcc
ARCH    ?= -arch=sm_86
NVFLAGS ?= $(ARCH) -O3 -std=c++17 -lineinfo -Xcompiler -fPIC -Iinclude
BUILD   := build

SRCS_CORE := src/gemv.cu
SRCS_CORE += src/misc.cu
SRCS_K1 := $(wildcard src/deltanet*.cu)
SRCS_K2 := $(wildcard src/attn*.cu)
SRCS_ENG := $(wildcard src/engine*.cu)
SRCS_MTP := $(wildcard src/mtp*.cu)
SRCS_PRE := $(wildcard src/prefill*.cu)
SRCS_HQ := $(wildcard src/head_q*.cu)
SRCS_PLD := $(wildcard src/pld*.cu)
SRCS_PROC := $(wildcard src/proc*.cu)

SRCS := $(SRCS_CORE) $(SRCS_K1) $(SRCS_K2) $(SRCS_ENG) $(SRCS_MTP) $(SRCS_PRE) $(SRCS_HQ) $(SRCS_PLD) $(SRCS_PROC)
HDRS := $(wildcard include/*.h) $(wildcard src/*.cuh) $(wildcard src/*.h)

all: $(BUILD)/libswiftmk.so
core: $(BUILD)/libswiftmk_core.so

$(BUILD)/%.o: src/%.cu $(HDRS) | $(BUILD)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BUILD)/libswiftmk.so: $(patsubst src/%.cu,$(BUILD)/%.o,$(SRCS))
	$(NVCC) $(ARCH) -shared -o $@ $^ -lcudart

$(BUILD)/libswiftmk_core.so: $(patsubst src/%.cu,$(BUILD)/%.o,$(SRCS_CORE))
	$(NVCC) $(ARCH) -shared -o $@ $^ -lcudart

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all core clean
