# The dense decode kernels (dense_common.cuh lists the files) built into $(BUILD)/libwarpc_glue.so as real SASS for
# one GPU architecture (a PTX ISA newer than the driver's is refused at load time, so no PTX fallback is shipped).
#
#   make NVCC=<nvcc> SM=86 BUILD=<dir> [CACHE=<dir>] [-jN]
#
# The entry points live in dispatching units that compile no template kernel; the templates are instantiated in the
# *_inst.cu units (one object per T set and mode, or per head dim and part). Every object is an independent translation
# unit (parallel make, -j nproc unless -j is given) compiled through a content-addressed object cache (nvcc_cached.sh),
# so after an edit only the objects whose preprocessed code changed are compiled.
NVCC ?= nvcc
SM ?= 86
NVFLAGS = -O3 -std=c++17 -gencode arch=compute_$(SM),code=sm_$(SM) -Xcompiler -fPIC
BUILD ?= build
CACHE ?= $(BUILD)/objcache
OBJ := $(BUILD)/obj
OUT := $(BUILD)/libwarpc_glue.so
ifeq ($(filter -j%,$(MAKEFLAGS)),)
MAKEFLAGS += -j$(shell nproc)
endif

HDR := $(wildcard *.cuh)
SRC := $(filter-out %_inst.cu,$(wildcard *.cu))
OBJS := $(patsubst %.cu,$(OBJ)/%.o,$(SRC))

# k_gemv and k_q4_gemv: one object per (T set, mode), rows per warp 1, 2, 4 in each
GEMV_SETS := T1 T2 T3_5 T6_8
GEMV_MODES := 0 1 2 3
GEMV_OBJS := $(foreach m,$(GEMV_MODES),$(foreach s,$(GEMV_SETS),$(OBJ)/dense_gemv_inst_$(s)_m$(m).o))
Q4GEMV_MODES := 0 1 3
Q4GEMV_OBJS := $(foreach m,$(Q4GEMV_MODES),$(foreach s,$(GEMV_SETS),$(OBJ)/dense_q4_gemv_inst_$(s)_m$(m).o))

# attention: one object per (head dim, part); parts 1 bf16 split + reduce, 2 chunked bf16, 4 quantised prep + merge,
# 8 / 16 quantised split with 8-bit / 4-bit V
ATTN_HDS := 64 128 256
ATTN_PARTS := 1 2 4 8 16
ATTN_OBJS := $(foreach h,$(ATTN_HDS),$(foreach p,$(ATTN_PARTS),$(OBJ)/dense_attn_inst_hd$(h)_p$(p).o))

CC_CACHED = sh ./nvcc_cached.sh "$(CACHE)" "$(NVCC)" $@ $< $(NVFLAGS) $(DEFS)

all: $(OUT)

$(OBJ)/%.o: %.cu $(HDR)
	@$(CC_CACHED)

$(GEMV_OBJS): $(OBJ)/%.o: dense_gemv_inst.cu $(HDR)
	@$(CC_CACHED)
$(foreach m,$(GEMV_MODES),$(foreach s,$(GEMV_SETS),\
  $(eval $(OBJ)/dense_gemv_inst_$(s)_m$(m).o: DEFS = -DGEMV_SET=$(s) -DGEMV_MODE=$(m))))

$(Q4GEMV_OBJS): $(OBJ)/%.o: dense_q4_gemv_inst.cu $(HDR)
	@$(CC_CACHED)
$(foreach m,$(Q4GEMV_MODES),$(foreach s,$(GEMV_SETS),\
  $(eval $(OBJ)/dense_q4_gemv_inst_$(s)_m$(m).o: DEFS = -DQ4GEMV_SET=$(s) -DQ4GEMV_MODE=$(m))))

$(ATTN_OBJS): $(OBJ)/%.o: dense_attn_inst.cu $(HDR)
	@$(CC_CACHED)
$(foreach h,$(ATTN_HDS),$(foreach p,$(ATTN_PARTS),\
  $(eval $(OBJ)/dense_attn_inst_hd$(h)_p$(p).o: DEFS = -DATTN_HD=$(h) -DATTN_PARTS=$(p))))

# link into a temporary name and rename: a loader never sees a half-written library
$(OUT): $(OBJS) $(GEMV_OBJS) $(Q4GEMV_OBJS) $(ATTN_OBJS)
	@mkdir -p $(BUILD)
	@$(NVCC) -gencode arch=compute_$(SM),code=sm_$(SM) -shared -o $@.tmp.$$$$ $^ && mv -f $@.tmp.$$$$ $@

clean:
	rm -rf $(OBJ) $(OUT)

.PHONY: all clean
