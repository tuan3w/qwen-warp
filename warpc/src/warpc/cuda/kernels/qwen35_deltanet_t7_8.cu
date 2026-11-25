// Instantiations of the Gated DeltaNet step for rows 7 and 8 (see qwen35_deltanet_impl.cuh).
#include "qwen35_deltanet_impl.cuh"

#define DN_T 7
DN_CONFIGS(DN_INST_ONE)
#undef DN_T

#define DN_T 8
DN_CONFIGS(DN_INST_ONE)
#undef DN_T
