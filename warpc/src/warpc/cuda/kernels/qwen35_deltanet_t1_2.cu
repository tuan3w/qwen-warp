// Instantiations of the Gated DeltaNet step for rows 1 and 2 (see qwen35_deltanet_impl.cuh).
#include "qwen35_deltanet_impl.cuh"

#define DN_T 1
DN_CONFIGS(DN_INST_ONE)
#undef DN_T

#define DN_T 2
DN_CONFIGS(DN_INST_ONE)
#undef DN_T
