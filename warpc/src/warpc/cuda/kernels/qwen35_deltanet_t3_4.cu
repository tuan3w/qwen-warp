// Instantiations of the Gated DeltaNet step for rows 3 and 4 (see qwen35_deltanet_impl.cuh).
#include "qwen35_deltanet_impl.cuh"

#define DN_T 3
DN_CONFIGS(DN_INST_ONE)
#undef DN_T

#define DN_T 4
DN_CONFIGS(DN_INST_ONE)
#undef DN_T
