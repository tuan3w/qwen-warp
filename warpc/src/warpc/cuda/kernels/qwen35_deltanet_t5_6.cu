// Instantiations of the Gated DeltaNet step for rows 5 and 6 (see qwen35_deltanet_impl.cuh).
#include "qwen35_deltanet_impl.cuh"

#define DN_T 5
DN_CONFIGS(DN_INST_ONE)
#undef DN_T

#define DN_T 6
DN_CONFIGS(DN_INST_ONE)
#undef DN_T
