// Runtime parameters of the sparsity campaign: environment variables SMK_SP_*, read with getenv AT CALL TIME (host launch code, i.e. at CUDA-graph capture time) and never
// cached in a static or a global, so the lab (lab/lab.py --sweep SMK_SP_A=1,2:SMK_SP_B=x,y) can change them inside one process: it sets the variable and drops the captured graphs
// (smk_engine_set_head), and the next step re-captures with the new value.  A parameter that must be fixed at engine creation is read in smk_engine_create_ex and can then only be swept
// with `--recreate`.  Device code must receive the value as a kernel argument or through a device scalar, not call getenv.
#pragma once
#include <cstdlib>

static inline double smk_sp_f(const char* name, double dflt) {
  const char* v = getenv(name);
  return (v && *v) ? atof(v) : dflt;
}
static inline long smk_sp_i(const char* name, long dflt) {
  const char* v = getenv(name);
  return (v && *v) ? atol(v) : dflt;
}
