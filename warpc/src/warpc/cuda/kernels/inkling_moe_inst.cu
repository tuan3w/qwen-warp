// k_moe instantiation: one object per (rows MOE_T, epilogue MOE_CONV), selected by -DMOE_T= / -DMOE_CONV= (Makefile).
#include "inkling_moe_kernel.cuh"

namespace moe {

#define MOE_CAT2(a, b, c, d) a##b##c##d
#define MOE_CAT(a, b, c, d) MOE_CAT2(a, b, c, d)

#ifndef GRID_BLOCKS
#define GRID_BLOCKS 288  // 288 blocks = 1152 warps = exactly 3 queue items per warp for T = 1 (8 experts x (288 gate|up + 144 down) = 3456 items): no wave quantisation
#endif

static int num_sms() {
  static int v = 0;
  if (!v) {
    int d = 0;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, d);
  }
  return v;
}

// blocks of a launch: GRID_BLOCKS, but never more than can be resident at once (k_moe's blocks wait for each other)
static int grid_blocks() {
  static int mb = -1;
  if (mb < 0) {
    mb = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, k_moe<MOE_T, MOE_CONV>, 128, 0);
  }
  const int cap = num_sms() * mb;
  return GRID_BLOCKS < cap ? GRID_BLOCKS : cap;
}

int MOE_CAT(grid_t, MOE_T, _c, MOE_CONV)() { return grid_blocks(); }

int MOE_CAT(launch_t, MOE_T, _c, MOE_CONV)(const Args& a, cudaStream_t st) {
  const int grid = grid_blocks();
  if (grid < 1) return cudaErrorInvalidConfiguration;
  k_moe<MOE_T, MOE_CONV><<<grid, 128, 0, st>>>(a.x, a.gw, a.bias, a.pg, a.sg, a.pu, a.su, a.pd, a.sd, a.out, a.idx, a.w, a.ws, a.E, a.route_scale, a.cv);
  return cudaGetLastError();
}

}  // namespace moe
