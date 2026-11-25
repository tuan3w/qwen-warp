// Entry points of the Inkling-2-Mini MoE kernel (inkling_moe_kernel.cuh): the whole layer in ONE persistent launch,
// router + gate|up + down, optionally with the MLP short convolution + residual as its epilogue. The kernel's blocks wait
// for each other, so nothing may run beside it (see the kernel header), and its workspace must be zero before the first
// launch (every launch leaves it zero).
#include "inkling_moe.cuh"

namespace moe {

static int grid_of(int T, int conv) {
#define MOE_CASE(t, c) \
  if (T == t && conv == c) return grid_t##t##_c##c();
  MOE_INSTANCES(MOE_CASE)
#undef MOE_CASE
  return 0;
}

static int launch(int T, int conv, const Args& a, cudaStream_t st) {
  if (T < 1 || T > MAXT || a.E > MAXE || a.E < TOPK) return (int)cudaErrorInvalidValue;
#define MOE_CASE(t, c) \
  if (T == t && conv == c) return launch_t##t##_c##c(a, st);
  MOE_INSTANCES(MOE_CASE)
#undef MOE_CASE
  return (int)cudaErrorInvalidValue;
}

}  // namespace moe

extern "C" {

// workspace size (bytes) for E experts (any T <= 8); the caller zeroes it once
size_t warpc_moe_ws_bytes(int E) { return moe::ws_layout(E).total; }

// byte offsets inside the workspace: counters, logits f32 [8][E], x f32 [8][H], a f32 [64][I], y bf16 [64][H] (tests read them)
int warpc_moe_ws_offsets(int E, size_t* out) {
  const moe::WsLayout L = moe::ws_layout(E);
  out[0] = L.ctr;
  out[1] = L.logits;
  out[2] = L.xf;
  out[3] = L.abuf;
  out[4] = L.ybuf;
  return 0;
}

// blocks of a launch with T rows (all must be resident at once)
int warpc_moe_grid(int T, int conv) { return moe::grid_of(T, conv); }

// The whole layer for T rows. x bf16 [T][2304]; router weight bf16 [E][2304], bias f32 [E]; packed int32 [E][1152][2304 / 8]
// (gate, up) and [E][2304][1152 / 8] (down) with their bf16 scales [E][N][K / 128]. Without the conv epilogue (h == NULL)
// the output bf16 [T][2304] is `out`. With it: res f32 [T][2304], cw bf16 [2304][4] (tap 3 = the current row), cst bf16
// [2304][3] (the conv state, oldest first, updated in place), h bf16 [T][2304] = bf16(res + f32(bf16((conv(mo) + mo) / div))),
// where mo is the bf16 expert output (not written). idx i32 [T][8] and w f32 [T][8] (scaled by route_scale) are written
// when idx is not NULL. ws: zero on entry, zero on exit.
int warpc_moe_forward(const void* x, const void* gate_w, const float* bias, const uint32_t* pg, const void* sg, const uint32_t* pu, const void* su, const uint32_t* pd,
                      const void* sd, void* out, int* idx, float* w, void* ws, const float* res, const void* cw, void* cst, void* h, int T, int E, float route_scale, float div,
                      cudaStream_t st) {
  using namespace moe;
  const Args a{reinterpret_cast<const bf16*>(x),  reinterpret_cast<const bf16*>(gate_w), bias, pg, pu, pd, reinterpret_cast<const bf16*>(sg), reinterpret_cast<const bf16*>(su),
               reinterpret_cast<const bf16*>(sd), reinterpret_cast<bf16*>(out),          idx,  w,  ws, E,  route_scale,
               Conv{res, reinterpret_cast<const bf16*>(cw), reinterpret_cast<bf16*>(cst), reinterpret_cast<bf16*>(h), div}};
  return launch(T, h != nullptr ? 1 : 0, a, st);
}

}  // extern "C"
