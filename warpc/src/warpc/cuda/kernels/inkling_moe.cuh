// Inkling-2-Mini mixture-of-experts decode kernel: shared declarations of the persistent kernel (inkling_moe_kernel.cuh,
// instantiated per (T, epilogue) by inkling_moe_inst.cu) and its entry points (inkling_moe.cu).
//
// Shapes are compile-time constants: hidden 2304, expert width 1152, at most 128 experts, top 8, at most 8 rows. The
// weights are the checkpoint's W4G128 stacked experts: packed int32 [E][N][K / 8] (nibble e of word j = column 8j + e,
// value (nibble - 8) * scale) and bf16 scales [E][N][K / 128]; the router weight is the stored bf16 [E][H].
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

namespace moe {

typedef __nv_bfloat16 bf16;

constexpr int TOPK = 8;
constexpr int MAXT = 8;
constexpr int MAXPAIRS = TOPK * MAXT;
constexpr int MAXE = 128;
constexpr int HID = 2304;    // hidden size
constexpr int INTER = 1152;  // expert width
constexpr int CONV_TAPS = 4;  // the MLP short convolution: W taps, W - 1 state columns

// the conv epilogue (h != nullptr): h[t] = bf16(res[t] + f32(bf16((conv(out) + out) / div))), the conv over the state
// cst bf16 [H][3] (oldest first, updated in place) and the taps cw bf16 [H][4]
struct Conv {
  const float* res;  // [T][H]
  const bf16* cw;
  bf16* cst;
  bf16* h;  // [T][H]
  float div;
};

struct Args {
  const bf16* x;                // [T][H]
  const bf16* gw;               // router weight [E][H]
  const float* bias;            // [E]
  const uint32_t *pg, *pu, *pd;  // packed gate / up [E][I][H / 8], down [E][H][I / 8]
  const bf16 *sg, *su, *sd;      // their scales [E][N][K / 128]
  bf16* out;                    // [T][H] expert output (written without the conv epilogue)
  int* idx;                     // [T][8] routing, optional (nullptr: not written)
  float* w;                     // [T][8] route weights, optional
  void* ws;                     // workspace, zero on entry and on exit
  int E;
  float route_scale;
  Conv cv;
};

// workspace layout (bytes, 256-aligned parts)
struct WsLayout {
  size_t ctr, logits, xf, abuf, ybuf, total;
};
__host__ __device__ constexpr size_t align256(size_t v) { return (v + 255) & ~(size_t)255; }
__host__ __device__ inline WsLayout ws_layout(int E) {
  WsLayout L;
  L.ctr = 0;
  L.logits = align256(L.ctr + 1024 * sizeof(unsigned));
  L.xf = align256(L.logits + (size_t)MAXT * E * sizeof(float));
  L.abuf = align256(L.xf + (size_t)MAXT * HID * sizeof(float));
  L.ybuf = align256(L.abuf + (size_t)MAXPAIRS * INTER * sizeof(float));
  L.total = align256(L.ybuf + (size_t)MAXPAIRS * HID * sizeof(bf16));
  return L;
}

// one launcher and one occupancy query per (rows, conv epilogue), each defined in its own object (inkling_moe_inst.cu)
#define MOE_INSTANCES(X) \
  X(1, 0) X(1, 1) X(2, 0) X(2, 1) X(3, 0) X(3, 1) X(4, 0) X(4, 1) X(5, 0) X(5, 1) X(6, 0) X(6, 1) X(7, 0) X(7, 1) X(8, 0) X(8, 1)
#define MOE_DECLARE(T, CONV)                                  \
  int launch_t##T##_c##CONV(const Args& a, cudaStream_t st); \
  int grid_t##T##_c##CONV();
MOE_INSTANCES(MOE_DECLARE)
#undef MOE_DECLARE

}  // namespace moe
