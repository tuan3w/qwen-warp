// swift-mk prefill: fused gate|up GEMM + SiLU*mul (CUTLASS DualGemm, examples/45_dual_gemm copied into src/dualgemm/).
//   act[M][N] = bf16( bf16(silu(bf16(X @ Wg^T))) * bf16(X @ Wu^T) )   X [M][K] row-major (ld = ldx), Wg / Wu [N][K] row-major (ld = K), act ld = N.
// Same rounding chain as k_silu_mul on the cuBLAS gate|up output (bf16 GEMM outputs, bf16 silu, bf16 product); fp32 accumulation in K order like the
// cuBLAS 256x128 tile it replaces, but the 2 x I wide gate|up activations never touch DRAM.  The tile (256 x 64 per matrix, 8 warps of 64 x 32 x 2) moves
// the same operand bytes per FLOP as cuBLAS's 256x128x32 3-stage kernel.
#include <cuda_runtime.h>
#include <math.h>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "dualgemm/device/dual_gemm.h"

namespace {

using bf16 = cutlass::bfloat16_t;

// D2 = bf16(bf16(silu(g)) * u) on the bf16 outputs of the two GEMMs (k_silu_mul semantics: silu(x) = x / (1 + expf(-x)) in fp32)
template <int Count>
struct SiluMulBf16 {
  struct Params {};
  CUTLASS_HOST_DEVICE SiluMulBf16(Params const&) {}
  CUTLASS_HOST_DEVICE void set_k_partition(int, int) {}
  CUTLASS_DEVICE cutlass::Array<bf16, Count> operator()(cutlass::Array<bf16, Count> const& g, cutlass::Array<bf16, Count> const& u) const {
    cutlass::Array<bf16, Count> o;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Count; i++) {
      const float x = float(g[i]);
      const float s = float(bf16(x / (1.0f + expf(-x))));
      o[i] = bf16(s * float(u[i]));
    }
    return o;
  }
};

constexpr int kVec = 128 / cutlass::sizeof_bits<bf16>::value;  // 8
using Op01 = cutlass::epilogue::thread::LinearCombination<bf16, kVec, float, float, cutlass::epilogue::thread::ScaleType::Nothing>;
using Op2 = SiluMulBf16<kVec>;

// 256 x 64 per matrix, 8 warps (64 x 32 x 2), 4 stages (96 KB smem): per 2048-row call (N = 17408, K = 5120) 6019 us against 6051 us for 3 stages,
// 6027 us for 128 x 128 x 3 stages and 6415 us for 128 x 128 x 4 stages (one 32k prefill profile, layers rotating over the four configs).
using SwiGlu = cutlass::gemm::device::DualGemm<
    bf16, cutlass::layout::RowMajor,                                   // X
    bf16, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor,  // Wg^T, Wu^T (= row-major [N][K])
    bf16, cutlass::layout::RowMajor,                                   // act
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>, cutlass::gemm::GemmShape<16, 8, 16>,
    Op01, Op01, Op2, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, 4,
    /*StoreD0=*/false, /*StoreD1=*/false, /*SplitKSerial=*/false>;

}  // namespace

extern "C" {

cudaError_t smkp_swiglu_gemm(const void* x, int ldx, const void* wg, const void* wu, void* act, int M, int N, int K, cudaStream_t st) {
  if (M < 1 || N % kVec || K % kVec || ldx % kVec) return cudaErrorInvalidValue;
  using RefA = cutlass::TensorRef<bf16 const, cutlass::layout::RowMajor>;
  using RefB = cutlass::TensorRef<bf16 const, cutlass::layout::ColumnMajor>;
  using RefC = cutlass::TensorRef<bf16 const, cutlass::layout::RowMajor>;
  using RefD = cutlass::TensorRef<bf16, cutlass::layout::RowMajor>;
  SwiGlu::Arguments args(cutlass::gemm::DualGemmMode::kGemm, cutlass::gemm::GemmCoord(M, N, K),
                         RefA((bf16 const*)x, cutlass::layout::RowMajor(ldx)), RefB((bf16 const*)wg, cutlass::layout::ColumnMajor(K)),
                         RefC(nullptr, cutlass::layout::RowMajor(N)), RefD(nullptr, cutlass::layout::RowMajor(N)),
                         RefB((bf16 const*)wu, cutlass::layout::ColumnMajor(K)), RefC(nullptr, cutlass::layout::RowMajor(N)),
                         RefD(nullptr, cutlass::layout::RowMajor(N)), RefD((bf16*)act, cutlass::layout::RowMajor(N)));
  if (SwiGlu::can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  SwiGlu op;
  if (op.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
  if (op.run(st) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  return cudaSuccess;
}

}  // extern "C"
