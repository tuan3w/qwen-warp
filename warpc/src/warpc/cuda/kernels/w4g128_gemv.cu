#include "w4g128_gemv.cuh"

// The one translation unit that instantiates k_w4g (3 configurations, selected by the K split KS = 1 / 4 / 8) and the
// extern "C" entry. The numerics are in w4g128_gemv.cuh.
namespace w4g {

// persistent grid: every warp does the same amount of work, so the largest divisor of `groups` that fits in the resident
// capacity (unless that divisor is below 60% of the capacity). The capacity counts at most MINB CTAs per SM, the launch
// bound the configurations were tuned for: with 3 resident CTAs the 16384-row shape gets an uneven 246-CTA grid and runs
// 7% slower than with the 128 CTAs of two per SM (measured).
static int resident_ctas(int occupancy_per_sm, int minb) {
  static const int sms = [] {
    int d = 0, n = 0;
    cudaGetDevice(&d);
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, d);
    return n;
  }();
  const int occ = occupancy_per_sm < minb ? occupancy_per_sm : minb;
  return (occ < 1 ? 1 : occ) * sms;
}

static int grid_for(int resident, int groups) {
  const int cap = groups < resident ? groups : resident;
  int d = cap;
  while (d > 1 && groups % d) --d;
  return d * 10 >= cap * 6 ? d : cap;
}

template <int NW, int KS, int U, int MINB>
static cudaError_t run(const void* w, const void* s, const void* x, int N, int K, int T, const Epi& epi, cudaStream_t st) {
  constexpr int TPC = NW / KS;
  if (((K >> 7) % KS) || ((N >> 4) % TPC)) return cudaErrorInvalidValue;
  static const int resident = [] {
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_w4g<NW, KS, U, MINB>, NW * 32, 0);
    return resident_ctas(occ, MINB);
  }();
  const int grid = grid_for(resident, (N >> 4) / TPC);
  k_w4g<NW, KS, U, MINB><<<grid, NW * 32, 0, st>>>((const uint4*)w, (const uint32_t*)s, (const uint4*)x, epi, N, K, T);
  return cudaGetLastError();
}

}  // namespace w4g

extern "C" {

// W: the repacked weights, S: their scales (the same buffer + N * K / 2 bytes), x bf16 [T][K]. kind 0: up to 3 output
// segments by row range (end[0..2] cumulative row ends, out[0..2] bf16 [T][width]); 1: interleaved gate | up pair, out[0]
// bf16 [T][N / 2]; 2: fp32 [T][N] unrounded sums (the gates). ks: K-split warps per tile (1, 4 or 8; part of the numerics). Invalid argument combinations return
// cudaErrorInvalidValue.
int warpc_w4g128_gemv(const void* w, const void* s, const void* x, int N, int K, int T, int kind, const int* end,
                      void* const* out, int ks, cudaStream_t st) {
  if ((K & 127) || (N & 15) || T < 1 || T > STEP_ROWS || (kind != w4g::EPI_SEGS && kind != w4g::EPI_SILU && kind != w4g::EPI_F32))
    return cudaErrorInvalidValue;
  w4g::Epi e;
  e.kind = kind;
  for (int i = 0; i < 3; i++) {
    e.end[i] = end ? end[i] : N;
    e.out[i] = out[i];
  }
  switch (ks) {
    case 4: return w4g::run<8, 4, 2, 2>(w, s, x, N, K, T, e, st);
    case 8: return w4g::run<8, 8, 2, 2>(w, s, x, N, K, T, e, st);
    case 1: return w4g::run<8, 1, 3, 2>(w, s, x, N, K, T, e, st);
  }
  return cudaErrorInvalidValue;
}

}  // extern "C"
