// Fused Q4_0 dequant GEMV (warpc_q4_gemv) against a CPU reference that reproduces the kernel's arithmetic: Q4_0 blocks dequantised in fp32, lane l of the
// warp accumulating blocks l, l + 32, ... with fmaf in column order, the 32 lane partials summed by the xor butterfly, the bf16 roundings of each epilogue.
// The comparison is bit for bit (maxabs 0) over modes 0 (three segments), 1 (gate/up) and 3 (residual add), T = 1..8 and rows per warp 1, 2, 4, on a row
// count that is not a multiple of a block's rows and on K with a partial last lane round. The arguments the kernel must refuse (K % 32, rows per warp) are checked.
// Build against a built kernel library (host code without contraction, so the reference's products and sums round like the kernel's):
//   nvcc -O2 -Xcompiler -ffp-contract=off -o test_q4_gemv test_q4_gemv.c -L<build dir> -l:libwarpc_glue.so -lm && LD_LIBRARY_PATH=<build dir> ./test_q4_gemv
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int warpc_q4_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, const int* end, void* const* out, int rpw, cudaStream_t st);

static uint16_t f16_bits(float f) {  // round to nearest even; normal values only
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0) return (uint16_t)sign;
  return (uint16_t)(sign | ((uint32_t)exp << 10) | ((man + 0x1000u + ((man >> 13) & 1u)) >> 13));
}
static float f16_value(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
  uint32_t x = exp ? (sign | (exp + 127 - 15) << 23 | man << 13) : sign;
  float f;
  memcpy(&f, &x, 4);
  return f;
}
static uint16_t bf16_bits(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  return (uint16_t)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}
static float bf16_value(uint16_t b) {
  uint32_t x = (uint32_t)b << 16;
  float f;
  memcpy(&f, &x, 4);
  return f;
}
static float rb(float f) { return bf16_value(bf16_bits(f)); }

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static float frand(float lo, float hi) {  // deterministic xorshift, in [lo, hi)
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return lo + (hi - lo) * (float)((double)(rng_state >> 11) * (1.0 / 9007199254740992.0));
}

static void make_q4(int N, int K, uint8_t* buf) {
  for (size_t i = 0; i < (size_t)N * (K / 32); i++) {
    uint8_t* blk = buf + i * 18;
    uint16_t s = f16_bits(frand(0.05f, 0.5f));
    blk[0] = (uint8_t)s;
    blk[1] = (uint8_t)(s >> 8);
    for (int j = 0; j < 16; j++) blk[2 + j] = (uint8_t)frand(0.f, 256.f);
  }
}

static void dequant_block(const uint8_t* b, float* w) {
  const float d = f16_value((uint16_t)(b[0] | (b[1] << 8)));
  for (int j = 0; j < 16; j++) {
    w[j] = d * ((float)(b[2 + j] & 0x0fu) - 8.f);
    w[j + 16] = d * ((float)(b[2 + j] >> 4) - 8.f);
  }
}

// the kernel's dot product of Q4_0 row `row` with x (K columns)
static float dot_q4(const uint8_t* row, const float* x, int K) {
  float lane[32];
  for (int l = 0; l < 32; l++) {
    float acc = 0.f;
    for (int b = l; b < K / 32; b += 32) {
      float qw[32];
      dequant_block(row + (size_t)b * 18, qw);
      for (int e = 0; e < 32; e++) acc = fmaf(qw[e], x[32 * b + e], acc);
    }
    lane[l] = acc;
  }
  for (int o = 16; o; o >>= 1) {
    float nxt[32];
    for (int l = 0; l < 32; l++) nxt[l] = lane[l] + lane[l ^ o];
    memcpy(lane, nxt, sizeof(lane));
  }
  return lane[0];
}

static int fail = 0;

static void check(const char* what, int rc, const uint16_t* got, const uint16_t* want, size_t n) {
  float maxabs = 0.f;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++)
    if (got[i] != want[i]) {
      bad++;
      maxabs = fmaxf(maxabs, fabsf(bf16_value(got[i]) - bf16_value(want[i])));
    }
  if (rc != 0 || bad) {
    printf("%s: rc=%d mismatches=%zu/%zu maxabs=%g\n", what, rc, bad, n, maxabs);
    fail = 1;
  }
}

static void run_case(int N, int K) {
  const int TM = 8, KB = K / 32;
  const size_t rbytes = (size_t)KB * 18, wbytes = (size_t)N * rbytes;
  uint8_t* wg = (uint8_t*)malloc(wbytes);
  uint8_t* wu = (uint8_t*)malloc(wbytes);
  make_q4(N, K, wg);
  make_q4(N, K, wu);
  float* x = (float*)malloc((size_t)TM * K * sizeof(float));
  uint16_t* xb = (uint16_t*)malloc((size_t)TM * K * sizeof(uint16_t));
  for (int i = 0; i < TM * K; i++) {
    xb[i] = bf16_bits(frand(-1.f, 1.f));
    x[i] = bf16_value(xb[i]);
  }
  uint16_t* h0 = (uint16_t*)malloc((size_t)TM * N * sizeof(uint16_t));
  for (int i = 0; i < TM * N; i++) h0[i] = bf16_bits(frand(-2.f, 2.f));
  float* g = (float*)malloc((size_t)TM * N * sizeof(float));
  float* u = (float*)malloc((size_t)TM * N * sizeof(float));
  for (int t = 0; t < TM; t++)
    for (int n = 0; n < N; n++) {
      g[t * N + n] = dot_q4(wg + n * rbytes, x + (size_t)t * K, K);
      u[t * N + n] = dot_q4(wu + n * rbytes, x + (size_t)t * K, K);
    }

  uint8_t *dw, *dw2;
  uint16_t *dx, *dout, *dseg[3];
  cudaMalloc((void**)&dw, wbytes);
  cudaMalloc((void**)&dw2, wbytes);
  cudaMalloc((void**)&dx, (size_t)TM * K * 2);
  cudaMalloc((void**)&dout, (size_t)TM * N * 2);
  for (int s = 0; s < 3; s++) cudaMalloc((void**)&dseg[s], (size_t)TM * N * 2);
  cudaMemcpy(dw, wg, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dw2, wu, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xb, (size_t)TM * K * 2, cudaMemcpyHostToDevice);
  uint16_t* got = (uint16_t*)malloc((size_t)TM * N * 2);
  uint16_t* want = (uint16_t*)malloc((size_t)TM * N * 2);
  const int end[3] = {N / 4, N / 4 + N / 3, N};
  const int rpws[3] = {1, 2, 4};
  char what[128];

  for (int T = 1; T <= TM; T++)
    for (int ri = 0; ri < 3; ri++) {
      const int rpw = rpws[ri];
      // mode 0: three segments, segment s is bf16 [T][width_s]
      void* segs[3] = {dseg[0], dseg[1], dseg[2]};
      int rc = warpc_q4_gemv(dw, NULL, dx, N, K, T, 0, end, segs, rpw, 0);
      cudaDeviceSynchronize();
      for (int s = 0; s < 3; s++) {
        const int start = s ? end[s - 1] : 0, width = end[s] - start;
        cudaMemcpy(got, dseg[s], (size_t)T * width * 2, cudaMemcpyDeviceToHost);
        for (int t = 0; t < T; t++)
          for (int r = 0; r < width; r++) want[t * width + r] = bf16_bits(g[t * N + start + r]);
        snprintf(what, sizeof(what), "N=%d K=%d T=%d rpw=%d mode 0 segment %d", N, K, T, rpw, s);
        check(what, rc, got, want, (size_t)T * width);
      }
      // mode 1: out = bf16(silu(bf16(gate))) * bf16(up)
      void* outs[3] = {dout, dout, dout};
      rc = warpc_q4_gemv(dw, dw2, dx, N, K, T, 1, NULL, outs, rpw, 0);
      cudaMemcpy(got, dout, (size_t)T * N * 2, cudaMemcpyDeviceToHost);
      for (int i = 0; i < T * N; i++) {
        const float a = rb(g[i]);
        const float sl = rb(a / (1.f + expf(-a)));
        want[i] = bf16_bits(sl * rb(u[i]));
      }
      snprintf(what, sizeof(what), "N=%d K=%d T=%d rpw=%d mode 1", N, K, T, rpw);
      check(what, rc, got, want, (size_t)T * N);
      // mode 3: h = bf16(h + bf16(y)) in place
      cudaMemcpy(dout, h0, (size_t)T * N * 2, cudaMemcpyHostToDevice);
      rc = warpc_q4_gemv(dw, NULL, dx, N, K, T, 3, NULL, outs, rpw, 0);
      cudaMemcpy(got, dout, (size_t)T * N * 2, cudaMemcpyDeviceToHost);
      for (int i = 0; i < T * N; i++) want[i] = bf16_bits(bf16_value(h0[i]) + rb(g[i]));
      snprintf(what, sizeof(what), "N=%d K=%d T=%d rpw=%d mode 3", N, K, T, rpw);
      check(what, rc, got, want, (size_t)T * N);
    }

  void* outs[3] = {dout, dout, dout};
  if (warpc_q4_gemv(dw, NULL, dx, N, K - 16, 1, 0, NULL, outs, 1, 0) == 0) {
    printf("K=%d (not a multiple of 32) was accepted\n", K - 16);
    fail = 1;
  }
  if (warpc_q4_gemv(dw, NULL, dx, N, K, 1, 0, NULL, outs, 3, 0) == 0) {
    printf("rows per warp 3 was accepted\n");
    fail = 1;
  }
  if (warpc_q4_gemv(dw, NULL, dx, N, K, 9, 0, NULL, outs, 1, 0) == 0) {
    printf("T=9 was accepted\n");
    fail = 1;
  }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("CUDA error %d\n", (int)e);
    fail = 1;
  }
  printf("N=%d K=%d: modes 0/1/3, T 1..8, rows per warp 1/2/4 %s\n", N, K, fail ? "FAIL" : "bit-identical (maxabs 0)");

  cudaFree(dw);
  cudaFree(dw2);
  cudaFree(dx);
  cudaFree(dout);
  for (int s = 0; s < 3; s++) cudaFree(dseg[s]);
  free(wg);
  free(wu);
  free(x);
  free(xb);
  free(h0);
  free(g);
  free(u);
  free(got);
  free(want);
}

int main(void) {
  run_case(203, 1056);  // 33 blocks per row: lane 0 takes two, the others one
  run_case(256, 4096);
  printf("%s\n", fail ? "FAIL" : "PASS");
  return fail;
}
