// Fused Q4_0 / Q4_1 dequant GEMV (warpc_q4_gemv) against a CPU reference that reproduces the kernel's arithmetic: the blocks dequantised in fp32
// (d * (q - 8); Q4_1: d * q + m rounded to bf16), lane l of the warp accumulating the 8-column chunks k = l + 32 j in order with fmaf in column order, the 32 lane
// partials summed by the xor butterfly, the bf16 roundings of each epilogue. The weights are repacked here from ggml blocks into the layout
// documented in dense_q4_gemv.cuh. The comparison is bit for bit (maxabs 0) over formats 0 / 1, modes 0 (three segments), 1 (gate/up) and 3
// (residual add), T = 1..8 and rows per warp 1, 2, on a row count that is not a multiple of a CTA's rows and on K with an odd number of pairs
// of rounds. The arguments the kernel must refuse (K % 512, rows per warp, T, format) are checked.
// Build against a built kernel library (host code without contraction, so the reference's products and sums round like the kernel's):
//   nvcc -O2 -Xcompiler -ffp-contract=off -o test_q4_gemv test_q4_gemv.c -L<build dir> -l:libwarpc_glue.so -lm && LD_LIBRARY_PATH=<build dir> ./test_q4_gemv
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int warpc_q4_gemv(const void* w, const void* w2, const void* x, int N, int K, int T, int mode, int fmt, const int* end, void* const* out, int rpw, cudaStream_t st);

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

// ggml blocks: fp16 d (Q4_1: then fp16 m) and 16 bytes of nibbles; column j of a block in the low nibble of byte j, j + 16 in the high nibble
static size_t block_bytes(int fmt) { return fmt ? 20 : 18; }

static void make_q4(int N, int K, int fmt, uint8_t* buf) {
  const size_t bb = block_bytes(fmt);
  for (size_t i = 0; i < (size_t)N * (K / 32); i++) {
    uint8_t* blk = buf + i * bb;
    uint16_t s = f16_bits(frand(0.05f, 0.5f));
    blk[0] = (uint8_t)s;
    blk[1] = (uint8_t)(s >> 8);
    if (fmt) {
      s = f16_bits(frand(-1.f, 1.f));
      blk[2] = (uint8_t)s;
      blk[3] = (uint8_t)(s >> 8);
    }
    for (int j = 0; j < 16; j++) blk[bb - 16 + j] = (uint8_t)frand(0.f, 256.f);
  }
}

static float weight(const uint8_t* row, int fmt, int col) {
  const uint8_t* b = row + (size_t)(col / 32) * block_bytes(fmt);
  const int c = col % 32;
  const float d = f16_value((uint16_t)(b[0] | (b[1] << 8)));
  const uint8_t byte = b[block_bytes(fmt) - 16 + (c % 16)];
  const float q = (float)(c < 16 ? byte & 0x0fu : byte >> 4);
  if (fmt) return rb(d * q + f16_value((uint16_t)(b[2] | (b[3] << 8))));
  return d * (q - 8.f);
}

// the kernel's dot product of the row with x (K columns): lane l takes the chunks 32 j + l, 8 columns each
static float dot_q4(const uint8_t* row, int fmt, const float* x, int K) {
  float lane[32];
  for (int l = 0; l < 32; l++) {
    float acc = 0.f;
    for (int j = 0; j < K / 256; j++)
      for (int e = 0; e < 8; e++) {
        const int col = 8 * (32 * j + l) + e;
        acc = fmaf(weight(row, fmt, col), x[col], acc);
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

// ggml blocks -> the device layout (row of K / 32 * block_bytes bytes): per pair of rounds, 32 lanes x 2 words of 8 nibbles, then the scales (and
// the minimums) of the 8 blocks each round's lanes read in (group, round) order
static void repack(const uint8_t* raw, int N, int K, int fmt, uint8_t* out) {
  const size_t bb = block_bytes(fmt), rs = (size_t)(K / 32) * bb;
  const int KB = K / 32, P = K / 512;
  for (int n = 0; n < N; n++) {
    const uint8_t* src = raw + n * rs;
    uint8_t* dst = out + n * rs;
    for (int p = 0; p < P; p++)
      for (int l = 0; l < 32; l++)
        for (int r = 0; r < 2; r++) {
          const int k = 32 * (2 * p + r) + l;
          uint32_t word = 0;
          for (int m = 0; m < 8; m++) {
            const int col = 8 * k + m, c = col % 32;
            const uint8_t byte = src[(size_t)(col / 32) * bb + bb - 16 + (c % 16)];
            word |= (uint32_t)(c < 16 ? byte & 0x0fu : byte >> 4) << (4 * m);
          }
          memcpy(dst + 256 * p + 8 * l + 4 * r, &word, 4);
        }
    for (int h = 0; h < (fmt ? 2 : 1); h++)
      for (int p = 0; p < P; p++)
        for (int g = 0; g < 8; g++)
          for (int r = 0; r < 2; r++)
            memcpy(dst + (size_t)KB * 16 + (size_t)h * KB * 2 + 32 * p + 4 * g + 2 * r, src + (size_t)(16 * p + 8 * r + g) * bb + 2 * h, 2);
  }
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

static void run_case(int N, int K, int fmt) {
  const int TM = 8;
  const size_t wbytes = (size_t)N * (K / 32) * block_bytes(fmt);
  uint8_t* wg = (uint8_t*)malloc(wbytes);
  uint8_t* wu = (uint8_t*)malloc(wbytes);
  uint8_t* pg = (uint8_t*)malloc(wbytes);
  uint8_t* pu = (uint8_t*)malloc(wbytes);
  make_q4(N, K, fmt, wg);
  make_q4(N, K, fmt, wu);
  repack(wg, N, K, fmt, pg);
  repack(wu, N, K, fmt, pu);
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
  const size_t rbytes = (size_t)(K / 32) * block_bytes(fmt);
  for (int t = 0; t < TM; t++)
    for (int n = 0; n < N; n++) {
      g[t * N + n] = dot_q4(wg + n * rbytes, fmt, x + (size_t)t * K, K);
      u[t * N + n] = dot_q4(wu + n * rbytes, fmt, x + (size_t)t * K, K);
    }

  uint8_t *dw, *dw2;
  uint16_t *dx, *dout, *dseg[3];
  cudaMalloc((void**)&dw, wbytes);
  cudaMalloc((void**)&dw2, wbytes);
  cudaMalloc((void**)&dx, (size_t)TM * K * 2);
  cudaMalloc((void**)&dout, (size_t)TM * N * 2);
  for (int s = 0; s < 3; s++) cudaMalloc((void**)&dseg[s], (size_t)TM * N * 2);
  cudaMemcpy(dw, pg, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dw2, pu, wbytes, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xb, (size_t)TM * K * 2, cudaMemcpyHostToDevice);
  uint16_t* got = (uint16_t*)malloc((size_t)TM * N * 2);
  uint16_t* want = (uint16_t*)malloc((size_t)TM * N * 2);
  const int end[3] = {N / 4, N / 4 + N / 3, N};
  const int rpws[2] = {1, 2};
  char what[128];

  for (int T = 1; T <= TM; T++)
    for (int ri = 0; ri < 2; ri++) {
      const int rpw = rpws[ri];
      // mode 0: three segments, segment s is bf16 [T][width_s]
      void* segs[3] = {dseg[0], dseg[1], dseg[2]};
      int rc = warpc_q4_gemv(dw, NULL, dx, N, K, T, 0, fmt, end, segs, rpw, 0);
      cudaDeviceSynchronize();
      for (int s = 0; s < 3; s++) {
        const int start = s ? end[s - 1] : 0, width = end[s] - start;
        cudaMemcpy(got, dseg[s], (size_t)T * width * 2, cudaMemcpyDeviceToHost);
        for (int t = 0; t < T; t++)
          for (int r = 0; r < width; r++) want[t * width + r] = bf16_bits(g[t * N + start + r]);
        snprintf(what, sizeof(what), "N=%d K=%d fmt=%d T=%d rpw=%d mode 0 segment %d", N, K, fmt, T, rpw, s);
        check(what, rc, got, want, (size_t)T * width);
      }
      // mode 1: out = bf16(silu(bf16(gate))) * bf16(up)
      void* outs[3] = {dout, dout, dout};
      rc = warpc_q4_gemv(dw, dw2, dx, N, K, T, 1, fmt, NULL, outs, rpw, 0);
      cudaMemcpy(got, dout, (size_t)T * N * 2, cudaMemcpyDeviceToHost);
      for (int i = 0; i < T * N; i++) {
        const float a = rb(g[i]);
        const float sl = rb(a / (1.f + expf(-a)));
        want[i] = bf16_bits(sl * rb(u[i]));
      }
      snprintf(what, sizeof(what), "N=%d K=%d fmt=%d T=%d rpw=%d mode 1", N, K, fmt, T, rpw);
      check(what, rc, got, want, (size_t)T * N);
      // mode 3: h = bf16(h + bf16(y)) in place
      cudaMemcpy(dout, h0, (size_t)T * N * 2, cudaMemcpyHostToDevice);
      rc = warpc_q4_gemv(dw, NULL, dx, N, K, T, 3, fmt, NULL, outs, rpw, 0);
      cudaMemcpy(got, dout, (size_t)T * N * 2, cudaMemcpyDeviceToHost);
      for (int i = 0; i < T * N; i++) want[i] = bf16_bits(bf16_value(h0[i]) + rb(g[i]));
      snprintf(what, sizeof(what), "N=%d K=%d fmt=%d T=%d rpw=%d mode 3", N, K, fmt, T, rpw);
      check(what, rc, got, want, (size_t)T * N);
    }

  void* outs[3] = {dout, dout, dout};
  if (warpc_q4_gemv(dw, NULL, dx, N, K - 256, 1, 0, fmt, NULL, outs, 1, 0) == 0) {
    printf("K=%d (not a multiple of 512) was accepted\n", K - 256);
    fail = 1;
  }
  if (warpc_q4_gemv(dw, NULL, dx, N, K, 1, 0, fmt, NULL, outs, 4, 0) == 0) {
    printf("rows per warp 4 was accepted\n");
    fail = 1;
  }
  if (warpc_q4_gemv(dw, NULL, dx, N, K, 9, 0, fmt, NULL, outs, 1, 0) == 0) {
    printf("T=9 was accepted\n");
    fail = 1;
  }
  if (warpc_q4_gemv(dw, NULL, dx, N, K, 1, 0, 2, NULL, outs, 1, 0) == 0) {
    printf("format 2 was accepted\n");
    fail = 1;
  }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("CUDA error %d\n", (int)e);
    fail = 1;
  }
  printf("N=%d K=%d fmt=%d: modes 0/1/3, T 1..8, rows per warp 1/2 %s\n", N, K, fmt, fail ? "FAIL" : "bit-identical (maxabs 0)");

  cudaFree(dw);
  cudaFree(dw2);
  cudaFree(dx);
  cudaFree(dout);
  for (int s = 0; s < 3; s++) cudaFree(dseg[s]);
  free(wg);
  free(wu);
  free(pg);
  free(pu);
  free(x);
  free(xb);
  free(h0);
  free(g);
  free(u);
  free(got);
  free(want);
}

int main(void) {
  for (int fmt = 0; fmt < 2; fmt++) {
    run_case(203, 1536, fmt);  // 3 pairs of rounds: a second pair is missing for the last of two in flight
    run_case(256, 4096, fmt);
  }
  printf("%s\n", fail ? "FAIL" : "PASS");
  return fail;
}
