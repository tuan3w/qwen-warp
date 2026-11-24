// Fused Q6_K dequant GEMV (warpc_q6_gemv, the lm_head of a Q4_0 file) against a CPU reference: ggml's Q6_K decoder (d * sc exact, one fp32 product with
// the code - 32) rounded to bf16, lane l of the warp accumulating the 8-column chunks k = l + 32 j in order with fmaf in column order, the 32 lane partials
// summed by the xor butterfly: k_gemv's mode 2 over the bf16-widened matrix. The weights are repacked here from ggml blocks into the layout documented
// in dense_q6_gemv.cuh. The comparison is bit for bit over the fp32 logits, with 1 and 2 blocks per lane in flight, on a row count that is not a
// multiple of a CTA's rows and on K with an odd number of blocks. The arguments the kernel must refuse are checked.
// Build against a built kernel library (host code without contraction):
//   nvcc -O2 -Xcompiler -ffp-contract=off -o test_q6_gemv test_q6_gemv.c -L<build dir> -l:libwarpc_glue.so -lm && LD_LIBRARY_PATH=<build dir> ./test_q6_gemv
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int warpc_q6_gemv(const void* w, const void* x, int N, int K, int T, void* out, int unroll, cudaStream_t st);

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

// ggml block (210 bytes): ql[128] low nibbles, qh[64] high pairs, sc[16] int8, fp16 d; element i of the 256 decoded as in ggml's dequantize_row_q6_K
static void decode_block(const uint8_t* b, float* y) {
  const uint8_t *ql = b, *qh = b + 128;
  const int8_t* sc = (const int8_t*)(b + 192);
  const float d = f16_value((uint16_t)(b[208] | (b[209] << 8)));
  for (int h = 0; h < 2; h++)
    for (int l = 0; l < 32; l++) {
      const uint8_t *qlh = ql + 64 * h, *qhh = qh + 32 * h;
      const int8_t* sch = sc + 8 * h;
      const int is = l / 16;
      const int q1 = ((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
      const int q2 = ((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
      const int q3 = ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
      const int q4 = ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
      float* yh = y + 128 * h;
      yh[l] = (d * (float)sch[is + 0]) * (float)q1;
      yh[l + 32] = (d * (float)sch[is + 2]) * (float)q2;
      yh[l + 64] = (d * (float)sch[is + 4]) * (float)q3;
      yh[l + 96] = (d * (float)sch[is + 6]) * (float)q4;
    }
}

static void make_q6(int N, int K, uint8_t* buf) {
  for (size_t i = 0; i < (size_t)N * (K / 256); i++) {
    uint8_t* b = buf + i * 210;
    for (int j = 0; j < 192; j++) b[j] = (uint8_t)frand(0.f, 256.f);
    for (int j = 0; j < 16; j++) b[192 + j] = (uint8_t)(int8_t)frand(-40.f, 40.f);
    const uint16_t s = f16_bits(frand(0.0005f, 0.004f));
    b[208] = (uint8_t)s;
    b[209] = (uint8_t)(s >> 8);
  }
}

// the kernel's logit of one row: bf16-rounded weights, lane l takes the chunks 32 j + l
static float dot_q6(const uint8_t* row, const float* x, int K) {
  float w[16384];
  for (int b = 0; b < K / 256; b++) decode_block(row + (size_t)b * 210, w + 256 * b);
  float lane[32];
  for (int l = 0; l < 32; l++) {
    float acc = 0.f;
    for (int j = 0; j < K / 256; j++)
      for (int e = 0; e < 8; e++) {
        const int col = 8 * (32 * j + l) + e;
        acc = fmaf(rb(w[col]), x[col], acc);
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

// ggml blocks -> the device row: ql plane (lane l of block j: 8 low nibbles as a u32), qh plane (8 pairs as a u16), scales, d; rows padded to 16 bytes
static void repack(const uint8_t* raw, int N, int K, uint8_t* out, size_t rs) {
  const int KB = K / 256;
  for (int n = 0; n < N; n++) {
    uint8_t* dst = out + n * rs;
    for (int b = 0; b < KB; b++) {
      const uint8_t* src = raw + ((size_t)n * KB + b) * 210;
      int code[256];
      for (int h = 0; h < 2; h++)
        for (int l = 0; l < 32; l++) {
          const uint8_t *qlh = src + 64 * h, *qhh = src + 128 + 32 * h;
          code[128 * h + l] = (qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4);
          code[128 * h + l + 32] = (qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4);
          code[128 * h + l + 64] = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
          code[128 * h + l + 96] = (qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4);
        }
      for (int lane = 0; lane < 32; lane++) {
        uint32_t lo = 0, hi = 0;
        for (int m = 0; m < 8; m++) {
          lo |= (uint32_t)(code[8 * lane + m] & 15) << (4 * m);
          hi |= (uint32_t)(code[8 * lane + m] >> 4) << (2 * m);
        }
        memcpy(dst + 128 * b + 4 * lane, &lo, 4);
        memcpy(dst + 128 * KB + 64 * b + 2 * lane, &hi, 2);
      }
      memcpy(dst + 192 * KB + 16 * b, src + 192, 16);
      memcpy(dst + 208 * KB + 2 * b, src + 208, 2);
    }
  }
}

static int fail = 0;

static void run_case(int N, int K) {
  const int KB = K / 256;
  const size_t rs = ((size_t)KB * 210 + 15) & ~(size_t)15;
  uint8_t* raw = (uint8_t*)malloc((size_t)N * KB * 210);
  uint8_t* packed = (uint8_t*)calloc((size_t)N, rs);
  make_q6(N, K, raw);
  repack(raw, N, K, packed, rs);
  float* x = (float*)malloc((size_t)K * sizeof(float));
  uint16_t* xb = (uint16_t*)malloc((size_t)K * 2);
  for (int i = 0; i < K; i++) {
    xb[i] = bf16_bits(frand(-1.f, 1.f));
    x[i] = bf16_value(xb[i]);
  }
  float* want = (float*)malloc((size_t)N * sizeof(float));
  float* got = (float*)malloc((size_t)N * sizeof(float));
  for (int n = 0; n < N; n++) want[n] = dot_q6(raw + (size_t)n * KB * 210, x, K);
  void *dw, *dx, *dout;
  cudaMalloc(&dw, (size_t)N * rs);
  cudaMalloc(&dx, (size_t)K * 2);
  cudaMalloc(&dout, (size_t)N * 4);
  cudaMemcpy(dw, packed, (size_t)N * rs, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xb, (size_t)K * 2, cudaMemcpyHostToDevice);
  for (int unroll = 1; unroll <= 2; unroll++) {
    cudaMemset(dout, 0, (size_t)N * 4);
    const int rc = warpc_q6_gemv(dw, dx, N, K, 1, dout, unroll, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(got, dout, (size_t)N * 4, cudaMemcpyDeviceToHost);
    size_t bad = 0;
    for (int n = 0; n < N; n++) bad += memcmp(&got[n], &want[n], 4) != 0;
    if (rc != 0 || bad) {
      printf("N=%d K=%d unroll=%d: rc=%d mismatches=%zu/%d\n", N, K, unroll, rc, bad, N);
      fail = 1;
    }
  }
  if (warpc_q6_gemv(dw, dx, N, K - 128, 1, dout, 1, 0) == 0 || warpc_q6_gemv(dw, dx, N, K, 2, dout, 1, 0) == 0 || warpc_q6_gemv(dw, dx, N, K, 1, dout, 3, 0) == 0) {
    printf("an argument the kernel must refuse was accepted\n");
    fail = 1;
  }
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("CUDA error %d\n", (int)e);
    fail = 1;
  }
  printf("N=%d K=%d: unroll 1 / 2 %s\n", N, K, fail ? "FAIL" : "bit-identical");
  cudaFree(dw);
  cudaFree(dx);
  cudaFree(dout);
  free(raw);
  free(packed);
  free(x);
  free(xb);
  free(want);
  free(got);
}

int main(void) {
  run_case(203, 2560);  // 10 blocks: an odd number of pairs of blocks
  run_case(130, 4096);
  run_case(70, 256);
  printf("%s\n", fail ? "FAIL" : "PASS");
  return fail;
}
