// ggml dequant oracle: write NBLK random blocks of Q8_0 / Q4_0 / Q4_1 to <dir>/<type>.raw, dequantize them with
// ggml's own dequantize_row_*, write the fp32 results to <dir>/<type>.out. tests/test_sources.py feeds the same raw
// bytes to the warpc.sources.gguf decoders and requires bit-identical output.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "ggml-quants.h"

// the f16->f32 lookup table ggml-quants.c references (normally in ggml.c)
float ggml_table_f32_f16[1 << 16];
static void init_f16_table(void) {
    for (int i = 0; i < (1 << 16); i++) {
        uint32_t bits = (uint32_t)(uint16_t)i;
        uint32_t s = bits & 0x8000u, e = (bits >> 10) & 0x1Fu, m = bits & 0x3FFu;
        uint32_t f;
        if (e == 0) {
            f = s << 16;
            if (m) { f |= (108u << 23) | (m << 13); }  // subnormal: exp = 127-15
        } else if (e == 31) {
            f = (s << 16) | 0x7F800000u | (m << 13);
        } else {
            f = (s << 16) | (((e + 112u) << 23)) | (m << 13);  // 127-15 = 112
        }
        memcpy(&ggml_table_f32_f16[i], &f, 4);
    }
}

static uint64_t seed = 0x9E3779B97F4A7C15ULL;
static uint32_t rnd(void) {
    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
    return (uint32_t)(seed >> 32);
}
static float frnd(float lo, float hi) { return lo + (hi - lo) * ((float)rnd() / 4294967295.0f); }

static void dump(const char *name, const float *y, int n) {
    FILE *f = fopen(name, "wb");
    if (!f) { perror(name); exit(1); }
    fwrite(y, sizeof(float), n, f);
    fclose(f);
}

#define NBLK 64

int main(int argc, char **argv) {
    const char *dir = argv[1];
    char raw[512], outp[512];
    init_f16_table();

    // Q8_0
    {
        block_q8_0 *b = malloc(sizeof(block_q8_0) * NBLK);
        for (int i = 0; i < NBLK; i++) {
            b[i].d = GGML_FP32_TO_FP16(frnd(-0.1f, 0.1f));
            for (int j = 0; j < QK8_0; j++) b[i].qs[j] = (int8_t)rnd() - 128;
        }
        snprintf(raw, sizeof raw, "%s/q8_0.raw", dir);
        snprintf(outp, sizeof outp, "%s/q8_0.out", dir);
        FILE *rf = fopen(raw, "wb"); fwrite(b, sizeof(block_q8_0), NBLK, rf); fclose(rf);
        float *y = malloc(sizeof(float) * NBLK * QK8_0);
        dequantize_row_q8_0(b, y, NBLK * QK8_0);
        dump(outp, y, NBLK * QK8_0);
        free(b); free(y);
    }
    // Q4_0
    {
        block_q4_0 *b = malloc(sizeof(block_q4_0) * NBLK);
        for (int i = 0; i < NBLK; i++) {
            b[i].d = GGML_FP32_TO_FP16(frnd(-0.1f, 0.1f));
            for (int j = 0; j < QK4_0 / 2; j++) b[i].qs[j] = rnd();
        }
        snprintf(raw, sizeof raw, "%s/q4_0.raw", dir);
        snprintf(outp, sizeof outp, "%s/q4_0.out", dir);
        FILE *rf = fopen(raw, "wb"); fwrite(b, sizeof(block_q4_0), NBLK, rf); fclose(rf);
        float *y = malloc(sizeof(float) * NBLK * QK4_0);
        dequantize_row_q4_0(b, y, NBLK * QK4_0);
        dump(outp, y, NBLK * QK4_0);
        free(b); free(y);
    }
    // Q4_1
    {
        block_q4_1 *b = malloc(sizeof(block_q4_1) * NBLK);
        for (int i = 0; i < NBLK; i++) {
            b[i].d = GGML_FP32_TO_FP16(frnd(-0.1f, 0.1f));
            b[i].m = GGML_FP32_TO_FP16(frnd(-0.5f, 0.5f));
            for (int j = 0; j < QK4_1 / 2; j++) b[i].qs[j] = rnd();
        }
        snprintf(raw, sizeof raw, "%s/q4_1.raw", dir);
        snprintf(outp, sizeof outp, "%s/q4_1.out", dir);
        FILE *rf = fopen(raw, "wb"); fwrite(b, sizeof(block_q4_1), NBLK, rf); fclose(rf);
        float *y = malloc(sizeof(float) * NBLK * QK4_1);
        dequantize_row_q4_1(b, y, NBLK * QK4_1);
        dump(outp, y, NBLK * QK4_1);
        free(b); free(y);
    }
    printf("ok %d blocks each\n", NBLK);
    return 0;
}
