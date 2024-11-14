#include "arith.h"

#include <string.h>

#define TS 512

uint32_t test_transpose() {
    //uint8_t d[TS]={0}, out[TS]={0};
    //for (uint32_t i = 0; i < TS/4; i++ ) {
    //    d[i] = i%16;
    //}

    //// gf16_matrix_print(d, 16, 32);
    //gf16_transpose_16x16(out +  0, d +   0, 16);
    //gf16_transpose_16x16(out +  8, d + 128, 16);
    //// gf16_matrix_print(out, 32, 16);


#ifdef USE_AVX2
    uint8_t *data1 = calloc(1, 2048);
    uint8_t *data2 = calloc(1, 2048);

    for (uint32_t i = 0; i < 32; i++ ) {
        data1[i] = i%16;
    }

    // gf16_matrix_print(data1, 64, 64);
    gf16_transpose_64x64_avx2(data2, data1, 32);
    // gf16_matrix_print(data2, 64, 64);
    free(data1); free(data2);
#endif
    return 0;
}

uint32_t test_transpose2() {
    const size_t s = 64;
    uint8_t *data1 = calloc(1, s*s/2);
    uint8_t *data2 = calloc(1, s*s/2);
    uint8_t *data3 = calloc(1, s*s/2);

    for (size_t i = 0; i < s; i++) {
        for (size_t j = 0; j < s; j++) {
            gf16_matrix_set_entry(data1, s, i, j, (i+j)%16);
        }
    }

    gf16_matrix_tranpose(data2, data1, s, s);
    gf16_matrix_tranpose_opt(data3, data1, s, s);

    // gf16_matrix_print(data1, s, s);
    // gf16_matrix_print(data2, s, s);
    // gf16_matrix_print(data3, s, s);

    if (memcmp(data2, data3, s*s/2) != 0) {
        printf("kek\n");
        return 1;
    }

    free(data1); free(data2); free(data3);
    return 0;
}

uint32_t test_solve() {
    const size_t s = 78;
    uint8_t *A = calloc(1, s*s/2);
    uint8_t *b = calloc(1, s/2);
    uint8_t *AT = calloc(1, s*s/2);

    gf16_matrix_rng_full_rank(A, s, s);

    gf16_matrix_print(A, s, s);
    // gf16_solve(A, b, s, s);
    const uint32_t srows = gf16_solve_transpose(A, AT, b, s, s);
    gf16_matrix_print(A, s, s);

    if (srows != s) {
        printf("wrong rank\n");
        return 1;
    }

    for (uint32_t i = 0; i < s; i++) {
        for (uint32_t j = 0; j < s; j++) {
             if (gf16_matrix_get_entry(A, s, i, j) != (i == j)) {
                 printf("kek\n");
                 return 1;
             }
        }
    }

    free(A); free(b); free(AT);
    return 0;
}

int main() {
    if (test_transpose()) { return 1; }
    if (test_transpose2()) { return 1; }
    if (test_solve()) { return 1; }

    return 0;
}
