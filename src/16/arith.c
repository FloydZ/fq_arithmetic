#include "arith.h"

#include <string.h>

#define TS 512

#ifdef USE_AVX2
#include <immintrin.h>
uint32_t test_vector_mul() {
    uint8_t tmp[16];
    for (uint32_t i = 1; i < (1u<<4); i++) {
        for (uint32_t j = 1; j < (1u<<4); j++) {
            const ff_t c = gf16_mul(i, j);
            __m256i a256 = _mm256_set1_epi8(i);
            __m256i b256 = _mm256_set1_epi8(j);
            __m256i c256 = gf16v_mul_u256(a256, b256);
            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (uint32_t k = 0; k < 16; k++) {
                if (c != tmp[k]) {
                    printf("test_vector_mul v1: error\n");
                    return 1;
                }
            }

            __m256i a256_v2 = _mm256_set1_epi8(i ^ (i<<4));
            __m256i b256_v2 = _mm256_set1_epi8(j ^ (j<<4));
            __m256i c256_v2 = gf16v_mul_full_u256(a256_v2, b256_v2);
            _mm256_storeu_si256((__m256i *)tmp, c256_v2);
            for (uint32_t k = 0; k < 16; k++) {
                if (c != (tmp[k]&0xF)) {
                    printf("test_vector_mul v21: error\n");
                    return 1;
                }
                if (c != (tmp[k]>>4)) {
                    printf("test_vector_mul v22: error\n");
                    return 1;
                }
            }
        }
    }
    return 0;
}

#endif

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
            gf16_matrix_set(data1, s, i, j, (i+j)%16);
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
             if (gf16_matrix_get(A, s, i, j) != (i == j)) {
                 printf("kek\n");
                 return 1;
             }
        }
    }

    free(A); free(b); free(AT);
    return 0;
}

int main() {
    // if (test_transpose()) { return 1; }
    // if (test_transpose2()) { return 1; }
    // if (test_solve()) { return 1; }

#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
#endif

    return 0;
}
