#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "arith.h"

bool test_add() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t = gf16to3_add(i, i);
        if (t != 0) {
            return 1;
        }
    }

    return 0;
}

bool test_mul() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t1 = gf16to3_mul(i, 1);
        if (t1 != i) { return 1; }

        const gf16to3 t2 = gf16to3_mul(1, i);
        if (t2 != i) { return 1; }
    }

    return 0;
}

#ifdef USE_AVX2

uint32_t test_vector_mul() {
    uint16_t tmp[32];
    for (int i = 0; i < 1u << 12; ++i) {
        for (int j = 0; j < 1u << 12; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf16to3v_mul_u256(a256, b256);
            const uint16_t c = gf16to3_mul(i, j);

            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf16to3 avx mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}


uint32_t test_matrix_mul() {
    const uint32_t nrows = 8;
    const uint32_t ncols = 8;
    const uint32_t ncols2 = 8;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols2);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols2);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols2);

    gf16to3_matrix_id(m1, nrows, ncols);
    gf16to3_matrix_id(m2, nrows, ncols);
    gf16to3_matrix_mul(m3, m1, m2, nrows, ncols, ncols);
    gf16to3_matrix_mul_8xX(m4, m1, m2, ncols, ncols2);

    gf16to3_matrix_print(m3, ncols, ncols2);
    gf16to3_matrix_print(m4, ncols, ncols2);

    uint32_t ret = 0;
    for (int i = 0; i < ncols; ++i) {
        for (int j = 0; j < ncols2; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, ncols, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, ncols, i, j);
            if (c != (i == j)) {
                printf("test matrix mul\n");
                ret = 1;
                goto finish;
            }

            if (d != (i == j)) {
                printf("test matrix mul v2\n");
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(m1); free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_gf16_add() {
    const uint32_t nrows = 17;
    const uint32_t ncols = 2;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16 *m2 = gf16_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16to3_matrix_zero(m3, nrows, ncols);
    gf16to3_matrix_zero(m4, nrows, ncols);
    // gf16to3_matrix_id(m1, nrows, ncols);
    // gf16_matrix_id(m2, nrows, ncols);
    gf16to3_matrix_rng(m1, nrows, ncols);
    gf16_matrix_rng(m2, nrows, ncols);
    gf16to3_matrix_add_gf16(m3, m1, m2, nrows, ncols);
    // gf16to3_matrix_add_gf16_16x16(m4, m1, m2, ncols, ncols);
    // gf16to3_matrix_add_gf16_8x8(m4, m1, m2, ncols, ncols);
    gf16to3_matrix_add_gf16_le16xle16(m4, m1, m2, nrows, ncols);

    gf16to3_matrix_print(m3, nrows, ncols);
    gf16to3_matrix_print(m4, nrows, ncols);

    uint32_t ret = 0;
    for (int i = 0; i < nrows; ++i) {
        for (int j = 0; j < ncols; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, ncols, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, ncols, i, j);
            if (c != d) {
                printf("test matrix gf16 add\n");
                ret = 1;
                goto finish;

            }
        }
    }

    finish:
    free(m1); free(m2); free(m3); free(m4);
    return ret;
}
#endif



int main() {
    // if (test_add()) { return 1; }
    // if (test_mul()) { return 1; }
#ifdef USE_AVX2
    // if (test_vector_mul()) { return 1; }
    // if (test_matrix_mul()) { return 1; }
    if (test_matrix_gf16_add()) { return 1; }
#endif
    return 0;
}
