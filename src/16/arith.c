// NOTE: currently the row echelon form need this
#define NROWS 127
#define NCOLS 128

#include <string.h>
#include <stdio.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"


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

/// tests the mayo code
uint32_t test_matrix_gaus_compressed() {
    uint8_t ret = 0;
    uint8_t *A = gf16_matrix_alloc(NROWS, NCOLS);
    gf16_matrix_rng_full_rank(A, NROWS, NCOLS);

    row_echelon_form_compressed(A, NROWS, NCOLS);
    //gf16_matrix_print(A, NCOLS, NROWS);
    for (uint32_t i = 0; i < NROWS; i++) {
        for (uint32_t j = 0; j < NCOLS; j++) {
            const gf16 t = gf16_matrix_get(A, NCOLS, j, i);
            if ((i==j) && (t != 1)) {
                gf16_matrix_print(A, NROWS, NCOLS);
                printf("error: test_matrix_gaus_compressed\n");
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    return ret;
}

uint32_t test_matrix_gaus() {
    // NOTE: this tests the expanded version
    uint8_t ret = 0;
    uint8_t *A = calloc(1, NCOLS*NROWS);

    // gf16_matrix_random(A, nrows, ncols);
    for (uint32_t i = 0; i < NROWS * NCOLS; i++) { A[i] = rand() & 0xF;  }
    for (uint32_t i1 = 0; i1 < NROWS; i1++) {
        for (uint32_t j1 = 0; j1 < NROWS; j1++) {
            if ((rand() & 1u) && (i1 != j1)) {
                for (uint32_t k1 = 0; k1 < NCOLS; k1++) {
                    A[i1*NCOLS + k1] ^= A[j1*NCOLS + k1];
                }
            }
        }
    }

    row_echelon_form(A, NROWS, NCOLS);

    // for (uint32_t i1 = 0; i1 < NROWS; i1++) {
    //     for (uint32_t j1 = 0; j1 < NCOLS; j1++) {
    //         printf("%2d", A[i1*NCOLS + j1]);
    //         if (j1 != (NCOLS - 1)) {
    //             printf(",");
    //         }
    //     }
    //     printf("\n");
    // }
    for (uint32_t i = 0; i < NROWS; i++) {
        for (uint32_t j = 0; j < NCOLS; j++) {
            const gf16 t = A[i*NCOLS + i]; //gf16_matrix_get(A, NCOLS, j, i);
            if ((i==j) && (t != 1)) {
                printf("\n");
                printf("error: test_matrix_gaus\n");
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    return ret;
}

uint32_t test_transpose_32x32() {
    const size_t s = 32;
    uint8_t *data1 = gf16_matrix_alloc(s, s);
    uint8_t *data2 = gf16_matrix_alloc(s, s);
    uint8_t *data3 = gf16_matrix_alloc(s, s);

    uint32_t ret = 0;
    for (uint32_t k = 0; k < 100; k++) {
        gf16_matrix_random(data1, s, s);
        gf16_matrix_tranpose(data2, data1, s, s);
        gf16_transpose_32x32_avx2(data3, data1, 16);

        for (uint32_t i = 0; i < s; i++) {
            for (uint32_t j = 0; j < s; j++) {
                const gf16 t1 = gf16_matrix_get(data2, s, i, j);
                const gf16 t2 = gf16_matrix_get(data3, s, i, j);
                if (t1 != t2) {
                    gf16_matrix_print(data2, s, s);
                    printf("\n");
                    gf16_matrix_print(data3, s, s);
                    printf("error test_transpose_32x32: %d %d\n", i, j);
                    ret = 1;
                    goto finish;
                }
            }
        }

    }

finish:
    free(data1); free(data2); free(data3);
    return ret;

}

uint32_t test_transpose_64x64() {
    const size_t s = 64;
    uint8_t *data1 = gf16_matrix_alloc(s, s);
    uint8_t *data2 = gf16_matrix_alloc(s, s);
    uint8_t *data3 = gf16_matrix_alloc(s, s);

    uint32_t ret = 0;
    for (uint32_t k = 0; k < 100; k++) {
        gf16_matrix_random(data1, s, s);
        gf16_matrix_tranpose(data2, data1, s, s);
        gf16_transpose_64x64_avx2(data3, data1, 32, 32);

        for (uint32_t i = 0; i < s; i++) {
            for (uint32_t j = 0; j < s; j++) {
                const gf16 t1 = gf16_matrix_get(data2, s, i, j);
                const gf16 t2 = gf16_matrix_get(data3, s, i, j);
                if (t1 != t2) {
                    gf16_matrix_print(data2, s, s);
                    printf("\n");
                    gf16_matrix_print(data3, s, s);
                    printf("error test_transpose_64x64: %d %d\n", i, j);
                    ret = 1;
                    goto finish;
                }
            }
        }

    }

finish:
    free(data1); free(data2); free(data3);
    return ret;

}

uint32_t test_transpose2() {
    const size_t s1 = 256;
    const size_t s2 = 256;
    uint8_t *data1 = gf16_matrix_alloc(s1, s2);
    uint8_t *data2 = gf16_matrix_alloc(s1, s2);
    uint8_t *data3 = gf16_matrix_alloc(s1, s2);

    for (size_t i = 0; i < s1; i++) {
        for (size_t j = 0; j < s2; j++) {
            gf16_matrix_set(data1, s1, i, j, rand()%16);
        }
    }

    gf16_matrix_tranpose(data2, data1, s1, s1);
    gf16_matrix_tranpose_opt(data3, data1, s1, s2);


    uint32_t ret = 0;
    for (uint32_t i = 0; i < s1; i++) {
        for (uint32_t j = 0; j < s2; j++) {
            const gf16 t1 = gf16_matrix_get(data2, s1, i, j);
            const gf16 t2 = gf16_matrix_get(data3, s2, i, j);
            if (t1 != t2) {
                gf16_matrix_print(data2, s1, s2);
                printf("\n");
                gf16_matrix_print(data3, s1, s2);
                printf("error test_transpose2: %d %d\n", i, j);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(data1); free(data2); free(data3);
    return ret;
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


uint32_t test_mul() {
    // just a development function
    const size_t n = 16, m = 4, k = 12;
    uint8_t *C1 = gf16_matrix_alloc(n, k);
    uint8_t *C2 = gf16_matrix_alloc(n, k);
    uint8_t *A = gf16_matrix_alloc(n, m);
    uint8_t *B = gf16_matrix_alloc(m, k);

    uint32_t ret = 0;
    for (uint32_t t = 0; t < 1; t++) {
        gf16_matrix_id(A, n, m);
        gf16_matrix_id(B, m, k);
        gf16_matrix_product(C1, A, B, n, m, k);
        // gf16mat_prod_16x4x12(C2, A, B);

        for (uint32_t i = 0; i < n; i++) {
            for (uint32_t j = 0; j < k; j++) {
                const gf16 t1 = gf16_matrix_get(C1, n, i, j);
                const gf16 t2 = gf16_matrix_get(C2, n, i, j);
                if (t1 != t2) {
                    gf16_matrix_print(C1, n, k);
                    printf("\n");
                    gf16_matrix_print(C2, n, k);
                    printf("error test_prod_16x4x12: %d %d\n", i, j);
                    ret = 1;
                    goto finish;
                }
            }
        }

    }

finish:
    free(C1); free(C2); free(A); free(B);
    return ret;
}


#endif

int main() {
#ifdef USE_AVX2
    // if (test_transpose_32x32()) { return 1; }
    // if (test_transpose_64x64()) { return 1; }
    // if (test_transpose2()) { return 1; }
    // if (test_matrix_gaus_compressed()) { return 1; }
    // if (test_matrix_gaus()) { return 1; }
    // if (test_solve()) { return 1; }

    // if (test_vector_mul()) { return 1; }
    // if (test_mul()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}

#undef ncols
#undef nrows
