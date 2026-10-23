// NOTE: currently the row echelon form need this
#define NROWS 127
#define NCOLS 128

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// {nrows1, ncols1, ncols2}
static const uint32_t test_shapes[][3] = {{1,1,1}, {3,5,2}, {17,9,4}, {33,40,7}, {64,64,64}, {7,130,3}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

uint32_t test_arith(void) {
    if (gf16_inv(0) != 0) { printf("error: test_arith inv(0)\n"); return 1; }
    for (uint32_t a = 0; a < 16; a++) {
        if (gf16_neg(a) != a) { printf("error: test_arith neg\n"); return 1; }
        if (gf16_add(a, gf16_neg(a)) != 0) { printf("error: test_arith a+neg(a)\n"); return 1; }
        if (a && (gf16_mul(gf16_inv(a), a) != 1)) { printf("error: test_arith inv\n"); return 1; }
        if (gf16_sqr(a) != gf16_mul(a, a)) { printf("error: test_arith sqr\n"); return 1; }

        for (uint32_t b = 0; b < 16; b++) {
            if (gf16_sub(gf16_add(a, b), b) != a) { printf("error: test_arith sub\n"); return 1; }
            if (gf16_mul(a, b) != gf16_mul_v3(a, b)) { printf("error: test_arith mul_v3\n"); return 1; }
            if (gf16_mul(a, b) != gf16_mul(b, a)) { printf("error: test_arith mul commutative\n"); return 1; }

            const uint8_t c = gf16v_mul(a ^ (b << 4), b ^ (a << 4));
            const uint8_t d = gf16_mul(a, b) ^ (gf16_mul(b, a) << 4);
            if (c != d) { printf("error: test_arith gf16v_mul\n"); return 1; }
        }
    }

    return 0;
}

uint32_t test_vector_ref(void) {
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t n = test_lengths[t];
        uint32_t ret = 0;
        gf16 *a = gf16_vector_alloc(n), *b = gf16_vector_alloc(n),
             *c = gf16_vector_alloc(n), *d = gf16_vector_alloc(n);
        gf16_vector_random(a, n);
        gf16_vector_random(b, n);
        const gf16 s = rand() & 0xF;

        gf16_vector_copy(c, a, n);
        for (uint32_t i = 0; i < n; i++) {
            if (gf16_vector_get(c, i) != gf16_vector_get(a, i)) { ret = 1; }
        }

        gf16_vector_add(c, b, n);
        for (uint32_t i = 0; i < n; i++) {
            if (gf16_vector_get(c, i) != (gf16_vector_get(a, i) ^ gf16_vector_get(b, i))) { ret = 1; }
        }

        gf16_vector_copy(c, a, n);
        gf16_vector_scalar_add(c, s, b, n);
        gf16_vector_zero(d, n);
        gf16_vector_scalar_add_v2(d, a, s, b, n);
        gf16 acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            const gf16 e = gf16_vector_get(a, i) ^ gf16_mul(s, gf16_vector_get(b, i));
            if (gf16_vector_get(c, i) != e) { ret = 1; }
            if (gf16_vector_get(d, i) != e) { ret = 1; }
            acc ^= gf16_mul(gf16_vector_get(a, i), gf16_vector_get(b, i));
        }

        if (gf16_vector_mul_acc(a, b, n) != acc) { ret = 1; }

        gf16_vector_zero(c, n);
        for (uint32_t i = 0; i < n; i++) {
            if (gf16_vector_get(c, i) != 0) { ret = 1; }
        }

        free(a); free(b); free(c); free(d);
        if (ret) {
            printf("error: test_vector_ref n=%u\n", n);
            return 1;
        }
    }
    return 0;
}

uint32_t test_matrix_ref(void) {
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {
        const uint32_t r = test_shapes[t][0], c = test_shapes[t][1], c2 = test_shapes[t][2];
        uint32_t ret = 0;
        gf16 *A  = gf16_matrix_alloc(r, c);
        gf16 *AT = gf16_matrix_alloc(c, r);
        gf16 *A2 = gf16_matrix_alloc(r, c);
        gf16 *I  = gf16_matrix_alloc(r, r);
        gf16 *B  = gf16_matrix_alloc(c, c2);
        gf16 *R0 = gf16_matrix_alloc(r, c2);
        gf16 *R1 = gf16_matrix_alloc(r, c2);
        gf16 *R2 = gf16_matrix_alloc(r, c2);
        gf16 *T  = gf16_matrix_alloc(r, c2);

        gf16_matrix_random(A, r, c);
        gf16_matrix_random(B, c, c2);
        gf16_matrix_random(R0, r, c2);

        // id * A == A
        gf16_matrix_id(I, r, r);
        gf16_matrix_mul(A2, I, A, r, r, c);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < c; j++) {
                if (gf16_matrix_get(A2, r, i, j) != gf16_matrix_get(A, r, i, j)) { ret = 1; }
            }
        }

        // transpose(transpose(A)) == A, and transpose == transpose_opt
        gf16_matrix_transpose(AT, A, r, c);
        gf16_matrix_zero(A2, r, c);
        gf16_matrix_transpose(A2, AT, c, r);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < c; j++) {
                if (gf16_matrix_get(A2, r, i, j) != gf16_matrix_get(A, r, i, j)) { ret = 1; }
                if (gf16_matrix_get(AT, c, j, i) != gf16_matrix_get(A, r, i, j)) { ret = 1; }
            }
        }
        gf16_matrix_zero(AT, c, r);
        gf16_matrix_transpose_opt(AT, A, r, c);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < c; j++) {
                if (gf16_matrix_get(AT, c, j, i) != gf16_matrix_get(A, r, i, j)) { ret = 1; }
            }
        }

        // add_mul == mul + add
        gf16_matrix_copy(R1, R0, r, c2);
        gf16_matrix_add_mul(R1, A, B, r, c, c2);
        gf16_matrix_mul(T, A, B, r, c, c2);
        gf16_matrix_add(R2, R0, T, r, c2);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < c2; j++) {
                if (gf16_matrix_get(R1, r, i, j) != gf16_matrix_get(R2, r, i, j)) { ret = 1; }
            }
        }

        // add_scalar
        const gf16 s = rand() & 0xF;
        gf16_matrix_copy(R1, R0, r, c2);
        gf16_matrix_add_scalar(R1, s, T, r, c2);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < c2; j++) {
                const gf16 e = gf16_matrix_get(R0, r, i, j) ^ gf16_mul(s, gf16_matrix_get(T, r, i, j));
                if (gf16_matrix_get(R1, r, i, j) != e) { ret = 1; }
            }
        }

        free(A); free(AT); free(A2); free(I); free(B);
        free(R0); free(R1); free(R2); free(T);
        if (ret) {
            printf("error: test_matrix_ref %ux%ux%u\n", r, c, c2);
            return 1;
        }
    }
    return 0;
}

/// solves A x = b for a random invertible A and checks x and that A is
/// reduced to the identity matrix
uint32_t test_solve(void) {
    const uint32_t s = 78;
    uint32_t ret = 0;
    uint8_t *A  = gf16_matrix_alloc(s, s);
    uint8_t *AT = gf16_matrix_alloc(s, s);
    uint8_t *x  = gf16_matrix_alloc(s, 1);
    uint8_t *b  = gf16_matrix_alloc(s, 1);

    gf16_matrix_rng_full_rank(A, s, s);
    gf16_matrix_random(x, s, 1);
    gf16_matrix_mul(b, A, x, s, s, 1);

    const uint32_t srows = gf16_solve_transpose(A, AT, b, s, s);
    if (srows != s) {
        printf("error: test_solve wrong rank\n");
        ret = 1;
        goto finish;
    }

    for (uint32_t i = 0; i < s; i++) {
        if (gf16_vector_get(b, i) != gf16_vector_get(x, i)) {
            printf("error: test_solve wrong solution\n");
            ret = 1;
            goto finish;
        }
        for (uint32_t j = 0; j < s; j++) {
             if (gf16_matrix_get(A, s, i, j) != (i == j)) {
                 printf("error: test_solve not reduced\n");
                 ret = 1;
                 goto finish;
             }
        }
    }

finish:
    free(A); free(AT); free(x); free(b);
    return ret;
}

/// tests a SIMD vector implementation (suffix SUF) against the reference
#define GF16_TEST_VECTOR_SIMD(SUF)                                              \
uint32_t test_vector_##SUF(void) {                                              \
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {                            \
        const uint32_t n = test_lengths[t];                                     \
        const uint32_t bytes = (n+1)/2;                                         \
        uint32_t ret = 0;                                                       \
        gf16 *a  = gf16_vector_alloc(n), *b  = gf16_vector_alloc(n),            \
             *c1 = gf16_vector_alloc(n), *c2 = gf16_vector_alloc(n);            \
        gf16_vector_random(a, n);                                               \
        gf16_vector_random(b, n);                                               \
        gf16_vector_random(c1, n);                                              \
        /* dirty the unused nibble, it must not be used/changed */              \
        if (n & 1u) { a[n/2] |= 0x30; b[n/2] |= 0xA0; c1[n/2] |= 0x50; }        \
        const gf16 s = rand() & 0xF;                                            \
                                                                                \
        memcpy(c2, c1, bytes);                                                  \
        gf16_vector_add(c1, b, n);                                              \
        gf16_vector_add_##SUF(c2, b, n);                                        \
        if (memcmp(c1, c2, bytes)) { printf("add "); ret = 1; }                 \
                                                                                \
        memcpy(c2, c1, bytes);                                                  \
        gf16_vector_copy(c1, a, n);                                             \
        gf16_vector_add(c1, b, n);                                              \
        gf16_vector_add_##SUF##_v2(c2, a, b, n);                                \
        if (memcmp(c1, c2, bytes)) { printf("add_v2 "); ret = 1; }              \
                                                                                \
        memcpy(c2, c1, bytes);                                                  \
        gf16_vector_scalar_add(c1, s, b, n);                                    \
        gf16_vector_scalar_add_##SUF(c2, s, b, n);                              \
        if (memcmp(c1, c2, bytes)) { printf("scalar_add "); ret = 1; }          \
                                                                                \
        memcpy(c2, c1, bytes);                                                  \
        gf16_vector_scalar_add_v2(c1, a, s, b, n);                              \
        gf16_vector_scalar_add_##SUF##_v2(c2, a, s, b, n);                      \
        if (memcmp(c1, c2, bytes)) { printf("scalar_add_v2 "); ret = 1; }       \
                                                                                \
        if (gf16_vector_mul_acc(a, b, n) != gf16_vector_mul_acc_##SUF(a, b, n)) {\
            printf("mul_acc "); ret = 1;                                        \
        }                                                                       \
                                                                                \
        free(a); free(b); free(c1); free(c2);                                   \
        if (ret) {                                                              \
            printf("error: test_vector_" #SUF " n=%u\n", n);                    \
            return 1;                                                           \
        }                                                                       \
    }                                                                           \
    return 0;                                                                   \
}

/// tests a SIMD matrix implementation (suffix SUF) against the reference
#define GF16_TEST_MATRIX_SIMD(SUF)                                              \
uint32_t test_matrix_##SUF(void) {                                              \
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {                             \
        const uint32_t r = test_shapes[t][0], c = test_shapes[t][1],            \
                       c2 = test_shapes[t][2];                                  \
        uint32_t ret = 0;                                                       \
        gf16 *A  = gf16_matrix_alloc(r, c),  *A2 = gf16_matrix_alloc(r, c),     \
             *B  = gf16_matrix_alloc(c, c2),                                    \
             *M1 = gf16_matrix_alloc(r, c),  *M2 = gf16_matrix_alloc(r, c),     \
             *R1 = gf16_matrix_alloc(r, c2), *R2 = gf16_matrix_alloc(r, c2);    \
        gf16_matrix_random(A, r, c);                                            \
        gf16_matrix_random(A2, r, c);                                           \
        gf16_matrix_random(B, c, c2);                                           \
        const gf16 s = rand() & 0xF;                                            \
        const uint32_t nbytes = gf16_matrix_bytes_size(r, c);                   \
                                                                                \
        gf16_matrix_add(M1, A, A2, r, c);                                       \
        gf16_matrix_add_##SUF(M2, A, A2, r, c);                                 \
        if (memcmp(M1, M2, nbytes)) { printf("add "); ret = 1; }                \
                                                                                \
        gf16_matrix_add_scalar(M1, s, A, r, c);                                 \
        gf16_matrix_add_scalar_##SUF(M2, s, A, r, c);                           \
        if (memcmp(M1, M2, nbytes)) { printf("add_scalar "); ret = 1; }         \
                                                                                \
        gf16_matrix_random(R2, r, c2);                                          \
        gf16_matrix_mul(R1, A, B, r, c, c2);                                    \
        gf16_matrix_mul_##SUF(R2, A, B, r, c, c2);                              \
        for (uint32_t i = 0; i < r; i++) {                                      \
            for (uint32_t j = 0; j < c2; j++) {                                 \
                if (gf16_matrix_get(R1, r, i, j) != gf16_matrix_get(R2, r, i, j)) { ret = 2; } \
            }                                                                   \
        }                                                                       \
                                                                                \
        gf16_matrix_random(R1, r, c2);                                          \
        gf16_matrix_copy(R2, R1, r, c2);                                        \
        gf16_matrix_add_mul(R1, A, B, r, c, c2);                                \
        gf16_matrix_add_mul_##SUF(R2, A, B, r, c, c2);                          \
        for (uint32_t i = 0; i < r; i++) {                                      \
            for (uint32_t j = 0; j < c2; j++) {                                 \
                if (gf16_matrix_get(R1, r, i, j) != gf16_matrix_get(R2, r, i, j)) { ret = 3; } \
            }                                                                   \
        }                                                                       \
                                                                                \
        free(A); free(A2); free(B); free(M1); free(M2); free(R1); free(R2);     \
        if (ret) {                                                              \
            printf("error: test_matrix_" #SUF " %ux%ux%u: %u\n", r, c, c2, ret);\
            return 1;                                                           \
        }                                                                       \
    }                                                                           \
    return 0;                                                                   \
}

#ifdef USE_AVX2
#include <immintrin.h>

GF16_TEST_VECTOR_SIMD(u256)
GF16_TEST_MATRIX_SIMD(u256)

uint32_t test_arith_u256(void) {
    uint8_t a[32], b[32], c[32];
    for (uint32_t t = 0; t < 1000; t++) {
        for (uint32_t k = 0; k < 32; k++) { a[k] = rand(); b[k] = rand(); }
        const __m256i va = _mm256_loadu_si256((const __m256i *)a);
        __m256i vb = _mm256_loadu_si256((const __m256i *)b);

        _mm256_storeu_si256((__m256i *)c, gf16v_mul_u256(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != gf16v_mul(a[k], b[k])) { printf("error: test_arith_u256 mul\n"); return 1; }
        }

        _mm256_storeu_si256((__m256i *)c, gf16v_add_u256(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u256 add\n"); return 1; }
        }

        _mm256_storeu_si256((__m256i *)c, gf16v_sub_u256(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u256 sub\n"); return 1; }
        }

        // restricted multiplication: only the lower nibble of each byte of b is used
        for (uint32_t k = 0; k < 32; k++) { b[k] &= 0xF; }
        vb = _mm256_loadu_si256((const __m256i *)b);
        _mm256_storeu_si256((__m256i *)c, gf16v_mul_lo_u256(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != gf16v_mul(a[k], b[k] ^ (b[k] << 4))) { printf("error: test_arith_u256 mul_lo\n"); return 1; }
        }

        const __m128i va128 = _mm_loadu_si128((const __m128i *)a);
        const __m128i vb128 = _mm_loadu_si128((const __m128i *)b);
        _mm_storeu_si128((__m128i *)c, gf16v_mul_lo_u128(va128, vb128));
        for (uint32_t k = 0; k < 16; k++) {
            if (c[k] != gf16v_mul(a[k], b[k] ^ (b[k] << 4))) { printf("error: test_arith_u256 mul_lo_u128\n"); return 1; }
        }
    }
    return 0;
}

uint32_t test_vector_mul() {
    uint8_t tmp[32] = {0};
    for (uint32_t i = 1; i < (1u<<4); i++) {
        for (uint32_t j = 1; j < (1u<<4); j++) {
            const ff_t c = gf16_mul(i, j);
            __m256i a256 = _mm256_set1_epi8(i);
            __m256i b256 = _mm256_set1_epi8(j);
            __m256i c256 = gf16v_mul_lo_u256(a256, b256);
            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (uint32_t k = 0; k < 16; k++) {
                if (c != tmp[k]) {
                    printf("test_vector_mul v1: error\n");
                    return 1;
                }
            }

            __m256i a256_v2 = _mm256_set1_epi8(i ^ (i<<4));
            __m256i b256_v2 = _mm256_set1_epi8(j ^ (j<<4));
            __m256i c256_v2 = gf16v_mul_u256(a256_v2, b256_v2);
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
    // NOTE: `gf16_matrix_rng_full_rank` produces M = P*[I | R] (col-major,
    // NROWS x NCOLS, P invertible), whose first NROWS columns (each 64 bytes)
    // form the row-major NROWS x NCOLS matrix [P^T | 0] used by the echelon
    // code. Its left NROWS x NROWS block is invertible, hence all pivots are
    // on the diagonal.
    uint8_t *A = gf16_matrix_alloc(NROWS, NCOLS);
    gf16_matrix_rng_full_rank(A, NROWS, NCOLS);

    row_echelon_form_compressed(A, NROWS, NCOLS);
    for (uint32_t i = 0; i < NROWS; i++) {
        const gf16 t = gf16_matrix_get(A, NCOLS, i, i);
        if (t != 1) {
            gf16_matrix_print(A, NCOLS, NROWS);
            printf("error: test_matrix_gaus_compressed\n");
            ret = 1;
            goto finish;
        }
    }

finish:
    free(A);
    return ret;
}

uint32_t test_matrix_gaus() {
    // NOTE: this tests the expanded version: row-major, one element per byte
    uint8_t ret = 0;
    uint8_t *A = calloc(1, NCOLS*NROWS);

    // upper unitriangular left NROWS x NROWS block (=> invertible), random rest
    for (uint32_t i = 0; i < NROWS; i++) {
        for (uint32_t j = 0; j < NCOLS; j++) {
            A[i*NCOLS + j] = (j == i) ? 1 : ((j > i) ? (rand() & 0xF) : 0);
        }
    }
    // random row operations, which keep the left block invertible
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

    for (uint32_t i = 0; i < NROWS; i++) {
        const gf16 t = A[i*NCOLS + i];
        if (t != 1) {
            printf("error: test_matrix_gaus\n");
            ret = 1;
            goto finish;
        }
    }

finish:
    free(A);
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
        gf16_matrix_transpose(data2, data1, s, s);
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
        gf16_matrix_transpose(data2, data1, s, s);
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
    const size_t s2 = 192;
    uint8_t *data1 = gf16_matrix_alloc(s1, s2);
    uint8_t *data2 = gf16_matrix_alloc(s2, s1);
    uint8_t *data3 = gf16_matrix_alloc(s2, s1);

    for (size_t i = 0; i < s1; i++) {
        for (size_t j = 0; j < s2; j++) {
            gf16_matrix_set(data1, s1, i, j, rand()%16);
        }
    }

    gf16_matrix_transpose(data2, data1, s1, s2);
    gf16_matrix_transpose_opt(data3, data1, s1, s2);


    uint32_t ret = 0;
    for (uint32_t i = 0; i < s2; i++) {
        for (uint32_t j = 0; j < s1; j++) {
            const gf16 t1 = gf16_matrix_get(data2, s2, i, j);
            const gf16 t2 = gf16_matrix_get(data3, s2, i, j);
            if (t1 != t2) {
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

/// compares the reference multiplication with the 16x4xk kernel
uint32_t test_mul() {
    const size_t n = 16, m = 4, k = 12;
    uint8_t *C1 = gf16_matrix_alloc(n, k);
    uint8_t *C2 = gf16_matrix_alloc(n, k);
    uint8_t *A = gf16_matrix_alloc(n, m);
    uint8_t *B = gf16_matrix_alloc(m, k);

    uint32_t ret = 0;
    for (uint32_t t = 0; t < 10; t++) {
        gf16_matrix_random(A, n, m);
        gf16_matrix_random(B, m, k);
        gf16_matrix_mul(C1, A, B, n, m, k);
        gf16mat_prod_16x4xk(C2, A, B, k);

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
#endif // USE_AVX2

#ifdef USE_NEON
GF16_TEST_VECTOR_SIMD(u128)
GF16_TEST_MATRIX_SIMD(u128)

uint32_t test_arith_u128(void) {
    uint8_t a[16], b[16], c[16];
    for (uint32_t t = 0; t < 1000; t++) {
        for (uint32_t k = 0; k < 16; k++) { a[k] = rand(); b[k] = rand(); }
        const uint8x16_t va = vld1q_u8(a);
        uint8x16_t vb = vld1q_u8(b);

        vst1q_u8(c, gf16v_mul_u128(va, vb));
        for (uint32_t k = 0; k < 16; k++) {
            if (c[k] != gf16v_mul(a[k], b[k])) { printf("error: test_arith_u128 mul\n"); return 1; }
        }

        vst1q_u8(c, gf16v_add_u128(va, vb));
        for (uint32_t k = 0; k < 16; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u128 add\n"); return 1; }
        }

        vst1q_u8(c, gf16v_sub_u128(va, vb));
        for (uint32_t k = 0; k < 16; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u128 sub\n"); return 1; }
        }

        for (uint32_t k = 0; k < 16; k++) { b[k] &= 0xF; }
        vb = vld1q_u8(b);
        vst1q_u8(c, gf16v_mul_lo_u128(va, vb));
        for (uint32_t k = 0; k < 16; k++) {
            if (c[k] != gf16v_mul(a[k], b[k] ^ (b[k] << 4))) { printf("error: test_arith_u128 mul_lo\n"); return 1; }
        }
    }
    return 0;
}
#endif // USE_NEON

#ifdef USE_AVX512
GF16_TEST_VECTOR_SIMD(u512)
GF16_TEST_MATRIX_SIMD(u512)

uint32_t test_arith_u512(void) {
    uint8_t a[64], b[64], c[64];
    for (uint32_t t = 0; t < 1000; t++) {
        for (uint32_t k = 0; k < 64; k++) { a[k] = rand(); b[k] = rand(); }
        const __m512i va = _mm512_loadu_si512((const void *)a);
        const __m512i vb = _mm512_loadu_si512((const void *)b);

        _mm512_storeu_si512((void *)c, gf16v_mul_u512(va, vb));
        for (uint32_t k = 0; k < 64; k++) {
            if (c[k] != gf16v_mul(a[k], b[k])) { printf("error: test_arith_u512 mul\n"); return 1; }
        }

        _mm512_storeu_si512((void *)c, gf16v_add_u512(va, vb));
        for (uint32_t k = 0; k < 64; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u512 add\n"); return 1; }
        }

        _mm512_storeu_si512((void *)c, gf16v_sub_u512(va, vb));
        for (uint32_t k = 0; k < 64; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u512 sub\n"); return 1; }
        }
    }
    return 0;
}
#endif // USE_AVX512

int main() {
    srand(1);
    if (test_arith()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }
    if (test_solve()) { return 1; }

#ifdef USE_AVX2
    if (test_arith_u256()) { return 1; }
    if (test_vector_u256()) { return 1; }
    if (test_matrix_u256()) { return 1; }

    if (test_transpose_32x32()) { return 1; }
    if (test_transpose_64x64()) { return 1; }
    if (test_transpose2()) { return 1; }
    if (test_matrix_gaus_compressed()) { return 1; }
    if (test_matrix_gaus()) { return 1; }

    if (test_vector_mul()) { return 1; }
    if (test_mul()) { return 1; }
#endif

#ifdef USE_NEON
    if (test_arith_u128()) { return 1; }
    if (test_vector_u128()) { return 1; }
    if (test_matrix_u128()) { return 1; }
#endif

#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
    if (test_vector_u512()) { return 1; }
    if (test_matrix_u512()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
