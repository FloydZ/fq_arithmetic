#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "../16/vector.h"
#include "arith.h"
#include "vector.h"
#include "matrix.h"

static const uint32_t test_lengths[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENGTHS (sizeof(test_lengths)/sizeof(test_lengths[0]))

/// {nrows1, ncols1, ncols2}
static const uint32_t test_shapes[][3] = {{1,1,1}, {3,5,2}, {17,9,4}, {33,40,7}, {64,64,64}, {7,130,3}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

bool test_add() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t = gf16to3_add(i, i);
        if (t != 0) {
            return 1;
        }

        if (gf16to3_add(i, gf16to3_neg(i)) != 0) { return 1; }
        if (gf16to3_sub(gf16to3_add(i, 0x123), 0x123) != i) { return 1; }
    }

    return 0;
}

/// schoolbook multiplication in GF(16)[x]/(x^3 + x + 1)
static gf16to3 gf16to3_mul_naive(const gf16to3 a, const gf16to3 b) {
    gf16 c[5] = {0};
    for (uint32_t i = 0; i < 3; i++) {
        for (uint32_t j = 0; j < 3; j++) {
            c[i+j] ^= gf16_mul((a >> (4*i)) & 0xF, (b >> (4*j)) & 0xF);
        }
    }
    // x^4 = x^2 + x, x^3 = x + 1
    c[2] ^= c[4]; c[1] ^= c[4];
    c[1] ^= c[3]; c[0] ^= c[3];
    return c[0] ^ (c[1] << 4) ^ (c[2] << 8);
}

bool test_mul() {
    for (uint32_t i = 0; i < 1u << 12; ++i) {
        const gf16to3 t1 = gf16to3_mul(i, 1);
        if (t1 != i) { return 1; }

        const gf16to3 t2 = gf16to3_mul(1, i);
        if (t2 != i) { return 1; }

        // inverse
        if (i && (gf16to3_mul(gf16to3_inv(i), i) != 1)) { printf("error: inv\n"); return 1; }

        for (uint32_t k = 0; k < 16; k++) {
            const gf16to3 j = rand() & 0xFFF;
            if (gf16to3_mul(i, j) != gf16to3_mul_naive(i, j)) { printf("error: mul\n"); return 1; }
            if (gf16to3_mul_gf16(i, j & 0xF) != gf16to3_mul(i, j & 0xF)) { printf("error: mul_gf16\n"); return 1; }
        }
    }

    if (gf16to3_inv(0) != 0) { return 1; }
    return 0;
}

uint32_t test_vector_ref(void) {
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {
        const uint32_t n = test_lengths[t];
        uint32_t ret = 0;
        gf16to3 *a = gf16to3_vector_alloc(n), *b = gf16to3_vector_alloc(n),
                *c = gf16to3_vector_alloc(n), *d = gf16to3_vector_alloc(n);
        gf16to3_vector_random(a, n);
        gf16to3_vector_random(b, n);
        const gf16to3 s = rand() & 0xFFF;

        gf16to3_vector_copy(c, a, n);
        gf16to3_vector_add(c, b, n);
        gf16to3_vector_copy(d, a, n);
        gf16to3_vector_scalar_add(d, s, b, n);
        gf16to3 acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (gf16to3_vector_get(c, i) != (a[i] ^ b[i])) { ret = 1; }
            if (gf16to3_vector_get(d, i) != (a[i] ^ gf16to3_mul(s, b[i]))) { ret = 1; }
            acc ^= gf16to3_mul(a[i], b[i]);
        }

        gf16to3_vector_zero(c, n);
        gf16to3_vector_scalar_add_v2(c, a, s, b, n);
        if (memcmp(c, d, n*sizeof(gf16to3))) { ret = 1; }
        if (gf16to3_vector_mul_acc(a, b, n) != acc) { ret = 1; }

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
        gf16to3 *A  = gf16to3_matrix_alloc(r, c);
        gf16to3 *AT = gf16to3_matrix_alloc(c, r);
        gf16to3 *A2 = gf16to3_matrix_alloc(r, c);
        gf16to3 *I  = gf16to3_matrix_alloc(r, r);
        gf16to3 *B  = gf16to3_matrix_alloc(c, c2);
        gf16to3 *R0 = gf16to3_matrix_alloc(r, c2);
        gf16to3 *R1 = gf16to3_matrix_alloc(r, c2);
        gf16to3 *R2 = gf16to3_matrix_alloc(r, c2);
        gf16to3 *T  = gf16to3_matrix_alloc(r, c2);

        gf16to3_matrix_random(A, r, c);
        gf16to3_matrix_random(B, c, c2);
        gf16to3_matrix_random(R0, r, c2);

        gf16to3_matrix_id(I, r, r);
        gf16to3_matrix_mul(A2, I, A, r, r, c);
        if (memcmp(A, A2, r*c*sizeof(gf16to3))) { ret = 1; }

        gf16to3_matrix_transpose(AT, A, r, c);
        gf16to3_matrix_transpose(A2, AT, c, r);
        if (memcmp(A, A2, r*c*sizeof(gf16to3))) { ret = 2; }

        gf16to3_matrix_copy(R1, R0, r, c2);
        gf16to3_matrix_add_mul(R1, A, B, r, c, c2);
        gf16to3_matrix_mul(T, A, B, r, c, c2);
        gf16to3_matrix_add(R2, R0, T, r, c2);
        if (memcmp(R1, R2, r*c2*sizeof(gf16to3))) { ret = 3; }

        const gf16to3 s = rand() & 0xFFF;
        gf16to3_matrix_copy(R1, R0, r, c2);
        gf16to3_matrix_add_scalar(R1, s, T, r, c2);
        for (uint32_t i = 0; i < r*c2; i++) {
            if (R1[i] != (R0[i] ^ gf16to3_mul(s, T[i]))) { ret = 4; }
        }

        free(A); free(AT); free(A2); free(I); free(B);
        free(R0); free(R1); free(R2); free(T);
        if (ret) {
            printf("error: test_matrix_ref %ux%ux%u: %u\n", r, c, c2, ret);
            return 1;
        }
    }
    return 0;
}

/// tests a SIMD vector implementation (suffix SUF) against the reference
#define GF16TO3_TEST_VECTOR_SIMD(SUF)                                           \
uint32_t test_vector_##SUF(void) {                                              \
    for (uint32_t t = 0; t < NR_TEST_LENGTHS; t++) {                            \
        const uint32_t n = test_lengths[t];                                     \
        const uint32_t bytes = n*sizeof(gf16to3);                               \
        uint32_t ret = 0;                                                       \
        gf16to3 *a  = gf16to3_vector_alloc(n), *b  = gf16to3_vector_alloc(n),   \
                *c1 = gf16to3_vector_alloc(n), *c2 = gf16to3_vector_alloc(n);   \
        gf16to3_vector_random(a, n);                                            \
        gf16to3_vector_random(b, n);                                            \
        gf16to3_vector_random(c1, n);                                           \
        const gf16to3 s = rand() & 0xFFF;                                       \
                                                                                \
        memcpy(c2, c1, bytes);                                                  \
        gf16to3_vector_add(c1, b, n);                                           \
        gf16to3_vector_add_##SUF(c2, b, n);                                     \
        if (memcmp(c1, c2, bytes)) { printf("add "); ret = 1; }                 \
                                                                                \
        gf16to3_vector_copy(c1, a, n);                                          \
        gf16to3_vector_add(c1, b, n);                                           \
        gf16to3_vector_add_##SUF##_v2(c2, a, b, n);                             \
        if (memcmp(c1, c2, bytes)) { printf("add_v2 "); ret = 1; }              \
                                                                                \
        gf16to3_vector_scalar_add(c1, s, b, n);                                 \
        gf16to3_vector_scalar_add_##SUF(c2, s, b, n);                           \
        if (memcmp(c1, c2, bytes)) { printf("scalar_add "); ret = 1; }          \
                                                                                \
        gf16to3_vector_scalar_add_v2(c1, a, s, b, n);                           \
        gf16to3_vector_scalar_add_##SUF##_v2(c2, a, s, b, n);                   \
        if (memcmp(c1, c2, bytes)) { printf("scalar_add_v2 "); ret = 1; }       \
                                                                                \
        if (gf16to3_vector_mul_acc(a, b, n) != gf16to3_vector_mul_acc_##SUF(a, b, n)) { \
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
#define GF16TO3_TEST_MATRIX_SIMD(SUF)                                           \
uint32_t test_matrix_##SUF(void) {                                              \
    for (uint32_t t = 0; t < NR_TEST_SHAPES; t++) {                             \
        const uint32_t r = test_shapes[t][0], c = test_shapes[t][1],            \
                       c2 = test_shapes[t][2];                                  \
        uint32_t ret = 0;                                                       \
        gf16to3 *A  = gf16to3_matrix_alloc(r, c),  *A2 = gf16to3_matrix_alloc(r, c), \
                *B  = gf16to3_matrix_alloc(c, c2),                              \
                *M1 = gf16to3_matrix_alloc(r, c),  *M2 = gf16to3_matrix_alloc(r, c), \
                *R1 = gf16to3_matrix_alloc(r, c2), *R2 = gf16to3_matrix_alloc(r, c2); \
        gf16to3_matrix_random(A, r, c);                                         \
        gf16to3_matrix_random(A2, r, c);                                        \
        gf16to3_matrix_random(B, c, c2);                                        \
        const gf16to3 s = rand() & 0xFFF;                                       \
                                                                                \
        gf16to3_matrix_add(M1, A, A2, r, c);                                    \
        gf16to3_matrix_add_##SUF(M2, A, A2, r, c);                              \
        if (memcmp(M1, M2, r*c*sizeof(gf16to3))) { ret = 1; }                   \
                                                                                \
        gf16to3_matrix_add_scalar(M1, s, A, r, c);                              \
        gf16to3_matrix_add_scalar_##SUF(M2, s, A, r, c);                        \
        if (memcmp(M1, M2, r*c*sizeof(gf16to3))) { ret = 2; }                   \
                                                                                \
        gf16to3_matrix_random(R2, r, c2);                                       \
        gf16to3_matrix_mul(R1, A, B, r, c, c2);                                 \
        gf16to3_matrix_mul_##SUF(R2, A, B, r, c, c2);                           \
        if (memcmp(R1, R2, r*c2*sizeof(gf16to3))) { ret = 3; }                  \
                                                                                \
        gf16to3_matrix_random(R1, r, c2);                                       \
        gf16to3_matrix_copy(R2, R1, r, c2);                                     \
        gf16to3_matrix_add_mul(R1, A, B, r, c, c2);                             \
        gf16to3_matrix_add_mul_##SUF(R2, A, B, r, c, c2);                       \
        if (memcmp(R1, R2, r*c2*sizeof(gf16to3))) { ret = 4; }                  \
                                                                                \
        free(A); free(A2); free(B); free(M1); free(M2); free(R1); free(R2);     \
        if (ret) {                                                              \
            printf("error: test_matrix_" #SUF " %ux%ux%u: %u\n", r, c, c2, ret);\
            return 1;                                                           \
        }                                                                       \
    }                                                                           \
    return 0;                                                                   \
}

#if defined(USE_NEON)
GF16TO3_TEST_VECTOR_SIMD(u128)
GF16TO3_TEST_MATRIX_SIMD(u128)

uint32_t test_gf16to3_vector_extend_gf16_x8() {
    const uint8_t in[4] = {0x10, 0x32, 0x54, 0x76};
    uint16_t out[8];
    vst1q_u16(out, gf16to3_vector_extend_gf16_x8(in));
    for (uint32_t i = 0; i < 8; i++) {
        if (out[i] != gf16_vector_get(in, i)) { printf("error: extend_x8\n"); return 1; }
    }

    vst1q_u16(out, gf16to3_vector_extend_gf16_x8_v2(in));
    for (uint32_t i = 0; i < 8; i++) {
        if (out[i] != gf16_vector_get(in, i)) { printf("error: extend_x8_v2\n"); return 1; }
    }
    return 0;
}

uint32_t test_gf16to3_vector_extend_gf16_x16() {
    const uint8_t in[8] = {0x10, 0x32, 0x54, 0x76, 0x89, 0x50, 0x41, 0x23};
    uint16_t out[16];
    vst1q_u16_x2(out, gf16to3_vector_extend_gf16_x16(in));
    for (uint32_t i = 0; i < 16; i++) {
        if (out[i] != gf16_vector_get(in, i)) { printf("error: extend_x16\n"); return 1; }
    }
    return 0;
}

uint32_t test_arith_vector_mul() {
    uint16_t tmp[16];
    for (uint64_t i = 1; i < 1u << 11; ++i) {
        for (uint64_t j = 2; j < 1u << 11; j += 7) {
            const uint16x8_t a128 = vdupq_n_u16(i);
            const uint16x8_t b128 = vdupq_n_u16(j);
            const uint16x8_t c128 = gf16to3v_mul_u128(a128, b128);
            vst1q_u16(tmp, c128);

            const gf16to3 c1 = gf16to3_mul(i, j);
            for (uint32_t k = 0; k < 8; ++k) {
                if (tmp[k] != c1) {
                    printf("gf16to3 neon 128 mul error\n");
                    return 1;
                }
            }

            uint16x8x2_t a256, b256, c256;
            a256.val[0] = vdupq_n_u16(i); a256.val[1] = vdupq_n_u16(i);
            b256.val[0] = vdupq_n_u16(j); b256.val[1] = vdupq_n_u16(j);
            c256 = gf16to3v_mul_u256(a256, b256);
            vst1q_u16_x2(tmp, c256);

            for (uint32_t k = 0; k < 16; ++k) {
                if (tmp[k] != c1) {
                    printf("gf16to3 neon 256 mul error\n");
                    return 1;
                }
            }

            vst1q_u16(tmp, gf16to3v_mul_gf16_u128(a128, j & 0xF));
            for (uint32_t k = 0; k < 8; ++k) {
                if (tmp[k] != gf16to3_mul_gf16(i, j & 0xF)) {
                    printf("gf16to3 neon mul_gf16 error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}

uint32_t test_vector_add_gf16() {
    const uint32_t N = 17;
    gf16 *m1 = gf16_vector_alloc(N);
    gf16to3 *m2 = gf16to3_vector_alloc(N);
    gf16to3 *m3 = gf16to3_vector_alloc(N);

    gf16_vector_random(m1, N);
    gf16to3_vector_random(m2, N);
    gf16to3_vector_copy(m3, m2, N);

    gf16to3_vector_add_gf16(m2, m1, N);
    gf16to3_vector_add_gf16_u128(m3, m1, N);

    uint32_t ret = 0;
    if (memcmp(m2, m3, N*sizeof(gf16to3))) {
        printf("test vector add gf16\n");
        ret = 1;
    }

    free(m1); free(m2); free(m3);
    return ret;
}
#endif

#ifdef USE_AVX2
GF16TO3_TEST_VECTOR_SIMD(u256)
GF16TO3_TEST_MATRIX_SIMD(u256)

uint32_t test_arith_vector_mul() {
    uint16_t tmp[32];
    for (uint16_t i = 1; i < 1u << 12; ++i) {
        for (uint16_t j = 1; j < 1u << 12; j += 3) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf16to3v_mul_u256(a256, b256);
            const uint16_t c = gf16to3_mul(i, j);

            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf16to3 avx mul error: %d %d\n", i, j);
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tmp, gf16to3v_mul_gf16_u256(a256, j & 0xF));
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != gf16to3_mul_gf16(i, j & 0xF)) {
                    printf("gf16to3 avx mul_gf16 error: %d %d\n", i, j);
                    return 1;
                }
            }
        }
    }

    return 0;
}

uint32_t test_vector_add_gf16() {
    const uint32_t N = 17;
    gf16 *m1 = gf16_vector_alloc(N);
    gf16to3 *m2 = gf16to3_vector_alloc(N);
    gf16to3 *m3 = gf16to3_vector_alloc(N);

    gf16_vector_random(m1, N);
    gf16to3_vector_random(m2, N);
    gf16to3_vector_copy(m3, m2, N);

    gf16to3_vector_add_gf16(m2, m1, N);
    gf16to3_vector_add_gf16_u256(m3, m1, N);

    uint32_t ret = 0;
    for (uint32_t j = 0; j < N; ++j) {
        const gf16to3 c = m2[j];
        const gf16to3 d = m3[j];
        if (d != c) {
            gf16_vector_print(m1, N);
            gf16to3_vector_print(m2, N);
            gf16to3_vector_print(m3, N);
            printf("test vector add gf16\n");
            ret = 1;
            goto finish;
        }
    }

    finish:
    free(m1); free(m2); free(m3);
    return ret;
}

uint32_t test_vector_scalar_add_gf16() {
    const uint32_t N = 33;
    gf16to3 *m1 = gf16to3_vector_alloc(N);
    gf16to3 *m2 = gf16to3_vector_alloc(N);
    gf16to3 *m3 = gf16to3_vector_alloc(N);

    gf16to3_vector_random(m1, N);
    gf16to3_vector_random(m2, N);
    gf16to3_vector_copy(m3, m2, N);

    const gf16 scalar = (rand() & 0xF) | 2;
    gf16to3_vector_scalar_add_gf16(m2, scalar, m1, N);
    gf16to3_vector_scalar_add_gf16_u256(m3, scalar, m1, N);

    uint32_t ret = 0;
    if (memcmp(m2, m3, N*sizeof(gf16to3))) {
        printf("test vector scalar add gf16\n");
        ret = 1;
    }

    free(m1); free(m2); free(m3);
    return ret;
}

uint32_t test_vector_scalar_gf16() {
    const uint32_t N = 17;
    gf16 *m1 = gf16_vector_alloc(N);
    gf16to3 *m2 = gf16to3_vector_alloc(N);
    gf16to3 *m3 = gf16to3_vector_alloc(N);

    gf16_vector_random(m1, N);

    const gf16to3 scalar = (rand() & 0xFFF) | 0x100;
    gf16to3_vector_scalar_gf16(m2, scalar, m1, N);
    gf16to3_vector_scalar_gf16_u256(m3, scalar, m1, N);

    uint32_t ret = 0;
    for (uint32_t j = 0; j < N; ++j) {
        const gf16to3 c = m2[j];
        const gf16to3 d = m3[j];
        if (d != c) {
            gf16_vector_print(m1, N);
            gf16to3_vector_print(m2, N);
            gf16to3_vector_print(m3, N);
            printf("test vector scalar gf16\n");
            ret = 1;
            goto finish;
        }
    }

    finish:
    free(m1); free(m2); free(m3);
    return ret;
}

uint32_t test_vector_add_scalar_gf16() {
    const uint32_t N = 17;
    gf16 *m1 = gf16_vector_alloc(N);
    gf16to3 *m2 = gf16to3_vector_alloc(N);
    gf16to3 *m3 = gf16to3_vector_alloc(N);

    gf16_vector_random(m1, N);

    const gf16to3 scalar = (rand() & 0xFFF) | 0x100;
    gf16to3_vector_add_scalar_gf16(m2, scalar, m1, N);
    gf16to3_vector_add_scalar_gf16_u256(m3, scalar, m1, N);

    uint32_t ret = 0;
    for (uint32_t j = 0; j < N; ++j) {
        const gf16to3 c = m2[j];
        const gf16to3 d = m3[j];
        if (d != c) {
            gf16_vector_print(m1, N);
            gf16to3_vector_print(m2, N);
            gf16to3_vector_print(m3, N);
            printf("test vector add scalar gf16\n");
            ret = 1;
            goto finish;
        }
    }

    finish:
    free(m1); free(m2); free(m3);
    return ret;
}


uint32_t test_matrix_mul() {
    const uint32_t nrows = 24;
    const uint32_t ncols = 6;
    const uint32_t ncols2 = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(ncols, ncols2);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols2);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols2);

    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, ncols, ncols2);
    gf16to3_matrix_mul(m3, m1, m2, nrows, ncols, ncols2);
    gf16to3_matrix_mul_le32xCxC(m4, m1, m2, nrows, ncols, ncols2);


    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols2; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix mul\n");
                gf16to3_matrix_print(m3, nrows, ncols2);
                gf16to3_matrix_print(m4, nrows, ncols2);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(m1); free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_mul_vector() {
    const uint32_t nrows = 22;
    const uint32_t ncols = 229;

    // hardcoded
    const uint32_t ncols2 = 1;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(ncols, ncols2);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols2);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols2);

    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, ncols, ncols2);
    gf16to3_matrix_mul(m3, m1, m2, nrows, ncols, ncols2);
    gf16to3_matrix_mul_vector(m4, m1, m2, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols2; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix mul vector\n");
                gf16to3_matrix_print(m3, nrows, ncols2);
                gf16to3_matrix_print(m4, nrows, ncols2);
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
    const uint32_t nrows = 37;
    const uint32_t ncols = 5;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16 *m2 = gf16_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16to3_matrix_zero(m3, nrows, ncols);
    gf16to3_matrix_zero(m4, nrows, ncols);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16_matrix_random(m2, nrows, ncols);
    gf16to3_matrix_add_gf16(m3, m1, m2, nrows, ncols);
    gf16to3_matrix_add_gf16_XxX(m4, m1, m2, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix gf16 add\n");
                gf16to3_matrix_print(m3, nrows, ncols);
                gf16to3_matrix_print(m4, nrows, ncols);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(m1); free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_gf16_map() {
    const uint32_t nrows = 35;
    const uint32_t ncols = 3;
    gf16 *m2 = gf16_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16to3_matrix_zero(m3, nrows, ncols);
    gf16to3_matrix_zero(m4, nrows, ncols);
    gf16_matrix_random(m2, nrows, ncols);
    gf16to3_matrix_map_gf16(m3, m2, nrows, ncols);
    gf16to3_matrix_map_gf16_u256(m4, m2, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix gf16 map\n");
                gf16to3_matrix_print(m3, nrows, ncols);
                gf16to3_matrix_print(m4, nrows, ncols);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_mul_gf16() {
    const uint32_t nrows = 16;
    const uint32_t ncols = 16;
    const uint32_t ncols2= 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16 *m2 = gf16_matrix_alloc(ncols, ncols2);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols2);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols2);

    gf16to3_matrix_zero(m3, nrows, ncols2);
    gf16to3_matrix_zero(m4, nrows, ncols2);
    gf16to3_matrix_random(m1, nrows, ncols);
    gf16_matrix_random(m2, ncols, ncols2);
    gf16to3_matrix_mul_gf16(m3, m1, m2, nrows, ncols, ncols2);
    gf16to3_matrix_mul_gf16_XxX(m4, m1, m2, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols2; ++j) {
            gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix gf16 mul\n");
                gf16to3_matrix_print(m3, nrows, ncols);
                gf16to3_matrix_print(m4, nrows, ncols);
                ret = 1;
                goto finish;
            }
        }
    }

    finish:
    free(m1); free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_mul_gf16_v2() {
    const uint32_t shapes[2][3] = {{229, 255, 1}, {37, 21, 5}};
    for (uint32_t s = 0; s < 2; s++) {
        const uint32_t nrows = shapes[s][0];
        const uint32_t ncols = shapes[s][1];
        const uint32_t ncols2= shapes[s][2];
        gf16 *m1    = gf16_matrix_alloc(nrows, ncols);
        gf16to3 *m2 = gf16to3_matrix_alloc(ncols, ncols2);
        gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols2);
        gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols2);

        gf16_matrix_random(m1, nrows, ncols);
        gf16to3_matrix_random(m2, ncols, ncols2);

        gf16to3_matrix_product_gf16_v2(m3, m1, m2, nrows, ncols, ncols2);
        gf16to3_matrix_mul_gf16_v2_u256(m4, m1, m2, nrows, ncols, ncols2);

        uint32_t ret = 0;
        if (memcmp(m3, m4, nrows*ncols2*sizeof(gf16to3))) {
            printf("test matrix gf16 mul_v2\n");
            ret = 1;
        }

        free(m1); free(m2); free(m3); free(m4);
        if (ret) { return 1; }
    }
    return 0;
}

uint32_t test_matrix_add_multiple() {
    const uint32_t nrows = 16;
    const uint32_t ncols = 16;
    gf16to3 *m1 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m2 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16to3_matrix_random(m1, nrows, ncols);
    gf16to3_matrix_random(m2, nrows, ncols);
    gf16to3 scalar = rand() & 0xFFF;
    gf16to3_matrix_add_multiple_3(m3, m2, scalar, m1, nrows, ncols);
    gf16to3_matrix_add_multiple_3_XxX(m4, m2, scalar, m1, nrows, ncols);

    uint32_t ret = 0;
    if (memcmp(m3, m4, nrows*ncols*sizeof(gf16to3))) {
        printf("test matrix add multiple\n");
        ret = 1;
    }

    free(m1); free(m2); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_scalar_gf16() {
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf16 *m1 = gf16_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16_matrix_random(m1, nrows, ncols);
    gf16to3 scalar = rand() & 0xFFF;
    gf16to3_matrix_scalar_gf16(m3, scalar,m1, nrows, ncols);
    gf16to3_matrix_scalar_gf16_u256(m4, scalar, m1, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols; ++j) {
            const gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            const gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix scalar gf16\n");
                gf16to3_matrix_print(m3, nrows, ncols);
                gf16to3_matrix_print(m4, nrows, ncols);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(m1); free(m3); free(m4);
    return ret;
}

uint32_t test_matrix_add_scalar_gf16() {
    const uint32_t nrows = 15;
    const uint32_t ncols = 15;
    gf16 *m1 = gf16_matrix_alloc(nrows, ncols);
    gf16to3 *m3 = gf16to3_matrix_alloc(nrows, ncols);
    gf16to3 *m4 = gf16to3_matrix_alloc(nrows, ncols);

    gf16_matrix_random(m1, nrows, ncols);
    gf16to3 scalar = rand() & 0xFFF;
    gf16to3_matrix_add_scalar_gf16(m3, scalar,m1, nrows, ncols);
    gf16to3_matrix_add_scalar_gf16_u256(m4, scalar, m1, nrows, ncols);

    uint32_t ret = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        for (uint32_t j = 0; j < ncols; ++j) {
            const gf16to3 c = gf16to3_matrix_get(m3, nrows, i, j);
            const gf16to3 d = gf16to3_matrix_get(m4, nrows, i, j);
            if (c != d) {
                printf("test matrix add scalar\n");
                gf16to3_matrix_print(m3, nrows, ncols);
                gf16to3_matrix_print(m4, nrows, ncols);
                ret = 1;
                goto finish;
            }
        }
    }

finish:
    free(m1); free(m3); free(m4);
    return ret;
}
#endif // USE_AVX2

#ifdef USE_AVX512
GF16TO3_TEST_VECTOR_SIMD(u512)
GF16TO3_TEST_MATRIX_SIMD(u512)

uint32_t test_arith_u512(void) {
    uint16_t a[32], b[32], c[32];
    for (uint32_t t = 0; t < 1000; t++) {
        for (uint32_t k = 0; k < 32; k++) { a[k] = rand() & 0xFFF; b[k] = rand() & 0xFFF; }
        const __m512i va = _mm512_loadu_si512((const void *)a);
        const __m512i vb = _mm512_loadu_si512((const void *)b);

        _mm512_storeu_si512((void *)c, gf16to3v_mul_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != gf16to3_mul(a[k], b[k])) { printf("error: test_arith_u512 mul\n"); return 1; }
        }

        _mm512_storeu_si512((void *)c, gf16to3v_add_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u512 add\n"); return 1; }
        }

        _mm512_storeu_si512((void *)c, gf16to3v_sub_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) {
            if (c[k] != (a[k] ^ b[k])) { printf("error: test_arith_u512 sub\n"); return 1; }
        }
    }
    return 0;
}
#endif // USE_AVX512


int main() {
    srand(1);
    if (test_add()) { printf("error: test_add\n"); return 1; }
    if (test_mul()) { printf("error: test_mul\n"); return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }

#if defined(USE_NEON)
    if (test_arith_vector_mul()) { return 1; }
    if (test_gf16to3_vector_extend_gf16_x8()) { return 1; }
    if (test_gf16to3_vector_extend_gf16_x16()) { return 1; }
    if (test_vector_add_gf16()) { return 1; }
    if (test_vector_u128()) { return 1; }
    if (test_matrix_u128()) { return 1; }
#endif

#if defined(USE_AVX2)
    if (test_arith_vector_mul()) { return 1; }
    if (test_vector_add_gf16()) { return 1; }
    if (test_vector_u256()) { return 1; }
    if (test_matrix_u256()) { return 1; }

    if (test_matrix_mul()) { return 1; }
    if (test_matrix_mul_vector()) { return 1; }
    if (test_matrix_gf16_add()) { return 1; }
    if (test_matrix_gf16_map()) { return 1; }
    if (test_matrix_mul_gf16()) { return 1; }
    if (test_matrix_mul_gf16_v2()) { return 1; }
    if (test_matrix_add_multiple()) { return 1; }
    if (test_matrix_scalar_gf16()) { return 1; }
    if (test_matrix_add_scalar_gf16()) { return 1; }

    if (test_vector_scalar_add_gf16()) { return 1; }
    if (test_vector_scalar_gf16()) { return 1; }
    if (test_vector_add_scalar_gf16()) { return 1; }
#endif

#ifdef USE_AVX512
    if (test_arith_u512()) { return 1; }
    if (test_vector_u512()) { return 1; }
    if (test_matrix_u512()) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
