#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

#include "../2/matrix.h"

/// vector lengths used by the SIMD tests (incl. non multiples of the lane count)
static const uint32_t test_lens[] = {0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257};
#define NR_TEST_LENS (sizeof(test_lens)/sizeof(test_lens[0]))

/// matrix shapes {n_rows1, n_cols1, n_cols2} used by the SIMD tests
static const uint32_t test_dims[][3] = {
    {1, 1, 1}, {3, 5, 2}, {17, 9, 1}, {33, 40, 5}, {16, 16, 16},
    {64, 3, 7}, {65, 7, 9}, {8, 8, 1}, {5, 45, 3}, {100, 4, 2},
};
#define NR_TEST_DIMS (sizeof(test_dims)/sizeof(test_dims[0]))

uint32_t test_arith() {
    if (gf2to12_inv(0) != 0) {
        printf("gf2to12 inv(0) error\n");
        return 1;
    }

    for (uint32_t a = 1; a < (1u << 12u); a++) {
        if (gf2to12_mul(a, gf2to12_inv(a)) != 1) {
            printf("gf2to12 inv error: %d\n", a);
            return 1;
        }
        if (gf2to12_add(a, gf2to12_neg(a)) != 0) {
            printf("gf2to12 neg error: %d\n", a);
            return 1;
        }
    }

    for (uint32_t t = 0; t < (1u << 16u); t++) {
        const gf2to12 a = rand() & 0xFFF;
        const gf2to12 b = rand() & 0xFFF;
        if (gf2to12_sub(gf2to12_add(a, b), b) != a) {
            printf("gf2to12 add/sub error\n");
            return 1;
        }
        if (gf2to12_mul(a, b) != gf2to12_mul_v2(a, b)) {
            printf("gf2to12 mul/mul_v2 error\n");
            return 1;
        }
        if (gf2to12_mul(a, b) != gf2to12_mul(b, a)) {
            printf("gf2to12 mul commutative error\n");
            return 1;
        }
    }

    return 0;
}

uint32_t test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        gf2to12 *a  = gf2to12_vector_alloc(n);
        gf2to12 *b  = gf2to12_vector_alloc(n);
        gf2to12 *c1 = gf2to12_vector_alloc(n);
        gf2to12 *c2 = gf2to12_vector_alloc(n);
        gf2to12_vector_random(a, n);
        gf2to12_vector_random(b, n);
        const gf2to12 s = rand() & 0xFFF;

        // c1 = a + s*b
        gf2to12_vector_scalar_add_v2(c1, a, s, b, n);
        gf2to12_vector_copy(c2, a, n);
        gf2to12_vector_scalar_add(c2, s, b, n);
        if (memcmp(c1, c2, n * sizeof(gf2to12)) != 0) {
            printf("error test_vector_ref scalar_add_v2\n");
            return 1;
        }

        gf2to12 acc = 0;
        for (uint32_t i = 0; i < n; i++) {
            gf2to12_vector_set(c1, i, gf2to12_mul(a[i], b[i]));
            acc ^= gf2to12_vector_get(c1, i);
        }
        if (acc != gf2to12_vector_mul_acc(a, b, n)) {
            printf("error test_vector_ref mul_acc\n");
            return 1;
        }

        free(a); free(b); free(c1); free(c2);
    }

    return 0;
}

uint32_t test_matrix_ref() {
    for (uint32_t d = 0; d < NR_TEST_DIMS; d++) {
        const uint32_t r = test_dims[d][0], k = test_dims[d][1], c = test_dims[d][2];
        gf2to12 *A  = gf2to12_matrix_alloc(r, k);
        gf2to12 *At = gf2to12_matrix_alloc(k, r);
        gf2to12 *A2 = gf2to12_matrix_alloc(r, k);
        gf2to12 *I  = gf2to12_matrix_alloc(k, k);
        gf2to12 *B  = gf2to12_matrix_alloc(k, c);
        gf2to12 *C1 = gf2to12_matrix_alloc(r, c);
        gf2to12 *C2 = gf2to12_matrix_alloc(r, c);
        gf2to12 *C3 = gf2to12_matrix_alloc(r, c);

        gf2to12_matrix_random(A, r, k);
        gf2to12_matrix_random(B, k, c);
        gf2to12_matrix_random(C1, r, c);

        // A * I == A
        gf2to12_matrix_id(I, k, k);
        gf2to12_matrix_mul(A2, A, I, r, k, k);
        if (memcmp(A, A2, r * k * sizeof(gf2to12)) != 0) {
            printf("error test_matrix_ref id\n");
            return 1;
        }

        // (A^T)^T == A
        gf2to12_matrix_transpose(At, A, r, k);
        gf2to12_matrix_zero(A2, r, k);
        gf2to12_matrix_transpose(A2, At, k, r);
        if (memcmp(A, A2, r * k * sizeof(gf2to12)) != 0) {
            printf("error test_matrix_ref transpose\n");
            return 1;
        }
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < k; j++) {
                if (gf2to12_matrix_get(A, r, i, j) != gf2to12_matrix_get(At, k, j, i)) {
                    printf("error test_matrix_ref transpose entry\n");
                    return 1;
                }
            }
        }

        // add_mul == mul + add
        gf2to12_matrix_copy(C2, C1, r, c);
        gf2to12_matrix_add_mul(C1, A, B, r, k, c);
        gf2to12_matrix_mul(C3, A, B, r, k, c);
        gf2to12_matrix_add(C2, C2, C3, r, c);
        if (memcmp(C1, C2, r * c * sizeof(gf2to12)) != 0) {
            printf("error test_matrix_ref add_mul\n");
            return 1;
        }

        // add_scalar == entry wise
        const gf2to12 s = rand() & 0xFFF;
        gf2to12_matrix_copy(A2, A, r, k);
        gf2to12_matrix_add_scalar(A2, s, A, r, k);
        for (uint32_t i = 0; i < r; i++) {
            for (uint32_t j = 0; j < k; j++) {
                const gf2to12 e = gf2to12_matrix_get(A, r, i, j);
                if (gf2to12_matrix_get(A2, r, i, j) != (e ^ gf2to12_mul(s, e))) {
                    printf("error test_matrix_ref add_scalar\n");
                    return 1;
                }
            }
        }

        free(A); free(At); free(A2); free(I); free(B); free(C1); free(C2); free(C3);
    }

    return 0;
}

/// tests all canonical SIMD vector functions with suffix `SUF` against the reference
#define GF2TO12_TEST_VECTOR_SIMD(SUF)                                               \
uint32_t test_vector_simd_##SUF() {                                                 \
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {                                   \
        const uint32_t n = test_lens[l];                                            \
        gf2to12 *a  = gf2to12_vector_alloc(n);                                      \
        gf2to12 *b  = gf2to12_vector_alloc(n);                                      \
        gf2to12 *c1 = gf2to12_vector_alloc(n);                                      \
        gf2to12 *c2 = gf2to12_vector_alloc(n);                                      \
        gf2to12_vector_random(a, n);                                                \
        gf2to12_vector_random(b, n);                                                \
        gf2to12_vector_random(c1, n);                                               \
        gf2to12_vector_copy(c2, c1, n);                                             \
        const gf2to12 s = rand() & 0xFFF;                                           \
                                                                                    \
        gf2to12_vector_add(c1, a, n);                                               \
        gf2to12_vector_add_##SUF(c2, a, n);                                         \
        if (memcmp(c1, c2, n*sizeof(gf2to12))) {                                    \
            printf("error vector_add_" #SUF " n=%u\n", n); return 1; }              \
                                                                                    \
        gf2to12_vector_copy(c1, a, n);                                              \
        gf2to12_vector_add(c1, b, n);                                               \
        gf2to12_vector_add_##SUF##_v2(c2, a, b, n);                                 \
        if (memcmp(c1, c2, n*sizeof(gf2to12))) {                                    \
            printf("error vector_add_" #SUF "_v2 n=%u\n", n); return 1; }           \
                                                                                    \
        gf2to12_vector_scalar_add(c1, s, a, n);                                     \
        gf2to12_vector_scalar_add_##SUF(c2, s, a, n);                               \
        if (memcmp(c1, c2, n*sizeof(gf2to12))) {                                    \
            printf("error vector_scalar_add_" #SUF " n=%u\n", n); return 1; }       \
                                                                                    \
        gf2to12_vector_scalar_add_v2(c1, a, s, b, n);                               \
        gf2to12_vector_scalar_add_##SUF##_v2(c2, a, s, b, n);                       \
        if (memcmp(c1, c2, n*sizeof(gf2to12))) {                                    \
            printf("error vector_scalar_add_" #SUF "_v2 n=%u\n", n); return 1; }    \
                                                                                    \
        if (gf2to12_vector_mul_acc(a, b, n) != gf2to12_vector_mul_acc_##SUF(a, b, n)) { \
            printf("error vector_mul_acc_" #SUF " n=%u\n", n); return 1; }          \
                                                                                    \
        free(a); free(b); free(c1); free(c2);                                       \
    }                                                                               \
    return 0;                                                                       \
}

/// tests all canonical SIMD matrix functions with suffix `SUF` against the reference
#define GF2TO12_TEST_MATRIX_SIMD(SUF)                                               \
uint32_t test_matrix_simd_##SUF() {                                                 \
    for (uint32_t d = 0; d < NR_TEST_DIMS; d++) {                                   \
        const uint32_t r = test_dims[d][0], k = test_dims[d][1], c = test_dims[d][2];\
        gf2to12 *A  = gf2to12_matrix_alloc(r, k);                                   \
        gf2to12 *A2 = gf2to12_matrix_alloc(r, k);                                   \
        gf2to12 *D1 = gf2to12_matrix_alloc(r, k);                                   \
        gf2to12 *D2 = gf2to12_matrix_alloc(r, k);                                   \
        gf2to12 *B  = gf2to12_matrix_alloc(k, c);                                   \
        gf2to12 *C1 = gf2to12_matrix_alloc(r, c);                                   \
        gf2to12 *C2 = gf2to12_matrix_alloc(r, c);                                   \
        gf2to12_matrix_random(A, r, k);                                             \
        gf2to12_matrix_random(A2, r, k);                                            \
        gf2to12_matrix_random(B, k, c);                                             \
        gf2to12_matrix_random(C1, r, c);                                            \
        gf2to12_matrix_copy(C2, C1, r, c);                                          \
        const gf2to12 s = rand() & 0xFFF;                                           \
                                                                                    \
        gf2to12_matrix_add(D1, A, A2, r, k);                                        \
        gf2to12_matrix_add_##SUF(D2, A, A2, r, k);                                  \
        if (memcmp(D1, D2, r*k*sizeof(gf2to12))) {                                  \
            printf("error matrix_add_" #SUF " %ux%u\n", r, k); return 1; }          \
                                                                                    \
        gf2to12_matrix_add_scalar(D1, s, A, r, k);                                  \
        gf2to12_matrix_add_scalar_##SUF(D2, s, A, r, k);                            \
        if (memcmp(D1, D2, r*k*sizeof(gf2to12))) {                                  \
            printf("error matrix_add_scalar_" #SUF " %ux%u\n", r, k); return 1; }   \
                                                                                    \
        gf2to12_matrix_add_mul(C1, A, B, r, k, c);                                  \
        gf2to12_matrix_add_mul_##SUF(C2, A, B, r, k, c);                            \
        if (memcmp(C1, C2, r*c*sizeof(gf2to12))) {                                  \
            printf("error matrix_add_mul_" #SUF " %ux%ux%u\n", r, k, c); return 1; }\
                                                                                    \
        gf2to12_matrix_mul(C1, A, B, r, k, c);                                      \
        gf2to12_matrix_mul_##SUF(C2, A, B, r, k, c);                                \
        if (memcmp(C1, C2, r*c*sizeof(gf2to12))) {                                  \
            printf("error matrix_mul_" #SUF " %ux%ux%u\n", r, k, c); return 1; }    \
                                                                                    \
        free(A); free(A2); free(D1); free(D2); free(B); free(C1); free(C2);         \
    }                                                                               \
    return 0;                                                                       \
}

#if defined(USE_NEON)
uint32_t test_arith_vector_mul() {
    uint16_t tmp[16];
    for (uint16_t i = 1; i < 1u << 11; ++i) {
        for (uint16_t j = 1; j < 1u << 11; ++j) {
            const uint16x8_t a128 = vdupq_n_u16(i);
            const uint16x8_t b128 = vdupq_n_u16(j);
            const uint16x8_t c128 = gf2to12v_mul_u128(a128, b128);
            vst1q_u16(tmp, c128);

            const gf2to12 c1 = gf2to12_mul(i, j);
            for (uint32_t k = 0; k < 8; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 neon 128 mul error\n");
                    return 1;
                }
            }

            uint16x8x2_t a256, b256, c256;
            a256.val[0] = vdupq_n_u16(i); a256.val[1] = vdupq_n_u16(i);
            b256.val[0] = vdupq_n_u16(j); b256.val[1] = vdupq_n_u16(j);
            c256 = gf2to12v_mul_u256(a256, b256);
            vst1q_u16_x2(tmp, c256);

            for (uint32_t k = 0; k < 16; ++k) {
                if (tmp[k] != c1) {
                    printf("gf2to12 neon 256 mul error\n");
                    return 1;
                }
            }
        }
    }

    return 0;
}

uint32_t test_arith_simd_u128() {
    uint16_t a[8], b[8], c[8];
    for (uint32_t t = 0; t < 10000; t++) {
        for (uint32_t k = 0; k < 8; k++) { a[k] = rand() & 0xFFF; b[k] = rand() & 0xFFF; }
        const uint16x8_t va = vld1q_u16(a), vb = vld1q_u16(b);

        vst1q_u16(c, gf2to12v_add_u128(va, vb));
        for (uint32_t k = 0; k < 8; k++) { if (c[k] != gf2to12_add(a[k], b[k])) { printf("error add_u128\n"); return 1; } }
        vst1q_u16(c, gf2to12v_sub_u128(va, vb));
        for (uint32_t k = 0; k < 8; k++) { if (c[k] != gf2to12_sub(a[k], b[k])) { printf("error sub_u128\n"); return 1; } }
        vst1q_u16(c, gf2to12v_mul_u128(va, vb));
        for (uint32_t k = 0; k < 8; k++) { if (c[k] != gf2to12_mul(a[k], b[k])) { printf("error mul_u128\n"); return 1; } }
    }
    return 0;
}

uint32_t test_expand_gf2_u128() {
    for (uint32_t t = 0; t < 256; t++) {
        uint8_t in[2] = {(uint8_t)t, (uint8_t)(t*7 + 3)};
        uint16_t out[16];
        vst1q_u16(out, gf2to12v_expand_gf2_x8_u128(in));
        for (uint32_t k = 0; k < 8; k++) {
            if (out[k] != ((in[0] >> k) & 1u)) { printf("error expand_gf2_x8_u128\n"); return 1; }
        }
        vst1q_u16_x2(out, gf2to12v_expand_gf2_x16_u256(in));
        for (uint32_t k = 0; k < 16; k++) {
            if (out[k] != ((in[k/8] >> (k%8)) & 1u)) { printf("error expand_gf2_x16_u256\n"); return 1; }
        }
    }
    return 0;
}

GF2TO12_TEST_VECTOR_SIMD(u128)
GF2TO12_TEST_MATRIX_SIMD(u128)
#endif

#ifdef USE_AVX2

#ifdef USE_AVX512
/// NOTE: two 256-bit stores instead of `_mm512_storeu_si512` into a stack
/// buffer, which crashed under gcc + ASan
static inline void test_store_u512(uint16_t *out, const __m512i a) {
    _mm256_storeu_si256((__m256i *)(out +  0), _mm512_castsi512_si256(a));
    _mm256_storeu_si256((__m256i *)(out + 16), _mm512_extracti64x4_epi64(a, 1));
}

static inline __m512i test_load_u512(const uint16_t *in) {
    const __m256i lo = _mm256_loadu_si256((const __m256i *)(in +  0));
    const __m256i hi = _mm256_loadu_si256((const __m256i *)(in + 16));
    return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}
#endif

uint32_t test_arith_vector_mul() {
    uint16_t tmp[32];
    for (int i = 0; i < 1u << 11; ++i) {
        for (int j = 0; j < 1u << 11; ++j) {
            const __m256i a256 = _mm256_set1_epi16(i);
            const __m256i b256 = _mm256_set1_epi16(j);
            const __m256i c256 = gf2to12v_mul_u256(a256, b256);
            const __m256i d256 = gf2to12v_mul_u256_v2(a256, b256);
            const gf2to12 c = gf2to12_mul(i, j);
            const gf2to12 d = gf2to12_mul_v2(i, j);
            if (c != d) {
                printf("gf2to12 mul error\n");
                return 1;
            }

            _mm256_storeu_si256((__m256i *)tmp, c256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx mul error\n");
                    return 1;
                }
            }

            _mm256_storeu_si256((__m256i *)tmp, d256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx mul_v2 error\n");
                    return 1;
                }
            }

#ifdef USE_AVX512
            const __m256i e256 = gf2to12v_mul_u256_avx512(a256, b256);
            _mm256_storeu_si256((__m256i *)tmp, e256);
            for (int k = 0; k < 16; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx512 mul256 error\n");
                    return 1;
                }
            }

            const __m512i a512 = _mm512_set1_epi16(i);
            const __m512i b512 = _mm512_set1_epi16(j);
            const __m512i e512 = gf2to12v_mul_u512(a512, b512);
            test_store_u512(tmp, e512);
            for (int k = 0; k < 32; ++k) {
                if (tmp[k] != c) {
                    printf("gf2to12 avx512 mul512 error\n");
                    return 1;
                }
            }
#endif
        }
    }

    return 0;
}

uint32_t test_arith_simd_u256() {
    uint16_t a[16], b[16], c[16];
    for (uint32_t t = 0; t < 10000; t++) {
        for (uint32_t k = 0; k < 16; k++) { a[k] = rand() & 0xFFF; b[k] = rand() & 0xFFF; }
        const __m256i va = _mm256_loadu_si256((const __m256i *)a);
        const __m256i vb = _mm256_loadu_si256((const __m256i *)b);

        _mm256_storeu_si256((__m256i *)c, gf2to12v_add_u256(va, vb));
        for (uint32_t k = 0; k < 16; k++) { if (c[k] != gf2to12_add(a[k], b[k])) { printf("error add_u256\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to12v_sub_u256(va, vb));
        for (uint32_t k = 0; k < 16; k++) { if (c[k] != gf2to12_sub(a[k], b[k])) { printf("error sub_u256\n"); return 1; } }
        _mm256_storeu_si256((__m256i *)c, gf2to12v_mul_u256(va, vb));
        for (uint32_t k = 0; k < 16; k++) { if (c[k] != gf2to12_mul(a[k], b[k])) { printf("error mul_u256\n"); return 1; } }
    }
    return 0;
}

#ifdef USE_AVX512
uint32_t test_arith_simd_u512() {
    uint16_t a[32], b[32], c[32];
    for (uint32_t t = 0; t < 10000; t++) {
        for (uint32_t k = 0; k < 32; k++) { a[k] = rand() & 0xFFF; b[k] = rand() & 0xFFF; }
        const __m512i va = test_load_u512(a);
        const __m512i vb = test_load_u512(b);

        test_store_u512(c, gf2to12v_add_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) { if (c[k] != gf2to12_add(a[k], b[k])) { printf("error add_u512\n"); return 1; } }
        test_store_u512(c, gf2to12v_sub_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) { if (c[k] != gf2to12_sub(a[k], b[k])) { printf("error sub_u512\n"); return 1; } }
        test_store_u512(c, gf2to12v_mul_u512(va, vb));
        for (uint32_t k = 0; k < 32; k++) { if (c[k] != gf2to12_mul(a[k], b[k])) { printf("error mul_u512\n"); return 1; } }
    }
    return 0;
}

GF2TO12_TEST_VECTOR_SIMD(u512)
GF2TO12_TEST_MATRIX_SIMD(u512)
#endif

GF2TO12_TEST_VECTOR_SIMD(u256)
GF2TO12_TEST_MATRIX_SIMD(u256)

uint32_t test_vector_add() {
    const uint32_t N = 256;

    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2to12_vector_random(v1, N);
    gf2to12_vector_zero(v2, N);
    gf2to12_vector_zero(v3, N);

    gf2to12_vector_add(v2, v1, N);
    gf2to12_vector_add_u256(v3, v1, N);

    // gf2to12_vector_print(v2, N);
    // gf2to12_vector_print(v3, N);
    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_add");
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_add_gf2() {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t N = test_lens[l];

        gf2 *v1 = gf2_vector_alloc(N);
        gf2to12 *v2 = gf2to12_vector_alloc(N);
        gf2to12 *v3 = gf2to12_vector_alloc(N);

        gf2_vector_random(v1, N);
        gf2to12_vector_random(v2, N);
        gf2to12_vector_copy(v3, v2, N);

        gf2to12_vector_add_gf2(v2, v1, N);
        gf2to12_vector_add_gf2_u256(v3, v1, N);

        for (uint32_t i = 0; i < N; i++) {
            if (v2[i] != v3[i]) {
                printf("error test_vector_add_gf2 n=%u\n", N);
                return 1;
            }
        }

        free(v1); free(v2); free(v3);
    }
    return 0;
}

uint32_t test_vector_scalar_add_gf2_v3() {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t N = test_lens[l];

        gf2 *v1     = gf2_vector_alloc(N);
        gf2to12 *v2 = gf2to12_vector_alloc(N);
        gf2to12 *v3 = gf2to12_vector_alloc(N);

        gf2_vector_random(v1, N);
        gf2to12_vector_random(v2, N);
        gf2to12_vector_copy(v3, v2, N);

        gf2to12 t = rand() & 0xFFF;
        gf2to12_vector_scalar_add_gf2_v3(v2, t, v1, N);
        gf2to12_vector_scalar_add_gf2_u256_v3(v3, t, v1, N);

        for (uint32_t i = 0; i < N; i++) {
            if (v2[i] != v3[i]) {
                printf("error test_vector_scalar_add_gf2_v3 n=%u\n", N);
                return 1;
            }
        }

        free(v1); free(v2); free(v3);
    }
    return 0;
}

uint32_t test_vector_set_to_gf2() {
    const uint32_t N = 321;

    gf2 *v1     = gf2_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);
    gf2to12 *v3 = gf2to12_vector_alloc(N);

    gf2_vector_random(v1, N);

    gf2to12_vector_set_to_gf2(v2, v1, N);
    gf2to12_vector_set_to_gf2_u256(v3, v1, N);

    for (uint32_t i = 0; i < N; i++) {
        if (v2[i] != v3[i]) {
            printf("error test_vector_set_to_gf2\n");
            gf2to12_vector_print(v2, N);
            gf2to12_vector_print(v3, N);
            return 1;
        }
    }

    free(v1); free(v2); free(v3);
    return 0;
}

uint32_t test_vector_mul_acc() {
    const uint32_t N = 256;

    gf2to12 *v1 = gf2to12_vector_alloc(N);
    gf2to12 *v2 = gf2to12_vector_alloc(N);

    gf2to12_vector_random(v1, N);
    gf2to12_vector_random(v2, N);

    const gf2to12 t1 = gf2to12_vector_mul_acc(v2, v1, N);
    const gf2to12 t2 = gf2to12_vector_mul_acc_u256(v2, v1, N);

    if (t1 != t2) {
        printf("error test_vector_mul_acc");
        return 1;
    }

    free(v1); free(v2);
    return 0;
}



uint32_t test_matrix_add_gf2() {
    const uint32_t nrows = 17;
    const uint32_t ncols = 17;

    gf2 *v1     = gf2_matrix_alloc(nrows, ncols);
    gf2to12 *v2 = gf2to12_matrix_alloc(nrows, ncols);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows, ncols);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows, ncols);

    gf2_matrix_random(v1, nrows, ncols);
    gf2to12_matrix_random(v2, nrows, ncols);

    gf2to12_matrix_add_gf2(C1, v1, v2, nrows, ncols);
    gf2to12_matrix_add_gf2_u256(C2, v1, v2, nrows, ncols);

    for (uint32_t i = 0; i < nrows; i++) {
        for (uint32_t j = 0; j < ncols; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows, i, j);
            if (t1 != t2) {
                printf("error test_matrix_add_gf2");
                gf2to12_matrix_print(C1, nrows, ncols);
                gf2to12_matrix_print(C2, nrows, ncols);
                return 1;
            }
        }
    }

    free(v1); free(v2); free(C1); free(C2);
    return 0;
}

uint32_t test_matrix_add_scalar_gf2() {
    // covers the n_rows == 4,5,6 special cases (incl. all column tails) and the generic code
    const uint32_t rows[] = {4, 5, 6, 7, 8, 13};
    const uint32_t cols[] = {1, 2, 3, 4, 5, 6, 7, 45};
    for (uint32_t r = 0; r < sizeof(rows)/sizeof(rows[0]); r++) {
        for (uint32_t c = 0; c < sizeof(cols)/sizeof(cols[0]); c++) {
            const uint32_t nrows = rows[r];
            const uint32_t ncols = cols[c];

            gf2 *v1     = gf2_matrix_alloc(nrows, ncols);
            gf2to12 *C1 = gf2to12_matrix_alloc(nrows, ncols);
            gf2to12 *C2 = gf2to12_matrix_alloc(nrows, ncols);

            gf2_matrix_random(v1, nrows, ncols);
            gf2to12_matrix_random(C1, nrows, ncols);
            gf2to12_matrix_copy(C2, C1, nrows, ncols);

            const gf2to12 v2 = rand() & 0xFFF;
            gf2to12_matrix_add_scalar_gf2(C1, v2, v1, nrows, ncols);
            gf2to12_matrix_add_scalar_gf2_u256(C2, v2, v1, nrows, ncols);

            for (uint32_t i = 0; i < nrows; i++) {
                for (uint32_t j = 0; j < ncols; j++) {
                    const gf2to12 t1 = gf2to12_matrix_get(C1, nrows, i, j);
                    const gf2to12 t2 = gf2to12_matrix_get(C2, nrows, i, j);
                    if (t1 != t2) {
                        printf("error test_matrix_add_scalar_gf2 %ux%u\n", nrows, ncols);
                        gf2to12_matrix_print(C1, nrows, ncols);
                        gf2to12_matrix_print(C2, nrows, ncols);
                        return 1;
                    }
                }
            }

            free(v1); free(C1); free(C2);
        }
    }
    return 0;
}

uint32_t test_matrix_mul_gf2() {
    const uint32_t dims[][3] = {{33, 32, 1}, {16, 9, 3}, {17, 5, 4}, {72, 3, 2}};
    for (uint32_t d = 0; d < sizeof(dims)/sizeof(dims[0]); d++) {
        const uint32_t nrows1 = dims[d][0];
        const uint32_t ncols1 = dims[d][1];
        const uint32_t ncols2 = dims[d][2];

        gf2 *v1     =     gf2_matrix_alloc(nrows1, ncols1);
        gf2to12 *v2 = gf2to12_matrix_alloc(ncols1, ncols2);
        gf2to12 *C1 = gf2to12_matrix_alloc(nrows1, ncols2);
        gf2to12 *C2 = gf2to12_matrix_alloc(nrows1, ncols2);

        gf2_matrix_random(v1, nrows1, ncols1);
        gf2to12_matrix_random(v2, ncols1, ncols2);
        gf2to12_matrix_random(C2, nrows1, ncols2);

        gf2to12_matrix_mul_gf2(C1, v1, v2, nrows1, ncols1, ncols2);
        gf2to12_matrix_mul_gf2_u256(C2, v1, v2, nrows1, ncols1, ncols2);

        for (uint32_t i = 0; i < nrows1; i++) {
            for (uint32_t j = 0; j < ncols2; j++) {
                const gf2to12 t1 = gf2to12_matrix_get(C1, nrows1, i, j);
                const gf2to12 t2 = gf2to12_matrix_get(C2, nrows1, i, j);
                if (t1 != t2) {
                    printf("error test_matrix_mul_gf2\n");
                    gf2to12_matrix_print(C1, nrows1, ncols2);
                    gf2to12_matrix_print(C2, nrows1, ncols2);
                    return 1;
                }
            }
        }

        free(v1); free(v2); free(C1); free(C2);
    }
    return 0;
}

uint32_t test_matrix_mul() {
    const uint32_t nrows1 = 22;
    const uint32_t ncols1 = 637;
    const uint32_t ncols2 = 1;

    gf2to12 *v1 = gf2to12_matrix_alloc(nrows1, ncols1);
    gf2to12 *v2 = gf2to12_matrix_alloc(ncols1, ncols2);
    gf2to12 *C1 = gf2to12_matrix_alloc(nrows1, ncols2);
    gf2to12 *C2 = gf2to12_matrix_alloc(nrows1, ncols2);

    gf2to12_matrix_random(v1, nrows1, ncols1);
    gf2to12_matrix_random(v2, ncols1, ncols2);

    gf2to12_matrix_mul(C1, v1, v2, nrows1, ncols1, ncols2);
    gf2to12_matrix_mul_u256(C2, v1, v2, nrows1, ncols1, ncols2);

    for (uint32_t i = 0; i < nrows1; i++) {
        for (uint32_t j = 0; j < ncols2; j++) {
            const gf2to12 t1 = gf2to12_matrix_get(C1, nrows1, i, j);
            const gf2to12 t2 = gf2to12_matrix_get(C2, nrows1, i, j);
            if (t1 != t2) {
                printf("error test_matrix_mul\n");
                gf2to12_matrix_print(C1, nrows1, ncols2);
                gf2to12_matrix_print(C2, nrows1, ncols2);
                return 1;
            }
        }
    }

    free(v1); free(v2); free(C1); free(C2);
    return 0;
}


#endif


int main() {
    srand(0x1234);
    if (test_arith()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }

#if defined(USE_AVX2) || defined(USE_NEON)
    if (test_arith_vector_mul()) { return 1; }
#if defined(USE_NEON)
    if (test_arith_simd_u128()) { return 1; }
    if (test_expand_gf2_u128()) { return 1; }
    if (test_vector_simd_u128()) { return 1; }
    if (test_matrix_simd_u128()) { return 1; }
#endif
#if defined(USE_AVX2)
    if (test_arith_simd_u256()) { return 1; }
    if (test_vector_simd_u256()) { return 1; }
    if (test_matrix_simd_u256()) { return 1; }
#if defined(USE_AVX512)
    if (test_arith_simd_u512()) { return 1; }
    if (test_vector_simd_u512()) { return 1; }
    if (test_matrix_simd_u512()) { return 1; }
#endif

    if (test_vector_add()) { return 1; }
    if (test_vector_add_gf2()) { return 1; }
    if (test_vector_scalar_add_gf2_v3()) { return 1; }
    if (test_vector_mul_acc()) { return 1; }
    if (test_vector_set_to_gf2()) { return 1; }

    if (test_matrix_add_gf2()) { return 1; }
    if (test_matrix_add_scalar_gf2()) { return 1; }
    if (test_matrix_mul_gf2()) { return 1; }
    if (test_matrix_mul()) { return 1; }
#endif
#endif

    printf("all good\n");
	return 0;
}
