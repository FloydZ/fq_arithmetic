#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "arith.h"
#include "vector.h"
#include "matrix.h"

#define N (1u << 10)

static const uint32_t test_lens[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENS (sizeof(test_lens)/sizeof(test_lens[0]))
/// {nrows1, ncols1, ncols2}
static const uint32_t test_shapes[][3] = {{1,1,1}, {3,5,2}, {17,9,13}, {33,40,7}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

static void gf256to16_random(gf256to16 a) {
    for (uint32_t i = 0; i < 16; i++) {
        a[i] = rand();
    }
}

/// a^(256^16) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    gf256to16 a, b;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        memcpy(b, a, 16);
        for (uint32_t k = 0; k < 128; ++k) {
            gf256to16_mul(b, b, b);
        }

        if (memcmp(a, b, 16) != 0) {
            printf("error test_mul_field\n");
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    gf256to16 a, b, c, one = {0}, t1, t2, t3;
    one[0] = 1;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(b);
        gf256to16_random(c);

        gf256to16_mul(t1, a, one);
        if (memcmp(t1, a, 16) != 0) {
            printf("error test_mul: a*1\n");
            return 1;
        }

        gf256to16_mul(t1, a, b);
        gf256to16_mul(t2, b, a);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: a*b\n");
            return 1;
        }

        gf256to16_mul(t2, t1, c);
        gf256to16_mul(t3, b, c);
        gf256to16_mul(t1, a, t3);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }

        gf256to16_add(t3, b, c);
        gf256to16_mul(t1, a, t3);
        gf256to16_mul(t2, a, b);
        gf256to16_mul(t3, a, c);
        gf256to16_add(t2, t2, t3);
        if (memcmp(t1, t2, 16) != 0) {
            printf("error test_mul: a*(b+c)\n");
            return 1;
        }
    }
    return 0;
}

/// add/sub/neg/inv
int test_arith_base() {
    gf256to16 a, b, t1, t2, zero = {0}, one = {0};
    one[0] = 1;

    gf256to16_inv(t1, zero);
    if (memcmp(t1, zero, 16) != 0) {
        printf("error test_arith_base: inv(0)\n");
        return 1;
    }

    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(b);
        // also test elements from the subfields
        if (t < 256) { memset(a, 0, 16); a[0] = t; }
        else if (t < 512) { memset(a, 0, 16); a[8] = t - 256; }
        if (memcmp(a, zero, 16) == 0) { continue; }

        gf256to16_add(t1, a, b);
        gf256to16_sub(t2, t1, b);
        if (memcmp(t2, a, 16) != 0) {
            printf("error test_arith_base: (a+b)-b\n");
            return 1;
        }

        gf256to16_neg(t1, a);
        gf256to16_add(t2, a, t1);
        if (memcmp(t2, zero, 16) != 0) {
            printf("error test_arith_base: a + (-a)\n");
            return 1;
        }

        gf256to16_inv(t1, a);
        gf256to16_mul(t2, a, t1);
        if (memcmp(t2, one, 16) != 0) {
            printf("error test_arith_base: a*inv(a)\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
int test_vector_mul() {
    uint8_t a[32], b[32], c[32], e[32], f[32];
    gf256to16 d;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(a + 16);
        gf256to16_random(b);
        gf256to16_random(b + 16);

        const __m256i a256 = _mm256_loadu_si256((const __m256i *)a);
        const __m256i b256 = _mm256_loadu_si256((const __m256i *)b);
        _mm256_storeu_si256((__m256i *)c, gf256to16v_mul_u256(a256, b256));
        _mm256_storeu_si256((__m256i *)e, gf256to16v_add_u256(a256, b256));
        _mm256_storeu_si256((__m256i *)f, gf256to16v_sub_u256(a256, b256));

        for (uint32_t i = 0; i < 2; ++i) {
            gf256to16_mul(d, a + 16*i, b + 16*i);
            if (memcmp(d, c + 16*i, 16) != 0) {
                printf("error test_vector_mul\n");
                return 1;
            }
            gf256to16_add(d, a + 16*i, b + 16*i);
            if (memcmp(d, e + 16*i, 16) || memcmp(d, f + 16*i, 16)) {
                printf("error test_vector_mul: add/sub\n");
                return 1;
            }
        }

        // SSE version
        _mm_storeu_si128((__m128i *)c, gf256to16v_mul_u128(_mm_loadu_si128((const __m128i *)a),
                                                           _mm_loadu_si128((const __m128i *)b)));
        gf256to16_mul(d, a, b);
        if (memcmp(d, c, 16) != 0) {
            printf("error test_vector_mul: u128\n");
            return 1;
        }
    }
    return 0;
}
#endif

#ifdef USE_NEON
int test_vector_mul_u128() {
    gf256to16 a, b, c, d, e, f;
    for (uint32_t t = 0; t < N; ++t) {
        gf256to16_random(a);
        gf256to16_random(b);
        const uint64x2_t a128 = vreinterpretq_u64_u8(vld1q_u8(a));
        const uint64x2_t b128 = vreinterpretq_u64_u8(vld1q_u8(b));
        vst1q_u8(c, vreinterpretq_u8_u64(gf256to16v_mul_u128(a128, b128)));
        vst1q_u8(e, vreinterpretq_u8_u64(gf256to16v_add_u128(a128, b128)));
        vst1q_u8(f, vreinterpretq_u8_u64(gf256to16v_sub_u128(a128, b128)));

        gf256to16_mul(d, a, b);
        if (memcmp(d, c, 16) != 0) {
            printf("error test_vector_mul_u128\n");
            return 1;
        }
        gf256to16_add(d, a, b);
        if (memcmp(d, e, 16) || memcmp(d, f, 16)) {
            printf("error test_vector_mul_u128: add/sub\n");
            return 1;
        }
    }
    return 0;
}
#endif

#ifdef USE_AVX512
int test_vector_mul_u512() {
    uint8_t a[64], b[64], c[64], e[64], f[64];
    gf256to16 d;
    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 4; ++i) {
            gf256to16_random(a + 16*i);
            gf256to16_random(b + 16*i);
        }

        const __m512i a512 = _mm512_loadu_si512((const void *)a);
        const __m512i b512 = _mm512_loadu_si512((const void *)b);
        _mm512_storeu_si512((void *)c, gf256to16v_mul_u512(a512, b512));
        _mm512_storeu_si512((void *)e, gf256to16v_add_u512(a512, b512));
        _mm512_storeu_si512((void *)f, gf256to16v_sub_u512(a512, b512));

        for (uint32_t i = 0; i < 4; ++i) {
            gf256to16_mul(d, a + 16*i, b + 16*i);
            if (memcmp(d, c + 16*i, 16) != 0) {
                printf("error test_vector_mul_u512\n");
                return 1;
            }
            gf256to16_add(d, a + 16*i, b + 16*i);
            if (memcmp(d, e + 16*i, 16) || memcmp(d, f + 16*i, 16)) {
                printf("error test_vector_mul_u512: add/sub\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

/// reference vector functions
int test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        const size_t bytes = n*sizeof(gf256to16);
        gf256to16 *a = gf256to16_vector_alloc(n), *b = gf256to16_vector_alloc(n),
                  *c = gf256to16_vector_alloc(n), *d = gf256to16_vector_alloc(n);
        gf256to16 s, t1, t2, acc = {0}, zero = {0};
        int ret = 0;
        for (uint32_t i = 0; i < n; i++) {
            gf256to16_vector_get(t1, c, i);
            if (memcmp(t1, zero, 16)) { ret = 1; }
        }

        gf256to16_vector_random(a, n);
        gf256to16_vector_random(b, n);
        gf256to16_vector_copy(c, a, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to16_vector_get(t1, a, i);
            gf256to16_vector_set(d, i, t1);
        }
        if (memcmp(c, a, bytes) || memcmp(d, a, bytes)) { ret = 2; }

        gf256to16_vector_add(c, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to16_add(t1, a[i], b[i]);
            if (memcmp(c[i], t1, 16)) { ret = 3; }
        }

        gf256to16_random(s);
        gf256to16_vector_copy(c, a, n);
        gf256to16_vector_scalar_add(c, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to16_mul(t1, s, b[i]);
            gf256to16_add(t1, t1, a[i]);
            if (memcmp(c[i], t1, 16)) { ret = 4; }
        }
        gf256to16_vector_scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 5; }

        for (uint32_t i = 0; i < n; i++) {
            gf256to16_mul(t1, a[i], b[i]);
            gf256to16_add(acc, acc, t1);
        }
        gf256to16_vector_mul_acc(t2, a, b, n);
        if (memcmp(acc, t2, 16)) { ret = 6; }

        gf256to16_vector_zero(c, n);
        for (uint32_t i = 0; i < n; i++) {
            if (memcmp(c[i], zero, 16)) { ret = 7; }
        }

        free(a); free(b); free(c); free(d);
        if (ret) {
            printf("error test_vector_ref: %d, n=%u\n", ret, n);
            return 1;
        }
    }
    return 0;
}

/// reference matrix functions
int test_matrix_ref() {
    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t n1 = test_shapes[l][0], c1 = test_shapes[l][1], c2 = test_shapes[l][2];
        gf256to16 *A  = gf256to16_matrix_alloc(n1, c1), *A2 = gf256to16_matrix_alloc(n1, c1),
                  *At = gf256to16_matrix_alloc(c1, n1), *B  = gf256to16_matrix_alloc(c1, c2),
                  *R1 = gf256to16_matrix_alloc(n1, c2), *R2 = gf256to16_matrix_alloc(n1, c2),
                  *R3 = gf256to16_matrix_alloc(n1, c2), *I  = gf256to16_matrix_alloc(n1, n1),
                  *Bc = gf256to16_vector_alloc(c1);
        gf256to16 t1, t2, s;
        int ret = 0;

        gf256to16_matrix_random(A, n1, c1);
        gf256to16_matrix_random(B, c1, c2);

        // get/set/copy
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c1; j++) {
                gf256to16_matrix_get(t1, A, n1, i, j);
                gf256to16_matrix_set(A2, n1, i, j, t1);
            }
        }
        if (memcmp(A, A2, n1*c1*sizeof(gf256to16))) { ret = 1; }
        gf256to16_matrix_zero(A2, n1, c1);
        gf256to16_matrix_copy(A2, A, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to16))) { ret = 2; }

        // transpose(transpose(A)) == A
        gf256to16_matrix_transpose(At, A, n1, c1);
        gf256to16_matrix_transpose(A2, At, c1, n1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to16))) { ret = 3; }

        // id*A == A
        gf256to16_matrix_id(I, n1, n1);
        gf256to16_matrix_mul(A2, I, A, n1, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to16))) { ret = 4; }

        // (A*B)[i, j] == <row i of A, column j of B>
        gf256to16_matrix_mul(R1, A, B, n1, c1, c2);
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c2; j++) {
                for (uint32_t k = 0; k < c1; k++) { gf256to16_matrix_get(Bc[k], B, c1, k, j); }
                gf256to16_vector_mul_acc(t1, At + i*c1, Bc, c1);
                gf256to16_matrix_get(t2, R1, n1, i, j);
                if (memcmp(t1, t2, 16)) { ret = 5; }
            }
        }

        // add_mul == mul + add
        gf256to16_matrix_random(R2, n1, c2);
        gf256to16_matrix_add(R3, R2, R1, n1, c2);
        gf256to16_matrix_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R2, R3, n1*c2*sizeof(gf256to16))) { ret = 6; }

        // add_scalar
        gf256to16_random(s);
        gf256to16_matrix_copy(R2, R1, n1, c2);
        gf256to16_matrix_add_scalar(R2, s, R3, n1, c2);
        for (uint32_t i = 0; i < n1*c2; i++) {
            gf256to16_mul(t1, s, R3[i]);
            gf256to16_add(t1, t1, R1[i]);
            if (memcmp(R2[i], t1, 16)) { ret = 7; }
        }

        free(A); free(A2); free(At); free(B); free(R1); free(R2); free(R3); free(I); free(Bc);
        if (ret) {
            printf("error test_matrix_ref: %d, shape %u %u %u\n", ret, n1, c1, c2);
            return 1;
        }
    }
    return 0;
}

/// set of SIMD implementations of the vector/matrix base functions
typedef struct {
    const char *name;
    void (*add)(gf256to16 *, const gf256to16 *, uint32_t);
    void (*add_v2)(gf256to16 *, const gf256to16 *, const gf256to16 *, uint32_t);
    void (*scalar_add)(gf256to16 *, const gf256to16, const gf256to16 *, uint32_t);
    void (*scalar_add_v2)(gf256to16 *, const gf256to16 *, const gf256to16, const gf256to16 *, uint32_t);
    void (*mul_acc)(gf256to16, const gf256to16 *, const gf256to16 *, uint32_t);
    void (*m_add)(gf256to16 *, const gf256to16 *, const gf256to16 *, uint32_t, uint32_t);
    void (*m_add_scalar)(gf256to16 *, const gf256to16, const gf256to16 *, uint32_t, uint32_t);
    void (*m_mul)(gf256to16 *, const gf256to16 *, const gf256to16 *, uint32_t, uint32_t, uint32_t);
    void (*m_add_mul)(gf256to16 *, const gf256to16 *, const gf256to16 *, uint32_t, uint32_t, uint32_t);
} gf256to16_simd_impl;

/// compares every SIMD implementation against the reference implementation
int test_simd(const gf256to16_simd_impl *impl) {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        const size_t bytes = n*sizeof(gf256to16);
        gf256to16 *a = gf256to16_vector_alloc(n), *b = gf256to16_vector_alloc(n),
                  *c = gf256to16_vector_alloc(n), *d = gf256to16_vector_alloc(n);
        gf256to16 s, t1, t2;
        int ret = 0;
        gf256to16_random(s);
        gf256to16_vector_random(a, n);
        gf256to16_vector_random(b, n);

        gf256to16_vector_copy(c, a, n);
        gf256to16_vector_copy(d, a, n);
        gf256to16_vector_add(c, b, n);
        impl->add(d, b, n);
        if (memcmp(c, d, bytes)) { ret = 1; }

        gf256to16_vector_random(d, n);
        impl->add_v2(d, a, b, n);
        if (memcmp(c, d, bytes)) { ret = 2; }

        gf256to16_vector_copy(c, a, n);
        gf256to16_vector_copy(d, a, n);
        gf256to16_vector_scalar_add(c, s, b, n);
        impl->scalar_add(d, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 3; }

        gf256to16_vector_random(d, n);
        impl->scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 4; }

        gf256to16_vector_mul_acc(t1, a, b, n);
        impl->mul_acc(t2, a, b, n);
        if (memcmp(t1, t2, 16)) { ret = 5; }

        free(a); free(b); free(c); free(d);
        if (ret) {
            printf("error test_simd %s (vector): %d, n=%u\n", impl->name, ret, n);
            return 1;
        }
    }

    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t n1 = test_shapes[l][0], c1 = test_shapes[l][1], c2 = test_shapes[l][2];
        gf256to16 *A  = gf256to16_matrix_alloc(n1, c1), *A2 = gf256to16_matrix_alloc(n1, c1),
                  *B  = gf256to16_matrix_alloc(c1, c2),
                  *R1 = gf256to16_matrix_alloc(n1, c2), *R2 = gf256to16_matrix_alloc(n1, c2),
                  *M1 = gf256to16_matrix_alloc(n1, c1), *M2 = gf256to16_matrix_alloc(n1, c1);
        gf256to16 s;
        int ret = 0;
        gf256to16_random(s);
        gf256to16_matrix_random(A, n1, c1);
        gf256to16_matrix_random(A2, n1, c1);
        gf256to16_matrix_random(B, c1, c2);

        gf256to16_matrix_add(M1, A, A2, n1, c1);
        impl->m_add(M2, A, A2, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to16))) { ret = 1; }

        gf256to16_matrix_copy(M2, M1, n1, c1);
        gf256to16_matrix_add_scalar(M1, s, A, n1, c1);
        impl->m_add_scalar(M2, s, A, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to16))) { ret = 2; }

        gf256to16_matrix_random(R2, n1, c2);
        gf256to16_matrix_mul(R1, A, B, n1, c1, c2);
        impl->m_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to16))) { ret = 3; }

        gf256to16_matrix_random(R1, n1, c2);
        gf256to16_matrix_copy(R2, R1, n1, c2);
        gf256to16_matrix_add_mul(R1, A, B, n1, c1, c2);
        impl->m_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to16))) { ret = 4; }

        free(A); free(A2); free(B); free(R1); free(R2); free(M1); free(M2);
        if (ret) {
            printf("error test_simd %s (matrix): %d, shape %u %u %u\n", impl->name, ret, n1, c1, c2);
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
static const gf256to16_simd_impl gf256to16_impl_u256 = {
    "u256",
    gf256to16_vector_add_u256, gf256to16_vector_add_u256_v2,
    gf256to16_vector_scalar_add_u256, gf256to16_vector_scalar_add_u256_v2,
    gf256to16_vector_mul_acc_u256,
    gf256to16_matrix_add_u256, gf256to16_matrix_add_scalar_u256,
    gf256to16_matrix_mul_u256, gf256to16_matrix_add_mul_u256,
};
#endif
#ifdef USE_NEON
static const gf256to16_simd_impl gf256to16_impl_u128 = {
    "u128",
    gf256to16_vector_add_u128, gf256to16_vector_add_u128_v2,
    gf256to16_vector_scalar_add_u128, gf256to16_vector_scalar_add_u128_v2,
    gf256to16_vector_mul_acc_u128,
    gf256to16_matrix_add_u128, gf256to16_matrix_add_scalar_u128,
    gf256to16_matrix_mul_u128, gf256to16_matrix_add_mul_u128,
};
#endif
#ifdef USE_AVX512
static const gf256to16_simd_impl gf256to16_impl_u512 = {
    "u512",
    gf256to16_vector_add_u512, gf256to16_vector_add_u512_v2,
    gf256to16_vector_scalar_add_u512, gf256to16_vector_scalar_add_u512_v2,
    gf256to16_vector_mul_acc_u512,
    gf256to16_matrix_add_u512, gf256to16_matrix_add_scalar_u512,
    gf256to16_matrix_mul_u512, gf256to16_matrix_add_mul_u512,
};
#endif

int main () {
    srand(0);
    if (test_mul_field()) { return 1; }
    if (test_mul()) { return 1; }
    if (test_arith_base()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
    if (test_simd(&gf256to16_impl_u256)) { return 1; }
#endif
#ifdef USE_NEON
    if (test_vector_mul_u128()) { return 1; }
    if (test_simd(&gf256to16_impl_u128)) { return 1; }
#endif
#ifdef USE_AVX512
    if (test_vector_mul_u512()) { return 1; }
    if (test_simd(&gf256to16_impl_u512)) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
