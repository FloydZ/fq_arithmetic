#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../helper.h"
#include "arith.h"
#include "vector.h"
#include "matrix.h"

#define N (1u << 12)

static const uint32_t test_lens[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENS (sizeof(test_lens)/sizeof(test_lens[0]))
/// {nrows1, ncols1, ncols2}
static const uint32_t test_shapes[][3] = {{1,1,1}, {3,5,2}, {17,9,13}, {33,40,7}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

static gf256to8 gf256to8_random() {
    return ((uint64_t)rand() << 42u) ^ ((uint64_t)rand() << 21u) ^ (uint64_t)rand();
}

/// a^(256^8) = a holds only if the extension polynomial is irreducible
int test_mul_field() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to8 a = gf256to8_random();
        gf256to8 b = a;
        for (uint32_t k = 0; k < 64; ++k) {
            b = gf256to8_mul(b, b);
        }

        if (a != b) {
            printf("error test_mul_field\n");
            return 1;
        }
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
int test_mul() {
    for (uint32_t t = 0; t < N; ++t) {
        const gf256to8 a = gf256to8_random(), b = gf256to8_random(), c = gf256to8_random();
        if (gf256to8_mul(a, 1) != a) {
            printf("error test_mul: a*1\n");
            return 1;
        }
        if (gf256to8_mul(a, b) != gf256to8_mul(b, a)) {
            printf("error test_mul: a*b\n");
            return 1;
        }
        if (gf256to8_mul(gf256to8_mul(a, b), c) != gf256to8_mul(a, gf256to8_mul(b, c))) {
            printf("error test_mul: (a*b)*c\n");
            return 1;
        }
        if (gf256to8_mul(a, b ^ c) != (gf256to8_mul(a, b) ^ gf256to8_mul(a, c))) {
            printf("error test_mul: a*(b+c)\n");
            return 1;
        }
    }
    return 0;
}

/// add/sub/neg/inv
int test_arith_base() {
    if (gf256to8_inv(0) != 0) {
        printf("error test_arith_base: inv(0)\n");
        return 1;
    }

    for (uint32_t t = 0; t < N; ++t) {
        // also test elements from the subfields
        gf256to8 a = gf256to8_random();
        if (t < 256) { a = t; }
        else if (t < 512) { a = ((gf256to8)(t - 256)) << 32; }
        const gf256to8 b = gf256to8_random();

        if (gf256to8_sub(gf256to8_add(a, b), b) != a) {
            printf("error test_arith_base: (a+b)-b\n");
            return 1;
        }
        if (gf256to8_add(a, gf256to8_neg(a)) != 0) {
            printf("error test_arith_base: a + (-a)\n");
            return 1;
        }
        if (a && (gf256to8_mul(a, gf256to8_inv(a)) != 1)) {
            printf("error test_arith_base: a*inv(a)\n");
            return 1;
        }
        if (gf256to8_addmul(b, a, b) != (b ^ gf256to8_mul(a, b))) {
            printf("error test_arith_base: addmul\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
int test_vector_mul() {
    v256 v1, v2, v3, v4, v5;
    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 4; ++i) {
            v1.v64[i] = gf256to8_random();
            v2.v64[i] = gf256to8_random();
        }

        v3.v256 = gf256to8v_mul_u256(v1.v256, v2.v256);
        v4.v256 = gf256to8v_add_u256(v1.v256, v2.v256);
        v5.v256 = gf256to8v_sub_u256(v1.v256, v2.v256);
        for (uint32_t i = 0; i < 4; ++i) {
            if (v3.v64[i] != gf256to8_mul(v1.v64[i], v2.v64[i])) {
                printf("error test_vector_mul\n");
                return 1;
            }
            if ((v4.v64[i] != gf256to8_add(v1.v64[i], v2.v64[i])) ||
                (v5.v64[i] != gf256to8_sub(v1.v64[i], v2.v64[i]))) {
                printf("error test_vector_mul: add/sub\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_NEON
int test_vector_mul_u128() {
    gf256to8 a[2], b[2], c[2], d[2], e[2];
    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 2; ++i) { a[i] = gf256to8_random(); b[i] = gf256to8_random(); }
        vst1q_u64(c, gf256to8v_mul_u128(vld1q_u64(a), vld1q_u64(b)));
        vst1q_u64(d, gf256to8v_add_u128(vld1q_u64(a), vld1q_u64(b)));
        vst1q_u64(e, gf256to8v_sub_u128(vld1q_u64(a), vld1q_u64(b)));
        for (uint32_t i = 0; i < 2; ++i) {
            if ((c[i] != gf256to8_mul(a[i], b[i])) ||
                (d[i] != gf256to8_add(a[i], b[i])) ||
                (e[i] != gf256to8_sub(a[i], b[i]))) {
                printf("error test_vector_mul_u128\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

#ifdef USE_AVX512
int test_vector_mul_u512() {
    gf256to8 a[8], b[8], c[8], d[8], e[8];
    for (uint32_t t = 0; t < N; ++t) {
        for (uint32_t i = 0; i < 8; ++i) { a[i] = gf256to8_random(); b[i] = gf256to8_random(); }
        const __m512i a512 = _mm512_loadu_si512((const void *)a);
        const __m512i b512 = _mm512_loadu_si512((const void *)b);
        _mm512_storeu_si512((void *)c, gf256to8v_mul_u512(a512, b512));
        _mm512_storeu_si512((void *)d, gf256to8v_add_u512(a512, b512));
        _mm512_storeu_si512((void *)e, gf256to8v_sub_u512(a512, b512));
        for (uint32_t i = 0; i < 8; ++i) {
            if ((c[i] != gf256to8_mul(a[i], b[i])) ||
                (d[i] != gf256to8_add(a[i], b[i])) ||
                (e[i] != gf256to8_sub(a[i], b[i]))) {
                printf("error test_vector_mul_u512\n");
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
        gf256to8 *a = gf256to8_vector_alloc(n), *b = gf256to8_vector_alloc(n),
                 *c = gf256to8_vector_alloc(n), *d = gf256to8_vector_alloc(n);
        int ret = 0;
        for (uint32_t i = 0; i < n; i++) {
            if (gf256to8_vector_get(c, i) != 0) { ret = 1; }
        }

        gf256to8_vector_random(a, n);
        gf256to8_vector_random(b, n);
        gf256to8_vector_copy(c, a, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to8_vector_set(d, i, gf256to8_vector_get(a, i));
        }
        if (memcmp(c, a, n*sizeof(gf256to8)) || memcmp(d, a, n*sizeof(gf256to8))) { ret = 2; }

        gf256to8_vector_add(c, b, n);
        for (uint32_t i = 0; i < n; i++) {
            if (c[i] != gf256to8_add(a[i], b[i])) { ret = 3; }
        }

        const gf256to8 s = gf256to8_random();
        gf256to8_vector_copy(c, a, n);
        gf256to8_vector_scalar_add(c, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            if (c[i] != gf256to8_add(a[i], gf256to8_mul(s, b[i]))) { ret = 4; }
        }
        gf256to8_vector_scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, n*sizeof(gf256to8))) { ret = 5; }

        gf256to8 acc = 0;
        for (uint32_t i = 0; i < n; i++) { acc ^= gf256to8_mul(a[i], b[i]); }
        if (acc != gf256to8_vector_mul_acc(a, b, n)) { ret = 6; }

        gf256to8_vector_zero(c, n);
        for (uint32_t i = 0; i < n; i++) {
            if (c[i] != 0) { ret = 7; }
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
        gf256to8 *A  = gf256to8_matrix_alloc(n1, c1), *A2 = gf256to8_matrix_alloc(n1, c1),
                 *At = gf256to8_matrix_alloc(c1, n1), *B  = gf256to8_matrix_alloc(c1, c2),
                 *R1 = gf256to8_matrix_alloc(n1, c2), *R2 = gf256to8_matrix_alloc(n1, c2),
                 *R3 = gf256to8_matrix_alloc(n1, c2), *I  = gf256to8_matrix_alloc(n1, n1),
                 *Bc = gf256to8_vector_alloc(c1);
        int ret = 0;

        gf256to8_matrix_random(A, n1, c1);
        gf256to8_matrix_random(B, c1, c2);

        // get/set/copy
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c1; j++) {
                gf256to8_matrix_set(A2, n1, i, j, gf256to8_matrix_get(A, n1, i, j));
            }
        }
        if (memcmp(A, A2, n1*c1*sizeof(gf256to8))) { ret = 1; }
        gf256to8_matrix_zero(A2, n1, c1);
        gf256to8_matrix_copy(A2, A, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to8))) { ret = 2; }

        // transpose(transpose(A)) == A
        gf256to8_matrix_transpose(At, A, n1, c1);
        gf256to8_matrix_transpose(A2, At, c1, n1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to8))) { ret = 3; }

        // id*A == A
        gf256to8_matrix_id(I, n1, n1);
        gf256to8_matrix_mul(A2, I, A, n1, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to8))) { ret = 4; }

        // (A*B)[i, j] == <row i of A, column j of B>
        gf256to8_matrix_mul(R1, A, B, n1, c1, c2);
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c2; j++) {
                for (uint32_t k = 0; k < c1; k++) { Bc[k] = gf256to8_matrix_get(B, c1, k, j); }
                if (gf256to8_matrix_get(R1, n1, i, j) != gf256to8_vector_mul_acc(At + i*c1, Bc, c1)) { ret = 5; }
            }
        }

        // add_mul == mul + add
        gf256to8_matrix_random(R2, n1, c2);
        gf256to8_matrix_add(R3, R2, R1, n1, c2);
        gf256to8_matrix_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R2, R3, n1*c2*sizeof(gf256to8))) { ret = 6; }

        // add_scalar
        const gf256to8 s = gf256to8_random();
        gf256to8_matrix_copy(R2, R1, n1, c2);
        gf256to8_matrix_add_scalar(R2, s, R3, n1, c2);
        for (uint32_t i = 0; i < n1*c2; i++) {
            if (R2[i] != gf256to8_add(R1[i], gf256to8_mul(s, R3[i]))) { ret = 7; }
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
    void (*add)(gf256to8 *, const gf256to8 *, uint32_t);
    void (*add_v2)(gf256to8 *, const gf256to8 *, const gf256to8 *, uint32_t);
    void (*scalar_add)(gf256to8 *, gf256to8, const gf256to8 *, uint32_t);
    void (*scalar_add_v2)(gf256to8 *, const gf256to8 *, gf256to8, const gf256to8 *, uint32_t);
    gf256to8 (*mul_acc)(const gf256to8 *, const gf256to8 *, uint32_t);
    void (*m_add)(gf256to8 *, const gf256to8 *, const gf256to8 *, uint32_t, uint32_t);
    void (*m_add_scalar)(gf256to8 *, gf256to8, const gf256to8 *, uint32_t, uint32_t);
    void (*m_mul)(gf256to8 *, const gf256to8 *, const gf256to8 *, uint32_t, uint32_t, uint32_t);
    void (*m_add_mul)(gf256to8 *, const gf256to8 *, const gf256to8 *, uint32_t, uint32_t, uint32_t);
} gf256to8_simd_impl;

/// compares every SIMD implementation against the reference implementation
int test_simd(const gf256to8_simd_impl *impl) {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        gf256to8 *a = gf256to8_vector_alloc(n), *b = gf256to8_vector_alloc(n),
                 *c = gf256to8_vector_alloc(n), *d = gf256to8_vector_alloc(n);
        int ret = 0;
        const gf256to8 s = gf256to8_random();
        gf256to8_vector_random(a, n);
        gf256to8_vector_random(b, n);

        gf256to8_vector_copy(c, a, n);
        gf256to8_vector_copy(d, a, n);
        gf256to8_vector_add(c, b, n);
        impl->add(d, b, n);
        if (memcmp(c, d, n*sizeof(gf256to8))) { ret = 1; }

        gf256to8_vector_random(d, n);
        impl->add_v2(d, a, b, n);
        if (memcmp(c, d, n*sizeof(gf256to8))) { ret = 2; }

        gf256to8_vector_copy(c, a, n);
        gf256to8_vector_copy(d, a, n);
        gf256to8_vector_scalar_add(c, s, b, n);
        impl->scalar_add(d, s, b, n);
        if (memcmp(c, d, n*sizeof(gf256to8))) { ret = 3; }

        gf256to8_vector_random(d, n);
        impl->scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, n*sizeof(gf256to8))) { ret = 4; }

        if (gf256to8_vector_mul_acc(a, b, n) != impl->mul_acc(a, b, n)) { ret = 5; }

        free(a); free(b); free(c); free(d);
        if (ret) {
            printf("error test_simd %s (vector): %d, n=%u\n", impl->name, ret, n);
            return 1;
        }
    }

    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t n1 = test_shapes[l][0], c1 = test_shapes[l][1], c2 = test_shapes[l][2];
        gf256to8 *A  = gf256to8_matrix_alloc(n1, c1), *A2 = gf256to8_matrix_alloc(n1, c1),
                 *B  = gf256to8_matrix_alloc(c1, c2),
                 *R1 = gf256to8_matrix_alloc(n1, c2), *R2 = gf256to8_matrix_alloc(n1, c2),
                 *M1 = gf256to8_matrix_alloc(n1, c1), *M2 = gf256to8_matrix_alloc(n1, c1);
        int ret = 0;
        const gf256to8 s = gf256to8_random();
        gf256to8_matrix_random(A, n1, c1);
        gf256to8_matrix_random(A2, n1, c1);
        gf256to8_matrix_random(B, c1, c2);

        gf256to8_matrix_add(M1, A, A2, n1, c1);
        impl->m_add(M2, A, A2, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to8))) { ret = 1; }

        gf256to8_matrix_copy(M2, M1, n1, c1);
        gf256to8_matrix_add_scalar(M1, s, A, n1, c1);
        impl->m_add_scalar(M2, s, A, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to8))) { ret = 2; }

        gf256to8_matrix_random(R2, n1, c2);
        gf256to8_matrix_mul(R1, A, B, n1, c1, c2);
        impl->m_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to8))) { ret = 3; }

        gf256to8_matrix_random(R1, n1, c2);
        gf256to8_matrix_copy(R2, R1, n1, c2);
        gf256to8_matrix_add_mul(R1, A, B, n1, c1, c2);
        impl->m_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to8))) { ret = 4; }

        free(A); free(A2); free(B); free(R1); free(R2); free(M1); free(M2);
        if (ret) {
            printf("error test_simd %s (matrix): %d, shape %u %u %u\n", impl->name, ret, n1, c1, c2);
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
static const gf256to8_simd_impl gf256to8_impl_u256 = {
    "u256",
    gf256to8_vector_add_u256, gf256to8_vector_add_u256_v2,
    gf256to8_vector_scalar_add_u256, gf256to8_vector_scalar_add_u256_v2,
    gf256to8_vector_mul_acc_u256,
    gf256to8_matrix_add_u256, gf256to8_matrix_add_scalar_u256,
    gf256to8_matrix_mul_u256, gf256to8_matrix_add_mul_u256,
};
#endif
#ifdef USE_NEON
static const gf256to8_simd_impl gf256to8_impl_u128 = {
    "u128",
    gf256to8_vector_add_u128, gf256to8_vector_add_u128_v2,
    gf256to8_vector_scalar_add_u128, gf256to8_vector_scalar_add_u128_v2,
    gf256to8_vector_mul_acc_u128,
    gf256to8_matrix_add_u128, gf256to8_matrix_add_scalar_u128,
    gf256to8_matrix_mul_u128, gf256to8_matrix_add_mul_u128,
};
#endif
#ifdef USE_AVX512
static const gf256to8_simd_impl gf256to8_impl_u512 = {
    "u512",
    gf256to8_vector_add_u512, gf256to8_vector_add_u512_v2,
    gf256to8_vector_scalar_add_u512, gf256to8_vector_scalar_add_u512_v2,
    gf256to8_vector_mul_acc_u512,
    gf256to8_matrix_add_u512, gf256to8_matrix_add_scalar_u512,
    gf256to8_matrix_mul_u512, gf256to8_matrix_add_mul_u512,
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
    if (test_simd(&gf256to8_impl_u256)) { return 1; }
#endif
#ifdef USE_NEON
    if (test_vector_mul_u128()) { return 1; }
    if (test_simd(&gf256to8_impl_u128)) { return 1; }
#endif
#ifdef USE_AVX512
    if (test_vector_mul_u512()) { return 1; }
    if (test_simd(&gf256to8_impl_u512)) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
