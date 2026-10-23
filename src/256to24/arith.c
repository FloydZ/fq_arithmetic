#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "arith.h"
#include "vector.h"
#include "matrix.h"

static const uint32_t test_lens[] = {1,2,3,7,8,15,16,17,31,32,33,63,64,65,100,257};
#define NR_TEST_LENS (sizeof(test_lens)/sizeof(test_lens[0]))
/// {nrows1, ncols1, ncols2}
static const uint32_t test_shapes[][3] = {{1,1,1}, {3,5,2}, {17,9,13}, {33,40,7}};
#define NR_TEST_SHAPES (sizeof(test_shapes)/sizeof(test_shapes[0]))

static void gf256to24_random(gf256to24 a) {
    for (uint32_t i = 0; i < 24; i++) {
        a[i] = rand();
    }
}

/// X^3 = X + 1
static int test_mul_modulus() {
    gf256to24 x = {0}, t, r, e = {0};
    x[8] = 1;
    gf256to24_mul(t, x, x);
    gf256to24_mul(r, t, x);

    e[0] = 1;
    e[8] = 1;
    if (memcmp(r, e, 24) != 0) {
        printf("error gf256to24_mul: X^3 != X + 1\n");
        return 1;
    }
    return 0;
}

/// a*1 = a, a*b = b*a, (a*b)*c = a*(b*c), a*(b+c) = a*b + a*c
static int test_mul() {
    gf256to24 a, b, c, one = {0}, t1, t2, t3;
    one[0] = 1;
    for (uint32_t i = 0; i < (1u << 12); i++) {
        gf256to24_random(a);
        gf256to24_random(b);
        gf256to24_random(c);

        gf256to24_mul(t1, a, one);
        if (memcmp(t1, a, 24) != 0) {
            printf("error gf256to24_mul: a*1 != a\n");
            return 1;
        }

        gf256to24_mul(t1, a, b);
        gf256to24_mul(t2, b, a);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: a*b != b*a\n");
            return 1;
        }

        gf256to24_mul(t2, t1, c);
        gf256to24_mul(t3, b, c);
        gf256to24_mul(t1, a, t3);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: (a*b)*c != a*(b*c)\n");
            return 1;
        }

        gf256to24_add(t3, b, c);
        gf256to24_mul(t1, a, t3);
        gf256to24_mul(t2, a, b);
        gf256to24_mul(t3, a, c);
        gf256to24_add(t2, t2, t3);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error gf256to24_mul: a*(b+c) != a*b + a*c\n");
            return 1;
        }
    }
    return 0;
}

/// a^(256^24) = a holds only if the extension polynomial is irreducible
static int test_mul_field() {
    gf256to24 a, b;
    for (uint32_t t = 0; t < (1u << 8); t++) {
        gf256to24_random(a);
        memcpy(b, a, 24);
        for (uint32_t k = 0; k < 192; ++k) {
            gf256to24_mul(b, b, b);
        }

        if (memcmp(a, b, 24) != 0) {
            printf("error gf256to24_mul: a^(256^24) != a\n");
            return 1;
        }
    }
    return 0;
}

/// add/sub/neg/inv and the subfield multiplications
static int test_arith_base() {
    gf256to24 a, b, t1, t2, zero = {0}, one = {0};
    one[0] = 1;

    gf256to24_inv(t1, zero);
    if (memcmp(t1, zero, 24) != 0) {
        printf("error test_arith_base: inv(0)\n");
        return 1;
    }

    for (uint32_t t = 0; t < (1u << 8); t++) {
        gf256to24_random(a);
        gf256to24_random(b);
        // also test elements from the subfields
        if (t < 16) { memset(a, 0, 24); a[0] = t; }
        else if (t < 32) { memset(a, 0, 24); a[16] = t; }
        if (memcmp(a, zero, 24) == 0) { continue; }

        gf256to24_add(t1, a, b);
        gf256to24_sub(t2, t1, b);
        if (memcmp(t2, a, 24) != 0) {
            printf("error test_arith_base: (a+b)-b\n");
            return 1;
        }

        gf256to24_neg(t1, a);
        gf256to24_add(t2, a, t1);
        if (memcmp(t2, zero, 24) != 0) {
            printf("error test_arith_base: a + (-a)\n");
            return 1;
        }

        gf256to24_inv(t1, a);
        gf256to24_mul(t2, a, t1);
        if (memcmp(t2, one, 24) != 0) {
            printf("error test_arith_base: a*inv(a)\n");
            return 1;
        }

        // multiplication with an element of GF(256^2)
        const gf256to2 s = rand();
        gf256to24 s24 = {0};
        memcpy(s24, &s, 2);
        gf256to24_mul_gf256to2(t1, a, s);
        gf256to24_mul(t2, a, s24);
        if (memcmp(t1, t2, 24) != 0) {
            printf("error test_arith_base: mul_gf256to2\n");
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
static int test_vector_mul() {
    uint8_t a[32] = {0}, b[32] = {0}, c[32], e[32];
    gf256to24 d;
    for (uint32_t t = 0; t < (1u << 12); t++) {
        gf256to24_random(a);
        gf256to24_random(b);
        // the 4th limb must be ignored
        a[31] = rand();
        b[31] = rand();

        const __m256i a256 = _mm256_loadu_si256((const __m256i *)a);
        const __m256i b256 = _mm256_loadu_si256((const __m256i *)b);
        _mm256_storeu_si256((__m256i *)c, gf256to24v_mul_u256(a256, b256));

        gf256to24_mul(d, a, b);
        if (memcmp(d, c, 24) != 0) {
            printf("error gf256to24v_mul_u256\n");
            return 1;
        }
        for (uint32_t i = 24; i < 32; i++) {
            if (c[i]) {
                printf("error gf256to24v_mul_u256: 4th limb\n");
                return 1;
            }
        }

        _mm256_storeu_si256((__m256i *)c, gf256to24v_add_u256(a256, b256));
        _mm256_storeu_si256((__m256i *)e, gf256to24v_sub_u256(a256, b256));
        gf256to24_add(d, a, b);
        if (memcmp(d, c, 24) || memcmp(d, e, 24)) {
            printf("error gf256to24v_add_u256/sub_u256\n");
            return 1;
        }
    }
    return 0;
}
#endif

#ifdef USE_NEON
static int test_vector_mul_u128() {
    gf256to24 a, b, c, d;
    for (uint32_t t = 0; t < (1u << 12); t++) {
        gf256to24_random(a);
        gf256to24_random(b);

        uint64x2x2_t a128, b128;
        a128.val[0] = vreinterpretq_u64_u8(vld1q_u8(a));
        a128.val[1] = vcombine_u64(vreinterpret_u64_u8(vld1_u8(a + 16)), vdup_n_u64(rand()));
        b128.val[0] = vreinterpretq_u64_u8(vld1q_u8(b));
        b128.val[1] = vcombine_u64(vreinterpret_u64_u8(vld1_u8(b + 16)), vdup_n_u64(rand()));

        const uint64x2x2_t c128 = gf256to24v_mul_u128(a128, b128);
        vst1q_u8(c, vreinterpretq_u8_u64(c128.val[0]));
        vst1_u8(c + 16, vreinterpret_u8_u64(vget_low_u64(c128.val[1])));
        gf256to24_mul(d, a, b);
        if (memcmp(d, c, 24) || vgetq_lane_u64(c128.val[1], 1)) {
            printf("error gf256to24v_mul_u128\n");
            return 1;
        }

        const uint64x2x2_t e128 = gf256to24v_add_u128(a128, b128);
        const uint64x2x2_t f128 = gf256to24v_sub_u128(a128, b128);
        gf256to24_add(d, a, b);
        vst1q_u8(c, vreinterpretq_u8_u64(e128.val[0]));
        vst1_u8(c + 16, vreinterpret_u8_u64(vget_low_u64(e128.val[1])));
        if (memcmp(d, c, 24)) {
            printf("error gf256to24v_add_u128\n");
            return 1;
        }
        vst1q_u8(c, vreinterpretq_u8_u64(f128.val[0]));
        vst1_u8(c + 16, vreinterpret_u8_u64(vget_low_u64(f128.val[1])));
        if (memcmp(d, c, 24)) {
            printf("error gf256to24v_sub_u128\n");
            return 1;
        }
    }
    return 0;
}
#endif

#ifdef USE_AVX512
static int test_vector_mul_u512() {
    uint8_t a[64], b[64], c[64], e[64];
    gf256to24 d;
    for (uint32_t t = 0; t < (1u << 12); t++) {
        for (uint32_t i = 0; i < 64; i++) { a[i] = rand(); b[i] = rand(); }

        const __m512i a512 = _mm512_loadu_si512((const void *)a);
        const __m512i b512 = _mm512_loadu_si512((const void *)b);
        _mm512_storeu_si512((void *)c, gf256to24v_mul_u512(a512, b512));
        _mm512_storeu_si512((void *)e, gf256to24v_add_u512(a512, b512));

        for (uint32_t i = 0; i < 2; i++) {
            gf256to24_mul(d, a + 32*i, b + 32*i);
            if (memcmp(d, c + 32*i, 24) != 0) {
                printf("error gf256to24v_mul_u512\n");
                return 1;
            }
            for (uint32_t j = 24; j < 32; j++) {
                if (c[32*i + j]) {
                    printf("error gf256to24v_mul_u512: 4th limb\n");
                    return 1;
                }
            }
            gf256to24_add(d, a + 32*i, b + 32*i);
            if (memcmp(d, e + 32*i, 24) != 0) {
                printf("error gf256to24v_add_u512\n");
                return 1;
            }
        }

        _mm512_storeu_si512((void *)e, gf256to24v_sub_u512(a512, b512));
        for (uint32_t i = 0; i < 2; i++) {
            gf256to24_sub(d, a + 32*i, b + 32*i);
            if (memcmp(d, e + 32*i, 24) != 0) {
                printf("error gf256to24v_sub_u512\n");
                return 1;
            }
        }
    }
    return 0;
}
#endif

/// reference vector functions
static int test_vector_ref() {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        const size_t bytes = n*sizeof(gf256to24);
        gf256to24 *a = gf256to24_vector_alloc(n), *b = gf256to24_vector_alloc(n),
                  *c = gf256to24_vector_alloc(n), *d = gf256to24_vector_alloc(n);
        gf256to24 s, t1, t2, acc = {0}, zero = {0};
        int ret = 0;
        for (uint32_t i = 0; i < n; i++) {
            gf256to24_vector_get(t1, c, i);
            if (memcmp(t1, zero, 24)) { ret = 1; }
        }

        gf256to24_vector_random(a, n);
        gf256to24_vector_random(b, n);
        gf256to24_vector_copy(c, a, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to24_vector_get(t1, a, i);
            gf256to24_vector_set(d, i, t1);
        }
        if (memcmp(c, a, bytes) || memcmp(d, a, bytes)) { ret = 2; }

        gf256to24_vector_add(c, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to24_add(t1, a[i], b[i]);
            if (memcmp(c[i], t1, 24)) { ret = 3; }
        }

        gf256to24_random(s);
        gf256to24_vector_copy(c, a, n);
        gf256to24_vector_scalar_add(c, s, b, n);
        for (uint32_t i = 0; i < n; i++) {
            gf256to24_mul(t1, s, b[i]);
            gf256to24_add(t1, t1, a[i]);
            if (memcmp(c[i], t1, 24)) { ret = 4; }
        }
        gf256to24_vector_scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 5; }

        for (uint32_t i = 0; i < n; i++) {
            gf256to24_mul(t1, a[i], b[i]);
            gf256to24_add(acc, acc, t1);
        }
        gf256to24_vector_mul_acc(t2, a, b, n);
        if (memcmp(acc, t2, 24)) { ret = 6; }

        gf256to24_vector_zero(c, n);
        for (uint32_t i = 0; i < n; i++) {
            if (memcmp(c[i], zero, 24)) { ret = 7; }
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
static int test_matrix_ref() {
    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t n1 = test_shapes[l][0], c1 = test_shapes[l][1], c2 = test_shapes[l][2];
        gf256to24 *A  = gf256to24_matrix_alloc(n1, c1), *A2 = gf256to24_matrix_alloc(n1, c1),
                  *At = gf256to24_matrix_alloc(c1, n1), *B  = gf256to24_matrix_alloc(c1, c2),
                  *R1 = gf256to24_matrix_alloc(n1, c2), *R2 = gf256to24_matrix_alloc(n1, c2),
                  *R3 = gf256to24_matrix_alloc(n1, c2), *I  = gf256to24_matrix_alloc(n1, n1),
                  *Bc = gf256to24_vector_alloc(c1);
        gf256to24 t1, t2, s;
        int ret = 0;

        gf256to24_matrix_random(A, n1, c1);
        gf256to24_matrix_random(B, c1, c2);

        // get/set/copy
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c1; j++) {
                gf256to24_matrix_get(t1, A, n1, i, j);
                gf256to24_matrix_set(A2, n1, i, j, t1);
            }
        }
        if (memcmp(A, A2, n1*c1*sizeof(gf256to24))) { ret = 1; }
        gf256to24_matrix_zero(A2, n1, c1);
        gf256to24_matrix_copy(A2, A, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to24))) { ret = 2; }

        // transpose(transpose(A)) == A
        gf256to24_matrix_transpose(At, A, n1, c1);
        gf256to24_matrix_transpose(A2, At, c1, n1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to24))) { ret = 3; }

        // id*A == A
        gf256to24_matrix_id(I, n1, n1);
        gf256to24_matrix_mul(A2, I, A, n1, n1, c1);
        if (memcmp(A, A2, n1*c1*sizeof(gf256to24))) { ret = 4; }

        // (A*B)[i, j] == <row i of A, column j of B>
        gf256to24_matrix_mul(R1, A, B, n1, c1, c2);
        for (uint32_t i = 0; i < n1; i++) {
            for (uint32_t j = 0; j < c2; j++) {
                for (uint32_t k = 0; k < c1; k++) { gf256to24_matrix_get(Bc[k], B, c1, k, j); }
                gf256to24_vector_mul_acc(t1, At + i*c1, Bc, c1);
                gf256to24_matrix_get(t2, R1, n1, i, j);
                if (memcmp(t1, t2, 24)) { ret = 5; }
            }
        }

        // add_mul == mul + add
        gf256to24_matrix_random(R2, n1, c2);
        gf256to24_matrix_add(R3, R2, R1, n1, c2);
        gf256to24_matrix_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R2, R3, n1*c2*sizeof(gf256to24))) { ret = 6; }

        // add_scalar
        gf256to24_random(s);
        gf256to24_matrix_copy(R2, R1, n1, c2);
        gf256to24_matrix_add_scalar(R2, s, R3, n1, c2);
        for (uint32_t i = 0; i < n1*c2; i++) {
            gf256to24_mul(t1, s, R3[i]);
            gf256to24_add(t1, t1, R1[i]);
            if (memcmp(R2[i], t1, 24)) { ret = 7; }
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
    void (*add)(gf256to24 *, const gf256to24 *, uint32_t);
    void (*add_v2)(gf256to24 *, const gf256to24 *, const gf256to24 *, uint32_t);
    void (*scalar_add)(gf256to24 *, const gf256to24, const gf256to24 *, uint32_t);
    void (*scalar_add_v2)(gf256to24 *, const gf256to24 *, const gf256to24, const gf256to24 *, uint32_t);
    void (*mul_acc)(gf256to24, const gf256to24 *, const gf256to24 *, uint32_t);
    void (*m_add)(gf256to24 *, const gf256to24 *, const gf256to24 *, uint32_t, uint32_t);
    void (*m_add_scalar)(gf256to24 *, const gf256to24, const gf256to24 *, uint32_t, uint32_t);
    void (*m_mul)(gf256to24 *, const gf256to24 *, const gf256to24 *, uint32_t, uint32_t, uint32_t);
    void (*m_add_mul)(gf256to24 *, const gf256to24 *, const gf256to24 *, uint32_t, uint32_t, uint32_t);
} gf256to24_simd_impl;

/// compares every SIMD implementation against the reference implementation
static int test_simd(const gf256to24_simd_impl *impl) {
    for (uint32_t l = 0; l < NR_TEST_LENS; l++) {
        const uint32_t n = test_lens[l];
        const size_t bytes = n*sizeof(gf256to24);
        gf256to24 *a = gf256to24_vector_alloc(n), *b = gf256to24_vector_alloc(n),
                  *c = gf256to24_vector_alloc(n), *d = gf256to24_vector_alloc(n);
        gf256to24 s, t1, t2;
        int ret = 0;
        gf256to24_random(s);
        gf256to24_vector_random(a, n);
        gf256to24_vector_random(b, n);

        gf256to24_vector_copy(c, a, n);
        gf256to24_vector_copy(d, a, n);
        gf256to24_vector_add(c, b, n);
        impl->add(d, b, n);
        if (memcmp(c, d, bytes)) { ret = 1; }

        gf256to24_vector_random(d, n);
        impl->add_v2(d, a, b, n);
        if (memcmp(c, d, bytes)) { ret = 2; }

        gf256to24_vector_copy(c, a, n);
        gf256to24_vector_copy(d, a, n);
        gf256to24_vector_scalar_add(c, s, b, n);
        impl->scalar_add(d, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 3; }

        gf256to24_vector_random(d, n);
        impl->scalar_add_v2(d, a, s, b, n);
        if (memcmp(c, d, bytes)) { ret = 4; }

        gf256to24_vector_mul_acc(t1, a, b, n);
        impl->mul_acc(t2, a, b, n);
        if (memcmp(t1, t2, 24)) { ret = 5; }

        free(a); free(b); free(c); free(d);
        if (ret) {
            printf("error test_simd %s (vector): %d, n=%u\n", impl->name, ret, n);
            return 1;
        }
    }

    for (uint32_t l = 0; l < NR_TEST_SHAPES; l++) {
        const uint32_t n1 = test_shapes[l][0], c1 = test_shapes[l][1], c2 = test_shapes[l][2];
        gf256to24 *A  = gf256to24_matrix_alloc(n1, c1), *A2 = gf256to24_matrix_alloc(n1, c1),
                  *B  = gf256to24_matrix_alloc(c1, c2),
                  *R1 = gf256to24_matrix_alloc(n1, c2), *R2 = gf256to24_matrix_alloc(n1, c2),
                  *M1 = gf256to24_matrix_alloc(n1, c1), *M2 = gf256to24_matrix_alloc(n1, c1);
        gf256to24 s;
        int ret = 0;
        gf256to24_random(s);
        gf256to24_matrix_random(A, n1, c1);
        gf256to24_matrix_random(A2, n1, c1);
        gf256to24_matrix_random(B, c1, c2);

        gf256to24_matrix_add(M1, A, A2, n1, c1);
        impl->m_add(M2, A, A2, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to24))) { ret = 1; }

        gf256to24_matrix_copy(M2, M1, n1, c1);
        gf256to24_matrix_add_scalar(M1, s, A, n1, c1);
        impl->m_add_scalar(M2, s, A, n1, c1);
        if (memcmp(M1, M2, n1*c1*sizeof(gf256to24))) { ret = 2; }

        gf256to24_matrix_random(R2, n1, c2);
        gf256to24_matrix_mul(R1, A, B, n1, c1, c2);
        impl->m_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to24))) { ret = 3; }

        gf256to24_matrix_random(R1, n1, c2);
        gf256to24_matrix_copy(R2, R1, n1, c2);
        gf256to24_matrix_add_mul(R1, A, B, n1, c1, c2);
        impl->m_add_mul(R2, A, B, n1, c1, c2);
        if (memcmp(R1, R2, n1*c2*sizeof(gf256to24))) { ret = 4; }

        free(A); free(A2); free(B); free(R1); free(R2); free(M1); free(M2);
        if (ret) {
            printf("error test_simd %s (matrix): %d, shape %u %u %u\n", impl->name, ret, n1, c1, c2);
            return 1;
        }
    }
    return 0;
}

#ifdef USE_AVX2
static const gf256to24_simd_impl gf256to24_impl_u256 = {
    "u256",
    gf256to24_vector_add_u256, gf256to24_vector_add_u256_v2,
    gf256to24_vector_scalar_add_u256, gf256to24_vector_scalar_add_u256_v2,
    gf256to24_vector_mul_acc_u256,
    gf256to24_matrix_add_u256, gf256to24_matrix_add_scalar_u256,
    gf256to24_matrix_mul_u256, gf256to24_matrix_add_mul_u256,
};
#endif
#ifdef USE_NEON
static const gf256to24_simd_impl gf256to24_impl_u128 = {
    "u128",
    gf256to24_vector_add_u128, gf256to24_vector_add_u128_v2,
    gf256to24_vector_scalar_add_u128, gf256to24_vector_scalar_add_u128_v2,
    gf256to24_vector_mul_acc_u128,
    gf256to24_matrix_add_u128, gf256to24_matrix_add_scalar_u128,
    gf256to24_matrix_mul_u128, gf256to24_matrix_add_mul_u128,
};
#endif
#ifdef USE_AVX512
static const gf256to24_simd_impl gf256to24_impl_u512 = {
    "u512",
    gf256to24_vector_add_u512, gf256to24_vector_add_u512_v2,
    gf256to24_vector_scalar_add_u512, gf256to24_vector_scalar_add_u512_v2,
    gf256to24_vector_mul_acc_u512,
    gf256to24_matrix_add_u512, gf256to24_matrix_add_scalar_u512,
    gf256to24_matrix_mul_u512, gf256to24_matrix_add_mul_u512,
};
#endif

int main () {
    srand(0);
    if (test_mul_modulus()) { return 1; }
    if (test_mul()) { return 1; }
    if (test_mul_field()) { return 1; }
    if (test_arith_base()) { return 1; }
    if (test_vector_ref()) { return 1; }
    if (test_matrix_ref()) { return 1; }
#ifdef USE_AVX2
    if (test_vector_mul()) { return 1; }
    if (test_simd(&gf256to24_impl_u256)) { return 1; }
#endif
#ifdef USE_NEON
    if (test_vector_mul_u128()) { return 1; }
    if (test_simd(&gf256to24_impl_u128)) { return 1; }
#endif
#ifdef USE_AVX512
    if (test_vector_mul_u512()) { return 1; }
    if (test_simd(&gf256to24_impl_u512)) { return 1; }
#endif

    printf("all good\n");
    return 0;
}
